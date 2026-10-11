"""ALFWorld's TextDAgger policy (Shridhar et al., 2021) as a policy consumer, and its training.

    python -m expreuse.policies.alfworld_dagger prepare --exclude data/alfworld/tasks.csv --target data/alfworld_dagger_train
    DAGGER_TRAIN_DATA=data/alfworld_dagger_train/json_2.1.1/train DAGGER_MODELS=data/models/alfworld_dagger \\
        python -m expreuse.policies.alfworld_dagger_train src/expreuse/policies/alfworld_dagger.yaml
    python -m expreuse.policies.alfworld_dagger calibrate --checkpoint data/models/alfworld_dagger/dagger_admissible_id.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np

from ..consumers.base import History
from ..model import Task

CONFIG = Path(__file__).with_name("alfworld_dagger.yaml")
DATA = Path(os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld")))


def prepare_training_split(excluded: set[str], target: Path, source: Path = DATA / "json_2.1.1/train") -> int:
    """Copy (hard-link where possible) the training games except the excluded ones, given as task ids (game
    files relative to the data folder) or task-folder names."""
    kept = 0
    for game in sorted(source.rglob("game.tw-pddl")):
        trial = game.parent
        task_id = str(game.resolve().relative_to(DATA.resolve()))
        if task_id in excluded or trial.name in excluded or trial.parent.name in excluded:
            continue
        destination = target / trial.relative_to(source.parent.parent)
        destination.mkdir(parents=True, exist_ok=True)
        for f in trial.iterdir():
            if f.is_file() and not (destination / f.name).exists():
                try:
                    os.link(f, destination / f.name)
                except OSError:
                    shutil.copy(f, destination / f.name)
        kept += 1
    return kept


def load_agent(checkpoint: str | Path, config: str | Path = CONFIG, cuda: bool = False):
    """The trained TextDAgger agent with the checkpoint's weights (loaded strictly)."""
    import torch
    import yaml
    from alfworld.agents.agent import TextDAggerAgent

    settings = yaml.safe_load(Path(config).read_text())
    settings["general"]["use_cuda"] = cuda
    agent = TextDAggerAgent(settings)
    agent.online_net.load_state_dict(torch.load(checkpoint, map_location="cuda" if cuda else "cpu"))
    agent.eval()
    return agent


class DaggerPolicy:
    """`BasePolicy` over the admissible commands."""

    def __init__(self, checkpoint: str | Path, config: str | Path = CONFIG, temperature: float | None = None,
                 cuda: bool = False):
        self.agent = load_agent(checkpoint, config, cuda)
        calibration = Path(checkpoint).with_name("calibration.json")
        self.temperature = temperature or (json.loads(calibration.read_text())["temperature"]
                                           if calibration.exists() else 1.0)
        self._steps, self._admissible, self._candidates, self._observation, self._dynamics = -1, [], None, None, None

    def _start(self, observation: str, admissible: Sequence[str]) -> None:
        a = self.agent
        a.init(1)
        task, obs = a.get_task_and_obs([observation])
        self._task = a.preprocess_task(task)
        obs = a.preprocess_observation(obs)
        a.observation_pool.push_first_sight(obs)
        self._admissible = list(admissible)
        self._candidates = a.preprocess_action_candidates([self._admissible])
        self._observation = [obs[0] + " [SEP] restart"]
        self._dynamics, self._steps = None, 0

    def _advance(self, action: str, observation: str, admissible: Sequence[str]) -> None:
        a = self.agent
        chosen = self._candidates[0][self._admissible.index(action)]
        self._dynamics = self._next
        self._admissible = list(admissible)
        self._candidates = a.preprocess_action_candidates([self._admissible])
        self._observation = [a.preprocess_observation([observation])[0] + " [SEP] " + chosen]
        self._steps += 1

    def scores(self, history: History, admissible: Sequence[str]) -> np.ndarray:
        import torch

        if len(history) == 1 or self._steps < 0 or len(history) != self._steps + 2:
            self._start(history[0][1], admissible if len(history) == 1 else [history[1][0]])
            for i, (action, observation) in enumerate(history[1:]):
                self._scores()
                last = i == len(history) - 2
                self._advance(action, observation, admissible if last else [history[i + 2][0]])
        else:
            self._advance(history[-1][0], history[-1][1], admissible)
        return self._scores()

    def _scores(self) -> np.ndarray:
        import torch

        a = self.agent
        a.observation_pool.push_batch(self._observation)
        recent = a.observation_pool.get()
        with torch.no_grad():
            h_obs, obs_mask = a.encode(recent, use_model="online")
            h_td, td_mask = a.encode(self._task, use_model="online")
            scores, _, self._next = a.action_scoring(self._candidates, h_obs, obs_mask, h_td, td_mask,
                                                     self._dynamics, use_model="online")
        return scores[0, : len(self._admissible)].float().cpu().numpy()

    def log_probs(self, task: Task, history: History, admissible: Sequence[str]) -> Sequence[float]:
        scores = self.scores(history, admissible) / self.temperature
        return list(scores - np.log(np.exp(scores - scores.max()).sum()) - scores.max())


GRID = np.exp(np.linspace(np.log(0.05), np.log(100), 300))


def fit_temperature(logits: Sequence[np.ndarray]) -> float:
    """The temperature minimising the mean negative log-likelihood of the labelled action; each entry holds a
    state's scores minus the score of its labelled action."""
    def nll(t):
        return float(np.mean([np.log(np.exp(z / t - (z / t).max()).sum()) + (z / t).max() for z in logits]))

    return float(min(GRID, key=nll))


def calibrate(checkpoint: Path, app, tasks: Sequence[str], config: Path = CONFIG) -> float:
    """Fit the temperature on the given games: the policy plays, and at every state it visits the label is the
    expert's action (`app.expert_action`). Written to calibration.json beside the checkpoint."""
    policy = DaggerPolicy(checkpoint, config, temperature=1.0)
    logits = []
    for task_id in tasks:
        task = app.task(task_id)
        history = [("", app.reset(task))]
        for _ in range(app.max_steps(task)):
            admissible = list(app.admissible_actions())
            scores = policy.scores(history, admissible)
            expert = app.expert_action()
            if expert in admissible:
                logits.append(scores - scores[admissible.index(expert)])
            action = admissible[int(np.argmax(scores))]
            observation, done, _ = app.step(action)
            history.append((action, observation))
            if done:
                break
    if not logits:
        raise RuntimeError("no state with an expert action among the admissible ones: is the expert attached?")
    temperature = fit_temperature(logits)
    Path(checkpoint).with_name("calibration.json").write_text(json.dumps(
        {"temperature": round(temperature, 3), "steps": len(logits), "games": len(tasks),
         "at_grid_bound": bool(temperature in (GRID[0], GRID[-1]))}))
    return temperature


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="the training games without the store games")
    p.add_argument("--exclude", required=True, help="CSV or text file with the task ids (game files) to leave out")
    p.add_argument("--target", required=True)
    c = sub.add_parser("calibrate", help="fit the score temperature on valid_seen games")
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--games", type=int, default=140)
    args = parser.parse_args()
    if args.command == "prepare":
        text = Path(args.exclude).read_text()
        if args.exclude.endswith(".csv"):
            excluded = {row["task_id"] for row in csv.DictReader(text.splitlines())}
        else:
            excluded = {line.strip() for line in text.splitlines() if line.strip()}
        kept = prepare_training_split(excluded, Path(args.target))
        print(f"{kept} training games kept under {args.target}")
    else:
        from ..envs.alfworld import AlfworldApplication

        from .. import config

        # ALFWorld attaches its expert on the training split only; the games played are the calibration list
        app = AlfworldApplication(split="train", bindings="none")
        tasks = Path(config.APPLICATIONS["alfworld"]["calibration_tasks"]).read_text().split()
        temperature = calibrate(Path(args.checkpoint), app, tasks[: args.games])
        print(f"temperature {temperature:.2f}")


if __name__ == "__main__":
    main()
