"""An imitation policy for ScienceWorld, trained by behaviour cloning on the simulator's gold action
sequences.

    python -m expreuse.policies.scienceworld_imitation data --exclude store_games.txt --workers 24
    python -m expreuse.policies.scienceworld_imitation embed
    python -m expreuse.policies.scienceworld_imitation train --epochs 6
    python -m expreuse.policies.scienceworld_imitation select --dev-games 2
    python -m expreuse.policies.scienceworld_imitation calibrate
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from ..consumers.base import History
from ..model import Task

DATA = Path(os.environ.get("IMITATION_DATA", "data/imitation/scienceworld"))
MODELS = Path(os.environ.get("IMITATION_MODELS", "data/models/scienceworld_imitation"))
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
FIELDS = ("goal", "hist", "look", "inv")
HISTORY = 3
NEGATIVES_HARD, NEGATIVES_RANDOM = 32, 32
PERTURB = 0.3
SIMPLIFICATION = "easy"
_NOOP = re.compile(r"\balready\b|No known action matches", re.IGNORECASE)
_ENV = None


def _contains_in_order(words, longer):
    it = iter(longer)
    return all(w in it for w in words)


def canonical(action: str, valid: set[str]) -> str | None:
    """The valid-list action a gold step names: the step as written, its parser aliases ('examine' -> 'look
    at', 'X in inventory' -> 'X'), or the one valid action with the same verb that contains the step's
    words in order."""
    if action in valid:
        return action
    alias = re.sub(r"^examine ", "look at ", action).replace(" in inventory", "")
    if alias in valid:
        return alias
    words = alias.split()
    fits = [v for v in valid if v.split()[0] == words[0] and _contains_in_order(words, v.split())]
    return fits[0] if len(fits) == 1 else None


def history_text(actions: Sequence[str], observation: str) -> str:
    return " | ".join(actions[-HISTORY:]) + " || " + (observation or "")


def negatives(target: str, valid: Sequence[str], rng: random.Random) -> list[str]:
    """Half the negatives share the gold action's verb or an object word, half are drawn from the rest."""
    words = set(target.split()[1:]) - {"to", "in", "on", "into", "the", "a", "containing"}
    verb = target.split()[0]
    others = [v for v in valid if v != target]
    hard = [v for v in others if v.split()[0] == verb or words & set(v.split()[1:])]
    rng.shuffle(hard)
    hard = hard[:NEGATIVES_HARD]
    rest = [v for v in others if v not in set(hard)]
    return hard + rng.sample(rest, min(NEGATIVES_RANDOM, len(rest)))


def _harmless(valid, rng):
    if "look around" in valid and rng.random() < 0.5:
        return "look around"
    options = [v for v in valid if v.startswith("look at ") or v == "inventory"]
    return rng.choice(options) if options else "look around"


def _init():
    global _ENV
    from scienceworld import ScienceWorldEnv

    _ENV = ScienceWorldEnv(envStepLimit=100000)


def replay(job):
    """One game's gold path ((task type, variation, split)): one example per gold step the valid list names."""
    task, variation, split = job
    env = _ENV
    env.load(task, variation, simplificationStr=SIMPLIFICATION, generateGoldPath=True)
    obs, info = env.reset()
    goal = info.get("taskDesc") or ""
    rng = random.Random(f"{task}:{variation}")
    examples, actions, unmatched, perturbed = [], [], 0, 0

    def record(target, valid):
        example = {"task": task, "goal": goal, "hist": history_text(actions, obs), "look": info.get("look") or "",
                   "inv": info.get("inv") or "", "target": target}
        example["candidates"] = valid if split == "dev" else negatives(target, valid, rng)
        examples.append(example)

    for gold in env.get_gold_action_sequence():
        valid = list(dict.fromkeys(info.get("valid") or []))
        target = canonical(gold, set(valid))
        if target is None:
            unmatched += 1
        else:
            record(target, valid)
            if split == "train" and rng.random() < PERTURB:
                for _ in range(rng.choice((1, 2))):
                    detour = _harmless(valid, rng)
                    obs, _, _, info = env.step(detour)
                    actions.append(detour)
                    valid = list(dict.fromkeys(info.get("valid") or []))
                    still = canonical(gold, set(valid))
                    if still is not None:
                        record(still, valid)
                        perturbed += 1
        after, _, done, next_info = env.step(gold)
        if target is None and _NOOP.search(after or ""):
            info = next_info  # nothing happened: the history and the last observation stay as they were
        else:
            obs, info = after, next_info
            actions.append(gold)
        if done:
            break
    return {"task": task, "variation": variation, "score": info.get("score") or 0, "examples": examples,
            "unmatched": unmatched, "perturbed": perturbed, "steps": len(actions)}


def build_data(excluded: set[str], workers: int, dev_per_type: int) -> None:
    """Replay every training variation except `excluded` ('task:variation' ids) and `dev_per_type` dev
    variations per task type; writes train.jsonl, dev.jsonl and data_report.json under DATA."""
    from scienceworld import ScienceWorldEnv

    env = ScienceWorldEnv(envStepLimit=100000)
    jobs = []
    for task in env.get_task_names():
        env.load(task, 0, simplificationStr=SIMPLIFICATION)
        jobs += [(task, v, "train") for v in env.get_variations_train() if f"{task}:{v}" not in excluded]
        jobs += [(task, v, "dev") for v in env.get_variations_dev()[:dev_per_type]]
    env.close()
    DATA.mkdir(parents=True, exist_ok=True)
    report = {"left_out": len(excluded)}
    with mp.Pool(workers, initializer=_init) as pool:
        for split in ("train", "dev"):
            mine = [j for j in jobs if j[2] == split]
            counts = defaultdict(int)
            with open(DATA / f"{split}.jsonl", "w") as out:
                for game in pool.imap_unordered(replay, mine, chunksize=4):
                    out.writelines(json.dumps(example) + "\n" for example in game["examples"])
                    counts["examples"] += len(game["examples"])
                    counts["gold_steps_not_in_valid_list"] += game["unmatched"]
                    counts["detour_examples"] += game["perturbed"]
                    counts["gold_steps"] += game["steps"]
            report[split] = {"games": len(mine), **counts}
            print(f"{split}: {len(mine)} games, {dict(counts)}", flush=True)
    (DATA / "data_report.json").write_text(json.dumps(report, indent=1))


def _encoder(threads: int):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(threads)
    return SentenceTransformer(ENCODER, device=os.environ.get("IMITATION_DEVICE", "cpu"))


def embed(threads: int) -> None:
    """Embed every distinct text of the train and dev data once (per state field, and every action)."""
    texts = {f: set() for f in (*FIELDS, "action")}
    for split in ("train", "dev"):
        with open(DATA / f"{split}.jsonl") as f:
            for line in f:
                e = json.loads(line)
                for field in FIELDS:
                    texts[field].add(e[field])
                texts["action"].add(e["target"])
                texts["action"].update(e["candidates"])
    model = _encoder(threads)
    (DATA / "vectors").mkdir(parents=True, exist_ok=True)
    for field, values in texts.items():
        ordered = sorted(values)
        vectors = model.encode(ordered, batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
        np.save(DATA / "vectors" / f"{field}.npy", vectors.astype(np.float16))
        (DATA / "vectors" / f"{field}.json").write_text(json.dumps(ordered))
        print(f"{field}: {len(ordered)} texts", flush=True)


def make_head(dim: int = 384, hidden: int = 512):
    import torch
    from torch import nn

    class Head(nn.Module):
        """The four state vectors -> one state vector; an action vector -> the same space; the pair and their
        product -> a score."""

        def __init__(self):
            super().__init__()
            self.state_net = nn.Sequential(nn.Linear(4 * dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.action_net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.out = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def states(self, fields):
            return self.state_net(fields)

        def actions(self, vectors):
            return self.action_net(vectors)

        def score(self, state, actions):
            s = state.expand(len(actions), -1)
            return self.out(torch.cat([s, actions, s * actions], 1)).squeeze(-1)

    return Head()


def _vectors():
    import torch

    vec = {f: torch.from_numpy(np.load(DATA / "vectors" / f"{f}.npy").astype(np.float32)) for f in (*FIELDS, "action")}
    index = {f: {t: i for i, t in enumerate(json.loads((DATA / "vectors" / f"{f}.json").read_text()))} for f in vec}
    return vec, index


def train(threads: int, epochs: int, batches_per_epoch: int, batch: int, lr: float, seed: int) -> None:
    """Train the head on the embedded data; every epoch is saved under MODELS/epoch<N>/."""
    import torch

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vec, index = _vectors()

    def rows(split):
        states, candidates, tasks = [], [], []
        with open(DATA / f"{split}.jsonl") as f:
            for line in f:
                e = json.loads(line)
                states.append([index[fld][e[fld]] for fld in FIELDS])
                pool = [e["target"]] + [c for c in e["candidates"] if c != e["target"]]
                candidates.append(np.array([index["action"][c] for c in pool], dtype=np.int64))
                tasks.append(e["task"])
        return np.array(states, dtype=np.int64), candidates, tasks

    def state_matrix(ids):
        return torch.cat([vec[f][torch.from_numpy(ids[:, i])] for i, f in enumerate(FIELDS)], 1)

    train_states, train_cands, train_tasks = rows("train")
    dev_states, dev_cands, dev_tasks = rows("dev")
    by_type = defaultdict(list)
    for i, t in enumerate(train_tasks):
        by_type[t].append(i)
    types = sorted(by_type)
    head = make_head()
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    log = []
    for epoch in range(1, epochs + 1):
        started, losses = time.time(), []
        head.train()
        for _ in range(batches_per_epoch):
            # a task type is drawn uniformly, then an example within it: the games per type are very uneven
            chosen = np.array([rng.choice(by_type[types[rng.integers(len(types))]]) for _ in range(batch)])
            states = head.states(state_matrix(train_states[chosen]))
            loss = torch.stack([
                -torch.log_softmax(head.score(s, head.actions(vec["action"][torch.from_numpy(train_cands[i])])), 0)[0]
                for s, i in zip(states, chosen)]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        head.eval()
        per_type = defaultdict(list)
        with torch.no_grad():
            states = head.states(state_matrix(dev_states))
            for s, cands, task in zip(states, dev_cands, dev_tasks):
                per_type[task].append(int(head.score(s, head.actions(vec["action"][torch.from_numpy(cands)])).argmax()) == 0)
        accuracy = float(np.mean([np.mean(v) for v in per_type.values()]))
        log.append({"epoch": epoch, "loss": float(np.mean(losses)), "dev_accuracy": accuracy,
                    "seconds": round(time.time() - started)})
        print(f"epoch {epoch}: loss {log[-1]['loss']:.3f}  dev accuracy {accuracy:.3f}", flush=True)
        directory = MODELS / f"epoch{epoch}"
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(head.state_dict(), directory / "head.pt")
        (directory / "config.json").write_text(json.dumps({"encoder": ENCODER, "epoch": epoch, "dev_accuracy": accuracy}))
        (MODELS / "training_log.json").write_text(json.dumps(log, indent=1))


def select(dev_per_type: int, max_steps: int | None = None) -> str:
    """Play the dev games with every saved epoch and promote the one with the highest success (ties: mean
    score) to MODELS itself, where `ImitationPolicy` loads it."""
    from ..envs.scienceworld import ScienceWorldApplication

    app = ScienceWorldApplication()
    tasks = app.task_ids("dev", per_task=dev_per_type)
    results = {}
    for directory in sorted(MODELS.glob("epoch*")):
        policy = ImitationPolicy(directory, app)
        won, scores = 0, []
        for task_id in tasks:
            task = app.task(task_id)
            history = [("", app.reset(task))]
            success = False
            for _ in range(max_steps or app.max_steps(task)):
                admissible = list(app.admissible_actions())
                probabilities = policy.log_probs(task, history, admissible)
                action = admissible[int(np.argmax(probabilities))]
                observation, done, success = app.step(action)
                history.append((action, observation))
                if done:
                    break
            won += success
            scores.append(max(0.0, app._info["score"]))
        results[directory.name] = (won / len(tasks), float(np.mean(scores)))
        print(f"{directory.name}: dev success {100 * results[directory.name][0]:.1f}%, "
              f"mean score {results[directory.name][1]:.1f}", flush=True)
    chosen = max(results, key=results.get)
    for name in ("head.pt", "config.json"):
        shutil.copy(MODELS / chosen / name, MODELS / name)
    config = json.loads((MODELS / "config.json").read_text())
    config.update({"selected_on": "dev success", "dev_success": results[chosen][0], "dev_games": len(tasks)})
    (MODELS / "config.json").write_text(json.dumps(config))
    return chosen


def calibrate(models: Path = MODELS) -> float:
    """Fit the temperature that minimises the negative log-likelihood of the gold action over the valid list
    on the dev states (Guo et al., 2017); stored in the model's config."""
    import torch

    config = json.loads((models / "config.json").read_text())
    head = make_head()
    head.load_state_dict(torch.load(models / "head.pt", map_location="cpu"))
    head.eval()
    vec, index = _vectors()
    logits = []
    with torch.no_grad(), open(DATA / "dev.jsonl") as f:
        for line in f:
            e = json.loads(line)
            state = head.states(torch.cat([vec[fld][index[fld][e[fld]]] for fld in FIELDS]).unsqueeze(0))[0]
            pool = [e["target"]] + [c for c in e["candidates"] if c != e["target"]]
            logits.append(head.score(state, head.actions(vec["action"][[index["action"][c] for c in pool]])).numpy())

    def nll(t):
        return float(np.mean([np.log(np.exp(z / t - (z / t).max()).sum()) + (z / t).max() - z[0] / t for z in logits]))

    grid = np.exp(np.linspace(np.log(0.5), np.log(100), 200))
    temperature = float(min(grid, key=nll))
    config.update({"temperature": round(temperature, 3), "calibration": {"dev_steps": len(logits),
                                                                        "nll_raw": round(nll(1), 3), "nll_calibrated": round(nll(temperature), 3)}})
    (models / "config.json").write_text(json.dumps(config))
    return temperature


class ImitationPolicy:
    """`BasePolicy` over the valid actions."""

    def __init__(self, directory: str | Path, app=None, threads: int = 2):
        import torch

        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text())
        self.encoder = _encoder(threads)
        self.head = make_head()
        self.head.load_state_dict(torch.load(directory / "head.pt", map_location="cpu"))
        self.head.eval()
        self.temperature = float(config.get("temperature", 1.0))
        self.app = app
        self._texts: dict[str, object] = {}
        self._tried: dict[tuple, set[str]] = {}
        self._last: tuple | None = None

    def _embed(self, texts):
        import torch

        missing = [t for t in dict.fromkeys(texts) if t not in self._texts]
        if missing:
            for t, v in zip(missing, self.encoder.encode(missing, batch_size=256, normalize_embeddings=True,
                                                         convert_to_tensor=True)):
                self._texts[t] = v.float().cpu()
        return torch.stack([self._texts[t] for t in texts])

    def scores(self, task: Task, history: History, admissible: Sequence[str]) -> tuple[list[str], np.ndarray]:
        import torch

        fields = self.app.state_fields() if self.app is not None else {"look": "", "inv": ""}
        actions = [a for a, _ in history[1:]]
        last_observation = history[-1][1] if history else ""
        if len(history) == 1:
            self._tried, self._last = {}, None
        elif self._last is not None:
            self._tried.setdefault(self._last, set()).add(history[-1][0])
        here = (last_observation, fields["look"], fields["inv"])
        self._last = here
        candidates = list(dict.fromkeys(admissible)) or ["look around"]
        tried = self._tried.get(here, set())
        allowed = [c for c in candidates if c not in tried] or candidates
        with torch.no_grad():
            state = self.head.states(self._embed([task.goal, history_text(actions, last_observation),
                                                  fields["look"], fields["inv"]]).reshape(1, -1))[0]
            scores = self.head.score(state, self.head.actions(self._embed(allowed))).numpy()
        return allowed, scores

    def log_probs(self, task: Task, history: History, admissible: Sequence[str]) -> Sequence[float]:
        allowed, scores = self.scores(task, history, admissible)
        scaled = scores / self.temperature
        logp = scaled - np.log(np.exp(scaled - scaled.max()).sum()) - scaled.max()
        by_action = dict(zip(allowed, logp))
        return [by_action.get(a, -1e9) for a in admissible]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("data")
    d.add_argument("--exclude", default=None, help="file with the 'task:variation' ids of the store games")
    d.add_argument("--workers", type=int, default=24)
    d.add_argument("--dev-per-type", type=int, default=5)
    e = sub.add_parser("embed")
    e.add_argument("--threads", type=int, default=16)
    t = sub.add_parser("train")
    t.add_argument("--threads", type=int, default=16)
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--batches-per-epoch", type=int, default=1000)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("select")
    s.add_argument("--dev-games", type=int, default=5, help="dev variations per task type to play")
    sub.add_parser("calibrate")
    args = parser.parse_args()
    if args.command == "data":
        excluded = {line.strip() for line in open(args.exclude)} if args.exclude else set()
        build_data(excluded, args.workers, args.dev_per_type)
    elif args.command == "embed":
        embed(args.threads)
    elif args.command == "train":
        train(args.threads, args.epochs, args.batches_per_epoch, args.batch, args.lr, args.seed)
    elif args.command == "select":
        print("selected", select(args.dev_games))
    else:
        print(f"temperature {calibrate():.2f}")


if __name__ == "__main__":
    main()
