"""ALFWorld (Shridhar et al., 2021): text household tasks of six types, one room each.

Operators are the milestone actions of the game (take, put, heat, cool, clean, use); navigation and opening are
setup steps of the operator they precede. The task plan is read directly from the game's PDDL: the expert plan in
`traj_data.json` gives the operators, and the PDDL problem in `game.tw-pddl` names the objects they act on (the
game numbers its objects deterministically from that problem), so the plan's bindings are exact and no exploration
of the room is needed. Grounding substitutes these entities for those of the stored subtraces.
"""
from __future__ import annotations

import collections
import json
import os
import re
from pathlib import Path
from typing import Sequence

from ..model import Plan, PlanStep, Task
from ..operators import ALFWORLD as OPERATORS

DATA = Path(os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld")))
MAX_STEPS = 40
PLANNER_OPERATORS = {"PickupObject": "take", "PutObject": "put", "HeatObject": "heat", "CoolObject": "cool",
                     "CleanObject": "clean", "ToggleObject": "use"}

_TAKE = re.compile(r"^\s*(?:take|pick up)\s+(.+?)(?:\s+from\s+(.+?))?\s*$", re.I)
_PUT = re.compile(r"^\s*(?:put|move)\s+(.+?)\s+(?:in/on|into|in|on|to)\s+(.+?)\s*$", re.I)
_TRANSFORM = re.compile(r"^\s*(cool|heat|clean)\s+(.+?)\s+with\s+(.+?)\s*$", re.I)
_USE = re.compile(r"^\s*use\s+(.+?)\s*$", re.I)
_GOAL = re.compile(r"your task is to:\s*(.+)", re.I)
APPLIANCES = ("microwave", "fridge", "sinkbasin", "sink")

# The static action interface shown to an LLM consumer: the action forms, no per-step list of valid actions.
ACTION_TEMPLATES = (
    "go to [receptacle]",
    "open [receptacle]",
    "close [receptacle]",
    "take [object] from [receptacle]",
    "move [object] to [receptacle]",
    "clean [object] with [receptacle]",
    "heat [object] with [appliance]",
    "cool [object] with [appliance]",
    "use [lamp]",
    "examine [object or receptacle]",
    "inventory",
    "look",
)
SYSTEM_PROMPT = (
    "You are an agent solving a household task in a text world. Every action has one "
    "of these forms, with each bracket replaced by a numbered thing in this room "
    "such as 'cabinet 2' or 'mug 1':\n"
    + "\n".join(f"  {t}" for t in ACTION_TEMPLATES)
    + "\nYou must go to a receptacle before you can open it, examine it, take from it "
    "or put something in it, and an object is only named once you are at the "
    "receptacle holding it. The environment answers 'Nothing happens.' to an action "
    "that does not apply. Output exactly one line, 'ACTION: <the action>'."
)


def _type(name: str) -> str:
    """'garbagecan 1' -> 'garbagecan'."""
    return re.sub(r"\s+\d+\s*$", "", name.strip()).lower()


def _is_appliance(name: str | None) -> bool:
    return bool(name) and any(a in name.lower() for a in APPLIANCES)


def goal_text(observation: str) -> str:
    match = _GOAL.search(observation)
    return match.group(1).strip() if match else observation.strip()


def goal_entities(goal: str) -> tuple[str | None, str | None]:
    """(object type, destination type) named by the goal text, e.g. ('mug', 'cabinet'); used for stored traces
    whose game files are not at hand."""
    g = goal.lower().rstrip(".")
    destination = re.search(r"\b(?:in|on|into)\s+(?:the\s+)?(\w+)", g)
    if any(k in g for k in ("desklamp", "lamp", "look at", "examine")):
        obj = re.search(r"(?:look at|examine)\s+(?:the\s+|a\s+|an\s+)?(\w+)", g)
        return (obj.group(1) if obj else None), None
    obj = re.search(r"(?:put|place|find|clean|heat|cool)\s+(?:a|an|some|two|the)?\s*(?:hot|cool|cold|clean|heated|cooled|cleaned)?\s*(\w+)", g)
    return (obj.group(1) if obj else None), (destination.group(1) if destination else None)


# ---------------------------------------------------------------- the plan, from the game's PDDL

def mangle(object_id: str) -> str:
    """A simulator object id ('Apple|-00.59|+00.83|-00.40') as the object's name in the game's PDDL."""
    return (object_id.replace("|", "_bar_").replace("-", "_minus_").replace("+", "_plus_")
            .replace(".", "_dot_").replace(",", "_comma_").lower())


def entity_names(problem: str) -> dict[str, str]:
    """PDDL object name -> the numbered name the game shows ('apple 1'). The game sorts the instances of a type by
    PDDL name and numbers them from the highest number down; a sink's basin is its own type."""
    objects = problem.split("(:objects", 1)[1].split("(:init", 1)[0]
    ids = sorted(token.lower() for token in re.findall(r"\S+", objects) if "_bar_" in token)

    def base(name: str) -> str:
        head = name.split("_bar_", 1)[0]
        return head + "basin" if "basin" in name else head

    stacks = {b: list(range(1, c + 1)) for b, c in collections.Counter(map(base, ids)).items()}
    return {name: f"{base(name)} {stacks[base(name)].pop()}" for name in ids}


def _facts(problem: str, predicate: str) -> list[tuple[str, str]]:
    init = problem.split("(:init", 1)[1]
    return [(m.group(1).lower(), m.group(2).lower())
            for m in re.finditer(r"\(" + predicate + r"\s+([^\s()]+)\s+([^\s()]+)\)", init)]


def task_parameters(game: Path) -> dict[str, str]:
    """The task's PDDL parameters as lower-cased types: object, destination (parent), toggled appliance."""
    params = json.loads((game.parent / "traj_data.json").read_text())["pddl_params"]
    return {"object": params.get("object_target", "").lower(), "destination": params.get("parent_target", "").lower(),
            "appliance": params.get("toggle_target", "").lower()}


def pddl_plan(game: Path) -> Plan:
    """The task plan of a game: the manipulation actions of the expert plan as operators, bound to the game's names
    of the objects, receptacles and appliances they use. The receptacle an object is taken from is the one the
    initial state places it in or at; it is left unbound when the state does not determine it."""
    problem = json.loads(game.read_text())["pddl_problem"]
    expert = json.loads((game.parent / "traj_data.json").read_text())["plan"]["high_pddl"]
    names = entity_names(problem)
    inside = dict(_facts(problem, "inReceptacle"))
    at, receptacles_at = collections.defaultdict(list), collections.defaultdict(list)
    for obj, location in _facts(problem, "objectAtLocation"):
        at[obj].append(location)
    for receptacle, location in _facts(problem, "receptacleAtLocation"):
        receptacles_at[location].append(receptacle)

    def name(object_id: str | None) -> str | None:
        return names.get(mangle(object_id)) if object_id else None

    def source(action: dict) -> str | None:
        obj = mangle(action["objectId"])
        if obj in inside:
            return names.get(inside[obj])
        kind = (action.get("coordinateReceptacleObjectId") or [""])[0].lower()
        candidates = {r for location in at.get(obj, []) for r in receptacles_at.get(location, [])
                      if not kind or r.split("_bar_", 1)[0] == kind or (kind == "sinkbasin" and "basin" in r)}
        return names.get(candidates.pop()) if len(candidates) == 1 else None

    plan, held = [], None
    for step in expert:
        action = step["planner_action"]
        operator = PLANNER_OPERATORS.get(action["action"])
        if operator is None:
            continue
        if operator == "take":
            held = name(action["objectId"])
            binding = {"object": held, "source": source(action)}
        elif operator == "put":
            binding = {"object": name(action["objectId"]), "destination": name(action.get("receptacleObjectId"))}
        elif operator == "use":
            binding = {"appliance": name(action["objectId"])}
        else:
            binding = {"object": name(action.get("cleanObjectId")) or held, "appliance": name(action["objectId"])}
        plan.append(PlanStep(operator, {role: entity for role, entity in binding.items() if entity}))
    return tuple(plan)


# ---------------------------------------------------------------- segmentation rules

class AlfworldLabeler:
    """Which operator an action realizes, when it completed its subgoal, and the entities it bound. A put into an
    appliance is a milestone only when the appliance is the task's destination (`destination`, the type from the
    task's PDDL, or else the one the goal text names)."""

    def __init__(self, goal: str | None = None, destination: str | None = None):
        self.goal, self.destination = goal, destination

    def operator_of(self, action: str) -> str | None:
        match = _TAKE.match(action)
        if match:
            return None if _is_appliance(match.group(2)) else "take"  # taking from an appliance is inside a transform
        match = _PUT.match(action)
        if match:
            destination = _type(match.group(2))
            wanted = self.destination or (goal_entities(self.goal)[1] if self.goal else None)
            return None if _is_appliance(destination) and destination != wanted else "put"
        match = _TRANSFORM.match(action)
        if match:
            return match.group(1).lower()
        return "use" if _USE.match(action) else None

    def completes(self, operator: str, binding: dict[str, str], action: str, observation: str) -> bool:
        return self.operator_of(action) == operator and "nothing happens" not in observation.lower()

    def binding(self, operator: str, steps: Sequence[tuple[str, str]]) -> dict[str, str]:
        for action, _ in reversed(steps):
            if self.operator_of(action) != operator:
                continue
            if operator == "take":
                m = _TAKE.match(action)
                return {"object": m.group(1).strip(), **({"source": m.group(2).strip()} if m.group(2) else {})}
            if operator == "put":
                m = _PUT.match(action)
                return {"object": m.group(1).strip(), "destination": m.group(2).strip()}
            if operator == "use":
                return {"appliance": _USE.match(action).group(1).strip()}
            m = _TRANSFORM.match(action)
            return {"object": m.group(2).strip(), "appliance": m.group(3).strip()}
        return {}


# ---------------------------------------------------------------- the environment

class AlfworldApplication:
    """The ALFWorld text environment. Task ids are the game files relative to `ALFWORLD_DATA`."""

    name = "alfworld"
    operators = OPERATORS
    system_prompt = SYSTEM_PROMPT

    def __init__(self, split: str = "eval_out_of_distribution", seed: int = 42):
        from alfworld.agents.environment import get_environment

        self.split, self.seed = split, seed
        self.labeler = AlfworldLabeler()
        self._config = {
            "dataset": {"data_path": str(DATA / "json_2.1.1/train"), "eval_id_data_path": str(DATA / "json_2.1.1/valid_seen"),
                        "eval_ood_data_path": str(DATA / "json_2.1.1/valid_unseen"), "num_train_games": -1, "num_eval_games": -1},
            "logic": {"domain": str(DATA / "logic/alfred.pddl"), "grammar": str(DATA / "logic/alfred.twl2")},
            "env": {"task_types": [1, 2, 3, 4, 5, 6], "goal_desc_human_anns_prob": 0.0, "expert_type": "handcoded",
                    "domain_randomization": False, "regen_game_files": False},
            "general": {"training_method": "dagger", "random_seed": seed},
            "dagger": {"training": {"max_nb_steps_per_episode": 200}},
            "rl": {"training": {"max_nb_steps_per_episode": 200}},
        }
        self._tw = get_environment("AlfredTWEnv")(self._config, train_eval=split)
        self.game_files = list(self._tw.game_files)
        self._env, self._loaded, self._info = None, None, {}

    def task_ids(self) -> list[str]:
        return [str(Path(f).resolve().relative_to(DATA.resolve())) for f in self.game_files]

    def task(self, task_id: str) -> Task:
        """The task's goal, read from the game's first observation."""
        observation = self._load(task_id)
        return Task(task_id, self.name, goal_text(observation))

    def _load(self, task_id: str) -> str:
        if self._loaded != task_id or self._env is None:
            self._tw.game_files = [str(DATA / task_id)]
            self._env = self._tw.init_env(batch_size=1)
            self._loaded = task_id
        observations, info = self._env.reset()
        self._info = {"admissible_commands": info["admissible_commands"][0], "won": bool(info.get("won", [False])[0])}
        return observations[0]

    def plan(self, task: Task) -> Plan:
        game = DATA / task.task_id
        self.labeler.goal = task.goal
        self.labeler.destination = task_parameters(game)["destination"] or None
        return pddl_plan(game)

    def reset(self, task: Task) -> str:
        self.labeler.goal = task.goal
        return self._load(task.task_id)

    def step(self, action: str) -> tuple[str, bool, bool]:
        observations, _, dones, info = self._env.step([action])
        self._info = {"admissible_commands": info["admissible_commands"][0], "won": bool(info.get("won", [False])[0])}
        return observations[0], bool(dones[0]), self._info["won"]

    def admissible_actions(self) -> Sequence[str]:
        return list(self._info.get("admissible_commands", []))

    def max_steps(self, task: Task) -> int:
        return MAX_STEPS

    def operator_of(self, action: str) -> str | None:
        return self.labeler.operator_of(action)
