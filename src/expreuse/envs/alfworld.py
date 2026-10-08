"""ALFWorld (Shridhar et al., 2021): text household tasks of six types, one room each.

Operators are the milestone actions of the game (take, put, heat, cool, clean, use); navigation and opening are
setup steps of the operator they precede. The task plan follows from the goal text, which names the task type and
the destination; its bindings come from a scripted scan of the room (every receptacle visited and opened).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence

from ..model import Plan, PlanStep, Task
from ..operators import ALFWORLD as OPERATORS

DATA = Path(os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld")))
MAX_STEPS = 40
SCAN_BUDGET = 30  # receptacles visited by the scan

_TAKE = re.compile(r"^\s*(?:take|pick up)\s+(.+?)(?:\s+from\s+(.+?))?\s*$", re.I)
_PUT = re.compile(r"^\s*(?:put|move)\s+(.+?)\s+(?:in/on|into|in|on|to)\s+(.+?)\s*$", re.I)
_TRANSFORM = re.compile(r"^\s*(cool|heat|clean)\s+(.+?)\s+with\s+(.+?)\s*$", re.I)
_USE = re.compile(r"^\s*use\s+(.+?)\s*$", re.I)
_GOAL = re.compile(r"your task is to:\s*(.+)", re.I)
_SEE = re.compile(r"you see (.+?)\.?\s*$", re.I)
_ARTICLE = re.compile(r"^(?:a|an|some|the)\s+", re.I)
APPLIANCES = ("microwave", "fridge", "sinkbasin", "sink")
APPLIANCE_OF = {"heat": "microwave", "cool": "fridge", "clean": "sinkbasin", "use": "desklamp"}

# The prompt of the paper's LLM runs (the static action interface: no per-step candidates, no worked examples).
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


def goal_operators(goal: str) -> tuple[str, ...]:
    """The task plan's operators, from the goal: the task type and the destination it names."""
    g = goal.lower()
    if any(k in g for k in ("desklamp", "lamp", "look at", "examine")):
        return ("take", "use")
    middle = [op for op, keys in (("clean", ("clean",)), ("heat", ("hot", "heat")), ("cool", ("cool", "cold")))
              if any(k in g for k in keys)]
    if "two" in g:
        return ("take", "put", "take", "put")
    return ("take", *middle, "put")


def goal_entities(goal: str) -> tuple[str | None, str | None]:
    """(object type, destination type) named by the goal, e.g. ('mug', 'cabinet')."""
    g = goal.lower().rstrip(".")
    destination = re.search(r"\b(?:in|on|into)\s+(?:the\s+)?(\w+)", g)
    if any(k in g for k in ("desklamp", "lamp", "look at", "examine")):
        obj = re.search(r"(?:look at|examine)\s+(?:the\s+|a\s+|an\s+)?(\w+)", g)
        return (obj.group(1) if obj else None), None
    obj = re.search(r"(?:put|place|find|clean|heat|cool)\s+(?:a|an|some|two|the)?\s*(?:hot|cool|cold|clean|heated|cooled|cleaned)?\s*(\w+)", g)
    return (obj.group(1) if obj else None), (destination.group(1) if destination else None)


class AlfworldLabeler:
    """Which operator an action realizes, when it completed its subgoal, and the entities it bound."""

    def __init__(self, goal: str | None = None):
        self.goal = goal  # a put into an appliance is a milestone only when the goal asks for that destination

    def operator_of(self, action: str) -> str | None:
        match = _TAKE.match(action)
        if match:
            return None if _is_appliance(match.group(2)) else "take"  # taking from an appliance is inside a transform
        match = _PUT.match(action)
        if match:
            destination = _type(match.group(2))
            wanted = goal_entities(self.goal)[1] if self.goal else None
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


def _contents(observation: str) -> list[str]:
    match = _SEE.search(observation.strip())
    if not match or "nothing" in match.group(1).lower():
        return []
    return [o for o in (_ARTICLE.sub("", p).strip() for p in re.split(r",\s*|\s+and\s+", match.group(1))) if o]


def bind_plan(operators: Sequence[str], goal: str, room: dict[str, list[str]] | None) -> Plan:
    """Target bindings from the scanned room: the goal's object where the scan saw it, the destination of the
    goal's type, and the appliance each transform needs. Unknown entities leave the role unbound."""
    object_type, destination_type = goal_entities(goal)
    room = room or {}
    objects = [(obj, place) for place, contents in room.items() for obj in contents
               if object_type and _type(obj) == object_type]
    receptacles = {_type(place): place for place in room}
    plan, used = [], 0
    for operator in operators:
        binding: dict[str, str] = {}
        if operator in ("take", "put", "heat", "cool", "clean") and objects:
            obj, place = objects[min(used, len(objects) - 1)]
            binding["object"] = obj
            if operator == "take":
                binding["source"] = place
        if operator == "put":
            used += 1
            if destination_type and destination_type in receptacles:
                binding["destination"] = receptacles[destination_type]
        if operator in APPLIANCE_OF and APPLIANCE_OF[operator] in receptacles:
            binding["appliance"] = receptacles[APPLIANCE_OF[operator]]
        plan.append(PlanStep(operator, binding))
    return tuple(plan)


class AlfworldApplication:
    """The ALFWorld text environment. Task ids are the game files relative to `ALFWORLD_DATA`."""

    name = "alfworld"
    operators = OPERATORS
    system_prompt = SYSTEM_PROMPT

    def __init__(self, split: str = "eval_out_of_distribution", scan: bool = True, seed: int = 42):
        from alfworld.agents.environment import get_environment

        self.split, self.scan, self.seed = split, scan, seed
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
        self._env, self._loaded, self._info, self._room = None, None, {}, None

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

    def _scan(self) -> dict[str, list[str]]:
        room = {}
        for receptacle in [c[len("go to "):] for c in self._info["admissible_commands"] if c.startswith("go to ")][:SCAN_BUDGET]:
            observation, _, _ = self.step(f"go to {receptacle}")
            if "closed" in observation.lower() and f"open {receptacle}" in self._info["admissible_commands"]:
                observation, _, _ = self.step(f"open {receptacle}")
            room[receptacle] = _contents(observation)
        return room

    def plan(self, task: Task) -> Plan:
        self.labeler.goal = task.goal
        operators = goal_operators(task.goal)
        if self.scan:
            self._load(task.task_id)
            self._room = self._scan()
        return bind_plan(operators, task.goal, self._room if self.scan else None)

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
