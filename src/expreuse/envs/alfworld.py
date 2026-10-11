"""ALFWorld (Shridhar et al., 2021): text household tasks of six types, one room each.

    python -m expreuse.envs.alfworld confirm-plans --tasks data/tasks/alfworld_evaluation.txt
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path
from typing import Sequence

from ..model import Plan, PlanStep, Task
from ..operators import ALFWORLD as OPERATORS

DATA = Path(os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld")))
MAX_STEPS = 40
SCAN_BUDGET = 30
BINDINGS = ("scan", "pddl", "none")
PLANNER_OPERATORS = {"PickupObject": "take", "PutObject": "put", "HeatObject": "heat", "CoolObject": "cool",
                     "CleanObject": "clean", "ToggleObject": "use"}

_TAKE = re.compile(r"^\s*(?:take|pick up)\s+(.+?)(?:\s+from\s+(.+?))?\s*$", re.I)
_PUT = re.compile(r"^\s*(?:put|move)\s+(.+?)\s+(?:in/on|into|in|on|to)\s+(.+?)\s*$", re.I)
_TRANSFORM = re.compile(r"^\s*(cool|heat|clean)\s+(.+?)\s+with\s+(.+?)\s*$", re.I)
_USE = re.compile(r"^\s*use\s+(.+?)\s*$", re.I)
_GOAL = re.compile(r"your task is to:\s*(.+)", re.I)
_SEE = re.compile(r"you see (.+?)\.?\s*$", re.I)
_ARTICLE = re.compile(r"^(?:a|an|some|the)\s+", re.I)
APPLIANCE_OF = {"heat": "microwave", "cool": "fridge", "clean": "sinkbasin", "use": "desklamp"}
APPLIANCES = ("microwave", "fridge", "sinkbasin", "sink")

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
    """The task plan's operators, from the goal text: the task type it states."""
    g = goal.lower()
    if any(k in g for k in ("desklamp", "lamp", "look at", "examine")):
        return ("take", "use")
    if re.search(r"\btwo\b", g):
        return ("take", "put", "take", "put")
    middle = [op for op, keys in (("clean", ("clean",)), ("heat", ("hot", "heat")), ("cool", ("cool", "cold")))
              if any(re.search(rf"\b{k}", g) for k in keys)]
    return ("take", *middle, "put")


def goal_entities(goal: str) -> tuple[str | None, str | None]:
    """(object type, destination type) named by the goal text, e.g. ('mug', 'cabinet')."""
    g = goal.lower().rstrip(".")
    destination = re.search(r"\b(?:in|on|into)\s+(?:the\s+)?(\w+)", g)
    if any(k in g for k in ("desklamp", "lamp", "look at", "examine")):
        obj = re.search(r"(?:look at|examine)\s+(?:the\s+|a\s+|an\s+)?(\w+)", g)
        return (obj.group(1) if obj else None), None
    obj = re.search(r"(?:put|place|find|clean|heat|cool)\s+(?:a|an|some|two|the)?\s*(?:hot|cool|cold|clean|heated|cooled|cleaned)?\s*(\w+)", g)
    return (obj.group(1) if obj else None), (destination.group(1) if destination else None)


def _contents(observation: str) -> list[str]:
    match = _SEE.search(observation.strip())
    if not match or "nothing" in match.group(1).lower():
        return []
    return [o for o in (_ARTICLE.sub("", p).strip() for p in re.split(r",\s*|\s+and\s+", match.group(1))) if o]


def bind_plan(operators: Sequence[str], goal: str, room: dict[str, list[str]] | None) -> Plan:
    """Target bindings from the scanned room: the goal's object where the scan saw it, the destination of the
    goal's type, and the appliance each transform needs."""
    object_type, destination_type = goal_entities(goal)
    room = room or {}
    objects = [(obj, place) for place, contents in room.items() for obj in contents
               if object_type and _type(obj) == object_type]
    receptacles = {}
    for place in room:
        receptacles.setdefault(_type(place), place)
    seen = {}
    for contents in room.values():
        for obj in contents:
            seen.setdefault(_type(obj), obj)
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
        if operator in APPLIANCE_OF:
            appliance = receptacles.get(APPLIANCE_OF[operator]) or seen.get(APPLIANCE_OF[operator])
            if appliance:
                binding["appliance"] = appliance
        plan.append(PlanStep(operator, binding))
    return tuple(plan)


def mangle(object_id: str) -> str:
    """A simulator object id ('Apple|-00.59|+00.83|-00.40') as the object's name in the game's PDDL."""
    return (object_id.replace("|", "_bar_").replace("-", "_minus_").replace("+", "_plus_")
            .replace(".", "_dot_").replace(",", "_comma_").lower())


def entity_names(problem: str) -> dict[str, str]:
    """PDDL object name -> the numbered name the game shows ('apple 1')."""
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
    """The task plan of a game: the manipulation actions of the expert plan as operators, bound to the game's
    names of the objects, receptacles and appliances they use."""
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


class AlfworldLabeler:
    """Which operator an action realizes, when it completed its subgoal, and the entities it bound."""

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


SPLITS = {"train": "train", "valid_seen": "valid_seen", "valid_unseen": "valid_unseen",
          "eval_in_distribution": "valid_seen", "eval_out_of_distribution": "valid_unseen"}


def _patch_textworld() -> None:
    """TextWorld's grammar relies on `locals().update`, which stopped working in Python 3.13."""
    if sys.version_info < (3, 13):
        return
    import textworld.envs.pddl.textgen as textgen

    def derive(self, context=None):
        context = context or self.context
        return [textgen.TerminalSymbol(eval(self.expression, vars(textgen), dict(context["variables"])))]

    textgen.EvalSymbol.derive = derive


class AlfworldApplication:
    """Task ids are the game files relative to `ALFWORLD_DATA`."""

    name = "alfworld"
    operators = OPERATORS
    system_prompt = SYSTEM_PROMPT

    def __init__(self, split: str = "eval_out_of_distribution", bindings: str = "scan"):
        if bindings not in BINDINGS:
            raise ValueError(f"bindings must be one of {BINDINGS}")
        _patch_textworld()
        self.split, self.bindings = split, bindings
        self.labeler = AlfworldLabeler()
        self._env, self._loaded, self._info = None, None, {}

    def task_ids(self, split: str | None = None) -> list[str]:
        """The solvable games of a split (default: the one given at construction)."""
        folder = DATA / "json_2.1.1" / SPLITS.get(split or self.split, split or self.split)
        return [str(game.relative_to(DATA)) for game in sorted(folder.glob("*/*/game.tw-pddl"))
                if json.loads(game.read_text()).get("solvable", True)]

    def task(self, task_id: str) -> Task:
        """The task's goal, read from the game's first observation."""
        observation = self._load(task_id)
        return Task(task_id, self.name, goal_text(observation))

    def _load(self, task_id: str) -> str:
        if self._loaded != task_id or self._env is None:
            import textworld
            import textworld.gym
            from alfworld.agents.environment.alfred_tw_env import (AlfredDemangler, AlfredExpert, AlfredExpertType,
                                                                   AlfredInfos)

            self.close()
            extras, wrappers = ["gamefile"], [AlfredDemangler(shuffle=False), AlfredInfos]
            if self.split == "train":  # the handcoded expert, which ALFWorld attaches on this split
                wrappers.append(AlfredExpert(expert_type=AlfredExpertType.HANDCODED))
                extras.append("expert_plan")
            infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=extras)
            env_id = textworld.gym.register_games([str(DATA / task_id)], infos, batch_size=1, asynchronous=False,
                                                  max_episode_steps=10_000, wrappers=wrappers)
            self._env, self._loaded = textworld.gym.make(env_id), task_id
        observations, info = self._env.reset()
        self._info = self._read(info)
        return observations[0]

    @staticmethod
    def _read(info) -> dict:
        return {"admissible_commands": info["admissible_commands"][0], "won": bool(info.get("won", [False])[0]),
                "expert_plan": list((info.get("extra.expert_plan") or [[]])[0])}

    def expert_action(self) -> str | None:
        """The next action of ALFWorld's handcoded expert, which the game provides on the training split only."""
        plan = self._info.get("expert_plan") or []
        return plan[0] if plan else None

    def _scan(self) -> dict[str, list[str]]:
        room = {}
        places = [c[len("go to "):] for c in self._info["admissible_commands"] if c.startswith("go to ")]
        for receptacle in places[:SCAN_BUDGET]:
            observation, _, _ = self.step(f"go to {receptacle}")
            if "closed" in observation.lower() and f"open {receptacle}" in self._info["admissible_commands"]:
                observation, _, _ = self.step(f"open {receptacle}")
            room[receptacle] = _contents(observation)
        return room

    def plan(self, task: Task) -> Plan:
        """Operators from the goal text; bindings as `bindings` says."""
        self.labeler.goal, self.labeler.destination = task.goal, None
        operators = goal_operators(task.goal)
        if self.bindings == "pddl":
            expert = pddl_plan(DATA / task.task_id)
            if tuple(p.operator for p in expert) == operators:
                return expert
            return tuple(PlanStep(op) for op in operators)
        if self.bindings == "scan":
            self._load(task.task_id)
            return bind_plan(operators, task.goal, self._scan())
        return tuple(PlanStep(op) for op in operators)

    def reset(self, task: Task) -> str:
        self.labeler.goal = task.goal
        return self._load(task.task_id)

    def step(self, action: str) -> tuple[str, bool, bool]:
        observations, _, dones, info = self._env.step([action])
        self._info = self._read(info)
        return observations[0], bool(dones[0]) or self._info["won"], self._info["won"]

    def admissible_actions(self) -> Sequence[str]:
        return list(self._info.get("admissible_commands", []))

    def max_steps(self, task: Task) -> int:
        return MAX_STEPS

    def operator_of(self, action: str) -> str | None:
        return self.labeler.operator_of(action)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


def confirm_plans(task_ids: Sequence[str], goals: dict[str, str]) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    """Tasks whose goal-derived operators differ from those of the game's expert PDDL plan, as (task id, from
    the goal, from the PDDL)."""
    differing = []
    for task_id in task_ids:
        derived = goal_operators(goals[task_id])
        expert = tuple(p.operator for p in pddl_plan(DATA / task_id))
        if derived != expert:
            differing.append((task_id, derived, expert))
    return differing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("confirm-plans", help="compare the goal-derived plans with the games' expert PDDL plans")
    c.add_argument("--tasks", required=True, help="task list (game files relative to ALFWORLD_DATA)")
    args = parser.parse_args()
    task_ids = [line.strip() for line in Path(args.tasks).read_text().splitlines() if line.strip()]
    app = AlfworldApplication(bindings="none")
    goals = {task_id: app.task(task_id).goal for task_id in task_ids}
    differing = confirm_plans(task_ids, goals)
    for task_id, derived, expert in differing:
        print(f"{task_id}: goal gives {derived}, PDDL gives {expert}")
    print(f"{len(task_ids) - len(differing)}/{len(task_ids)} goal-derived plans confirmed by the PDDL")
    raise SystemExit(1 if differing else 0)


if __name__ == "__main__":
    main()
