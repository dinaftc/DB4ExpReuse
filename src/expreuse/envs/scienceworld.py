"""ScienceWorld (Wang et al., 2022): text science experiments in a house of rooms, 30 task types.

Operators are the simulator's manipulation commands (focus on, pick up, move, pour, mix, activate, deactivate, use,
connect, read); navigation, looking and waiting belong to the operator they precede. A plan cannot in general be
read off the goal, so each task type has one plan, derived from the benchmark's gold traces of training tasks
(`scienceworld_plans.json`); the type of a new task is recognised from its description.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

from ..model import Plan, PlanStep, Task
from ..operators import SCIENCEWORLD as OPERATORS

PLANS = json.loads((Path(__file__).with_name("scienceworld_plans.json")).read_text())
STEP_BUDGET = {"mendelian-genetics-known-plant": 150, "mendelian-genetics-unknown-plant": 150, "boil": 120, "freeze": 90,
               "grow-fruit": 80, "change-the-state-of-matter-of": 70, "inclined-plane-determine-angle": 70,
               "inclined-plane-friction-unnamed-surfaces": 70, "melt": 70}
DEFAULT_BUDGET = 50
SIMPLIFICATION = "easy"

_VERBS = (
    (re.compile(r"^focus on\b", re.I), "focus on"),
    (re.compile(r"^deactivate\b", re.I), "deactivate"),
    (re.compile(r"^activate\b", re.I), "activate"),
    (re.compile(r"^use\b", re.I), "measure"),
    (re.compile(r"^(?:move|put)\b", re.I), "put"),
    (re.compile(r"^pour\b", re.I), "pour"),
    (re.compile(r"^mix\b", re.I), "mix"),
    (re.compile(r"^connect\b", re.I), "connect"),
    (re.compile(r"^read\b", re.I), "read"),
    (re.compile(r"^pick up\b", re.I), "take"),
)
_BINDINGS = (
    (re.compile(r"^pick up (.+?) in (.+)$", re.I), {"object": 1, "source": 2}),
    (re.compile(r"^pick up (.+)$", re.I), {"object": 1}),
    (re.compile(r"^(?:move|put) (.+?) (?:to|in|into|on) (.+)$", re.I), {"object": 1, "destination": 2}),
    (re.compile(r"^pour (.+?) (?:into|in|to) (.+)$", re.I), {"object": 1, "destination": 2}),
    (re.compile(r"^use (.+?) on (.+)$", re.I), {"instrument": 1, "object": 2}),
    (re.compile(r"^connect (.+?) to (.+)$", re.I), {"component_1": 1, "component_2": 2}),
    (re.compile(r"^(?:de)?activate (.+)$", re.I), {"device": 1}),
    (re.compile(r"^mix (.+)$", re.I), {"container": 1}),
    (re.compile(r"^focus on (.+)$", re.I), {"object": 1}),
    (re.compile(r"^read (.+)$", re.I), {"object": 1}),
)
_REFUSALS = ("no known action matches that input", "that can't be done", "you can't do that", "not sure what")
_PICKED = re.compile(r"you move the (.+?) to the inventory\.", re.I)
_MOVED = re.compile(r"you move the (.+?) to the (.+?)\.", re.I)
_FOCUS = re.compile(r"focus on the ([\w\- ]+?)[.,]", re.I)
_WORD = re.compile(r"[a-z]+")

# The instruction header of the paper's LLM runs (ReAct prompt): rooms, action grammar and the rule for 'focus on'.
SYSTEM_PROMPT = (
    "Interact with the environment to solve a scientific experiment.\n"
    "In the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside,\n"
    "living room, bedroom, greenhouse, art studio, hallway.\n"
    "All containers in the environment have already been opened, you can directly get items from\n"
    "the containers.\n"
    "The available actions are:\n"
    "open OBJ: open a container\n"
    "close OBJ: close a container\n"
    "activate OBJ: activate a device\n"
    "deactivate OBJ: deactivate a device\n"
    "connect OBJ to OBJ: connect electrical components\n"
    "disconnect OBJ: disconnect electrical components\n"
    "use OBJ [on OBJ]: use a device/item\n"
    "look around: describe the current room\n"
    "examine OBJ: describe an object in detail\n"
    "look at OBJ: describe a container's contents\n"
    "read OBJ: read a note or book\n"
    "move OBJ to OBJ: move an object to a container\n"
    "pick up OBJ: move an object to the inventory\n"
    "pour OBJ into OBJ: pour a liquid into a container\n"
    "mix OBJ: chemically mix a container\n"
    "teleport to LOC: teleport to a specific room\n"
    "focus on OBJ: signal intent on a task object\n"
    "wait: task no action for 10 steps\n"
    "wait1: task no action for a step\n"
    "For the pick up action: the command is 'pick up <OBJ>'.\n"
    "Do not prefix the command with text.\n"
    "You can only perform these commands. The text command you generate must be strictly compliant\n"
    "with the command grammar. Only provide one single command in your response.\n"
    "'focus on OBJ' declares the object the task is about: use it only on that object (the one the task\n"
    "names, or the one you found that fits its description), and only once you have found it. Focusing\n"
    "on anything else ends the task immediately\n"
    "with the lowest possible score."
)


def task_type(description: str, known: str | None = None) -> str:
    """The task type: `known` when given, else the type whose training descriptions overlap most with this one."""
    if known in PLANS:
        return known
    words = set(_WORD.findall(description.lower()))
    best, best_score = None, -1.0
    for name, entry in PLANS.items():
        score = max((len(words & set(_WORD.findall(d.lower()))) / max(len(words | set(_WORD.findall(d.lower()))), 1)
                     for d in entry["descriptions"]), default=0.0)
        if score > best_score:
            best, best_score = name, score
    return best


def plan_operators(description: str, known: str | None = None) -> tuple[str, ...]:
    return tuple(PLANS[task_type(description, known)]["operators"])


class ScienceWorldLabeler:
    def operator_of(self, action: str) -> str | None:
        text = action.strip()
        return next((op for rx, op in _VERBS if rx.match(text)), None)

    def completes(self, operator: str, binding: dict[str, str], action: str, observation: str) -> bool:
        if self.operator_of(action) != operator or any(r in observation.lower() for r in _REFUSALS):
            return False
        if operator == "take":
            return bool(_PICKED.search(observation))
        if operator == "put":
            return bool(_MOVED.search(observation)) and not _PICKED.search(observation)
        return True

    def binding(self, operator: str, steps: Sequence[tuple[str, str]]) -> dict[str, str]:
        source = next((m.group(1) for a, _ in steps for m in [re.match(r"^teleport to (.+)$", a.strip(), re.I)] if m), None)
        for action, _ in reversed(steps):
            if self.operator_of(action) != operator:
                continue
            for pattern, roles in _BINDINGS:
                match = pattern.match(action.strip())
                if match:
                    binding = {role: match.group(g).strip() for role, g in roles.items()}
                    if operator == "take" and source and "source" not in binding:
                        binding["source"] = source
                    return binding
            return {}
        return {}


def bind_plan(operators: Sequence[str], description: str) -> Plan:
    """Target bindings read from the task description: the object the task is scored on."""
    focus = _FOCUS.search(description)
    return tuple(PlanStep(op, {"object": focus.group(1).strip()} if op == "focus on" and focus else {})
                 for op in operators)


class ScienceWorldApplication:
    """The ScienceWorld simulator. Task ids are '<task type>:<variation>'."""

    name = "scienceworld"
    operators = OPERATORS
    system_prompt = SYSTEM_PROMPT

    def __init__(self, step_limit: int = 150):
        from scienceworld import ScienceWorldEnv

        self._env = ScienceWorldEnv(envStepLimit=step_limit)
        self.labeler = ScienceWorldLabeler()
        self._info: dict = {}

    def task_ids(self, split: str = "test", per_task: int | None = 5, types: Sequence[str] | None = None) -> list[str]:
        pick = {"train": self._env.get_variations_train, "dev": self._env.get_variations_dev,
                "test": self._env.get_variations_test}[split]
        ids = []
        for name in types or self._env.get_task_names():
            self._env.load(name, 0, simplificationStr=SIMPLIFICATION)
            ids += [f"{name}:{v}" for v in pick()[:per_task]]
        return ids

    def task(self, task_id: str) -> Task:
        self.reset(Task(task_id, self.name, ""))
        return Task(task_id, self.name, self._info["task_desc"])

    def plan(self, task: Task) -> Plan:
        name = task.task_id.rsplit(":", 1)[0]
        return bind_plan(plan_operators(task.goal, name), task.goal)

    def reset(self, task: Task) -> str:
        name, variation = task.task_id.rsplit(":", 1)
        self._env.load(name, int(variation), simplificationStr=SIMPLIFICATION)
        observation, info = self._env.reset()
        self._info = {"task_desc": info.get("taskDesc") or "", "valid": info.get("valid") or [], "score": 0.0,
                      "look": info.get("look") or "", "inv": info.get("inv") or ""}
        return observation

    def step(self, action: str) -> tuple[str, bool, bool]:
        observation, _, done, info = self._env.step(action)
        score = float(info.get("score") or 0)
        self._info.update(valid=info.get("valid") or [], score=score, look=info.get("look") or "", inv=info.get("inv") or "")
        return observation, bool(done), score >= 100

    def state_fields(self) -> dict[str, str]:
        """What the simulator reports about the current state: the room ('look') and the inventory ('inv')."""
        return {"look": self._info.get("look", ""), "inv": self._info.get("inv", "")}

    def admissible_actions(self) -> Sequence[str]:
        return list(self._info.get("valid", []))

    def max_steps(self, task: Task) -> int:
        return STEP_BUDGET.get(task.task_id.rsplit(":", 1)[0], DEFAULT_BUDGET)

    def operator_of(self, action: str) -> str | None:
        return self.labeler.operator_of(action)
