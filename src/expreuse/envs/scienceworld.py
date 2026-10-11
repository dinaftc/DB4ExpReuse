"""ScienceWorld (Wang et al., 2022): text science experiments in a house of rooms, 30 task types.

    python -m expreuse.envs.scienceworld build-plans --variations 20 --workers 8
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
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
HEAT_SOURCES = ("stove", "oven", "blast furnace", "hot plate", "lighter")
COLD_PLACES = ("freezer", "fridge", "refrigerator")
SCAN_BUDGET = 25

_VERBS = (
    (re.compile(r"^focus on\b", re.I), "focus on"),
    (re.compile(r"^activate (?:the )?(?:" + "|".join(HEAT_SOURCES) + r")\b", re.I), "heat"),
    (re.compile(r"^(?:move|put) .+ (?:to|in|into|on) (?:the )?(?:" + "|".join(COLD_PLACES) + r")\b", re.I), "cool"),
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
    (re.compile(r"^activate (.+)$", re.I), {"appliance": 1}),
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
_DESTINATION = re.compile(r"move it to the ([\w\- ]+?)(?: in the [\w\- ]+?)?\s*\.", re.I)
_SUBSTANCE = re.compile(r"(?:boil|melt|freeze|change the state of matter of|melting point of|temperature of)\s+([\w\- ]+?)\s*[.,]", re.I)
_WORD = re.compile(r"[a-z]+")
_ROOM = re.compile(r"this [\w ]*?is called the ([\w ]+?)\.", re.I)
_ARTICLE = re.compile(r"^(?:a|an|the|some)\s+", re.I)
_SUBSTANCE_PREFIX = re.compile(r"^substance called\s+", re.I)
_GENERIC = {"substance", "thing", "object", "animal", "plant", "living thing", "non-living thing"}

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
    """`known` when given, else the type whose training descriptions overlap most with this one."""
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
                    if operator == "cool":
                        binding = {"object": binding["object"], "appliance": binding["destination"]}
                    if operator == "heat":
                        heated = next((m.group(1) for a, _ in steps for m in [re.match(r"^(?:move|put) (.+?) (?:to|in|into|on) ", a.strip(), re.I)] if m), None)
                        binding = {**({"object": heated} if heated else {}), "appliance": binding["appliance"]}
                    return binding
            return {}
        return {}


def _room_contents(look: str) -> list[str]:
    """Objects named in a room's 'you see:' block (the agent, air and doors left out), with what they contain."""
    match = re.search(r"you see:\s*(.+?)(?:you also see:|$)", look or "", re.S | re.I)
    if not match:
        return []
    items = []
    for line in match.group(1).splitlines():
        s = _ARTICLE.sub("", line.strip())
        if not s or s.startswith("agent") or "substance called air" in s.lower():
            continue
        name = _SUBSTANCE_PREFIX.sub("", re.split(r",|\.|\(", s)[0].strip())
        if name:
            items.append(name)
        for inner in re.findall(r"\((?:containing|that contains)\s+(.+?)\)", s, re.I) + \
                re.findall(r"(?:On|In) the [\w\- ]+? is:\s*(.+?)\.\s*$", s, re.I):
            items += [p for p in (_SUBSTANCE_PREFIX.sub("", _ARTICLE.sub("", x.strip()).strip(" ."))
                                  for x in re.split(r"\s*\([^)]*\)|,| and ", inner))
                      if p and p.lower() not in ("nothing", "air") and not p.lower().startswith("which is")]
    return list(dict.fromkeys(items))


def bind_plan(operators: Sequence[str], description: str, rooms: dict[str, list[str]] | None = None) -> Plan:
    """Target bindings: the object the task is scored on and the destination it names, from the description;
    the substance to heat or cool, the heat source, the cold place and the containers from the scanned
    rooms."""
    rooms = rooms or {}
    everything = [o for objects in rooms.values() for o in objects]
    focus = _FOCUS.search(description)
    destination = _DESTINATION.search(description)
    substance = _SUBSTANCE.search(description)

    def first(kinds):
        return next((o for o in everything if any(k in o.lower() for k in kinds)), None)

    def located(name):
        return next((o for o in everything if name and name.lower() in o.lower()), name)

    focused = focus.group(1).strip() if focus else None
    if focused in _GENERIC:  # "focus on the substance": the substance is the one the task names, if any
        focused = substance.group(1).strip() if substance else None

    plan = []
    for op in operators:
        binding: dict[str, str] = {}
        if op == "focus on" and focused:
            binding["object"] = located(focused)
        elif op == "put" and destination:
            binding["destination"] = located(destination.group(1).strip())
        elif op == "heat":
            heat_source = first(HEAT_SOURCES)
            binding = {k: v for k, v in (("object", located(substance.group(1).strip()) if substance else None),
                                         ("appliance", heat_source)) if v}
        elif op == "cool":
            cold = first(COLD_PLACES)
            binding = {k: v for k, v in (("object", located(substance.group(1).strip()) if substance else None),
                                         ("appliance", cold)) if v}
        elif op == "measure":
            thermometer = first(("thermometer",))
            binding = {k: v for k, v in (("object", located(substance.group(1).strip()) if substance else None),
                                         ("instrument", thermometer)) if v}
        plan.append(PlanStep(op, binding))
    return tuple(plan)


def gold_operators(trajectory: Sequence[tuple[str, str, float, str | None]], start_inventory: str | None,
                   labeler: "ScienceWorldLabeler") -> list[str]:
    """The operator sequence of one gold trace: a subtrace closes at a confirmed change of the inventory (a
    pick up or a move out of it) or at a score increase, and is named by the last action of the window that
    realizes an operator."""
    operators, window, best = [], [], 0.0
    before = _inventory_count(start_inventory)
    for action, observation, score, inventory in trajectory:
        window.append(action)
        after = _inventory_count(inventory)
        confirmed = None
        if before is not None and after is not None:
            if action.strip().lower().startswith("pick up") and _PICKED.search(observation or "") and after == before + 1:
                confirmed = "take"
            elif action.strip().lower().startswith("move") and _MOVED.search(observation or "") \
                    and not _PICKED.search(observation or "") and after == before - 1:
                confirmed = labeler.operator_of(action)
        before = after
        rewarded = score > best
        best = max(best, score)
        if confirmed or rewarded:
            named = confirmed or next((labeler.operator_of(a) for a in reversed(window) if labeler.operator_of(a)), None)
            if named:
                operators.append(named)
            window = []
    return operators


def _inventory_count(text: str | None) -> int | None:
    if text is None:
        return None
    body = text.split("you see:", 1)[-1]
    return sum(1 for part in body.split("\t") if part.strip())


def _replay_gold(job):
    task, variation = job
    env = _replay_gold.env
    env.load(task, variation, simplificationStr=SIMPLIFICATION, generateGoldPath=True)
    _, info = env.reset()
    start, trajectory = info.get("inv"), []
    for action in env.get_gold_action_sequence():
        observation, _, done, info = env.step(action)
        trajectory.append((action, observation, float(info.get("score") or 0), info.get("inv")))
        if done:
            break
    description = (info.get("taskDesc") or "").replace("Task Description:\n", "").strip()
    return task, description, gold_operators(trajectory, start, ScienceWorldLabeler()), trajectory[-1][2] if trajectory else 0.0


def _init_replay():
    from scienceworld import ScienceWorldEnv

    _replay_gold.env = ScienceWorldEnv(envStepLimit=100000)


def build_plans(variations: int, workers: int, out: Path | None = None) -> dict:
    """One plan per task type: replay the gold traces of up to `variations` training variations, segment them,
    and keep the most frequent operator sequence; the training descriptions recognise the type of a new
    task."""
    from scienceworld import ScienceWorldEnv

    env = ScienceWorldEnv(envStepLimit=100000)
    jobs = []
    for task in env.get_task_names():
        env.load(task, 0, simplificationStr=SIMPLIFICATION)
        jobs += [(task, v) for v in env.get_variations_train()[:variations]]
    env.close()
    sequences, descriptions, reached = collections.defaultdict(collections.Counter), collections.defaultdict(set), collections.Counter()
    with mp.Pool(workers, initializer=_init_replay) as pool:
        for task, description, operators, score in pool.imap_unordered(_replay_gold, jobs, chunksize=2):
            sequences[task][tuple(operators)] += 1
            descriptions[task].add(description)
            reached[task] += score >= 100
    table = {task: {"operators": list(max(counter, key=lambda s: (counter[s], -len(s)))),
                    "replayed": sum(counter.values()), "with_this_sequence": max(counter.values()),
                    "descriptions": sorted(descriptions[task])} for task, counter in sorted(sequences.items())}
    (out or Path(__file__).with_name("scienceworld_plans.json")).write_text(json.dumps(table, indent=1))
    return table


class ScienceWorldApplication:
    """Task ids are '<task type>:<variation>'."""

    name = "scienceworld"
    operators = OPERATORS
    system_prompt = SYSTEM_PROMPT

    def __init__(self, step_limit: int = 150, scan: bool = True):
        from scienceworld import ScienceWorldEnv

        self._env = ScienceWorldEnv(envStepLimit=step_limit)
        self.scan_rooms = scan
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

    def scan(self) -> dict[str, list[str]]:
        """Teleport to every room and record what it holds (not charged to the task's step budget)."""
        rooms = {}
        look = self._info.get("look") or ""
        rooms[_room_name(look)] = _room_contents(look)
        for room in sorted({c[len("teleport to "):] for c in self._info.get("valid", []) if c.startswith("teleport to ")}):
            if room in rooms or len(rooms) > SCAN_BUDGET:
                continue
            self.step(f"teleport to {room}")
            look = self._info.get("look") or ""
            rooms[_room_name(look)] = _room_contents(look)
        return rooms

    def plan(self, task: Task) -> Plan:
        name = task.task_id.rsplit(":", 1)[0]
        self.reset(task)
        rooms = self.scan() if self.scan_rooms else None
        return bind_plan(plan_operators(task.goal, name), task.goal, rooms)

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

    def close(self) -> None:
        self._env.close()


def _room_name(look: str) -> str:
    match = _ROOM.search(look or "")
    return match.group(1).strip() if match else "?"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build-plans")
    b.add_argument("--variations", type=int, default=20, help="training variations replayed per task type")
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--out", default=None)
    args = parser.parse_args()
    table = build_plans(args.variations, args.workers, Path(args.out) if args.out else None)
    for task, entry in table.items():
        print(f"{task}: {entry['with_this_sequence']}/{entry['replayed']} gold traces give {entry['operators']}")


if __name__ == "__main__":
    main()
