"""Synthetic household-style traces in the shape of the data model."""
from __future__ import annotations

import random

from .embedding import Embedder
from .model import Step, StoredTrace, Subtrace, Task, Trace

OBJECTS = ["mug", "cup", "apple", "potato", "plate", "bowl", "egg", "tomato",
           "knife", "pan", "bread", "lettuce"]
SOURCES = ["countertop", "diningtable", "shelf", "drawer", "sinkbasin"]
DESTINATIONS = ["cabinet", "fridge", "garbagecan", "sidetable", "desk"]
# task type -> (goal template, middle operator, appliance)
TYPES = {
    "put": ("put a {obj} in {dst}", None, None),
    "heat": ("put a hot {obj} in {dst}", "heat", "microwave"),
    "cool": ("put a cool {obj} in {dst}", "cool", "fridge"),
    "clean": ("put a clean {obj} in {dst}", "clean", "sinkbasin"),
}


def make_trace(index: int, rng: random.Random, producer: str = "synthetic",
               application: str = "alfworld", success_rate: float = 0.8,
               embedder: Embedder | None = None) -> StoredTrace:
    kind = rng.choice(list(TYPES))
    template, middle, appliance = TYPES[kind]
    obj, src, dst = rng.choice(OBJECTS), rng.choice(SOURCES), rng.choice(DESTINATIONS)
    o, s, d = (f"{name} {rng.randint(1, 4)}" for name in (obj, src, dst))
    goal = template.format(obj=obj, dst=dst)
    trace_id, task_id = f"trace-{producer}-{index}", f"task-{producer}-{index}"

    segments = [("take", {"object": o, "source": s},
                 [f"go to {s}"] * rng.randint(1, 3) + [f"take {o} from {s}"])]
    if middle:
        a = f"{appliance} 1"
        segments.append((middle, {"object": o, "appliance": a},
                         [f"go to {a}", f"{middle} {o} with {a}"]))
    segments.append(("put", {"object": o, "destination": d},
                     [f"go to {d}", f"open {d}", f"put {o} in/on {d}"]))

    success = rng.random() < success_rate
    failed_at = None if success else rng.randrange(len(segments))
    steps, subtraces = [], []
    for subgoal_id, (operator, binding, actions) in enumerate(segments):
        if failed_at is not None and subgoal_id > failed_at:
            break
        finished = subgoal_id != failed_at
        if not finished:
            actions = actions[:-1] + ["look"]
        for action in actions:
            steps.append(Step(trace_id, len(steps), subgoal_id, action,
                              f"You {action}." if action != "look" else "Nothing happens."))
        subtraces.append(Subtrace(trace_id, subgoal_id, operator, binding, finished, True))
    embedding = embedder.encode([goal])[0] if embedder else None
    return StoredTrace(Task(task_id, application, goal, embedding),
                       Trace(trace_id, task_id, producer, success),
                       tuple(steps), tuple(subtraces))


def make_traces(n: int, seed: int = 0, **kwargs) -> list[StoredTrace]:
    rng = random.Random(seed)
    return [make_trace(i, rng, **kwargs) for i in range(n)]
