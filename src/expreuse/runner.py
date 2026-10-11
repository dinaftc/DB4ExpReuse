"""Plan, reuse query, act, store the trace."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Callable

from .consumers.base import Consumer
from .embedding import Embedder
from .envs.base import Application
from .grounding import ground
from .model import Experience, Plan, ReuseRequest, StoredTrace, Task, Trace
from .segmentation import segment
from .stores.base import ExperienceStore


@dataclass(frozen=True)
class ReuseConfig:
    mode: str = "none"
    producer: str = ""
    applications: tuple[str, ...] = ()
    theta: float | None = None
    k: int = 1
    ground: bool = True


@dataclass
class Episode:
    task: Task
    success: bool
    experience: Experience
    stored: StoredTrace
    seconds: dict[str, float] = field(default_factory=dict)
    plan: Plan = ()


def retrieve(store: ExperienceStore, task: Task, plan: Plan, config: ReuseConfig
             ) -> tuple[Experience, bool]:
    """Returns the answer and whether it comes from another application (such answers are not grounded)."""
    applications = config.applications or (task.application,)
    cross = task.application not in applications
    request = ReuseRequest(task.embedding, config.producer, applications,
                           tuple(p.operator for p in plan), config.k)
    experience = store.query(config.mode, request, config.theta)
    if experience and config.ground and not cross:
        experience = ground(experience, plan)
    return experience, cross


def run_task(app: Application, consumer: Consumer, task: Task, embedder: Embedder,
             store: ExperienceStore | None = None,
             config: ReuseConfig = ReuseConfig(), store_as: str | None = None,
             shape: Callable[[Experience], Experience] | None = None
             ) -> Episode:
    """With `store_as`, the trace is segmented and stored as that producer's experience; `shape` edits the
    query's answer before the consumer sees it."""
    seconds = {"retrieval": 0.0, "environment": 0.0, "consumer": 0.0}
    if task.embedding is None:
        task = replace(task, embedding=embedder.encode([task.goal])[0])
    plan = app.plan(task)

    experience, cross = Experience("none"), False
    if store is not None and config.mode != "none":
        start = time.perf_counter()
        experience, cross = retrieve(store, task, plan, config)
        seconds["retrieval"] = time.perf_counter() - start
        if shape is not None:
            experience = shape(experience)

    consumer.begin(task, plan, experience, cross_application=cross)
    start = time.perf_counter()
    history = [("", app.reset(task))]
    seconds["environment"] += time.perf_counter() - start
    success = False
    for _ in range(app.max_steps(task)):
        start = time.perf_counter()
        admissible = app.admissible_actions()
        seconds["environment"] += time.perf_counter() - start
        start = time.perf_counter()
        action = consumer.act(history, admissible)
        seconds["consumer"] += time.perf_counter() - start
        start = time.perf_counter()
        observation, done, success = app.step(action)
        seconds["environment"] += time.perf_counter() - start
        history.append((action, observation))
        if done:
            break

    trace_id = uuid.uuid4().hex
    steps, subtraces = segment(trace_id, plan, history[1:], app.labeler)
    stored = StoredTrace(task, Trace(trace_id, task.task_id, store_as or "", success),
                         steps, subtraces)
    if store is not None and store_as:
        store.insert(stored)
    return Episode(task, success, experience, stored, seconds, plan)
