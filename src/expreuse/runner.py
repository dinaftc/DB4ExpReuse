"""End-to-end loop of Figure 1: plan, reuse query, act, store the trace."""
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
    mode: str = "none"  # none | whole | subtraces | cascade
    producer: str = ""  # r: whose experience is requested
    applications: tuple[str, ...] = ()  # default: the target application
    theta: float | None = None  # CASCADE threshold
    k: int = 1
    ground: bool = True  # within-application only


@dataclass
class Episode:
    task: Task
    success: bool
    experience: Experience
    stored: StoredTrace
    seconds: dict[str, float] = field(default_factory=dict)  # per component


def retrieve(store: ExperienceStore, task: Task, plan: Plan, config: ReuseConfig
             ) -> tuple[Experience, bool]:
    """Run the reuse query for a task. Returns the answer and whether it comes
    from another application; cross-application answers are not grounded."""
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
    """Solve one task. With `store_as`, the resulting trace is segmented and
    inserted as experience of that producer. `shape` edits the query's answer
    before the consumer sees it (the coverage ablation withholds subtraces)."""
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
    return Episode(task, success, experience, stored, seconds)
