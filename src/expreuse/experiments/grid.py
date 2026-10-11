"""The conditions of every experiment, as data: one `Condition` is one result file.

    python -m expreuse.experiments.reproduce same_application --list
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .. import config

LLMS = ("gemma", "ministral")
POLICY = {"alfworld": "dagger", "scienceworld": "imitation"}
OTHER = {"alfworld": "scienceworld", "scienceworld": "alfworld"}
MODES = ("none", "whole", "subtraces", "cascade")
QUERIES = ("whole", "subtraces", "cascade")
PLAN_SETTINGS = (False, True)


@dataclass(frozen=True)
class Condition:
    experiment: str
    application: str
    consumer: str
    mode: str = "none"
    source: str | None = None
    cross_application: str | None = None
    hide_plan: bool = False
    ground: bool = True
    theta: float | None = None
    k: int = config.K
    withhold: float = 0.0
    schedule: str = "all"
    tasks: str | None = None
    bindings: str | None = None
    collect: bool = False

    @property
    def store_consumer(self) -> str:
        return self.source or self.consumer

    @property
    def store_application(self) -> str:
        return self.cross_application or self.application

    @property
    def producer(self) -> str:
        """Whose traces the query asks for."""
        return config.CONSUMERS[self.store_consumer]["producer"]

    def store_file(self, stores: str | Path) -> Path:
        return Path(stores) / f"{self.store_application}_{self.store_consumer}.jsonl"

    def own_store_file(self, stores: str | Path) -> Path:
        return Path(stores) / f"{self.application}_{self.consumer}.jsonl"

    def task_list(self) -> str:
        return self.tasks or config.APPLICATIONS[self.application]["tasks"]

    def name(self) -> str:
        if self.collect:
            return f"{self.application}_{self.consumer}_collect"
        parts = [self.application, self.consumer, self.mode]
        if self.source or self.cross_application:
            parts.append("from_" + (f"{self.cross_application}_" if self.cross_application else "") + self.store_consumer)
        if self.hide_plan:
            parts.append("noplan")
        if not self.ground and self.mode != "none" and not self.cross_application:
            parts.append("ungrounded")
        if self.theta is not None:
            parts.append(f"theta{self.theta:g}")
        if self.k != config.K:
            parts.append(f"k{self.k}")
        if self.withhold:
            parts.append(f"withhold{self.withhold:g}")
        if self.bindings:
            parts.append(f"bindings_{self.bindings}")
        if self.tasks:
            parts.append(Path(self.tasks).stem.removeprefix(self.application + "_"))
        return "_".join(parts)


def collect_stores() -> Iterator[Condition]:
    """Every producer solves the store task list once, without experience and without being shown the plan
    (the plan only segments the trace it leaves)."""
    for app in config.APPLICATIONS:
        for consumer in LLMS + (POLICY[app],):
            yield Condition("collect_stores", app, consumer, "none", hide_plan=True,
                            tasks=config.APPLICATIONS[app]["store_tasks"], collect=True)


def expert_store() -> Iterator[Condition]:
    """The base store of the system benchmarks."""
    yield Condition("expert_store", "alfworld", "expert", "none", hide_plan=True, tasks="all", bindings="none",
                    collect=True)


def same_application() -> Iterator[Condition]:
    """Each consumer with its own store: no reuse, Whole, Subtraces, Cascade."""
    for app in config.APPLICATIONS:
        for consumer in LLMS:
            for hide in PLAN_SETTINGS:
                for mode in MODES:
                    yield Condition("same_application", app, consumer, mode, hide_plan=hide)
        for mode in MODES:
            yield Condition("same_application", app, POLICY[app], mode)


def cross_agent() -> Iterator[Condition]:
    """A consumer reads the store of a producer of the other agent type."""
    for app in config.APPLICATIONS:
        policy = POLICY[app]
        for mode in QUERIES:
            for hide in PLAN_SETTINGS:
                yield Condition("cross_agent", app, "gemma", mode, source=policy, hide_plan=hide)
            yield Condition("cross_agent", app, policy, mode, source="gemma")


def cross_application() -> Iterator[Condition]:
    """A consumer of one application reads traces of the other, ungrounded, on the tasks with shared operators."""
    for app in config.APPLICATIONS:
        tasks, other, policy = config.APPLICATIONS[app]["cross_application_tasks"], OTHER[app], POLICY[app]
        for hide in PLAN_SETTINGS:
            yield Condition("cross_application", app, "gemma", "none", hide_plan=hide, tasks=tasks)
        yield Condition("cross_application", app, policy, "none", tasks=tasks)
        for mode in ("whole", "subtraces"):
            for hide in PLAN_SETTINGS:
                yield Condition("cross_application", app, "gemma", mode, source="gemma", cross_application=other,
                                hide_plan=hide, tasks=tasks)
            for source in ("gemma", POLICY[other]):
                yield Condition("cross_application", app, policy, mode, source=source, cross_application=other,
                                tasks=tasks)


def theta() -> Iterator[Condition]:
    """Cascade's threshold: Gemma plays Cascade at every level, with the plan shown and hidden."""
    for app in config.APPLICATIONS:
        for hide in PLAN_SETTINGS:
            for level in config.THETA_LEVELS:
                yield Condition("theta", app, "gemma", "cascade", hide_plan=hide, theta=level)


def depth() -> Iterator[Condition]:
    """Retrieval depth K beyond config.K (whose conditions are same_application's)."""
    for k in config.K_LEVELS["llm"]:
        if k != config.K:
            for mode in ("whole", "subtraces"):
                yield Condition("depth", "scienceworld", "ministral", mode, k=k)
    for app in config.APPLICATIONS:
        for k in config.K_LEVELS["policy"]:
            if k != config.K:
                for mode in ("whole", "subtraces"):
                    yield Condition("depth", app, POLICY[app], mode, k=k)


def coverage() -> Iterator[Condition]:
    """Partial subtrace coverage: a share of each Subtraces answer withheld, for Gemma (plan shown and hidden)
    and the policy."""
    for app in config.APPLICATIONS:
        schedule = "all" if app == "alfworld" else "nested"
        for fraction in config.COVERAGE_LEVELS:
            if fraction:
                for hide in PLAN_SETTINGS:
                    yield Condition("coverage", app, "gemma", "subtraces", hide_plan=hide, withhold=fraction,
                                    schedule=schedule)
                yield Condition("coverage", app, POLICY[app], "subtraces", withhold=fraction, schedule=schedule)


def grounding() -> Iterator[Condition]:
    """Grounding switched off on ALFWorld, for the language models (plan shown and hidden) and the policy; the
    grounded runs are those of same_application."""
    for mode in QUERIES:
        for consumer in LLMS:
            for hide in PLAN_SETTINGS:
                yield Condition("grounding", "alfworld", consumer, mode, hide_plan=hide, ground=False)
        yield Condition("grounding", "alfworld", POLICY["alfworld"], mode, ground=False)


EXPERIMENTS = {f.__name__: f for f in (collect_stores, expert_store, same_application, cross_agent,
                                       cross_application, theta, depth, coverage, grounding)}
