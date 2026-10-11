"""Run one condition: a consumer on a task list, with or without a reuse query, repeated `runs` times.

    python -m expreuse.experiments.run --application alfworld --consumer gemma --mode cascade
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

from .. import config
from ..ablations import all_removals, nested_removals, withhold
from ..embedding import HashEmbedder, SentenceTransformerEmbedder
from ..envs import load_application
from ..io import read_jsonl, to_json
from ..runner import ReuseConfig, retrieve, run_task
from ..stores import REALIZATIONS, make_store
from .grid import MODES, Condition


@dataclass(frozen=True)
class Settings:
    stores: str = "stores"
    store_files: tuple[str, ...] = ()
    realization: str = "memory"
    dsn: str | None = None
    url: str | None = None
    embedder: str = "mpnet"
    policy_path: str | None = None
    max_tasks: int | None = None
    runs: int | None = None


def read_list(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip() and not line.startswith("#")]


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def make_embedder(name: str):
    return SentenceTransformerEmbedder(config.EMBEDDER) if name == "mpnet" else HashEmbedder(768)


def open_application(cond: Condition):
    if cond.application == "alfworld":
        spec = config.CONSUMERS[cond.consumer]
        return load_application("alfworld", split=spec.get("split") or config.APPLICATIONS["alfworld"]["split"],
                                bindings=cond.bindings or config.APPLICATIONS["alfworld"]["bindings"])
    return load_application("scienceworld")


def make_consumer(name: str, app, embedder, with_plan: bool, policy_path: str | None = None):
    """The consumer and, for a language model, its client (whose `usage` counts tokens)."""
    spec = config.CONSUMERS[name]
    if spec["kind"] == "llm":
        from ..consumers.llm import BedrockChatModel, LLMConsumer, OpenAIChatModel, openrouter_options

        settings = dict(temperature=config.LLM["temperature"], max_tokens=config.LLM["max_tokens"][app.name],
                        timeout=config.LLM["timeout_seconds"][app.name], retries=config.LLM["retries"][app.name])
        if spec["serving"] == "bedrock":
            model = BedrockChatModel(spec["bedrock_model_id"], **settings)
        else:
            model = OpenAIChatModel(spec["model"], base_url=spec["base_url"],
                                    api_key=os.environ.get(config.LLM["api_key_variable"]["openrouter"]),
                                    extra=openrouter_options(spec["providers"][app.name], config.LLM["reasoning"]),
                                    **settings)
        return LLMConsumer(model, app.system_prompt, with_plan=with_plan), model
    if spec["kind"] == "expert":
        from ..consumers.expert import ExpertConsumer

        return ExpertConsumer(app), None
    from ..consumers.policy import ShapedPolicy

    if name == "dagger":
        from ..policies.alfworld_dagger import DaggerPolicy

        base = DaggerPolicy(policy_path or spec["checkpoint"])
    else:
        from ..policies.scienceworld_imitation import ImitationPolicy

        base = ImitationPolicy(policy_path or spec["directory"], app)
    return ShapedPolicy(base, embedder, m=config.M, lam=config.LAMBDA, operator_of=app.operator_of), None


def open_store(files, embedder, settings: Settings):
    if not files:
        return None
    store = make_store(settings.realization, embedder.dim, dsn=settings.dsn, url=settings.url)
    store.setup()
    for path in files:
        store.insert_many(read_jsonl(path, embedder))
    return store


class Withholding:
    """The coverage ablation's edit of a query answer: the subtraces at `positions` are withheld."""

    def __init__(self, positions: tuple[int, ...]):
        self.positions = positions

    def __call__(self, experience):
        return withhold(experience, self.positions)


def withholdings(cond: Condition, app, task, store, reuse, embedder) -> list[Withholding | None]:
    """The removals to play for a task: none, every subset of the level's size, or the task's nested ordering."""
    if not cond.withhold:
        return [None]
    task = replace(task, embedding=embedder.encode([task.goal])[0])
    experience, _ = retrieve(store, task, app.plan(task), reuse)
    if cond.schedule == "nested":
        return [Withholding(nested_removals(experience, cond.withhold, task.task_id))]
    return [Withholding(tuple(s)) for s in all_removals(experience, cond.withhold) or [()]]


def usage_delta(model, before: dict | None) -> dict | None:
    return None if model is None else {k: model.usage[k] - before[k] for k in model.usage}


def execute(cond: Condition, out: Path, settings: Settings = Settings()) -> Path:
    """Play (or resume) one condition and write its episodes to `out`."""
    spec = config.CONSUMERS[cond.consumer]
    if spec.get("application", cond.application) != cond.application:
        raise ValueError(f"{cond.consumer} only plays {spec['application']}")
    embedder = make_embedder(settings.embedder)
    app = open_application(cond)
    files = settings.store_files or ([str(cond.store_file(settings.stores))] if cond.mode != "none" else [])
    store = open_store(files, embedder, settings)
    reuse = ReuseConfig(mode=cond.mode, producer=cond.producer,
                        applications=(cond.cross_application,) if cond.cross_application else (),
                        theta=config.THETA[cond.application] if cond.theta is None else cond.theta,
                        k=cond.k, ground=config.GROUND and cond.ground)
    consumer, model = make_consumer(cond.consumer, app, embedder, not cond.hide_plan, settings.policy_path)
    store_as = spec["producer"] if cond.collect else None
    tasks = app.task_ids() if cond.task_list() == "all" else read_list(cond.task_list())
    tasks = tasks[: settings.max_tasks]
    runs = settings.runs or (config.RUNS if spec["kind"] == "llm" and not cond.collect else 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["task_id"], r["run"], tuple(r.get("withheld") or ())) for r in read_records(out)}
    with open(out, "a") as f:
        for run in range(runs):
            for task_id in tasks:
                task = app.task(task_id)
                for shape in withholdings(cond, app, task, store, reuse, embedder):
                    withheld = () if shape is None else tuple(shape.positions)
                    if (task_id, run, withheld) in done:
                        continue
                    before = dict(model.usage) if model else None
                    episode = run_task(app, consumer, task, embedder, store, reuse, store_as=store_as, shape=shape)
                    experience = episode.experience
                    record = {"task_id": task_id, "run": run, "success": episode.success,
                              "route": experience.route, "similarity": experience.similarity,
                              "plan": [p.operator for p in episode.plan],
                              "covered": [i for i, hits in enumerate(experience.subtraces) if hits],
                              "returned_steps": len(experience.steps()), "steps": episode.stored.length,
                              "seconds": {k: round(v, 3) for k, v in episode.seconds.items()},
                              "tokens": usage_delta(model, before), "withheld": list(withheld),
                              "condition": cond.name()}
                    if store_as:
                        with open(cond.own_store_file(settings.stores), "a") as g:
                            g.write(json.dumps(to_json(episode.stored)) + "\n")
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    done.add((task_id, run, withheld))
                    print(f"{cond.name()} run {run} {task_id} {'success' if episode.success else 'failure'} "
                          f"route={experience.route} steps={episode.stored.length}", flush=True)
    if store is not None:
        store.close()
    if hasattr(app, "close"):
        app.close()
    return out


def add_settings(p: argparse.ArgumentParser) -> None:
    p.add_argument("--stores", default="stores", help="directory of the producers' trace files")
    p.add_argument("--store", action="append", default=[], help="trace file read instead of the condition's store")
    p.add_argument("--realization", default="memory", choices=list(REALIZATIONS))
    p.add_argument("--dsn", help="PostgreSQL connection string (relational, jsonb)")
    p.add_argument("--url", help="Qdrant endpoint")
    p.add_argument("--embedder", default="mpnet", choices=["mpnet", "hash"])
    p.add_argument("--policy-path", help="checkpoint or model directory of a policy consumer")
    p.add_argument("--max-tasks", type=int, help="only the first tasks of the list")
    p.add_argument("--runs", type=int, help="repetitions (default: config.RUNS for a language model, else 1)")


def settings_from(args) -> Settings:
    return Settings(args.stores, tuple(args.store), args.realization, args.dsn, args.url, args.embedder,
                    args.policy_path, args.max_tasks, args.runs)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--application", required=True, choices=list(config.APPLICATIONS))
    p.add_argument("--consumer", required=True, choices=list(config.CONSUMERS))
    p.add_argument("--mode", default="none", choices=list(MODES))
    p.add_argument("--source", choices=list(config.CONSUMERS), help="read this consumer's store")
    p.add_argument("--cross-application", choices=list(config.APPLICATIONS),
                   help="read that application's traces; such answers are not grounded")
    p.add_argument("--hide-plan", action="store_true", help="the query uses the plan but the consumer is not shown it")
    p.add_argument("--no-ground", action="store_true", help="show stored entities instead of the task's")
    p.add_argument("--theta", type=float, help="Cascade's threshold (default: config.THETA of the application)")
    p.add_argument("--k", type=int, default=config.K, help="answers per query")
    p.add_argument("--withhold", type=float, default=0.0, help="share of the answer's subtraces withheld")
    p.add_argument("--withhold-schedule", default="all", choices=["all", "nested"])
    p.add_argument("--tasks", help="task list (default: the application's evaluation list); 'all' plays every "
                                  "task of the opened split")
    p.add_argument("--bindings", choices=["scan", "pddl", "none"],
                   help="ALFWorld: where the plan's target entities come from (default: config, the scripted scan)")
    p.add_argument("--collect", action="store_true", help="store the traces as the consumer's own")
    p.add_argument("--out", help="JSONL of episodes (default: results/<condition>.jsonl)")
    add_settings(p)
    return p.parse_args(argv)


def condition_from(args) -> Condition:
    return Condition("run", args.application, args.consumer, args.mode, args.source, args.cross_application,
                     args.hide_plan, not args.no_ground, args.theta, args.k, args.withhold, args.withhold_schedule,
                     args.tasks, args.bindings, args.collect)


def main(argv=None):
    args = parse(argv)
    cond = condition_from(args)
    execute(cond, Path(args.out or f"results/{cond.name()}.jsonl"), settings_from(args))


if __name__ == "__main__":
    main()
