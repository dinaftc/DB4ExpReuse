"""One experimental condition: a consumer on a task list, with or without a reuse query, repeated `runs` times.

    python -m expreuse.experiments.run --application alfworld --consumer gemma --mode cascade \\
        --store stores/alfworld_gemma.jsonl --out results/alfworld_gemma_cascade.jsonl

Writes one JSON line per episode (task, run, success, route taken, best whole-trace similarity, steps, seconds per
component, tokens) and skips the episodes already in the output, so an interrupted condition resumes. Producers'
stores are collected with `--mode none --store-as --store-out <jsonl>`. The settings not given on the command line
come from `expreuse.config`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from .. import config
from ..ablations import all_removals, nested_removals, withhold
from ..embedding import HashEmbedder, SentenceTransformerEmbedder
from ..envs import load_application
from ..io import read_jsonl, to_json
from ..model import ReuseRequest
from ..runner import ReuseConfig, retrieve, run_task
from ..stores import REALIZATIONS, make_store


def read_list(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip() and not line.startswith("#")]


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def open_application(args):
    if args.application == "alfworld":
        split = args.split or config.CONSUMERS[args.consumer].get("split") or config.APPLICATIONS["alfworld"]["split"]
        return load_application("alfworld", split=split,
                                seed=config.SEEDS["alfworld_environment"],
                                bindings=args.bindings or config.APPLICATIONS["alfworld"]["bindings"])
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


def open_store(args, embedder):
    if not args.store:
        return None
    store = make_store(args.realization, embedder.dim, dsn=args.dsn, url=args.url)
    store.setup()
    for path in args.store:
        for item in read_jsonl(path, embedder):
            store.insert(item)
    return store


def best_similarity(store, episode, reuse: ReuseConfig) -> float | None:
    """Similarity of the most similar stored successful trace to the task: what Cascade compares with theta."""
    if store is None or reuse.mode == "none":
        return None
    operators = tuple(s.operator for s in sorted(episode.stored.subtraces, key=lambda s: s.subgoal_id))
    request = ReuseRequest(episode.task.embedding, reuse.producer,
                           reuse.applications or (episode.task.application,), operators, 1)
    hits = store.whole(request)
    return round(hits[0].similarity, 4) if hits else None


class Withholding:
    """The coverage ablation's edit of a query answer: the subtraces at `positions` are withheld."""

    def __init__(self, positions: tuple[int, ...]):
        self.positions = positions

    def __call__(self, experience):
        return withhold(experience, self.positions)


def withholdings(args, app, task, store, reuse, embedder) -> list[Withholding | None]:
    """The removals to play for a task: none, every subset of the level's size, or the task's nested ordering.
    The query is run once here to see which plan positions the answer covers."""
    if not args.withhold:
        return [None]
    task = replace(task, embedding=embedder.encode([task.goal])[0])
    experience, _ = retrieve(store, task, app.plan(task), reuse)
    if args.withhold_schedule == "nested":
        return [Withholding(nested_removals(experience, args.withhold, task.task_id))]
    return [Withholding(tuple(s)) for s in all_removals(experience, args.withhold) or [()]]


def usage_delta(model, before: dict | None) -> dict | None:
    if model is None:
        return None
    return {k: model.usage[k] - before[k] for k in model.usage}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--application", required=True, choices=list(config.APPLICATIONS))
    p.add_argument("--consumer", required=True, choices=list(config.CONSUMERS))
    p.add_argument("--mode", default="none", choices=["none", "whole", "subtraces", "cascade"])
    p.add_argument("--producer", help="whose traces the query asks for (default: the consumer's own)")
    p.add_argument("--store", action="append", default=[], help="JSONL of stored traces (repeatable)")
    p.add_argument("--realization", default="memory", choices=list(REALIZATIONS))
    p.add_argument("--dsn", help="PostgreSQL connection string (relational, jsonb)")
    p.add_argument("--url", help="Qdrant endpoint")
    p.add_argument("--cross-application", choices=list(config.APPLICATIONS),
                   help="read that application's traces instead; such answers are not grounded")
    p.add_argument("--theta", type=float, help="Cascade's threshold (default: config.THETA of the application)")
    p.add_argument("--k", type=int, default=config.K, help="answers per query")
    p.add_argument("--no-ground", action="store_true", help="show stored entities instead of the task's")
    p.add_argument("--bindings", choices=["scan", "pddl", "none"],
                   help="ALFWorld: where the plan's target entities come from (default: config, the scripted scan)")
    p.add_argument("--hide-plan", action="store_true", help="the query uses the plan but the consumer is not shown it")
    p.add_argument("--tasks", help="task list (default: the application's evaluation list); 'all' plays every task "
                                  "of the opened split")
    p.add_argument("--max-tasks", type=int, help="only the first tasks of the list")
    p.add_argument("--split", help="ALFWorld split the application lists (the task list decides what is played)")
    p.add_argument("--runs", type=int, help="repetitions (default: config.RUNS for a language model, 1 for a policy)")
    p.add_argument("--withhold", type=float, default=0.0, help="share of the answer's subtraces withheld")
    p.add_argument("--withhold-schedule", default="all", choices=["all", "nested"],
                   help="every subset of that size, or one nested random ordering per task")
    p.add_argument("--store-as", nargs="?", const="", help="store the traces under this producer (default: own name)")
    p.add_argument("--store-out", help="JSONL the stored traces are appended to")
    p.add_argument("--policy-path", help="checkpoint or model directory of a policy consumer")
    p.add_argument("--embedder", default="mpnet", choices=["mpnet", "hash"])
    p.add_argument("--out", required=True, help="JSONL of episodes")
    return p.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    spec = config.CONSUMERS[args.consumer]
    if spec.get("application", args.application) != args.application:
        sys.exit(f"{args.consumer} only plays {spec['application']}")
    embedder = SentenceTransformerEmbedder(config.EMBEDDER) if args.embedder == "mpnet" else HashEmbedder(768)
    app = open_application(args)
    store = open_store(args, embedder)
    reuse = ReuseConfig(mode=args.mode, producer=args.producer or spec["producer"],
                        applications=(args.cross_application,) if args.cross_application else (),
                        theta=config.THETA[args.application] if args.theta is None else args.theta,
                        k=args.k, ground=config.GROUND and not args.no_ground)
    consumer, model = make_consumer(args.consumer, app, embedder, not args.hide_plan, args.policy_path)
    store_as = (spec["producer"] if args.store_as == "" else args.store_as) if args.store_as is not None else None
    tasks = app.task_ids() if args.tasks == "all" else read_list(args.tasks or config.APPLICATIONS[args.application]["tasks"])
    tasks = tasks[: args.max_tasks]
    runs = args.runs or (config.RUNS if spec["kind"] == "llm" else 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["task_id"], r["run"], tuple(r.get("withheld") or ())) for r in read_records(out)}
    settings = {k: v for k, v in vars(args).items() if k != "out"}
    with open(out, "a") as f:
        for run in range(runs):
            for task_id in tasks:
                task = app.task(task_id)
                for shape in withholdings(args, app, task, store, reuse, embedder):
                    withheld = () if shape is None else tuple(shape.positions)
                    if (task_id, run, withheld) in done:
                        continue
                    before = dict(model.usage) if model else None
                    episode = run_task(app, consumer, task, embedder, store, reuse, store_as=store_as, shape=shape)
                    record = {"task_id": task_id, "run": run, "success": episode.success,
                              "route": episode.experience.route,
                              "similarity": best_similarity(store, episode, reuse),
                              "steps": episode.stored.length,
                              "seconds": {k: round(v, 3) for k, v in episode.seconds.items()},
                              "tokens": usage_delta(model, before), "withheld": list(withheld),
                              "settings": settings}
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    done.add((task_id, run, withheld))
                    if args.store_out and store_as:
                        with open(args.store_out, "a") as g:
                            g.write(json.dumps(to_json(episode.stored)) + "\n")
                    print(f"run {run} {task_id} {'success' if episode.success else 'failure'} "
                          f"route={episode.experience.route} steps={episode.stored.length}", flush=True)
    if store is not None:
        store.close()


if __name__ == "__main__":
    main()
