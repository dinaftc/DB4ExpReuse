"""Task success with bootstrap standard errors, routes, steps, time and tokens of result files.

    python -m expreuse.experiments.summarize results/*.jsonl [--baseline results/none.jsonl] [--csv table.csv]

Runs (and, in the coverage ablation, the subsets of one task) are averaged per task before tasks are resampled.
With a baseline, the difference to it is paired by task, with wins and losses counted over tasks.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np

from .. import config
from ..evaluation import success_rate


def per_task(path: Path) -> dict[str, float]:
    outcomes = collections.defaultdict(list)
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            outcomes[r["task_id"]].append(float(r["success"]))
    return {t: float(np.mean(v)) for t, v in outcomes.items()}


def summarize(path: Path, baseline: dict[str, float] | None = None) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    tasks = per_task(path)
    rate, se = success_rate(list(tasks.values()), config.BOOTSTRAP_RESAMPLES, config.SEEDS["bootstrap"])
    routes = collections.Counter(r["route"] for r in records)
    seconds = {k: np.mean([r["seconds"].get(k, 0.0) for r in records]) for k in ("retrieval", "consumer", "environment")}
    tokens = [r["tokens"] for r in records if r.get("tokens")]
    row = {"file": path.name, "tasks": len(tasks), "episodes": len(records), "success": round(rate, 1), "se": round(se, 1),
           "whole": round(100 * routes["whole"] / len(records), 1), "subtraces": round(100 * routes["subtraces"] / len(records), 1),
           "steps": round(float(np.mean([r["steps"] for r in records])), 1),
           "retrieval_s": round(float(seconds["retrieval"]), 3), "consumer_s": round(float(seconds["consumer"]), 1),
           "environment_s": round(float(seconds["environment"]), 2),
           "prompt_tokens": round(float(np.mean([t["prompt_tokens"] for t in tokens])), 0) if tokens else None,
           "completion_tokens": round(float(np.mean([t["completion_tokens"] for t in tokens])), 0) if tokens else None}
    if baseline:
        common = sorted(set(tasks) & set(baseline))
        diff = np.array([tasks[t] - baseline[t] for t in common])
        rng = np.random.default_rng(config.SEEDS["bootstrap"])
        index = rng.integers(0, len(diff), size=(config.BOOTSTRAP_RESAMPLES, len(diff)))
        row.update({"diff": round(100 * diff.mean(), 1), "diff_se": round(100 * diff[index].mean(axis=1).std(ddof=1), 1),
                    "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum()), "paired_tasks": len(common)})
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--baseline", help="result file the differences are paired against")
    p.add_argument("--csv")
    args = p.parse_args(argv)
    baseline = per_task(Path(args.baseline)) if args.baseline else None
    rows = [summarize(Path(f), baseline) for f in args.files]
    columns = list(rows[0])
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    print("  ".join(c.rjust(widths[c]) for c in columns))
    for r in rows:
        print("  ".join(str(r.get(c, "")).rjust(widths[c]) for c in columns))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
