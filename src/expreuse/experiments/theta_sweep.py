"""Cascade's threshold: task success at every theta, assembled from the Whole, Subtraces and no-reuse
conditions.

    python -m expreuse.experiments.theta_sweep --whole w.jsonl --subtraces s.jsonl --none n.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from .. import config
from ..evaluation import success_rate


def episodes(path: str) -> dict[str, list[dict]]:
    by_task = collections.defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by_task[r["task_id"]].append(r)
    return by_task


def assemble(whole, subtraces, none, theta: float) -> list[float]:
    outcomes = []
    for task in sorted(set(whole) & set(subtraces) & set(none)):
        similarity = max((r["similarity"] or -1.0) for r in whole[task])
        if similarity >= theta:
            chosen = whole[task]
        elif any(r["route"] == "subtraces" for r in subtraces[task]):
            chosen = subtraces[task]
        else:
            chosen = none[task]
        outcomes.append(float(np.mean([r["success"] for r in chosen])))
    return outcomes


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--whole", required=True)
    p.add_argument("--subtraces", required=True)
    p.add_argument("--none", required=True)
    p.add_argument("--levels", type=float, nargs="+", default=config.THETA_LEVELS)
    args = p.parse_args(argv)
    whole, subtraces, none = episodes(args.whole), episodes(args.subtraces), episodes(args.none)
    base, base_se = success_rate([float(np.mean([r["success"] for r in v])) for v in none.values()])
    print(f"no reuse: {base:.1f} ± {base_se:.1f}")
    print("theta  success    se  whole-routed")
    for theta in args.levels:
        outcomes = assemble(whole, subtraces, none, theta)
        rate, se = success_rate(outcomes, config.BOOTSTRAP_RESAMPLES, config.SEEDS["bootstrap"])
        routed = np.mean([max((r["similarity"] or -1.0) for r in whole[t]) >= theta for t in whole])
        print(f"{theta:5.2f}  {rate:7.1f}  {se:4.1f}  {100 * routed:5.1f}%")


if __name__ == "__main__":
    main()
