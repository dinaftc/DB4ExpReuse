"""Run the conditions of an experiment, or list them.

    python -m expreuse.experiments.reproduce same_application --list
    python -m expreuse.experiments.reproduce collect_stores
    python -m expreuse.experiments.reproduce theta               # Gemma at every theta; the policies assembled
    python -m expreuse.experiments.reproduce expert_store        # base store of the system benchmarks
    python -m expreuse.experiments.reproduce all

Results go to results/<experiment>/<condition>.jsonl and are summarized in results/<experiment>/summary.csv.
The producers' stores (`collect_stores`) must exist before the other experiments run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .. import config
from . import summarize, theta_sweep
from .grid import EXPERIMENTS, POLICY
from .run import add_settings, execute, settings_from

ORDER = ("collect_stores", "same_application", "cross_agent", "cross_application", "theta", "depth", "coverage",
         "grounding")


def assemble_theta(results: Path) -> None:
    """The policies' threshold curves, assembled exactly from their same_application results."""
    folder = results / "same_application"
    for app in config.APPLICATIONS:
        consumer = POLICY[app]
        files = {mode: folder / f"{app}_{consumer}_{mode}.jsonl" for mode in ("whole", "subtraces", "none")}
        if all(f.exists() for f in files.values()):
            print(f"== {app} {consumer}")
            theta_sweep.main(["--whole", str(files["whole"]), "--subtraces", str(files["subtraces"]),
                              "--none", str(files["none"])])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", choices=list(ORDER) + ["expert_store", "all"])
    p.add_argument("--list", action="store_true", help="print the conditions instead of running them")
    p.add_argument("--applications", nargs="+", choices=list(config.APPLICATIONS))
    p.add_argument("--consumers", nargs="+", choices=list(config.CONSUMERS))
    p.add_argument("--results", default="results")
    add_settings(p)
    args = p.parse_args(argv)
    results = Path(args.results)
    for name in (ORDER if args.experiment == "all" else (args.experiment,)):
        conditions = [c for c in EXPERIMENTS[name]()
                      if (not args.applications or c.application in args.applications)
                      and (not args.consumers or c.consumer in args.consumers)]
        if args.list:
            for c in conditions:
                print(f"{name}: {c.name()}")
            continue
        files = []
        for i, c in enumerate(conditions, 1):
            print(f"[{name} {i}/{len(conditions)}] {c.name()}", flush=True)
            files.append(str(execute(c, results / name / f"{c.name()}.jsonl", settings_from(args))))
        if files and not name.endswith("store") and name != "collect_stores":
            summarize.main(files + ["--csv", str(results / name / "summary.csv")])
        if name == "theta":
            assemble_theta(results)


if __name__ == "__main__":
    main()
