# Sample of the stores

A small subset of each application's store in the data model of Section 3.1: a few tasks per application, their
traces from an LLM producer (Gemma-26B) and from a trained policy (DAgger on ALFWorld, the imitation policy on
ScienceWorld), successful and failed, segmented by the applications' own rules. The task lists of the experiments
are in `tasks/` (see `tasks/README.md`).

| File | Content |
|---|---|
| `operators.json`, `operators.csv` | the application's operator library: canonical name, the form the application writes it in, signature (roles), description, example binding |
| `traces.jsonl` | one stored trace per line in the format of `expreuse.io` (task, trace, steps, subtraces); `expreuse.io.read_jsonl` loads it, recomputing goal embeddings with the given embedder |
| `goal_embeddings.json` | the all-mpnet-base-v2 embedding of each task's goal (768 floats), as used in the paper |
| `tasks.csv`, `plans.csv`, `traces.csv`, `steps.csv`, `subtraces.csv` | the same data as the relations Task, Plan, Trace, Step and Subtrace (lists and objects as JSON strings) |

A subtrace is the steps of a trace that share a `subgoal_id`, the position of its operator in the task plan.
`finished` is 1 when the environment carried out the operator; `fits` is 1 when the subtrace contains only the
steps of its own operator. `plans.csv` gives each task's plan with target bindings: on ALFWorld those of the
game's expert PDDL plan (`bindings="pddl"`; the experiments bind from a room scan instead), on ScienceWorld those read from the task description (the room scan adds the rest at run
time).

```python
from expreuse.embedding import HashEmbedder
from expreuse.io import read_jsonl
from expreuse.stores import MemoryStore

store = MemoryStore()
for item in read_jsonl("data/alfworld/traces.jsonl", HashEmbedder(64)):
    store.insert(item)
```
