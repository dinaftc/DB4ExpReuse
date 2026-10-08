# DB4ExpReuse

Code for *A Data Abstraction for Agent Experience Reuse in Sequential Decision Making*.

Agents solving sequential decision-making tasks leave interaction traces. This
library stores those traces, and the operator-level subtraces they decompose
into, under one logical data model, and expresses experience reuse as queries
over it:

- **Whole** returns complete successful traces whose task goal is most similar to the target goal.
- **Subtraces** returns, for each operator of the target plan, the most similar finished subtraces.
- **Cascade** picks between the two at query time, using the similarity of the best complete trace.

> **Status.** The framework (data model, reuse queries, grounding, consumers, the three physical realizations,
> the system benchmarks) and the two applications (ALFWorld and ScienceWorld: environment wrappers, operator
> libraries, task plans, trace segmentation rules). `data/` holds a sample of each application's store. The trained
> policies and the scripts that produce the task-success tables are not included.

## Install

```bash
pip install -e ".[all]"        # or pick extras: postgres, qdrant, embeddings, llm, dev, alfworld, scienceworld
```

The PostgreSQL realizations need PostgreSQL 13+ with the
[pgvector](https://github.com/pgvector/pgvector) extension. The Qdrant
realization runs in-process when no server URL is given.

## Quick start

```python
from expreuse import MemoryStore, ReuseRequest
from expreuse.embedding import SentenceTransformerEmbedder
from expreuse.synthetic import make_traces

embedder = SentenceTransformerEmbedder()          # all-mpnet-base-v2
store = MemoryStore()                             # or make_store("relational", embedder.dim, dsn=...)
store.setup()
for item in make_traces(200, producer="demo", embedder=embedder):
    store.insert(item)

request = ReuseRequest(
    embedding=embedder.encode(["put a hot cup in cabinet"])[0],
    producer="demo",                # whose experience to reuse
    applications=("alfworld",),     # eligible stored tasks
    plan=("take", "heat", "put"),   # operators of the target plan
)
experience = store.cascade(request, theta=0.6)
print(experience.route)             # "whole", "subtraces" or "none"
```

`expreuse.runner.run_task` runs the full loop of Figure 1 for an application:
plan, reuse query, grounding, acting, segmentation, and storing the new trace.

## Where things are

| Paper | Code |
|---|---|
| Data model (§3.1) | `src/expreuse/model.py` |
| Operators and signatures (Table 2) | `src/expreuse/operators.py` |
| Segmenting traces into subtraces (§3.2) | `src/expreuse/segmentation.py` |
| Whole, Subtraces, Cascade (§3.3) | `src/expreuse/stores/base.py`, reference semantics in `stores/memory.py` |
| Grounding (§3.3) | `src/expreuse/grounding.py` |
| LLM consumer | `src/expreuse/consumers/llm.py` |
| Policy consumer, Q_ITD shaping | `src/expreuse/consumers/policy.py` |
| Normalized relational realization (§4.5) | `src/expreuse/stores/relational.py`, `relational.sql` |
| JSONB realization | `src/expreuse/stores/jsonb.py`, `jsonb.sql` |
| Qdrant realization | `src/expreuse/stores/qdrant.py` |
| Bootstrap standard errors (§4.1) | `src/expreuse/evaluation.py` |
| Scaling, latency, concurrency, approximate search (§4.5) | `src/expreuse/bench/` |
| Interface an application implements | `src/expreuse/envs/base.py` |
| ALFWorld: operators, plan from the goal, scan, segmentation rules | `src/expreuse/envs/alfworld.py` |
| ScienceWorld: operators, plan per task type from gold traces, segmentation rules | `src/expreuse/envs/scienceworld.py`, `envs/scienceworld_plans.json` |
| Sample of the stores (operators, tasks, plans, traces, steps, subtraces) | `data/alfworld/`, `data/scienceworld/` |

## Applications

`expreuse.envs.load_application("alfworld")` and `load_application("scienceworld")` open the two applications of
the paper (install the `alfworld` or `scienceworld` extra; ALFWorld's data goes to `$ALFWORLD_DATA`, ScienceWorld
needs Java). Each one lists its tasks (`task_ids()`), reads a task's goal (`task(task_id)`), derives the task plan
(`plan(task)`) and runs the environment loop for `run_task`.

- **ALFWorld**: the plan's operators follow from the goal (`goal_operators`); a scripted scan of the room binds
  them to the objects, destination and appliances it finds. Subtraces close at the milestone actions take, put,
  heat, cool, clean and use.
- **ScienceWorld**: one plan per task type, derived from the benchmark's gold traces of training tasks
  (`envs/scienceworld_plans.json`); the type of a new task is recognised from its description. Subtraces close at
  the manipulation commands (focus on, pick up, move, pour, mix, activate, deactivate, use, connect, read), a pick
  up or move being complete when the simulator confirms it.

```python
from expreuse import ReuseConfig, run_task
from expreuse.consumers import LLMConsumer, OpenAIChatModel
from expreuse.embedding import SentenceTransformerEmbedder
from expreuse.envs import load_application
from expreuse.io import read_jsonl
from expreuse.stores import MemoryStore

app, embedder = load_application("alfworld"), SentenceTransformerEmbedder()
store = MemoryStore()
for item in read_jsonl("data/alfworld/traces.jsonl", embedder):
    store.insert(item)
consumer = LLMConsumer(OpenAIChatModel("google/gemma-4-26b-a4b-it", base_url="https://openrouter.ai/api/v1"),
                       app.system_prompt, with_plan=True)
task = app.task(app.task_ids()[0])
episode = run_task(app, consumer, task, embedder, store,
                   ReuseConfig(mode="cascade", producer="Gemma-26B", theta=0.6), store_as="Gemma-26B")
print(episode.success, episode.experience.route)
```

`data/README.md` describes the sample stores: for each application, the operator library with signatures
(Table 2) and a few tasks with their plans, traces, steps and subtraces, as JSON Lines and CSV.

## Adding an application

Implement `expreuse.envs.Application`: the environment loop (`reset`, `step`,
`admissible_actions`, `max_steps`), `plan(task)` returning the task plan with
its target bindings, and a `labeler` that tells the segmenter which operator an
action realizes and when a subgoal is completed. Consumers and stores need no
change.

## Physical realizations

All realizations implement `ExperienceStore` and return the same experience
for the same request; the test suite checks each one against the in-memory
reference.

- **Relational**: `goal`, `task`, `trace`, `subtrace`, `step`. The logical
  `Task` relation is split into `task` and `goal` so each distinct goal is
  embedded and compared once.
- **JSONB**: one `trace_doc` row per trace with its subtraces and steps in a
  document. Cascade reads only the goal similarity to choose a route and
  loads documents afterwards.
- **Qdrant**: goal embeddings in one collection; traces and subtraces in
  payload-only collections fetched by indexed keys.

Goal search is exact by default. `create_goal_index()` on the PostgreSQL
stores adds an HNSW index.

## Benchmarks

Each script prints one JSON line per measurement. Without `--traces FILE.jsonl`
they run on synthetic traces (`expreuse.synthetic`), which exercise the system
but are not the traces used in the paper.

```bash
DSN=postgresql:///expreuse

# Retrieval and insertion latency as stored subtraces grow (Figure 3a, Table 6)
python -m expreuse.bench.latency --realization relational --dsn $DSN --sizes 2000,13000,100000,1000000
python -m expreuse.bench.latency --realization jsonb      --dsn $DSN
python -m expreuse.bench.latency --realization qdrant     --url http://localhost:6333

# Throughput with concurrent clients, query-only and 90/10 query/insert (Figure 3b)
python -m expreuse.bench.concurrency --dsn $DSN --clients 1,4,16,32,64
python -m expreuse.bench.concurrency --dsn $DSN --clients 1,4,16,32,64 --insert-fraction 0.1

# Exact vs. HNSW goal lookup
python -m expreuse.bench.ann --dsn $DSN --goals 1000000 --ef 40,400
```

Use `--embedder mpnet` for all-mpnet-base-v2 embeddings; the default is a fast
hashing embedder of the same dimension.

## Tests

```bash
pytest                                                    # in-memory and Qdrant
EXPREUSE_TEST_DSN=postgresql:///expreuse_test pytest      # plus PostgreSQL realizations
```

The PostgreSQL tests create and drop the schemas `expreuse_test_relational`
and `expreuse_test_jsonb` in that database.

## Trace files

`expreuse.io` reads and writes stored traces as JSON Lines, one trace per line:

```json
{"task_id": "t1", "application": "alfworld", "goal": "put a hot mug in cabinet",
 "trace_id": "tr1", "producer": "gemma", "success": true,
 "steps": [{"subgoal_id": 0, "action": "take mug 2 from countertop 1", "observation": "..."}],
 "subtraces": [{"subgoal_id": 0, "operator": "take",
                "binding": {"object": "mug 2", "source": "countertop 1"},
                "finished": true, "fits": true}]}
```
