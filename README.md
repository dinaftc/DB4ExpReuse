# DB4ExpReuse

Code for *A Data Abstraction for Agent Experience Reuse in Sequential Decision Making*.

Agents solving sequential decision-making tasks leave interaction traces. This
library stores those traces, and the operator-level subtraces they decompose
into, under one logical data model, and expresses experience reuse as queries
over it:

- **Whole** returns complete successful traces whose task goal is most similar to the target goal.
- **Subtraces** returns, for each operator of the target plan, the most similar finished subtraces.
- **Cascade** picks between the two at query time, using the similarity of the best complete trace.

> **Contents.** The framework (data model, reuse queries, grounding, consumers, the three physical realizations,
> the system benchmarks); the two applications (ALFWorld and ScienceWorld: environment wrappers, operator
> libraries, task plans, trace segmentation rules); the two policy consumers with their training code; the
> scripts of every experiment with their settings (`experiments/`, `src/expreuse/config.py`) and task lists
> (`data/tasks/`); and in `data/` a sample of each application's store. The full stores, the trained policies and
> the API keys are not included: `experiments/collect_stores.sh` rebuilds the stores by running the producers.

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
| Partial subtrace coverage (withholding) | `src/expreuse/ablations.py` |
| Settings: models, serving, seeds, θ, K, m, λ, budgets | `src/expreuse/config.py` |
| Running a condition, summaries, θ sweep | `src/expreuse/experiments/` |
| The experiments, one script each | `experiments/*.sh` |
| Task lists (evaluation, store, calibration, cross-application) | `data/tasks/` |
| Scaling, latency, concurrency, approximate search (§4.5) | `src/expreuse/bench/` |
| Interface an application implements | `src/expreuse/envs/base.py` |
| ALFWorld: operators, plan and bindings from the game's PDDL, segmentation rules | `src/expreuse/envs/alfworld.py` |
| ScienceWorld: operators (with heat and cool), plan per task type from gold traces, room scan, segmentation rules | `src/expreuse/envs/scienceworld.py`, `envs/scienceworld_plans.json` |
| Sample of the stores (operators, tasks, plans, traces, steps, subtraces) | `data/alfworld/`, `data/scienceworld/` |
| TextDAgger policy on ALFWorld and its training | `src/expreuse/policies/alfworld_dagger.py`, `alfworld_dagger_train.py`, `alfworld_dagger.yaml` |
| Imitation policy on ScienceWorld and its training | `src/expreuse/policies/scienceworld_imitation.py` |

## Applications

`expreuse.envs.load_application("alfworld")` and `load_application("scienceworld")` open the two applications of
the paper (install the `alfworld` or `scienceworld` extra; ALFWorld's data goes to `$ALFWORLD_DATA`, ScienceWorld
needs Java). Each one lists its tasks (`task_ids()`), reads a task's goal (`task(task_id)`), derives the task plan
(`plan(task)`) and runs the environment loop for `run_task`.

- **ALFWorld**: the plan is read directly from the game's PDDL (`pddl_plan`): the expert plan in `traj_data.json`
  gives the operators, and the PDDL problem in `game.tw-pddl` names the object, source, destination and appliance
  of each one (the game numbers its objects deterministically from that problem, `entity_names`). These bindings
  are what grounding substitutes into stored subtraces; no exploration of the room is needed. Subtraces close at
  the milestone actions take, put, heat, cool, clean and use.
- **ScienceWorld**: one plan per task type, derived from the benchmark's gold traces of training variations
  (`python -m expreuse.envs.scienceworld build-plans`, table in `envs/scienceworld_plans.json`); the type of a new
  task is recognised from its description, and a scripted scan of the rooms (`scan`, teleport to each room and read
  it, not charged to the task) binds the plan to the substance, heat source, cold place, container and instrument
  it finds. Subtraces close at the manipulation commands (focus on, pick up, move, pour, mix, activate, deactivate,
  use, connect, read), a pick up or move being complete when the simulator confirms it; activating a heat source
  realizes `heat` and moving an object into a freezer or fridge realizes `cool`, the operators shared with ALFWorld
  besides take and put.

ALFWorld's planner library is copied to a temporary directory at start-up; if the home directory is on a network
file system, point `TMPDIR` to a local disk.

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

## Policy consumers

Both trained policies implement `BasePolicy` (`log_probs` over the admissible actions) and are shaped by
`ShapedPolicy`. Install the `policies` extra.

**ALFWorld, TextDAgger.** ALFWorld's own DAgger recipe (`alfworld_dagger_train.py` is ALFWorld 0.4.2's training
script with four marked changes; `alfworld_dagger.yaml` its configuration), trained on the training games minus
the ones the stores were collected on, with `valid_seen` for checkpoint selection, 50,000 episodes on a GPU.

```bash
python -m expreuse.policies.alfworld_dagger prepare --exclude store_games.txt --target data/alfworld_dagger_train
DAGGER_TRAIN_DATA=data/alfworld_dagger_train/json_2.1.1/train DAGGER_MODELS=data/models/alfworld_dagger \
  python -m expreuse.policies.alfworld_dagger_train src/expreuse/policies/alfworld_dagger.yaml
python -m expreuse.policies.alfworld_dagger calibrate --checkpoint data/models/alfworld_dagger/dagger_admissible_id.pt
```

```python
from expreuse.consumers import ShapedPolicy
from expreuse.policies.alfworld_dagger import DaggerPolicy
policy = ShapedPolicy(DaggerPolicy("data/models/alfworld_dagger/dagger_admissible_id.pt"), embedder, m=1, lam=1.0)
```

**ScienceWorld, imitation.** A scoring head over frozen all-MiniLM-L6-v2 embeddings of the state (task, last
actions and observation, room, inventory) and of each valid action, trained by behaviour cloning on the gold paths
of the training variations (store games left out), with harmless detours as in DART; the epoch is selected by
playing the dev games and the score temperature is fitted on the dev states.

```bash
python -m expreuse.policies.scienceworld_imitation data --exclude store_games.txt --workers 24
python -m expreuse.policies.scienceworld_imitation embed
python -m expreuse.policies.scienceworld_imitation train --epochs 6
python -m expreuse.policies.scienceworld_imitation select
python -m expreuse.policies.scienceworld_imitation calibrate
```

```python
from expreuse.policies.scienceworld_imitation import ImitationPolicy
policy = ShapedPolicy(ImitationPolicy("data/models/scienceworld_imitation", app), embedder, m=1, lam=1.0,
                      operator_of=app.operator_of)
```

## Experiments

Every condition is one call of `expreuse.experiments.run` (application, consumer, query, store, task list), which
writes one JSON line per episode and resumes where it stopped; `expreuse.experiments.summarize` turns result files
into task success with bootstrap standard errors (runs averaged per task, tasks resampled), route shares, steps,
time per component and tokens, with paired differences, wins and losses against a baseline file. The scripts in
`experiments/` run the conditions of each experiment:

| Script | Experiment |
|---|---|
| `collect_stores.sh` | the producers solve the store task lists once (plan shown, no experience); their traces are the stores |
| `same_application.sh` | each consumer with its own store: no reuse, Whole, Subtraces, Cascade; language models with the plan shown and in the no-plan setting (plan hidden, answers ungrounded) |
| `cross_agent.sh` | a consumer reads another producer's store |
| `cross_application.sh` | ALFWorld consumers read ScienceWorld traces and the reverse, on the tasks with shared operators, ungrounded; the same-application answers on those tasks are the reference |
| `ablations.sh` | Cascade's threshold (`theta_sweep`, assembled from the Whole, Subtraces and no-reuse runs), retrieval depth K, partial subtrace coverage (`--withhold`, every subset on ALFWorld, one nested random ordering per task on ScienceWorld), grounding on and off |
| `systems.sh` | the physical realizations: latency as the store grows, throughput under concurrent clients, approximate goal search |

```bash
export OPENROUTER_API_KEY=...        # Gemma; Ministral is served by Amazon Bedrock from the AWS environment
PY=.venv/bin/python STORES=stores RESULTS=results bash experiments/collect_stores.sh
bash experiments/same_application.sh
python -m expreuse.experiments.summarize results/same_application/alfworld_gemma_plan_*.jsonl \
    --baseline results/same_application/alfworld_gemma_plan_none.jsonl
```

The settings live in `src/expreuse/config.py`; a command-line option of `run` overrides a value for one condition.

| Setting | Value |
|---|---|
| Goal embeddings | `all-mpnet-base-v2`, precomputed per goal |
| Query | K = 1 answer; policies: m = 1 voting step, λ = 1; grounding on within an application |
| Cascade threshold θ | 0.6 on ALFWorld (selected on `valid_seen`), 0.8 on ScienceWorld (selected on the dev variations) |
| Language models | Gemma 4 26B-A4B-IT: `google/gemma-4-26b-a4b-it` through OpenRouter, providers Cloudflare then DeepInfra (ALFWorld) or Cloudflare (ScienceWorld); Ministral 8B: `mistralai/ministral-8b-2512` through Amazon Bedrock (`mistral.ministral-3-8b-instruct`) |
| Serving | temperature 0, reasoning mode off, 4000 output tokens on ALFWorld and 1200 on ScienceWorld, request timeout 300 s / 600 s with 4 / 6 retries, static action interface (the action forms once, no per-step list of valid actions), an invalid action costs a step |
| Repetitions | 3 runs of every language-model condition; the trained policies are deterministic and run once |
| Seeds | ALFWorld environment 42; bootstrap 0; policy training 0; the coverage ordering of a task is seeded by its id |
| Budgets | 40 steps on ALFWorld; per task type on ScienceWorld (`STEP_BUDGET`); scans are not charged |
| Standard errors | bootstrap over tasks, 10,000 resamples |
| Task lists | `data/tasks/README.md` |

**Stores.** A producer's store is the set of traces it left on its store task list (256 ALFWorld training games,
149 ScienceWorld training variations), segmented into subtraces; `collect_stores.sh` writes one JSON Lines file
per producer and application. The scaling store of the system benchmarks (about 3.6K ALFWorld traces, grown
synthetically to a million subtraces) was produced the same way by two further agents, a DeepSeek model and
Ministral 3B, on ALFWorld's public training games (the `train` split of the `json_2.1.1` data distributed with
[ALFWorld](https://github.com/alfworld/alfworld)); `systems.sh` grows any producer's store in its place.

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
