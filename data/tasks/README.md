# Task lists

One task id per line. ALFWorld ids are game files relative to `$ALFWORLD_DATA` (ALFWorld's `json_2.1.1` data,
fetched with `alfworld-download`); ScienceWorld ids are `task type:variation` in the simulator's own numbering.
`expreuse.experiments.run --tasks <file>` plays a list; `expreuse.config.APPLICATIONS` names the default one of
each use.

| File | Tasks | Use |
|---|---|---|
| `alfworld_evaluation.txt` | 134 | the `valid_unseen` games: every ALFWorld task-success number |
| `alfworld_store.txt` | 256 | training games the producers solved to fill their stores; left out of the policy's training games |
| `alfworld_valid_seen.txt` | 140 | the `valid_seen` games: Cascade's threshold and the policy's score temperature are selected here |
| `alfworld_cross_application.txt` | 85 | evaluation games whose plan contains an operator shared with ScienceWorld (take, put, heat, cool) |
| `scienceworld_evaluation.txt` | 149 | the first five test variations of each of the 30 task types (one type has four) |
| `scienceworld_store.txt` | 149 | training variations the producers solved to fill their stores; left out of the imitation policy's training data |
| `scienceworld_dev.txt` | 145 | the first five dev variations per type: Cascade's threshold, the imitation policy's epoch and score temperature |
| `scienceworld_plan_training.txt` | 518 | the training variations whose gold traces give the per-type plans (`envs/scienceworld.py build-plans`, up to 20 per type) |
| `scienceworld_cross_application.txt` | 55 | evaluation tasks whose plan contains an operator shared with ALFWorld |

The evaluation and store lists never overlap, and no test variation of ScienceWorld is replayed when the plans
are built.
