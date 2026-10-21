# Mini Transformer Outer Loop for the Closed-Loop DTN

This trains a small **Set-Transformer** that plays the role of the paper's
**outer DTN-control layer / regional projection step** (Sec. V-D,
Eq. 102–103): given the ~10 candidate APs your QUBO/BQM inner loop already
scored for a task (with twin fidelity/age/uncertainty, trust, coordination
duals, and the QUBO-derived delay/energy/risk/fidelity costs), it predicts
which destination the regional coordinator should pick — learned directly
from your `pso_training_dataset_extended.xlsx` labels (`pso_selected`).

```
inner loop (per AP)          outer loop (this repo)
QUBO/BQM candidate  ───►  Set-Transformer over the task's
generation, one row           K candidate rows  ───►  softmax over
per (task, candidate AP)      destinations = learned projection x*_u,m(t)
```

## Files

| file        | purpose |
|---|---|
| `data.py`   | loads the xlsx (caches as parquet), builds per-task candidate sets, scenario-stratified train/val/test split |
| `model.py`  | `OuterLoopSetTransformer` — 2–4 layer permutation-invariant transformer encoder + selection/regression heads |
| `train.py`  | training loop (CLI), reports top-1/top-3 accuracy against the PSO label and average cost regret |
| `infer.py`  | `OuterLoopPolicy` for inference, plus `features_from_repo_objects` to bridge live `Task`/`APNode`/`CandidateScore` objects from `Sagin_DT-DQO-main` into the same feature schema |

## Data shape 

* 199,540 rows = 19,954 task-groups × up to 10 candidates each
* every group has exactly one `pso_selected == 1` row
* no missing values

Each group → one training example: a **set** of candidate feature vectors +
the index of the PSO-chosen candidate.

## Why a Transformer here (and not an MLP per candidate)

The regional decision is inherently a *comparison across candidates* — "is
this AP's fidelity/queue/dual-price better than the alternatives *for this
task, right now*" — not a fixed per-candidate score. Self-attention over the
candidate set lets every candidate's score depend on the others (relative
load, relative fidelity, etc.), which is exactly what the paper's arg-min
projection rule (Eq. 102) and the QUBO coupling terms (Eq. 94, coordination
penalty) are doing by hand. A learned coordinator token (`coord_token` in
`model.py`) also gives you a pooled embedding you could feed into the outer
consensus/dual-price update if you want to go further.

## Train

```bash
pip install torch pandas openpyxl pyarrow

python train.py \
  --data pso_training_dataset_extended.xlsx \
  --epochs 25 --batch-size 128 \
  --out outer_loop_ckpt.pt
```

First run parses the xlsx (~1–2 min) and caches a `.parquet` next to it so
subsequent runs load in under a second. On CPU, 15 epochs over the full
19,954 task-groups takes ~2 minutes.

Reference run (15 epochs, default hyperparameters, scenario-stratified
70/15/15 split so val/test scenarios are never seen in training):

```
test top1_acc=0.46   test top3_acc=0.76   avg_cost_regret=+0.45
```

* **top1_acc** — exact match with the PSO-selected AP.
* **top3_acc** — PSO's choice is in the model's top 3.
* **avg_cost_regret** — `local_cost(model choice) − local_cost(PSO choice)`,
  averaged. Near zero (and sometimes *negative* on the val set) means the
  model's picks are about as good as — occasionally better than — PSO's,
  since PSO is itself a metaheuristic, not a guaranteed optimum.

If you want tighter top-1 accuracy, the easiest levers are: more epochs
(loss was still slowly improving at 15), a larger `--d-model`/`--n-layers`,
or lowering `--aux-weight` if the regression head is competing too hard with
the selection head.

## Generate a new 1M-row PSO dataset

From the repository root, run:

```bash
python scripts/generate_pso_dataset.py
```

This writes `pso_training_dataset_1m.csv` and a matching
`pso_training_dataset_1m.summary.json`.  It starts with 500 diverse scenarios
(10 slots each) and, because traffic profiles produce different numbers of
tasks, automatically adds mixed scenarios until the CSV contains **at least
1,000,000 candidate rows**.  Rows are streamed directly to disk; it does not
hold the whole dataset in memory.

The labels are produced by the repository's offline discrete PSO teacher.
Each `(scenario_id, slot, task_id)` group contains all 10 candidate AP rows
and exactly one `pso_selected == 1` label, so it is ready for this training
pipeline.  For a different target or output path:

```bash
python scripts/generate_pso_dataset.py \
  --scenarios 500 --min-rows 1000000 \
  --output pso_training_dataset_1m.csv
```

Then train from the repository root using the generated CSV (the current
loader also accepts CSV files):

```bash
python Outer_loop_with_transformer/train.py \
  --data pso_training_dataset_1m.csv \
  --epochs 25 --batch-size 128 \
  --out Outer_loop_with_transformer/outer_loop_ckpt_1m.pt
```

## Option 1: train the current per-task model on aligned labels

The existing Set-Transformer sees one task and its candidate APs, so use a
per-task label rather than the global PSO label when the goal is to learn an
observable local policy.  This leaves the PSO dataset unchanged and creates a
second CSV with `policy_selected == 1` for the candidate minimizing
`candidate_local_cost + coord_coupling_penalty`:

```bash
python scripts/create_local_policy_dataset.py \
  --input pso_training_dataset_extended.csv \
  --output per_task_policy_dataset_1m.csv
```

Train with the explicit label column:

```bash
cd Outer_loop_with_transformer
python train.py \
  --data ../per_task_policy_dataset_1m.csv \
  --label-col policy_selected \
  --epochs 50 --batch-size 128 \
  --out outer_loop_policy_ckpt.pt
```

Keep the original PSO CSV for global-policy comparisons.  Do not describe the
option-1 checkpoint as an imitation of PSO; it learns the per-task observable
cost policy instead.

## Use the trained policy at simulation time

```python
from infer import OuterLoopPolicy, features_from_repo_objects

policy = OuterLoopPolicy.load("outer_loop_ckpt.pt")

# Option A: you already log candidates in the training schema
result = policy.select_from_dataframe(candidate_rows_df)
print(result["selected_destination_id"], result["ranked"])

# Option B: call it straight from RegionalController with live objects
df = features_from_repo_objects(task, ap_lookup, summary.candidate_scores)
result = policy.select_from_dataframe(df)
```

To actually replace the hand-written projection in
`control/regional.py::RegionalController._rank_scores_with_projected_load`,
swap its cost-based `sorted(...)` key for `policy`'s ranked probabilities
(or blend: `learned_score = -log(prob) + lambda_f * fidelity_violation`,
keeping the hard fidelity gate from Eq. (64)/(102) rather than trusting the
network to learn constraint satisfaction perfectly).

