# uedrlhf — learning to score environment difficulty from pixels

Bachelor's thesis project (LMU Munich). It trains convolutional networks to
predict **how difficult a procedurally generated RL environment is, from its
rendered image alone**, using preference labels produced by a vision LLM
instead of human annotators.

The trained scorer is meant to be dropped into an [Unsupervised Environment
Design](https://arxiv.org/abs/2012.02096) loop as the replacement for a
hand-written difficulty heuristic — specifically as the scorer in Sampling For
Learnability (SFL).

---

## The idea

Curriculum methods for RL need to rank candidate environments by how useful
they are to train on. That ranking is normally a hand-tuned heuristic tied to
one environment's internal state.

This project asks whether the ranking can instead be **learned from pixels**:

1. Generate pairs of environments.
2. Ask a vision LLM which one is harder — a cheap, scalable stand-in for human
   preference feedback.
3. Train a CNN on those pairwise preferences (Bradley–Terry style) to output a
   scalar difficulty score.
4. Use the resulting score as a drop-in, environment-agnostic scorer.

Two environment families are supported, through one shared interface:
[JaxNav](https://github.com/FLAIROx/JaxMARL) (continuous 2D navigation) and
[XLand-MiniGrid](https://github.com/dunnolab/xland-minigrid) (gridworld with
rulesets).

## Pipeline

```
  generate            feedback              dataset            train              analysis
┌────────────┐   ┌────────────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────────┐
│ sample env │   │ ask the LLM    │   │ SQLite  ->   │   │ CNN /       │   │ paired stats │
│ pairs, JAX │──>│ which is       │──>│ HuggingFace  │──>│ ResNet on   │──>│ + figures    │
│ render PNG │   │ harder (batch) │   │ Array3D u8   │   │ preferences │   │ (W&B runs)   │
└────────────┘   └────────────────┘   └──────────────┘   └─────────────┘   └──────────────┘
 scripts/generate  scripts/feedback     scripts/dataset    scripts/train     scripts/analysis
```

Each stage has a matching SLURM job in `scripts/slurm_scripts/`, numbered in
execution order (`v3_1_generate_mixed.sh` … `v3_9_finetune_mixed.sh`).

## Layout

```
uedrlhf/                  library code
  config.py                 loads config.yaml once, exposes validated constants
  networks.py               CNN and LearnabilityResNet (flax.nnx)
  environments/             EnvBackend interface + jaxnav / xland backends
  llm_providers/            LLMProvider interface + gemini / openwebui providers
  jax_renderer.py           JIT tile-cache renderer
  fast_data.py, db.py       fast data loading, SQLite feedback store

scripts/
  generate/                 environment-pair generation
  feedback/                 LLM preference collection (batch submit / collect)
  dataset/                  compile SQLite -> HuggingFace, uint8 re-encoding
  train/                    pretrain, and three finetuning strategies
                            (*_fast.py variants are env-var parameterized)
  analysis/                 statistics, figures, solvability and balance checks
  slurm_scripts/            cluster jobs, numbered per pipeline stage
  debug/                    manual inspection utilities (not a test suite)

app/                      small FastAPI app for browsing/labelling environments
cloud/                    VM provisioning helpers
config.yaml               single source of truth for every run
```

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/malikdurmus/uedrlhf.git
cd uedrlhf
uv sync
cp .env.example .env    # then fill in your API keys
```

`.env` supplies `GEMINI_API_KEY`, `WANDB_API_KEY` and `OPENWEBUI_API_KEY`. It is
git-ignored; nothing in this repo should ever contain a real key.

Gemini batch mode additionally needs a Google service-account JSON and a GCS
bucket, both named in the `gemini:` block of `config.yaml`.

## Configuration

`config.yaml` drives everything. `uedrlhf/config.py` reads it once at import,
validates the enum fields, and exposes the result as module constants — scripts
never parse YAML themselves.

```yaml
environment: xland_minigrid      # or jaxnav
llm_provider: gemini             # or openwebui
feedback_type: difficulty_feedback
training:
  learning_rate: 0.001
  batch_size: 32
  epochs: 50
```

The `*_fast.py` trainers additionally read `DATASET_PATH`, `PROJECT_NAME`,
`RUN_NAME`, `SEED`, `LEARNING_RATE` and `AUGMENT_SYMMETRY` from the
environment, so a single script covers a whole SLURM array;
`train_pixels_fast.py` also honours `INIT_SEED` (for ensemble members) and
`ABORT_ON_COLLAPSE`. Their defaults reproduce `config.yaml` behaviour exactly.

## Running it

```bash
# 1. generate environment pairs and render them
python scripts/generate/generate.py

# 2. submit them to the LLM, then collect the results
python scripts/feedback/llm_batch_submit.py
python scripts/feedback/llm_batch_collect.py --job batches/<JOB_ID>

# 3. compile into a HuggingFace dataset
python scripts/dataset/compile.py
python scripts/dataset/convert_to_array3d.py

# 4. train
python scripts/train/train_pixels.py        # direct
python scripts/train/finetune_full.py       # from a pretrained checkpoint

# 5. reproduce the statistical comparison and figures
python scripts/analysis/compare_methods.py --outdir outputs/analysis
```

On a cluster, use the numbered SLURM scripts instead. They `cd` to
`${REPO_ROOT:-$HOME/Desktop/GIT/new/uedrlhf}`, so set `REPO_ROOT` if your
checkout lives elsewhere.

## Method notes

A few decisions that shaped the results:

- **Epoch selection.** An 80/10/10 train/val/test split; the epoch with minimum
  *raw* validation loss is selected and test metrics are reported at that epoch.
  The test split is never used for selection. This matters here because
  BatchNorm makes per-epoch validation metrics oscillate sharply, so
  final-epoch numbers are meaningless and "best accuracy over epochs"
  double-dips the reported metric.
- **Paired statistics, not means.** Conditions are compared across 10 seeds
  sharing a split and initialization per seed: Wilcoxon signed-rank for two
  conditions, Friedman followed by Holm-corrected pairwise Wilcoxon for three
  (Demšar, 2006). A single collapsed run can flip a mean while 9/10 seeds point
  the other way — that happened, which is why conclusions come from paired
  tests and per-seed plots rather than bar charts.
- **One learning rate everywhere** (1e-3, batch 32), fixed in advance, so the
  finetuning comparison isn't confounded by per-arm tuning. Declared as a
  limitation rather than tuned away.
- **Deep ensemble** of 5 networks for deployment, differing only in
  initialization seed; inference averages the scalar scores (never the
  weights), and member spread doubles as an uncertainty signal.

## Results

Finetuning strategies on the JaxNav solvable dataset, 10 paired seeds, test
accuracy at the min-validation-loss epoch:

| Method | Test accuracy |
| --- | --- |
| Full finetune | **0.8700 ± 0.0105** |
| Differential LR | 0.8239 ± 0.0098 |
| Linear probe | 0.8007 ± 0.0091 |

Friedman p = 4.5e-5; all pairwise Holm-adjusted p = 0.0059; the ordering holds
in 10/10 seeds and under both selection rules.

Solvability pretraining reaches ~100% accuracy — expected, since solvability is
a deterministic function of a fully visible grid — but measurably did *not* help
downstream: on identical data, from-scratch training matched or beat finetuning
from the pretrained checkpoint (Wilcoxon p = 0.0098).

Numbers are regenerated by `scripts/analysis/compare_methods.py`; figures and
tables land in `outputs/analysis/`.

## Related repositories

- **SFL loop** — where the scorer gets deployed:
  [`sampling-for-learnability`](https://github.com/malikdurmus/sampling-for-learnability)
- **JaxMARL fork** providing the JaxNav environment:
  [`malikdurmus/JaxMARL`](https://github.com/malikdurmus/JaxMARL) (branch `experimental`)

## Notes

Datasets, checkpoints, W&B artefacts and the thesis working notes are not part
of this repository. `scripts/debug/` holds manual utilities rather than tests;
`pytest` is configured to skip it.

## Acknowledgements

Built on [JaxMARL](https://github.com/FLAIROx/JaxMARL),
[XLand-MiniGrid](https://github.com/dunnolab/xland-minigrid),
[JAXUED](https://github.com/DramaCow/jaxued) and
[flax.nnx](https://flax.readthedocs.io/). This repository's history begins from
a fork of JaxMARL.
