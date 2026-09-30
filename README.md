# uedrlhf

Train a CNN to score how difficult a procedurally generated RL environment is,
using only its rendered image. The training labels come from pairwise
comparisons made by a vision LLM.

Bachelor's thesis project, LMU Munich.

## What it does

Curriculum methods for RL need to rank candidate environments. Ranking them by
difficulty usually requires a hand-written heuristic over internal simulator
state. That heuristic has to be rewritten for every new environment.

This project learns the ranking from pixels:

1. Generate pairs of environments and render them.
2. Ask a vision LLM which one is harder.
3. Train a CNN on those pairwise preferences to produce a difficulty score.
4. Use an ensemble of the trained scorers inside a Sampling For Learnability
   (SFL) loop.

Because the input is an image, the same architecture works across environments.
Two are supported: [JaxNav](https://github.com/FLAIROx/JaxMARL) (2D navigation)
and [XLand-MiniGrid](https://github.com/dunnolab/xland-minigrid) (gridworld).

## Pipeline

```
 generate          feedback           dataset          train          analysis
┌──────────┐    ┌──────────────┐    ┌───────────┐    ┌──────────┐    ┌─────────┐
│ sample   │    │ submit pairs │    │ SQLite to │    │ pretrain │    │ stats   │
│ env pair │───>│ to the LLM,  │───>│ HuggingF., │───>│ finetune │───>│ and     │
│ + render │    │ collect      │    │ uint8      │    │ direct   │    │ figures │
└──────────┘    └──────────────┘    └───────────┘    └──────────┘    └─────────┘
```

Each stage has a matching SLURM job in `scripts/slurm_scripts/`, numbered in
execution order (`v3_1_generate_mixed.sh` through `v3_9_finetune_mixed.sh`).

This repository covers generation, labelling, training and evaluation. The SFL
loop that consumes the scorer lives in
[sampling-for-learnability](https://github.com/malikdurmus/sampling-for-learnability).

## Layout

```
uedrlhf/                  library code
  config.py                 reads and validates config.yaml, exposes constants
  networks.py               CNN and LearnabilityResNet (flax.nnx)
  environments/             EnvBackend interface, jaxnav and xland backends
  llm_providers/            LLMProvider interface, gemini and openwebui
  jax_renderer.py           JIT tile-cache renderer
  fast_data.py, db.py       data loading, SQLite feedback store

scripts/
  generate/                 environment pair generation
  feedback/                 LLM preference collection
  dataset/                  compile to HuggingFace, uint8 re-encoding
  train/                    pretraining and three finetuning strategies
  analysis/                 statistics, figures, solvability checks
  slurm_scripts/            cluster jobs
  debug/                    manual inspection utilities, not a test suite

app/                      FastAPI app for collecting human preference labels
outputs/analysis/         result tables and figures
config.yaml               settings for every run
```

`EnvBackend` and `LLMProvider` are abstract base classes. Adding an environment
or an LLM provider means implementing one interface. No pipeline code changes.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/malikdurmus/uedrlhf.git
cd uedrlhf
uv sync
cp .env.example .env
```

Fill in `.env` with `GEMINI_API_KEY`, `WANDB_API_KEY` and `OPENWEBUI_API_KEY`.
The file is git-ignored. Gemini batch mode also needs a Google service-account
JSON and a GCS bucket, both named in the `gemini:` block of `config.yaml`.

## Configuration

`config.yaml` holds all settings. `uedrlhf/config.py` reads it once at import
and validates it. Scripts import from there and never parse YAML themselves.

```yaml
environment: xland_minigrid      # or jaxnav
llm_provider: gemini             # or openwebui
feedback_type: difficulty_feedback
training:
  learning_rate: 0.001
  batch_size: 32
  epochs: 50
```

The `*_fast.py` trainers also read these environment variables:
`DATASET_PATH`, `PROJECT_NAME`, `RUN_NAME`, `SEED`, `LEARNING_RATE` and
`AUGMENT_SYMMETRY`. This lets one script serve a whole SLURM array.
`train_pixels_fast.py` also reads `INIT_SEED` and `ABORT_ON_COLLAPSE`. The
defaults match `config.yaml`.

## Running

```bash
# generate environment pairs and render them
python scripts/generate/generate.py

# submit to the LLM, then collect the results
python scripts/feedback/llm_batch_submit.py
python scripts/feedback/llm_batch_collect.py --job batches/<JOB_ID>

# compile into a HuggingFace dataset
python scripts/dataset/compile.py
python scripts/dataset/convert_to_array3d.py

# train
python scripts/train/train_pixels.py        # direct
python scripts/train/finetune_full.py       # from a pretrained checkpoint

# regenerate the statistics and figures
python scripts/analysis/compare_methods.py --outdir outputs/analysis

# optional: collect human labels
uvicorn app.main:app --reload
```

On a cluster, use the numbered SLURM scripts. They change into
`${REPO_ROOT:-$HOME/Desktop/GIT/new/uedrlhf}`, so set `REPO_ROOT` if your
checkout is elsewhere.

## Notes

Datasets, checkpoints and W&B artefacts are not included in this repository.
`scripts/debug/` holds manual utilities, and `pytest` is configured to skip it.

## Acknowledgements

Built on [JaxMARL](https://github.com/FLAIROx/JaxMARL),
[XLand-MiniGrid](https://github.com/dunnolab/xland-minigrid),
[JAXUED](https://github.com/DramaCow/jaxued) and
[flax.nnx](https://flax.readthedocs.io/). This repository's history starts from
a fork of JaxMARL.

## License

MIT, see [LICENSE](LICENSE).
