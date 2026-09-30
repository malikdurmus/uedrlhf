# uedrlhf

**Learning to score environment difficulty from pixels, with LLM preference feedback.**

Bachelor's thesis, LMU Munich. A convolutional network is trained to rate how
difficult a procedurally generated RL environment is *from its rendered image
alone*, supervised by pairwise preferences from a vision LLM rather than by
human annotators. The resulting scorer replaces a hand-written heuristic inside
an Unsupervised Environment Design loop.

---

## Why

Curriculum methods for RL need to rank candidate environments by how useful they
are to train on. Sampling For Learnability (SFL) ranks them by learnability,
`ℓ = p(1−p)`, where `p` is the agent's success rate — a quantity that is cheap to
estimate but says nothing about *how hard* a level is intrinsically. Adding a
difficulty term normally means writing an environment-specific heuristic over
internal simulator state, which has to be redesigned for every new environment.

This project replaces that heuristic with a learned, pixel-based scorer:

1. **Generate** pairs of environments and render them.
2. **Label** each pair with a vision LLM: *which of these two is harder?* — a
   scalable stand-in for human preference feedback.
3. **Train** a CNN on those pairwise preferences under a Bradley–Terry model, so
   it emits a scalar difficulty score.
4. **Deploy** an ensemble of these scorers as the difficulty signal in SFL.

Because the input is a rendered image, the same scorer architecture transfers
across environments. Two families are supported through one interface:
[JaxNav](https://github.com/FLAIROx/JaxMARL) (continuous 2D navigation) and
[XLand-MiniGrid](https://github.com/dunnolab/xland-minigrid) (gridworld with
rulesets).

## System

```
 OFFLINE — this repository
┌──────────────────────────────────────────────────────────────────────────┐
│                         ┌──────────────────┐                             │
│                         │ Human annotation │  validation subset          │
│                         │  (FastAPI app)   │  ─ agreement vs the LLM     │
│                         └────────┬─────────┘                             │
│                                  ┆                                       │
│  ┌───────────┐   ┌───────────────▼──────┐   ┌────────────────────────┐   │
│  │   Level   │──>│    LLM annotator     │──>│   Deep ensemble CNN    │   │
│  │ generator │   │ pairwise "which is   │   │ K=5, Bradley–Terry     │   │
│  │  (JAX)    │   │  harder?" via batch  │   │ scalar difficulty      │   │
│  └───────────┘   └──────────────────────┘   └───────────┬────────────┘   │
└─────────────────────────────────────────────────────────┼────────────────┘
                                                          │ score
 ONLINE — sampling-for-learnability repo                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  ┌──────────────────────┐        ┌──────────────────────────────┐        │
│  │ Percentile           │───c───>│ Compound score               │        │
│  │ normalization        │        │ S = H(ℓ, c, μ)               │        │
│  │ (frozen reference)   │        └──────────┬───────────────────┘        │
│  └──────────────────────┘                   │                            │
│                                             ▼                            │
│  sample N ──> rollouts ℓ=p(1−p) ──> top-K buffer ──> PPO update          │
│      ▲                                                    │              │
│      └────────────────────────────────────────────────────┘              │
│                        target controller μ ± 0.05 ◄── success rate       │
└──────────────────────────────────────────────────────────────────────────┘
```

This repository is the **offline** half: generation, labelling, training and
evaluation of the scorer. The online SFL loop — percentile normalization, the
compound score and the target controller — lives in
[`sampling-for-learnability`](https://github.com/malikdurmus/sampling-for-learnability).

## Pipeline

```
 scripts/generate     scripts/feedback      scripts/dataset    scripts/train    scripts/analysis
┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐  ┌────────────┐  ┌──────────────┐
│ sample env      │  │ submit batch to │  │ SQLite ->      │  │ pretrain / │  │ paired stats │
│ pairs, render   │─>│ the LLM, then   │─>│ HuggingFace,   │─>│ finetune / │─>│ + figures    │
│ via JAX         │  │ collect results │  │ uint8 Array3D  │  │ direct     │  │ from W&B     │
└─────────────────┘  └─────────────────┘  └────────────────┘  └────────────┘  └──────────────┘
```

Every stage has a matching SLURM job in `scripts/slurm_scripts/`, numbered in
execution order (`v3_1_generate_mixed.sh` … `v3_9_finetune_mixed.sh`).

## Layout

```
uedrlhf/                  library code
  config.py                 loads config.yaml once, validates it, exposes constants
  networks.py               CNN and LearnabilityResNet (flax.nnx)
  environments/             EnvBackend ABC + jaxnav / xland_minigrid backends
  llm_providers/            LLMProvider ABC + gemini / openwebui providers
  jax_renderer.py           JIT tile-cache renderer
  fast_data.py, db.py       fast data loading, SQLite feedback store

scripts/
  generate/                 environment-pair generation
  feedback/                 LLM preference collection (batch submit / collect)
  dataset/                  compile SQLite -> HuggingFace, uint8 re-encoding
  train/                    pretraining + three finetuning strategies
                            (*_fast.py variants are env-var parameterized)
  analysis/                 statistics, figures, solvability / balance checks
  slurm_scripts/            cluster jobs, numbered per pipeline stage
  debug/                    manual inspection utilities (not a test suite)

app/                      FastAPI app for collecting human preference labels
cloud/                    VM provisioning helpers
outputs/analysis/         result tables and figures backing the numbers below
config.yaml               single source of truth for every run
```

Both extension points are abstract base classes, so adding a third environment
or a third LLM provider means implementing one interface, not editing the
pipeline: `EnvBackend` (`create_env`, `generate_states`, `render_batch`,
`state_to_dict`, `get_prompts`) and `LLMProvider` (`get_preference`,
`build_batch_request`).

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/malikdurmus/uedrlhf.git
cd uedrlhf
uv sync
cp .env.example .env    # then fill in your keys
```

`.env` holds `GEMINI_API_KEY`, `WANDB_API_KEY` and `OPENWEBUI_API_KEY`. It is
git-ignored; nothing in this repository should ever contain a real key. Gemini
batch mode additionally needs a Google service-account JSON and a GCS bucket,
both named in the `gemini:` block of `config.yaml`.

## Configuration

`config.yaml` drives everything. `uedrlhf/config.py` reads it once at import,
validates the enum fields with explicit errors, and exposes the result as module
constants — no script parses YAML itself.

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
`RUN_NAME`, `SEED`, `LEARNING_RATE` and `AUGMENT_SYMMETRY` from the environment
so one script covers a whole SLURM array; `train_pixels_fast.py` also honours
`INIT_SEED` (ensemble members) and `ABORT_ON_COLLAPSE`. Defaults reproduce
`config.yaml` behaviour exactly.

## Running it

```bash
# 1. generate environment pairs and render them
python scripts/generate/generate.py

# 2. submit to the LLM, then collect
python scripts/feedback/llm_batch_submit.py
python scripts/feedback/llm_batch_collect.py --job batches/<JOB_ID>

# 3. compile into a HuggingFace dataset
python scripts/dataset/compile.py
python scripts/dataset/convert_to_array3d.py

# 4. train
python scripts/train/train_pixels.py        # direct
python scripts/train/finetune_full.py       # from a pretrained checkpoint

# 5. reproduce the statistics and figures
python scripts/analysis/compare_methods.py --outdir outputs/analysis

# optional: collect human labels for the validation subset
uvicorn app.main:app --reload
```

On a cluster use the numbered SLURM scripts. They `cd` to
`${REPO_ROOT:-$HOME/Desktop/GIT/new/uedrlhf}`, so set `REPO_ROOT` if your
checkout lives elsewhere.

## Method

The scorer is trained directly on preferences. Both images of a pair go through
the network in one batch; the difference of their scalar scores is the logit of
a Bradley–Terry model, optimized with binary cross-entropy:

```python
scores            = model(concat(img0, img1))
logits            = score1 - score0          # higher score = more difficult
loss              = sigmoid_binary_cross_entropy(logits, preference)
```

Four decisions shaped the results:

- **Epoch selection.** An 80/10/10 train/val/test split; the epoch with minimum
  *raw* validation loss is chosen and test metrics are reported at that epoch.
  The test split is never used for selection. This matters because BatchNorm
  makes per-epoch validation metrics oscillate sharply, so final-epoch numbers
  are meaningless and "best accuracy over epochs" double-dips the metric being
  reported. Every headline number is re-checked under a smoothed-validation rule.
- **Paired statistics, never means alone.** Conditions are compared over 10 seeds
  sharing a split and initialization per seed: Wilcoxon signed-rank for two
  conditions, Friedman then Holm-corrected pairwise Wilcoxon for three (Demšar,
  2006). This was not academic: one augmentation comparison had a *negative* mean
  delta while winning in 9 of 10 seeds, because a single collapsed run dominated
  the average.
- **One learning rate everywhere** (1e-3, batch 32), fixed in advance, so the
  finetuning comparison is not confounded by per-arm tuning. Recorded as a
  limitation rather than tuned away.
- **Deep ensemble** of K = 5 networks differing only in initialization seed, on
  one shared split. Inference averages the scalar scores — never the weights,
  since independently trained networks sit in different loss basins and averaged
  BatchNorm statistics match no member. Member spread doubles as an epistemic
  uncertainty signal.

## Results

Finetuning strategies, JaxNav solvable dataset, 10 paired seeds, test accuracy
at the minimum-validation-loss epoch:

| Method | Test accuracy |
| --- | --- |
| Full finetune | **0.8700 ± 0.0105** |
| Differential LR | 0.8239 ± 0.0098 |
| Linear probe | 0.8007 ± 0.0091 |

Friedman p = 4.5e-5; all pairwise Holm-adjusted p = 0.0059; the ordering holds in
10/10 seeds and under both selection rules.

Two findings worth stating plainly:

- **Pretraining did not pay off.** Solvability pretraining reaches ~100% accuracy
  — expected, since solvability is a deterministic function of a fully visible
  grid — but on identical data, training from scratch *matched or beat*
  finetuning from the pretrained checkpoint (Wilcoxon p = 0.0098). The
  pretraining stage was consequently dropped for XLand-MiniGrid.
- **An input-domain mismatch was caught and corrected.** An earlier campaign
  trained on inverted images; re-running in the true domain shifted absolute
  accuracies but left the method ordering and every significance conclusion
  intact.

Numbers are regenerated by `scripts/analysis/compare_methods.py`; tables and
figures land in `outputs/analysis/`.

## Notes

Datasets, checkpoints and W&B artefacts are not part of this repository.
`scripts/debug/` holds manual utilities rather than tests, and `pytest` is
configured to skip it.

## Acknowledgements

Built on [JaxMARL](https://github.com/FLAIROx/JaxMARL),
[XLand-MiniGrid](https://github.com/dunnolab/xland-minigrid),
[JAXUED](https://github.com/DramaCow/jaxued) and
[flax.nnx](https://flax.readthedocs.io/). This repository's history begins from a
fork of JaxMARL.

## License

MIT — see [LICENSE](LICENSE).
