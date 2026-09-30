# Hyperparameter sensitivity: learning rate (XLand-MiniGrid difficult, direct)

Pre-registered VALIDATION-ONLY sensitivity analysis; the test split is not
consulted here, and the main campaigns keep lr = 1e-3 regardless of the
outcome (Rule A, THESIS_NOTES.MD). Within-run selection: epoch of minimum raw
val loss. Runs execute with ABORT_ON_COLLAPSE=0 so collapses are counted
rather than hidden; means are over trained (non-collapsed) runs only.

| learning rate | n trained | collapses | val loss (mean +/- std) | val acc (mean +/- std) | selected epochs |
|---|---|---|---|---|---|
| 3e-4 | 3 | 0/3 | 0.4437 +/- 0.0158 | 0.7808 +/- 0.0187 | 12, 14, 18 |
| 1e-3 | 2 | 1/3 | 0.4396 +/- 0.0260 | 0.7833 +/- 0.0184 | 17, 14 |
| 3e-3 | 1 | 2/3 | 0.4539 (n=1) | 0.7836 (n=1) | 25 |

Lowest mean validation loss: **lr = 1e-3** (val loss 0.4396, val acc 0.7833).
This confirms the configuration used throughout the thesis.

## Collapse incidence (secondary result)

Across the grid: 3/9 runs collapsed (3e-4: 0/3, 1e-3: 1/3, 3e-3: 2/3).

The rate increases monotonically with the learning rate, which is the direction predicted by the dying-ReLU account of the collapse (larger first Adam steps -> higher chance the scoring head's ReLU is driven into the all-negative regime, from which the pairwise-difference loss provides no gradient path back). Report as SUGGESTIVE, not established: with 3 runs per cell the comparison is badly underpowered (Fisher exact, lowest vs highest lr: p ~ 0.4). A properly powered test would need ~20 runs per cell and is out of scope; state the trend and its limits.
