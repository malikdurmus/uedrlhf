# Ensemble-vs-single ablation: xmg_medium
Shared split SEED=0; member epoch = min raw val loss (from W&B).

| Set | member accs | single mean ± std | ensemble acc | ensemble F1 |
|---|---|---|---|---|
| no_aug | 0.8126, 0.7916, 0.8016, 0.7939, 0.8039 | 0.8007 ± 0.0084 | **0.8057** | 0.8155 |

Members used (no_aug): ens-xmgmed-init100@15, ens-xmgmed-init101@18, ens-xmgmed-init102@19, ens-xmgmed-init103@24, ens-xmgmed-init104@10

| aug | 0.8135, 0.8094, 0.8094, 0.8144, 0.8221 | 0.8137 ± 0.0052 | **0.8171** | 0.8277 |

Members used (aug): ens-xmgmed-aug-init100@34, ens-xmgmed-aug-init101@41, ens-xmgmed-aug-init102@49, ens-xmgmed-aug-init103-restart@47, ens-xmgmed-aug-init104-restart2@32

