# Ensemble-vs-single ablation: xmg_difficult
Shared split SEED=0; member epoch = min raw val loss (from W&B).

| Set | member accs | single mean ± std | ensemble acc | ensemble F1 |
|---|---|---|---|---|
| aug | 0.7985, 0.8053, 0.7957, 0.8044, 0.8062 | 0.8020 ± 0.0046 | **0.8048** | 0.8066 |

Members used (aug): ens-xmg-aug-init101@49, ens-xmg-aug-init102@46, ens-xmg-aug-init103@46, ens-xmg-aug-init104@50, ens-xmg-aug-init105@41

