# Ensemble-vs-single ablation: jaxnav
Shared split SEED=0; member epoch = min raw val loss (from W&B).

| Set | member accs | single mean ± std | ensemble acc | ensemble F1 |
|---|---|---|---|---|
| no_aug | 0.8772, 0.8795, 0.8809, 0.8890, 0.8813 | 0.8816 ± 0.0045 | **0.8863** | 0.8938 |

Members used (no_aug): ens-jaxnav-init100@7, ens-jaxnav-init101@9, ens-jaxnav-init102@9, ens-jaxnav-init103@10, ens-jaxnav-init104@6

| aug | 0.8981, 0.8927, 0.8936, 0.8899, 0.8936 | 0.8936 ± 0.0029 | **0.9000** | 0.9078 |

Members used (aug): ens-jaxnav-aug-init100@26, ens-jaxnav-aug-init101@22, ens-jaxnav-aug-init102@41, ens-jaxnav-aug-init103@32, ens-jaxnav-aug-init104@28

