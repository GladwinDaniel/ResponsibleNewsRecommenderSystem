# Experiment Results — Responsible News Recommender

> Generated: 2026-10-19 13:47:59
> Framework: GNN + RoBERTa-23 + EMA Susceptibility + Provider Diversity Cap
> Dataset: MINDsmall (synthetic mode — published baseline numbers)

---

## 1. Baseline Comparison

### Baseline Comparison (MINDsmall dev)

| System | AUC ↑ | nDCG@5 ↑ | nDCG@10 ↑ | MRR ↑ | P@10 ↑ | PER ↑ | Parity Gap ↓ | Explain. ↑ | HR Suppressed ↑ |
|--------|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| NRMS (Wu et al., 2019) | 0.6804 | 0.4090 | 0.5195 | 0.3761 | 0.1635 | 0.0000 | 0.452 | 0.000 | 0.000 |
| NAML (Wu et al., 2019) | 0.6662 | 0.3817 | 0.4952 | 0.3455 | 0.1620 | 0.0000 | 0.468 | 0.000 | 0.000 |
| LSTUR (An et al., 2019) | 0.6753 | 0.3899 | 0.5078 | 0.3602 | 0.1655 | 0.0000 | 0.441 | 0.000 | 0.000 |
| GLORY (Phan et al., 2023) | 0.6894 | 0.4302 | 0.5375 | 0.3984 | 0.1638 | 0.0000 | 0.418 | 0.000 | 0.000 |
| ProFairRec (Li et al., 2022) | 0.6679 | 0.3869 | 0.5055 | 0.3553 | 0.1675 | 0.0000 | 0.124 | 0.124 | 0.020 |
| RoBERTa Text-Only | 0.6284 | 0.4607 | 0.5342 | 0.6678 | 0.3349 | 0.0003 | 0.002 | 1.000 | 1.000 |
| MuRIL Text-Only | 0.6325 | 0.4592 | 0.5393 | 0.6589 | 0.3409 | 0.0002 | 0.001 | 1.000 | 1.000 |
| EfficientNet-B0 Vision-Only | 0.6215 | 0.4486 | 0.5271 | 0.6507 | 0.3339 | 0.0000 | 0.002 | 1.000 | 1.000 |
| Xception Vision-Only | 0.6265 | 0.4533 | 0.5294 | 0.6515 | 0.3337 | 0.0000 | 0.001 | 1.000 | 1.000 |
| Multimodal: RoBERTa + EfficientNet | 0.6365 | 0.4668 | 0.5468 | 0.6710 | 0.3430 | 0.0001 | 0.002 | 1.000 | 1.000 |
| Multimodal: RoBERTa + Xception | 0.6333 | 0.4613 | 0.5409 | 0.6674 | 0.3408 | 0.0003 | 0.002 | 1.000 | 1.000 |
| Multimodal: MuRIL + EfficientNet | 0.6393 | 0.4718 | 0.5470 | 0.6752 | 0.3465 | 0.0001 | 0.002 | 1.000 | 1.000 |
| Multimodal: MuRIL + Xception | 0.6390 | 0.4676 | 0.5441 | 0.6630 | 0.3443 | 0.0002 | 0.000 | 1.000 | 1.000 |
| **Ours (RespNews Multimodal GNN)** | 0.6357 | 0.4673 | 0.5454 | 0.6782 | 0.3403 | 0.0000 | 0.002 | 1.000 | 1.000 |

### Key Findings

**vs NRMS:**
- AUC: -0.0447 (-6.6%)
- nDCG@10: +0.0260 (+5.0%)
- PER improvement: +0.0000 (from 0.000 → 0.000)
- Provider parity improvement: +0.450 (from 0.452 → 0.002)
- Explainability: 0.0% → 100.0%

**vs GLORY:**
- AUC: -0.0537 (-7.8%)
- nDCG@10: +0.0080 (+1.5%)
- PER improvement: +0.0000 (from 0.000 → 0.000)
- Provider parity improvement: +0.416 (from 0.418 → 0.002)
- Explainability: 0.0% → 100.0%


---

## 2. Ablation Study

### Ablation Study

| System | AUC ↑ | nDCG@5 ↑ | nDCG@10 ↑ | MRR ↑ | P@10 ↑ | PER ↑ | Parity Gap ↓ | Explain. ↑ | HR Suppressed ↑ |
|--------|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| **Full System** | 0.6361 | 0.4669 | 0.5416 | 0.6669 | 0.3431 | 0.0002 | 0.002 | 1.000 | 1.000 |
| w/o Persuasion Penalty | 0.6381 | 0.4613 | 0.5440 | 0.6587 | 0.3474 | 0.0000 | 0.002 | 1.000 | 1.000 |
| w/o Susceptibility | 0.6402 | 0.4665 | 0.5465 | 0.6650 | 0.3449 | 0.0002 | 0.001 | 1.000 | 1.000 |
| w/o GNN Embeddings | 0.6334 | 0.4626 | 0.5393 | 0.6618 | 0.3426 | 0.0002 | 0.001 | 1.000 | 1.000 |
| w/o Vision Backbone (MuRIL Text-Only) | 0.6300 | 0.4551 | 0.5335 | 0.6542 | 0.3374 | 0.0001 | 0.002 | 1.000 | 1.000 |
| w/o MuRIL (RoBERTa Text Backbone) | 0.6314 | 0.4582 | 0.5372 | 0.6617 | 0.3406 | 0.0000 | 0.001 | 1.000 | 1.000 |
| w/o Provider Cap | 0.6351 | 0.4611 | 0.5419 | 0.6658 | 0.3368 | 0.0000 | 0.001 | 1.000 | 1.000 |

### Component Contribution Analysis

| Component Removed | AUC Δ | nDCG@10 Δ | PER Δ | Parity Gap Δ |
|---|---:|---:|---:|---:|
| Persuasion Penalty | +0.0020 | +0.0024 | -0.0002 | -0.000 |
| Susceptibility | +0.0041 | +0.0049 | -0.0000 | -0.001 |
| GNN Embeddings | -0.0027 | -0.0024 | -0.0000 | -0.001 |
| Vision Backbone (MuRIL Text-Only) | -0.0061 | -0.0082 | -0.0001 | -0.000 |
| MuRIL (RoBERTa Text Backbone) | -0.0047 | -0.0044 | -0.0002 | -0.001 |
| Provider Cap | -0.0010 | +0.0002 | -0.0002 | -0.001 |

> A **negative** PER Δ means removing that component *reduces* persuasion protection.
> A **positive** Parity Gap Δ means removing that component *worsens* provider diversity.

---

## 3. Pareto Frontier

### Pareto Frontier: α/β Sensitivity Analysis

| β | α | Avg Relevance | Avg Risk | PER % | High-Risk | Parity Gap | Optimal? |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.00 | 1.00 | 0.7870 | 0.1422 | 0.0% | 0.0 | 0.400 |  |
| 0.10 | 0.97 | 0.7865 | 0.1416 | 0.3% | 0.0 | 0.400 |  |
| 0.20 | 0.94 | 0.7853 | 0.1412 | 0.6% | 0.0 | 0.400 |  |
| 0.30 | 0.91 | 0.7869 | 0.1410 | 0.7% | 0.0 | 0.400 | ⭐ |
| 0.40 | 0.88 | 0.7891 | 0.1409 | 0.8% | 0.0 | 0.400 |  |
| 0.50 | 0.85 | 0.7861 | 0.1408 | 0.9% | 0.0 | 0.400 |  |
| 0.60 | 0.82 | 0.7864 | 0.1408 | 0.9% | 0.0 | 0.400 |  |
| 0.70 | 0.79 | 0.7885 | 0.1407 | 1.0% | 0.0 | 0.400 |  |
| 0.80 | 0.76 | 0.7841 | 0.1406 | 1.0% | 0.0 | 0.400 |  |
| 0.90 | 0.73 | 0.7862 | 0.1406 | 1.0% | 0.0 | 0.400 |  |
| 1.00 | 0.70 | 0.7848 | 0.1405 | 1.1% | 0.0 | 0.400 |  |

---

## Methodology Notes

- **Baseline numbers** are sourced from published papers on MINDsmall dev split
- **IR metrics** (AUC, nDCG, MRR) are simulated to match published aggregates
- **SDG metrics** (PER, Parity Gap, Explainability, HR Suppressed) are computed from our pipeline
- **Ablation** uses the same article pool with components selectively disabled
- **Pareto frontier** sweeps β from 0.0 (pure relevance) to 1.0 (maximum safety)
- Recommended operating point: **α=0.70, β=0.30** (marked with ⭐)