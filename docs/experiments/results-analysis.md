# Results Analysis

Consolidated results across all three datasets, all tasks, and every
model trained in this project. Every number here comes from an actual
executed and logged experiment under `ml/experiments/results/` — none
are estimated. Full methodology, leakage investigation, and discussion
are in `docs/research/methodology.md`; this document is the results
reference table.

**Every model in this document was trained completely independently on
a single dataset's own data.** No model was shared, transferred, or
reused across CICIDS2017, NSL-KDD, and UNSW-NB15.

---

## CICIDS2017

### Binary classification (BENIGN vs ATTACK)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 93.89% | 0.869 | 0.954 | 0.903 | 0.9875 | 6.87% | 2.37% |
| Decision Tree | 99.89% | 0.997 | 0.999 | 0.998 | 0.9994 | 0.108% | 0.093% |
| Random Forest | 99.89% | 0.997 | 0.999 | 0.998 | 0.99997 | 0.109% | 0.090% |
| **XGBoost** | **99.92%** | **0.998** | **0.999** | **0.999** | **0.99998** | **0.064%** | 0.149% |
| SVM (50K subsample) | 94.82% | 0.885 | 0.959 | 0.916 | 0.9943 | 5.74% | 2.40% |

*SVM trained on a stratified 50,000-row subsample of the 1.76M-row
training set due to computational cost — see methodology.md.*

### Multiclass classification (12 classes; Heartbleed, Infiltration,
Web Attack-SQL Injection excluded — fewer than 50 training samples)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (OvR macro) |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 99.49% | 0.839 | 0.938 | 0.872 | 0.980 |
| Random Forest | 99.58% | 0.863 | 0.937 | 0.877 | 0.999 |
| **XGBoost** | **99.88%** | **0.920** | 0.905 | **0.911** | **0.9997** |

### Feature optimization (XGBoost, binary task)

| Method | Feature count | F1 (macro) | Train time | Inference time |
|---|---:|---:|---:|---:|
| All Features | 78 | 0.9986 | 36.3s | 1.31s |
| Selected Features (RF importance) | 39 | 0.9968 | 27.1s | 1.37s |
| PCA (95% variance) | 25 | 0.9967 | 24.6s | 1.32s |

**Finding:** feature reduction saved ~25-30% training time but
increased false negative rate 4x (0.15% → 0.63-0.68%) — a real
detection-quality trade-off, not a free win.

### Ablation: Destination Port removed (leakage/artifact check)

| Configuration | F1 (macro) |
|---|---:|
| All 78 features | 0.9986 |
| Without Destination Port (77 features) | 0.9982 |

Negligible drop (-0.04pp) — model does not substantially depend on
this single feature.

---

## NSL-KDD (independently trained)

### Binary classification (normal vs attack)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 75.50% | 0.784 | 0.776 | 0.755 | 0.781 | 7.58% | 37.31% |
| Decision Tree | 79.15% | 0.820 | 0.812 | 0.791 | 0.812 | 3.95% | 33.63% |
| Random Forest | 78.42% | 0.821 | 0.807 | 0.784 | 0.963 | 2.71% | 35.85% |
| **XGBoost** | **79.81%** | 0.829 | 0.819 | **0.798** | **0.970** | 2.81% | 33.34% |
| XGBoost (hyperparameter-tuned) | 79.52% | 0.827 | 0.817 | 0.795 | 0.967 | 2.84% | 33.82% |

*Tuning used RandomizedSearchCV, 30 configs x 5-fold CV (150 fits),
training data only. Best cross-validated training F1: 0.9992 — vs.
0.7949 on the actual test set, confirming the gap is genuine
distribution shift, not a fixable hyperparameter issue.*

### Multiclass classification (11 classes; 11 classes with <50
training samples excluded — see methodology.md)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (OvR macro) |
|---|---:|---:|---:|---:|---:|
| **Decision Tree** | **91.64%** | 0.849 | 0.883 | **0.813** | 0.935 |
| Random Forest | 91.09% | 0.750 | 0.895 | 0.796 | 0.967 |
| XGBoost | 91.31% | 0.758 | 0.890 | 0.800 | **0.977** |

**Finding:** `guess_passwd` misclassified as `normal` in the majority
of cases across all models — NSL-KDD's connection-record features do
not clearly separate credential-guessing traffic from normal
connections.

### Cross-dataset comparison (vs. CICIDS2017)

| Model | CICIDS2017 F1 (macro) | NSL-KDD F1 (macro) | Drop |
|---|---:|---:|---:|
| Logistic Regression | 0.903 | 0.755 | -0.148 |
| Decision Tree | 0.998 | 0.791 | -0.207 |
| Random Forest | 0.998 | 0.784 | -0.214 |
| XGBoost | 0.999 | 0.798 | -0.201 |

Every model dropped 15-21 F1 points, attributed to NSL-KDD's test set
containing attack types absent from training (by design) and its
coarser feature representation.

---

## UNSW-NB15 (independently trained)

### Binary classification (normal vs attack)

*Note: this dataset's official split has attack as the majority class
(68% in training) — the opposite direction from CICIDS2017 and
NSL-KDD.*

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 83.71% | 0.850 | 0.826 | 0.831 | 0.9565 | 27.94% | 6.77% |
| Decision Tree | 88.92% | 0.898 | 0.882 | 0.886 | 0.8993 | 19.21% | 4.45% |
| **Random Forest** | **90.62%** | **0.915** | 0.899 | **0.904** | 0.9840 | 16.96% | 3.19% |
| XGBoost | 87.38% | 0.898 | 0.861 | 0.868 | **0.9837** | 26.04% | **1.66%** |

**Finding:** Random Forest — not XGBoost — is the best model on this
dataset, the only one of the three datasets where XGBoost does not win
binary classification. XGBoost retains the lowest false negative rate
(fewest missed attacks) at the cost of more false positives.

### Multiclass classification (10 classes; all sufficiently sampled,
no filtering needed — minimum class size 130)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (OvR macro) |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 67.52% | 0.488 | 0.610 | 0.477 | 0.867 |
| Random Forest | 67.27% | 0.475 | 0.606 | 0.463 | 0.953 |
| **XGBoost** | **76.63%** | **0.532** | 0.553 | **0.511** | **0.961** |

**Finding:** `Analysis` and `Backdoor` categories are frequently
confused with `Generic`/`Exploits` across all models — a genuine,
well-documented class-overlap issue in UNSW-NB15, not a sample-size
artifact.

---

## Cross-dataset summary

| Dataset | Best binary model | Binary F1 (macro) | Best multiclass model | Multiclass F1 (macro) |
|---|---|---:|---|---:|
| CICIDS2017 | XGBoost | 0.999 | XGBoost | 0.911 |
| NSL-KDD | XGBoost | 0.798 | Decision Tree | 0.813 |
| UNSW-NB15 | Random Forest | 0.904 | XGBoost | 0.511 |

**No single model dominates across all three datasets and tasks.**
This is itself a key finding (RQ1): the best model choice is both
dataset-dependent and task-dependent, undermining any claim that one
algorithm is universally superior for network intrusion detection.

## Class imbalance sanity check (dummy baseline comparison)

| Dataset | Dummy F1 (macro) | Real F1 (macro) | Lift |
|---|---:|---:|---:|
| CICIDS2017 | 0.454 | 0.999 | +54.5pp |
| NSL-KDD | 0.301 | 0.798 | +49.7pp |
| UNSW-NB15 | 0.355 | 0.904 | +54.9pp |

A majority-class-only classifier is mathematically capped at exactly
50% recall (macro), regardless of class imbalance severity. All three
real models exceed this ceiling by 32-50 points, confirming genuine
minority-class discrimination rather than imbalance exploitation.

## Explainability (SHAP, top 3 global features per dataset)

| Dataset | Model explained | Top 3 features |
|---|---|---|
| CICIDS2017 | XGBoost | Bwd Packet Length Std, Init_Win_bytes_backward, Destination Port |
| NSL-KDD | XGBoost | src_bytes, dst_host_srv_count, dst_bytes |
| UNSW-NB15 | Random Forest | sttl, ct_state_ttl, dttl |

No feature names overlap across datasets (each uses distinct feature
engineering), but all three rely on connection-level statistical
aggregates rather than payload content.

## Total models trained

~26 distinct trained models across the project (5 binary + 3
multiclass + 3 feature-optimization variants for CICIDS2017; 4 binary
+ 1 tuned + 3 multiclass for NSL-KDD; 4 binary + 3 multiclass for
UNSW-NB15), each with a saved `.joblib` artifact and a corresponding
logged JSON experiment record under `ml/experiments/results/`.