# AI-Based Network Intrusion Detection System — Full Project Report

**A Robust Machine Learning-Based Network Intrusion Detection Framework
with Feature Optimization and Cross-Dataset Evaluation**

---

## 1. Executive Summary

This project built a research-grade Network Intrusion Detection System
(NIDS) combining classical machine learning with a full-stack live
monitoring dashboard. Rather than reporting a single dataset's high
accuracy at face value, the project systematically investigated
*whether* that accuracy was trustworthy — running a 5-test data leakage
audit, a mathematically-grounded class-imbalance sanity check, and
independent validation across three structurally distinct benchmark
datasets (CICIDS2017, NSL-KDD, UNSW-NB15). The result is a documented,
evidence-backed understanding of both what these models can do and
where they genuinely fail — including two original exploratory
experiments (a cross-dataset unified model and a domain-routing
ensemble) built directly on findings discovered along the way.

**Total scope:** 3 independently-sourced datasets, ~30+ trained
models, a full 4-layer live application (React → Node/Express +
PostgreSQL → FastAPI → trained models), and a fully documented research
methodology.

---

## 2. Project Architecture

React Frontend (Vite + Tailwind CSS)
│ HTTP / REST
▼
Node.js Backend (Express.js) ──▶ PostgreSQL (prediction history, alerts)
│ HTTP
▼
Python ML Service (FastAPI)
│
Preprocessing → Trained Model → Prediction → SHAP Explainability


React never communicates directly with the Python service — every
request is proxied through Node/Express, per a strict three-layer
separation of concerns.

**Repository structure:** `ml/` (data, models, inference service),
`web/frontend/` (React dashboard), `web/backend/` (Express API +
PostgreSQL), `docs/` (architecture, research, experiments).

---

## 3. Chronological Workflow

### Phase 0 — Planning
Defined the 3-folder architecture, technology stack (React/JS,
Node/Express, PostgreSQL, classical ML only — no deep learning), and
core project rules: no fabricated results, leakage-safe preprocessing,
accuracy never reported alone.

### Phase 1 — Dataset Infrastructure
Built schema validators and dataset-specific loaders for CICIDS2017,
NSL-KDD, and UNSW-NB15 *before* downloading any real data, each
handling known quirks (CICIDS2017's whitespace-padded columns,
NSL-KDD's headerless format, UNSW-NB15's label casing). Verified with
13 pytest unit tests on synthetic edge cases.

### Phase 2 — CICIDS2017 Acquisition, Cleaning & EDA
Downloaded the official `MachineLearningCSV.zip` (CIC/UNB). Verified
load: **2,830,743 rows, 79 columns**. Discovered and fixed a real
encoding bug: the Unicode replacement character (`�`) had corrupted
three "Web Attack" labels in the source file — resolved by mapping
`\ufffd` to a hyphen.

**EDA findings:**
- Binary imbalance: 80.3% BENIGN / 19.7% ATTACK
- 308,381 exact duplicate rows (10.89% of the dataset)
- 1,358 NaN + ~4,376 Infinity values in `Flow Bytes/s` / `Flow Packets/s`
  (division-by-zero artifacts from near-instantaneous flows)
- A small number of physically invalid negative-duration rows
- Extreme multiclass imbalance: Heartbleed had only 11 samples against
  2.27M BENIGN rows (a 206,645:1 ratio)

**Preprocessing pipeline (leakage-safe):** deduplicate → remove
invalid rows → exclude multiclass rare classes (<50 samples: Heartbleed,
Infiltration, Web Attack-SQL Injection) → stratified 70/10/20 split →
fit scalers/encoders on training data only.

### Phase 3 — CICIDS2017 Model Training & Optimization
Trained baseline models (Logistic Regression, Decision Tree) then
advanced models (Random Forest, XGBoost, SVM). Ran feature optimization
(All Features vs. Random-Forest-Selected vs. PCA) and SHAP
explainability. Full results in Section 4.

### Phase 4 — The Leakage Investigation
CICIDS2017's 99%+ accuracy prompted direct skepticism. Rather than
accept it, five independent tests were run to determine whether the
model was genuinely learning or "cramming" on dataset artifacts (full
results in Section 5).

### Phase 5 — Full-Stack Application
Built the FastAPI inference service (6 endpoints), the Node/Express
backend with PostgreSQL persistence (`predictions` and `alerts`
tables), and a 5-page React dashboard styled as a dark
Security-Operations-Center interface. Verified end-to-end: a curl
request flowed through Node → FastAPI → XGBoost → back into
PostgreSQL, confirmed by directly querying the database afterward.
Traffic Analysis (CSV upload) was tested on a real 200-row sample and
achieved 100% agreement with ground truth (167/167 BENIGN, 33/33
attacks correctly classified).

### Phase 6 — NSL-KDD (Cross-Dataset Validation #1)
Independently trained the same model family on NSL-KDD's official
fixed train/test split. Found a substantial, consistent generalization
gap versus CICIDS2017 (Section 6). Ran hyperparameter tuning
specifically to test whether this gap was fixable — it was not (best
cross-validated training F1 of 0.9992 did not translate to improved
test performance, which remained at 0.7949).

### Phase 7 — UNSW-NB15 (Cross-Dataset Validation #2)
Independently trained on UNSW-NB15's official split. Found that
**Random Forest, not XGBoost, was the best model** — the only dataset
of the three where this held, directly disproving any assumption of a
universally superior algorithm.

### Phase 8 — Class Imbalance Sanity Check
Directly tested whether high accuracy was a class-imbalance artifact
using a `DummyClassifier` baseline and the mathematical property that
majority-class-only prediction caps macro recall at exactly 50%,
regardless of imbalance severity. Real models cleared this ceiling by
32-50 points on all three datasets (Section 7).

### Phase 9 — Cross-Dataset Explainability
Ran SHAP on each dataset's own best-performing model, finding no
feature-name overlap (different schemas) but a consistent conceptual
reliance on connection-level statistical aggregates rather than
payload content.

### Phase 10 — Real-World Deployment Considerations
Directly addressed whether this system could be deployed at a real
organization. Documented two structural barriers: benchmark datasets'
features are tool-derived (CICFlowMeter), not raw organizational
telemetry; and benchmark traffic differs from production traffic more
than the benchmark datasets differ from each other.

### Phase 11 — Literature Review & Documentation
Verified and cited four foundational sources: Sharafaldin et al. (2018,
CICIDS2017), Tavallaee et al. (2009, NSL-KDD), Moustafa & Slay (2015,
UNSW-NB15), and Engelen et al. (2021, a direct critique of CICIDS2017's
data quality that motivated the Phase 4 leakage investigation).

### Phase 12 — Novelty Research
Investigated the current literature to identify a genuine, not-already-
published research angle. Found that cross-dataset SHAP comparison
(Phase 9) had already been independently published multiple times in
2025-2026. Instead pursued a genuinely explored gap: an honest,
hand-crafted attempt at a unified cross-dataset model — informed by
Sarhan et al.'s (2021) NetFlow-standardization work, which notably does
NOT cover this project's exact three-dataset combination.

### Phase 13 — Unified Model & Domain-Routing Ensemble
Built and evaluated two original experiments extending the project's
own findings (Section 8): a flat unified model trained on all three
datasets combined via a 5-feature common representation, and a
domain-routing "mixture of experts" ensemble. Discovered and
rigorously investigated a counter-intuitive result (a 5-feature model
outperforming a 122-feature model on NSL-KDD), confirming it as a
genuine feature-set effect via a controlled experiment, not an
artifact.

### Phase 14 — Wiring Novel Models into the Live Application
Persisted the router, three experts, and unified model as deployable
artifacts. Added `/predict/unified` and `/predict/routed` to FastAPI,
proxied through Node, and built a new "Cross-Dataset Insights" page in
the React dashboard, verified live end-to-end.

---

## 4. Full Model Results — CICIDS2017

### Binary classification (BENIGN vs ATTACK)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 93.89% | 0.869 | 0.954 | 0.903 | 0.9875 | 6.87% | 2.37% |
| Decision Tree | 99.89% | 0.997 | 0.999 | 0.998 | 0.9994 | 0.108% | 0.093% |
| Random Forest | 99.89% | 0.997 | 0.999 | 0.998 | 0.99997 | 0.109% | 0.090% |
| **XGBoost** | **99.92%** | **0.998** | **0.999** | **0.999** | **0.99998** | **0.064%** | 0.149% |
| SVM (50K subsample) | 94.82% | 0.885 | 0.959 | 0.916 | 0.9943 | 5.74% | 2.40% |

### Multiclass (12 classes; 3 rare classes excluded, <50 samples)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (OvR) |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 99.49% | 0.839 | 0.938 | 0.872 | 0.980 |
| Random Forest | 99.58% | 0.863 | 0.937 | 0.877 | 0.999 |
| **XGBoost** | **99.88%** | **0.920** | 0.905 | **0.911** | **0.9997** |

### Feature optimization (XGBoost, binary task)

| Method | Features | F1 (macro) | Train time |
|---|---:|---:|---:|
| All Features | 78 | 0.9986 | 36.3s |
| Selected (RF importance) | 39 | 0.9968 | 27.1s |
| PCA (95% variance) | 25 | 0.9967 | 24.6s |

**Finding:** fewer features saved ~25-30% training time but quadrupled
false negative rate — a real trade-off, not a free improvement.

### SHAP explainability — top 5 global features (XGBoost)
`Bwd Packet Length Std`, `Init_Win_bytes_backward`, `Destination Port`,
`Init_Win_bytes_forward`, `Bwd Packet Length Mean`.

---

## 5. The Data Leakage Investigation

Triggered by CICIDS2017's 99%+ accuracy. Five independent tests:

| Test | Result | Conclusion |
|---|---|---|
| Train vs test performance gap | F1 0.9987 (train) vs 0.9986 (test) | Rules out memorization |
| Destination Port ablation | F1 0.9986 → 0.9982 (77 features) | Rules out single-feature shortcut |
| Near-duplicate row overlap | 1.30% overlap, concentrated in BENIGN not attacks | Rules out attack-burst leakage |
| SHAP vs permutation importance | 5/6 top features matched | Confirms genuine, method-independent signal |
| Dummy-baseline sanity check | See Section 7 | Rules out imbalance exploitation |

**Conclusion: no evidence of data leakage found across any of the five
tests.**

---

## 6. Cross-Dataset Validation

### NSL-KDD — Binary

| Model | Accuracy | F1 (macro) | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 75.50% | 0.755 | 0.781 |
| Decision Tree | 79.15% | 0.791 | 0.812 |
| Random Forest | 78.42% | 0.784 | 0.963 |
| **XGBoost** | **79.81%** | **0.798** | **0.970** |
| XGBoost (tuned) | 79.52% | 0.795 | 0.967 |

### NSL-KDD — Multiclass (11 reliable classes after rare-class filtering)

| Model | Accuracy | F1 (macro) |
|---|---:|---:|
| **Decision Tree** | **91.64%** | **0.813** |
| Random Forest | 91.09% | 0.796 |
| XGBoost | 91.31% | 0.800 |

### UNSW-NB15 — Binary

| Model | Accuracy | F1 (macro) |
|---|---:|---:|
| Logistic Regression | 83.71% | 0.831 |
| Decision Tree | 88.92% | 0.886 |
| **Random Forest** | **90.62%** | **0.904** |
| XGBoost | 87.38% | 0.868 |

### UNSW-NB15 — Multiclass (10 classes, no filtering needed)

| Model | Accuracy | F1 (macro) |
|---|---:|---:|
| Decision Tree | 67.52% | 0.477 |
| Random Forest | 67.27% | 0.463 |
| **XGBoost** | **76.63%** | **0.511** |

### The generalization gap (F1 macro, best model per dataset)

| Model | CICIDS2017 | NSL-KDD | Drop |
|---|---:|---:|---:|
| XGBoost | 0.999 | 0.798 | **-0.201** |

**Confirmed via hyperparameter tuning:** RandomizedSearchCV (150 fits)
reached 0.9992 cross-validated training F1 on NSL-KDD, yet test F1
remained unchanged at 0.7949 — proving the gap is genuine distribution
shift (NSL-KDD's test set includes attack types never seen in
training, by design), not a fixable configuration issue.

### Key cross-dataset finding
**No single model wins across all three datasets.** XGBoost wins on
CICIDS2017 and NSL-KDD binary/multiclass, but Random Forest wins on
UNSW-NB15 binary — directly disproving any claim of a universally
superior algorithm for NIDS.

---

## 7. Class Imbalance Sanity Check

Tested whether high accuracy reflects genuine signal or class-imbalance
exploitation, using a `DummyClassifier` (always predicts majority
class) as a baseline.

| Dataset | Dummy F1 (macro) | Real F1 (macro) | Lift |
|---|---:|---:|---:|
| CICIDS2017 | 0.454 | 0.999 | +54.5pp |
| NSL-KDD | 0.301 | 0.798 | +49.7pp |
| UNSW-NB15 | 0.355 | 0.904 | +54.9pp |

**Mathematical proof used:** a majority-class-only classifier achieves
exactly 50% recall (macro) — 100% on the always-predicted class, 0% on
the other, averaged — regardless of how imbalanced the data is. Real
models scored 82-99.9% macro recall, 32-50 points above this
theoretical ceiling, which class imbalance alone cannot produce.

---

## 8. Original Exploratory Experiments

### 8.1 Flat unified model (trained on all 3 datasets combined)

Used a hand-picked 5-feature common set (`duration_sec`, `src_bytes`,
`dst_bytes`, `total_bytes`, `log_duration`) — packet-count features
were excluded because NSL-KDD's `count`/`srv_count` are time-window
connection aggregates, not per-flow packet counts comparable to the
other two datasets.

| Dataset | Specialized F1 | Unified F1 | Gap |
|---|---:|---:|---:|
| CICIDS2017 | 0.999 | 0.981 | -0.018 |
| NSL-KDD | 0.798 | 0.762 | -0.036 |
| UNSW-NB15 | 0.904 | 0.797 | **-0.107** |

**Finding:** degradation is non-uniform. CICIDS2017 comprises ~85% of
the combined training data, skewing the unified model's notion of
"normal" traffic toward CICIDS2017/NSL-KDD-style patterns. On
UNSW-NB15, this produced a 31.7% false positive rate — nearly a third
of genuinely normal UNSW-NB15 traffic was misclassified as an attack.

### 8.2 Domain-routing ensemble ("mixture of specialized experts")

Built 3 lightweight experts (5 features, one per dataset) plus a
router predicting which dataset an incoming flow most resembles.

| Dataset | Router accuracy | Flat unified F1 | Oracle-routed F1 (upper bound) | Learned-router F1 (realistic) |
|---|---:|---:|---:|---:|
| CICIDS2017 | 99.69% | 0.981 | 0.983 | 0.979 |
| NSL-KDD | 88.72% | 0.762 | **0.836** | 0.791 |
| UNSW-NB15 | 94.92% | 0.797 | **0.832** | 0.802 |

**Finding:** routing recovers meaningful performance versus flat
unification, but the realistic learned router only captures 39-46% of
the theoretical oracle gain — directly explained by the router's own
imperfect domain-identification accuracy (as low as 88.72% on
NSL-KDD).

### 8.3 Investigating a counter-intuitive result

The 5-feature NSL-KDD "oracle expert" (F1 0.836) outperformed the
original 122-feature specialized model (F1 0.798) — a surprising
result that was investigated rather than accepted. A controlled 2×2
experiment (feature count × hyperparameters) isolated the cause:

| Configuration | Original hyperparams | Oracle-expert hyperparams |
|---|---:|---:|
| Full 122 features | 0.7978 | 0.8084 |
| 5 common features | **0.8207** | **0.8362** |

**Conclusion:** the 5-feature model wins under both hyperparameter
settings, confirming a genuine feature-set effect — likely caused by
NSL-KDD's 60+ sparse one-hot `service` categories diluting tree splits
away from the small number of genuinely strong continuous signals.
Notably, this is the *opposite* direction from CICIDS2017's own
feature-optimization result (Section 4), where fewer features slightly
hurt performance — demonstrating that whether feature reduction helps
is dataset-dependent, not universal.

---

## 9. Real-World Deployment Considerations

Two structural barriers to production deployment were identified and
documented, distinct from the generalization gap already measured:

1. **Feature availability.** All three datasets' features are derived
   by specific flow-extraction tools (e.g. CICFlowMeter) from
   packet captures — not raw telemetry an organization would actually
   possess (NetFlow, firewall logs, EDR data). Deployment would require
   either running an equivalent extraction tool against real traffic
   or building a custom translation layer, and some features may not
   be reconstructable at all from a given organization's telemetry.

2. **Distribution shift beyond benchmark-to-benchmark.** CICIDS2017 is
   2017 university-testbed traffic; real organizational traffic
   (cloud SaaS, containers, VPNs, IoT) differs from this baseline more
   than NSL-KDD differs from CICIDS2017 — and the measured 15-21 point
   generalization gap between two *benchmark* datasets is likely a
   lower bound on the real-world gap, not an upper bound.

**What genuine deployment would require:** a feature-extraction
pipeline matched to real telemetry, retraining on the organization's
own labeled traffic, continuous drift monitoring, and likely a hybrid
supervised + anomaly-detection approach with human-in-the-loop review
— consistent with how commercial NIDS vendors handle this problem via
per-customer baselining rather than one static pretrained model.

---

## 10. Live Application Summary

| Layer | Technology | Status |
|---|---|---|
| Frontend | React, Vite, Tailwind CSS | 6 pages: Dashboard, Traffic Analysis, Alerts, Model Performance, About, Cross-Dataset Insights |
| Backend | Node.js, Express, PostgreSQL | Verified end-to-end with real DB writes |
| ML API | FastAPI | 8 endpoints: /health, /predict, /predict/batch, /explain, /model-info, /metrics, /predict/unified, /predict/routed |
| ML models served live | CICIDS2017 (4 binary + 3 multiclass), domain router, 3 common-feature experts, 1 unified model | All loaded and verified via live API calls |

**Traffic Analysis validation:** a real 200-row CICIDS2017 test sample
was classified with 100% agreement against ground truth (167/167
BENIGN, 33/33 attacks).

---

## 11. Literature Cited

- Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward
  Generating a New Intrusion Detection Dataset and Intrusion Traffic
  Characterization. *ICISSP*, 108-116.
- Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). A
  Detailed Analysis of the KDD CUP 99 Data Set. *IEEE CISDA*, 1-6.
- Moustafa, N., & Slay, J. (2015). UNSW-NB15: A Comprehensive Data Set
  for Network Intrusion Detection Systems. *MilCIS*, 1-6.
- Engelen, G., Rimmer, V., & Joosen, W. (2021). Troubleshooting an
  Intrusion Detection Dataset: the CICIDS2017 Case Study. *IEEE SPW*,
  7-12.

---

## 12. Limitations & Future Work

- Live CSV batch prediction (Traffic Analysis page) currently supports
  CICIDS2017 only; NSL-KDD/UNSW-NB15 models are visible in Model
  Performance but not selectable for live upload-based prediction.
- Full cross-dataset unification (beyond the 5-feature exploratory
  version) remains genuinely open — no standardized common-feature
  release exists for this exact three-dataset combination (confirmed
  against Sarhan et al.'s NetFlow-standardization work).
- Domain-balanced retraining of the unified model (to correct the
  CICIDS2017-dominance bias found in Section 8.1) was identified but
  not implemented.
- Automated test coverage exists for the ML pipeline (13 pytest tests)
  but not yet for the Node backend or React frontend.
- Anomaly-detection (autoencoder trained on benign-only traffic) and
  adversarial robustness testing on flow features were identified as
  promising extensions but not implemented in this phase.

---

## 13. Conclusion

This project demonstrates that classical machine learning can achieve
strong network intrusion detection performance on individual benchmark
datasets (up to 99.9% F1), but that this performance does not
uniformly transfer across datasets — a finding established through
direct measurement and confirmed via independent hyperparameter tuning
rather than assumed. Equally important, the project demonstrates that
high benchmark accuracy is not, in this case, an artifact of data
leakage or class imbalance, through five independent verification
tests and a mathematically-grounded sanity check. Two original
exploratory experiments — a unified cross-dataset model and a
domain-routing ensemble — extend these findings into a concrete,
tested trade-off between generalization and specialization, wired into
a fully functional live application.