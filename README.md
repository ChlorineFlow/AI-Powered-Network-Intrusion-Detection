# 🛡️ AI-Based Network Intrusion Detection System

**A research-grade, full-stack Network Intrusion Detection System** combining classical machine learning with a live monitoring dashboard — built with the same rigor expected in published NIDS research: leakage-safe evaluation, cross-dataset validation, explainability, and honestly documented limitations.

> **Research title:** *A Robust Machine Learning-Based Network Intrusion Detection Framework with Feature Optimization and Cross-Dataset Evaluation*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![React](https://img.shields.io/badge/React-18-61DAFB)
![Node.js](https://img.shields.io/badge/Node.js-Express-green)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-database-336791)

---

## ✨ Highlights

- 🎯 **3 independently-evaluated benchmark datasets** — CICIDS2017, NSL-KDD, UNSW-NB15 — no dataset shortcuts, no shared/transferred models
- 🧠 **~26 trained models** across binary and multiclass tasks, baseline through advanced (Logistic Regression → Decision Tree → Random Forest → XGBoost → SVM)
- 🔬 **A 5-test data leakage investigation** run specifically because 99%+ accuracy looked "too good" — exact-duplicate removal, train/test gap analysis, feature ablation, near-duplicate-flow overlap check, cross-method SHAP + permutation importance verification
- 📉 **Honest cross-dataset generalization findings** — every model dropped 15-21 F1 points moving from CICIDS2017 to NSL-KDD, confirmed via hyperparameter tuning to be genuine distribution shift, not a fixable config issue
- 🧮 **Mathematically-grounded imbalance sanity check** — real models exceed the 50% macro-recall ceiling that class-imbalance exploitation alone can never cross, by 32-50 points, across all three datasets
- 🖥️ **A fully functional 4-layer live system** — React dashboard → Express/PostgreSQL → FastAPI → XGBoost, verified end-to-end with real predictions
- 📊 **SHAP explainability**, cross-validated against permutation importance, run independently on all three datasets

This isn't a tutorial clone — every number here comes from an executed, logged experiment, and every limitation is disclosed rather than hidden.

---

## 🏗️ Architecture

React Frontend (Vite + Tailwind CSS)
│ HTTP / REST
▼
Node.js Backend (Express.js) ──▶ PostgreSQL (prediction history, alerts)
│ HTTP
▼
Python ML Service (FastAPI)
│
Preprocessing → Trained Model → Prediction → SHAP Explainability


Strict separation of concerns: React never talks to Python directly — every request is proxied through Node/Express, per a mandated three-layer architecture.

---

## 📊 Real Results (Nothing Fabricated)

Full breakdown for every model, every dataset, every task: **[`docs/experiments/results-analysis.md`](docs/experiments/results-analysis.md)**

### Best model per dataset — no single algorithm dominates

| Dataset | Best binary model | F1 (macro) | Best multiclass model | F1 (macro) |
|---|---|---:|---|---:|
| CICIDS2017 | XGBoost | **0.999** | XGBoost | 0.911 |
| NSL-KDD | XGBoost | 0.798 | Decision Tree | 0.813 |
| UNSW-NB15 | **Random Forest** | 0.904 | XGBoost | 0.511 |

*Random Forest — not XGBoost — wins on UNSW-NB15, a genuine, non-cherry-picked finding that directly challenges "just use XGBoost" assumptions.*

### The generalization gap — measured, not assumed

| Model | CICIDS2017 F1 | NSL-KDD F1 | Drop |
|---|---:|---:|---:|
| XGBoost | 0.999 | 0.798 | **-0.201** |
| Random Forest | 0.998 | 0.784 | -0.214 |
| Decision Tree | 0.998 | 0.791 | -0.207 |

Confirmed via independent hyperparameter tuning: best cross-validated training F1 reached **0.9992**, yet test F1 stayed at **0.7949** — proving the gap is genuine distribution shift, not a tunable configuration problem.

### Is the high accuracy just class imbalance? Tested, not assumed.

| Dataset | Dummy baseline F1 | Real model F1 | Lift |
|---|---:|---:|---:|
| CICIDS2017 | 0.454 | 0.999 | **+54.5 points** |
| NSL-KDD | 0.301 | 0.798 | +49.7 points |
| UNSW-NB15 | 0.355 | 0.904 | +54.9 points |

A majority-class-only classifier is mathematically capped at exactly 50% macro recall. Real models clear this ceiling by 32-50 points on all three datasets — evidence that cannot be produced by imbalance exploitation alone.

---

## 🖥️ Live Dashboard

Five fully functional pages, all backed by real data flowing through the entire stack:

| Page | What it does |
|---|---|
| **Dashboard** | Real-time attack ratio, traffic totals, active model, attack distribution, recent alerts |
| **Traffic Analysis** | Upload a CSV → live batch prediction through the actual trained XGBoost model |
| **Alerts** | Filterable list of detected attacks with severity and status |
| **Model Performance** | Real experiment comparisons across all 4 trained models |
| **About** | Project summary, generalization findings, and real-world deployment limitations |

Styled as a dark SOC (Security Operations Center) interface — not a generic SaaS template.

---

## 🧰 Tech Stack

| Layer | Technologies |
|---|---|
| **Frontend** | React.js, Vite, JavaScript, Tailwind CSS, React Router, Axios, Recharts, Lucide React, PapaParse |
| **Backend** | Node.js, Express.js, PostgreSQL, Helmet, Morgan, express-rate-limit |
| **ML** | Python, Pandas, NumPy, Scikit-learn, XGBoost, imbalanced-learn, SHAP, Joblib |
| **ML API** | FastAPI, Uvicorn, Pydantic |
| **Testing** | Pytest (13+ tests — schema validation, dataset loading, encoding fixes, leakage-safe splitting) |

---

## 🔬 Engineering & Research Practices

- **Leakage prevention by design** — deduplication and invalid-row cleaning happen before any train/test split; scalers, encoders, and feature selectors fit on training data only.
- **Skepticism toward good-looking results** — a 99%+ accuracy score triggered a full independent investigation rather than being reported at face value.
- **Reproducibility** — every experiment configured via `ml/config.yaml`, logged with full traceability (model, hyperparameters, random seed, timing, metrics) under `ml/experiments/results/`.
- **Documented, not hidden, limitations** — generalization gaps, rare-class exclusions, SVM subsampling, and real-world deployment barriers are all explicitly written up, not glossed over.
- **Multi-dataset validation** — every model retrained from scratch per dataset; conclusions checked for consistency across three independently-sourced, differently-structured datasets before being trusted.

---

## 📚 Datasets

| Dataset | Role | Rows (train/test) |
|---|---|---|
| **CICIDS2017** | Primary | 1.76M / 0.50M (post-cleaning) |
| **NSL-KDD** | Cross-dataset validation | 126K / 22.5K |
| **UNSW-NB15** | Cross-dataset validation | 175K / 82K |

Datasets are not bundled in this repository (large, license-gated). See [`ml/data/README.md`](ml/data/README.md) for download and setup instructions.

---

## 🚀 Getting Started

Three services run simultaneously:

```bash
# Terminal 1 — ML inference service (FastAPI)
cd ml
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
uvicorn src.inference.app:app --reload --port 8000

# Terminal 2 — Backend (Express + PostgreSQL)
cd web/backend
npm install
cp .env.example .env         # configure DATABASE_URL
node src/db/migrate.js       # create tables (run once)
node src/app.js

# Terminal 3 — Frontend (React)
cd web/frontend
npm install
cp .env.example .env
npm run dev
```

Then open `http://localhost:5173`.

---

## 📖 Key Documentation

- **[`docs/experiments/results-analysis.md`](docs/experiments/results-analysis.md)** — every model, every metric, every dataset, in one place
- **[`docs/research/methodology.md`](docs/research/methodology.md)** — full methodology: leakage investigation, cross-dataset findings, deployment considerations
- **[`docs/research/research-questions.md`](docs/research/research-questions.md)** — RQ1-RQ7, answered with evidence
- **[`docs/architecture/system-architecture.md`](docs/architecture/system-architecture.md)** — full system design

---

## 🗺️ Roadmap

- [x] Baseline + advanced models across 3 independent datasets
- [x] Feature optimization (All Features vs Selected vs PCA)
- [x] SHAP explainability + permutation importance verification (all 3 datasets)
- [x] 5-test data leakage investigation
- [x] Cross-dataset generalization study + hyperparameter tuning proof
- [x] Class-imbalance mathematical sanity check
- [x] Full-stack app: FastAPI + Node/Express + PostgreSQL + React (5 pages)
- [ ] Wire NSL-KDD/UNSW-NB15 models into the live dashboard/API (currently CICIDS2017 only)
- [ ] Unified cross-dataset model (documented as future work — see methodology.md)
- [ ] Automated backend/frontend test coverage
- [ ] Anomaly-detection model (autoencoder) for novel-attack detection
- [ ] Statistical significance testing between models (McNemar's test)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).