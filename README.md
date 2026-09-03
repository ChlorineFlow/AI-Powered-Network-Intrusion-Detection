# 🛡️ AI-Based Network Intrusion Detection System

A research-oriented Network Intrusion Detection System (NIDS) combining classical machine learning with a full-stack monitoring dashboard. Built with a strict three-layer architecture (React → Node/Express + PostgreSQL → Python/FastAPI ML service) and research-grade data hygiene: leakage-safe splitting, documented class-imbalance handling, cross-dataset validation, and no fabricated results.

> **Research title:** *A Robust Machine Learning-Based Network Intrusion Detection Framework with Feature Optimization and Cross-Dataset Evaluation*

---

## 🎯 Project Status

**Feature-complete and fully functional end-to-end.** Every layer — ML pipeline, inference API, backend, and dashboard — is built, tested, and verified with real data.

| Component | Status |
|---|---|
| Dataset loaders, validators, EDA (CICIDS2017) | ✅ Complete |
| Leakage-safe preprocessing pipeline | ✅ Complete |
| Baseline + advanced ML models (binary & multiclass) | ✅ Complete |
| Feature optimization (All Features vs Selected vs PCA) | ✅ Complete |
| SHAP explainability + permutation importance verification | ✅ Complete |
| Data leakage investigation (5 independent tests) | ✅ Complete |
| Cross-dataset validation (NSL-KDD) + hyperparameter tuning | ✅ Complete |
| FastAPI inference service (6 endpoints) | ✅ Complete |
| Node.js/Express backend + PostgreSQL | ✅ Complete |
| React dashboard (5 pages) | ✅ Complete |
| UNSW-NB15 (third dataset) | ⏳ Not yet started |
| Automated backend/frontend tests | ⏳ Partial (ML: 13 pytest tests passing) |

---

## 📊 Real Results — Nothing Fabricated

### Binary classification (CICIDS2017, leakage-safe split)

| Model | Accuracy | F1 (macro) | ROC-AUC | FPR | FNR |
|---|---|---|---|---|---|
| Logistic Regression | 93.89% | 0.903 | 0.9875 | 6.87% | 2.37% |
| Decision Tree | 99.89% | 0.998 | 0.9994 | 0.108% | 0.093% |
| Random Forest | 99.89% | 0.998 | 0.99997 | 0.109% | 0.090% |
| **XGBoost** | **99.92%** | **0.999** | **0.99998** | **0.064%** | **0.149%** |

### Cross-dataset generalization (NSL-KDD, independently trained)

| Model | Accuracy | F1 (macro) |
|---|---|---|
| Logistic Regression | 75.50% | 0.755 |
| Decision Tree | 79.15% | 0.791 |
| Random Forest | 78.42% | 0.784 |
| XGBoost | 79.81% | 0.798 |

**Every model dropped 15-21 F1 points moving from CICIDS2017 to NSL-KDD** — confirmed via independent hyperparameter tuning (near-perfect 0.999 cross-validated training score, unchanged ~0.795 test score) to be a genuine train/test distribution shift, not a fixable configuration issue. This finding is documented in full in [`docs/research/methodology.md`](docs/research/methodology.md), including a 5-test investigation that explicitly ruled out data leakage as the cause of CICIDS2017's high scores (exact-duplicate removal, train/test performance gap analysis, feature-ablation testing, near-duplicate-flow overlap checking, and cross-method feature-importance verification via SHAP + permutation importance).

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
Preprocessing → Trained Model (XGBoost) → Prediction → SHAP Explainability


React never communicates directly with the Python service — every request is proxied through the Node/Express layer.

## 🖥️ Dashboard

Five fully functional pages, all backed by live data:
- **Dashboard** — real-time attack ratio, traffic totals, active model, attack distribution, recent alerts
- **Traffic Analysis** — upload a CSV, run live batch predictions through the actual trained XGBoost model
- **Alerts** — filterable list of detected attacks with severity and status
- **Model Performance** — real experiment comparisons across all 4 trained models (binary + multiclass)
- **About** — project summary and documented limitations

## 🧰 Tech Stack

| Layer | Technologies |
|---|---|
| **Frontend** | React.js, Vite, JavaScript, Tailwind CSS, React Router, Axios, Recharts, Lucide React, PapaParse |
| **Backend** | Node.js, Express.js, PostgreSQL, Helmet, Morgan, express-rate-limit |
| **ML** | Python, Pandas, NumPy, Scikit-learn, XGBoost, imbalanced-learn, SHAP, Joblib |
| **ML API** | FastAPI, Uvicorn, Pydantic |
| **Testing** | Pytest (13 tests — schema validation, dataset loading, encoding fixes, leakage-safe splitting) |

## 📁 Repository Structure

network-intrusion-detection/
├── ml/ # Data loaders, validators, EDA, preprocessing, models, FastAPI service
├── web/
│ ├── frontend/ # React dashboard (5 pages)
│ └── backend/ # Express API + PostgreSQL
└── docs/ # Architecture, research methodology, experiment logs, API docs


## 🔬 Engineering Practices

- **Leakage prevention by design:** deduplication and invalid-row cleaning happen before any train/test split; scalers, encoders, and feature selectors are fit on the training split only.
- **Rigor over convenient numbers:** a 99%+ accuracy result on CICIDS2017 triggered a 5-test independent leakage investigation and a full cross-dataset validation on NSL-KDD before being trusted — not just reported at face value.
- **Reproducibility:** every experiment is configured via `ml/config.yaml` and logged with full traceability (model, hyperparameters, random seed, timing, metrics) under `ml/experiments/results/`.
- **No fabricated results:** every metric in this repository and dashboard comes from an actual executed, logged experiment. Known limitations (generalization gap, rare-class exclusion, SVM subsampling) are explicitly documented rather than hidden.

## 📚 Datasets

- **CICIDS2017** (primary) — [CIC, University of New Brunswick](https://www.unb.ca/cic/datasets/ids-2017.html)
- **NSL-KDD** (cross-dataset validation) — trained and evaluated independently
- **UNSW-NB15** (planned, not yet integrated)

Datasets are not bundled in this repository (large, license-gated). See [`ml/data/README.md`](ml/data/README.md) for download and setup instructions.

## 🚀 Getting Started

Three services must run simultaneously:

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
cp .env.example .env   # configure DATABASE_URL
node src/db/migrate.js  # create tables (run once)
node src/app.js

# Terminal 3 — Frontend (React)
cd web/frontend
npm install
cp .env.example .env
npm run dev
```

Then open `http://localhost:5173`.

## 📖 Key Documentation

- [`docs/research/methodology.md`](docs/research/methodology.md) — full methodology, including the data leakage investigation and cross-dataset generalization findings
- [`docs/research/research-questions.md`](docs/research/research-questions.md) — RQ1-RQ7 and their answers
- [`docs/architecture/system-architecture.md`](docs/architecture/system-architecture.md) — full system design

## 🗺️ Roadmap

- [x] Baseline models: Logistic Regression, Decision Tree
- [x] Advanced models: Random Forest, XGBoost, subsampled SVM
- [x] Feature selection & PCA comparison
- [x] SHAP-based explainability + permutation importance verification
- [x] Data leakage investigation (5 independent tests)
- [x] Cross-dataset validation on NSL-KDD + hyperparameter tuning
- [x] FastAPI inference service
- [x] Node/Express API + PostgreSQL persistence
- [x] React SOC-style dashboard (5 pages)
- [ ] UNSW-NB15 integration
- [ ] Unified cross-dataset model (documented as future work — see methodology.md)
- [ ] Automated backend/frontend test coverage
- [ ] Research paper–style final writeup

## 📄 License

This project is licensed under the [MIT License](LICENSE).