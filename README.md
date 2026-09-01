# 🛡️ AI-Based Network Intrusion Detection System

A research-oriented Network Intrusion Detection System (NIDS) combining classical machine learning with a full-stack monitoring dashboard. Built with a strict three-layer architecture (React → Node/Express → FastAPI ML service) and research-grade data hygiene: leakage-safe splitting, documented handling of class imbalance, and no fabricated results.

> **Research title:** *A Robust Machine Learning-Based Network Intrusion Detection Framework with Feature Optimization and Cross-Dataset Evaluation*

---

## 🎯 Project Status

**Actively in development.** Dataset pipeline, EDA, and leakage-safe preprocessing are complete and verified against the real CICIDS2017 dataset (2.83M flow records). Model training, the FastAPI inference service, and the dashboard are in progress.

| Phase | Status |
|---|---|
| Project architecture & scaffolding | ✅ Complete |
| Dataset loaders & schema validators | ✅ Complete — tested |
| Exploratory Data Analysis (CICIDS2017) | ✅ Complete |
| Leakage-safe preprocessing pipeline | ✅ Complete — tested |
| Baseline ML models | 🔄 In progress |
| Advanced ML models (RF, XGBoost, SVM) | ⏳ Planned |
| Feature selection & optimization | ⏳ Planned |
| Explainable AI (SHAP) | ⏳ Planned |
| FastAPI inference service | ⏳ Planned |
| Node/Express backend + PostgreSQL | 🔄 Scaffolded |
| React dashboard | 🔄 Scaffolded |

---

## 📊 Real Findings from EDA (2,830,743 flow records)

These are actual measured statistics from the full CICIDS2017 dataset — not estimates:

- **Class imbalance:** 80.3% benign traffic vs 19.7% attack traffic at the binary level; a **206,645:1** ratio between the majority class and the rarest attack type (Heartbleed, 11 samples) at the multiclass level.
- **Data quality artifacts identified and handled:** 308,381 exact duplicate rows (10.9%) removed *before* splitting to prevent train/test leakage; infinite and NaN values in flow-rate features (from division-by-zero in near-zero-duration flows) explicitly cleaned; a small number of physically invalid negative-duration records filtered out.
- **Methodological decision, documented:** three attack classes with fewer than 50 total samples are excluded from multiclass evaluation (though retained in binary classification) because a stratified split would leave 1–4 test samples per class — not enough for a statistically meaningful F1 score. See [`docs/research/methodology.md`](docs/research/methodology.md).

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
Preprocessing → Trained Model → Prediction → Explainability


React never communicates directly with the Python service — every request is proxied through the Node/Express layer, keeping ML inference, business logic, and presentation cleanly separated.

## 🧰 Tech Stack

| Layer | Technologies |
|---|---|
| **Frontend** | React.js, Vite, JavaScript, Tailwind CSS, React Router, Axios, Recharts, Lucide React |
| **Backend** | Node.js, Express.js, PostgreSQL, Helmet, Morgan, express-rate-limit |
| **ML** | Python, Pandas, NumPy, Scikit-learn, XGBoost, imbalanced-learn, SHAP, Joblib |
| **ML API** | FastAPI, Uvicorn, Pydantic |
| **Testing** | Pytest (13 tests passing — schema validation, dataset loading, encoding fixes, leakage-safe splitting) |

## 📁 Repository Structure

network-intrusion-detection/
├── ml/ # Data loaders, validators, EDA, preprocessing, models, FastAPI service
├── web/
│ ├── frontend/ # React dashboard
│ └── backend/ # Express API + PostgreSQL
└── docs/ # Architecture, research methodology, experiment logs, API docs


## 🔬 Engineering Practices

- **Leakage prevention by design:** deduplication and invalid-row cleaning happen *before* any train/test split; scalers, encoders, and resampling are fit on the training split only — never on validation or test data.
- **Reproducibility:** every experiment is configured via `ml/config.yaml` (random seed, split ratios, model list) rather than hardcoded values.
- **Test coverage on real data paths:** loaders and cleaning logic are unit-tested with synthetic edge cases (corrupted encodings, infinite values, negative durations) *and* verified against the full real dataset.
- **No fabricated results:** every metric in this repository comes from an actual executed experiment, logged under `ml/experiments/results/`. Planned/future work is explicitly marked as such.

## 📚 Datasets

- **CICIDS2017** (primary) — [CIC, University of New Brunswick](https://www.unb.ca/cic/datasets/ids-2017.html)
- **NSL-KDD** (secondary benchmark)
- **UNSW-NB15** (cross-dataset generalization)

Datasets are not bundled in this repository (large, license-gated). See [`ml/data/README.md`](ml/data/README.md) for download and setup instructions.

## 🚀 Getting Started

```bash
# ML environment
cd ml
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
pytest tests/ -v             # 13 tests should pass

# Backend
cd ../web/backend
npm install
cp .env.example .env

# Frontend
cd ../frontend
npm install
cp .env.example .env
```

Full setup instructions for datasets, environment variables, and running each service are in each subfolder's README.

## 🗺️ Roadmap

- [ ] Baseline models: Logistic Regression, Decision Tree
- [ ] Advanced models: Random Forest, XGBoost, SVM
- [ ] Feature selection & PCA comparison
- [ ] SHAP-based explainability
- [ ] FastAPI inference service (`/predict`, `/explain`, `/metrics`)
- [ ] Node/Express API + PostgreSQL persistence
- [ ] React SOC-style dashboard
- [ ] Cross-dataset generalization evaluation (NSL-KDD, UNSW-NB15)
- [ ] Research paper–style results writeup

## 📄 License

This project is licensed under the [MIT License](LICENSE).