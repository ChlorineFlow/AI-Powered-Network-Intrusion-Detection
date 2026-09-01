# AI-Based Network Intrusion Detection System

Research-oriented NIDS built with classical machine learning, a FastAPI inference
service, a Node.js/Express backend, and a React dashboard.

**Full title:** A Robust Machine Learning-Based Network Intrusion Detection
Framework with Feature Optimization and Cross-Dataset Evaluation.

## Repository layout

```
network-intrusion-detection/
├── ml/      # Datasets, preprocessing, feature engineering, model training,
│            # evaluation, explainability, and the FastAPI inference service
├── web/
│   ├── frontend/   # React (JavaScript) + Vite + Tailwind CSS dashboard
│   └── backend/    # Node.js + Express API orchestration layer + PostgreSQL
└── docs/    # Architecture, research, experiment, and API documentation
```

## Architecture

```
React (Vite, :5173)
      ↓ HTTP/REST
Node.js / Express (:5000)  ──▶ PostgreSQL (prediction history, alerts)
      ↓ HTTP
Python / FastAPI ML service (:8000)
      ↓
Trained classical ML model + preprocessing pipeline
```

React never talks to the Python service directly — all requests are proxied
through the Node/Express layer.

## Technology stack

- **Frontend:** React.js, Vite, JavaScript, Tailwind CSS, React Router, Axios,
  Recharts, Lucide React
- **Backend:** Node.js, Express.js, JavaScript, PostgreSQL, Axios, dotenv,
  CORS, Helmet, morgan
- **ML:** Python, Pandas, NumPy, Scikit-learn, XGBoost, imbalanced-learn,
  Matplotlib, Seaborn, SHAP, Joblib
- **ML API:** FastAPI, Uvicorn, Pydantic

Deep learning is out of scope for now — classical ML models only
(Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM).

## Status

Phase 1 complete: repository scaffolding only. No models have been trained,
no datasets are bundled, and no dashboard statistics are wired up yet. See
`docs/architecture/` for the full plan and `ml/data/README.md` for dataset
setup instructions.

## Development order

See `docs/research/methodology.md` and the phase plan in
`docs/architecture/system-architecture.md` for the mandated build order:
architecture → dataset pipeline → EDA → preprocessing → baseline ML →
advanced ML → feature optimization → evaluation → explainability →
FastAPI → Node/Express → React → integration → testing → research
analysis → documentation.
