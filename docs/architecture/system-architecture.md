# System Architecture

## Three-layer architecture

```
React Frontend (Vite + Tailwind, :5173)
        | HTTP / REST
        v
Node.js Backend (Express, :5000)
  - Authentication / validation / routing
  - File upload / API / business logic
  - PostgreSQL (prediction history, alerts)
        | HTTP
        v
Python ML Service (FastAPI, :8000)
  - Preprocessing -> Feature engineering -> Trained model
  - Prediction -> Explainability
```

React never communicates with the Python service directly. All traffic is
proxied through the Node/Express layer.

## Request flow

```
React -> Node/Express -> Python/FastAPI -> Trained model -> Prediction
      <- Node/Express <- Python/FastAPI <-
```

## Environment variables

| Variable          | Used by  | Purpose                                |
|-------------------|----------|-----------------------------------------|
| FRONTEND_URL       | backend  | CORS allow-list                         |
| ML_API_URL         | backend  | Base URL of the FastAPI service         |
| DATABASE_URL       | backend  | PostgreSQL connection string            |
| VITE_NODE_API_URL  | frontend | Base URL of the Node backend            |
| PORT               | backend  | Express listen port                     |

No production URLs are hardcoded anywhere in the codebase.

## Persistence

PostgreSQL, owned entirely by the Node backend:
- Prediction history
- Alerts

Trained models and preprocessing artifacts are **not** stored in the
database — they live under `ml/saved_models/` and are loaded by the FastAPI
service at startup.

## Repository layout

```
network-intrusion-detection/
├── ml/      # datasets, preprocessing, models, training, inference (FastAPI)
├── web/
│   ├── frontend/   # React + Vite + Tailwind
│   └── backend/    # Node.js + Express + PostgreSQL
└── docs/    # architecture, research, experiments, API docs
```

## Development order

1. Project architecture (this document)
2. Dataset pipeline (loaders/validators)
3. EDA
4. Preprocessing (leakage-safe)
5. Baseline ML (Logistic Regression, Decision Tree)
6. Advanced ML (Random Forest, XGBoost, SVM)
7. Feature optimization (selection, PCA comparison)
8. Evaluation (multi-metric, macro/weighted)
9. Explainable AI (SHAP, feature importance)
10. Python ML API (FastAPI)
11. Node/Express backend
12. React + Tailwind frontend
13. Integration
14. Testing
15. Research analysis
16. Final documentation

Deep learning is explicitly out of scope for the current phase; the project
uses classical ML models only.
