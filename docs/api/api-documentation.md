# API Documentation

Two API layers exist, per the mandated three-layer architecture: the
Python FastAPI ML service (internal — not intended for direct external
use) and the Node/Express backend (the actual public-facing API,
proxying to FastAPI).

## FastAPI ML service (`ml/src/inference/app.py`, default port 8000)

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Reports loaded model names for binary/multiclass tasks |
| GET | `/model-info` | Default model, available models, training dataset note |
| GET | `/metrics` | Latest logged experiment result per (dataset, task, model) |
| POST | `/predict` | Single-row binary prediction. Body: `{"features": {...}, "model": "xgboost"}` |
| POST | `/predict/batch` | Batch binary prediction. Body: `{"rows": [{...}, ...], "model": "xgboost"}` |
| POST | `/explain` | SHAP top-contributing-features for one prediction (tree-based models only) |

`/predict` and `/predict/batch` return `prediction` ("BENIGN"/"ATTACK"),
`confidence` (probability of the predicted class), and `model_used`.
Missing required feature columns return HTTP 422 with the list of
missing columns (truncated to 10). An unrecognized `model` name returns
HTTP 400.

## Node/Express backend (`web/backend/src/app.js`, default port 5000)

All routes are prefixed with `/api` (except the top-level `/health`).

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Backend liveness check |
| POST | `/api/predict` | Proxies to FastAPI `/predict`; persists result to `predictions` table (and `alerts` if ATTACK) |
| POST | `/api/predict/batch` | Proxies to FastAPI `/predict/batch`; persists each row's result |
| POST | `/api/explain` | Proxies to FastAPI `/explain` |
| GET | `/api/predictions?limit=N` | Recent prediction history from PostgreSQL |
| GET | `/api/dashboard/stats` | Aggregated stats: totals, benign/attack counts, attack percentage, attack-type distribution |
| GET | `/api/model-info` | Proxies to FastAPI `/model-info` |
| GET | `/api/metrics` | Proxies to FastAPI `/metrics` |
| GET | `/api/alerts?status=&limit=` | Alert list, optionally filtered by status (`new`/`acknowledged`/`resolved`) |

If the FastAPI service is unreachable, prediction-related endpoints
return HTTP 503 with an explanatory message rather than a generic
failure, since this is a common local-development state (FastAPI not
yet started).

## Database schema (PostgreSQL, `web/backend/src/db/schema.sql`)

**`predictions`**: `id`, `prediction`, `confidence`, `model_used`,
`attack_type`, `source` (`api`/`csv_upload`), `created_at`.

**`alerts`**: `id`, `prediction_id` (FK), `attack_type`, `severity`
(derived from confidence: critical ≥95%, high ≥85%, medium ≥70%, else
low), `confidence`, `status`, `created_at`. Only created automatically
when a prediction's result is `ATTACK`.

## Current scope limitation

Both API layers currently serve **CICIDS2017-trained models only**.
NSL-KDD and UNSW-NB15 models exist (trained, evaluated, logged — see
`docs/experiments/results-analysis.md`) but are not yet loaded by the
FastAPI service or selectable via these endpoints; their results are
visible only in the frontend's Model Performance page (which reads
directly from `/api/metrics`'s full experiment log) and not via live
`/predict` calls. This is documented as a known gap, not an oversight.