# web/backend — Node.js / Express API layer

The application/API orchestration layer. Handles routing, validation, file
uploads, and forwards inference requests to the Python/FastAPI ML service.
No ML training or inference logic lives here. Prediction history and alerts
are persisted to **PostgreSQL**.

## Status

Scaffolding only — no routes, controllers, or DB schema implemented yet.

## Setup (once dependencies are added in a later phase)

```bash
cd web/backend
npm install
cp .env.example .env   # then fill in real values
npm run dev
```

## Structure

- `src/config/` — environment & app configuration
- `src/routes/` — Express route definitions
- `src/controllers/` — request/response handling
- `src/services/` — business logic, including the client that calls the
  FastAPI ML service
- `src/db/` — PostgreSQL connection pool, migrations/schema, and query
  modules for prediction history and alerts
- `src/middleware/` — auth, error handling, rate limiting, etc.
- `src/validators/` — request payload validation
- `src/utils/` — shared helpers
- `uploads/` — temporary storage for uploaded CSVs (not committed)
- `logs/` — application logs (not committed)
- `tests/` — backend tests

## Persistence

PostgreSQL is used for:
- Prediction history
- Alerts

Trained ML models are **not** stored in the database — they live in
`ml/saved_models/` and are loaded by the FastAPI service.
