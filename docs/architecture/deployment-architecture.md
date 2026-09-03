# Deployment Architecture

## Current: local development deployment

This project currently runs as three independently-started local
services, matching the architecture in `system-architecture.md`:

Terminal 1: uvicorn src.inference.app:app --reload --port 8000 (FastAPI, ml/)
Terminal 2: node src/app.js (Express, web/backend/)
Terminal 3: npm run dev (Vite, web/frontend/)


PostgreSQL runs as a local service, connected via `DATABASE_URL` in
`web/backend/.env`. No containerization, orchestration, or cloud
infrastructure is currently used — per project scope (see root
`README.md` technology stack decisions), Docker/Kubernetes/cloud
services were deliberately excluded unless a genuine need arose, and
none did for a project of this scope.

## What would be required for a genuine production deployment

This is documented explicitly as a gap, not implemented, and is
distinct from the model-generalization concerns already covered in
`docs/research/methodology.md` under "Real-world deployment
considerations":

1. **Containerization** — Dockerfiles for the FastAPI service and
   Express backend, to ensure consistent Python/Node runtime versions
   across environments.
2. **Environment-specific configuration** — the current `.env`-based
   configuration is adequate for local development; a production
   deployment would need secrets management (e.g. a vault service)
   rather than plaintext `.env` files.
3. **Managed PostgreSQL** — a managed database service rather than a
   local instance, with backup/restore and connection pooling
   appropriate for concurrent load.
4. **Process management / reverse proxy** — a process manager (e.g.
   systemd, PM2) for the Node backend and a reverse proxy (e.g. Nginx)
   in front of both the frontend build and the API, rather than Vite's
   development server and `uvicorn --reload`.
5. **Model serving considerations at scale** — the current FastAPI
   service loads all models into memory at startup and serves
   inference synchronously; a higher-throughput deployment might
   require a dedicated model-serving layer (e.g. batching, a queue for
   `/predict/batch` on very large uploads) not needed at this project's
   scale.
6. **CI/CD** — no automated build/test/deploy pipeline currently exists;
   `ml/tests/` has pytest coverage (13 tests) but is not wired into any
   CI system.

None of the above changes the model-generalization findings documented
in `methodology.md` — infrastructure improvements would make the
*system* deployable, but would not by themselves close the measured
15-21 F1-point generalization gap between benchmark datasets, or the
feature-schema mismatch between benchmark datasets and real
organizational telemetry.