# web/frontend — React Dashboard

Professional SOC-style dashboard: React (JavaScript) + Vite + Tailwind CSS.
Talks only to the Node/Express backend — never directly to the Python ML
service.

## Status

Scaffolding only — pages and components not yet implemented.

## Setup (once dependencies are added in a later phase)

```bash
cd web/frontend
npm install
cp .env.example .env
npm run dev
```

## Planned pages

- Dashboard — traffic totals, attack %, current model, recent alerts,
  attack distribution, traffic timeline
- Traffic Analysis — CSV upload, record inspection, run prediction
- Predictions — timestamp, prediction, attack type, confidence, severity
- Model Performance — accuracy/precision/recall/F1/ROC-AUC, confusion
  matrix, model comparison
- Alerts — attack type, severity, confidence, timestamp, status

## Structure

- `src/components/` — common, dashboard, charts, alerts, predictions
- `src/pages/` — top-level route components
- `src/services/api.js` — Axios client for the Node backend
- `src/hooks/`, `src/utils/`, `src/context/` — shared frontend logic
