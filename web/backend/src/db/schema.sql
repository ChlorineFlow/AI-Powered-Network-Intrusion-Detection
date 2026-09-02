-- Prediction history: every prediction made through the API, whether
-- via single /predict or /predict/batch.
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    prediction VARCHAR(20) NOT NULL,           -- 'BENIGN' or 'ATTACK'
    confidence DOUBLE PRECISION NOT NULL,
    model_used VARCHAR(50) NOT NULL,
    attack_type VARCHAR(100),                   -- populated if multiclass was run
    source VARCHAR(50) DEFAULT 'api',            -- 'api', 'csv_upload', etc.
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_prediction ON predictions (prediction);

-- Alerts: generated when a prediction is classified as ATTACK, for the
-- dashboard's Alerts page.
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER REFERENCES predictions(id) ON DELETE CASCADE,
    attack_type VARCHAR(100),
    severity VARCHAR(20) NOT NULL DEFAULT 'medium',  -- 'low', 'medium', 'high', 'critical'
    confidence DOUBLE PRECISION NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'new',       -- 'new', 'acknowledged', 'resolved'
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts (status);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts (created_at DESC);