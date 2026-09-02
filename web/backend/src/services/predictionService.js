/**
 * Persists predictions and alerts to PostgreSQL. Called after a
 * successful ML service prediction — the ML service itself has no
 * knowledge of the database, keeping the layers cleanly separated.
 */

import { pool } from "../config/db.js";

function severityFromConfidence(confidence) {
  if (confidence >= 0.95) return "critical";
  if (confidence >= 0.85) return "high";
  if (confidence >= 0.7) return "medium";
  return "low";
}

export async function savePrediction({ prediction, confidence, modelUsed, attackType = null, source = "api" }) {
  const result = await pool.query(
    `INSERT INTO predictions (prediction, confidence, model_used, attack_type, source)
     VALUES ($1, $2, $3, $4, $5)
     RETURNING id, created_at`,
    [prediction, confidence, modelUsed, attackType, source]
  );
  const { id, created_at } = result.rows[0];

  if (prediction === "ATTACK") {
    await pool.query(
      `INSERT INTO alerts (prediction_id, attack_type, severity, confidence, status)
       VALUES ($1, $2, $3, $4, 'new')`,
      [id, attackType, severityFromConfidence(confidence), confidence]
    );
  }

  return { id, created_at };
}

export async function getRecentPredictions(limit = 50) {
  const result = await pool.query(
    `SELECT * FROM predictions ORDER BY created_at DESC LIMIT $1`,
    [limit]
  );
  return result.rows;
}

export async function getDashboardStats() {
  const totals = await pool.query(`
    SELECT
      COUNT(*) AS total,
      COUNT(*) FILTER (WHERE prediction = 'BENIGN') AS benign,
      COUNT(*) FILTER (WHERE prediction = 'ATTACK') AS attack
    FROM predictions
  `);

  const distribution = await pool.query(`
    SELECT attack_type, COUNT(*) AS count
    FROM predictions
    WHERE prediction = 'ATTACK' AND attack_type IS NOT NULL
    GROUP BY attack_type
    ORDER BY count DESC
  `);

  const row = totals.rows[0];
  const total = parseInt(row.total, 10);
  const attack = parseInt(row.attack, 10);

  return {
    total_analyzed: total,
    benign_count: parseInt(row.benign, 10),
    attack_count: attack,
    attack_percentage: total > 0 ? (attack / total) * 100 : 0,
    attack_distribution: distribution.rows,
  };
}

export async function getAlerts(status = null, limit = 50) {
  const query = status
    ? `SELECT a.*, p.model_used FROM alerts a
       JOIN predictions p ON a.prediction_id = p.id
       WHERE a.status = $1 ORDER BY a.created_at DESC LIMIT $2`
    : `SELECT a.*, p.model_used FROM alerts a
       JOIN predictions p ON a.prediction_id = p.id
       ORDER BY a.created_at DESC LIMIT $1`;
  const params = status ? [status, limit] : [limit];
  const result = await pool.query(query, params);
  return result.rows;
}