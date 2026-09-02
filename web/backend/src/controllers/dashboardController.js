import * as mlService from "../services/mlService.js";
import * as predictionService from "../services/predictionService.js";

export async function getDashboardStats(req, res) {
  try {
    const stats = await predictionService.getDashboardStats();
    res.json(stats);
  } catch (err) {
    console.error("[dashboardController] getDashboardStats failed:", err.message);
    res.status(500).json({ error: "Failed to fetch dashboard stats." });
  }
}

export async function getModelInfo(req, res) {
  try {
    const info = await mlService.getModelInfo();
    res.json(info);
  } catch (err) {
    res.status(503).json({ error: "ML inference service is unavailable." });
  }
}

export async function getModelMetrics(req, res) {
  try {
    const metrics = await mlService.getModelMetrics();
    res.json(metrics);
  } catch (err) {
    res.status(503).json({ error: "ML inference service is unavailable." });
  }
}

export async function getAlerts(req, res) {
  try {
    const { status, limit } = req.query;
    const alerts = await predictionService.getAlerts(status, parseInt(limit, 10) || 50);
    res.json({ count: alerts.length, alerts });
  } catch (err) {
    console.error("[dashboardController] getAlerts failed:", err.message);
    res.status(500).json({ error: "Failed to fetch alerts." });
  }
}