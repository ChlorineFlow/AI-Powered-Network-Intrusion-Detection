import * as mlService from "../services/mlService.js";
import * as predictionService from "../services/predictionService.js";

export async function predict(req, res) {
  try {
    const { features, model } = req.body;
    if (!features) {
      return res.status(422).json({ error: "Missing 'features' in request body." });
    }

    const result = await mlService.predictSingle(features, model);

    const saved = await predictionService.savePrediction({
      prediction: result.prediction,
      confidence: result.confidence,
      modelUsed: result.model_used,
      source: "api",
    });

    res.json({ ...result, id: saved.id, created_at: saved.created_at });
  } catch (err) {
    handleMlError(err, res);
  }
}

export async function predictBatch(req, res) {
  try {
    const { rows, model } = req.body;
    if (!rows || !Array.isArray(rows) || rows.length === 0) {
      return res.status(422).json({ error: "Missing or empty 'rows' array in request body." });
    }

    const result = await mlService.predictBatch(rows, model);

    // Persist each row's prediction — sequential for simplicity; batch
    // insert could be added later if upload volume grows large.
    for (const r of result.results) {
      await predictionService.savePrediction({
        prediction: r.prediction,
        confidence: r.confidence,
        modelUsed: result.model_used,
        source: "csv_upload",
      });
    }

    res.json(result);
  } catch (err) {
    handleMlError(err, res);
  }
}

export async function explain(req, res) {
  try {
    const { features, model, top_n } = req.body;
    if (!features) {
      return res.status(422).json({ error: "Missing 'features' in request body." });
    }
    const result = await mlService.explainPrediction(features, model, top_n);
    res.json(result);
  } catch (err) {
    handleMlError(err, res);
  }
}

export async function getRecentPredictions(req, res) {
  try {
    const limit = parseInt(req.query.limit, 10) || 50;
    const predictions = await predictionService.getRecentPredictions(limit);
    res.json({ count: predictions.length, predictions });
  } catch (err) {
    console.error("[predictionController] getRecentPredictions failed:", err.message);
    res.status(500).json({ error: "Failed to fetch prediction history." });
  }
}

function handleMlError(err, res) {
  if (err.response) {
    // The ML service responded with an error status (4xx/5xx)
    console.error("[predictionController] ML service error:", err.response.data);
    return res.status(err.response.status).json(err.response.data);
  }
  console.error("[predictionController] ML service unreachable:", err.message);
  res.status(503).json({
    error: "ML inference service is unavailable. Ensure FastAPI is running on ML_API_URL.",
  });
}