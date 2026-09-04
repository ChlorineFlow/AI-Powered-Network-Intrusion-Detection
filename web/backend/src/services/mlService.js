/**
 * Client for the Python FastAPI ML inference service. This is the ONLY
 * place in the backend that talks to Python, per the mandated
 * three-layer architecture (React never calls FastAPI directly).
 */

import axios from "axios";
import dotenv from "dotenv";

dotenv.config();

const ML_API_URL = process.env.ML_API_URL || "http://localhost:8000";

const client = axios.create({
  baseURL: ML_API_URL,
  timeout: 15000,
});

export async function checkMlServiceHealth() {
  const response = await client.get("/health");
  return response.data;
}

export async function getModelInfo() {
  const response = await client.get("/model-info");
  return response.data;
}

export async function getModelMetrics() {
  const response = await client.get("/metrics");
  return response.data;
}

export async function predictSingle(features, model = "xgboost") {
  const response = await client.post("/predict", { features, model });
  return response.data;
}

export async function predictBatch(rows, model = "xgboost") {
  const response = await client.post("/predict/batch", { rows, model });
  return response.data;
}

export async function explainPrediction(features, model = "xgboost", topN = 10) {
  const response = await client.post("/explain", {
    features,
    model,
    top_n: topN,
  });
  return response.data;
}

export async function predictUnified(duration_sec, src_bytes, dst_bytes) {
  const response = await client.post("/predict/unified", { duration_sec, src_bytes, dst_bytes });
  return response.data;
}

export async function predictRouted(duration_sec, src_bytes, dst_bytes) {
  const response = await client.post("/predict/routed", { duration_sec, src_bytes, dst_bytes });
  return response.data;
}