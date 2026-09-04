import axios from "axios";

const BASE_URL = import.meta.env.VITE_NODE_API_URL || "http://localhost:5000";

const client = axios.create({ baseURL: BASE_URL, timeout: 15000 });

export const getDashboardStats = () => client.get("/api/dashboard/stats").then((r) => r.data);
export const getModelInfo = () => client.get("/api/model-info").then((r) => r.data);
export const getModelMetrics = () => client.get("/api/metrics").then((r) => r.data);
export const getAlerts = (params = {}) => client.get("/api/alerts", { params }).then((r) => r.data);
export const getRecentPredictions = (limit = 20) =>
  client.get("/api/predictions", { params: { limit } }).then((r) => r.data);
export const predict = (features, model = "xgboost") =>
  client.post("/api/predict", { features, model }).then((r) => r.data);
export const predictBatch = (rows, model = "xgboost") =>
  client.post("/api/predict/batch", { rows, model }).then((r) => r.data);

export const predictUnified = (duration_sec, src_bytes, dst_bytes) =>
  client.post("/api/predict/unified", { duration_sec, src_bytes, dst_bytes }).then((r) => r.data);

export const predictRouted = (duration_sec, src_bytes, dst_bytes) =>
  client.post("/api/predict/routed", { duration_sec, src_bytes, dst_bytes }).then((r) => r.data);