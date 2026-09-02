import express from "express";
import cors from "cors";
import helmet from "helmet";
import morgan from "morgan";
import dotenv from "dotenv";

import predictionRoutes from "./routes/predictionRoutes.js";
import dashboardRoutes from "./routes/dashboardRoutes.js";
import { testConnection } from "./config/db.js";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;
const FRONTEND_URL = process.env.FRONTEND_URL || "http://localhost:5173";

app.use(helmet());
app.use(cors({ origin: FRONTEND_URL }));
app.use(morgan("dev"));
app.use(express.json({ limit: "10mb" })); // batch CSV uploads can be sizeable

app.get("/health", (req, res) => {
  res.json({ status: "ok", service: "nids-backend" });
});

app.use("/api", predictionRoutes);
app.use("/api", dashboardRoutes);

app.use((req, res) => {
  res.status(404).json({ error: "Not found" });
});

app.use((err, req, res, next) => {
  console.error("[app] Unhandled error:", err);
  res.status(500).json({ error: "Internal server error" });
});

async function start() {
  try {
    await testConnection();
  } catch (err) {
    console.error("[app] Failed to connect to PostgreSQL on startup:", err.message);
    console.error("[app] Server will still start, but DB-dependent routes will fail.");
  }

  app.listen(PORT, () => {
    console.log(`[app] NIDS backend listening on http://localhost:${PORT}`);
    console.log(`[app] CORS allowed origin: ${FRONTEND_URL}`);
  });
}

start();