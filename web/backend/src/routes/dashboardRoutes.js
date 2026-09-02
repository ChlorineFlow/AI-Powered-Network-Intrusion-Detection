import express from "express";
import * as dashboardController from "../controllers/dashboardController.js";

const router = express.Router();

router.get("/dashboard/stats", dashboardController.getDashboardStats);
router.get("/model-info", dashboardController.getModelInfo);
router.get("/metrics", dashboardController.getModelMetrics);
router.get("/alerts", dashboardController.getAlerts);

export default router;