import express from "express";
import * as predictionController from "../controllers/predictionController.js";

const router = express.Router();

router.post("/predict", predictionController.predict);
router.post("/predict/batch", predictionController.predictBatch);
router.post("/explain", predictionController.explain);
router.get("/predictions", predictionController.getRecentPredictions);
router.post("/predict/unified", predictionController.predictUnified);
router.post("/predict/routed", predictionController.predictRouted);

export default router;