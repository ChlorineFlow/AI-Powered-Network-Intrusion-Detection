import { useEffect, useState } from "react";
import ModelComparisonChart from "../components/charts/ModelComparisonChart.jsx";
import MetricsTable from "../components/dashboard/MetricsTable.jsx";
import { getModelMetrics } from "../services/api.js";

const CORE_MODELS = ["logistic_regression", "decision_tree", "random_forest", "xgboost"];

export default function ModelPerformance() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    getModelMetrics()
      .then((data) => setResults(data.results || []))
      .catch(() => setError("Could not load metrics from the backend."))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-8 text-muted text-sm">Loading experiment results...</div>;
  if (error)
    return (
      <div className="p-8">
        <div className="bg-raised border border-alert/40 rounded-md p-4 text-alert text-sm">{error}</div>
      </div>
    );

  const binaryResults = results.filter(
    (r) => r.task === "binary" && CORE_MODELS.includes(r.model)
  );
  const multiclassResults = results.filter(
    (r) => r.task === "multiclass" && CORE_MODELS.includes(r.model)
  );

  const chartData = (rows) =>
    rows.map((r) => ({
      model: r.model.replace("_", " "),
      f1_macro: r.metrics.f1_macro,
      accuracy: r.metrics.accuracy,
    }));

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="font-display text-xl text-ink">Model Performance</h1>
        <p className="text-muted text-sm mt-1">
          Real, logged results from experiments run on CICIDS2017 — no fabricated numbers.
        </p>
      </div>

      <div className="bg-surface border border-hairline rounded-md p-5">
        <h2 className="font-display text-sm text-ink mb-1">Binary classification (BENIGN vs ATTACK)</h2>
        <p className="text-muted text-xs mb-4">Trained and evaluated on the full leakage-safe CICIDS2017 split.</p>
        <ModelComparisonChart data={chartData(binaryResults)} />
        <div className="mt-4">
          <MetricsTable rows={binaryResults} />
        </div>
      </div>

      <div className="bg-surface border border-hairline rounded-md p-5">
        <h2 className="font-display text-sm text-ink mb-1">Multiclass attack classification</h2>
        <p className="text-muted text-xs mb-4">
          12 attack classes (3 rare classes excluded — see documented methodology).
        </p>
        <ModelComparisonChart data={chartData(multiclassResults)} />
        <div className="mt-4">
          <MetricsTable rows={multiclassResults} />
        </div>
      </div>
    </div>
  );
}