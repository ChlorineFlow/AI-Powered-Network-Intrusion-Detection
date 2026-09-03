import { useEffect, useState } from "react";
import ModelComparisonChart from "../components/charts/ModelComparisonChart.jsx";
import MetricsTable from "../components/dashboard/MetricsTable.jsx";
import { getModelMetrics } from "../services/api.js";

const CORE_MODELS = ["logistic_regression", "decision_tree", "random_forest", "xgboost"];
const DATASETS = [
  { key: "cicids2017", label: "CICIDS2017" },
  { key: "nsl_kdd", label: "NSL-KDD" },
  { key: "unsw_nb15", label: "UNSW-NB15" },
];

export default function ModelPerformance() {
  const [results, setResults] = useState([]);
  const [dataset, setDataset] = useState("cicids2017");
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

  const forDataset = (task) =>
    results.filter(
      (r) => r.dataset === dataset && r.task === task && CORE_MODELS.includes(r.model)
    );

  const binaryResults = forDataset("binary");
  const multiclassResults = forDataset("multiclass");

  const chartData = (rows) =>
    rows.map((r) => ({
      model: r.model.replace("_", " "),
      f1_macro: r.metrics.f1_macro,
      accuracy: r.metrics.accuracy,
    }));

  return (
    <div className="p-8 space-y-6">
      <div>
        <h1 className="font-display text-xl text-ink">Model Performance</h1>
        <p className="text-muted text-sm mt-1">
          Real, logged results — each dataset trained and evaluated independently.
        </p>
      </div>

      <div className="flex gap-2">
        {DATASETS.map((d) => (
          <button
            key={d.key}
            onClick={() => setDataset(d.key)}
            className={`px-4 py-2 rounded-md text-sm font-mono border transition-colors ${
              dataset === d.key
                ? "bg-raised border-info/40 text-info"
                : "border-hairline text-muted hover:text-ink"
            }`}
          >
            {d.label}
          </button>
        ))}
      </div>

      <div className="bg-surface border border-hairline rounded-md p-5">
        <h2 className="font-display text-sm text-ink mb-1">Binary classification</h2>
        <p className="text-muted text-xs mb-4">
          {binaryResults.length > 0
            ? "Normal/Benign vs Attack, evaluated on this dataset's own held-out test set."
            : "No binary results logged for this dataset yet."}
        </p>
        {binaryResults.length > 0 && (
          <>
            <ModelComparisonChart data={chartData(binaryResults)} />
            <div className="mt-4">
              <MetricsTable rows={binaryResults} />
            </div>
          </>
        )}
      </div>

      <div className="bg-surface border border-hairline rounded-md p-5">
        <h2 className="font-display text-sm text-ink mb-1">Multiclass classification</h2>
        <p className="text-muted text-xs mb-4">
          {multiclassResults.length > 0
            ? "Attack-category classification, with low-sample classes excluded per documented methodology."
            : "No multiclass results logged for this dataset yet."}
        </p>
        {multiclassResults.length > 0 && (
          <>
            <ModelComparisonChart data={chartData(multiclassResults)} />
            <div className="mt-4">
              <MetricsTable rows={multiclassResults} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}