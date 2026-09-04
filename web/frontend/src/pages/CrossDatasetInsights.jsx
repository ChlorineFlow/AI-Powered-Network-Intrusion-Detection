import { useState } from "react";
import { Play } from "lucide-react";
import { predictUnified, predictRouted } from "../services/api.js";

const expertLabels = {
  cicids2017: "CICIDS2017",
  nsl_kdd: "NSL-KDD",
  unsw_nb15: "UNSW-NB15",
};

export default function CrossDatasetInsights() {
  const [duration, setDuration] = useState("2.5");
  const [srcBytes, setSrcBytes] = useState("500");
  const [dstBytes, setDstBytes] = useState("1200");
  const [unifiedResult, setUnifiedResult] = useState(null);
  const [routedResult, setRoutedResult] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);

  async function runComparison() {
    setRunning(true);
    setError(null);
    try {
      const d = parseFloat(duration), s = parseFloat(srcBytes), b = parseFloat(dstBytes);
      const [unified, routed] = await Promise.all([
        predictUnified(d, s, b),
        predictRouted(d, s, b),
      ]);
      setUnifiedResult(unified);
      setRoutedResult(routed);
    } catch (err) {
      setError("Prediction failed — check that FastAPI and Node are running.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="p-8 space-y-6 max-w-3xl">
      <div>
        <h1 className="font-display text-xl text-ink">Cross-Dataset Insights</h1>
        <p className="text-muted text-sm mt-1">
          Two experimental approaches to generalizing across CICIDS2017, NSL-KDD, and
          UNSW-NB15, using a small common feature set (duration and byte counts).
        </p>
      </div>

      <div className="bg-surface border border-hairline rounded-md p-5">
        <h2 className="font-display text-sm text-ink mb-4">Try it</h2>
        <div className="grid grid-cols-3 gap-3">
          <div>
            <label className="text-muted text-xs">Duration (seconds)</label>
            <input
              value={duration}
              onChange={(e) => setDuration(e.target.value)}
              className="w-full mt-1 bg-void border border-hairline rounded-md px-3 py-2 text-ink font-mono text-sm"
            />
          </div>
          <div>
            <label className="text-muted text-xs">Source bytes</label>
            <input
              value={srcBytes}
              onChange={(e) => setSrcBytes(e.target.value)}
              className="w-full mt-1 bg-void border border-hairline rounded-md px-3 py-2 text-ink font-mono text-sm"
            />
          </div>
          <div>
            <label className="text-muted text-xs">Destination bytes</label>
            <input
              value={dstBytes}
              onChange={(e) => setDstBytes(e.target.value)}
              className="w-full mt-1 bg-void border border-hairline rounded-md px-3 py-2 text-ink font-mono text-sm"
            />
          </div>
        </div>
        <button
          onClick={runComparison}
          disabled={running}
          className="mt-4 flex items-center gap-2 px-4 py-2 bg-safe/15 text-safe border border-safe/40 rounded-md text-sm hover:bg-safe/25 transition-colors disabled:opacity-50"
        >
          <Play size={14} strokeWidth={2} />
          {running ? "Running..." : "Compare approaches"}
        </button>
        {error && (
          <div className="mt-4 bg-alert/10 border border-alert/40 rounded-md p-3 text-alert text-sm">
            {error}
          </div>
        )}
      </div>

      {(unifiedResult || routedResult) && (
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-raised border border-hairline rounded-md p-5">
            <h3 className="font-display text-sm text-ink mb-1">Flat unified model</h3>
            <p className="text-muted text-xs mb-3">One model, trained on all 3 datasets combined.</p>
            {unifiedResult && (
              <>
                <span
                  className={`font-mono text-xs px-2 py-0.5 rounded ${
                    unifiedResult.prediction === "ATTACK" ? "bg-alert/15 text-alert" : "bg-safe/15 text-safe"
                  }`}
                >
                  {unifiedResult.prediction}
                </span>
                <div className="font-mono text-ink text-lg mt-2">
                  {(unifiedResult.confidence * 100).toFixed(1)}% confidence
                </div>
              </>
            )}
          </div>

          <div className="bg-raised border border-hairline rounded-md p-5">
            <h3 className="font-display text-sm text-ink mb-1">Domain-routing ensemble</h3>
            <p className="text-muted text-xs mb-3">Routes to the dataset-specific expert.</p>
            {routedResult && (
              <>
                <span
                  className={`font-mono text-xs px-2 py-0.5 rounded ${
                    routedResult.prediction === "ATTACK" ? "bg-alert/15 text-alert" : "bg-safe/15 text-safe"
                  }`}
                >
                  {routedResult.prediction}
                </span>
                <div className="font-mono text-ink text-lg mt-2">
                  {(routedResult.confidence * 100).toFixed(1)}% confidence
                </div>
                <div className="text-muted text-xs mt-3">
                  Routed to <span className="text-info">{expertLabels[routedResult.routed_to_expert]}</span> expert
                  {" "}({(routedResult.router_confidence * 100).toFixed(1)}% router confidence)
                </div>
              </>
            )}
          </div>
        </div>
      )}

      <div className="bg-raised border border-hairline rounded-md p-5">
        <h2 className="font-display text-sm text-ink mb-2">What this demonstrates</h2>
        <p className="text-muted text-xs leading-relaxed">
          A flat model trained on all 3 datasets combined suffers from training-set
          dominance — CICIDS2017 makes up 85% of the combined data, skewing the model's
          notion of "normal" traffic. A domain-routing ensemble instead predicts which
          dataset a flow most resembles, then defers to that dataset's own specialized
          expert. On held-out test data, oracle (perfect) routing recovers most of the
          performance gap versus flat unification, while the realistic learned router
          recovers roughly 40-60% of that gain, limited by its own domain-identification
          accuracy (88.7%-99.7% depending on dataset). Full results in{" "}
          <code className="text-info">docs/research/methodology.md</code>.
        </p>
      </div>
    </div>
  );
}