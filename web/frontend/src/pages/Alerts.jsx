import { useEffect, useState } from "react";
import { getAlerts } from "../services/api.js";

const severityStyle = {
  critical: "bg-alert/15 text-alert border-alert/40",
  high: "bg-alert/15 text-alert border-alert/40",
  medium: "bg-warn/15 text-warn border-warn/40",
  low: "bg-safe/15 text-safe border-safe/40",
};

const statusStyle = {
  new: "bg-info/15 text-info",
  acknowledged: "bg-warn/15 text-warn",
  resolved: "bg-safe/15 text-safe",
};

export default function Alerts() {
  const [alerts, setAlerts] = useState([]);
  const [filter, setFilter] = useState("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    load();
  }, [filter]);

  function load() {
    setLoading(true);
    const params = filter === "all" ? { limit: 100 } : { status: filter, limit: 100 };
    getAlerts(params)
      .then((data) => setAlerts(data.alerts || []))
      .catch(() => setError("Could not load alerts from the backend."))
      .finally(() => setLoading(false));
  }

  const filters = ["all", "new", "acknowledged", "resolved"];

  return (
    <div className="p-8 space-y-6">
      <div>
        <h1 className="font-display text-xl text-ink">Alerts</h1>
        <p className="text-muted text-sm mt-1">
          Attack detections generated from live predictions and CSV analysis.
        </p>
      </div>

      <div className="flex gap-2">
        {filters.map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3 py-1.5 rounded-md text-xs font-mono border transition-colors ${
              filter === f
                ? "bg-raised border-info/40 text-info"
                : "border-hairline text-muted hover:text-ink"
            }`}
          >
            {f}
          </button>
        ))}
      </div>

      {loading && <div className="text-muted text-sm">Loading...</div>}
      {error && (
        <div className="bg-raised border border-alert/40 rounded-md p-4 text-alert text-sm">{error}</div>
      )}

      {!loading && !error && alerts.length === 0 && (
        <div className="bg-surface border border-hairline rounded-md p-8 text-center text-muted text-sm">
          No alerts match this filter.
        </div>
      )}

      {!loading && alerts.length > 0 && (
        <div className="bg-surface border border-hairline rounded-md overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-raised">
              <tr>
                <th className="text-left px-4 py-2 text-muted font-normal">Attack type</th>
                <th className="text-left px-4 py-2 text-muted font-normal">Severity</th>
                <th className="text-left px-4 py-2 text-muted font-normal">Confidence</th>
                <th className="text-left px-4 py-2 text-muted font-normal">Status</th>
                <th className="text-left px-4 py-2 text-muted font-normal">Time</th>
              </tr>
            </thead>
            <tbody>
              {alerts.map((a) => (
                <tr key={a.id} className="border-t border-hairline">
                  <td className="px-4 py-2 text-ink">{a.attack_type || "Unclassified"}</td>
                  <td className="px-4 py-2">
                    <span
                      className={`text-xs font-mono px-2 py-0.5 rounded border ${severityStyle[a.severity] || ""}`}
                    >
                      {a.severity}
                    </span>
                  </td>
                  <td className="px-4 py-2 font-mono text-ink">{(a.confidence * 100).toFixed(1)}%</td>
                  <td className="px-4 py-2">
                    <span className={`text-xs font-mono px-2 py-0.5 rounded ${statusStyle[a.status] || ""}`}>
                      {a.status}
                    </span>
                  </td>
                  <td className="px-4 py-2 font-mono text-muted text-xs">
                    {new Date(a.created_at).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}