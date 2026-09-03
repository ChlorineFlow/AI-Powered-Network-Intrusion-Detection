export default function MetricsTable({ rows }) {
  if (!rows || rows.length === 0) {
    return <div className="text-muted text-sm">No experiments logged yet.</div>;
  }

  return (
    <div className="border border-hairline rounded-md overflow-x-auto">
      <table className="w-full text-sm min-w-[640px]">
        <thead className="bg-raised">
          <tr>
            <th className="text-left px-4 py-2 text-muted font-normal">Model</th>
            <th className="text-left px-4 py-2 text-muted font-normal">Accuracy</th>
            <th className="text-left px-4 py-2 text-muted font-normal">F1 (macro)</th>
            <th className="text-left px-4 py-2 text-muted font-normal">ROC-AUC</th>
            <th className="text-left px-4 py-2 text-muted font-normal">Train time</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-hairline">
              <td className="px-4 py-2 font-mono text-ink">{r.model}</td>
              <td className="px-4 py-2 font-mono text-ink">{(r.metrics.accuracy * 100).toFixed(2)}%</td>
              <td className="px-4 py-2 font-mono text-ink">{r.metrics.f1_macro?.toFixed(4)}</td>
              <td className="px-4 py-2 font-mono text-ink">
                {(r.metrics.roc_auc ?? r.metrics.roc_auc_ovr_macro)?.toFixed(4) ?? "—"}
              </td>
              <td className="px-4 py-2 font-mono text-muted">{r.training_time_seconds?.toFixed(1)}s</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}