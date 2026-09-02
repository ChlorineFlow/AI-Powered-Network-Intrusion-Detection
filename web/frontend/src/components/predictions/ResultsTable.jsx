export default function ResultsTable({ results }) {
  if (!results || results.length === 0) return null;

  return (
    <div className="border border-hairline rounded-md overflow-hidden">
      <div className="max-h-96 overflow-y-auto">
        <table className="w-full text-sm">
          <thead className="bg-raised sticky top-0">
            <tr>
              <th className="text-left px-4 py-2 text-muted font-normal">#</th>
              <th className="text-left px-4 py-2 text-muted font-normal">Prediction</th>
              <th className="text-left px-4 py-2 text-muted font-normal">Confidence</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r, i) => (
              <tr key={i} className="border-t border-hairline">
                <td className="px-4 py-2 font-mono text-muted">{i + 1}</td>
                <td className="px-4 py-2">
                  <span
                    className={`font-mono text-xs px-2 py-0.5 rounded ${
                      r.prediction === "ATTACK"
                        ? "bg-alert/15 text-alert"
                        : "bg-safe/15 text-safe"
                    }`}
                  >
                    {r.prediction}
                  </span>
                </td>
                <td className="px-4 py-2 font-mono text-ink">
                  {(r.confidence * 100).toFixed(2)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}