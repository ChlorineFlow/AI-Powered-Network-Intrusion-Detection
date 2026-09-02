const severityColor = {
  critical: "bg-alert",
  high: "bg-alert",
  medium: "bg-warn",
  low: "bg-safe",
};

export default function RecentAlertsList({ alerts }) {
  if (!alerts || alerts.length === 0) {
    return (
      <div className="text-muted text-sm flex items-center justify-center h-64">
        No alerts yet — traffic is clean.
      </div>
    );
  }

  return (
    <div className="space-y-2 max-h-80 overflow-y-auto pr-1">
      {alerts.map((alert) => (
        <div
          key={alert.id}
          className="flex items-center gap-3 px-3 py-2 bg-void border border-hairline rounded-md"
        >
          <span
            className={`w-2 h-2 rounded-full shrink-0 ${severityColor[alert.severity] || "bg-muted"}`}
          />
          <div className="flex-1 min-w-0">
            <div className="text-ink text-sm truncate">
              {alert.attack_type || "Unclassified attack"}
            </div>
            <div className="font-mono text-muted text-xs">
              {new Date(alert.created_at).toLocaleString()}
            </div>
          </div>
          <div className="font-mono text-xs text-muted shrink-0">
            {(alert.confidence * 100).toFixed(1)}%
          </div>
        </div>
      ))}
    </div>
  );
}