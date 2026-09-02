import { useEffect, useState } from "react";
import StatHero from "../components/dashboard/StatHero.jsx";
import StatSmall from "../components/dashboard/StatSmall.jsx";
import AttackDistributionChart from "../components/charts/AttackDistributionChart.jsx";
import RecentAlertsList from "../components/alerts/RecentAlertsList.jsx";
import { getDashboardStats, getModelInfo, getAlerts } from "../services/api.js";

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const [statsData, modelData, alertsData] = await Promise.all([
          getDashboardStats(),
          getModelInfo().catch(() => null),
          getAlerts({ limit: 10 }).catch(() => ({ alerts: [] })),
        ]);
        setStats(statsData);
        setModelInfo(modelData);
        setAlerts(alertsData.alerts || []);
      } catch (err) {
        setError("Could not reach the backend. Is the Node server running on port 5000?");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) {
    return <div className="p-8 text-muted text-sm">Loading dashboard...</div>;
  }

  if (error) {
    return (
      <div className="p-8">
        <div className="bg-raised border border-alert/40 rounded-md p-4 text-alert text-sm">
          {error}
        </div>
      </div>
    );
  }

  const attackPct = stats?.attack_percentage ?? 0;

  return (
    <div className="p-8 space-y-6">
      <div>
        <h1 className="font-display text-xl text-ink">Dashboard</h1>
        <p className="text-muted text-sm mt-1">Live overview of analyzed network traffic.</p>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div className="col-span-2">
          <StatHero
            label="Attack ratio"
            value={`${attackPct.toFixed(2)}%`}
            sublabel={`${stats?.attack_count ?? 0} of ${stats?.total_analyzed ?? 0} flows flagged`}
            tone={attackPct > 10 ? "alert" : "safe"}
          />
        </div>
        <div className="space-y-4">
          <StatSmall label="Total analyzed" value={stats?.total_analyzed ?? 0} />
          <StatSmall label="Active model" value={modelInfo?.default_model ?? "—"} />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-surface border border-hairline rounded-md p-5">
          <h2 className="font-display text-sm text-ink mb-4">Attack distribution</h2>
          <AttackDistributionChart data={stats?.attack_distribution} />
        </div>
        <div className="bg-surface border border-hairline rounded-md p-5">
          <h2 className="font-display text-sm text-ink mb-4">Recent alerts</h2>
          <RecentAlertsList alerts={alerts} />
        </div>
      </div>
    </div>
  );
}