import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import { LayoutDashboard, Activity, ShieldAlert, BarChart3, Info } from "lucide-react";
import Dashboard from "./pages/Dashboard.jsx";
import TrafficAnalysis from "./pages/TrafficAnalysis.jsx";

const navItems = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/traffic", label: "Traffic Analysis", icon: Activity },
  { to: "/alerts", label: "Alerts", icon: ShieldAlert },
  { to: "/model-performance", label: "Model Performance", icon: BarChart3 },
  { to: "/about", label: "About", icon: Info },
];

function Sidebar() {
  return (
    <aside className="w-56 shrink-0 bg-surface border-r border-hairline flex flex-col">
      <div className="px-5 py-6 border-b border-hairline">
        <div className="font-display font-semibold text-ink text-lg leading-tight">
          NIDS
        </div>
        <div className="text-muted text-xs mt-1">Intrusion Detection</div>
      </div>
      <nav className="flex-1 px-3 py-4 space-y-1">
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors ${
                isActive
                  ? "bg-raised text-ink border border-hairline"
                  : "text-muted hover:text-ink hover:bg-raised/50"
              }`
            }
          >
            <Icon size={16} strokeWidth={1.75} />
            {label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}

function ComingSoon({ title }) {
  return (
    <div className="p-8">
      <h1 className="font-display text-xl text-ink mb-2">{title}</h1>
      <p className="text-muted text-sm">This page is being built in a later phase.</p>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex min-h-screen bg-void">
        <Sidebar />
        <main className="flex-1 overflow-y-auto">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/traffic" element={<TrafficAnalysis />} />
            <Route path="/alerts" element={<ComingSoon title="Alerts" />} />
            <Route path="/model-performance" element={<ComingSoon title="Model Performance" />} />
            <Route path="/about" element={<ComingSoon title="About" />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}