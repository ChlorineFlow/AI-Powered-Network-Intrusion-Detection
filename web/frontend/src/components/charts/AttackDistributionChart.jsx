import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

export default function AttackDistributionChart({ data }) {
  if (!data || data.length === 0) {
    return (
      <div className="text-muted text-sm flex items-center justify-center h-64">
        No attack traffic recorded yet.
      </div>
    );
  }

  const chartData = data.map((d) => ({
    name: d.attack_type || "Unknown",
    count: parseInt(d.count, 10),
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={chartData} layout="vertical" margin={{ left: 24 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#232B3A" horizontal={false} />
        <XAxis type="number" stroke="#8592A6" fontSize={12} />
        <YAxis dataKey="name" type="category" stroke="#8592A6" fontSize={12} width={140} />
        <Tooltip
          contentStyle={{ background: "#161C29", border: "1px solid #232B3A", borderRadius: 6 }}
          labelStyle={{ color: "#E4EAF2" }}
        />
        <Bar dataKey="count" fill="#FF5C5C" radius={[0, 3, 3, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}