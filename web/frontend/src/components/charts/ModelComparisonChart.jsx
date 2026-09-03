import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from "recharts";

export default function ModelComparisonChart({ data }) {
  if (!data || data.length === 0) {
    return <div className="text-muted text-sm flex items-center justify-center h-64">No data.</div>;
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data} margin={{ left: 0, right: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#232B3A" vertical={false} />
        <XAxis dataKey="model" stroke="#8592A6" fontSize={12} />
        <YAxis stroke="#8592A6" fontSize={12} domain={[0, 1]} />
        <Tooltip
          contentStyle={{ background: "#161C29", border: "1px solid #232B3A", borderRadius: 6 }}
          labelStyle={{ color: "#E4EAF2" }}
        />
        <Legend wrapperStyle={{ fontSize: 12, color: "#8592A6" }} />
        <Bar dataKey="f1_macro" name="F1 (macro)" fill="#2DD4A7" radius={[3, 3, 0, 0]} />
        <Bar dataKey="accuracy" name="Accuracy" fill="#5B8DEF" radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}