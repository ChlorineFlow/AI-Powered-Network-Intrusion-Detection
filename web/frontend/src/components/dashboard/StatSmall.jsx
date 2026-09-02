export default function StatSmall({ label, value }) {
  return (
    <div className="bg-raised border border-hairline rounded-md p-4">
      <div className="text-muted text-xs">{label}</div>
      <div className="font-mono text-ink text-xl mt-1">{value}</div>
    </div>
  );
}