export default function StatHero({ label, value, sublabel, tone = "safe" }) {
  const toneColor = { safe: "text-safe", alert: "text-alert", warn: "text-warn" }[tone];

  return (
    <div className="bg-raised border border-hairline rounded-md p-6 flex flex-col justify-between h-full">
      <div className="text-muted text-sm">{label}</div>
      <div className={`font-mono font-semibold text-5xl mt-3 ${toneColor}`}>
        {value}
      </div>
      {sublabel && <div className="text-muted text-xs mt-2">{sublabel}</div>}
    </div>
  );
}