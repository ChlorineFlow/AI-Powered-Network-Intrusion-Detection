export default function About() {
  return (
    <div className="p-8 max-w-2xl space-y-6">
      <div>
        <h1 className="font-display text-xl text-ink">About this project</h1>
        <p className="text-muted text-sm mt-1">
          AI-Based Network Intrusion Detection System
        </p>
      </div>

      <div className="bg-surface border border-hairline rounded-md p-5 space-y-3 text-sm text-ink">
        <p>
          A research-oriented Network Intrusion Detection System combining classical
          machine learning with a full-stack monitoring dashboard, trained and
          evaluated on the CICIDS2017 dataset with rigorous leakage-safe methodology.
        </p>
        <p className="text-muted">
          Architecture: React (Vite + Tailwind) → Node.js/Express + PostgreSQL →
          Python/FastAPI ML service.
        </p>
        <p className="text-muted">
          Models: Logistic Regression, Decision Tree, Random Forest, and XGBoost —
          trained on both binary (BENIGN vs ATTACK) and multiclass attack
          classification tasks, with SHAP-based explainability and documented
          cross-dataset generalization testing on NSL-KDD.
        </p>
      </div>

      <div className="bg-raised border border-hairline rounded-md p-5">
        <h2 className="font-display text-sm text-ink mb-2">Cross-dataset generalization</h2>
        <p className="text-muted text-xs leading-relaxed">
          Performance on CICIDS2017 (99%+ F1) does not directly generalize to other
          traffic distributions — cross-dataset evaluation on NSL-KDD showed a
          15-21 point F1 drop, confirmed via independent hyperparameter tuning to
          reflect genuine distribution shift rather than a fixable configuration
          issue.
        </p>
      </div>

      <div className="bg-raised border border-hairline rounded-md p-5">
        <h2 className="font-display text-sm text-ink mb-2">Real-world deployment considerations</h2>
        <p className="text-muted text-xs leading-relaxed">
          This system is a research prototype, not a deployment-ready product. Its
          78 input features are flow statistics computed by CICFlowMeter, not raw
          network data — a real organization's telemetry (NetFlow, firewall logs,
          EDR data) would need translation into this format, and some features may
          not be reconstructable at all from a given environment. Even with matching
          features, real production traffic differs from this 2017 research-testbed
          baseline far more than NSL-KDD does, and modern attack techniques are not
          represented in this training data. Genuine deployment would require
          per-organization retraining on labeled traffic, continuous drift
          monitoring, and likely a hybrid supervised + anomaly-detection approach —
          consistent with how commercial NIDS vendors handle this problem today.
        </p>
      </div>

      <div className="bg-raised border border-hairline rounded-md p-5">
        <h2 className="font-display text-sm text-ink mb-2">Full documentation</h2>
        <p className="text-muted text-xs leading-relaxed">
          See <code className="text-info">docs/research/methodology.md</code> in the
          project repository for the complete methodology, including the 5-test data
          leakage investigation and cross-dataset validation findings.
        </p>
      </div>
    </div>
  );
}