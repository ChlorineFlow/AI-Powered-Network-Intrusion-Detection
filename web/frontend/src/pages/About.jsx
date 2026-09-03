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
        <h2 className="font-display text-sm text-ink mb-2">Known limitations</h2>
        <p className="text-muted text-xs leading-relaxed">
          Performance on CICIDS2017 (99%+ F1) does not directly generalize to other
          traffic distributions — cross-dataset evaluation on NSL-KDD showed a
          15-21 point F1 drop, confirmed via independent hyperparameter tuning to
          reflect genuine distribution shift rather than a fixable configuration
          issue. See the project documentation for full methodology.
        </p>
      </div>
    </div>
  );
}