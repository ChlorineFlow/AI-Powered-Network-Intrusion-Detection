# Research Gap

Based on the literature reviewed (see `literature-review.md`), this
project identifies and addresses the following gaps in how NIDS machine
learning results are commonly reported:

## Gap 1: Single-dataset evaluation presented as general capability

Most reported NIDS results are evaluated on one benchmark dataset, with
no test of whether performance holds on independently-sourced traffic.
**Addressed by:** independently training and evaluating on three
structurally distinct datasets (CICIDS2017, NSL-KDD, UNSW-NB15), with
results compared directly rather than in isolation (see
`docs/experiments/results-analysis.md`).

## Gap 2: High accuracy accepted without investigating its source

A 99%+ accuracy result can arise from genuine detection capability,
data leakage, or class-imbalance exploitation — these are rarely
distinguished in typical project reporting. **Addressed by:** a 5-test
independent leakage investigation (exact-duplicate removal, train/test
performance gap analysis, feature ablation, near-duplicate-flow overlap
detection, cross-method feature-importance verification) and a
mathematically-grounded class-imbalance sanity check (macro-recall
ceiling analysis against a dummy majority-class baseline).

## Gap 3: Generalization gaps are assumed rather than measured

Claims that a model "generalizes well" are rarely tested against a
second dataset with a different traffic-generation methodology.
**Addressed by:** direct cross-dataset comparison showing a measured
15-21 F1-point drop from CICIDS2017 to NSL-KDD, further confirmed via
independent hyperparameter tuning to be a distribution-shift effect
rather than a fixable configuration issue (best cross-validated
training F1: 0.9992; test F1 unchanged at 0.7949).

## Gap 4: Benchmark performance conflated with deployment readiness

Benchmark datasets' features are derived from specific tools
(CICFlowMeter, Argus/Bro-derived pipelines) applied to controlled
testbed or synthetic traffic — not the raw telemetry an organization
would actually have available. **Addressed by:** an explicit "Real-world
deployment considerations" analysis (`methodology.md`) covering feature
availability barriers and traffic distribution shift beyond what
benchmark-to-benchmark comparison already demonstrates.

## Gap 5: No single "best" model claim examined across contexts

Papers proposing a specific algorithm often do not test whether that
algorithm remains best across datasets or tasks. **Addressed by:**
finding that XGBoost — while best on CICIDS2017 and NSL-KDD binary
classification — is *not* the best model on UNSW-NB15 (Random Forest
wins), demonstrating that "best model" claims are dataset- and
task-dependent rather than universal.