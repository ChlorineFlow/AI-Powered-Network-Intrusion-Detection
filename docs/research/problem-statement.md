# Problem Statement

Network intrusion detection systems (NIDS) aim to distinguish malicious
network traffic from legitimate traffic, and — where possible — to
classify the specific type of attack observed. Signature-based
detection, the traditional approach, cannot recognize attacks that do
not match a known signature, motivating machine learning-based
approaches that can, in principle, generalize to novel traffic patterns
by learning statistical properties of attack behavior rather than exact
signatures.

However, published ML-based NIDS research commonly reports very high
accuracy (often 99%+) on a single benchmark dataset without testing
whether that performance holds on independently-sourced traffic, and
without investigating whether the reported accuracy reflects genuine
learned signal versus dataset-specific artifacts (class imbalance,
labeling errors, near-duplicate flows) — a concern directly raised by
prior critiques of widely-used benchmarks such as CICIDS2017 (Engelen et
al., 2021).

## This project's problem statement

This project investigates two connected questions:

1. **Can classical machine learning models effectively distinguish
   malicious from benign network traffic, and classify attack types,
   using flow-level statistical features?** Answered via rigorous,
   leakage-safe training and evaluation across binary and multiclass
   tasks on CICIDS2017.

2. **How much of that performance is dataset-specific, and how much
   reflects genuinely transferable detection capability?** Answered by
   independently training and evaluating the same modeling approach on
   two additional, structurally distinct benchmark datasets (NSL-KDD,
   UNSW-NB15), and by directly testing — rather than assuming — whether
   high accuracy on the primary dataset could be explained by data
   leakage or class imbalance exploitation.

The project additionally investigates the practical gap between
benchmark performance and real-world deployment readiness, given that
benchmark datasets' features are derived from specific flow-extraction
tools and controlled-testbed traffic that differ substantially from
production network telemetry.