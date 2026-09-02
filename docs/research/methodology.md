# Methodology

## Multiclass label scope

CICIDS2017 exhibits severe class imbalance at the multiclass level. Based
on EDA of the full dataset (2,830,743 rows), three attack categories have
extremely low sample counts:

| Class                        | Count |
|-------------------------------|------:|
| Heartbleed                    |    11 |
| Web Attack - Sql Injection    |    21 |
| Infiltration                  |    36 |

A stratified train/val/test split (80/10/10) would leave roughly 1-4
samples per split for these classes. Any precision/recall/F1 computed on
so few test samples is not statistically meaningful — a single
misclassification can swing the reported metric by 25-100 percentage
points.

**Decision:** classes with fewer than `min_class_samples` (50, set in
`ml/config.yaml`) total occurrences are excluded from the multiclass
classification task and its evaluation. This is a deliberate scope
limitation, not a data quality fix — the excluded classes are real,
correctly-labeled attacks; they are simply too rare in this dataset for
reliable multiclass evaluation.

These attacks are **not discarded from the project**: they remain
included in the binary classification task (BENIGN vs ATTACK), where
sample size is not a limiting factor.

This threshold and its justification should be stated explicitly in the
final report as a documented limitation, not omitted.

## Duplicate rows

10.89% of CICIDS2017 rows (308,381 of 2,830,743) are exact duplicates —
a known artifact of the dataset's flow-generation process. Duplicates are
removed **before** the train/test split to prevent the same flow record
from appearing in both splits, which would otherwise leak information
from train into test and inflate reported performance.

## Missing and infinite values

`Flow Bytes/s` and `Flow Packets/s` contain a small number of NaN and
Infinity values (well under 0.1% of rows), caused by near-zero-duration
flows producing division by zero in the original feature computation.
These rows are excluded during preprocessing rather than imputed, since
the affected row count is small enough that dropping introduces
negligible bias.

## Data leakage prevention

Per project requirements, the following order is strictly enforced:
1. Deduplicate raw data
2. Remove invalid rows (NaN/Infinity/negative-duration)
3. Train/validation/test split (stratified)
4. Fit scalers, encoders, and feature selection — training split only
5. Apply class imbalance handling (class weights / SMOTE) — training
   split only
6. Evaluate on the untouched test split


## SVM computational scope

Support Vector Machines scale poorly with dataset size (roughly O(n²) to
O(n³) for the RBF kernel used here). Training on the full 1,764,483-row
CICIDS2017 training set is computationally impractical in this project's
timeframe. SVM is therefore trained and evaluated on a stratified
subsample of 50,000 rows from the training set, preserving the original
class balance. This is disclosed explicitly as a scope limitation — SVM
results in this project are not directly comparable to Random
Forest/XGBoost/baseline results trained on the full dataset, and any
reported SVM metrics should be read with that caveat.


## Data leakage investigation

Given the high performance metrics observed across all models (>99%
accuracy on binary and multiclass tasks), two specific leakage
hypotheses were tested rather than assumed absent:

**1. Classical overfitting (memorization).** Train-set vs test-set
performance was compared for the best-performing model (XGBoost,
binary). F1 (macro) gap: 0.9987 (train) vs 0.9986 (test) — a gap of
0.0001. This rules out memorization-based overfitting: a model that had
simply memorized training rows would show materially worse performance
on held-out test data.

**2. Feature-artifact dependence.** CICIDS2017's testbed generated
specific attacks against fixed destination ports (e.g. FTP-Patator
against port 21), raising the concern that the model might be learning
a port-to-label lookup rather than genuine flow-behavior patterns.
An ablation test retrained XGBoost with `Destination Port` removed
entirely: F1 (macro) dropped from 0.9986 to 0.9982 — a negligible
0.04 percentage point change. This indicates the model's performance
is not substantially dependent on this feature.

**Acknowledged, unresolved limitation: near-duplicate flow bursts.**
Neither test above rules out a structural property of CICIDS2017: attack
tools used to generate the dataset (e.g. Hulk, GoldenEye) fire large
numbers of near-identical requests in rapid succession. Row-level exact
deduplication (already applied, Step 4) cannot catch flows that differ
only slightly (e.g. by timestamp-derived fields) while representing
the same underlying attack burst. If such near-duplicates are split
across train and test, the model could appear to generalize well while
actually recognizing minor variations of patterns it has already seen.
This is a documented, known critique of CICIDS2017-based research
(e.g. Engelen et al., "Troubleshooting an Intrusion Detection Dataset").
It is not fully resolvable through row-level techniques and is instead
addressed empirically via cross-dataset evaluation (RQ5): if performance
holds up on NSL-KDD and UNSW-NB15 — datasets with entirely different
generation processes — that is stronger evidence of genuine
generalization than same-dataset test-set performance alone.