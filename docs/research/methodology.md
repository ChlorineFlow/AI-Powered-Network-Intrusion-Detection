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