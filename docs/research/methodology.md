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


## Further leakage and importance verification

Two additional diagnostics were run to address the near-duplicate-flow
concern raised above and to independently verify feature importance.

**Near-duplicate overlap check.** Numeric features were rounded to 1
decimal place and used as an approximate fingerprint to detect
near-identical rows shared between train and test. Only 1.30% of test
rows (6,547 of 504,139) had a near-duplicate match in the training set,
and this overlap was concentrated almost entirely in BENIGN traffic
(1.53%) — nearly every attack class showed 0.0% overlap. This weighs
against the near-duplicate-attack-burst leakage hypothesis: if attack
tools' rapid-fire requests were being split across train/test, high
overlap would be expected specifically in attack classes, which is not
observed.

**Permutation importance cross-check.** SHAP's global feature ranking
(Step 10) was cross-validated against permutation importance — an
independent method measuring actual F1-macro degradation when each
feature is randomly shuffled. Five of the top six features matched
between both methods (Destination Port, Init_Win_bytes_backward,
Init_Win_bytes_forward, Bwd Packet Length Std, Average Packet Size),
providing strong, method-independent evidence that the model's reliance
on these features reflects genuine learned signal rather than an
artifact of one particular explainability technique.

**Conclusion.** Across five checks (exact-duplicate removal, train/test
performance gap, single-feature ablation, near-duplicate overlap, and
cross-method importance verification), no evidence of data leakage or
memorization was found. The remaining open question is not leakage but
generalization: `Destination Port` remains the single most important
feature by both methods, and CICIDS2017's testbed used fixed ports for
specific attacks. Whether this reflects genuine transferable attack
behavior or a dataset-specific artifact can only be resolved by
cross-dataset evaluation (RQ5) on NSL-KDD and UNSW-NB15.


## Cross-dataset generalization findings (RQ5)

Models were independently trained and evaluated on NSL-KDD using its
own standard fixed train/test split (KDDTrain+.txt / KDDTest+.txt),
following the same rigorous methodology as CICIDS2017 (leakage-safe
encoding fit only on training data, full multi-metric evaluation).
Note: models are NOT transferred between datasets — CICIDS2017's 78
flow-statistics features and NSL-KDD's 41 connection-record features
are structurally different and not interchangeable. Instead, results
and conclusions are compared across the two independently-trained
pipelines.

| Model               | CICIDS2017 F1 (macro) | NSL-KDD F1 (macro) | Drop   |
|----------------------|-----------------------:|---------------------:|-------:|
| Logistic Regression  | 0.903                  | 0.755                 | -0.148 |
| Decision Tree        | 0.998                  | 0.791                 | -0.207 |
| Random Forest        | 0.998                  | 0.784                 | -0.214 |
| XGBoost              | 0.999                  | 0.798                 | -0.201 |

**Every model showed a 15-21 point F1 drop on NSL-KDD relative to
CICIDS2017.** This is attributed to two known, documented properties
of NSL-KDD rather than a flaw in the modeling approach:

1. NSL-KDD's test set intentionally includes attack subtypes absent
   from the training set (e.g. `apache2`, `mailbomb`, `snmpguess`,
   `processtable`), by design, to test generalization to unknown
   attacks — unlike CICIDS2017's stratified split, where every class
   present in test was also present in training.
2. NSL-KDD's 41 KDD-style connection-record features are coarser and
   more aggregated than CICIDS2017's 78 fine-grained flow-timing
   statistics, offering less discriminative signal per record.

Notably, ROC-AUC remained comparatively high for the tree-based models
(0.96-0.97) even as F1 dropped sharply, indicating the models retain
reasonable probabilistic ranking ability but the default classification
threshold is miscalibrated for NSL-KDD's harder distribution — high
false negative rates (33-37%) drove most of the F1 loss.

**Interpretation relative to the data leakage investigation above.**
This finding provides independent, external support for the conclusion
that CICIDS2017's near-perfect results are not an artifact of leakage:
if the CICIDS2017 pipeline had been exploiting a leakage bug, that bug
would not predict or explain a consistent, large performance drop when
the same rigorous methodology is applied to an honestly-harder dataset.
The magnitude and consistency of the drop across every model is
instead the expected signature of genuine train/test difficulty
differences between the two datasets — a legitimate answer to RQ5, and
a caution against treating CICIDS2017 benchmark numbers as a general
measure of real-world NIDS performance.


## Hyperparameter tuning confirms the gap is generalization, not configuration

To test whether the NSL-KDD performance gap (Step: cross-dataset
findings above) was simply due to using CICIDS2017-tuned hyperparameters
on NSL-KDD, XGBoost was independently tuned for NSL-KDD using
RandomizedSearchCV (30 configurations, 5-fold stratified cross-validation,
150 total fits, training data only — test labels never touched during
search).

**Best cross-validated F1 (macro) on training folds: 0.9992**
**Same tuned model's F1 (macro) on the held-out test set: 0.7949**

This ~20-point gap between cross-validated training performance and
actual test performance — despite extensive tuning reaching near-perfect
scores on the training distribution — is strong direct evidence that
the earlier-observed NSL-KDD performance drop is not a hyperparameter
configuration problem. No tuning of an XGBoost model's internal
parameters can supply information about attack patterns absent from the
training data entirely, and NSL-KDD's test set is explicitly
constructed to include exactly such novel attack types. Tuning
therefore confirms rather than resolves the earlier finding: the
gap reflects a genuine train/test distribution shift built into the
dataset's design, not a fixable modeling shortcoming.


## Future work: unified cross-dataset model

A single model trained across CICIDS2017, NSL-KDD, and UNSW-NB15
simultaneously was considered but not attempted in this project's
scope. The three datasets do not share a feature space (78 vs 41 vs ~49
features, measuring different things), and their label taxonomies are
not directly comparable (e.g. CICIDS2017's "DoS Hulk" vs NSL-KDD's
"neptune"). A unified model would require: (1) identifying a common
feature subset across all three, discarding each dataset's richest
dataset-specific features in the process, and (2) manually constructing
and justifying a cross-dataset attack-category label mapping. Both are
themselves open research problems in NIDS literature (domain
generalization / cross-dataset transfer), and a naive attempt would be
expected to underperform the dataset-specific models already trained
in this project. This is documented as a limitation and a direction
for future work rather than attempted here.


## Real-world deployment considerations

The models trained in this project achieve strong results on CICIDS2017
(99%+ F1) and demonstrate a measured, honest generalization gap on
NSL-KDD (15-21 F1 points). Neither result should be read as evidence
that this system is ready for direct deployment on an arbitrary
organization's live network. Two distinct barriers exist.

**1. Feature availability.** CICIDS2017's 78 features are not raw
network data — they are flow statistics computed by CICFlowMeter from
packet captures (e.g. `Flow Bytes/s`, `Init_Win_bytes_backward`,
`Fwd Packet Length Std`). A deploying organization's raw telemetry
(NetFlow/sFlow records, firewall logs, Zeek/Bro logs, cloud VPC flow
logs, EDR telemetry) is not in this format. Deployment would require
either running CICFlowMeter (or an equivalent tool) against the
organization's traffic to reproduce matching columns, or building a
custom translation layer — and some features may not be reconstructable
from a given organization's available telemetry at all.

**2. Distribution shift beyond what NSL-KDD demonstrated.** CICIDS2017
represents synthetic 2017 university-testbed traffic generated with
specific attack tools (Hulk, GoldenEye, Slowloris) in a controlled lab
environment. A real organization's traffic — cloud SaaS calls,
containerized microservices, VPN tunnels, IoT devices, encrypted
channels — differs from this baseline far more than NSL-KDD's benchmark
traffic does, and this project's own cross-dataset experiment already
showed a 15-21 point F1 drop moving between two *research benchmark*
datasets. Additionally, attack techniques evolve: 2017-era volumetric
DoS tools bear little resemblance to modern techniques such as
encrypted C2 channels, cloud credential abuse, or living-off-the-land
attacks, none of which appear in this training data.

**What genuine deployment would require**, consistent with how
commercial NIDS/EDR vendors approach this problem (per-customer
baselining rather than one static pretrained model):
1. A feature-extraction pipeline matched to the organization's actual
   available telemetry.
2. Retraining or fine-tuning on the organization's own labeled traffic
   — in practice the primary bottleneck, since labeled attack examples
   are rarely available.
3. Continuous retraining / drift monitoring as traffic and attack
   techniques evolve over time.
4. Likely a hybrid detection approach combining supervised
   classification (for known attack signatures) with anomaly detection
   (to catch novel, unlabeled attack patterns the supervised model was
   never trained on).
5. Human-in-the-loop alert review rather than fully automated blocking,
   given realistic false-positive and false-negative rates on
   previously-unseen traffic distributions.

This project is scoped as an academic research system demonstrating
rigorous methodology (leakage-safe evaluation, cross-dataset validation,
explainability, documented limitations) rather than a
deployment-ready product, and this section is presented as a known,
explicitly acknowledged limitation rather than an oversight.