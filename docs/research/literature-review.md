# Literature Review

## Foundational dataset papers

**CICIDS2017.** Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018).
"Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic
Characterization." *4th International Conference on Information Systems
Security and Privacy (ICISSP)*, Portugal, pp. 108-116. Introduces
CICIDS2017: five days of labeled traffic (July 3-7, 2017) combining
B-Profile-generated benign background traffic with common contemporary
attacks (Brute Force, DoS, Heartbleed, Web Attacks, Infiltration, Botnet,
DDoS, PortScan), processed into 80+ flow features via the CICFlowMeter
tool. This is the primary dataset used in this project.

**NSL-KDD.** Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009).
"A Detailed Analysis of the KDD CUP 99 Data Set." *IEEE Symposium on
Computational Intelligence for Security and Defense Applications
(CISDA)*, pp. 1-6. Identifies statistical redundancy problems in the
original KDD Cup 99 dataset (duplicate records causing classifiers to
bias toward frequent attack types) and proposes NSL-KDD as a corrected,
de-duplicated successor with no shared records between its train and
test sets.

**UNSW-NB15.** Moustafa, N., & Slay, J. (2015). "UNSW-NB15: A
Comprehensive Data Set for Network Intrusion Detection Systems."
*Military Communications and Information Systems Conference (MilCIS)*,
pp. 1-6. Introduces UNSW-NB15, generated via the IXIA PerfectStorm tool
to produce a hybrid of real modern normal traffic and synthetic
contemporary attack behaviors, with 49 features spanning flow, content,
time, and additional generated attributes, labeled with both a binary
class and a 9-category `attack_cat` taxonomy.

## Dataset quality critique

**Engelen, G., Rimmer, V., & Joosen, W. (2021).** "Troubleshooting an
Intrusion Detection Dataset: the CICIDS2017 Case Study." *IEEE Security
and Privacy Workshops (SPW)*, pp. 7-12. This paper directly motivated
the leakage investigation performed in this project. Engelen et al.
identify multiple issues in CICIDS2017's original construction: labeling
and flow-timing-window inaccuracies, a duplicated `Fwd Header Length`
feature caused by a CICFlowMeter bug (a bug this project's own EDA
independently observed as `Fwd Header Length.1` in the raw column list —
consistent with their finding), and flow-construction artifacts from
improper TCP connection termination handling. Later work (Liu et al.,
2022; Lanvin et al., 2023, as cited in subsequent NIDS literature)
extends this critique with further documentation of packet misordering,
duplicate flows, and labeling errors across CIC-produced datasets.

These findings are the direct reason this project did not treat
CICIDS2017's headline 99%+ accuracy at face value, and instead conducted
an independent 5-test leakage investigation (exact-duplicate removal,
train/test performance gap analysis, single-feature ablation, near-
duplicate-flow overlap detection, and cross-method feature-importance
verification) before accepting the result — see
`docs/research/methodology.md`.

## Positioning of this project relative to the literature

Much published NIDS research reports strong results on a single
benchmark dataset without cross-dataset validation, a gap explicitly
identified as a limitation in prior work (e.g. Ring et al.'s widely-cited
survey of network-based intrusion detection datasets discusses the lack
of standardized cross-dataset evaluation practice in the field). This
project addresses that gap directly by independently training and
evaluating models on three structurally distinct datasets (CICIDS2017,
NSL-KDD, UNSW-NB15) rather than reporting a single dataset's results as
if they were representative of general NIDS performance, and by
explicitly measuring — rather than assuming — the resulting
generalization gap (15-21 F1 points between CICIDS2017 and NSL-KDD,
confirmed via independent hyperparameter tuning to reflect genuine
distribution shift rather than a fixable modeling choice).

## Scope note

This is a project-level literature review supporting the methodology
and interpretation choices made in this repository, not an exhaustive
systematic survey of the NIDS machine learning literature. Citations
above were independently verified against their original publication
venues.