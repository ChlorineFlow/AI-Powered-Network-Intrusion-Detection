"""
Expected schema definitions for each supported dataset.

These schemas describe what a RAW dataset file should look like before
any preprocessing happens. They intentionally do NOT enumerate every one
of CICIDS2017's ~78 flow-based columns (the exact set varies slightly by
which CIC release/CSV you download) — instead we validate a required
subset of critical columns plus general structural checks.

NSL-KDD and UNSW-NB15 have small, stable, well-documented column sets,
so those are listed in full.
"""

# ---------------------------------------------------------------------------
# CICIDS2017
# ---------------------------------------------------------------------------
# CICIDS2017 CSVs are known to have inconsistent column naming across files
# (leading/trailing whitespace, e.g. ' Label' instead of 'Label'). The loader
# strips whitespace from column names before this check runs.
CICIDS2017_REQUIRED_COLUMNS = {
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Label",
}

# ---------------------------------------------------------------------------
# NSL-KDD
# ---------------------------------------------------------------------------
# NSL-KDD's KDDTrain+.txt / KDDTest+.txt files have NO header row.
# This is the standard 41-feature + label + difficulty column order.
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]

# ---------------------------------------------------------------------------
# UNSW-NB15
# ---------------------------------------------------------------------------
# Column names as published by ACCS/UNSW Canberra for the CSV distribution.
UNSW_NB15_REQUIRED_COLUMNS = {
    "dur", "proto", "service", "state", "spkts", "dpkts",
    "sbytes", "dbytes", "rate", "sttl", "dttl",
    "attack_cat", "label",
}