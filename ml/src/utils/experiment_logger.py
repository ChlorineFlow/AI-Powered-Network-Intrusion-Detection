"""
Records every experiment's configuration and results to a JSON file
under ml/experiments/results/, per project rules (§24) — every result
must be traceable to a specific run.
"""

import json
import time
from pathlib import Path


def log_experiment(
    results_dir: Path,
    dataset: str,
    task: str,
    model_name: str,
    metrics: dict,
    training_time_seconds: float,
    inference_time_seconds: float,
    random_seed: int,
    extra: dict | None = None,
):
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    record = {
        "timestamp": timestamp,
        "dataset": dataset,
        "task": task,
        "model": model_name,
        "random_seed": random_seed,
        "training_time_seconds": training_time_seconds,
        "inference_time_seconds": inference_time_seconds,
        "metrics": metrics,
    }
    if extra:
        record["extra"] = extra

    filename = f"{dataset}_{task}_{model_name}_{timestamp}.json"
    path = results_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print(f"[experiment_logger] Saved: {path}")
    return path