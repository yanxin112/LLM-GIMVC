import json
from pathlib import Path

import numpy as np


def normalize_missing_rate(missing_rate):
    value = float(missing_rate)
    if value > 1.0:
        percent = int(round(value))
        fraction = percent / 100.0
    else:
        fraction = value
        percent = int(round(value * 100))
    return {
        "percent": int(percent),
        "fraction": float(fraction),
        "dir_name": f"missing_{int(percent)}",
    }


def get_block1_job_dir(output_root, dataset, missing_pattern, missing_rate, method, seed):
    rate = normalize_missing_rate(missing_rate)
    return Path(output_root) / dataset / missing_pattern / rate["dir_name"] / method / f"seed_{seed}"


def get_block1_metrics_path(output_root, dataset, missing_pattern, missing_rate, method, seed):
    return get_block1_job_dir(output_root, dataset, missing_pattern, missing_rate, method, seed) / "metrics.json"


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def make_json_safe(obj):
    try:
        import torch
    except Exception:
        torch = None

    if isinstance(obj, dict):
        return {str(key): make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(value) for value in obj]
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch is not None and torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    return obj


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(obj), f, indent=2, ensure_ascii=False)


def method_to_stage4b_args(method):
    method = str(method).lower()
    if method == "llm_gimvc":
        return {
            "input_source": "fusion",
            "requires_stage4b": True,
            "requires_external_baseline": False,
        }
    if method == "statistical_only":
        return {
            "input_source": "statistical_only",
            "requires_stage4b": True,
            "requires_external_baseline": False,
        }
    if method in ["mica", "jga_imvc", "freecsl"]:
        return {
            "input_source": None,
            "requires_stage4b": False,
            "requires_external_baseline": True,
        }
    raise ValueError(f"Unknown Block 1 method: {method}")


def _metric_value(metrics, name):
    for key, value in metrics.items():
        if key.lower() == name.lower():
            return float(value)
    raise KeyError(f"Metric '{name}' not found. Available metrics: {list(metrics.keys())}")


def extract_metrics_from_method_metrics(method_metrics, primary="kmeans_on_head_representation"):
    if primary not in method_metrics:
        raise KeyError(f"Primary metric group '{primary}' not found in method metrics.")
    metrics = method_metrics[primary]
    return {
        "NMI": _metric_value(metrics, "NMI"),
        "ARI": _metric_value(metrics, "ARI"),
        "ACC": _metric_value(metrics, "ACC"),
        "Purity": _metric_value(metrics, "Purity"),
    }


def extract_metric_group(method_metrics, group_name):
    if group_name not in method_metrics:
        raise KeyError(f"Metric group '{group_name}' not found in method metrics.")
    metrics = method_metrics[group_name]
    return {
        "NMI": _metric_value(metrics, "NMI"),
        "ARI": _metric_value(metrics, "ARI"),
        "ACC": _metric_value(metrics, "ACC"),
        "Purity": _metric_value(metrics, "Purity"),
    }
