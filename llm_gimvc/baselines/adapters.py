import csv
import json
from pathlib import Path

import numpy as np

from clustering_eval import clustering_metrics


REQUIRED_METRICS = ["NMI", "ARI", "ACC", "Purity"]


def load_labels_from_data_bundle(data_path):
    data_path = Path(data_path)
    if data_path.suffix.lower() == ".npz":
        with np.load(data_path, allow_pickle=True) as data:
            return np.asarray(data["labels"]).reshape(-1)
    if data_path.suffix.lower() == ".mat":
        try:
            import scipy.io as sio
        except Exception as exc:
            raise ImportError("scipy is required to read .mat baseline bundles") from exc
        mat = sio.loadmat(data_path)
        if "labels" not in mat:
            raise KeyError(f"labels not found in {data_path}")
        return np.asarray(mat["labels"]).reshape(-1)
    raise ValueError(f"Unsupported data bundle format: {data_path}")


def normalize_metric_keys(metrics):
    aliases = {
        "nmi": "NMI",
        "ari": "ARI",
        "acc": "ACC",
        "accuracy": "ACC",
        "purity": "Purity",
        "pur": "Purity",
    }
    normalized = {}
    for key, value in metrics.items():
        canonical = aliases.get(str(key).strip().lower())
        if canonical is not None:
            normalized[canonical] = float(np.asarray(value).reshape(-1)[0])

    missing = [key for key in REQUIRED_METRICS if key not in normalized]
    if missing:
        raise ValueError(f"Missing required metric keys: {missing}. Available keys: {list(metrics.keys())}")
    return {key: float(normalized[key]) for key in REQUIRED_METRICS}


def compute_metrics_from_predictions(labels, pred_labels):
    labels = np.asarray(labels).reshape(-1)
    pred_labels = np.asarray(pred_labels).reshape(-1)
    if labels.shape[0] != pred_labels.shape[0]:
        raise ValueError(f"Prediction length mismatch: pred={pred_labels.shape[0]}, labels={labels.shape[0]}")
    return clustering_metrics(labels, pred_labels)


def _read_json_metrics(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        obj = json.load(f)
    if "metrics" in obj and isinstance(obj["metrics"], dict):
        obj = obj["metrics"]
    return normalize_metric_keys(obj)


def _read_csv_metrics(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    selected = rows[-1]
    for row in rows:
        marker = " ".join(str(value).lower() for value in row.values())
        if "final" in marker or "best" in marker:
            selected = row
    return normalize_metric_keys(selected)


def _read_mat_metrics(path):
    try:
        import scipy.io as sio
    except Exception:
        return None
    mat = sio.loadmat(path)
    candidates = {}
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        candidates[key] = value
    try:
        return normalize_metric_keys(candidates)
    except Exception:
        expanded = {}
        for key, value in candidates.items():
            lower = key.lower()
            if lower.endswith("_mean"):
                expanded[lower[:-5]] = value
            else:
                expanded[key] = value
        return normalize_metric_keys(expanded)


def read_raw_metrics(raw_output_dir, return_path=False):
    raw_output_dir = Path(raw_output_dir)
    candidates = [
        ("metrics.json", _read_json_metrics),
        ("result.json", _read_json_metrics),
        ("metrics.csv", _read_csv_metrics),
        ("result.csv", _read_csv_metrics),
        ("result.mat", _read_mat_metrics),
    ]
    errors = []
    for filename, reader in candidates:
        path = raw_output_dir / filename
        if not path.exists():
            continue
        try:
            metrics = reader(path)
            if metrics is not None:
                return (metrics, path) if return_path else metrics
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    if return_path:
        return None, None
    return None


def _read_vector_csv(path):
    values = []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        return None
    header = [cell.strip().lower() for cell in rows[0]]
    start_idx = 1 if any(cell in ["pred", "y_pred", "label", "labels"] for cell in header) else 0
    pred_col = 0
    if start_idx == 1:
        for idx, cell in enumerate(header):
            if cell in ["pred", "y_pred", "labels_pred", "cluster", "label"]:
                pred_col = idx
                break
    for row in rows[start_idx:]:
        if not row:
            continue
        values.append(float(row[pred_col]))
    return np.asarray(values)


def _read_mat_predictions(path):
    try:
        import scipy.io as sio
    except Exception:
        return None
    mat = sio.loadmat(path)
    for key in ["pred", "y_pred", "labels", "y_pred_final"]:
        if key in mat:
            return np.asarray(mat[key]).reshape(-1)
    return None


def read_pred_labels(raw_output_dir, return_path=False):
    raw_output_dir = Path(raw_output_dir)
    npy_candidates = ["pred_labels.npy", "y_pred.npy", "labels_pred.npy"]
    for filename in npy_candidates:
        path = raw_output_dir / filename
        if path.exists():
            pred = np.load(path, allow_pickle=True)
            return (np.asarray(pred).reshape(-1), path) if return_path else np.asarray(pred).reshape(-1)

    csv_candidates = ["pred_labels.csv", "pred.csv", "y_pred.csv"]
    for filename in csv_candidates:
        path = raw_output_dir / filename
        if path.exists():
            pred = _read_vector_csv(path)
            if pred is not None:
                return (pred, path) if return_path else pred

    for filename in ["pred_labels.mat", "result.mat"]:
        mat_path = raw_output_dir / filename
        if mat_path.exists():
            pred = _read_mat_predictions(mat_path)
            if pred is not None:
                return (pred, mat_path) if return_path else pred

    if return_path:
        return None, None
    return None


def adapt_external_result(method, data_path, raw_output_dir):
    raw_output_dir = Path(raw_output_dir)
    pred_labels, pred_labels_path = read_pred_labels(raw_output_dir, return_path=True)
    if pred_labels is not None:
        labels = load_labels_from_data_bundle(data_path)
        return {
            "metrics": compute_metrics_from_predictions(labels, pred_labels),
            "adapter_source": "pred_labels",
            "raw_metrics_path": None,
            "pred_labels_path": pred_labels_path.as_posix(),
            "pred_labels": np.asarray(pred_labels).reshape(-1),
        }

    metrics, raw_metrics_path = read_raw_metrics(raw_output_dir, return_path=True)
    if metrics is not None:
        raise FileNotFoundError(
            f"{method} produced metrics but no prediction labels under {raw_output_dir.as_posix()}. "
            "Stage 5B requires pred_labels for independent verification."
        )

    raise FileNotFoundError(
        f"No prediction labels found for {method} under {raw_output_dir.as_posix()}"
    )
