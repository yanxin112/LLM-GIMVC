#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from clustering_eval import clustering_metrics


METHODS = ("mica", "jga_imvc", "freecsl")
METRICS = ("NMI", "ARI", "ACC", "Purity")
ENTRYPOINTS = {
    "mica": Path("external_baselines/MICA/run_mica.py"),
    "jga_imvc": Path("external_baselines/JGA-IMVC/run_jga_imvc.py"),
    "freecsl": Path("external_baselines/FreeCSL/run_freecsl.py"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Independently verify Stage 5C external baseline outputs.")
    parser.add_argument("--dataset", default="BDGP")
    parser.add_argument("--missing-pattern", default="MCAR")
    parser.add_argument("--missing-rate", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-root", default="results/block1")
    parser.add_argument("--baseline-data-root", default="results/baseline_data")
    parser.add_argument("--raw-output-root", default="results/external_baselines")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--report-json", default="refine-logs/stage5c/external_baseline_verification.json")
    parser.add_argument("--report-md", default="refine-logs/stage5c/external_baseline_verification.md")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path, payload, overwrite):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Report already exists: {path}. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_text(path, text, overwrite):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Report already exists: {path}. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def path_variants(root, dataset, pattern, rate, tail):
    patterns = []
    for value in (pattern, pattern.lower(), pattern.upper()):
        if value not in patterns:
            patterns.append(value)
    return [Path(root) / dataset / value / f"missing_{rate}" / tail for value in patterns]


def first_existing(candidates):
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_metrics(path):
    payload = load_json(path)
    values = payload.get("metrics", payload)
    aliases = {str(key).lower(): value for key, value in values.items()}
    normalized = {}
    for name in METRICS:
        key = name.lower()
        if key not in aliases:
            raise KeyError(f"Metric {name} missing from {path}.")
        normalized[name] = float(aliases[key])
    return normalized, payload


def read_csv_vector(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"Prediction CSV is empty: {path}")
    header = [cell.strip().lower() for cell in rows[0]]
    known = ("pred_labels", "pred", "y_pred", "labels_pred", "cluster", "label")
    has_header = any(cell in known for cell in header)
    column = next((header.index(name) for name in known if name in header), 0)
    data_rows = rows[1:] if has_header else rows
    return np.asarray([float(row[column]) for row in data_rows if row])


def read_mat_vector(path):
    try:
        import scipy.io as sio
    except Exception as exc:
        raise ImportError("scipy is required to read .mat predictions.") from exc
    payload = sio.loadmat(path)
    for key in ("pred_labels", "pred", "y_pred", "labels_pred", "cluster", "labels", "y_pred_final"):
        if key in payload:
            return np.asarray(payload[key]).reshape(-1)
    raise KeyError(f"No prediction vector found in {path}.")


def read_predictions(path):
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False)).reshape(-1)
    if suffix == ".csv":
        return read_csv_vector(path).reshape(-1)
    if suffix == ".mat":
        return read_mat_vector(path).reshape(-1)
    raise ValueError(f"Unsupported prediction format: {path}")


def prediction_candidates(result_dir, raw_dirs):
    names = ("pred_labels.npy", "pred_labels.csv", "pred_labels.mat")
    candidates = [result_dir / name for name in names]
    for raw_dir in raw_dirs:
        candidates.extend(raw_dir / name for name in names)
        if raw_dir.is_dir():
            for suffix in ("*.npy", "*.csv", "*.mat"):
                candidates.extend(sorted(raw_dir.rglob(suffix)))
    seen = set()
    unique = []
    for candidate in candidates:
        key = candidate.resolve() if candidate.exists() else candidate
        if str(key) not in seen:
            seen.add(str(key))
            unique.append(candidate)
    return unique


def placeholder_entrypoint(path):
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    markers = ("fake mica", "fake jga-imvc", "fake freecsl", "rng.integers")
    return any(marker in text for marker in markers)


def status_for(errors, mismatch, one_cluster, invalid_clusters):
    if errors:
        return "FAILED"
    if invalid_clusters:
        return "INVALID_CLUSTER_COUNT"
    if one_cluster:
        return "DEGENERATE_ONE_CLUSTER"
    if mismatch:
        return "METRIC_MISMATCH"
    return "PASS"


def verify_method(args, method, labels, expected_clusters, data_error):
    result_dir = first_existing(
        path_variants(
            args.results_root,
            args.dataset,
            args.missing_pattern,
            args.missing_rate,
            Path(method) / f"seed_{args.seed}",
        )
    )
    metrics_path = result_dir / "metrics.json"
    command_path = result_dir / "command_log.json"
    summary_path = result_dir / "job_summary.json"
    errors = []
    notes = []
    entrypoint = Path(args.repo_root) / ENTRYPOINTS[method]
    placeholder_detected = placeholder_entrypoint(entrypoint)
    metrics = None
    metrics_payload = {}
    if metrics_path.is_file():
        try:
            metrics, metrics_payload = load_metrics(metrics_path)
        except Exception as exc:
            errors.append(f"metrics unreadable: {exc}")

    raw_dirs = [result_dir / "raw"]
    source = metrics_payload.get("source", {}) if isinstance(metrics_payload, dict) else {}
    if source.get("raw_output_dir"):
        raw_dirs.append(Path(source["raw_output_dir"]))
    raw_dirs.extend(
        path_variants(
            Path(args.raw_output_root) / method,
            args.dataset,
            args.missing_pattern,
            args.missing_rate,
            Path(f"seed_{args.seed}"),
        )
    )
    wrapper_status_path = next(
        (raw_dir / "wrapper_status.json" for raw_dir in raw_dirs if (raw_dir / "wrapper_status.json").is_file()),
        None,
    )
    wrapper_status = load_json(wrapper_status_path) if wrapper_status_path else None
    if wrapper_status and wrapper_status.get("error"):
        notes.append(str(wrapper_status["error"]))
    prediction_path = next((path for path in prediction_candidates(result_dir, raw_dirs) if path.is_file()), None)

    if placeholder_detected:
        errors.append(f"placeholder implementation detected: {entrypoint.as_posix()}")
    if data_error:
        errors.append(data_error)
    if not result_dir.is_dir():
        errors.append("result directory missing")
    if not metrics_path.is_file():
        errors.append("metrics.json missing")
    if not command_path.is_file():
        errors.append("command_log.json missing")
    if not summary_path.is_file():
        errors.append("job_summary.json missing")
    if prediction_path is None:
        errors.append("prediction file missing")

    pred_len = None
    num_clusters = None
    recomputed = {}
    mismatch = False
    one_cluster = False
    invalid_clusters = False
    if prediction_path is not None and labels is not None:
        try:
            predictions = read_predictions(prediction_path)
            pred_len = int(predictions.size)
            num_clusters = int(np.unique(predictions).size)
            if pred_len != int(labels.size):
                errors.append(f"prediction length mismatch: {pred_len} != {labels.size}")
            else:
                one_cluster = num_clusters == 1
                upper_bound = max(int(expected_clusters or 0) * 4, int(np.sqrt(labels.size)) + 1)
                invalid_clusters = num_clusters < 1 or num_clusters > labels.size or num_clusters > upper_bound
                recomputed = clustering_metrics(labels, predictions)
                if metrics is not None:
                    differences = {name: abs(recomputed[name] - metrics[name]) for name in METRICS}
                    mismatch = any(value > args.tolerance for value in differences.values())
                    notes.append("metric_differences=" + json.dumps(differences, sort_keys=True))
        except Exception as exc:
            errors.append(f"prediction read/recompute failed: {exc}")

    status = status_for(errors, mismatch, one_cluster, invalid_clusters)
    notes.extend(errors)
    return {
        "method": method,
        "result_dir": result_dir.as_posix(),
        "result_dir_exists": result_dir.is_dir(),
        "metrics_exists": metrics_path.is_file(),
        "command_log_exists": command_path.is_file(),
        "job_summary_exists": summary_path.is_file(),
        "placeholder_detected": placeholder_detected,
        "wrapper_status_file": wrapper_status_path.as_posix() if wrapper_status_path else None,
        "wrapper_status": wrapper_status.get("status") if wrapper_status else None,
        "prediction_file": prediction_path.as_posix() if prediction_path else None,
        "pred_len": pred_len,
        "label_len": int(labels.size) if labels is not None else None,
        "num_pred_clusters": num_clusters,
        "nmi_recomputed": recomputed.get("NMI"),
        "nmi_metrics_json": metrics.get("NMI") if metrics else None,
        "metrics_recomputed": recomputed,
        "metrics_json": metrics,
        "status": status,
        "notes": notes,
    }


def markdown_report(payload):
    rows = [
        "# Stage 5C External Baseline Verification",
        "",
        f"- Data bundle: `{payload['data_path']}`",
        f"- Data keys valid: `{payload['data_keys_valid']}`",
        "",
        "| method | result_dir_exists | metrics_exists | command_log_exists | prediction_file | pred_len | label_len | num_pred_clusters | nmi_recomputed | nmi_metrics_json | status | notes |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in payload["results"]:
        values = [
            item["method"],
            str(item["result_dir_exists"]),
            str(item["metrics_exists"]),
            str(item["command_log_exists"]),
            item["prediction_file"] or "",
            item["pred_len"] if item["pred_len"] is not None else "",
            item["label_len"] if item["label_len"] is not None else "",
            item["num_pred_clusters"] if item["num_pred_clusters"] is not None else "",
            f"{item['nmi_recomputed']:.8f}" if item["nmi_recomputed"] is not None else "",
            f"{item['nmi_metrics_json']:.8f}" if item["nmi_metrics_json"] is not None else "",
            item["status"],
            "; ".join(item["notes"]).replace("|", "\\|"),
        ]
        rows.append("| " + " | ".join(str(value) for value in values) + " |")
    rows.extend(["", payload["conclusion"], ""])
    return "\n".join(rows)


def main():
    args = parse_args()
    data_path = first_existing(
        path_variants(
            args.baseline_data_root,
            args.dataset,
            args.missing_pattern,
            args.missing_rate,
            Path(f"seed_{args.seed}") / "data.npz",
        )
    )
    labels = None
    expected_clusters = None
    data_error = None
    required_keys = ("labels", "available_mask", "missing_mask")
    data_keys = []
    if not data_path.is_file():
        data_error = f"baseline data missing: {data_path.as_posix()}"
    else:
        try:
            with np.load(data_path, allow_pickle=False) as bundle:
                data_keys = sorted(bundle.files)
                missing = [key for key in required_keys if key not in bundle]
                if missing:
                    data_error = f"baseline data missing required keys: {missing}"
                else:
                    labels = np.asarray(bundle["labels"]).reshape(-1)
                    expected_clusters = int(np.unique(labels).size)
        except Exception as exc:
            data_error = f"baseline data unreadable: {exc}"

    results = [
        verify_method(args, method, labels, expected_clusters, data_error)
        for method in METHODS
    ]
    passed = data_error is None and all(item["status"] == "PASS" for item in results)
    conclusion = (
        "External baseline verification passed."
        if passed
        else "External baseline verification failed. Do not launch Stage 5C full matrix."
    )
    payload = {
        "dataset": args.dataset,
        "missing_pattern": args.missing_pattern,
        "missing_rate": args.missing_rate,
        "seed": args.seed,
        "data_path": data_path.as_posix(),
        "data_keys": data_keys,
        "data_keys_valid": data_error is None,
        "data_error": data_error,
        "tolerance": args.tolerance,
        "passed": passed,
        "results": results,
        "conclusion": conclusion,
    }
    write_json(args.report_json, payload, args.overwrite)
    write_text(args.report_md, markdown_report(payload), args.overwrite)
    print(conclusion)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
