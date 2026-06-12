import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np


PREDICTION_NAMES = (
    "pred_labels.npy",
    "y_pred.npy",
    "labels_pred.npy",
    "pred_labels.csv",
    "pred.csv",
    "y_pred.csv",
    "pred_labels.mat",
    "result.mat",
)


def parse_args(display_name):
    parser = argparse.ArgumentParser(
        description=f"Adapter wrapper for the official {display_name} implementation."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--missing-rate", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--official-command",
        default=None,
        help=(
            "Official command template. Supported placeholders: {data}, {output}, "
            "{dataset}, {missing_rate}, {seed}, {device}."
        ),
    )
    return parser.parse_args()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_and_validate_bundle(path):
    required = {
        "complete_views",
        "masked_views",
        "available_mask",
        "missing_mask",
        "labels",
        "metadata",
    }
    with np.load(path, allow_pickle=True) as bundle:
        missing = sorted(required - set(bundle.files))
        if missing:
            raise ValueError(f"Unified data bundle is missing required keys: {missing}")
        payload = {
            key: np.asarray(bundle[key])
            for key in bundle.files
            if key != "labels"
        }
        labels_length = int(np.asarray(bundle["labels"]).reshape(-1).size)
    return payload, labels_length


def find_prediction(output_dir):
    for name in PREDICTION_NAMES:
        path = output_dir / name
        if path.is_file():
            return path
    for suffix in ("*.npy", "*.csv", "*.mat"):
        for path in sorted(output_dir.rglob(suffix)):
            if path.name != "training_data_no_labels.npz":
                return path
    return None


def normalize_prediction(source, target):
    suffix = source.suffix.lower()
    if suffix == ".npy":
        values = np.asarray(np.load(source, allow_pickle=False)).reshape(-1)
    elif suffix == ".csv":
        values = np.loadtxt(source, delimiter=",", skiprows=1)
        values = np.asarray(values).reshape(-1)
    elif suffix == ".mat":
        try:
            import scipy.io as sio
        except Exception as exc:
            raise ImportError("scipy is required to normalize .mat predictions.") from exc
        payload = sio.loadmat(source)
        values = None
        for key in ("pred_labels", "pred", "y_pred", "labels_pred", "cluster", "y_pred_final"):
            if key in payload:
                values = np.asarray(payload[key]).reshape(-1)
                break
        if values is None:
            raise KeyError(f"No prediction vector found in {source}.")
    else:
        raise ValueError(f"Unsupported prediction file: {source}")
    np.save(target, values)
    return values


def run_official_wrapper(method, display_name):
    args = parse_args(display_name)
    data_path = Path(args.data).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "wrapper_status.json"
    if not data_path.is_file():
        raise FileNotFoundError(f"Unified data bundle not found: {data_path}")

    training_payload, expected_length = load_and_validate_bundle(data_path)
    training_data_path = output_dir / "training_data_no_labels.npz"
    np.savez_compressed(training_data_path, **training_payload)

    env_name = f"{method.upper().replace('-', '_')}_OFFICIAL_COMMAND"
    command_template = args.official_command or os.environ.get(env_name)
    if not command_template:
        message = (
            f"Official implementation missing for {display_name}. "
            f"Place the official repository locally and set {env_name} to its training/inference "
            "command template. Random-label fallback is disabled."
        )
        write_json(
            status_path,
            {
                "method": method,
                "status": "failed",
                "error": message,
                "source_data": data_path.as_posix(),
                "training_data": training_data_path.as_posix(),
                "labels_passed_to_training": False,
            },
        )
        raise RuntimeError(message)

    values = {
        "data": training_data_path.as_posix(),
        "output": output_dir.as_posix(),
        "dataset": args.dataset,
        "missing_rate": args.missing_rate,
        "seed": args.seed,
        "device": args.device,
    }
    command = shlex.split(command_template.format(**values), posix=True)
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent / display_name,
        text=True,
        capture_output=True,
        check=False,
    )
    (output_dir / "official_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "official_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        write_json(
            status_path,
            {
                "method": method,
                "status": "failed",
                "returncode": completed.returncode,
                "command": command,
                "labels_passed_to_training": False,
            },
        )
        raise RuntimeError(
            f"Official {display_name} command failed with return code {completed.returncode}. "
            f"See {output_dir.as_posix()}."
        )

    prediction_source = find_prediction(output_dir)
    if prediction_source is None:
        raise FileNotFoundError(
            f"Official {display_name} completed but produced no prediction file under {output_dir}."
        )
    normalized_path = output_dir / "pred_labels.npy"
    predictions = normalize_prediction(prediction_source, normalized_path)
    if predictions.size != expected_length:
        raise ValueError(
            f"Official {display_name} prediction length {predictions.size} "
            f"does not match {expected_length} samples."
        )
    write_json(
        status_path,
        {
            "method": method,
            "status": "complete",
            "returncode": completed.returncode,
            "command": command,
            "prediction_source": prediction_source.as_posix(),
            "prediction_file": normalized_path.as_posix(),
            "prediction_length": int(predictions.size),
            "labels_passed_to_training": False,
        },
    )
    if prediction_source != normalized_path and prediction_source.is_file():
        shutil.copy2(prediction_source, output_dir / f"original_{prediction_source.name}")
