import argparse
from pathlib import Path

from block1_utils import normalize_missing_rate, write_json
from llm_gimvc.baselines.adapters import adapt_external_result
from llm_gimvc.baselines.runner import LABEL_USAGE_WARNING


def _parse_args():
    parser = argparse.ArgumentParser(description="Adapt an existing external baseline raw output to Block 1 format.")
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--missing-rate", type=float, required=True)
    parser.add_argument("--missing-pattern", type=str, default="mcar")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--raw-output-dir", type=str, required=True)
    parser.add_argument("--block1-job-dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    rate = normalize_missing_rate(args.missing_rate)
    data_path = Path(args.data_path)
    raw_output_dir = Path(args.raw_output_dir)
    block1_job_dir = Path(args.block1_job_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data bundle not found: {data_path}")
    if not raw_output_dir.exists():
        raise FileNotFoundError(f"Raw output directory not found: {raw_output_dir}")

    adapter_result = adapt_external_result(args.method, data_path, raw_output_dir)
    metrics = adapter_result["metrics"]
    metrics_path = block1_job_dir / "metrics.json"
    summary_path = block1_job_dir / "job_summary.json"
    command_log_path = block1_job_dir / "command_log.json"
    metrics_obj = {
        "block": "block1",
        "stage": "stage5b_external_baseline",
        "method": args.method,
        "dataset": args.dataset,
        "missing_pattern": args.missing_pattern,
        "missing_rate": rate["percent"],
        "missing_rate_fraction": rate["fraction"],
        "seed": args.seed,
        "metrics": metrics,
        "primary_metric": {
            "name": "NMI",
            "value": metrics["NMI"],
        },
        "source": {
            "data_path": data_path.as_posix(),
            "metadata_path": (data_path.parent / "metadata.json").as_posix(),
            "raw_output_dir": raw_output_dir.as_posix(),
            "adapter_source": adapter_result["adapter_source"],
            "raw_metrics_path": adapter_result.get("raw_metrics_path"),
            "pred_labels_path": adapter_result.get("pred_labels_path"),
        },
        "config": {
            "repo_root": None,
            "repo_dir": None,
            "entrypoint": None,
            "export_format": data_path.suffix.lstrip("."),
        },
        "debug_only": False,
        "reused": False,
    }
    job_summary = {
        "block": "block1",
        "stage": "stage5b_external_baseline",
        "status": "complete",
        "method": args.method,
        "dataset": args.dataset,
        "missing_pattern": args.missing_pattern,
        "missing_rate": rate["percent"],
        "seed": args.seed,
        "job_dir": block1_job_dir.as_posix(),
        "metrics_path": metrics_path.as_posix(),
        "command_log_path": command_log_path.as_posix(),
        "data_path": data_path.as_posix(),
        "raw_output_dir": raw_output_dir.as_posix(),
        "external_command": None,
        "external_returncode": None,
        "metrics": metrics,
        "adapter_source": adapter_result["adapter_source"],
        "reused": False,
        "debug_only": False,
        "label_usage_warning": LABEL_USAGE_WARNING,
    }
    command_log = {
        "stage": "stage5b_external_baseline",
        "method": args.method,
        "cmd": None,
        "cwd": None,
        "returncode": None,
        "ok": True,
        "stdout_tail": "adapter-only conversion",
        "stderr_tail": "",
    }
    write_json(metrics_path, metrics_obj)
    write_json(summary_path, job_summary)
    write_json(command_log_path, command_log)
    print("final metrics:")
    print(f"  NMI: {metrics['NMI']:.6f}")
    print(f"  ARI: {metrics['ARI']:.6f}")
    print(f"  ACC: {metrics['ACC']:.6f}")
    print(f"  Purity: {metrics['Purity']:.6f}")
    print("saved:")
    print(f"  metrics.json: {metrics_path.as_posix()}")
    print(f"  job_summary.json: {summary_path.as_posix()}")


if __name__ == "__main__":
    main()
