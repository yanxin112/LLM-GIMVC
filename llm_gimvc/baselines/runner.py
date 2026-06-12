import shutil
from pathlib import Path

import numpy as np

from block1_utils import get_block1_job_dir, normalize_missing_rate, read_json, write_json
from configure import get_default_config
from pipeline_utils import run_command

from .base import BaselineJob
from .data_export import export_block1_data_bundle
from .registry import get_baseline_adapter


LABEL_USAGE_WARNING = "labels are included in the exported bundle only for post-hoc evaluation"


def _format_metrics(metrics):
    return (
        f"  NMI: {metrics['NMI']:.6f}\n"
        f"  ARI: {metrics['ARI']:.6f}\n"
        f"  ACC: {metrics['ACC']:.6f}\n"
        f"  Purity: {metrics['Purity']:.6f}"
    )


def _command_record(result, method, cwd):
    return {
        "stage": "stage5b_external_baseline",
        "method": method,
        "cmd": result["cmd"],
        "cwd": Path(cwd).as_posix(),
        "returncode": int(result["returncode"]),
        "ok": bool(result["ok"]),
        "stdout_tail": result.get("stdout_tail", "")[-4000:],
        "stderr_tail": result.get("stderr_tail", "")[-4000:],
    }


def _build_config(dataset, repo_root, data_root, raw_output_root, timeout_seconds):
    config = get_default_config(dataset)
    config["ExternalBaselines"]["repo_root"] = repo_root
    config["ExternalBaselines"]["data_root"] = data_root
    config["ExternalBaselines"]["output_root"] = raw_output_root
    config["ExternalBaselines"]["timeout_seconds"] = timeout_seconds
    return config


def _save_standard_outputs(
    job,
    adapter_result,
    command_result,
    repo_dir,
    entrypoint,
    export_format,
    reused,
):
    metrics_path = job.block1_job_dir / "metrics.json"
    prediction_path = job.block1_job_dir / "pred_labels.npy"
    standard_raw_dir = job.block1_job_dir / "raw"
    summary_path = job.block1_job_dir / "job_summary.json"
    command_log_path = job.block1_job_dir / "command_log.json"
    metrics = adapter_result["metrics"]
    job.block1_job_dir.mkdir(parents=True, exist_ok=True)
    predictions = adapter_result.get("pred_labels")
    if predictions is None:
        raise ValueError("External baseline adapter did not return prediction labels.")
    np.save(prediction_path, np.asarray(predictions).reshape(-1))
    if job.raw_output_dir.resolve() != standard_raw_dir.resolve():
        shutil.copytree(job.raw_output_dir, standard_raw_dir, dirs_exist_ok=True)
    metrics_obj = {
        "block": "block1",
        "stage": "stage5b_external_baseline",
        "method": job.method,
        "dataset": job.dataset,
        "missing_pattern": job.missing_pattern,
        "missing_rate": job.missing_rate,
        "missing_rate_fraction": job.missing_rate_fraction,
        "seed": job.seed,
        "metrics": metrics,
        "primary_metric": {
            "name": "NMI",
            "value": metrics["NMI"],
        },
        "source": {
            "data_path": job.data_path.as_posix(),
            "metadata_path": job.metadata_path.as_posix(),
            "raw_output_dir": job.raw_output_dir.as_posix(),
            "adapter_source": adapter_result["adapter_source"],
            "raw_metrics_path": adapter_result.get("raw_metrics_path"),
            "pred_labels_path": adapter_result.get("pred_labels_path"),
            "standard_pred_labels_path": prediction_path.as_posix(),
        },
        "config": {
            "repo_root": job.config["ExternalBaselines"]["repo_root"],
            "repo_dir": Path(repo_dir).as_posix(),
            "entrypoint": entrypoint,
            "export_format": export_format,
        },
        "debug_only": False,
        "reused": bool(reused),
    }
    job_summary = {
        "block": "block1",
        "stage": "stage5b_external_baseline",
        "status": "complete",
        "method": job.method,
        "dataset": job.dataset,
        "missing_pattern": job.missing_pattern,
        "missing_rate": job.missing_rate,
        "seed": job.seed,
        "job_dir": job.block1_job_dir.as_posix(),
        "metrics_path": metrics_path.as_posix(),
        "command_log_path": command_log_path.as_posix(),
        "data_path": job.data_path.as_posix(),
        "raw_output_dir": job.raw_output_dir.as_posix(),
        "standard_raw_dir": standard_raw_dir.as_posix(),
        "pred_labels_path": prediction_path.as_posix(),
        "external_command": command_result["cmd"],
        "external_returncode": int(command_result["returncode"]),
        "metrics": metrics,
        "adapter_source": adapter_result["adapter_source"],
        "reused": bool(reused),
        "debug_only": False,
        "label_usage_warning": LABEL_USAGE_WARNING,
    }
    command_log = _command_record(command_result, job.method, repo_dir)
    write_json(metrics_path, metrics_obj)
    write_json(summary_path, job_summary)
    write_json(command_log_path, command_log)
    return {
        "metrics": metrics,
        "metrics_path": metrics_path,
        "summary_path": summary_path,
        "command_log_path": command_log_path,
        "prediction_path": prediction_path,
    }


def run_external_baseline_job(
    method,
    dataset,
    missing_rate,
    missing_pattern,
    seed,
    output_dir="results/block1",
    data_root="results/baseline_data",
    raw_output_root="results/external_baselines",
    repo_root="external_baselines",
    export_format="npz",
    device="cuda:0",
    force=False,
    reuse_existing=True,
    dry_run=False,
    fail_fast=True,
    timeout_seconds=None,
):
    method = method.lower()
    rate = normalize_missing_rate(missing_rate)
    block1_job_dir = get_block1_job_dir(output_dir, dataset, missing_pattern, rate["percent"], method, seed)
    metrics_path = block1_job_dir / "metrics.json"
    if metrics_path.exists() and reuse_existing and not force:
        return {
            "reused": True,
            "metrics": read_json(metrics_path)["metrics"],
            "metrics_path": metrics_path,
            "job_dir": block1_job_dir,
        }

    data_result = export_block1_data_bundle(
        dataset=dataset,
        missing_rate=rate["percent"],
        missing_pattern=missing_pattern,
        seed=seed,
        output_root=data_root,
        export_format=export_format,
        device="cpu",
        force=force,
    )
    raw_output_dir = block1_job_dir / "raw"
    config = _build_config(dataset, repo_root, data_root, raw_output_root, timeout_seconds)
    adapter = get_baseline_adapter(method)
    job = BaselineJob(
        method=method,
        dataset=dataset,
        missing_rate=rate["percent"],
        missing_rate_fraction=rate["fraction"],
        missing_pattern=missing_pattern,
        seed=int(seed),
        data_path=data_result["data_path"],
        metadata_path=data_result["metadata_path"],
        raw_output_dir=raw_output_dir,
        block1_job_dir=block1_job_dir,
        device=device,
        config=config,
    )
    command = adapter.build_command(job)
    repo_dir = adapter._repo_dir(job)
    entrypoint = adapter._method_cfg(job)["entrypoint"]

    if dry_run:
        return {
            "dry_run": True,
            "cmd": " ".join(command),
            "data_path": job.data_path,
            "raw_output_dir": raw_output_dir,
            "job_dir": block1_job_dir,
        }

    adapter.validate_environment(job)
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    result = run_command(
        command,
        cwd=repo_dir,
        fail_fast=False,
        timeout_seconds=timeout_seconds,
    )
    if not result["ok"]:
        block1_job_dir.mkdir(parents=True, exist_ok=True)
        failed_log_path = block1_job_dir / "failed_command_log.json"
        command_record = _command_record(result, method, repo_dir)
        write_json(failed_log_path, command_record)
        write_json(command_log_path := block1_job_dir / "command_log.json", command_record)
        write_json(
            block1_job_dir / "job_summary.json",
            {
                "block": "block1",
                "stage": "stage5b_external_baseline",
                "status": "failed",
                "method": method,
                "dataset": dataset,
                "missing_pattern": missing_pattern,
                "missing_rate": rate["percent"],
                "seed": int(seed),
                "job_dir": block1_job_dir.as_posix(),
                "command_log_path": command_log_path.as_posix(),
                "data_path": job.data_path.as_posix(),
                "raw_output_dir": raw_output_dir.as_posix(),
                "error": command_record.get("stderr_tail") or command_record.get("error"),
                "debug_only": False,
            },
        )
        raise RuntimeError(
            f"External baseline {method} failed with returncode {result['returncode']}. "
            f"See {failed_log_path.as_posix()}."
        )

    try:
        adapter_result = adapter.adapt_results(job)
    except Exception as exc:
        block1_job_dir.mkdir(parents=True, exist_ok=True)
        failed_log_path = block1_job_dir / "failed_command_log.json"
        failed_record = _command_record(result, method, repo_dir)
        failed_record["adapter_error"] = str(exc)
        write_json(failed_log_path, failed_record)
        raise

    saved = _save_standard_outputs(
        job=job,
        adapter_result=adapter_result,
        command_result=result,
        repo_dir=repo_dir,
        entrypoint=entrypoint,
        export_format=export_format,
        reused=False,
    )
    saved["job_dir"] = block1_job_dir
    saved["raw_output_dir"] = raw_output_dir
    return saved
