import argparse
import subprocess
import sys
from pathlib import Path

from block1_utils import (
    extract_metric_group,
    extract_metrics_from_method_metrics,
    get_block1_job_dir,
    normalize_missing_rate,
    read_json,
    write_json,
)
from pipeline_utils import get_method_dir, run_command


STAGE4B_METHODS = {
    "llm_gimvc": "fusion",
    "statistical_only": "statistical_only",
}

EXTERNAL_BASELINE_METHODS = {
    "mica",
    "jga_imvc",
    "freecsl",
}


def _resolve_block1_method(method):
    method = str(method).lower()
    if method in STAGE4B_METHODS:
        return {
            "input_source": STAGE4B_METHODS[method],
            "requires_stage4b": True,
            "requires_external_baseline": False,
        }
    if method in EXTERNAL_BASELINE_METHODS:
        return {
            "input_source": None,
            "requires_stage4b": False,
            "requires_external_baseline": True,
        }
    raise ValueError(f"Unknown Block 1 method: {method}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Run one Block 1 job for Stage 5B-Fix.")
    parser.add_argument("--dataset", type=str, default="BDGP")
    parser.add_argument("--missing-rate", type=float, default=50)
    parser.add_argument("--missing-pattern", type=str, default="mcar")
    parser.add_argument("--method", type=str, default="llm_gimvc")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics", nargs="+", default=["nmi", "ari", "acc", "purity"])
    parser.add_argument("--output-dir", type=str, default="results/block1")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--provider", type=str, default="mock")
    parser.add_argument("--embedding-provider", type=str, default="mock")
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--embedding-model", type=str, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--max-llm-queries", type=int, default=None)
    parser.add_argument("--qwen-base-url", type=str, default=None)
    parser.add_argument("--gate-mode", choices=["heuristic", "mlp"], default="heuristic")
    parser.add_argument("--abstention-threshold", type=float, default=0.3)
    parser.add_argument("--head-type", choices=["dcp", "completer"], default="dcp")
    parser.add_argument("--representation", choices=["mean", "sum", "concat"], default="mean")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--lambda-contrastive", type=float, default=1.0)
    parser.add_argument("--lambda-kl", type=float, default=0.1)
    parser.add_argument("--lambda-balance", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--reuse-existing", dest="reuse_existing", action="store_true")
    parser.add_argument("--no-reuse-existing", dest="reuse_existing", action="store_false")
    parser.set_defaults(reuse_existing=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-unsafe-stage2a", action="store_true")
    parser.add_argument("--allow-debug-fusion", action="store_true")
    parser.add_argument("--llm-missing-fallback", choices=["zero", "stat"], default="zero")
    parser.add_argument("--external-repo-root", type=str, default="external_baselines")
    parser.add_argument("--baseline-data-root", type=str, default="results/baseline_data")
    parser.add_argument("--external-raw-output-root", type=str, default="results/external_baselines")
    parser.add_argument("--export-format", choices=["npz", "mat"], default="npz")
    parser.add_argument("--baseline-timeout-seconds", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _command_to_text(cmd):
    return subprocess.list2cmdline([str(part) for part in cmd])


def _stage4b_command(args, input_source, rate):
    if input_source is None:
        raise ValueError("Stage 5A-Fix cannot build a Stage 4B command with input_source=None.")
    cmd = [
        sys.executable,
        "run_llm_gimvc_method.py",
        "--dataset",
        args.dataset,
        "--missing-rate",
        str(rate["fraction"]),
        "--seed",
        str(args.seed),
        "--provider",
        args.provider,
        "--embedding-provider",
        args.embedding_provider,
        "--gate-mode",
        args.gate_mode,
    ]
    if args.llm_model:
        cmd.extend(["--llm-model", args.llm_model])
    if args.embedding_model:
        cmd.extend(["--embedding-model", args.embedding_model])
    if args.embedding_dim is not None:
        cmd.extend(["--embedding-dim", str(args.embedding_dim)])
    if args.max_llm_queries is not None:
        cmd.extend(["--max-llm-queries", str(args.max_llm_queries)])
    if args.qwen_base_url:
        cmd.extend(["--qwen-base-url", args.qwen_base_url])
    cmd.extend([
        "--abstention-threshold",
        str(args.abstention_threshold),
        "--input-source",
        input_source,
        "--head-type",
        args.head_type,
        "--representation",
        args.representation,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--temperature",
        str(args.temperature),
        "--lambda-contrastive",
        str(args.lambda_contrastive),
        "--lambda-kl",
        str(args.lambda_kl),
        "--lambda-balance",
        str(args.lambda_balance),
        "--warmup-epochs",
        str(args.warmup_epochs),
        "--llm-missing-fallback",
        args.llm_missing_fallback,
        "--device",
        args.device,
    ])
    if args.allow_unsafe_stage2a:
        cmd.append("--allow-unsafe-stage2a")
    if args.allow_debug_fusion:
        cmd.append("--allow-debug-fusion")
    if args.force:
        cmd.append("--force-stage4a")
    return cmd


def _external_baseline_command(args, method, rate):
    cmd = [
        sys.executable,
        "run_external_baseline.py",
        "--method",
        method,
        "--dataset",
        args.dataset,
        "--missing-rate",
        str(rate["percent"]),
        "--missing-pattern",
        args.missing_pattern,
        "--seed",
        str(args.seed),
        "--output-dir",
        args.output_dir,
        "--data-root",
        args.baseline_data_root,
        "--raw-output-root",
        args.external_raw_output_root,
        "--repo-root",
        args.external_repo_root,
        "--export-format",
        args.export_format,
        "--device",
        args.device,
    ]
    if args.force:
        cmd.append("--force")
    if not args.reuse_existing:
        cmd.append("--no-reuse-existing")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.baseline_timeout_seconds is not None:
        cmd.extend(["--timeout-seconds", str(args.baseline_timeout_seconds)])
    return cmd


def _print_header(args, rate, method_args, job_dir):
    print("=" * 60)
    print("Block 1 Single Job - Stage 5B-Fix")
    print("=" * 60)
    print(f"dataset: {args.dataset}")
    print(f"missing pattern: {args.missing_pattern}")
    print(f"missing rate: {rate['percent']}")
    print(f"missing rate fraction: {rate['fraction']}")
    print(f"method: {args.method}")
    print(f"seed: {args.seed}")
    print(f"metrics: {' '.join(args.metrics)}")
    print(f"output dir: {args.output_dir}")
    print("resolved method:")
    if method_args["requires_external_baseline"]:
        print("  External baseline runner")
    else:
        print(f"  Stage 4B input_source: {method_args['input_source']}")
    print(f"expected output path: {(job_dir / 'metrics.json').as_posix()}")


def _format_metric_lines(metrics):
    return (
        f"  NMI: {metrics['NMI']:.6f}\n"
        f"  ARI: {metrics['ARI']:.6f}\n"
        f"  ACC: {metrics['ACC']:.6f}\n"
        f"  Purity: {metrics['Purity']:.6f}"
    )


def _write_outputs(args, rate, job_dir, command_result, method_metrics, method_summary, stage4b_dir, reused):
    metrics_path = job_dir / "metrics.json"
    job_summary_path = job_dir / "job_summary.json"
    command_log_path = job_dir / "command_log.json"
    stage4b_metrics_path = stage4b_dir / "method_metrics.json"
    stage4b_summary_path = stage4b_dir / "method_summary.json"

    primary_metrics = extract_metrics_from_method_metrics(method_metrics)
    head_assignment = extract_metric_group(method_metrics, "head_assignment")
    kmeans_metrics = extract_metric_group(method_metrics, "kmeans_on_head_representation")
    debug_only = bool(method_summary.get("debug_only", False))

    metrics_obj = {
        "block": "block1",
        "stage": "stage5a_fix_block1_single_job",
        "method": args.method,
        "dataset": args.dataset,
        "missing_pattern": args.missing_pattern,
        "missing_rate": rate["percent"],
        "missing_rate_fraction": rate["fraction"],
        "seed": args.seed,
        "metrics": primary_metrics,
        "primary_metric": {
            "name": "NMI",
            "value": primary_metrics["NMI"],
        },
        "diagnostics": {
            "head_assignment": head_assignment,
            "kmeans_on_head_representation": kmeans_metrics,
        },
        "source": {
            "stage4b_method_dir": stage4b_dir.as_posix(),
            "stage4b_method_metrics_path": stage4b_metrics_path.as_posix(),
            "stage4b_method_summary_path": stage4b_summary_path.as_posix(),
        },
        "config": {
            "provider": args.provider,
            "embedding_provider": args.embedding_provider,
            "llm_model": args.llm_model,
            "embedding_model": args.embedding_model,
            "embedding_dim": args.embedding_dim,
            "max_llm_queries": args.max_llm_queries,
            "qwen_base_url": args.qwen_base_url,
            "gate_mode": args.gate_mode,
            "abstention_threshold": args.abstention_threshold,
            "head_type": args.head_type,
            "representation": args.representation,
            "epochs": args.epochs,
        },
        "debug_only": debug_only,
        "reused": bool(reused),
    }
    job_summary = {
        "block": "block1",
        "stage": "stage5a_fix_block1_single_job",
        "status": "complete",
        "dataset": args.dataset,
        "missing_pattern": args.missing_pattern,
        "missing_rate": rate["percent"],
        "seed": args.seed,
        "method": args.method,
        "job_dir": job_dir.as_posix(),
        "metrics_path": metrics_path.as_posix(),
        "command_log_path": command_log_path.as_posix(),
        "stage4b_command": command_result["cmd"],
        "stage4b_returncode": command_result["returncode"],
        "metrics": primary_metrics,
        "reused": bool(reused),
        "debug_only": debug_only,
    }
    command_log = {
        "stage": "stage5a_fix_block1_single_job",
        "method": args.method,
        "cmd": command_result["cmd"],
        "returncode": command_result["returncode"],
        "ok": bool(command_result["ok"]),
        "stdout_tail": command_result.get("stdout_tail", "")[-4000:],
        "stderr_tail": command_result.get("stderr_tail", "")[-4000:],
    }

    write_json(metrics_path, metrics_obj)
    write_json(job_summary_path, job_summary)
    write_json(command_log_path, command_log)
    return metrics_obj, metrics_path, job_summary_path, command_log_path


def main():
    args = _parse_args()
    method = args.method.lower()
    method_args = _resolve_block1_method(method)
    rate = normalize_missing_rate(args.missing_rate)
    job_dir = get_block1_job_dir(args.output_dir, args.dataset, args.missing_pattern, rate["percent"], method, args.seed)
    metrics_path = job_dir / "metrics.json"
    _print_header(args, rate, method_args, job_dir)

    if args.dry_run:
        if method_args["requires_external_baseline"]:
            cmd = _external_baseline_command(args, method, rate)
        else:
            cmd = _stage4b_command(args, method_args["input_source"], rate)
        print("running:")
        print(f"  {_command_to_text(cmd)}")
        print("dry-run: no command executed and no metrics written")
        return

    if metrics_path.exists() and args.reuse_existing and not args.force:
        metrics_obj = read_json(metrics_path)
        print("reused=true")
        print("final metrics:")
        print(_format_metric_lines(metrics_obj["metrics"]))
        write_json(
            job_dir / "job_summary.json",
            {
                "block": "block1",
                "stage": "stage5a_fix_block1_single_job",
                "status": "complete",
                "dataset": args.dataset,
                "missing_pattern": args.missing_pattern,
                "missing_rate": rate["percent"],
                "seed": args.seed,
                "method": method,
                "job_dir": job_dir.as_posix(),
                "metrics_path": metrics_path.as_posix(),
                "command_log_path": (job_dir / "command_log.json").as_posix(),
                "stage4b_command": "",
                "stage4b_returncode": 0,
                "metrics": metrics_obj["metrics"],
                "reused": True,
                "debug_only": bool(metrics_obj.get("debug_only", False)),
            },
        )
        write_json(
            job_dir / "command_log.json",
            {
                "stage": "stage5a_fix_block1_single_job",
                "method": method,
                "cmd": "",
                "returncode": 0,
                "ok": True,
                "stdout_tail": "reused existing Block 1 metrics.json",
                "stderr_tail": "",
            },
        )
        print("saved:")
        print(f"  metrics.json: {metrics_path.as_posix()}")
        print(f"  job_summary.json: {(job_dir / 'job_summary.json').as_posix()}")
        print(f"  command_log.json: {(job_dir / 'command_log.json').as_posix()}")
        return

    if method_args["requires_external_baseline"]:
        cmd = _external_baseline_command(args, method, rate)
        print("running:")
        print(f"  {_command_to_text(cmd)}")
        result = run_command(
            cmd,
            cwd=Path.cwd(),
            fail_fast=False,
            timeout_seconds=args.baseline_timeout_seconds,
        )
        if not result["ok"]:
            result["stage"] = "stage5b_external_baseline_bridge"
            job_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                job_dir / "command_log.json",
                {"stage": "stage5b_external_baseline_bridge", "method": method, **result},
            )
            raise RuntimeError(
                f"External baseline command failed for {method}. "
                f"returncode={result['returncode']}. stderr_tail:\n{result.get('stderr_tail', '')}"
            )
        metrics_obj = read_json(metrics_path)
        if metrics_obj is None:
            raise FileNotFoundError(f"External baseline did not produce metrics.json: {metrics_path.as_posix()}")
        print("final metrics:")
        print(_format_metric_lines(metrics_obj["metrics"]))
        print("saved:")
        print(f"  metrics.json: {metrics_path.as_posix()}")
        print(f"  job_summary.json: {(job_dir / 'job_summary.json').as_posix()}")
        print(f"  command_log.json: {(job_dir / 'command_log.json').as_posix()}")
        return

    cmd = _stage4b_command(args, method_args["input_source"], rate)
    print("running:")
    print(f"  {_command_to_text(cmd)}")
    result = run_command(cmd, cwd=Path.cwd(), fail_fast=False)
    if not result["ok"]:
        result["stage"] = "stage5a_fix_block1_single_job"
        job_dir.mkdir(parents=True, exist_ok=True)
        write_json(job_dir / "command_log.json", {"stage": "stage5a_fix_block1_single_job", "method": method, **result})
        raise RuntimeError(f"Stage 4B command failed. See command_log.json in {job_dir.as_posix()}")

    stage4b_dir = get_method_dir(
        args.dataset,
        rate["fraction"],
        args.seed,
        args.gate_mode,
        method_args["input_source"],
        args.head_type,
        args.representation,
    )
    method_metrics = read_json(stage4b_dir / "method_metrics.json")
    method_summary = read_json(stage4b_dir / "method_summary.json")
    if method_metrics is None:
        raise RuntimeError(f"Stage 4B method_metrics.json not found: {stage4b_dir / 'method_metrics.json'}")
    if method_summary is None:
        raise RuntimeError(f"Stage 4B method_summary.json not found: {stage4b_dir / 'method_summary.json'}")

    metrics_obj, metrics_path, job_summary_path, command_log_path = _write_outputs(
        args,
        rate,
        job_dir,
        result,
        method_metrics,
        method_summary,
        stage4b_dir,
        reused=False,
    )
    print("final metrics:")
    print(_format_metric_lines(metrics_obj["metrics"]))
    print("saved:")
    print(f"  metrics.json: {metrics_path.as_posix()}")
    print(f"  job_summary.json: {job_summary_path.as_posix()}")
    print(f"  command_log.json: {command_log_path.as_posix()}")


if __name__ == "__main__":
    main()
