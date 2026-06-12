import argparse
import sys
from datetime import datetime
from pathlib import Path

from block1_utils import get_block1_metrics_path, normalize_missing_rate, write_json
from pipeline_utils import run_command


def _bool_arg(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ["1", "true", "yes", "y"]:
        return True
    if value in ["0", "false", "no", "n"]:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Run a local Stage 5A Block 1 sweep.")
    parser.add_argument("--datasets", nargs="+", default=["BDGP"])
    parser.add_argument("--missing-rates", nargs="+", type=float, default=[50])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--methods", nargs="+", default=["llm_gimvc", "statistical_only"])
    parser.add_argument("--missing-pattern", type=str, default="mcar")
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
    parser.add_argument("--epochs", type=int, default=20)
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
    parser.add_argument("--external-repo-root", type=str, default="external_baselines")
    parser.add_argument("--baseline-data-root", type=str, default="results/baseline_data")
    parser.add_argument("--external-raw-output-root", type=str, default="results/external_baselines")
    parser.add_argument("--export-format", choices=["npz", "mat"], default="npz")
    parser.add_argument("--baseline-timeout-seconds", type=float, default=None)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--fail-fast", nargs="?", const=True, default=True, type=_bool_arg)
    parser.add_argument("--no-fail-fast", dest="fail_fast", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _build_jobs(args):
    jobs = []
    for dataset in args.datasets:
        for missing_rate in args.missing_rates:
            rate = normalize_missing_rate(missing_rate)
            for method in args.methods:
                for seed in args.seeds:
                    jobs.append(
                        {
                            "dataset": dataset,
                            "missing_rate": rate["percent"],
                            "missing_rate_fraction": rate["fraction"],
                            "method": method,
                            "seed": seed,
                        }
                    )
    if args.max_jobs is not None:
        jobs = jobs[: max(args.max_jobs, 0)]
    return jobs


def _job_command(args, job):
    cmd = [
        sys.executable,
        "-m",
        "llm_gimvc.experiments.block1",
        "--dataset",
        job["dataset"],
        "--missing-rate",
        str(job["missing_rate"]),
        "--missing-pattern",
        args.missing_pattern,
        "--method",
        job["method"],
        "--seed",
        str(job["seed"]),
        "--metrics",
        *args.metrics,
        "--output-dir",
        args.output_dir,
        "--device",
        args.device,
        "--provider",
        args.provider,
        "--embedding-provider",
        args.embedding_provider,
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
        "--gate-mode",
        args.gate_mode,
        "--abstention-threshold",
        str(args.abstention_threshold),
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
        "--external-repo-root",
        args.external_repo_root,
        "--baseline-data-root",
        args.baseline_data_root,
        "--external-raw-output-root",
        args.external_raw_output_root,
        "--export-format",
        args.export_format,
    ])
    if args.baseline_timeout_seconds is not None:
        cmd.extend(["--baseline-timeout-seconds", str(args.baseline_timeout_seconds)])
    if not args.reuse_existing:
        cmd.append("--no-reuse-existing")
    if args.force:
        cmd.append("--force")
    if args.allow_unsafe_stage2a:
        cmd.append("--allow-unsafe-stage2a")
    if args.allow_debug_fusion:
        cmd.append("--allow-debug-fusion")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def main():
    args = _parse_args()
    jobs = _build_jobs(args)
    print("=" * 60)
    print("Block 1 Sweep - Stage 5A")
    print("=" * 60)
    print(f"datasets: {', '.join(args.datasets)}")
    print(f"missing rates: {', '.join(str(normalize_missing_rate(x)['percent']) for x in args.missing_rates)}")
    print(f"methods: {', '.join(args.methods)}")
    print(f"seeds: {', '.join(str(x) for x in args.seeds)}")
    print(f"num jobs: {len(jobs)}")

    summary_jobs = []
    failed = []
    success = 0
    skipped = 0
    for index, job in enumerate(jobs, start=1):
        metrics_path = get_block1_metrics_path(
            args.output_dir,
            job["dataset"],
            args.missing_pattern,
            job["missing_rate"],
            job["method"],
            job["seed"],
        )
        print(
            f"[{index}/{len(jobs)}] {job['dataset']} missing_{job['missing_rate']} "
            f"{job['method']} seed_{job['seed']}"
        )
        cmd = _job_command(args, job)
        if args.dry_run:
            result = run_command(cmd, cwd=Path.cwd(), fail_fast=False)
            if result["ok"]:
                print("status: dry-run")
                if result.get("stdout_tail"):
                    print(result["stdout_tail"])
                skipped += 1
            else:
                print("status: failed")
                failed.append({**job, "returncode": result["returncode"], "stderr_tail": result["stderr_tail"]})
                if args.fail_fast:
                    summary_jobs.append({**job, "returncode": result["returncode"], "ok": False, "metrics_path": metrics_path.as_posix()})
                    break
        else:
            result = run_command(cmd, cwd=Path.cwd(), fail_fast=False)
            if result["ok"]:
                print("status: complete")
                success += 1
            else:
                print("status: failed")
                failed.append({**job, "returncode": result["returncode"], "stderr_tail": result["stderr_tail"]})
                if args.fail_fast:
                    summary_jobs.append({**job, "returncode": result["returncode"], "ok": False, "metrics_path": metrics_path.as_posix()})
                    break
        summary_jobs.append(
            {
                **job,
                "returncode": int(result["returncode"]),
                "ok": bool(result["ok"]),
                "metrics_path": metrics_path.as_posix(),
            }
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.output_dir) / "sweep_runs" / timestamp
    summary_path = sweep_dir / "sweep_summary.json"
    summary = {
        "stage": "stage5a_block1_sweep",
        "datasets": args.datasets,
        "missing_rates": [normalize_missing_rate(x)["percent"] for x in args.missing_rates],
        "methods": args.methods,
        "seeds": args.seeds,
        "num_jobs": len(jobs),
        "num_success": success,
        "num_failed": len(failed),
        "num_skipped": skipped,
        "jobs": summary_jobs,
        "failed_jobs": failed,
        "dry_run": bool(args.dry_run),
    }
    write_json(summary_path, summary)
    print(f"saved sweep summary: {summary_path.as_posix()}")

    if failed and args.fail_fast:
        raise RuntimeError(f"Block 1 sweep stopped after failed job: {failed[0]}")
    if failed:
        print(f"failed jobs: {len(failed)}")


if __name__ == "__main__":
    main()
