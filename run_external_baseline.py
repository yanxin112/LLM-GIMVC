import argparse

from llm_gimvc.baselines.runner import run_external_baseline_job


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
    parser = argparse.ArgumentParser(description="Run one external baseline and adapt it to Block 1 format.")
    parser.add_argument("--method", type=str, default="mica")
    parser.add_argument("--dataset", type=str, default="BDGP")
    parser.add_argument("--missing-rate", type=float, default=50)
    parser.add_argument("--missing-pattern", type=str, default="mcar")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results/block1")
    parser.add_argument("--data-root", type=str, default="results/baseline_data")
    parser.add_argument("--raw-output-root", type=str, default="results/external_baselines")
    parser.add_argument("--repo-root", type=str, default="external_baselines")
    parser.add_argument("--export-format", choices=["npz", "mat"], default="npz")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--reuse-existing", dest="reuse_existing", action="store_true")
    parser.add_argument("--no-reuse-existing", dest="reuse_existing", action="store_false")
    parser.set_defaults(reuse_existing=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", nargs="?", const=True, default=True, type=_bool_arg)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    return parser.parse_args()


def _print_metrics(metrics):
    print("final metrics:")
    print(f"  NMI: {metrics['NMI']:.6f}")
    print(f"  ARI: {metrics['ARI']:.6f}")
    print(f"  ACC: {metrics['ACC']:.6f}")
    print(f"  Purity: {metrics['Purity']:.6f}")


def main():
    args = _parse_args()
    print("=" * 60)
    print("External Baseline Runner - Stage 5B")
    print("=" * 60)
    print(f"method: {args.method}")
    print(f"dataset: {args.dataset}")
    print(f"missing pattern: {args.missing_pattern}")
    print(f"missing rate: {int(round(args.missing_rate)) if args.missing_rate > 1 else int(round(args.missing_rate * 100))}")
    print(f"missing rate fraction: {args.missing_rate / 100.0 if args.missing_rate > 1 else args.missing_rate}")
    print(f"seed: {args.seed}")
    print("exporting data...")
    result = run_external_baseline_job(
        method=args.method,
        dataset=args.dataset,
        missing_rate=args.missing_rate,
        missing_pattern=args.missing_pattern,
        seed=args.seed,
        output_dir=args.output_dir,
        data_root=args.data_root,
        raw_output_root=args.raw_output_root,
        repo_root=args.repo_root,
        export_format=args.export_format,
        device=args.device,
        force=args.force,
        reuse_existing=args.reuse_existing,
        dry_run=args.dry_run,
        fail_fast=args.fail_fast,
        timeout_seconds=args.timeout_seconds,
    )
    if result.get("dry_run"):
        print(f"data path: {result['data_path'].as_posix()}")
        print(f"raw output dir: {result['raw_output_dir'].as_posix()}")
        print(f"block1 job dir: {result['job_dir'].as_posix()}")
        print("validating environment: skipped for dry-run")
        print("running external baseline:")
        print(f"  {result['cmd']}")
        print("dry-run: no external command executed and no metrics written")
        return
    if result.get("reused"):
        print("status: reused")
        _print_metrics(result["metrics"])
        print(f"metrics.json: {result['metrics_path'].as_posix()}")
        return

    print("validating environment: complete")
    print("running external baseline: complete")
    print("adapting results: complete")
    _print_metrics(result["metrics"])
    print("saved:")
    print(f"  metrics.json: {result['metrics_path'].as_posix()}")
    print(f"  job_summary.json: {result['summary_path'].as_posix()}")
    print(f"  command_log.json: {result['command_log_path'].as_posix()}")


if __name__ == "__main__":
    main()
