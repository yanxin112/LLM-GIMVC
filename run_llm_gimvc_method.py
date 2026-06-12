import argparse
from pathlib import Path

from configure import get_default_config
from pipeline_utils import (
    clustering_head_complete,
    fusion_complete,
    get_clustering_head_dir,
    get_fusion_dir,
    get_method_dir,
    get_missing_rate_dir,
    get_stage1_dir,
    get_stage2a_dir,
    make_json_safe,
    python_cmd,
    read_json,
    run_command,
    stage1_complete,
    stage2a_complete,
    write_json,
)


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the full LLM-GIMVC method pipeline.")
    parser.add_argument("--dataset", type=str, default="BDGP")
    parser.add_argument("--missing-rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--provider", type=str, default="mock")
    parser.add_argument("--embedding-provider", type=str, default="mock")
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--embedding-model", type=str, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--max-llm-queries", type=int, default=None)
    parser.add_argument("--qwen-base-url", type=str, default=None)
    parser.add_argument("--dry-run-api", action="store_true")
    parser.add_argument("--gate-mode", choices=["heuristic", "mlp"], default="heuristic")
    parser.add_argument("--abstention-threshold", type=float, default=0.3)
    parser.add_argument(
        "--input-source",
        choices=["observed_only", "statistical_only", "llm_only", "fusion"],
        default="fusion",
    )
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
    parser.add_argument("--force-stage1", action="store_true")
    parser.add_argument("--force-stage2a", action="store_true")
    parser.add_argument("--force-stage2b", action="store_true")
    parser.add_argument("--force-stage4a", action="store_true")
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-stage2a", action="store_true")
    parser.add_argument("--skip-stage2b", action="store_true")
    parser.add_argument("--skip-stage4a", action="store_true")
    parser.add_argument("--allow-unsafe-stage2a", action="store_true")
    parser.add_argument("--allow-debug-fusion", action="store_true")
    parser.add_argument("--llm-missing-fallback", choices=["zero", "stat"], default="zero")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def _format_metrics(metrics):
    return (
        f"NMI: {metrics['NMI']:.6f}\n"
        f"  ARI: {metrics['ARI']:.6f}\n"
        f"  ACC: {metrics['ACC']:.6f}\n"
        f"  Purity: {metrics['Purity']:.6f}"
    )


def _print_header(args):
    print("=" * 60)
    print("LLM-GIMVC Method Runner - Stage 4B")
    print("=" * 60)
    print(f"dataset: {args.dataset}")
    print(f"missing rate: {args.missing_rate}")
    print(f"seed: {args.seed}")
    print(f"device: {args.device}")
    print(f"provider: {args.provider}")
    print(f"embedding provider: {args.embedding_provider}")
    print(f"gate mode: {args.gate_mode}")
    print(f"input source: {args.input_source}")
    print(f"head type: {args.head_type}")
    print(f"representation: {args.representation}")


def _ensure_complete(check, error_message):
    if not check["complete"]:
        raise RuntimeError(f"{error_message} Missing: {check['missing']} in {check['dir']}")


def _record_stage(enabled, reused, check, **extra):
    record = {
        "enabled": bool(enabled),
        "reused": bool(reused),
        "complete": bool(check["complete"]),
        "dir": check["dir"],
    }
    record.update(extra)
    return record


def _run_stage(stage_name, cmd, command_log, cwd, fail_fast):
    result = run_command(cmd, cwd=cwd, fail_fast=fail_fast)
    result["stage"] = stage_name
    command_log.append(result)
    return result


def _stage1(args, stage1_dir, command_log, cwd, fail_fast):
    print("[1/4] Stage 1 statistical path")
    check = stage1_complete(stage1_dir)
    reused = False
    if args.skip_stage1:
        _ensure_complete(check, "Stage 1 is skipped but required artifacts are missing.")
        print("status: complete")
        return _record_stage(False, False, check, skipped=True)
    if check["complete"] and args.reuse_existing and not args.force_stage1:
        reused = True
        print("status: reused")
    else:
        print("status: running")
        cmd = python_cmd(
            "run_statistical_path.py",
            "--dataset",
            args.dataset,
            "--missing-rate",
            args.missing_rate,
            "--seed",
            args.seed,
            "--device",
            args.device,
        )
        _run_stage("stage1_statistical_path", cmd, command_log, cwd, fail_fast)
        check = stage1_complete(stage1_dir)
        _ensure_complete(check, "Stage 1 did not produce required artifacts.")
        print("status: complete")
    return _record_stage(True, reused, check)


def _stage2a(args, stage2a_dir, command_log, cwd, fail_fast):
    print("[2/4] Stage 2A LLM semantic path")
    check = stage2a_complete(stage2a_dir)
    existing_summary = read_json(stage2a_dir / "run_summary.json") or {}
    existing_embedding_dim = existing_summary.get("embedding_dim")
    summary_matches_args = (
        existing_summary.get("provider") == args.provider
        and existing_summary.get("embedding_provider") == args.embedding_provider
        and (args.llm_model is None or existing_summary.get("llm_model") == args.llm_model)
        and (args.embedding_model is None or existing_summary.get("embedding_model") == args.embedding_model)
        and (
            args.embedding_dim is None
            or (existing_embedding_dim is not None and int(existing_embedding_dim) == int(args.embedding_dim))
        )
        and (
            args.max_llm_queries is None
            or existing_summary.get("max_llm_queries") == args.max_llm_queries
        )
    )
    reused = False
    if args.skip_stage2a:
        _ensure_complete(check, "Stage 2A is skipped but required artifacts are missing.")
        print("status: complete")
    elif check["complete"] and args.reuse_existing and not args.force_stage2a and summary_matches_args:
        reused = True
        print("status: reused")
    else:
        print("status: running")
        cmd = python_cmd(
            "run_llm_semantic_path.py",
            "--dataset",
            args.dataset,
            "--missing-rate",
            args.missing_rate,
            "--seed",
            args.seed,
            "--provider",
            args.provider,
            "--embedding-provider",
            args.embedding_provider,
            "--device",
            args.device,
        )
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
        if args.dry_run_api:
            cmd.append("--dry-run-api")
        _run_stage("stage2a_llm_semantic_path", cmd, command_log, cwd, fail_fast)
        if args.dry_run_api:
            print("status: dry-run-api")
            return _record_stage(True, False, check, dry_run_api=True, is_partial=True, safe_for_fusion=False)
        check = stage2a_complete(stage2a_dir)
        _ensure_complete(check, "Stage 2A did not produce required artifacts.")
        print("status: complete")

    summary = read_json(stage2a_dir / "run_summary.json") or {}
    is_partial = bool(summary.get("is_partial", False))
    safe_for_fusion = bool(summary.get("safe_for_fusion", False))
    print(f"safe_for_fusion: {str(safe_for_fusion).lower()}")
    print(f"is_partial: {str(is_partial).lower()}")
    if (is_partial or not safe_for_fusion) and not args.allow_unsafe_stage2a:
        raise RuntimeError(
            "Stage 2A output is partial or unsafe. Re-run full Stage 2A or pass "
            "--allow-unsafe-stage2a for debugging only."
        )
    return _record_stage(
        not args.skip_stage2a,
        reused,
        check,
        skipped=bool(args.skip_stage2a),
        is_partial=is_partial,
        safe_for_fusion=safe_for_fusion,
    )


def _stage2b(args, fusion_dir, command_log, cwd, fail_fast):
    print("[3/4] Stage 2B fusion gate")
    check = fusion_complete(fusion_dir)
    reused = False
    if args.skip_stage2b:
        _ensure_complete(check, "Stage 2B is skipped but required artifacts are missing.")
        print("status: complete")
    elif check["complete"] and args.reuse_existing and not args.force_stage2b:
        reused = True
        print("status: reused")
    else:
        print("status: running")
        cmd = python_cmd(
            "run_fusion_gate.py",
            "--dataset",
            args.dataset,
            "--missing-rate",
            args.missing_rate,
            "--seed",
            args.seed,
            "--gate-mode",
            args.gate_mode,
            "--abstention-threshold",
            args.abstention_threshold,
            "--device",
            args.device,
        )
        if args.allow_unsafe_stage2a:
            cmd.append("--allow-partial-stage2a")
        _run_stage("stage2b_fusion_gate", cmd, command_log, cwd, fail_fast)
        check = fusion_complete(fusion_dir)
        _ensure_complete(check, "Stage 2B did not produce required artifacts.")
        print("status: complete")

    summary = read_json(fusion_dir / "fusion_summary.json") or {}
    debug_only = bool(summary.get("debug_only", False))
    used_partial = bool(summary.get("used_partial_stage2a", False))
    stage2a_safe = bool(summary.get("stage2a_safe_for_fusion", True))
    print(f"debug_only: {str(debug_only).lower()}")
    if (debug_only or used_partial or not stage2a_safe) and not args.allow_debug_fusion:
        raise RuntimeError(
            "Fusion output is debug-only. Re-run Fusion Gate with full safe Stage 2A output and heuristic mode, "
            "or pass --allow-debug-fusion for debugging only."
        )
    return _record_stage(
        not args.skip_stage2b,
        reused,
        check,
        skipped=bool(args.skip_stage2b),
        debug_only=debug_only,
        used_partial_stage2a=used_partial,
        stage2a_safe_for_fusion=stage2a_safe,
    )


def _stage4a(args, clustering_head_dir, command_log, cwd, fail_fast):
    print("[4/4] Stage 4A clustering head")
    check = clustering_head_complete(clustering_head_dir)
    reused = False
    if args.skip_stage4a:
        _ensure_complete(check, "Stage 4A is skipped but required artifacts are missing.")
        print("status: complete")
    elif check["complete"] and args.reuse_existing and not args.force_stage4a:
        reused = True
        print("status: reused")
    else:
        print("status: running")
        cmd = python_cmd(
            "run_clustering_head.py",
            "--dataset",
            args.dataset,
            "--missing-rate",
            args.missing_rate,
            "--seed",
            args.seed,
            "--gate-mode",
            args.gate_mode,
            "--input-source",
            args.input_source,
            "--head-type",
            args.head_type,
            "--representation",
            args.representation,
            "--epochs",
            args.epochs,
            "--batch-size",
            args.batch_size,
            "--lr",
            args.lr,
            "--weight-decay",
            args.weight_decay,
            "--temperature",
            args.temperature,
            "--lambda-contrastive",
            args.lambda_contrastive,
            "--lambda-kl",
            args.lambda_kl,
            "--lambda-balance",
            args.lambda_balance,
            "--warmup-epochs",
            args.warmup_epochs,
            "--llm-missing-fallback",
            args.llm_missing_fallback,
            "--device",
            args.device,
        )
        if args.allow_unsafe_stage2a:
            cmd.append("--allow-unsafe-stage2a")
        if args.allow_debug_fusion:
            cmd.append("--allow-debug-fusion")
        _run_stage("stage4a_clustering_head", cmd, command_log, cwd, fail_fast)
        check = clustering_head_complete(clustering_head_dir)
        _ensure_complete(check, "Stage 4A did not produce required artifacts.")
        print("status: complete")
    return _record_stage(not args.skip_stage4a, reused, check, skipped=bool(args.skip_stage4a))


def _build_config(args):
    config = get_default_config(args.dataset)
    config["Dataset"]["name"] = args.dataset
    config["Dataset"]["missing_rate"] = args.missing_rate
    config["training"]["seed"] = args.seed
    config["LLM"]["provider"] = args.provider
    if args.llm_model:
        config["LLM"]["model"] = args.llm_model
        config["LLM"]["llm_model"] = args.llm_model
    config["LLM"]["embedding_provider"] = args.embedding_provider
    config.setdefault("Embedding", {})["provider"] = args.embedding_provider
    if args.embedding_model:
        config["Embedding"]["model"] = args.embedding_model
        config["Embedding"]["embedding_model"] = args.embedding_model
        config["LLM"]["embedding_model"] = args.embedding_model
    if args.embedding_dim is not None:
        config["Embedding"]["target_dim"] = args.embedding_dim
        config["LLM"]["output_dim"] = args.embedding_dim
    config["LLM"]["max_llm_queries"] = args.max_llm_queries
    if args.qwen_base_url:
        config["LLM"]["qwen_base_url"] = args.qwen_base_url
        config["Embedding"]["qwen_base_url"] = args.qwen_base_url
    config["Fusion"]["mode"] = args.gate_mode
    config["Fusion"]["abstention_threshold"] = args.abstention_threshold
    config["ClusteringHead"]["input_source"] = args.input_source
    config["ClusteringHead"]["head_type"] = args.head_type
    config["ClusteringHead"]["representation"] = args.representation
    config["ClusteringHead"]["epochs"] = args.epochs
    config["ClusteringHead"]["batch_size"] = args.batch_size
    config["ClusteringHead"]["lr"] = args.lr
    config["ClusteringHead"]["weight_decay"] = args.weight_decay
    config["ClusteringHead"]["temperature"] = args.temperature
    config["ClusteringHead"]["lambda_contrastive"] = args.lambda_contrastive
    config["ClusteringHead"]["lambda_kl"] = args.lambda_kl
    config["ClusteringHead"]["lambda_balance"] = args.lambda_balance
    config["ClusteringHead"]["warmup_epochs"] = args.warmup_epochs
    config["MethodRunner"]["provider"] = args.provider
    config["MethodRunner"]["embedding_provider"] = args.embedding_provider
    config["MethodRunner"]["gate_mode"] = args.gate_mode
    config["MethodRunner"]["input_source"] = args.input_source
    config["MethodRunner"]["head_type"] = args.head_type
    config["MethodRunner"]["representation"] = args.representation
    config["MethodRunner"]["llm_missing_fallback"] = args.llm_missing_fallback
    config["MethodRunner"]["reuse_existing"] = args.reuse_existing
    config["MethodRunner"]["force_stage1"] = args.force_stage1
    config["MethodRunner"]["force_stage2a"] = args.force_stage2a
    config["MethodRunner"]["force_stage2b"] = args.force_stage2b
    config["MethodRunner"]["force_stage4a"] = args.force_stage4a
    config["MethodRunner"]["run_stage1"] = not args.skip_stage1
    config["MethodRunner"]["run_stage2a"] = not args.skip_stage2a
    config["MethodRunner"]["run_stage2b"] = not args.skip_stage2b
    config["MethodRunner"]["run_stage4a"] = not args.skip_stage4a
    config["MethodRunner"]["allow_unsafe_stage2a"] = args.allow_unsafe_stage2a
    config["MethodRunner"]["allow_debug_fusion"] = args.allow_debug_fusion
    return config


def main():
    args = _parse_args()
    config = _build_config(args)
    fail_fast = bool(config["MethodRunner"].get("fail_fast", True))
    cwd = Path(__file__).resolve().parent

    stage1_dir = get_stage1_dir(args.dataset, args.missing_rate, args.seed)
    stage2a_dir = get_stage2a_dir(args.dataset, args.missing_rate, args.seed)
    fusion_dir = get_fusion_dir(args.dataset, args.missing_rate, args.seed, args.gate_mode)
    clustering_head_dir = get_clustering_head_dir(
        args.dataset,
        args.missing_rate,
        args.seed,
        args.gate_mode,
        args.input_source,
        args.head_type,
        args.representation,
    )
    method_dir = (
        Path(args.output_dir)
        if args.output_dir
        else get_method_dir(
            args.dataset,
            args.missing_rate,
            args.seed,
            args.gate_mode,
            args.input_source,
            args.head_type,
            args.representation,
        )
    )
    method_dir.mkdir(parents=True, exist_ok=True)

    method_metrics_path = method_dir / "method_metrics.json"
    method_summary_path = method_dir / "method_summary.json"
    command_log_path = method_dir / "command_log.json"
    resolved_config_path = method_dir / "resolved_config.json"

    _print_header(args)
    command_log = []
    stages = {}

    if args.dry_run_api:
        print("[dry-run-api] Stage 4B will only print/run the Stage 2C API dry-run command.")
        cmd = python_cmd(
            "run_llm_semantic_path.py",
            "--dataset",
            args.dataset,
            "--missing-rate",
            args.missing_rate,
            "--seed",
            args.seed,
            "--provider",
            args.provider,
            "--embedding-provider",
            args.embedding_provider,
            "--device",
            args.device,
            "--dry-run-api",
        )
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
        print("running:")
        print("  " + " ".join(str(part) for part in cmd))
        return

    stages["stage1_statistical_path"] = _stage1(args, stage1_dir, command_log, cwd, fail_fast)
    stages["stage2a_llm_semantic_path"] = _stage2a(args, stage2a_dir, command_log, cwd, fail_fast)
    stages["stage2b_fusion_gate"] = _stage2b(args, fusion_dir, command_log, cwd, fail_fast)
    stages["stage4a_clustering_head"] = _stage4a(args, clustering_head_dir, command_log, cwd, fail_fast)

    metrics = read_json(clustering_head_dir / "metrics.json")
    head_summary = read_json(clustering_head_dir / "head_summary.json")
    if metrics is None:
        raise RuntimeError(f"Stage 4A metrics not found: {clustering_head_dir / 'metrics.json'}")
    if head_summary is None:
        raise RuntimeError(f"Stage 4A head summary not found: {clustering_head_dir / 'head_summary.json'}")

    primary_value = metrics["kmeans_on_head_representation"]["NMI"]
    method_metrics = {
        "method": "LLM-GIMVC",
        "dataset": args.dataset,
        "missing_rate": args.missing_rate,
        "seed": args.seed,
        "gate_mode": args.gate_mode,
        "input_source": args.input_source,
        "head_type": args.head_type,
        "representation": args.representation,
        "head_assignment": metrics["head_assignment"],
        "kmeans_on_head_representation": metrics["kmeans_on_head_representation"],
        "primary_metric": {
            "name": "kmeans_on_head_representation.NMI",
            "value": primary_value,
        },
    }

    stage2a_unsafe_allowed = bool(args.allow_unsafe_stage2a)
    fusion_debug_allowed = bool(args.allow_debug_fusion)
    debug_only = bool(
        stage2a_unsafe_allowed
        or fusion_debug_allowed
        or stages["stage2a_llm_semantic_path"].get("is_partial", False)
        or not stages["stage2a_llm_semantic_path"].get("safe_for_fusion", True)
        or stages["stage2b_fusion_gate"].get("debug_only", False)
        or stages["stage2b_fusion_gate"].get("used_partial_stage2a", False)
        or not stages["stage2b_fusion_gate"].get("stage2a_safe_for_fusion", True)
    )

    method_summary = {
        "stage": "stage4b_method_runner",
        "method": "LLM-GIMVC",
        "dataset": args.dataset,
        "missing_rate": args.missing_rate,
        "missing_rate_dir": get_missing_rate_dir(args.missing_rate),
        "seed": args.seed,
        "device": args.device,
        "provider": args.provider,
        "embedding_provider": args.embedding_provider,
        "llm_model": args.llm_model,
        "embedding_model": args.embedding_model,
        "embedding_dim": args.embedding_dim,
        "max_llm_queries": args.max_llm_queries,
        "qwen_base_url": args.qwen_base_url,
        "gate_mode": args.gate_mode,
        "abstention_threshold": args.abstention_threshold,
        "input_source": args.input_source,
        "head_type": args.head_type,
        "representation": args.representation,
        "stages": stages,
        "debug_only": debug_only,
        "stage2a_unsafe_allowed": stage2a_unsafe_allowed,
        "fusion_debug_allowed": fusion_debug_allowed,
        "metrics": method_metrics,
        "paths": {
            "stage1_dir": stage1_dir.as_posix(),
            "stage2a_dir": stage2a_dir.as_posix(),
            "fusion_dir": fusion_dir.as_posix(),
            "clustering_head_dir": clustering_head_dir.as_posix(),
            "method_dir": method_dir.as_posix(),
            "method_metrics_path": method_metrics_path.as_posix(),
            "method_summary_path": method_summary_path.as_posix(),
            "command_log_path": command_log_path.as_posix(),
            "resolved_config_path": resolved_config_path.as_posix(),
        },
        "stage4a_head_summary": head_summary if config["MethodRunner"].get("save_stage_summaries", True) else None,
    }

    write_json(method_metrics_path, method_metrics)
    write_json(method_summary_path, method_summary)
    if config["MethodRunner"].get("save_command_log", True):
        write_json(command_log_path, command_log)
    write_json(resolved_config_path, make_json_safe(config))

    print("Final metrics:")
    print("head_assignment:")
    print(f"  {_format_metrics(metrics['head_assignment'])}")
    print("kmeans_on_head_representation:")
    print(f"  {_format_metrics(metrics['kmeans_on_head_representation'])}")
    print("Primary metric:")
    print(f"  kmeans_on_head_representation.NMI = {primary_value:.6f}")
    print("Saved:")
    print(f"method_metrics.json: {method_metrics_path.as_posix()}")
    print(f"method_summary.json: {method_summary_path.as_posix()}")
    print(f"command_log.json: {command_log_path.as_posix()}")


if __name__ == "__main__":
    main()
