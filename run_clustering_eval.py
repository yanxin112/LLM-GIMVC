import argparse
import json
from pathlib import Path

import numpy as np
import torch

from clustering_eval import build_sample_representation, evaluate_representation
from configure import get_default_config
from datasets import get_loader
from llm_semantic_path import format_missing_rate


def _device_from_arg(device_arg):
    if device_arg == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _default_fusion_dir(config):
    dataset = config["Dataset"]["name"]
    rate = format_missing_rate(config["Dataset"]["missing_rate"])
    seed = config["training"]["seed"]
    gate_mode = config["Fusion"]["mode"]
    return Path(config["Fusion"]["output_root"]) / dataset / f"missing_{rate}" / f"seed_{seed}" / gate_mode


def _output_dir(config, representation):
    dataset = config["Dataset"]["name"]
    rate = format_missing_rate(config["Dataset"]["missing_rate"])
    seed = config["training"]["seed"]
    gate_mode = config["Fusion"]["mode"]
    return (
        Path(config["Evaluation"]["output_root"])
        / dataset
        / f"missing_{rate}"
        / f"seed_{seed}"
        / gate_mode
        / representation
    )


def _read_fusion_summary(fusion_dir, require_summary=True):
    summary_path = fusion_dir / "fusion_summary.json"
    if not summary_path.exists():
        if require_summary:
            raise RuntimeError(f"Fusion summary not found: {summary_path}")
        return {}, summary_path
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f), summary_path


def _check_fusion_safety(fusion_summary, allow_debug):
    debug_only = bool(fusion_summary.get("debug_only", False))
    used_partial = bool(fusion_summary.get("used_partial_stage2a", False))
    stage2a_safe = bool(fusion_summary.get("stage2a_safe_for_fusion", True))

    if (debug_only or used_partial or not stage2a_safe) and not allow_debug:
        raise RuntimeError(
            "Fusion output is debug-only. Re-run Fusion Gate with full safe Stage 2A output and heuristic mode, "
            "or pass --allow-debug-fusion for debugging only."
        )


def _load_y_final(y_final_path, device):
    y_final = _torch_load(y_final_path, device)
    if isinstance(y_final, dict):
        y_final = y_final["y_final"]
    if not torch.is_tensor(y_final):
        raise ValueError(f"Expected tensor y_final at {y_final_path}")
    return y_final.float()


def main():
    parser = argparse.ArgumentParser(description="Run Stage 3A KMeans clustering evaluation.")
    parser.add_argument("--dataset", type=str, default="BDGP")
    parser.add_argument("--missing-rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gate-mode", type=str, default="heuristic")
    parser.add_argument("--fusion-dir", type=str, default=None)
    parser.add_argument("--y-final", type=str, default=None)
    parser.add_argument("--representation", choices=["mean", "sum", "concat"], default="mean")
    parser.add_argument("--allow-debug-fusion", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = get_default_config(args.dataset)
    config["Dataset"]["name"] = args.dataset
    config["Dataset"]["missing_rate"] = args.missing_rate
    config["training"]["seed"] = args.seed
    config["Fusion"]["mode"] = args.gate_mode
    config["Evaluation"]["representation"] = args.representation

    device = _device_from_arg(args.device)
    fusion_dir = Path(args.fusion_dir) if args.fusion_dir else _default_fusion_dir(config)
    y_final_path = Path(args.y_final) if args.y_final else fusion_dir / "y_final.pt"
    fusion_summary, fusion_summary_path = _read_fusion_summary(
        fusion_dir,
        require_summary=config["Evaluation"]["require_fusion_summary"],
    )
    if config["Evaluation"]["require_non_debug_fusion"]:
        _check_fusion_safety(fusion_summary, args.allow_debug_fusion)

    if not y_final_path.exists():
        raise FileNotFoundError(f"y_final not found: {y_final_path}")

    loader, features, labels, inc_mask, masked_x = get_loader(config, device)
    labels = np.asarray(labels).reshape(-1)
    num_clusters = int(config["Dataset"].get("num_classes") or np.unique(labels).size)
    if num_clusters <= 0:
        num_clusters = int(np.unique(labels).size)

    y_final = _load_y_final(y_final_path, device)
    rep = build_sample_representation(
        y_final,
        mode=args.representation,
        normalize=config["Evaluation"]["normalize_representation"],
    )
    metrics, y_pred = evaluate_representation(
        rep,
        labels,
        num_clusters=num_clusters,
        seed=args.seed,
        n_init=config["Evaluation"]["kmeans_n_init"],
        max_iter=config["Evaluation"]["kmeans_max_iter"],
    )

    output_dir = Path(args.output_dir) if args.output_dir else _output_dir(config, args.representation)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    y_pred_path = output_dir / "y_pred.npy"
    representation_path = output_dir / "representation.npy"
    eval_summary_path = output_dir / "eval_summary.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    np.save(y_pred_path, y_pred.astype(np.int64))
    np.save(representation_path, rep.astype("float32"))

    eval_summary = {
        "stage": "stage3a_clustering_eval",
        "dataset": config["Dataset"]["name"],
        "missing_rate": config["Dataset"]["missing_rate"],
        "missing_rate_dir": f"missing_{format_missing_rate(config['Dataset']['missing_rate'])}",
        "seed": config["training"]["seed"],
        "gate_mode": args.gate_mode,
        "representation": args.representation,
        "num_samples": int(y_final.shape[0]),
        "num_views": int(y_final.shape[1]),
        "dim": int(y_final.shape[2]),
        "representation_shape": [int(x) for x in rep.shape],
        "num_clusters": int(num_clusters),
        "metrics": metrics,
        "fusion_dir": fusion_dir.as_posix(),
        "y_final_path": y_final_path.as_posix(),
        "fusion_summary_path": fusion_summary_path.as_posix(),
        "fusion_debug_only": bool(fusion_summary.get("debug_only", False)),
        "fusion_used_partial_stage2a": bool(fusion_summary.get("used_partial_stage2a", False)),
        "fusion_stage2a_safe_for_fusion": bool(fusion_summary.get("stage2a_safe_for_fusion", True)),
        "metrics_path": metrics_path.as_posix(),
        "y_pred_path": y_pred_path.as_posix(),
        "representation_path": representation_path.as_posix(),
        "eval_summary_path": eval_summary_path.as_posix(),
    }
    with open(eval_summary_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"dataset: {args.dataset}")
    print(f"missing rate: {args.missing_rate}")
    print(f"seed: {args.seed}")
    print(f"gate mode: {args.gate_mode}")
    print(f"representation: {args.representation}")
    print(f"fusion dir: {fusion_dir.as_posix()}")
    print(f"y_final path: {y_final_path.as_posix()}")
    print(f"fusion summary: {fusion_summary_path.as_posix()}")
    print(f"y_final shape: {tuple(y_final.shape)}")
    print(f"representation shape: {tuple(rep.shape)}")
    print(f"num clusters: {num_clusters}")
    print(f"NMI: {metrics['NMI']:.6f}")
    print(f"ARI: {metrics['ARI']:.6f}")
    print(f"ACC: {metrics['ACC']:.6f}")
    print(f"Purity: {metrics['Purity']:.6f}")
    print(f"saved metrics: {metrics_path.as_posix()}")
    print(f"saved y_pred: {y_pred_path.as_posix()}")
    print(f"saved representation: {representation_path.as_posix()}")
    print(f"saved summary: {eval_summary_path.as_posix()}")


if __name__ == "__main__":
    main()
