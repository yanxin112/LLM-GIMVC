import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from clustering_eval import evaluate_representation
from comparison_eval import build_method_representation, build_method_tensors, compute_metric_deltas
from configure import get_default_config
from datasets import get_loader
from llm_semantic_path import format_missing_rate
from model import get_statistical_generator
from util import set_seed


def _device_from_arg(device_arg):
    if device_arg == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _infer_config_from_data(config, features, labels):
    config["Module"]["in_dim"] = [feature.shape[1] for feature in features]
    config["Dataset"]["num_views"] = len(features)
    config["Dataset"]["num_sample"] = features[0].shape[0]
    config["Dataset"]["num_classes"] = int(np.unique(labels).size)
    return config


def _stage1_dir(config):
    dataset = config["Dataset"]["name"]
    rate = format_missing_rate(config["Dataset"]["missing_rate"])
    seed = config["training"]["seed"]
    return Path("outputs") / "statistical_path" / dataset / f"missing_{rate}" / f"seed_{seed}"


def _stage2a_dir(config):
    dataset = config["Dataset"]["name"]
    rate = format_missing_rate(config["Dataset"]["missing_rate"])
    seed = config["training"]["seed"]
    return Path("outputs") / "llm_semantic_path" / dataset / f"missing_{rate}" / f"seed_{seed}"


def _fusion_dir(config):
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
        Path(config["Evaluation"]["comparison_output_root"])
        / dataset
        / f"missing_{rate}"
        / f"seed_{seed}"
        / gate_mode
        / representation
    )


def _read_json(path):
    if not path.exists():
        raise RuntimeError(f"Required summary not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _check_stage2a_safety(summary, allow_unsafe):
    is_partial = bool(summary.get("is_partial", False))
    safe_for_fusion = bool(summary.get("safe_for_fusion", False))
    if (is_partial or not safe_for_fusion) and not allow_unsafe:
        raise RuntimeError(
            "Stage 2A output is partial or unsafe. Re-run full Stage 2A or pass --allow-unsafe-stage2a for debugging only."
        )


def _check_fusion_safety(summary, allow_debug):
    debug_only = bool(summary.get("debug_only", False))
    used_partial = bool(summary.get("used_partial_stage2a", False))
    stage2a_safe = bool(summary.get("stage2a_safe_for_fusion", True))
    if (debug_only or used_partial or not stage2a_safe) and not allow_debug:
        raise RuntimeError(
            "Fusion output is debug-only. Re-run Fusion Gate with full safe Stage 2A output and heuristic mode, "
            "or pass --allow-debug-fusion for debugging only."
        )


def _load_stage1_model(config, model_path, device):
    checkpoint = _torch_load(model_path, device)
    model_cfg = checkpoint.get("config", config)
    input_dims = checkpoint.get("input_dims") or config["Module"]["in_dim"]
    model = get_statistical_generator(
        input_dims,
        d_model=model_cfg["Module"].get("trans_dim", config["Module"]["trans_dim"]),
        n_layers=model_cfg["Module"].get("trans_layers", config["Module"]["trans_layers"]),
        heads=model_cfg["Module"].get("trans_headers", config["Module"]["trans_headers"]),
        dropout=model_cfg["Module"].get("trans_dropout", config["Module"]["trans_dropout"]),
        device=device,
    )
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()
    return model


def _build_latent_fea(model, features, available_mask, device):
    all_x = [torch.from_numpy(feature).to(device) for feature in features]
    with torch.no_grad():
        return model.encode_views(all_x, available_mask)


def _validate_shapes(latent_fea, available_mask, y_stat, y_llm, query_mask, y_final, labels):
    expected_3d = latent_fea.shape
    expected_2d = latent_fea.shape[:2]
    if y_stat.shape != expected_3d or y_llm.shape != expected_3d or y_final.shape != expected_3d:
        raise ValueError(
            "3D shape mismatch: "
            f"latent_fea={tuple(latent_fea.shape)}, y_stat={tuple(y_stat.shape)}, "
            f"y_llm={tuple(y_llm.shape)}, y_final={tuple(y_final.shape)}"
        )
    if query_mask.shape != expected_2d or available_mask.shape != expected_2d:
        raise ValueError(
            "2D shape mismatch: "
            f"available_mask={tuple(available_mask.shape)}, query_mask={tuple(query_mask.shape)}, "
            f"expected={tuple(expected_2d)}"
        )
    if len(labels) != latent_fea.shape[0]:
        raise ValueError(f"Label length {len(labels)} does not match sample count {latent_fea.shape[0]}")


def _write_csv(path, methods, metrics_by_method, deltas):
    fieldnames = [
        "method",
        "NMI",
        "ARI",
        "ACC",
        "Purity",
        "NMI_delta_vs_statistical_only",
        "ARI_delta_vs_statistical_only",
        "ACC_delta_vs_statistical_only",
        "Purity_delta_vs_statistical_only",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method in methods:
            row = {"method": method}
            row.update(metrics_by_method[method])
            row.update(
                {
                    f"{metric}_delta_vs_statistical_only": deltas[method][metric]
                    for metric in ["NMI", "ARI", "ACC", "Purity"]
                }
            )
            writer.writerow(row)


def _print_table(methods, metrics_by_method, deltas):
    print("Comparison metrics:")
    print(f"{'method':<18} {'NMI':>9} {'ARI':>9} {'ACC':>9} {'Purity':>9} {'DeltaNMI vs stat':>18}")
    for method in methods:
        metrics = metrics_by_method[method]
        print(
            f"{method:<18} {metrics['NMI']:>9.6f} {metrics['ARI']:>9.6f} "
            f"{metrics['ACC']:>9.6f} {metrics['Purity']:>9.6f} {deltas[method]['NMI']:>18.6f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Run Stage 3B comparison evaluation.")
    parser.add_argument("--dataset", type=str, default="BDGP")
    parser.add_argument("--missing-rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gate-mode", type=str, default="heuristic")
    parser.add_argument("--representation", choices=["mean", "sum", "concat"], default="mean")
    parser.add_argument("--stage1-dir", type=str, default=None)
    parser.add_argument("--stage2a-dir", type=str, default=None)
    parser.add_argument("--fusion-dir", type=str, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--allow-debug-fusion", action="store_true")
    parser.add_argument("--allow-unsafe-stage2a", action="store_true")
    parser.add_argument("--llm-missing-fallback", choices=["zero", "stat"], default="zero")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-save-representations", action="store_true")
    args = parser.parse_args()

    config = get_default_config(args.dataset)
    config["Dataset"]["name"] = args.dataset
    config["Dataset"]["missing_rate"] = args.missing_rate
    config["training"]["seed"] = args.seed
    config["Fusion"]["mode"] = args.gate_mode
    config["Evaluation"]["representation"] = args.representation
    config["Evaluation"]["llm_missing_fallback"] = args.llm_missing_fallback
    if args.no_save_representations:
        config["Evaluation"]["save_method_representations"] = False

    methods = args.methods or config["Evaluation"]["comparison_methods"]
    device = _device_from_arg(args.device)
    stage1_dir = Path(args.stage1_dir) if args.stage1_dir else _stage1_dir(config)
    stage2a_dir = Path(args.stage2a_dir) if args.stage2a_dir else _stage2a_dir(config)
    fusion_dir = Path(args.fusion_dir) if args.fusion_dir else _fusion_dir(config)

    stage2a_summary = _read_json(stage2a_dir / "run_summary.json")
    fusion_summary = _read_json(fusion_dir / "fusion_summary.json")
    _check_stage2a_safety(stage2a_summary, args.allow_unsafe_stage2a)
    _check_fusion_safety(fusion_summary, args.allow_debug_fusion)

    set_seed(config["training"]["seed"])
    loader, features, labels, inc_mask, masked_x = get_loader(config, device)
    labels = np.asarray(labels).reshape(-1)
    config = _infer_config_from_data(config, features, labels)
    num_clusters = int(config["Dataset"].get("num_classes") or np.unique(labels).size)
    if num_clusters <= 0:
        num_clusters = int(np.unique(labels).size)

    available_mask = torch.from_numpy(inc_mask).to(device).float()
    stage1_model = _load_stage1_model(config, stage1_dir / "model.pt", device)
    latent_fea = _build_latent_fea(stage1_model, features, available_mask, device)
    y_stat = _torch_load(stage1_dir / "y_stat.pt", device).float()
    y_llm = _torch_load(stage2a_dir / "y_llm.pt", device).float()
    query_mask = _torch_load(stage2a_dir / "query_mask.pt", device).float()
    y_final = _torch_load(fusion_dir / "y_final.pt", device).float()

    _validate_shapes(latent_fea, available_mask, y_stat, y_llm, query_mask, y_final, labels)

    method_tensors = build_method_tensors(
        latent_fea=latent_fea,
        available_mask=available_mask,
        y_stat=y_stat,
        y_llm=y_llm,
        query_mask=query_mask,
        y_final=y_final,
        llm_missing_fallback=args.llm_missing_fallback,
    )

    output_dir = Path(args.output_dir) if args.output_dir else _output_dir(config, args.representation)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_method = {}
    method_representation_shapes = {}
    for method in methods:
        if method not in method_tensors:
            raise ValueError(f"Unknown comparison method: {method}")
        rep = build_method_representation(
            method_tensors[method],
            method_name=method,
            available_mask=available_mask,
            mode=args.representation,
            normalize=config["Evaluation"]["normalize_representation"],
        )
        metrics, y_pred = evaluate_representation(
            rep,
            labels,
            num_clusters,
            seed=args.seed,
            n_init=config["Evaluation"]["kmeans_n_init"],
            max_iter=config["Evaluation"]["kmeans_max_iter"],
        )
        metrics_by_method[method] = metrics
        method_representation_shapes[method] = [int(x) for x in rep.shape]

        method_dir = output_dir / "methods" / method
        method_dir.mkdir(parents=True, exist_ok=True)
        with open(method_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        np.save(method_dir / "y_pred.npy", y_pred.astype(np.int64))
        if config["Evaluation"]["save_method_representations"]:
            np.save(method_dir / "representation.npy", rep.astype("float32"))

    deltas = compute_metric_deltas(metrics_by_method, reference="statistical_only")
    comparison_metrics = dict(metrics_by_method)
    comparison_metrics["deltas_vs_statistical_only"] = deltas

    comparison_metrics_path = output_dir / "comparison_metrics.json"
    comparison_table_path = output_dir / "comparison_table.csv"
    comparison_summary_path = output_dir / "comparison_summary.json"

    with open(comparison_metrics_path, "w", encoding="utf-8") as f:
        json.dump(comparison_metrics, f, indent=2)
    _write_csv(comparison_table_path, methods, metrics_by_method, deltas)

    fusion_minus_stat_nmi = deltas.get("fusion", {}).get("NMI")
    best_method_by_nmi = max(methods, key=lambda method: metrics_by_method[method]["NMI"])
    debug_only = bool(args.allow_debug_fusion or args.allow_unsafe_stage2a)
    comparison_summary = {
        "stage": "stage3b_comparison_eval",
        "dataset": config["Dataset"]["name"],
        "missing_rate": config["Dataset"]["missing_rate"],
        "missing_rate_dir": f"missing_{format_missing_rate(config['Dataset']['missing_rate'])}",
        "seed": config["training"]["seed"],
        "gate_mode": args.gate_mode,
        "representation": args.representation,
        "methods": methods,
        "reference_method": "statistical_only",
        "num_samples": int(latent_fea.shape[0]),
        "num_views": int(latent_fea.shape[1]),
        "dim": int(latent_fea.shape[2]),
        "num_clusters": int(num_clusters),
        "method_representation_shapes": method_representation_shapes,
        "metrics_by_method": metrics_by_method,
        "deltas_vs_statistical_only": deltas,
        "fusion_minus_statistical_nmi": fusion_minus_stat_nmi,
        "best_method_by_nmi": best_method_by_nmi,
        "stage2a_is_partial": bool(stage2a_summary.get("is_partial", False)),
        "stage2a_safe_for_fusion": bool(stage2a_summary.get("safe_for_fusion", False)),
        "fusion_debug_only": bool(fusion_summary.get("debug_only", False)),
        "fusion_used_partial_stage2a": bool(fusion_summary.get("used_partial_stage2a", False)),
        "debug_only": debug_only,
        "stage1_dir": stage1_dir.as_posix(),
        "stage2a_dir": stage2a_dir.as_posix(),
        "fusion_dir": fusion_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "comparison_metrics_path": comparison_metrics_path.as_posix(),
        "comparison_table_path": comparison_table_path.as_posix(),
        "comparison_summary_path": comparison_summary_path.as_posix(),
    }
    with open(comparison_summary_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)

    print(f"dataset: {args.dataset}")
    print(f"missing rate: {args.missing_rate}")
    print(f"seed: {args.seed}")
    print(f"gate mode: {args.gate_mode}")
    print(f"representation: {args.representation}")
    print(f"methods: {', '.join(methods)}")
    print(f"latent_fea shape: {tuple(latent_fea.shape)}")
    print(f"y_stat shape: {tuple(y_stat.shape)}")
    print(f"y_llm shape: {tuple(y_llm.shape)}")
    print(f"y_final shape: {tuple(y_final.shape)}")
    print(f"available_mask shape: {tuple(available_mask.shape)}")
    print(f"query_mask shape: {tuple(query_mask.shape)}")
    print(f"num clusters: {num_clusters}")
    _print_table(methods, metrics_by_method, deltas)
    if "fusion" in deltas:
        print("fusion - statistical_only:")
        print(f"NMI delta: {deltas['fusion']['NMI']:.6f}")
        print(f"ARI delta: {deltas['fusion']['ARI']:.6f}")
        print(f"ACC delta: {deltas['fusion']['ACC']:.6f}")
        print(f"Purity delta: {deltas['fusion']['Purity']:.6f}")
    print(f"best method by NMI: {best_method_by_nmi}")
    print(f"saved comparison metrics: {comparison_metrics_path.as_posix()}")
    print(f"saved comparison table: {comparison_table_path.as_posix()}")
    print(f"saved comparison summary: {comparison_summary_path.as_posix()}")


if __name__ == "__main__":
    main()
