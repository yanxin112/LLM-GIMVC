import argparse
import json
from pathlib import Path

import numpy as np
import torch

from configure import get_default_config
from datasets import get_loader
from fusion_gate import FusionGateMLP, HeuristicFusionGate, fuse_views
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


def _default_stage1_dir(config):
    dataset = config["Dataset"]["name"]
    rate = format_missing_rate(config["Dataset"]["missing_rate"])
    seed = config["training"]["seed"]
    return Path("outputs") / "statistical_path" / dataset / f"missing_{rate}" / f"seed_{seed}"


def _default_stage2a_dir(config):
    dataset = config["Dataset"]["name"]
    rate = format_missing_rate(config["Dataset"]["missing_rate"])
    seed = config["training"]["seed"]
    return Path("outputs") / "llm_semantic_path" / dataset / f"missing_{rate}" / f"seed_{seed}"


def _output_dir(config):
    dataset = config["Dataset"]["name"]
    rate = format_missing_rate(config["Dataset"]["missing_rate"])
    seed = config["training"]["seed"]
    gate_mode = config["Fusion"]["mode"]
    return Path(config["Fusion"]["output_root"]) / dataset / f"missing_{rate}" / f"seed_{seed}" / gate_mode


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


def _read_stage2a_summary(stage2a_dir, allow_partial):
    summary_path = stage2a_dir / "run_summary.json"
    if not summary_path.exists():
        raise RuntimeError(
            f"Stage 2A run_summary.json not found: {summary_path}. "
            "Run full Stage 2A first. Fusion Gate requires Stage 2A safety metadata."
        )

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    if "partial_max_samples" in stage2a_dir.as_posix():
        if not allow_partial:
            raise RuntimeError(
                "Stage 2A path is a partial_max_samples directory. "
                "Partial Stage 2A outputs are not safe for Fusion Gate. "
                "Run run_llm_semantic_path.py without --max-samples, or pass --allow-partial-stage2a for debugging only."
            )

    if "is_partial" not in summary or "safe_for_fusion" not in summary:
        raise RuntimeError(
            "Stage 2A run_summary.json is missing safety fields: is_partial / safe_for_fusion. "
            "Re-run run_llm_semantic_path.py after applying the partial-output safety patch."
        )

    is_partial = bool(summary.get("is_partial", False))
    safe_for_fusion = bool(summary.get("safe_for_fusion", False))

    if (is_partial or not safe_for_fusion) and not allow_partial:
        raise RuntimeError(
            "Stage 2A output is partial or unsafe for Fusion Gate. "
            "Run run_llm_semantic_path.py without --max-samples, or pass --allow-partial-stage2a for debugging only."
        )

    used_partial = bool(is_partial or not safe_for_fusion)

    if used_partial and allow_partial:
        print("WARNING: using partial/unsafe Stage 2A output because --allow-partial-stage2a was set.")
        print("WARNING: Fusion result is debug-only and must not be used as a formal experiment result.")

    return summary, is_partial, used_partial


def _shape_error(name, actual, expected):
    return f"{name} shape {tuple(actual)} does not match expected {tuple(expected)}"


def _validate_shapes(latent_fea, y_stat, y_llm, c_llm, s_cons, query_mask, available_mask):
    if latent_fea.ndim != 3:
        raise ValueError(f"latent_fea must be 3D, got shape {tuple(latent_fea.shape)}")
    if y_stat.shape != latent_fea.shape:
        raise ValueError(_shape_error("y_stat", y_stat.shape, latent_fea.shape))
    if y_llm.shape != latent_fea.shape:
        raise ValueError(_shape_error("y_llm", y_llm.shape, latent_fea.shape))
    expected_2d = latent_fea.shape[:2]
    for name, tensor in [
        ("c_llm", c_llm),
        ("s_cons", s_cons),
        ("query_mask", query_mask),
        ("available_mask", available_mask),
    ]:
        if tensor.shape != expected_2d:
            raise ValueError(_shape_error(name, tensor.shape, expected_2d))


def _stats_or_none(values):
    if values.numel() == 0:
        return None, None, None, None
    return (
        float(values.mean().detach().cpu()),
        float(values.std(unbiased=False).detach().cpu()),
        float(values.min().detach().cpu()),
        float(values.max().detach().cpu()),
    )


def _run_gate(config, y_stat, y_llm, c_llm, s_cons, query_mask, missing_mask):
    gate_mode = config["Fusion"]["mode"]
    threshold = config["Fusion"]["abstention_threshold"]
    if gate_mode == "heuristic":
        gate = HeuristicFusionGate(abstention_threshold=threshold).to(y_stat.device)
        return gate(c_llm, s_cons, query_mask, missing_mask), False
    if gate_mode == "mlp":
        print("WARNING: MLP FusionGate is untrained in Stage 2B. Use heuristic mode for meaningful debugging.")
        gate = FusionGateMLP(
            d_model=y_stat.shape[-1],
            hidden_dim=config["Fusion"]["gate_hidden_dim"],
        ).to(y_stat.device)
        gate.eval()
        with torch.no_grad():
            return gate(y_stat, y_llm, c_llm, s_cons, query_mask, missing_mask, threshold), True
    raise ValueError(f"Unknown gate mode: {gate_mode}")


def main():
    parser = argparse.ArgumentParser(description="Run Stage 2B Fusion Gate.")
    parser.add_argument("--dataset", type=str, default="BDGP")
    parser.add_argument("--missing-rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--stage1-model", type=str, default=None)
    parser.add_argument("--stage1-y-stat", type=str, default=None)
    parser.add_argument("--stage2a-dir", type=str, default=None)
    parser.add_argument("--gate-mode", choices=["heuristic", "mlp"], default="heuristic")
    parser.add_argument("--abstention-threshold", type=float, default=0.3)
    parser.add_argument("--allow-partial-stage2a", action="store_true")
    args = parser.parse_args()

    config = get_default_config(args.dataset)
    config["Dataset"]["name"] = args.dataset
    config["Dataset"]["missing_rate"] = args.missing_rate
    config["training"]["seed"] = args.seed
    config["Fusion"]["mode"] = args.gate_mode
    config["Fusion"]["abstention_threshold"] = args.abstention_threshold
    config["Fusion"]["allow_partial_stage2a"] = args.allow_partial_stage2a

    device = _device_from_arg(args.device)
    stage1_dir = _default_stage1_dir(config)
    stage1_model_path = Path(args.stage1_model) if args.stage1_model else stage1_dir / "model.pt"
    y_stat_path = Path(args.stage1_y_stat) if args.stage1_y_stat else stage1_dir / "y_stat.pt"
    stage2a_dir = Path(args.stage2a_dir) if args.stage2a_dir else _default_stage2a_dir(config)

    if not stage1_model_path.exists():
        raise FileNotFoundError(f"Stage 1 model not found: {stage1_model_path}")
    if not y_stat_path.exists():
        raise FileNotFoundError(f"Stage 1 y_stat not found: {y_stat_path}")
    if not stage2a_dir.exists():
        raise FileNotFoundError(f"Stage 2A directory not found: {stage2a_dir}")

    print(f"dataset: {args.dataset}")
    print(f"missing rate: {args.missing_rate}")
    print(f"seed: {args.seed}")
    print(f"device: {device}")
    print(f"gate mode: {args.gate_mode}")
    print(f"abstention threshold: {args.abstention_threshold}")
    print(f"stage1 model: {stage1_model_path.as_posix()}")
    print(f"y_stat path: {y_stat_path.as_posix()}")
    print(f"stage2a dir: {stage2a_dir.as_posix()}")

    stage2a_summary, is_partial_stage2a, used_partial_stage2a = _read_stage2a_summary(
        stage2a_dir,
        args.allow_partial_stage2a,
    )

    set_seed(config["training"]["seed"])
    loader, features, labels, inc_mask, masked_x = get_loader(config, device)
    config = _infer_config_from_data(config, features, labels)
    available_mask = torch.from_numpy(inc_mask).to(device).float()
    missing_mask = 1 - available_mask

    stage1_model = _load_stage1_model(config, stage1_model_path, device)
    latent_fea = _build_latent_fea(stage1_model, features, available_mask, device)

    y_stat = _torch_load(y_stat_path, device).float()
    y_llm = _torch_load(stage2a_dir / "y_llm.pt", device).float()
    c_llm = _torch_load(stage2a_dir / "c_llm.pt", device).float()
    s_cons = _torch_load(stage2a_dir / "s_cons.pt", device).float()
    query_mask = _torch_load(stage2a_dir / "query_mask.pt", device).float()

    _validate_shapes(latent_fea, y_stat, y_llm, c_llm, s_cons, query_mask, available_mask)
    eligible_mask = missing_mask * query_mask

    gate_weight, untrained_mlp_debug_only = _run_gate(
        config,
        y_stat,
        y_llm,
        c_llm,
        s_cons,
        query_mask,
        missing_mask,
    )
    y_final, source_mask = fuse_views(
        latent_fea=latent_fea,
        y_stat=y_stat,
        y_llm=y_llm,
        gate_weight=gate_weight,
        available_mask=available_mask,
        preserve_observed=config["Fusion"].get("preserve_observed", True),
    )

    if config["Fusion"].get("preserve_observed", True) and available_mask.bool().any():
        observed_diff = torch.abs(y_final[available_mask.bool()] - latent_fea[available_mask.bool()]).max()
        observed_diff = float(observed_diff.detach().cpu())
        if observed_diff > 1e-5:
            raise RuntimeError(f"Observed views were not preserved. max abs diff={observed_diff}")
    else:
        observed_diff = 0.0

    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    y_final_path = output_dir / "y_final.pt"
    gate_weight_path = output_dir / "gate_weight.pt"
    source_mask_path = output_dir / "source_mask.pt"
    eligible_mask_path = output_dir / "eligible_mask.pt"
    summary_path = output_dir / "fusion_summary.json"

    torch.save(y_final.cpu(), y_final_path)
    torch.save(gate_weight.cpu(), gate_weight_path)
    torch.save(source_mask.cpu(), source_mask_path)
    torch.save(eligible_mask.cpu(), eligible_mask_path)

    eligible_bool = eligible_mask.bool()
    eligible_gate = gate_weight[eligible_bool]
    eligible_c = c_llm[eligible_bool]
    eligible_s = s_cons[eligible_bool]
    gate_mean, gate_std, gate_min, gate_max = _stats_or_none(eligible_gate)
    c_mean, _, _, _ = _stats_or_none(eligible_c)
    s_mean, _, _, _ = _stats_or_none(eligible_s)

    num_observed_entries = int(available_mask.sum().detach().cpu())
    num_missing_entries = int(missing_mask.sum().detach().cpu())
    num_queried_entries = int(query_mask.sum().detach().cpu())
    num_eligible_entries = int(eligible_mask.sum().detach().cpu())
    num_fused_entries = int(((gate_weight > 0) * eligible_bool).sum().detach().cpu())
    num_abstained_entries = int(((c_llm < args.abstention_threshold) * eligible_bool).sum().detach().cpu())
    num_statistical_fallback_entries = int(num_missing_entries - num_fused_entries)

    summary = {
        "stage": "stage2b_fusion_gate",
        "dataset": config["Dataset"]["name"],
        "missing_rate": config["Dataset"]["missing_rate"],
        "missing_rate_dir": f"missing_{format_missing_rate(config['Dataset']['missing_rate'])}",
        "seed": config["training"]["seed"],
        "device": str(device),
        "gate_mode": args.gate_mode,
        "abstention_threshold": args.abstention_threshold,
        "preserve_observed": config["Fusion"].get("preserve_observed", True),
        "num_samples": latent_fea.shape[0],
        "num_views": latent_fea.shape[1],
        "dim": latent_fea.shape[2],
        "num_observed_entries": num_observed_entries,
        "num_missing_entries": num_missing_entries,
        "num_queried_entries": num_queried_entries,
        "num_eligible_entries": num_eligible_entries,
        "num_abstained_entries": num_abstained_entries,
        "num_fused_entries": num_fused_entries,
        "num_statistical_fallback_entries": num_statistical_fallback_entries,
        "gate_weight_mean_all": float(gate_weight.mean().detach().cpu()),
        "gate_weight_mean_eligible": gate_mean,
        "gate_weight_std_eligible": gate_std,
        "gate_weight_min_eligible": gate_min,
        "gate_weight_max_eligible": gate_max,
        "c_llm_mean_eligible": c_mean,
        "s_cons_mean_eligible": s_mean,
        "observed_preservation_max_abs_diff": observed_diff,
        "latent_fea_shape": list(latent_fea.shape),
        "y_stat_shape": list(y_stat.shape),
        "y_llm_shape": list(y_llm.shape),
        "c_llm_shape": list(c_llm.shape),
        "s_cons_shape": list(s_cons.shape),
        "query_mask_shape": list(query_mask.shape),
        "gate_weight_shape": list(gate_weight.shape),
        "y_final_shape": list(y_final.shape),
        "source_mask_shape": list(source_mask.shape),
        "stage1_model_path": stage1_model_path.as_posix(),
        "y_stat_path": y_stat_path.as_posix(),
        "stage2a_dir": stage2a_dir.as_posix(),
        "y_llm_path": (stage2a_dir / "y_llm.pt").as_posix(),
        "c_llm_path": (stage2a_dir / "c_llm.pt").as_posix(),
        "s_cons_path": (stage2a_dir / "s_cons.pt").as_posix(),
        "query_mask_path": (stage2a_dir / "query_mask.pt").as_posix(),
        "y_final_path": y_final_path.as_posix(),
        "gate_weight_path": gate_weight_path.as_posix(),
        "source_mask_path": source_mask_path.as_posix(),
        "eligible_mask_path": eligible_mask_path.as_posix(),
        "stage2a_is_partial": bool(is_partial_stage2a),
        "stage2a_safe_for_fusion": bool(stage2a_summary.get("safe_for_fusion", False)),
        "stage2a_max_samples": stage2a_summary.get("max_samples"),
        "stage2a_num_processed_samples": stage2a_summary.get("num_processed_samples"),
        "stage2a_num_total_samples": stage2a_summary.get("num_total_samples"),
        "used_partial_stage2a": bool(used_partial_stage2a or is_partial_stage2a),
        "untrained_mlp_debug_only": bool(untrained_mlp_debug_only),
        "debug_only": bool(untrained_mlp_debug_only or used_partial_stage2a),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"latent_fea shape: {tuple(latent_fea.shape)}")
    print(f"available_mask shape: {tuple(available_mask.shape)}")
    print(f"missing_mask shape: {tuple(missing_mask.shape)}")
    print(f"y_stat shape: {tuple(y_stat.shape)}")
    print(f"y_llm shape: {tuple(y_llm.shape)}")
    print(f"c_llm shape: {tuple(c_llm.shape)}")
    print(f"s_cons shape: {tuple(s_cons.shape)}")
    print(f"query_mask shape: {tuple(query_mask.shape)}")
    print(f"gate_weight shape: {tuple(gate_weight.shape)}")
    print(f"y_final shape: {tuple(y_final.shape)}")
    print(f"source_mask shape: {tuple(source_mask.shape)}")
    print(f"num observed entries: {num_observed_entries}")
    print(f"num missing entries: {num_missing_entries}")
    print(f"num queried entries: {num_queried_entries}")
    print(f"num eligible entries: {num_eligible_entries}")
    print(f"num fused entries: {num_fused_entries}")
    print(f"num statistical fallback entries: {num_statistical_fallback_entries}")
    print(f"gate weight mean eligible: {gate_mean}")
    print(f"observed_preservation_max_abs_diff: {observed_diff}")
    print(f"saved y_final: {y_final_path.as_posix()}")
    print(f"saved gate_weight: {gate_weight_path.as_posix()}")
    print(f"saved source_mask: {source_mask_path.as_posix()}")
    print(f"saved eligible_mask: {eligible_mask_path.as_posix()}")
    print(f"saved summary: {summary_path.as_posix()}")


if __name__ == "__main__":
    main()
