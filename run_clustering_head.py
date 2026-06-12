import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from clustering_eval import clustering_metrics, evaluate_representation
from clustering_head import (
    DCPStyleClusteringHead,
    cluster_balance_loss,
    clustering_kl_loss,
    initialize_cluster_centers,
    target_distribution,
    view_contrastive_loss,
)
from comparison_eval import build_method_tensors
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


def _output_dir(config):
    dataset = config["Dataset"]["name"]
    rate = format_missing_rate(config["Dataset"]["missing_rate"])
    seed = config["training"]["seed"]
    gate_mode = config["Fusion"]["mode"]
    head_cfg = config["ClusteringHead"]
    return (
        Path(head_cfg["output_root"])
        / dataset
        / f"missing_{rate}"
        / f"seed_{seed}"
        / gate_mode
        / head_cfg["input_source"]
        / head_cfg["head_type"]
        / head_cfg["representation"]
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
            "Stage 2A output is partial or unsafe. Re-run full Stage 2A or pass "
            "--allow-unsafe-stage2a for debugging only."
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


def _optional_load(path, device):
    if not path.exists():
        return None
    return _torch_load(path, device).float()


def _shape_or_none(tensor):
    if tensor is None:
        return None
    return [int(x) for x in tensor.shape]


def _metrics_row(epoch, losses, head_metrics, kmeans_metrics):
    row = {
        "epoch": int(epoch),
        "total_loss": float(losses["total_loss"]),
        "contrastive_loss": float(losses["contrastive_loss"]),
        "kl_loss": float(losses["kl_loss"]),
        "balance_loss": float(losses["balance_loss"]),
    }
    for prefix, metrics in [("head", head_metrics), ("kmeans", kmeans_metrics)]:
        row[f"{prefix}_NMI"] = float(metrics["NMI"])
        row[f"{prefix}_ARI"] = float(metrics["ARI"])
        row[f"{prefix}_ACC"] = float(metrics["ACC"])
        row[f"{prefix}_Purity"] = float(metrics["Purity"])
    return row


def _write_train_log(path, rows):
    fieldnames = [
        "epoch",
        "total_loss",
        "contrastive_loss",
        "kl_loss",
        "balance_loss",
        "head_NMI",
        "head_ARI",
        "head_ACC",
        "head_Purity",
        "kmeans_NMI",
        "kmeans_ARI",
        "kmeans_ACC",
        "kmeans_Purity",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _evaluate_head(model, x_views, labels, num_clusters, seed, kmeans_n_init, kmeans_max_iter):
    model.eval()
    with torch.no_grad():
        output = model(x_views)
    sample_rep = output["sample_rep"].detach().cpu().numpy().astype("float32")
    q = output["q"].detach().cpu().numpy().astype("float32")
    y_pred_head = q.argmax(axis=1).astype(np.int64)
    head_metrics = clustering_metrics(labels, y_pred_head)
    kmeans_metrics, y_pred_kmeans = evaluate_representation(
        sample_rep,
        labels,
        num_clusters,
        seed=seed,
        n_init=kmeans_n_init,
        max_iter=kmeans_max_iter,
    )
    return output, head_metrics, kmeans_metrics, y_pred_head, y_pred_kmeans


def _print_metrics(label, metrics):
    print(
        f"  {label} NMI/ARI/ACC/Purity: "
        f"{metrics['NMI']:.6f} / {metrics['ARI']:.6f} / {metrics['ACC']:.6f} / {metrics['Purity']:.6f}"
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="Run Stage 4A trainable clustering head.")
    parser.add_argument("--dataset", type=str, default="BDGP")
    parser.add_argument("--missing-rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gate-mode", type=str, default="heuristic")
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
    parser.add_argument("--stage1-dir", type=str, default=None)
    parser.add_argument("--stage2a-dir", type=str, default=None)
    parser.add_argument("--fusion-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--allow-debug-fusion", action="store_true")
    parser.add_argument("--allow-unsafe-stage2a", action="store_true")
    parser.add_argument("--llm-missing-fallback", choices=["zero", "stat"], default="zero")
    parser.add_argument("--no-save-representations", action="store_true")
    parser.add_argument("--projection-dim", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--no-normalize-input", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.head_type == "completer":
        raise NotImplementedError("Completer-style head is reserved for Stage 4B. Stage 4A implements dcp.")

    config = get_default_config(args.dataset)
    config["Dataset"]["name"] = args.dataset
    config["Dataset"]["missing_rate"] = args.missing_rate
    config["training"]["seed"] = args.seed
    config["Fusion"]["mode"] = args.gate_mode
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
    config["Evaluation"]["llm_missing_fallback"] = args.llm_missing_fallback
    if args.projection_dim is not None:
        config["ClusteringHead"]["projection_dim"] = args.projection_dim
    if args.eval_interval is not None:
        config["ClusteringHead"]["eval_interval"] = args.eval_interval
    if args.no_normalize_input:
        config["ClusteringHead"]["normalize_input"] = False
    if args.no_save_representations:
        config["ClusteringHead"]["save_representations"] = False
    config["ClusteringHead"]["allow_debug_fusion"] = args.allow_debug_fusion

    device = _device_from_arg(args.device)
    stage1_dir = Path(args.stage1_dir) if args.stage1_dir else _stage1_dir(config)
    stage2a_dir = Path(args.stage2a_dir) if args.stage2a_dir else _stage2a_dir(config)
    fusion_dir = Path(args.fusion_dir) if args.fusion_dir else _fusion_dir(config)

    if not (stage1_dir / "model.pt").exists():
        raise FileNotFoundError(f"Stage 1 model not found: {stage1_dir / 'model.pt'}")
    if not (stage1_dir / "y_stat.pt").exists():
        raise FileNotFoundError(f"Stage 1 y_stat not found: {stage1_dir / 'y_stat.pt'}")
    if not (stage2a_dir / "y_llm.pt").exists():
        raise FileNotFoundError(f"Stage 2A y_llm not found: {stage2a_dir / 'y_llm.pt'}")
    if not (stage2a_dir / "query_mask.pt").exists():
        raise FileNotFoundError(f"Stage 2A query_mask not found: {stage2a_dir / 'query_mask.pt'}")
    if not (fusion_dir / "y_final.pt").exists():
        raise FileNotFoundError(f"Stage 2B y_final not found: {fusion_dir / 'y_final.pt'}")

    stage2a_summary = _read_json(stage2a_dir / "run_summary.json")
    fusion_summary = _read_json(fusion_dir / "fusion_summary.json")
    if config["ClusteringHead"]["require_safe_stage2a"]:
        _check_stage2a_safety(stage2a_summary, args.allow_unsafe_stage2a)
    if config["ClusteringHead"]["require_non_debug_fusion"]:
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

    c_llm = _optional_load(stage2a_dir / "c_llm.pt", device)
    s_cons = _optional_load(stage2a_dir / "s_cons.pt", device)
    gate_weight = _optional_load(fusion_dir / "gate_weight.pt", device)
    source_mask = _optional_load(fusion_dir / "source_mask.pt", device)

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
    if args.input_source not in method_tensors:
        raise ValueError(f"Unknown input_source: {args.input_source}")

    x_views = method_tensors[args.input_source].detach().float()
    if config["ClusteringHead"]["normalize_input"]:
        x_views = F.normalize(x_views, p=2, dim=-1)
    if config["ClusteringHead"]["train_input_tensor"]:
        raise RuntimeError("train_input_tensor=True is not supported in Stage 4A.")
    x_views = x_views.to(device)

    n_samples, num_views, input_dim = x_views.shape
    print(f"dataset: {args.dataset}")
    print(f"missing rate: {args.missing_rate}")
    print(f"seed: {args.seed}")
    print(f"device: {device}")
    print(f"gate mode: {args.gate_mode}")
    print(f"input source: {args.input_source}")
    print(f"head type: {args.head_type}")
    print(f"representation: {args.representation}")
    print(f"latent_fea shape: {tuple(latent_fea.shape)}")
    print(f"x_views shape: {tuple(x_views.shape)}")
    print(f"available_mask shape: {tuple(available_mask.shape)}")
    print(f"num clusters: {num_clusters}")

    head_cfg = config["ClusteringHead"]
    model = DCPStyleClusteringHead(
        input_dim=input_dim,
        num_views=num_views,
        num_clusters=num_clusters,
        projection_dim=head_cfg["projection_dim"],
        use_projection=head_cfg["use_projection"],
        representation=args.representation,
        temperature=args.temperature,
    ).to(device)

    init_summary = initialize_cluster_centers(
        model=model,
        x_views=x_views,
        num_clusters=num_clusters,
        seed=args.seed,
        batch_size=args.batch_size,
        device=device,
    )

    dataset = TensorDataset(
        x_views.detach().cpu(),
        available_mask.detach().cpu(),
        torch.arange(n_samples, dtype=torch.long),
    )
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    eval_interval = max(int(head_cfg["eval_interval"]), 1)
    train_log_rows = []
    final_losses = {
        "total_loss": 0.0,
        "contrastive_loss": 0.0,
        "kl_loss": 0.0,
        "balance_loss": 0.0,
    }
    final_output = None
    final_head_metrics = None
    final_kmeans_metrics = None
    final_y_pred_head = None
    final_y_pred_kmeans = None

    print("training:")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_sums = {
            "total_loss": 0.0,
            "contrastive_loss": 0.0,
            "kl_loss": 0.0,
            "balance_loss": 0.0,
        }
        num_batches = 0

        for batch_x_views, batch_available_mask, batch_idx in train_loader:
            batch_x_views = batch_x_views.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_x_views)
            z_views = output["z_views"]
            q = output["q"]

            loss_contrastive = view_contrastive_loss(z_views, temperature=args.temperature)
            loss_balance = cluster_balance_loss(q)
            if epoch <= args.warmup_epochs:
                loss_kl = q.new_tensor(0.0)
            else:
                p = target_distribution(q)
                loss_kl = clustering_kl_loss(q, p)

            total_loss = (
                args.lambda_contrastive * loss_contrastive
                + args.lambda_kl * loss_kl
                + args.lambda_balance * loss_balance
            )
            if not torch.isfinite(total_loss):
                raise RuntimeError(
                    "NaN/Inf loss encountered: "
                    f"total={float(total_loss.detach().cpu())}, "
                    f"contrastive={float(loss_contrastive.detach().cpu())}, "
                    f"kl={float(loss_kl.detach().cpu())}, "
                    f"balance={float(loss_balance.detach().cpu())}"
                )

            total_loss.backward()
            optimizer.step()

            epoch_sums["total_loss"] += float(total_loss.detach().cpu())
            epoch_sums["contrastive_loss"] += float(loss_contrastive.detach().cpu())
            epoch_sums["kl_loss"] += float(loss_kl.detach().cpu())
            epoch_sums["balance_loss"] += float(loss_balance.detach().cpu())
            num_batches += 1

        final_losses = {key: value / max(num_batches, 1) for key, value in epoch_sums.items()}
        should_eval = (epoch % eval_interval == 0) or (epoch == args.epochs)
        if epoch == 1 or should_eval:
            print(
                f"epoch {epoch}/{args.epochs} "
                f"total={final_losses['total_loss']:.6f} "
                f"contrastive={final_losses['contrastive_loss']:.6f} "
                f"kl={final_losses['kl_loss']:.6f} "
                f"balance={final_losses['balance_loss']:.6f}"
            )

        if should_eval:
            (
                final_output,
                final_head_metrics,
                final_kmeans_metrics,
                final_y_pred_head,
                final_y_pred_kmeans,
            ) = _evaluate_head(
                model,
                x_views,
                labels,
                num_clusters,
                seed=args.seed,
                kmeans_n_init=config["Evaluation"]["kmeans_n_init"],
                kmeans_max_iter=config["Evaluation"]["kmeans_max_iter"],
            )
            train_log_rows.append(
                _metrics_row(epoch, final_losses, final_head_metrics, final_kmeans_metrics)
            )
            print(f"eval epoch {epoch}:")
            _print_metrics("head", final_head_metrics)
            _print_metrics("kmeans-on-head-rep", final_kmeans_metrics)

    if final_output is None:
        (
            final_output,
            final_head_metrics,
            final_kmeans_metrics,
            final_y_pred_head,
            final_y_pred_kmeans,
        ) = _evaluate_head(
            model,
            x_views,
            labels,
            num_clusters,
            seed=args.seed,
            kmeans_n_init=config["Evaluation"]["kmeans_n_init"],
            kmeans_max_iter=config["Evaluation"]["kmeans_max_iter"],
        )

    metrics = {
        "head_assignment": final_head_metrics,
        "kmeans_on_head_representation": final_kmeans_metrics,
    }

    output_dir = Path(args.output_dir) if args.output_dir else _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "head_summary.json"
    train_log_path = output_dir / "train_log.csv"
    y_pred_head_path = output_dir / "y_pred_head.npy"
    y_pred_kmeans_path = output_dir / "y_pred_kmeans.npy"
    sample_representation_path = output_dir / "sample_representation.npy"
    q_path = output_dir / "q.npy"
    z_views_path = output_dir / "z_views.npy"

    final_sample_rep = final_output["sample_rep"].detach().cpu().numpy().astype("float32")
    final_q = final_output["q"].detach().cpu().numpy().astype("float32")
    final_z_views = final_output["z_views"].detach().cpu().numpy().astype("float32")

    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "metrics": metrics,
            "init_summary": init_summary,
        },
        model_path,
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(metrics), f, indent=2)
    _write_train_log(train_log_path, train_log_rows)
    np.save(y_pred_head_path, final_y_pred_head.astype(np.int64))
    np.save(y_pred_kmeans_path, final_y_pred_kmeans.astype(np.int64))
    np.save(sample_representation_path, final_sample_rep)
    np.save(q_path, final_q)
    if head_cfg["save_representations"]:
        np.save(z_views_path, final_z_views)

    best_record = max(train_log_rows, key=lambda row: row["kmeans_NMI"]) if train_log_rows else None
    stage2a_is_partial = bool(stage2a_summary.get("is_partial", False))
    stage2a_safe_for_fusion = bool(stage2a_summary.get("safe_for_fusion", False))
    fusion_debug_only = bool(fusion_summary.get("debug_only", False))
    fusion_used_partial = bool(fusion_summary.get("used_partial_stage2a", False))
    fusion_stage2a_safe = bool(fusion_summary.get("stage2a_safe_for_fusion", True))
    debug_only = bool(
        stage2a_is_partial
        or not stage2a_safe_for_fusion
        or fusion_debug_only
        or fusion_used_partial
        or not fusion_stage2a_safe
    )

    head_summary = {
        "stage": "stage4a_clustering_head",
        "dataset": args.dataset,
        "missing_rate": args.missing_rate,
        "missing_rate_dir": f"missing_{format_missing_rate(args.missing_rate)}",
        "seed": args.seed,
        "gate_mode": args.gate_mode,
        "input_source": args.input_source,
        "head_type": args.head_type,
        "representation": args.representation,
        "num_samples": int(n_samples),
        "num_views": int(num_views),
        "input_dim": int(input_dim),
        "projection_dim": int(head_cfg["projection_dim"]),
        "num_clusters": int(num_clusters),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "temperature": float(args.temperature),
        "lambda_contrastive": float(args.lambda_contrastive),
        "lambda_kl": float(args.lambda_kl),
        "lambda_balance": float(args.lambda_balance),
        "warmup_epochs": int(args.warmup_epochs),
        "init_cluster_centers": head_cfg["init_cluster_centers"],
        "init_summary": init_summary,
        "final_losses": final_losses,
        "metrics": metrics,
        "best_epoch_by_kmeans_nmi": int(best_record["epoch"]) if best_record else None,
        "best_kmeans_nmi": float(best_record["kmeans_NMI"]) if best_record else None,
        "best_head_nmi": float(best_record["head_NMI"]) if best_record else None,
        "stage2a_is_partial": stage2a_is_partial,
        "stage2a_safe_for_fusion": stage2a_safe_for_fusion,
        "fusion_debug_only": fusion_debug_only,
        "fusion_used_partial_stage2a": fusion_used_partial,
        "debug_only": debug_only,
        "optional_tensor_shapes": {
            "c_llm": _shape_or_none(c_llm),
            "s_cons": _shape_or_none(s_cons),
            "gate_weight": _shape_or_none(gate_weight),
            "source_mask": _shape_or_none(source_mask),
        },
        "stage1_dir": stage1_dir.as_posix(),
        "stage2a_dir": stage2a_dir.as_posix(),
        "fusion_dir": fusion_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "model_path": model_path.as_posix(),
        "metrics_path": metrics_path.as_posix(),
        "train_log_path": train_log_path.as_posix(),
        "summary_path": summary_path.as_posix(),
        "y_pred_head_path": y_pred_head_path.as_posix(),
        "y_pred_kmeans_path": y_pred_kmeans_path.as_posix(),
        "sample_representation_path": sample_representation_path.as_posix(),
        "q_path": q_path.as_posix(),
        "z_views_path": z_views_path.as_posix() if head_cfg["save_representations"] else None,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(head_summary), f, indent=2)

    print("final metrics:")
    print("head_assignment:")
    print(f"  NMI: {metrics['head_assignment']['NMI']:.6f}")
    print(f"  ARI: {metrics['head_assignment']['ARI']:.6f}")
    print(f"  ACC: {metrics['head_assignment']['ACC']:.6f}")
    print(f"  Purity: {metrics['head_assignment']['Purity']:.6f}")
    print("kmeans_on_head_representation:")
    print(f"  NMI: {metrics['kmeans_on_head_representation']['NMI']:.6f}")
    print(f"  ARI: {metrics['kmeans_on_head_representation']['ARI']:.6f}")
    print(f"  ACC: {metrics['kmeans_on_head_representation']['ACC']:.6f}")
    print(f"  Purity: {metrics['kmeans_on_head_representation']['Purity']:.6f}")
    print(f"saved model: {model_path.as_posix()}")
    print(f"saved metrics: {metrics_path.as_posix()}")
    print(f"saved train log: {train_log_path.as_posix()}")
    print(f"saved summary: {summary_path.as_posix()}")


if __name__ == "__main__":
    main()
