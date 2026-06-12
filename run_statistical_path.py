import argparse
import json
from pathlib import Path

import numpy as np
import torch

from configure import get_default_config
from datasets import get_loader, get_transformer_loader2
from loss import StatisticalPathLoss
from model import get_statistical_generator
from util import set_seed


def _device_from_arg(device_arg):
    if device_arg == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def _infer_config_from_data(config, features, labels):
    config["Module"]["in_dim"] = [feature.shape[1] for feature in features]
    config["Dataset"]["num_views"] = len(features)
    config["Dataset"]["num_sample"] = features[0].shape[0]
    config["Dataset"]["num_classes"] = int(np.unique(labels).size)
    return config


def _format_missing_rate(missing_rate):
    rate = float(missing_rate)
    if rate <= 1.0:
        return str(int(round(rate * 100)))
    return str(int(round(rate)))


def _output_dir(config):
    dataset = config["Dataset"]["name"]
    missing_rate = _format_missing_rate(config["Dataset"]["missing_rate"])
    seed = config["training"]["seed"]
    return Path("outputs") / "statistical_path" / dataset / f"missing_{missing_rate}" / f"seed_{seed}"


def _build_target_mask(in_mask, out_mask, loss_target):
    if loss_target == "visible":
        return out_mask
    if loss_target == "heldout":
        target_mask = (out_mask - in_mask).clamp(min=0)
        if torch.sum(target_mask) == 0:
            return out_mask
        return target_mask
    raise ValueError(f"Unknown loss_target: {loss_target}")


def _build_latent_features(model, features, available_mask, device):
    all_x = [torch.from_numpy(feature).to(device) for feature in features]
    with torch.no_grad():
        fea = model.encode_views(all_x, available_mask)
    return fea


def generate_y_stat(model, fea, in_mask):
    model.eval()
    with torch.no_grad():
        y_stat = model(fea, in_mask)
    return y_stat


def train_statistical_generator(config, device=None, return_data=False):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seed(config["training"]["seed"])
    loader, features, labels, inc_mask, masked_x = get_loader(config, device)
    config = _infer_config_from_data(config, features, labels)

    model = get_statistical_generator(
        config["Module"]["in_dim"],
        d_model=config["Module"]["trans_dim"],
        n_layers=config["Module"]["trans_layers"],
        heads=config["Module"]["trans_headers"],
        dropout=config["Module"]["trans_dropout"],
        device=device,
        load_weights=config.get("load_model"),
    )

    freeze_embeddings = config["training"].get("freeze_embedding_layers", True)
    if freeze_embeddings:
        for p in model.embedding_layers.parameters():
            p.requires_grad = False
        optimizer_params = model.ETrans.parameters()
    else:
        raise NotImplementedError(
            "Training embedding_layers is not supported in Stage 1 because latent_fea is precomputed with no_grad. "
            "Set freeze_embedding_layers=True or refactor training to compute embeddings inside the batch loop."
        )

    print(f"freeze_embedding_layers: {freeze_embeddings}")
    print("trainable modules: ETrans only")

    available_mask = torch.from_numpy(inc_mask).to(device)
    fea = _build_latent_features(model, features, available_mask, device)
    data_loader_tf = get_transformer_loader2(
        [fea[:, view_idx].cpu().numpy() for view_idx in range(config["Dataset"]["num_views"])],
        available_mask,
        config["training"]["batch_size_tf"],
        device,
    )

    loss_model = StatisticalPathLoss()
    optimizer = torch.optim.Adam(optimizer_params, lr=config["training"]["lr"])
    epochs = config["training"]["epoch_tf"]
    loss_target = config["training"].get("loss_target", "heldout")
    epoch_losses = []
    last_batch_loss = None
    last_epoch_mean_loss = None
    first_batch_logged = False

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        for mask_fea, fea_target, in_mask, out_mask, idx in data_loader_tf:
            y_stat = model.forward_transformer(mask_fea, in_mask)
            target_mask = _build_target_mask(in_mask, out_mask, loss_target)
            loss = loss_model.weighted_wmse_loss(y_stat, fea_target, target_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last_batch_loss = float(loss.detach().cpu())
            epoch_loss += last_batch_loss
            batch_count += 1

            if not first_batch_logged:
                print(f"loss_target: {loss_target}")
                print(f"fea shape: {tuple(fea_target.shape)}")
                print(f"mask_fea shape: {tuple(mask_fea.shape)}")
                print(f"in_mask shape: {tuple(in_mask.shape)}")
                print(f"out_mask shape: {tuple(out_mask.shape)}")
                print(f"target_mask shape: {tuple(target_mask.shape)}")
                print(f"y_stat shape: {tuple(y_stat.shape)}")
                first_batch_logged = True

        mean_loss = epoch_loss / max(batch_count, 1)
        epoch_losses.append(mean_loss)
        last_epoch_mean_loss = mean_loss
        print(f"epoch {epoch + 1}/{epochs} reconstruction loss: {mean_loss:.6f}")

    data = dict(
        features=features,
        labels=labels,
        inc_mask=inc_mask,
        available_mask=available_mask,
        latent_fea=fea,
        masked_x=masked_x,
        last_batch_loss=last_batch_loss,
        last_epoch_mean_loss=last_epoch_mean_loss,
        epoch_losses=epoch_losses,
        loss_target=loss_target,
        freeze_embedding_layers=freeze_embeddings,
    )
    return (model, data) if return_data else model


def main():
    parser = argparse.ArgumentParser(description="Train the statistical Transformer y_stat path.")
    parser.add_argument("--dataset", type=str, default="BDGP")
    parser.add_argument("--missing-rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--loss-target", choices=["visible", "heldout"], default=None)
    parser.add_argument("--train-embeddings", action="store_true")
    args = parser.parse_args()

    config = get_default_config(args.dataset)
    config["Dataset"]["name"] = args.dataset
    config["Dataset"]["missing_rate"] = args.missing_rate
    config["training"]["seed"] = args.seed
    if args.epochs is not None:
        config["training"]["epoch_tf"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size_tf"] = args.batch_size
    if args.load_model is not None:
        config["load_model"] = args.load_model
    if args.loss_target is not None:
        config["training"]["loss_target"] = args.loss_target
    if args.train_embeddings:
        config["training"]["freeze_embedding_layers"] = False

    device = _device_from_arg(args.device)
    print(f"dataset: {args.dataset}")
    print(f"missing rate: {args.missing_rate}")
    print(f"seed: {args.seed}")
    print(f"device: {device}")

    model, data = train_statistical_generator(config, device=device, return_data=True)

    y_stat = generate_y_stat(model, data["latent_fea"], data["available_mask"])
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    y_stat_path = output_dir / "y_stat.pt"
    model_path = output_dir / "model.pt"
    summary_path = output_dir / "run_summary.json"

    torch.save(y_stat.cpu(), y_stat_path)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "input_dims": config["Module"]["in_dim"],
        },
        model_path,
    )

    summary = {
        "dataset": config["Dataset"]["name"],
        "missing_rate": config["Dataset"]["missing_rate"],
        "missing_rate_dir": f"missing_{_format_missing_rate(config['Dataset']['missing_rate'])}",
        "seed": config["training"]["seed"],
        "device": str(device),
        "num_samples": config["Dataset"]["num_sample"],
        "num_views": config["Dataset"]["num_views"],
        "trans_dim": config["Module"]["trans_dim"],
        "loss_target": data["loss_target"],
        "freeze_embedding_layers": data["freeze_embedding_layers"],
        "trainable_modules": ["ETrans"],
        "epoch_losses": data["epoch_losses"],
        "final_reconstruction_loss": data["last_epoch_mean_loss"],
        "y_stat_shape": list(y_stat.shape),
        "y_stat_path": y_stat_path.as_posix(),
        "model_path": model_path.as_posix(),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"final y_stat shape: {tuple(y_stat.shape)}")
    print(f"final reconstruction loss: {data['last_epoch_mean_loss']:.6f}")
    print(f"saved y_stat: {y_stat_path.as_posix()}")
    print(f"saved model: {model_path.as_posix()}")
    print(f"saved summary: {summary_path.as_posix()}")


if __name__ == "__main__":
    main()
