import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from torch.nn.functional import normalize


OFFICIAL_DIR = Path(__file__).resolve().parent / "official"
sys.path.insert(0, str(OFFICIAL_DIR))

from dataloader import TrainDataset_All
from network import FreeCSL


def parse_args():
    parser = argparse.ArgumentParser(description="Run official FreeCSL on a unified label-free data bundle.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--missing-rate", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pre-epochs", type=int, default=int(os.environ.get("FREECSL_PRE_EPOCHS", "50")))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("FREECSL_EPOCHS", "100")))
    return parser.parse_args()


def load_bundle(path):
    with np.load(path, allow_pickle=True) as bundle:
        required = {"complete_views", "available_mask", "metadata"}
        missing = sorted(required - set(bundle.files))
        if missing:
            raise ValueError(f"FreeCSL input bundle is missing keys: {missing}")
        views = [np.asarray(view, dtype=np.float32) for view in bundle["complete_views"]]
        available_mask = np.asarray(bundle["available_mask"], dtype=np.float32)
        metadata = json.loads(str(np.asarray(bundle["metadata"]).item()))
    return views, available_mask, metadata


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_features(model, loader, device, contrastive):
    outputs = [[] for _ in range(model.args.view_num)]
    model.eval()
    with torch.no_grad():
        for xs, _, _, _ in loader:
            xs = [item.to(device) for item in xs]
            features = (
                model.get_Single_constrZs(xs)
                if contrastive
                else model.get_Single_reconHs(xs)
            )
            for view_index in range(model.args.view_num):
                outputs[view_index].append(features[view_index].cpu())
    return [torch.cat(parts, dim=0).to(device) for parts in outputs]


def common_features(features, masks):
    numerator = sum(feature * mask.unsqueeze(1) for feature, mask in zip(features, masks))
    denominator = sum(masks).clamp_min(1.0).unsqueeze(1)
    return numerator / denominator


def main():
    cli = parse_args()
    output_dir = Path(cli.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    views, available_mask, metadata = load_bundle(cli.data)
    if available_mask.shape != (views[0].shape[0], len(views)):
        raise ValueError("FreeCSL available_mask shape does not match the view data.")

    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    seed_everything(cli.seed)
    views = [
        torch.from_numpy(MinMaxScaler().fit_transform(view).astype(np.float32))
        for view in views
    ]
    masks = [
        torch.from_numpy(available_mask[:, view_index]).float()
        for view_index in range(available_mask.shape[1])
    ]
    dummy_targets = [
        np.zeros(views[0].shape[0], dtype=np.int64)
        for _ in views
    ]
    indices = np.arange(views[0].shape[0])
    cluster_count = int(metadata["num_clusters"])
    batch_size = min(512, views[0].shape[0])
    args = SimpleNamespace(
        device=device,
        dataset=cli.dataset,
        cluster_num=cluster_count,
        data_num=views[0].shape[0],
        view_num=len(views),
        view_dims=[view.shape[1] for view in views],
        missrate=float(cli.missing_rate),
        Pre_epochs=cli.pre_epochs,
        epochs=cli.epochs,
        batch_pre=batch_size,
        batch=batch_size,
        lr_pre=3.0e-4,
        lr_train=5.0e-4,
        recon_fea_dim=64,
        gamma=1.0,
        alpha=1.0,
        tau=0.2,
        epsilon=0.05,
        sinkhorn_iterations=3,
        z_dim=64,
        K_neighber=3,
        collapse_regularization=0.2,
        lamda=0.1,
        graph_out_dim=64,
        graph_h_dim=128,
    )
    dataset = TrainDataset_All(views, dummy_targets, masks, indices)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    model = FreeCSL(args).to(device)
    optimizer_pre = torch.optim.Adam(model.parameters(), lr=args.lr_pre)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_train)

    model.train()
    for _ in range(cli.pre_epochs):
        for xs, _, batch_masks, _ in train_loader:
            xs = [item.to(device) for item in xs]
            batch_masks = [item.to(device) for item in batch_masks]
            loss = model.train_Recon(xs, batch_masks)
            optimizer_pre.zero_grad()
            loss.backward()
            optimizer_pre.step()

    all_masks = [mask.to(device) for mask in masks]
    estimator = KMeans(n_clusters=cluster_count, n_init=20, random_state=cli.seed)
    for _ in range(cli.epochs):
        h_features = get_features(model, loader, device, contrastive=False)
        z_features = get_features(model, loader, device, contrastive=True)
        h_common = common_features(h_features, all_masks).cpu().numpy()
        z_common = common_features(z_features, all_masks).cpu().numpy()
        estimator.fit(h_common)
        centroids_h = estimator.cluster_centers_
        estimator.fit(z_common)
        centroids_z = estimator.cluster_centers_
        with torch.no_grad():
            model.clu_H_layer.data = normalize(
                torch.tensor(centroids_h, device=device, dtype=torch.float32),
                dim=1,
                p=2,
            )
            model.clu_Z_layer.data = normalize(
                torch.tensor(centroids_z, device=device, dtype=torch.float32),
                dim=1,
                p=2,
            )
        model.train()
        for xs, _, batch_masks, _ in train_loader:
            xs = [item.to(device) for item in xs]
            batch_masks = [item.to(device) for item in batch_masks]
            reconstruction = model.train_Recon(xs, batch_masks)
            _, contrastive = model.train_Constr(xs, batch_masks)
            _, graph = model.train_Graph(xs, batch_masks)
            loss = reconstruction + contrastive + graph
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    final_features = get_features(model, loader, device, contrastive=True)
    final_common = common_features(final_features, all_masks).cpu().numpy()
    predictions = KMeans(
        n_clusters=cluster_count,
        n_init=20,
        random_state=cli.seed,
    ).fit_predict(final_common)
    np.save(output_dir / "pred_labels.npy", predictions.astype(np.int64))
    with open(output_dir / "official_adapter_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": "freecsl",
                "official_repository": "https://github.com/zoyadai/2025_CVPR_FreeCSL",
                "dataset": cli.dataset,
                "missing_rate": cli.missing_rate,
                "seed": cli.seed,
                "device": str(device),
                "pre_epochs": cli.pre_epochs,
                "epochs": cli.epochs,
                "num_samples": int(predictions.size),
                "num_clusters": int(np.unique(predictions).size),
                "labels_read": False,
                "hyperparameter_note": "BDGP is absent from the official config; official generic defaults were used.",
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
