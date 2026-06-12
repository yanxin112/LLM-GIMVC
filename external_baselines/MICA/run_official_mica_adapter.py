import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


OFFICIAL_DIR = Path(__file__).resolve().parent / "official"
sys.path.insert(0, str(OFFICIAL_DIR))

from layers import MainNetwork
from loss import DeepMVCLoss
from train import contrastive_train, inference, pre_train


class LabelFreeMicaDataset(Dataset):
    def __init__(self, views, missing_mask, device):
        self.data_views = [torch.from_numpy(view).to(device).float() for view in views]
        self.missing_matrix = torch.from_numpy(missing_mask).to(device).float()
        self.labels = np.zeros(views[0].shape[0], dtype=np.int64)

    def __len__(self):
        return self.data_views[0].shape[0]

    def __getitem__(self, index):
        return (
            [view[index] for view in self.data_views],
            0,
            self.missing_matrix[index],
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run official MICA on a unified label-free data bundle.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--missing-rate", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mse-epochs", type=int, default=int(os.environ.get("MICA_MSE_EPOCHS", "200")))
    parser.add_argument("--con-epochs", type=int, default=int(os.environ.get("MICA_CON_EPOCHS", "200")))
    return parser.parse_args()


def load_bundle(path):
    with np.load(path, allow_pickle=True) as bundle:
        required = {"complete_views", "missing_mask", "metadata"}
        missing = sorted(required - set(bundle.files))
        if missing:
            raise ValueError(f"MICA input bundle is missing keys: {missing}")
        views = [np.asarray(view, dtype=np.float32) for view in bundle["complete_views"]]
        missing_mask = np.asarray(bundle["missing_mask"], dtype=np.float32)
        metadata = json.loads(str(np.asarray(bundle["metadata"]).item()))
    return views, missing_mask, metadata


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    views, missing_mask, metadata = load_bundle(args.data)
    if args.dataset != "BDGP":
        raise ValueError("This adapter currently has verified official hyperparameters only for BDGP.")
    if missing_mask.shape != (views[0].shape[0], len(views)):
        raise ValueError("MICA missing_mask shape does not match the view data.")

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")
    set_seed(args.seed)

    # Values are the BDGP settings recorded by the official repository.
    batch_size = 250
    learning_rate = 1.0e-4
    dim_high_feature = 2000
    dim_low_feature = 1024
    dims = [256, 512]
    alpha = 0.01
    beta = 0.01
    lmd = 0.01
    gamma = 0.01
    omega = 0.001
    temperature = 1.0
    normalized = False
    num_clusters = int(metadata["num_clusters"])

    dataset = LabelFreeMicaDataset(views, missing_mask, requested_device)
    model = MainNetwork(
        len(views),
        [view.shape[1] for view in views],
        dims,
        dim_high_feature,
        dim_low_feature,
        num_clusters,
        batch_size,
    ).to(requested_device)
    loss_fn = DeepMVCLoss(batch_size, num_clusters)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.con_epochs,
        eta_min=1.0e-5,
    )

    pre_train(model, dataset, batch_size, args.mse_epochs, optimizer)
    for epoch in range(args.con_epochs):
        contrastive_train(
            model,
            dataset,
            loss_fn,
            batch_size,
            alpha,
            beta,
            lmd,
            gamma,
            omega,
            temperature,
            normalized,
            epoch,
            optimizer,
        )
        scheduler.step()

    predictions, _, _ = inference(model, dataset, batch_size)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if predictions.size != views[0].shape[0]:
        raise ValueError(
            f"MICA produced {predictions.size} predictions for {views[0].shape[0]} samples."
        )
    np.save(output_dir / "pred_labels.npy", predictions)
    with open(output_dir / "official_adapter_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": "mica",
                "official_repository": "https://github.com/sunway677/MICA",
                "dataset": args.dataset,
                "missing_rate": args.missing_rate,
                "seed": args.seed,
                "device": str(requested_device),
                "mse_epochs": args.mse_epochs,
                "con_epochs": args.con_epochs,
                "num_samples": int(predictions.size),
                "num_clusters": int(np.unique(predictions).size),
                "labels_read": False,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
