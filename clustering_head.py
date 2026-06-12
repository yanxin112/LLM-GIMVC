import contextlib
import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from clustering_eval import run_kmeans


class DCPStyleClusteringHead(nn.Module):
    """
    Minimal DCP-style trainable clustering head for recovered multi-view latent tensors.

    Input:
        x_views: Tensor [B,V,D]

    Output:
        {
            "z_views": Tensor [B,V,H],
            "sample_rep": Tensor [B,H] or [B,V*H],
            "q": Tensor [B,C],
            "logits": Tensor [B,C],
        }
    """

    def __init__(
        self,
        input_dim,
        num_views,
        num_clusters,
        projection_dim=512,
        use_projection=True,
        representation="mean",
        temperature=0.5,
    ):
        super().__init__()
        if representation not in ["mean", "sum", "concat"]:
            raise ValueError(f"Unknown representation mode: {representation}")
        if num_clusters <= 0:
            raise ValueError("num_clusters must be positive")
        if num_views <= 0:
            raise ValueError("num_views must be positive")

        self.input_dim = int(input_dim)
        self.num_views = int(num_views)
        self.num_clusters = int(num_clusters)
        self.projection_dim = int(projection_dim)
        self.use_projection = bool(use_projection)
        self.representation = representation
        self.temperature = float(temperature)

        if self.use_projection:
            self.projector = nn.Sequential(
                nn.Linear(self.input_dim, self.projection_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(self.projection_dim),
            )
            view_dim = self.projection_dim
        else:
            self.projector = nn.Identity()
            view_dim = self.input_dim

        self.view_dim = int(view_dim)
        if self.representation == "concat":
            self.rep_dim = self.num_views * self.view_dim
        else:
            self.rep_dim = self.view_dim

        self.cluster_centers = nn.Parameter(torch.empty(self.num_clusters, self.rep_dim))
        nn.init.xavier_uniform_(self.cluster_centers)

    def _aggregate(self, z_views):
        if self.representation == "mean":
            return z_views.mean(dim=1)
        if self.representation == "sum":
            return z_views.sum(dim=1)
        if self.representation == "concat":
            return z_views.reshape(z_views.shape[0], self.num_views * self.view_dim)
        raise ValueError(f"Unknown representation mode: {self.representation}")

    def forward(self, x_views):
        if x_views.ndim != 3:
            raise ValueError(f"x_views must be a 3D tensor [B,V,D], got {tuple(x_views.shape)}")
        if x_views.shape[1] != self.num_views or x_views.shape[2] != self.input_dim:
            raise ValueError(
                "x_views shape mismatch: "
                f"expected [B,{self.num_views},{self.input_dim}], got {tuple(x_views.shape)}"
            )

        z_views = self.projector(x_views.float())
        z_views = F.normalize(z_views, p=2, dim=-1)
        sample_rep = self._aggregate(z_views)
        sample_rep = F.normalize(sample_rep, p=2, dim=1)

        centers = F.normalize(self.cluster_centers, p=2, dim=1)
        dist = torch.cdist(sample_rep, centers, p=2).pow(2)
        q = 1.0 / (1.0 + dist)
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        q = q.clamp_min(1.0e-8)
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1.0e-8)

        return {
            "z_views": z_views,
            "sample_rep": sample_rep,
            "q": q,
            "logits": -dist,
        }


def view_contrastive_loss(z_views, temperature=0.5):
    """
    Multi-view instance contrastive loss.

    z_views: Tensor [B,V,H]
    """
    if z_views.ndim != 3:
        raise ValueError(f"z_views must be a 3D tensor [B,V,H], got {tuple(z_views.shape)}")
    batch_size, num_views, _ = z_views.shape
    if num_views < 2:
        return z_views.new_tensor(0.0)

    z_views = F.normalize(z_views, p=2, dim=-1)
    labels = torch.arange(batch_size, device=z_views.device)
    total_loss = z_views.new_tensor(0.0)
    pair_count = 0

    for view_i in range(num_views):
        for view_j in range(view_i + 1, num_views):
            logits = z_views[:, view_i, :] @ z_views[:, view_j, :].T
            logits = logits / float(temperature)
            total_loss = total_loss + F.cross_entropy(logits, labels)
            total_loss = total_loss + F.cross_entropy(logits.T, labels)
            pair_count += 2

    return total_loss / max(pair_count, 1)


def target_distribution(q):
    """
    DEC target distribution.
    p_ij = q_ij^2 / f_j
    p = p / p.sum(dim=1)
    """
    weight = q ** 2 / q.sum(dim=0, keepdim=True).clamp_min(1.0e-8)
    p = weight / weight.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    return p.detach()


def clustering_kl_loss(q, p):
    """KL(P || Q)."""
    return F.kl_div(q.clamp_min(1.0e-8).log(), p, reduction="batchmean")


def cluster_balance_loss(q):
    """Minimize KL between the batch cluster distribution and uniform distribution."""
    mean_q = q.mean(dim=0)
    num_clusters = q.shape[1]
    uniform = torch.full_like(mean_q, 1.0 / num_clusters)
    return F.kl_div(mean_q.clamp_min(1.0e-8).log(), uniform, reduction="batchmean")


def _compute_sample_representation(model, x_views, batch_size=512, device="cpu"):
    dataset = torch.utils.data.TensorDataset(x_views.detach().float().cpu())
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    reps = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            output = model(batch_x.to(device))
            reps.append(output["sample_rep"].detach().cpu())
    if was_training:
        model.train()
    return torch.cat(reps, dim=0)


def _centers_from_labels(rep, labels, num_clusters, seed=0):
    rep = np.asarray(rep, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    centers = np.zeros((num_clusters, rep.shape[1]), dtype=np.float32)
    for cluster_idx in range(num_clusters):
        members = rep[labels == cluster_idx]
        if members.size == 0:
            centers[cluster_idx] = rep[rng.integers(0, rep.shape[0])]
        else:
            centers[cluster_idx] = members.mean(axis=0)
    return centers


def initialize_cluster_centers(model, x_views, num_clusters, seed=0, batch_size=512, device="cpu"):
    """Use current model sample representations and KMeans to initialize cluster centers."""
    rep_tensor = _compute_sample_representation(model, x_views, batch_size=batch_size, device=device)
    rep = rep_tensor.numpy().astype("float32")
    backend = "sklearn"

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=num_clusters, n_init=20, max_iter=300, random_state=seed)
        kmeans.fit(rep)
        centers = kmeans.cluster_centers_.astype("float32")
    except Exception:
        backend = "fallback"
        labels = run_kmeans(rep, num_clusters, seed=seed, n_init=20, max_iter=300)
        centers = _centers_from_labels(rep, labels, num_clusters, seed=seed)

    centers_tensor = torch.from_numpy(centers).to(model.cluster_centers.device).float()
    if centers_tensor.shape != model.cluster_centers.shape:
        raise ValueError(
            "KMeans center shape mismatch: "
            f"centers={tuple(centers_tensor.shape)}, model={tuple(model.cluster_centers.shape)}"
        )

    with torch.no_grad():
        model.cluster_centers.data.copy_(centers_tensor)

    return {
        "init_method": "kmeans",
        "backend": backend,
        "num_clusters": int(num_clusters),
        "rep_shape": [int(x) for x in rep.shape],
    }
