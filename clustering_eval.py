import contextlib
import io

import numpy as np
import torch
import torch.nn.functional as F


def build_sample_representation(y_final, mode="mean", normalize=True):
    if y_final.ndim != 3:
        raise ValueError(f"y_final must be a 3D tensor [N,V,D], got shape {tuple(y_final.shape)}")

    n_samples, num_views, dim = y_final.shape
    if mode == "mean":
        rep = y_final.mean(dim=1)
    elif mode == "sum":
        rep = y_final.sum(dim=1)
    elif mode == "concat":
        rep = y_final.reshape(n_samples, num_views * dim)
    else:
        raise ValueError(f"Unknown representation mode: {mode}")

    if normalize:
        rep = F.normalize(rep, p=2, dim=1)
    return rep.detach().cpu().numpy().astype("float32")


def _to_contiguous(labels):
    labels = np.asarray(labels).reshape(-1)
    unique = np.unique(labels)
    mapping = {label: idx for idx, label in enumerate(unique)}
    return np.asarray([mapping[label] for label in labels], dtype=np.int64), unique


def _contingency_matrix(y_true, y_pred):
    y_true, true_labels = _to_contiguous(y_true)
    y_pred, pred_labels = _to_contiguous(y_pred)
    matrix = np.zeros((len(true_labels), len(pred_labels)), dtype=np.int64)
    for true_idx, pred_idx in zip(y_true, y_pred):
        matrix[true_idx, pred_idx] += 1
    return matrix


def _comb2(x):
    x = np.asarray(x, dtype=np.float64)
    return x * (x - 1.0) / 2.0


def _hungarian_max(profit):
    profit = np.asarray(profit, dtype=np.float64)
    rows, cols = profit.shape
    size = max(rows, cols)
    padded = np.zeros((size, size), dtype=np.float64)
    padded[:rows, :cols] = profit
    cost = padded.max() - padded

    u = np.zeros(size + 1)
    v = np.zeros(size + 1)
    p = np.zeros(size + 1, dtype=np.int64)
    way = np.zeros(size + 1, dtype=np.int64)

    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = np.full(size + 1, np.inf)
        used = np.zeros(size + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, size + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = np.zeros(size, dtype=np.int64)
    for j in range(1, size + 1):
        assignment[p[j] - 1] = j - 1
    return assignment[:rows]


def _cluster_accuracy(y_true, y_pred):
    contingency = _contingency_matrix(y_true, y_pred)
    assignment = _hungarian_max(contingency)
    matched = 0
    for row, col in enumerate(assignment):
        if row < contingency.shape[0] and col < contingency.shape[1]:
            matched += contingency[row, col]
    return float(matched / max(np.sum(contingency), 1))


def _purity_score(y_true, y_pred):
    contingency = _contingency_matrix(y_true, y_pred)
    return float(np.max(contingency, axis=0).sum() / max(np.sum(contingency), 1))


def _nmi_score(y_true, y_pred):
    contingency = _contingency_matrix(y_true, y_pred).astype(np.float64)
    total = contingency.sum()
    if total == 0:
        return 0.0
    pi = contingency.sum(axis=1) / total
    pj = contingency.sum(axis=0) / total
    pij = contingency / total
    nonzero = pij > 0
    mi = np.sum(pij[nonzero] * np.log(pij[nonzero] / (pi[:, None] * pj[None, :])[nonzero]))
    h_true = -np.sum(pi[pi > 0] * np.log(pi[pi > 0]))
    h_pred = -np.sum(pj[pj > 0] * np.log(pj[pj > 0]))
    denom = h_true + h_pred
    return float(0.0 if denom == 0 else (2.0 * mi / denom))


def _ari_score(y_true, y_pred):
    contingency = _contingency_matrix(y_true, y_pred)
    sum_comb = _comb2(contingency).sum()
    row_comb = _comb2(contingency.sum(axis=1)).sum()
    col_comb = _comb2(contingency.sum(axis=0)).sum()
    total_comb = _comb2(contingency.sum())
    if total_comb == 0:
        return 0.0
    expected = row_comb * col_comb / total_comb
    max_index = 0.5 * (row_comb + col_comb)
    denom = max_index - expected
    return float(0.0 if denom == 0 else (sum_comb - expected) / denom)


def clustering_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"Label length mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            from sklearn import metrics

        nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
        ari = metrics.adjusted_rand_score(y_true, y_pred)
    except Exception:
        nmi = _nmi_score(y_true, y_pred)
        ari = _ari_score(y_true, y_pred)

    return {
        "NMI": float(nmi),
        "ARI": float(ari),
        "ACC": _cluster_accuracy(y_true, y_pred),
        "Purity": _purity_score(y_true, y_pred),
    }


def _numpy_kmeans(rep, num_clusters, seed=0, n_init=20, max_iter=300):
    rep = np.asarray(rep, dtype=np.float32)
    rng = np.random.default_rng(seed)
    best_labels = None
    best_inertia = np.inf

    for _ in range(n_init):
        init_idx = rng.choice(rep.shape[0], size=num_clusters, replace=False)
        centers = rep[init_idx].copy()
        labels = np.zeros(rep.shape[0], dtype=np.int64)
        for _ in range(max_iter):
            distances = (
                np.sum(rep ** 2, axis=1, keepdims=True)
                - 2.0 * rep @ centers.T
                + np.sum(centers ** 2, axis=1, keepdims=True).T
            )
            new_labels = np.argmin(distances, axis=1)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            for cluster_idx in range(num_clusters):
                members = rep[labels == cluster_idx]
                if members.size == 0:
                    centers[cluster_idx] = rep[rng.integers(0, rep.shape[0])]
                else:
                    centers[cluster_idx] = members.mean(axis=0)

        inertia = float(np.sum((rep - centers[labels]) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    return best_labels


def run_kmeans(rep, num_clusters, seed=0, n_init=20, max_iter=300):
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            from sklearn.cluster import KMeans

        model = KMeans(
            n_clusters=num_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_state=seed,
        )
        return model.fit_predict(rep)
    except Exception:
        print("WARNING: sklearn KMeans unavailable; using deterministic NumPy KMeans fallback.")
        return _numpy_kmeans(rep, num_clusters, seed=seed, n_init=n_init, max_iter=max_iter)


def evaluate_representation(rep, labels, num_clusters, seed=0, n_init=20, max_iter=300):
    y_pred = run_kmeans(rep, num_clusters, seed=seed, n_init=n_init, max_iter=max_iter)
    metrics = clustering_metrics(labels, y_pred)
    return metrics, y_pred
