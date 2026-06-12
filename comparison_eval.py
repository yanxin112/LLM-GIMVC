import torch
import torch.nn.functional as F

from clustering_eval import build_sample_representation


def build_method_tensors(
    latent_fea,
    available_mask,
    y_stat,
    y_llm,
    query_mask,
    y_final,
    llm_missing_fallback="zero",
):
    available = available_mask.unsqueeze(-1).float()
    missing = 1.0 - available

    observed_tensor = available * latent_fea
    stat_tensor = available * latent_fea + missing * y_stat

    query = query_mask.unsqueeze(-1).float()
    if llm_missing_fallback == "zero":
        missing_llm = query * y_llm
    elif llm_missing_fallback == "stat":
        missing_llm = query * y_llm + (1.0 - query) * y_stat
    else:
        raise ValueError(f"Unknown llm_missing_fallback: {llm_missing_fallback}")
    llm_tensor = available * latent_fea + missing * missing_llm

    return {
        "observed_only": observed_tensor,
        "statistical_only": stat_tensor,
        "llm_only": llm_tensor,
        "fusion": y_final,
    }


def build_method_representation(
    method_tensor,
    method_name,
    available_mask,
    mode="mean",
    normalize=True,
):
    if method_name == "observed_only" and mode in ["mean", "sum"]:
        mask = available_mask.unsqueeze(-1).float()
        observed_count = mask.sum(dim=1).clamp(min=1.0)
        if mode == "mean":
            rep = method_tensor.sum(dim=1) / observed_count
        elif mode == "sum":
            rep = method_tensor.sum(dim=1)
        if normalize:
            rep = F.normalize(rep, p=2, dim=1)
        return rep.detach().cpu().numpy().astype("float32")

    return build_sample_representation(method_tensor, mode=mode, normalize=normalize)


def compute_metric_deltas(metrics_by_method, reference="statistical_only"):
    if reference not in metrics_by_method:
        raise ValueError(f"Reference method not found: {reference}")

    ref_metrics = metrics_by_method[reference]
    deltas = {}
    for method, metrics in metrics_by_method.items():
        deltas[method] = {}
        for metric_name, value in metrics.items():
            deltas[method][metric_name] = float(value - ref_metrics[metric_name])
    return deltas
