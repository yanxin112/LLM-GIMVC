import torch
import torch.nn as nn


class HeuristicFusionGate(nn.Module):
    def __init__(self, abstention_threshold=0.3):
        super().__init__()
        self.abstention_threshold = abstention_threshold

    def forward(self, c_llm, s_cons, query_mask, missing_mask):
        w = c_llm * s_cons
        eligible_mask = query_mask.float() * missing_mask.float()
        w = w * eligible_mask
        w = torch.where(c_llm >= self.abstention_threshold, w, torch.zeros_like(w))
        return w.clamp(0.0, 1.0)


class FusionGateMLP(nn.Module):
    def __init__(self, d_model=512, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_model + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, y_stat, y_llm, c_llm, s_cons, query_mask, missing_mask, abstention_threshold=0.3):
        scalar_inputs = torch.stack([c_llm, s_cons], dim=-1)
        gate_input = torch.cat([y_stat, y_llm, scalar_inputs], dim=-1)
        w = self.net(gate_input).squeeze(-1)

        eligible_mask = query_mask.float() * missing_mask.float()
        w = w * eligible_mask
        w = torch.where(c_llm >= abstention_threshold, w, torch.zeros_like(w))
        return w.clamp(0.0, 1.0)


def fuse_views(
    latent_fea,
    y_stat,
    y_llm,
    gate_weight,
    available_mask,
    preserve_observed=True,
):
    w = gate_weight.unsqueeze(-1)
    y_recovered = w * y_llm + (1 - w) * y_stat

    if preserve_observed:
        observed = available_mask.unsqueeze(-1).float()
        y_final = observed * latent_fea + (1 - observed) * y_recovered
    else:
        y_final = y_recovered

    source_mask = torch.ones_like(available_mask, dtype=torch.long)
    source_mask = torch.where(available_mask.bool(), torch.zeros_like(source_mask), source_mask)
    fused_mask = (~available_mask.bool()) & (gate_weight > 0)
    source_mask = torch.where(fused_mask, torch.full_like(source_mask, 2), source_mask)
    return y_final, source_mask
