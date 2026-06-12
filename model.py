import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_clones(module, n_views):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n_views)])


def set_embedding_model(d_list, d_out):
    return nn.ModuleList([nn.Linear(d, d_out) for d in d_list])


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        return self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (
            x.std(dim=-1, keepdim=True) + self.eps
        ) + self.bias


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1).float()
        _, _, views = mask.size()
        mask = mask.repeat(1, 1, views).reshape(-1, 1, views, views)
        identity_matrix = torch.eye(views, device=mask.device).unsqueeze(0).unsqueeze(0)
        mask = mask * (1 - identity_matrix)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)
    return torch.matmul(scores, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.out(concat)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.2):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout_1(F.relu(self.linear_1(x)))
        return self.dropout_2(self.linear_2(x))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        return x + self.dropout_2(self.ff(x2))


class Encoder(nn.Module):
    def __init__(self, d_model, n_views, heads, dropout):
        super().__init__()
        self.n_views = n_views
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), n_views)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = src
        for i in range(self.n_views):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, d_model, n_views, heads, dropout):
        super().__init__()
        self.encoder = Encoder(d_model, n_views, heads, dropout)

    def forward(self, src, src_mask):
        return self.encoder(src, src_mask)


class StatisticalViewGenerator(nn.Module):
    """Transformer cross-view generator for the LLM-GIMVC y_stat branch."""

    def __init__(self, input_dims, d_model=512, n_layers=1, heads=4, dropout=0.0):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads")
        self.input_dims = list(input_dims)
        self.num_views = len(self.input_dims)
        self.d_model = d_model
        self.embedding_layers = set_embedding_model(self.input_dims, d_model)
        self.ETrans = Transformer(d_model, n_layers, heads, dropout)

    def encode_views(self, x_list, available_mask=None):
        z = []
        for view_idx, x in enumerate(x_list):
            z.append(self.embedding_layers[view_idx](x.float()))
        fea = torch.stack(z, dim=1)
        if available_mask is not None:
            fea = fea * available_mask.unsqueeze(-1).float()
        return fea

    def forward_transformer(self, fea, in_mask):
        return self.ETrans(fea, in_mask)

    def forward(self, fea, in_mask):
        y_stat = self.forward_transformer(fea, in_mask)
        return y_stat


def get_statistical_generator(
    input_dims,
    d_model=512,
    n_layers=1,
    heads=4,
    dropout=0.0,
    device=torch.device("cpu"),
    load_weights=None,
):
    model = StatisticalViewGenerator(input_dims, d_model, n_layers, heads, dropout)

    if load_weights is not None:
        state = torch.load(load_weights, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state)
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    return model.to(device)
