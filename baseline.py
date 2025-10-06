# switch_transformer_minimal.py
import copy
from typing import List, Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F


# -----------------------------
# Minimal building blocks
# -----------------------------
class FeedForward(nn.Module):
    """Simple 2-layer feed-forward network used as an expert (replaceable)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, d_model]
        return self.fc2(self.dropout(self.act(self.fc1(x))))


def clone_module_list(module: nn.Module, n: int) -> nn.ModuleList:
    """Deep-copy a module `n` times into an nn.ModuleList."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class SimpleMultiHeadAttention(nn.Module):
    """Thin wrapper around nn.MultiheadAttention with convenient call signature."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # PyTorch MultiheadAttention expects seq_len x batch x embed
        # mask argument here is optional and passed as attn_mask if provided
        # attn_mask expected shape: (target_len, source_len) or broadcastable
        attn_out, attn_weights = self.mha(query, key, value, attn_mask=mask)
        return attn_out


# -----------------------------
# Switch Feed-Forward (MoE) Layer
# -----------------------------
class SwitchFeedForward(nn.Module):
    """
    Switch-style MoE FFN.
    Input shape: [seq_len, batch_size, d_model]
    Returns:
      final_output: [seq_len, batch_size, d_model]
      counts: tensor of ints (#tokens assigned to each expert)
      route_prob_sums: tensor of floats (sum of router probs per expert)
      n_dropped: int (how many tokens were dropped because of capacity)
      route_prob_max: tensor [tokens] (top-1 prob per token)
    """

    def __init__(self,
                 *,
                 capacity_factor: float,
                 drop_tokens: bool,
                 is_scale_prob: bool,
                 n_experts: int,
                 expert: FeedForward,
                 d_model: int):
        super().__init__()

        self.capacity_factor = capacity_factor
        self.is_scale_prob = is_scale_prob
        self.n_experts = n_experts
        self.drop_tokens = drop_tokens

        # clone experts
        self.experts = clone_module_list(expert, n_experts)

        # router: maps d_model -> n_experts
        self.switch = nn.Linear(d_model, n_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        # x: [seq_len, batch_size, d_model]
        seq_len, batch_size, d_model = x.shape
        tokens = x.view(-1, d_model)  # [T, d_model], T = seq_len * batch_size
        T = tokens.size(0)
        device = tokens.device

        # compute router probabilities
        logits = self.switch(tokens)  # [T, E]
        route_prob = self.softmax(logits)  # [T, E]
        route_prob_max, routes = torch.max(route_prob, dim=-1)  # [T], [T] (expert index)

        # prepare lists of token indices for each expert
        indexes_list: List[torch.Tensor] = []
        for i in range(self.n_experts):
            # routes == i -> positions assigned to expert i
            idx = torch.nonzero(routes == i, as_tuple=True)[0]
            indexes_list.append(idx)

        # allocate final output
        final_output = tokens.new_zeros(tokens.shape)  # [T, d_model]

        # capacity per expert
        capacity = int(self.capacity_factor * (T / self.n_experts))

        # counts before trimming
        counts = torch.tensor([idx.numel() for idx in indexes_list], device=device, dtype=torch.long)

        dropped_indices = []

        # if drop_tokens is True, trim each expert's assigned tokens to capacity
        if self.drop_tokens:
            for i in range(self.n_experts):
                idxs = indexes_list[i]
                n = idxs.numel()
                if n <= capacity:
                    continue
                # randomly permute and keep first `capacity` tokens
                perm = torch.randperm(n, device=device)
                keep_perm = perm[:capacity]
                kept = idxs[keep_perm]
                # tokens that won't be processed (dropped)
                drop_perm = perm[capacity:]
                dropped = idxs[drop_perm]
                indexes_list[i] = kept
                dropped_indices.append(dropped)

        # run experts on their assigned tokens (skip empty)
        for i in range(self.n_experts):
            idxs = indexes_list[i]
            if idxs.numel() == 0:
                continue
            sub_x = tokens[idxs, :]  # [n_i, d_model]
            out = self.experts[i](sub_x)  # [n_i, d_model]
            final_output[idxs, :] = out

        # Collate dropped tokens: set output to original token (residual passthrough)
        n_dropped = 0
        if len(dropped_indices) > 0:
            dropped = torch.cat(dropped_indices) if len(dropped_indices) > 1 else dropped_indices[0]
            if dropped.numel() > 0:
                final_output[dropped, :] = tokens[dropped, :]
                n_dropped = dropped.numel()

        # scale outputs by gating probability (top-1 prob)
        if self.is_scale_prob:
            final_output = final_output * route_prob_max.unsqueeze(-1)
        else:
            # normalized scaling (as in some implementations)
            final_output = final_output * (route_prob_max / (route_prob_max.detach() + 1e-12)).unsqueeze(-1)

        # reshape back to [seq_len, batch_size, d_model]
        final_output = final_output.view(seq_len, batch_size, d_model)

        # recompute counts and route_prob_sums for stats
        final_counts = torch.tensor([idx.numel() for idx in indexes_list], device=device, dtype=torch.long)
        route_prob_sums = route_prob.sum(dim=0).detach()  # [E]

        return final_output, final_counts, route_prob_sums, n_dropped, route_prob_max.detach()


# -----------------------------
# Switch Transformer Layer / Model
# -----------------------------
class SwitchTransformerLayer(nn.Module):
    def __init__(self, *,
                 d_model: int,
                 attn: SimpleMultiHeadAttention,
                 feed_forward: SwitchFeedForward,
                 dropout_prob: float = 0.1):
        super().__init__()
        self.size = d_model
        self.attn = attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)

    def forward(self, *, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x: [seq_len, batch_size, d_model]
        z = self.norm_self_attn(x)
        self_attn = self.attn(query=z, key=z, value=z, mask=mask)  # [seq_len, batch, d_model]
        x = x + self.dropout(self_attn)

        z2 = self.norm_ff(x)
        ff_out, counts, route_prob_sums, n_dropped, route_prob_max = self.feed_forward(z2)
        x = x + self.dropout(ff_out)

        return x, counts, route_prob_sums, n_dropped, route_prob_max


class SwitchTransformer(nn.Module):
    def __init__(self, layer: SwitchTransformerLayer, n_layers: int):
        super().__init__()
        # deep-copy layer n_layers times
        self.layers = clone_module_list(layer, n_layers)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        all_counts = []
        all_route_prob_sums = []
        all_n_dropped = []
        all_route_prob_max = []
        for layer in self.layers:
            x, counts, route_prob_sums, n_dropped, route_prob_max = layer(x=x, mask=mask)
            all_counts.append(counts)
            all_route_prob_sums.append(route_prob_sums)
            all_n_dropped.append(n_dropped)
            all_route_prob_max.append(route_prob_max)

        x = self.norm(x)
        # stack results for convenience
        counts_stack = torch.stack(all_counts)  # [n_layers, n_experts]
        probs_stack = torch.stack(all_route_prob_sums)  # [n_layers, n_experts]
        route_max_stack = torch.stack(all_route_prob_max)  # [n_layers, T]
        return x, counts_stack, probs_stack, all_n_dropped, route_max_stack


# -----------------------------
# Quick smoke test / example
# -----------------------------
def smoke_test():
    torch.manual_seed(42)
    device = torch.device("cpu")

    # hyperparams for small test
    d_model = 64
    d_ff = 256
    n_heads = 4
    n_experts = 4
    seq_len = 10
    batch_size = 2
    n_layers = 2

    # build expert FFN prototype
    base_expert = FeedForward(d_model=d_model, d_ff=d_ff, dropout=0.1)

    # Switch feed-forward
    switch_ff = SwitchFeedForward(
        capacity_factor=1.0,
        drop_tokens=True,
        is_scale_prob=True,
        n_experts=n_experts,
        expert=base_expert,
        d_model=d_model
    )

    # attention and layer
    attn = SimpleMultiHeadAttention(d_model=d_model, n_heads=n_heads)
    layer = SwitchTransformerLayer(d_model=d_model, attn=attn, feed_forward=switch_ff, dropout_prob=0.1)
    model = SwitchTransformer(layer, n_layers=n_layers).to(device)

    # random input: [seq_len, batch_size, d_model]
    x = torch.randn(seq_len, batch_size, d_model, device=device)

    out, counts_stack, probs_stack, n_dropped_list, route_max_stack = model(x, mask=None)

    print("Output shape:", out.shape)
    print("Counts per layer (n_experts):")
    print(counts_stack)
    print("Route-prob sums per layer:")
    print(probs_stack)
    print("Dropped tokens per layer (list):", n_dropped_list)
    # route_max_stack shape: [n_layers, T]
    print("Top-1 route-prob (sample):", route_max_stack.shape)


if __name__ == "__main__":
    smoke_test()
