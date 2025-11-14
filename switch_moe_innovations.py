# switch_moe_innovations.py
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

# ----- simple FeedForward used as expert (replace with your real FFN) -----
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


def clone_module_list(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([type(module)(*getattr(module, "_init_args", ())) if False else
                           torch.clone(module) if False else 
                           # fallback: use deepcopy-like behaviour
                           torch.nn.utils._stateless._deepcopy_module(module, {}, device=None)  # private helper may not always exist
                           for _ in range(n)])


# If torch's private deep copy fails in some envs, fallback to this:
def safe_clone_module_list(module: nn.Module, n: int) -> nn.ModuleList:
    from copy import deepcopy
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


# ----------------- Modified SwitchFeedForward with our innovations -----------------
class SwitchFeedForward(nn.Module):
    """
    Switch-style FFN with:
      - token-importance aware routing (TIA-R)
      - simbal regularizer on router weights
      - adaptive capacity + reassignment (NTLB)
    Assumes input shape [seq_len, batch_size, d_model].
    """

    def __init__(self,
                 *,
                 n_experts: int,
                 expert: FeedForward,
                 d_model: int,
                 capacity_factor: float = 1.0,
                 drop_tokens: bool = True,
                 is_scale_prob: bool = True,
                 aux_loss_coef: float = 1e-2,
                 simbal_coef: float = 1e-3,
                 importance_lambda: float = 0.0,
                 adaptive_k: float = 0.2,
                 max_reassign_per_batch: Optional[int] = None,
                 safe_clone: bool = True):
        super().__init__()

        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        self.is_scale_prob = is_scale_prob

        self.aux_loss_coef = aux_loss_coef          # original load balancing coefficient
        self.simbal_coef = simbal_coef              # new simbal regularizer coefficient
        self.importance_lambda = importance_lambda  # weight for importance biasing of logits
        self.adaptive_k = adaptive_k                # how much slack to give overloaded experts
        self.max_reassign_per_batch = max_reassign_per_batch

        # clone experts (deepcopy fallback)
        if safe_clone:
            self.experts = safe_clone_module_list(expert, n_experts)
        else:
            self.experts = clone_module_list(expert, n_experts)

        # Router: linear map -> softmax over experts
        # Note: weight shape [n_experts, d_model] in PyTorch Linear
        self.switch = nn.Linear(d_model, n_experts, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def _importance_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token importance score.
        Default: L2 norm of token embedding (cheap).
        x: [tokens, d_model]
        returns: [tokens] (float)
        """
        # L2 norm as a simple importance proxy
        return x.norm(p=2, dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        x: [seq_len, batch_size, d_model]
        returns:
          - output: [seq_len, batch_size, d_model]
          - stats: dict with counts, route_prob_sums, n_dropped, n_reassigned, aux_losses
        """
        seq_len, batch_size, d_model = x.shape
        tokens = x.view(-1, d_model)     # [T, d]
        T = tokens.size(0)
        device = tokens.device

        # Router logits
        logits = self.switch(tokens)     # [T, n_experts]

        # ---- Token Importance Aware Routing (TIA-R) ----
        # compute small importance score and bias logits by it (scaled by importance_lambda)
        if self.importance_lambda and self.importance_lambda != 0.0:
            imp = self._importance_score(tokens)  # [T]
            # normalize importance to zero mean / unit std to keep scale reasonable
            imp_norm = (imp - imp.mean()) / (imp.std(unbiased=False) + 1e-6)
            # expand and add to logits (we add to each expert with same score)
            logits = logits + (self.importance_lambda * imp_norm.unsqueeze(-1))

        route_prob = self.softmax(logits)  # [T, E]
        route_prob_max, routes = torch.max(route_prob, dim=-1)  # [T], [T]

        # Token indices per expert (initial assignment)
        indexes_list: List[torch.Tensor] = []
        for i in range(self.n_experts):
            idxs = torch.nonzero(routes == i, as_tuple=False).view(-1)
            indexes_list.append(idxs)

        # Base capacity (paper formula)
        base_capacity = int(self.capacity_factor * (T / self.n_experts))
        # counts (initial)
        counts = torch.tensor([idx.numel() for idx in indexes_list], device=device, dtype=torch.long)

        # Adaptive slack: small extra capacity for overloaded experts
        mean_count = T / float(self.n_experts)
        # delta_i = clamp(k * (counts_i - mean), 0, base_capacity*0.5)
        delta = ((counts.float() - mean_count) * self.adaptive_k).clamp(min=0.0)
        max_delta = base_capacity * 0.5
        delta = delta.clamp(max=max_delta).to(torch.long)
        capacity_per_expert = (torch.tensor(base_capacity, device=device, dtype=torch.long) + delta).tolist()

        # Keep track of dropped and reassigned indices
        dropped_tokens = []
        reassign_pairs = []  # list of (token_idx, old_expert, attempts)

        # First pass: enforce capacities using importance-aware trimming (keep most important)
        # For each expert, if assigned tokens exceed capacity, keep top-k by importance (bias)
        importance_scores = None
        if self.importance_lambda and self.importance_lambda != 0.0:
            importance_scores = self._importance_score(tokens)  # [T]
        else:
            # fallback small jitter so ordering is deterministic
            importance_scores = torch.zeros(T, device=device)

        # We'll collect overflow tokens to try reassigning
        overflow_tokens = []  # list of token indices that overflowed their expert
        for i in range(self.n_experts):
            idxs = indexes_list[i]
            cap = capacity_per_expert[i]
            if idxs.numel() == 0:
                continue
            if idxs.numel() <= cap:
                continue
            # sort indexes by importance (descending), keep top cap
            scores = importance_scores[idxs]
            # get topk indices relative to idxs
            _, order = torch.sort(scores, descending=True)
            kept = idxs[order[:cap]]
            overflow = idxs[order[cap:]]
            indexes_list[i] = kept
            overflow_tokens.append((i, overflow))  # (from_expert, tensor of token idxs)

        # Attempt reassignment for overflow tokens (NTLB)
        n_reassigned = 0
        n_dropped = 0
        # build current counts list (after trimming)
        counts = torch.tensor([idx.numel() for idx in indexes_list], device=device, dtype=torch.long)

        # Compute spare capacities
        spare = []
        for i in range(self.n_experts):
            spare_i = max(0, capacity_per_expert[i] - counts[i].item())
            spare.append(spare_i)

        # Flatten overflow into token-level list with their original experts
        overflow_flat = []
        for from_exp, overflow_idxs in overflow_tokens:
            if overflow_idxs is None or overflow_idxs.numel() == 0:
                continue
            # convert to Python list for iteration (small-scale ok)
            for t in overflow_idxs.tolist():
                overflow_flat.append((t, int(from_exp)))

        # Optionally limit reassignments per batch to avoid huge overhead
        max_reassign = self.max_reassign_per_batch if self.max_reassign_per_batch is not None else len(overflow_flat)
        reassign_count = 0

        # Greedy reassignment: for each overflow token, pick best target expert with spare capacity
        for (t_idx, old_e) in overflow_flat:
            if reassign_count >= max_reassign:
                # stop trying to reassign beyond limit
                dropped_tokens.append(t_idx)
                n_dropped += 1
                continue

            # scores for candidate experts: route_prob[t_idx, j] / (1 + counts[j])
            probs = route_prob[t_idx]  # [E]
            counts_float = counts.float()
            denom = (1.0 + counts_float)
# Detach before converting to NumPy
            candidate_scores = (probs / denom).detach().cpu().numpy()


            # order candidates by descending score but do not choose original old if it has no spare
            best_j = None
            best_score = -1.0
            for j, sc in enumerate(candidate_scores):
                if spare[j] <= 0:
                    continue
                if sc > best_score:
                    best_score = float(sc)
                    best_j = j

            if best_j is not None:
                # assign token t_idx to best_j
                indexes_list[best_j] = torch.cat([indexes_list[best_j], torch.tensor([t_idx], device=device)], dim=0)
                spare[best_j] -= 1
                counts[best_j] += 1
                reassign_count += 1
                n_reassigned += 1
            else:
                # no spare capacity anywhere -> drop or keep as residual (paper dropped; here we optionally drop)
                if self.drop_tokens:
                    dropped_tokens.append(t_idx)
                    n_dropped += 1
                else:
                    # keep it unchanged (no expert processing) - emulate paper's fallback
                    dropped_tokens.append(t_idx)
                    n_dropped += 1

        # Final counts
        counts = torch.tensor([idx.numel() for idx in indexes_list], device=device, dtype=torch.long)

        # Expert forward passes (handle empty index sets)
        final_output = tokens.new_zeros(tokens.shape)  # [T, d]
        for i in range(self.n_experts):
            idxs = indexes_list[i]
            if idxs.numel() == 0:
                # nothing for this expert
                continue
            sub_x = tokens[idxs, :]  # [n_i, d]
            out = self.experts[i](sub_x)  # [n_i, d]
            final_output[idxs, :] = out

        # For dropped tokens (if any), we keep original tokens unchanged (residual)
        if len(dropped_tokens) > 0:
            dropped_tensor = torch.tensor(dropped_tokens, device=device, dtype=torch.long)
            final_output[dropped_tensor, :] = tokens[dropped_tensor, :]

        # Scale outputs by gating probability (top-1 prob)
        if self.is_scale_prob:
            final_output = final_output * route_prob_max.unsqueeze(-1)
        else:
            final_output = final_output * (route_prob_max / (route_prob_max.detach() + 1e-12)).unsqueeze(-1)

        # reshape back
        final_output = final_output.view(seq_len, batch_size, d_model)

        # ---- Auxiliary losses ----
        # Load balancing loss (paper)
        f_i = counts.float() / float(max(1, T))
        P_i = route_prob.mean(dim=0)  # [E]
        load_bal_loss = self.aux_loss_coef * self.n_experts * torch.sum(f_i * P_i)

        # SimBal: similarity-preserving / diversity loss on router weight matrix
        # Encourage W W^T to be close to diagonal (reduce cross-correlation)
        W = self.switch.weight  # shape [E, d]
        G = W @ W.t()           # [E, E]
        diag_G = torch.diag(torch.diag(G))
        off_diag = G - diag_G
        simbal_loss = self.simbal_coef * torch.sum(off_diag * off_diag)

        aux_loss = load_bal_loss + simbal_loss

        stats = {
            "counts": counts.detach().cpu(),
            "route_prob_sums": route_prob.sum(0).detach().cpu(),
            "n_dropped": n_dropped,
            "n_reassigned": n_reassigned,
            "load_bal_loss": load_bal_loss.detach().item(),
            "simbal_loss": simbal_loss.detach().item(),
            "aux_loss": aux_loss.detach().item(),
        }

        return final_output, stats


# ----------------- Minimal Transformer Layer and Model Wrappers -----------------
class SimpleMHAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_head)

    def forward(self, x, mask=None):
        # x: [seq_len, batch, d_model]
        attn_out, _ = self.mha(x, x, x, attn_mask=mask)
        return attn_out


class SwitchTransformerLayer(nn.Module):
    def __init__(self, d_model: int, attn: SimpleMHAttention, feed_forward: SwitchFeedForward, dropout_prob: float = 0.1):
        super().__init__()
        self.size = d_model
        self.attn = attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Pre-norm style
        z = self.norm1(x)
        self_attn = self.attn(z, mask)
        x = x + self.dropout(self_attn)

        z2 = self.norm2(x)
        ff_out, stats = self.feed_forward(z2)
        x = x + self.dropout(ff_out)
        return x, stats


class SwitchTransformer(nn.Module):
    def __init__(self, layer: SwitchTransformerLayer, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([type(layer)(layer.size, layer.attn, layer.feed_forward) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        all_counts = []
        all_route_prob_sums = []
        dropped = 0
        reassigned = 0
        aux_losses = []
        simbal_losses = []
        for layer in self.layers:
            x, stats = layer(x, mask)
            all_counts.append(stats["counts"])
            all_route_prob_sums.append(stats["route_prob_sums"])
            dropped += stats["n_dropped"]
            reassigned += stats["n_reassigned"]
            aux_losses.append(stats["load_bal_loss"])
            simbal_losses.append(stats["simbal_loss"])
        x = self.norm(x)
        out_stats = {
            "counts": all_counts,
            "route_prob_sums": all_route_prob_sums,
            "n_dropped_total": dropped,
            "n_reassigned_total": reassigned,
            "aux_losses": aux_losses,
            "simbal_losses": simbal_losses,
        }
        return x, out_stats


# ----------------- Quick smoke test -----------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cpu")

    # small toy model
    d_model = 64
    d_ff = 256
    n_experts = 4
    seq_len = 8
    batch = 2
    n_layers = 2

    base_expert = FeedForward(d_model, d_ff)
    switch_ff = SwitchFeedForward(
        n_experts=n_experts,
        expert=base_expert,
        d_model=d_model,
        capacity_factor=1.0,
        drop_tokens=True,
        is_scale_prob=True,
        aux_loss_coef=1e-2,
        simbal_coef=1e-3,
        importance_lambda=0.5,   # enable token importance
        adaptive_k=0.2,
        max_reassign_per_batch=100,
        safe_clone=True
    )

    attn = SimpleMHAttention(d_model, n_head=4)
    layer = SwitchTransformerLayer(d_model=d_model, attn=attn, feed_forward=switch_ff, dropout_prob=0.1)
    model = SwitchTransformer(layer, n_layers=n_layers).to(device)

    # random input
    x = torch.randn(seq_len, batch, d_model, device=device)
    out, stats = model(x, mask=None)

    print("Output shape:", out.shape)
    print("Stats summary:")
    print("  n_dropped_total:", stats["n_dropped_total"])
    print("  n_reassigned_total:", stats["n_reassigned_total"])
    print("  counts per layer:")
    for i, c in enumerate(stats["counts"]):
        print(f"    layer {i}: {c.tolist()}")
    print("  aux losses (per layer):", stats["aux_losses"])
    print("  simbal losses (per layer):", stats["simbal_losses"])
