# train_baseline_only.py
"""
Train baseline Switch Transformer (paper-style) on Tiny Shakespeare.
This version includes:
 - loss curve
 - drop rate curve
 - expert distribution tracking
 - plots and logs for paper comparison
"""

import os
import csv
import math
import random
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split

# Import baseline implementation
from switch_moe_innovations import (
    FeedForward,
    SimpleMHAttention as SimpleMultiHeadAttention,  # since name differs slightly
    SwitchFeedForward,
    SwitchTransformerLayer,
    SwitchTransformer
)


# -----------------------------
# Seeding
# -----------------------------
SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------
# Dataset
# -----------------------------
class CharDataset(Dataset):
    def __init__(self, text, seq_len):
        self.seq_len = seq_len
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y

def load_tiny_shakespeare(seq_len=64, batch_size=32, val_frac=0.05):
    file_path = "tiny_shakespeare.txt"
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    if not os.path.exists(file_path):
        import requests
        print("Downloading Tiny Shakespeare...", flush=True)
        r = requests.get(url)
        r.raise_for_status()
        open(file_path, "w", encoding="utf-8").write(r.text)

    text = open(file_path, "r", encoding="utf-8").read()
    dataset = CharDataset(text, seq_len)
    n = len(dataset)
    n_val = max(1, int(val_frac * n))
    n_train = n - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)
    return dataset, train_loader, val_loader

# -----------------------------
# Model
# -----------------------------
def build_baseline_model(vocab_size, seq_len, d_model=64, d_ff=256, n_heads=4, n_experts=4, n_layers=2, capacity_factor=1.25):
    base_expert = FeedForward(d_model, d_ff, dropout=0.1)
    switch_ff = SwitchFeedForward(
        capacity_factor=capacity_factor,
        drop_tokens=True,
        is_scale_prob=True,
        n_experts=n_experts,
        expert=base_expert,
        d_model=d_model
    )
    # ✅ Corrected here
    attn = SimpleMultiHeadAttention(d_model=d_model, n_head=n_heads)
    layer = SwitchTransformerLayer(d_model=d_model, attn=attn, feed_forward=switch_ff, dropout_prob=0.1)
    model = SwitchTransformer(layer, n_layers=n_layers)
    model.embedding = nn.Embedding(vocab_size, d_model)
    model.fc_out = nn.Linear(d_model, vocab_size)
    return model


# -----------------------------
# Helper: parse baseline forward
# -----------------------------
# -----------------------------
# 3. Helpers for parsing baseline forward outputs (✅ FIXED)
# -----------------------------
def parse_baseline_forward(ret):
    """
    The baseline forward returns:
      (out, counts_stack, probs_stack, n_dropped, route_max_stack)
    where:
      out: [seq_len, batch_size, d_model]
      counts_stack: [n_layers, n_experts] (tensor)
      probs_stack: [n_layers, n_experts] (tensor)
      n_dropped: int or list per layer
      route_max_stack: [n_layers, T] (top-1 probs)
    We'll normalize these into a stats dict.
    """
    stats = {
        "counts": None,               # tensor [n_layers, n_experts]
        "route_prob_sums": None,      # tensor [n_layers, n_experts]
        "n_dropped_per_layer": None,  # list
        "n_dropped_total": 0,
        "route_max": None             # tensor
    }

    if not isinstance(ret, tuple):
        # unexpected; return minimal
        return ret, stats

    out = ret[0]

    # counts
    if len(ret) >= 2:
        counts = ret[1]
        if isinstance(counts, torch.Tensor):
            stats["counts"] = counts.detach().cpu()

    # route prob sums
    if len(ret) >= 3:
        probs = ret[2]
        if isinstance(probs, torch.Tensor):
            stats["route_prob_sums"] = probs.detach().cpu()

    # n_dropped ✅ FIXED HERE
    if len(ret) >= 4:
        n_dropped = ret[3]
        if isinstance(n_dropped, torch.Tensor):
            try:
                stats["n_dropped_total"] = int(n_dropped.sum().item())
                stats["n_dropped_per_layer"] = n_dropped.tolist()
            except Exception:
                stats["n_dropped_total"] = int(n_dropped.item())
                stats["n_dropped_per_layer"] = [int(n_dropped.item())]
        elif isinstance(n_dropped, list):
            stats["n_dropped_total"] = int(sum(n_dropped))
            stats["n_dropped_per_layer"] = [int(x) for x in n_dropped]
        else:
            try:
                stats["n_dropped_total"] = int(n_dropped)
                stats["n_dropped_per_layer"] = [int(n_dropped)]
            except Exception:
                stats["n_dropped_total"] = 0
                stats["n_dropped_per_layer"] = []

    # route_max
    if len(ret) >= 5 and isinstance(ret[4], torch.Tensor):
        stats["route_max"] = ret[4].detach().cpu()

    return out, stats


# -----------------------------
# Training loop with metrics
# -----------------------------
def train_baseline(epochs=5, seq_len=64, batch_size=32):
    set_seed()
    dataset, train_loader, val_loader = load_tiny_shakespeare(seq_len, batch_size)

    model = build_baseline_model(dataset.vocab_size, seq_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Device:", device, flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Metrics for plots
    train_losses = []
    val_losses = []
    drop_rates = []
    expert_dist_start = None
    expert_dist_end = None

    total_tokens_all = 0
    total_dropped_all = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_tokens = 0
        total_dropped = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            x_t = x.transpose(0, 1)
            emb = model.embedding(x_t)
            forward_ret = model(emb)
            out, stats = parse_baseline_forward(forward_ret)

            logits = model.fc_out(out)
            loss = criterion(logits.view(-1, dataset.vocab_size), y.transpose(0, 1).reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_tokens += x.numel()
            total_dropped += stats["n_dropped_total"]

            # capture expert distribution
            if epoch == 1 and batch_idx == 0 and stats["counts"] is not None:
                expert_dist_start = stats["counts"].sum(dim=0).numpy()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        drop_rates.append(total_dropped / total_tokens)
        total_tokens_all += total_tokens
        total_dropped_all += total_dropped

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                x_t = x_val.transpose(0, 1)
                emb = model.embedding(x_t)
                forward_ret = model(emb)
                out, _ = parse_baseline_forward(forward_ret)
                logits = model.fc_out(out)
                loss = criterion(logits.view(-1, dataset.vocab_size), y_val.transpose(0, 1).reshape(-1))
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

        print(f"[Epoch {epoch}] Train loss={avg_train_loss:.4f}, Val loss={val_losses[-1]:.4f}, Drop rate={drop_rates[-1]:.4f}")

    # After training, get expert distribution at end
    expert_dist_end = stats["counts"].sum(dim=0).numpy() if stats["counts"] is not None else None

    # -----------------------------
    # PLOTS
    # -----------------------------
    epochs_axis = np.arange(1, epochs + 1)

    plt.figure(figsize=(8,5))
    plt.plot(epochs_axis, train_losses, label="Train Loss")
    plt.plot(epochs_axis, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve_baseline.png")
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(epochs_axis, drop_rates, color="red", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Drop Rate")
    plt.title("Dropped Token Rate per Epoch")
    plt.grid(True)
    plt.savefig("drop_rate_curve.png")
    plt.show()

    if expert_dist_start is not None and expert_dist_end is not None:
        experts = np.arange(len(expert_dist_start))
        plt.figure(figsize=(8,5))
        plt.bar(experts, expert_dist_start)
        plt.title("Expert Token Distribution - Start")
        plt.xlabel("Expert ID")
        plt.ylabel("Token Count")
        plt.savefig("expert_dist_start.png")
        plt.show()

        plt.figure(figsize=(8,5))
        plt.bar(experts, expert_dist_end)
        plt.title("Expert Token Distribution - End")
        plt.xlabel("Expert ID")
        plt.ylabel("Token Count")
        plt.savefig("expert_dist_end.png")
        plt.show()

    print(f"\nTraining finished. Total drop rate across training: {total_dropped_all/total_tokens_all:.4f}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    train_baseline(epochs=5)