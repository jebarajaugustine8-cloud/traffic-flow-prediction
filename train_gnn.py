"""
train_gnn.py
------------
GRAPH NEURAL NETWORK MODULE — Smart Traffic Flow Prediction using GNNs

Roads naturally form a graph, so this module models Chennai as one:

    NODES  = the 16 zones (Vadapalani, T. Nagar, Guindy, ...)
    EDGES  = road connectivity between zones (connected if within
             6.5 km of each other — a distance-based road graph)

A 2-layer Graph Convolutional Network (GCN, Kipf & Welling 2017) is
implemented from scratch in pure PyTorch so every line is explainable:

    H1 = ReLU( Â · X  · W1 )        Â = D^-1/2 (A + I) D^-1/2
    H2 = ReLU( Â · H1 · W2 )        (normalized adjacency with self-loops)
    y  = Linear(H2)                  one traffic prediction PER NODE

Unlike Gradient Boosting — which predicts one zone at a time — the GCN
predicts ALL 16 zones simultaneously, letting each zone "see" its
neighbours through message passing. That is why GNNs are state-of-the-art
for traffic forecasting.

Outputs:
    model/metrics_gnn.json         (results shown on the website)
    static/graphs/zone_graph.png   (the Chennai road graph visualised)
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from locations import CHENNAI_ZONES

torch.manual_seed(42)
np.random.seed(42)

N_ZONES = len(CHENNAI_ZONES)
CONNECT_KM = 6.5   # zones within this road distance are connected


# ------------------------------------------------------------------
# 1. Build the Chennai road graph
# ------------------------------------------------------------------

def haversine(a, b):
    R = 6371
    p1, p2 = np.radians(a["lat"]), np.radians(b["lat"])
    dphi = np.radians(b["lat"] - a["lat"])
    dlmb = np.radians(b["lng"] - a["lng"])
    x = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(x))


def build_adjacency():
    A = np.zeros((N_ZONES, N_ZONES))
    for i in range(N_ZONES):
        for j in range(i + 1, N_ZONES):
            if haversine(CHENNAI_ZONES[i], CHENNAI_ZONES[j]) <= CONNECT_KM:
                A[i, j] = A[j, i] = 1.0
    # every zone must reach the network: connect isolated zones to
    # their nearest neighbour (e.g. Sholinganallur via OMR)
    for i in range(N_ZONES):
        if A[i].sum() == 0:
            dists = [haversine(CHENNAI_ZONES[i], CHENNAI_ZONES[j]) if j != i else 1e9
                     for j in range(N_ZONES)]
            j = int(np.argmin(dists))
            A[i, j] = A[j, i] = 1.0
    # GCN normalisation: Â = D^-1/2 (A + I) D^-1/2
    A_hat = A + np.eye(N_ZONES)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(A_hat.sum(axis=1)))
    return A, torch.tensor(D_inv_sqrt @ A_hat @ D_inv_sqrt, dtype=torch.float32)


def plot_graph(A):
    plt.rcParams.update({
        "figure.facecolor": "#05060f", "axes.facecolor": "#05060f",
        "text.color": "#eef1ff", "font.size": 9,
    })
    plt.figure(figsize=(9, 8))
    for i in range(N_ZONES):
        for j in range(i + 1, N_ZONES):
            if A[i, j]:
                plt.plot([CHENNAI_ZONES[i]["lng"], CHENNAI_ZONES[j]["lng"]],
                         [CHENNAI_ZONES[i]["lat"], CHENNAI_ZONES[j]["lat"]],
                         color="#00e5ff", alpha=.35, lw=1.2, zorder=1)
    for z in CHENNAI_ZONES:
        plt.scatter(z["lng"], z["lat"], s=z["busyness"] * 160,
                    color="#ffb020", edgecolors="#ff2d95", zorder=2)
        plt.annotate(z["name"].split(" (")[0], (z["lng"], z["lat"]),
                     textcoords="offset points", xytext=(6, 6), color="#eef1ff")
    plt.title("Chennai Zone Road Graph — nodes sized by busyness",
              color="#eef1ff", fontsize=13, pad=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("static/graphs/zone_graph.png", dpi=110)
    plt.close()


# ------------------------------------------------------------------
# 2. Build graph samples from the dataset
#    Each sample = one (hour, day, weather, holiday) situation with
#    a 16-node feature matrix and 16 traffic targets.
# ------------------------------------------------------------------

def build_samples():
    df = pd.read_csv("dataset/traffic.csv")
    # average duplicates (weather can shift mid-day in the generator)
    g = df.groupby(["Hour", "Day", "Weather", "Holiday", "Zone"])["Traffic"].mean().reset_index()
    pivot = g.pivot_table(index=["Hour", "Day", "Weather", "Holiday"],
                          columns="Zone", values="Traffic").dropna()

    keys = pivot.index.to_frame(index=False)
    X_time = np.column_stack([
        np.sin(2 * np.pi * keys["Hour"] / 24),
        np.cos(2 * np.pi * keys["Hour"] / 24),
        keys["Day"] / 7.0,
        (keys["Weather"] == 0).astype(float),
        (keys["Weather"] == 1).astype(float),
        (keys["Weather"] == 2).astype(float),
        keys["Holiday"].astype(float),
        (keys["Day"] >= 6).astype(float),
        keys["Hour"].isin([8, 9, 17, 18, 19]).astype(float),
    ])                                          # (S, 9) situation features
    Y = pivot.values.astype(np.float32)         # (S, 16) traffic per zone
    return (torch.tensor(X_time, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32))


# ------------------------------------------------------------------
# 3. The GCN, from scratch
# ------------------------------------------------------------------

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, A_hat):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.A_hat = A_hat                      # (N, N) normalised adjacency

    def forward(self, X):                       # X: (batch, N, in_dim)
        return self.A_hat @ self.W(X)           # message passing


class TrafficGCN(nn.Module):
    def __init__(self, A_hat, time_dim=9, emb_dim=8, hidden=64):
        super().__init__()
        self.node_emb = nn.Embedding(N_ZONES, emb_dim)   # learned zone identity
        self.gcn1 = GCNLayer(time_dim + emb_dim, hidden, A_hat)
        self.gcn2 = GCNLayer(hidden, hidden, A_hat)
        self.head = nn.Linear(hidden, 1)
        self.act = nn.ReLU()

    def forward(self, x_time):                  # x_time: (batch, time_dim)
        B = x_time.shape[0]
        node_ids = torch.arange(N_ZONES)
        emb = self.node_emb(node_ids).unsqueeze(0).expand(B, -1, -1)   # (B,N,emb)
        time = x_time.unsqueeze(1).expand(-1, N_ZONES, -1)             # (B,N,time)
        h = torch.cat([time, emb], dim=-1)
        h = self.act(self.gcn1(h))
        h = self.act(self.gcn2(h))
        return self.head(h).squeeze(-1)          # (B, N) traffic per zone


# ------------------------------------------------------------------
# 4. Train & evaluate
# ------------------------------------------------------------------

def main():
    A, A_hat = build_adjacency()
    plot_graph(A)
    print(f"Chennai zone graph: {N_ZONES} nodes, {int(A.sum() / 2)} edges")

    X, Y = build_samples()
    print(f"Graph samples: {X.shape[0]} situations × {N_ZONES} zones")

    # normalise targets for stable training
    y_mean, y_std = Y.mean(), Y.std()
    Yn = (Y - y_mean) / y_std

    idx = torch.randperm(X.shape[0])
    split = int(0.8 * len(idx))
    tr, te = idx[:split], idx[split:]

    model = TrafficGCN(A_hat)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()

    for epoch in range(400):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X[tr]), Yn[tr])
        loss.backward()
        opt.step()
        if (epoch + 1) % 100 == 0:
            print(f"  epoch {epoch+1:>3}  train MSE {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(X[te]) * y_std + y_mean

    y_true = Y[te].numpy().ravel()
    y_pred = pred.numpy().ravel()
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    mae = np.abs(y_true - y_pred).mean()

    print(f"\nGCN test R²: {r2*100:.2f}%   MAE: {mae:.2f} vehicles/hour")

    metrics = {
        "model": "Graph Convolutional Network (2-layer GCN, from scratch in PyTorch)",
        "graph": {"nodes": N_ZONES, "edges": int(A.sum() / 2),
                  "rule": f"zones within {CONNECT_KM} km are connected"},
        "r2": round(float(r2) * 100, 2),
        "mae": round(float(mae), 2),
        "params": int(sum(p.numel() for p in model.parameters())),
        "note": ("Predicts all 16 zones simultaneously via message passing; "
                 "each zone's prediction is informed by its road neighbours."),
    }
    with open("model/metrics_gnn.json", "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save(model.state_dict(), "model/traffic_gcn.pt")
    print("✅ Saved model/metrics_gnn.json, model/traffic_gcn.pt, static/graphs/zone_graph.png")


if __name__ == "__main__":
    main()
