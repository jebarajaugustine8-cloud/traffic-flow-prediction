"""
graphs.py
---------
Generates the analytics dashboard charts into static/graphs/:

  1. hourly_traffic.png   - weekday vs weekend hourly pattern
  2. weather.png          - average traffic by weather condition
  3. heatmap.png          - Day x Hour traffic heatmap
  4. importance.png       - which features the model relies on most
"""

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "#101418",
    "axes.facecolor": "#101418",
    "axes.edgecolor": "#3a4450",
    "axes.labelcolor": "#e8edf2",
    "text.color": "#e8edf2",
    "xtick.color": "#aab4bf",
    "ytick.color": "#aab4bf",
    "grid.color": "#242c34",
    "font.size": 11,
})

GREEN, AMBER, RED, BLUE = "#37d67a", "#ffb020", "#ff5a5a", "#4da3ff"

df = pd.read_csv("dataset/traffic.csv")

# ------------------------------------------------------------------
# 1. Hourly traffic: weekday vs weekend
# ------------------------------------------------------------------
weekday = df[df["Day"] <= 5].groupby("Hour")["Traffic"].mean()
weekend = df[df["Day"] >= 6].groupby("Hour")["Traffic"].mean()

plt.figure(figsize=(9, 5))
plt.plot(weekday.index, weekday.values, marker="o", color=AMBER, label="Weekday")
plt.plot(weekend.index, weekend.values, marker="s", color=BLUE, label="Weekend")
plt.fill_between(weekday.index, weekday.values, alpha=0.12, color=AMBER)
plt.title("Average Traffic by Hour — Weekday vs Weekend", fontsize=13, pad=12)
plt.xlabel("Hour of Day")
plt.ylabel("Vehicles / Hour")
plt.xticks(range(0, 24, 2))
plt.grid(True, alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("static/graphs/hourly_traffic.png", dpi=110)
plt.close()

# ------------------------------------------------------------------
# 2. Weather analysis (bar chart is clearer than a pie here)
# ------------------------------------------------------------------
weather = df.groupby("Weather")["Traffic"].mean()
labels = ["Clear", "Rain", "Fog"]

plt.figure(figsize=(7, 5))
bars = plt.bar(labels, weather.values, color=[GREEN, BLUE, "#9aa5b1"], width=0.55)
for b in bars:
    plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 6,
             f"{b.get_height():.0f}", ha="center", fontsize=11, color="#e8edf2")
plt.title("Average Traffic by Weather Condition", fontsize=13, pad=12)
plt.ylabel("Vehicles / Hour")
plt.grid(True, axis="y", alpha=0.5)
plt.tight_layout()
plt.savefig("static/graphs/weather.png", dpi=110)
plt.close()

# ------------------------------------------------------------------
# 3. Day x Hour heatmap
# ------------------------------------------------------------------
pivot = df.pivot_table(index="Day", columns="Hour", values="Traffic", aggfunc="mean")

plt.figure(figsize=(11, 4.5))
im = plt.imshow(pivot.values, aspect="auto", cmap="inferno")
plt.colorbar(im, label="Vehicles / Hour")
plt.yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
plt.xticks(range(0, 24, 2))
plt.xlabel("Hour of Day")
plt.title("Traffic Heatmap — Day × Hour", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig("static/graphs/heatmap.png", dpi=110)
plt.close()

# ------------------------------------------------------------------
# 4. Feature importance (from metrics.json written by train_model.py)
# ------------------------------------------------------------------
try:
    with open("model/metrics.json") as f:
        metrics = json.load(f)
    imp = metrics.get("feature_importance", {})
    if imp:
        names = list(imp.keys())
        vals = list(imp.values())
        order = np.argsort(vals)
        names = [names[i] for i in order]
        vals = [vals[i] for i in order]

        plt.figure(figsize=(8, 5))
        plt.barh(names, vals, color=AMBER)
        plt.title(f"Feature Importance — {metrics['best_model']}", fontsize=13, pad=12)
        plt.xlabel("Importance")
        plt.grid(True, axis="x", alpha=0.5)
        plt.tight_layout()
        plt.savefig("static/graphs/importance.png", dpi=110)
        plt.close()
except FileNotFoundError:
    print("⚠ model/metrics.json not found — run train_model.py first for the importance chart")

print("✅ Graphs created in static/graphs/")
