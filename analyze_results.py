import pandas as pd
import numpy as np

# === Charger les logs RL ===
mappo_gnn = pd.read_csv("logs/train_log.csv")
mappo_no_gnn = pd.read_csv("logs/train_log_no_gnn.csv")

# === Calculs ===
def summarize(df, name):
    avg = df["avg_queue"].mean()
    std = df["avg_queue"].std()
    best = df["avg_queue"].min()
    print(f"{name}:")
    print(f"  Mean AvgQueue = {avg:.2f}")
    print(f"  Std  AvgQueue = {std:.2f}")
    print(f"  Best AvgQueue = {best:.2f}\n")
    return avg, std

mappo_gnn_mean, mappo_gnn_std = summarize(mappo_gnn, "MAPPO + GNN")
mappo_no_gnn_mean, mappo_no_gnn_std = summarize(mappo_no_gnn, "MAPPO no GNN")

# === Baselines ===
fixed_avg = 89.16
actuated_avg = 148.82

print("Baselines:")
print(f"  FIXED     AvgQueue = {fixed_avg:.2f}")
print(f"  ACTUATED  AvgQueue = {actuated_avg:.2f}")
