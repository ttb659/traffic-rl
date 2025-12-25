import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Charger les logs RL ===
mappo_gnn = pd.read_csv("logs/train_log.csv")
mappo_no_gnn = pd.read_csv("logs/train_log_no_gnn.csv")

# === Moyennes ===
methods = [
    "Fixed TLS",
    "Actuated TLS",
    "MAPPO\n(no GNN)",
    "MAPPO\n+ GNN"
]

values = [
    89.16,
    148.82,
    mappo_no_gnn["avg_queue"].mean(),
    mappo_gnn["avg_queue"].mean()
]

errors = [
    0,
    0,
    mappo_no_gnn["avg_queue"].std(),
    mappo_gnn["avg_queue"].std()
]

# === Plot ===
plt.figure(figsize=(8, 5))
bars = plt.bar(methods, values, yerr=errors, capsize=6)

plt.ylabel("Average Queue Length")
plt.title("Comparison of Traffic Signal Control Methods")

plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("Comparison_of_Traffic_Signal_Control_Methods.png", dpi=300, bbox_inches='tight')
plt.show()
