import pandas as pd
import matplotlib.pyplot as plt

mappo_gnn = pd.read_csv("logs/train_log.csv")
mappo_no_gnn = pd.read_csv("logs/train_log_no_gnn.csv")

plt.figure(figsize=(8, 5))

plt.plot(
    mappo_gnn["episode"],
    mappo_gnn["avg_queue"],
    label="MAPPO + GNN",
    linewidth=2
)

plt.plot(
    mappo_no_gnn["episode"],
    mappo_no_gnn["avg_queue"],
    label="MAPPO no GNN",
    linestyle="--"
)

plt.xlabel("Episode")
plt.ylabel("Average Queue Length")
plt.title("Learning Curves")
plt.legend()
plt.grid(alpha=0.6)

plt.tight_layout()
plt.show()
