import csv
import os

class Logger:
    def __init__(self, log_dir="logs", filename="train_log.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, filename)

        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode",
                "total_reward",
                "avg_reward_per_step",
                "actor_loss",
                "critic_loss",
                "entropy",
                "avg_queue"
            ])

    def log(self, data):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(data)
