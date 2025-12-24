import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, embed_dim, n_agents):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim * n_agents, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, embeddings):
        """
        embeddings : (N, embed_dim)
        """
        x = embeddings.view(1, -1)  # concat global
        value = self.net(x)
        return value
