import torch
import torch.nn as nn
import torch.distributions as dist


class Actor(nn.Module):
    def __init__(self, embed_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, embeddings):
        """
        embeddings : (N, embed_dim)
        """
        logits = self.net(embeddings)
        return dist.Categorical(logits=logits)
