import torch
import torch.nn as nn


class LightGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        self.activation = nn.ReLU()

    def forward(self, x, adj):
        """
        x   : (N, in_dim)    node features
        adj : (N, N)         normalized adjacency matrix
        """

        # Layer 1
        h = torch.matmul(adj, x)
        h = self.activation(self.fc1(h))

        # Layer 2
        h = torch.matmul(adj, h)
        h = self.fc2(h)

        return h
