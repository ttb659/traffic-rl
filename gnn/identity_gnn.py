import torch
import torch.nn as nn

class IdentityGNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, adj=None):
        # Pas de message passing
        return x
