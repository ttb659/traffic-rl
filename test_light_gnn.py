import torch
import numpy as np

from gnn.light_gnn import LightGNN
from graph.adjacency import build_adjacency_matrix
from graph.normalize import normalize_adjacency

# Paramètres
N_AGENTS = 9
OBS_DIM = 6
EMB_DIM = 32

# Faux états (comme sortis de l'env)
x = torch.randn(N_AGENTS, OBS_DIM)

# Graphe
A = build_adjacency_matrix()
A_norm = normalize_adjacency(A)
A_norm = torch.tensor(A_norm, dtype=torch.float32)

# Modèle
gnn = LightGNN(OBS_DIM, 32, EMB_DIM)

# Forward
embeddings = gnn(x, A_norm)

print("Shape embeddings :", embeddings.shape)
