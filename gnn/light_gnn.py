import torch
import torch.nn as nn


class LightGNN(nn.Module): # Une architecture GNN légère avec deux couches pour la comparaison avec un GNN classique
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

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class LightGNN(nn.Module):
    def __init__(self, num_nodes, embedding_dim, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        # Les seuls paramètres appris sont les embeddings
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        
        # Poids d'importance appris pour l'élagage (Spécifique à LightGNN)
        self.edge_weights = nn.Parameter(torch.ones(1)) # Simplifié ici

    def forward(self, adj):
        
        #adj : Matrice d'adjacence normalisée
        
        e_0 = self.embedding.weight
        all_layers = [e_0]
        h = e_0

        for i in range(self.num_layers):
            # Propagation linéaire (sans Linear ni ReLU)
            # Dans un vrai LightGNN, on appliquerait ici un masque d'élagage
            h = torch.matmul(adj, h)
            all_layers.append(h)

        # Combinaison finale de toutes les couches (évite l'over-smoothing)
        final_embeddings = torch.mean(torch.stack(all_layers, dim=1), dim=1)
        return final_embeddings
"""