import numpy as np
from graph.mapping import IDX_TO_POS

def build_adjacency_matrix():
    adj = np.zeros((9, 9), dtype=np.float32)

    for i, (r, c) in IDX_TO_POS.items():
        for j, (rr, cc) in IDX_TO_POS.items():
            if abs(r - rr) + abs(c - cc) == 1:
                adj[i, j] = 1.0

    return adj

#from adjacency import build_adjacency_matrix
#print(build_adjacency_matrix())
