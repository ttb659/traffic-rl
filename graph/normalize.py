import numpy as np

def normalize_adjacency(A):
    I = np.eye(A.shape[0])
    A_hat = A + I
    D = np.sum(A_hat, axis=1)
    D_inv = np.diag(1.0 / D)
    return D_inv @ A_hat
