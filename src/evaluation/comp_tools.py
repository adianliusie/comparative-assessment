import numpy as np
import os

from typing import Tuple
from scipy.linalg import solve

#== Loading Comparative Matrices =======================================================#
def comparatisons_to_tensors(comparisons)->Tuple[np.ndarray, np.ndarray]:
    num_examples = max([int(i.split('-')[0]) for i in comparisons.keys()])+1
    num_cands    = max([int(i.split('-')[1]) for i in comparisons.keys()])+1
    C_tensor = np.zeros((num_examples, num_cands, num_cands))
    M_tensor = np.zeros((num_examples, num_cands, num_cands))
    for k, v in comparisons.items():
        n, i1, i2 = [int(i) for i in k.split('-')]
        if v == -1:
            continue

        if v == 0:
            C_tensor[n, i1, i2] += 1
        elif v == 1:
            C_tensor[n, i2, i1] += 1

        M_tensor[n, i1, i2] += 1
        M_tensor[n, i2, i1] += 1
    return C_tensor, M_tensor

def matrices_to_scores(C_tensor:np.ndarray, M_tensor:np.ndarray):
    output_ranks = []
    for C, M in zip(C_tensor, M_tensor):
        ranks = win_ratio(C, M)
        output_ranks.append(ranks)
    return output_ranks

#== Scoring Methods of going from comparison matrix to ranks ===========================#
def win_ratio(C, M):  
    C = C*M  
    M = M + M.T
    x = np.sum(C, axis=1)/np.sum(M, axis=1)
    return x

def colley_scoring(C, M):    
    C = C*M
    M = M + M.T
    n = C.shape[0]
    B = 2 * np.eye(n) + np.diag(np.sum(M, axis=0)) - M
    b = 1 + 0.5*(np.sum(C, axis=1)-np.sum(C, axis=0)) 
    x = solve(B, b)
    return x

#== Backwards methods of going from ranks to matrix ===========================#
def true_scores_to_matrix(true_scores):
    N_cand = len(true_scores)
    label_matrix = np.zeros((N_cand, N_cand))
    for i in range(N_cand):
        for j in range(N_cand):
            if true_scores[i] > true_scores[j]:
                label_matrix[i][j] += 1
    return label_matrix

