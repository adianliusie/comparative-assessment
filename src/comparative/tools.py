import numpy as np
import os
import random

from typing import Tuple
from scipy.linalg import solve

#== Formatting methods ============================================================================#
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

    return C_tensor, M_tensor

#== Analytical Scoring Methods of going from comparison matrix to ranks ===========================#
def win_ratio(C_tensor, M_tensor):  
    wins = (M_tensor*C_tensor).sum(axis=-1) + (M_tensor*(1-C_tensor)).sum(axis=-2)
    games = M_tensor.sum(axis=-1) + M_tensor.sum(axis=-2)
    win_ratio = wins/games
    return win_ratio

def colley_scoring(C, M):    
    C = C*M
    M = M + M.T
    n = C.shape[0]
    B = 2 * np.eye(n) + np.diag(np.sum(M, axis=0)) - M
    b = 1 + 0.5*(np.sum(C, axis=1)-np.sum(C, axis=0)) 
    x = solve(B, b)
    return x

def matrices_to_scores(C_tensor:np.ndarray, M_tensor:np.ndarray):
    output_ranks = []
    for C, M in zip(C_tensor, M_tensor):
        ranks = win_ratio(C, M)
        output_ranks.append(ranks)
    return output_ranks

#== Backwards methods of going from ranks to matrix ===============================================#
def true_scores_to_matrix(true_scores):
    N_cand = len(true_scores)
    label_matrix = np.zeros((N_cand, N_cand))
    for i in range(N_cand):
        for j in range(N_cand):
            if true_scores[i] > true_scores[j]:
                label_matrix[i][j] += 1
    return label_matrix

#== base mask creation methods ====================================================================#
def generate_random_mask(N:int, num_comp:int=0):
    assert num_comp <= N*(N-1), "number of comparisons cannot be greater than Nx(N-1)"

    # ranodmly select positions from possible matrix
    possible_indices = [(i, j) for i, j in np.ndindex(N,N) if i!=j]
    rand_indices = np.random.choice(range(len(possible_indices)), num_comp, replace=False)
    selected = [possible_indices[i] for i in rand_indices]
    
    # Assign 1 to the randomly selected indices
    M = np.zeros((N, N), dtype=int)
    for i, j in selected:
        M[i, j] = 1

    return M

def generate_no_repeat_mask(N:int, num_comp:int=0):
    assert num_comp <= N*(N-1)/2, "number of comparisons cannot be greater than Nx(N-1)/2"

    # ranodmly select positions from possible matrix
    possible_indices = [(i, j) for i, j in np.ndindex(N,N) if i<j]
    rand_indices = np.random.choice(range(len(possible_indices)), num_comp, replace=False)
    selected = [possible_indices[i] for i in rand_indices]
    
    # Assign 1 to the selected indices, and randomize order
    M = np.zeros((N, N), dtype=int)
    for i, j in selected:
        if random.randrange(2):
            M[i, j] = 1
        else:
            M[j, i] = 1

    return M

def generate_symmetric_mask(N:int, num_comp:int=0):
    assert num_comp%2 == 0, "number of comparisons must be even"
    M = generate_no_repeat_mask(N, int(num_comp/2))
    M = M + M.T
    return M

def generate_mask_tensor(N:int, num_comp:int=0, num_contexts:int=1, mode='random'):
    if   mode == 'random':    mask_fn = generate_random_mask
    elif mode == 'no-repeat': mask_fn = generate_no_repeat_mask
    elif mode == 'symmetric': mask_fn = generate_symmetric_mask

    M_tensor = np.zeros((num_contexts, N, N))
    for i in range(num_contexts):
        M_tensor[i] = mask_fn(N=N, num_comp=num_comp)
    return M_tensor

