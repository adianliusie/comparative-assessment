from functools import lru_cache
import itertools 

import numpy as np
import random

#== Base utility methods ==========================================================================#
def ranks_to_comparisons(ranks):
    # Convert ranks to a NumPy array (assuming ranks is a list or 1D array)
    ranks = np.array(ranks)

    # Use broadcasting to create the comparison matrix efficiently
    C = ranks > ranks[:, None]
    return C.astype(int)

def _default_mask(C):
    M = np.ones_like(C)
    np.fill_diagonal(M, 0)
    return M

#== Loss functions for ranks and comparison matrices ==============================================#
def consistent_loss_fn(C, ranks, M=None):
    if M == None: M = _default_mask(C)
    C_r = ranks_to_comparisons(ranks)
    loss = np.sum(np.abs(C - C_r)*M)
    return loss

def corpus_prob_loss_fn(C, ranks, M=None, A_A=0.8):
    if M == None: M = _default_mask(C)
    C_r = ranks_to_comparisons(ranks)

    # calculate priors, P_A refers to P(\hat{A})
    P_A = np.sum(C)/np.sum(M)
    P_B = 1 - P_A

    # A_B refers to P(\hat{A}|B) etc.
    A_A = 0.8
    B_A = 1 - A_A
    A_B = (2*P_A) - A_A
    B_B = 1 - A_B

    # Note that C is now a NxN tensor of log probs, s.t. C_ij = P(y_i > y_j)
    probs = A_A * (C*C_r) + A_B * (C*(1-C_r)) + B_A * ((1-C)*C_r) + B_B * (1-C)*(1-C_r)
    print("check ordering convention of above line")
    return np.sum(probs*M)

def prob_loss_fn(C, ranks, M=None):
    if M == None: M = _default_mask(C)
    C_r = ranks_to_comparisons(ranks)

    # Note that C is now a NxN tensor of log probs, s.t. C_ij = P(y_i > y_j)
    probs = C_r * C + (1-C_r) * (1-C)
    #print("check ordering convention of above line")
    return np.sum(probs*M)

#== Search methods for finding ranks ==============================================================#
def brute_force(loss_fn, C, ranks, M=None, maxsize=10000):
    """first optimization stage of tring last of random ordering"""
    best_loss = loss_fn(C=C, ranks=new_ranks, M=M)
    perms = list(itertools.permutations(ranks))
    random.shuffle(perms)
    for k, new_ranks in enumerate(perms):
        new_ranks = list(new_ranks)

        if k > maxsize: break
        
        loss = loss_fn(C=C, ranks=new_ranks, M=M)
        if loss < best_loss:
            best_loss = loss
            ranks = new_ranks.copy()
    return ranks

def rand_search(loss_fn, C, ranks, M=None, maxsize=10000):
    """first optimization stage of tring last of random ordering"""
    new_ranks = ranks.copy()
    best_loss = loss_fn(C=C, ranks=new_ranks, M=M)

    k = 0
    while k < maxsize:
        random.shuffle(new_ranks)
        loss = loss_fn(C=C, ranks=new_ranks, M=M)

        if loss < best_loss:
            best_loss = loss
            ranks = new_ranks.copy()
        k += 1
    return ranks

def greedy(loss_fn, C, ranks):
    """first optimization stage of tring last of random ordering"""
    best_loss = loss_fn(C, ranks)
    new_ranks = ranks.copy()
    
    k, N = 0, len(ranks)
    while k < (N*N):
        for i, j in np.ndindex((N,N)):
            if i == j: continue

            new_ranks = ranks.copy()
            new_ranks[i], new_ranks[j] = new_ranks[j], new_ranks[i]
            loss = loss_fn(C, new_ranks)

            if loss < best_loss:
                best_loss = loss
                ranks = new_ranks.copy()
                k = 0
            else:
                k += 1
    return ranks

#== Current Recipe ================================================================================#
def find_optimal_ranks(C, loss='consistent'):
    if   loss == 'consistent': loss_fn=consistent_loss_fn
    elif loss == 'probs': loss_fn=prob_loss_fn

    N = C.shape[0]
    ranks = list(range(N))
    ranks = rand_search(loss_fn, C, ranks, maxsize=100_000)
    ranks = greedy(loss_fn, C, ranks)
    return ranks
