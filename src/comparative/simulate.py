import random
import numpy as np

from typing import List, Tuple

def create_comparison_set(N_cand:int, N_comp:int=None, N_total_comp:int=None):
    """ function selects comparisons to make with the specified properties"""

    # set (num comparisons per item) or (num total comparisons)
    if N_comp: 
        assert (N_comp < N_cand) and (N_comp*N_cand%2==0)
        N_total_comp = N_comp*N_cand/2
    elif N_total_comp: 
        assert N_total_comp <= N_cand*(N_cand-1)/2
        N_comp=N_cand-1
        
    # begin search for valid set of pairs
    pairs = {k:[] for k in range(N_cand)}
    pairs_made = 0
    while pairs_made < N_total_comp:  
        i, j = None, None
        
        possible_i = [x for x in range(N_cand) if len(pairs[x])<N_comp]
        i = random.choice(possible_i)
        possible_j = [x for x in range(N_cand) if (x!=i) and (x not in pairs[i]) and (len(pairs[x])<N_comp)]

        if possible_j == []:
            possible_j = [x for x in range(N_cand) if (x!=i) and (x not in pairs[i])]
            j = random.choice(possible_j)
            i_del = random.choice(pairs[j])
            
            pairs[i_del].remove(j)
            pairs[j].remove(i_del)
            pairs_made -= 1
        else:
            j = random.choice(possible_j)
            
        # if at this stage, we can compare i and j
        pairs[i].append(j)
        pairs[j].append(i)
        pairs_made += 1

    comparisons = set([(i, j) for i in pairs for j in pairs[i] if i<j])
    return sorted(comparisons)

def order_comparisons(comparisons:List[Tuple[int]], ordering:str='random'):   
    """ unfinished """
    if ordering == 'random':
        comparisons = [tuple(random.sample(x, len(x))) for x in comparisons]
    elif ordering == 'symmetric':
        comparisons = comparisons + [(j,i) for (i,j) in comparisons]
    elif ordering == 'balanced':
        N_cand = max([x[1] for x in comparisons])+1
        N_comp = len(comparisons)/N_cand
        outputs = {k:[] for k in range(N_cand)}
        
        for (i, j) in comparisons:
            if random.randint(0,1) == 1:
                x, y = i, j
            else:
                y, x = i, j
                
            if len(outputs[x]) < N_comp:
                outputs[x].append(y)
            elif len(outputs[y]) < N_comp:
                outputs[y].append(x)
            else:
                print(i,j)
                print(outputs)
                raise ValueError('damn man')
        
        comparisons = set([(i, j) for i in outputs for j in pairs[i]])
    return comparisons

def create_comparison_mask(N_cand:int, N_comp:int=None, N_total_comp:int=None, ordering:str='random'):
    comparisons = create_comparison_set(N_cand=N_cand, N_comp=N_comp, N_total_comp=N_total_comp)
    comparisons = order_comparisons(comparisons=comparisons, ordering=ordering)
    
    # create matrix of true labels
    mask_matrix = np.zeros((N_cand, N_cand))
    for (i, j) in comparisons:
        mask_matrix[i,j] += 1
    return mask_matrix

def generate_random_binary_matrix(N_cand:int, N_total_comp:int=0):
    if N_total_comp > N_cand*(N_cand-1):
        raise ValueError("K cannot be greater than the total number of elements (256) in a 16x16 matrix.")

    # Create an empty matrix
    matrix = np.zeros((N_cand, N_cand), dtype=int)

    # Generate K random unique indices excluding the diagonal indices
    indices = np.random.choice(N_cand*(N_cand-1), N_total_comp, replace=False)

    # Assign 1 to the randomly selected indices
    for index in indices:
        row = index // 15
        col = index % 15
        if col >= row:
            col +=1 
        matrix[row, col] = 1

    return matrix

def generate_random_binary_tensor(N_samples:int, N_cand:int, N_total_comp:int=0):
    mask_tensor = np.zeros((N_samples, N_cand, N_cand))
    for i in range(N_samples):
        mask_tensor[i] = generate_random_binary_matrix(N_cand=N_cand, N_total_comp=N_total_comp)
    return mask_tensor

def create_comparison_tensor_mask(N_samples:int, N_cand:int, N_comp:int=None, N_total_comp:int=None, ordering:str='random'):
    mask_tensor = np.zeros((N_samples, N_cand, N_cand))
    for i in range(N_samples):
        mask_tensor[i] = create_comparison_mask(
            N_cand=N_cand, 
            N_comp=N_comp, 
            N_total_comp=N_total_comp,
            ordering=ordering
        )
    return mask_tensor