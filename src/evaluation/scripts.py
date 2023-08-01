import numpy as np
from collections import Counter
from typing import List
from collections import defaultdict

from .loader import SystemLoader
from .eval_tools import Evaluater
from ..comparative.tools import comparatisons_to_tensors, matrices_to_scores

# def evaluate_comparisons(comparisons, dataset, score_type):
#     C_tensor, M_tensor = comparatisons_to_tensors(comparisons)
#     scores = matrices_to_scores(C_tensor, M_tensor)
#     scores_dict = {k:v for k, v in enumerate(scores)}
#     labels = Evaluater.load_ratings_labels(dataset=dataset, score_type=score_type)
#     sys_pear = Evaluater.calc_system_pearson(scores_dict, labels)
#     sys_spear = Evaluater.calc_system_spearman(scores_dict, labels)
#     pear = Evaluater.calc_pearson(scores_dict, labels)
#     spear = Evaluater.calc_spearman(scores_dict, labels)
    
#     comp_labels = Evaluater.load_comparative_labels(dataset, score_type=score_type)
#     acc = Evaluater.calc_accuracy(comparisons, comp_labels)
    
#     sys_pear = round(sys_pear, 2)
#     sys_spear = round(sys_spear, 2) 
#     pear = round(pear, 2)
#     spear = round(spear, 2)
#     return({'acc':acc, 'sys_spear':sys_spear, 'sys_pear':sys_pear, 'spear':spear, 'pear':pear})

def evaluate_system(system, dataset, score_type):
    labels = Evaluater.load_ratings_labels(dataset=dataset, score_type=score_type)
    sys_pear = Evaluater.calc_system_pearson(system.ratings, labels)
    sys_spear = Evaluater.calc_system_spearman(system.ratings, labels)
    pear = Evaluater.calc_pearson(system.ratings, labels)
    spear = Evaluater.calc_spearman(system.ratings, labels)
    
    comp_labels = Evaluater.load_comparative_labels(dataset, score_type=score_type)
    acc = Evaluater.calc_accuracy(system.comparisons, comp_labels)
    
    sys_pear = round(sys_pear, 2)
    sys_spear = round(sys_spear, 2) 
    pear = round(pear, 2)
    spear = round(spear, 2)
    return({'acc':acc, 'sys_spear':sys_spear, 'sys_pear':sys_pear, 'spear':spear, 'pear':pear})

def get_system_bias(comparisons):
    counts = Counter(comparisons.values())
    probs = {k:round(v/len(comparisons), 2) for k, v in counts.items()}
    return probs

def load_system(path:str, balanced:bool=False)->SystemLoader:
    system = SystemLoader()
    if 'scoring' in path:
        system.load_ratings(path)
    elif 'probs' in path:
        system.load_comparisons_probs(path, balanced=balanced)
    else:
        system.load_comparisons(path)
    return system

def system_evaluation(path:str, dataset:str, score_type:str, balanced=False):
    system = load_system(path, balanced)
    return evaluate_system(system, dataset, score_type)

def multi_system_evaluation(paths:List[str], dataset:str, score_type:str, balanced=False):
    outputs = defaultdict(list)
    for path in paths:
        out = system_evaluation(path, dataset, score_type, balanced)
        for k, v in out.items():
            outputs[k].append(v)

    means = {k:np.mean(v) for k, v in outputs.items()}
    stds = {k:np.std(v) for k, v in outputs.items()}
    return {'means':means, 'stds':stds}