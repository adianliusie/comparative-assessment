import os
import scipy
import numpy as np

from collections import defaultdict
from typing import Dict

from src.utils.general import load_text_line
from src.data_handler import DataHandler

class Evaluater:
    @staticmethod
    def load_comparative_labels(dataset:str, score_type:str='consistency'):
        data_handler = DataHandler('', dataset)
        data = data_handler.comparative_texts(score_type)
        labels = {}
        for ex in data:
            labels[ex.ex_id] = ex.label
        return dict(labels)
    
    @staticmethod
    def load_ratings_labels(dataset:str, score_type:str='consistency'):
        data_handler = DataHandler('', dataset)
        data = data_handler.scoring_texts(score_type)
        labels = defaultdict(dict)
        for ex in data:
            passage_id, ex_id = ex.ex_id.split('-')
            labels[int(passage_id)][int(ex_id)] = ex.label
        return dict(labels)
    
    @staticmethod
    def calc_accuracy(comparisons, labels):
        ex_ids = [k for k, v in labels.items() if v != -1]
        hits = [(comparisons[ex_id] == labels[ex_id]) for ex_id in ex_ids]
        correct = sum(hits) 
        
        unsure = 0.5*sum([1 for k, v in comparisons.items() if (v==-1) and (k in ex_ids)])
        return 100*(correct + unsure)/len(ex_ids)
    
    @staticmethod
    def calc_spearman(ratings:Dict[str, Dict[str, float]], labels:Dict[str, Dict[str, float]]):
        spearmans = []
        for passage_id in labels:
            keys = labels[passage_id].keys()
            true_scores = [labels[passage_id][k] for k in keys]
            pred_scores = [ratings[passage_id][k] for k in keys]
            
            # skip if all true scores same value, as spearman is undefined
            if len(set(true_scores)) == 1:
                continue 

            elif len(set(pred_scores)) == 1:
                spearmans.append(0)

            else:
                s = scipy.stats.spearmanr(pred_scores, true_scores)[0]  
                spearmans.append(s)
        #spearmans = [s for s in spearmans if not np.isnan(s)]
        return 100*np.mean(spearmans)
    
    @staticmethod
    def calc_pearson(ratings:Dict[str, Dict[str, float]], labels:Dict[str, Dict[str, float]]):
        pearsons = []
        for passage_id in labels:
            keys = labels[passage_id].keys()
            true_scores = [labels[passage_id][k] for k in keys]
            pred_scores = [ratings[passage_id][k] for k in keys]
            p = scipy.stats.pearsonr(pred_scores, true_scores)[0]  
            pearsons.append(p)
        pearsons = [p for p in pearsons if not np.isnan(p)]
        return 100*np.mean(pearsons)
 
    @staticmethod
    def calc_system_spearman(ratings:Dict[str, Dict[str, float]], labels:Dict[str, Dict[str, float]]):
        sys_keys = labels[0].keys()
        
        true_scores = [[labels[passage_id][k] for k in sys_keys] for passage_id in labels.keys()]
        pred_scores = [[ratings[passage_id][k] for k in sys_keys] for passage_id in labels.keys()]

        avg_true_scores = np.mean(true_scores, axis=0)
        avg_pred_scores = np.mean(pred_scores, axis=0)

        spearman = scipy.stats.spearmanr(avg_pred_scores, avg_true_scores)[0]  
        return 100*spearman
    
    @staticmethod
    def calc_system_pearson(ratings:Dict[str, Dict[str, float]], labels:Dict[str, Dict[str, float]]):
        sys_keys = labels[0].keys()
        
        true_scores = [[labels[passage_id][k] for k in sys_keys] for passage_id in labels.keys()]
        pred_scores = [[ratings[passage_id][k] for k in sys_keys] for passage_id in labels.keys()]

        avg_true_scores = np.mean(true_scores, axis=0)
        avg_pred_scores = np.mean(pred_scores, axis=0)

        pearson = scipy.stats.pearsonr(avg_pred_scores, avg_true_scores)[0]  
        return 100*pearson
    