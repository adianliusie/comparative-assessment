import os
import scipy
import numpy as np

from collections import defaultdict
from typing import Dict

from src.utils.general import load_text_line
from src.data_handler import DataHandler

class Evaluater:
    #== Loading labels ============================================================================#
    @staticmethod
    def load_comparative_labels(dataset:str, score_type:str='consistency')->np.ndarray:
        data_handler = DataHandler('', dataset)
        data = data_handler.comparative_texts(score_type)

        #convert dict to matrix
        ex_ids = [ex.ex_id.split('-') for ex in data] # ex_id="docid-candid"
        num_docs  = max([int(x[0]) for x in ex_ids]) + 1
        num_cands = max([int(x[1]) for x in ex_ids]) + 1

        comparisons = -1*np.ones((num_docs, num_cands, num_cands))
        for ex in data:
            doc_id, sys_id1, sys_id2 = [int(i) for i in ex.ex_id.split('-')]
            comparisons[doc_id, sys_id1, sys_id2] = ex.label

        return comparisons
    
    @staticmethod
    def load_ratings_labels(dataset:str, score_type:str='consistency')->np.ndarray:
        data_handler = DataHandler('', dataset)
        data = data_handler.scoring_texts(score_type)

        ex_ids = [ex.ex_id.split('-') for ex in data] # ex_id="docid-candid"
        num_docs  = max([int(x[0]) for x in ex_ids]) + 1
        num_cands = max([int(x[1]) for x in ex_ids]) + 1

        label_ratings = np.zeros((num_docs, num_cands))

        for ex in data:
            passage_id, ex_id = [int(i) for i in ex.ex_id.split('-')]
            label_ratings[passage_id, ex_id] = ex.label

        return label_ratings
    
    #== Calculating Metrics =======================================================================#
    @staticmethod
    def calc_accuracy(comparisons:np.ndarray, labels:np.ndarray)->float:
        hits = (comparisons == labels)[labels != -1]
        correct = sum(hits) 
        total = np.sum(labels != 1)
        return 100*(correct)/total
    
    @staticmethod
    def calc_spearman(ratings:np.ndarray, labels:np.ndarray):
        spearmans = []
        for pred_k, label_k in zip(ratings, labels):
            # skip if all true scores same value, as spearman is undefined
            if len(set(label_k)) == 1: 
                continue 
            elif len(set(pred_k)) == 1:
                spearmans.append(0)
            else:
                s = scipy.stats.spearmanr(pred_k, label_k)[0]  
                spearmans.append(s)

        return 100*np.mean(spearmans)
    
    @staticmethod
    def calc_pearson(ratings:np.ndarray, labels:np.ndarray):
        pearsons = []
        for pred_k, label_k in zip(ratings, labels):
            if len(set(label_k)) == 1: 
                continue 
            elif len(set(pred_k)) == 1:
                pearsons.append(0)
            else:
                p = scipy.stats.pearsonr(pred_k, label_k)[0]  
                pearsons.append(p)
        return 100*np.mean(pearsons)
 
    @staticmethod
    def calc_system_spearman(ratings:np.ndarray, labels:np.ndarray):        
        avg_true_scores = np.mean(labels, axis=0)
        avg_pred_scores = np.mean(ratings, axis=0)

        spearman = scipy.stats.spearmanr(avg_pred_scores, avg_true_scores)[0]  
        return 100*spearman
    
    @staticmethod
    def calc_system_pearson(ratings:Dict[str, Dict[str, float]], labels:Dict[str, Dict[str, float]]):
        avg_true_scores = np.mean(labels, axis=0)
        avg_pred_scores = np.mean(ratings, axis=0)

        pearson = scipy.stats.pearsonr(avg_pred_scores, avg_true_scores)[0]  
        return 100*pearson
    
    #== Miscellaneous ==============================================================================#
    def reliability_plot(comparison_probs, labels):
        pass
    
    
    
    def scores_to_class(ratings, labels, thresh=None, K=None):
        """given scores and true labels, finds optimal tresholds that split the classes"""
        assert len(ratings) == len(labels) == 1, "only implemented for grading exams"
        ratings = ratings[0]
        labels = labels[0]

        if K==None:
            K = len(set(labels))
      
        assert set(labels) == set(range(K))

        # get ranks of decisions, and sorted scores
        ranks = [k for k, v in sorted(enumerate(ratings), key=lambda x: x[1])]
        ordered_labels = np.array([labels[i] for i in ranks])
        ordered_scores = [v for k, v in sorted(enumerate(ratings), key=lambda x: x[1])]
            
        # initialise thresholds 
        if thresh == None:
            thresh = [int(x) for x in np.linspace(0, len(ordered_labels), K+1)[1:-1]]

        # get cumaltive counts of each class
        cum_list = {}
        for i in range(K):
            cum_list[i] = np.cumsum(ordered_labels==i)
            
        old_thresh = np.zeros_like(thresh)
        #for _ in range(10):
        while not np.array_equal(old_thresh, thresh):
            print(old_thresh, '\n', thresh)
            old_thresh = thresh.copy()
            for i in range(K-1):
                lower = thresh[i-1] if (i != 0) else 0
                higher = thresh[i+1] if (i+1 != K-1) else len(x)-1

                diff = cum_list[i] - cum_list[i+1]

                t = lower + np.argmax(diff[lower:higher]) + 1
                thresh[i] =  t
        
        # convert ranks to threhsolds in terms of input scores
        print(thresh)
        score_thresh = [ordered_scores[i] for i in thresh]
        return score_thresh

    