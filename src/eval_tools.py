import os
import scipy
import numpy as np

from collections import defaultdict
from typing import Dict

from src.utils import load_text_line
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
        print(len(labels), len(ex_ids))
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
            s = scipy.stats.spearmanr(pred_scores, true_scores)[0]  
            spearmans.append(s)
        spearmans = [s for s in spearmans if not np.isnan(s)]
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
        def system_score(ratings, sys_id):
            return np.mean([ratings[doc_id][sys_id] for doc_id in labels])
                                   
        system_ids = ratings[0].keys()                 
        system_preds = [system_score(ratings, sys_id) for sys_id in system_ids]
        system_labels = [system_score(labels, sys_id) for sys_id in system_ids]
        
        spearman = scipy.stats.spearmanr(system_preds, system_labels)[0]  
        return 100*spearman
    
    @staticmethod
    def calc_system_pearson(ratings:Dict[str, Dict[str, float]], labels:Dict[str, Dict[str, float]]):
        def system_score(ratings, sys_id):
            return np.mean([ratings[doc_id][sys_id] for doc_id in labels])
                                   
        system_ids = ratings[0].keys()                 
        system_preds = [system_score(ratings, sys_id) for sys_id in system_ids]
        system_labels = [system_score(labels, sys_id) for sys_id in system_ids]
        
        pearson = scipy.stats.pearsonr(system_preds, system_labels)[0]  
        return 100*pearson
    
class MarkerLoader:
    def __init__(self, ratings_path=None, comparison_path=None):
        if ratings_path:
            self.load_ratings(ratings_path)
        elif comparison_path:
            self.load_comparisons(comparison_path)

    def load_comparisons(self, path):
        self.comparisons = self._load_comparisons(path)
        self.ratings = self.comparisons_to_ratings(self.comparisons)

    def load_ratings(self, path):
        self.ratings = self._load_ratings(path)
        self.comparisons = self.ratings_to_comparisons(self.ratings)
    
    #== Load Files by category ================================================#
    def _load_ratings(self, path)->Dict[str, Dict[str, float]]:        
        ratings = defaultdict(dict)
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            
            #read contexts file and convert to float
            score = load_text_line(file_path)[0]
            score = int(score) if score.isdigit() else -1 
                
            #save to dictionary
            ex_id = file.replace('.txt', '')
            passage_id, summary_id = ex_id.split('-')
            ratings[int(passage_id)][int(summary_id)] = score
            
        ratings = dict(ratings)
        
        # keep an eye on how often the scores were invalid
        fails = sum([(v==-1) for doc in ratings.values() for v in doc.values() ])
        total = sum([1       for doc in ratings.values() for v in doc.values() ])
        print(f"loaded ratings with {fails} failures out of {total}")
        
        return ratings
    
    def _load_comparisons(self, path)->Dict[str, int]:
        comparisons = {}
        # load the information from text files into a single dictionary
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            output = load_text_line(file_path)            
            
            # determine which of A and B is preferred
            if 'Summary A' in output and 'Summary B' not in output:
                score = 0
            elif 'Summary B' in output and not 'Summary A' in output:
                score = 1
            else:
                score = -1
            
            # save to dictionary
            ex_id = file.replace('.txt', '')
            comparisons[ex_id] = score
        
        # keep an eye on how often the scores were invalid
        fails = sum([(v==-1) for v in comparisons.values()])
        total = sum([1       for v in comparisons.values()])
        print(f"loaded ratings with {fails} failures out of {total}")
        
        comparisons = {k: v for k, v in sorted(comparisons.items())}
        return comparisons
    
    #== Methods to Convert ==================================================#
    def ratings_to_comparisons(self, ratings):
        comparisons = defaultdict(dict)
        
        for passage_id in ratings:
            passage_scores = ratings[passage_id]
            N = len(passage_scores)
            for i in range(N):
                for j in range(N):
                    if i==j: continue   
                    score_1 = passage_scores[i]
                    score_2 = passage_scores[j]

                    # select the passage with the highest score
                    if score_1 > score_2:   
                        score = 0
                    elif score_2 > score_1: 
                        score = 1
                    else:                   
                        score = -1

                    # append input to dictionary
                    comparisons[f"{passage_id}-{i}-{j}"] = score
                    
        comparisons = {k: v for k, v in sorted(comparisons.items())}
        return comparisons

    def comparisons_to_ratings(self, comparisons):
        ratings = defaultdict(dict)
        
        def increase_rating(passage_id:int, ex_id:int, value=1):
            if ex_id not in ratings[passage_id]:
                ratings[passage_id][ex_id] = 0
            ratings[int(passage_id)][ex_id] += value
            return 
        
        for ex_id, v in comparisons.items():
            passage_id, ex_1_id, ex_2_id = ex_id.split('-')
            passage_id, ex_1_id, ex_2_id = int(passage_id), int(ex_1_id), int(ex_2_id)
            if v == -1:
                increase_rating(passage_id, ex_1_id, value=0.5)
                increase_rating(passage_id, ex_2_id, value=0.5)
            elif v == 0:
                increase_rating(passage_id, ex_1_id, value=1)
                increase_rating(passage_id, ex_2_id, value=0)
            elif v == 1:
                increase_rating(passage_id, ex_1_id, value=0)
                increase_rating(passage_id, ex_2_id, value=1)

        return dict(ratings)
    