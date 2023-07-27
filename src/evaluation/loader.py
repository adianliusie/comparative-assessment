import os
import scipy
import numpy as np
import re

from collections import defaultdict
from typing import Dict

from src.utils.general import load_text_line, load_json
from src.data_handler import DataHandler

class SystemLoader:
    def load_ratings(self, path):
        self.ratings = self._load_ratings(path)
        self.comparisons = self.ratings_to_comparisons(self.ratings)
    
    def load_comparisons(self, path, lim=None):
        self.comparisons = self._load_comparisons(path, lim=lim)
        self.ratings = self.comparisons_to_ratings(self.comparisons)

    def load_comparisons_logits(self, path, lim=None, balanced=False):
        comparison_logits = self._load_comparison_logits(path, lim=lim)
        t = self.get_balanced_thresholds(comparison_logits) if balanced else None
        self.comparisons = self.logits_to_comparisons(comparison_logits, t=t)
        self.ratings = self.comparisons_to_ratings(self.comparisons)

    #== Load Files by category =======================================================#
    @staticmethod
    def _load_ratings(path)->Dict[str, Dict[str, float]]:        
        ratings = defaultdict(dict)

        data = load_json(path)

        for ex_id, output in data.items():
            output_text = output['output_text']

            # extract numerical prediction
            score = re.split(r'\D+',output_text)[0]
            score = int(score) if score.isdigit() else -1 

            # save to dictionary
            passage_id, summary_id = ex_id.split('-')
            ratings[int(passage_id)][int(summary_id)] = score
            
        ratings = dict(ratings)
        
        # check how often the scores were invalid
        fails = sum([(v==-1) for doc in ratings.values() for v in doc.values() ])
        total = sum([1       for doc in ratings.values() for v in doc.values() ])
        #print(f"loaded ratings with {fails} failures out of {total}")
        return ratings
    
    @staticmethod
    def _load_comparisons(path:str, lim:int=None)->Dict[str, int]:
        comparisons = {}
        # load the information from text files into a single dictionary
        data = load_json(path)

        for ex_id, output in data.items():
            output_text = output['output_text']
            # determine which of A and B is preferred
            if (' A' in output_text) and (not ' B' in output_text):
                score = 0
            elif (' B' in output_text) and (not ' A' in output_text):
                score = 1
            else:
                score = -1
                
            # save to dictionary
            comparisons[ex_id] = score
        
        # check how often the scores were invalid
        fails = sum([(v==-1) for v in comparisons.values()])
        total = sum([1      for v in comparisons.values()])
        print(f"loaded ratings with {fails} failures out of {total}")
        
        comparisons = {k: v for k, v in sorted(comparisons.items())}
        return comparisons

    @staticmethod
    def _load_comparison_logits(path:str, lim:int):
        comparisons_logits = {}
        # load the information from text files into a single dictionary
        data = load_json(path)

        #print('there is a fix in the code to ignore the neutral class')

        for ex_id, output in data.items():
            output_logits = output['logits'][:2]
            comparisons_logits[ex_id] = output_logits

        comparisons_logits = {k: v for k, v in sorted(comparisons_logits.items())}

        return comparisons_logits

    #== Methods to Convert ===========================================================#
    @staticmethod
    def ratings_to_comparisons(ratings):
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

    @staticmethod
    def comparisons_to_ratings(comparisons):
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

    #== Methods dealing with prompt-based classifier =================================#
    @staticmethod
    def logits_to_comparisons(comparisons_logits, t:np.ndarray=None):
        if t is None: t=0
        comparisons = {}
        for ex_id, logits in comparisons_logits.items():
            weighted_logits = logits + np.array([0,t])
            #print(weighted_logits)
            #import time; time.sleep(2)
            pred = np.argmax(weighted_logits, axis=0)
            comparisons[ex_id] = pred
        return comparisons

    @staticmethod
    def get_balanced_thresholds(comparisons_logits):
        logits_array = np.array(list(comparisons_logits.values()))

        for t in np.arange(-5,5,0.01):
            reweighted_logits = logits_array + np.array([[0,t]])
            preds = np.argmax(reweighted_logits, axis=-1)
            if np.mean(preds) >= 0.5:
                break 
        return t