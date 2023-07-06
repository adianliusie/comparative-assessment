import os
import scipy
import numpy as np
import re

from collections import defaultdict
from typing import Dict

from src.utils.general import load_text_line, load_json
from src.data_handler import DataHandler

class SystemLoader:
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
            output_text = load_json(file_path)['output_text']
            score = re.split(r'\D+',output_text)[0]
            score = int(score) if score.isdigit() else -1 

            #save to dictionary
            ex_id = file.replace('.json', '')
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
        
        # keep an eye on how often the scores were invalid
        fails = sum([(v==-1) for v in comparisons.values()])
        total = sum([1      for v in comparisons.values()])
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

class ProbsSystemLoader(SystemLoader):
    def __init__(self,comparison_path=None):
        # load logits from comparisons
        self.comparisons_logits = self._load_comparison_logits(comparison_path)
        
    def _load_comparison_logits(self, path):
        comparisons_logits = {}
        # load the information from text files into a single dictionary
        data = load_json(path)

        for ex_id, output in data.items():
            output_logits = output['logits']

            # save to dictionary
            comparisons_logits[ex_id] = output_logits

        comparisons_logits = {k: v for k, v in sorted(comparisons_logits.items())}
        return comparisons_logits
    
    def get_comparisons(self, alphas:np.ndarray=None):
        if alphas is None: alphas=np.zeros(3)
        comparisons = {}
        for ex_id, logits in self.comparisons_logits.items():
            weighted_logits = alphas + logits
            pred = np.argmax(weighted_logits, axis=0)
            comparisons[ex_id] = pred
        return comparisons

    def get_balanced_thresholds(self):
        self.alphas = np.zeros(3)
        logits_array = np.array(list(self.comparisons_logits.values()))

        for a in np.arange(-3,3,0.01):
            reweighted_logits = logits_array + np.array([[0,a,0]])
            preds = np.argmax(reweighted_logits, axis=-1)
            if np.mean(preds) >= 0.5:
                break 

        return np.array([0,a,0])