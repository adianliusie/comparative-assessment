import json
import numpy as np
from datasets import load_dataset
from types import SimpleNamespace
from typing import List
from functools import lru_cache

class DataHandler:
    def __init__(self, prompt_template:str, dataset:str='summeval'):
        self.prompt_template = prompt_template
        self.documents = self.load_data(dataset)

    def scoring_texts(self, score_type):
        outputs = []
        for doc in self.documents:
            num_responses = len(doc.responses)
            for k in range(num_responses):
                # relevant information need for text filling
                context = doc.context
                response = doc.responses[k]
                fact = getattr(doc, 'fact', None)
                
                # fill in the prompt template
                text_info = SimpleNamespace(
                    context=context,
                    response_A=response,
                    fact=fact
                )
                input_text = self.fill_template(text_info)

                # get labels for scoring
                label = doc.scores[score_type][k]

                # add example to output
                ex_id = doc.context_id + '-' + str(k)
                ex = SimpleNamespace(
                    ex_id=ex_id,
                    input_text=input_text,
                    label=label
                )
                outputs.append(ex)
        return outputs

    def comparative_texts(self, score_type):
        outputs = []
        for doc in self.documents:
            num_responses = len(doc.responses)
            for i in range(num_responses):
                for j in range(num_responses):
                    # skip the same document
                    if i == j: continue

                    # relevant information need for text filling
                    context = doc.context
                    response_A = doc.responses[i]
                    response_B = doc.responses[j]
                    fact = getattr(doc, 'fact', None)

                    # fill in the prompt template
                    text_info = SimpleNamespace(
                        context=context,
                        response_A=response_A,
                        response_B=response_B,
                        fact=fact
                    )
                    input_text = self.fill_template(text_info)

                    # get comparative labels
                    score_1 = doc.scores[score_type][i]
                    score_2 = doc.scores[score_type][j]
                    score_diff = score_1-score_2

                    if   score_diff  > 0: label = 0
                    elif score_diff  < 0: label = 1
                    elif score_diff == 0: label = -1

                    # add example to output
                    ex_id = doc.context_id + '-' + str(i) + '-' + str(j)
                    ex = SimpleNamespace(
                        ex_id=ex_id,
                        input_text=input_text,
                        label=label,
                        score_diff=score_diff
                    )

                    outputs.append(ex)
        return outputs

    def fill_template(self, text_info):
        text = self.prompt_template
        if '<context>' in text:
            text = text.replace('<context>', text_info.context)
        if '<A>' in text:
            text = text.replace('<A>', text_info.response_A)
        if '<B>' in text:
            text = text.replace('<B>', text_info.response_B)
        if '<topic>' in text:
            text = text.replace('<topic>', text_info.topic)
        if '<fact>' in text:
            text = text.prompt_template.replace("<fact>", text_info.fact)
        return text

    #== Data Loading Methods ===========================================================#
    @classmethod
    def load_data(cls, dataset):
        if dataset=='summeval':
            documents = cls.load_summeval()
        elif dataset=='summeval-s':
            documents = cls.load_summeval()[:20]
        elif dataset=='summeval-t':
            documents = cls.load_summeval()[:5]
        elif dataset=='topicalchat':
            path = "/rds/project/rds-8YSp2LXTlkY/data/nlg_evaluation/topicalchat_usr/tc_usr_data.json"
            documents = cls.load_topicalchat(path)
        return documents

    @staticmethod
    @lru_cache(maxsize=3)
    def load_summeval()->List[SimpleNamespace]:
        output = []
        summ_eval = load_dataset('mteb/summeval')['test']
        for k, row in enumerate(summ_eval):
            ex = SimpleNamespace(
                context_id=str(k),
                context=row['text'],
                responses=row['machine_summaries'],
                scores={
                    'coherency':row['coherence'],
                    'fluency':row['fluency'],
                    'consistency':row['consistency'],
                    'relevance':row['relevance']
                }
            )
            output.append(ex)
        return output

    @staticmethod
    def load_topicalchat(path_to_json) -> List[SimpleNamespace]:
        # rds-altaslp-8YSp2LXTlkY/data/nlg_evaluation/topicalchat_usr/tc_usr_data.json
        with open(path_to_json, "r") as f:
            x = f.read()
        data = json.loads(x)
        output = []
        for k, row in enumerate(data):
            responses = row['responses']
            ex = SimpleNamespace(
                context_id=str(k),
                context=row['context'],
                responses=[x['response'] for x in responses],
                fact=row['fact'],
                scores={
                    'coherency': [np.mean(x['Understandable']) for x in responses],
                    'naturalness': [np.mean(x['Natural']) for x in responses],
                    'continuity': [np.mean(x['Maintains Context']) for x in responses],
                    'engagingness': [np.mean(x['Engaging']) for x in responses],
                    'groundedness': [np.mean(x['Uses Knowledge']) for x in responses],
                    'overall': [np.mean(x['Overall']) for x in responses],
                }
            )
            output.append(ex)
        return output
