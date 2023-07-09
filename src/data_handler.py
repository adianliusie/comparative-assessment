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
            num_sum = len(doc.machine_summaries)
            passage = doc.passage
            for k in range(num_sum):
                summary = doc.machine_summaries[k]
                text_info = SimpleNamespace(
                    context=passage,
                    summary_1=summary,
                )
                input_text = self.fill_template(text_info)

                label = doc.scores[score_type][k]
                # create example
                ex_id = doc.passage_id + '-' + str(k)
                ex = SimpleNamespace(
                    ex_id=ex_id,
                    input_text=input_text,
                    label=label
                )
                outputs.append(ex)
        return outputs

    def scoring_texts_topicalchat(self, score_type):
        outputs = []
        for doc in self.documents:
            num_response = len(doc.responses)
            context = doc.context
            fact = doc.fact
            for k in range(num_response):
                response = doc.responses[k]
                input_text = self.prompt_template.replace("<fact>", doc.fact)
                input_text = input_text.replace("<context>", doc.context)
                input_text = input_text.replace("<response>", response)
                # print(input_text)
                label = doc.scores[score_type][k]
                # create example
                ex_id = str(doc.dialogue_id) + '-' + str(k)
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
            num_sum = len(doc.machine_summaries)
            passage = doc.passage
            for i in range(num_sum):
                for j in range(num_sum):
                    # skip the same document
                    if i == j: continue

                    # create input text by filling in template
                    summary_1 = doc.machine_summaries[i]
                    summary_2 = doc.machine_summaries[j]

                    text_info = SimpleNamespace(
                        context=passage,
                        summary_1=summary_1,
                        summary_2=summary_2
                    )
                    input_text = self.fill_template(text_info)

                    # get label based on which summary is more fluent
                    score_1 = doc.scores[score_type][i]
                    score_2 = doc.scores[score_type][j]
                    score_diff = score_1-score_2

                    if   score_diff  > 0: label = 0
                    elif score_diff  < 0: label = 1
                    elif score_diff == 0: label = -1

                    # create example
                    ex_id = doc.passage_id + '-' + str(i) + '-' + str(j)
                    ex = SimpleNamespace(
                        ex_id=ex_id,
                        input_text=input_text,
                        label=label,
                        score_diff=score_diff
                    )

                    outputs.append(ex)
                    # add to outputs provided scores are not the same
                    # if score_1 != score_2:
                    #    outputs.append(ex)
        return outputs

    def comparative_texts_topicalchat(self, score_type):
        outputs = []
        for doc in self.documents:
            num_response = len(doc.responses)
            context = doc.context
            fact = doc.fact
            for i in range(num_response):
                for j in range(num_response):
                    # skip the same document
                    if i == j: continue

                    # create input text by filling in template
                    response_A = doc.responses[i]
                    response_B = doc.responses[j]

                    input_text = self.prompt_template.replace("<fact>", doc.fact)
                    input_text = input_text.replace("<context>", doc.context)
                    input_text = input_text.replace("<response_A>", response_A)
                    input_text = input_text.replace("<response_B>", response_B)

                    # get label based on which summary is more fluent
                    score_1 = doc.scores[score_type][i]
                    score_2 = doc.scores[score_type][j]
                    score_diff = score_1-score_2

                    if   score_diff  > 0: label = 0
                    elif score_diff  < 0: label = 1
                    elif score_diff == 0: label = -1

                    # create example
                    ex_id = str(doc.dialogue_id) + '-' + str(i) + '-' + str(j)
                    ex = SimpleNamespace(
                        ex_id=ex_id,
                        input_text=input_text,
                        label=label,
                        score_diff=score_diff
                    )

                    outputs.append(ex)
                    # add to outputs provided scores are not the same
                    # if score_1 != score_2:
                    #    outputs.append(ex)
        return outputs

    def fill_template(self, text_info):
        text = self.prompt_template
        if '<context>' in text:
            text = text.replace('<context>', text_info.context)
        if '<summary_1>' in text:
            text = text.replace('<summary_1>', text_info.summary_1)
        if '<summary_2>' in text:
            text = text.replace('<summary_2>', text_info.summary_2)
        if '<topic>' in text:
            text = text.replace('<topic>', text_info.topic)
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
            # warning: you will have to change this
            path = "/home/pm574/rds/rds-altaslp-8YSp2LXTlkY/data/nlg_evaluation/topicalchat_usr/tc_usr_data.json"
            documents = cls.load_topicalchat(path)
        return documents

    @staticmethod
    @lru_cache(maxsize=3)
    def load_summeval()->List[SimpleNamespace]:
        data = []
        summ_eval = load_dataset('mteb/summeval')['test']
        for k, row in enumerate(summ_eval):
            ex = SimpleNamespace(
                passage_id=str(k),
                passage=row['text'],
                machine_summaries=row['machine_summaries'],
                scores={
                    'coherency':row['coherence'],
                    'fluency':row['fluency'],
                    'consistency':row['consistency'],
                    'relevance':row['relevance']
                }
            )
            data.append(ex)
        return data

    @staticmethod
    def load_topicalchat(path_to_json) -> List[SimpleNamespace]:
        # rds-altaslp-8YSp2LXTlkY/data/nlg_evaluation/topicalchat_usr/tc_usr_data.json
        with open(path_to_json, "r") as f:
            x = f.read()
        data = json.loads(x)
        data_list = []
        for i in range(len(data)):
            data_i = data[i]
            # for j, response in enumerate(data_i['responses']):
            responses_text = []
            scores_understand = []
            scores_natural   = []
            scores_maintain  = []
            scores_engaging  = []
            scores_knowledge = []
            scores_overall   = []
            for response in data_i['responses']:
                responses_text.append(response['response'])
                scores_understand.append(np.mean(response['Understandable']))
                scores_natural.append(np.mean(response['Natural']))
                scores_maintain.append(np.mean(response['Maintains Context']))
                scores_engaging.append(np.mean(response['Engaging']))
                scores_knowledge.append(np.mean(response['Uses Knowledge']))
                scores_overall.append(np.mean(response['Overall']))

            ex = SimpleNamespace(
                dialogue_id=i,
                context=data_i['context'],
                fact=data_i['fact'],
                responses=responses_text,
                scores={
                    'understandable': scores_understand,
                    'natural': scores_natural,
                    'maintain_context': scores_maintain,
                    'engaging': scores_engaging,
                    'use_knowledge': scores_knowledge,
                    'overall': scores_overall,
                }
            )
            data_list.append(ex)
        return data_list
