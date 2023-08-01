import json
import os
import numpy as np

from datasets import load_dataset
from types import SimpleNamespace
from typing import List
from functools import lru_cache

from .utils.general import load_json_files

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

                # get prompt input text
                input_text = self.fill_template(text_info) if self.prompt_template else None

                # get labels for scoring
                label = doc.scores[score_type][k]

                # add example to output
                ex_id = doc.context_id + '-' + str(k)
                ex = SimpleNamespace(
                    ex_id=ex_id,
                    input_text=input_text,
                    label=label,
                    response=response,
                    reference=getattr(doc, 'reference', None),
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
        elif dataset=='podcast':
            documents = cls.load_podcast()
        elif dataset=='topicalchat':
            documents = cls.load_topicalchat()
        elif dataset=='webnlg':
            documents = cls.load_webnlg()
        elif dataset=='wi-train':
            documents = cls.load_write_and_improve(split='train')
        elif dataset=='wi-dev':
            documents = cls.load_write_and_improve(split='dev')
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
                reference=row['human_summaries'][0],
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
    def load_topicalchat() -> List[SimpleNamespace]:
        data_path = "/rds/project/rds-8YSp2LXTlkY/data/nlg_evaluation/topicalchat_usr/tc_usr_data.json"
        with open(data_path, "r") as f:
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

    @staticmethod
    def load_webnlg() -> List[SimpleNamespace]:
        # dataset downloaded from https://github.com/ufal/nlgi_eval
        data_path = "/rds/project/rds-8YSp2LXTlkY/data/nlg_evaluation/data-to-text/webnlg.processed.json"
        with open(data_path, "r") as f:
            x = f.read()
        data = json.loads(x)

        output = []
        for k, row in data.items():
            generated_texts, fluency, grammar, semantics = [], [], [], []
            for system, value in row.items():
                generated_texts.append(value['text'])
                fluency.append(value['fluency'])
                grammar.append(value['grammar'])
                semantics.append(value['semantics'])
                triples = value['data'] # triples concatenated as string- same for all systems

            context = f"The following are semantic triples of the form (subject|relation|object)\n\n{triples}"
            ex = SimpleNamespace(
                context_id=str(k),
                context=context,
                responses=generated_texts,
                scores={
                    'fluency': fluency,
                    'grammar': grammar,
                    'semantic': semantics,
                }
            )
            output.append(ex)
        return output

    @staticmethod
    def load_write_and_improve(split='train') -> List[SimpleNamespace]:
        base_path = "/rds/project/rds-8YSp2LXTlkY/data/nlg_evaluation/write-improve"

        paths = [os.path.join(base_path, f"{level}.{split}.json") for level in ['A', 'B', 'C']]
        jsons = [load_json_files(path) for path in paths]
        data = [ex for json in jsons for ex in json]

        responses = [ex['text'] for ex in data]
        detailed_raw_scores = [ex['cefr'] for ex in data]

        detailed_cefr_to_scores = {cefr:k for k, cefr in enumerate(sorted(list(set(detailed_raw_scores))))}
        cefr_to_scores = {'A1':0, 'A2':1, 'B1':2, 'B2':3, 'C1':4, 'C2':5}
        scores = [detailed_cefr_to_scores[score] for score in detailed_raw_scores]
        raw_cefr = [score[:2] for score in detailed_raw_scores]
        cefr = [cefr_to_scores[score] for score in raw_cefr]

        out = SimpleNamespace(
                context_id='0',
                context=None,
                responses=responses,
                scores={'overall':scores,
                        'detailed_raw':detailed_raw_scores,
                        'cefr_raw':raw_cefr,
                        'cefr':cefr}
        )
        return [out]

    @staticmethod
    def load_podcast()->List[SimpleNamespace]:
        podcast_data = load_dataset("potsawee/podcast_summary_assessment")['evaluation']
        # splitting 3580 -> 179 * 20
        podcast_179 = {}
        score_mapping = {'B':0, 'F': 1, 'G': 2, 'E': 3} # Bad, Fair, Good, Excellent
        for k, row in enumerate(podcast_data):
            episode_id = row['episode_id']
            if episode_id not in podcast_179:
                podcast_179[episode_id] = SimpleNamespace(
                    context_id=row['episode_id'],
                    context=row['transcript'],
                    responses=[],
                    scores={'overall': []},
                )
            assert podcast_179[episode_id].context_id == row['episode_id'] # sanity check
            assert podcast_179[episode_id].context == row['transcript'] # sanity check
            podcast_179[episode_id].responses.append(row['summary'])
            podcast_179[episode_id].scores['overall'].append(score_mapping[row['score']])
        # dict to list
        podcast_179 = [v for v in podcast_179.values()]
        return podcast_179
