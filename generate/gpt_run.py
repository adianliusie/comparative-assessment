import os
import argparse
import pickle
import random
import openai
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset

from src.data_handler import DataHandler
from src.utils import save_json, load_json

import time

openai.organization = "org-DlbzLzT5CDhoGm9GuaIkleGI"
openai.api_key = os.getenv("OPENAI_API_KEY")

# python gpt_run.py --dataset summeval-s --score-type consistency --shuffle --ranking --prompt-num 1 --output-path output_texts/summeval-consistency/prompt_1
# python gpt_run.py --dataset summeval-s --score-type consistency --shuffle --prompt-num 4 --output-path output_texts/summeval-consistency/prompt_4

def main(
    output_path:str,
    dataset:str='sumevall',
    score_type:str='consistency',
    prompt_num:int=1,
    shuffle:bool=True,
    ranking=False
):
    
    #load prompt from default, or choose your own prompt
    #prompt_template = ...
    prompt_template = get_default_template(prompt_num, ranking, score_type)
    
    # get input text data to feed to chatgpt
    data_handler = DataHandler(prompt_template, dataset=dataset)
    if ranking:
        proc_inputs = data_handler.comparative_texts(score_type)
    else:
        proc_inputs = data_handler.scoring_texts(score_type)
    
    # save experiment settings 
    info = {
        'prompt_num':prompt_num,
        'prompt':prompt_template,
        'dataset':dataset,
        'score_type':score_type,
        'shuffle':shuffle,
        'ranking':ranking
    }
    info_path = f"{output_path}/info.json"
    if not os.path.isfile(info_path):
        save_json(info, info_path) 
    else:
        log_info = load_json(info_path)
        assert all([info[k] == log_info[k] for k in info.keys() if k != 'dataset'])
        
    # select evaluation order for examples (shuffling allows parallelisation)
    ids = [x for x in range(len(proc_inputs))]
    if shuffle:
        random.shuffle(ids)

    # process all inputs to chatgpt
    for idx in ids:
        ex = proc_inputs[idx]
        outpath = f"{output_path}/outputs/{ex.ex_id}.txt"
        exist = os.path.isfile(outpath)
        if exist:
            print(f"skipping {ex.ex_id}")
            continue
        
        # ChatGPT
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-0301',
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": ex.input_text},
            ],
            temperature=0.0, # 0.0 = deterministic
            max_tokens=10, # max_tokens is the generated one,
        )
        
        # get and print generated text
        gen_text = response.choices[0].message.content
        print("ChatGPT:", gen_text)
        with open(outpath, "w") as f:
            f.write(gen_text)
        print(f"[{datetime.now()}] {ex.ex_id} wrote: {outpath}")

def add_arguments(parser):
    '''Build Argument Parser'''
    #parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--output-path', type=str, required=True, help='where to save chatgpt outputs')
    parser.add_argument('--dataset', type=str, default='summeval', help='which evaluation dataset to use')
    parser.add_argument('--score-type', type=str, default='consistency', help='which score to use of the dataset')
    parser.add_argument('--shuffle', action='store_true', help='whether to shuffling order of samples')
    parser.add_argument('--ranking', action='store_true', help='whether to do comparative evaluation')
    parser.add_argument('--prompt-num', type=int, default=1, help='which prompt to use')
    #parser.add_argument('--shuffle', type="bool", nargs="?", const=True, default=False)
    return parser

def get_default_template(prompt_num, ranking, score_type):
    if prompt_num == 1:
        assert (ranking == True) and (score_type=='consistency')
        prompt_template = "Assess the following two summaries given the corresponding article, and determine which passage is more consistent.\n\n<context>\n\nSummary A: <summary_1>\n\nSummary B: <summary_2>\n\nWhich Summary is more consistent relative to the passage, Summary A or Summary B?\n\nAnswer:"
    elif prompt_num == 2:
        assert (ranking == True) and (score_type=='consistency')
        prompt_template = "Determine which of the two summaries is more consistent given the following article.\n\n<context>\n\nSummary A: <summary_1>\n\nSummary B: <summary_2>\n\nWhich Summary is more consistent relative to the passage, Summary A or Summary B?\n\nAnswer:"
    elif prompt_num == 3:
        assert (ranking == True) and (score_type=='consistency')
        prompt_template = "Assess the following two summaries given the corresponding article, and determine which passage is more consistent. Note that consistency measures how much information included in the summary is present in the source article.\n\n<context>\n\nSummary A: <summary_1>\n\nSummary B: <summary_2>\n\nWhich Summary is more consistent relative to the passage, Summary A or Summary B?\n\nAnswer:"
    elif prompt_num == 4:
        assert (ranking == False) and (score_type=='consistency')
        prompt_template = "Score the following summary given the corresponding article with respect to consistency from 1 to 10.\n\nSummary:<summary_1>\n\nSource Article:<context>\n\nMarks:"  
    elif prompt_num == 5:
        assert (ranking == False) and (score_type=='consistency')
        prompt_template = "Determine a score between 1 and 100 for how consistent the following summary is with respect to the article\n\nSource Article:<context>\n\nSummary:<summary_1>\n\nScore:"  
    elif prompt_num == 6:
        assert (ranking == False) and (score_type=='consistency')
        prompt_template = "Score the following summary given the corresponding article with respect to consistency from 1 to 10. Note that consistency measures how much information included in the summary is present in the source article. 10 points indicate the summary contains only statements that are entailed by the source document\n\nSummary:<summary_1>\n\nSource Article:<context>\n\nMarks:"  
    return prompt_template

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    kwargs = vars(parser.parse_args())
    for counter in range(1, 5):
        try:
            main(**kwargs)
            time.sleep(1)
        except openai.error.RateLimitError:
            print("openai.error.RateLimitError... #{}".format(counter))
            print("restart in 10 seconds")
            time.sleep(10)