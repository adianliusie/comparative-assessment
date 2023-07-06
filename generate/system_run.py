import os
import argparse
import pickle
import random
import openai
import os

from tqdm import tqdm
from datetime import datetime

from src.data_handler import DataHandler
from src.utils.general import save_json, load_json
from src.models import load_interface
from src.models.prompts import get_prompt_template

import time
# python system_run.py --output-path output_texts/falcon-7b/summeval-consistency/prompt_c1 --system falcon-7b --dataset summeval-s --score consistency --prompt-id c1 --shuffle --comparative
# python system_run.py --output-path output_texts/falcon-7b/summeval-consistency/prompt_c1 --system falcon-7b --dataset summeval-s --score-type consistency --prompt-id comparative-1 --shuffle --comparative
# python system_run.py --output-path output_texts2/chatgpt/summeval-relevance/comparative-1 --system chatgpt --dataset summeval-s --score-type relevance --prompt-id comparative-1 --shuffle --comparative

def main(
    system:str,
    output_path:str,
    dataset:str='sumevall',
    score_type:str='consistency',
    prompt_id:int='c1',
    shuffle:bool=True,
    comparative=False,
    max_len=None
):
    #load prompt from default, or choose your own prompt
    assert ('comparative' in prompt_id) == comparative
    prompt_template = get_prompt_template(prompt_id, score_type)
    
    # get input text data to feed to chatgpt
    data_handler = DataHandler(prompt_template, dataset=dataset)
    if comparative:
        proc_inputs = data_handler.comparative_texts(score_type)
    else:
        proc_inputs = data_handler.scoring_texts(score_type)
    
    # create directory if not already existing
    system_output_path = f"{output_path}/outputs"
    if not os.path.isdir(system_output_path):
        os.makedirs(system_output_path)   

    # save experiment settings 
    info = {
        'prompt_id':prompt_id,
        'prompt':prompt_template,
        'system':system, 
        'dataset':dataset,
        'score_type':score_type,
        'shuffle':shuffle,
        'comparative':comparative
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

    # determine examples already run (and processed)
    done_ids = []
    if os.path.isfile(f"{system_output_path}/combined.json"):
        done_ids = set(list(load_json(f"{system_output_path}/combined.json").keys()))

    # select the model to run the inputs through
    interface = load_interface(system=system)

    # process all inputs to chatgpt
    for idx in ids:
        ex = proc_inputs[idx]
        outpath = f"{output_path}/outputs/{ex.ex_id}.json"

        # skip outputs already computed
        if (os.path.isfile(outpath)) or (ex.ex_id in done_ids):
            continue
        
        # Get LLM outputs
        response = interface.response(input_text=ex.input_text, 
                                      do_sample=False,
                                      max_new_tokens=max_len)

        # get and print generated text
        gen_text = response.output_text
        current_time = datetime.now().strftime('%H:%M:%S')
        print(f"[{current_time}] {ex.ex_id} : {gen_text}")

        # save output file
        save_json(response.__dict__, outpath)

        # with open(outpath, "w") as f:
        #     f.write(gen_text)

def generation_parser():
    """ Build Argument Parser """
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str, required=True, help='where to save chatgpt outputs')
    parser.add_argument('--system', type=str, default='falcon-7b', help='which transformer to use')
    parser.add_argument('--dataset', type=str, default='summeval-s', help='which evaluation dataset to use')
    parser.add_argument('--score-type', type=str, default='consistency', help='which score to use of the dataset')
    parser.add_argument('--shuffle', action='store_true', help='whether to shuffling order of samples')
    parser.add_argument('--comparative', action='store_true', help='whether to do comparative evaluation')
    parser.add_argument('--prompt-id', type=str, default='c1', help='which prompt to use')
    parser.add_argument('--max-len', type=int, default=10, help='number of maximum tokens to be generated')
    return parser

# def get_default_template(prompt_num, comparative, score_type):
#     if prompt_num == 'c1':
#         assert (comparative == True) and (score_type=='consistency')
#         prompt_template = "Assess the following two summaries given the corresponding article, and determine which passage is more consistent.\n\n<context>\n\nSummary A: <summary_1>\n\nSummary B: <summary_2>\n\nWhich Summary is more consistent relative to the passage, Summary A or Summary B?\n\nAnswer:"
#     elif prompt_num == 'c2':
#         assert (comparative == True) and (score_type=='consistency')
#         prompt_template = "Determine which of the two summaries is more consistent given the following article.\n\n<context>\n\nSummary A: <summary_1>\n\nSummary B: <summary_2>\n\nWhich Summary is more consistent relative to the passage, Summary A or Summary B?\n\nAnswer:"
#     elif prompt_num == 'c3':
#         assert (comparative == True) and (score_type=='consistency')
#         prompt_template = "Assess the following two summaries given the corresponding article, and determine which passage is more consistent. Note that consistency measures how much information included in the summary is present in the source article.\n\n<context>\n\nSummary A: <summary_1>\n\nSummary B: <summary_2>\n\nWhich Summary is more consistent relative to the passage, Summary A or Summary B?\n\nAnswer:"
#     elif prompt_num == 'r1':
#         assert (comparative == False) and (score_type=='consistency')
#         prompt_template = "Score the following summary given the corresponding article with respect to consistency from 1 to 10.\n\nSummary:<summary_1>\n\nSource Article:<context>\n\nMarks:"  
#     elif prompt_num == 'r2':
#         assert (comparative == False) and (score_type=='consistency')
#         prompt_template = "Determine a score between 1 and 100 for how consistent the following summary is with respect to the article\n\nSource Article:<context>\n\nSummary:<summary_1>\n\nScore:"  
#     elif prompt_num == 'r3':
#         assert (comparative == False) and (score_type=='consistency')
#         prompt_template = "Score the following summary given the corresponding article with respect to consistency from 1 to 10. Note that consistency measures how much information included in the summary is present in the source article. 10 points indicate the summary contains only statements that are entailed by the source document\n\nSummary:<summary_1>\n\nSource Article:<context>\n\nMarks:"  
#     return prompt_template

if __name__ == "__main__":
    parser = generation_parser()
    kwargs = vars(parser.parse_args())
    for counter in range(1, 5):
        try:
            main(**kwargs)
        except openai.error.RateLimitError:
            print("openai.error.RateLimitError... #{}".format(counter))
            print("restart in 10 seconds")
            time.sleep(10)
