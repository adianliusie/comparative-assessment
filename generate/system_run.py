import os
import argparse
import pickle
import random
import re
import openai

from tqdm import tqdm
from datetime import datetime

from src.data_handler import DataHandler
from src.utils.general import save_json, load_json
from src.models import load_interface
from src.prompts.load_prompt import get_prompt_template
from src.utils.post_processing import save_combined_json, delete_leftover_files

import time
# python system_run.py --output-path output_texts/falcon-7b/summeval-consistency/prompt_c1 --system falcon-7b --dataset summeval-s --score consistency --prompt-id c1 --shuffle --comparative
# python system_run.py --output-path output_texts/falcon-7b/summeval-consistency/prompt_c1 --system falcon-7b --dataset summeval-s --score-type consistency --prompt-id comparative-1 --shuffle --comparative
# python system_run.py --output-path output_texts2/chatgpt/summeval-relevance/comparative-1 --system chatgpt --dataset summeval-s --score-type relevance --prompt-id comparative-1 --shuffle --comparative

def main(
    system:str,
    output_path:str,
    dataset:str='sumevall',
    score_type:str='consistency',
    prompt_id:int=None,
    shuffle:bool=True,
    comparative=False,
    max_len=None,
    device=None,
    probs=False,
    num_comparisons=None
):
    print(output_path)

    #load prompt from default, or choose your own prompt
    if prompt_id:
        assert bool(re.search(r'c\d+', prompt_id)) == comparative

        prompt_template = get_prompt_template(prompt_id, score_type)
        if dataset == 'topicalchat':
            prompt_template = prompt_template.replace('Summary', 'Response')
            prompt_template = prompt_template.replace('summary', 'response')
            prompt_template = prompt_template.replace('Passage', 'Dialogue')
            prompt_template = prompt_template.replace('passage', 'dialogue')
    else:
        prompt_template = None

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

    # select the model to run the inputs through
    interface = load_interface(system=system, device=device)

    # set decoder_prefix (only used for prob mode)
    decoder_prefix='Summary'
    if (dataset == 'topicalchat') and (probs):
        decoder_prefix='Response'
    elif (dataset in ['wi-train', 'wi-dev']) and (probs):
        decoder_prefix='Text'

    # save experiment settings
    info = {
        'prompt_id':prompt_id,
        'prompt':prompt_template,
        'system':system,
        'dataset':dataset,
        'score_type':score_type,
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

    # process all inputs to chatgpt
    for k, idx in enumerate(ids):
        # break early if sufficient comparisons have been done
        if num_comparisons and (k + len(done_ids) > num_comparisons):
            break

        ex = proc_inputs[idx]
        outpath = f"{system_output_path}/{ex.ex_id}.json"

        # skip outputs already computed
        if (os.path.isfile(outpath)) or (ex.ex_id in done_ids):
            continue

        # get text response
        # print(ex.input_text)

        if system == 'bertscore':
            response = interface.bert_score(
                response=ex.response,
                reference=ex.reference
            )

        elif probs:
            response = interface.prompt_template_response(
                input_text=ex.input_text,
                decoder_prefix=decoder_prefix
            )

        else:
            if comparative:
                mid_string = "\n\nSummary A: "
            else:
                mid_string="Summary: "

            response = interface.text_response(
                input_text=ex.input_text,
                do_sample=False,
                max_new_tokens=max_len,
                mid_string=mid_string # change this according to the prompt -- Podcast dataset & LLaMA only
            )

        #print(response)
        #import time; time.sleep(2)

        # get and print generated text
        gen_text = response.output_text
        current_time = datetime.now().strftime('%H:%M:%S')
        print(f"[{current_time}] {ex.ex_id} : {gen_text}")

        # save output file
        save_json(response.__dict__, outpath)

        # with open(outpath, "w") as f:
        #     f.write(gen_text)

    save_combined_json(system_output_path)
    delete_leftover_files(system_output_path)

def generation_parser():
    """ Build Argument Parser """
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--output-path', type=str, required=True, help='where to save chatgpt outputs')

    parser.add_argument('--system', type=str, default='flant5-large', help='which transformer to use')
    parser.add_argument('--probs', type="bool", default=True, help='whether prompt templates should be used')
    parser.add_argument('--prompt-id', type=str, default=None, help='which prompt to use')

    parser.add_argument('--dataset', type=str, default='summeval', help='which evaluation dataset to use')
    parser.add_argument('--score-type', type=str, default='consistency', help='which score to use of the dataset')

    parser.add_argument('--max-len', type=int, default=10, help='number of maximum tokens to be generated')
    parser.add_argument('--num-comparisons', type=int, default=None, help='number of comparisons to do for the dataset')

    parser.add_argument('--device', type=str, default=None, help='device to run experiments')

    parser.add_argument('--shuffle', type="bool", default=True, help='whether to shuffling order of samples')
    parser.add_argument('--comparative', type="bool", default=True, help='whether to do comparative evaluation')

    return parser

if __name__ == "__main__":
    parser = generation_parser()
    kwargs = vars(parser.parse_args())
    main(**kwargs)
