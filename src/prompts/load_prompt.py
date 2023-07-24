import os

def get_prompt_template(prompt_id:str, score_type:str=None):
    if prompt_id.endswith('.txt'):
        prompt_template = load_prompt_from_txt(prompt_id)

    else:
        prompt_template = create_prompt(prompt_id, score_type)

    return prompt_template

def create_prompt(prompt_id, score_type):
    SCORE_TO_ADJECTIVE = {'consistency':'consistent', 'coherency':'coherent', 'fluency':'fluent', 'relevance':'relevant',
                          'naturalness':'natural', 'continuity':'good a continuation', 'engagingness':'engaging'}
    
    adjective = SCORE_TO_ADJECTIVE[score_type]

    # scoring prompts
    if prompt_id   == 's1': prompt_template = f"<context>\n\nSummary: <A>\n\nProvide a score between 1 and 10 that measures the summary's {score_type}"
    elif prompt_id == 's2': prompt_template = f"Passage: <context>\n\nSummary: <A>\n\nScore the response between 1 and 10 based on how {adjective} the summary is"
    elif prompt_id == 's3': prompt_template = f"<context>\n\nSummary: <A>\n\nScore the summary between 1 and 100 in terms of {score_type}"
    elif prompt_id == 's4': prompt_template = f"Score the following summary with respect to {score_type} from 1 to 10. \n\nPassage:<context>\n\nSummary:<A>"  

    # comparative prompts
    elif prompt_id == 'c1': prompt_template = f"<context>\n\nSummary A: <A>\n\nSummary B: <B>\n\nWhich Summary is more {adjective}, Summary A or Summary B?"
    elif prompt_id == 'c2': prompt_template = f"Passage:\n<context>\n\nSummary A: <A>\n\nSummary B: <B>\n\nWhich Summary is more {adjective} relative to the passage, Summary A or Summary B?"
    elif prompt_id == 'c3': prompt_template = f"<context>\n\nSummary A: <A>\n\nSummary B: <B>\n\nWhich Summary is more {adjective}, Summary A or Summary B? Output either Summary A or Summary B"
    elif prompt_id == 'c4': prompt_template = f"Assess the following two summaries given the corresponding passage, and determine which summary is more {adjective}.\n\nPassage:\n<context>\n\nSummary A: <A>\n\nSummary B: <B>\n\nWhich Summary is more {adjective}, Summary A or Summary B?"

    prompt_template = prompt_template.replace('is more good', 'is better')

    return prompt_template

def load_prompt_from_txt(prompt_path:str):
    if os.path.isfile(prompt_path):
        with open(prompt_path, 'r') as f:
            prompt_template = f.read()

    return prompt_template
