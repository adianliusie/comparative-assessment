import os

def get_prompt_template(prompt_id:str, score_type:str=None):
    if prompt_id.endswith('.txt'):
        prompt_template = load_prompt_from_txt(prompt_id)

    else:
        prompt_template = create_prompt(prompt_id, score_type)

    return prompt_template

def create_prompt(prompt_id, score_type):
    SCORE_TO_ADJECTIVE = {'consistency':'consistent', 'coherency':'coherent', 'fluency':'fluent', 'relevance':'relevant',
                          'naturalness':'natural', 'continuity':'good a continuation', 'engagingness':'engaging',
                          'grammar':'grammatically correct', 'overall':'good', 'semantic':None, 'good': 'good'}

    adjective = SCORE_TO_ADJECTIVE[score_type]
    if score_type == 'overall':
        score_type = 'overall quality'

    # scoring prompts
    if prompt_id   == 's1': prompt_template = f"<context>\n\nSummary: <A>\n\nProvide a score between 1 and 10 that measures the summary's {score_type}"
    elif prompt_id == 's2': prompt_template = f"Passage: <context>\n\nSummary: <A>\n\nScore the response between 1 and 10 based on how {adjective} the summary is"
    elif prompt_id == 's3': prompt_template = f"<context>\n\nSummary: <A>\n\nScore the summary between 1 and 100 in terms of {score_type}"
    elif prompt_id == 's4': prompt_template = f"Score the following summary with respect to {score_type} from 1 to 10. \n\nPassage:<context>\n\nSummary:<A>"
    elif prompt_id == 's5': prompt_template = f"Passage: <context>\n\nSummary: <A>\n\nScore the summary between 1 and 10 based on how well it summarizes the passage"

    # comparative prompts
    elif prompt_id == 'c1': prompt_template = f"<context>\n\nSummary A: <A>\n\nSummary B: <B>\n\nWhich Summary is more {adjective}, Summary A or Summary B?"
    elif prompt_id == 'c2': prompt_template = f"Passage:\n<context>\n\nSummary A: <A>\n\nSummary B: <B>\n\nWhich Summary is more {adjective} relative to the passage, Summary A or Summary B?"
    elif prompt_id == 'c3': prompt_template = f"<context>\n\nSummary A: <A>\n\nSummary B: <B>\n\nWhich Summary is more {adjective}, Summary A or Summary B? Output either Summary A or Summary B"
    elif prompt_id == 'c4': prompt_template = f"Assess the following two summaries given the corresponding passage, and determine which summary is more {adjective}.\n\nPassage:\n<context>\n\nSummary A: <A>\n\nSummary B: <B>\n\nWhich Summary is more {adjective}, Summary A or Summary B?"

    #Prompts that don't take the context into account
    elif prompt_id == 'wi-s1': prompt_template = f"<A>\n\nProvide a score between 1 and 10 that measures the text's quality"
    elif prompt_id == 'wi-s2': prompt_template = f"Text:<A>\n\nScore the above text between 1 and 10"
    elif prompt_id == 'wi-c1': prompt_template = f"Text A: <A>\n\nText B: <B>\n\nWhich text is better, Text A or Text B?"
    elif prompt_id == 'wi-c2': prompt_template = f"Text A: <A>\n\nText B: <B>\n\nWhich text is of a higher quality, Text A or Text B? Output either Text A or Text B."

    # special prompts for particular properties
    elif prompt_id == 'sem-s1': prompt_template = f"<context>\n\nResponse: <A>\n\nProvide a score between 1 and 10 that measures how well the response captures the information of the semantic triples"
    elif prompt_id == 'sem-s2': prompt_template = f"<context>\n\nResponse: <A>\n\Score the response out of 10 based on how well the response relates with the semantic triples"
    elif prompt_id == 'sem-c1': prompt_template = f"<context>\n\nResponse A: <A>\n\nResponse B: <B>\n\nWhich Response is best aligns with the given semantic triples, Response A or Response B?"
    elif prompt_id == 'sem-c2': prompt_template = f"<context>\n\nResponse A: <A>\n\nResponse B: <B>\n\nWhich Response captures the most information from the semantic triples, Response A or Response B?"

    prompt_template = prompt_template.replace('is more good', 'is better')
    return prompt_template

def load_prompt_from_txt(prompt_path:str):
    if os.path.isfile(prompt_path):
        with open(prompt_path, 'r') as f:
            prompt_template = f.read()

    return prompt_template
