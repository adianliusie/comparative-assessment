def get_prompt_template(prompt_id, score_type):
    SCORE_TO_ADJECTIVE = {'consistency':'consistent', 'coherency':'coherent', 'fluency':'fluent', 'relevance':'relevant'}
    if score_type:
        assert score_type in ['consistency', 'coherency', 'fluency', 'relevance']
        adjective = SCORE_TO_ADJECTIVE[score_type]

    if prompt_id == 'scoring-1':
        prompt_template = f"<context>\n\nSummary: <summary_1>\n\Provide a score between 1 and 10 that measures the summaries' {adjective}\n\nAnswer:"

    elif prompt_id == 'scoring-2':
        prompt_template = f"<context>\n\nSummary: <summary_1>\n\n Score the summary between 1 and 10 in terms of {adjective}\n\nScore:"

    elif prompt_id == 'scoring-3':
        prompt_template = f"<context>\n\nSummary: <summary_1>\n\n Score the summary between 1 and 100 in terms of {adjective}\n\nScore:"

    elif prompt_id == 'scoring-4':
        prompt_template = f"Score the following summary with respect to {adjective} from 1 to 10. \n\nSource Article:<context>\n\nSummary:<summary_1>\n\nMarks:"  

    elif prompt_id == 'comparative-1':
        prompt_template = f"<context>\n\nSummary A: <summary_1>\n\nSummary B: <summary_2>\n\nWhich Summary is more {adjective}, Summary A or Summary B?\n\nAnswer:"

    elif prompt_id == 'comparative-2':
        prompt_template = f"<context>\n\nSummary A: <summary_1>\n\nSummary B: <summary_2>\n\nWhich Summary is more {adjective} relative to the passage, Summary A or Summary B?\n\nAnswer:"

    elif prompt_id == 'comparative-3':
        prompt_template = f"<context>\n\nSummary A: <summary_1>\n\nSummary B: <summary_2>\n\nWhich Summary is more {adjective} relative to the passage, Summary A or Summary B?\n\nOutput either Summary A or Summary B\n\nAnswer:"

    elif prompt_id == 'comparative-4':
        prompt_template = f"Assess the following two summaries given the corresponding article, and determine which passage is more {adjective}.\n\n<context>\n\nSummary A: <summary_1>\n\nSummary B: <summary_2>\n\nWhich Summary is more {adjective}, Summary A or Summary B?\n\nAnswer:"

    return prompt_template

