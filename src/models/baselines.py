import bert_score
from types import SimpleNamespace

class BertScore:
    def __init__(self):
        pass

    def bert_score(self, response:str, reference:str):
        score = bert_score.score([response], [reference], lang='en')
        score = score[0].item()
        return SimpleNamespace(output_text=score)
