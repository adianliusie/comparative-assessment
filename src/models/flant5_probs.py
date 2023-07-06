import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from types import SimpleNamespace
from functools import lru_cache
from typing import List

class FlanT5BaseProbsInterface:
    @classmethod 
    def load_model(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        cls.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", return_dict=True)

    @classmethod
    def load_system(cls):
        cls.load_model()
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cls.model.to(cls.device)
        cls.setup_label_words()
        cls.response_prefix = 'Summary'
        cls.decoder_input_ids = cls.get_decoder_ids()

    @classmethod
    def setup_label_words(cls):
        # Set Up label words
        label_words = [' A', ' B', 'both']
        label_ids = [cls.tokenizer(word, add_special_tokens=False).input_ids for word in label_words]
        if any([len(i)>1 for i in label_ids]):
            print('warning: some label words are tokenized to multiple words')
        cls.label_ids   = [int(cls.tokenizer(word, add_special_tokens=False).input_ids[0]) for word in label_words]

    @classmethod
    def response(cls, input_text, top_k:int=10, do_sample:bool=False, max_new_tokens:int=None):
        if not hasattr(cls, 'tokenizer'):
            cls.load_system()

        if cls.response_prefix:
            input_text = input_text + cls.response_prefix

        #print(input_text)
        inputs = cls.tokenizer(input_text, return_tensors="pt").to(cls.device)
        output = cls.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=cls.decoder_input_ids
        )
        
        vocab_logits = output.logits[:,-1]
        class_logits = vocab_logits[0, tuple(cls.label_ids)]

        # Debug function to see what outputs would be
        indices = vocab_logits.topk(k=5).indices[0]
        # print(indices)
        # print(cls.label_ids)
        # print(cls.tokenizer.decode(indices))

        raw_class_probs = F.softmax(vocab_logits, dim=-1)[0, tuple(cls.label_ids)]
        pred = int(torch.argmax(class_logits))

        pred_to_output = {0:'Summary A', 1:'Summary B', 2:'Neither'}
        output_text = pred_to_output[pred]

        return SimpleNamespace(
            output_text=output_text, 
            logits=[float(i) for i in class_logits],
            raw_probs=[float(i) for i in raw_class_probs]
        )

    @classmethod
    def get_decoder_ids(cls, bsz=1) -> List[int]:
        if cls.response_prefix:
            # repeat template bsz times
            decoder_input_ids = cls.tokenizer(
                [cls.response_prefix for _ in range(bsz)], 
                return_tensors="pt",
            ).input_ids
            
            # add start token
            decoder_input_ids = cls.model._shift_right(decoder_input_ids)
        else:
            # set input to start of sentence token
            decoder_input_ids = cls.model.config.decoder_start_token_id * torch.ones(bsz, 1, dtype=torch.long)

        return decoder_input_ids.to(cls.device)

class FlanT5LargeProbsInterface(FlanT5BaseProbsInterface):
    @classmethod 
    def load_model(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        cls.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", return_dict=True)
 
class FlanT5XLProbsInterface(FlanT5BaseProbsInterface):
    @classmethod 
    def load_model(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
        cls.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl", return_dict=True)
 
class FlanT5XXLProbsInterface(FlanT5BaseProbsInterface):
    @classmethod 
    def load_model(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
        cls.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl", return_dict=True)
 




