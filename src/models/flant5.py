import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from types import SimpleNamespace
from functools import lru_cache
from typing import List

class FlanT5BaseInterface:
    @classmethod 
    def load_model(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        cls.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", return_dict=True)

    @classmethod
    def load_system(cls):
        cls.load_model()
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cls.model.to(cls.device)

    @classmethod
    def response(cls, input_text, top_k:int=10, do_sample:bool=False, max_new_tokens:int=None):
        if not hasattr(cls, 'tokenizer'):
            cls.load_system()
        
        inputs = cls.tokenizer(input_text, return_tensors="pt").to(cls.device)
        output = cls.model.generate(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],
            top_k=top_k,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            pad_token_id=cls.tokenizer.eos_token_id
        )

        output_tokens = output[0]
        output_text = cls.tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
        return SimpleNamespace(output_text=output_text)
        

class FlanT5LargeInterface(FlanT5BaseInterface):
    @classmethod 
    def load_model(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        cls.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", return_dict=True)
 
class FlanT5XLInterface(FlanT5BaseInterface):
    @classmethod 
    def load_model(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
        cls.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl", return_dict=True)
 
class FlanT5XXLInterface(FlanT5BaseInterface):
    @classmethod 
    def load_model(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
        cls.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl", return_dict=True)
 




