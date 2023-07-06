import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from types import SimpleNamespace

class Falcon7BInterface:
    @classmethod
    def load_system(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
        cls.model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct", trust_remote_code=True)
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
        input_tokens = inputs.input_ids[0]

        new_tokens = output_tokens[len(input_tokens):]
        assert torch.equal(output_tokens[:len(input_tokens)], input_tokens)
        output_text = cls.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return SimpleNamespace(output_text=output_text)
        
class Falcon40BInterface(Falcon7BInterface):
    @classmethod
    def load_system(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-40b-instruct")
        cls.model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b-instruct", trust_remote_code=True) 
        cls.device = 'cpu'