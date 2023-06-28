from transformers import AutoTokenizer, AutoModelForCausalLM
from types import SimpleNamespace

class Falcon7BInterface:
    @classmethod
    def load_system(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
        cls.model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct", trust_remote_code=True)

    @classmethod
    def response(cls, input_text, top_k:int=10, do_sample:bool=False, max_new_tokens:int=None):
        if not hasattr(cls, 'tokenizer'):
            cls.load_system()

        inputs = cls.tokenizer(input_text, return_tensors="pt")
        output = cls.model.generate(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],
            top_k=top_k,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
        )

        output_text = cls.tokenizer.decode(output[0], skip_special_tokens=True)
        return SimpleNamespace(output_text=output_text)
        
class Falcon40BInterface(Falcon7BInterface):
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-40b-instruct")
    model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b-instruct", trust_remote_code=True)