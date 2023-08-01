import torch

from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_URLS = {
    'llama2-7b':'meta-llama/Llama-2-7b-hf',
    'llama2-13b':'meta-llama/Llama-2-13b-hf',
    'llama2-7b-chat':'meta-llama/Llama-2-7b-chat-hf',
    'llama2-13b-chat':'meta-llama/Llama-2-13b-chat-hf',
}

class Llama2Interface:
    def __init__(self, system, device=None):
        system_url = MODEL_URLS[system]
        self.tokenizer = AutoTokenizer.from_pretrained(system_url)
        self.model = AutoModelForCausalLM.from_pretrained(system_url)
        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)

    def text_response(self, input_text, top_k:int=10, do_sample:bool=False, max_new_tokens:int=None):
        input_text = input_text + '\n\nAnswer:'
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        #print(input_text)
        #import time; time.sleep(2)
        output = self.model.generate(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],
            top_k=top_k,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id
        )

        output_tokens = output[0]
        
        input_tokens = inputs.input_ids[0]
        new_tokens = output_tokens[len(input_tokens):]
        assert torch.equal(output_tokens[:len(input_tokens)], input_tokens)
        
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return SimpleNamespace(output_text=output_text)
    
    def to(self, device):
        self.device = device
        self.model.to(self.device)

