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
        # inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

         # quick fix to make input text being short than 4k
        LLAMA_MAX_LEN = 4000 # just slightly below 4096 for generated tokens
        input_pair = input_text.split("\n\nSummary A: ")
        input_pair[1] = "\n\nSummary A: " + input_pair[1]
        inputs_1 = self.tokenizer(input_pair[1], return_tensors="pt", add_special_tokens=False) # will manually add <s> (ID=1)
        inputs_1_len = inputs_1.input_ids.shape[1]
        inputs_0 = self.tokenizer(input_pair[0], return_tensors="pt", truncation=True, max_length=LLAMA_MAX_LEN-inputs_1_len, add_special_tokens=False) # will manually add <s> (ID=1)
        input_ids_concat = torch.cat([torch.tensor([[1]]), inputs_0.input_ids, inputs_1.input_ids], dim=-1)
        token_type_ids_concat = torch.cat([torch.tensor([[0]]), inputs_0.token_type_ids, inputs_1.token_type_ids], dim=-1).to(self.device)
        attention_mask_concat = torch.cat([torch.tensor([[1]]), inputs_0.attention_mask, inputs_1.attention_mask], dim=-1).to(self.device)

        output = self.model.generate(
            input_ids=input_ids_concat,
            attention_mask=attention_mask_concat,
            top_k=top_k,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output_tokens = output[0]
        input_tokens = input_ids_concat[0]
        new_tokens = output_tokens[len(input_tokens):]
        assert torch.equal(output_tokens[:len(input_tokens)], input_tokens)

        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return SimpleNamespace(output_text=output_text)

    def to(self, device):
        self.device = device
        self.model.to(self.device)
