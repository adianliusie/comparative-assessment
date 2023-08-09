import torch
import torch.nn.functional as F

from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

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
        if device == 'cuda':
            self.model = self.model.half()
        self.to(device)

    def to(self, device):
        self.device = device
        self.model.to(self.device)

    def text_response(self, input_text, top_k:int=10, do_sample:bool=False, max_new_tokens:int=None):
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
    
    def prompt_template_response(self, input_text, decoder_prefix=None):
        if not hasattr(self, 'label_ids'):
            self.set_up_prompt_classifier(decoder_prefix)

        input_text = input_text + f" {decoder_prefix}"
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        output = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
        
        vocab_logits = output.logits[:,-1]
        #self.debug_output_logits(input_text, vocab_logits)

        class_logits = vocab_logits[0, tuple(self.label_ids)]
        raw_class_probs = F.softmax(vocab_logits, dim=-1)[0, tuple(self.label_ids)]
        pred = int(torch.argmax(class_logits))

        pred_to_output = {0:'A', 1:'B'}
        output_text = pred_to_output[pred]

        return SimpleNamespace(
            output_text=output_text, 
            logits=[float(i) for i in class_logits],
            raw_probs=[float(i) for i in raw_class_probs]
        )

    def prompt_template_response_2(self, input_text, decoder_prefix=None):
        if not hasattr(self, 'label_ids'):
            self.set_up_prompt_classifier(decoder_prefix)

        input_text = input_text + f" {decoder_prefix}"
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        output = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
        
        vocab_logits = output.logits[:,-1]
        self.debug_output_logits(input_text, vocab_logits)

        class_logits = vocab_logits[0, tuple(self.label_ids)]
        raw_class_probs = F.softmax(vocab_logits, dim=-1)[0, tuple(self.label_ids)]
        pred = int(torch.argmax(class_logits))

        pred_to_output = {0:'A', 1:'B'}
        output_text = pred_to_output[pred]

        return SimpleNamespace(
            output_text=output_text, 
            logits=[float(i) for i in class_logits],
            raw_probs=[float(i) for i in raw_class_probs]
        )
    
    def debug_output_logits(self, input_text, logits):
        # Debug function to see what outputs would be
        indices = logits.topk(k=5).indices[0]
        print('\n\n', '-'*50, "\nINPUT TEXT\n", '-'*50, '\n')
        print(input_text)
        print('\n\n', '-'*50, "\nSET LABEL IDS \n", '-'*50, '\n')
        print(self.label_ids)

        print('\n\n', '-'*50, "\nTOP K IDS \n", '-'*50, '\n')
        print(indices, '\n')
        print(logits[0, indices], '\n')
        print(self.tokenizer.decode(indices), '\n')
        print('\n\n')
        import time; time.sleep(1)

    #== Setup methods =============================================================================#
    def set_up_prompt_classifier(self, decoder_prefix='Summary'):
        self.setup_label_words()
        print(f"decoder prefix is {decoder_prefix}")

    def setup_label_words(self):
        # Set Up label words
        label_words = ['A', 'B']
        label_ids = [self.tokenizer(word, add_special_tokens=False).input_ids for word in label_words]
        if any([len(i)>1 for i in label_ids]):
            print('warning: some label words are tokenized to multiple words')
        self.label_ids = [int(self.tokenizer(word, add_special_tokens=False).input_ids[0]) for word in label_words]

