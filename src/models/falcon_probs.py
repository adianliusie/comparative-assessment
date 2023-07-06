import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from types import SimpleNamespace

class Falcon7BProbsInterface:
    @classmethod
    def load_system(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
        cls.model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct", trust_remote_code=True)
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cls.model.to(cls.device)
        cls.setup_label_words()
        cls.response_prefix = 'Summary'

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
        )
        
        vocab_logits = output.logits[:,-1]
        class_logits = vocab_logits[0, tuple(cls.label_ids)]

        # Debug function to see what outputs would be
        # indices = vocab_logits.topk(k=5).indices[0]
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

class Falcon40BProbsInterface(Falcon7BProbsInterface):
    @classmethod
    def load_system(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-40b-instruct")
        cls.model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b-instruct", trust_remote_code=True) 
        cls.device = 'cpu'
        cls.setup_label_words()
        cls.response_prefix = 'Summary'

