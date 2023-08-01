from .flant5 import FlanT5Interface
from .llama import Llama2Interface
from .openai import ChatGptInterface
from .baselines import BertScore

def load_interface(system:str, device:str):
    if system in ['flant5-base', 'flant5-large', 'flant5-xl', 'flant5-xxl']:
        interface = FlanT5Interface(system, device)
    elif system in ['llama2-7b', 'llama2-13b', 'llama2-7b-chat', 'llama2-13b-chat']:
        interface = Llama2Interface(system, device)
    elif system in ['chatgpt']:
        interface = ChatGptInterface()
    elif system == 'bertscore':
        interface = BertScore()
    return interface