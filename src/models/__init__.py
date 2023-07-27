from .flant5 import FlanT5Interface
from .openai import ChatGptInterface
from .baselines import BertScore

def load_interface(system:str, device:str):
    if system in ['flant5-base', 'flant5-large', 'flant5-xl', 'flant5-xxl']:
        interface = FlanT5Interface(system, device)
    elif system in ['chatgpt']:
        interface = ChatGptInterface()
    elif system == 'bertscore':
        interface = BertScore()
    return interface