# from .openai import ChatGptInterface
# from .falcon import Falcon7BInterface, Falcon40BInterface
# from .falcon_probs import Falcon7BProbsInterface, Falcon40BProbsInterface
# from .flant5_probs import FlanT5BaseProbsInterface, FlanT5LargeProbsInterface
# from .flant5_probs import FlanT5XLProbsInterface, FlanT5XXLProbsInterface
# from .flant5 import FlanT5BaseInterface, FlanT5LargeInterface
# from .flant5 import FlanT5XLInterface, FlanT5XXLInterface
from .flant5 import FlanT5Interface
from .openai import ChatGptInterface

def load_interface(system:str, device:str):
    if system in ['flant5-base', 'flant5-large', 'flant5-xl', 'flant5-xxl']:
        interface = FlanT5Interface(system, device)
    elif system in ['chatgpt']:
        interface = ChatGptInterface()
    return interface