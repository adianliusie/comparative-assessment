from .openai import ChatGptInterface
from .falcon import Falcon7BInterface, Falcon40BInterface
from .falcon_probs import Falcon7BProbsInterface, Falcon40BProbsInterface
from .flant5_probs import FlanT5BaseProbsInterface, FlanT5LargeProbsInterface
from .flant5_probs import FlanT5XLProbsInterface, FlanT5XXLProbsInterface
from .flant5 import FlanT5BaseInterface, FlanT5LargeInterface
from .flant5 import FlanT5XLInterface, FlanT5XXLInterface

def load_interface(system:str):
    if system=='chatgpt':
        interface = ChatGptInterface
    elif system=='flant5-base':
        interface = FlanT5BaseInterface
    elif system=='flant5-large':
        interface = FlanT5LargeInterface
    elif system=='flant5-xl':
        interface = FlanT5XLInterface
    elif system=='flant5-xxl':
        interface = FlanT5XXLInterface
    elif system=='falcon-7b':
        interface = Falcon7BInterface
    elif system=='falcon-40b':
        interface = Falcon40BInterface
    elif system=='flant5-base-probs':
        interface = FlanT5BaseProbsInterface
    elif system=='flant5-large-probs':
        interface = FlanT5LargeProbsInterface
    elif system=='flant5-xl-probs':
        interface = FlanT5XLProbsInterface
    elif system=='flant5-xxl-probs':
        interface = FlanT5XXLProbsInterface
    elif system=='falcon-7b-probs':
        interface = Falcon7BProbsInterface
    elif system=='falcon-40b-probs':
        interface = Falcon40BProbsInterface
    
    return interface