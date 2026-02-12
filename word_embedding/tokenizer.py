import re

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preproccessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preproccessed = [item.strip() for item in preproccessed if item.strip()]
        ids = [self.str_to_int[s] for s in preproccessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preproccessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preproccessed = [item.strip() for item in preproccessed if item.strip()]
        preproccessed = [item if item in self.str_to_int else "<|unk|>" for item in preproccessed]
        ids = [self.str_to_int[s] for s in preproccessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text