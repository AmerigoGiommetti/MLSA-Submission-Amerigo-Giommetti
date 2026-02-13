import random
import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Function to train the tokenizer on sub-words
def train_bpe_tokenizer(data, vocab_size):
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"]
    )
    tokenizer.train_from_iterator(data, trainer)
    return tokenizer

# Encoding function
def encode_bpe(text, tokenizer, max_len):
    encoded = tokenizer.encode(text)
    ids = encoded.ids
    ids = [1] + ids + [2] # Adding <sos> and <eos>
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids)) # Padding
    else:
        ids = ids[:max_len-1] + [2]       # Truncating
    return ids

def save_tokenizer(tokenizer, path):
    tokenizer.save(path)

def load_tokenizer(path):
    return Tokenizer.from_file(path)