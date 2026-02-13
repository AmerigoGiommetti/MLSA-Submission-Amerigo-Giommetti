import random
import numpy as np
import torch
import re
from collections import Counter

# Special tokens definition
SPECIAL_TOKENS = {
    "<pad>": 0,       # For length normalization
    "<sos>": 1,       # Start of sequence
    "<eos>": 2,       # End of sequence
    "<unk>": 3        # For words not found in the vocabulary
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def tokenize(sentence):
    if not isinstance(sentence, str):
        return []
    sentence = sentence.lower()
    # Divinig word and special characters
    return re.findall(r"\w+|[^\w\s]", sentence)

def build_vocab(sentences, max_size):
    # Initialize vocab with special tokens
    vocab = dict(SPECIAL_TOKENS)
    
    # Count all tokens in the dataset
    word_counts = Counter()
    for sent in sentences:
        word_counts.update(tokenize(sent))
    
    # Get the most common words, leaving room for the 4 special tokens
    most_common = word_counts.most_common(max_size - len(SPECIAL_TOKENS))
    
    # Add the most common words to our dictionary
    for word, count in most_common:
        if word not in vocab:
            vocab[word] = len(vocab)
            
    return vocab

def encode(sentence, vocab, max_len):
    tokens = tokenize(sentence)
    ids = [vocab["<sos>"]] + [vocab.get(tok, vocab["<unk>"]) for tok in tokens] + [vocab["<eos>"]]
    # Padding
    return ids[:max_len] + [vocab["<pad>"]] * max(0, max_len - len(ids))