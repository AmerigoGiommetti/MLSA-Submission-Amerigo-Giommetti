import torch
import torch.nn as nn
import math
from .config import *

class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size):
        super().__init__()

        self.src_embedding = nn.Embedding(src_vocab_size, EMBEDDING_DIM)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, EMBEDDING_DIM)
        self.positional_encoding = PositionalEncoding(EMBEDDING_DIM)

        self.transformer = nn.Transformer(
            d_model=EMBEDDING_DIM,
            nhead=8,                 
            num_encoder_layers= ENC_LAYER,    
            num_decoder_layers= DEC_LAYER,    
            dim_feedforward= FEED_FWD,
            dropout= DROPOUT,
            batch_first=True
        )

        self.fc_out = nn.Linear(EMBEDDING_DIM, tgt_vocab_size)

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None):
        src_emb = self.positional_encoding(self.src_embedding(src))
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt))

        # Mask to avoid the model being able to see in the future
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(DEVICE)

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask, # Ignores <pad> in input
            tgt_key_padding_mask=tgt_padding_mask  # Ignores <pad> in the target
        )

        return self.fc_out(out)