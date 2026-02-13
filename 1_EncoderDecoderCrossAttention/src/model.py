import torch.nn as nn
import torch
import random
from src.config import HIDDEN_DIM, EMBEDDING_DIM

# Attention class
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Since the encoder is bidirectional, encoder_outputs has HIDDEN_DIM * 2
        # The decoder state (hidden) has HIDDEN_DIM.
        # Total input to the linear layer is (HIDDEN_DIM * 2) + HIDDEN_DIM = HIDDEN_DIM * 3
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask ):
        # hidden: [1, batch_size, HIDDEN_DIM] (current decoder state)
        # encoder_outputs: [batch_size, src_len, HIDDEN_DIM * 2] (from Bi-Encoder)

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # Repeat decoder hidden state for every source token
        # hidden becomes: [batch_size, src_len, HIDDEN_DIM]
        hidden = hidden.repeat(src_len, 1, 1).transpose(0, 1)

        # Calculate alignment energy
        # Concatenate decoder state with bidirectional encoder outputs
        # Shape: [batch_size, src_len, HIDDEN_DIM * 3]
        combined = torch.cat((hidden, encoder_outputs), dim=2)

        energy = torch.tanh(self.attn(combined)) # [batch_size, src_len, HIDDEN_DIM]

        # attention: [batch_size, src_len]
        attention = self.v(energy).squeeze(2)

        # Apply mask: where mask is False (padding), we fill with a very small number
        # This makes the softmax result practically 0 for those tokens
        # Namely ignoring padding tokens in the attention mechanism
        attention = attention.masked_fill(mask == 0, -1e10)

        # Softmax ensures weights sum to 1
        return torch.softmax(attention, dim=1)

# Bi-directional encoder class
class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)

        # Set bidirectional=True to capture context from both directions
        # This doubles the output dimension of encoder_outputs to HIDDEN_DIM * 2
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM,
                            batch_first=True,
                            bidirectional=True)

        # Linear layers to bridge the gap between Bi-Encoder and Uni-Decoder
        # We transform the concatenated forward and backward states back to HIDDEN_DIM
        self.fc_hidden = nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM)
        self.fc_cell = nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM)

    def forward(self, x):
        # x: [batch_size, seq_len]
        emb = self.embedding(x)

        # encoder_outputs: [batch, seq_len, HIDDEN_DIM * 2]
        # h/c: [num_layers * num_directions, batch, HIDDEN_DIM]
        encoder_outputs, (h, c) = self.lstm(emb)

        # Concatenate the final forward (h[-2]) and backward (h[-1]) hidden states
        # then pass through the linear layer and than activation
        h_combined = torch.cat((h[-2,:,:], h[-1,:,:]), dim=1)
        c_combined = torch.cat((c[-2,:,:], c[-1,:,:]), dim=1)

        h_final = torch.tanh(self.fc_hidden(h_combined))
        c_final = torch.tanh(self.fc_cell(c_combined))

        # Returns outputs for attention, and resized hidden/cell for decoder start
        # h_final.unsqueeze(0) shape: [1, batch_size, HIDDEN_DIM]
        return encoder_outputs, h_final.unsqueeze(0), c_final.unsqueeze(0)

# Decoder class
class Decoder(nn.Module):
    def __init__(self, vocab_size, attention):
        super().__init__()
        self.attention = attention
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        # L'input dell'LSTM ora riceve Embedding + Vettore di Contesto
        self.lstm = nn.LSTM(HIDDEN_DIM * 2 + EMBEDDING_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM * 3 + EMBEDDING_DIM, vocab_size)

    def forward(self, x, hidden, cell, encoder_outputs, mask):
        # x: [batch_size, 1]
        emb = self.embedding(x) # [batch_size, 1, emb_dim]

        # Calculating attention weights but passing mask to ignore padding
        a = self.attention(hidden, encoder_outputs, mask)
        a = a.unsqueeze(1)

        # Creating context vector
        weighted = torch.bmm(a, encoder_outputs) # [batch_size, 1, hidden_dim]

        # Input and context concatenation for LSTM
        rnn_input = torch.cat((emb, weighted), dim=2)
        out, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))

        # Final prediction using out, context and original embedding
        prediction = self.fc(torch.cat((out, weighted, emb), dim=2))

        return prediction.squeeze(1), hidden, cell

# Encoder-Decoder seq2seq class (with attention)
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def create_mask(self, src):
        # mask is 1 for real tokens and 0 for <pad> tokens
        mask = (src != 0)
        return mask

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # Getting all outputs from the encoder
        encoder_outputs, hidden, cell = self.encoder(src)

        # Create padding mask based on the source input
        mask = self.create_mask(src)

        input = trg[:, 0]

        for t in range(1, trg_len):
            # Passing encoder_outputs at each step
            output, hidden, cell = self.decoder(input.unsqueeze(1), hidden, cell, encoder_outputs, mask)
            outputs[:, t, :] = output

            top1 = output.argmax(1)
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else top1

        return outputs