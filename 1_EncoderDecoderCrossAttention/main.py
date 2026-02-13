import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from src.config import *
from src.utils import set_seed, build_vocab
from src.data_loader import load_data_final, get_loaders
from src.model import Encoder, Decoder, Attention, Seq2Seq

def train():
    set_seed(SEED)
    
    # Data loading
    train_df, val_df, test_df = load_data_final()
    
    # Vocab building
    special_tokens = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    code_vocab = build_vocab(train_df['code'], MAX_VOCAB_SIZE)
    summary_vocab = build_vocab(train_df['summary'], MAX_VOCAB_SIZE)

    # Vocab saving in its vocabulary folder
    if not os.path.exists('vocabulary'):
        os.makedirs('vocabulary')
        print("Created vocabulary folder")

    with open('vocabulary/code_vocab.pkl', 'wb') as f:
        pickle.dump(code_vocab, f)
    with open('vocabulary/summary_vocab.pkl', 'wb') as f:
        pickle.dump(summary_vocab, f)
    print("Saved vocabulary in 'vocabulary/'.")

    # Getting data loaders
    train_loader, val_loader, _ = get_loaders(train_df, val_df, test_df, code_vocab, summary_vocab)

    # Model Init
    attn = Attention(HIDDEN_DIM)
    enc = Encoder(len(code_vocab)).to(DEVICE)
    dec = Decoder(len(summary_vocab), attn).to(DEVICE)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # ignora il padding

    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(NUM_EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        epoch_train_loss = 0
        
        # Use tqdm to monitor batches
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for batch_x, batch_y in pbar:
            optimizer.zero_grad()
            outputs = model(batch_x, batch_y)
            output_dim = outputs.shape[-1]

            loss = criterion(
                outputs[:, 1:, :].reshape(-1, output_dim),
                batch_y[:, 1:].reshape(-1)
            )

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- VALIDATION PHASE ---
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x, batch_y, teacher_forcing_ratio=0.0)
                output_dim = outputs.shape[-1]
                loss = criterion(
                    outputs[:, 1:, :].reshape(-1, output_dim),
                    batch_y[:, 1:].reshape(-1)
                )
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1} Results -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Saving model weights at each step
    if not os.path.exists('models'): os.makedirs('models') # Makes the folder if not existing
    torch.save(model.state_dict(), "models/summarization.pt")
    print("Saved model: models/summarization.pt")
    
    return train_losses, val_losses

def plot_training_results(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    train_perplexity = [np.exp(l) for l in train_losses]
    val_perplexity = [np.exp(l) for l in val_losses]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(epochs, train_losses, 'royalblue', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'orange', label='Validation Loss', linewidth=2, linestyle='--')
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_perplexity, 'crimson', label='Train Perplexity', linewidth=2)
    ax2.plot(epochs, val_perplexity, 'darkred', label='Val Perplexity', linewidth=2, linestyle='--')
    ax2.set_title('Perplexity per Epoch', fontsize=14)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Perplexity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    t_losses, v_losses = train()
    plot_training_results(t_losses, v_losses)