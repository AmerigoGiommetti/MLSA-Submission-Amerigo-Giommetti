# General imports
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Local imports
from src.config import *
from src.utils import set_seed, train_bpe_tokenizer
from src.data_loader import load_data_final, get_dataloaders
from src.model import TransformerSeq2Seq

def train():
    set_seed(SEED)
    
    # Data loading
    train_df, val_df, test_df = load_data_final()
    
    # Train tokenizers and save them
    print("Training BPE Tokenizers...")
    code_tk = train_bpe_tokenizer(train_df['code'].tolist(), SRC_VOCAB_SIZE)
    sum_tk = train_bpe_tokenizer(train_df['summary'].tolist(), TGT_VOCAB_SIZE)
    # Create the folder if not existing
    import os
    if not os.path.exists('tokenizers'): os.makedirs('tokenizers')
    code_tk.save("tokenizers/code_bpe.json")
    sum_tk.save("tokenizers/summary_bpe.json")
    print("Saving BPE Tokenizers...")
    code_tk.save("tokenizers/code_bpe.json")
    sum_tk.save("tokenizers/summary_bpe.json")
    
    # Get Loaders
    train_loader, val_loader, _ = get_dataloaders(train_df, val_df, test_df, code_tk, sum_tk)
    
    # Model initialization (and optimizer and criterion)
    model = TransformerSeq2Seq(
        src_vocab_size=code_tk.get_vocab_size(),
        tgt_vocab_size=sum_tk.get_vocab_size()
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # 0 è <pad>
    
    # Plotting variables
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')	# Initializing validation loss to infinite
    patience = 3			# Epochs to wait for an early stopping
    patience_counter = 0		# Initializing patience counter to 0

    print(f"Starting training on {DEVICE}...")

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            src_mask = (batch_x == 0)
            tgt_pad_mask = (batch_y[:, :-1] == 0)

            output = model(batch_x, batch_y[:, :-1], src_padding_mask=src_mask, tgt_padding_mask=tgt_pad_mask)
            
            loss = criterion(output.reshape(-1, output.size(-1)), batch_y[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                src_mask = (batch_x == 0)
                tgt_pad_mask = (batch_y[:, :-1] == 0)
                output = model(batch_x, batch_y[:, :-1], src_padding_mask=src_mask, tgt_padding_mask=tgt_pad_mask)
                loss = criterion(output.reshape(-1, output.size(-1)), batch_y[:, 1:].reshape(-1))
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Early Stopping e Saving of best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if not os.path.exists('models'): os.makedirs('models') # Makes the folder if not existing
            torch.save(model.state_dict(), 'models/best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
    
    return train_losses, val_losses # For graphs


if __name__ == "__main__":
    train_losses, val_losses = train()
    print("Training completed!")

    # Epochs (x axis)
    epochs = range(1, len(train_losses) + 1)

    # perplexity
    val_perplexity = [np.exp(l) for l in val_losses]
    train_perplexity = [np.exp(l) for l in train_losses]

    # Big plot where we will put 2 subplots
    plt.figure(figsize=(12, 5))

    # Plot 1: Training & Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)

    # Plot 2: Perplexity
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_perplexity, label="Training Perplexity")
    plt.plot(epochs, val_perplexity, label="Validation Perplexity")

    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Perplexity")
    plt.legend()
    plt.grid(True)

    # Layout più pulito
    plt.tight_layout()
    plt.show()