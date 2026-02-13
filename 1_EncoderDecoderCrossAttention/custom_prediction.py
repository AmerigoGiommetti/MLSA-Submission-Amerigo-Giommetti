import sys
import torch
import pickle
import os
from src.model import Encoder, Decoder, Attention, Seq2Seq
from src.config import DEVICE, HIDDEN_DIM, EMBEDDING_DIM, MAX_CODE_LEN, MAX_SUMMARY_LEN
from src.utils import tokenize, encode

def decode_string(ids, inv_vocab):
    """
    Converts a list of token IDs back into a readable string.
    """
    words = []
    for idx in ids:
        word = inv_vocab.get(idx, "<unk>")
        if word == "<eos>": 
            break
        if word not in ["<sos>", "<pad>"]:
            words.append(word)
    return " ".join(words)

def summarize_custom_code(model, code_text, code_vocab, summary_vocab):
    """
    Main inference logic using the Bidirectional Seq2Seq model with Attention.
    """
    model.eval()
    inv_summary_vocab = {v: k for k, v in summary_vocab.items()}
    
    # Preprocess and Encode input
    # We use the utility from src.utils to ensure consistency with training
    input_ids = encode(code_text, code_vocab, MAX_CODE_LEN)
    src_tensor = torch.tensor(input_ids).unsqueeze(0).to(DEVICE)
    
    # Create the Mask for Attention
    mask = (src_tensor != 0) 

    with torch.no_grad():
        # Pass through Bidirectional Encoder
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
        
        # Prepare Decoder input (<sos>)
        current_input = torch.tensor([[summary_vocab["<sos>"]]]).to(DEVICE)
        result_ids = []
        
        # Greedy Generation Loop
        for _ in range(MAX_SUMMARY_LEN):
            # The decoder now requires encoder_outputs and mask for the attention mechanism
            output, hidden, cell = model.decoder(
                current_input, 
                hidden, 
                cell, 
                encoder_outputs, 
                mask
            )
            
            # Pick the most probable next word
            top_idx = output.argmax(1).item()
            
            if top_idx == summary_vocab["<eos>"]:
                break
                
            result_ids.append(top_idx)
            current_input = torch.tensor([[top_idx]]).to(DEVICE)
            
    return decode_string(result_ids, inv_summary_vocab)

def main():
    # Check if code snippet is provided as command line argument
    if len(sys.argv) < 2:
        print("\n[Error] No code snippet provided.")
        print('Usage: python custom_prediction.py "def add(a, b): return a + b"')
        return

    my_code = sys.argv[1]

    # Loading Vocabularies from the 'vocabulary' folder
    try:
        with open("vocabulary/code_vocab.pkl", "rb") as f:
            code_vocab = pickle.load(f)
        with open("vocabulary/summary_vocab.pkl", "rb") as f:
            summary_vocab = pickle.load(f)
    except FileNotFoundError:
        print("[Error] Vocabulary files not found in 'vocabulary/'. Did you run main.py?")
        return

    # Initializing the Full Architecture
    attn = Attention(HIDDEN_DIM)
    encoder = Encoder(len(code_vocab)).to(DEVICE)
    decoder = Decoder(len(summary_vocab), attn).to(DEVICE)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    # Loading model weights
    model_path = 'models/summarization.pt'
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
        except Exception as e:
            print(f"[Error] Failed to load model weights: {e}")
            return
    else:
        print(f"[Error] Model weights '{model_path}' not found.")
        return

    # Run Inference
    print("\n" + "="*50)
    print(f"{'SEQ2SEQ CUSTOM PREDICTION':^50}")
    print("="*50)
    print(f"INPUT CODE:\n{my_code}")
    print("-" * 50)
    
    summary = summarize_custom_code(model, my_code, code_vocab, summary_vocab)
    
    print(f"GENERATED SUMMARY:\n> {summary}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()