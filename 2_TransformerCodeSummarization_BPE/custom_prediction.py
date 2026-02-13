import sys
import torch
from src.utils import load_tokenizer
from src.model import TransformerSeq2Seq
from src.config import DEVICE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE
from evaluate import summarize_code_transformer

def main():
    # Checking the presence of the arguments
    if len(sys.argv) < 2:
        print("Errore: Nessun codice fornito.")
        print('Utilizzo: python custom_prediction.py "def nome_funzione(): ..."')
        return

    # Taking the second argument as code input
    my_code = sys.argv[1]

    # Loading tokens saved during training
    try:
        code_tk = load_tokenizer("tokenizers/code_bpe.json")
        sum_tk = load_tokenizer("tokenizers/summary_bpe.json")
    except Exception as e:
        print(f"Errore nel caricamento dei tokenizer: {e}")
        return

    # Initializing model
    model = TransformerSeq2Seq(
        src_vocab_size=code_tk.get_vocab_size(), 
        tgt_vocab_size=sum_tk.get_vocab_size()
    ).to(DEVICE)

    # Loading model weights
    try:
        model.load_state_dict(torch.load('models/best_model.pth', map_location=DEVICE))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Printing results
    print("\n--- Code analysis ---")
    print(f"Input: {my_code}")
    
    summary = summarize_code_transformer(model, my_code, code_tk, sum_tk, beam_size=3)
    
    print(f"Generated summary: {summary}")
    print("----------------------\n")

if __name__ == "__main__":
    main()