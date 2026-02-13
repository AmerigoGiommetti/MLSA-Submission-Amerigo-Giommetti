import torch
import pickle
import os
from src.config import *
from src.utils import set_seed, tokenize
from src.data_loader import load_data_final
from src.model import Encoder, Decoder, Attention, Seq2Seq
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tqdm import tqdm

# Create inference examples
def summarize_code(model, code_sentence, src_vocab, tgt_vocab, max_len=MAX_SUMMARY_LEN):
    model.eval()
    with torch.no_grad():
        tokens = tokenize(code_sentence)
        ids = [src_vocab["<sos>"]] + [src_vocab.get(tok, src_vocab["<unk>"]) for tok in tokens] + [src_vocab["<eos>"]]
        src_tensor = torch.tensor(ids).unsqueeze(0).to(DEVICE)
        
        mask = (src_tensor != 0)
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
        
        inputs = [tgt_vocab["<sos>"]]
        for _ in range(max_len):
            input_tensor = torch.tensor([inputs[-1]]).to(DEVICE).unsqueeze(0)
            output, hidden, cell = model.decoder(input_tensor, hidden, cell, encoder_outputs, mask)
            
            predicted_id = output.argmax(1).item()
            if predicted_id == tgt_vocab["<eos>"]:
                break
            inputs.append(predicted_id)

        inv_vocab = {v: k for k, v in tgt_vocab.items()}
        return " ".join([inv_vocab.get(i, "<unk>") for i in inputs[1:]])

# Calculates score metrics for BLEU and ROGUE
def calculate_metrics(model, test_df, src_vocab, tgt_vocab, n_samples=100):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smoothing = SmoothingFunction().method1
    
    bleu_scores = []
    rouge_l_scores = []
    samples = test_df.sample(n=min(n_samples, len(test_df)), random_state=SEED)
    
    print(f"Calculating metrics on {len(samples)} samples...")
    with torch.no_grad():
        for _, row in tqdm(samples.iterrows(), total=len(samples)):
            generated = summarize_code(model, row['code'], src_vocab, tgt_vocab)
            reference = row['summary']
            
            ref_tokens = [tokenize(reference)]
            cand_tokens = tokenize(generated)
            bleu_scores.append(sentence_bleu(ref_tokens, cand_tokens, smoothing_function=smoothing))
            rouge_l_scores.append(scorer.score(reference, generated)['rougeL'].fmeasure)
            
    print(f"\nQUANTITATIVE RESULTS:")
    print(f"Average BLEU:    {sum(bleu_scores)/len(bleu_scores):.4f}")
    print(f"Average ROUGE-L: {sum(rouge_l_scores)/len(rouge_l_scores):.4f}")

# Shows inference samples
def show_sample_report(model, test_df, src_vocab, tgt_vocab, n=3):
    print(f"\n{'='*80}\n{'QUALITATIVE INFERENCE REPORT':^80}\n{'='*80}")
    samples = test_df.sample(n, random_state=SEED)
    for i, (_, row) in enumerate(samples.iterrows()):
        generated = summarize_code(model, row['code'], src_vocab, tgt_vocab)
        print(f"--- [ EXAMPLE {i+1} ] ---")
        print(f"SOURCE CODE:\n{row['code']}\n")
        print(f"EXPECTED: {row['summary']}")
        print(f"MODEL:    {generated}\n")
        print(f"{'-'*80}")

def main():
    set_seed(SEED)
    
    # Loading of vocabulary created during training
    print("Loading vocabularies from /vocabulary...")
    try:
        with open('vocabulary/code_vocab.pkl', 'rb') as f:
            code_vocab = pickle.load(f)
        with open('vocabulary/summary_vocab.pkl', 'rb') as f:
            summary_vocab = pickle.load(f)
    except FileNotFoundError:
        print("Error: Vocabulary files not found. Run main.py first.")
        return

    # Init of model with same dimension of training
    attn = Attention(HIDDEN_DIM)
    enc = Encoder(len(code_vocab)).to(DEVICE)
    dec = Decoder(len(summary_vocab), attn).to(DEVICE)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    
    # Loading of model weights calculated during training
    model_path = "models/summarization.pt" 
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Weights loaded from {model_path}")
    else:
        print(f"Error: {model_path} not found.")
        return

    # Loading test set for evaluation
    _, _, test_df = load_data_final()

    # Execution
    calculate_metrics(model, test_df, code_vocab, summary_vocab)
    show_sample_report(model, test_df, code_vocab, summary_vocab)

if __name__ == "__main__":
    main()