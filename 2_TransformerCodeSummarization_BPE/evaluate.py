# General imports
import torch
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Local imports
from src.config import *
from src.utils import load_tokenizer, encode_bpe
from src.model import TransformerSeq2Seq
from src.data_loader import load_data_final

def summarize_code_transformer(model, code_sentence, code_tokenizer, summary_tokenizer, beam_size=3, max_len=MAX_SUMMARY_LEN):
    """Inference using beam search."""
    model.eval()
    with torch.no_grad():
        src_ids = encode_bpe(code_sentence, code_tokenizer, MAX_CODE_LEN)
        src_tensor = torch.tensor(src_ids).unsqueeze(0).to(DEVICE)
        
        beams = [(0.0, [1])] # 1 is <sos>
        
        for _ in range(max_len):
            new_beams = []
            for score, seq in beams:
                if seq[-1] == 2: # 2 is <eos>
                    new_beams.append((score, seq))
                    continue
                
                tgt_tensor = torch.tensor(seq).unsqueeze(0).to(DEVICE)
                output = model(src=src_tensor, tgt=tgt_tensor)
                probs = torch.log_softmax(output[0, -1, :], dim=-1)
                top_v, top_i = probs.topk(beam_size)
                
                for i in range(beam_size):
                    new_beams.append((score + top_v[i].item(), seq + [top_i[i].item()]))
            
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
            if all(seq[-1] == 2 for score, seq in beams):
                break
        
        best_seq = beams[0][1]
        return summary_tokenizer.decode(best_seq, skip_special_tokens=True)

def run_evaluation_report(n_samples=100):
    print("Starting final evaluation...")
    
    # Tokenizers and data loading
    try:
        code_tk = load_tokenizer("tokenizers/code_bpe.json")
        sum_tk = load_tokenizer("tokenizers/summary_bpe.json")
    except Exception as e:
        print(f"Errore: Tokenizer non trovati. Esegui prima main.py. {e}")
        return

    _, _, test_df = load_data_final()

    # Model init and loading (NOT TRAINING!)
    model = TransformerSeq2Seq(
        src_vocab_size=code_tk.get_vocab_size(),
        tgt_vocab_size=sum_tk.get_vocab_size()
    ).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load('models/best_model.pth', map_location=DEVICE))
        print("Modello caricato con successo!")
    except Exception as e:
        print(f"Errore nel caricamento del modello: {e}")
        return

    # Evaluation (BLEU e ROUGE)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    bleu_scores, rouge_scores = [], []
    
    samples = test_df.sample(n=min(n_samples, len(test_df)))
    
    print(f"Evaluating on {len(samples)} samples...")
    for _, row in samples.iterrows():
        reference = row['summary']
        prediction = summarize_code_transformer(model, row['code'], code_tk, sum_tk)

        # Bleu tokenization based on bpe
        ref_tokens = sum_tk.encode(reference).tokens
        pred_tokens = sum_tk.encode(prediction).tokens

        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=SmoothingFunction().method1)
        bleu_scores.append(bleu)
        rouge_scores.append(scorer.score(reference, prediction)['rougeL'].fmeasure)

    print("\n" + "="*30)
    print(f"FINAL REPORT")
    print(f"AVG BLEU:   {np.mean(bleu_scores):.4f}")
    print(f"AVG ROUGE-L: {np.mean(rouge_scores):.4f}")
    print("="*30 + "\n")

    # Inference
    print("Model generated samples evaluation:")
    for i in range(3):
        row = test_df.iloc[np.random.randint(len(test_df))]
        gen = summarize_code_transformer(model, row['code'], code_tk, sum_tk)
        print(f"\n[CODE]: {row['code'][:100]}...")
        print(f"[REAL]: {row['summary']}")
        print(f"[GEN ]: {gen}")

if __name__ == "__main__":
    run_evaluation_report(n_samples=100)