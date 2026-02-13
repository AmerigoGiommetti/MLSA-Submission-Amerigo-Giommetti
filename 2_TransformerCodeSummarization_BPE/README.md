# ML-Based Python Code Summarization
**Course:** Machine Learning for Software Analysis  
**Academic Year:** 2025/2026  
**Author:** Amerigo Giommetti

## Project Overview
This project implements a Sequence-to-Sequence (Seq2Seq) Transformer model to automatically generate natural language summaries (docstrings) from Python code snippets. It leverages the **CodeXGLUE** dataset and modern NLP techniques to bridge the gap between source code and human-readable documentation.

### Key Features
* **Architecture:** Multi-head Attention Transformer (Encoder-Decoder).
* **Tokenization:** Byte Pair Encoding (BPE) to handle Out-of-Vocabulary (OOV) code identifiers.
* **Optimization:** Early Stopping based on Validation Loss and Label Smoothing.
* **Inference:** Beam Search decoding for higher quality summary generation.
* **Metrics:** Evaluation via BLEU score and ROUGE-L.

---

## Project Structure
```text
2_TransformerCodeSummarization_BPE/
├── src/
│   ├── config.py                 # Hyperparameters and global settings
│   ├── model.py                  # Transformer & Positional Encoding definitions
│   ├── utils.py                  # BPE Training, Encoding, and Seed management
│   └── data_loader.py            # CodeXGLUE loading and DataLoader pipeline
├── models/
│   └── best_model.pth            # Contains the weights of the best model computed during training
├── tokenizers/
│   ├── code_bpe.json             # Contains the tokenizer for code trained during last session
│   └── summary_bpe.json          # Contains the tokenizer for summary trained during last session
├── main.py                             # Entry point for Training
├── evaluate.py                         # Entry point for Evaluation and Inference
├── requirements.txt                    # Software dependencies
├── transformerPlot.png                 # Plot drawn after last training
├── EncoderDecoderDocumentation.docx    # Project documentation
├── transformers_BPE_BeamSearch.ipynb   # Google colab notebook, for detail chek documentation
└── README.md                           # Project documentation