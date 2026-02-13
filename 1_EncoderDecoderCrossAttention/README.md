# Python Code Summarization with Seq2Seq & Attention
**Course:** Machine Learning for Software Analysis (MLSA)  
**Academic Year:** 2025/2026  
**Author:** Amerigo Giommetti

## 1. Project Overview
This project implements a Machine Learning pipeline to automatically generate natural language summaries (docstrings) from Python code snippets. It uses the **CodeXGLUE** dataset (specifically the `code_to_text` Python split) and a Sequence-to-Sequence (Seq2Seq) architecture based on Recurrent Neural Networks (RNNs).

### Key Features
* **Bidirectional Encoder:** Captures context from both directions of the source code.
* **Bahdanau Attention:** Implements a cross-attention mechanism to focus on relevant code tokens during summary generation.
* **Padding Masking:** Advanced optimization that forces the attention mechanism to ignore `<pad>` tokens, improving training stability and performance.
* **Persistence:** Automated saving of model weights (`.pt`) and vocabularies (`.pkl`).

---

## 2. Project Structure
The project is modularized to ensure separation of concerns:

```text
4_EncoderDecoderCrossAttention/
├── src/
│   ├── config.py                       # Hyperparameters and global settings
│   ├── model.py                        # Encoder, Decoder, and Attention definitions
│   ├── utils.py                        # Tokenization, Encoding, and Seed management
│   └── data_loader.py                  # Data pipeline (HuggingFace integration)
├── vocabulary/            
│   ├── code_vocab.plk                  # code vocabulary saved at each training
│   └── summary_vocab.plk               # summary vocabulary saved at each training
├── models/                
│   └── summarization.pt                # Saved model weights (after training)
├── main.py                             # Entry point for training and validation
├── evaluate.py                         # Entry point for quantitative and qualitative evaluation
├── requirements.txt                    # Software dependencies
├── EncoderDecoderPlot.png              # Plot drawn after last training
├── EncoderDecoderDocumentation.docx    # Project documentation
└── README.md                           # Project documentation