import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader
from .config import *
from .utils import encode_bpe

def load_data_final():
    # Code_x_glue dataset loading
    print("Caricamento dataset CodeXGLUE...")
    dataset = load_dataset("code_x_glue_ct_code_to_text", "python")

    # Data gathering using already present split of training, validation and test in code_x_glue
    train_raw = dataset['train'].select(range(min(MAX_SAMPLES_TRAIN, len(dataset['train']))))
    val_raw = dataset['validation'].select(range(min(MAX_SAMPLES_VALIDATION, len(dataset['validation']))))
    test_raw  = dataset['test'].select(range(min(MAX_SAMPLES_TEST, len(dataset['test']))))
        
    # Data shaping into dataframes and minor code cleaning
    def to_df(ds):
        df = pd.DataFrame({'code': ds['code'], 'summary': ds['docstring']})
        df['code'] = df['code'].str.replace(r'\s+', ' ', regex=True).str.strip()
        df['summary'] = df['summary'].str.replace(r'\s+', ' ', regex=True).str.strip()
        return df

    return to_df(train_raw), to_df(val_raw), to_df(test_raw)

def get_dataloaders(train_df, val_df, test_df, code_tokenizer, summary_tokenizer):

    print("Encoding datasets...")
    
    # Encoding using BPE
    # Train set
    X_train = torch.tensor([
        encode_bpe(c, code_tokenizer, MAX_CODE_LEN) for c in train_df['code']
    ]).to(DEVICE)
    Y_train = torch.tensor([
        encode_bpe(s, summary_tokenizer, MAX_SUMMARY_LEN) for s in train_df['summary']
    ]).to(DEVICE)

    # Validation set
    X_val = torch.tensor([
        encode_bpe(c, code_tokenizer, MAX_CODE_LEN) for c in val_df['code']
    ]).to(DEVICE)
    Y_val = torch.tensor([
        encode_bpe(s, summary_tokenizer, MAX_SUMMARY_LEN) for s in val_df['summary']
    ]).to(DEVICE)

    # Test set
    X_test = torch.tensor([
        encode_bpe(c, code_tokenizer, MAX_CODE_LEN) for c in test_df['code']
    ]).to(DEVICE)
    Y_test = torch.tensor([
        encode_bpe(s, summary_tokenizer, MAX_SUMMARY_LEN) for s in test_df['summary']
    ]).to(DEVICE)

    # Creation of TensorDatasets
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    test_dataset = TensorDataset(X_test, Y_test)

    # Creation of dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"DataLoader succesfully created (Batch Size: {BATCH_SIZE})")
    
    return train_loader, val_loader, test_loader