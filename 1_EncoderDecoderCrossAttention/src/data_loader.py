import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from src.config import *
from src.utils import encode

def load_data_final():
    """
    Downloads the CodeXGLUE dataset from HuggingFace and performs 
    basic cleaning (removing extra whitespaces).
    """
    print(f"[Data] Loading CodeXGLUE (python) from HuggingFace...")
    dataset = load_dataset("code_x_glue_ct_code_to_text", "python")
    
    def to_df(ds, size):
        # Limit the samples to the size defined in config.py
        subset = ds.select(range(min(size, len(ds))))
        df = pd.DataFrame({
            'code': subset['code'], 
            'summary': subset['docstring']
        })
        
        # Cleaning: normalize whitespaces and strip strings
        df['code'] = df['code'].str.replace(r'\s+', ' ', regex=True).str.strip()
        df['summary'] = df['summary'].str.replace(r'\s+', ' ', regex=True).str.strip()
        return df

    train_df = to_df(dataset['train'], MAX_SAMPLES_TRAIN)
    val_df = to_df(dataset['validation'], MAX_SAMPLES_VALIDATION)
    test_df = to_df(dataset['test'], MAX_SAMPLES_TEST)

    print(f"[Data] Loaded Samples -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df

def get_loaders(train_df, val_df, test_df, code_vocab, summary_vocab):
    """
    Encodes the dataframes into PyTorch tensors and returns 
    the DataLoaders for Training, Validation, and Testing.
    """
    
    def create_dataset(df, name):
        print(f"[Data] Encoding {name} set...")
        
        # Encoding code (source)
        x_encoded = [
            encode(row['code'], code_vocab, MAX_CODE_LEN) 
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Encoding {name} (X)")
        ]
        
        # Encoding summary (target)
        y_encoded = [
            encode(row['summary'], summary_vocab, MAX_SUMMARY_LEN) 
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Encoding {name} (Y)")
        ]
        
        # Convert to tensors and move to DEVICE (if needed, or handle in loop)
        x_tensor = torch.tensor(x_encoded).to(DEVICE)
        y_tensor = torch.tensor(y_encoded).to(DEVICE)
        
        return TensorDataset(x_tensor, y_tensor)

    # Create TensorDatasets
    train_ds = create_dataset(train_df, "Train")
    val_ds = create_dataset(val_df, "Validation")
    test_ds = create_dataset(test_df, "Test")

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader