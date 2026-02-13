import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Hyperparameters
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3

# Sequence Lengths
MAX_VOCAB_SIZE = 3000
MAX_CODE_LEN = 150    
MAX_SUMMARY_LEN = 50  

# Datasets limits
MAX_SAMPLES_TRAIN = 10000
MAX_SAMPLES_VALIDATION = 500
MAX_SAMPLES_TEST = 500