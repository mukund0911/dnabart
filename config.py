"""
Configuration settings.
"""

import os
import torch
from transformers import BartTokenizerFast

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
HYPERPARAMETERS = {
    "epochs": 3,
    "batch_size": 8,
    "lr": 1e-4
}

# Other configuration settings
BATCH_SIZE = 8

# Tokenize sequences using BPE
TOKENIZER = BartTokenizerFast.from_pretrained('bpe_tokenization')