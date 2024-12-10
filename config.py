"""
Configuration settings.
"""

import os
import torch
from itertools import product

from transformers import BartTokenizerFast

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
HYPERPARAMETERS = {
    "epochs": 3,
    "batch_size": 5,
    "lr": 1e-4
}

# Other configuration settings
BATCH_SIZE = 5
K = 3

# Tokenize sequences using BPE
TOKENIZER = BartTokenizerFast.from_pretrained('bpe_tokenization')
KMER_VOCAB = {'<pad>': 1, '<unk>': 2, **{''.join(kmer): idx+2 for idx, kmer in enumerate(product('ATCG', repeat=K))}}