"""
Data loading and preprocessing functions.
"""
import pandas as pd
from tqdm import tqdm
from typing import List
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Dataset

from config import DEVICE, BATCH_SIZE, TOKENIZER

class BatchedGenomeDataset(Dataset):
    def __init__(self, input_encodings, target_encodings):
        self.input_encodings = input_encodings
        self.target_encodings = target_encodings
    
    def __getitem__(self, idx):
        batch_idx, item_idx = divmod(idx, len(self.input_encodings[0]['input_ids']))
        item = {key: val[item_idx].clone().detach() for key, val in self.input_encodings[batch_idx].items()}
        item['labels'] = self.target_encodings[batch_idx]['input_ids'][item_idx].clone().detach()
        return item
    
    def __len__(self):
        return sum(len(batch['input_ids']) for batch in self.input_encodings)

def load_data():
    """
    Load corrupted and true sequences, store in pandas.DataFrame. 
    Partition into 80/20% train-test splits.

    Returns:
        tuple: (train_data, test_data) as pandas DataFrames
    """
    with open('reference_reads/R1_sequences.txt', 'r') as f:
        inputs = [line.strip() for line in f]

    with open('reference_reads/R1_true_sequences.txt', 'r') as f:
        targets = [line.strip() for line in f]

    data = pd.DataFrame({'input_text': inputs, 'target_text': targets})
    return train_test_split(data, test_size=0.2, random_state=42)

def batch_tokenize(seq: List, batch_size=1000):
    """
    Tokenize input sequences based on trained BPE tokenizer.

    Args:
        tokenizer: BPE tokenizer trained on Saccharomyces reference sequence.
        seq (List): list of genome sequences
        batch_size (int): specified number of sequences for batch-wise tokenization

    Returns:
        List: Tokenized sequences based on BPE.
    """
    encodings = []
    for i in tqdm(range(0, len(seq), batch_size)):
        batch = seq[i:i+batch_size]
        batch_encodings = TOKENIZER(batch, return_tensors='pt', max_length=20, 
                                    truncation=True, padding="max_length")
        batch_encodings = {k: v.to(DEVICE) for k, v in batch_encodings.items()}
        encodings.append(batch_encodings)
    return encodings

def make_data_loaders(train_input_encodings, train_target_encodings, 
                      test_input_encodings, test_target_encodings):
    """
    Create DataLoaders for train and test datasets.

    Returns:
        tuple: (train_loader, test_loader)
    """
    train_dataset = BatchedGenomeDataset(train_input_encodings, train_target_encodings)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = BatchedGenomeDataset(test_input_encodings, test_target_encodings)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return train_loader, test_loader