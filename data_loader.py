"""
Data loading and preprocessing functions.
"""
import pandas as pd
from tqdm import tqdm
from typing import List
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Dataset

from tokenizer import kmer_tokenize
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

def load_data(corruption_type):
    """
    Load corrupted and true sequences, store in pandas.DataFrame. 
    Partition into 80/10% train-validation splits.

    Returns:
        tuple: (train_data, val_data) as pandas DataFrames
    """
    with open(f'reference_reads_{corruption_type}/R1_sequences.txt', 'r') as f:
        inputs = [line.strip() for line in f]

    with open(f'reference_reads_{corruption_type}/R1_true_sequences.txt', 'r') as f:
        targets = [line.strip() for line in f]

    data = pd.DataFrame({'input_text': inputs, 'target_text': targets})
    data = data[:100000]
    return train_test_split(data, test_size=0.2, random_state=42)

def batch_tokenize(seq: List, enc_type='bpe', batch_size=1000):
    """
    Tokenize input sequences based on trained BPE tokenizer.

    Args:
        tokenizer: BPE tokenizer trained on Saccharomyces reference sequence.
        seq (List): list of genome sequences
        batch_size (int): specified number of sequences for batch-wise tokenization
        enc_type (str): type of encoding to use (bpe or kmer)

    Returns:
        List: Tokenized sequences based on BPE.
    """
    encodings = []
    for i in tqdm(range(0, len(seq), batch_size)):
        batch = seq[i:i+batch_size]
        if enc_type == 'bpe':
            batch_encodings = TOKENIZER(batch, return_tensors='pt', max_length=150, 
                                        truncation=True, padding="max_length")
            batch_encodings = {k: v.to(DEVICE) for k, v in batch_encodings.items()}
        else:
            encoded = [kmer_tokenize(s, max_length=100) for s in batch]
            input_ids = torch.tensor([e[0] for e in encoded])
            attention_mask = torch.tensor([e[1] for e in encoded])
            batch_encodings = {'input_ids': input_ids.to(DEVICE), 'attention_mask': attention_mask.to(DEVICE)}

        encodings.append(batch_encodings)
    return encodings

def make_data_loaders(train_input_encodings, train_target_encodings, 
                      val_input_encodings, val_target_encodings):
    """
    Create DataLoaders for train and val datasets.

    Returns:
        tuple: (train_loader, val_loader)
    """
    train_dataset = BatchedGenomeDataset(train_input_encodings, train_target_encodings)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = BatchedGenomeDataset(val_input_encodings, val_target_encodings)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    return train_loader, val_loader