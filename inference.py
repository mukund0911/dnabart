import wandb
import pandas as pd
from tqdm import tqdm
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BartForConditionalGeneration

from config import BATCH_SIZE, TOKENIZER, HYPERPARAMETERS
from data_loader import BatchedGenomeDataset

wandb.init(project="dnabart", config=HYPERPARAMETERS)

## Load test sequences and tokenize

def batch_tokenize(seq: List, batch_size=1000):
    encodings = []
    for i in tqdm(range(0, len(seq), batch_size)):
        batch = seq[i:i+batch_size]
        batch_encodings = TOKENIZER(batch, return_tensors='pt', max_length=20, 
                                    truncation=True, padding="max_length")
        batch_encodings = {k: v for k, v in batch_encodings.items()}
        encodings.append(batch_encodings)
    return encodings

with open('reference_reads/R1_sequences.txt', 'r') as f:
    inputs = [line.strip() for line in f]

with open('reference_reads/R1_true_sequences.txt', 'r') as f:
    targets = [line.strip() for line in f]

test_data = pd.DataFrame({'input_text': inputs, 'target_text': targets})
test_data = test_data.tail(1000)

test_input_encodings = batch_tokenize(list(test_data['input_text']))
test_target_encodings = batch_tokenize(list(test_data['target_text']))

test_dataset = BatchedGenomeDataset(test_input_encodings, test_target_encodings)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)



# Load pretrained model and test
def decode_sequence(seq):
    """Decode a tokenized sequence."""
    return TOKENIZER.decode(seq, skip_special_tokens=True)

def log_image_table(predictions, input, labels):
    "Log a wandb.Table with (pred, target)"
    table = wandb.Table(columns=["pred", "input", "target"])
    for pred, ip, targ in zip(predictions, input, labels):
        table.add_data(pred, ip, targ)
    wandb.log({"predictions_table":table})


model = BartForConditionalGeneration.from_pretrained('trained_model')

total_eval_loss = 0
total_correct = 0
total_samples = 0
batch_idx = 0
model.eval()
with torch.inference_mode():
    for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
        # print(batch)
        # exit()
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])

        generated_ids = model.generate(batch['input_ids'], max_length=20)

        predictions = [decode_sequence(g) for g in generated_ids]
        input = [decode_sequence(i) for i in batch['input_ids']]
        labels = [decode_sequence(l) for l in batch['labels']]

        correct = sum(p == l for p, l in zip(predictions, labels))
        total_correct += correct
        total_samples += len(predictions)

        # Log one batch of images to the dashboard, always same batch_idx.
        if i==batch_idx:
            log_image_table(predictions, input, labels)

print(f"Loss: {total_eval_loss / len(test_loader)}, Accuracy: {total_correct / total_samples}")