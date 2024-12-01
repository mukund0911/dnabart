import wandb
import torch
from tqdm import tqdm
from transformers import BartForConditionalGeneration

from config import DEVICE

def make_model(config, train_input_encodings, train_target_encodings, 
               test_input_encodings, test_target_encodings):
    """
    Create and initialize the model, dataloaders, and optimizer.

    Args:
        config (dict): Configuration parameters
        train_input_encodings (List): Tokenized train input sequences
        train_target_encodings (List): Tokenized train target sequences
        test_input_encodings (List): Tokenized test input sequences
        test_target_encodings (List): Tokenized test target sequences

    Returns:
        tuple: (model, train_loader, test_loader, optimizer)
    """
    from data_loader import make_data_loaders

    train_loader, test_loader = make_data_loaders(
        train_input_encodings, train_target_encodings,
        test_input_encodings, test_target_encodings
    )

    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(DEVICE)
    print('Num parameters: ', model.num_parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    
    return model, train_loader, test_loader, optimizer

def train_model(model, train_loader, test_loader, optimizer, config):
    """
    Train the BART model.

    Args:
        model (BartForConditionalGeneration): The BART model
        train_loader (DataLoader): DataLoader for training data
        test_loader (DataLoader): DataLoader for validation data
        optimizer (torch.optim.Optimizer): The optimizer
        config (dict): Configuration parameters
    """
    wandb.watch(model, log="all", log_freq=10)

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            optimizer.zero_grad()
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            metrics = {"train/train_loss": loss, "train/epoch": (epoch+1)/config.epochs}
            wandb.log(metrics)

        val_loss, accuracy = validate_model(model, test_loader)

        val_metrics = {"val/val_loss": val_loss, "val/val_accuracy": accuracy}
        wandb.log({**metrics, **val_metrics})
        
        torch.save(model, f"checkpoints/ckpt_ep{epoch}_b{config.batch_size}_lr{config.lr}.pt")

        print(f"Epoch: {epoch+1}, Train Loss: {total_loss / len(train_loader):.3f}, Valid Loss: {val_loss:3f}, Valid Accuracy: {accuracy:.2f}")

def validate_model(model, valid_dl):
    """
    Compute performance of the model on the validation dataset.

    Args:
        model (BartForConditionalGeneration): The BART model
        valid_dl (DataLoader): DataLoader for validation data

    Returns:
        tuple: (average_loss, accuracy)
    """
    total_eval_loss = 0
    total_correct = 0
    total_samples = 0
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(valid_dl, desc="Evaluating"):
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            loss = outputs.loss
            total_eval_loss += loss.item()
            
            generated_ids = model.generate(batch['input_ids'], max_length=20)
            
            predictions = [decode_sequence(g) for g in generated_ids]
            labels = [decode_sequence(l) for l in batch['labels']]
            
            correct = sum(p == l for p, l in zip(predictions, labels))
            total_correct += correct
            total_samples += len(predictions)
    return total_eval_loss / len(valid_dl), total_correct / total_samples

def decode_sequence(seq):
    """Decode a tokenized sequence."""
    from main import tokenizer
    return tokenizer.decode(seq, skip_special_tokens=True)