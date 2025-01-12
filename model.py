import wandb
import dill
import math
from tqdm import tqdm

import torch
from torch.cuda.amp import autocast, GradScaler
from transformers import BartForConditionalGeneration, BartTokenizerFast

from config import DEVICE

def make_model(config, train_input_encodings, train_target_encodings, 
               val_input_encodings, val_target_encodings):
    """
    Create and initialize the model, dataloaders, and optimizer.

    Args:
        config (dict): Configuration parameters
        train_input_encodings (List): Tokenized train input sequences
        train_target_encodings (List): Tokenized train target sequences
        val_input_encodings (List): Tokenized val input sequences
        val_target_encodings (List): Tokenized val target sequences

    Returns:
        tuple: (model, train_loader, val_loader, optimizer)
    """
    from data_loader import make_data_loaders

    train_loader, val_loader = make_data_loaders(
        train_input_encodings, train_target_encodings,
        val_input_encodings, val_target_encodings
    )

    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(DEVICE)
    print('Num parameters: ', model.num_parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    
    return model, train_loader, val_loader, optimizer

def train_model(model, train_loader, val_loader, optimizer, 
                config, corruption_type, enc_type):
    """
    Train the BART model with optimizations for speed.

    Args:
        model (BartForConditionalGeneration): The BART model
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        optimizer (torch.optim.Optimizer): The optimizer
        config (dict): Configuration parameters
    """
    wandb.watch(model, log="all", log_freq=10)
    scaler = GradScaler()  # for mixed precision training

    # Load checkpoint until epoch 2
    model.load_state_dict(torch.load(f'checkpoints/{corruption_type}/ckpt_ep2_b{config.batch_size}_lr{config.lr}.pt'))

    for epoch in range(2, config.epochs):
        model.train()
        total_loss = 0
        correct, total_samples = 0, 0

        TP, FP, FN = 0, 0, 0 
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            with autocast():  # enables mixed precision
                outputs = model(input_ids=batch['input_ids'], 
                                attention_mask=batch['attention_mask'], 
                                labels=batch['labels'])
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()
            
            # Compute accuracy using logits
            pred_tokens = outputs.logits.argmax(dim=-1)
            correct_preds = (pred_tokens == batch['labels']).sum().item()
            total_tokens = batch['labels'].numel()

            correct += correct_preds
            total_samples += total_tokens

            TP += correct_preds
            # For every incorrect token, that's both a FN and a FP.
            incorrect = total_tokens - correct_preds
            FP += incorrect
            FN += incorrect

            step_metrics = {
                "train/step_loss": loss.item(),
                "train/step_accuracy": correct_preds / total_tokens
            }
            wandb.log(step_metrics)

        train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total_samples
        
        # Compute token-level F1 score
        if TP == 0:
            train_f1 = 0.0
        else:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            if precision + recall == 0:
                train_f1 = 0.0
            else:
                train_f1 = 2 * (precision * recall) / (precision + recall)

        # Compute perplexity for training set
        train_perplexity = math.exp(train_loss) if train_loss < 20 else float('inf')

        # Validate Model
        val_loss, val_accuracy = validate_model(model, val_loader)
        
        epoch_metrics = {
            "train/epoch_loss": train_loss,
            "train/epoch_accuracy": train_accuracy,
            "train/epoch_f1": train_f1,
            "train/epoch_ppl": train_perplexity,
            "valid/loss": val_loss,
            "valid/accuracy": val_accuracy,
            "epoch": epoch
        }
        wandb.log(epoch_metrics)
        
        print(f"Epoch {epoch+1}/{config.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}, Train Perplexity: {train_perplexity:.4f}")
        print(f"Valid Loss: {val_loss:.4f}, Valid Accuracy: {val_accuracy:.4f}")
        
        # Save model checkpoint
        # torch.save(model.state_dict(), f"checkpoints/{corruption_type}/ckpt_ep{epoch+1}_b{config.batch_size}_lr{config.lr}.pt")
    
    model.save_pretrained(f"trained_models_100k_{enc_type}/{corruption_type}")
    

def validate_model(model, val_loader):
    """
    Compute performance of the model on the validation dataset.

    Args:
        model (BartForConditionalGeneration): The BART model
        val_loader (DataLoader): DataLoader for validation data

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0
    correct, total_samples = 0, 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation"):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], 
                            attention_mask=batch['attention_mask'], 
                            labels=batch['labels'])
            
            total_loss += outputs.loss.item()
            
            pred_tokens = outputs.logits.argmax(dim=-1)
            correct += (pred_tokens == batch['labels']).sum().item()
            total_samples += batch['labels'].numel()

    return total_loss / len(val_loader), correct / total_samples