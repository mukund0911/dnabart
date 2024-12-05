import wandb
import dill
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from transformers import BartForConditionalGeneration, BartTokenizerFast

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
    Train the BART model with optimizations for speed.

    Args:
        model (BartForConditionalGeneration): The BART model
        train_loader (DataLoader): DataLoader for training data
        test_loader (DataLoader): DataLoader for validation data
        optimizer (torch.optim.Optimizer): The optimizer
        config (dict): Configuration parameters
    """
    wandb.watch(model, log="all", log_freq=10)
    scaler = GradScaler()  # for mixed precision training

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        correct, total_samples = 0, 0
        
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
            correct += (pred_tokens == batch['labels']).sum().item()
            total_samples += batch['labels'].numel()

            step_metrics = {
                "train/step_loss": loss.item(),
                "train/step_accuracy": correct / total_samples
            }
            wandb.log(step_metrics)

        train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total_samples
        
        # Validate Model
        val_loss, val_accuracy = validate_model(model, test_loader)
        
        epoch_metrics = {
            "train/epoch_loss": train_loss,
            "train/epoch_accuracy": train_accuracy,
            "valid/loss": val_loss,
            "valid/accuracy": val_accuracy,
            "epoch": epoch
        }
        wandb.log(epoch_metrics)
        
        print(f"Epoch {epoch+1}/{config.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Valid Loss: {val_loss:.4f}, Valid Accuracy: {val_accuracy:.4f}")
        
        # Save model checkpoint
        torch.save(model.state_dict(), f"checkpoints/ckpt_ep{epoch+1}_b{config.batch_size}_lr{config.lr}.pt")
    
    model.save_pretrained("trained_model")
    

def validate_model(model, test_loader):
    """
    Compute performance of the model on the validation dataset.

    Args:
        model (BartForConditionalGeneration): The BART model
        test_loader (DataLoader): DataLoader for validation data

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0
    correct, total_samples = 0, 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Validation"):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], 
                            attention_mask=batch['attention_mask'], 
                            labels=batch['labels'])
            
            total_loss += outputs.loss.item()
            
            pred_tokens = outputs.logits.argmax(dim=-1)
            correct += (pred_tokens == batch['labels']).sum().item()
            total_samples += batch['labels'].numel()

    return total_loss / len(test_loader), correct / total_samples

# def decode_sequence(seq):
#     """Decode a tokenized sequence."""
#     return TOKENIZER.decode(seq, skip_special_tokens=True)