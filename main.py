import os
import sys
import wandb
import torch
from transformers import BartTokenizerFast

from config import HYPERPARAMETERS, DEVICE
from data_loader import load_data, batch_tokenize, make_data_loaders
from model import make_model, train_model

def main():
    corruption_type = sys.argv[1]
    enc_type = sys.argv[2]

    print(f"Using device: {DEVICE}; {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Load Dataset
    train_data, val_data = load_data(corruption_type)

    print("Tokenizing train data...")
    train_input_encodings = batch_tokenize(list(train_data['input_text']), enc_type)
    train_target_encodings = batch_tokenize(list(train_data['target_text']), enc_type)

    print("Tokenizing validation data...")
    val_input_encodings = batch_tokenize(list(val_data['input_text']), enc_type)
    val_target_encodings = batch_tokenize(list(val_data['target_text']), enc_type)

    with wandb.init(project=f"dnabart_{corruption_type}", config=HYPERPARAMETERS):
        config = wandb.config

        # Define the model, data, and optimization problem
        model, train_loader, val_loader, optimizer = make_model(config, 
                                                                 train_input_encodings, train_target_encodings,
                                                                 val_input_encodings, val_target_encodings)

        # Train the model
        train_model(model, train_loader, val_loader, optimizer, config, corruption_type)
        


if __name__ == "__main__":
    main()