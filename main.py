import os
import wandb
import torch
from transformers import BartTokenizerFast

from config import HYPERPARAMETERS, DEVICE
from data_loader import load_data, batch_tokenize, make_data_loaders
from model import make_model, train_model

def main():
    print(f"Using device: {DEVICE}; {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Load Dataset
    train_data, test_data = load_data()

    print("Tokenizing train data...")
    train_input_encodings = batch_tokenize(list(train_data['input_text']))
    train_target_encodings = batch_tokenize(list(train_data['target_text']))

    print("Tokenizing test data...")
    test_input_encodings = batch_tokenize(list(test_data['input_text']))
    test_target_encodings = batch_tokenize(list(test_data['target_text']))

    with wandb.init(project="dnabart", config=HYPERPARAMETERS):
        config = wandb.config

        # Define the model, data, and optimization problem
        model, train_loader, test_loader, optimizer = make_model(config, 
                                                                 train_input_encodings, train_target_encodings,
                                                                 test_input_encodings, test_target_encodings)

        # Train the model
        train_model(model, train_loader, test_loader, optimizer, config)

    # Save the trained model
    model.save_pretrained("trained_model")


if __name__ == "__main__":
    main()