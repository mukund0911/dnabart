"""
Configuration settings.
"""

import os
import torch

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
HYPERPARAMETERS = {
    "epochs": 5,
    "batch_size": 4,
    "lr": 1e-5
}

# Other configuration settings
BATCH_SIZE = 4