"""
Configuration file for Fingerprint Recognition using Siamese Neural Network.
Stores paths, model parameters, and other global settings.
"""

import os
from pathlib import Path

# Paths
DATA_PATH = "../dataset/"
MODEL_PATH = "siamese_model.h5"

# Image settings
IMG_HEIGHT = 90
IMG_WIDTH = 90
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Model training parameters
EPOCHS = 10
BATCH_SIZE = 32

# Random seed for reproducibility
SEED = 42

# Train/Test split, we are using 10% test partition
TEST_SIZE=0.1