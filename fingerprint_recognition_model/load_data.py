import numpy as np
import os
from config import DATA_PATH

def load_dataset():
    """
    Load fingerprint datasets from .npy files.
    
    Returns:
        data_dict: Dictionary with keys ['real', 'easy', 'medium', 'hard']
                   Each key maps to a tuple (X, y) of numpy arrays.
    """
    datasets = ['real', 'easy', 'medium', 'hard']
    data_dict = {}

    for dataset in datasets:
        x_path = os.path.join(DATA_PATH, f"x_{dataset}.npy")
        y_path = os.path.join(DATA_PATH, f"y_{dataset}.npy")

        x_data = np.load(x_path)
        y_data = np.load(y_path)

        data_dict[dataset] = (x_data, y_data)
        print(f"{dataset}: X shape {x_data.shape}, Y shape {y_data.shape}")

    return data_dict
