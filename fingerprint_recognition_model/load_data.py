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

from config import TEST_SIZE
from sklearn.model_selection import train_test_split
    
def preprocess_data():
    data_dict = load_dataset()

    # Example: combine all datasets (you can change logic here)
    x_data = np.concatenate([data_dict['easy'][0], data_dict['medium'][0], data_dict['hard'][0]], axis=0)
    label_data = np.concatenate([data_dict['easy'][1], data_dict['medium'][1], data_dict['hard'][1]], axis=0)

    x_train, x_val, label_train, label_val = train_test_split(x_data, label_data, test_size=TEST_SIZE)
    
    return x_train, x_val, label_train, label_val 