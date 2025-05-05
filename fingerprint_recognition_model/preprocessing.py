# preprocessing.py

"""
Data preprocessing for SOCOFing fingerprint dataset. 
Includes label extraction and batch processing for real and altered subsets.
"""
import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Label Extraction Functions
# -----------------------------------------------------------------------------

def extract_label(img_path: str) -> np.ndarray:
    """
    Extract subject ID, gender, hand, and finger from a Real-image filename.

    Args:
        img_path (str): Path to the BMP image file.

    Returns:
        np.ndarray: [subject_id, gender, hand, finger] as uint16 array.
    """
    filename = os.path.splitext(os.path.basename(img_path))[0]
    subject_id, meta = filename.split('__')
    gender_str, hand_str, finger_str, _ = meta.split('_')

    gender = 0 if gender_str == 'M' else 1
    hand = 0 if hand_str == 'Left' else 1
    finger_map = {'thumb': 0, 'index': 1, 'middle': 2, 'ring': 3, 'little': 4}
    finger = finger_map.get(finger_str.lower(), -1)

    return np.array([int(subject_id), gender, hand, finger], dtype=np.uint16)


def extract_label_alt(img_path: str) -> np.ndarray:
    """
    Extract subject ID, gender, hand, and finger from an Altered-image filename.
    Slight variant in splitting to handle extra underscores.

    Args:
        img_path (str): Path to the BMP image file.

    Returns:
        np.ndarray: [subject_id, gender, hand, finger] as uint16 array.
    """
    filename = os.path.splitext(os.path.basename(img_path))[0]
    subject_id, meta = filename.split('__')
    gender_str, hand_str, finger_str, *_ = meta.split('_')

    gender = 0 if gender_str == 'M' else 1
    hand = 0 if hand_str == 'Left' else 1
    finger_map = {'thumb': 0, 'index': 1, 'middle': 2, 'ring': 3, 'little': 4}
    finger = finger_map.get(finger_str.lower(), -1)

    return np.array([int(subject_id), gender, hand, finger], dtype=np.uint16)

# -----------------------------------------------------------------------------
# Generic Directory Preprocessing
# -----------------------------------------------------------------------------

def preprocess_directory(
    pattern: str,
    label_func,
    output_x: str,
    output_y: str,
    target_size: tuple = (90, 90),
    visualize: bool = False
) -> None:
    """
    Load images matching a glob pattern, extract labels, and save numpy arrays.

    Args:
        pattern (str): Glob pattern for image files (e.g., 'dataset/SOCOFing/Real/*.BMP').
        label_func (callable): Function to extract label array from image path.
        output_x (str): Path to save images .npy file.
        output_y (str): Path to save labels .npy file.
        target_size (tuple): (width, height) to resize images.
        visualize (bool): If True, display the last image and label.
    """
    image_paths = sorted(glob.glob(pattern))
    if not image_paths:
        raise ValueError(f"No images found for pattern: {pattern}")

    num_images = len(image_paths)
    imgs = np.empty((num_images, *target_size), dtype=np.uint8)
    labels = np.empty((num_images, 4), dtype=np.uint16)

    for i, path in enumerate(image_paths):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, target_size)
        imgs[i] = img
        labels[i] = label_func(path)

    # Save arrays
    np.save(output_x, imgs)
    np.save(output_y, labels)

    if visualize:
        plt.figure(figsize=(2,2))
        plt.title(f"Label: {labels[-1]}")
        plt.imshow(imgs[-1], cmap='gray')
        plt.axis('off')
        plt.show()

# -----------------------------------------------------------------------------
# Convenience Wrappers for Each Subset
# -----------------------------------------------------------------------------

def preprocess_real(data_dir: str = 'dataset/SOCOFing/Real', **kwargs) -> None:
    pattern = os.path.join(data_dir, '*.BMP')
    preprocess_directory(
        pattern,
        extract_label,
        output_x=os.path.join(data_dir, '../x_real.npy'),
        output_y=os.path.join(data_dir, '../y_real.npy'),
        **kwargs
    )


def preprocess_altered(
    difficulty: str,
    data_dir: str = 'dataset/SOCOFing/Altered',
    **kwargs
) -> None:
    subset = f"Altered-{difficulty.capitalize()}"
    pattern = os.path.join(data_dir, subset, '*.BMP')
    output_prefix = difficulty.lower()
    preprocess_directory(
        pattern,
        extract_label_alt,
        output_x=os.path.join(data_dir, '../x_{}.npy'.format(output_prefix)),
        output_y=os.path.join(data_dir, '../y_{}.npy'.format(output_prefix)),
        **kwargs
    )


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

from load_data import load_dataset
from config import TEST_SIZE
from sklearn.model_selection import train_test_split
    
def preprocess_data():
    data_dict = load_dataset()

    # Example: combine all datasets (you can change logic here)
    x_data = np.concatenate([data_dict['easy'][0], data_dict['medium'][0], data_dict['hard'][0]], axis=0)
    label_data = np.concatenate([data_dict['easy'][1], data_dict['medium'][1], data_dict['hard'][1]], axis=0)

    x_train, x_val, label_train, label_val = train_test_split(x_data, label_data, test_size=TEST_SIZE)
    
    return x_train, x_val, label_train, label_val 


# -----------------------------------------------------------------------------
# If run as script, process all subsets
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    preprocess_real(visualize=True)
    for lvl in ['Easy', 'Medium', 'Hard']:
        preprocess_altered(lvl, visualize=True)


