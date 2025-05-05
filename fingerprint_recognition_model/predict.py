# predict.py

"""
Utility for running inference with a trained Siamese fingerprint model.
"""
import numpy as np
from tensorflow.keras.models import load_model
from config import MODEL_PATH


def load_trained_model():
    """
    Load the trained Siamese model from disk.

    Returns:
        keras.Model: Loaded Siamese model.
    """
    model = load_model(MODEL_PATH)
    return model


def predict_similarity(model, pairs: np.ndarray) -> np.ndarray:
    """
    Predict similarity scores for fingerprint image pairs.

    Args:
        model (keras.Model): Trained Siamese model.
        pairs (np.ndarray): Array of shape (N, 2, H, W, C) or (2, H, W, C).

    Returns:
        np.ndarray: Similarity scores between 0 and 1 for each pair.
    """
    # Ensure batch dimension
    if pairs.ndim == 4:
        # Single pair without batch dimension
        pairs = np.expand_dims(pairs, axis=0)
    if pairs.ndim != 5:
        raise ValueError(f"Input pairs array must be 4D or 5D, got shape {pairs.shape}")

    # Split into two inputs
    x1 = pairs[:, 0, ...]
    x2 = pairs[:, 1, ...]

    # Predict similarity
    scores = model.predict([x1, x2], verbose=0)
    return scores.squeeze()
