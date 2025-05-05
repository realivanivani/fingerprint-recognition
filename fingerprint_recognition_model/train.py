# train.py

"""
Training pipeline for the Siamese fingerprint recognition model.
Imports hyperparameters and paths from config.py, builds the model, trains with callbacks, and saves best weights.
"""
import os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from config import BATCH_SIZE, EPOCHS, MODEL_PATH, IMG_SIZE
from model import build_siamese_model


def compile_and_train(
    pairs_train: np.ndarray,
    labels_train: np.ndarray,
    pairs_val: np.ndarray,
    labels_val: np.ndarray
) -> any:
    """
    Build, compile, and train the Siamese model using global config parameters.

    Args:
        pairs_train (np.ndarray): Array of shape (N, 2, H, W, C) for training pairs.
        labels_train (np.ndarray): Binary labels for training pairs.
        pairs_val (np.ndarray): Validation pair array.
        labels_val (np.ndarray): Validation labels.

    Returns:
        History: Keras History object from model.fit().
    """
    # Unpack paired inputs
    x1_train, x2_train = pairs_train[:, 0], pairs_train[:, 1]
    x1_val, x2_val = pairs_val[:, 0], pairs_val[:, 1]

    # Build Siamese model with shape from config
    model = build_siamese_model(input_shape=(*IMG_SIZE, 1))

    # Ensure checkpoint directory exists
    ckpt_dir = os.path.dirname(MODEL_PATH)
    if ckpt_dir and not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    earlystop_cb = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    # Train
    history = model.fit(
        [x1_train, x2_train], labels_train,
        validation_data=([x1_val, x2_val], labels_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, earlystop_cb]
    )
    return history

