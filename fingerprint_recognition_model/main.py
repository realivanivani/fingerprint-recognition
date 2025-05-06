import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
import config
from load_data import preprocess_data
from train import train_model
from predict import predict_model
from plots import plot_history, plot_confusion_matrix, plot_roc_curve

def main():
    # Load and preprocess the data
    print("Loading and preprocessing data...")
    x_train, x_val, label_train, label_val = preprocess_data()

    # Train the model
    print("Training the model...")
    model = train_model(x_train, label_train, config)

    # Save the model
    if config['save_model']:
        model.save(config['model_save_path'])
        print(f"Model saved to {config['model_save_path']}")

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = predict_model(model, X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Plot the training history
    if config['plot_history']:
        plot_history(model.history)

    # Plot confusion matrix
    if config['plot_confusion_matrix']:
        plot_confusion_matrix(y_test, y_pred, classes=config['class_names'])

    # Plot ROC curve
    if config['plot_roc_curve']:
        fpr, tpr, _ = roc_curve(y_test, model.predict(X_test)[:, 1])
        auc_score = auc(fpr, tpr)
        plot_roc_curve(fpr, tpr, auc_score)

if __name__ == "__main__":
    main()


