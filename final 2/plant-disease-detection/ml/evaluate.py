"""
evaluate.py — Model Evaluation & Performance Metrics
Generates confusion matrix, classification report, and comparison charts.
CVRT Engineering College, Tadipatri | JNTUA B.Tech CSE 2024-25
"""

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import create_data_generators, DATASET_DIR

ML_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ML_DIR, 'plant_disease_model.h5')
LABELS_PATH = os.path.join(ML_DIR, 'class_labels.json')


def load_model_and_labels():
    """Load the trained model and class label dictionary."""
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run 'python ml/train.py' first."
        )
    print("[INFO] Loading trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABELS_PATH, 'r') as f:
        label_map = json.load(f)
    # Convert string keys (from JSON) to int
    label_map = {int(k): v for k, v in label_map.items()}
    class_names = [label_map[i] for i in sorted(label_map.keys())]
    return model, class_names


def evaluate_model(model, test_gen, class_names: list) -> dict:
    """
    Run the model on the test set and compute all evaluation metrics.

    Metrics computed:
        - Overall accuracy
        - Precision (weighted average across all classes)
        - Recall (weighted average across all classes)
        - F1-Score (weighted average across all classes)
        - Per-class precision, recall, F1 (via classification_report)
        - Confusion matrix

    Args:
        model: Loaded Keras model
        test_gen: Test data generator
        class_names (list): List of class name strings

    Returns:
        dict: All computed metrics
    """
    print("[INFO] Running predictions on test set...")
    print(f"       Test samples: {test_gen.samples}")

    test_gen.reset()
    raw_predictions = model.predict(test_gen, verbose=1)
    predicted_classes = np.argmax(raw_predictions, axis=1)

    # Get true labels from generator
    true_classes = test_gen.classes
    # Trim to generator length (in case last batch is incomplete)
    min_len = min(len(true_classes), len(predicted_classes))
    true_classes = true_classes[:min_len]
    predicted_classes = predicted_classes[:min_len]

    # ── Core metrics ──────────────────────────────────────────────────────────
    accuracy  = accuracy_score(true_classes, predicted_classes)
    precision = precision_score(true_classes, predicted_classes, average='weighted', zero_division=0)
    recall    = recall_score(true_classes, predicted_classes, average='weighted', zero_division=0)
    f1        = f1_score(true_classes, predicted_classes, average='weighted', zero_division=0)
    cm        = confusion_matrix(true_classes, predicted_classes)
    report    = classification_report(true_classes, predicted_classes,
                                       target_names=class_names, zero_division=0)

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy  : {accuracy*100:.2f}%")
    print(f"  Precision : {precision*100:.2f}%")
    print(f"  Recall    : {recall*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print("=" * 60)
    print("\nPer-Class Report:\n")
    print(report)

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'num_test_samples': min_len
    }


def plot_confusion_matrix(cm: list, class_names: list, top_n: int = 15):
    """
    Plot a heatmap of the confusion matrix for the top-N most confused classes.
    (Plotting all 38 classes makes the plot unreadable.)

    Args:
        cm (list): Confusion matrix (list of lists)
        class_names (list): Class name strings
        top_n (int): Number of classes to show (default 15 most common)
    """
    cm_array = np.array(cm)

    # Show top_n classes by total count
    totals = cm_array.sum(axis=1)
    top_indices = np.argsort(totals)[-top_n:][::-1]
    sub_cm = cm_array[np.ix_(top_indices, top_indices)]
    sub_names = [class_names[i].replace('___', '\n').replace('_', ' ')[:25]
                 for i in top_indices]

    plt.figure(figsize=(16, 12))
    sns.heatmap(sub_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sub_names, yticklabels=sub_names,
                linewidths=0.5)
    plt.title(f'Confusion Matrix — Top {top_n} Classes\nPlant Disease CNN',
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    output_path = os.path.join(ML_DIR, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Confusion matrix saved → {output_path}")


def plot_per_class_accuracy(cm: list, class_names: list):
    """
    Bar chart of per-class accuracy (true positives / total per class).

    Args:
        cm (list): Confusion matrix
        class_names (list): Class name strings
    """
    cm_array = np.array(cm)
    per_class_acc = cm_array.diagonal() / cm_array.sum(axis=1).clip(min=1)
    short_names = [c.replace('___', ' — ').replace('_', ' ')
                   for c in class_names]

    plt.figure(figsize=(18, 7))
    colors = ['#27ae60' if acc >= 0.9 else '#f39c12' if acc >= 0.7 else '#e74c3c'
              for acc in per_class_acc]
    bars = plt.bar(range(len(class_names)), per_class_acc * 100, color=colors, edgecolor='white')
    plt.xticks(range(len(class_names)), short_names, rotation=90, fontsize=7)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xlabel('Disease Class', fontsize=12)
    plt.title('Per-Class Accuracy — Plant Disease CNN', fontsize=14, fontweight='bold')
    plt.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90% threshold')
    plt.ylim(0, 110)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(ML_DIR, 'per_class_accuracy.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Per-class accuracy chart saved → {output_path}")


def save_metrics(metrics: dict):
    """Save evaluation metrics to JSON for use by the Flask API statistics endpoint."""
    metrics_path = os.path.join(ML_DIR, 'evaluation_metrics.json')
    save_metrics_out = {k: v for k, v in metrics.items() if k != 'classification_report'}
    with open(metrics_path, 'w') as f:
        json.dump(save_metrics_out, f, indent=2)
    print(f"[✓] Metrics saved → {metrics_path}")


def evaluate():
    """Main evaluation pipeline — runs after training is complete."""
    print("=" * 60)
    print("  Plant Disease CNN — Model Evaluation")
    print("  CVRT Engineering College | JNTUA CSE 2024-25")
    print("=" * 60)

    if not os.path.isdir(DATASET_DIR):
        print(f"\n[ERROR] Dataset not found at: {DATASET_DIR}")
        return

    model, class_names = load_model_and_labels()
    _, _, test_gen, _ = create_data_generators(DATASET_DIR)

    metrics = evaluate_model(model, test_gen, class_names)
    plot_confusion_matrix(metrics['confusion_matrix'], class_names)
    plot_per_class_accuracy(metrics['confusion_matrix'], class_names)
    save_metrics(metrics)

    print("\n[✓] Evaluation complete.")
    print("    Run 'python run.py' to start the web application.")


if __name__ == '__main__':
    evaluate()
