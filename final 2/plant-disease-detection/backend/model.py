"""
model.py — ML Model Prediction Wrapper
Loads the trained CNN model and exposes a clean predict() interface.
CVRT Engineering College, Tadipatri | JNTUA B.Tech CSE 2024-25
"""

import os
import json
import numpy as np
import tensorflow as tf

ML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ml')
MODEL_PATH = os.path.join(ML_DIR, 'plant_disease_model.h5')
LABELS_PATH = os.path.join(ML_DIR, 'class_labels.json')

# Singleton model instance — loaded once and reused for all requests
_model = None
_class_labels = None


def load_model():
    """
    Load the trained CNN model and class labels into memory.
    Uses a singleton pattern — model is loaded only once and cached globally.

    Returns:
        tuple: (model, class_labels_dict)

    Raises:
        FileNotFoundError: If model or labels file does not exist
    """
    global _model, _class_labels

    if _model is not None:
        return _model, _class_labels

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at: {MODEL_PATH}\n"
            f"Please run 'python ml/train.py' first to train the model."
        )

    if not os.path.isfile(LABELS_PATH):
        raise FileNotFoundError(
            f"Class labels not found at: {LABELS_PATH}\n"
            f"Please run 'python ml/train.py' first."
        )

    print("[INFO] Loading CNN model (first request may take a few seconds)...")
    _model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABELS_PATH, 'r') as f:
        label_map = json.load(f)
    _class_labels = {int(k): v for k, v in label_map.items()}

    print(f"[✓] Model loaded. Supports {len(_class_labels)} disease classes.")
    return _model, _class_labels


def predict(image_array: np.ndarray) -> dict:
    """
    Run CNN prediction on a preprocessed image array.

    Args:
        image_array (np.ndarray): Preprocessed image of shape (1, 224, 224, 3),
                                  values normalized to [-1, 1] by MobileNetV2 preprocess_input.

    Returns:
        dict: {
            'predicted_class': str,       # e.g., 'Tomato___Late_blight'
            'confidence': float,           # e.g., 0.9734 (probability)
            'top3': [                      # Top 3 predictions
                {'label': str, 'confidence': float},
                ...
            ]
        }
    """
    model, class_labels = load_model()

    # Run forward pass through the CNN
    predictions = model.predict(image_array, verbose=0)[0]   # shape: (num_classes,)

    # Get top-3 predictions by confidence
    top3_indices = np.argsort(predictions)[-3:][::-1]
    top3 = [
        {
            'label': class_labels.get(int(idx), f'Class {idx}'),
            'confidence': float(predictions[idx])
        }
        for idx in top3_indices
    ]

    return {
        'predicted_class': top3[0]['label'],
        'confidence': top3[0]['confidence'],
        'top3': top3
    }


def get_model_info() -> dict:
    """Return basic information about the loaded model for the /api/stats endpoint."""
    try:
        model, class_labels = load_model()
        return {
            'model_type': 'MobileNetV2 CNN (Transfer Learning)',
            'num_classes': len(class_labels),
            'input_shape': str(model.input_shape),
            'model_loaded': True
        }
    except Exception as e:
        return {'model_loaded': False, 'error': str(e)}
