"""
preprocess.py — Data Preprocessing for Plant Disease Detection CNN
CVRT Engineering College, Tadipatri | JNTUA B.Tech CSE 2024-25
Student: Chaganti Nithin — HT NO: 239B1A0528
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# ─── Configuration ───────────────────────────────────────────────────────────
DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'plantvillage')
IMG_SIZE = (224, 224)       # MobileNetV2 input size
BATCH_SIZE = 32
RANDOM_STATE = 42
TRAIN_SPLIT = 0.8           # 80% training, 20% testing
VALIDATION_SPLIT = 0.2      # 20% validation from training data


def get_class_labels(dataset_dir: str) -> list:
    """
    Scan the dataset directory and return a sorted list of class label names.
    Each subdirectory in the dataset directory represents one disease class.

    Args:
        dataset_dir (str): Path to the root dataset directory

    Returns:
        list: Sorted list of class names (e.g., ['Apple___Apple_scab', ...])
    """
    train_dir = os.path.join(dataset_dir, 'train')
    if os.path.isdir(train_dir):
        # Pre-split dataset
        classes = sorted([d for d in os.listdir(train_dir)
                          if os.path.isdir(os.path.join(train_dir, d))])
    else:
        # Single directory — get all subdirectories as classes
        classes = sorted([d for d in os.listdir(dataset_dir)
                          if os.path.isdir(os.path.join(dataset_dir, d))])
    return classes


def save_class_labels(class_labels: list, output_path: str) -> None:
    """
    Save class labels to a JSON file so that the prediction API can decode
    model output indices back to human-readable disease names.

    Args:
        class_labels (list): List of class name strings
        output_path (str): File path to save the JSON (e.g., 'ml/class_labels.json')
    """
    label_map = {i: label for i, label in enumerate(class_labels)}
    with open(output_path, 'w') as f:
        json.dump(label_map, f, indent=2)
    print(f"[✓] Class labels saved → {output_path}")
    print(f"    Total classes: {len(class_labels)}")


def create_data_generators(dataset_dir: str):
    """
    Create Keras ImageDataGenerator objects with data augmentation for training
    and standard normalization for validation/testing.

    Data augmentation prevents overfitting by artificially expanding the training set
    through random transformations of existing images.

    Augmentation applied to TRAINING data:
        - Horizontal + vertical flips
        - Rotation up to 20 degrees
        - Width/height shift up to 10%
        - Zoom up to 15%
        - Brightness adjustment ±20%
        - Fill mode: nearest (fill gaps after transformation)

    Normalization applied to ALL data:
        - MobileNetV2 preprocess_input: scales pixels from [0,255] to [-1,1]

    Args:
        dataset_dir (str): Root path to dataset directory

    Returns:
        tuple: (train_generator, val_generator, test_generator, class_labels)
    """
    # ── Training augmentation ────────────────────────────────────────────────
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=VALIDATION_SPLIT
    )

    # ── Test/prediction: only normalization, NO augmentation ─────────────────
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    # ── Determine directory structure ────────────────────────────────────────
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')

    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        # Pre-split (Kaggle vipoooool split version)
        print("[INFO] Detected pre-split dataset (train/ and val/ directories)")

        train_generator = train_datagen.flow_from_directory(
            directory=train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=True,
            seed=RANDOM_STATE
        )

        val_generator = test_datagen.flow_from_directory(
            directory=val_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )

        test_generator = val_generator   # Use val as test for single-split dataset

    else:
        # Single root directory — use Keras validation_split to create train/val
        print("[INFO] Detected single-directory dataset, applying train/val split")

        train_generator = train_datagen.flow_from_directory(
            directory=dataset_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=RANDOM_STATE
        )

        val_generator = train_datagen.flow_from_directory(
            directory=dataset_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

        test_generator = val_generator

    # ── Extract class labels ─────────────────────────────────────────────────
    class_labels = list(train_generator.class_indices.keys())
    print(f"\n[✓] Dataset loaded successfully")
    print(f"    Training samples  : {train_generator.samples}")
    print(f"    Validation samples: {val_generator.samples}")
    print(f"    Number of classes : {len(class_labels)}")

    return train_generator, val_generator, test_generator, class_labels


def analyze_dataset(dataset_dir: str) -> dict:
    """
    Analyze dataset statistics including class distribution and image counts.
    Generates a bar plot of class distribution to detect class imbalance.

    Args:
        dataset_dir (str): Root path to dataset directory

    Returns:
        dict: Statistics {class_name: image_count, total: int, min: int, max: int}
    """
    print("\n[INFO] Analyzing dataset statistics...")

    stats = {}
    root = os.path.join(dataset_dir, 'train') if os.path.isdir(
        os.path.join(dataset_dir, 'train')) else dataset_dir

    if not os.path.isdir(root):
        print(f"[ERROR] Dataset directory not found: {root}")
        print("        Please download the PlantVillage dataset first.")
        print("        See datasets/README.md for instructions.")
        return {}

    for cls in sorted(os.listdir(root)):
        cls_path = os.path.join(root, cls)
        if os.path.isdir(cls_path):
            count = len([f for f in os.listdir(cls_path)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            stats[cls] = count

    if not stats:
        return {}

    total = sum(stats.values())
    print(f"\n    Total images  : {total:,}")
    print(f"    Total classes : {len(stats)}")
    print(f"    Min per class : {min(stats.values())} ({min(stats, key=stats.get)})")
    print(f"    Max per class : {max(stats.values())} ({max(stats, key=stats.get)})")
    print(f"    Avg per class : {total // len(stats)}")

    # Plot class distribution
    plt.figure(figsize=(20, 8))
    classes = list(stats.keys())
    counts = list(stats.values())
    short_names = [c.replace('___', '\n').replace('_', ' ') for c in classes]

    bars = plt.bar(range(len(classes)), counts, color='#2ecc71', edgecolor='#27ae60')
    plt.xticks(range(len(classes)), short_names, rotation=90, fontsize=7)
    plt.xlabel('Plant Disease Class', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.title('PlantVillage Dataset — Class Distribution', fontsize=14, fontweight='bold')
    plt.ylim(0, max(counts) * 1.15)
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), 'class_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[✓] Class distribution chart saved → {output_path}")

    return {
        'class_counts': stats,
        'total': total,
        'num_classes': len(stats),
        'min_count': min(stats.values()),
        'max_count': max(stats.values()),
        'avg_count': total // len(stats)
    }


def preprocess_single_image(image_path: str) -> np.ndarray:
    """
    Preprocess a single image for prediction by the trained CNN model.
    Applies the same normalization as used during training (MobileNetV2 preprocess_input).

    Steps:
        1. Load image using Keras (handles JPG, PNG, WEBP)
        2. Resize to (224, 224) to match model input
        3. Convert to numpy array (float32)
        4. Expand dims to add batch dimension: shape becomes (1, 224, 224, 3)
        5. Apply MobileNetV2 preprocess_input: scales to [-1, 1]

    Args:
        image_path (str): Absolute path to the image file

    Returns:
        np.ndarray: Preprocessed image array of shape (1, 224, 224, 3)

    Raises:
        FileNotFoundError: If the image file does not exist
        ValueError: If the image cannot be loaded or has invalid format
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)      # shape: (224,224,3)
        img_array = np.expand_dims(img_array, axis=0)                    # shape: (1,224,224,3)
        img_array = preprocess_input(img_array)                          # normalize to [-1,1]
        return img_array
    except Exception as e:
        raise ValueError(f"Failed to load image '{image_path}': {str(e)}")


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess an image uploaded via HTTP multipart form (bytes) for prediction.
    Used by the Flask API to process images from the web frontend.

    Args:
        image_bytes (bytes): Raw image bytes from an HTTP upload (JPG/PNG)

    Returns:
        np.ndarray: Preprocessed image array of shape (1, 224, 224, 3)
    """
    import io
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB (handles RGBA PNGs, grayscale, etc.)
        img = img.convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        raise ValueError(f"Failed to process uploaded image: {str(e)}")


if __name__ == '__main__':
    """
    Run this script directly to verify dataset structure and view statistics.
    Usage: python ml/preprocess.py
    """
    print("=" * 60)
    print("  Plant Disease Detection — Dataset Preprocessing")
    print("  CVRT Engineering College | JNTUA CSE 2024-25")
    print("=" * 60)

    stats = analyze_dataset(DATASET_DIR)

    if stats:
        # Save class labels
        class_labels = get_class_labels(DATASET_DIR)
        labels_path = os.path.join(os.path.dirname(__file__), 'class_labels.json')
        save_class_labels(class_labels, labels_path)

        print("\n[✓] Preprocessing analysis complete.")
        print("    Run 'python ml/train.py' to train the CNN model.")
    else:
        print("\n[!] Dataset not found. Please download it first.")
        print("    See datasets/README.md for instructions.")
