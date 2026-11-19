"""
train.py — CNN Model Training for Plant Disease Detection
Uses MobileNetV2 transfer learning for high accuracy with limited compute.
CVRT Engineering College, Tadipatri | JNTUA B.Tech CSE 2024-25
Student: Chaganti Nithin — HT NO: 239B1A0528
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from datetime import datetime

# Import our preprocessing utilities
from preprocess import (
    create_data_generators,
    save_class_labels,
    DATASET_DIR,
    IMG_SIZE,
    BATCH_SIZE
)

# ─── Configuration ────────────────────────────────────────────────────────────
ML_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ML_DIR, 'plant_disease_model.h5')
LABELS_PATH = os.path.join(ML_DIR, 'class_labels.json')
HISTORY_PATH = os.path.join(ML_DIR, 'training_history.json')

EPOCHS_PHASE1 = 10          # Phase 1: train only top layers (frozen base)
EPOCHS_PHASE2 = 20          # Phase 2: fine-tune top layers of base model
LEARNING_RATE_PHASE1 = 1e-3
LEARNING_RATE_PHASE2 = 1e-4
FINE_TUNE_AT = 100          # Unfreeze from this layer onwards in MobileNetV2


def build_model(num_classes: int) -> Model:
    """
    Build the CNN model using MobileNetV2 as a pre-trained backbone with
    custom classification head for plant disease classification.

    Architecture:
        MobileNetV2 (pre-trained on ImageNet, weights frozen in Phase 1)
        → GlobalAveragePooling2D (converts feature maps to vector)
        → BatchNormalization (stabilize training)
        → Dense(512, relu, L2 regularizer) → Dropout(0.4)
        → Dense(256, relu, L2 regularizer) → Dropout(0.3)
        → Dense(num_classes, softmax) → Output: probability per class

    Why MobileNetV2?
        - Lightweight yet accurate (efficient for CPU/low-end GPU)
        - Pre-trained features from ImageNet transfer well to leaf images
        - Achieves >92% accuracy with significantly less training time vs VGG/ResNet

    Args:
        num_classes (int): Number of disease classes (38 for PlantVillage)

    Returns:
        tf.keras.Model: Compiled Keras model ready for training
    """
    print("\n[INFO] Building MobileNetV2-based CNN model...")

    # ── Base model: MobileNetV2 pre-trained on ImageNet ──────────────────────
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,          # Remove original ImageNet classification head
        weights='imagenet'          # Start with ImageNet pre-trained weights
    )

    # Phase 1: Freeze ALL base model layers — only train the custom head
    base_model.trainable = False

    # ── Custom classification head ────────────────────────────────────────────
    x = base_model.output
    x = GlobalAveragePooling2D()(x)              # Reduce (7,7,1280) → (1280,)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu',
               kernel_regularizer=l2(1e-4),
               name='dense_512')(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu',
               kernel_regularizer=l2(1e-4),
               name='dense_256')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    # ── Compile: Phase 1 (higher learning rate — only training head) ──────────
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_PHASE1),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )

    print(f"    Base model trainable params  : 0 (frozen)")
    print(f"    Custom head trainable params : {model.count_params() - base_model.count_params():,}")
    print(f"    Total params                 : {model.count_params():,}")

    return model, base_model


def get_callbacks(phase: int) -> list:
    """
    Create Keras training callbacks for monitoring and optimization.

    Callbacks:
        - ModelCheckpoint: Save best model automatically based on val_accuracy
        - EarlyStopping: Stop training if no improvement for N epochs
        - ReduceLROnPlateau: Halve learning rate if val_loss plateaus
        - TensorBoard: Training visualization logs (optional)

    Args:
        phase (int): Training phase (1 = head only, 2 = fine-tuning)

    Returns:
        list: List of Keras callback objects
    """
    callbacks = [
        # Save the best model checkpoint based on validation accuracy
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        # Stop early if validation accuracy doesn't improve
        EarlyStopping(
            monitor='val_accuracy',
            patience=5 if phase == 1 else 7,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when validation loss stops improving
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    return callbacks


def fine_tune_model(model: Model, base_model: Model) -> Model:
    """
    Phase 2: Unfreeze the top layers of MobileNetV2 for fine-tuning.

    Unfreezes layers from index FINE_TUNE_AT onwards (top layers of the base model).
    Uses a much lower learning rate to make small weight adjustments without
    destroying the pre-trained features learned from ImageNet.

    Strategy:
        - Layers 0 to FINE_TUNE_AT: Keep frozen (early features = edges, textures)
        - Layers FINE_TUNE_AT onwards: Unfreeze (higher-level semantic features)

    Args:
        model (Model): Complete model (base + head)
        base_model (Model): MobileNetV2 base model component

    Returns:
        Model: Recompiled model with unfrozen top layers
    """
    print(f"\n[INFO] Phase 2: Fine-tuning from layer {FINE_TUNE_AT} onwards...")

    base_model.trainable = True

    # Freeze all layers BEFORE fine_tune_at
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_PHASE2),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )

    trainable_count = sum(1 for layer in model.layers if layer.trainable)
    print(f"    Trainable layers: {trainable_count} / {len(model.layers)}")

    return model


def plot_training_history(history_phase1, history_phase2):
    """
    Plot and save training vs validation accuracy and loss curves for both phases.

    Generates a 2×2 subplot figure:
        Top-left    : Accuracy curves (Phase 1)
        Top-right   : Loss curves (Phase 1)
        Bottom-left : Accuracy curves (Phase 2 fine-tuning)
        Bottom-right: Loss curves (Phase 2 fine-tuning)

    Args:
        history_phase1: Keras History object from Phase 1 training
        history_phase2: Keras History object from Phase 2 fine-tuning
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Plant Disease CNN — Training History', fontsize=16, fontweight='bold')

    for idx, (history, title_prefix) in enumerate([
        (history_phase1, 'Phase 1 (Head Training)'),
        (history_phase2, 'Phase 2 (Fine-Tuning)')
    ]):
        epochs = range(1, len(history.history['accuracy']) + 1)

        # Accuracy plot
        axes[idx][0].plot(epochs, history.history['accuracy'], 'b-o', label='Training', linewidth=2)
        axes[idx][0].plot(epochs, history.history['val_accuracy'], 'r-s', label='Validation', linewidth=2)
        axes[idx][0].set_title(f'{title_prefix} — Accuracy')
        axes[idx][0].set_xlabel('Epoch')
        axes[idx][0].set_ylabel('Accuracy')
        axes[idx][0].legend()
        axes[idx][0].grid(True, alpha=0.3)

        # Loss plot
        axes[idx][1].plot(epochs, history.history['loss'], 'b-o', label='Training', linewidth=2)
        axes[idx][1].plot(epochs, history.history['val_loss'], 'r-s', label='Validation', linewidth=2)
        axes[idx][1].set_title(f'{title_prefix} — Loss')
        axes[idx][1].set_xlabel('Epoch')
        axes[idx][1].set_ylabel('Loss')
        axes[idx][1].legend()
        axes[idx][1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(ML_DIR, 'training_history.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Training history plot saved → {plot_path}")


def train():
    """
    Main training function — executes the complete two-phase training pipeline.

    Pipeline:
        1. Load and augment dataset via data generators
        2. Save class labels for later decoding predictions
        3. Build MobileNetV2-based model
        4. Phase 1: Train custom head with frozen base (10 epochs)
        5. Phase 2: Fine-tune top layers with lower learning rate (20 epochs)
        6. Save final model, training history, and evaluation plots
    """
    print("=" * 60)
    print("  Plant Disease CNN — Model Training")
    print("  CVRT Engineering College | JNTUA CSE 2024-25")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Step 1: Load dataset ──────────────────────────────────────────────────
    if not os.path.isdir(DATASET_DIR):
        print(f"\n[ERROR] Dataset not found at: {DATASET_DIR}")
        print("        Please download PlantVillage dataset first.")
        print("        See datasets/README.md for instructions.")
        return

    train_gen, val_gen, test_gen, class_labels = create_data_generators(DATASET_DIR)
    num_classes = len(class_labels)

    # ── Step 2: Save class labels ─────────────────────────────────────────────
    save_class_labels(class_labels, LABELS_PATH)

    # ── Step 3: Build model ───────────────────────────────────────────────────
    model, base_model = build_model(num_classes)

    # ── Step 4: Phase 1 — Train custom head ──────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  PHASE 1: Training custom classification head")
    print(f"  Epochs: {EPOCHS_PHASE1} | LR: {LEARNING_RATE_PHASE1}")
    print(f"{'─'*50}")

    steps_per_epoch = train_gen.samples // BATCH_SIZE
    validation_steps = val_gen.samples // BATCH_SIZE

    history_phase1 = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS_PHASE1,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=get_callbacks(phase=1),
        verbose=1
    )

    phase1_val_acc = max(history_phase1.history['val_accuracy'])
    print(f"\n[✓] Phase 1 complete. Best val accuracy: {phase1_val_acc:.4f} ({phase1_val_acc*100:.2f}%)")

    # ── Step 5: Phase 2 — Fine-tune ───────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  PHASE 2: Fine-tuning (unfreeze top layers)")
    print(f"  Epochs: {EPOCHS_PHASE2} | LR: {LEARNING_RATE_PHASE2}")
    print(f"{'─'*50}")

    model = fine_tune_model(model, base_model)

    history_phase2 = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS_PHASE1 + EPOCHS_PHASE2,
        initial_epoch=EPOCHS_PHASE1,      # Continue from where Phase 1 ended
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=get_callbacks(phase=2),
        verbose=1
    )

    phase2_val_acc = max(history_phase2.history['val_accuracy'])
    print(f"\n[✓] Phase 2 complete. Best val accuracy: {phase2_val_acc:.4f} ({phase2_val_acc*100:.2f}%)")

    # ── Step 6: Save training history ─────────────────────────────────────────
    full_history = {
        'phase1_accuracy': history_phase1.history['accuracy'],
        'phase1_val_accuracy': history_phase1.history['val_accuracy'],
        'phase1_loss': history_phase1.history['loss'],
        'phase1_val_loss': history_phase1.history['val_loss'],
        'phase2_accuracy': history_phase2.history['accuracy'],
        'phase2_val_accuracy': history_phase2.history['val_accuracy'],
        'phase2_loss': history_phase2.history['loss'],
        'phase2_val_loss': history_phase2.history['val_loss'],
        'final_accuracy': float(phase2_val_acc),
        'num_classes': num_classes,
        'training_samples': train_gen.samples,
        'validation_samples': val_gen.samples,
        'completed_at': datetime.now().isoformat()
    }

    with open(HISTORY_PATH, 'w') as f:
        json.dump(full_history, f, indent=2)
    print(f"[✓] Training history saved → {HISTORY_PATH}")

    # ── Step 7: Plot training curves ─────────────────────────────────────────
    plot_training_history(history_phase1, history_phase2)

    print("\n" + "=" * 60)
    print(f"  TRAINING COMPLETE")
    print(f"  Final Validation Accuracy : {phase2_val_acc*100:.2f}%")
    print(f"  Model saved to            : {MODEL_PATH}")
    print(f"  Class labels saved to     : {LABELS_PATH}")
    print("=" * 60)
    print("\nNext step: Run 'python ml/evaluate.py' to get full metrics.")


if __name__ == '__main__':
    # Set TensorFlow to use GPU if available, otherwise CPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[✓] GPU detected: {len(gpus)} GPU(s). Using GPU acceleration.")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("[INFO] No GPU detected. Training on CPU (slower, ~20-30 min).")

    train()
