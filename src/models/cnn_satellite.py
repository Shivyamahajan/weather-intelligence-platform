"""
CNN for Satellite Cloud Image Classification
Project: Weather Intelligence Platform
Author: Shivya
Date: July 2026

Description:
    Trains a CNN to classify satellite cloud images into categories
    that correlate with rainfall probability:
    
    - Clear Sky: No significant clouds, very low rainfall probability
    - Low Cloud: Stratus/stratocumulus, light rainfall possible
    - Medium Cloud: Altocumulus/altostratus, moderate rainfall
    - Deep Convection: Cumulonimbus, heavy rainfall likely
    - Cyclonic Pattern: Spiral cloud formation, extreme weather
    
What is a CNN?
    Convolutional Neural Networks detect spatial patterns in images.
    Instead of looking at raw pixel values, CNNs learn to detect
    features like edges, textures, shapes, and complex patterns.
    
    Layers:
    - Conv2D: detects local patterns (edges, cloud boundaries)
    - MaxPooling: reduces image size, keeps important features
    - BatchNorm: stabilizes training
    - Dense: combines learned features for classification
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import requests
import zipfile
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import (MobileNetV2, ResNet50, 
                                            EfficientNetB0)

# ─── Configuration ───
IMG_SIZE    = 224      # resize all images to 224×224
BATCH_SIZE  = 32
EPOCHS      = 30
NUM_CLASSES = 4        # Clear, Cloudy, Overcast, Rain
RANDOM_SEED = 42


def download_sample_data():
    """
    Download a sample cloud image dataset.
    We use the CCSN (Cirrus Cumulus Stratus Nimbus) dataset
    which is publicly available.
    
    If download fails, we create synthetic sample images
    to demonstrate the pipeline.
    """
    data_dir = Path('data/external/cloud_images')
    
    if data_dir.exists() and len(list(data_dir.rglob('*.jpg'))) > 10:
        print(f"Cloud images already exist: {data_dir}")
        return data_dir
    
    print("Setting up cloud image dataset...")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to download from Kaggle
    try:
        import kaggle
        print("Downloading cloud classification dataset from Kaggle...")
        os.system('kaggle datasets download -d sshikamaru/cloud-type-classification -p data/external/cloud_images --unzip')
        print("Dataset downloaded!")
        return data_dir
    except Exception as e:
        print(f"Kaggle download not available: {e}")
        print("Creating synthetic dataset for demonstration...")
        return create_synthetic_dataset(data_dir)


def create_synthetic_dataset(data_dir):
    """
    Create synthetic cloud images if real data not available.
    This demonstrates the pipeline even without internet access.
    """
    from PIL import Image
    import random
    
    classes = {
        'clear_sky': (135, 206, 235),       # light blue
        'light_cloud': (200, 200, 200),      # light gray
        'heavy_cloud': (100, 100, 100),      # dark gray
        'rain_cloud': (50, 50, 80),          # dark blue-gray
    }
    
    print("Creating synthetic cloud images...")
    
    for split in ['train', 'validation', 'test']:
        for class_name, base_color in classes.items():
            class_dir = data_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            n_images = 80 if split == 'train' else 20
            
            for i in range(n_images):
                # Create image with noise to simulate clouds
                img_array = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                
                # Base color
                for c, val in enumerate(base_color):
                    img_array[:,:,c] = val
                
                # Add noise (simulates cloud texture)
                noise = np.random.randint(-30, 30, (IMG_SIZE, IMG_SIZE, 3))
                img_array = np.clip(img_array.astype(int) + noise, 0, 255).astype(np.uint8)
                
                # Add cloud-like circles
                for _ in range(random.randint(3, 10)):
                    cx = random.randint(20, IMG_SIZE-20)
                    cy = random.randint(20, IMG_SIZE-20)
                    r  = random.randint(10, 50)
                    y_grid, x_grid = np.ogrid[:IMG_SIZE, :IMG_SIZE]
                    mask = (x_grid - cx)**2 + (y_grid - cy)**2 <= r**2
                    img_array[mask] = np.clip(
                        img_array[mask].astype(int) + random.randint(-20, 20),
                        0, 255
                    ).astype(np.uint8)
                
                img = Image.fromarray(img_array)
                img.save(class_dir / f'cloud_{i:04d}.jpg')
    
    print(f"Synthetic dataset created in {data_dir}")
    return data_dir


def build_cnn_scratch(input_shape, num_classes):
    """
    Build a CNN from scratch.
    
    Architecture:
    Input Image (224 × 224 × 3)
        ↓
    Conv2D(32) → BatchNorm → MaxPool    ← detect edges and simple textures
        ↓
    Conv2D(64) → BatchNorm → MaxPool    ← detect cloud shapes
        ↓
    Conv2D(128) → BatchNorm → MaxPool   ← detect complex cloud patterns
        ↓
    Conv2D(256) → BatchNorm → MaxPool   ← detect high-level cloud types
        ↓
    GlobalAveragePooling                ← convert feature maps to vector
        ↓
    Dense(256) → Dropout(0.5)           ← learn classification rules
        ↓
    Dense(num_classes) → Softmax        ← output class probabilities
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3,3), activation='relu', 
                      padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        # Block 2
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        # Block 3
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        # Block 4
        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        # Classifier head
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='CNN_Scratch')
    
    return model


def build_transfer_learning_model(input_shape, num_classes):
    """
    Transfer learning with MobileNetV2.
    
    What is Transfer Learning?
    Instead of training from scratch, we use a model that was already
    trained on 1.4 million images (ImageNet). The model has already
    learned to detect edges, textures, shapes, and objects.
    
    We take this pre-trained model, freeze its weights (so we don't
    lose what it learned), and add our own classification layers on top.
    Then we only train our new layers on cloud images.
    
    This works amazingly well even with small datasets because
    the features learned from natural images transfer well to
    satellite cloud images.
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,     # exclude ImageNet classification head
        weights='imagenet'     # use pre-trained weights
    )
    
    # Freeze base model
    base_model.trainable = False
    print(f"MobileNetV2 base: {len(base_model.layers)} layers (frozen)")
    
    # Add custom head
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs, name='MobileNetV2_Transfer')
    
    return model


def prepare_data_generators(data_dir):
    """
    Create data generators with augmentation.
    
    Data Augmentation: artificially increases dataset size by
    creating modified versions of existing images:
    - Rotation: rotate images slightly
    - Flip: mirror images horizontally
    - Zoom: zoom in slightly
    - Brightness: adjust image brightness
    
    This prevents overfitting and makes the model more robust.
    """
    # Training generator with augmentation
    train_gen = ImageDataGenerator(
        rescale=1./255,           # normalize pixel values to [0,1]
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Validation/test generator — NO augmentation, just rescale
    val_gen = ImageDataGenerator(rescale=1./255)
    
    train_dir = data_dir / 'train'
    val_dir   = data_dir / 'validation'
    test_dir  = data_dir / 'test'
    
    if not train_dir.exists():
        # Flat structure — create split
        train_dir = data_dir
        val_dir   = data_dir
    
    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=RANDOM_SEED
    )
    
    val_data = val_gen.flow_from_directory(
        val_dir if val_dir != train_dir else train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"\nClass mapping: {train_data.class_indices}")
    print(f"Training samples: {train_data.samples}")
    print(f"Validation samples: {val_data.samples}")
    
    return train_data, val_data


def train_cnn(model, train_data, val_data, model_name):
    """Compile and train CNN model."""
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        )
    ]
    
    print(f"\nTraining {model_name}...")
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def plot_cnn_history(history, model_name):
    """Plot training history for CNN."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history.history['loss'], label='Train Loss', 
             color='steelblue', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss',
             color='red', linewidth=2)
    ax1.set_title(f'{model_name} — Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['accuracy'], label='Train Accuracy',
             color='steelblue', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy',
             color='red', linewidth=2)
    ax2.set_title(f'{model_name} — Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'reports/figures/cnn_{model_name.lower()}_history.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"CNN training history saved!")


if __name__ == "__main__":
    print("=" * 60)
    print("CNN SATELLITE IMAGE CLASSIFICATION")
    print("=" * 60)
    
    # Setup data
    data_dir   = download_sample_data()
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    
    train_data, val_data = prepare_data_generators(data_dir)
    num_classes = len(train_data.class_indices)
    
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {list(train_data.class_indices.keys())}")
    
    # Build models
    cnn_scratch    = build_cnn_scratch(input_shape, num_classes)
    cnn_transfer   = build_transfer_learning_model(input_shape, num_classes)
    
    # Train CNN from scratch
    print("\n--- Training CNN from Scratch ---")
    history_scratch = train_cnn(cnn_scratch, train_data, val_data, 'CNN_Scratch')
    plot_cnn_history(history_scratch, 'CNN_Scratch')
    
    # Evaluate
    val_loss, val_acc = cnn_scratch.evaluate(val_data, verbose=0)
    print(f"\nCNN Scratch — Val Accuracy: {val_acc:.4f}")
    
    # Train Transfer Learning model
    print("\n--- Training MobileNetV2 Transfer Learning ---")
    history_transfer = train_cnn(cnn_transfer, train_data, val_data, 'MobileNetV2')
    plot_cnn_history(history_transfer, 'MobileNetV2')
    
    val_loss2, val_acc2 = cnn_transfer.evaluate(val_data, verbose=0)
    print(f"\nMobileNetV2 — Val Accuracy: {val_acc2:.4f}")
    
    # Save models
    os.makedirs('models', exist_ok=True)
    cnn_scratch.save('models/cnn_scratch.keras')
    cnn_transfer.save('models/cnn_mobilenet.keras')
    
    print("\n" + "="*55)
    print("CNN RESULTS SUMMARY")
    print("="*55)
    print(f"CNN from Scratch:     {val_acc:.4f} accuracy")
    print(f"MobileNetV2 Transfer: {val_acc2:.4f} accuracy")
    print(f"\nExpected: Transfer learning should outperform scratch CNN")
    print(f"because it leverages pre-trained image features.")
    print("\n CNN training complete!")