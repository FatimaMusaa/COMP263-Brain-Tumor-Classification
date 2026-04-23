from __future__ import annotations

import tensorflow as tf
from config import TRAIN_DIR, TEST_DIR, IMAGE_SIZE, BATCH_SIZE, SEED, VAL_SPLIT


def create_datasets():
    """
    Create TensorFlow datasets for training, validation, and testing.
    Uses training folder with validation split and separate testing folder.
    """
    # Load training dataset (subset of TRAIN_DIR)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
# Load validation dataset (remaining portion of TRAIN_DIR)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
# Load test dataset from separate folder (TEST_DIR)
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="categorical",
    )
# Get class names (e.g., glioma, meningioma, etc.) from training dataset
    class_names = train_ds.class_names
# Optimize dataset loading performance using prefetching (overlap data preprocessing and model execution)
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)
    test_ds = test_ds.prefetch(buffer_size=autotune)
# Return datasets and class labels for use in training and evaluation
    return train_ds, val_ds, test_ds, class_names


def get_augmentation_layer() -> tf.keras.Sequential:
    """Data augmentation for better generalization."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )