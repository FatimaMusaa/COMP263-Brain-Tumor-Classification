from __future__ import annotations

import tensorflow as tf
from config import TRAIN_DIR, TEST_DIR, IMAGE_SIZE, BATCH_SIZE, SEED, VAL_SPLIT


def create_datasets():
    """
    Create TensorFlow datasets for training, validation, and testing.
    Uses training folder with validation split and separate testing folder.
    """
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="categorical",
    )

    class_names = train_ds.class_names

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)
    test_ds = test_ds.prefetch(buffer_size=autotune)

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