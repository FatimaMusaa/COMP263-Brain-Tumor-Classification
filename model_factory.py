from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from config import IMAGE_SIZE, NUM_CLASSES, LEARNING_RATE
from data_utils import get_augmentation_layer

# Register this custom loss so it can be saved/loaded with the model
@tf.keras.utils.register_keras_serializable()
class WeightedCategoricalCrossentropy(tf.keras.losses.Loss):
    """Categorical crossentropy with per-class weights for one-hot labels."""
# Initialize the loss with class weights (list of floats) and an optional name
    def __init__(self, class_weights: list[float], name: str = "weighted_cce"):
        super().__init__(name=name)
        # Convert class weights to Tensor for computation
        self.class_weights = tf.constant(class_weights, dtype=tf.float32)
# Main loss calculation: compute standard categorical crossentropy and multiply by class weights
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # Clip predictions to avoid log(0) errors (numerical stability)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        # Compute standard categorical crossentropy
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
         # Get the weight for each sample based on its true class
        weights = tf.reduce_sum(y_true * self.class_weights, axis=-1)
        # Multiply loss by class weights (important classes get higher penalty)
        return ce * weights
 # Required for saving/loading the model with this custom loss
    def get_config(self):
        return {
            "class_weights": self.class_weights.numpy().tolist(),
            "name": self.name,
        }

# Data structure to store model information in one place
@dataclass
class ModelBundle:
    name: str
    display_name: str
    family: str
    model: tf.keras.Model
    backbone: tf.keras.Model | None = None
    recommendation: str = ""

# Select loss function (normal or weighted) based on whether class weights are provided
def get_loss(class_weights: list[float] | None = None):
    if class_weights is None:
        return "categorical_crossentropy"
    return WeightedCategoricalCrossentropy(class_weights=class_weights)

# Compile the model with optimizer, loss, and evaluation metrics
def compile_model(
    model: tf.keras.Model,
    learning_rate: float,
    class_weights: list[float] | None = None,
) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=get_loss(class_weights),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model

# Build a simple CNN model (baseline for comparison)
def build_baseline_cnn(
    class_weights: list[float] | None = None,
    learning_rate: float = LEARNING_RATE,
) -> ModelBundle:
    # Data augmentation layer (random transformations)
    augmentation = get_augmentation_layer()
 # Sequential CNN model with convolutional layers, batch normalization, pooling, and dense layers
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)), # Normalize pixel values
            augmentation,# Apply augmentation
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(32, 3, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(256, 3, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(),
             # Classification head
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            # Output layer (multi-class classification)
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="baseline_cnn",
    )

    compile_model(model, learning_rate=learning_rate, class_weights=class_weights)
    return ModelBundle(
        name="baseline_cnn",
        display_name="Baseline CNN",
        family="cnn",
        model=model,
        recommendation="Simple reference model for comparison in the report.",
    )

# Generic function to build transfer learning models
def build_transfer_model(
    *,
    name: str,
    display_name: str,
    family: str,
    backbone_cls,# Pretrained model class (EfficientNet, DenseNet)
    preprocess_fn,# Preprocessing function for that model
    class_weights: list[float] | None = None,
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.35,
    dense_units: int = 256,
    recommendation: str = "",
) -> ModelBundle:
    augmentation = get_augmentation_layer()
    # Load pretrained backbone (ImageNet weights) and freeze it for initial training
    backbone = backbone_cls(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    )
     # Freeze backbone (transfer learning step 1)
    backbone.trainable = False
# Build model architecture: input -> augmentation -> preprocessing -> backbone -> custom head
    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = augmentation(inputs)
    x = preprocess_fn(x) # Model-specific preprocessing
    x = backbone(x, training=False) # Extract features
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
     # Dense classification layers
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
# Compile model with appropriate loss and metrics
    model = tf.keras.Model(inputs, outputs, name=name)
    compile_model(model, learning_rate=learning_rate, class_weights=class_weights)

    return ModelBundle(
        name=name,
        display_name=display_name,
        family=family,
        model=model,
        backbone=backbone,
        recommendation=recommendation,
    )

# EfficientNet model (recommended)
def build_efficientnet_transfer(
    class_weights: list[float] | None = None,
    learning_rate: float = 1e-3,
) -> ModelBundle:
    return build_transfer_model(
        name="efficientnet_b0_transfer",
        display_name="EfficientNetB0 Transfer",
        family="transfer_learning",
        backbone_cls=tf.keras.applications.EfficientNetB0,
        preprocess_fn=tf.keras.applications.efficientnet.preprocess_input,
        class_weights=class_weights,
        learning_rate=learning_rate,
        dropout_rate=0.4,
        dense_units=128,
        recommendation=(
            "Recommended default: strong accuracy on small MRI datasets with "
            "good efficiency for local demos."
        ),
    )





MODEL_BUILDERS = {
    "baseline_cnn": build_baseline_cnn,
    "efficientnet_b0_transfer": build_efficientnet_transfer
}


def list_supported_models() -> list[str]:
    return list(MODEL_BUILDERS.keys())


def create_model(
    model_name: str,
    class_weights: list[float] | None = None,
    learning_rate: float | None = None,
) -> ModelBundle:
    if model_name not in MODEL_BUILDERS:
        available = ", ".join(list_supported_models())
        raise ValueError(f"Unsupported model '{model_name}'. Available: {available}")

    lr = LEARNING_RATE if learning_rate is None else learning_rate
    return MODEL_BUILDERS[model_name](class_weights=class_weights, learning_rate=lr)
