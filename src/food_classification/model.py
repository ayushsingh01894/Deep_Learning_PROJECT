"""
Model definition — Module 4: Food Image Classification
===========================================================
A CNN built via Transfer Learning on top of MobileNetV2 (ImageNet-pretrained).
"""

from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = 224  # MobileNetV2's native input resolution


def build_food_cnn_model(
    num_classes: int, learning_rate: float = 1e-3, base_trainable: bool = False
) -> keras.Model:
    """
    Architecture:
        Input(224, 224, 3)
        -> MobileNetV2 base (ImageNet weights, frozen by default)
        -> GlobalAveragePooling2D
        -> Dense(128, ReLU) -> Dropout(0.3)
        -> Dense(num_classes, Softmax)

    Set base_trainable=True (with a much lower learning_rate, e.g. 1e-5) for
    the optional fine-tuning phase after initial training of the head.
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = base_trainable

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # MobileNetV2 expects inputs preprocessed to [-1, 1]
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=base_trainable)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="food_classification_mobilenetv2")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
