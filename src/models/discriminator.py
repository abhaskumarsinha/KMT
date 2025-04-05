import keras
import keras.ops as ops
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def build_discriminator(input_shape=(256, 256, 1), name="Discriminator"):
    """
    Builds a convolutional discriminator model for GANs.

    Args:
        input_shape (tuple): Shape of the input image (H, W, C).
        name (str): Name of the Keras model.

    Returns:
        keras.Model: The compiled discriminator model.
    """
    inputs = keras.layers.Input(shape=input_shape)

    x = keras.layers.Conv2D(32, kernel_size=4, strides=2, padding='same')(inputs)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)

    x = keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)

    x = keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)

    x = keras.layers.Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1)(x)
    outputs = keras.layers.Activation('sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model
