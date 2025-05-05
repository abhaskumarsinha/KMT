import keras
import keras.ops as ops
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras import layers


def build_discriminator(input_shape=(256, 256, 1),
                        base_filters=16,
                        num_blocks=2,
                        use_batchnorm=True,
                        name="Discriminator"):
    """
    Builds a convolutional discriminator model with tunable complexity.

    Args:
        input_shape (tuple): Shape of the input image (H, W, C).
        base_filters (int): Number of filters in the first Conv2D layer.
        num_blocks (int): Number of downsampling blocks (conv + BN + LeakyReLU).
        use_batchnorm (bool): Whether to use BatchNormalization.
        name (str): Name of the Keras model.

    Returns:
        keras.Model: The discriminator model.
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs

    for i in range(num_blocks):
        filters = base_filters * (2 ** i)
        x = layers.Conv2D(filters, kernel_size=4, strides=2, padding='same')(x)
        if i > 0 and use_batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    outputs = layers.Activation('sigmoid')(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)
