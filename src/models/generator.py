import keras
import keras.ops as ops
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils.import *


class KeypointBasedTransform(keras.layers.Layer):
    def __init__(self, batch_size, target_size=(60, 60), model=None, upscaler=None, **kwargs):
        """
        Args:
            batch_size (int): Default training batch size.
            target_size (tuple): Target resolution for feature processing (H, W).
            model (tf.keras.Model or tf.keras.Layer, optional): Custom motion prediction model.
            upscaler (tf.keras.Model or tf.keras.Layer, optional): Custom upscaling model.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.target_size = target_size
        self.height, self.width = target_size

        self.resize_layer = keras.layers.Resizing(self.height, self.width, interpolation="bilinear")

        # Default motion field model
        self.model = model if model is not None else keras.Sequential([
            keras.layers.Input(shape=(self.height, self.width, 22)),
            keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            keras.layers.Conv2D(64, 3, padding="same", activation="tanh"),
            keras.layers.Conv2D(64, 3, padding="same", activation="tanh"),
            keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            keras.layers.Conv2D(11, 1, activation="linear")
        ])

        self.mask = keras.layers.Conv2D(11, 3, padding="same", activation="relu")
        self.occlusion = keras.layers.Conv2D(1, 3, padding="same", activation="relu")

        # Default upscaler
        self.upscaler = upscaler if upscaler is not None else keras.Sequential([
            keras.layers.Input(shape=(self.height, self.width, 1)),
            keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            keras.layers.UpSampling2D(size=2),
            keras.layers.Conv2D(128, 3, padding="same", activation="tanh"),
            keras.layers.UpSampling2D(size=2),
            keras.layers.Conv2D(256, 3, padding="same", activation="tanh"),
            keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            keras.layers.Conv2D(1, 3, padding="same", activation="linear"),
            keras.layers.Resizing(256, 256, interpolation="bilinear")
        ])

    def call(self, inputs, inference=False):
        source_image, source_kp, source_jac, driving_kp, driving_jac = inputs

        resized_image = self.resize_layer(source_image)

        sparse_motion_tensor = sparse_motion(
            keras.ops.shape(resized_image[..., 0]),
            source_kp,
            driving_kp,
            source_jac,
            driving_jac
        )

        batch_size = 1 if inference else self.batch_size

        deformed_images = apply_sparse_motion(source_image, sparse_motion_tensor, batch_size=batch_size)
        zeros = ops.zeros((batch_size, self.height, self.width, 1), dtype=deformed_images.dtype)
        deformed_images = ops.concatenate([deformed_images, zeros], axis=-1)

        source_density = kp2gaussian(source_kp, 0.01, (2, self.height, self.width), batch_size=batch_size)
        driving_density = kp2gaussian(driving_kp, 0.01, (2, self.height, self.width), batch_size=batch_size)

        motion_field = source_density - driving_density
        motion_field = keras.ops.concatenate([
            keras.ops.zeros((batch_size, 1, self.height, self.width)), 
            motion_field
        ], axis=1)
        motion_field = keras.ops.transpose(motion_field, (0, 2, 3, 1))

        model_input = keras.ops.concatenate([motion_field, deformed_images], axis=-1)
        density_output = self.model(model_input)

        mask = self.mask(density_output)

        zero_tensor = ops.zeros((batch_size, 1, self.height, self.width, 2), dtype=sparse_motion_tensor.dtype)
        sparse_motion_tensor = ops.concatenate([sparse_motion_tensor, zero_tensor], axis=1)

        grid_flow_map = ops.einsum("bnhwc,bhwn->bhwc", sparse_motion_tensor, mask)
        occlusion = self.occlusion(density_output)

        warped = grid_transform(resized_image, grid_flow_map, batch=batch_size)
        warped *= occlusion

        if inference:
            return (
                self.upscaler(warped),
                grid_flow_map,
                keras.ops.mean(motion_field, axis=-1),
                occlusion,
                warped
            )
        else:
            return self.upscaler(warped)
