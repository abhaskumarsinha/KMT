import keras
import keras.ops as ops
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils import *


class KeypointDetector(keras.layers.Layer):
    def __init__(self, num_keypoints=10, **kwargs):
        """
        Layer for detecting keypoints and local Jacobian transformations.

        Args:
            num_keypoints (int): Number of keypoints to detect.
        """
        super().__init__(**kwargs)
        self.num_keypoints = num_keypoints

        self.keypoint_branch = keras.Sequential([
            keras.layers.Conv2D(32, 3, strides=2, activation='relu', padding='same'),
            keras.layers.Conv2D(32, 3, strides=2, activation='relu', padding='same'),
            keras.layers.Conv2D(num_keypoints, 3, strides=1, activation=None, padding='same'),
        ])

        self.jacobian_branch = keras.Sequential([
            keras.layers.Conv2D(32, 3, strides=2, activation='relu', padding='same'),
            keras.layers.Conv2D(32, 3, strides=2, activation='relu', padding='same'),
            keras.layers.Conv2D(4 * num_keypoints, 3, strides=1, activation=None, padding='same'),
        ])

    def call(self, inputs):
        """
        Args:
            inputs (Tensor): Input image of shape (B, 256, 256, 1)

        Returns:
            keypoints (Tensor): (B, num_keypoints, 2) normalized (x, y) keypoint coordinates.
            jacobians (Tensor): (B, num_keypoints, 2, 2) local affine Jacobian matrices.
        """
        batch_size = keras.ops.shape(inputs)[0]

        key_feat = self.keypoint_branch(inputs)
        jac_feat = self.jacobian_branch(inputs)

        H_feat, W_feat = key_feat.shape[1], key_feat.shape[2]
        grid = generate_grid((batch_size, H_feat, W_feat))

        key_feat_flat = ops.reshape(key_feat, (batch_size, -1, self.num_keypoints))
        weights = ops.softmax(key_feat_flat, axis=1)
        weights = ops.reshape(weights, (batch_size, H_feat, W_feat, self.num_keypoints))

        grid_x = ops.expand_dims(grid[..., 0], axis=-1)
        grid_y = ops.expand_dims(grid[..., 1], axis=-1)
        kp_x = ops.sum(weights * grid_x, axis=[1, 2])
        kp_y = ops.sum(weights * grid_y, axis=[1, 2])
        keypoints = ops.stack([kp_x, kp_y], axis=-1)

        jac_feat = ops.reshape(jac_feat, (batch_size, H_feat, W_feat, self.num_keypoints, 4))
        jac_feat = ops.transpose(jac_feat, [0, 4, 1, 2, 3])
        weights_exp = ops.expand_dims(weights, axis=1)
        weighted_jac = jac_feat * weights_exp
        weighted_jac = ops.transpose(weighted_jac, [0, 2, 3, 4, 1])
        weighted_jac = ops.reshape(weighted_jac, (batch_size, H_feat, W_feat, 4 * self.num_keypoints))
        jac_sum = ops.sum(weighted_jac, axis=[1, 2])
        jacobians = ops.reshape(jac_sum, (batch_size, self.num_keypoints, 2, 2))

        return keypoints, jacobians
