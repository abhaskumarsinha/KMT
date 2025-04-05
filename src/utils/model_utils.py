import keras
import keras.ops as ops
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def grid_transform(inputs, grid, order=1, fill_mode="constant", fill_value=0, batch = None):
    """
    Applies a spatial transformation to the input tensor using a sampling grid.

    This function performs image warping by sampling the `inputs` tensor at 
    specified `grid` coordinates using interpolation. The grid values are expected 
    to be normalized to the range [-1, 1], where -1 and 1 correspond to the 
    top-left and bottom-right corners respectively.

    Parameters:
        inputs (tf.Tensor): Input tensor of shape (B, H, W, C), representing a batch 
                            of images.
        grid (tf.Tensor): Sampling grid of shape (B, H_out, W_out, 2), containing 
                          normalized coordinates (x, y) to sample from the input.
        order (int): Interpolation order. 1 for bilinear, 0 for nearest neighbor.
        fill_mode (str): Points outside the boundaries are filled according to this 
                         mode. Supported: "constant", "nearest", "reflect", etc.
        fill_value (float): Value to use for points sampled outside the boundaries 
                            when `fill_mode="constant"`.
        batch (int, optional): Manually specify batch size if it cannot be inferred 
                               from inputs (e.g., in a tracing context).

    Returns:
        tf.Tensor: Warped output tensor of shape (B, H_out, W_out, C), where each 
                   image has been resampled using the provided grid.
    """

    # Assume inputs has static shape (B, H, W, C)

    if batch is not None:
        B = batch
    else:
        B = inputs.shape[0]
    H = inputs.shape[1]
    W = inputs.shape[2]
    C = inputs.shape[3]

    # Dynamic output grid dims
    grid_shape = ops.shape(grid)
    H_out = grid_shape[1]
    W_out = grid_shape[2]

    # Convert normalized grid coordinates to pixel coordinates.
    # grid[...,0] is x, grid[...,1] is y.
    x = (grid[..., 0] + 1.0) * (ops.cast(W, "float32") - 1) / 2.0
    y = (grid[..., 1] + 1.0) * (ops.cast(H, "float32") - 1) / 2.0

    outputs_list = []

    for b in range(B):
        channels_out = []
        # Swap coordinate order: (y, x) for image indexing.
        coords = ops.stack([y[b], x[b]], axis=-1)  # shape: (H_out, W_out, 2)
        coords_flat = ops.reshape(coords, (-1, 2))   # shape: (H_out*W_out, 2)
        # Transpose to shape (2, N)
        coords_flat = ops.transpose(coords_flat, [1, 0])

        for c in range(C):
            channel_img = inputs[b, :, :, c]  # shape: (H, W)
            if fill_mode == "constant":
                # Pad image with one pixel on each side.
                padded_img = ops.pad(channel_img, [[1, 1], [1, 1]], constant_values=fill_value)
                padded_H = H + 2
                padded_W = W + 2
                # Adjust coordinates by +1 to index into padded image.
                coords_adj = coords_flat + 1.0
                # For bilinear interpolation, we need floor and ceil of each coordinate.
                y_coords = coords_adj[0]
                x_coords = coords_adj[1]
                y0 = ops.floor(y_coords)
                y1 = ops.ceil(y_coords)
                x0 = ops.floor(x_coords)
                x1 = ops.ceil(x_coords)
                # valid if both floor and ceil are within [0, padded_dim - 1].
                valid = (y0 >= 0) & (y1 < padded_H) & (x0 >= 0) & (x1 < padded_W)

                # For sampling, clip coordinates into valid range.
                y_clip = ops.clip(y_coords, 0, padded_H - 1)
                x_clip = ops.clip(x_coords, 0, padded_W - 1)
                coords_clipped = ops.stack([y_clip, x_clip], axis=0)

                # Use map_coordinates with a safe fill_mode.
                sampled = keras.ops.image.map_coordinates(
                    padded_img,
                    coords_clipped,
                    order=order,
                    fill_mode="nearest",  # We'll override invalid positions below.
                    fill_value=fill_value
                )
                # Replace values at invalid positions with fill_value.
                valid = ops.cast(valid, "float32")  # shape: (N,)
                sampled = valid * sampled + (1 - valid) * fill_value
            else:
                # For nonconstant fill modes, we use the coordinates as is.
                sampled = keras.ops.image.map_coordinates(
                    channel_img,
                    coords_flat,
                    order=order,
                    fill_mode=fill_mode,
                    fill_value=fill_value
                )
            sampled_reshaped = ops.reshape(sampled, (H_out, W_out))
            channels_out.append(sampled_reshaped)
        batch_out = ops.stack(channels_out, axis=-1)  # (H_out, W_out, C)
        outputs_list.append(batch_out)

    outputs = ops.stack(outputs_list, axis=0)  # (B, H_out, W_out, C)
    return outputs

def generate_grid(shape):

    """
    Generates a normalized 2D sampling grid for spatial transformations.

    The grid contains coordinates in the range [-1, 1], where:
      - x varies from -1 (left) to 1 (right)
      - y varies from 1 (top) to -1 (bottom), following image coordinates

    This grid is used for resampling operations like spatial warping or 
    optical flow-based motion transfer.

    Parameters:
        shape (tuple): A 3-tuple (batch_size, height, width) specifying the 
                       desired grid shape.

    Returns:
        tf.Tensor: A grid tensor of shape (batch_size, height, width, 2), where 
                   each position contains (x, y) coordinates normalized to [-1, 1].
    """
    batch_size, h, w = shape

    # Generate normalized coordinates
    x = ops.linspace(-1.0, 1.0, w)  # Linearly spaced x-coordinates
    y = ops.linspace(1.0, -1.0, h)  # Linearly spaced y-coordinates (flip y-axis)

    # Create meshgrid
    X, Y = ops.meshgrid(x, y, indexing='xy')  # Shape (h, w)

    # Stack X and Y into (h, w, 2)
    grid = ops.stack([X, Y], axis=-1)

    # Expand for batch dimension (batch, h, w, 2)
    grid = ops.expand_dims(grid, axis=0)
    grid = ops.repeat(grid, batch_size, axis=0)

    return grid  # Shape: (batch_size, h, w, 2)



def grid_to_gaussian(grid, variance=0.01):

    """
    Converts a coordinate grid into a 2D isotropic Gaussian heatmap.

    Each point in the grid is interpreted as a coordinate offset from the center,
    and the function computes the corresponding Gaussian activation based on 
    the squared distance from the origin.

    This is commonly used to convert keypoint offsets or sampling grids into 
    soft spatial attention maps.

    Parameters:
        grid (tf.Tensor): Tensor of shape (batch, h, w, 2), where each (x, y) pair 
                          represents an offset from the center.
        variance (float): Scalar controlling the spread of the Gaussian (default 0.01). 
                          Smaller values produce sharper peaks.

    Returns:
        tf.Tensor: A tensor of shape (batch, h, w, 1) representing the Gaussian 
                   probability distribution over the grid.
    """

    squared_distance = ops.sum(ops.square(grid), axis=-1, keepdims=True)  # x^2 + y^2
    gaussian = ops.exp(-0.5 * squared_distance / variance)  # Apply Gaussian formula
    return gaussian  # Shape: (batch, h, w, 1)


def kp2gaussian(keypoints, variance, image_shape, batch_size=None):
    """
    Converts normalized keypoint coordinates into 2D Gaussian heatmaps.

    Parameters:
        keypoints (tf.Tensor): Tensor of shape (B, n, 2), where each keypoint 
                               is (x, y) in normalized coordinates.
        variance (float): Scalar controlling the spread of the Gaussian.
        image_shape (tuple): Tuple (B, h, w) specifying the output heatmap dimensions.
        batch_size (int, optional): Optional override for batch size, used when 
                                    tracing or shape inference is needed.

    Returns:
        tf.Tensor: A tensor of shape (B, n, h, w) containing a Gaussian heatmap 
                   for each keypoint.
    """
    if batch_size is None:
        B, h, w = image_shape
    else:
        B = batch_size
        _, h, w = image_shape

    grid = generate_grid((B, h, w))                           # (B, h, w, 2)
    grid_exp = ops.expand_dims(grid, axis=1)                 # (B, 1, h, w, 2)
    kp_exp = ops.expand_dims(ops.expand_dims(keypoints, axis=2), axis=2)  # (B, n, 1, 1, 2)

    delta = grid_exp - kp_exp                                # (B, n, h, w, 2)
    squared_distance = ops.sum(ops.square(delta), axis=-1, keepdims=True)  # (B, n, h, w, 1)
    gaussian = ops.exp(-0.5 * squared_distance / variance)   # (B, n, h, w, 1)

    return ops.squeeze(gaussian, axis=-1)                    # (B, n, h, w)


def get_keypoint_coordinates(image, grid):
    """
    Computes the keypoint coordinates from a weighted image and spatial grid.

    This function estimates keypoint positions by computing the spatial 
    expectation over the grid, weighted by pixel values in the input image.

    Parameters:
        image (tf.Tensor): Tensor of shape (B, H, W, C), typically a heatmap 
                           or attention map per keypoint.
        grid (tf.Tensor): Tensor of shape (B, H, W, 2), containing normalized 
                          (x, y) coordinates in the range [-1, 1].

    Returns:
        tf.Tensor: Tensor of shape (B, 2, C) containing keypoint coordinates 
                   (x, y) for each channel (usually one channel per keypoint).
    """
    grid_x = ops.expand_dims(grid[..., 0], axis=-1)  # (B, H, W, 1)
    grid_y = ops.expand_dims(grid[..., 1], axis=-1)  # (B, H, W, 1)

    latent_x = image * grid_x  # (B, H, W, C)
    latent_y = image * grid_y  # (B, H, W, C)

    kp_x = ops.sum(latent_x, axis=[1, 2])  # (B, C)
    kp_y = ops.sum(latent_y, axis=[1, 2])  # (B, C)

    return ops.stack([kp_x, kp_y], axis=1)  # (B, 2, C)

def sparse_motion(image_shape,
                  sparse_image_keypoints,
                  driving_image_keypoints,
                  sparse_image_jacobians,
                  driving_image_jacobians):
    """
    Computes a per-keypoint motion field by transforming pixel-wise coordinates 
    from the driving frame to the source (sparse) frame using Jacobian-based warping.

    Parameters:
        image_shape (tuple): Tuple (B, h, w) representing batch size and output resolution.
        sparse_image_keypoints (tf.Tensor): Tensor of shape (B, n, 2), source keypoints.
        driving_image_keypoints (tf.Tensor): Tensor of shape (B, n, 2), driving keypoints.
        sparse_image_jacobians (tf.Tensor): Tensor of shape (B, n, 2, 2), Jacobians at source keypoints.
        driving_image_jacobians (tf.Tensor): Tensor of shape (B, n, 2, 2), Jacobians at driving keypoints.

    Returns:
        tf.Tensor: Transformed coordinate grid of shape (B, n, h, w, 2).
    """
    B, h, w = image_shape
    n = ops.shape(sparse_image_keypoints)[1]

    grid = ops.expand_dims(generate_grid(image_shape), axis=1)  # (B, 1, h, w, 2)
    grid = ops.repeat(grid, n, axis=1)                          # (B, n, h, w, 2)

    driving_kp = ops.expand_dims(ops.expand_dims(driving_image_keypoints, axis=2), axis=3)  # (B, n, 1, 1, 2)
    diff = grid - driving_kp  # (B, n, h, w, 2)

    inv_driving = ops.linalg.inv(driving_image_jacobians + ops.eye(2, 2) * 1e-6)  # (B, n, 2, 2)
    T = ops.matmul(sparse_image_jacobians, inv_driving)  # (B, n, 2, 2)

    diff_transformed = ops.einsum('bnij,bnxyj->bnxyi', T, diff)  # (B, n, h, w, 2)

    sparse_kp = ops.expand_dims(ops.expand_dims(sparse_image_keypoints, axis=2), axis=3)
    return diff_transformed + sparse_kp  # (B, n, h, w, 2)

def apply_sparse_motion(image, motion_field, order=1, fill_mode="constant", fill_value=0, batch_size=None):
    """
    Applies a sparse motion field to an image using interpolation.

    Parameters:
        image (Tensor): Input tensor of shape (B, h, w, 1) or (B, h, w).
        motion_field (Tensor): Motion field of shape (B, n, h, w, 2), one per keypoint.
        order (int): Interpolation order.
        fill_mode (str): Fill mode for sampling.
        fill_value (float): Fill value for constant mode.
        batch_size (int, optional): Batch size override.

    Returns:
        Tensor: Output of shape (B, h, w, n), one warped image per keypoint.
    """
    if len(image.shape) == 3:
        image = ops.expand_dims(image, axis=-1)

    B = batch_size if batch_size is not None else image.shape[0]
    n = motion_field.shape[1]

    deformed = []
    for i in range(n):
        grid_i = motion_field[:, i]
        warped = grid_transform(image, grid_i, order=order, fill_mode=fill_mode, fill_value=fill_value, batch=B)
        deformed.append(warped)

    return ops.concatenate(deformed, axis=-1)

