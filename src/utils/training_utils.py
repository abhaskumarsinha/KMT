import keras
import keras.ops as ops
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_identity_jacobians(batch, num_kp):
    """
    Generate a batch of identity Jacobian matrices for keypoints.

    Each Jacobian is a 2x2 identity matrix, indicating no local transformation.
    The output is used in motion transfer models where identity Jacobians are
    assigned to keypoints that do not contribute to deformation.

    Parameters:
        batch (int): Number of batches (B), typically corresponding to the batch size of input images.
        num_kp (int): Number of keypoints (n) per image.

    Returns:
        tf.Tensor: A tensor of shape (B, n, 2, 2) containing identity matrices for each keypoint in the batch.
    """
    identity_matrix = ops.convert_to_tensor([[1, 0], [0, 1]], dtype="float32")  # (2, 2)

    # Expand to (B, n, 2, 2)
    identity_jacobians = ops.expand_dims(identity_matrix, axis=0)  # (1, 2, 2)
    identity_jacobians = ops.repeat(identity_jacobians, batch * num_kp, axis=0)
    identity_jacobians = ops.reshape(identity_jacobians, (batch, num_kp, 2, 2))

    return identity_jacobians
