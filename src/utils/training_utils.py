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


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow import keras

def train_motion_model(X, Y,
                       generator,
                       gan_model,
                       discriminator,
                       batch_size=16,
                       epochs=10,
                       save_path='./working',
                       preview=True):
    """
    Trains the generator and GAN model using the provided datasets.

    Args:
        X (tuple): Tuple of (X0, X1) input arrays.
        Y (np.ndarray): Ground truth output images.
        generator (keras.Model): Generator model.
        gan_model (keras.Model): GAN model combining generator and discriminator.
        discriminator (keras.Model): Discriminator model.
        batch_size (int): Number of samples per training batch.
        epochs (int): Number of training epochs.
        save_path (str): Path to save model weights.
        preview (bool): Whether to show output images during training.
    """
    total_inst = (Y.shape[0] - 1) // batch_size * batch_size
    X0, X1 = X[0][:total_inst], X[1][:total_inst]
    Y = Y[:total_inst]

    generator.layers[-1].batch_size = batch_size

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch_number in tqdm(range(0, total_inst, batch_size), desc="Batches"):
            x_batch_0 = X0[batch_number: batch_number + batch_size]
            x_batch_1 = X1[batch_number: batch_number + batch_size]
            y_batch = Y[batch_number: batch_number + batch_size]

            # Train generator directly (supervised)
            generator.train_on_batch((x_batch_0, x_batch_1), y_batch)

            # Train discriminator to distinguish real vs fake
            loss_real = discriminator.train_on_batch(x_batch_1, keras.ops.zeros((batch_size, 1)))
            print('Discriminator Loss (real):', loss_real)

            # Freeze discriminator for generator adversarial training
            gan_model.trainable = False
            gan_model.layers[-1].trainable = True

            loss_gen = gan_model.train_on_batch((x_batch_0, x_batch_1), keras.ops.ones((batch_size, 1)))
            print('GAN Loss (G):', loss_gen)

            # Re-enable discriminator for adversarial step
            gan_model.trainable = True
            gan_model.layers[-1].trainable = False

            loss_fake = gan_model.train_on_batch((x_batch_0, x_batch_1), keras.ops.zeros((batch_size, 1)))
            print('GAN Loss (D):', loss_fake)

        # Visualization after each epoch
        if preview:
            pred = generator((x_batch_0, x_batch_1))[0, ..., 0]
            plt.imshow(pred, cmap='gray')
            plt.title(f"Epoch {epoch + 1}")
            plt.axis('off')
            plt.show()

        # Save model weights
        generator.save_weights(f'{save_path}/generator.weights.h5')
        gan_model.save_weights(f'{save_path}/GAN.weights.h5')

    print("Training complete.")

