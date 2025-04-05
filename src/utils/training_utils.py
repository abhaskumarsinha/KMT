import keras
import keras.ops as ops
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.discriminator import *
from src.models.generator import *
from src.models.keypoint_detector import *

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


def setup_keypoint_pipeline(
    keypoint_detector,
    generator,
    discriminator_model,
    image_size=(256, 256, 1),
    batch_size=16,
    training_epochs=250,
    num_keypoints=10,
    learning_rate=1e-4
):
    """
    Sets up a general training pipeline for keypoint-based image generation using a GAN architecture.

    Args:
        keypoint_detector (keras.Model): A model that detects keypoints from images.
        generator (keras.Model): A keypoint-based image generator (e.g., KeypointBasedTransform).
        discriminator_model (keras.Model): Discriminator network to distinguish real from fake images.
        image_size (tuple): Shape of the input images. Default is (256, 256, 1).
        batch_size (int): Training batch size. Default is 16.
        training_epochs (int): Number of epochs for training. Default is 250.
        num_keypoints (int): Number of keypoints to detect. Default is 10.
        learning_rate (float): Learning rate for all optimizers. Default is 1e-4.

    Returns:
        dict: {
            "gan": GAN model,
            "generator_model": Full generator training model,
            "keypoint_detector": Keypoint detector model,
            "discriminator": Discriminator model
        }
    """

    # ------------------------------
    #   Set up Generator pipeline
    # ------------------------------
    src_input = keras.Input(shape=image_size)
    drv_input = keras.Input(shape=image_size)

    src_kp = keypoint_detector(src_input)
    drv_kp = keypoint_detector(drv_input)

    gen_out = generator((src_input, src_kp[0], src_kp[1], drv_kp[0], drv_kp[1]))

    generator_model = keras.Model(inputs=[src_input, drv_input], outputs=gen_out)
    generator_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='mse',
        run_eagerly=False
    )

    # ------------------------------
    #   Discriminator setup
    # ------------------------------
    discriminator_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        run_eagerly=False
    )

    # ------------------------------
    #   GAN pipeline
    # ------------------------------
    gan_output = discriminator_model(generator((src_input, src_kp[0], src_kp[1], drv_kp[0], drv_kp[1])))
    gan_model = keras.Model(inputs=[src_input, drv_input], outputs=gan_output)
    gan_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        run_eagerly=False
    )

    # Debug summaries
    print("\n🧱 GAN Summary:")
    gan_model.summary()
    print("\n🧱 Generator Backbone Summary:")
    generator.model.summary()
    print("\n🧱 Generator Upscaler Summary:")
    generator.upscaler.summary()

    return {
        "gan": gan_model,
        "generator_model": generator_model,
        "keypoint_detector": keypoint_detector,
        "discriminator": discriminator_model
    }
