import keras
import keras.ops as ops
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

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
                       preview=True,
                       preview_interval=50):  # new argument added
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
        preview_interval (int): Interval (in epochs) at which preview and saving occur.
    """
    total_inst = (Y.shape[0] - 1) // batch_size * batch_size
    X0, X1 = X[0][:total_inst], X[1][:total_inst]
    Y = Y[:total_inst]

    generator.layers[-1].batch_size = batch_size

    for epoch in tqdm(range(epochs), desc="Epochs"):
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch_number in range(0, total_inst, batch_size):
            x_batch_0 = X0[batch_number: batch_number + batch_size]
            x_batch_1 = X1[batch_number: batch_number + batch_size]
            y_batch = Y[batch_number: batch_number + batch_size]

            # Train generator directly (supervised)
            generator.train_on_batch((x_batch_0, x_batch_1), y_batch)

            # Train discriminator to distinguish real vs fake
            loss_real = discriminator.train_on_batch(x_batch_1, keras.ops.zeros((batch_size, 1)))
            

            # Freeze discriminator for generator adversarial training
            gan_model.trainable = False
            gan_model.layers[-1].trainable = True

            loss_gen = gan_model.train_on_batch((x_batch_0, x_batch_1), keras.ops.ones((batch_size, 1)))
            
            # Re-enable discriminator for adversarial step
            gan_model.trainable = True
            gan_model.layers[-1].trainable = False

            loss_fake = gan_model.train_on_batch((x_batch_0, x_batch_1), keras.ops.zeros((batch_size, 1)))
            

        # Execute preview and saving only every `preview_interval` epochs
        if (epoch) % preview_interval == 0:
            print('Discriminator Loss (real):', loss_real)
            print('GAN Loss (G):', loss_gen)
            print('GAN Loss (D):', loss_fake)

            
            if preview:
                # Create directory for images if it does not exist
                
                image_dir = os.path.join(save_path, "images")
                os.makedirs(image_dir, exist_ok=True)
                
                pred = generator((x_batch_0, x_batch_1))[0, ..., 0]
                plt.imshow(pred, cmap='gray')
                plt.title(f"Epoch {epoch + 1}")
                plt.axis('off')
                plt.show()
                plt.savefig(os.path.join(image_dir, f"epoch_{epoch + 1}.png"))
                plt.close()


            generator.save_weights(f'{save_path}/generator.weights.h5')
            gan_model.save_weights(f'{save_path}/GAN.weights.h5')

    print("Training complete.")

def setup_keypoint_pipeline(
    keypoint_detector,
    generator,
    discriminator_model,
    image_size=(256, 256, 1),
    batch_size=16,
    warmup_samples=500,
    warmup_epochs=10,
    training_epochs=250,
    num_keypoints=10,
    learning_rate=1e-4,
):
    """
    Sets up a general training pipeline for keypoint-based image generation using GAN.

    Returns:
        - GAN model
        - Generator model
        - Keypoint detector
        - Discriminator
        - Aligner (warmup model)
    """


    # ------------------------------
    #   Warmup keypoint detector (align jacobians)
    # ------------------------------
    kp_input = keras.Input(shape=image_size)
    kp_output = keypoint_detector(kp_input)
    kp_aligner = keras.Model(inputs=kp_input, outputs=kp_output[1])
    kp_aligner.compile(optimizer='adam', loss='mse')

    # Dummy warmup training
    kp_aligner.fit(
        keras.random.normal((warmup_samples, *image_size)),
        generate_identity_jacobians(warmup_samples, num_keypoints),
        batch_size=50,
        epochs=warmup_epochs
    )

    # ------------------------------
    #   Set up GAN pipeline
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

    discriminator_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        run_eagerly=False
    )

    # GAN pipeline with frozen discriminator
    disc_out = discriminator_model(generator((src_input, src_kp[0], src_kp[1], drv_kp[0], drv_kp[1])))
    gan_model = keras.Model(inputs=[src_input, drv_input], outputs=disc_out)
    gan_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        run_eagerly=False
    )

    # Debugging summaries
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
        "discriminator": discriminator_model,
        "aligner": kp_aligner
    }
