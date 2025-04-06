
"""
`src.models` – Core model components for keypoint-based image transformation

This module contains the core models used in the pipeline for keypoint-based image transformation,
including the generator, discriminator, and keypoint detector layers. These models can be used 
independently or together as part of a GAN or motion transfer pipeline.

Modules:
--------

- `generator.py`:
    Contains `KeypointBasedTransform`, a configurable Keras layer that warps an image using
    keypoints and local affine Jacobians. It supports optional custom motion models and upscalers.

- `discriminator.py`:
    Defines `build_discriminator`, a simple CNN-based discriminator suitable for use in GANs.
    It distinguishes between real and generated images.

- `keypoint_detector.py`:
    Defines `KeypointDetector`, a Keras layer for learning keypoint locations and their corresponding
    Jacobian matrices using separate convolutional branches. Supports custom sub-models.

Usage Example:
--------------

```python
from src.models.discriminator import build_discriminator
from src.models.generator import KeypointBasedTransform
from src.models.keypoint_detector import KeypointDetector

# Discriminator model
discriminator = build_discriminator()

# Generator model layer
transform_layer = KeypointBasedTransform(batch_size=16)

# Keypoint detector layer
kp_detector = KeypointDetector(num_keypoints=10)
