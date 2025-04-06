[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Unlicense License][license-shield]][license-url]

# KMT: Keras Motion Transfer

![Keras Motion Transfer](https://github.com/user-attachments/assets/add57cc8-647e-4957-97a0-63ea638e585b)

[`[Keras]`](https://keras.io/) . [`[First Order Animation]`](https://aliaksandrsiarohin.github.io/first-order-model-website/) . [`[Notebook]`](https://www.github.com/abhaskumarsinha/KMT/example.ipynb) . [`[Logo]`](https://logo.com/) . [`[Vox Dataset]`](https://github.com/AliaksandrSiarohin/video-preprocessing)


A Keras-based motion transfer model leveraging first-order Jacobian-based motion estimation to animate a static image using keypoints, inspired by Aliaksandr Siarohin's first-order motion model.

## 📌 About The Project
KMT: Keras Motion Transfer is a deep learning pipeline for animating a static image using the motion from a driving video. Inspired by recent advances in self-supervised keypoint-based motion transfer, this project implements a generative model in Keras that can transfer motion dynamics from video frames to a target image, generating a coherent, animated sequence.

KMT consists of two core components:

**Keypoint Detector**
- Detects keypoint positions and estimates local motion (via `2×2` Jacobians) in both the source (static image) and driving (video frame) domains.

**Generator**
- Takes a grayscale source image of shape `(256, 256, 1)` and the keypoint information from both the source and driving frames. It computes the transformation using first-order motion approximations and synthesizes the animated output frame-by-frame.

## 🔍 How It Works
- The keypoint detector extracts `(x, y)` coordinates and local Jacobians for each keypoint from both the source image and each frame of the driving video.

- Motion transfer is modeled based on a first-order approximation, as described in _Eq. `(4)`_ of the original motion transfer paper.

- The relative motion between keypoints is estimated using the Jacobians and keypoint offsets.

- A dense motion field is constructed, which is then fed into the generator along with the static source image.

- The generator synthesizes the animated frame corresponding to each video frame's motion.

- A GAN training setup refines both the keypoint detector and the generator for realistic motion and appearance.

## ⚙️ Technical Highlights
- Built entirely using the Keras deep learning framework.

- Backend agnostic — designed to work (in theory) with TensorFlow, JAX, and PyTorch backends.

- Grayscale only (B/W) generation to avoid misuse in real-world facial applications.

- Currently single-device only — no support for distributed training yet.

# Getting Started

## Installation
```
pip install -r requirements. txt
```
## Getting Started
Check [example.ipynb](https://www.github.com/abhaskumarsinha/KMT/example.ipynb) notebook to get started.

# 🚀 Features
- Supports video dataloading for training and inference

- Includes a dataset creator to build custom datasets

- Automatically switches between GPU and CPU based on availability

- Backend-agnostic design (theoretically compatible with JAX, PyTorch, and TensorFlow)

- Modular and clean codebase for easy debugging and rapid experimentation

# References
- Siarohin, Aliaksandr, et al. "*First order motion model for image animation.*" Advances in neural information processing systems 32 (2019).
- Chollet, François and others. Keras, 2015. Available at: https://keras.io

# Cite
```
@misc{KMT2025,
  author       = {Abhas Kumar Sinha},
  title        = {KMT: Keras Motion Transfer},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://www.github.com/abhaskumarsinha/KMT}},
  note         = {MIT License. Copyright (c) 2025 Abhas Kumar Sinha}
}
```





[contributors-shield]: https://img.shields.io/github/contributors/abhaskumarsinha/KMT?style=for-the-badge
[contributors-url]: https://github.com/abhaskumarsinha/KMT/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/abhaskumarsinha/KMT?style=for-the-badge
[forks-url]: https://github.com/abhaskumarsinha/KMT/network/members
[stars-shield]: https://img.shields.io/github/stars/abhaskumarsinha/KMT?style=for-the-badge
[stars-url]: https://github.com/abhaskumarsinha/KMT/stargazers
[license-shield]: https://img.shields.io/github/license/abhaskumarsinha/KMT?style=for-the-badge
[license-url]: https://github.com/abhaskumarsinha/KMT/blob/master/LICENSE.txt

