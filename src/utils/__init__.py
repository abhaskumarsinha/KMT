"""
Video Motion Transfer Utilities

This package contains core utility functions for video motion transfer tasks, including:

1. Video Data Loading & Processing
2. Spatial Transformation Operations
3. Training Pipelines & Model Setup

Modules:
    dataset_utils.py: Handles video loading, frame extraction, and dataset preparation
    model_utils.py: Provides geometric transformations and keypoint operations
    training_utils.py: Contains training loops and model configuration helpers

Key Functionality Groups:

## Data Handling
- `VideoDataLoader`: Batch video frame extraction with automatic padding
- `create_dataset`: Prepares (source, driving) frame pairs for training

## Geometric Operations
- `grid_transform`: Differentiable image warping using sampling grids
- `sparse_motion`: Computes per-keypoint motion fields
- `kp2gaussian`: Converts keypoints to spatial heatmaps

## Training Infrastructure
- `train_motion_model`: End-to-end GAN training loop
- `setup_keypoint_pipeline`: Configures full model architecture
- `generate_identity_jacobians`: Initializes keypoint transformations

Typical Usage Flow:
    1. Load videos → VideoDataLoader
    2. Prepare batches → create_dataset
    3. Configure models → setup_keypoint_pipeline
    4. Train → train_motion_model

Coordinate Systems:
    All spatial operations use normalized coordinates [-1, 1] where:
    - x: -1 = left, 1 = right
    - y: 1 = top, -1 = bottom (image convention)
"""


from src.utils.dataset_utils import *
from src.utils.model_utils import *
from src.utils.training_utils import *
