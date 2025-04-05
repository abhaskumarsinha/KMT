import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import keras

class VideoDataLoader:
    """
    A class to load and process videos into grayscale frame batches for model training.

    Attributes:
        folder_path (str): Directory containing video files.
        max_frames (int): Maximum number of frames to extract per video.
        num_vids (int): Number of videos to process for a batch.
        video_files (list): List of .mp4 video file paths in the folder.
    """
    
    def __init__(self, folder_path, max_frames, num_vids):
        """
        Initializes the loader with folder path and limits.

        Args:
            folder_path (str): Path to the folder containing .mp4 files.
            max_frames (int): Number of frames to extract per video.
            num_vids (int): Number of videos to load for each batch.
        """
        self.folder_path = folder_path
        self.max_frames = max_frames
        self.num_vids = num_vids
        self.video_files = glob.glob(f"{folder_path}/*.mp4")  # MP4 format

    def process_video(self, video_path):
        """
        Loads a single video and converts it into a fixed number of grayscale frames.

        Args:
            video_path (str): Path to the video file.

        Returns:
            np.ndarray: Array of shape (max_frames, 256, 256) representing grayscale frames.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []

        while len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Resize frame and convert to grayscale by averaging across color channels
            frame = cv2.resize(frame, (256, 256))
            gray_frame = np.mean(frame, axis=2).astype(np.uint8)
            frames.append(gray_frame)

        cap.release()

        # Pad with last frame if video has fewer frames than max_frames
        if len(frames) < self.max_frames:
            last_frame = frames[-1] if frames else np.zeros((256, 256), dtype=np.uint8)
            frames.extend([last_frame] * (self.max_frames - len(frames)))

        return np.array(frames)

    def get_batch(self):
        """
        Retrieves a batch of videos, each with max_frames grayscale frames.

        Returns:
            np.ndarray: Array of shape (num_vids, max_frames, 256, 256).
        """
        batch_videos = []
        for video_file in self.video_files[:self.num_vids]:
            video_data = self.process_video(video_file)
            batch_videos.append(video_data)

        return np.array(batch_videos)

    def get_stats(self):
        """
        Calculates statistics about the videos in the dataset.

        Returns:
            dict: Dictionary with total, average, min, and max frame counts across all videos.
        """
        frame_counts = []

        for video_file in self.video_files:
            cap = cv2.VideoCapture(video_file)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_counts.append(frame_count)
            cap.release()

        avg_frames = np.mean(frame_counts) if frame_counts else 0
        min_frames = np.min(frame_counts) if frame_counts else 0
        max_frames = np.max(frame_counts) if frame_counts else 0

        return {
            "total_videos": len(self.video_files),
            "average_frames": avg_frames,
            "min_frames": min_frames,
            "max_frames": max_frames
        }


def create_dataset(batch):
    """
    Prepares training data from a batch of grayscale video frames.

    Args:
        batch (np.ndarray): Input array of shape (num_videos, num_frames, height, width).

    Returns:
        tuple: ((X0, X1), Y)
            - X0: Repeated first frame of each video across time (shape: (num_videos*num_frames, height, width, 1))
            - X1: All frames from the video as input sequence
            - Y: Target output frames (same as X1 for reconstruction)
    """
    num_vids, num_frames, h, w = batch.shape

    # Extract the first frame of each video and repeat it across all frames
    first_frames = batch[:, 0:1, :, :]
    X0 = np.repeat(first_frames, num_frames, axis=1)  # shape: (num_vids, num_frames, h, w)

    # Input frames (could be motion or target-dependent)
    X1 = batch.copy()

    # Ground truth (target) frames
    Y = batch.copy()

    # Reshape for model input: (num_vids*num_frames, h, w, 1) and normalize to [0, 1]
    X0 = X0.reshape(-1, h, w, 1) / 255
    X1 = X1.reshape(-1, h, w, 1) / 255
    Y  = Y.reshape(-1, h, w, 1) / 255

    return (X0, X1), Y

