"""
Video Processor: Verification, Metadata Extraction, and Temporal Frame Sampling.
"""

import hashlib
import os
from pathlib import Path
from typing import Generator, List, Optional, Tuple
import cv2
import numpy as np

from ml.data.schemas import VideoMetadata


class VideoProcessor:
    """Robust video processing with corruption detection and temporal sampling."""

    def __init__(self, target_fps: int = 25, max_frames: Optional[int] = 300):
        self.target_fps = target_fps
        self.max_frames = max_frames

    @staticmethod
    def compute_sha256(filepath: str) -> str:
        """Compute SHA-256 checksum of video file for dataset versioning."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def inspect_video(self, video_path: str) -> VideoMetadata:
        """Inspect video file, check corruption, and extract structural metadata."""
        if not os.path.exists(video_path):
            return VideoMetadata(is_corrupt=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return VideoMetadata(is_corrupt=True)

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Test reading the first frame to confirm real decoding
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                return VideoMetadata(is_corrupt=True)

            duration = frame_count / fps if fps > 0 else 0.0
            checksum = self.compute_sha256(video_path)

            return VideoMetadata(
                duration_sec=round(duration, 2),
                fps=round(fps, 2),
                frame_count=frame_count,
                width=width,
                height=height,
                is_corrupt=False,
                sha256_checksum=checksum,
            )
        except Exception:
            return VideoMetadata(is_corrupt=True)
        finally:
            cap.release()

    def sample_frames(
        self, video_path: str, target_size: Optional[Tuple[int, int]] = None
    ) -> List[np.ndarray]:
        """
        Extract normalized RGB frames uniformly sampled to match target_fps.
        Returns list of (H, W, 3) RGB numpy arrays.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if src_fps <= 0 or np.isnan(src_fps):
            src_fps = 30.0

        stride = max(1, round(src_fps / self.target_fps))

        frames = []
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if frame_idx % stride == 0:
                    # Convert BGR (OpenCV) to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if target_size:
                        rgb_frame = cv2.resize(rgb_frame, target_size, interpolation=cv2.INTER_LINEAR)
                    frames.append(rgb_frame)

                    if self.max_frames and len(frames) >= self.max_frames:
                        break

                frame_idx += 1
        finally:
            cap.release()

        return frames
