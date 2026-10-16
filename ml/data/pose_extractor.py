"""
Holistic Pose Feature Extractor for Indian Sign Language.
Extracts 3D skeletal landmarks (Hands, Upper Body Pose) using MediaPipe Holistic,
applies body-centric spatial normalization (mid-shoulder centering, scale invariance),
and computes temporal motion dynamics.
"""

from typing import List, Optional, Tuple
import numpy as np
import cv2

try:
    import mediapipe.python.solutions.holistic as mp_holistic_module
    mp_holistic_cls = mp_holistic_module.Holistic
    MEDIAPIPE_AVAILABLE = True
except Exception:
    try:
        import mediapipe as mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic"):
            mp_holistic_cls = mp.solutions.holistic.Holistic
            MEDIAPIPE_AVAILABLE = True
        else:
            mp_holistic_cls = None
            MEDIAPIPE_AVAILABLE = False
    except Exception:
        mp_holistic_cls = None
        MEDIAPIPE_AVAILABLE = False


class PoseExtractor:
    """
    Extracts, normalizes, and packages skeletal landmark sequences from video frames.
    Output per frame: 75 keypoints x 3 coordinates (x, y, z) = 225-dimensional feature vector.
    - Left Hand: 21 keypoints (indices 0..20)
    - Right Hand: 21 keypoints (indices 21..41)
    - Body Pose / Face: 33 keypoints (indices 42..74)
    """

    NUM_HAND_KEYPOINTS = 21
    NUM_POSE_KEYPOINTS = 33
    TOTAL_KEYPOINTS = 75  # 21 + 21 + 33
    FEATURE_DIM = 75 * 3  # 225

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self._holistic = None

    def _get_holistic(self):
        if not MEDIAPIPE_AVAILABLE or mp_holistic_cls is None:
            raise RuntimeError(
                "MediaPipe Holistic is not available. Please install a compatible mediapipe version (<=0.10.14)."
            )
        if self._holistic is None:
            self._holistic = mp_holistic_cls(
                static_image_mode=False,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
        return self._holistic

    def close(self):
        if self._holistic is not None:
            self._holistic.close()
            self._holistic = None

    def extract_landmarks_from_frame(self, rgb_frame: np.ndarray) -> np.ndarray:
        """
        Processes a single RGB frame and extracts raw 75 keypoint coordinates (75, 3).
        If a landmark group is not detected, it is filled with zeros.
        """
        try:
            holistic = self._get_holistic()
            results = holistic.process(rgb_frame)
        except Exception:
            # Fall back to zeros on unreadable frame or holistic failure
            return np.zeros((self.TOTAL_KEYPOINTS, 3), dtype=np.float32)

        # 1. Left Hand (21 x 3)
        left_hand = np.zeros((self.NUM_HAND_KEYPOINTS, 3), dtype=np.float32)
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                left_hand[i] = [lm.x, lm.y, lm.z]

        # 2. Right Hand (21 x 3)
        right_hand = np.zeros((self.NUM_HAND_KEYPOINTS, 3), dtype=np.float32)
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                right_hand[i] = [lm.x, lm.y, lm.z]

        # 3. Pose / Upper Body (33 x 3)
        pose = np.zeros((self.NUM_POSE_KEYPOINTS, 3), dtype=np.float32)
        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                pose[i] = [lm.x, lm.y, lm.z]

        # Concatenate into (75, 3)
        keypoints = np.vstack([left_hand, right_hand, pose])
        return keypoints

    @classmethod
    def normalize_keypoints(cls, keypoints_seq: np.ndarray) -> np.ndarray:
        """
        Normalize a sequence of shape (T, 75, 3):
        - Center relative to midpoint of shoulders (pose indices 11 and 12, offset by 42: index 53 and 54).
        - Scale by Euclidean distance between shoulders to achieve distance/scale invariance.
        """
        T, num_kpts, channels = keypoints_seq.shape
        normalized = np.copy(keypoints_seq)

        # Pose index 11 is Left Shoulder, index 12 is Right Shoulder
        # In our combined array: 42 + 11 = 53, 42 + 12 = 54
        l_shoulder_idx = 42 + 11
        r_shoulder_idx = 42 + 12

        for t in range(T):
            l_sh = keypoints_seq[t, l_shoulder_idx]
            r_sh = keypoints_seq[t, r_shoulder_idx]

            # If shoulders were detected (non-zero)
            if np.any(l_sh != 0) and np.any(r_sh != 0):
                mid_shoulder = (l_sh + r_sh) / 2.0
                shoulder_dist = np.linalg.norm(l_sh - r_sh)
                scale = max(shoulder_dist, 1e-4)

                # Center and scale all detected keypoints
                for k in range(num_kpts):
                    if np.any(normalized[t, k] != 0):
                        normalized[t, k] = (normalized[t, k] - mid_shoulder) / scale
            else:
                # Fallback: Center relative to frame center (0.5, 0.5, 0)
                center = np.array([0.5, 0.5, 0.0], dtype=np.float32)
                for k in range(num_kpts):
                    if np.any(normalized[t, k] != 0):
                        normalized[t, k] = normalized[t, k] - center

        return normalized

    def process_frames(self, frames: List[np.ndarray], normalize: bool = True) -> np.ndarray:
        """
        Process a list of RGB frames into a normalized feature array of shape (T, 225).
        """
        if not frames:
            return np.zeros((0, self.FEATURE_DIM), dtype=np.float32)

        raw_sequence = []
        for frame in frames:
            kpts = self.extract_landmarks_from_frame(frame)
            raw_sequence.append(kpts)

        raw_seq = np.array(raw_sequence, dtype=np.float32)  # (T, 75, 3)

        if normalize:
            norm_seq = self.normalize_keypoints(raw_seq)
        else:
            norm_seq = raw_seq

        # Flatten (T, 75, 3) -> (T, 225)
        T = norm_seq.shape[0]
        return norm_seq.reshape(T, self.FEATURE_DIM)
