"""
Unified Data Schemas for Indian Sign Language (ISL) Datasets and Manifests.
Provides strict validation, serialization, and typing across continuous and isolated tasks.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ISLTaskType(str, Enum):
    CONTINUOUS_TRANSLATION = "continuous_translation"
    GLOSS_RECOGNITION = "gloss_recognition"
    ISOLATED_SIGN = "isolated_sign"


class SplitType(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class VideoMetadata(BaseModel):
    duration_sec: float = Field(0.0, description="Duration of video in seconds")
    fps: float = Field(0.0, description="Frames per second")
    frame_count: int = Field(0, description="Total frame count")
    width: int = Field(0, description="Video width in pixels")
    height: int = Field(0, description="Video height in pixels")
    is_corrupt: bool = Field(False, description="True if video could not be decoded")
    sha256_checksum: Optional[str] = Field(None, description="SHA-256 hash of raw video")


class LandmarkMetadata(BaseModel):
    total_frames: int = Field(0, description="Number of sampled frames with extracted landmarks")
    num_keypoints: int = Field(75, description="Number of skeletal landmarks per frame (e.g., 75)")
    channels: int = Field(3, description="Coordinate channels (x, y, z)")
    mean_confidence: float = Field(0.0, description="Mean landmark visibility/confidence")
    is_valid: bool = Field(True, description="Whether extracted sequence meets quality criteria")


class SignSample(BaseModel):
    sample_id: str = Field(..., description="Unique sample identifier (e.g., islcsltr_0001)")
    dataset_name: str = Field(..., description="Source dataset (e.g., ISLTranslate, ISL-CSLTR)")
    video_path: Optional[str] = Field(None, description="Relative path to raw video file")
    pose_path: Optional[str] = Field(None, description="Relative path to extracted numpy/h5 pose array")
    signer_id: Optional[str] = Field("unknown", description="Unique identifier of signer for leakage prevention")
    task_type: ISLTaskType = Field(ISLTaskType.CONTINUOUS_TRANSLATION, description="Modality task type")
    
    # Text Annotations
    english_text: str = Field(..., description="Ground-truth natural English translation")
    telugu_text: Optional[str] = Field(None, description="Ground-truth native Telugu translation if available")
    gloss_tokens: List[str] = Field(default_factory=list, description="Sign gloss sequence if annotated")
    
    # Partitioning & Metadata
    split: SplitType = Field(SplitType.TRAIN, description="Split partition (train/val/test)")
    video_meta: Optional[VideoMetadata] = None
    landmark_meta: Optional[LandmarkMetadata] = None


class ManifestSummary(BaseModel):
    dataset_name: str
    total_samples: int
    split_distribution: Dict[str, int]
    total_duration_hours: float
    signer_count: int
    english_vocab_size: int
    gloss_vocab_size: int
    leakage_check_passed: bool
    created_at: str
