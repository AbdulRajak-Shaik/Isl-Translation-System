"""
Pydantic API Schemas for Request/Response payloads.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    model_loaded: bool
    device: str
    timestamp: str


class ModelStatusResponse(BaseModel):
    model_name: str
    parameter_count: int
    checkpoint_path: str
    vocab_size: int
    device: str
    confidence_threshold: float
    is_trained_checkpoint: bool


class TranslationRequest(BaseModel):
    english_text: str = Field(..., description="English text to translate into native Telugu")
    is_uncertain: bool = Field(False, description="Whether upstream sign prediction had high uncertainty")


class TranslationResponse(BaseModel):
    english_source: str
    telugu_text: str
    script_valid: bool
    is_uncertain: bool
    engine: str
    warning: Optional[str] = None


class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    lang: str = Field("en", description="Target language ('en' or 'te')")


class TTSResponse(BaseModel):
    success: bool
    audio_base64: Optional[str] = None
    format: str = "audio/mp3"
    lang: str
    error: Optional[str] = None


class JobCreateResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float = Field(0.0, ge=0.0, le=1.0)
    english_translation: Optional[str] = None
    telugu_translation: Optional[str] = None
    confidence: Optional[float] = None
    is_uncertain: Optional[bool] = None
    english_audio_base64: Optional[str] = None
    telugu_audio_base64: Optional[str] = None
    error: Optional[str] = None


class FeedbackSubmission(BaseModel):
    session_id: Optional[str] = None
    english_predicted: str
    telugu_predicted: str
    correct_english: Optional[str] = None
    rating: int = Field(..., ge=1, le=5)
    comments: Optional[str] = None


class LiveInferenceResult(BaseModel):
    english_text: str
    telugu_text: str
    confidence: float
    is_uncertain: bool
    provisional: bool
    mean_entropy: float
    audio_en: Optional[str] = None
    audio_te: Optional[str] = None
    timestamp: float


class TrainJobCreateResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    english_text: str
    message: str


class TrainJobStatusResponse(BaseModel):
    job_id: str
    status: str  # "pending", "extracting", "training", "completed", "failed"
    progress: float = Field(0.0, ge=0.0, le=1.0)
    stage: str = "Initialized"
    current_epoch: int = 0
    total_epochs: int = 0
    loss: Optional[float] = None
    bleu: Optional[float] = None
    english_ground_truth: Optional[str] = None
    telugu_ground_truth: Optional[str] = None
    test_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
