"""
Application Configuration Settings.
Loads environment variables with validated defaults.
"""

import os
from typing import List
from pydantic import BaseModel, Field


class Settings(BaseModel):
    PROJECT_NAME: str = "Indian Sign Language Translation System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    CORS_ORIGINS: List[str] = (
        [origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()]
        if os.getenv("CORS_ORIGINS")
        else ["*"]
    )

    # Model & Inference
    CHECKPOINT_PATH: str = os.getenv("MODEL_CHECKPOINT_PATH", "ml/checkpoints/isl_transformer_best.pt")
    VOCAB_PATH: str = os.getenv("VOCAB_PATH", "data/manifests/vocab.json")
    DEVICE: str = os.getenv("MODEL_DEVICE", "cpu")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))

    # Uploads & Scratch
    UPLOAD_TEMP_DIR: str = os.getenv("UPLOAD_TEMP_DIR", "data/scratch/uploads")
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
    MAX_VIDEO_DURATION_SECONDS: int = int(os.getenv("MAX_VIDEO_DURATION_SECONDS", "180"))
    AUTO_DELETE_PROCESSED_VIDEOS: bool = os.getenv("AUTO_DELETE_PROCESSED_VIDEOS", "true").lower() == "true"


settings = Settings()
os.makedirs(settings.UPLOAD_TEMP_DIR, exist_ok=True)
