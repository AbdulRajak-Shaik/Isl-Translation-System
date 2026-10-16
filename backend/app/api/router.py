"""
FastAPI REST API Routes.
"""

from datetime import datetime
import os
import shutil
import uuid
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from backend.app.core.config import settings
from backend.app.schemas.api_models import (
    FeedbackSubmission,
    HealthResponse,
    JobCreateResponse,
    JobStatusResponse,
    ModelStatusResponse,
    TrainJobCreateResponse,
    TrainJobStatusResponse,
    TranslationRequest,
    TranslationResponse,
    TTSRequest,
    TTSResponse,
)
from backend.app.services.inference_service import InferenceService
from backend.app.services.job_manager import VideoJobManager
from backend.app.services.training_job_manager import TrainingJobManager

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """System health status and device availability."""
    service = InferenceService.get_instance()
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        model_loaded=service.predictor.model_loaded,
        device=str(service.predictor.device),
        timestamp=datetime.utcnow().isoformat(),
    )


@router.get("/model/status", response_model=ModelStatusResponse)
def get_model_status():
    """Active model architecture, checkpoint status, and parameter details."""
    service = InferenceService.get_instance()
    model = service.predictor.model
    num_params = sum(p.numel() for p in model.parameters())

    return ModelStatusResponse(
        model_name="ISLSeq2SeqTransformer (4x4 Encoder-Decoder + CTC)",
        parameter_count=num_params,
        checkpoint_path=settings.CHECKPOINT_PATH,
        vocab_size=len(service.predictor.vocab),
        device=str(service.predictor.device),
        confidence_threshold=service.predictor.confidence_threshold,
        is_trained_checkpoint=service.predictor.model_loaded,
    )


@router.post("/translate", response_model=TranslationResponse)
def translate_english_to_telugu(request: TranslationRequest):
    """Direct English-to-Telugu translation with script verification and uncertainty handling."""
    service = InferenceService.get_instance()
    res = service.translator.translate(request.english_text, is_uncertain=request.is_uncertain)
    return TranslationResponse(
        english_source=res.english_source,
        telugu_text=res.telugu_text,
        script_valid=res.script_valid,
        is_uncertain=res.is_uncertain,
        engine=res.engine,
        warning=res.warning,
    )


@router.post("/tts", response_model=TTSResponse)
def synthesize_speech(request: TTSRequest):
    """Generates independent English or Telugu speech audio."""
    service = InferenceService.get_instance()
    res = service.tts.generate_audio_base64(request.text, lang=request.lang)
    return TTSResponse(
        success=res["success"],
        audio_base64=res["audio_base64"],
        format=res["format"],
        lang=res["lang"],
        error=res["error"],
    )


@router.post("/video/upload", response_model=JobCreateResponse)
async def upload_video(file: UploadFile = File(...)):
    """Uploads continuous ISL video file and enqueues async recognition and translation."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    allowed_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported video format '{ext}'. Supported: {', '.join(allowed_exts)}",
        )

    # Save to sandboxed temp directory
    safe_filename = f"{uuid.uuid4()}{ext}"
    dest_path = os.path.join(settings.UPLOAD_TEMP_DIR, safe_filename)

    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    # Enqueue in Job Manager
    job_mgr = VideoJobManager.get_instance()
    job_id = job_mgr.create_job(file.filename)
    await job_mgr.submit_video_job(job_id, dest_path)

    return JobCreateResponse(
        job_id=job_id,
        status="pending",
        filename=file.filename,
        message="Video accepted for processing.",
    )


@router.get("/video/job/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """Polls status and translation results of an uploaded video job."""
    job_mgr = VideoJobManager.get_instance()
    job = job_mgr.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


@router.post("/video/train", response_model=TrainJobCreateResponse)
async def upload_and_train_video(
    file: UploadFile = File(...),
    english_text: str = Form(...),
    telugu_text: Optional[str] = Form(None),
    signer_id: str = Form("custom_signer"),
    epochs: int = Form(20),
    batch_size: int = Form(8),
):
    """Uploads continuous ISL video, extracts 3D landmarks, updates manifests, and trains model asynchronously."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    if not english_text or not english_text.strip():
        raise HTTPException(status_code=400, detail="English translation text is required.")

    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    allowed_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported video format '{ext}'. Supported: {', '.join(allowed_exts)}",
        )

    # Save to sandboxed temp directory
    safe_filename = f"train_{uuid.uuid4()}{ext}"
    dest_path = os.path.join(settings.UPLOAD_TEMP_DIR, safe_filename)

    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    # Enqueue in Training Job Manager
    train_mgr = TrainingJobManager.get_instance()
    job_id = train_mgr.create_job(file.filename, english_text=english_text, telugu_text=telugu_text)
    await train_mgr.submit_train_job(
        job_id=job_id,
        video_path=dest_path,
        english_text=english_text,
        telugu_text=telugu_text,
        signer_id=signer_id,
        epochs=epochs,
        batch_size=batch_size,
    )

    return TrainJobCreateResponse(
        job_id=job_id,
        status="pending",
        filename=file.filename,
        english_text=english_text,
        message="Video accepted for landmark extraction and model training.",
    )


@router.get("/video/train/{job_id}", response_model=TrainJobStatusResponse)
def get_train_job_status(job_id: str):
    """Polls progress, training metrics, and test results of a video training job."""
    train_mgr = TrainingJobManager.get_instance()
    job = train_mgr.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job '{job_id}' not found.")
    return job


@router.post("/feedback")
def submit_feedback(feedback: FeedbackSubmission):
    """Submits user correction and adequacy rating for model improvement tracking."""
    feedback_dir = "data/reports/feedback"
    os.makedirs(feedback_dir, exist_ok=True)
    feedback_file = os.path.join(feedback_dir, "user_feedback.jsonl")

    import json
    with open(feedback_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(feedback.model_dump(), ensure_ascii=False) + "\n")

    return {"status": "success", "message": "Feedback recorded for model refinement."}
