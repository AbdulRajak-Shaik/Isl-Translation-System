"""
Asynchronous Video Job Manager.
Handles uploaded video processing, background execution, progress tracking,
and privacy-compliant cleanup of temporary files.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import uuid
from typing import Dict, Optional

from backend.app.core.config import settings
from backend.app.schemas.api_models import JobStatusResponse
from backend.app.services.inference_service import InferenceService
from ml.data.pose_extractor import PoseExtractor
from ml.data.video_processor import VideoProcessor


class VideoJobManager:
    """Manages asynchronous processing jobs for uploaded video files."""

    _instance = None

    @classmethod
    def get_instance(cls) -> "VideoJobManager":
        if cls._instance is None:
            cls._instance = VideoJobManager()
        return cls._instance

    def __init__(self):
        self.jobs: Dict[str, JobStatusResponse] = {}
        self.video_processor = VideoProcessor(target_fps=25)
        self.pose_extractor = PoseExtractor()
        self.executor = ThreadPoolExecutor(max_workers=2)

    def create_job(self, filename: str) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = JobStatusResponse(
            job_id=job_id,
            status="pending",
            progress=0.0,
        )
        return job_id

    def get_job(self, job_id: str) -> Optional[JobStatusResponse]:
        return self.jobs.get(job_id)

    def _process_video_sync(self, job_id: str, video_path: str):
        try:
            job = self.jobs[job_id]
            job.status = "processing"
            job.progress = 0.15

            # 1. Inspect Video Integrity
            meta = self.video_processor.inspect_video(video_path)
            if meta.is_corrupt:
                job.status = "failed"
                job.error = "Video file is corrupt or has unsupported codec."
                return

            if meta.duration_sec > settings.MAX_VIDEO_DURATION_SECONDS:
                job.status = "failed"
                job.error = f"Video exceeds maximum allowed duration ({settings.MAX_VIDEO_DURATION_SECONDS}s)."
                return

            job.progress = 0.35

            # 2. Extract Holistic Landmarks (Streaming, zero-copy frames)
            pose_seq = self.pose_extractor.extract_from_video(
                video_path,
                target_fps=25,
                max_frames=300,
                max_dimension=480,
                normalize=True,
            )
            if pose_seq is None or len(pose_seq) == 0:
                job.status = "failed"
                job.error = "Could not extract landmark features from video."
                return

            job.progress = 0.80

            # 3. Run Continuous Translation & Speech Synthesis
            service = InferenceService.get_instance()
            trans_res = service.translate_pose_sequence(pose_seq, provisional=False)

            en_text = trans_res["english_text"]
            te_text = trans_res["telugu_text"]

            # Generate TTS Audio
            audio_en = service.tts.generate_audio_base64(en_text, lang="en")
            audio_te = service.tts.generate_audio_base64(te_text, lang="te")

            # 5. Populate Result
            job.status = "completed"
            job.progress = 1.0
            job.english_translation = en_text
            job.telugu_translation = te_text
            job.confidence = trans_res["confidence"]
            job.is_uncertain = trans_res["is_uncertain"]
            job.english_audio_base64 = audio_en.get("audio_base64")
            job.telugu_audio_base64 = audio_te.get("audio_base64")

        except Exception as e:
            job = self.jobs.get(job_id)
            if job:
                job.status = "failed"
                job.error = f"Processing error: {str(e)}"
        finally:
            # Privacy Compliance: Discard temporary uploaded video
            if settings.AUTO_DELETE_PROCESSED_VIDEOS and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except OSError:
                    pass

    async def submit_video_job(self, job_id: str, video_path: str):
        loop = asyncio.get_event_loop()
        loop.run_in_executor(self.executor, self._process_video_sync, job_id, video_path)
