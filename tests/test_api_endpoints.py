"""
Integration Tests for FastAPI REST Endpoints.
"""

from io import BytesIO
from fastapi.testclient import TestClient
import pytest

from backend.app.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestAPIEndpoints:

    def test_health_endpoint(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "model_loaded" in data

    def test_model_status_endpoint(self, client):
        response = client.get("/api/v1/model/status")
        assert response.status_code == 200
        data = response.json()
        assert "ISLSeq2SeqTransformer" in data["model_name"]
        assert data["parameter_count"] > 0
        assert data["vocab_size"] > 0

    def test_translation_endpoint(self, client):
        payload = {"english_text": "Good morning doctor.", "is_uncertain": False}
        response = client.post("/api/v1/translate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["script_valid"] is True
        assert len(data["telugu_text"]) > 0

    def test_tts_endpoint(self, client):
        payload = {"text": "Hello world", "lang": "en"}
        response = client.post("/api/v1/tts", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        if data["success"]:
            assert data["audio_base64"] is not None
        else:
            assert data["error"] is not None

    def test_invalid_video_upload_rejected(self, client):
        # Uploading an invalid text file pretending to be video
        fake_file = BytesIO(b"not a video")
        files = {"file": ("malicious.exe", fake_file, "application/octet-stream")}
        response = client.post("/api/v1/video/upload", files=files)
        assert response.status_code == 415

    def test_feedback_endpoint(self, client):
        payload = {
            "english_predicted": "Where is the hospital?",
            "telugu_predicted": "ఆసుపత్రి ఎక్కడ ఉంది?",
            "rating": 5,
            "comments": "Accurate prediction on smoke test",
        }
        response = client.post("/api/v1/feedback", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_train_video_missing_fields_rejected(self, client):
        fake_file = BytesIO(b"fake video content")
        files = {"file": ("test.mp4", fake_file, "video/mp4")}
        response = client.post("/api/v1/video/train", files=files)
        assert response.status_code == 422

    def test_train_video_invalid_format_rejected(self, client):
        fake_file = BytesIO(b"fake text")
        files = {"file": ("test.txt", fake_file, "text/plain")}
        data = {"english_text": "Good morning."}
        response = client.post("/api/v1/video/train", files=files, data=data)
        assert response.status_code == 415
