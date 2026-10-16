# API Reference Manual: Indian Sign Language Translation System

**Version:** 1.0.0  
**Base URL:** `http://localhost:8000/api/v1`  
**WebSocket URL:** `ws://localhost:8000/api/v1/live/ws`  

---

## 1. System Health & Model Inspection

### 1.1 `GET /health`
Returns system status, active version, device, and model checkpoint readiness.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "device": "cpu",
  "timestamp": "2026-10-15T06:05:27.839250"
}
```

### 1.2 `GET /model/status`
Inspects active neural network architecture, parameter counts, and vocabulary.

**Response:**
```json
{
  "model_name": "ISLSeq2SeqTransformer (4x4 Encoder-Decoder + CTC)",
  "parameter_count": 7516206,
  "checkpoint_path": "ml/checkpoints/isl_transformer_best.pt",
  "vocab_size": 23,
  "device": "cpu",
  "confidence_threshold": 0.55,
  "is_trained_checkpoint": true
}
```

---

## 2. Translation & Speech Synthesis

### 2.1 `POST /translate`
Translates English text into native Telugu script (`tel_Telu`).

**Request Body:**
```json
{
  "english_text": "Where is the hospital?",
  "is_uncertain": false
}
```

**Response:**
```json
{
  "english_source": "Where is the hospital?",
  "telugu_text": "ఆసుపత్రి ఎక్కడ ఉంది?",
  "script_valid": true,
  "is_uncertain": false,
  "engine": "IndicTrans2-Fallback-Engine",
  "warning": null
}
```

### 2.2 `POST /tts`
Generates natural neural audio in English (`en`) or Telugu (`te`).

**Request Body:**
```json
{
  "text": "ఆసుపత్రి ఎక్కడ ఉంది?",
  "lang": "te"
}
```

**Response:**
```json
{
  "success": true,
  "audio_base64": "data:audio/mp3;base64,//uQx...",
  "format": "audio/mp3",
  "lang": "te",
  "error": null
}
```

---

## 3. Video Upload and Async Processing

### 3.1 `POST /video/upload`
Uploads a continuous sign language video file for asynchronous decoding and translation.

*Content-Type:* `multipart/form-data`  
*Form Field:* `file` (MP4, WebM, AVI, MOV up to 50 MB)

**Response:**
```json
{
  "job_id": "89e5fc81-ef07-4286-9a25-c6bf3dc74a0c",
  "status": "pending",
  "filename": "sample_signing.mp4",
  "message": "Video accepted for processing."
}
```

### 3.2 `GET /video/job/{job_id}`
Polls status and retrieves final translations and audio buffers.

**Response (Completed):**
```json
{
  "job_id": "89e5fc81-ef07-4286-9a25-c6bf3dc74a0c",
  "status": "completed",
  "progress": 1.0,
  "english_translation": "Good morning doctor.",
  "telugu_translation": "శుభోదయం డాక్టర్.",
  "confidence": 0.88,
  "is_uncertain": false,
  "english_audio_base64": "data:audio/mp3;base64,...",
  "telugu_audio_base64": "data:audio/mp3;base64,...",
  "error": null
}
```

---

## 4. Live Camera WebSocket Protocol

### `WebSocket /live/ws`
Streaming protocol for real-time webcam frame ingestion.

**Client Frame Message:**
```json
{
  "type": "frame",
  "frame": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
}
```

**Server Translation Event:**
```json
{
  "type": "translation_update",
  "english_text": "Where is the hospital?",
  "telugu_text": "ఆసుపత్రి ఎక్కడ ఉంది?",
  "confidence": 0.84,
  "is_uncertain": false,
  "provisional": false,
  "is_new_utterance": true,
  "timestamp": 1729000123.45
}
```
