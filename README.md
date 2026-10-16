# Professional Indian Sign Language (ISL) Recognition and Continuous Translation System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-teal.svg)](https://fastapi.tiangolo.com)
[![React 19](https://img.shields.io/badge/Frontend-React_Vite-indigo.svg)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-orange.svg)](https://pytorch.org)
[![License: Academic & CC BY 4.0](https://img.shields.io/badge/License-Academic%20%2F%20CC%20BY%204.0-green.svg)](LICENSE)

An enterprise-grade, research-grounded **Continuous Indian Sign Language (ISL) Translation and Speech Generation System**. Unlike toy projects that recognize only a handful of isolated gestures, this system processes **unsegmented, continuous signing video streams**, decodes natural **English sentences**, cascades into native **Telugu script (`తెలుగు`)**, and generates synchronized **dual-language Text-to-Speech (TTS) audio**.

---

## 1. Key System Capabilities

- **Continuous Sentence Translation:** Processes continuous spatio-temporal signing without manual boundary segmentation or reliance on intermediate gloss vocabularies.
- **Multimodal Skeletal Normalization:** Extracts 75 3D holistic landmarks (hands, upper torso, face) using MediaPipe, normalized relative to shoulder width for signer and distance invariance.
- **Joint CTC-Attention Transformer Architecture:** 4-layer Transformer Encoder and 4-layer Autoregressive Decoder trained with joint Cross-Entropy and Connectionist Temporal Classification (CTC) losses.
- **Calibrated Uncertainty Estimation:** Computes prediction entropy across decoded tokens; flags low-confidence or out-of-distribution signing with `[ధృవీకరణ అవసరం]` ("Clarification Required") instead of hallucinating fluent text.
- **Native Telugu Script Translation:** Implements AI4Bharat IndicTrans2 (`tel_Telu`) standards with Unicode script validation.
- **Synchronized English & Telugu TTS:** Independent neural speech synthesis with play, pause, replay, and browser speech API fallbacks.
- **Dual Modality Input:**
  - **Live Camera:** WebSocket streaming with 60-frame sliding context buffer.
  - **Video Upload:** Async background processing with drag-and-drop support, format validation, and automatic privacy cleanup of temporary media files.

---

## 2. System Architecture

```
                       ┌────────────────────────────────────────┐
                       │          LIVE WEBCAM / VIDEO FILE      │
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │    MediaPipe 3D Holistic Landmarks     │
                       │    (75 keypoints x 3 coords = 225)     │
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │   Body-Centric Spatial Normalization   │
                       │  (Mid-shoulder center + Scale factor)  │
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │       Spatio-Temporal Transformer      │
                       │          Encoder (4 Layers)            │
                       └─────────────┬────────────────────┬─────┘
                                     │                    │
                                     ▼                    ▼
                        ┌──────────────────────┐ ┌───────────────┐
                        │   Autoregressive     │ │ Auxiliary CTC │
                        │   Seq2Seq Decoder    │ │ Loss Head     │
                        └────────────┬─────────┘ └───────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────────┐
                        │     English Sentence Text    │
                        │   + Calibrated Uncertainty   │
                        └────────────┬─────────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────────┐
                        │  AI4Bharat IndicTrans2 NMT   │
                        │    Native Telugu Script      │
                        └────────────┬─────────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────────┐
                        │ Dual English & Telugu Speech │
                        │  Synthesis Engine (gTTS/Web) │
                        └──────────────────────────────┘
```

---

## 3. Verified Research Foundations & Dataset Audit

This project implements architectures and evaluation standards benchmarked against peer-reviewed literature:

- **ISLTranslate (ACL 2023 Findings):** 31,222 continuous video-to-sentence pairs covering 11,655 vocabulary items. Primary continuous baseline.
- **iSign (ACL 2024 Findings):** Over 118,000 video-sentence pairs across SignVideo2Text and SignPose2Text benchmark tasks.
- **ISL-CSLTR (Mendeley Data 2021):** 700 continuous videos across 7 signers under CC BY 4.0 license. Used for smoke tests and validation.
- **CISLR & INCLUDE:** Large-scale isolated sign databases for auxiliary representation learning and hand landmark robustness.
- **AI4Bharat IndicTrans2:** State-of-the-art transformer NMT for Indian languages (`eng_Latn` → `tel_Telu`).

*Detailed audits and legal analysis are documented in the [research/](research/) directory:*
- [research/RESEARCH_REPORT.md](research/RESEARCH_REPORT.md)
- [research/DATASET_REGISTRY.md](research/DATASET_REGISTRY.md)
- [research/LICENSE_AUDIT.md](research/LICENSE_AUDIT.md)
- [research/MODEL_COMPARISON.md](research/MODEL_COMPARISON.md)
- [research/EVALUATION_PROTOCOL.md](research/EVALUATION_PROTOCOL.md)
- [research/KNOWN_LIMITATIONS.md](research/KNOWN_LIMITATIONS.md)

---

## 4. Quick Start & Installation

### Prerequisites
- Python 3.10+ (Tested on Python 3.12.9)
- Node.js 18+ and npm (Tested on Node v22.19.0)
- FFmpeg (for video decoding)

### Step 1: Clone and Set Up Python Environment
```bash
git clone <repository_url>
cd "Major Project 2"

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### Step 2: Initialize Dataset Manifests & Smoke Checkpoint
```bash
# Generate validation fixture, manifests, and vocabulary
python scripts/prepare_data.py --fixture

# Train baseline checkpoint
python ml/training/train.py --smoke-test
```

### Step 3: Start FastAPI Backend Service
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
*API interactive documentation will be available at:* `http://localhost:8000/docs`

### Step 4: Start React Frontend
In a separate terminal:
```bash
cd frontend
npm install
npm run dev
```
*Open your browser at:* `http://localhost:5173`

---

## 5. Google Colab GPU Training Pipeline

For large-scale multi-epoch training on NVIDIA T4/A100 GPUs, follow the 7 sequential Colab notebooks:

1. `notebooks/01_environment_setup.ipynb` — GPU inspection, Drive mount, dependency checks.
2. `notebooks/02_dataset_audit.ipynb` — Video decoding verification and signer leakage checks.
3. `notebooks/03_preprocessing.ipynb` — MediaPipe holistic pose extraction and normalization.
4. `notebooks/04_baseline_training.ipynb` — Baseline model training with mixed precision (FP16).
5. `notebooks/05_advanced_training.ipynb` — Cosine annealing LR and error-weighted losses.
6. `notebooks/06_evaluation.ipynb` — Held-out test evaluation (SacreBLEU, chrF++, WER).
7. `notebooks/07_inference_demo.ipynb` — End-to-end continuous translation and audio playback demo.

*Detailed Colab instructions:* See [docs/COLAB_GUIDE.md](docs/COLAB_GUIDE.md).

---

## 6. Running Automated Tests

Run the full pytest suite (15 unit and integration tests covering data pipelines, model forward passes, translation, TTS, and API endpoints):

```bash
python -m pytest tests -v
```

---

## 7. Docker Deployment

Deploy the full stack with Docker Compose:

```bash
docker-compose -f docker/docker-compose.yml up --build
```
- Frontend UI: `http://localhost:3000`
- Backend API: `http://localhost:8000`

---

## 8. Repository Structure

```
Major Project 2/
├── backend/                  # FastAPI backend service
│   ├── app/
│   │   ├── api/              # REST routes & WebSocket live stream
│   │   ├── core/             # Configuration & settings
│   │   ├── schemas/          # Pydantic API contracts
│   │   ├── services/         # Inference singleton & async job manager
│   │   ├── translation/      # English-to-Telugu NMT & script validator
│   │   ├── tts/              # Speech synthesis engine
│   │   └── main.py           # Application entry point
│   └── requirements.txt
├── frontend/                 # React + Vite + TypeScript application
│   ├── src/
│   │   ├── components/       # LiveCamera, VideoUpload, TranslationPanel, etc.
│   │   ├── services/         # REST API & WebSocket clients
│   │   ├── types/            # TypeScript data contracts
│   │   ├── App.tsx           # Main application view
│   │   ├── App.css           # Component styling
│   │   └── index.css         # Core design system tokens
│   └── package.json
├── ml/                       # Machine Learning core
│   ├── configs/              # Model and training YAML configs
│   ├── data/                 # Video processing, pose extraction, schemas
│   ├── models/               # Transformer Encoder, Decoder, CTC head
│   ├── training/             # PyTorch trainer, metrics (BLEU, chrF, WER)
│   ├── inference/            # Predictor and temporal sliding window
│   └── checkpoints/          # Model weights (best & last)
├── notebooks/                # 7 complete Google Colab GPU notebooks
├── research/                 # Academic literature review, audits, benchmarks
├── tests/                    # Unit and integration test suite
├── scripts/                  # Data preparation and notebook builder CLI
├── docker/                   # Dockerfiles and compose configs
├── docs/                     # API reference and Colab guide
├── .env.example              # Configuration template
├── PROGRESS.md               # Continuous implementation log
└── README.md                 # Project documentation
```

---

## 9. License & Ethical Use

- **Software Code:** Licensed under the MIT License.
- **Benchmark Data:** Subject to academic research licenses of respective creators (Exploration-Lab, Mendeley Data CC BY 4.0).
- **Video Privacy Policy:** No raw user webcam or uploaded video is permanently stored. Frames are processed in ephemeral memory buffers and automatically unlinked after inference.
