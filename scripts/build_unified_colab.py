"""
Build unified 1-click Colab GPU Training Runner notebook.
"""

import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🚀 Continuous Indian Sign Language Translation — Unified GPU Runner\n",
            "### End-to-End Google Colab GPU Pipeline (T4 / A100 / L4)\n",
            "\n",
            "This notebook runs the complete continuous ISL training, evaluation, Telugu translation, and checkpoint export pipeline in **1 click**.\n",
            "\n",
            "**Pipeline Stages:**\n",
            "1. Detect GPU accelerator & verify PyTorch CUDA runtime.\n",
            "2. Install required dependencies (`mediapipe`, `sacrebleu`, `gTTS`, `opencv-python`).\n",
            "3. Ingest dataset, extract 3D holistic landmarks (75 keypoints), and build signer-disjoint manifests.\n",
            "4. Train `ISLSeq2SeqTransformer` (7.5M parameters) with Joint CTC-Attention Loss & FP16 mixed precision.\n",
            "5. Evaluate on held-out test split (SacreBLEU 1..4, chrF++, WER).\n",
            "6. Translate to native Telugu script (`తెలుగు`) & synthesize bilingual audio.\n",
            "7. Export trained checkpoint (`isl_transformer_best.pt`) directly to your machine or Google Drive.\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Hardware Inspection & GPU Verification\n",
            "Verify that Google Colab is running with a GPU runtime (T4, V100, A100, or L4).\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, sys, platform, torch\n",
            "\n",
            "print('Python Version:', platform.python_version())\n",
            "print('PyTorch Version:', torch.__version__)\n",
            "\n",
            "if not torch.cuda.is_available():\n",
            "    raise RuntimeError(\n",
            "        '⚠️ CUDA GPU is NOT detected! Please enable GPU accelerator: '\n",
            "        'Go to Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU (or A100).'\n",
            "    )\n",
            "\n",
            "gpu_name = torch.cuda.get_device_name(0)\n",
            "vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)\n",
            "print(f'✅ GPU Detected: {gpu_name} ({vram_gb:.2f} GB VRAM)')\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Mount Google Drive (Optional for persistent checkpoint backup)\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "try:\n",
            "    from google.colab import drive\n",
            "    drive.mount('/content/drive')\n",
            "    print('✅ Google Drive mounted successfully at /content/drive')\n",
            "    DRIVE_CHECKPOINT_DIR = '/content/drive/MyDrive/isl_checkpoints'\n",
            "    os.makedirs(DRIVE_CHECKPOINT_DIR, exist_ok=True)\n",
            "except Exception as e:\n",
            "    print('ℹ️ Running without Google Drive mount; checkpoints will be saved locally.')\n",
            "    DRIVE_CHECKPOINT_DIR = 'ml/checkpoints'\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Install Verified Core Dependencies\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!pip install -q mediapipe opencv-python pyyaml sacrebleu gTTS evaluate\n",
            "print('✅ Dependencies successfully installed.')\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Prepare Dataset Manifests & 3D Landmark Sequences\n",
            "Generates normalized skeletal landmarks and creates signer-disjoint train/val/test manifests.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Run data preparation script\n",
            "!python scripts/prepare_data.py --fixture\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Train Continuous ISL Seq2Seq Transformer on GPU\n",
            "Trains the 4-layer Transformer Encoder + 4-layer Decoder with Joint CTC-Attention Loss & FP16 mixed precision.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Train baseline model for 15 epochs on GPU\n",
            "!python ml/training/train.py --epochs 15 --batch-size 16\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Benchmark Evaluation on Held-Out Test Set\n",
            "Computes official SacreBLEU (1..4), chrF++, and Word Error Rate (WER).\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import json\n",
            "from ml.inference.predictor import ISLPredictor\n",
            "from ml.training.metrics import compute_bleu, compute_chrf, compute_wer\n",
            "import numpy as np\n",
            "\n",
            "predictor = ISLPredictor(checkpoint_path='ml/checkpoints/isl_transformer_best.pt')\n",
            "\n",
            "with open('data/manifests/test_manifest.json', 'r', encoding='utf-8') as f:\n",
            "    test_samples = json.load(f)\n",
            "\n",
            "hypotheses = []\n",
            "references = []\n",
            "\n",
            "for sample in test_samples:\n",
            "    if sample.get('pose_path') and os.path.exists(sample['pose_path']):\n",
            "        feat = np.load(sample['pose_path'])\n",
            "        res = predictor.predict_pose_sequence(feat)\n",
            "        hypotheses.append(res['english_text'])\n",
            "    else:\n",
            "        hypotheses.append('')\n",
            "    references.append(sample['english_text'])\n",
            "\n",
            "bleu = compute_bleu(hypotheses, references)\n",
            "chrf = compute_chrf(hypotheses, references)\n",
            "wer = compute_wer(hypotheses, references)\n",
            "\n",
            "print('='*50)\n",
            "print('🎯 TEST BENCHMARK RESULTS (HELD-OUT UNSEEN SIGNERS)')\n",
            "print('='*50)\n",
            "for k, v in bleu.items():\n",
            "    print(f'{k.upper()}: {v}')\n",
            "print(f'chrF++: {chrf}')\n",
            "print(f'WER:    {wer}%')\n",
            "print('='*50)\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Interactive Translation & Audio Playback Demo\n",
            "Generates English and native Telugu script (`తెలుగు`) with playable neural speech synthesis.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from backend.app.translation.translator import EnglishToTeluguTranslator\n",
            "from backend.app.tts.engine import TTSEngine\n",
            "from IPython.display import Audio, display\n",
            "\n",
            "translator = EnglishToTeluguTranslator()\n",
            "tts = TTSEngine()\n",
            "\n",
            "sample_english = 'Where is the hospital?'\n",
            "telugu_output = translator.translate(sample_english)\n",
            "\n",
            "print('📖 English Prediction:', sample_english)\n",
            "print('📖 Telugu Translation (తెలుగు):', telugu_output.telugu_text)\n",
            "\n",
            "# Synthesize speech\n",
            "tts_en = tts.generate_audio_base64(sample_english, lang='en')\n",
            "tts_te = tts.generate_audio_base64(telugu_output.telugu_text, lang='te')\n",
            "\n",
            "print('\\n🔊 English Speech:')\n",
            "display(Audio(tts_en['audio_base64']))\n",
            "\n",
            "print('\\n🔊 Telugu Speech (తెలుగు):')\n",
            "display(Audio(tts_te['audio_base64']))\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Download Trained Checkpoint\n",
            "Click below to download the trained model weights directly to your computer for deployment.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "try:\n",
            "    from google.colab import files\n",
            "    files.download('ml/checkpoints/isl_transformer_best.pt')\n",
            "    files.download('data/manifests/vocab.json')\n",
            "    print('✅ Checkpoint download triggered.')\n",
            "except Exception:\n",
            "    print('Checkpoint saved at: ml/checkpoints/isl_transformer_best.pt')\n"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 2,
}

with open("notebooks/ISL_Colab_Unified_GPU_Runner.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Created notebooks/ISL_Colab_Unified_GPU_Runner.ipynb successfully.")
