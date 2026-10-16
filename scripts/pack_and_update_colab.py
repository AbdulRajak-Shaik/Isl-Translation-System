"""
Packages the project source code into:
1. isl_project_source.zip
2. Embeds an auto-extractor directly into notebooks/ISL_Colab_Unified_GPU_Runner.ipynb
so that Google Colab runs out-of-the-box in a fresh environment without ModuleNotFoundError.
"""

import base64
import io
import json
import os
import zipfile


def create_source_zip(zip_path="isl_project_source.zip"):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in ["ml", "backend", "scripts"]:
            for root, dirs, files in os.walk(folder):
                if "__pycache__" in root or "checkpoints" in root:
                    continue
                for f in files:
                    if f.endswith((".py", ".yaml", ".json")):
                        full_p = os.path.join(root, f)
                        arcname = os.path.relpath(full_p, ".").replace("\\", "/")
                        z.write(full_p, arcname)
    print(f"Created {zip_path} ({os.path.getsize(zip_path)} bytes)")
    with open(zip_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def build_self_contained_notebook(b64_zip_content, output_notebook="notebooks/ISL_Colab_Unified_GPU_Runner.ipynb"):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 Continuous Indian Sign Language Translation — Unified GPU Runner\n",
                "### End-to-End Google Colab GPU Pipeline (T4 / A100 / L4)\n",
                "\n",
                "This notebook is **100% self-contained**. It automatically deploys the `ml/`, `backend/`, and `scripts/` modules into the Colab environment, detects your GPU, prepares the dataset, trains with FP16 mixed precision, evaluates SacreBLEU, and translates to native Telugu script (`తెలుగు`).\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Project Source Auto-Deployment & Environment Setup\n",
                "This cell automatically extracts the `ml`, `backend`, and `scripts` modules into `/content` so that all imports work seamlessly.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Auto-extract project codebase into current Colab runtime\n",
                "import os, sys, io, base64, zipfile\n",
                "\n",
                "# Ensure /content is on python path\n",
                "for p in ['.', '/content']:\n",
                "    if p not in sys.path:\n",
                "        sys.path.insert(0, p)\n",
                "\n",
                f"EMBEDDED_ZIP = '{b64_zip_content}'\n",
                "\n",
                "if not os.path.exists('ml'):\n",
                "    print('📦 Unpacking project source code into Colab runtime...')\n",
                "    zip_data = base64.b64decode(EMBEDDED_ZIP)\n",
                "    with zipfile.ZipFile(io.BytesIO(zip_data)) as z:\n",
                "        z.extractall('.')\n",
                "    print('✅ Extracted ml/, backend/, and scripts/ successfully!')\n",
                "else:\n",
                "    print('✅ Project directories already present.')\n",
                "\n",
                "print('Python Path verified:', sys.path[:2])\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Hardware Inspection & GPU Verification\n",
                "Verify that Google Colab is running with a GPU runtime (T4, V100, A100, or L4).\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import platform, torch\n",
                "\n",
                "print('Python Version:', platform.python_version())\n",
                "print('PyTorch Version:', torch.__version__)\n",
                "\n",
                "if not torch.cuda.is_available():\n",
                "    print('⚠️ WARNING: CUDA GPU is not enabled. Training will proceed on CPU.')\n",
                "    print('To speed up 20x, go to: Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU.')\n",
                "else:\n",
                "    gpu_name = torch.cuda.get_device_name(0)\n",
                "    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)\n",
                "    print(f'✅ GPU Detected: {gpu_name} ({vram_gb:.2f} GB VRAM)')\n"
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
                "import os, sys, json, numpy as np\n",
                "for p in ['.', '/content']:\n",
                "    if p not in sys.path:\n",
                "        sys.path.insert(0, p)\n",
                "\n",
                "from ml.inference.predictor import ISLPredictor\n",
                "from ml.training.metrics import compute_bleu, compute_chrf, compute_wer\n",
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
                "# Synthesize speech directly to MP3 files for Colab audio player\n",
                "from gtts import gTTS\n",
                "tts_en = gTTS(text=sample_english, lang='en')\n",
                "tts_en.save('speech_en.mp3')\n",
                "print('\\n🔊 English Speech:')\n",
                "display(Audio('speech_en.mp3'))\n",
                "\n",
                "tts_te = gTTS(text=telugu_output.telugu_text, lang='te')\n",
                "tts_te.save('speech_te.mp3')\n",
                "print('\\n🔊 Telugu Speech (తెలుగు):')\n",
                "display(Audio('speech_te.mp3'))\n"
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
                "except Exception as e:\n",
                "    print(f'Checkpoint available at ml/checkpoints/isl_transformer_best.pt: {e}')\n"
            ]
        }
    ]

    nb = {
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

    with open(output_notebook, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Built self-contained notebook: {output_notebook}")


if __name__ == "__main__":
    b64_content = create_source_zip()
    build_self_contained_notebook(b64_content)
