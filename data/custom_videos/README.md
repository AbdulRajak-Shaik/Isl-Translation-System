# Custom Video Training Guide

This directory is designed for you to easily add your own Indian Sign Language (ISL) video files and train the neural model to recognize them.

---

## 1. Quick Ingestion (Single Video)

To add a video and automatically train the model:

```bash
python scripts/ingest_video.py \
    --video "path/to/your_video.mp4" \
    --english "Sentence translated in English" \
    --train
```

* `--video`: The path to your video file (`.mp4`, `.avi`, `.mov`, `.webm`).
* `--english`: What the person is signing in English.
* `--telugu`: *(Optional)* Native Telugu script translation (if omitted, the system generates it automatically).
* `--signer`: *(Optional)* Signer name (e.g., `signer_01`, `signer_ramesh`).
* `--train`: Retrains the model immediately for 30 epochs and reloads your running backend application.

---

## 2. Batch Ingestion (Multiple Videos via CSV)

If you have multiple videos:

1. Place your video files in this directory or anywhere in the workspace.
2. Fill out [`annotations.csv`](./annotations.csv):
   ```csv
   video_path,english_text,telugu_text,signer_id
   data/custom_videos/video1.mp4,What is the time?,సమయం ఎంత?,signer_01
   data/custom_videos/video2.mp4,I am going home.,నేను ఇంటికి వెళ్తున్నాను.,signer_02
   ```
3. Run the batch ingestion and training command:
   ```bash
   python scripts/ingest_video.py --csv "data/custom_videos/annotations.csv" --train --epochs 30
   ```

---

## 3. What the System Does Automatically

1. **Integrity Check**: Inspects frame rate, resolution, and decodability.
2. **3D Landmark Extraction**: Runs MediaPipe Holistic to extract 75 3D keypoints (hands, face, torso) per frame.
3. **Scale Normalization**: Centers keypoints at mid-shoulder and applies shoulder-width normalization.
4. **Augmentations**: Generates spatial jitter and temporal speed variations to ensure generalization without overfitting.
5. **Vocabulary Extension**: Adds any new words into `data/manifests/vocab.json`.
6. **Model Training**: Runs `ISLSeq2SeqTransformer` with Joint CTC-Attention Loss.
7. **Instant Deployment**: Deploys the trained checkpoint to `ml/checkpoints/isl_transformer_best.pt` and reloads your running FastAPI web app.
