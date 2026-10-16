"""
WebSocket Endpoint for Real-Time Live Camera Inference.
Receives streamed camera frames, maintains sliding temporal context,
and yields synchronized English and Telugu translations.
"""

import base64
import json
import time
import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from backend.app.services.inference_service import InferenceService
from ml.data.pose_extractor import PoseExtractor
from ml.inference.predictor import StreamingWindowBuffer


async def live_translation_websocket(websocket: WebSocket):
    await websocket.accept()
    print("[WebSocket] Client connected for live ISL translation stream.")

    service = InferenceService.get_instance()
    pose_extractor = PoseExtractor()
    window_buffer = StreamingWindowBuffer(window_size=60, stride=15)

    last_emitted_text = ""

    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue

            msg_type = payload.get("type", "frame")

            if msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "timestamp": time.time()}))
                continue

            if msg_type == "reset":
                window_buffer.clear()
                last_emitted_text = ""
                await websocket.send_text(json.dumps({"type": "reset_ack"}))
                continue

            if msg_type == "frame":
                b64_image = payload.get("frame", "")
                if not b64_image:
                    continue

                # Strip base64 header if present (e.g., "data:image/jpeg;base64,")
                if "," in b64_image:
                    b64_image = b64_image.split(",", 1)[1]

                try:
                    img_bytes = base64.b64decode(b64_image)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame_bgr is None:
                        continue
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                except Exception:
                    continue

                # Extract Holistic landmarks for single frame
                keypoints = pose_extractor.extract_landmarks_from_frame(frame_rgb)
                norm_keypoints = pose_extractor.normalize_keypoints(keypoints.reshape(1, 75, 3)).reshape(225)

                ready = window_buffer.add_frame(norm_keypoints)

                if ready:
                    window_seq = window_buffer.get_current_window()
                    trans_res = service.translate_pose_sequence(window_seq, provisional=True)
                    en_text = trans_res["english_text"]
                    te_text = trans_res["telugu_text"]

                    # Avoid redundant repetition if prediction hasn't changed
                    is_new = en_text and en_text != last_emitted_text
                    if is_new:
                        last_emitted_text = en_text

                    response = {
                        "type": "translation_update",
                        "english_text": en_text,
                        "telugu_text": te_text,
                        "confidence": trans_res["confidence"],
                        "is_uncertain": trans_res["is_uncertain"],
                        "provisional": trans_res["provisional"],
                        "is_new_utterance": is_new,
                        "timestamp": time.time(),
                    }
                    await websocket.send_text(json.dumps(response, ensure_ascii=False))

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected.")
    except Exception as e:
        print(f"[WebSocket] Stream error: {e}")
    finally:
        pose_extractor.close()
