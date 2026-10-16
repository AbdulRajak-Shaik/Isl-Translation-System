import { useEffect, useRef, useState } from 'react';
import { Camera, CameraOff, Play, Pause, Square, AlertCircle, Wifi, WifiOff } from 'lucide-react';
import { LiveWebSocketClient } from '../services/websocket';
import type { TranslationResult } from '../types/translation';

interface LiveCameraProps {
  onNewTranslation: (res: TranslationResult) => void;
  confidenceThreshold: number;
}

export const LiveCamera: React.FC<LiveCameraProps> = ({
  onNewTranslation,
  confidenceThreshold,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [wsStatus, setWsStatus] = useState<'connected' | 'connecting' | 'disconnected' | 'error'>('disconnected');
  const [cameraError, setCameraError] = useState<string | null>(null);

  const wsClientRef = useRef<LiveWebSocketClient | null>(null);
  const frameIntervalRef = useRef<number | null>(null);

  // Initialize WebSocket client
  useEffect(() => {
    const isViteDev = location.port === '5173';
    const wsDefaultHost = isViteDev ? 'localhost:8000/api/v1/live/ws' : `${location.host}/api/v1/live/ws`;
    const wsHost =
      import.meta.env.VITE_WS_URL ||
      (import.meta.env.VITE_API_HOST
        ? `${import.meta.env.VITE_API_HOST}/api/v1/live/ws`
        : wsDefaultHost);
    const wsUrl = (location.protocol === 'https:' ? 'wss://' : 'ws://') + wsHost;

    const client = new LiveWebSocketClient(wsUrl, {
      onStatusChange: (status) => setWsStatus(status),
      onTranslation: (data) => {
        if (!data.english_text) return;
        const result: TranslationResult = {
          id: `live_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
          englishText: data.english_text,
          teluguText: data.telugu_text,
          confidence: data.confidence,
          isUncertain: data.confidence < confidenceThreshold,
          provisional: data.provisional,
          timestamp: Date.now(),
          source: 'live',
        };
        onNewTranslation(result);
      },
      onError: (err) => console.error('WebSocket error:', err),
    });

    wsClientRef.current = client;
    client.connect();

    return () => {
      client.disconnect();
    };
  }, [confidenceThreshold, onNewTranslation]);

  // Start Camera Stream
  const startCamera = async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 25 },
        },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setIsStreaming(true);
      setIsPaused(false);
      startFrameStreaming();
    } catch (err: any) {
      setCameraError(
        err.name === 'NotAllowedError'
          ? 'Camera access denied. Please allow camera permissions in your browser.'
          : 'Could not access camera. Please check your video device.'
      );
    }
  };

  // Stop Camera Stream
  const stopCamera = () => {
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }

    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }

    if (wsClientRef.current) {
      wsClientRef.current.reset();
    }

    setIsStreaming(false);
    setIsPaused(false);
  };

  // Toggle Pause
  const togglePause = () => {
    if (isPaused) {
      wsClientRef.current?.resume();
      setIsPaused(false);
    } else {
      wsClientRef.current?.pause();
      setIsPaused(true);
    }
  };

  // Frame Streaming Loop (12 FPS to balance bandwidth and temporal resolution)
  const startFrameStreaming = () => {
    if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);

    frameIntervalRef.current = window.setInterval(() => {
      if (!videoRef.current || !canvasRef.current || isPaused) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (video.videoWidth === 0 || video.videoHeight === 0) return;

      canvas.width = 320;
      canvas.height = 240;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const base64Data = canvas.toDataURL('image/jpeg', 0.65);

      if (wsClientRef.current) {
        wsClientRef.current.sendFrame(base64Data);
      }
    }, 85); // ~12 FPS
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  return (
    <div className="camera-card glass-panel" id="live-camera-section">
      <div className="camera-header">
        <div className="camera-title-group">
          <Camera size={20} className="camera-header-icon" />
          <h2 className="camera-heading">Live ISL Recognition</h2>
          {isStreaming && !isPaused && (
            <span className="live-status-pill">
              <span className="recording-dot" />
              <span>LIVE</span>
            </span>
          )}
        </div>

        <div className="connection-indicator">
          {wsStatus === 'connected' ? (
            <span className="badge badge-success" title="WebSocket connected">
              <Wifi size={13} /> Model Ready
            </span>
          ) : (
            <span className="badge badge-danger" title="WebSocket disconnected">
              <WifiOff size={13} /> Disconnected
            </span>
          )}
        </div>
      </div>

      <div className="video-viewport-wrapper">
        <video
          ref={videoRef}
          playsInline
          muted
          className={`webcam-feed ${isStreaming ? 'active' : 'hidden'}`}
          aria-label="Webcam Video Stream"
        />
        <canvas ref={canvasRef} className="frame-capture-canvas" style={{ display: 'none' }} />

        {!isStreaming && (
          <div className="camera-placeholder">
            <div className="placeholder-icon-ring">
              <CameraOff size={36} />
            </div>
            <h3>Camera is Turned Off</h3>
            <p>Allow camera permissions to begin real-time continuous Indian Sign Language translation.</p>
            <button
              id="btn-start-camera"
              className="btn-primary"
              onClick={startCamera}
            >
              <Play size={16} /> Enable Camera
            </button>
          </div>
        )}

        {cameraError && (
          <div className="camera-error-banner" role="alert">
            <AlertCircle size={18} />
            <span>{cameraError}</span>
          </div>
        )}
      </div>

      {isStreaming && (
        <div className="camera-controls">
          <button
            id="btn-pause-stream"
            className="btn-secondary"
            onClick={togglePause}
          >
            {isPaused ? <Play size={16} /> : <Pause size={16} />}
            <span>{isPaused ? 'Resume' : 'Pause'}</span>
          </button>

          <button
            id="btn-stop-camera"
            className="btn-danger"
            onClick={stopCamera}
          >
            <Square size={16} />
            <span>Stop Camera</span>
          </button>
        </div>
      )}
    </div>
  );
};
