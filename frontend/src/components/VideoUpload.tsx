import { useRef, useState } from 'react';
import { UploadCloud, FileVideo, AlertCircle, Loader2, Sparkles } from 'lucide-react';
import { uploadVideo, pollJobStatus } from '../services/api';
import type { TranslationResult } from '../types/translation';

interface VideoUploadProps {
  onNewTranslation: (res: TranslationResult) => void;
  confidenceThreshold: number;
  onOpenTrainStudio?: () => void;
}

export const VideoUpload: React.FC<VideoUploadProps> = ({
  onNewTranslation,
  confidenceThreshold,
  onOpenTrainStudio,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [processingStatus, setProcessingStatus] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processSelectedFile(e.target.files[0]);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const processSelectedFile = (file: File) => {
    setErrorMessage(null);
    // Validate format
    const validExtensions = ['.mp4', '.webm', '.avi', '.mov'];
    const hasValidExt = validExtensions.some((ext) => file.name.toLowerCase().endsWith(ext));
    if (!hasValidExt) {
      setErrorMessage(`Invalid format. Please upload MP4, WebM, AVI, or MOV.`);
      return;
    }

    // Validate size (max 50 MB)
    if (file.size > 50 * 1024 * 1024) {
      setErrorMessage(`File exceeds 50 MB limit.`);
      return;
    }

    setSelectedFile(file);
    setVideoPreviewUrl(URL.createObjectURL(file));
  };

  const handleStartUpload = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setUploadProgress(0);
    setProcessingStatus('Uploading video to processing service...');
    setErrorMessage(null);

    try {
      const uploadRes = await uploadVideo(selectedFile, (percent) => {
        setUploadProgress(percent);
        if (percent >= 100) {
          setProcessingStatus('Extracting 3D holistic landmarks and decoding signing...');
        }
      });

      const jobId = uploadRes.job_id;
      pollJob(jobId);
    } catch (err: any) {
      setIsProcessing(false);
      setErrorMessage(err.message || 'Failed to upload video');
    }
  };

  const pollJob = (jobId: string) => {
    let consecutiveErrors = 0;
    const interval = window.setInterval(async () => {
      try {
        const job = await pollJobStatus(jobId);
        consecutiveErrors = 0;
        if (job.status === 'processing') {
          const progressPercent = Math.round(job.progress * 100);
          setProcessingStatus(`Processing temporal sequence (${progressPercent}%)...`);
        } else if (job.status === 'completed') {
          clearInterval(interval);
          setIsProcessing(false);
          setProcessingStatus(null);

          const result: TranslationResult = {
            id: `upload_${Date.now()}`,
            englishText: job.english_translation || '',
            teluguText: job.telugu_translation || '',
            confidence: job.confidence ?? 0.85,
            isUncertain: (job.confidence ?? 0.85) < confidenceThreshold,
            provisional: false,
            audioEn: job.english_audio_base64,
            audioTe: job.telugu_audio_base64,
            timestamp: Date.now(),
            source: 'upload',
          };
          onNewTranslation(result);
        } else if (job.status === 'failed') {
          clearInterval(interval);
          setIsProcessing(false);
          setProcessingStatus(null);
          setErrorMessage(job.error || 'Video recognition failed.');
        }
      } catch (err: any) {
        consecutiveErrors++;
        if (consecutiveErrors >= 3) {
          clearInterval(interval);
          setIsProcessing(false);
          setErrorMessage(err.message || 'Failed to monitor processing job');
        }
      }
    }, 1200);
  };

  return (
    <div className="upload-card glass-panel" id="video-upload-section">
      <div className="upload-header">
        <FileVideo size={20} className="camera-header-icon" />
        <h2 className="camera-heading">Upload Sign Language Video</h2>
      </div>

      {!selectedFile ? (
        <div
          className="drop-zone"
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="video/mp4,video/webm,video/quicktime,video/avi"
            style={{ display: 'none' }}
            onChange={handleFileChange}
          />
          <div className="drop-icon-ring">
            <UploadCloud size={40} />
          </div>
          <h3>Select or Drop Sign Language Video</h3>
          <p>Supports MP4, WebM, AVI, and MOV up to 50 MB / 3 minutes</p>
          <button className="btn-secondary" type="button">
            Browse Video Files
          </button>
        </div>
      ) : (
        <div className="preview-zone">
          {videoPreviewUrl && (
            <video
              src={videoPreviewUrl}
              controls
              className="upload-preview-player"
            />
          )}

          <div className="selected-file-meta">
            <div className="meta-info">
              <span className="file-name">{selectedFile.name}</span>
              <span className="file-size">{(selectedFile.size / (1024 * 1024)).toFixed(1)} MB</span>
            </div>

            {!isProcessing && (
              <div className="file-action-buttons">
                <button
                  className="btn-primary"
                  onClick={handleStartUpload}
                  id="btn-process-video"
                >
                  Translate Video
                </button>
                {onOpenTrainStudio && (
                  <button
                    className="btn-secondary"
                    onClick={onOpenTrainStudio}
                    title="Open Training Studio to teach the model this video"
                  >
                    <Sparkles size={16} />
                    Train Mode
                  </button>
                )}
                <button
                  className="btn-secondary"
                  onClick={() => {
                    setSelectedFile(null);
                    setVideoPreviewUrl(null);
                  }}
                >
                  Change File
                </button>
              </div>
            )}
          </div>

          {isProcessing && (
            <div className="processing-progress-card">
              <div className="progress-label-row">
                <span className="status-text">
                  <Loader2 size={16} className="spin-icon" />
                  {processingStatus}
                </span>
                <span className="percent-text">{uploadProgress}%</span>
              </div>
              <div className="progress-bar-track">
                <div
                  className="progress-bar-fill"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          )}
        </div>
      )}

      {errorMessage && (
        <div className="camera-error-banner" role="alert">
          <AlertCircle size={18} />
          <span>{errorMessage}</span>
        </div>
      )}
    </div>
  );
};
