import { useRef, useState } from 'react';
import {
  UploadCloud,
  AlertCircle,
  Loader2,
  CheckCircle2,
  Volume2,
  Sparkles,
  Cpu,
  Languages,
} from 'lucide-react';
import { uploadAndTrainVideo, pollTrainJobStatus, translateText } from '../services/api';
import type { TrainJobStatus, TranslationResult } from '../types/translation';

interface TrainingStudioProps {
  onNewTranslation?: (res: TranslationResult) => void;
  onModelRetrained?: () => void;
}

export const TrainingStudio: React.FC<TrainingStudioProps> = ({
  onNewTranslation,
  onModelRetrained,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);

  // Form Fields
  const [englishText, setEnglishText] = useState<string>('');
  const [teluguText, setTeluguText] = useState<string>('');
  const [signerId, setSignerId] = useState<string>('signer_custom');
  const [epochs, setEpochs] = useState<number>(20);

  // Training state
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainStatus, setTrainStatus] = useState<TrainJobStatus | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Audio Playback
  const [isPlayingEn, setIsPlayingEn] = useState<boolean>(false);
  const [isPlayingTe, setIsPlayingTe] = useState<boolean>(false);

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
    const validExtensions = ['.mp4', '.webm', '.avi', '.mov'];
    const hasValidExt = validExtensions.some((ext) => file.name.toLowerCase().endsWith(ext));
    if (!hasValidExt) {
      setErrorMessage('Invalid format. Please upload MP4, WebM, AVI, or MOV.');
      return;
    }

    if (file.size > 100 * 1024 * 1024) {
      setErrorMessage('File exceeds 100 MB limit.');
      return;
    }

    setSelectedFile(file);
    setVideoPreviewUrl(URL.createObjectURL(file));
  };

  const handleAutoTranslate = async () => {
    if (!englishText.trim()) return;
    try {
      const data = await translateText(englishText);
      setTeluguText(data.telugu_text);
    } catch {
      // Fallback
    }
  };

  const handleStartTraining = async () => {
    if (!selectedFile) {
      setErrorMessage('Please select a video file.');
      return;
    }
    if (!englishText.trim()) {
      setErrorMessage('Please provide the English ground-truth translation.');
      return;
    }

    setIsTraining(true);
    setErrorMessage(null);
    setTrainStatus({
      job_id: '',
      status: 'pending',
      progress: 0.05,
      stage: 'Uploading video to training pipeline...',
      current_epoch: 0,
      total_epochs: epochs,
    });

    try {
      const res = await uploadAndTrainVideo(
        selectedFile,
        englishText.trim(),
        teluguText.trim() || undefined,
        epochs,
        (percent) => {
          if (percent >= 100) {
            setTrainStatus((prev) =>
              prev ? { ...prev, stage: 'Extracting 3D holistic landmarks (hands, face, pose)...' } : null
            );
          } else {
            setTrainStatus((prev) =>
              prev ? { ...prev, stage: `Uploading video (${percent}%)...` } : null
            );
          }
        }
      );

      pollTraining(res.job_id);
    } catch (err: any) {
      setIsTraining(false);
      setErrorMessage(err.message || 'Failed to start video training job');
    }
  };

  const pollTraining = (jobId: string) => {
    let consecutiveErrors = 0;
    const interval = window.setInterval(async () => {
      try {
        const job: TrainJobStatus = await pollTrainJobStatus(jobId);
        consecutiveErrors = 0;
        setTrainStatus(job);

        if (job.status === 'completed') {
          clearInterval(interval);
          setIsTraining(false);
          if (onModelRetrained) onModelRetrained();

          // Push to history/translation panel if callback available
          if (job.test_result && onNewTranslation) {
            const result: TranslationResult = {
              id: `trained_test_${Date.now()}`,
              englishText: job.test_result.english_text,
              teluguText: job.test_result.telugu_text,
              confidence: job.test_result.confidence,
              isUncertain: job.test_result.is_uncertain,
              provisional: false,
              audioEn: job.test_result.audio_en,
              audioTe: job.test_result.audio_te,
              timestamp: Date.now(),
              source: 'upload',
            };
            onNewTranslation(result);
          }
        } else if (job.status === 'failed') {
          clearInterval(interval);
          setIsTraining(false);
          setErrorMessage(job.error || 'Training encountered an error.');
        }
      } catch (err: any) {
        consecutiveErrors++;
        if (consecutiveErrors >= 3) {
          clearInterval(interval);
          setIsTraining(false);
          setErrorMessage(err.message || 'Failed to poll training progress');
        }
      }
    }, 1500);
  };

  const playAudio = (audioBase64?: string, isTelugu: boolean = false) => {
    if (!audioBase64) return;
    const audio = new Audio(audioBase64);
    if (isTelugu) {
      setIsPlayingTe(true);
      audio.onended = () => setIsPlayingTe(false);
      audio.onerror = () => setIsPlayingTe(false);
    } else {
      setIsPlayingEn(true);
      audio.onended = () => setIsPlayingEn(false);
      audio.onerror = () => setIsPlayingEn(false);
    }
    audio.play().catch(() => {
      setIsPlayingEn(false);
      setIsPlayingTe(false);
    });
  };

  return (
    <div className="upload-card glass-panel" id="training-studio-section">
      <div className="upload-header">
        <Cpu size={22} className="camera-header-icon" />
        <div>
          <h2 className="camera-heading">Video Training & Fine-Tuning Studio</h2>
          <p className="camera-subheading">
            Teach the continuous ISL model new videos and sentences in 1 click
          </p>
        </div>
      </div>

      <div className="studio-layout">
        {/* Left Side: Video Selection & Preview */}
        <div className="studio-video-col">
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
                <UploadCloud size={36} />
              </div>
              <h3>Select ISL Video to Train</h3>
              <p>Supports MP4, WebM, AVI up to 100 MB</p>
              <button className="btn-secondary" type="button">
                Browse Video
              </button>
            </div>
          ) : (
            <div className="preview-zone">
              {videoPreviewUrl && (
                <video src={videoPreviewUrl} controls className="upload-preview-player" />
              )}
              <div className="selected-file-meta">
                <div className="meta-info">
                  <span className="file-name">{selectedFile.name}</span>
                  <span className="file-size">
                    {(selectedFile.size / (1024 * 1024)).toFixed(1)} MB
                  </span>
                </div>
                {!isTraining && (
                  <button
                    className="btn-secondary btn-sm"
                    onClick={() => {
                      setSelectedFile(null);
                      setVideoPreviewUrl(null);
                      setTrainStatus(null);
                    }}
                  >
                    Change Video
                  </button>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Right Side: Translation Annotations & Training Settings */}
        <div className="studio-form-col">
          <div className="form-group">
            <label htmlFor="english-ground-truth">
              English Ground-Truth Sentence <span className="req">*</span>
            </label>
            <input
              id="english-ground-truth"
              type="text"
              className="studio-input"
              placeholder='e.g. "Who are you? I am Nital."'
              value={englishText}
              disabled={isTraining}
              onChange={(e) => setEnglishText(e.target.value)}
            />
            <span className="field-hint">
              The exact English sentence conveyed in this sign video.
            </span>
          </div>

          <div className="form-group">
            <div className="label-with-action">
              <label htmlFor="telugu-ground-truth">
                Telugu Translation (తెలుగు)
              </label>
              <button
                type="button"
                className="btn-text-action"
                onClick={handleAutoTranslate}
                disabled={isTraining || !englishText.trim()}
                title="Auto-generate native Telugu script"
              >
                <Languages size={14} /> Auto-Generate
              </button>
            </div>
            <input
              id="telugu-ground-truth"
              type="text"
              className="studio-input telugu-font"
              placeholder='e.g. "మీరు ఎవరు? నేను నితల్."'
              value={teluguText}
              disabled={isTraining}
              onChange={(e) => setTeluguText(e.target.value)}
            />
            <span className="field-hint">
              Native script translation (optional: auto-generated if left blank).
            </span>
          </div>

          <div className="form-row-2">
            <div className="form-group">
              <label htmlFor="signer-name">Signer ID</label>
              <input
                id="signer-name"
                type="text"
                className="studio-input"
                value={signerId}
                disabled={isTraining}
                onChange={(e) => setSignerId(e.target.value)}
              />
            </div>

            <div className="form-group">
              <label htmlFor="epoch-slider">
                Training Epochs: <strong>{epochs}</strong>
              </label>
              <input
                id="epoch-slider"
                type="range"
                min={10}
                max={50}
                step={5}
                value={epochs}
                disabled={isTraining}
                onChange={(e) => setEpochs(Number(e.target.value))}
                className="studio-range"
              />
            </div>
          </div>

          {!isTraining && (
            <button
              id="btn-start-training"
              className="btn-primary btn-full-width"
              onClick={handleStartTraining}
              disabled={!selectedFile || !englishText.trim()}
            >
              <Sparkles size={18} />
              Train Model On This Video
            </button>
          )}

          {/* Active Training Progress Card */}
          {isTraining && trainStatus && (
            <div className="training-progress-card">
              <div className="train-header-row">
                <span className="train-stage-text">
                  <Loader2 size={16} className="spin-icon" />
                  {trainStatus.stage}
                </span>
                <span className="train-percent-text">
                  {Math.round(trainStatus.progress * 100)}%
                </span>
              </div>

              <div className="progress-bar-track">
                <div
                  className="progress-bar-fill progress-glow"
                  style={{ width: `${Math.round(trainStatus.progress * 100)}%` }}
                />
              </div>

              {trainStatus.current_epoch > 0 && (
                <div className="epoch-stats-grid">
                  <div className="stat-pill">
                    <span>Epoch</span>
                    <strong>
                      {trainStatus.current_epoch} / {trainStatus.total_epochs}
                    </strong>
                  </div>
                  {trainStatus.loss !== undefined && (
                    <div className="stat-pill">
                      <span>Loss</span>
                      <strong>{trainStatus.loss.toFixed(2)}</strong>
                    </div>
                  )}
                  {trainStatus.bleu !== undefined && (
                    <div className="stat-pill">
                      <span>Val BLEU-4</span>
                      <strong>{trainStatus.bleu.toFixed(1)}</strong>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Completed Test Results Card */}
          {trainStatus?.status === 'completed' && trainStatus.test_result && (
            <div className="trained-result-card" role="region" aria-label="Trained Model Verification">
              <div className="result-success-badge">
                <CheckCircle2 size={18} />
                <span>Model Trained, Verified & Deployed!</span>
              </div>

              <div className="result-test-body">
                <div className="test-label">Instant Video Test Output:</div>
                <div className="test-prediction-english">
                  "{trainStatus.test_result.english_text}"
                </div>
                <div className="test-prediction-telugu telugu-font">
                  {trainStatus.test_result.telugu_text}
                </div>

                <div className="test-meta-row">
                  <span className="confidence-pill high">
                    Confidence: {Math.round(trainStatus.test_result.confidence * 100)}%
                  </span>
                  <div className="audio-btns-group">
                    {trainStatus.test_result.audio_en && (
                      <button
                        className="btn-audio-mini"
                        onClick={() => playAudio(trainStatus.test_result?.audio_en, false)}
                        title="Speak English"
                      >
                        <Volume2 size={14} /> {isPlayingEn ? 'Speaking...' : 'EN'}
                      </button>
                    )}
                    {trainStatus.test_result.audio_te && (
                      <button
                        className="btn-audio-mini"
                        onClick={() => playAudio(trainStatus.test_result?.audio_te, true)}
                        title="వినండి"
                      >
                        <Volume2 size={14} /> {isPlayingTe ? 'వింటున్నారు...' : 'TE'}
                      </button>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {errorMessage && (
            <div className="camera-error-banner" role="alert">
              <AlertCircle size={18} />
              <span>{errorMessage}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
