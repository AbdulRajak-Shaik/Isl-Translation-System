import { useState, useRef } from 'react';
import { Volume2, VolumeX, Copy, Check, AlertTriangle, ShieldCheck, ThumbsUp, ThumbsDown } from 'lucide-react';
import type { TranslationResult } from '../types/translation';
import { fetchTTS, submitFeedback } from '../services/api';

interface TranslationPanelProps {
  currentTranslation: TranslationResult | null;
}

export const TranslationPanel: React.FC<TranslationPanelProps> = ({
  currentTranslation,
}) => {
  const [isPlayingEn, setIsPlayingEn] = useState<boolean>(false);
  const [isPlayingTe, setIsPlayingTe] = useState<boolean>(false);
  const [copiedEn, setCopiedEn] = useState<boolean>(false);
  const [copiedTe, setCopiedTe] = useState<boolean>(false);
  const [feedbackSent, setFeedbackSent] = useState<boolean>(false);

  const audioEnRef = useRef<HTMLAudioElement | null>(null);
  const audioTeRef = useRef<HTMLAudioElement | null>(null);

  // Play English Speech
  const playEnglishAudio = async () => {
    if (!currentTranslation?.englishText) return;

    if (currentTranslation.audioEn) {
      playAudioUri(currentTranslation.audioEn, 'en');
      return;
    }

    try {
      setIsPlayingEn(true);
      const res = await fetchTTS(currentTranslation.englishText, 'en');
      if (res.audio_base64) {
        playAudioUri(res.audio_base64, 'en');
      }
    } catch (err) {
      console.error('Failed to generate English speech:', err);
      // Client-side fallback to Web Speech API
      if ('speechSynthesis' in window) {
        const utter = new SpeechSynthesisUtterance(currentTranslation.englishText);
        utter.lang = 'en-IN';
        utter.onend = () => setIsPlayingEn(false);
        window.speechSynthesis.speak(utter);
      } else {
        setIsPlayingEn(false);
      }
    }
  };

  // Play Telugu Speech
  const playTeluguAudio = async () => {
    if (!currentTranslation?.teluguText) return;

    if (currentTranslation.audioTe) {
      playAudioUri(currentTranslation.audioTe, 'te');
      return;
    }

    try {
      setIsPlayingTe(true);
      const res = await fetchTTS(currentTranslation.teluguText, 'te');
      if (res.audio_base64) {
        playAudioUri(res.audio_base64, 'te');
      }
    } catch (err) {
      console.error('Failed to generate Telugu speech:', err);
      // Client-side fallback to Web Speech API
      if ('speechSynthesis' in window) {
        const utter = new SpeechSynthesisUtterance(currentTranslation.teluguText);
        utter.lang = 'te-IN';
        utter.onend = () => setIsPlayingTe(false);
        window.speechSynthesis.speak(utter);
      } else {
        setIsPlayingTe(false);
      }
    }
  };

  const playAudioUri = (uri: string, lang: 'en' | 'te') => {
    const audio = new Audio(uri);
    if (lang === 'en') {
      audioEnRef.current = audio;
      setIsPlayingEn(true);
      audio.onended = () => setIsPlayingEn(false);
      audio.play().catch(() => setIsPlayingEn(false));
    } else {
      audioTeRef.current = audio;
      setIsPlayingTe(true);
      audio.onended = () => setIsPlayingTe(false);
      audio.play().catch(() => setIsPlayingTe(false));
    }
  };

  const copyText = (text: string, type: 'en' | 'te') => {
    navigator.clipboard.writeText(text);
    if (type === 'en') {
      setCopiedEn(true);
      setTimeout(() => setCopiedEn(false), 2000);
    } else {
      setCopiedTe(true);
      setTimeout(() => setCopiedTe(false), 2000);
    }
  };

  const handleRating = async (rating: number) => {
    if (!currentTranslation || feedbackSent) return;
    try {
      await submitFeedback({
        english_predicted: currentTranslation.englishText,
        telugu_predicted: currentTranslation.teluguText,
        rating,
      });
      setFeedbackSent(true);
    } catch (e) {
      console.error('Feedback error:', e);
    }
  };

  if (!currentTranslation) {
    return (
      <div className="translation-empty-card glass-panel" id="translation-empty-view">
        <div className="empty-content">
          <div className="empty-icon-circle">
            <ShieldCheck size={32} />
          </div>
          <h3>Continuous Translation Ready</h3>
          <p>
            Start signing in front of the camera or upload a video. Continuous sentences will appear
            here in English and Telugu with synchronized speech synthesis.
          </p>
        </div>
      </div>
    );
  }

  const confidencePercent = Math.round(currentTranslation.confidence * 100);

  return (
    <div className="translation-section" id="translation-display-section">
      {/* Confidence and Status Header */}
      <div className="translation-status-bar glass-panel">
        <div className="status-meta">
          <span className="meta-badge">
            {currentTranslation.provisional ? (
              <span className="badge badge-warning">Provisional Stream</span>
            ) : (
              <span className="badge badge-success">Finalized Translation</span>
            )}
          </span>

          <div className="confidence-meter">
            <span className="confidence-label">Confidence:</span>
            <span className="confidence-value">{confidencePercent}%</span>
            <div className="confidence-track">
              <div
                className={`confidence-fill ${
                  currentTranslation.isUncertain ? 'uncertain' : 'high'
                }`}
                style={{ width: `${confidencePercent}%` }}
              />
            </div>
          </div>
        </div>

        {currentTranslation.isUncertain && (
          <div className="uncertainty-alert" role="alert">
            <AlertTriangle size={15} />
            <span>Low Confidence: Signer clarification or repetition recommended</span>
          </div>
        )}
      </div>

      {/* Side-by-Side Dual Language Translation Cards */}
      <div className="cards-grid">
        {/* English Translation Card */}
        <div className="translation-card glass-panel" id="english-translation-panel">
          <div className="card-top-row">
            <div className="card-lang-tag">
              <span className="lang-code">EN</span>
              <span className="lang-title">English Output</span>
            </div>
            <div className="card-actions">
              <button
                className="action-icon-btn"
                onClick={() => copyText(currentTranslation.englishText, 'en')}
                title="Copy English text"
                aria-label="Copy English"
              >
                {copiedEn ? <Check size={16} /> : <Copy size={16} />}
              </button>
              <button
                className={`audio-btn ${isPlayingEn ? 'playing' : ''}`}
                onClick={playEnglishAudio}
                title="Listen to English speech"
                aria-label="Play English Audio"
              >
                {isPlayingEn ? <VolumeX size={17} /> : <Volume2 size={17} />}
                <span>{isPlayingEn ? 'Speaking' : 'Speak'}</span>
              </button>
            </div>
          </div>

          <div className="translation-body">
            <p className="translation-text english-text">
              {currentTranslation.englishText || 'Waiting for signing...'}
            </p>
          </div>
        </div>

        {/* Telugu Translation Card */}
        <div className="translation-card glass-panel" id="telugu-translation-panel">
          <div className="card-top-row">
            <div className="card-lang-tag telugu-tag">
              <span className="lang-code">TE</span>
              <span className="lang-title">తెలుగు అనువాదం</span>
            </div>
            <div className="card-actions">
              <button
                className="action-icon-btn"
                onClick={() => copyText(currentTranslation.teluguText, 'te')}
                title="Copy Telugu text"
                aria-label="Copy Telugu"
              >
                {copiedTe ? <Check size={16} /> : <Copy size={16} />}
              </button>
              <button
                className={`audio-btn telugu-audio-btn ${isPlayingTe ? 'playing' : ''}`}
                onClick={playTeluguAudio}
                title="Listen to Telugu speech"
                aria-label="Play Telugu Audio"
              >
                {isPlayingTe ? <VolumeX size={17} /> : <Volume2 size={17} />}
                <span>{isPlayingTe ? 'వినబడుతోంది' : 'వినండి'}</span>
              </button>
            </div>
          </div>

          <div className="translation-body">
            <p className="translation-text telugu-text font-telugu">
              {currentTranslation.teluguText || 'సంకేతాల కోసం వేచి చూస్తోంది...'}
            </p>
          </div>
        </div>
      </div>

      {/* Human Review and Adequacy Feedback Row */}
      <div className="feedback-row glass-panel">
        <span className="feedback-prompt">Rate this translation accuracy:</span>
        <div className="feedback-buttons">
          <button
            className={`feedback-btn ${feedbackSent ? 'disabled' : ''}`}
            onClick={() => handleRating(5)}
            disabled={feedbackSent}
            title="Accurate Meaning"
          >
            <ThumbsUp size={15} />
            <span>Accurate</span>
          </button>
          <button
            className={`feedback-btn ${feedbackSent ? 'disabled' : ''}`}
            onClick={() => handleRating(1)}
            disabled={feedbackSent}
            title="Inaccurate Meaning"
          >
            <ThumbsDown size={15} />
            <span>Needs Correction</span>
          </button>
        </div>
        {feedbackSent && <span className="feedback-thanks">✓ Verified feedback submitted</span>}
      </div>
    </div>
  );
};
