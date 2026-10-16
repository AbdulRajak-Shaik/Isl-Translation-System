import { useState, useEffect, useCallback } from 'react';
import './App.css';
import { Navbar } from './components/Navbar';
import { LiveCamera } from './components/LiveCamera';
import { VideoUpload } from './components/VideoUpload';
import { TrainingStudio } from './components/TrainingStudio';
import { TranslationPanel } from './components/TranslationPanel';
import { HistoryDrawer } from './components/HistoryDrawer';
import { ModelStatusModal } from './components/ModelStatusModal';
import { SettingsModal } from './components/SettingsModal';
import type { TranslationResult, ModelStatus, AppSettings } from './types/translation';
import { fetchModelStatus, fetchTTS } from './services/api';

export function App() {
  const [activeTab, setActiveTab] = useState<'live' | 'upload' | 'train'>('live');
  const [currentTranslation, setCurrentTranslation] = useState<TranslationResult | null>(null);
  const [history, setHistory] = useState<TranslationResult[]>([]);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);

  // Modals state
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);
  const [isHistoryOpen, setIsHistoryOpen] = useState<boolean>(false);
  const [isModelStatusOpen, setIsModelStatusOpen] = useState<boolean>(false);

  // App Settings
  const [settings, setSettings] = useState<AppSettings>({
    confidenceThreshold: 0.55,
    frameSamplingRate: 25,
    autoSpeak: true,
    activeLanguage: 'both',
    highContrastMode: false,
  });

  const refreshModelStatus = useCallback(() => {
    fetchModelStatus()
      .then((status) => setModelStatus(status))
      .catch((err) => console.warn('Could not refresh model status:', err));
  }, []);

  // Fetch Model Status on Mount
  useEffect(() => {
    refreshModelStatus();
  }, [refreshModelStatus]);

  // Handle new translation emitted from LiveCamera or VideoUpload
  const handleNewTranslation = useCallback(
    (res: TranslationResult) => {
      setCurrentTranslation(res);

      // Only push non-provisional or meaningful sentences to history
      if (!res.provisional && res.englishText) {
        setHistory((prev) => [res, ...prev.slice(0, 49)]); // keep recent 50

        // Auto Speak if configured
        if (settings.autoSpeak) {
          if (settings.activeLanguage === 'en' || settings.activeLanguage === 'both') {
            fetchTTS(res.englishText, 'en')
              .then((ttsRes) => {
                if (ttsRes.audio_base64) {
                  const audio = new Audio(ttsRes.audio_base64);
                  audio.play().catch(() => {});
                }
              })
              .catch(() => {});
          }
        }
      }
    },
    [settings.autoSpeak, settings.activeLanguage]
  );

  const handleClearHistory = () => {
    setHistory([]);
  };

  const handleUpdateSettings = (newSettings: Partial<AppSettings>) => {
    setSettings((prev) => ({ ...prev, ...newSettings }));
  };

  return (
    <div className="app-root">
      {/* Navigation Header */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onOpenHistory={() => setIsHistoryOpen(true)}
        onOpenModelStatus={() => setIsModelStatusOpen(true)}
        modelStatus={modelStatus}
      />

      {/* Main Workspace Grid */}
      <main className="main-content" id="main-content-area">
        {/* Hidden Accessible H1 for SEO Standard */}
        <h1 style={{ position: 'absolute', width: 1, height: 1, padding: 0, margin: -1, overflow: 'hidden', clip: 'rect(0, 0, 0, 0)', border: 0 }}>
          Indian Sign Language Continuous Recognition & Translation System
        </h1>

        <div className="workspace-grid">
          {/* Left Column: Visual Capture Mode */}
          <section className="viewport-column" aria-label="Sign Language Video Input">
            {activeTab === 'live' && (
              <LiveCamera
                onNewTranslation={handleNewTranslation}
                confidenceThreshold={settings.confidenceThreshold}
              />
            )}
            {activeTab === 'upload' && (
              <VideoUpload
                onNewTranslation={handleNewTranslation}
                confidenceThreshold={settings.confidenceThreshold}
                onOpenTrainStudio={() => setActiveTab('train')}
              />
            )}
            {activeTab === 'train' && (
              <TrainingStudio
                onNewTranslation={handleNewTranslation}
                onModelRetrained={refreshModelStatus}
              />
            )}
          </section>

          {/* Right Column: Translation & Audio Speech Panels */}
          <section className="translation-column" aria-label="Bilingual Translation & Speech">
            <TranslationPanel
              currentTranslation={currentTranslation}
            />
          </section>
        </div>
      </main>

      {/* Modals & Drawers */}
      <HistoryDrawer
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
        history={history}
        onClearHistory={handleClearHistory}
      />

      <ModelStatusModal
        isOpen={isModelStatusOpen}
        onClose={() => setIsModelStatusOpen(false)}
        status={modelStatus}
      />

      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        onUpdateSettings={handleUpdateSettings}
      />
    </div>
  );
}

export default App;
