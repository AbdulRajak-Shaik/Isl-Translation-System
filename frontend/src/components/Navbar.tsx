import { Video, Upload, Settings, History, Sparkles, Activity, Cpu } from 'lucide-react';
import type { ModelStatus } from '../types/translation';

interface NavbarProps {
  activeTab: 'live' | 'upload' | 'train';
  setActiveTab: (tab: 'live' | 'upload' | 'train') => void;
  onOpenSettings: () => void;
  onOpenHistory: () => void;
  onOpenModelStatus: () => void;
  modelStatus: ModelStatus | null;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  onOpenSettings,
  onOpenHistory,
  onOpenModelStatus,
  modelStatus,
}) => {
  return (
    <header className="navbar-container">
      <div className="navbar-left">
        <div className="brand-logo">
          <Sparkles className="brand-icon" />
          <div className="brand-titles">
            <span className="brand-name">ISL TRANSLATOR</span>
            <span className="brand-tagline">Continuous Sign Language AI</span>
          </div>
        </div>

        <nav className="mode-nav" aria-label="Mode Selection">
          <button
            id="btn-mode-live"
            className={`nav-tab ${activeTab === 'live' ? 'active' : ''}`}
            onClick={() => setActiveTab('live')}
          >
            <Video size={18} />
            <span>Live Camera</span>
          </button>
          <button
            id="btn-mode-upload"
            className={`nav-tab ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            <Upload size={18} />
            <span>Upload Video</span>
          </button>
          <button
            id="btn-mode-train"
            className={`nav-tab ${activeTab === 'train' ? 'active' : ''}`}
            onClick={() => setActiveTab('train')}
          >
            <Cpu size={18} />
            <span>Train Studio</span>
          </button>
        </nav>
      </div>

      <div className="navbar-right">
        {modelStatus && (
          <button
            id="btn-model-status"
            className="status-pill"
            onClick={onOpenModelStatus}
            title="Inspect Model Status"
          >
            <Activity size={14} className="status-icon-live" />
            <span className="status-label">
              {modelStatus.is_trained_checkpoint ? 'Trained Model' : 'Baseline'}: {modelStatus.device.toUpperCase()}
            </span>
          </button>
        )}

        <button
          id="btn-history-toggle"
          className="icon-btn"
          onClick={onOpenHistory}
          title="Session Translation History"
          aria-label="History"
        >
          <History size={20} />
        </button>

        <button
          id="btn-settings-toggle"
          className="icon-btn"
          onClick={onOpenSettings}
          title="System Settings"
          aria-label="Settings"
        >
          <Settings size={20} />
        </button>
      </div>
    </header>
  );
};
