import { X, Sliders } from 'lucide-react';
import type { AppSettings } from '../types/translation';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  settings: AppSettings;
  onUpdateSettings: (newSettings: Partial<AppSettings>) => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  settings,
  onUpdateSettings,
}) => {
  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="settings-modal-card glass-panel" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-group">
            <Sliders size={22} className="modal-header-icon" />
            <h2>System Settings</h2>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close">
            <X size={20} />
          </button>
        </div>

        <div className="modal-body settings-body">
          {/* Confidence Threshold */}
          <div className="setting-row">
            <div className="setting-info">
              <label className="setting-label">Confidence Acceptance Threshold</label>
              <p className="setting-desc">
                Predictions below this calibrated confidence level will be flagged as uncertain.
              </p>
            </div>
            <div className="slider-group">
              <input
                type="range"
                min="0.30"
                max="0.85"
                step="0.05"
                value={settings.confidenceThreshold}
                onChange={(e) =>
                  onUpdateSettings({ confidenceThreshold: parseFloat(e.target.value) })
                }
                className="range-slider"
              />
              <span className="slider-value">
                {Math.round(settings.confidenceThreshold * 100)}%
              </span>
            </div>
          </div>

          {/* Auto Speak Toggle */}
          <div className="setting-row">
            <div className="setting-info">
              <label className="setting-label">Automatic Speech Synthesis</label>
              <p className="setting-desc">
                Automatically speak finalized sentences via neural text-to-speech.
              </p>
            </div>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={settings.autoSpeak}
                onChange={(e) => onUpdateSettings({ autoSpeak: e.target.checked })}
              />
              <span className="toggle-slider" />
            </label>
          </div>

          {/* Active Audio Channel */}
          <div className="setting-row">
            <div className="setting-info">
              <label className="setting-label">Default Audio Language</label>
              <p className="setting-desc">Select which speech synthesis channel to prioritize.</p>
            </div>
            <select
              className="select-dropdown"
              value={settings.activeLanguage}
              onChange={(e) =>
                onUpdateSettings({ activeLanguage: e.target.value as 'both' | 'en' | 'te' })
              }
            >
              <option value="both">Both English & Telugu</option>
              <option value="en">English Only</option>
              <option value="te">Telugu Only (తెలుగు)</option>
            </select>
          </div>
        </div>

        <div className="modal-footer">
          <button className="btn-primary" onClick={onClose}>
            Save Preferences
          </button>
        </div>
      </div>
    </div>
  );
};
