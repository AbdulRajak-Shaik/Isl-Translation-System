import { X, Trash2, Download, History as HistoryIcon, Clock } from 'lucide-react';
import type { TranslationResult } from '../types/translation';

interface HistoryDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  history: TranslationResult[];
  onClearHistory: () => void;
}

export const HistoryDrawer: React.FC<HistoryDrawerProps> = ({
  isOpen,
  onClose,
  history,
  onClearHistory,
}) => {
  if (!isOpen) return null;

  const exportHistoryJson = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(history, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `isl_translation_history_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <aside className="drawer-panel glass-panel" onClick={(e) => e.stopPropagation()}>
        <div className="drawer-header">
          <div className="drawer-title-group">
            <HistoryIcon size={20} />
            <h2>Translation History</h2>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close">
            <X size={20} />
          </button>
        </div>

        <div className="privacy-notice">
          <span>🔒 Privacy Protected: No raw video data is stored. Only translated text is retained for your session.</span>
        </div>

        <div className="drawer-body">
          {history.length === 0 ? (
            <div className="drawer-empty-state">
              <Clock size={32} />
              <p>No translations recorded in this session yet.</p>
            </div>
          ) : (
            <div className="history-list">
              {history.map((item) => (
                <div key={item.id} className="history-item glass-panel">
                  <div className="history-item-meta">
                    <span className="history-source-pill">{item.source.toUpperCase()}</span>
                    <span className="history-time">
                      {new Date(item.timestamp).toLocaleTimeString()}
                    </span>
                    <span className="history-confidence">
                      {Math.round(item.confidence * 100)}% conf
                    </span>
                  </div>
                  <div className="history-english">{item.englishText}</div>
                  <div className="history-telugu font-telugu">{item.teluguText}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        {history.length > 0 && (
          <div className="drawer-footer">
            <button className="btn-secondary" onClick={exportHistoryJson}>
              <Download size={16} /> Export JSON
            </button>
            <button className="btn-danger" onClick={onClearHistory}>
              <Trash2 size={16} /> Clear History
            </button>
          </div>
        )}
      </aside>
    </div>
  );
};
