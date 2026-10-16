import { X, Cpu, CheckCircle2, Shield } from 'lucide-react';
import type { ModelStatus } from '../types/translation';

interface ModelStatusModalProps {
  isOpen: boolean;
  onClose: () => void;
  status: ModelStatus | null;
}

export const ModelStatusModal: React.FC<ModelStatusModalProps> = ({
  isOpen,
  onClose,
  status,
}) => {
  if (!isOpen || !status) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="status-modal-card glass-panel" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-group">
            <Cpu size={22} className="modal-header-icon" />
            <h2>Active Model Architecture</h2>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close">
            <X size={20} />
          </button>
        </div>

        <div className="modal-body">
          <div className="status-grid">
            <div className="status-spec-item">
              <span className="spec-label">Model Architecture</span>
              <span className="spec-value highlight">{status.model_name}</span>
            </div>

            <div className="status-spec-item">
              <span className="spec-label">Trainable Parameters</span>
              <span className="spec-value">{status.parameter_count.toLocaleString()} weights</span>
            </div>

            <div className="status-spec-item">
              <span className="spec-label">Active Inference Device</span>
              <span className="spec-value">{status.device.toUpperCase()}</span>
            </div>

            <div className="status-spec-item">
              <span className="spec-label">Vocabulary Size</span>
              <span className="spec-value">{status.vocab_size} tokens</span>
            </div>

            <div className="status-spec-item">
              <span className="spec-label">Checkpoint Status</span>
              <span className="spec-value status-badge-ok">
                <CheckCircle2 size={15} />
                {status.is_trained_checkpoint ? 'Verified Checkpoint' : 'Baseline Weights'}
              </span>
            </div>

            <div className="status-spec-item">
              <span className="spec-label">Calibrated Threshold</span>
              <span className="spec-value">{(status.confidence_threshold * 100).toFixed(0)}%</span>
            </div>
          </div>

          <div className="architecture-notes">
            <h4>
              <Shield size={16} /> Architectural Guarantees
            </h4>
            <ul>
              <li><strong>Continuous Translation:</strong> Sequence-to-sequence autoregressive transformer with temporal cross-attention.</li>
              <li><strong>Signer Invariant:</strong> 3D Holistic skeletal landmarks normalized by shoulder width and nasal center.</li>
              <li><strong>Native Telugu NMT:</strong> Aligned with AI4Bharat IndicTrans2 (`tel_Telu`) unicode script standards.</li>
              <li><strong>Zero Mock Predictions:</strong> Direct execution from PyTorch neural network checkpoint.</li>
            </ul>
          </div>
        </div>

        <div className="modal-footer">
          <button className="btn-primary" onClick={onClose}>
            Close Inspector
          </button>
        </div>
      </div>
    </div>
  );
};
