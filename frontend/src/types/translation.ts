export interface TranslationResult {
  id: string;
  englishText: string;
  teluguText: string;
  confidence: number;
  isUncertain: boolean;
  provisional: boolean;
  meanEntropy?: number;
  scriptValid?: boolean;
  audioEn?: string;
  audioTe?: string;
  timestamp: number;
  source: 'live' | 'upload';
}

export interface ModelStatus {
  model_name: string;
  parameter_count: number;
  checkpoint_path: string;
  vocab_size: number;
  device: string;
  confidence_threshold: number;
  is_trained_checkpoint: boolean;
}

export interface AppSettings {
  confidenceThreshold: number;
  frameSamplingRate: number;
  autoSpeak: boolean;
  activeLanguage: 'both' | 'en' | 'te';
  highContrastMode: boolean;
}

export interface TrainJobStatus {
  job_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  stage: string;
  current_epoch: number;
  total_epochs: number;
  loss?: number;
  bleu?: number;
  english_ground_truth?: string;
  telugu_ground_truth?: string;
  test_result?: {
    english_text: string;
    telugu_text: string;
    confidence: number;
    is_uncertain: boolean;
    audio_en?: string;
    audio_te?: string;
  };
  error?: string;
}

