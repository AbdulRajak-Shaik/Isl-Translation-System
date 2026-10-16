import type { ModelStatus } from '../types/translation';

const apiHost = import.meta.env.VITE_API_HOST;
const isViteDev = typeof window !== 'undefined' && window.location.port === '5173';
const API_BASE_URL =
  import.meta.env.VITE_API_URL ||
  (apiHost ? `https://${apiHost}/api/v1` : (isViteDev ? 'http://localhost:8000/api/v1' : '/api/v1'));

export async function fetchHealth() {
  const res = await fetch(`${API_BASE_URL}/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}

export async function fetchModelStatus(): Promise<ModelStatus> {
  const res = await fetch(`${API_BASE_URL}/model/status`);
  if (!res.ok) throw new Error('Failed to fetch model status');
  return res.json();
}

export async function translateText(englishText: string): Promise<{ telugu_text: string }> {
  const res = await fetch(`${API_BASE_URL}/translate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ english_text: englishText, is_uncertain: false }),
  });
  if (!res.ok) throw new Error('Translation failed');
  return res.json();
}

export async function uploadVideo(
  file: File,
  onProgress?: (percent: number) => void
): Promise<{ job_id: string; message: string }> {
  const formData = new FormData();
  formData.append('file', file);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE_URL}/video/upload`);

    if (xhr.upload && onProgress) {
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const percent = Math.round((event.loaded / event.total) * 100);
          onProgress(percent);
        }
      };
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        try {
          const err = JSON.parse(xhr.responseText);
          reject(new Error(err.detail || 'Upload failed'));
        } catch {
          reject(new Error(`Server error: ${xhr.status}`));
        }
      }
    };

    xhr.onerror = () => reject(new Error('Network error during upload'));
    xhr.send(formData);
  });
}

export async function pollJobStatus(jobId: string) {
  const res = await fetch(`${API_BASE_URL}/video/job/${jobId}`);
  if (!res.ok) {
    let msg = `Job status check failed (${res.status})`;
    try {
      const data = await res.json();
      if (data.detail) msg = data.detail;
    } catch {
      // ignore JSON parse error
    }
    throw new Error(msg);
  }
  return res.json();
}

export async function fetchTTS(text: string, lang: 'en' | 'te') {
  const res = await fetch(`${API_BASE_URL}/tts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, lang }),
  });
  if (!res.ok) throw new Error('TTS generation failed');
  return res.json();
}

export async function submitFeedback(payload: {
  english_predicted: string;
  telugu_predicted: string;
  correct_english?: string;
  rating: number;
  comments?: string;
}) {
  const res = await fetch(`${API_BASE_URL}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return res.json();
}

export async function uploadAndTrainVideo(
  file: File,
  englishText: string,
  teluguText?: string,
  epochs: number = 20,
  onProgress?: (percent: number) => void
): Promise<{ job_id: string; message: string }> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('english_text', englishText);
  if (teluguText) formData.append('telugu_text', teluguText);
  formData.append('epochs', epochs.toString());
  formData.append('batch_size', '8');

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE_URL}/video/train`);

    if (xhr.upload && onProgress) {
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const percent = Math.round((event.loaded / event.total) * 100);
          onProgress(percent);
        }
      };
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        try {
          const err = JSON.parse(xhr.responseText);
          reject(new Error(err.detail || 'Training submission failed'));
        } catch {
          reject(new Error(`Server error: ${xhr.status}`));
        }
      }
    };

    xhr.onerror = () => reject(new Error('Network error during training submission'));
    xhr.send(formData);
  });
}

export async function pollTrainJobStatus(jobId: string) {
  const res = await fetch(`${API_BASE_URL}/video/train/${jobId}`);
  if (!res.ok) {
    let msg = `Training job status check failed (${res.status})`;
    try {
      const data = await res.json();
      if (data.detail) msg = data.detail;
    } catch {
      // ignore JSON parse error
    }
    throw new Error(msg);
  }
  return res.json();
}

