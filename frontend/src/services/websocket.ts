export interface LiveWSCallbacks {
  onTranslation: (data: {
    english_text: string;
    telugu_text: string;
    confidence: number;
    is_uncertain: boolean;
    provisional: boolean;
    is_new_utterance: boolean;
    timestamp: number;
  }) => void;
  onStatusChange: (status: 'connected' | 'connecting' | 'disconnected' | 'error') => void;
  onError?: (err: Event | string) => void;
}

export class LiveWebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private callbacks: LiveWSCallbacks;
  private isPaused: boolean = false;

  constructor(url: string, callbacks: LiveWSCallbacks) {
    this.url = url;
    this.callbacks = callbacks;
  }

  public connect() {
    this.callbacks.onStatusChange('connecting');
    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.callbacks.onStatusChange('connected');
      };

      this.ws.onmessage = (event) => {
        if (this.isPaused) return;
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'translation_update') {
            this.callbacks.onTranslation(data);
          }
        } catch (e) {
          console.error('[WS Parse Error]', e);
        }
      };

      this.ws.onerror = (e) => {
        this.callbacks.onStatusChange('error');
        if (this.callbacks.onError) this.callbacks.onError(e);
      };

      this.ws.onclose = () => {
        this.callbacks.onStatusChange('disconnected');
      };
    } catch (e) {
      this.callbacks.onStatusChange('error');
    }
  }

  public sendFrame(base64Frame: string) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN && !this.isPaused) {
      this.ws.send(JSON.stringify({ type: 'frame', frame: base64Frame }));
    }
  }

  public pause() {
    this.isPaused = true;
  }

  public resume() {
    this.isPaused = false;
  }

  public reset() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'reset' }));
    }
  }

  public disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}
