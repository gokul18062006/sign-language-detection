import { useEffect, useMemo, useRef, useState } from 'react';

type BackendStatus = {
  sign_active: boolean;
  voice_active: boolean;
  camera_ready: boolean;
  detected_word: string | null;
  detected_label: string | null;
  gesture_label: string;
  voice_transcript: string | null;
  matched_word: string | null;
  matched_label: string | null;
  last_event: string;
  last_detection_time: number | null;
  logs: string[];
  supported_words: string[];
};

type VoiceMatchResponse = {
  transcript: string;
  normalized: string;
  matched_word: string | null;
  matched_label: string | null;
  description: string | null;
  supported_words: string[];
};

type SignTheme = {
  title: string;
  subtitle: string;
  accent: string;
  glow: string;
  symbol: string;
};

declare global {
  interface Window {
    speechSynthesis: SpeechSynthesis;
  }
}

const DEFAULT_API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000';

const SIGN_THEMES: Record<string, SignTheme> = {
  hello: {
    title: 'Hello',
    subtitle: 'Open hand greeting',
    accent: '#22d3ee',
    glow: 'rgba(34, 211, 238, 0.22)',
    symbol: '01',
  },
  help: {
    title: 'Help',
    subtitle: 'Request assistance',
    accent: '#a78bfa',
    glow: 'rgba(167, 139, 250, 0.22)',
    symbol: '02',
  },
  yes: {
    title: 'Yes',
    subtitle: 'Positive confirmation',
    accent: '#34d399',
    glow: 'rgba(52, 211, 153, 0.22)',
    symbol: '03',
  },
  no: {
    title: 'No',
    subtitle: 'Negative response',
    accent: '#fb7185',
    glow: 'rgba(251, 113, 133, 0.24)',
    symbol: '04',
  },
  'thank you': {
    title: 'Thank You',
    subtitle: 'Polite gratitude',
    accent: '#f59e0b',
    glow: 'rgba(245, 158, 11, 0.22)',
    symbol: '05',
  },
  stop: {
    title: 'Stop',
    subtitle: 'Firm pause signal',
    accent: '#f97316',
    glow: 'rgba(249, 115, 22, 0.22)',
    symbol: '06',
  },
};

const HUMAN_SIGN_IMAGE_MAP: Partial<
  Record<
    string,
    {
      src: string;
      alt: string;
      caption: string;
    }
  >
> = {
  hello: {
    src: '/signs/hello-photo.png?v=2',
    alt: 'Open human hand for hello sign',
    caption: 'Photo hello hand sign',
  },
  help: {
    src: '/signs/help-human-hand-v2.svg',
    alt: 'Helping gesture using human hand form',
    caption: 'Human-form help hand sign',
  },
  yes: {
    src: '/signs/yes-photo.png?v=2',
    alt: 'Affirmative yes gesture using human hand form',
    caption: 'Photo yes hand sign',
  },
  no: {
    src: '/signs/no-photo.png?v=2',
    alt: 'Negative no gesture using human hand form',
    caption: 'Photo no hand sign',
  },
  'thank you': {
    src: '/signs/thank-you-human-hand.svg',
    alt: 'Thank you gesture using human hand form',
    caption: 'Human-form thank you hand sign',
  },
  stop: {
    src: '/signs/stop-photo.png?v=2',
    alt: 'Stop gesture using human hand form',
    caption: 'Photo stop hand sign',
  },
};

function apiUrl(baseUrl: string, path: string): string {
  return `${baseUrl.replace(/\/$/, '')}${path}`;
}

function formatTimestamp(value: number | null): string {
  if (!value) {
    return 'waiting';
  }
  return new Intl.DateTimeFormat('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }).format(new Date(value * 1000));
}

function SignCard({ word }: { word: string }) {
  const theme = SIGN_THEMES[word] ?? {
    title: word,
    subtitle: 'Recognized phrase',
    accent: '#7dd3fc',
    glow: 'rgba(125, 211, 252, 0.18)',
    symbol: '00',
  };

  return (
    <div className="sign-card" style={{ '--accent': theme.accent, '--glow': theme.glow } as React.CSSProperties}>
      <div className="sign-card__symbol">{theme.symbol}</div>
      <div className="sign-card__body">
        <div className="sign-card__title">{theme.title}</div>
        <div className="sign-card__subtitle">{theme.subtitle}</div>
      </div>
      <div className="sign-card__grid" aria-hidden="true">
        <span />
        <span />
        <span />
        <span />
      </div>
    </div>
  );
}

function SignImageCard({
  word,
  image,
}: {
  word: string;
  image: { src: string; alt: string; caption: string };
}) {
  const theme = SIGN_THEMES[word] ?? {
    title: word,
    subtitle: 'Recognized phrase',
    accent: '#7dd3fc',
    glow: 'rgba(125, 211, 252, 0.18)',
    symbol: '00',
  };

  return (
    <div className="sign-image-card" style={{ '--accent': theme.accent, '--glow': theme.glow } as React.CSSProperties}>
      <div className="sign-image-card__meta">
        <div className="sign-image-card__title">{theme.title}</div>
        <div className="sign-image-card__caption">{image.caption}</div>
      </div>
      <img src={image.src} alt={image.alt} className="sign-image-card__art" />
    </div>
  );
}

function App() {
  const apiBase = DEFAULT_API_BASE;
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const shouldKeepListeningRef = useRef(false);
  const spokenWordRef = useRef<string | null>(null);
  const speechRecognitionSupported = Boolean(window.SpeechRecognition ?? window.webkitSpeechRecognition);
  const [status, setStatus] = useState<BackendStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [listening, setListening] = useState(false);
  const [voiceResponse, setVoiceResponse] = useState<VoiceMatchResponse | null>(null);
  const [localNotice, setLocalNotice] = useState('Ready to connect');

  const logs = useMemo(() => status?.logs ?? [], [status?.logs]);
  const currentWord = status?.detected_word ?? null;
  const signFeedUrl = apiUrl(apiBase, '/api/video-feed');

  useEffect(() => {
    let active = true;

    const fetchStatus = async () => {
      try {
        const response = await fetch(apiUrl(apiBase, '/api/status'));
        if (!response.ok) {
          throw new Error(`Status request failed with ${response.status}`);
        }
        const data = (await response.json()) as BackendStatus;
        if (active) {
          setStatus(data);
          setError(null);
          setLoading(false);
        }
      } catch (fetchError) {
        if (active) {
          setError(fetchError instanceof Error ? fetchError.message : 'Unable to reach backend');
          setLoading(false);
        }
      }
    };

    fetchStatus();
    const interval = window.setInterval(fetchStatus, 900);

    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [apiBase]);

  useEffect(() => {
    if (!status?.sign_active || !currentWord || spokenWordRef.current === currentWord) {
      return;
    }

    spokenWordRef.current = currentWord;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(currentWord);
    utterance.rate = 0.95;
    utterance.pitch = 1;
    utterance.lang = 'en-US';
    window.speechSynthesis.speak(utterance);
  }, [currentWord, status?.sign_active]);

  useEffect(() => {
    return () => {
      recognitionRef.current?.abort();
      window.speechSynthesis.cancel();
    };
  }, []);

  const refreshStatusFromResponse = async (url: string) => {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Request failed with ${response.status}`);
    }
    const data = (await response.json()) as BackendStatus;
    setStatus(data);
    return data;
  };

  const callSimpleEndpoint = async (path: string) => {
    const data = await refreshStatusFromResponse(apiUrl(apiBase, path));
    setLocalNotice(data.last_event);
    return data;
  };

  const startSign = async () => {
    try {
      setError(null);
      spokenWordRef.current = null;
      await callSimpleEndpoint('/api/start-sign');
    } catch (startError) {
      setError(startError instanceof Error ? startError.message : 'Unable to start sign detection');
    }
  };

  const stopSign = async () => {
    try {
      await callSimpleEndpoint('/api/stop-sign');
      window.speechSynthesis.cancel();
      spokenWordRef.current = null;
    } catch (stopError) {
      setError(stopError instanceof Error ? stopError.message : 'Unable to stop sign detection');
    }
  };

  const startListening = async () => {
    try {
      const Recognition = window.SpeechRecognition ?? window.webkitSpeechRecognition;
      if (!Recognition) {
        throw new Error('Speech recognition is not available in this browser');
      }

      setError(null);
      await callSimpleEndpoint('/api/start-voice');
      shouldKeepListeningRef.current = true;

      const recognition = recognitionRef.current ?? new Recognition();
      recognitionRef.current = recognition;
      recognition.lang = 'en-US';
      recognition.continuous = false;
      recognition.interimResults = false;

      recognition.onresult = async event => {
        const transcript = event.results[event.results.length - 1][0].transcript.trim();
        setLocalNotice(`Recognized: ${transcript}`);
        try {
          const response = await fetch(apiUrl(apiBase, '/api/voice-text'), {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ transcript }),
          });
          if (!response.ok) {
            throw new Error(`Voice translation failed with ${response.status}`);
          }
          const data = (await response.json()) as VoiceMatchResponse;
          setVoiceResponse(data);
          await refreshStatusFromResponse(apiUrl(apiBase, '/api/status'));
          setLocalNotice(data.matched_word ? `Matched ${data.matched_word}` : 'No supported word matched');
        } catch (voiceError) {
          setError(voiceError instanceof Error ? voiceError.message : 'Voice translation failed');
        }
      };

      recognition.onerror = event => {
        setError(event.error ?? 'Speech recognition error');
        shouldKeepListeningRef.current = false;
        setListening(false);
      };

      recognition.onend = () => {
        if (shouldKeepListeningRef.current) {
          try {
            recognition.start();
          } catch {
            shouldKeepListeningRef.current = false;
            setListening(false);
          }
          return;
        }
        setListening(false);
      };

      setListening(true);
      recognition.start();
    } catch (startError) {
      setError(startError instanceof Error ? startError.message : 'Unable to start voice recognition');
      shouldKeepListeningRef.current = false;
      setListening(false);
    }
  };

  const stopListening = async () => {
    shouldKeepListeningRef.current = false;
    setListening(false);
    recognitionRef.current?.stop();
    try {
      await callSimpleEndpoint('/api/stop-voice');
    } catch (stopError) {
      setError(stopError instanceof Error ? stopError.message : 'Unable to stop voice recognition');
    }
  };

  const supportedWords = status?.supported_words ?? Object.keys(SIGN_THEMES);
  const matchedWord = voiceResponse?.matched_word ?? status?.matched_word ?? null;
  const matchedImage = matchedWord ? HUMAN_SIGN_IMAGE_MAP[matchedWord] ?? null : null;

  return (
    <main className="app-shell">
      <header className="hero">
        <div>
          <div className="eyebrow">React + TypeScript frontend · FastAPI backend</div>
          <h1>Two-way sign language bridge</h1>
          <p className="hero-copy">
            Stream webcam gestures into live speech, and turn spoken words into sign cards without
            relying on a heavy native desktop stack.
          </p>
        </div>
        <div className="hero-stats">
          <div className="stat-card">
            <span>Status</span>
            <strong>{loading ? 'Loading' : error ? 'Disconnected' : 'Online'}</strong>
          </div>
          <div className="stat-card">
            <span>Backend event</span>
            <strong>{status?.last_event ?? 'Waiting'}</strong>
          </div>
        </div>
      </header>

      <section className="pills">
        <div className={`pill ${status?.sign_active ? 'pill--active' : ''}`}>Sign {status?.sign_active ? 'active' : 'idle'}</div>
        <div className={`pill ${listening || status?.voice_active ? 'pill--active' : ''}`}>Voice {listening || status?.voice_active ? 'listening' : 'idle'}</div>
        <div className="pill">Camera {status?.camera_ready ? 'ready' : 'fallback stream'}</div>
        <div className="pill">Last update {formatTimestamp(status?.last_detection_time ?? null)}</div>
      </section>

      {error ? <section className="banner banner--error">{error}</section> : null}
      <section className="banner banner--info">{localNotice}</section>

      <section className="grid-layout">
        <article className="panel panel--wide">
          <div className="panel__header">
            <div>
              <div className="panel__eyebrow">Sign to voice</div>
              <h2>Live gesture recognition</h2>
            </div>
            <div className="panel__status">{status?.gesture_label ?? 'Detection paused'}</div>
          </div>

          <div className="media-frame">
            <img src={signFeedUrl} alt="Live webcam feed for sign detection" />
            <div className="media-frame__chip">{status?.detected_label ?? 'Awaiting gesture'}</div>
          </div>

          <div className="control-row">
            <button className="button button--primary" onClick={startSign}>
              Start Detection
            </button>
            <button className="button button--ghost" onClick={stopSign}>
              Stop Detection
            </button>
          </div>

          <div className="grid-mini">
            <div className="info-card">
              <span>Detected word</span>
              <strong>{status?.detected_label ?? 'None'}</strong>
            </div>
            <div className="info-card">
              <span>Gesture label</span>
              <strong>{status?.gesture_label ?? 'Waiting'}</strong>
            </div>
            <div className="info-card">
              <span>Supported gestures</span>
              <strong>{supportedWords.join(' · ')}</strong>
            </div>
          </div>
        </article>

        <aside className="panel panel--stack">
          <div className="panel__header">
            <div>
              <div className="panel__eyebrow">Voice to sign</div>
              <h2>Browser speech input</h2>
            </div>
            <div className="panel__status">{listening ? 'Listening' : 'Stopped'}</div>
          </div>

          <div className="voice-actions">
              <button
                className="button button--primary"
                onClick={startListening}
                disabled={listening || !speechRecognitionSupported}
              >
              Start Listening
            </button>
            <button className="button button--danger" onClick={stopListening} disabled={!listening}>
              Stop Listening
            </button>
          </div>

          <div className="info-card info-card--tall">
            <span>Transcript</span>
            <strong>{voiceResponse?.transcript ?? status?.voice_transcript ?? 'Say something like hello or thank you'}</strong>
          </div>

          <div className="voice-match">
            {matchedWord ? (
              matchedImage ? (
                <SignImageCard word={matchedWord} image={matchedImage} />
              ) : (
                <SignCard word={matchedWord} />
              )
            ) : (
              <div className="voice-placeholder">Matched sign will appear here.</div>
            )}
          </div>

          <div className="grid-mini grid-mini--voice">
            <div className="info-card">
              <span>Match</span>
              <strong>{voiceResponse?.matched_label ?? status?.matched_label ?? 'None'}</strong>
            </div>
            <div className="info-card">
              <span>Speech support</span>
              <strong>{speechRecognitionSupported ? 'Browser supported' : 'Not supported in this browser'}</strong>
            </div>
          </div>
        </aside>
      </section>

      <section className="panel panel--logs">
        <div className="panel__header">
          <div>
            <div className="panel__eyebrow">Activity</div>
            <h2>Live status log</h2>
          </div>
          <div className="panel__status">{logs.length} events</div>
        </div>
        <ul className="log-list">
          {logs.length > 0 ? (
            logs.map(entry => <li key={entry}>{entry}</li>)
          ) : (
            <li className="log-list__empty">No events yet. Start detection or voice listening to populate the log.</li>
          )}
        </ul>
      </section>
    </main>
  );
}

export default App;
