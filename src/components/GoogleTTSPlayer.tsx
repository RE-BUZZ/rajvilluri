'use client';

import React, { useState, useRef, useEffect } from 'react';

interface GoogleTTSPlayerProps {
  text: string;
  languageCode?: string;
  onPlay?: () => void;
  onEnd?: () => void;
  autoPlay?: boolean;
}

/**
 * Google TTS Audio Player Component
 * Handles playback of Tamil TTS audio
 */
export default function GoogleTTSPlayer({
  text,
  languageCode = 'ta-IN',
  onPlay,
  onEnd,
  autoPlay = false,
}: GoogleTTSPlayerProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    if (text && text.trim().length > 0) {
      generateAndPlayAudio();
    }
  }, [text, languageCode]);

  const generateAndPlayAudio = async () => {
    if (!text || text.trim().length === 0) return;

    setIsLoading(true);
    setError(null);

    try {
      console.log('🎙️ Generating Tamil TTS audio...', { text: text.substring(0, 50), languageCode });

      const response = await fetch('/api/google/tts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text,
          languageCode,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'TTS generation failed');
      }

      const data = await response.json();
      
      // Convert base64 audio to blob URL
      const audioBytes = Uint8Array.from(atob(data.audioContent), c => c.charCodeAt(0));
      const blob = new Blob([audioBytes], { type: 'audio/mp3' });
      const url = URL.createObjectURL(blob);
      
      setAudioUrl(url);

      // Auto-play if enabled
      if (autoPlay && audioRef.current) {
        audioRef.current.play().catch(err => {
          console.error('Auto-play failed:', err);
        });
      }

      if (onPlay) {
        onPlay();
      }

    } catch (err: any) {
      console.error('Google TTS Error:', err);
      setError(err.message || 'Failed to generate speech');
    } finally {
      setIsLoading(false);
    }
  };

  const playAudio = () => {
    if (audioRef.current) {
      audioRef.current.play().catch(err => {
        console.error('Play failed:', err);
        setError('Failed to play audio');
      });
    }
  };

  useEffect(() => {
    // Cleanup blob URL on unmount
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  return (
    <div className="google-tts-player">
      {isLoading && (
        <div className="text-xs text-blue-500">Generating Tamil speech...</div>
      )}
      
      {error && (
        <div className="text-xs text-red-500">Error: {error}</div>
      )}

      {audioUrl && (
        <audio
          ref={audioRef}
          src={audioUrl}
          onEnded={() => {
            if (onEnd) onEnd();
          }}
          onError={(e) => {
            console.error('Audio playback error:', e);
            setError('Audio playback failed');
          }}
          preload="auto"
        />
      )}
    </div>
  );
}

