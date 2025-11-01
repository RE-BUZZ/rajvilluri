'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';

interface GoogleSTTRecorderProps {
  languageCode?: string;
  onTranscript: (transcript: string, confidence: number) => void;
  onError?: (error: string) => void;
  isActive?: boolean;
}

/**
 * Google Speech-to-Text Recorder Component
 * Records audio and sends to Google STT API for Tamil recognition
 */
export default function GoogleSTTRecorder({
  languageCode = 'ta-IN',
  onTranscript,
  onError,
  isActive = false,
}: GoogleSTTRecorderProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  // Start recording
  const startRecording = useCallback(async () => {
    try {
      console.log('🎤 Starting Google STT recording for Tamil...');
      
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
      });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        console.log('🎤 Recording stopped, processing with Google STT...');
        setIsProcessing(true);

        try {
          // Combine audio chunks
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
          
          // Send to Google STT API
          const formData = new FormData();
          formData.append('audio', audioBlob, 'recording.webm');
          formData.append('language', languageCode);

          const response = await fetch('/api/google/stt', {
            method: 'POST',
            body: formData,
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'STT recognition failed');
          }

          const data = await response.json();
          console.log('✅ Google STT Result:', data);

          if (data.transcript && data.transcript.trim().length > 0) {
            onTranscript(data.transcript, data.confidence || 0);
          }

        } catch (err: any) {
          console.error('Google STT Error:', err);
          if (onError) {
            onError(err.message || 'Speech recognition failed');
          }
        } finally {
          setIsProcessing(false);
          audioChunksRef.current = [];
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
      console.log('✅ Recording started');

    } catch (err: any) {
      console.error('Failed to start recording:', err);
      if (onError) {
        onError(err.message || 'Failed to access microphone');
      }
    }
  }, [languageCode, onTranscript, onError]);

  // Stop recording
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);

      // Stop all tracks
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
    }
  }, [isRecording]);

  // Auto start/stop based on isActive prop
  useEffect(() => {
    if (isActive && !isRecording) {
      startRecording();
    } else if (!isActive && isRecording) {
      stopRecording();
    }

    // Cleanup on unmount
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, [isActive, isRecording, startRecording, stopRecording]);

  return (
    <div className="google-stt-recorder">
      {isRecording && (
        <div className="text-xs text-green-500 animate-pulse">
          🎤 Recording Tamil speech...
        </div>
      )}
      {isProcessing && (
        <div className="text-xs text-blue-500">
          🔄 Processing with Google STT...
        </div>
      )}
    </div>
  );
}

