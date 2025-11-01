'use client';

import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import StreamingAvatar, { 
  AvatarQuality, 
  StreamingEvents, 
  TaskType,
  TaskMode,
  VoiceEmotion,
  STTProvider
} from '@heygen/streaming-avatar';

import { 
  Mic, 
  Languages
} from 'lucide-react';
import GoogleSTTRecorder from './GoogleSTTRecorder';
import GoogleTTSPlayer from './GoogleTTSPlayer';

interface AvatarVideoProps {
  avatarId?: string;
  voiceId?: string;
}

// AvatarVideo component with dynamic language support (11 languages)
export default function AvatarVideo({ avatarId, voiceId }: AvatarVideoProps) {
  const [streamingAvatar, setStreamingAvatar] = useState<StreamingAvatar | null>(null);
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [isVoiceChatActive, setIsVoiceChatActive] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [isAvatarTalking, setIsAvatarTalking] = useState(false);
  const [text, setText] = useState('');
  const [error, setError] = useState<string | null>(null);
  
  // Basic configuration
  const [quality] = useState<AvatarQuality>(AvatarQuality.High);
  const [connectionQuality, setConnectionQuality] = useState<string>('unknown');
  const [isListening, setIsListening] = useState(false);
  const [lastActivityTime, setLastActivityTime] = useState<Date | null>(null);
  const [isVideoPlaying, setIsVideoPlaying] = useState(false);
  

  
  // Browser Speech Recognition as backup (for fallback only)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [browserSpeechRecognition, setBrowserSpeechRecognition] = useState<any>(null);
  
  // Missing state variables
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [sessionInfo, setSessionInfo] = useState<any>(null);
  const [useBrowserSTT, setUseBrowserSTT] = useState(false);
  
  // Language selection for STT - Default to English
  const [selectedLanguage, setSelectedLanguage] = useState<string>('en');
  console.log('🌍 Avatar configured to speak in English (en) - use language selector to change');
  const [confidenceThreshold] = useState<number>(0.65);
  const [enableConfidenceFiltering] = useState<boolean>(true);
  const [lastConfidenceScore, setLastConfidenceScore] = useState<number | null>(null);
  const [isLanguageSelectorOpen, setIsLanguageSelectorOpen] = useState(false);
  const [isVoiceChatLoading, setIsVoiceChatLoading] = useState(false);
  const [isLanguageChanging, setIsLanguageChanging] = useState(false);
  
  // Google TTS/STT for Tamil
  const [tamilResponseText, setTamilResponseText] = useState<string>('');
  const [isGoogleSTTActive, setIsGoogleSTTActive] = useState(false);
  
  // Check if Tamil is selected
  const isTamilLanguage = selectedLanguage === 'ta';
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  
  // Language options for HeyGen Avatar - 12 important languages (including Tamil)
  const languageOptions = [
    { code: 'en', name: 'English', flag: '🇺🇸' },
    { code: 'ja', name: 'Japanese', flag: '🇯🇵' },
    { code: 'hi', name: 'Hindi', flag: '🇮🇳' },
    { code: 'ta', name: 'Tamil', flag: '🇮🇳' },
    { code: 'zh', name: 'Chinese', flag: '🇨🇳' },
    { code: 'es', name: 'Spanish', flag: '🇪🇸' },
    { code: 'fr', name: 'French', flag: '🇫🇷' },
    { code: 'de', name: 'German', flag: '🇩🇪' },
    { code: 'ru', name: 'Russian', flag: '🇷🇺' },
    { code: 'ko', name: 'Korean', flag: '🇰🇷' },
    { code: 'it', name: 'Italian', flag: '🇮🇹' },
    { code: 'pt', name: 'Portuguese', flag: '🇵🇹' }
  ];

  // Language-to-Voice ID mapping for HeyGen
  // Each language requires specific voice IDs that support it
  // To find Tamil voice IDs, use HeyGen API: https://docs.heygen.com/reference/list-voices-v2
  // Filter by language='ta' or check voice metadata for supported languages
  const languageToVoiceMapping = useMemo<Record<string, string>>(() => {
    const defaultVoice = voiceId || 'default';
    
    // IMPORTANT: Replace the Tamil voice ID with an actual HeyGen Tamil-compatible voice ID
    // To find Tamil voices, use the HeyGen API endpoint:
    // GET https://api.heygen.com/v2/voices?language=ta
    // Or check the voice metadata for supported languages
    
    return {
      'en': defaultVoice, // Use provided voiceId or default for English
      'ja': defaultVoice,  // Japanese - update with Japanese voice ID if needed
      'hi': defaultVoice,  // Hindi - update with Hindi voice ID if needed
      'ta': defaultVoice,  // Tamil - REPLACE 'default' with Tamil-compatible voice ID
      // Example: 'ta': 'YOUR_TAMIL_VOICE_ID_HERE', // Get from HeyGen API
      'zh': defaultVoice,  // Chinese
      'es': defaultVoice,  // Spanish
      'fr': defaultVoice,  // French
      'de': defaultVoice,  // German
      'ru': defaultVoice,  // Russian
      'ko': defaultVoice,  // Korean
      'it': defaultVoice,  // Italian
      'pt': defaultVoice,  // Portuguese
    };
  }, [voiceId]);

  // Get the appropriate voice ID for the selected language
  const getVoiceIdForLanguage = useCallback((langCode: string): string => {
    return languageToVoiceMapping[langCode] || voiceId || 'default';
  }, [voiceId, languageToVoiceMapping]);
  
  // Handle language selection
  const handleLanguageSelect = useCallback((languageCode: string) => {
    setSelectedLanguage(languageCode);
    setIsLanguageSelectorOpen(false);
    console.log('🌍 Language changed to:', languageCode);
    
    // Update browser STT language if it's active (fallback only)
    if (browserSpeechRecognition && useBrowserSTT) {
      console.log('🔄 Updating Browser STT language to:', languageCode);
      browserSpeechRecognition.lang = languageCode;
    }
  }, [browserSpeechRecognition, useBrowserSTT]);
  
  // Helper function to set up video/audio streams
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const setupVideoStream = useCallback((detail: any) => {
    console.log('Setting up video stream with detail:', detail);
    
    if (videoRef.current) {
      if (detail?.video) {
        console.log('Setting video stream from detail.video');
        videoRef.current.srcObject = detail.video;
        videoRef.current.play().catch(console.error);
      } else if (detail?.stream) {
        console.log('Setting video stream from detail.stream');
        videoRef.current.srcObject = detail.stream;
        videoRef.current.play().catch(console.error);
      }
    }
    
    if (audioRef.current) {
      if (detail?.audio) {
        console.log('Setting audio stream from detail.audio');
        audioRef.current.srcObject = detail.audio;
        audioRef.current.play().catch(console.error);
      } else if (detail?.stream) {
        console.log('Setting audio stream from detail.stream');
        audioRef.current.srcObject = detail.stream;
        audioRef.current.play().catch(console.error);
      }
    }
  }, []);
  

  
  
  // Browser Speech Recognition Setup
  const setupBrowserSTT = useCallback(() => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      console.log('❌ Browser speech recognition not supported');
      return null;
    }
    
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = selectedLanguage; // Use selected language directly
    
    recognition.onstart = () => {
      console.log('🎤 Browser STT started');
      setIsListening(true);
    };
    
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    recognition.onresult = async (event: any) => {
      console.log('🗣️ Browser STT result:', event);
      const transcript = event.results[event.results.length - 1][0].transcript;
      
      if (event.results[event.results.length - 1].isFinal) {
        console.log('🎯 Final browser transcript:', transcript);
        setIsListening(false);
        
        // Send user speech directly to HeyGen's knowledge base via TALK
        if (streamingAvatar) {
          await streamingAvatar.speak({
            text: transcript,
            task_type: TaskType.TALK, // This will trigger the knowledge base response
            taskMode: TaskMode.SYNC,
          });
        }
      }
    };
    
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    recognition.onerror = (event: any) => {
      console.error('❌ Browser STT error:', event.error);
      setIsListening(false);
    };
    
    recognition.onend = () => {
      console.log('🎤 Browser STT ended');
      setIsListening(false);
    };
    
    return recognition;
  }, [streamingAvatar, selectedLanguage]);
  

  
  // Initialize avatar session
  const startSession = useCallback(async (languageOverride?: string) => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Generate access token
      const tokenResponse = await fetch('/api/generate-token', {
        method: 'POST',
      });
      
      if (!tokenResponse.ok) {
        throw new Error('Failed to generate access token');
      }
      
      const { token } = await tokenResponse.json();
      
      // Initialize streaming avatar
      const avatar = new StreamingAvatar({ token });
      
      // Debug: Log ALL events to find video stream
      const originalOn = avatar.on.bind(avatar);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      avatar.on = (event: any, handler: any) => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const wrappedHandler = (eventData: any) => {
          console.log(`🎯 Event fired: ${event}`, eventData);
          if (eventData?.detail) {
            console.log(`🎯 Event detail for ${event}:`, eventData.detail);
            // Check for video/stream in any event
            if (eventData.detail.video || eventData.detail.stream || eventData.detail.audio) {
              console.log(`🎬 FOUND STREAM IN ${event}!`, {
                video: !!eventData.detail.video,
                stream: !!eventData.detail.stream,
                audio: !!eventData.detail.audio
              });
              setupVideoStream(eventData.detail);
            }
          }
          return handler(eventData);
        };
        return originalOn(event, wrappedHandler);
      };
      
      // Set up event listeners
      avatar.on(StreamingEvents.AVATAR_START_TALKING, (event) => {
        console.log('🎭 Avatar started talking:', event);
        console.log('🎭 Avatar event detail:', event.detail);
        
        // AGGRESSIVE blocking of ANY HeyGen responses during voice chat
        if (isVoiceChatActive) {
          console.log('🚨 BLOCKING ALL AVATAR RESPONSES DURING VOICE CHAT! Interrupting...');
          avatar.interrupt().catch(console.error);
          return;
        }
        
        // Check if this is HeyGen's automatic response
        if (event.detail?.text && (
          event.detail.text.toLowerCase().includes('heygen') ||
          event.detail.text.toLowerCase().includes('streaming') ||
          event.detail.text.toLowerCase().includes('avatar') ||
          event.detail.text.toLowerCase().includes('help you')
        )) {
          console.log('🚨 DETECTED HEYGEN AUTO-RESPONSE! Content:', event.detail.text);
          avatar.interrupt().catch(console.error);
          return;
        }
        
        setIsAvatarTalking(true);
        
        // Check if video stream is in avatar events
        if (event.detail?.video || event.detail?.stream) {
          console.log('Video found in avatar event!');
          setupVideoStream(event.detail);
        }
      });
      
      avatar.on(StreamingEvents.AVATAR_STOP_TALKING, (event) => {
        console.log('🎭 Avatar stopped talking:', event);
        console.log('🎭 Avatar stop event detail:', event.detail);
        setIsAvatarTalking(false);
      });
      
      avatar.on(StreamingEvents.STREAM_READY, (event) => {
        console.log('Stream ready event:', event);
        console.log('Event detail:', event.detail);
        console.log('Video stream:', event.detail?.video);
        console.log('Audio stream:', event.detail?.audio);
        
        setIsConnected(true);
        
        // Enhanced video stream handling
        if (videoRef.current) {
          console.log('Video ref available');
          
          if (event.detail?.video) {
            console.log('Setting video stream...');
            videoRef.current.srcObject = event.detail.video;
            videoRef.current.onloadedmetadata = () => {
              console.log('Video metadata loaded, playing...');
              videoRef.current?.play().catch(console.error);
            };
            videoRef.current.play().catch(console.error);
          } else if (event.detail?.stream) {
            // Alternative stream property
            console.log('Using alternative stream property...');
            videoRef.current.srcObject = event.detail.stream;
            videoRef.current.play().catch(console.error);
          } else {
            console.log('No video stream found in event');
          }
        } else {
          console.log('Video ref not available');
        }
        
        // Enhanced audio stream handling
        if (audioRef.current) {
          if (event.detail?.audio) {
            console.log('Setting audio stream...');
            audioRef.current.srcObject = event.detail.audio;
            audioRef.current.play().catch(console.error);
          } else if (event.detail?.stream) {
            // Sometimes audio is in the same stream
            audioRef.current.srcObject = event.detail.stream;
            audioRef.current.play().catch(console.error);
          }
        }
      });
      
      avatar.on(StreamingEvents.STREAM_DISCONNECTED, () => {
        console.log('Stream disconnected');
        setIsConnected(false);
        setIsSessionActive(false);
      });
      
      avatar.on(StreamingEvents.USER_START, (event) => {
        console.log('🎤 User started talking:', event);
        setIsListening(true);
        setLastActivityTime(new Date());
      });
      
      avatar.on(StreamingEvents.USER_STOP, (event) => {
        console.log('🎤 User stopped talking:', event);
        setIsListening(false);
        setLastActivityTime(new Date());
        // HeyGen will handle the speech processing and knowledge base response automatically
      });
      
      // Note: Speech recognition and knowledge base responses are now handled automatically by HeyGen

      // New SDK 2.0.16 Events
      avatar.on('stream_quality', (event) => {
        console.log('Stream quality update:', event);
        setConnectionQuality(event.detail?.quality || 'good');
      });
      
      avatar.on(StreamingEvents.AVATAR_STOP_TALKING, (event) => {
        console.log('Avatar stopped talking with duration:', event.detail.duration_ms);
        setIsAvatarTalking(false);
        setLastActivityTime(new Date());
      });
      
      // Use language override if provided, otherwise use selected language
      const sessionLanguage = languageOverride || selectedLanguage;
      
      // Get the appropriate voice ID for the selected language
      const languageVoiceId = getVoiceIdForLanguage(sessionLanguage);
      
      // Create avatar session with selected language configuration
      console.log('Creating avatar session with language settings:', {
        avatarId: avatarId || 'Katya_ProfessionalLook2_public',
        voiceId: languageVoiceId, // Use language-specific voice ID
        quality: quality,
        language: sessionLanguage // Use session language
      });
      
      // Try multiple configurations to find what works
      let sessionData;
      
      // Special handling for Tamil - HeyGen may not support 'ta' language code
      // If Tamil is selected, try without language parameter first if it fails
      const isTamil = sessionLanguage === 'ta';
      let tamilFallbackUsed = false;
      
      // First try: Minimal configuration with session language
      try {
        console.log(`Trying minimal configuration with ${sessionLanguage} language...`);
        sessionData = await avatar.createStartAvatar({
          quality: quality,
          avatarName: 'Katya_CasualLook_public',
          knowledgeId: '73785c3e335945d7a80c032fc7d58067',
          language: sessionLanguage,
          voice: {
            voiceId: languageVoiceId,
            rate: 1.0
          },
          sttSettings: {
            provider: STTProvider.DEEPGRAM,
            confidence: 0.55
          }
        });
        console.log(`Minimal config with ${sessionLanguage} worked!`);
      } catch (error1: any) {
        console.log(`Minimal config with ${sessionLanguage} failed:`, error1);
        
        // Special handling for Tamil 400 errors
        if (isTamil && (error1?.message?.includes('400') || error1?.code === 400 || error1?.statusCode === 400)) {
          console.log('⚠️ Tamil language may not be supported by HeyGen API');
          console.log('🔄 Trying Tamil without language parameter (fallback)...');
          
          try {
            // Try without language parameter for Tamil
            sessionData = await avatar.createStartAvatar({
              quality: quality,
              avatarName: 'Katya_CasualLook_public',
              knowledgeId: '73785c3e335945d7a80c032fc7d58067',
              // Don't include language parameter for Tamil
              voice: {
                voiceId: languageVoiceId,
                rate: 1.0
              },
              sttSettings: {
                provider: STTProvider.DEEPGRAM,
                confidence: 0.55
              }
            });
            console.log('✅ Tamil session created without language parameter (fallback mode)');
            tamilFallbackUsed = true;
            setError('Tamil language code may not be supported. Using voice-only mode. Some features may be limited.');
          } catch (tamilFallbackError) {
            console.log('❌ Tamil fallback also failed:', tamilFallbackError);
            
            // Last resort: Try with English language as fallback
            console.log('🔄 Trying with English language as fallback for Tamil...');
            try {
              sessionData = await avatar.createStartAvatar({
                quality: quality,
                avatarName: 'Katya_CasualLook_public',
                knowledgeId: '73785c3e335945d7a80c032fc7d58067',
                language: 'en', // Fallback to English
                voice: {
                  voiceId: languageVoiceId,
                  rate: 1.0
                }
              });
              console.log('✅ Tamil session created with English fallback');
              setError('Tamil language not supported by HeyGen. Using English language with Tamil voice ID.');
            } catch (englishFallbackError) {
              console.log('❌ English fallback also failed:', englishFallbackError);
              throw error1; // Re-throw original error
            }
          }
          
          // Skip other retry attempts if Tamil fallback worked
          if (tamilFallbackUsed || sessionData) {
            // Continue to success path
          } else {
            // Continue to next retry attempts
            // Second try: With voice and session language
            try {
              console.log(`Trying with voice and ${sessionLanguage} language...`);
              sessionData = await avatar.createStartAvatar({
                quality: quality,
                avatarName: 'Katya_CasualLook_public',
                knowledgeId: '73785c3e335945d7a80c032fc7d58067',
                language: sessionLanguage,
                voice: {
                  voiceId: languageVoiceId,
                  rate: 1.0
                }
              });
              console.log(`Config with voice and ${sessionLanguage} worked!`);
            } catch (error2) {
              console.log(`Config with voice and ${sessionLanguage} failed:`, error2);
              
              // Third try: Minimal configuration with knowledge base and session language
              console.log(`Trying minimal config with knowledge base and ${sessionLanguage} language...`);
              sessionData = await avatar.createStartAvatar({
                quality: quality,
                avatarName: 'Katya_CasualLook_public',
                knowledgeId: '73785c3e335945d7a80c032fc7d58067',
                language: sessionLanguage
              });
            }
          }
        } else {
          // Normal error handling for non-Tamil languages
          // Second try: With voice and session language
          try {
            console.log(`Trying with voice and ${sessionLanguage} language...`);
            sessionData = await avatar.createStartAvatar({
              quality: quality,
              avatarName: 'Katya_CasualLook_public',
              knowledgeId: '73785c3e335945d7a80c032fc7d58067',
              language: sessionLanguage,
              voice: {
                voiceId: languageVoiceId,
                rate: 1.0
              }
            });
            console.log(`Config with voice and ${sessionLanguage} worked!`);
          } catch (error2) {
            console.log(`Config with voice and ${sessionLanguage} failed:`, error2);
            
            // Third try: Minimal configuration with knowledge base and session language
            console.log(`Trying minimal config with knowledge base and ${sessionLanguage} language...`);
            sessionData = await avatar.createStartAvatar({
              quality: quality,
              avatarName: 'Katya_CasualLook_public',
              knowledgeId: '73785c3e335945d7a80c032fc7d58067',
              language: sessionLanguage
            });
          }
        }
      }
      
      console.log('Avatar session created successfully:', sessionData);
      
      // Try to get video stream directly from avatar instance
      console.log('🔍 Checking avatar instance for video methods...');
      console.log('Avatar instance:', avatar);
      console.log('Avatar methods:', Object.getOwnPropertyNames(avatar));
      console.log('Avatar prototype methods:', Object.getOwnPropertyNames(Object.getPrototypeOf(avatar)));
      
      // Check for common video stream properties  
      const checkAvatarStreams = () => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        if ((avatar as any).mediaStream) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          console.log('📹 Found avatar.mediaStream:', (avatar as any).mediaStream);
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          setupVideoStream({ stream: (avatar as any).mediaStream });
        }
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        if ((avatar as any).videoStream) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          console.log('📹 Found avatar.videoStream:', (avatar as any).videoStream);
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          setupVideoStream({ video: (avatar as any).videoStream });
        }
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        if ((avatar as any).getVideoStream && typeof (avatar as any).getVideoStream === 'function') {
          console.log('📹 Found avatar.getVideoStream method');
          try {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const stream = (avatar as any).getVideoStream();
            console.log('📹 Got video stream:', stream);
            setupVideoStream({ stream });
          } catch (e) {
            console.log('❌ getVideoStream failed:', e);
          }
        }
      };
      
      checkAvatarStreams();
      
      // Also check for streams after a delay (sometimes they appear later)
      setTimeout(() => {
        console.log('⏰ Checking for video streams after 2 seconds...');
        checkAvatarStreams();
      }, 2000);
      
      setTimeout(() => {
        console.log('⏰ Checking for video streams after 5 seconds...');
        checkAvatarStreams();
      }, 5000);
      
      setStreamingAvatar(avatar);
      setSessionInfo(sessionData);
      setIsSessionActive(true);
      
      console.log('✅ Avatar session started successfully - ready for interaction');
      
      // Start voice chat mode for microphone access
      console.log('🎤 Starting voice chat mode...');
      try {
        await avatar.startVoiceChat({
          isInputAudioMuted: false,
        });
        setIsVoiceChatActive(true);
        console.log('✅ Voice chat mode started - microphone should be active');
      } catch (voiceError) {
        console.error('❌ Failed to start voice chat:', voiceError);
        console.log('💡 You can still use text input or try the voice chat button');
      }
      
    } catch (err) {
      console.error('Failed to start session:', err);
      setError(err instanceof Error ? err.message : 'Failed to start session');
    } finally {
      setIsLoading(false);
    }
  }, [avatarId, voiceId, isVoiceChatActive, quality, setupVideoStream, getVoiceIdForLanguage]);
  
  // Stop avatar session
  const stopSession = useCallback(async () => {
    try {
      if (streamingAvatar) {
        if (isVoiceChatActive) {
          await streamingAvatar.closeVoiceChat();
        }
        await streamingAvatar.stopAvatar();
        setStreamingAvatar(null);
      }
      setIsSessionActive(false);
      setIsConnected(false);
      setIsVoiceChatActive(false);
      setIsAvatarTalking(false);
      setSessionInfo(null);
    } catch (err) {
      console.error('Failed to stop session:', err);
      setError(err instanceof Error ? err.message : 'Failed to stop session');
    }
  }, [streamingAvatar, isVoiceChatActive]);
  
  // Updated language selection handler with session restart
  const handleLanguageSelectWithRestart = useCallback(async (languageCode: string) => {
    console.log('🌍 Language changing to:', languageCode);
    setIsLanguageSelectorOpen(false);
    
    // Update browser STT language if it's active (fallback only)
    if (browserSpeechRecognition && useBrowserSTT) {
      console.log('🔄 Updating Browser STT language to:', languageCode);
      browserSpeechRecognition.lang = languageCode;
    }
    
    // Restart avatar session with new language if session is active
    if (streamingAvatar && isSessionActive) {
      console.log('🔄 Restarting avatar session with new language:', languageCode);
      try {
        setIsLanguageChanging(true);
        await stopSession();
        // Update the language state immediately
        setSelectedLanguage(languageCode);
        // Start session with the new language directly
        await startSession(languageCode);
        console.log('✅ Session restarted with language:', languageCode);
      } catch (error) {
        console.error('❌ Failed to restart session with new language:', error);
        setError('Failed to change language. Please try again.');
        // Revert language state on error
        setSelectedLanguage(selectedLanguage);
      } finally {
        setIsLanguageChanging(false);
      }
    } else {
      // If no active session, just update the language state
      setSelectedLanguage(languageCode);
      console.log('🌍 Language changed to:', languageCode);
    }
  }, [browserSpeechRecognition, useBrowserSTT, streamingAvatar, isSessionActive, stopSession, startSession, selectedLanguage]);
  
  // Handle Google STT transcript for Tamil
  const handleGoogleSTTTranscript = useCallback(async (transcript: string, confidence: number) => {
    console.log('🗣️ Google STT Tamil transcript:', transcript, 'Confidence:', confidence);
    
    if (transcript.trim().length === 0) return;
    
    // Generate response using simple logic (can be replaced with AI/LLM)
    const response = await generateTamilResponse(transcript);
    setTamilResponseText(response);
  }, []);

  // Simple Tamil response generator (replace with your AI/LLM service)
  const generateTamilResponse = async (userInput: string): Promise<string> => {
    // TODO: Replace with your AI service or knowledge base
    // For now, return a simple echo response
    return `நீங்கள் சொன்னது: ${userInput}. இது ஒரு எடுத்துக்காட்டு பதில்.`;
  };

  // Send text to avatar (using HeyGen for non-Tamil, Google TTS for Tamil)
  const sendText = useCallback(async () => {
    if (!text.trim()) return;
    
    const userMessage = text.trim();
    setText(''); // Clear input immediately
    
    console.log('📝 User typed:', userMessage);
    
    // For Tamil, use Google TTS
    if (isTamilLanguage) {
      console.log('🇮🇳 Using Google TTS for Tamil response');
      const response = await generateTamilResponse(userMessage);
      setTamilResponseText(response);
      return;
    }
    
    // For other languages, use HeyGen
    if (!streamingAvatar) {
      setError('Avatar session not active');
      return;
    }
    
    try {
      // Use HeyGen's speak method with TALK type - the avatar will use its built-in knowledge base
      console.log('🎭 Sending to HeyGen knowledge base:', userMessage);
      await streamingAvatar.speak({
        text: userMessage,
        task_type: TaskType.TALK, // This will trigger the knowledge base response
        taskMode: TaskMode.SYNC,
      });
    } catch (err) {
      console.error('Failed to send text:', err);
      setError(err instanceof Error ? err.message : 'Failed to send text');
    }
  }, [streamingAvatar, text, isTamilLanguage]);
  
  // Toggle voice chat (HeyGen for non-Tamil, Google STT for Tamil)
  const toggleVoiceChat = useCallback(async () => {
    // For Tamil, use Google STT
    if (isTamilLanguage) {
      if (isGoogleSTTActive) {
        console.log('🔇 Stopping Google STT for Tamil...');
        setIsGoogleSTTActive(false);
      } else {
        console.log('🎤 Starting Google STT for Tamil...');
        setIsVoiceChatLoading(true);
        
        try {
          // Request microphone permission
          await navigator.mediaDevices.getUserMedia({ audio: true });
          console.log('✅ Microphone permission granted');
          setIsGoogleSTTActive(true);
        } catch (permError) {
          console.error('❌ Microphone permission denied:', permError);
          setError('Microphone permission denied. Please allow microphone access and try again.');
        } finally {
          setIsVoiceChatLoading(false);
        }
      }
      return;
    }
    
    // For other languages, use HeyGen
    if (!streamingAvatar) {
      console.log('❌ No streaming avatar available');
      return;
    }
    
    try {
      if (isVoiceChatActive) {
        console.log('🔇 Stopping voice chat...');
        await streamingAvatar.closeVoiceChat();
        setIsVoiceChatActive(false);
        console.log('✅ Voice chat stopped');
      } else {
        console.log('🎤 Starting voice chat...');
        setIsVoiceChatLoading(true);
        
        try {
          console.log('🔍 Requesting microphone permissions...');
          
          // Request microphone permission first
          try {
            await navigator.mediaDevices.getUserMedia({ audio: true });
            console.log('✅ Microphone permission granted');
          } catch (permError) {
            console.error('❌ Microphone permission denied:', permError);
            setError('Microphone permission denied. Please allow microphone access and try again.');
            setIsVoiceChatLoading(false);
            return;
          }
          
          // Start voice chat with STT
          await streamingAvatar.startVoiceChat({
            isInputAudioMuted: false,
          });
          setIsVoiceChatActive(true);
          console.log('✅ Voice chat started - speak now!');
        } finally {
          setIsVoiceChatLoading(false);
        }
      }
    } catch (err) {
      console.error('❌ Failed to toggle voice chat:', err);
      setError(err instanceof Error ? err.message : 'Failed to toggle voice chat');
      setIsVoiceChatLoading(false);
    }
  }, [streamingAvatar, isVoiceChatActive, isTamilLanguage, isGoogleSTTActive]);
  
  // Toggle microphone mute
  const toggleMute = useCallback(async () => {
    if (!streamingAvatar) return;
    
    try {
      if (isMuted) {
        await streamingAvatar.unmuteInputAudio();
      } else {
        await streamingAvatar.muteInputAudio();
      }
      setIsMuted(!isMuted);
    } catch (err) {
      console.error('Failed to toggle mute:', err);
      setError(err instanceof Error ? err.message : 'Failed to toggle mute');
    }
  }, [streamingAvatar, isMuted]);
  
  // Interrupt avatar
  const interruptAvatar = useCallback(async () => {
    if (!streamingAvatar) return;
    
    try {
      await streamingAvatar.interrupt();
    } catch (err) {
      console.error('Failed to interrupt avatar:', err);
      setError(err instanceof Error ? err.message : 'Failed to interrupt avatar');
    }
  }, [streamingAvatar]);
  
  // Toggle Browser Speech Recognition
  const toggleBrowserSTT = useCallback(async () => {
    if (useBrowserSTT) {
      // Stop browser STT
      if (browserSpeechRecognition) {
        browserSpeechRecognition.stop();
        setBrowserSpeechRecognition(null);
      }
      setUseBrowserSTT(false);
      console.log('🔇 Browser STT stopped');
    } else {
      // Start browser STT
      const recognition = setupBrowserSTT();
      if (recognition) {
        setBrowserSpeechRecognition(recognition);
        recognition.start();
        setUseBrowserSTT(true);
        console.log('🎤 Browser STT started');
      }
    }
  }, [useBrowserSTT, browserSpeechRecognition, setupBrowserSTT]);
  

  
  // Keep session alive (SDK 2.0.16 feature)
  const keepSessionAlive = useCallback(async () => {
    if (!streamingAvatar) return;
    
    try {
      await streamingAvatar.keepAlive();
      setLastActivityTime(new Date());
      console.log('Session kept alive');
    } catch (err) {
      console.error('Failed to keep session alive:', err);
      setError(err instanceof Error ? err.message : 'Failed to keep session alive');
    }
  }, [streamingAvatar]);
  
  // Start avatar listening (for text mode)
  const startAvatarListening = useCallback(async () => {
    if (!streamingAvatar || isVoiceChatActive) return;
    
    try {
      await streamingAvatar.startListening();
      setIsListening(true);
    } catch (err) {
      console.error('Failed to start listening:', err);
      setError(err instanceof Error ? err.message : 'Failed to start listening');
    }
  }, [streamingAvatar, isVoiceChatActive]);
  
  // Stop avatar listening (for text mode)
  const stopAvatarListening = useCallback(async () => {
    if (!streamingAvatar || isVoiceChatActive) return;
    
    try {
      await streamingAvatar.stopListening();
      setIsListening(false);
    } catch (err) {
      console.error('Failed to stop listening:', err);
      setError(err instanceof Error ? err.message : 'Failed to stop listening');
    }
  }, [streamingAvatar, isVoiceChatActive]);
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamingAvatar) {
        streamingAvatar.stopAvatar().catch(console.error);
      }
    };
  }, [streamingAvatar]);
  
  return (
    <div className="w-full h-screen bg-black overflow-hidden">
      {/* Video Container - Full screen height */}
      <div className="relative w-full h-full">
        
        {/* Error Display - Positioned relative to video */}
        {error && (
          <div className="absolute top-2 left-2 right-2 bg-red-500 bg-opacity-90 text-white px-2 py-1.5 rounded z-50">
            <p className="text-xs">{error}</p>
            <button 
              onClick={() => setError(null)}
              className="text-red-200 hover:text-white text-xs underline"
            >
              Dismiss
            </button>
          </div>
        )}
        
                 <video
           ref={videoRef}
           autoPlay
           playsInline
           muted={false}
           controls={false}
           className="w-full h-full object-contain"
           style={{ backgroundColor: '#000' }}
           onLoadStart={() => console.log('Video load started')}
           onCanPlay={() => console.log('Video can play')}
           onPlay={() => {
             console.log('Video playing');
             setIsVideoPlaying(true);
           }}
           onPause={() => {
             console.log('Video paused');
             setIsVideoPlaying(false);
           }}
           onError={(e) => console.error('Video error:', e)}
         />
        <audio 
          ref={audioRef} 
          autoPlay 
          onPlay={() => console.log('Audio playing')}
          onError={(e) => console.error('Audio error:', e)}
        />
        
                 {/* Logo Overlay - Top Left of Video */}
         <div className="absolute top-4 left-4 z-30">
           <img 
             src="/logo/Sobha.png" 
             alt="Sobha Logo" 
             className="h-16 w-auto opacity-80 sm:h-18 md:h-20 lg:h-22 xl:h-24"
           />
         </div>
         
         {/* Compact Status Indicators - Top Left of Video (below logo) */}
         <div className="absolute top-20 left-4 flex flex-col gap-2 sm:top-22 sm:left-4 sm:gap-2 md:top-24 md:left-4 md:gap-2 lg:top-26 lg:left-4 lg:gap-2 xl:top-28 xl:left-4 xl:gap-2">
                       {isConnected && (
              <span className="bg-green-500 bg-opacity-80 text-white px-2 py-1 rounded text-xs flex items-center gap-1.5 sm:px-2.5 sm:py-1.5 sm:text-sm sm:gap-2 md:px-3 md:py-2 md:text-sm md:gap-2 lg:px-3.5 lg:py-2.5 lg:text-sm lg:gap-2.5 xl:px-4 xl:py-3 xl:text-sm xl:gap-3">
                <div className="w-1.5 h-1.5 bg-white rounded-full sm:w-2 sm:h-2 md:w-2.5 md:h-2.5 lg:w-3 lg:h-3 xl:w-3.5 xl:h-3.5"></div>
                <span className="text-xs sm:text-sm md:text-sm lg:text-sm xl:text-sm">Live</span>
              </span>
            )}
            {(isAvatarTalking || (isTamilLanguage && tamilResponseText)) && (
              <span className="bg-blue-500 bg-opacity-80 text-white px-2 py-1 rounded text-xs flex items-center gap-1.5 sm:px-2.5 sm:py-1.5 sm:text-sm sm:gap-2 md:px-3 md:py-2 md:text-sm md:gap-2 lg:px-3.5 lg:py-2.5 lg:text-sm lg:gap-2.5 xl:px-4 xl:py-3 xl:text-sm xl:gap-3">
                <div className="w-1.5 h-1.5 bg-white rounded-full animate-pulse sm:w-2 sm:h-2 md:w-2.5 md:h-2.5 lg:w-3 lg:h-3 xl:w-3.5 xl:h-3.5"></div>
                <span className="text-xs sm:text-sm md:text-sm lg:text-sm xl:text-sm">
                  {isTamilLanguage ? 'Speaking (Tamil)' : 'Speaking'}
                </span>
              </span>
            )}
            {isGoogleSTTActive && (
              <span className="bg-purple-500 bg-opacity-80 text-white px-2 py-1 rounded text-xs flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 bg-white rounded-full animate-pulse"></div>
                <span className="text-xs">Listening (Tamil)</span>
              </span>
            )}

         </div>
        
        {/* Loading Overlay */}
        {isLoading && (
          <div className="absolute inset-0 bg-black bg-opacity-75 flex items-center justify-center">
            <div className="text-white text-center px-4">
              <div className="text-sm mb-2">Loading Avatar...</div>
              <div className="text-xs">Connecting to Agent</div>
            </div>
          </div>
        )}
        
                 {/* No video placeholder */}
         {!isConnected && !isLoading && !isTamilLanguage && (
           <div className="absolute inset-0 flex items-center justify-center">
             <div className="text-white text-center px-4">
               <button
                 onClick={() => startSession(selectedLanguage)}
                 className="px-3 py-1.5 rounded-lg bg-red-500 hover:bg-red-600 text-white text-sm font-medium transition-colors shadow-lg sm:px-3.5 sm:py-1.5 sm:text-sm md:px-4 md:py-2 md:text-sm lg:px-4.5 lg:py-2.5 lg:text-sm"
                 title="Start Session"
               >
                 Start Session
               </button>
             </div>
           </div>
         )}
         
         {/* Tamil mode - no HeyGen session needed */}
         {isTamilLanguage && !isConnected && (
           <div className="absolute inset-0 flex items-center justify-center">
             <div className="text-white text-center px-4">
               <div className="text-sm mb-2">🇮🇳 Tamil Mode Active</div>
               <div className="text-xs text-gray-300">Using Google TTS & STT</div>
               <div className="text-xs text-gray-400 mt-2">Click microphone to start speaking Tamil</div>
             </div>
           </div>
         )}
        
        {/* Connected but no video placeholder */}
        {isConnected && !isLoading && !isVideoPlaying && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="bg-black bg-opacity-50 text-white text-center px-4 py-2 rounded">
              <div className="text-xs">Avatar Connected</div>
              <div className="text-xs">Waiting for video...</div>
            </div>
          </div>
        )}
                 {/* Voice Chat Toggle Button - Right side of video */}
         {(isSessionActive || isTamilLanguage) && (
           <button
             onClick={toggleVoiceChat}
             disabled={isVoiceChatLoading}
             className={`absolute right-3 top-1/3 p-2 rounded-full shadow-lg sm:right-4 sm:p-2 md:right-6 md:p-2 lg:right-8 lg:p-2 xl:right-10 xl:p-2 ${
               (isVoiceChatActive || isGoogleSTTActive)
                 ? 'bg-red-500 hover:bg-red-600 text-white scale-110 animate-pulse' 
                 : isVoiceChatLoading
                 ? 'bg-orange-500 text-white scale-105 animate-pulse'
                 : 'bg-gray-500 hover:bg-gray-600 text-white'
             }`}
             title={
               (isVoiceChatActive || isGoogleSTTActive)
                 ? 'Stop Voice Chat' 
                 : isVoiceChatLoading 
                 ? 'Starting Voice Chat...' 
                 : 'Start Voice Chat'
             }
           >
             <Mic size={16} className="sm:w-4 sm:h-4 md:w-4 md:h-4 lg:w-4 lg:h-4 xl:w-4 xl:h-4" />
           </button>
         )}
         
         {/* Google STT Recorder for Tamil */}
         {isTamilLanguage && (
           <div className="absolute right-3 top-1/3 mt-12 sm:right-4 sm:mt-14 md:right-6 md:mt-16">
             <GoogleSTTRecorder
               languageCode="ta-IN"
               onTranscript={handleGoogleSTTTranscript}
               onError={(err) => setError(err)}
               isActive={isGoogleSTTActive}
             />
           </div>
         )}
         
         {/* Google TTS Player for Tamil */}
         {isTamilLanguage && tamilResponseText && (
           <div className="absolute bottom-4 left-4 right-4 bg-black bg-opacity-75 text-white p-3 rounded z-40">
             <GoogleTTSPlayer
               text={tamilResponseText}
               languageCode="ta-IN"
               autoPlay={true}
               onEnd={() => {
                 console.log('✅ Tamil TTS playback completed');
                 setIsAvatarTalking(false);
               }}
               onPlay={() => {
                 console.log('🎙️ Tamil TTS playing...');
                 setIsAvatarTalking(true);
               }}
             />
             <div className="mt-2 text-xs text-gray-300">
               <div className="font-semibold mb-1">Response:</div>
               <div>{tamilResponseText}</div>
             </div>
           </div>
         )}
         
         {/* Language Selector Button - Below Google STT recorder */}
         <div className="absolute right-3 top-1/3 mt-24 sm:right-4 sm:mt-28 md:right-6 md:mt-32 lg:right-8 lg:mt-36 xl:right-10 xl:mt-40">
           <button
             onClick={() => setIsLanguageSelectorOpen(!isLanguageSelectorOpen)}
             disabled={isLanguageChanging}
             className={`p-2 rounded-full bg-blue-600 hover:bg-blue-700 text-white transition-all duration-200 shadow-lg sm:p-2 md:p-2 lg:p-2 xl:p-2 ${
               isLanguageChanging ? 'opacity-50 cursor-not-allowed animate-pulse' : ''
             }`}
             title={isLanguageChanging ? 'Changing Language...' : 'Select Language'}
           >
             <Languages size={16} className="sm:w-4 sm:h-4 md:w-4 md:h-4 lg:w-4 lg:h-4 xl:w-4 xl:h-4" />
           </button>
          
                     {/* Language Dropdown */}
           {isLanguageSelectorOpen && (
             <div className="absolute right-0 top-full mt-2 bg-white rounded-lg shadow-xl border border-gray-200 min-w-40 max-h-48 overflow-y-auto z-50 sm:min-w-40 sm:max-h-48 md:min-w-40 md:max-h-48 lg:min-w-40 lg:max-h-48 xl:min-w-40 xl:max-h-48">
               <div className="py-1 sm:py-1 md:py-1">
                 <div className="px-3 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wide border-b border-gray-100 sm:px-3 sm:py-2 sm:text-xs md:px-3 md:py-2 md:text-xs lg:px-3 lg:py-2 lg:text-xs">
                   Select Language
                 </div>
                 {languageOptions.map((language) => (
                   <button
                     key={language.code}
                     onClick={() => handleLanguageSelectWithRestart(language.code)}
                     disabled={isLanguageChanging}
                     className={`w-full text-left px-3 py-1.5 text-xs hover:bg-blue-50 transition-colors sm:px-3 sm:py-1.5 sm:text-xs md:px-3 md:py-1.5 md:text-xs lg:px-3 lg:py-1.5 lg:text-xs ${
                       selectedLanguage === language.code 
                         ? 'bg-blue-100 text-blue-700 font-medium' 
                         : 'text-gray-700'
                     } ${isLanguageChanging ? 'opacity-50 cursor-not-allowed' : ''}`}
                   >
                     <span className="mr-1.5 sm:mr-1.5 md:mr-1.5 lg:mr-1.5">{language.flag}</span>
                     <span className="truncate">
                       {language.name} {selectedLanguage === language.code ? '(Active)' : ''}
                       {isLanguageChanging && selectedLanguage === language.code ? ' (Changing...)' : ''}
                     </span>
                   </button>
                 ))}
               </div>
             </div>
           )}
        </div>
      </div>
    </div>
  );
}
