import { NextRequest, NextResponse } from 'next/server';

/**
 * Google Cloud Speech-to-Text API Route (Enhanced)
 * Handles Tamil and other language speech recognition
 * Supports multiple audio formats and improved error handling
 */
export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const audioFile = formData.get('audio') as File;
    const languageCode = (formData.get('language') as string) || 'ta-IN'; // Default to Tamil
    const sampleRate = formData.get('sampleRate') ? parseInt(formData.get('sampleRate') as string) : null;
    
    if (!audioFile) {
      return NextResponse.json(
        { error: 'No audio file provided' },
        { status: 400 }
      );
    }

    // Check file size (limit to 10MB)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (audioFile.size > maxSize) {
      return NextResponse.json(
        { error: `Audio file too large. Maximum size is ${maxSize / 1024 / 1024}MB` },
        { status: 400 }
      );
    }

    // Check for Google Cloud credentials
    const googleCredentials = process.env.GOOGLE_APPLICATION_CREDENTIALS;
    const googleProjectId = process.env.GOOGLE_CLOUD_PROJECT_ID;
    
    if (!googleCredentials && !googleProjectId) {
      const apiKey = process.env.GOOGLE_CLOUD_API_KEY;
      if (!apiKey) {
        return NextResponse.json(
          { 
            error: 'Google Cloud credentials not configured',
            message: 'Set GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_CLOUD_API_KEY in environment variables',
            docs: 'See GOOGLE_TTS_STT_SETUP.md for setup instructions'
          },
          { status: 500 }
        );
      }
    }

    // Convert audio file to buffer
    const audioBuffer = Buffer.from(await audioFile.arrayBuffer());
    
    // Detect audio format from file type
    const fileName = audioFile.name.toLowerCase();
    const mimeType = audioFile.type.toLowerCase();
    
    let encoding: 'WEBM_OPUS' | 'LINEAR16' | 'MP3' | 'FLAC' | 'MULAW' | 'AMR' | 'AMR_WB' | 'OGG_OPUS' = 'WEBM_OPUS';
    let detectedSampleRate = sampleRate || 48000;
    
    // Auto-detect format
    if (mimeType.includes('webm') || mimeType.includes('opus') || fileName.endsWith('.webm')) {
      encoding = 'WEBM_OPUS';
      detectedSampleRate = sampleRate || 48000;
    } else if (mimeType.includes('wav') || fileName.endsWith('.wav')) {
      encoding = 'LINEAR16';
      detectedSampleRate = sampleRate || 16000;
    } else if (mimeType.includes('flac') || fileName.endsWith('.flac')) {
      encoding = 'FLAC';
      detectedSampleRate = sampleRate || 16000;
    } else if (mimeType.includes('mp3') || fileName.endsWith('.mp3')) {
      encoding = 'MP3';
      detectedSampleRate = sampleRate || 44100;
    }
    
    console.log('Audio format detected:', { encoding, sampleRate: detectedSampleRate, mimeType, fileName });

    // Import Google Cloud Speech dynamically (only on server)
    const { SpeechClient } = await import('@google-cloud/speech');
    
    // Initialize client with credentials
    const clientConfig: any = {};
    if (googleCredentials) {
      // Will use GOOGLE_APPLICATION_CREDENTIALS env var automatically
    } else if (googleProjectId) {
      clientConfig.projectId = googleProjectId;
    }
    
    const client = new SpeechClient(clientConfig);

    // Configure recognition with enhanced settings
    const config = {
      encoding: encoding,
      sampleRateHertz: detectedSampleRate,
      languageCode: languageCode,
      alternativeLanguageCodes: languageCode.startsWith('ta') ? ['en-US', 'hi-IN'] : ['en-US'], // Better fallbacks for Tamil
      enableAutomaticPunctuation: true,
      enableWordTimeOffsets: false, // Set to true if you need word-level timestamps
      model: 'latest_long', // Best for longer audio, or use 'latest_short' for <1min
      useEnhanced: true, // Use enhanced model if available
      enableSpokenPunctuation: true, // Better for Tamil
      enableSpokenEmojis: false,
    };

    const audio = {
      content: audioBuffer.toString('base64'),
    };

    const request_config = {
      config,
      audio,
    };

    console.log('Sending to Google STT:', { 
      languageCode, 
      encoding, 
      sampleRate: detectedSampleRate,
      size: `${(audioBuffer.length / 1024).toFixed(2)}KB`
    });

    // Perform speech recognition
    const [response] = await client.recognize(request_config);
    
    if (!response.results || response.results.length === 0) {
      console.log('No speech detected in audio');
      return NextResponse.json(
        { 
          transcript: '', 
          confidence: 0,
          message: 'No speech detected in audio'
        },
        { status: 200 }
      );
    }

    // Get the best result (most confident)
    const result = response.results[0];
    const transcript = result.alternatives[0]?.transcript || '';
    const confidence = result.alternatives[0]?.confidence || 0;

    // Get all alternatives if available (for better accuracy)
    const alternatives = result.alternatives?.map(alt => ({
      transcript: alt.transcript || '',
      confidence: alt.confidence || 0
    })) || [];

    console.log('✅ Google STT Result:', { 
      transcript: transcript.substring(0, 100), 
      confidence: confidence.toFixed(2),
      languageCode,
      alternatives: alternatives.length
    });

    return NextResponse.json({
      transcript,
      confidence,
      languageCode,
      alternatives: alternatives.slice(1), // Return other alternatives excluding the first (best) one
      detectedLanguage: result.languageCode || languageCode,
    });

  } catch (error: any) {
    console.error('❌ Google STT Error:', error);
    
    // Provide more helpful error messages
    let errorMessage = 'Speech recognition failed';
    let statusCode = 500;
    
    if (error.message?.includes('permission') || error.message?.includes('credentials')) {
      errorMessage = 'Google Cloud credentials error';
      statusCode = 401;
    } else if (error.message?.includes('quota') || error.message?.includes('limit')) {
      errorMessage = 'Google Cloud quota exceeded';
      statusCode = 429;
    } else if (error.message?.includes('invalid') || error.message?.includes('format')) {
      errorMessage = 'Invalid audio format';
      statusCode = 400;
    }
    
    return NextResponse.json(
      { 
        error: errorMessage,
        message: error.message,
        code: error.code,
        details: process.env.NODE_ENV === 'development' ? error.stack : undefined
      },
      { status: statusCode }
    );
  }
}

