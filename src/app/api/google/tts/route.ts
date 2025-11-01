import { NextRequest, NextResponse } from 'next/server';

/**
 * Google Cloud Text-to-Speech API Route (Enhanced)
 * Converts Tamil text to speech audio with improved voice selection and configuration
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { 
      text, 
      languageCode = 'ta-IN', 
      voiceName, 
      ssmlGender = 'NEUTRAL',
      speakingRate = 1.0, // 0.25 to 4.0
      pitch = 0, // -20.0 to 20.0 semitones
      volumeGainDb = 0, // -96.0 to 16.0 dB
      audioEncoding = 'MP3' as 'MP3' | 'LINEAR16' | 'OGG_OPUS'
    } = body;

    // Validate text input
    if (!text || text.trim().length === 0) {
      return NextResponse.json(
        { error: 'No text provided' },
        { status: 400 }
      );
    }

    // Validate text length (Google limit is ~5000 characters)
    if (text.length > 5000) {
      return NextResponse.json(
        { error: 'Text too long. Maximum 5000 characters allowed' },
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

    // Import Google Cloud Text-to-Speech dynamically
    const { TextToSpeechClient } = await import('@google-cloud/text-to-speech');
    
    // Initialize client with credentials
    const clientConfig: any = {};
    if (googleCredentials) {
      // Will use GOOGLE_APPLICATION_CREDENTIALS env var automatically
    } else if (googleProjectId) {
      clientConfig.projectId = googleProjectId;
    }
    
    const client = new TextToSpeechClient(clientConfig);

    // Select Tamil voice with smart fallback
    let selectedVoiceName = voiceName;
    let selectedGender: 'NEUTRAL' | 'MALE' | 'FEMALE' = ssmlGender as 'NEUTRAL' | 'MALE' | 'FEMALE';
    
    if (!selectedVoiceName) {
      try {
        // List available Tamil voices
        const [voices] = await client.listVoices({
          languageCode: languageCode,
        });

        // Filter Tamil voices
        const tamilVoices = (voices.voices || []).filter(
          (voice) => voice.languageCodes?.includes(languageCode)
        );

        if (tamilVoices.length === 0) {
          // Fallback to default Tamil voices if API doesn't return any
          const defaultVoices = [
            'ta-IN-Wavenet-A', // Neural female (best quality)
            'ta-IN-Wavenet-B', // Neural male (best quality)
            'ta-IN-Standard-A', // Standard female
            'ta-IN-Standard-B', // Standard male
          ];
          
          selectedVoiceName = defaultVoices[0];
          console.log('Using default Tamil voice:', selectedVoiceName);
        } else {
          // Prefer neural/Wavenet voices for better quality
          const neuralVoice = tamilVoices.find(
            (voice) => voice.name?.includes('neural') || 
                       voice.name?.includes('Wavenet') ||
                       voice.name?.includes('wavenet')
          );

          // If gender specified, try to match
          if (ssmlGender !== 'NEUTRAL') {
            const genderMatch = tamilVoices.find(voice => {
              const name = voice.name?.toLowerCase() || '';
              if (ssmlGender === 'FEMALE' && (name.includes('a') || name.includes('female'))) return true;
              if (ssmlGender === 'MALE' && (name.includes('b') || name.includes('male'))) return true;
              return false;
            });
            
            if (genderMatch) {
              selectedVoiceName = genderMatch.name || tamilVoices[0].name || '';
              selectedGender = genderMatch.ssmlGender as 'NEUTRAL' | 'MALE' | 'FEMALE' || ssmlGender as 'NEUTRAL' | 'MALE' | 'FEMALE';
            } else {
              selectedVoiceName = neuralVoice?.name || tamilVoices[0].name || 'ta-IN-Standard-A';
            }
          } else {
            selectedVoiceName = neuralVoice?.name || tamilVoices[0].name || 'ta-IN-Standard-A';
          }

          console.log(`Found ${tamilVoices.length} Tamil voices, selected:`, selectedVoiceName);
        }
      } catch (voiceError: any) {
        console.warn('Failed to list voices, using default:', voiceError.message);
        // Use known good default Tamil voices
        selectedVoiceName = 'ta-IN-Wavenet-A'; // Best quality Tamil voice
      }
    }

    // Validate audio encoding
    const validEncodings = ['MP3', 'LINEAR16', 'OGG_OPUS'];
    const encoding = validEncodings.includes(audioEncoding) ? audioEncoding : 'MP3';

    // Validate and clamp parameters
    const validatedRate = Math.max(0.25, Math.min(4.0, speakingRate));
    const validatedPitch = Math.max(-20.0, Math.min(20.0, pitch));
    const validatedVolume = Math.max(-96.0, Math.min(16.0, volumeGainDb));

    // Configure TTS request with enhanced settings
    const request_config = {
      input: { text: text.trim() },
      voice: {
        languageCode,
        name: selectedVoiceName,
        ssmlGender: selectedGender,
      },
      audioConfig: {
        audioEncoding: encoding as 'MP3' | 'LINEAR16' | 'OGG_OPUS',
        speakingRate: validatedRate,
        pitch: validatedPitch,
        volumeGainDb: validatedVolume,
        effectsProfileId: ['telephony-class-application'], // Optimize for phone/voice chat
      },
    };

    console.log('Sending to Google TTS:', { 
      textLength: text.length, 
      languageCode, 
      voiceName: selectedVoiceName,
      encoding,
      speakingRate: validatedRate
    });

    // Perform TTS
    const [response] = await client.synthesizeSpeech(request_config);
    
    if (!response.audioContent) {
      return NextResponse.json(
        { error: 'Failed to generate audio from Google TTS' },
        { status: 500 }
      );
    }

    // Convert audio content to base64 for transmission
    const audioContent = response.audioContent as Uint8Array;
    const audioBase64 = Buffer.from(audioContent).toString('base64');
    const audioSizeKB = (audioContent.length / 1024).toFixed(2);

    console.log('✅ Google TTS Success:', { 
      textLength: text.length, 
      languageCode, 
      voiceName: selectedVoiceName,
      audioSize: `${audioSizeKB}KB`,
      encoding
    });

    return NextResponse.json({
      audioContent: audioBase64,
      audioEncoding: encoding,
      languageCode,
      voiceName: selectedVoiceName,
      audioSize: parseInt(audioSizeKB),
      textLength: text.length,
    });

  } catch (error: any) {
    console.error('❌ Google TTS Error:', error);
    
    // Provide more helpful error messages
    let errorMessage = 'Text-to-speech failed';
    let statusCode = 500;
    
    if (error.message?.includes('permission') || error.message?.includes('credentials')) {
      errorMessage = 'Google Cloud credentials error';
      statusCode = 401;
    } else if (error.message?.includes('quota') || error.message?.includes('limit')) {
      errorMessage = 'Google Cloud quota exceeded';
      statusCode = 429;
    } else if (error.message?.includes('invalid') || error.message?.includes('text')) {
      errorMessage = 'Invalid text input';
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

