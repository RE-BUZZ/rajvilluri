# Google TTS & STT Setup for Tamil Language

## Overview
This repository now supports Tamil language using Google Cloud Text-to-Speech (TTS) and Speech-to-Text (STT) APIs, since HeyGen SDK doesn't support Tamil natively.

## Architecture
- **Other Languages**: Use HeyGen SDK (as before)
- **Tamil Language**: Uses Google Cloud TTS & STT APIs

## Prerequisites

### 1. Google Cloud Account
- Create a Google Cloud account at https://cloud.google.com
- Enable billing (free tier available)

### 2. Enable APIs
Enable the following APIs in Google Cloud Console:
- **Cloud Text-to-Speech API**
- **Cloud Speech-to-Text API**

Commands:
```bash
gcloud services enable texttospeech.googleapis.com
gcloud services enable speech.googleapis.com
```

### 3. Create Service Account
1. Go to Google Cloud Console → IAM & Admin → Service Accounts
2. Click "Create Service Account"
3. Name it (e.g., "tamil-tts-stt")
4. Grant roles:
   - Cloud Text-to-Speech API User
   - Cloud Speech-to-Text API User
5. Create and download JSON key

### 4. Configuration

#### Option A: Service Account Key (Recommended for Production)
```bash
# Set environment variable pointing to service account JSON
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"

# Or in .env.local
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

#### Option B: API Key (For Development)
```bash
# In .env.local
GOOGLE_CLOUD_API_KEY=your_api_key_here
GOOGLE_CLOUD_PROJECT_ID=your-project-id
```

### 5. Install Dependencies
```bash
npm install
```

The following packages are already included:
- `@google-cloud/speech` - Speech-to-Text
- `@google-cloud/text-to-speech` - Text-to-Speech

## Usage

### Tamil Language Flow

1. **Select Tamil** from language selector
2. **Click microphone button** to start recording
3. **Speak in Tamil** - Google STT will transcribe
4. **Get response** - Generated using `generateTamilResponse()` function
5. **Hear Tamil TTS** - Google TTS will speak the response

### Code Structure

#### API Routes
- `/api/google/stt` - Speech-to-Text endpoint
- `/api/google/tts` - Text-to-Speech endpoint

#### Components
- `GoogleSTTRecorder.tsx` - Records audio and sends to Google STT
- `GoogleTTSPlayer.tsx` - Plays Tamil TTS audio
- `AvatarVideo.tsx` - Main component with Tamil integration

### Customizing Tamil Responses

Edit `generateTamilResponse()` function in `AvatarVideo.tsx`:

```typescript
const generateTamilResponse = async (userInput: string): Promise<string> => {
  // Replace with your AI service, LLM, or knowledge base
  // Examples:
  
  // 1. Simple echo
  return `நீங்கள் சொன்னது: ${userInput}`;
  
  // 2. Use OpenAI/Anthropic/etc
  // const response = await fetch('/api/ai-chat', {
  //   method: 'POST',
  //   body: JSON.stringify({ message: userInput, language: 'ta' })
  // });
  // return await response.json();
  
  // 3. Use knowledge base
  // return await queryKnowledgeBase(userInput);
};
```

## Testing

### Test Tamil STT
1. Start development server: `npm run dev`
2. Select Tamil from language selector
3. Click microphone button
4. Speak in Tamil
5. Check browser console for transcript

### Test Tamil TTS
1. Type text in input field (if visible)
2. Submit - TTS should play automatically
3. Check browser console for audio generation logs

## Cost Considerations

### Google Cloud Pricing (as of 2024)
- **Speech-to-Text**: 
  - First 60 minutes/month: Free
  - After: $0.006 per 15 seconds
- **Text-to-Speech**:
  - First 4 million characters/month: Free
  - After: $4-$16 per 1 million characters (depends on voice)

### Tips to Reduce Costs
- Use streaming recognition efficiently
- Cache common responses
- Implement rate limiting
- Monitor usage in Google Cloud Console

## Troubleshooting

### Error: "Google Cloud credentials not configured"
- Check `GOOGLE_APPLICATION_CREDENTIALS` is set
- Verify service account JSON file path is correct
- Ensure JSON file has proper permissions

### Error: "API not enabled"
- Enable Text-to-Speech and Speech-to-Text APIs in Google Cloud Console

### Error: "Microphone permission denied"
- Check browser permissions for microphone access
- Allow microphone access when prompted

### Tamil TTS not playing
- Check browser console for errors
- Verify audio codec support (MP3)
- Check network tab for API responses

### Tamil STT not working
- Verify audio format (WebM Opus)
- Check microphone is working
- Review API response in network tab

## Environment Variables

Add to `.env.local`:

```env
# Option 1: Service Account (Recommended)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# Option 2: API Key (Alternative)
GOOGLE_CLOUD_API_KEY=your_api_key_here
GOOGLE_CLOUD_PROJECT_ID=your-project-id

# HeyGen (for other languages)
HEYGEN_API_KEY=your_heygen_key
```

## Tamil Voice Options

Google Cloud supports multiple Tamil voices:
- `ta-IN-Standard-A` - Female, Standard
- `ta-IN-Standard-B` - Male, Standard
- `ta-IN-Wavenet-A` - Female, Neural (better quality)
- `ta-IN-Wavenet-B` - Male, Neural (better quality)

To change voice, modify `GoogleTTSPlayer.tsx` or pass `voiceName` in API call.

## Next Steps

1. ✅ Install dependencies: `npm install`
2. ✅ Set up Google Cloud credentials
3. ✅ Test Tamil STT recording
4. ✅ Test Tamil TTS playback
5. ⚠️ Customize `generateTamilResponse()` with your AI/knowledge base
6. ⚠️ Deploy with proper environment variables

## Support

- Google Cloud Speech-to-Text Docs: https://cloud.google.com/speech-to-text/docs
- Google Cloud Text-to-Speech Docs: https://cloud.google.com/text-to-speech/docs
- HeyGen SDK Docs: https://docs.heygen.com (for other languages)

