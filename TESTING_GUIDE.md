# Testing Guide: Google TTS & STT for Tamil

## Quick Test Status

Run the test script to check setup:
```bash
npm run test-google
```

## Testing Options

### Option 1: Full Test (Requires Google Cloud Setup) ✅ Recommended

**Prerequisites:**
- Google Cloud account with billing enabled
- Service account key file
- APIs enabled

**Steps:**
1. **Install dependencies**
   ```bash
   npm install
   ```

2. **Configure Google Cloud**
   - See `GOOGLE_TTS_STT_SETUP.md` for detailed instructions
   - Set `GOOGLE_APPLICATION_CREDENTIALS` in `.env.local`

3. **Start dev server**
   ```bash
   npm run dev
   ```

4. **Test in browser**
   - Open http://localhost:3000
   - Select **Tamil** from language selector
   - Click microphone button
   - Speak in Tamil
   - See transcription appear
   - Response will be generated and played

### Option 2: Code Test (No Google Cloud Needed) ✅ Quick Check

Test that all code is properly integrated:

```bash
# Run test script
npm run test-google
```

**Expected Results:**
- ✅ Dependencies found
- ✅ API routes exist
- ✅ Components exist
- ✅ Integration checks pass
- ⚠️ Configuration needs setup (expected if credentials not set)

### Option 3: Mock Test (Development Testing)

You can test the UI flow without actual API calls:

1. **Start dev server**
   ```bash
   npm run dev
   ```

2. **Test UI Flow**
   - Select Tamil language
   - Verify Tamil mode indicator appears
   - Verify microphone button shows
   - Verify status indicators work

3. **Check Console Logs**
   - Open browser DevTools (F12)
   - Look for Tamil mode activation logs
   - Check for component mounting

## Testing Checklist

### Code Integration ✅
- [x] Dependencies installed
- [x] API routes created
- [x] Components created
- [x] AvatarVideo integration complete

### Google Cloud Setup ⚠️
- [ ] Google Cloud account created
- [ ] Billing enabled
- [ ] APIs enabled (TTS & STT)
- [ ] Service account created
- [ ] Credentials file downloaded
- [ ] Environment variable set

### Functional Testing
- [ ] Tamil language selection works
- [ ] Tamil mode indicator shows
- [ ] Microphone button appears for Tamil
- [ ] Google STT recording works
- [ ] Transcription appears
- [ ] Response generation works
- [ ] Google TTS playback works

## Test Scenarios

### Scenario 1: Speech-to-Text Test
1. Select Tamil
2. Click microphone
3. Speak: "வணக்கம், எப்படி இருக்கிறீர்கள்?"
4. Expected: Transcription appears in console/logs

### Scenario 2: Text-to-Speech Test
1. Select Tamil
2. Type: "வணக்கம்"
3. Submit (if text input visible)
4. Expected: Tamil audio plays

### Scenario 3: Full Conversation Test
1. Select Tamil
2. Click microphone
3. Speak a question
4. Expected: 
   - Transcription
   - Response generated
   - Tamil TTS plays response

## Browser Testing

### Chrome/Edge (Recommended)
- Best WebRTC support
- Full microphone API support

### Firefox
- Good support
- May need permission prompts

### Safari
- Limited WebRTC support
- May have issues with audio recording

## Troubleshooting Tests

### Test: Check Dependencies
```bash
npm list @google-cloud/speech @google-cloud/text-to-speech
```

### Test: Check Environment Variables
```bash
# Windows PowerShell
$env:GOOGLE_APPLICATION_CREDENTIALS

# Linux/Mac
echo $GOOGLE_APPLICATION_CREDENTIALS
```

### Test: API Route Directly
```bash
# Start dev server first
npm run dev

# Test STT endpoint (requires audio file)
curl -X POST http://localhost:3000/api/google/stt \
  -F "audio=@test-audio.webm" \
  -F "language=ta-IN"

# Test TTS endpoint
curl -X POST http://localhost:3000/api/google/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"வணக்கம்","languageCode":"ta-IN"}'
```

## Expected Test Results

### ✅ All Tests Pass
```
Configuration: ✅ Ready
Dependencies: 2/2 ✅
API Routes: 2/2 ✅
Components: 2/2 ✅
Integration: 7/7 ✅
```

### ⚠️ Partial Setup
```
Configuration: ⚠️ Needs Setup (credentials missing)
Dependencies: 2/2 ✅
API Routes: 2/2 ✅
Components: 2/2 ✅
Integration: 7/7 ✅
```

This is normal if you haven't set up Google Cloud yet.

## Quick Start Testing (Without Full Setup)

If you just want to verify the code works:

1. **Verify code integration:**
   ```bash
   npm run test-google
   ```

2. **Start dev server:**
   ```bash
   npm run dev
   ```

3. **Check browser:**
   - Tamil appears in language selector ✅
   - Selecting Tamil shows Tamil mode ✅
   - UI elements appear correctly ✅
   - Console shows Tamil mode logs ✅

4. **For actual STT/TTS:**
   - Configure Google Cloud credentials
   - Test with real API calls

## Test Files

- `scripts/test-google-tts-stt.js` - Main test script
- `GOOGLE_TTS_STT_SETUP.md` - Setup documentation
- `TESTING_GUIDE.md` - This file

## Support

If tests fail:
1. Check error messages in test output
2. Review `GOOGLE_TTS_STT_SETUP.md` for setup steps
3. Check browser console for runtime errors
4. Verify all dependencies installed: `npm install`

