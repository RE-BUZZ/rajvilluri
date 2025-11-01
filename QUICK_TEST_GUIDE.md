# Quick Test Guide

## 🚀 Application is Ready for Testing!

### Test URL
**http://localhost:3000**

### Quick Start
1. **Make sure dev server is running:**
   ```bash
   npm run dev
   ```

2. **Open in browser:**
   ```
   http://localhost:3000
   ```

### Testing Checklist

#### ✅ Test English (HeyGen)
1. Select **English** from language selector
2. Click **"Start Session"** button
3. Wait for avatar to load
4. Click microphone button for voice chat
5. Speak in English
6. Verify avatar responds

#### ✅ Test Tamil (Google TTS & STT)
1. Select **Tamil** from language selector
2. You'll see "Tamil Mode Active" message
3. Click **microphone button**
4. Speak in Tamil
5. Google STT will transcribe your speech
6. Response will be generated
7. Google TTS will speak the response in Tamil

### Status Indicators

**Top left of video:**
- 🟢 **Live** - Connection active
- 🔵 **Speaking** - Avatar is talking
- 🟣 **Listening (Tamil)** - Google STT active for Tamil

### API Endpoints (for testing)

- **HeyGen Token:** `POST http://localhost:3000/api/generate-token`
- **Google TTS:** `POST http://localhost:3000/api/google/tts`
- **Google STT:** `POST http://localhost:3000/api/google/stt`

### Troubleshooting

**Server not starting?**
- Check if port 3000 is in use
- Check for errors in terminal
- Verify all dependencies installed: `npm install`

**HeyGen not working?**
- Check `HEYGEN_API_KEY` in `.env.local`
- Verify API key is correct

**Tamil not working?**
- Check `GOOGLE_CLOUD_API_KEY` in `.env.local`
- May need service account JSON instead of API key
- See `GOOGLE_TTS_STT_SETUP.md` for full setup

### Browser Console

Open DevTools (F12) to see:
- Language changes
- API calls
- Error messages
- Debug logs

### Expected Behavior

**English Mode:**
- HeyGen avatar appears
- Video stream works
- Voice chat functional

**Tamil Mode:**
- No HeyGen session needed
- Google STT records Tamil speech
- Google TTS speaks Tamil responses
- All in Tamil language!

---

**Ready to test!** Open http://localhost:3000 and start testing! 🎉

