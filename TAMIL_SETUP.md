# Tamil Language Setup for HeyGen SDK

## Overview
Tamil language support has been added to the HeyGen SDK integration. When Tamil is selected, the system will automatically use a Tamil-compatible voice ID.

## How to Find Tamil Voice IDs

### Method 1: Using HeyGen API
1. Use the HeyGen API endpoint to list voices:
   ```bash
   GET https://api.heygen.com/v2/voices?language=ta
   ```

2. You'll need to authenticate with your HeyGen API key:
   ```bash
   curl -X GET "https://api.heygen.com/v2/voices?language=ta" \
     -H "X-API-KEY: YOUR_API_KEY"
   ```

3. The response will contain voice objects with `voice_id` and metadata showing supported languages.

### Method 2: Using HeyGen Dashboard
1. Go to [HeyGen Labs](https://labs.heygen.com/interactive-avatar)
2. Navigate to Voice settings
3. Filter by language: Tamil (ta)
4. Copy the Voice ID for Tamil-compatible voices

### Method 3: Check Voice Metadata
Use the List Voices API endpoint:
```
GET https://api.heygen.com/v2/voices
```

Then filter the results for voices where:
- `language` includes `'ta'` (Tamil)
- Or `supported_languages` array includes `'ta'`

## Updating the Code

Once you have a Tamil voice ID, update `src/components/AvatarVideo.tsx`:

Find this section around line 98:
```typescript
'ta': defaultVoice,  // Tamil - REPLACE 'default' with Tamil-compatible voice ID
```

Replace it with:
```typescript
'ta': 'YOUR_TAMIL_VOICE_ID_HERE',  // Replace with actual Tamil voice ID from HeyGen
```

For example:
```typescript
'ta': 'abc123-def456-ghi789',  // Tamil-compatible voice ID
```

## Example Voice ID Format
HeyGen voice IDs typically look like:
- UUID format: `95856005-0332-41b0-935f-352e296aa0df`
- Or custom IDs provided by HeyGen

## Testing
After updating the voice ID:
1. Restart your development server
2. Select Tamil from the language selector
3. The avatar should use the Tamil voice when speaking
4. Check browser console for language/voice configuration logs

## Additional Notes
- The language code for Tamil is `'ta'`
- Make sure the voice ID you use actually supports Tamil language
- Some voices may support multiple languages - verify this in the voice metadata
- The system will automatically switch to the Tamil voice when Tamil is selected

## Troubleshooting
If Tamil voice doesn't work:
1. Verify the voice ID is correct
2. Check that the voice supports Tamil language
3. Review browser console for error messages
4. Check HeyGen API documentation for latest voice IDs: https://docs.heygen.com/reference/list-voices-v2


