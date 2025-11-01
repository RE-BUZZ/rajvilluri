# Vercel Deployment Guide

## 🚀 Deploy to Vercel

### Repository Information
- **GitHub Repository:** `RE-BUZZ/sobha_dev`
- **URL:** https://github.com/RE-BUZZ/sobha_dev

### Quick Deploy Steps

#### Option 1: Deploy via Vercel Dashboard (Recommended)

1. **Go to Vercel:**
   - Visit: https://vercel.com
   - Sign in with your GitHub account

2. **Import Project:**
   - Click "Add New..." → "Project"
   - Find and select: `RE-BUZZ/sobha_dev`
   - Click "Import"

3. **Configure Build Settings:**
   - **Framework Preset:** Next.js (auto-detected)
   - **Root Directory:** `./` (default)
   - **Build Command:** `npm run build` (default)
   - **Output Directory:** `.next` (default)
   - **Install Command:** `npm install` (default)

4. **Add Environment Variables:**
   Click "Environment Variables" and add:

   ```
   Name: HEYGEN_API_KEY
   Value: sk_V2_hgu_kgNq3ZDoeYw_H18hzlKm68SJbUYs7LmzCA2MyFJphEjA
   Environment: Production, Preview, Development
   ```

   ```
   Name: GOOGLE_CLOUD_API_KEY
   Value: AQ.Ab8RN6JX4NXFGaMU4LphXKszjmxcZ1D_EHh53OsQc86fH8EYUw
   Environment: Production, Preview, Development
   ```

5. **Deploy:**
   - Click "Deploy"
   - Wait for build to complete (~2-5 minutes)
   - Your app will be live at: `https://your-project-name.vercel.app`

#### Option 2: Deploy via Vercel CLI

1. **Login to Vercel:**
   ```bash
   vercel login
   ```

2. **Deploy:**
   ```bash
   vercel
   ```

3. **Set Environment Variables:**
   ```bash
   vercel env add HEYGEN_API_KEY
   vercel env add GOOGLE_CLOUD_API_KEY
   ```

4. **Deploy to Production:**
   ```bash
   vercel --prod
   ```

### Environment Variables Required

| Variable | Value | Description |
|----------|-------|-------------|
| `HEYGEN_API_KEY` | `sk_V2_hgu_kgNq3ZDoeYw_H18hzlKm68SJbUYs7LmzCA2MyFJphEjA` | HeyGen API key for avatar streaming |
| `GOOGLE_CLOUD_API_KEY` | `AQ.Ab8RN6JX4NXFGaMU4LphXKszjmxcZ1D_EHh53OsQc86fH8EYUw` | Google Cloud API key for Tamil TTS/STT |

### Important Notes

⚠️ **Google Cloud Libraries:**
- The `@google-cloud/speech` and `@google-cloud/text-to-speech` libraries may require service account JSON files
- If you encounter credential issues, you may need to:
  1. Create a service account in Google Cloud Console
  2. Download the JSON key file
  3. Convert it to an environment variable or use Vercel's file upload feature
  4. See `GOOGLE_TTS_STT_SETUP.md` for detailed instructions

### Post-Deployment Testing

1. **Test English (HeyGen):**
   - Open your Vercel URL
   - Select English language
   - Click "Start Session"
   - Test voice chat

2. **Test Tamil (Google TTS/STT):**
   - Select Tamil language
   - Click microphone button
   - Speak in Tamil
   - Verify response in Tamil

### Troubleshooting

**Build Fails:**
- Check build logs in Vercel dashboard
- Ensure all dependencies are in `package.json`
- Verify Node.js version compatibility

**API Errors:**
- Verify environment variables are set correctly
- Check API keys are valid
- Review Vercel function logs

**Google Cloud Errors:**
- May need service account JSON instead of API key
- Check Google Cloud Console for quota/billing issues
- Enable required APIs (TTS, STT) in Google Cloud Console

### Vercel Configuration

The `vercel.json` file is already configured with:
- Next.js framework preset
- Build commands
- Environment variable references

### Deployment URL

After successful deployment, your app will be available at:
- Production: `https://your-project-name.vercel.app`
- Preview: `https://your-project-name-git-branch-username.vercel.app`

---

**Ready to deploy!** Follow the steps above to get your app live on Vercel. 🚀

