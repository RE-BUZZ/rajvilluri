# Deploy KON_Real_Estate to Vercel

## 🚀 Quick Deploy Steps

### Method 1: Vercel Dashboard (Recommended - Easiest)

1. **Go to Vercel Dashboard:**
   - Visit: https://vercel.com
   - Sign in with GitHub

2. **Import Project:**
   - Click **"Add New..."** → **"Project"**
   - Select repository: **`RE-BUZZ/sobha_dev`**
   - Click **"Import"**

3. **Configure Project:**
   - **Project Name:** `KON_Real_Estate`
   - **Framework Preset:** Next.js (auto-detected)
   - **Root Directory:** `./`
   - **Build Command:** `npm run build`
   - **Output Directory:** `.next`

4. **Environment Variables:**
   Before deploying, click **"Environment Variables"** and add:

   ```
   Name: HEYGEN_API_KEY
   Value: sk_V2_hgu_kgNq3ZDoeYw_H18hzlKm68SJbUYs7LmzCA2MyFJphEjA
   ✅ Production
   ✅ Preview  
   ✅ Development
   ```

   ```
   Name: GOOGLE_CLOUD_API_KEY
   Value: AQ.Ab8RN6JX4NXFGaMU4LphXKszjmxcZ1D_EHh53OsQc86fH8EYUw
   ✅ Production
   ✅ Preview
   ✅ Development
   ```

5. **Deploy:**
   - Click **"Deploy"** button
   - Wait 2-5 minutes for build
   - Your app will be live!

### Method 2: Vercel CLI

**Step 1: Authenticate**
```bash
vercel login
```
- Press Enter to open browser
- Complete authentication in browser

**Step 2: Link Project**
```bash
vercel link
```
- When prompted for project name, enter: `KON_Real_Estate`

**Step 3: Set Environment Variables**
```bash
vercel env add HEYGEN_API_KEY
# Paste: sk_V2_hgu_kgNq3ZDoeYw_H18hzlKm68SJbUYs7LmzCA2MyFJphEjA
# Select: Production, Preview, Development

vercel env add GOOGLE_CLOUD_API_KEY
# Paste: AQ.Ab8RN6JX4NXFGaMU4LphXKszjmxcZ1D_EHh53OsQc86fH8EYUw
# Select: Production, Preview, Development
```

**Step 4: Deploy**
```bash
vercel --prod
```

## 📋 Project Information

- **Repository:** `RE-BUZZ/sobha_dev`
- **Project Name:** `KON_Real_Estate`
- **Framework:** Next.js 15.4.5
- **Node Version:** Auto-detected (20+)

## ✅ Post-Deployment Checklist

After deployment completes:

1. **Test English (HeyGen):**
   - Visit your Vercel URL
   - Select English → Start Session
   - Test voice chat

2. **Test Tamil (Google TTS/STT):**
   - Select Tamil
   - Click microphone
   - Speak in Tamil

## 🔗 Expected URLs

After deployment, your app will be available at:
- **Production:** `https://kon-real-estate.vercel.app`
- **Custom Domain:** (Can be added in Vercel dashboard)

## ⚠️ Important Notes

1. **Google Cloud Libraries:**
   - May require service account JSON file
   - If errors occur, see `GOOGLE_TTS_STT_SETUP.md`
   - May need to enable APIs in Google Cloud Console

2. **Build Time:**
   - First deployment: ~3-5 minutes
   - Subsequent: ~1-2 minutes

3. **Environment Variables:**
   - Must be set before first deploy
   - Can be updated later in dashboard

---

**Ready to deploy!** Use Method 1 (Dashboard) for easiest deployment. 🚀

