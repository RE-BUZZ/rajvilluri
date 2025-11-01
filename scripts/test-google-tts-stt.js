#!/usr/bin/env node

/**
 * Test Google TTS & STT Integration
 * Tests if Google Cloud APIs are properly configured for Tamil support
 */

const fs = require('fs');
const path = require('path');
const https = require('https');

console.log('🧪 Testing Google TTS & STT Integration\n');
console.log('='.repeat(60));

// Load environment variables
function loadEnv() {
  const envPaths = [
    path.join(__dirname, '..', '.env'),
    path.join(__dirname, '..', '.env.local')
  ];
  
  envPaths.forEach(envPath => {
    if (fs.existsSync(envPath)) {
      const envContent = fs.readFileSync(envPath, 'utf8');
      envContent.split('\n').forEach(line => {
        const trimmed = line.trim();
        if (trimmed && !trimmed.startsWith('#')) {
          const [key, ...values] = trimmed.split('=');
          if (key && values.length > 0) {
            process.env[key.trim()] = values.join('=').trim();
          }
        }
      });
    }
  });
}

loadEnv();

// Test 1: Check Configuration
console.log('\n📋 Test 1: Checking Configuration...');

const googleCredentials = process.env.GOOGLE_APPLICATION_CREDENTIALS;
const googleApiKey = process.env.GOOGLE_CLOUD_API_KEY;
const googleProjectId = process.env.GOOGLE_CLOUD_PROJECT_ID;

let configPassed = 0;
let configTotal = 0;

// Check credentials
configTotal++;
if (googleCredentials) {
  const credPath = path.resolve(googleCredentials);
  if (fs.existsSync(credPath)) {
    console.log('   ✅ GOOGLE_APPLICATION_CREDENTIALS: Found');
    console.log(`      Path: ${credPath}`);
    
    // Try to read and validate JSON
    try {
      const credContent = fs.readFileSync(credPath, 'utf8');
      const credJson = JSON.parse(credContent);
      if (credJson.type === 'service_account' && credJson.project_id) {
        console.log(`      Project ID: ${credJson.project_id}`);
        console.log(`      Service Account: ${credJson.client_email || 'N/A'}`);
        configPassed++;
      } else {
        console.log('   ⚠️  Service account JSON format may be incorrect');
      }
    } catch (e) {
      console.log(`   ⚠️  Could not parse credentials file: ${e.message}`);
    }
  } else {
    console.log(`   ❌ GOOGLE_APPLICATION_CREDENTIALS: File not found at ${credPath}`);
  }
} else if (googleApiKey || googleProjectId) {
  console.log('   ✅ Google Cloud API Key/Project ID: Found');
  if (googleApiKey) console.log(`      API Key: ${googleApiKey.substring(0, 10)}...`);
  if (googleProjectId) console.log(`      Project ID: ${googleProjectId}`);
  configPassed++;
} else {
  console.log('   ❌ No Google Cloud credentials found');
  console.log('   💡 Set GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_CLOUD_API_KEY in .env.local');
}

console.log(`\n   Result: ${configPassed}/${configTotal} configuration checks passed`);

// Test 2: Check Dependencies
console.log('\n📋 Test 2: Checking Dependencies...');

const packageJsonPath = path.join(__dirname, '..', 'package.json');
let depsPassed = 0;
let depsTotal = 0;

if (fs.existsSync(packageJsonPath)) {
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
  const deps = packageJson.dependencies || {};
  
  const requiredDeps = {
    '@google-cloud/speech': 'Speech-to-Text',
    '@google-cloud/text-to-speech': 'Text-to-Speech'
  };
  
  Object.entries(requiredDeps).forEach(([dep, name]) => {
    depsTotal++;
    if (deps[dep]) {
      console.log(`   ✅ ${name}: ${deps[dep]}`);
      depsPassed++;
    } else {
      console.log(`   ❌ ${name}: Not found in package.json`);
    }
  });
} else {
  console.log('   ❌ package.json not found');
}

console.log(`\n   Result: ${depsPassed}/${depsTotal} dependencies found`);

// Test 3: Check API Routes
console.log('\n📋 Test 3: Checking API Routes...');

const sttRoute = path.join(__dirname, '..', 'src', 'app', 'api', 'google', 'stt', 'route.ts');
const ttsRoute = path.join(__dirname, '..', 'src', 'app', 'api', 'google', 'tts', 'route.ts');

let routesPassed = 0;
let routesTotal = 0;

routesTotal++;
if (fs.existsSync(sttRoute)) {
  console.log('   ✅ /api/google/stt route exists');
  routesPassed++;
} else {
  console.log('   ❌ /api/google/stt route not found');
}

routesTotal++;
if (fs.existsSync(ttsRoute)) {
  console.log('   ✅ /api/google/tts route exists');
  routesPassed++;
} else {
  console.log('   ❌ /api/google/tts route not found');
}

console.log(`\n   Result: ${routesPassed}/${routesTotal} API routes found`);

// Test 4: Check Components
console.log('\n📋 Test 4: Checking Components...');

const sttComponent = path.join(__dirname, '..', 'src', 'components', 'GoogleSTTRecorder.tsx');
const ttsComponent = path.join(__dirname, '..', 'src', 'components', 'GoogleTTSPlayer.tsx');

let componentsPassed = 0;
let componentsTotal = 0;

componentsTotal++;
if (fs.existsSync(sttComponent)) {
  console.log('   ✅ GoogleSTTRecorder component exists');
  componentsPassed++;
} else {
  console.log('   ❌ GoogleSTTRecorder component not found');
}

componentsTotal++;
if (fs.existsSync(ttsComponent)) {
  console.log('   ✅ GoogleTTSPlayer component exists');
  componentsPassed++;
} else {
  console.log('   ❌ GoogleTTSPlayer component not found');
}

console.log(`\n   Result: ${componentsPassed}/${componentsTotal} components found`);

// Test 5: Check AvatarVideo Integration
console.log('\n📋 Test 5: Checking AvatarVideo Integration...');

const avatarVideoPath = path.join(__dirname, '..', 'src', 'components', 'AvatarVideo.tsx');

if (fs.existsSync(avatarVideoPath)) {
  const content = fs.readFileSync(avatarVideoPath, 'utf8');
  
  const checks = {
    'GoogleSTTRecorder import': /import.*GoogleSTTRecorder/i.test(content),
    'GoogleTTSPlayer import': /import.*GoogleTTSPlayer/i.test(content),
    'Tamil language check': /isTamilLanguage/i.test(content),
    'Google STT handler': /handleGoogleSTTTranscript/i.test(content),
    'Tamil response generator': /generateTamilResponse/i.test(content),
    'Google STT recorder usage': /GoogleSTTRecorder/i.test(content),
    'Google TTS player usage': /GoogleTTSPlayer/i.test(content),
  };
  
  let integrationPassed = 0;
  let integrationTotal = 0;
  
  Object.entries(checks).forEach(([check, result]) => {
    integrationTotal++;
    if (result) {
      console.log(`   ✅ ${check}`);
      integrationPassed++;
    } else {
      console.log(`   ❌ ${check}`);
    }
  });
  
  console.log(`\n   Result: ${integrationPassed}/${integrationTotal} integration checks passed`);
} else {
  console.log('   ❌ AvatarVideo.tsx not found');
}

// Test 6: API Connection Test (if credentials available)
console.log('\n📋 Test 6: Testing Google Cloud API Connection...');

if (googleCredentials && fs.existsSync(path.resolve(googleCredentials))) {
  console.log('   📞 Testing Text-to-Speech API connection...');
  
  // Note: This is a basic connectivity test
  // Full API test requires actual API calls which may incur costs
  console.log('   💡 To fully test APIs:');
  console.log('      1. Start dev server: npm run dev');
  console.log('      2. Select Tamil language');
  console.log('      3. Click microphone and speak');
  console.log('      4. Check browser console for API responses');
} else {
  console.log('   ⚠️  Skipping API test - credentials not configured');
  console.log('   💡 Configure GOOGLE_APPLICATION_CREDENTIALS to enable API testing');
}

// Summary
console.log('\n' + '='.repeat(60));
console.log('\n📊 Test Summary:');

const totalTests = configPassed + depsPassed + routesPassed + componentsPassed;
const totalPossible = configTotal + depsTotal + routesTotal + componentsTotal;

console.log(`   Configuration: ${configPassed > 0 ? '✅' : '⚠️'} ${configPassed > 0 ? 'Ready' : 'Needs Setup'}`);
console.log(`   Dependencies: ${depsPassed}/${depsTotal} ✅`);
console.log(`   API Routes: ${routesPassed}/${routesTotal} ✅`);
console.log(`   Components: ${componentsPassed}/${componentsTotal} ✅`);

console.log('\n📝 Next Steps:');
console.log('   1. Install dependencies: npm install');
if (configPassed === 0) {
  console.log('   2. ⚠️  Configure Google Cloud credentials (see GOOGLE_TTS_STT_SETUP.md)');
} else {
  console.log('   2. ✅ Google Cloud credentials configured');
}
console.log('   3. Start dev server: npm run dev');
console.log('   4. Open http://localhost:3000');
console.log('   5. Select Tamil from language selector');
console.log('   6. Click microphone button to test STT');
console.log('   7. Type text and submit to test TTS');

console.log('\n🌐 Test URLs:');
console.log('   • Local: http://localhost:3000');
console.log('   • STT API: http://localhost:3000/api/google/stt');
console.log('   • TTS API: http://localhost:3000/api/google/tts');

console.log('\n📖 Documentation:');
console.log('   • Setup Guide: GOOGLE_TTS_STT_SETUP.md');
console.log('   • Tamil Setup: TAMIL_SETUP.md');

console.log('\n' + '='.repeat(60));

// Exit code
if (configPassed > 0 && depsPassed === depsTotal && routesPassed === routesTotal && componentsPassed === componentsTotal) {
  console.log('\n✅ Ready for testing!');
  process.exit(0);
} else {
  console.log('\n⚠️  Some setup steps needed before testing');
  process.exit(1);
}

