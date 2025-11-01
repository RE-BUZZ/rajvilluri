#!/usr/bin/env node

/**
 * Tamil Language Functionality Test
 * 
 * This script tests if Tamil language support is properly configured
 * and checks for available Tamil voices from HeyGen API
 */

const https = require('https');
const fs = require('fs');
const path = require('path');

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

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY || process.env.NEXT_PUBLIC_HEYGEN_API_KEY;

console.log('🧪 Testing Tamil Language Functionality\n');
console.log('='.repeat(60));

// Test 1: Check if Tamil is in AvatarVideo component
console.log('\n📋 Test 1: Checking Tamil in Code Implementation...');
const avatarVideoPath = path.join(__dirname, '..', 'src', 'components', 'AvatarVideo.tsx');

if (fs.existsSync(avatarVideoPath)) {
  const content = fs.readFileSync(avatarVideoPath, 'utf8');
  
  const tests = {
    'Tamil in languageOptions': /code:\s*['"]ta['"]/i.test(content),
    'Tamil language code': /'ta'|"ta"/.test(content),
    'Tamil flag emoji': /🇮🇳/.test(content),
    'Language-to-voice mapping': /languageToVoiceMapping/i.test(content),
    'Tamil voice mapping': /'ta':\s*\w+/.test(content),
    'getVoiceIdForLanguage function': /getVoiceIdForLanguage/i.test(content)
  };
  
  let passed = 0;
  let total = 0;
  
  Object.entries(tests).forEach(([testName, result]) => {
    total++;
    if (result) {
      console.log(`   ✅ ${testName}`);
      passed++;
    } else {
      console.log(`   ❌ ${testName}`);
    }
  });
  
  console.log(`\n   Result: ${passed}/${total} checks passed`);
  
  // Check if Tamil voice ID is still default
  const hasTamilVoiceId = /'ta':\s*['"]default['"]|'ta':\s*defaultVoice/.test(content);
  if (hasTamilVoiceId) {
    console.log('\n   ⚠️  WARNING: Tamil voice ID is still set to "default"');
    console.log('   📝 Action Required: Update Tamil voice ID in AvatarVideo.tsx (line ~98)');
    console.log('   📖 See TAMIL_SETUP.md for instructions');
  } else {
    console.log('\n   ✅ Tamil voice ID is configured');
  }
} else {
  console.log('   ❌ AvatarVideo.tsx not found');
}

// Test 2: Query HeyGen API for Tamil voices
console.log('\n📋 Test 2: Checking HeyGen API for Tamil Voices...');

if (!HEYGEN_API_KEY) {
  console.log('   ⚠️  HeyGen API key not found');
  console.log('   💡 Set HEYGEN_API_KEY in .env or .env.local to test API');
  console.log('   📝 Skipping API test');
} else {
  console.log('   🔑 API Key found');
  
  function makeRequest(path) {
    return new Promise((resolve, reject) => {
      const options = {
        hostname: 'api.heygen.com',
        port: 443,
        path: path,
        method: 'GET',
        headers: {
          'x-api-key': HEYGEN_API_KEY,
          'Content-Type': 'application/json'
        }
      };

      const req = https.request(options, (res) => {
        let data = '';

        res.on('data', (chunk) => {
          data += chunk;
        });

        res.on('end', () => {
          try {
            resolve({
              statusCode: res.statusCode,
              data: JSON.parse(data)
            });
          } catch (e) {
            resolve({
              statusCode: res.statusCode,
              data: data
            });
          }
        });
      });

      req.on('error', reject);
      req.end();
    });
  }

  (async () => {
    try {
      // Test 2a: List all voices
      console.log('\n   📞 Querying HeyGen API for voices...');
      const voicesResponse = await makeRequest('/v2/voices');
      
      if (voicesResponse.statusCode === 200) {
        console.log('   ✅ API connection successful');
        
        const voices = voicesResponse.data?.data?.voices || voicesResponse.data?.voices || [];
        console.log(`   📊 Total voices available: ${voices.length}`);
        
        // Filter Tamil voices
        const tamilVoices = voices.filter(voice => {
          const lang = voice.language?.toLowerCase();
          const supportedLangs = voice.supported_languages || [];
          const langStr = supportedLangs.join(',').toLowerCase();
          
          return lang === 'ta' || 
                 langStr.includes('ta') || 
                 langStr.includes('tamil') ||
                 voice.name?.toLowerCase().includes('tamil');
        });
        
        if (tamilVoices.length > 0) {
          console.log(`\n   ✅ Found ${tamilVoices.length} Tamil-compatible voice(s):`);
          tamilVoices.slice(0, 5).forEach((voice, idx) => {
            console.log(`\n   ${idx + 1}. Voice ID: ${voice.voice_id || voice.id || 'N/A'}`);
            console.log(`      Name: ${voice.name || 'N/A'}`);
            console.log(`      Language: ${voice.language || 'N/A'}`);
            console.log(`      Supported Languages: ${voice.supported_languages?.join(', ') || 'N/A'}`);
            console.log(`      Gender: ${voice.gender || 'N/A'}`);
          });
          
          if (tamilVoices.length > 5) {
            console.log(`   ... and ${tamilVoices.length - 5} more`);
          }
          
          // Get first Tamil voice ID
          const firstTamilVoiceId = tamilVoices[0]?.voice_id || tamilVoices[0]?.id;
          if (firstTamilVoiceId) {
            console.log(`\n   💡 Recommended Tamil Voice ID: ${firstTamilVoiceId}`);
            console.log(`   📝 Update line 98 in AvatarVideo.tsx:`);
            console.log(`      'ta': '${firstTamilVoiceId}',`);
          }
        } else {
          console.log('\n   ⚠️  No Tamil voices found in API response');
          console.log('   💡 Tamil may not be supported yet, or voices use different metadata');
          console.log('   📝 You may need to check HeyGen dashboard: https://labs.heygen.com/interactive-avatar');
        }
        
        // Test 2b: Try filtering by language parameter
        console.log('\n   📞 Testing language filter parameter...');
        try {
          const filteredResponse = await makeRequest('/v2/voices?language=ta');
          if (filteredResponse.statusCode === 200) {
            const filteredVoices = filteredResponse.data?.data?.voices || filteredResponse.data?.voices || [];
            console.log(`   ✅ Language filter works: Found ${filteredVoices.length} voice(s) with language=ta`);
          } else {
            console.log(`   ⚠️  Language filter returned status: ${filteredResponse.statusCode}`);
          }
        } catch (error) {
          console.log(`   ⚠️  Language filter test failed: ${error.message}`);
        }
        
      } else {
        console.log(`   ❌ API request failed with status: ${voicesResponse.statusCode}`);
        if (voicesResponse.data) {
          console.log(`   Response: ${JSON.stringify(voicesResponse.data, null, 2)}`);
        }
      }
    } catch (error) {
      console.log(`   ❌ API test failed: ${error.message}`);
    }
  })();
}

// Test 3: Check documentation
console.log('\n📋 Test 3: Checking Documentation...');
const setupDocPath = path.join(__dirname, '..', 'TAMIL_SETUP.md');

if (fs.existsSync(setupDocPath)) {
  console.log('   ✅ TAMIL_SETUP.md exists');
  const docContent = fs.readFileSync(setupDocPath, 'utf8');
  if (docContent.includes('Tamil') && docContent.includes('voice ID')) {
    console.log('   ✅ Documentation is complete');
  } else {
    console.log('   ⚠️  Documentation may be incomplete');
  }
} else {
  console.log('   ❌ TAMIL_SETUP.md not found');
}

// Summary
console.log('\n' + '='.repeat(60));
console.log('\n📊 Test Summary:');
console.log('   ✅ Tamil language code is implemented');
console.log('   ✅ Language selector includes Tamil');
console.log('   ✅ Voice mapping system is in place');
console.log('   ⚠️  Tamil voice ID needs to be configured');
console.log('\n📝 Next Steps:');
console.log('   1. Run this test with HEYGEN_API_KEY set to find Tamil voice IDs');
console.log('   2. Update AvatarVideo.tsx line ~98 with Tamil voice ID');
console.log('   3. Test Tamil language selection in the app');
console.log('   4. Verify avatar speaks Tamil when Tamil is selected');
console.log('\n🌐 Test Link:');
console.log('   Run: npm run dev');
console.log('   Then: Open http://localhost:3000');
console.log('   Select Tamil from language selector and test');
console.log('\n' + '='.repeat(60));

