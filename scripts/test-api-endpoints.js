#!/usr/bin/env node

/**
 * Test API Endpoints
 * Tests HeyGen token generation and Google TTS/STT endpoints
 */

const https = require('https');
const http = require('http');

const BASE_URL = 'http://localhost:3000';

console.log('🧪 Testing API Endpoints\n');
console.log('='.repeat(60));

// Test 1: Check if server is running
console.log('\n📋 Test 1: Server Connectivity...');
testEndpoint('/', 'GET', null, (success, data) => {
  if (success) {
    console.log('   ✅ Server is running and accessible');
  } else {
    console.log('   ❌ Server is not accessible');
    console.log('   💡 Make sure dev server is running: npm run dev');
    process.exit(1);
  }
});

// Test 2: Test HeyGen Token Generation
console.log('\n📋 Test 2: HeyGen Token Generation...');
setTimeout(() => {
  testEndpoint('/api/generate-token', 'POST', {}, (success, data) => {
    if (success && data.token) {
      console.log('   ✅ HeyGen token generated successfully');
      console.log(`   Token preview: ${data.token.substring(0, 20)}...`);
    } else if (success && data.error) {
      console.log(`   ⚠️  HeyGen API error: ${data.error}`);
      if (data.error.includes('not configured')) {
        console.log('   💡 Check HEYGEN_API_KEY in .env.local');
      }
    } else {
      console.log('   ❌ Failed to generate token');
    }
  });
}, 1000);

// Test 3: Test Google TTS (if configured)
console.log('\n📋 Test 3: Google TTS Endpoint...');
setTimeout(() => {
  const ttsPayload = {
    text: 'வணக்கம்',
    languageCode: 'ta-IN'
  };
  
  testEndpoint('/api/google/tts', 'POST', ttsPayload, (success, data) => {
    if (success && data.audioContent) {
      console.log('   ✅ Google TTS working!');
      console.log(`   Voice: ${data.voiceName || 'N/A'}`);
      console.log(`   Audio size: ${data.audioSize || 'N/A'}KB`);
    } else if (success && data.error) {
      console.log(`   ⚠️  Google TTS error: ${data.error}`);
      if (data.error.includes('credentials')) {
        console.log('   💡 Check Google Cloud credentials in .env.local');
        console.log('   💡 You may need service account JSON file');
      }
    } else {
      console.log('   ❌ Google TTS failed');
    }
  });
}, 2000);

// Test 4: Test Google STT (requires audio file - skip for now)
console.log('\n📋 Test 4: Google STT Endpoint...');
console.log('   ⏭️  Skipping (requires audio file upload)');
console.log('   💡 Test manually by selecting Tamil and speaking');

// Summary
setTimeout(() => {
  console.log('\n' + '='.repeat(60));
  console.log('\n📊 Testing Summary:');
  console.log('   🌐 Application URL: http://localhost:3000');
  console.log('   📝 API Endpoints tested');
  console.log('\n💡 Next Steps:');
  console.log('   1. Open http://localhost:3000 in your browser');
  console.log('   2. Select English → Test HeyGen avatar');
  console.log('   3. Select Tamil → Test Google TTS & STT');
  console.log('\n' + '='.repeat(60));
}, 3000);

// Helper function to test endpoints
function testEndpoint(path, method, body, callback) {
  const url = new URL(path, BASE_URL);
  
  const options = {
    hostname: url.hostname,
    port: url.port || 3000,
    path: url.pathname,
    method: method,
    headers: {
      'Content-Type': 'application/json',
    },
    timeout: 5000
  };

  const req = http.request(options, (res) => {
    let data = '';
    
    res.on('data', (chunk) => {
      data += chunk;
    });
    
    res.on('end', () => {
      try {
        const parsed = JSON.parse(data);
        callback(res.statusCode === 200 || res.statusCode === 201, parsed);
      } catch (e) {
        callback(res.statusCode === 200 || res.statusCode === 201, { html: data.substring(0, 100) });
      }
    });
  });

  req.on('error', (error) => {
    callback(false, { error: error.message });
  });

  req.on('timeout', () => {
    req.destroy();
    callback(false, { error: 'Request timeout' });
  });

  if (body) {
    req.write(JSON.stringify(body));
  }

  req.end();
}

