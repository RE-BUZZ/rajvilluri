// Simple test to check HeyGen API connectivity
const https = require('https');

async function testHeyGenAPI() {
  // Try to read from .env.local
  const fs = require('fs');
  const path = require('path');
  
  try {
    const envPath = path.join(__dirname, '.env.local');
    if (fs.existsSync(envPath)) {
      const envContent = fs.readFileSync(envPath, 'utf8');
      const lines = envContent.split('\n');
      
      lines.forEach(line => {
        const [key, value] = line.split('=');
        if (key && value) {
          process.env[key.trim()] = value.trim();
        }
      });
    }
  } catch (error) {
    console.log('Could not read .env.local:', error.message);
  }

  const apiKey = process.env.HEYGEN_API_KEY;
  
  console.log('🔍 Testing HeyGen API...');
  console.log('API Key present:', !!apiKey);
  console.log('API Key length:', apiKey ? apiKey.length : 0);
  
  if (!apiKey) {
    console.log('❌ No HEYGEN_API_KEY found in environment');
    console.log('Please create .env.local with:');
    console.log('HEYGEN_API_KEY=your_key_here');
    return;
  }

  // Test token generation
  const data = JSON.stringify({});
  
  const options = {
    hostname: 'api.heygen.com',
    port: 443,
    path: '/v1/streaming.create_token',
    method: 'POST',
    headers: {
      'x-api-key': apiKey,
      'Content-Type': 'application/json',
      'Content-Length': data.length
    }
  };

  return new Promise((resolve, reject) => {
    const req = https.request(options, (res) => {
      let responseData = '';
      
      res.on('data', (chunk) => {
        responseData += chunk;
      });
      
      res.on('end', () => {
        console.log('Response status:', res.statusCode);
        console.log('Response headers:', res.headers);
        
        try {
          const parsedData = JSON.parse(responseData);
          console.log('Response data:', parsedData);
          
          if (res.statusCode === 200) {
            console.log('✅ HeyGen API is working!');
            console.log('Token generated successfully');
          } else {
            console.log('❌ HeyGen API error:', parsedData);
          }
        } catch (error) {
          console.log('❌ Failed to parse response:', responseData);
        }
        
        resolve();
      });
    });

    req.on('error', (error) => {
      console.log('❌ Request error:', error.message);
      reject(error);
    });

    req.write(data);
    req.end();
  });
}

testHeyGenAPI().catch(console.error);