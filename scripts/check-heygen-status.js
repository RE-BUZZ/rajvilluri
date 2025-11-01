#!/usr/bin/env node

/**
 * HeyGen API Status and Credits Checker
 * 
 * This script checks:
 * 1. HeyGen API connectivity
 * 2. Account credits/quota
 * 3. Streaming avatar availability
 * 4. API key validity
 * 
 * Based on HeyGen's official API documentation:
 * https://docs.heygen.com/reference/
 */

const https = require('https');
const fs = require('fs');
const path = require('path');

// Load environment variables from .env file
function loadEnv() {
  const envPath = path.join(__dirname, '..', '.env');
  if (fs.existsSync(envPath)) {
    const envContent = fs.readFileSync(envPath, 'utf8');
    envContent.split('\n').forEach(line => {
      const [key, value] = line.split('=');
      if (key && value) {
        process.env[key.trim()] = value.trim();
      }
    });
  }
}

loadEnv();

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY || process.env.NEXT_PUBLIC_HEYGEN_API_KEY;
const HEYGEN_BASE_URL = 'api.heygen.com';

if (!HEYGEN_API_KEY) {
  console.error('❌ HeyGen API key not found in environment variables');
  console.error('   Make sure HEYGEN_API_KEY is set in your .env file');
  process.exit(1);
}

/**
 * Make HTTP request to HeyGen API
 */
function makeRequest(path, method = 'GET', data = null) {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: HEYGEN_BASE_URL,
      port: 443,
      path: path,
      method: method,
      headers: {
        'x-api-key': HEYGEN_API_KEY,
        'Content-Type': 'application/json',
        'User-Agent': 'HeyGen-Status-Checker/1.0'
      }
    };

    if (data && method !== 'GET') {
      const jsonData = JSON.stringify(data);
      options.headers['Content-Length'] = Buffer.byteLength(jsonData);
    }

    const req = https.request(options, (res) => {
      let responseData = '';

      res.on('data', (chunk) => {
        responseData += chunk;
      });

      res.on('end', () => {
        try {
          const parsed = JSON.parse(responseData);
          resolve({
            statusCode: res.statusCode,
            headers: res.headers,
            data: parsed
          });
        } catch (e) {
          resolve({
            statusCode: res.statusCode,
            headers: res.headers,
            data: responseData
          });
        }
      });
    });

    req.on('error', (err) => {
      reject(err);
    });

    if (data && method !== 'GET') {
      req.write(JSON.stringify(data));
    }

    req.end();
  });
}

/**
 * Check API key validity and account info
 */
async function checkAccountInfo() {
  console.log('🔍 Checking HeyGen account information...');
  
  try {
    // According to HeyGen docs, this endpoint checks account status
    const response = await makeRequest('/v1/user/remaining_quota');
    
    if (response.statusCode === 200) {
      console.log('✅ Account Status: Active');
      
      if (response.data.data) {
        const quota = response.data.data;
        console.log(`📊 Remaining Credits: ${quota.credit || 'N/A'}`);
        console.log(`🎬 Video Generation Quota: ${quota.video_generation || 'N/A'}`);
        console.log(`🎭 Avatar Streaming Quota: ${quota.streaming || 'N/A'}`);
        
        // Check if credits are running low
        if (quota.credit !== undefined) {
          if (quota.credit < 10) {
            console.log('⚠️  WARNING: Low credits remaining!');
          } else if (quota.credit < 50) {
            console.log('🟡 Credits are getting low');
          } else {
            console.log('🟢 Credit levels look good');
          }
        }
      }
      
      return true;
    } else if (response.statusCode === 401) {
      console.log('❌ API Key Invalid or Expired');
      console.log(`   Response: ${JSON.stringify(response.data, null, 2)}`);
      return false;
    } else if (response.statusCode === 403) {
      console.log('❌ Access Forbidden - Check API permissions');
      console.log(`   Response: ${JSON.stringify(response.data, null, 2)}`);
      return false;
    } else {
      console.log(`⚠️  Unexpected response: ${response.statusCode}`);
      console.log(`   Response: ${JSON.stringify(response.data, null, 2)}`);
      return false;
    }
  } catch (error) {
    console.log('❌ Failed to check account info');
    console.error(`   Error: ${error.message}`);
    return false;
  }
}

/**
 * Test streaming avatar token generation
 */
async function checkStreamingAvatar() {
  console.log('\n🎭 Testing Streaming Avatar functionality...');
  
  try {
    // Test token creation for streaming avatars
    const response = await makeRequest('/v1/streaming.create_token', 'POST', {});
    
    if (response.statusCode === 200) {
      console.log('✅ Streaming Avatar: Available');
      
      if (response.data.data && response.data.data.token) {
        console.log('🎫 Token generation: Working');
        console.log(`   Token length: ${response.data.data.token.length} characters`);
        
        // Don't log the actual token for security
        const tokenPreview = response.data.data.token.substring(0, 20) + '...';
        console.log(`   Token preview: ${tokenPreview}`);
      }
      
      return true;
    } else if (response.statusCode === 402) {
      console.log('💳 Payment Required - Insufficient credits for streaming');
      console.log('   Please add credits to your HeyGen account');
      return false;
    } else if (response.statusCode === 429) {
      console.log('🚫 Rate Limited - Too many requests');
      console.log('   Wait a few minutes before trying again');
      return false;
    } else {
      console.log(`❌ Streaming Avatar Error: ${response.statusCode}`);
      console.log(`   Response: ${JSON.stringify(response.data, null, 2)}`);
      return false;
    }
  } catch (error) {
    console.log('❌ Failed to test streaming avatar');
    console.error(`   Error: ${error.message}`);
    return false;
  }
}

/**
 * Check available avatars
 */
async function checkAvailableAvatars() {
  console.log('\n👥 Checking available avatars...');
  
  try {
    // List available avatars (public ones)
    const response = await makeRequest('/v2/avatars');
    
    if (response.statusCode === 200) {
      console.log('✅ Avatar List: Accessible');
      
      if (response.data.data && response.data.data.avatars) {
        const avatars = response.data.data.avatars;
        console.log(`📋 Available Avatars: ${avatars.length}`);
        
        // Show first few avatars
        const publicAvatars = avatars.filter(a => a.avatar_name).slice(0, 5);
        if (publicAvatars.length > 0) {
          console.log('   Sample avatars:');
          publicAvatars.forEach(avatar => {
            console.log(`   - ${avatar.avatar_name} (${avatar.gender || 'Unknown'})`);
          });
        }
      }
      
      return true;
    } else {
      console.log(`❌ Avatar List Error: ${response.statusCode}`);
      console.log(`   Response: ${JSON.stringify(response.data, null, 2)}`);
      return false;
    }
  } catch (error) {
    console.log('❌ Failed to check avatars');
    console.error(`   Error: ${error.message}`);
    return false;
  }
}

/**
 * Check API connectivity and health
 */
async function checkAPIHealth() {
  console.log('\n🏥 Checking API health...');
  
  try {
    // Simple API health check
    const startTime = Date.now();
    const response = await makeRequest('/v2/avatars');
    const endTime = Date.now();
    const responseTime = endTime - startTime;
    
    if (response.statusCode === 200 || response.statusCode === 401) {
      console.log('✅ API Connectivity: Good');
      console.log(`⚡ Response Time: ${responseTime}ms`);
      
      if (responseTime > 5000) {
        console.log('🟡 API response is slow (>5s)');
      } else if (responseTime > 2000) {
        console.log('🟡 API response is moderate (>2s)');
      } else {
        console.log('🟢 API response is fast (<2s)');
      }
      
      return true;
    } else {
      console.log(`❌ API Health Issue: ${response.statusCode}`);
      return false;
    }
  } catch (error) {
    console.log('❌ API Connection Failed');
    console.error(`   Error: ${error.message}`);
    
    if (error.code === 'ENOTFOUND') {
      console.log('🌐 DNS resolution failed - check internet connection');
    } else if (error.code === 'ECONNREFUSED') {
      console.log('🚫 Connection refused - HeyGen API might be down');
    }
    
    return false;
  }
}

/**
 * Main execution
 */
async function main() {
  console.log('🚀 HeyGen API Status Checker');
  console.log('================================\n');
  
  const results = {
    apiHealth: false,
    accountInfo: false,
    streamingAvatar: false,
    avatarList: false
  };
  
  // Run all checks
  results.apiHealth = await checkAPIHealth();
  results.accountInfo = await checkAccountInfo();
  results.streamingAvatar = await checkStreamingAvatar();
  results.avatarList = await checkAvailableAvatars();
  
  // Summary
  console.log('\n📋 SUMMARY');
  console.log('===========');
  
  const passedChecks = Object.values(results).filter(Boolean).length;
  const totalChecks = Object.keys(results).length;
  
  console.log(`✅ Passed: ${passedChecks}/${totalChecks} checks`);
  
  if (passedChecks === totalChecks) {
    console.log('🎉 All systems operational!');
  } else if (passedChecks >= totalChecks - 1) {
    console.log('🟡 Minor issues detected');
  } else {
    console.log('🔴 Major issues detected');
  }
  
  // Specific recommendations
  console.log('\n💡 RECOMMENDATIONS');
  console.log('===================');
  
  if (!results.apiHealth) {
    console.log('• Check your internet connection');
    console.log('• Verify HeyGen API status at status.heygen.com');
  }
  
  if (!results.accountInfo) {
    console.log('• Verify your API key is correct');
    console.log('• Check if your account is active');
    console.log('• Ensure API key has necessary permissions');
  }
  
  if (!results.streamingAvatar) {
    console.log('• Add credits to your HeyGen account');
    console.log('• Check if streaming features are enabled');
    console.log('• Contact HeyGen support if credits are available');
  }
  
  if (!results.avatarList) {
    console.log('• Verify API permissions for avatar access');
  }
  
  // Exit code based on results
  process.exit(passedChecks === totalChecks ? 0 : 1);
}

// Handle unhandled rejections
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection:', reason);
  process.exit(1);
});

// Run the script
main().catch(error => {
  console.error('Script failed:', error);
  process.exit(1);
});
