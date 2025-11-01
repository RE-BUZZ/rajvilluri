#!/usr/bin/env python3
"""
HeyGen API Status and Credits Checker (Python)

Simple script to check:
1. HeyGen API connectivity
2. Account credits/quota
3. Streaming avatar availability
4. API key validity
"""

import os
import json
import time
import requests
from pathlib import Path

# Load environment variables from .env file
def load_env():
    env_path = Path(__file__).parent.parent / '.env.local'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

load_env()

HEYGEN_API_KEY = os.getenv('HEYGEN_API_KEY') or os.getenv('NEXT_PUBLIC_HEYGEN_API_KEY')
HEYGEN_BASE_URL = 'https://api.heygen.com'

if not HEYGEN_API_KEY:
    print('❌ HeyGen API key not found in environment variables')
    print('   Make sure HEYGEN_API_KEY is set in your .env file')
    exit(1)

headers = {
    'x-api-key': HEYGEN_API_KEY,
    'Content-Type': 'application/json',
    'User-Agent': 'HeyGen-Status-Checker-Python/1.0'
}

def make_request(endpoint, method='GET', data=None):
    """Make HTTP request to HeyGen API"""
    url = f"{HEYGEN_BASE_URL}{endpoint}"
    
    try:
        if method == 'GET':
            response = requests.get(url, headers=headers, timeout=10)
        else:
            response = requests.post(url, headers=headers, json=data or {}, timeout=10)
        
        return {
            'status_code': response.status_code,
            'data': response.json() if response.content else {}
        }
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

def check_account_info():
    """Check API key validity and account info"""
    print('🔍 Checking HeyGen account information...')
    
    response = make_request('/v1/user/remaining_quota')
    if not response:
        return False
    
    if response['status_code'] == 200:
        print('✅ Account Status: Active')
        
        data = response['data'].get('data', {})
        credit = data.get('credit', 'N/A')
        streaming = data.get('streaming', 'N/A')
        
        print(f'📊 Remaining Credits: {credit}')
        print(f'🎭 Streaming Quota: {streaming}')
        
        # Check credit levels
        if isinstance(credit, (int, float)):
            if credit < 10:
                print('⚠️  WARNING: Low credits remaining!')
            elif credit < 50:
                print('🟡 Credits are getting low')
            else:
                print('🟢 Credit levels look good')
        
        return True
    elif response['status_code'] == 401:
        print('❌ API Key Invalid or Expired')
        print(f"   Response: {json.dumps(response['data'], indent=2)}")
        return False
    elif response['status_code'] == 403:
        print('❌ Access Forbidden - Check API permissions')
        print(f"   Response: {json.dumps(response['data'], indent=2)}")
        return False
    else:
        print(f"⚠️  Unexpected response: {response['status_code']}")
        print(f"   Response: {json.dumps(response['data'], indent=2)}")
        return False

def check_streaming_token():
    """Test streaming avatar token generation"""
    print('\n� Step 1: Testing Token Generation...')
    
    response = make_request('/v1/streaming.create_token', 'POST')
    if not response:
        return False, None
    
    if response['status_code'] == 200:
        print('✅ Token Generation: Success')
        
        token = response['data'].get('data', {}).get('token', '')
        if token:
            print(f'   Token length: {len(token)} characters')
            print(f'   Token preview: {token[:20]}...')
            return True, token
        else:
            print('❌ Token Generation: No token in response')
            return False, None
    elif response['status_code'] == 402:
        print('💳 Token Generation: Insufficient credits')
        print(f"   Response: {json.dumps(response['data'], indent=2)}")
        return False, None
    elif response['status_code'] == 429:
        print('🚫 Token Generation: Rate limited')
        print(f"   Response: {json.dumps(response['data'], indent=2)}")
        return False, None
    else:
        print(f"❌ Token Generation Error: {response['status_code']}")
        print(f"   Response: {json.dumps(response['data'], indent=2)}")
        return False, None

def check_avatar_session(token, avatar_id):
    """Test avatar session creation"""
    print('\n🎭 Step 2: Testing Avatar Session Creation...')
    
    if not token:
        print('❌ Avatar Session: No token available')
        return False, None
    
    session_data = {
        "quality": "high",
        "avatar_name": avatar_id,
        "knowledge_base": "",
        "voice": {
            "type": "text",
            "input_text": "Hello, this is a test message for HeyGen API status check."
        }
    }
    
    response = make_request('/v1/streaming.new', 'POST', session_data)
    if not response:
        return False, None
    
    if response['status_code'] == 200:
        print('✅ Avatar Session: Created successfully')
        
        session_info = response['data'].get('data', {})
        session_id = session_info.get('session_id', '')
        
        if session_id:
            print(f'   Session ID: {session_id}')
            print(f'   Avatar: {avatar_id}')
            return True, session_id
        else:
            print('❌ Avatar Session: No session ID in response')
            return False, None
    elif response['status_code'] == 402:
        print('💳 Avatar Session: Insufficient credits')
        print(f"   Response: {json.dumps(response['data'], indent=2)}")
        return False, None
    elif response['status_code'] == 400:
        print('⚠️  Avatar Session: Invalid parameters (avatar might not support streaming)')
        print(f"   Response: {json.dumps(response['data'], indent=2)}")
        return False, None
    else:
        print(f"❌ Avatar Session Error: {response['status_code']}")
        print(f"   Response: {json.dumps(response['data'], indent=2)}")
        return False, None

def check_streaming_capabilities():
    """Test complete streaming workflow"""
    print('\n🔄 Step 3: Testing Complete Streaming Workflow...')
    
    # Get token
    token_success, token = check_streaming_token()
    if not token_success:
        return False
    
    # Get available streaming avatars
    avatars_response = make_request('/v2/avatars')
    if not avatars_response or avatars_response['status_code'] != 200:
        print('❌ Streaming Workflow: Cannot fetch avatars')
        if avatars_response:
            print(f"   Status Code: {avatars_response['status_code']}")
            print(f"   Response: {json.dumps(avatars_response['data'], indent=2)}")
        return False
    
    # Find a streaming-capable avatar
    avatars = avatars_response['data'].get('data', {}).get('avatars', [])
    streaming_avatars = [a for a in avatars if a.get('avatar_name') and 'public' in str(a.get('avatar_id', '')).lower()]
    
    if not streaming_avatars:
        print('⚠️  Streaming Workflow: No suitable avatars found for testing')
        return False
    
    # Test with first available avatar
    test_avatar = streaming_avatars[0]
    avatar_name = test_avatar.get('avatar_name', '')
    
    print(f'   Testing with avatar: {avatar_name}')
    
    # Try to create session
    session_success, session_id = check_avatar_session(token, avatar_name)
    if not session_success:
        return False
    
    # Test session status
    if session_id:
        print('\n📊 Step 4: Testing Session Status...')
        time.sleep(1)  # Wait a moment for session to initialize
        
        status_response = make_request(f'/v1/streaming.session/{session_id}')
        if status_response and status_response['status_code'] == 200:
            print('✅ Session Status: Accessible')
            
            status_data = status_response['data'].get('data', {})
            session_status = status_data.get('status', 'unknown')
            print(f'   Session Status: {session_status}')
            
            return True
        else:
            print('⚠️  Session Status: Cannot verify session')
            if status_response:
                print(f"   Status Code: {status_response['status_code']}")
                print(f"   Response: {json.dumps(status_response['data'], indent=2)}")
            return True  # Session was created successfully anyway
    
    return True

def test_voice_generation():
    """Test voice generation capabilities"""
    print('\n🔊 Step 5: Testing Voice Generation...')
    
    # Test voice cloning endpoint (if available)
    voices_response = make_request('/v2/voices')
    if voices_response and voices_response['status_code'] == 200:
        print('✅ Voice API: Accessible')
        
        voices = voices_response['data'].get('data', {}).get('voices', [])
        print(f'   Available voices: {len(voices)}')
        
        # Show sample voices
        sample_voices = voices[:3]
        if sample_voices:
            print('   Sample voices:')
            for voice in sample_voices:
                voice_id = voice.get('voice_id', 'Unknown')
                language = voice.get('language', 'Unknown')
                gender = voice.get('gender', 'Unknown')
                print(f'   - {voice_id} ({language}, {gender})')
        
        return True
    else:
        print('⚠️  Voice API: Not accessible or not available')
        if voices_response:
            print(f"   Status Code: {voices_response['status_code']}")
            print(f"   Response: {json.dumps(voices_response['data'], indent=2)}")
        return False

def test_complete_workflow():
    """Test complete end-to-end workflow"""
    print('\n🔄 Step 6: Testing Complete Workflow...')
    
    try:
        # Step 1: Get token
        print('   → Creating streaming token...')
        token_success, token = check_streaming_token()
        if not token_success:
            print('❌ Complete Workflow: Failed at token generation')
            return False
        
        # Step 2: Get avatars
        print('   → Fetching available avatars...')
        avatars_response = make_request('/v2/avatars')
        if not avatars_response or avatars_response['status_code'] != 200:
            print('❌ Complete Workflow: Failed to fetch avatars')
            if avatars_response:
                print(f"   Status Code: {avatars_response['status_code']}")
                print(f"   Response: {json.dumps(avatars_response['data'], indent=2)}")
            return False
        
        avatars = avatars_response['data'].get('data', {}).get('avatars', [])
        if not avatars:
            print('❌ Complete Workflow: No avatars available')
            return False
        
        # Step 3: Test session creation with first available avatar
        test_avatar = avatars[0]
        avatar_name = test_avatar.get('avatar_name', 'test_avatar')
        
        print(f'   → Testing session with {avatar_name}...')
        session_success, session_id = check_avatar_session(token, avatar_name)
        
        if session_success and session_id:
            print('✅ Complete Workflow: Success!')
            print(f'   ✓ Token created: {token[:20]}...')
            print(f'   ✓ Session created: {session_id}')
            print(f'   ✓ Avatar tested: {avatar_name}')
            
            # Test session cleanup (optional)
            print('   → Testing session cleanup...')
            cleanup_response = make_request(f'/v1/streaming.stop', 'POST', {'session_id': session_id})
            if cleanup_response and cleanup_response['status_code'] == 200:
                print('   ✓ Session cleanup: Success')
            else:
                print('   ⚠️ Session cleanup: May need manual cleanup')
                if cleanup_response:
                    print(f"      Cleanup Status: {cleanup_response['status_code']}")
                    print(f"      Cleanup Response: {json.dumps(cleanup_response['data'], indent=6)}")
            
            return True
        else:
            print('❌ Complete Workflow: Failed at session creation')
            return False
            
    except Exception as e:
        print(f'❌ Complete Workflow: Exception occurred - {e}')
        return False

def check_available_avatars():
    """Check available avatars"""
    print('\n👥 Checking available avatars...')
    
    response = make_request('/v2/avatars')
    if not response:
        return False
    
    if response['status_code'] == 200:
        print('✅ Avatar List: Accessible')
        
        avatars = response['data'].get('data', {}).get('avatars', [])
        print(f'📋 Available Avatars: {len(avatars)}')
        
        # Show sample avatars
        sample_avatars = [a for a in avatars if a.get('avatar_name')][:5]
        if sample_avatars:
            print('   Sample avatars:')
            for avatar in sample_avatars:
                name = avatar.get('avatar_name', 'Unknown')
                gender = avatar.get('gender', 'Unknown')
                print(f'   - {name} ({gender})')
        
        return True
    else:
        print(f"❌ Avatar List Error: {response['status_code']}")
        print(f"   Response: {json.dumps(response['data'], indent=2)}")
        return False

def check_api_health():
    """Check API connectivity and health"""
    print('\n🏥 Checking API health...')
    
    start_time = time.time()
    response = make_request('/v2/avatars')
    end_time = time.time()
    
    if not response:
        return False
    
    response_time = int((end_time - start_time) * 1000)  # Convert to ms
    
    if response['status_code'] in [200, 401]:
        print('✅ API Connectivity: Good')
        print(f'⚡ Response Time: {response_time}ms')
        
        if response_time > 5000:
            print('🟡 API response is slow (>5s)')
        elif response_time > 2000:
            print('🟡 API response is moderate (>2s)')
        else:
            print('🟢 API response is fast (<2s)')
        
        return True
    else:
        print(f"❌ API Health Issue: {response['status_code']}")
        print(f"   Response: {json.dumps(response['data'], indent=2)}")
        return False

def main():
    """Main execution"""
    print('🚀 HeyGen API Status Checker (Python) - Detailed Mode')
    print('=====================================================\n')
    
    results = {
        'api_health': check_api_health(),
        'account_info': check_account_info(),
        'avatar_list': check_available_avatars(),
        'streaming_workflow': check_streaming_capabilities(),
        'voice_generation': test_voice_generation(),
        'complete_workflow': test_complete_workflow()
    }
    
    # Summary
    print('\n📋 DETAILED SUMMARY')
    print('===================')
    
    passed_checks = sum(results.values())
    total_checks = len(results)
    
    print(f'✅ Passed: {passed_checks}/{total_checks} checks')
    
    # Detailed status for each component
    print('\n🔍 Component Status:')
    status_icons = {True: '✅', False: '❌'}
    print(f'   API Health: {status_icons[results["api_health"]]}')
    print(f'   Account Info: {status_icons[results["account_info"]]}')
    print(f'   Avatar List: {status_icons[results["avatar_list"]]}')
    print(f'   Streaming Workflow: {status_icons[results["streaming_workflow"]]}')
    print(f'   Voice Generation: {status_icons[results["voice_generation"]]}')
    print(f'   Complete Workflow: {status_icons[results["complete_workflow"]]}')
    
    if passed_checks == total_checks:
        print('\n🎉 All systems operational!')
        print('   Your HeyGen setup is fully functional!')
    elif passed_checks >= total_checks - 1:
        print('\n🟡 Minor issues detected')
        print('   Most features are working correctly')
    elif passed_checks >= total_checks - 2:
        print('\n🟠 Some issues detected')
        print('   Core functionality may be affected')
    else:
        print('\n🔴 Major issues detected')
        print('   Significant problems found')
    
    # Detailed recommendations
    print('\n💡 DETAILED RECOMMENDATIONS')
    print('============================')
    
    if not results['api_health']:
        print('🌐 API Connectivity Issues:')
        print('   • Check your internet connection')
        print('   • Verify HeyGen API status at status.heygen.com')
        print('   • Try running the script from a different network')
    
    if not results['account_info']:
        print('🔑 Account/Authentication Issues:')
        print('   • Verify your API key is correct in .env.local')
        print('   • Check if your account is active')
        print('   • Ensure API key has necessary permissions')
        print('   • Try regenerating your API key from HeyGen dashboard')
    
    if not results['streaming_workflow']:
        print('🎭 Streaming Avatar Issues:')
        print('   • Add credits to your HeyGen account')
        print('   • Check if streaming features are enabled in your plan')
        print('   • Verify avatar permissions')
        print('   • Contact HeyGen support if credits are available')
    
    if not results['voice_generation']:
        print('🔊 Voice Generation Issues:')
        print('   • Voice API might not be available in your plan')
        print('   • Check voice generation permissions')
        print('   • This feature might be in beta')
    
    if not results['avatar_list']:
        print('👥 Avatar Access Issues:')
        print('   • Verify API permissions for avatar access')
        print('   • Check if avatar library is accessible in your region')
    
    # Performance notes
    print('\n⚡ Performance Notes:')
    if all(results.values()):
        print('   • All API endpoints are responding normally')
        print('   • Token generation is working correctly')
        print('   • Avatar sessions can be created successfully')
        print('   • Your setup is ready for production use')
    else:
        print('   • Some features may have degraded performance')
        print('   • Consider testing again in a few minutes')
        print('   • Monitor HeyGen status page for ongoing issues')
    
    # Exit code based on results
    exit(0 if passed_checks == total_checks else 1)

if __name__ == '__main__':
    main()
