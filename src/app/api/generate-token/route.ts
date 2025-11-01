import { NextResponse } from 'next/server';

export async function POST() {
  try {
    const apiKey = process.env.HEYGEN_API_KEY;
    
    // Debug logging
    console.log('Environment check:', {
      hasApiKey: !!apiKey,
      keyLength: apiKey?.length || 0,
      keyStart: apiKey?.substring(0, 10) + '...',
      envKeys: Object.keys(process.env).filter(key => key.includes('HEYGEN'))
    });
    
    // Check if API key looks like base64 and might need decoding
    const isBase64 = apiKey && /^[A-Za-z0-9+/]+=*$/.test(apiKey);
    console.log('API key format check:', {
      looksLikeBase64: isBase64,
      keyLength: apiKey?.length || 0
    });
    
    if (!apiKey) {
      console.error('HeyGen API key not found in environment variables');
      return NextResponse.json(
        { error: 'HeyGen API key not configured' },
        { status: 500 }
      );
    }

    console.log('Making request to HeyGen API...');
    
    const response = await fetch('https://api.heygen.com/v1/streaming.create_token', {
      method: 'POST',
      headers: {
        'x-api-key': apiKey,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({}), // Empty body for token creation
    });
    
    console.log('HeyGen API response status:', response.status);
    console.log('HeyGen API response headers:', Object.fromEntries(response.headers.entries()));

    if (!response.ok) {
      const errorData = await response.text();
      console.error('HeyGen API error response:', errorData);
      console.error('Response status:', response.status);
      
      // Try alternative endpoint if the first one fails
      if (response.status === 404 || response.status === 400) {
        console.log('Trying alternative endpoint: streaming.new');
        
        const altResponse = await fetch('https://api.heygen.com/v1/streaming.new', {
          method: 'POST',
          headers: {
            'x-api-key': apiKey,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({}),
        });
        
        console.log('Alternative API response status:', altResponse.status);
        
        if (altResponse.ok) {
          const altData = await altResponse.json();
          console.log('Alternative API success:', altData);
          return NextResponse.json({ 
            token: altData.data?.token || altData.token
          });
        } else {
          const altErrorData = await altResponse.text();
          console.error('Alternative API error:', altErrorData);
        }
      }
      
      return NextResponse.json(
        { 
          error: 'Failed to generate token from HeyGen API', 
          details: errorData,
          status: response.status 
        },
        { status: response.status }
      );
    }

    const data = await response.json();
    console.log('HeyGen API success response:', data);
    
    return NextResponse.json({ 
      token: data.data?.token || data.token
    });
  } catch (error) {
    console.error('Token generation error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}