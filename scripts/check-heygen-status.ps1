# HeyGen API Status Checker (PowerShell)
# Quick check for HeyGen credits and API status

param(
    [switch]$Detailed
)

# Load environment variables
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^([^=]+)=(.*)$") {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
        }
    }
}

$apiKey = $env:HEYGEN_API_KEY
if (-not $apiKey) {
    $apiKey = $env:NEXT_PUBLIC_HEYGEN_API_KEY
}

if (-not $apiKey) {
    Write-Host "❌ HeyGen API key not found!" -ForegroundColor Red
    Write-Host "   Set HEYGEN_API_KEY in your .env file" -ForegroundColor Yellow
    exit 1
}

$headers = @{
    'x-api-key' = $apiKey
    'Content-Type' = 'application/json'
}

Write-Host "🚀 HeyGen Quick Status Check" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan

# Quick API health check
Write-Host "🔍 Checking API connectivity..." -ForegroundColor Yellow

try {
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $response = Invoke-RestMethod -Uri "https://api.heygen.com/v2/avatars" -Headers $headers -Method GET -TimeoutSec 10
    $stopwatch.Stop()
    
    Write-Host "✅ API: Online ($($stopwatch.ElapsedMilliseconds)ms)" -ForegroundColor Green
    
    # Check remaining quota
    Write-Host "💰 Checking account credits..." -ForegroundColor Yellow
    
    try {
        $quotaResponse = Invoke-RestMethod -Uri "https://api.heygen.com/v1/user/remaining_quota" -Headers $headers -Method GET -TimeoutSec 10
        
        if ($quotaResponse.data) {
            $credit = $quotaResponse.data.credit
            $streaming = $quotaResponse.data.streaming
            
            Write-Host "💳 Credits: $credit" -ForegroundColor Green
            Write-Host "🎭 Streaming Quota: $streaming" -ForegroundColor Green
            
            if ($credit -lt 10) {
                Write-Host "⚠️  WARNING: Low credits!" -ForegroundColor Red
            } elseif ($credit -lt 50) {
                Write-Host "🟡 Credits getting low" -ForegroundColor Yellow
            }
        }
    }
    catch {
        Write-Host "⚠️  Could not fetch quota info" -ForegroundColor Yellow
    }
    
    # Test streaming token
    Write-Host "🎬 Testing streaming functionality..." -ForegroundColor Yellow
    
    try {
        $tokenResponse = Invoke-RestMethod -Uri "https://api.heygen.com/v1/streaming.create_token" -Headers $headers -Method POST -Body "{}" -TimeoutSec 10
        
        if ($tokenResponse.data.token) {
            Write-Host "✅ Streaming: Available" -ForegroundColor Green
        }
    }
    catch {
        $statusCode = $_.Exception.Response.StatusCode.value__
        if ($statusCode -eq 402) {
            Write-Host "💳 Streaming: Insufficient credits" -ForegroundColor Red
        } elseif ($statusCode -eq 429) {
            Write-Host "🚫 Streaming: Rate limited" -ForegroundColor Yellow
        } else {
            Write-Host "❌ Streaming: Error ($statusCode)" -ForegroundColor Red
        }
    }
    
    Write-Host "`n🎉 Status check complete!" -ForegroundColor Green
    
    if ($Detailed) {
        Write-Host "`n📋 Available Avatars:" -ForegroundColor Cyan
        $response.data.avatars | Select-Object -First 5 | ForEach-Object {
            Write-Host "   • $($_.avatar_name) ($($_.gender))" -ForegroundColor White
        }
    }
}
catch {
    $statusCode = $_.Exception.Response.StatusCode.value__
    
    if ($statusCode -eq 401) {
        Write-Host "❌ Invalid API key!" -ForegroundColor Red
    } elseif ($statusCode -eq 403) {
        Write-Host "❌ Access forbidden!" -ForegroundColor Red
    } else {
        Write-Host "❌ API Error: $($_.Exception.Message)" -ForegroundColor Red
    }
    
    Write-Host "`n💡 Troubleshooting:" -ForegroundColor Yellow
    Write-Host "   • Check your API key in .env file" -ForegroundColor White
    Write-Host "   • Verify internet connection" -ForegroundColor White
    Write-Host "   • Check HeyGen service status" -ForegroundColor White
    
    exit 1
}
