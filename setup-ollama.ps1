# ==============================================================================
# 🚀 Offline AI Agent - One-Click Ollama Setup & GPU Accelerator
# Hardware Target: NVIDIA GeForce RTX 4060 (8GB VRAM)
# ==============================================================================

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   🧠 OFFLINE AI AGENT - LOCAL MODEL SETUP SCRIPT" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check NVIDIA GPU
Write-Host "[1/4] Checking GPU Hardware Acceleration..." -ForegroundColor Yellow
$gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
if ($gpuInfo) {
    Write-Host "✔ Detected GPU: $gpuInfo" -ForegroundColor Green
    Write-Host "  RTX 4060 with 8GB VRAM will provide full local hardware acceleration!" -ForegroundColor Gray
} else {
    Write-Host "⚠ nvidia-smi not found or GPU busy. Models will run on CPU fallback if GPU is unavailable." -ForegroundColor Yellow
}

# 2. Check or Install Ollama
Write-Host "`n[2/4] Checking Ollama installation..." -ForegroundColor Yellow
$ollamaExists = Get-Command ollama -ErrorAction SilentlyContinue

if (-not $ollamaExists) {
    # Check default install path
    $defaultOllamaPath = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
    if (Test-Path $defaultOllamaPath) {
        $env:Path += ";$env:LOCALAPPDATA\Programs\Ollama"
        Write-Host "✔ Found Ollama in default directory: $defaultOllamaPath" -ForegroundColor Green
        $ollamaExists = $true
    }
}

if (-not $ollamaExists) {
    Write-Host "Ollama not found. Installing via winget..." -ForegroundColor Cyan
    winget install Ollama.Ollama --accept-source-agreements --accept-package-agreements
    
    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
} else {
    Write-Host "✔ Ollama is already installed!" -ForegroundColor Green
}

# 3. Ensure Ollama Service is Running
Write-Host "`n[3/4] Testing connection to local Ollama server (http://localhost:11434)..." -ForegroundColor Yellow
$isListening = $false
try {
    $resp = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 3 -ErrorAction Stop
    $isListening = $true
    Write-Host "✔ Ollama server is actively running!" -ForegroundColor Green
} catch {
    Write-Host "Starting Ollama background process..." -ForegroundColor Cyan
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
    try {
        $resp = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 3 -ErrorAction Stop
        $isListening = $true
        Write-Host "✔ Ollama server started successfully!" -ForegroundColor Green
    } catch {
        Write-Host "⚠ Could not confirm server. Please run 'ollama serve' in a separate terminal if needed." -ForegroundColor Yellow
    }
}

# 4. Pull Recommended Model
Write-Host "`n[4/4] Recommended Models for your 8GB RTX 4060:" -ForegroundColor Yellow
Write-Host "  1. qwen2.5:7b      (Alibaba - Best tool calling, structured JSON, coding) [RECOMMENDED]" -ForegroundColor White
Write-Host "  2. llama3.1:8b     (Meta - Top tier general reasoning & instruction following)" -ForegroundColor White
Write-Host "  3. mistral:7b      (Mistral AI - Fast, snappy responses)" -ForegroundColor White
Write-Host "  4. gemma2:9b       (Google - High efficiency, state of the art)" -ForegroundColor White
Write-Host "  5. Skip for now    (You can pull models anytime from the web UI dashboard)" -ForegroundColor Gray

$choice = Read-Host "`nEnter model number to pull (1-5, default 1)"
if (-not $choice) { $choice = "1" }

$targetModel = "qwen2.5:7b"
switch ($choice) {
    "1" { $targetModel = "qwen2.5:7b" }
    "2" { $targetModel = "llama3.1:8b" }
    "3" { $targetModel = "mistral:7b" }
    "4" { $targetModel = "gemma2:9b" }
    "5" { $targetModel = "" }
    default { $targetModel = "qwen2.5:7b" }
}

if ($targetModel) {
    Write-Host "`nPulling $targetModel into local offline storage..." -ForegroundColor Cyan
    ollama pull $targetModel
    Write-Host "`n✔ $targetModel downloaded and ready for 100% offline execution!" -ForegroundColor Green
} else {
    Write-Host "`nSkipped model download. You can download anytime in the web dashboard or with 'ollama pull'." -ForegroundColor Gray
}

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "   🎉 SETUP COMPLETE! HOW TO LAUNCH YOUR AGENT:" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  1. Web Cockpit UI:   npm start    (then open http://localhost:3000)" -ForegroundColor Cyan
Write-Host "  2. Terminal CLI:     npm run cli" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Green
Write-Host ""
