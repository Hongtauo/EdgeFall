param(
    [ValidateSet("auto", "live", "replay")]
    [string]$Mode = "auto",
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
$applicationRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontendRoot = Join-Path $applicationRoot "frontend"
$backendRoot = Join-Path $applicationRoot "backend"
$venvRoot = Join-Path $applicationRoot ".venv"
$liveVenvRoot = Join-Path $applicationRoot ".venv-live"

Write-Host "[1/3] Preparing Vue frontend..." -ForegroundColor Cyan
Push-Location $frontendRoot
try {
    if (-not (Test-Path -LiteralPath (Join-Path $frontendRoot "node_modules"))) {
        npm install
    }
    npm run build
}
finally {
    Pop-Location
}

if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $livePythonExecutable = Join-Path $liveVenvRoot "Scripts\python.exe"
    if ($Mode -in @("auto", "live") -and (Test-Path -LiteralPath $livePythonExecutable)) {
        $pythonExecutable = $livePythonExecutable
        Write-Host "[2/3] Using the GPU inference environment..." -ForegroundColor Cyan
    }
    else {
        $pythonExecutable = Join-Path $venvRoot "Scripts\python.exe"
    }
    if (-not (Test-Path -LiteralPath $pythonExecutable)) {
        Write-Host "[2/3] Creating local Python environment..." -ForegroundColor Cyan
        python -m venv $venvRoot
        & $pythonExecutable -m pip install -r (Join-Path $backendRoot "requirements.txt")
    }
    elseif ($pythonExecutable -eq (Join-Path $venvRoot "Scripts\python.exe")) {
        Write-Host "[2/3] Using the local Python environment..." -ForegroundColor Cyan
    }
}
else {
    $pythonExecutable = $PythonPath
    Write-Host "[2/3] Using the selected Python environment..." -ForegroundColor Cyan
}

& $pythonExecutable -c "import fastapi, cv2, numpy" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "FastAPI backend dependencies are missing. Install backend/requirements.txt in the selected Python environment."
}

if ($Mode -eq "live") {
    & $pythonExecutable -c "import torch, ultralytics; assert torch.cuda.is_available(), 'CUDA is unavailable'" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "The selected Python environment cannot run CUDA inference."
    }
}

$env:EDGEFALL_ENGINE = $Mode
Write-Host "[3/3] EdgeFall-ST demo is ready at http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "      Engine mode: $Mode  (press Ctrl+C to stop)" -ForegroundColor DarkGray
Push-Location $backendRoot
try {
    & $pythonExecutable -m uvicorn app.main:app --host 127.0.0.1 --port 8000
}
finally {
    Pop-Location
}
