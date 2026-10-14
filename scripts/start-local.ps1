$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$localRoot = Join-Path $env:LOCALAPPDATA "PetMind"
$pgRoot = Join-Path $localRoot "postgres\16"
$pgData = Join-Path $localRoot "postgres\cluster"
$pgCtl = Join-Path $pgRoot "bin\pg_ctl.exe"
$python = Join-Path $projectRoot ".venv-1\Scripts\python.exe"
$logs = Join-Path $localRoot "logs"

if (-not (Test-Path $pgCtl)) { throw "Не знайдено PostgreSQL у $pgRoot" }
if (-not (Test-Path $python)) { throw "Не знайдено Python оточення .venv-1. Спершу встановіть залежності backend." }
if (-not (Test-Path (Join-Path $projectRoot "frontend\node_modules"))) { throw "Не знайдено frontend/node_modules. Спершу виконайте corepack pnpm install у frontend." }
New-Item -ItemType Directory -Force -Path $logs | Out-Null

# Load KEY=VALUE pairs from the repository .env without overriding variables already set in the session.
$envFile = Join-Path $projectRoot ".env"
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile -Encoding UTF8) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$' -and -not (Test-Path "Env:$($Matches[1])")) {
            Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2].Trim().Trim('"')
        }
    }
}
$apiPort = if ($env:API_PORT) { [int]$env:API_PORT } else { 8000 }
$env:API_PORT = "$apiPort"

& $pgCtl status -D $pgData *> $null
if ($LASTEXITCODE -ne 0) {
    & $pgCtl start -D $pgData -l (Join-Path $localRoot "postgres\postgres.log") -o "-h 127.0.0.1 -p 5432"
    if ($LASTEXITCODE -ne 0) { throw "Не вдалося запустити PostgreSQL." }
}

$env:DATABASE_URL = "postgresql+psycopg://petly:petly@127.0.0.1:5432/petly"
$env:PGPASSWORD = "petly"
$env:PYTHONPATH = "$projectRoot\backend;$projectRoot\src"
if (-not $env:LLM_PROVIDER) { $env:LLM_PROVIDER = "disabled" }

Push-Location $projectRoot
try {
    & $python -m alembic -c backend/alembic.ini upgrade head
    if ($LASTEXITCODE -ne 0) { throw "Міграції бази даних завершилися помилкою." }
    & $python backend/scripts/seed.py --demo-only
    if ($LASTEXITCODE -ne 0) { throw "Завантаження демо-даних завершилося помилкою." }
} finally {
    Pop-Location
}

$apiListening = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $apiPort -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($apiListening) {
    $owner = Get-Process -Id $apiListening.OwningProcess -ErrorAction SilentlyContinue
    $started = if ($owner) { $owner.StartTime } else { $null }
    $latestChange = (Get-ChildItem (Join-Path $projectRoot "backend\app"), (Join-Path $projectRoot "src\petrec") -Recurse -Filter *.py | Measure-Object LastWriteTime -Maximum).Maximum
    if ($started -and $latestChange -gt $started) {
        Write-Warning "API на порту $apiPort (PID $($apiListening.OwningProcess)) запущено до останніх змін коду. Зупиніть його: Stop-Process -Id $($apiListening.OwningProcess), і запустіть скрипт знову."
    } else {
        Write-Host "API вже працює на порту $apiPort (PID $($apiListening.OwningProcess))."
    }
} else {
    $apiOut = Join-Path $logs "api.out.log"
    $apiErr = Join-Path $logs "api.err.log"
    Start-Process -FilePath $python -ArgumentList @("-m", "uvicorn", "app.main:app", "--app-dir", "backend", "--host", "127.0.0.1", "--port", "$apiPort") -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $apiOut -RedirectStandardError $apiErr
    Write-Host "API запускається на http://127.0.0.1:$apiPort (логи: $logs)."
}

$webListening = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 5173 -State Listen -ErrorAction SilentlyContinue
if ($webListening) {
    Write-Host "PetMind вже запущений: http://localhost:5173"
    exit 0
}

Write-Host "Запускаю PetMind. Вебзастосунок: http://localhost:5173 (Ctrl+C зупинить frontend)."
Push-Location (Join-Path $projectRoot "frontend")
try {
    corepack pnpm dev -- --host 127.0.0.1 --port 5173
} finally {
    Pop-Location
}
