param([switch]$NoBrowser, [ValidateSet('real','stub')][string]$Backend='real', [int]$ServicePort=8089, [int]$WorkerPort=8789)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$jarPath = Join-Path $projectRoot 'service\target\docmind-service-0.1.0-phase4.jar'
$logDir = Join-Path $projectRoot 'logs\launcher'
$baseUrl = "http://127.0.0.1:$ServicePort"

function Read-Health([string]$Url) {
    try { return Invoke-RestMethod -Uri $Url -TimeoutSec 2 } catch { return $null }
}
function Port-InUse([int]$Port) {
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}
function Wait-Health([string]$Url, [string]$Service) {
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        $health = Read-Health $Url
        if ($health -and $health.service -eq $Service) { return $health }
        Start-Sleep -Seconds 1
    }
    throw "Startup timed out. Check logs in $logDir"
}

function Prepare-RuntimeJar {
    if (-not (Test-Path -LiteralPath $jarPath)) { throw "Missing built service: $jarPath" }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [IO.Compression.ZipFile]::OpenRead($jarPath)
    try {
        $entries = @($archive.Entries.FullName)
        if (-not ($entries -like 'BOOT-INF/lib/*.jar') -or -not ($entries -contains 'BOOT-INF/classes/static/index.html')) {
            throw 'Service JAR is incomplete. Rebuild with Maven package before starting.'
        }
    } finally { $archive.Dispose() }
    $digest = (Get-FileHash -LiteralPath $jarPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $runtimeDir = Join-Path $logDir 'runtime'
    New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
    $runtimeJar = Join-Path $runtimeDir "docmind-$digest.jar"
    if (-not (Test-Path -LiteralPath $runtimeJar)) {
        $temp = Join-Path $runtimeDir (([guid]::NewGuid().ToString()) + '.tmp')
        try {
            Copy-Item -LiteralPath $jarPath -Destination $temp
            if ((Get-FileHash -LiteralPath $temp -Algorithm SHA256).Hash.ToLowerInvariant() -ne $digest) {
                throw 'Build changed during startup. Wait for Maven to finish and retry.'
            }
            Move-Item -LiteralPath $temp -Destination $runtimeJar
        } finally {
            if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp }
        }
    }
    if ((Get-FileHash -LiteralPath $runtimeJar -Algorithm SHA256).Hash.ToLowerInvariant() -ne $digest) {
        throw 'Runtime JAR integrity check failed.'
    }
    return $runtimeJar
}

function Assert-Workbench {
    $page = Invoke-WebRequest -Uri "$baseUrl/" -UseBasicParsing -TimeoutSec 5
    if ($page.StatusCode -ne 200 -or $page.Content -notmatch '/assets/[^" ]+\.js') {
        throw 'Workbench page is unavailable. Rebuild and restart the DocMind Java service.'
    }
    $asset = $Matches[0]
    $script = Invoke-WebRequest -Uri ($baseUrl + $asset) -UseBasicParsing -TimeoutSec 5
    if ($script.StatusCode -ne 200 -or $script.RawContentLength -eq 0) {
        throw 'Workbench JavaScript assets are unavailable.'
    }
}

try {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $worker = Read-Health "http://127.0.0.1:$WorkerPort/v1/health"
    if (-not $worker) {
        if (Port-InUse $WorkerPort) { throw "Port $WorkerPort is occupied by another application." }
        if (-not (Test-Path -LiteralPath $pythonExe)) { throw "Missing Python: $pythonExe" }
        $env:PYTHONIOENCODING = 'utf-8'
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
        Start-Process -FilePath $pythonExe -ArgumentList "-m docmind.server.worker --backend $Backend --port $WorkerPort --journal-dir logs/service/$Backend" -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir "worker-$stamp.out.log") -RedirectStandardError (Join-Path $logDir "worker-$stamp.err.log") | Out-Null
        $worker = Wait-Health "http://127.0.0.1:$WorkerPort/v1/health" 'docmind-python-worker'
    }
    if ($worker.service -ne 'docmind-python-worker' -or $worker.backend -ne $Backend -or -not $worker.ready) {
        throw 'The worker does not match the requested DocMind backend. Check its health and logs.'
    }
    $service = Read-Health "$baseUrl/api/health"
    if (-not $service) {
        if (Port-InUse $ServicePort) { throw "Port $ServicePort is occupied by another application." }
        $runtimeJar = Prepare-RuntimeJar
        $javaExe = (Get-Command java -ErrorAction Stop).Source
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
        Start-Process -FilePath $javaExe -ArgumentList "-jar `"$runtimeJar`" --server.port=$ServicePort --docmind.worker.base-url=http://127.0.0.1:$WorkerPort --docmind.expected-backend=$Backend" -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir "java-$stamp.out.log") -RedirectStandardError (Join-Path $logDir "java-$stamp.err.log") | Out-Null
        $service = Wait-Health "$baseUrl/api/health" 'docmind-service'
    }
    if ($service.service -ne 'docmind-service' -or $service.backend -ne $Backend -or $service.backend_mismatch -or -not $service.worker.ready) {
        throw 'DocMind service health check failed. Check the local service logs.'
    }
    Assert-Workbench
    if (-not $NoBrowser) { Start-Process $baseUrl }
    Write-Host "DocMind is ready: $baseUrl"
} catch {
    Write-Error $_.Exception.Message -ErrorAction Continue
    if ($NoBrowser) { exit 1 }
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show($_.Exception.Message, 'DocMind startup failed', 'OK', 'Error') | Out-Null
    exit 1
}
