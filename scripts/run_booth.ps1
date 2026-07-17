[CmdletBinding()]
param(
    [switch]$Windowed,
    [string]$Browser,
    [ValidateRange(1, 300)][int]$ReadyTimeoutSeconds = 30
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
Import-Module (Join-Path $PSScriptRoot 'booth_helpers.psm1') -Force

$ConsoleUrl = 'http://127.0.0.1:3600'
$HealthUrls = @(
    "$ConsoleUrl/api/health",
    'http://127.0.0.1:8600/v1/health'
)
$ProfileDirectory = Join-Path (Get-Location) '.booth-browser-profile'
$BrowserPath = Resolve-BoothBrowser -Browser $Browser
$BrowserProcess = $null
$ShouldStopServices = $false
$BoothHandedOff = $false

function Wait-ModelDeckBoothReady {
    $Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    while ($Stopwatch.Elapsed.TotalSeconds -lt $ReadyTimeoutSeconds) {
        $Ready = $true
        foreach ($HealthUrl in $HealthUrls) {
            try {
                $Response = Invoke-WebRequest -Uri $HealthUrl -TimeoutSec 1 -UseBasicParsing
                if ($Response.StatusCode -ne 200) { $Ready = $false }
            }
            catch { $Ready = $false }
        }
        if ($Ready) { return }

        foreach ($ServiceName in @('management', 'gateway')) {
            $PidPath = "var/run/$ServiceName.pid"
            if ((Test-Path $PidPath) -and
                -not (Get-Process -Id ([int](Get-Content $PidPath)) -ErrorAction SilentlyContinue)) {
                throw "The $ServiceName service exited before booth mode became ready. Check var/log/$ServiceName-error.log."
            }
        }
        Start-Sleep -Milliseconds 200
    }
    throw "ModelDeck booth mode did not become ready within $ReadyTimeoutSeconds seconds."
}

try {
    Write-Host 'Stopping any earlier ModelDeck session…'
    & (Join-Path $PSScriptRoot 'stop.ps1')

    $ShouldStopServices = $true
    Write-Host "Starting ModelDeck booth services at $ConsoleUrl"
    & (Join-Path $PSScriptRoot 'run.ps1') -OpenDay
    Wait-ModelDeckBoothReady

    New-Item -ItemType Directory -Force -Path $ProfileDirectory | Out-Null
    $BrowserArguments = Get-BoothBrowserArguments `
        -Url $ConsoleUrl `
        -ProfileDirectory $ProfileDirectory `
        -Windowed:$Windowed
    Write-Host 'ModelDeck is ready. Opening the dedicated booth browser.'
    $BrowserProcess = Start-Process `
        -FilePath $BrowserPath `
        -ArgumentList $BrowserArguments `
        -RedirectStandardOutput 'var/log/booth-browser.log' `
        -RedirectStandardError 'var/log/booth-browser-error.log' `
        -PassThru
    $PowerShellPath = (Get-Process -Id $PID).Path
    $WatcherArguments = @(
        '-NoProfile',
        '-File',
        (Join-Path $PSScriptRoot 'watch_booth.ps1'),
        '-BrowserProcessId',
        $BrowserProcess.Id
    )
    Start-Process `
        -FilePath $PowerShellPath `
        -ArgumentList $WatcherArguments `
        -RedirectStandardOutput 'var/log/booth-watcher.log' `
        -RedirectStandardError 'var/log/booth-watcher-error.log' | Out-Null
    $BoothHandedOff = $true
    Write-Host 'Booth mode is running in the background. Close the booth or run scripts/stop.ps1 to stop ModelDeck.'
    exit 0
}
catch {
    Write-Error "Could not start booth mode: $($_.Exception.Message)"
    exit 1
}
finally {
    if (-not $BoothHandedOff -and $BrowserProcess -and -not $BrowserProcess.HasExited) {
        Stop-Process -Id $BrowserProcess.Id -ErrorAction SilentlyContinue
    }
    if ($ShouldStopServices -and -not $BoothHandedOff) {
        & (Join-Path $PSScriptRoot 'stop.ps1')
    }
}
