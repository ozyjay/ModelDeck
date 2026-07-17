[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateRange(1, [int]::MaxValue)]
    [int]$BrowserProcessId
)

$ErrorActionPreference = 'Continue'
Set-Location (Join-Path $PSScriptRoot '..')

try {
    $BrowserProcess = Get-Process -Id $BrowserProcessId -ErrorAction Stop
    $BrowserProcess.WaitForExit()
}
catch {
    Write-Verbose "The booth browser process $BrowserProcessId has already exited."
}
finally {
    & (Join-Path $PSScriptRoot 'stop.ps1')
}
