[CmdletBinding()]
param(
    [string]$Wheelhouse = 'packaging/fedora/wheelhouse',
    [string]$WheelhouseManifest = 'packaging/fedora/wheelhouse.sha256',
    [string]$OutputDirectory = 'dist/fedora',
    [string]$Python = 'python3.12'
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '../..')

& (Join-Path $PSScriptRoot 'build_fedora_rpm.ps1') @PSBoundParameters
if ($LASTEXITCODE -ne 0) { throw 'Fedora standalone RPM build failed.' }

$Packages = @(
    Get-ChildItem -Path $OutputDirectory -Filter 'modeldeck-*.x86_64.rpm' -File -Recurse |
        Sort-Object FullName
)
if (-not $Packages.Count) { throw "The Fedora build completed but no x86_64 ModelDeck RPM was found in: $OutputDirectory" }

Write-Host 'Unsigned standalone Fedora RPM:'
$Packages.FullName | ForEach-Object { Write-Host "  $_" }
Write-Host 'Sign the release RPM before distribution with scripts/packaging/sign_fedora_rpm.ps1.'
