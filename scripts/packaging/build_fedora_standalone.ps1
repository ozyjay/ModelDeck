[CmdletBinding()]
param(
    [string]$Wheelhouse = 'packaging/fedora/wheelhouse',
    [string]$WheelhouseManifest = 'packaging/fedora/wheelhouse.sha256',
    [string]$OutputDirectory = 'dist/fedora',
    [string]$Python = 'python3.12',
    [switch]$PrepareWheelhouse
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '../..')

function Prepare-OfflineWheelhouse {
    param(
        [string]$Path,
        [string]$Manifest,
        [string]$RuntimePython
    )

    if (Test-Path $Path -PathType Leaf) { throw "Wheelhouse path is a file, not a directory: $Path" }
    if (Test-Path $Path -PathType Container) {
        $ExistingItems = @(Get-ChildItem -Path $Path -Force)
        if ($ExistingItems.Count) {
            throw "Wheelhouse is not empty: $Path. Use a new empty directory so a release inventory cannot mix artefacts."
        }
    }
    else {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }

    & $RuntimePython --version
    if ($LASTEXITCODE -ne 0) { throw "Python interpreter is unavailable: $RuntimePython" }

    foreach ($Requirements in @(
        'packaging/fedora/requirements-control.txt',
        'runtime/requirements-rocm72.txt',
        'requirements-rocm72-q4-gptqmodel.txt'
    )) {
        & $RuntimePython -m pip download --only-binary=:all: --dest $Path -r $Requirements
        if ($LASTEXITCODE -ne 0) { throw "Could not download required binary wheels: $Requirements" }
    }

    $Wheels = @(Get-ChildItem -Path $Path -File -Filter '*.whl' | Sort-Object Name)
    if (-not $Wheels.Count) { throw "No wheels were downloaded to: $Path" }
    $UnexpectedFiles = @(Get-ChildItem -Path $Path -File | Where-Object { $_.Extension -ne '.whl' })
    if ($UnexpectedFiles.Count) {
        throw "Wheelhouse contains non-wheel artefacts: $($UnexpectedFiles.Name -join ', ')"
    }

    $ManifestDirectory = Split-Path -Parent $Manifest
    if ($ManifestDirectory) { New-Item -ItemType Directory -Path $ManifestDirectory -Force | Out-Null }
    $Wheels | ForEach-Object {
        '{0}  {1}' -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant(), $_.Name
    } | Set-Content -Path $Manifest
    Write-Host "Prepared $($Wheels.Count) reviewed-wheel candidates and SHA-256 inventory: $Manifest"
}

if ($PrepareWheelhouse) {
    Prepare-OfflineWheelhouse -Path $Wheelhouse -Manifest $WheelhouseManifest -RuntimePython $Python
}

$BuildParameters = @{
    Wheelhouse = $Wheelhouse
    WheelhouseManifest = $WheelhouseManifest
    OutputDirectory = $OutputDirectory
    Python = $Python
}
& (Join-Path $PSScriptRoot 'build_fedora_rpm.ps1') @BuildParameters
if ($LASTEXITCODE -ne 0) { throw 'Fedora standalone RPM build failed.' }

$Packages = @(
    Get-ChildItem -Path $OutputDirectory -Filter 'modeldeck-*.x86_64.rpm' -File -Recurse |
        Sort-Object FullName
)
if (-not $Packages.Count) { throw "The Fedora build completed but no x86_64 ModelDeck RPM was found in: $OutputDirectory" }

Write-Host 'Unsigned standalone Fedora RPM:'
$Packages.FullName | ForEach-Object { Write-Host "  $_" }
Write-Host 'Sign the release RPM before distribution with scripts/packaging/sign_fedora_rpm.ps1.'
