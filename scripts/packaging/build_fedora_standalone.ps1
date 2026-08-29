[CmdletBinding()]
param(
    [string]$Wheelhouse = 'packaging/fedora/wheelhouse',
    [string]$WheelhouseManifest = 'packaging/fedora/wheelhouse.sha256',
    [string]$OutputDirectory = 'dist/fedora',
    [string]$Python = '',
    [switch]$PrepareWheelhouse,
    [switch]$ReplaceWheelhouse
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '../..')

function Get-PythonVersion {
    param([string]$RuntimePython)

    try {
        $Version = & $RuntimePython -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>$null
        if ($LASTEXITCODE -ne 0) { return $null }
        return "$Version".Trim()
    }
    catch {
        return $null
    }
}

function Assert-Python312 {
    param([string]$RuntimePython)

    $Version = Get-PythonVersion -RuntimePython $RuntimePython
    if (-not $Version) {
        throw "Python 3.12 is required. Could not run '$RuntimePython'; pass a working interpreter using -Python."
    }
    if ($Version -notmatch '^3\.12\.\d+$') {
        throw "Python 3.12 is required for the pinned ROCm wheels, but '$RuntimePython' is Python $Version. Pass a Python 3.12 interpreter using -Python."
    }
    return $Version
}

function Resolve-Python312 {
    param([string]$RequestedPython)

    if ($RequestedPython) {
        $Version = Assert-Python312 -RuntimePython $RequestedPython
        Write-Host "Using Python ${Version}: $RequestedPython"
        return $RequestedPython
    }

    $Candidates = [System.Collections.Generic.List[string]]::new()
    $Python312Command = Get-Command python3.12 -ErrorAction SilentlyContinue
    if ($Python312Command) { $Candidates.Add($Python312Command.Source) }

    if (Get-Command pyenv -ErrorAction SilentlyContinue) {
        $PyenvVersions = @(
            & pyenv versions --bare 2>$null |
                Where-Object { $_ -match '^3\.12\.\d+$' } |
                Sort-Object { [version]$_ } -Descending
        )
        foreach ($PyenvVersion in $PyenvVersions) {
            $PyenvPrefix = & pyenv prefix $PyenvVersion 2>$null
            if ($LASTEXITCODE -eq 0 -and $PyenvPrefix) {
                $Candidates.Add((Join-Path $PyenvPrefix 'bin/python'))
            }
        }
    }

    $Python3Command = Get-Command python3 -ErrorAction SilentlyContinue
    if ($Python3Command) { $Candidates.Add($Python3Command.Source) }

    $Seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    foreach ($Candidate in $Candidates) {
        if (-not $Seen.Add($Candidate)) { continue }
        $Version = Get-PythonVersion -RuntimePython $Candidate
        if ($Version -match '^3\.12\.\d+$') {
            Write-Host "Using Python ${Version}: $Candidate"
            return $Candidate
        }
    }

    throw 'Python 3.12 was not found. Install it with pyenv or pass its full path using -Python.'
}

function Prepare-OfflineWheelhouse {
    param(
        [string]$Path,
        [string]$Manifest,
        [string]$RuntimePython,
        [switch]$Replace
    )

    $FullPath = [System.IO.Path]::GetFullPath($Path)
    $RepositoryRoot = [System.IO.Path]::GetFullPath((Get-Location).Path)
    if ($FullPath -eq [System.IO.Path]::GetPathRoot($FullPath) -or $FullPath -eq $RepositoryRoot) {
        throw "Unsafe wheelhouse path: $FullPath"
    }
    if (Test-Path $FullPath -PathType Leaf) { throw "Wheelhouse path is a file, not a directory: $FullPath" }

    $ExistingItems = @()
    if (Test-Path $FullPath -PathType Container) {
        $ExistingItems = @(Get-ChildItem -Path $FullPath -Force)
        if ($ExistingItems.Count) {
            if (-not $Replace) {
                throw "Wheelhouse is not empty: $FullPath. Use the verified offline build or rerun preparation with -ReplaceWheelhouse."
            }
        }
    }

    & $RuntimePython -m pip --version
    if ($LASTEXITCODE -ne 0) { throw "pip is unavailable for the selected Python interpreter: $RuntimePython" }

    $StagingPath = Join-Path ([System.IO.Path]::GetTempPath()) "modeldeck-wheelhouse-$([guid]::NewGuid().ToString('N'))"
    New-Item -ItemType Directory -Path $StagingPath | Out-Null
    try {
        foreach ($Requirements in @(
            'packaging/fedora/requirements-control.txt',
            'runtime/requirements-rocm72.txt',
            'requirements-rocm72-q4-gptqmodel.txt'
        )) {
            Write-Host "Preparing pinned wheels for $Requirements..."
            & $RuntimePython -m pip wheel --quiet --disable-pip-version-check --no-cache-dir --progress-bar off --wheel-dir $StagingPath -r $Requirements
            if ($LASTEXITCODE -ne 0) {
                throw "Could not prepare required wheels for $Requirements. The existing wheelhouse was not changed."
            }
        }

        $Wheels = @(Get-ChildItem -Path $StagingPath -File -Filter '*.whl' | Sort-Object Name)
        if (-not $Wheels.Count) { throw 'No wheels were downloaded.' }
        $UnexpectedFiles = @(Get-ChildItem -Path $StagingPath -File | Where-Object { $_.Extension -ne '.whl' })
        if ($UnexpectedFiles.Count) {
            throw "Prepared wheelhouse contains non-wheel artefacts: $($UnexpectedFiles.Name -join ', ')"
        }
        $ManifestLines = @($Wheels | ForEach-Object {
            '{0}  {1}' -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant(), $_.Name
        })

        $WheelhouseParent = Split-Path -Parent $FullPath
        New-Item -ItemType Directory -Path $WheelhouseParent -Force | Out-Null
        if (Test-Path $FullPath -PathType Container) {
            if ($ExistingItems.Count) {
                $BackupSuffix = "$([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))-$([guid]::NewGuid().ToString('N').Substring(0, 8))"
                $BackupPath = "$FullPath.backup-$BackupSuffix"
                Move-Item -Path $FullPath -Destination $BackupPath
                Write-Host "Moved the previous wheelhouse to: $BackupPath"
            }
            else {
                Remove-Item -Path $FullPath
            }
        }
        Move-Item -Path $StagingPath -Destination $FullPath
        $StagingPath = $null

        $ManifestPath = [System.IO.Path]::GetFullPath($Manifest)
        $ManifestDirectory = Split-Path -Parent $ManifestPath
        if ($ManifestDirectory) { New-Item -ItemType Directory -Path $ManifestDirectory -Force | Out-Null }
        $ManifestLines | Set-Content -Path $ManifestPath
        Write-Host "Prepared $($Wheels.Count) wheel candidates and SHA-256 inventory: $ManifestPath"
    }
    finally {
        if ($StagingPath -and (Test-Path $StagingPath -PathType Container)) {
            Remove-Item -Path $StagingPath -Recurse -Force
        }
    }
}

$ResolvedPython = Resolve-Python312 -RequestedPython $Python

if ($PrepareWheelhouse) {
    Prepare-OfflineWheelhouse -Path $Wheelhouse -Manifest $WheelhouseManifest -RuntimePython $ResolvedPython -Replace:$ReplaceWheelhouse
}
elseif ($ReplaceWheelhouse) {
    throw '-ReplaceWheelhouse requires -PrepareWheelhouse.'
}

$BuildParameters = @{
    Wheelhouse = $Wheelhouse
    WheelhouseManifest = $WheelhouseManifest
    OutputDirectory = $OutputDirectory
    Python = $ResolvedPython
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
