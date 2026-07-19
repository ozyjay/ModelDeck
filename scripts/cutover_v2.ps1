[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$DataDirectory = '.modeldeck',
    [string]$BackupDirectory = 'var/backups',
    [switch]$SkipStop
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

$RepositoryRoot = (Get-Location).Path
$DataPath = [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $DataDirectory))
$RepositoryPrefix = $RepositoryRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
if ($DataPath -eq $RepositoryRoot -or -not $DataPath.StartsWith($RepositoryPrefix, [System.StringComparison]::Ordinal)) {
    throw 'The v2 data directory must be a specific directory below the repository.'
}

if (-not $SkipStop) {
    & (Join-Path $PSScriptRoot 'stop.ps1')
}

$Timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$BackupRoot = [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $BackupDirectory))
if ($BackupRoot -eq $RepositoryRoot -or -not $BackupRoot.StartsWith($RepositoryPrefix, [System.StringComparison]::Ordinal)) {
    throw 'The backup directory must be a specific directory below the repository.'
}
$BackupPath = Join-Path -Path $BackupRoot -ChildPath "modeldeck-v1-$Timestamp"
New-Item -ItemType Directory -Force -Path $BackupPath | Out-Null

$DatabaseFiles = @('modeldeck.sqlite3', 'modeldeck.sqlite3-wal', 'modeldeck.sqlite3-shm')
foreach ($Name in $DatabaseFiles) {
    $Source = Join-Path $DataPath $Name
    if (Test-Path -LiteralPath $Source) {
        if ($PSCmdlet.ShouldProcess($Source, "Move legacy database file to $BackupPath")) {
            Move-Item -LiteralPath $Source -Destination (Join-Path $BackupPath $Name)
        }
    }
}

New-Item -ItemType Directory -Force -Path $DataPath | Out-Null
$Python = if (Test-Path '.venv/bin/python') { '.venv/bin/python' } else { '.venv/Scripts/python.exe' }
if (-not (Test-Path $Python)) { throw 'Run scripts/setup.ps1 before the v2 cut-over.' }

if ($PSCmdlet.ShouldProcess((Join-Path $DataPath 'modeldeck.sqlite3'), 'Initialise the empty v2 database')) {
    $PreviousDataDirectory = $Env:MODELDECK_DATA_DIR
    try {
        $Env:MODELDECK_DATA_DIR = $DataPath
        & $Python -c 'from modeldeck.compatibility import CompatibilityStore; from modeldeck.config import Settings; s=Settings.from_env(); CompatibilityStore(s.data_dir / "modeldeck.sqlite3").initialise_v2()'
        if ($LASTEXITCODE -ne 0) { throw 'The empty v2 database could not be initialised.' }
    }
    finally {
        if ($null -eq $PreviousDataDirectory) { Remove-Item Env:MODELDECK_DATA_DIR -ErrorAction SilentlyContinue }
        else { $Env:MODELDECK_DATA_DIR = $PreviousDataDirectory }
    }
}

Write-Host "ModelDeck v2 is ready. Legacy database backup: $BackupPath"
Write-Host 'Model caches, benchmark output, runtime manifests and logs were not changed.'
