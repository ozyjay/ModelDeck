[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$DataDirectory = '.modeldeck',
    [string]$BackupDirectory = 'var/backups',
    [switch]$SkipStop
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '../..')

$RepositoryRoot = (Get-Location).Path
$DataPath = [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $DataDirectory))
$RepositoryPrefix = $RepositoryRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
if ($DataPath -eq $RepositoryRoot -or -not $DataPath.StartsWith($RepositoryPrefix, [System.StringComparison]::Ordinal)) {
    throw 'The ModelDeck data directory must be a specific directory below the repository.'
}

$DatabasePath = Join-Path $DataPath 'modeldeck.sqlite3'
if (-not (Test-Path -LiteralPath $DatabasePath)) { throw "No ModelDeck database exists at $DatabasePath" }

$Timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$BackupRoot = [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $BackupDirectory))
if ($BackupRoot -eq $RepositoryRoot -or -not $BackupRoot.StartsWith($RepositoryPrefix, [System.StringComparison]::Ordinal)) {
    throw 'The backup directory must be a specific directory below the repository.'
}
$BackupPath = Join-Path -Path $BackupRoot -ChildPath "modeldeck-v3-$Timestamp"
$Python = if (Test-Path '.venv/bin/python') { '.venv/bin/python' } else { '.venv/Scripts/python.exe' }
if (-not (Test-Path $Python)) { throw 'Run scripts/setup/setup.ps1 before migrating ModelDeck.' }

if ($PSCmdlet.ShouldProcess($DatabasePath, "Back up and migrate to ModelDeck v4 at $BackupPath")) {
    if (-not $SkipStop) { & (Join-Path $PSScriptRoot '../operations/stop.ps1') }
    New-Item -ItemType Directory -Force -Path $BackupPath | Out-Null
    foreach ($Name in @('modeldeck.sqlite3', 'modeldeck.sqlite3-wal', 'modeldeck.sqlite3-shm')) {
        $Source = Join-Path $DataPath $Name
        if (Test-Path -LiteralPath $Source) { Copy-Item -LiteralPath $Source -Destination (Join-Path $BackupPath $Name) }
    }
    & $Python -m modeldeck.migrate_v3_to_v4 $DatabasePath
    if ($LASTEXITCODE -ne 0) { throw 'The v3 to v4 migration failed. The database backup was retained.' }
    Write-Host "ModelDeck v4 migration complete. Backup: $BackupPath"
    Write-Host 'Workers, Routing Profiles, model policy and compatibility evidence were preserved.'
}
else {
    Write-Host 'ModelDeck v4 migration was not performed; no services were stopped and no backup was created.'
}
