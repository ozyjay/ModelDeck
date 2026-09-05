[CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'High')]
param(
    [Parameter(Mandatory)][string]$Source,
    [string]$DataDirectory = '.modeldeck',
    [switch]$ReplaceExisting
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '../..')

$RepositoryRoot = (Get-Location).Path
Import-Module (Join-Path $PSScriptRoot '../lib/environment_helpers.psm1') -Force
Import-Module (Join-Path $PSScriptRoot '../lib/state_operations_helpers.psm1') -Force
Import-ModelDeckEnvironment -Path (Join-Path $RepositoryRoot '.env')
Assert-ModelDeckConfigurationMutable
Assert-ModelDeckServicesStopped -RepositoryRoot $RepositoryRoot
$DataPath = Resolve-CheckoutStateDirectory -RepositoryRoot $RepositoryRoot -DataDirectory $DataDirectory
$Python = if (Test-Path '.venv/bin/python') { '.venv/bin/python' } else { '.venv/Scripts/python.exe' }
if (-not (Test-Path -LiteralPath $Python)) { throw 'Run scripts/setup/setup.ps1 before importing ModelDeck state.' }

$SourcePath = [System.IO.Path]::GetFullPath($Source)
if (-not (Test-Path -LiteralPath $SourcePath -PathType Leaf)) { throw "No state archive exists at $SourcePath" }
$ExistingState = (Test-Path -LiteralPath $DataPath) -and @(Get-ChildItem -LiteralPath $DataPath -Force).Count -gt 0
if ($ExistingState -and -not $ReplaceExisting) {
    throw 'The checkout state is not empty. Re-run with -ReplaceExisting to create a timestamped backup and replace it.'
}

if ($PSCmdlet.ShouldProcess($DataPath, "Back up and replace inactive ModelDeck state from $SourcePath")) {
    if ($ReplaceExisting) {
        & $Python -m modeldeck.state_import $SourcePath $DataPath --replace-existing
    }
    else {
        & $Python -m modeldeck.state_import $SourcePath $DataPath
    }
    if ($LASTEXITCODE -ne 0) { throw 'The ModelDeck state import failed.' }
    Write-Host "ModelDeck state imported into $DataPath. Start services with scripts/operations/run.ps1 when ready."
}
