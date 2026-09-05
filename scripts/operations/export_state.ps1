[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory)][string]$Destination,
    [string]$DataDirectory = '.modeldeck'
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
if (-not (Test-Path -LiteralPath $Python)) { throw 'Run scripts/setup/setup.ps1 before exporting ModelDeck state.' }

$DestinationPath = [System.IO.Path]::GetFullPath($Destination)
if ($PSCmdlet.ShouldProcess($DestinationPath, "Export inactive ModelDeck state from $DataPath")) {
    & $Python -m modeldeck.state_export $DataPath $DestinationPath
    if ($LASTEXITCODE -ne 0) { throw 'The ModelDeck state export failed.' }
    Write-Host "ModelDeck state exported to $DestinationPath"
}
