[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateSet('schemas', 'prepare-workers', 'matrix', 'freeze-installation', 'register', 'export-review', 'report', 'finalists', 'release-decision')]
    [string]$Action,
    [Parameter(ValueFromRemainingArguments)][string[]]$ToolArguments
)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '../..')
$Interpreter = if ($Action -eq 'freeze-installation') { '.venv-rocm72/bin/python' } else { '.venv/bin/python' }
if (-not (Test-Path $Interpreter)) { throw "Required project interpreter is missing: $Interpreter" }
& $Interpreter -m modeldeck.scenechat_experiment_cli $Action @ToolArguments
if ($LASTEXITCODE -ne 0) { throw "SceneChat evidence command failed: $Action" }
