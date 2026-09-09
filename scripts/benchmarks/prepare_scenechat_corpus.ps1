[CmdletBinding()]
param(
    [string]$Plan,
    [string]$Sources,
    [string]$Candidate,
    [string]$Review,
    [Parameter(Mandatory)][string]$Output
)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '../..')
$Arguments = @('scripts/benchmarks/prepare_scenechat_corpus.py', '--output', $Output)
if ($Plan) { $Arguments += @('--plan', $Plan) }
if ($Sources) { $Arguments += @('--sources', $Sources) }
if ($Candidate) { $Arguments += @('--candidate', $Candidate) }
if ($Review) { $Arguments += @('--review', $Review) }
& .venv/bin/python @Arguments
if ($LASTEXITCODE -ne 0) { throw 'SceneChat corpus preparation or approval import failed.' }
