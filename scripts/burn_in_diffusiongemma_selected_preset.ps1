[CmdletBinding()]
param(
    [string]$JsonOutput,
    [string]$MarkdownOutput,
    [switch]$ValidateOnly
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

$Stamp = [datetime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
if (-not $JsonOutput) {
    $JsonOutput = "var/benchmarks/diffusiongemma-selected-preset-burn-in-$Stamp.json"
}
if (-not $MarkdownOutput) { $MarkdownOutput = [IO.Path]::ChangeExtension($JsonOutput, '.md') }

$Parameters = @{
    DurationMinutes = 120
    IntervalSeconds = 5
    JsonOutput = $JsonOutput
    MarkdownOutput = $MarkdownOutput
    ValidateOnly = $ValidateOnly
}
& (Join-Path $PSScriptRoot 'stability_rocm_text_diffusion.ps1') @Parameters
