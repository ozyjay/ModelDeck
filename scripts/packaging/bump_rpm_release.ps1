[CmdletBinding(DefaultParameterSetName = 'Increment', SupportsShouldProcess)]
param(
    [Parameter(Mandatory, ParameterSetName = 'Increment')]
    [switch]$Increment,
    [Parameter(Mandatory, ParameterSetName = 'Release')]
    [string]$Release,
    [string]$ReleaseFile = 'packaging/fedora/rpm-release'
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '../..')

if (-not (Test-Path $ReleaseFile -PathType Leaf)) { throw "RPM release file was not found: $ReleaseFile" }
$Current = (Get-Content -Path $ReleaseFile -Raw).Trim()
if ($Current -notmatch '^[1-9]\d*$') { throw "RPM release must be a positive integer: $ReleaseFile" }
if ($PSCmdlet.ParameterSetName -eq 'Increment') {
    $Next = ([int]$Current + 1).ToString()
}
else {
    if ($Release -notmatch '^[1-9]\d*$') { throw "RPM release must be a positive integer: $Release" }
    $Next = $Release
}
if ([int]$Next -le [int]$Current) { throw "RPM release bump must be greater than ${Current}: $Next" }

if ($PSCmdlet.ShouldProcess($ReleaseFile, "Set RPM release to $Next")) {
    Set-Content -Path $ReleaseFile -Value $Next
}
Write-Host "RPM release: $Current -> $Next"
