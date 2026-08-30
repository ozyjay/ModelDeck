<#
.SYNOPSIS
Advances ModelDeck's Fedora RPM release number.

.DESCRIPTION
Updates the canonical positive-integer RPM release in packaging/fedora/rpm-release without
changing the application MAJOR.MINOR.PATCH version. Use this for a packaging-only rebuild.

.PARAMETER Increment
Increases the canonical RPM release by one.

.PARAMETER Release
Sets an exact, greater positive-integer RPM release instead of incrementing it.

.PARAMETER ReleaseFile
Overrides the canonical RPM-release file. Intended for tests and controlled tooling.

.EXAMPLE
pwsh -NoProfile -File scripts/packaging/bump_rpm_release.ps1 -Increment

Advances the RPM release from 1 to 2 without changing the application version.

.EXAMPLE
pwsh -NoProfile -File scripts/packaging/bump_rpm_release.ps1 -Release 3 -WhatIf

Shows the proposed RPM-release change without changing files.

.NOTES
The requested RPM release must be greater than the current release. Use Get-Help with -Full
to view this help from PowerShell.
#>
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
