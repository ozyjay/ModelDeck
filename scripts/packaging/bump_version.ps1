<#
.SYNOPSIS
Advances ModelDeck's application version.

.DESCRIPTION
Updates the canonical MAJOR.MINOR.PATCH version in backend/modeldeck/__init__.py.
A major, minor, or patch bump can be selected, or an exact greater version can be supplied.
Every application-version change resets the canonical Fedora RPM release to 1.

.PARAMETER Part
Selects the semantic-version component to advance: Major resets minor and patch to zero;
Minor resets patch to zero; Patch increments only the patch component.

.PARAMETER Version
Sets an exact, greater MAJOR.MINOR.PATCH application version instead of selecting Part.

.PARAMETER VersionFile
Overrides the canonical application-version file. Intended for tests and controlled tooling.

.PARAMETER ReleaseFile
Overrides the canonical RPM-release file. Intended for tests and controlled tooling.

.EXAMPLE
pwsh -NoProfile -File scripts/packaging/bump_version.ps1 -Part Patch

Advances 0.1.1 to 0.1.2 and resets the RPM release to 1.

.EXAMPLE
pwsh -NoProfile -File scripts/packaging/bump_version.ps1 -Part Minor -WhatIf

Shows the proposed minor release without changing files.

.EXAMPLE
pwsh -NoProfile -File scripts/packaging/bump_version.ps1 -Version 1.0.0

Sets the application version to 1.0.0 and resets the RPM release to 1.

.NOTES
The requested version must be greater than the current version. Use Get-Help with -Full
to view this help from PowerShell.
#>
[CmdletBinding(DefaultParameterSetName = 'Part', SupportsShouldProcess)]
param(
    [Parameter(Mandatory, ParameterSetName = 'Part')]
    [ValidateSet('Major', 'Minor', 'Patch')]
    [string]$Part,
    [Parameter(Mandatory, ParameterSetName = 'Version')]
    [string]$Version,
    [string]$VersionFile = 'backend/modeldeck/__init__.py',
    [string]$ReleaseFile = 'packaging/fedora/rpm-release'
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '../..')

function Get-CanonicalVersion {
    param([string]$Path)

    if (-not (Test-Path $Path -PathType Leaf)) { throw "Version file was not found: $Path" }
    $Content = Get-Content -Path $Path -Raw
    $Matches = [regex]::Matches($Content, '(?m)^__version__\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"\s*$')
    if ($Matches.Count -ne 1) {
        throw "Version file must contain exactly one canonical __version__ assignment: $Path"
    }
    return [pscustomobject]@{ Content = $Content; Value = $Matches[0].Groups[1].Value }
}

function Assert-ReleaseFile {
    param([string]$Path)

    if (-not (Test-Path $Path -PathType Leaf)) { throw "RPM release file was not found: $Path" }
    $Release = (Get-Content -Path $Path -Raw).Trim()
    if ($Release -notmatch '^[1-9]\d*$') { throw "RPM release must be a positive integer: $Path" }
}

$Current = Get-CanonicalVersion -Path $VersionFile
Assert-ReleaseFile -Path $ReleaseFile
$CurrentVersion = [version]$Current.Value
if ($PSCmdlet.ParameterSetName -eq 'Part') {
    $NextVersion = switch ($Part) {
        'Major' { '{0}.0.0' -f ($CurrentVersion.Major + 1) }
        'Minor' { '{0}.{1}.0' -f $CurrentVersion.Major, ($CurrentVersion.Minor + 1) }
        'Patch' { '{0}.{1}.{2}' -f $CurrentVersion.Major, $CurrentVersion.Minor, ($CurrentVersion.Build + 1) }
    }
}
else {
    if ($Version -notmatch '^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$') {
        throw "Version must use release SemVer MAJOR.MINOR.PATCH: $Version"
    }
    $NextVersion = $Version
}

if ([version]$NextVersion -le $CurrentVersion) {
    throw "Version bump must be greater than $($Current.Value): $NextVersion"
}

$Updated = [regex]::new('(?m)^__version__\s*=\s*"[0-9]+\.[0-9]+\.[0-9]+"\s*$').Replace(
    $Current.Content,
    "__version__ = `"$NextVersion`"",
    1
)
if ($PSCmdlet.ShouldProcess("$VersionFile and $ReleaseFile", "Set application version to $NextVersion and reset RPM release to 1")) {
    Set-Content -Path $VersionFile -Value $Updated
    Set-Content -Path $ReleaseFile -Value '1'
}
Write-Host "Application version: $($Current.Value) -> $NextVersion; RPM release: 1"
