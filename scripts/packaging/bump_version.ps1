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
