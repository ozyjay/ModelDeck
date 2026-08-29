[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$RpmPath,
    [Parameter(Mandatory)][string]$KeyId
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $RpmPath -PathType Leaf)) { throw "RPM was not found: $RpmPath" }
if (-not (Get-Command rpmsign -ErrorAction SilentlyContinue)) { throw 'rpmsign is required; install the Fedora rpm-sign package.' }
if (-not (Get-Command gpg -ErrorAction SilentlyContinue)) { throw 'gpg is required to sign a release RPM.' }
& gpg --list-secret-keys --with-colons $KeyId | Out-Null
if ($LASTEXITCODE -ne 0) { throw "No usable private GPG key was found for $KeyId." }
& rpmsign --addsign --key-id $KeyId $RpmPath
if ($LASTEXITCODE -ne 0) { throw "RPM signing failed: $RpmPath" }
& rpm --checksig --verbose $RpmPath
if ($LASTEXITCODE -ne 0) { throw "RPM signature verification failed: $RpmPath" }
