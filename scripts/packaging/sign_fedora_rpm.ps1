[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$RpmPath,
    [Parameter(Mandatory)][string]$KeyId,
    [switch]$VerifyOnly
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $RpmPath -PathType Leaf)) { throw "RPM was not found: $RpmPath" }
if (-not (Get-Command gpg -ErrorAction SilentlyContinue)) { throw 'gpg is required to sign a release RPM.' }
if (-not (Get-Command rpm -ErrorAction SilentlyContinue)) { throw 'rpm is required to verify a release RPM.' }
if (-not $VerifyOnly) {
    if (-not (Get-Command rpmsign -ErrorAction SilentlyContinue)) { throw 'rpmsign is required; install the Fedora rpm-sign package.' }
    & gpg --list-secret-keys --with-colons $KeyId | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "No usable private GPG key was found for $KeyId." }
    & rpmsign --addsign --key-id $KeyId $RpmPath
    if ($LASTEXITCODE -ne 0) { throw "RPM signing failed: $RpmPath" }
}

$VerificationDirectory = Join-Path ([System.IO.Path]::GetTempPath()) "modeldeck-rpm-signing-$([guid]::NewGuid().ToString('N'))"
try {
    $PublicKeyPath = Join-Path $VerificationDirectory 'signing-key.asc'
    $VerificationDatabase = Join-Path $VerificationDirectory 'rpmdb'
    New-Item -ItemType Directory -Path $VerificationDirectory | Out-Null
    & gpg --batch --armor --export $KeyId > $PublicKeyPath
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $PublicKeyPath -PathType Leaf) -or (Get-Item $PublicKeyPath).Length -eq 0) {
        throw "Could not export the public GPG key for $KeyId."
    }
    & rpm --dbpath $VerificationDatabase --initdb
    if ($LASTEXITCODE -ne 0) { throw 'Could not initialise the temporary RPM verification database.' }
    & rpm --dbpath $VerificationDatabase --import $PublicKeyPath
    if ($LASTEXITCODE -ne 0) { throw "Could not import the public GPG key into the temporary verification database: $KeyId" }
    & rpm --dbpath $VerificationDatabase --checksig --verbose $RpmPath
    if ($LASTEXITCODE -ne 0) { throw "RPM signature verification failed: $RpmPath" }
}
finally {
    if (Test-Path $VerificationDirectory) { Remove-Item -Recurse -Force $VerificationDirectory }
}
