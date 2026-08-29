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

$VerificationOutput = @(& rpm --checksig --verbose $RpmPath 2>&1)
$VerificationExitCode = $LASTEXITCODE
$VerificationOutput | ForEach-Object { Write-Host $_ }
if ($VerificationExitCode -ne 0) {
    $VerificationText = $VerificationOutput -join [Environment]::NewLine
    if ($VerificationText -match '\bNOKEY\b') {
        Write-Warning 'The RPM has a signature, but this machine has not imported the corresponding public key into its RPM trust database.'
        Write-Host "For full local verification, export the public key with 'gpg --armor --export $KeyId' and import it using 'sudo rpm --import <public-key-file>'."
        $global:LASTEXITCODE = 0
    }
    else {
        throw "RPM signature verification failed: $RpmPath"
    }
}
