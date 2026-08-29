[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$RpmPath,
    [string]$KeyId,
    [switch]$CreateKey,
    [string]$SigningName,
    [string]$SigningEmail,
    [string]$KeyExpiry = '2y',
    [switch]$VerifyOnly
)

$ErrorActionPreference = 'Stop'

function Get-SecretKeyFingerprints {
    param([string]$Lookup)

    $Arguments = @('--batch', '--with-colons', '--list-secret-keys', '--fingerprint')
    if ($Lookup) { $Arguments += $Lookup }
    $Records = @(& gpg @Arguments 2>$null)
    if ($LASTEXITCODE -ne 0) { throw 'Could not inspect the local GPG secret-key ring.' }

    $Fingerprints = [System.Collections.Generic.List[string]]::new()
    $AwaitingPrimaryFingerprint = $false
    foreach ($Record in $Records) {
        $Fields = $Record -split ':'
        if ($Fields[0] -eq 'sec') {
            $AwaitingPrimaryFingerprint = $true
            continue
        }
        if ($AwaitingPrimaryFingerprint -and $Fields[0] -eq 'fpr' -and $Fields.Count -gt 9 -and $Fields[9]) {
            $Fingerprints.Add($Fields[9])
            $AwaitingPrimaryFingerprint = $false
        }
    }
    return @($Fingerprints)
}

function New-ReleaseSigningKey {
    param(
        [string]$Name,
        [string]$Email,
        [string]$Expiry
    )

    if (-not $Name) { throw 'SigningName is required when creating a new GPG signing key.' }
    if ($Name -match '[\r\n]' -or $Email -match '[\r\n]') { throw 'SigningName and SigningEmail must be single-line values.' }
    $Identity = if ($Email) { "$Name <$Email>" } else { $Name }

    Write-Host "Creating a protected RSA signing key for: $Identity"
    Write-Host 'GPG may open pinentry to request a passphrase. Store the recovery material outside this repository.'
    & gpg --quick-generate-key $Identity rsa4096 sign $Expiry
    if ($LASTEXITCODE -ne 0) { throw 'GPG key creation failed.' }

    $Fingerprints = @(Get-SecretKeyFingerprints -Lookup $Identity)
    if ($Fingerprints.Count -ne 1) { throw "Could not uniquely identify the signing key created for: $Identity" }
    return $Fingerprints[0]
}

if (-not (Test-Path $RpmPath -PathType Leaf)) { throw "RPM was not found: $RpmPath" }
if (-not (Get-Command gpg -ErrorAction SilentlyContinue)) { throw 'gpg is required; install the Fedora gnupg2 package.' }

$SelectedKey = $KeyId
if (-not $SelectedKey) {
    $AvailableKeys = @(Get-SecretKeyFingerprints)
    if ($AvailableKeys.Count -eq 1) {
        $SelectedKey = $AvailableKeys[0]
    }
    elseif ($AvailableKeys.Count -eq 0) {
        if (-not $CreateKey) {
            throw 'No secret GPG key is available. Rerun with -CreateKey -SigningName "Your release identity", or import the existing release key and pass -KeyId.'
        }
        $SelectedKey = New-ReleaseSigningKey -Name $SigningName -Email $SigningEmail -Expiry $KeyExpiry
    }
    else {
        throw 'Multiple secret GPG keys are available. Pass -KeyId explicitly so the intended release key is used.'
    }
}

$SigningScript = Join-Path $PSScriptRoot 'sign_fedora_rpm.ps1'
if ($VerifyOnly) {
    Write-Host "Verifying release RPM with GPG key: $SelectedKey"
    & $SigningScript -RpmPath $RpmPath -KeyId $SelectedKey -VerifyOnly
    if ($LASTEXITCODE -ne 0) { throw "Release RPM signature verification failed: $RpmPath" }
    Write-Host "Verified release RPM: $RpmPath"
}
else {
    Write-Host "Signing release RPM with GPG key: $SelectedKey"
    & $SigningScript -RpmPath $RpmPath -KeyId $SelectedKey
    if ($LASTEXITCODE -ne 0) { throw "Release RPM signing failed: $RpmPath" }
    Write-Host "Signed and verified release RPM: $RpmPath"
}
