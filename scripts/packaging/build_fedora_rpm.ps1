[CmdletBinding()]
param(
    [string]$Wheelhouse = 'packaging/fedora/wheelhouse',
    [string]$WheelhouseManifest = 'packaging/fedora/wheelhouse.sha256',
    [string]$OutputDirectory = 'dist/fedora',
    [string]$Python = 'python3.12'
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '../..')

function Assert-OfflineWheelhouse {
    param([string]$Path, [string]$Manifest)

    if (-not (Test-Path $Path -PathType Container)) { throw "Offline wheelhouse was not found: $Path" }
    if (-not (Test-Path $Manifest -PathType Leaf)) { throw "Wheelhouse SHA-256 manifest was not found: $Manifest" }
    $Entries = @(
        Get-Content $Manifest | Where-Object { $_.Trim() -and -not $_.Trim().StartsWith('#') }
    )
    if (-not $Entries.Count) { throw 'Wheelhouse SHA-256 manifest has no entries.' }
    $WheelhouseRoot = [System.IO.Path]::GetFullPath($Path)
    foreach ($Entry in $Entries) {
        if ($Entry -notmatch '^([a-f0-9]{64})\s{2,}(.+)$') { throw "Invalid wheelhouse SHA-256 entry: $Entry" }
        $Expected = $Matches[1]
        $Name = $Matches[2]
        $Candidate = [System.IO.Path]::GetFullPath((Join-Path $WheelhouseRoot $Name))
        if (-not $Candidate.StartsWith($WheelhouseRoot + [System.IO.Path]::DirectorySeparatorChar)) {
            throw "Unsafe wheelhouse filename: $Name"
        }
        if (-not (Test-Path $Candidate -PathType Leaf)) { throw "Missing wheelhouse file: $Name" }
        $Actual = (Get-FileHash $Candidate -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($Actual -ne $Expected) { throw "Wheelhouse SHA-256 mismatch: $Name" }
    }
}

function Install-OfflineRequirements {
    param([string]$RuntimePython, [string]$Requirements)
    & $RuntimePython -m pip install --no-index --find-links $Wheelhouse -r $Requirements
    if ($LASTEXITCODE -ne 0) { throw "Offline dependency installation failed: $Requirements" }
}

function Install-ModelDeckCode {
    param([string]$RuntimePython, [switch]$BuildEntrypoints)
    if ($BuildEntrypoints) {
        & $RuntimePython -m pip install --no-index --no-deps --no-build-isolation .
        if ($LASTEXITCODE -ne 0) { throw 'Could not install ModelDeck into the control runtime.' }
        return
    }
    $PureLib = & $RuntimePython -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    if ($LASTEXITCODE -ne 0) { throw 'Could not locate the isolated runtime site-packages directory.' }
    Copy-Item backend/modeldeck -Destination $PureLib -Recurse -Force -Exclude '__pycache__', '*.pyc'
}

function New-OfflineRequirements {
    param([string]$Source, [string]$Destination)
    $Lines = Get-Content $Source | ForEach-Object {
        if ($_ -match '^torch @ ') { 'torch==2.9.1+rocm7.2.1.lw.gitff65f5bc' }
        elseif ($_ -match '^torchvision @ ') { 'torchvision==0.24.0+rocm7.2.1.gitb919bd0c' }
        elseif ($_ -match '^torchaudio @ ') { 'torchaudio==2.9.0+rocm7.2.1.gite3c6ee2b' }
        elseif ($_ -match '^triton @ ') { 'triton==3.5.1+rocm7.2.1.gita272dfa8' }
        else { $_ }
    }
    if ($Lines | Where-Object { $_ -match '\s@\shttps?://' }) { throw "Network URL remains in staged requirements: $Source" }
    Set-Content -Path $Destination -Value $Lines
}

Assert-OfflineWheelhouse -Path $Wheelhouse -Manifest $WheelhouseManifest
if (-not (Get-Command rpmbuild -ErrorAction SilentlyContinue)) { throw 'rpmbuild is required to create the Fedora package.' }
& (Join-Path $PSScriptRoot '../operations/build_frontend.ps1') -Check
& $Python --version
if ($LASTEXITCODE -ne 0) { throw "Python interpreter is unavailable: $Python" }

$Version = (Select-String -Path pyproject.toml -Pattern '^version = "([^"]+)"').Matches[0].Groups[1].Value
if (-not $Version) { throw 'Could not determine the ModelDeck package version.' }
$BuildId = "$Version-1"
$Stage = Join-Path ([System.IO.Path]::GetTempPath()) "modeldeck-rpm-stage-$([guid]::NewGuid().ToString('N'))"
$RpmTop = Join-Path ([System.IO.Path]::GetTempPath()) "modeldeck-rpmbuild-$([guid]::NewGuid().ToString('N'))"

try {
    $PayloadRoot = Join-Path $Stage 'usr'
    $Libexec = Join-Path $PayloadRoot 'libexec/modeldeck'
    $Control = Join-Path $Libexec 'control'
    $Rocm = Join-Path $Libexec 'rocm72'
    $Q4 = Join-Path $Libexec 'rocm72-q4'
    foreach ($Runtime in @($Control, $Rocm, $Q4)) {
        if ($Runtime -eq $Control) {
            & $Python -m venv --system-site-packages $Runtime
        }
        else {
            & $Python -m venv $Runtime
        }
        if ($LASTEXITCODE -ne 0) { throw "Could not create isolated runtime: $Runtime" }
    }
    Install-OfflineRequirements -RuntimePython "$Control/bin/python" -Requirements 'packaging/fedora/requirements-control.txt'
    Install-ModelDeckCode -RuntimePython "$Control/bin/python" -BuildEntrypoints
    $PreparedPrimary = Join-Path $Stage 'requirements-rocm72.txt'
    $PreparedQ4 = Join-Path $Stage 'requirements-rocm72-q4.txt'
    New-OfflineRequirements -Source 'runtime/requirements-rocm72.txt' -Destination $PreparedPrimary
    New-OfflineRequirements -Source 'requirements-rocm72-q4-gptqmodel.txt' -Destination $PreparedQ4
    Install-OfflineRequirements -RuntimePython "$Rocm/bin/python" -Requirements $PreparedPrimary
    Install-OfflineRequirements -RuntimePython "$Q4/bin/python" -Requirements $PreparedQ4
    Install-ModelDeckCode -RuntimePython "$Rocm/bin/python"
    Install-ModelDeckCode -RuntimePython "$Q4/bin/python"

    New-Item -ItemType Directory -Force -Path "$PayloadRoot/bin", "$PayloadRoot/lib/systemd/user", "$PayloadRoot/share/applications", "$PayloadRoot/share/icons/hicolor/scalable/apps", "$PayloadRoot/share/modeldeck", "$PayloadRoot/share/doc/modeldeck" | Out-Null
    Copy-Item 'packaging/fedora/com.modeldeck.ModelDeck.desktop' "$PayloadRoot/share/applications/"
    Copy-Item 'packaging/fedora/modeldeck.svg' "$PayloadRoot/share/icons/hicolor/scalable/apps/"
    Copy-Item 'docs/licenses/APACHE-2.0.txt' "$PayloadRoot/share/doc/modeldeck/"
    Copy-Item 'packaging/fedora/modeldeck.target' "$PayloadRoot/lib/systemd/user/"
    foreach ($Service in @('modeldeck-management.service', 'modeldeck-gateway.service')) {
        (Get-Content "packaging/fedora/$Service.in" -Raw).Replace('@BUILD_ID@', $BuildId) | Set-Content "$PayloadRoot/lib/systemd/user/$Service"
    }
    @{ build_id = $BuildId; package_version = $Version; architecture = 'x86_64'; models_included = $false } | ConvertTo-Json | Set-Content "$PayloadRoot/share/modeldeck/release.json"
    @"
#!/usr/bin/sh
exec /usr/libexec/modeldeck/control/bin/modeldeck-desktop "`$@"
"@ | Set-Content "$PayloadRoot/bin/modeldeck-desktop" -NoNewline
    & chmod 0755 "$PayloadRoot/bin/modeldeck-desktop"
    if ($LASTEXITCODE -ne 0) { throw 'Could not mark the desktop launcher executable.' }

    New-Item -ItemType Directory -Force -Path "$RpmTop/SOURCES", "$RpmTop/BUILD", "$RpmTop/BUILDROOT", "$RpmTop/RPMS", "$RpmTop/SRPMS" | Out-Null
    & tar -C $Stage -czf "$RpmTop/SOURCES/modeldeck-payload.tar.gz" usr
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the RPM payload archive.' }
    New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
    & rpmbuild -bb packaging/fedora/modeldeck.spec --define "_topdir $RpmTop" --define "_rpmdir $((Resolve-Path $OutputDirectory).Path)"
    if ($LASTEXITCODE -ne 0) { throw 'RPM build failed.' }
    Write-Host "Built unsigned ModelDeck RPMs in $OutputDirectory. Sign them separately with scripts/packaging/sign_fedora_rpm.ps1."
}
finally {
    Remove-Item $Stage -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item $RpmTop -Recurse -Force -ErrorAction SilentlyContinue
}
