[CmdletBinding()]
param(
    [string]$Wheelhouse = 'packaging/fedora/wheelhouse',
    [string]$WheelhouseManifest = 'packaging/fedora/wheelhouse.sha256',
    [string]$OutputDirectory = 'dist/fedora',
    [string]$Python = 'python3.12',
    [string]$RpmRelease = '1'
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
    $ManifestNames = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    foreach ($Entry in $Entries) {
        if ($Entry -notmatch '^([a-f0-9]{64})\s{2,}(.+)$') { throw "Invalid wheelhouse SHA-256 entry: $Entry" }
        $Expected = $Matches[1]
        $Name = $Matches[2]
        if (-not $ManifestNames.Add($Name)) { throw "Duplicate wheelhouse SHA-256 entry: $Name" }
        $Candidate = [System.IO.Path]::GetFullPath((Join-Path $WheelhouseRoot $Name))
        if (-not $Candidate.StartsWith($WheelhouseRoot + [System.IO.Path]::DirectorySeparatorChar)) {
            throw "Unsafe wheelhouse filename: $Name"
        }
        if (-not (Test-Path $Candidate -PathType Leaf)) { throw "Missing wheelhouse file: $Name" }
        $Actual = (Get-FileHash $Candidate -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($Actual -ne $Expected) { throw "Wheelhouse SHA-256 mismatch: $Name" }
    }
    $UnlistedFiles = @(
        Get-ChildItem -Path $WheelhouseRoot -File |
            Where-Object { -not $ManifestNames.Contains($_.Name) }
    )
    if ($UnlistedFiles.Count) {
        throw "Wheelhouse contains unlisted files: $($UnlistedFiles.Name -join ', ')"
    }
}

function Install-OfflineRequirements {
    param([string]$RuntimePython, [string]$Requirements)
    Write-Host "Installing offline runtime requirements: $Requirements"
    & $RuntimePython -m pip install --quiet --disable-pip-version-check --no-cache-dir --progress-bar off --no-index --find-links $Wheelhouse -r $Requirements
    if ($LASTEXITCODE -ne 0) { throw "Offline dependency installation failed: $Requirements" }
}

function Install-ModelDeckCode {
    param([string]$RuntimePython, [switch]$BuildEntrypoints)
    if ($BuildEntrypoints) {
        Write-Host 'Installing ModelDeck into the packaged control runtime.'
        & $RuntimePython -m pip install --quiet --disable-pip-version-check --no-cache-dir --progress-bar off --no-index --no-deps --no-build-isolation .
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

function New-BundledPythonRuntime {
    param([string]$SourcePython, [string]$Destination)

    $PythonPrefix = & $SourcePython -c 'import sys; print(sys.base_prefix)'
    if ($LASTEXITCODE -ne 0 -or -not $PythonPrefix) { throw 'Could not locate the Python 3.12 base runtime.' }
    $PythonVersion = & $SourcePython -c 'import platform; print(platform.python_version())'
    if ($LASTEXITCODE -ne 0 -or -not $PythonVersion) { throw 'Could not determine the Python 3.12 runtime version.' }
    $PythonPrefix = $PythonPrefix.Trim()
    $PythonVersion = $PythonVersion.Trim()
    New-Item -ItemType Directory -Force -Path "$Destination/bin", "$Destination/lib" | Out-Null
    Copy-Item "$PythonPrefix/bin/python3.12" "$Destination/bin/python3.12"
    Get-ChildItem "$PythonPrefix/lib" -File -Filter 'libpython3.12.so*' | Copy-Item -Destination "$Destination/lib/"
    Copy-Item "$PythonPrefix/lib/python3.12" -Destination "$Destination/lib/" -Recurse
    Remove-Item "$Destination/lib/python3.12/site-packages" -Recurse -Force -ErrorAction SilentlyContinue
    if (Test-Path "$PythonPrefix/LICENSE" -PathType Leaf) { Copy-Item "$PythonPrefix/LICENSE" "$Destination/LICENSE.python" }
    $BundledElves = @(
        Get-Item "$Destination/bin/python3.12"
        Get-ChildItem "$Destination/lib" -File -Filter 'libpython3.12.so*'
        Get-ChildItem "$Destination/lib/python3.12/lib-dynload" -File -Filter '*.so'
    )
    foreach ($Elf in $BundledElves) {
        & patchelf --remove-rpath $Elf.FullName
        if ($LASTEXITCODE -ne 0) { throw "Could not remove the build-host RPATH from: $($Elf.FullName)" }
    }
    return $PythonVersion
}

function Set-PackagedRuntimeLauncher {
    param(
        [string]$Runtime,
        [string]$InstalledRuntime,
        [string]$PythonVersion
    )

    $Bin = Join-Path $Runtime 'bin'
    Get-ChildItem $Bin -Force | Remove-Item -Recurse -Force
    $PythonLauncher = @'
#!/usr/bin/sh
export PYTHONHOME=/usr/libexec/modeldeck/python312
export PYTHONPATH=@INSTALLED_RUNTIME@/lib/python3.12/site-packages
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH=/usr/libexec/modeldeck/python312/lib:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=/usr/libexec/modeldeck/python312/lib
fi
exec /usr/libexec/modeldeck/python312/bin/python3.12 "$@"
'@.Replace('@INSTALLED_RUNTIME@', $InstalledRuntime)
    Set-Content -Path "$Bin/python" -Value $PythonLauncher -NoNewline
    & chmod 0755 "$Bin/python"
    if ($LASTEXITCODE -ne 0) { throw "Could not create the packaged Python launcher: $InstalledRuntime" }
    New-Item -ItemType SymbolicLink -Path "$Bin/python3" -Target 'python' | Out-Null
    New-Item -ItemType SymbolicLink -Path "$Bin/python3.12" -Target 'python' | Out-Null
    @(
        'home = /usr/libexec/modeldeck/python312/bin',
        'include-system-site-packages = false',
        "version = $PythonVersion",
        "executable = $InstalledRuntime/bin/python"
    ) | Set-Content -Path (Join-Path $Runtime 'pyvenv.cfg')
}

function New-PythonModuleLauncher {
    param(
        [string]$Path,
        [string]$Runtime,
        [string]$Module
    )

    $Launcher = @'
#!/usr/bin/sh
exec @RUNTIME@/bin/python -m @MODULE@ "$@"
'@.Replace('@RUNTIME@', $Runtime).Replace('@MODULE@', $Module)
    Set-Content -Path $Path -Value $Launcher -NoNewline
    & chmod 0755 $Path
    if ($LASTEXITCODE -ne 0) { throw "Could not create packaged launcher: $Path" }
}

Assert-OfflineWheelhouse -Path $Wheelhouse -Manifest $WheelhouseManifest
if (-not (Get-Command rpmbuild -ErrorAction SilentlyContinue)) { throw 'rpmbuild is required to create the Fedora package.' }
if (-not (Get-Command patchelf -ErrorAction SilentlyContinue)) { throw 'patchelf is required to create a relocatable Python runtime.' }
& (Join-Path $PSScriptRoot '../operations/build_frontend.ps1') -Check
& $Python --version
if ($LASTEXITCODE -ne 0) { throw "Python interpreter is unavailable: $Python" }

$Version = (Select-String -Path backend/modeldeck/__init__.py -Pattern '^__version__ = "([^"]+)"').Matches[0].Groups[1].Value
if (-not $Version) { throw 'Could not determine the ModelDeck package version.' }
if ($Version -notmatch '^\d+\.\d+\.\d+(?:[A-Za-z0-9.+~_-]+)?$') { throw "Invalid ModelDeck package version: $Version" }
if ($RpmRelease -notmatch '^[1-9]\d*$') { throw "RpmRelease must be a positive integer, not: $RpmRelease" }
$BuildId = "$Version-$RpmRelease"
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

    $BundledPythonVersion = New-BundledPythonRuntime -SourcePython $Python -Destination (Join-Path $Libexec 'python312')
    Set-PackagedRuntimeLauncher -Runtime $Control -InstalledRuntime '/usr/libexec/modeldeck/control' -PythonVersion $BundledPythonVersion
    Set-PackagedRuntimeLauncher -Runtime $Rocm -InstalledRuntime '/usr/libexec/modeldeck/rocm72' -PythonVersion $BundledPythonVersion
    Set-PackagedRuntimeLauncher -Runtime $Q4 -InstalledRuntime '/usr/libexec/modeldeck/rocm72-q4' -PythonVersion $BundledPythonVersion
    New-PythonModuleLauncher -Path "$Control/bin/modeldeck" -Runtime '/usr/libexec/modeldeck/control' -Module 'modeldeck'
    New-PythonModuleLauncher -Path "$Control/bin/modeldeck-gateway" -Runtime '/usr/libexec/modeldeck/control' -Module 'modeldeck.gateway.app'
    New-PythonModuleLauncher -Path "$Control/bin/modeldeck-import-state" -Runtime '/usr/libexec/modeldeck/control' -Module 'modeldeck.state_import'
    New-PythonModuleLauncher -Path "$Control/bin/modeldeck-export-state" -Runtime '/usr/libexec/modeldeck/control' -Module 'modeldeck.state_export'
    New-PythonModuleLauncher -Path "$Control/bin/modeldeck-probe" -Runtime '/usr/libexec/modeldeck/control' -Module 'modeldeck.hardware.probe'

    New-Item -ItemType Directory -Force -Path "$PayloadRoot/bin", "$PayloadRoot/lib/systemd/user", "$PayloadRoot/share/applications", "$PayloadRoot/share/icons/hicolor/scalable/apps", "$PayloadRoot/share/modeldeck", "$PayloadRoot/share/doc/modeldeck" | Out-Null
    Copy-Item 'packaging/fedora/com.modeldeck.ModelDeck.desktop' "$PayloadRoot/share/applications/"
    Copy-Item 'packaging/fedora/modeldeck.svg' "$PayloadRoot/share/icons/hicolor/scalable/apps/"
    Copy-Item 'docs/licenses/APACHE-2.0.txt' "$PayloadRoot/share/doc/modeldeck/"
    Copy-Item 'packaging/fedora/modeldeck.target' "$PayloadRoot/lib/systemd/user/"
    foreach ($Service in @('modeldeck-management.service', 'modeldeck-gateway.service')) {
        (Get-Content "packaging/fedora/$Service.in" -Raw).Replace('@BUILD_ID@', $BuildId) | Set-Content "$PayloadRoot/lib/systemd/user/$Service"
    }
    $DesktopPythonPath = Join-Path $Libexec 'desktop-python'
    New-Item -ItemType Directory -Force -Path "$DesktopPythonPath/modeldeck" | Out-Null
    Copy-Item 'backend/modeldeck/__init__.py' "$DesktopPythonPath/modeldeck/"
    Copy-Item 'backend/modeldeck/desktop' "$DesktopPythonPath/modeldeck/" -Recurse
    @{ build_id = $BuildId; package_version = $Version; architecture = 'x86_64'; models_included = $false } | ConvertTo-Json | Set-Content "$PayloadRoot/share/modeldeck/release.json"
    @"
#!/usr/bin/sh
export PYTHONPATH=/usr/libexec/modeldeck/desktop-python
exec /usr/bin/python3 -m modeldeck.desktop.app "`$@"
"@ | Set-Content "$PayloadRoot/bin/modeldeck-desktop" -NoNewline
    & chmod 0755 "$PayloadRoot/bin/modeldeck-desktop"
    if ($LASTEXITCODE -ne 0) { throw 'Could not mark the desktop launcher executable.' }

    Get-ChildItem $Libexec -Recurse -File -Filter 'direct_url.json' | Remove-Item -Force
    Get-ChildItem $Libexec -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force
    $AbsoluteRuntimeLinks = @(
        Get-ChildItem $Libexec -Recurse -Force |
            Where-Object { $_.LinkType -and [System.IO.Path]::IsPathRooted($_.Target) }
    )
    if ($AbsoluteRuntimeLinks.Count) {
        throw "Packaged runtime contains absolute symbolic links: $($AbsoluteRuntimeLinks.FullName -join ', ')"
    }

    New-Item -ItemType Directory -Force -Path "$RpmTop/SOURCES", "$RpmTop/BUILD", "$RpmTop/BUILDROOT", "$RpmTop/RPMS", "$RpmTop/SRPMS", "$RpmTop/TMP" | Out-Null
    & tar -C $Stage -czf "$RpmTop/SOURCES/modeldeck-payload.tar.gz" usr
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the RPM payload archive.' }
    New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
    & rpmbuild --quiet -bb packaging/fedora/modeldeck.spec --define "_topdir $RpmTop" --define "_tmppath $RpmTop/TMP" --define "_rpmdir $((Resolve-Path $OutputDirectory).Path)" --define "modeldeck_version $Version" --define "modeldeck_release $RpmRelease"
    if ($LASTEXITCODE -ne 0) { throw 'RPM build failed.' }
    Write-Host "Built unsigned ModelDeck RPMs in $OutputDirectory. Sign them separately with scripts/packaging/sign_fedora_rpm.ps1."
}
finally {
    Remove-Item $Stage -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item $RpmTop -Recurse -Force -ErrorAction SilentlyContinue
}
