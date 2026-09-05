[CmdletBinding()]
param(
    [string]$RuntimeRoot = ''
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '../..')

$Revision = '9d77fa17254e1dee4b9e92504c91611a60b1359f'
$Root = if ($RuntimeRoot) { [System.IO.Path]::GetFullPath($RuntimeRoot) } else { '.runtime-tools/llama.cpp' }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'git is required to acquire llama.cpp.' }
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) { throw 'cmake is required to build llama.cpp.' }
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) { throw 'ninja is required to build llama.cpp.' }
if (-not (Get-Command glslc -ErrorAction SilentlyContinue) -or
    -not (Test-Path '/usr/include/vulkan/vulkan.h')) {
    throw 'The Fedora Vulkan development tools are missing; install glslc and vulkan-loader-devel.'
}

if (-not (Test-Path "$Root/.git")) {
    New-Item -ItemType Directory -Force '.runtime-tools' | Out-Null
    git clone https://github.com/ggml-org/llama.cpp.git $Root
    if ($LASTEXITCODE -ne 0) { throw 'Could not clone the pinned llama.cpp source.' }
}

git -C $Root fetch --depth 1 origin $Revision
if ($LASTEXITCODE -ne 0) { throw 'Could not fetch the pinned llama.cpp revision.' }
git -C $Root checkout --detach $Revision
if ($LASTEXITCODE -ne 0) { throw 'Could not check out the pinned llama.cpp revision.' }
cmake -S $Root -B "$Root/build" -G Ninja -DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { throw 'Could not configure the llama.cpp Vulkan build.' }
cmake --build "$Root/build" --target llama-server llama-bench
if ($LASTEXITCODE -ne 0) { throw 'Could not build the llama.cpp Vulkan tools.' }
New-Item -ItemType Directory -Force "$Root/bin" | Out-Null
Copy-Item "$Root/build/bin/llama-server" "$Root/bin/llama-server" -Force
Copy-Item "$Root/build/bin/llama-bench" "$Root/bin/llama-bench" -Force
$Help = & "$Root/bin/llama-server" --help 2>&1 | Out-String
foreach ($RequiredFlag in @('--spec-type', '--spec-draft-model', '--spec-draft-n-max', '--mmproj')) {
    if (-not $Help.Contains($RequiredFlag)) {
        throw "The pinned llama-server build does not expose required flag $RequiredFlag."
    }
}
$Receipt = [ordered]@{
    format = 'modeldeck-llama-build'
    version = 1
    commit = $Revision
    executable_sha256 = (Get-FileHash "$Root/bin/llama-server" -Algorithm SHA256).Hash.ToLowerInvariant()
    backend = 'Vulkan'
    operating_system = 'linux'
    architecture = 'x86_64'
}
$Receipt | ConvertTo-Json | Set-Content "$Root/bin/modeldeck-build.json" -Encoding utf8
Write-Host "Pinned llama.cpp Vulkan runtime $Revision is ready."
