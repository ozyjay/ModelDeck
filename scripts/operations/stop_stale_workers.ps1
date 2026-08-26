param(
    [switch]$Quiet
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '../..')
$Root = (Get-Location).Path
$WorkerModules = @(
    'modeldeck.workers.mock_worker',
    'modeldeck.workers.autoregressive_worker',
    'modeldeck.workers.embedding_worker',
    'modeldeck.workers.llama_vulkan_worker',
    'modeldeck.workers.moshiko_worker',
    'modeldeck.workers.qwen35_chat_worker',
    'modeldeck.workers.qwen35_worker',
    'modeldeck.workers.scenechat_worker',
    'modeldeck.workers.speech_recognition_worker',
    'modeldeck.workers.text_diffusion_worker',
    'modeldeck.workers.translation_worker',
    'modeldeck.workers.tts_worker'
)
$WorkerPorts = @(8610..8624)
$TrustedLlamaServer = "$Root/.runtime-tools/llama.cpp/bin/llama-server"
$Stopped = @()

function Stop-StaleProcess {
    param(
        [int]$ProcessId,
        [string]$Description,
        [string]$ShutdownUrl
    )

    if (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) { return $false }
    if ($ShutdownUrl) {
        try {
            Invoke-RestMethod -Method Post -Uri $ShutdownUrl -TimeoutSec 2 | Out-Null
            try { Wait-Process -Id $ProcessId -Timeout 5 -ErrorAction Stop }
            catch { Stop-Process -Id $ProcessId -ErrorAction SilentlyContinue }
        }
        catch { Stop-Process -Id $ProcessId -ErrorAction SilentlyContinue }
    }
    else {
        Stop-Process -Id $ProcessId -ErrorAction SilentlyContinue
    }
    try { Wait-Process -Id $ProcessId -Timeout 5 -ErrorAction Stop }
    catch {
        Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
        Wait-Process -Id $ProcessId -Timeout 5 -ErrorAction SilentlyContinue
    }
    $script:Stopped += "$ProcessId ($Description)"
    return $true
}

foreach ($ProcessDirectory in Get-ChildItem /proc -Directory -ErrorAction SilentlyContinue) {
    if ($ProcessDirectory.Name -notmatch '^\d+$') { continue }
    $ProcessId = [int]$ProcessDirectory.Name
    try { $Arguments = (Get-Content "$($ProcessDirectory.FullName)/cmdline" -Raw -ErrorAction Stop) -split [char]0 }
    catch { continue }
    $CommandLine = $Arguments -join ' '
    $IsTrustedLlamaServer = $Arguments[0] -eq $TrustedLlamaServer -and
        $Arguments -contains '--model' -and $Arguments -contains '--host' -and
        $Arguments[([array]::IndexOf($Arguments, '--host') + 1)] -eq '127.0.0.1'
    if ($IsTrustedLlamaServer) {
        Stop-StaleProcess -ProcessId $ProcessId -Description 'private llama-server' | Out-Null
        continue
    }
    if ($CommandLine -notmatch [regex]::Escape("$Root/.venv")) { continue }
    if (-not ($WorkerModules | Where-Object { $Arguments -contains $_ })) { continue }
    $PortMatch = [regex]::Match($CommandLine, '--port\s+(\d+)')
    if (-not $PortMatch.Success) { continue }
    $Port = [int]$PortMatch.Groups[1].Value
    if ($Port -notin $WorkerPorts) { continue }
    Stop-StaleProcess -ProcessId $ProcessId -Description "Worker port $Port" `
        -ShutdownUrl "http://127.0.0.1:$Port/shutdown" | Out-Null
}

if (-not $Quiet) {
    if ($Stopped.Count) { Write-Host "Stopped stale ModelDeck workers: $($Stopped -join ', ')" }
    else { Write-Host 'No stale ModelDeck workers found.' }
}
