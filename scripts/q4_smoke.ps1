[CmdletBinding()]
param(
    [ValidateSet("text-diffusion")]
    [string]$Model = "text-diffusion"
)

$ErrorActionPreference = "Stop"

$baseUri = "http://127.0.0.1:8600"

$body = @{
    model = $Model
    prompt = "Explain why the sky appears blue in three concise sentences."
    max_length = 256
    denoising_steps = 48
    block_length = 256
    temperature = 0.8
    seed = 11
    stream_intermediate_frames = $false
} | ConvertTo-Json -Depth 10

$timer = [System.Diagnostics.Stopwatch]::StartNew()

$queued = Invoke-RestMethod `
    -Uri "$baseUri/v1/diffuse" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body

$jobId = $queued.job_id

if (-not $jobId) {
    throw "Gateway did not return a job_id."
}

Write-Host "Job: $jobId"

do {
    Start-Sleep -Milliseconds 1000

    $job = Invoke-RestMethod `
        -Uri "$baseUri/v1/jobs/$jobId" `
        -Method Get

    Write-Progress `
        -Activity "DiffusionGemma generation" `
        -Status "State: $($job.state); frames: $($job.frame_count)"

    if ($timer.Elapsed.TotalMinutes -gt 10) {
        throw "Generation timed out after 10 minutes."
    }
}
while ($job.state -notin @("complete", "failed", "cancelled"))

$timer.Stop()
Write-Progress -Activity "DiffusionGemma generation" -Completed

$job | ConvertTo-Json -Depth 20
"Elapsed: $([math]::Round($timer.Elapsed.TotalSeconds, 2)) seconds"

if ($job.state -ne "complete") {
    throw "Generation ended with state: $($job.state)"
}

$terminal = $job.frames[-1]
if ($terminal.step -gt $terminal.total_steps) {
    throw "Terminal frame step exceeds total steps."
}
if ($terminal.finish_reason -eq "length") {
    throw "Generation exhausted its response length before producing a complete answer."
}
if ($job.text -match '^\s*(thought\b|<\|channel>thought\b)') {
    throw "Generation leaked a private reasoning channel."
}
