$ErrorActionPreference = 'Continue'
Set-Location (Join-Path $PSScriptRoot '..')
foreach ($Name in @('gateway', 'management')) {
    $Path = "var/run/$Name.pid"
    if (-not (Test-Path $Path)) { continue }
    $ProcessId = [int](Get-Content $Path)
    $Process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($Process) {
        Stop-Process -Id $ProcessId -ErrorAction SilentlyContinue
        try { Wait-Process -Id $ProcessId -Timeout 10 -ErrorAction Stop }
        catch {
            Write-Warning "$Name did not stop gracefully; forcing process $ProcessId to exit."
            Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
            Wait-Process -Id $ProcessId -Timeout 5 -ErrorAction SilentlyContinue
        }
    }
    Remove-Item $Path -ErrorAction SilentlyContinue
}
Write-Host 'ModelDeck services stopped.'
