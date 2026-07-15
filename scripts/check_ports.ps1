$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$ServiceBusy = @()
foreach ($Port in @(3600, 8600)) {
    $Listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
    try { $Listener.Start() } catch { $ServiceBusy += $Port } finally { $Listener.Stop() }
}
if ($ServiceBusy.Count) {
    throw "ModelDeck service ports are occupied: $($ServiceBusy -join ', '). Run scripts/stop.ps1 first."
}

& (Join-Path $PSScriptRoot 'stop_stale_workers.ps1') -Quiet
$Busy = @()
foreach ($Port in @(3600, 8000, 8600, 8610, 8611, 8620, 8621, 8622, 8623, 8624)) {
    $Listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
    try { $Listener.Start() } catch { $Busy += $Port } finally { $Listener.Stop() }
}
if ($Busy.Count) { throw "Fixed ports are occupied: $($Busy -join ', ')" }
Write-Host 'ModelDeck fixed ports are available: 3600, 8000, 8600, 8610, 8611, 8620, 8621, 8622, 8623, 8624'
