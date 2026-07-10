$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$Busy = @()
foreach ($Port in @(3600, 8600, 8610, 8611, 8620)) {
    $Listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
    try { $Listener.Start() } catch { $Busy += $Port } finally { $Listener.Stop() }
}
if ($Busy.Count) { throw "Fixed ports are occupied: $($Busy -join ', ')" }
Write-Host 'ModelDeck fixed ports are available: 3600, 8600, 8610, 8611, 8620'
