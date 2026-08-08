$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
Import-Module (Join-Path $PSScriptRoot 'environment_helpers.psm1') -Force
Import-ModelDeckEnvironment -Path (Join-Path (Get-Location) '.env')
@(
    [pscustomobject]@{
        Name = 'gateway'
        Host = if ($Env:MODELDECK_GATEWAY_HOST) { $Env:MODELDECK_GATEWAY_HOST } else { '127.0.0.1' }
        Port = if ($Env:MODELDECK_GATEWAY_PORT) { [int]$Env:MODELDECK_GATEWAY_PORT } else { 8600 }
    },
    [pscustomobject]@{
        Name = 'management'
        Host = if ($Env:MODELDECK_HOST) { $Env:MODELDECK_HOST } else { '127.0.0.1' }
        Port = if ($Env:MODELDECK_MANAGEMENT_PORT) { [int]$Env:MODELDECK_MANAGEMENT_PORT } else { 3600 }
    }
) | ForEach-Object {
    $Address = [System.Net.IPAddress]::None
    if (-not [System.Net.IPAddress]::TryParse($_.Host, [ref]$Address)) {
        throw "Invalid $($_.Name) bind address '$($_.Host)'. Use an IP address literal."
    }
    if ($_.Name -eq 'gateway' -and -not ($Address.IsLoopback -or $Address.ToString() -eq '172.17.0.1')) {
        throw "Unsafe gateway bind address '$($_.Host)'. Use loopback or Docker's default bridge 172.17.0.1."
    }
    $Listener = [System.Net.Sockets.TcpListener]::new($Address, $_.Port)
    try {
        $Listener.Start()
    } catch {
        throw "ModelDeck $($_.Name) cannot bind $($_.Host):$($_.Port): $($_.Exception.Message)"
    } finally {
        $Listener.Stop()
    }
}

& (Join-Path $PSScriptRoot 'stop_stale_workers.ps1') -Quiet
$Busy = @()
foreach ($Port in @(8000, 8610, 8611, 8620, 8621, 8622, 8623, 8624)) {
    $Listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
    try { $Listener.Start() } catch { $Busy += $Port } finally { $Listener.Stop() }
}
if ($Busy.Count) { throw "Fixed ports are occupied: $($Busy -join ', ')" }
Write-Host 'ModelDeck fixed Worker ports are available: 8000, 8610, 8611, 8620, 8621, 8622, 8623, 8624'
