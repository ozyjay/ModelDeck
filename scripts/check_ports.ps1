$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
Import-Module (Join-Path $PSScriptRoot 'environment_helpers.psm1') -Force
Import-ModelDeckEnvironment -Path (Join-Path (Get-Location) '.env')
$Bindings = @(
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
)
$BridgeEnabled = $Env:MODELDECK_ENABLE_DOCKER_BRIDGE -and $Env:MODELDECK_ENABLE_DOCKER_BRIDGE.Trim().ToLowerInvariant() -in @('1', 'true', 'yes', 'on')
if ($BridgeEnabled) {
    $Bindings += [pscustomobject]@{ Name = 'gateway-docker-bridge'; Host = '172.17.0.1'; Port = $Bindings[0].Port }
}
$Bindings | ForEach-Object {
    $Binding = $_
    $Address = [System.Net.IPAddress]::None
    if (-not [System.Net.IPAddress]::TryParse($Binding.Host, [ref]$Address)) {
        throw "Invalid $($Binding.Name) bind address '$($Binding.Host)'. Use an IP address literal."
    }
    if ($Binding.Name -eq 'gateway' -and -not [System.Net.IPAddress]::IsLoopback($Address)) {
        throw "Unsafe primary gateway bind address '$($Binding.Host)'. Use a loopback address."
    }
    if ($Binding.Name -eq 'gateway-docker-bridge' -and $Address.ToString() -ne '172.17.0.1') {
        throw "Unsafe Docker bridge address '$($Binding.Host)'."
    }
    $Listener = [System.Net.Sockets.TcpListener]::new($Address, $Binding.Port)
    try {
        $Listener.Start()
    } catch {
        throw "ModelDeck $($Binding.Name) cannot bind $($Binding.Host):$($Binding.Port): $($_.Exception.Message)"
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
