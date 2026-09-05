function Assert-ModelDeckServicesStopped {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$RepositoryRoot)

    $ProcessIds = @{}
    foreach ($PidFile in @(
        'var/run/management.pid',
        'var/run/gateway.pid',
        'var/run/gateway-docker-bridge.pid',
        'var/run/gateway-loopback.pid'
    )) {
        $Path = Join-Path $RepositoryRoot $PidFile
        if (-not (Test-Path -LiteralPath $Path)) { continue }
        $RecordedId = 0
        if ([int]::TryParse((Get-Content -LiteralPath $Path -Raw).Trim(), [ref]$RecordedId) -and
            (Get-Process -Id $RecordedId -ErrorAction SilentlyContinue)) {
            $ProcessIds[$RecordedId] = $true
        }
    }

    if (Test-Path '/proc') {
        $ExpectedPython = @(
            (Join-Path $RepositoryRoot '.venv/bin/python'),
            (Join-Path $RepositoryRoot '.venv/Scripts/python.exe')
        ) | Where-Object { Test-Path -LiteralPath $_ } | ForEach-Object { (Resolve-Path -LiteralPath $_).Path }
        foreach ($ProcessDirectory in Get-ChildItem /proc -Directory -ErrorAction SilentlyContinue) {
            if ($ProcessDirectory.Name -notmatch '^\d+$') { continue }
            try {
                $CommandLine = (Get-Content "$($ProcessDirectory.FullName)/cmdline" -Raw -ErrorAction Stop) -split [char]0
            }
            catch { continue }
            if ($CommandLine.Count -lt 3 -or $ExpectedPython -notcontains $CommandLine[0]) { continue }
            $IsManagedModule = @($CommandLine | Where-Object {
                $_ -in @('modeldeck', 'modeldeck.gateway.app', 'modeldeck.gateway.docker_bridge') -or
                $_ -like 'modeldeck.workers.*'
            }).Count -gt 0
            $IsService = ($CommandLine -contains '-m') -and $IsManagedModule
            if ($IsService) { $ProcessIds[[int]$ProcessDirectory.Name] = $true }
        }
    }

    if ($ProcessIds.Count -gt 0) {
        throw "ModelDeck is active (process IDs: $($ProcessIds.Keys -join ', ')). Stop it with scripts/operations/stop.ps1 before importing or exporting state."
    }
}

function Resolve-CheckoutStateDirectory {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$RepositoryRoot,
        [Parameter(Mandatory)][string]$DataDirectory
    )

    $DataPath = [System.IO.Path]::GetFullPath([System.IO.Path]::Combine($RepositoryRoot, $DataDirectory))
    $Prefix = $RepositoryRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
    if ($DataPath -eq $RepositoryRoot -or -not $DataPath.StartsWith($Prefix, [System.StringComparison]::Ordinal)) {
        throw 'The ModelDeck data directory must be a specific directory below the repository.'
    }
    return $DataPath
}

function Assert-ModelDeckConfigurationMutable {
    [CmdletBinding()]
    param()

    $LockValue = if ($null -ne $Env:MODELDECK_CONFIGURATION_LOCKED) {
        $Env:MODELDECK_CONFIGURATION_LOCKED
    }
    else {
        $Env:MODELDECK_OPEN_DAY
    }
    if ($LockValue -and $LockValue.Trim().ToLowerInvariant() -in @('1', 'true', 'yes', 'on')) {
        throw 'ModelDeck configuration is locked. State import and export are disabled.'
    }
}

Export-ModuleMember -Function Assert-ModelDeckServicesStopped, Resolve-CheckoutStateDirectory, Assert-ModelDeckConfigurationMutable
