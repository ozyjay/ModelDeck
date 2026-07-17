Set-StrictMode -Version Latest

$script:AllowedEnvironmentNames = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::Ordinal
)
@(
    'HF_HOME',
    'HF_HUB_CACHE',
    'MODELDECK_ALLOW_DOWNLOADS',
    'MODELDECK_DATA_DIR',
    'MODELDECK_DIAGNOSTIC_CAPTURE',
    'MODELDECK_DIFFUSION_TIMEOUT_SECONDS',
    'MODELDECK_GATEWAY_PORT',
    'MODELDECK_HOST',
    'MODELDECK_LOG_DIR',
    'MODELDECK_MANAGEMENT_PORT',
    'MODELDECK_OPEN_DAY',
    'MODELDECK_ROCM72_PYTHON',
    'MODELDECK_ROCM72_Q4_PYTHON',
    'MODELDECK_SCENECHAT_API_KEY',
    'MODELDECK_SCENECHAT_TIMEOUT_SECONDS'
) | ForEach-Object { [void]$script:AllowedEnvironmentNames.Add($_) }

function Import-ModelDeckEnvironment {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return }

    $SeenNames = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::Ordinal
    )
    $LineNumber = 0
    foreach ($RawLine in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $LineNumber++
        $Line = $RawLine.Trim()
        if (-not $Line -or $Line.StartsWith('#')) { continue }

        $Separator = $Line.IndexOf('=')
        if ($Separator -lt 1) {
            throw "Invalid .env entry at line $LineNumber. Expected NAME=VALUE."
        }
        $Name = $Line.Substring(0, $Separator).Trim()
        if (-not $script:AllowedEnvironmentNames.Contains($Name)) {
            throw "Unsupported .env variable at line ${LineNumber}: $Name"
        }
        if (-not $SeenNames.Add($Name)) {
            throw "Duplicate .env variable at line ${LineNumber}: $Name"
        }

        $Value = $Line.Substring($Separator + 1).Trim()
        if ($Value.StartsWith('"') -or $Value.StartsWith("'")) {
            $Quote = $Value.Substring(0, 1)
            if ($Value.Length -lt 2 -or -not $Value.EndsWith($Quote)) {
                throw "Unterminated quoted .env value at line $LineNumber for $Name"
            }
            $Value = $Value.Substring(1, $Value.Length - 2)
        }

        if ($null -eq (Get-Item -LiteralPath "Env:$Name" -ErrorAction SilentlyContinue)) {
            [Environment]::SetEnvironmentVariable($Name, $Value, 'Process')
        }
    }
}

Export-ModuleMember -Function Import-ModelDeckEnvironment
