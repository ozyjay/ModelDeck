Set-StrictMode -Version Latest

$script:BoothBrowserCandidates = @(
    'chromium',
    'chromium-browser',
    'google-chrome',
    'google-chrome-stable',
    'microsoft-edge',
    'microsoft-edge-stable'
)

function Resolve-BoothBrowser {
    [CmdletBinding()]
    param([string]$Browser)

    $RequestedBrowser = if ($Browser) { $Browser } else { $Env:BOOTH_BROWSER }
    if ($RequestedBrowser) {
        $Command = Get-Command $RequestedBrowser -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($Command) { return $Command.Source }
        if (Test-Path -LiteralPath $RequestedBrowser -PathType Leaf) {
            return (Resolve-Path -LiteralPath $RequestedBrowser).Path
        }
        throw "Configured booth browser was not found: $RequestedBrowser"
    }

    foreach ($Candidate in $script:BoothBrowserCandidates) {
        $Command = Get-Command $Candidate -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($Command) { return $Command.Source }
    }

    throw 'No supported Chromium browser was found. Install Chromium, Chrome, or Edge, or set BOOTH_BROWSER to its executable path.'
}

function Get-BoothBrowserArguments {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Url,
        [Parameter(Mandatory)][string]$ProfileDirectory,
        [switch]$Windowed
    )

    $Arguments = @(
        "--user-data-dir=$ProfileDirectory",
        '--no-first-run',
        '--disable-background-networking',
        '--disable-session-crashed-bubble',
        '--disable-infobars'
    )
    if ($Windowed) {
        $Arguments += "--app=$Url"
    }
    else {
        $Arguments += '--kiosk'
        $Arguments += $Url
    }
    return $Arguments
}

Export-ModuleMember -Function Resolve-BoothBrowser, Get-BoothBrowserArguments
