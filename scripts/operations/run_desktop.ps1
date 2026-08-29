[CmdletBinding()]
param([Alias('OpenDay')][switch]$LockConfiguration)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '../..')

if (-not (Test-Path '.venv/bin/modeldeck')) { throw 'Run scripts/setup/setup.ps1 first.' }
if (-not (Get-Command python3 -ErrorAction SilentlyContinue)) { throw 'python3 with GTK4, libadwaita, and WebKitGTK support is required.' }

& python3 -c 'import gi; gi.require_version("Adw", "1"); gi.require_version("Gtk", "4.0"); gi.require_version("WebKit", "6.0")'
if ($LASTEXITCODE -ne 0) { throw 'python3 requires GTK4, libadwaita, and WebKitGTK 6.0 support to launch the ModelDeck desktop shell.' }

& (Join-Path $PSScriptRoot 'run.ps1') -LockConfiguration:$LockConfiguration

$Version = (Select-String -Path 'backend/modeldeck/__init__.py' -Pattern '^__version__ = "([^"]+)"').Matches[0].Groups[1].Value
if (-not $Version) { throw 'Could not determine the ModelDeck development build version.' }
$ExistingPythonPath = $Env:PYTHONPATH
$Env:PYTHONPATH = (Join-Path (Get-Location) 'backend')
if ($ExistingPythonPath) { $Env:PYTHONPATH += ":$ExistingPythonPath" }
$Env:MODELDECK_DESKTOP_DEVELOPMENT = '1'
$Env:MODELDECK_DESKTOP_BUILD_ID = 'development'

Write-Host "Opening ModelDeck Desktop ${Version} from this checkout."
Write-Host 'Services remain running after the window closes; stop them with scripts/operations/stop.ps1.'
& python3 -m modeldeck.desktop.app
if ($LASTEXITCODE -ne 0) { throw 'ModelDeck Desktop exited with an error.' }
