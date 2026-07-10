$ErrorActionPreference = 'Continue'
Set-Location (Join-Path $PSScriptRoot '..')
foreach ($Name in @('gateway','management')) { $Path = "var/run/$Name.pid"; if (Test-Path $Path) { Stop-Process -Id (Get-Content $Path) -ErrorAction SilentlyContinue; Remove-Item $Path } }
Write-Host 'ModelDeck services stopped.'

