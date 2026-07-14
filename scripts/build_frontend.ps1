[CmdletBinding()]
param([switch]$Check)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw 'Node.js is required to build the operator console.' }
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { throw 'npm is required to build the operator console.' }
if (-not (Test-Path 'frontend/node_modules')) { throw 'Run npm --prefix frontend ci before building the operator console.' }

$Committed = Join-Path (Get-Location) 'backend/modeldeck/api/static'
$Temporary = $null
try {
    if ($Check) {
        $Temporary = Join-Path ([System.IO.Path]::GetTempPath()) "modeldeck-frontend-$([guid]::NewGuid().ToString('N'))"
        $Env:MODELDECK_FRONTEND_OUT_DIR = $Temporary
    }
    & npm --prefix frontend run build
    if ($LASTEXITCODE -ne 0) { throw 'The operator console build failed.' }

    if ($Check) {
        $Expected = @(Get-ChildItem $Committed -File -Recurse | ForEach-Object {
            $Relative = [System.IO.Path]::GetRelativePath($Committed, $_.FullName)
            "$Relative`t$((Get-FileHash $_.FullName -Algorithm SHA256).Hash)"
        } | Sort-Object)
        $Actual = @(Get-ChildItem $Temporary -File -Recurse | ForEach-Object {
            $Relative = [System.IO.Path]::GetRelativePath($Temporary, $_.FullName)
            "$Relative`t$((Get-FileHash $_.FullName -Algorithm SHA256).Hash)"
        } | Sort-Object)
        $Difference = Compare-Object $Expected $Actual
        if ($Difference) {
            $Difference | Format-Table -AutoSize | Out-String | Write-Host
            throw 'Committed operator console assets are stale; run scripts/build_frontend.ps1.'
        }
        Write-Host 'Committed operator console assets match the frontend source.'
    }
}
finally {
    Remove-Item Env:MODELDECK_FRONTEND_OUT_DIR -ErrorAction SilentlyContinue
    if ($Temporary -and (Test-Path $Temporary)) { Remove-Item $Temporary -Recurse -Force }
}
