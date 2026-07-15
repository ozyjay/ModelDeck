[CmdletBinding()]
param(
    [string]$CacheRoot = '/mnt/work/models/huggingface/hub'
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$ModelId = 'google/gemma-4-E2B-it'
$Revision = '9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf'
$Snapshot = Join-Path $CacheRoot "models--google--gemma-4-E2B-it/snapshots/$Revision"
$Required = @(
    'config.json',
    'processor_config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'chat_template.jinja',
    'generation_config.json'
)
$Expected = @{
    'chat_template.jinja' = @{ size = 17336; blob = 'c19999a347da729cf62806a8ddb7eb8e315223b5' }
    'config.json' = @{ size = 4954; blob = '923b5e9405e7d319572b0c1b1a89291512262aa3' }
    'generation_config.json' = @{ size = 208; blob = 'e605bb4523b1462ea9d9a3810b9e3ecf7ab7b1f6' }
    'model.safetensors' = @{ size = 10246621918; blob = '2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550' }
    'processor_config.json' = @{ size = 1689; blob = '5465974d23e1eca2c46c2809b26c997946ce0d90' }
    'tokenizer_config.json' = @{ size = 2095; blob = '375b25dc8be85705251e41be1c25310d24932051' }
    'tokenizer.json' = @{ size = 32169626; blob = 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f' }
}

if (-not (Test-Path $Snapshot -PathType Container)) {
    throw "The pinned SceneChat snapshot is missing: $ModelId@$Revision"
}
$Missing = @($Required | Where-Object { -not (Test-Path (Join-Path $Snapshot $_) -PathType Leaf) })
if (-not (Get-ChildItem $Snapshot -Filter '*.safetensors' -File)) { $Missing += '*.safetensors' }
if ($Missing.Count) { throw "The pinned SceneChat snapshot is incomplete: $($Missing -join ', ')" }

$Config = Get-Content (Join-Path $Snapshot 'config.json') -Raw | ConvertFrom-Json
$Processor = Get-Content (Join-Path $Snapshot 'processor_config.json') -Raw | ConvertFrom-Json
if ($Config.architectures -notcontains 'Gemma4ForConditionalGeneration') {
    throw 'The pinned configuration does not declare Gemma4ForConditionalGeneration.'
}
if ($Processor.processor_class -ne 'Gemma4Processor') {
    throw 'The pinned processor configuration does not declare Gemma4Processor.'
}
if (-not (Test-Path '.venv-rocm72/bin/python')) {
    throw 'Run pwsh -NoProfile -File scripts/setup.ps1 first.'
}

$Runtime = & .venv-rocm72/bin/python -c "import json, torch, transformers, PIL; from transformers import AutoConfig, AutoProcessor; p=r'$Snapshot'; c=AutoConfig.from_pretrained(p,local_files_only=True,trust_remote_code=False); print(json.dumps({'torch':torch.__version__,'hip':torch.version.hip,'transformers':transformers.__version__,'pillow':PIL.__version__,'config_class':type(c).__name__,'processor_class':type(AutoProcessor.from_pretrained(p,local_files_only=True,trust_remote_code=False)).__name__}))"
if ($LASTEXITCODE -ne 0) { throw 'The pinned SceneChat processor and dependency fingerprint probe failed.' }
$Files = Get-ChildItem $Snapshot -File | ForEach-Object {
    $Item = Get-Item $_.FullName -Force
    $Physical = if ($Item.LinkType) {
        Get-Item ([System.IO.Path]::GetFullPath((Join-Path $Snapshot $Item.Target)))
    }
    else { $Item }
    [ordered]@{
        name = $_.Name
        size_bytes = $Physical.Length
        blob_identifier = if ($Item.LinkType) { Split-Path $Item.Target -Leaf } else { $null }
    }
}
foreach ($Name in $Expected.Keys) {
    $Actual = $Files | Where-Object name -eq $Name
    if (-not $Actual) { throw "The immutable SceneChat file is missing: $Name" }
    if ($Actual.size_bytes -ne $Expected[$Name].size -or
        $Actual.blob_identifier -ne $Expected[$Name].blob) {
        throw "The immutable SceneChat file does not match its approved blob and size: $Name"
    }
}

[ordered]@{
    model_id = $ModelId
    revision = $Revision
    snapshot = $Snapshot
    runtime = $Runtime | ConvertFrom-Json
    files = $Files
    read_only = $true
} | ConvertTo-Json -Depth 8
