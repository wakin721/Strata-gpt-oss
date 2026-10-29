param(
    [Parameter(Mandatory = $true)][string]$Prompt,
    [int]$MaxNewTokens = 64,
    [int]$Context = 256,
    [int]$ExpertCacheMiB = 4096,
    [int]$CpuExpertWorkers = 4,
    [switch]$Thinking,
    [switch]$GroupHotExperts,
    [string]$Output
)
$ErrorActionPreference = 'Stop'
$gptSourceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$gptModelRoot = Join-Path $gptSourceRoot '..\strata_data\models\gpt-oss-120b-MXFP4'
$gptArguments = @(
    '-u', (Join-Path $PSScriptRoot 'run_prompt.py'),
    '--model', (Join-Path $gptModelRoot 'gpt-oss-120b-MXFP4.gguf'),
    '--tokenizer', (Join-Path $gptModelRoot 'tokenizer'),
    '--engine', (Join-Path $gptSourceRoot 'build-gpt-oss-ninja\experimental\gpt-oss\strata-gpt-oss-cpu.exe'),
    '--expert-ptx', (Join-Path $gptSourceRoot 'build-gpt-oss\gpu_expert.ptx'),
    '--fast', '--stream', '--prompt', $Prompt,
    '--max-new-tokens', "$MaxNewTokens", '--context', "$Context",
    '--expert-cache-mib', "$ExpertCacheMiB", '--cpu-expert-workers', "$CpuExpertWorkers"
)
if ($Thinking) { $gptArguments += '--thinking' }
if ($GroupHotExperts) { $gptArguments += '--group-hot-experts' }
if ($Output) { $gptArguments += @('--output', $Output) }
& python @gptArguments
exit $LASTEXITCODE
