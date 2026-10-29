param(
    [int]$Port = 8097,
    [int]$Context = 2048,
    [int]$ExpertCacheMiB = 4096,
    [int]$CpuExpertWorkers = 4,
    [switch]$GroupHotExperts,
    [string]$ApiKeyEnv
)
$ErrorActionPreference = 'Stop'
$gptApiArguments = @('-u', (Join-Path $PSScriptRoot 'api_server.py'), '--port', "$Port", '--context', "$Context",
    '--expert-cache-mib', "$ExpertCacheMiB", '--cpu-expert-workers', "$CpuExpertWorkers")
if ($GroupHotExperts) { $gptApiArguments += '--group-hot-experts' }
if ($ApiKeyEnv) { $gptApiArguments += @('--api-key-env', $ApiKeyEnv) }
& python @gptApiArguments
exit $LASTEXITCODE
