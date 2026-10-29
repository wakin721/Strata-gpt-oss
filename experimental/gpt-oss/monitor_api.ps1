param(
    [int]$Port = 8097,
    [string]$SharedProfile,
    [string]$ApiKeyEnv,
    [switch]$Once,
    [switch]$Json
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'profile_helpers.ps1')
$gptSourceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$gptWorkspace = Split-Path -Parent $gptSourceRoot
$gptPython = Join-Path $gptWorkspace 'strata\python\python.exe'
if (-not (Test-Path -LiteralPath $gptPython)) { $gptPython = (Get-Command python).Source }
$gptStrata = Join-Path $gptWorkspace 'strata\strata.py'
if (-not (Test-Path -LiteralPath $gptStrata)) { $gptStrata = Join-Path $gptSourceRoot 'strata.py' }
$gptKeys = if ($ApiKeyEnv) { [Environment]::GetEnvironmentVariable($ApiKeyEnv) } else { $null }
if (-not $gptKeys -and $SharedProfile) {
    $gptShared = Get-StrataProfileSettings -Path $SharedProfile
    if ($gptShared['UnresolvedApiKey']) { throw 'Shared ApiKey must be a literal; use -ApiKeyEnv for a computed key.' }
    $gptKeys = $gptShared['ApiKey']
}
$gptPreviousMonitorKey = [Environment]::GetEnvironmentVariable('STRATA_GPT_OSS_MONITOR_KEY')
$gptMonitorArguments = @($gptStrata, 'monitor', '--url', "http://127.0.0.1:$Port")
if ($gptKeys) {
    $env:STRATA_GPT_OSS_MONITOR_KEY = ($gptKeys -split ',' | Where-Object { $_.Trim() } | Select-Object -First 1).Trim()
    $gptMonitorArguments += @('--credential-env', 'STRATA_GPT_OSS_MONITOR_KEY')
}
if ($Once -or $Json) { $gptMonitorArguments += '--once' }
if ($Json) { $gptMonitorArguments += '--json' }
try {
    & $gptPython @gptMonitorArguments
    $gptMonitorExitCode = $LASTEXITCODE
} finally {
    [Environment]::SetEnvironmentVariable('STRATA_GPT_OSS_MONITOR_KEY', $gptPreviousMonitorKey)
}
exit $gptMonitorExitCode
