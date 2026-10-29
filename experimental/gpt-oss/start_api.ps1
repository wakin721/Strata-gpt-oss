param(
    [int]$Port = 8097,
    [string]$Context = '2k',
    [ValidateSet('local','lan')][string]$Access = 'local',
    [int]$ExpertCacheMiB = 4096,
    [int]$CpuExpertWorkers = 4,
    [int]$RequestTimeout = 900,
    [switch]$GroupHotExperts,
    [switch]$PersistentEngine,
    [string]$ApiKeyEnv,
    [string]$SharedProfile
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'profile_helpers.ps1')
$gptPreviousProfileKeys = [Environment]::GetEnvironmentVariable('STRATA_GPT_OSS_PROFILE_KEYS')
if ($SharedProfile) {
    $gptShared = Get-StrataProfileSettings -Path $SharedProfile
    if (-not $PSBoundParameters.ContainsKey('Access') -and $gptShared['UnresolvedAccess']) { throw 'Shared Access must be a literal; use -Access explicitly.' }
    if (-not $ApiKeyEnv -and $gptShared['UnresolvedApiKey']) { throw 'Shared ApiKey must be a literal; use -ApiKeyEnv for a computed key.' }
    if (-not $PSBoundParameters.ContainsKey('Access') -and $gptShared['Access']) { $Access = $gptShared['Access'] }
    if (-not $ApiKeyEnv -and $gptShared['ApiKey']) {
        $env:STRATA_GPT_OSS_PROFILE_KEYS = $gptShared['ApiKey']
        $ApiKeyEnv = 'STRATA_GPT_OSS_PROFILE_KEYS'
    }
}
$gptApiArguments = @('-u', (Join-Path $PSScriptRoot 'api_server.py'), '--port', "$Port", '--context', "$Context", '--access', $Access,
    '--expert-cache-mib', "$ExpertCacheMiB", '--cpu-expert-workers', "$CpuExpertWorkers", '--request-timeout', "$RequestTimeout")
if ($GroupHotExperts) { $gptApiArguments += '--group-hot-experts' }
if ($PersistentEngine) { $gptApiArguments += '--persistent-engine' }
if ($ApiKeyEnv) { $gptApiArguments += @('--api-key-env', $ApiKeyEnv) }
try {
    & python @gptApiArguments
    $gptApiExitCode = $LASTEXITCODE
} finally {
    [Environment]::SetEnvironmentVariable('STRATA_GPT_OSS_PROFILE_KEYS', $gptPreviousProfileKeys)
}
exit $gptApiExitCode
