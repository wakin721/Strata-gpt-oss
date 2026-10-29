$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'profile_helpers.ps1')
$gptTestRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\build-gpt-oss-profile-test-temp'))
$gptCase = [IO.Path]::GetFullPath((Join-Path $gptTestRoot ([guid]::NewGuid().ToString('N'))))
New-Item -ItemType Directory -Path $gptCase -Force | Out-Null
try {
    $gptMarker = Join-Path $gptCase 'executed.txt'
    $gptProfile = Join-Path $gptCase 'shared.ps1'
    @'
$Access = 'lan'
$ApiKey = 'profile-test-first,profile-test-second'
throw 'This launch script must never execute.'
'@ | Set-Content -LiteralPath $gptProfile -Encoding utf8
    $gptSettings = Get-StrataProfileSettings -Path $gptProfile
    if ($gptSettings['Access'] -ne 'lan' -or ($gptSettings['ApiKey'] -split ',').Count -ne 2) { throw 'Literal settings were not read.' }
    if ($gptSettings['UnresolvedApiKey']) { throw 'A literal key was rejected.' }
    $gptDynamic = '$Access = ''lan''; $ApiKey = $(Set-Content -LiteralPath ''' + $gptMarker + ''' -Value ''bad''; ''computed'')'
    $gptDynamic | Set-Content -LiteralPath $gptProfile -Encoding utf8
    $gptSettings = Get-StrataProfileSettings -Path $gptProfile
    if (-not $gptSettings['UnresolvedApiKey'] -or (Test-Path -LiteralPath $gptMarker)) { throw 'Computed key was evaluated or not flagged.' }
    Write-Output 'Shared profile literals read; scripts and computed keys never executed.'
} finally {
    if (-not $gptCase.StartsWith($gptTestRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Test cleanup path escaped its root.' }
    Remove-Item -LiteralPath $gptCase -Recurse -Force
}
