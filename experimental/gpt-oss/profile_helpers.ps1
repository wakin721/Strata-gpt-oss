function Get-StrataProfileSettings {
    param([string]$Path)
    $settings = @{}
    if (-not $Path) { return $settings }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw 'The shared Strata script was not found.' }
    $Path = (Resolve-Path -LiteralPath $Path).ProviderPath
    $tokens = $null
    $parseErrors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$tokens, [ref]$parseErrors)
    if ($parseErrors.Count) { throw 'The shared Strata script has syntax errors.' }
    $assignments = $ast.FindAll({ param($node) $node -is [System.Management.Automation.Language.AssignmentStatementAst] }, $true)
    foreach ($assignment in $assignments) {
        if ($assignment.Left -isnot [System.Management.Automation.Language.VariableExpressionAst]) { continue }
        $name = $assignment.Left.VariablePath.UserPath
        if ($name -notin @('Access', 'ApiKey')) { continue }
        try {
            # SafeGetValue only accepts constant AST expressions. It never runs
            # the launch script, its commands or arbitrary PowerShell code.
            $expression = $assignment.Right
            if ($expression -is [System.Management.Automation.Language.PipelineAst] -and $expression.PipelineElements.Count -eq 1 -and
                $expression.PipelineElements[0] -is [System.Management.Automation.Language.CommandExpressionAst]) {
                $expression = $expression.PipelineElements[0].Expression
            }
            if ($expression -is [System.Management.Automation.Language.CommandExpressionAst]) { $expression = $expression.Expression }
            $value = $expression.SafeGetValue()
            if ($value -is [string]) {
                $settings[$name] = $value
                $settings.Remove("Unresolved$name")
            } else { $settings["Unresolved$name"] = $true }
        } catch { $settings["Unresolved$name"] = $true }
    }
    return $settings
}
