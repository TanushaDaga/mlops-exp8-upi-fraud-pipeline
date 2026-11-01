# Prints the project structure as a tree (folders first), skipping caches.
param([string]$Root = ".", [int]$MaxDepth = 2)
$skip = '^(__pycache__|catboost_info|\.pytest_cache|mlruns|\.venv|\.git)$'

function Show-Tree([string]$Path, [string]$Prefix, [int]$Depth) {
    $items = Get-ChildItem -LiteralPath $Path -Force | Where-Object { $_.Name -notmatch $skip } |
        Sort-Object @{ Expression = { -not $_.PSIsContainer } }, Name
    for ($i = 0; $i -lt $items.Count; $i++) {
        $item = $items[$i]
        $last = ($i -eq $items.Count - 1)
        $branch = if ($last) { '\-- ' } else { '+-- ' }
        $name = if ($item.PSIsContainer) { "$($item.Name)/" } else { $item.Name }
        "$Prefix$branch$name"
        if ($item.PSIsContainer -and $Depth -lt $MaxDepth -and $item.FullName -notmatch 'artifacts[\\/]models$') {
            $childPrefix = if ($last) { "$Prefix    " } else { "$Prefix|   " }
            Show-Tree $item.FullName $childPrefix ($Depth + 1)
        }
    }
}
(Resolve-Path $Root).Path | Split-Path -Leaf | ForEach-Object { "$_/" }
Show-Tree (Resolve-Path $Root).Path '' 1
