$ErrorActionPreference = "Continue"
$env:GIT_TERMINAL_PROMPT = "0"
$targets = @(
  @{ n="click";          b="main" },
  @{ n="more-itertools"; b="master" },
  @{ n="pluggy";         b="main" },
  @{ n="boltons";        b="master" }
)
foreach ($t in $targets) {
  $d = ".\d0\src\$($t.n)"
  Write-Output "===== $($t.n) deepen ====="
  git -C $d -c http.sslBackend=openssl fetch --deepen=2000 origin $t.b 2>&1 | Select-Object -Last 2
  $rc = $LASTEXITCODE
  $n = (git -C $d log --oneline "origin/$($t.b)" 2>$null | Measure-Object).Count
  $oldest = (git -C $d log --format=%cI "origin/$($t.b)" 2>$null | Select-Object -Last 1)
  Write-Output "  exit=$rc commits=$n oldest=$oldest"
}
Write-Output "DEEPEN DONE"
