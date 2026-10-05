param()
$ErrorActionPreference = "Continue"
$env:GIT_TERMINAL_PROMPT = "0"
$root = (Get-Location).Path
$src = Join-Path $root "d0\src"
New-Item -ItemType Directory -Force -Path $src | Out-Null

# name, owner/repo, pinned tag, expected peeled sha
$repos = @(
  @{ n="click";          r="pallets/click";                       tag="8.5.0";  sha="8b19813f2bfca99f1018a587a8cf54fc959f2e5d" },
  @{ n="more-itertools"; r="more-itertools/more-itertools";       tag="v11.1.0"; sha="64be96ceb2a6e836f76f069f4a96d2394d59fd0c" },
  @{ n="pluggy";         r="pytest-dev/pluggy";                   tag="1.6.0";  sha="fd08ab5f811a9b2fa9124ae8cbbd393221151e2c" },
  @{ n="boltons";        r="mahmoud/boltons";                     tag="26.2.0"; sha="4332b35a278d694f30c99881faa61cde695c7a96" },
  @{ n="attrs";          r="python-attrs/attrs";                  tag="26.1.0"; sha="7bfc49e9b22d5ba25b6e429524c3d49fee27cb36" },
  @{ n="dateutil";       r="dateutil/dateutil";                   tag="2.9.0";  sha="db9d018944c41ddc740015cf5f64717c2ba64a5c" },
  @{ n="packaging";      r="pypa/packaging";                      tag="26.3";   sha="929fd4b1410ac7ef61ef3f45b2f5d7e87711a9b5" },
  @{ n="marshmallow";    r="marshmallow-code/marshmallow";        tag="4.3.1";  sha="c7b559a1fa3aba57ca6dba0ab336841c5038a782" }
)

foreach ($p in $repos) {
  $dir = Join-Path $src $p.n
  Write-Output "===== $($p.n) ====="
  if (-not (Test-Path (Join-Path $dir ".git"))) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    git -C $dir init -q 2>&1 | Out-Null
    git -C $dir remote add origin "https://ghfast.top/https://github.com/$($p.r).git" 2>&1 | Out-Null
  }
  # depth-1 fetch of the exact tag ref
  git -C $dir -c http.sslBackend=openssl fetch --depth 1 --force origin "refs/tags/$($p.tag)" 2>&1 | Select-Object -Last 2
  if ($LASTEXITCODE -ne 0) {
    Write-Output "  TAGFETCH_FAIL exit=$LASTEXITCODE ; trying SHA fetch"
    git -C $dir -c http.sslBackend=openssl fetch --depth 1 --force origin $p.sha 2>&1 | Select-Object -Last 2
  }
  git -C $dir checkout -q --detach FETCH_HEAD 2>&1 | Select-Object -Last 2
  $head = (git -C $dir rev-parse HEAD).Trim()
  $tree = (git -C $dir rev-parse "HEAD^{tree}").Trim()
  $date = (git -C $dir log -1 --format=%cI).Trim()
  Write-Output "  head=$head tree=$tree date=$date expected=$($p.sha) match=$($head -eq $p.sha)"
  $files = (git -C $dir ls-tree -r --name-only HEAD | Measure-Object).Count
  $size = (Get-ChildItem $dir -Recurse -File -Exclude *.pack,*.idx | Where-Object { $_.FullName -notlike "*\.git\*" } | Measure-Object -Property Length -Sum).Sum
  Write-Output "  files=$files checkout_bytes=$size"
}
Write-Output "ALL DONE"
