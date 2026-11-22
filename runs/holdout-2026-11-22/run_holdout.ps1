# Official holdout run, Sprint 01 (RULES "The two official runs"). Run once.
# Waits until the network is off, checks the freeze, runs the continuation check,
# then every strategy once, each as its own command, and writes everything to this folder.

$ErrorActionPreference = "Stop"
$Repo = "D:\syndicate"
$Out = Join-Path $Repo "runs\holdout-2026-11-22"
$Py = "C:\Users\user\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe"
$Tag = "186a4afc20ae06059aa9180238b9ad1d94d818a4"
$Strategies = @("angelo_closeloc", "arhaan_contrarian", "arnav_carry", "baseline_flow", "hongyu_zhu_strategy",
    "janice_gradual_catchup", "janice_leadlag", "janice_slow_momentum", "leaky_example", "tanmay_flow", "template")

$env:PYTHONUTF8 = "1"
Set-Location $Repo
New-Item -ItemType Directory -Force (Join-Path $Out "reports") | Out-Null
$Log = Join-Path $Out "run_log.txt"
function Write-Step($msg) { $line = "{0:yyyy-MM-dd HH:mm:ss}Z  {1}" -f (Get-Date).ToUniversalTime(), $msg; Write-Host $line; Add-Content -Path $Log -Value $line -Encoding ascii }

# Keep the machine awake while this window runs (no system setting is changed).
Add-Type -Namespace Oqs -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[Oqs.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null

function Test-Online {
    try { Invoke-WebRequest -Uri "https://api.github.com" -Method Head -TimeoutSec 5 -UseBasicParsing | Out-Null; return $true }
    catch { return ($null -ne $_.Exception.Response) }
}

Write-Host "Waiting for the network to go off. Switch Wi-Fi off now."
$off = 0
while ($off -lt 3) {
    if (Test-Online) { $off = 0; Write-Host ("{0:HH:mm:ss}  still online" -f (Get-Date)) } else { $off++ }
    Start-Sleep -Seconds 5
}
Write-Step "network off, starting"

function Invoke-Py([string[]]$PyArgs, [string]$StdOut) {
    $err = "$StdOut.stderr"
    $p = Start-Process -FilePath $Py -ArgumentList $PyArgs -NoNewWindow -Wait -PassThru -RedirectStandardOutput $StdOut -RedirectStandardError $err
    if ((Get-Item $err).Length -eq 0) { Remove-Item $err }
    return $p.ExitCode
}

$head = (git rev-parse HEAD).Trim()
$dirty = git status --porcelain --untracked-files=no
Write-Step "HEAD $head"
if ($head -ne $Tag -or $dirty) { Write-Step "STOP: the checkout is not the freeze tag or has changes"; Read-Host "Press Enter to close"; exit 1 }

$code = Invoke-Py @("-m", "src.freeze", "check") (Join-Path $Out "freeze_check.txt")
Write-Step "freeze check exit $code"
if ($code -ne 0) { Write-Step "STOP: freeze check failed"; Read-Host "Press Enter to close"; exit 1 }

Get-FileHash -Algorithm SHA256 data\train_1h.parquet, data\holdout_1h.parquet |
    ForEach-Object { "{0}  {1}" -f $_.Hash.ToLower(), (Resolve-Path -Relative $_.Path) } |
    Set-Content -Path (Join-Path $Out "data_sha256.txt") -Encoding ascii
Write-Step "data hashes written"

$code = Invoke-Py @("-m", "src.contrib.yousof", "--continuation", "data/train_1h.parquet", "data/holdout_1h.parquet", "--end", "2026-10-18T14:00") (Join-Path $Out "continuation.txt")
Write-Step "continuation check exit $code"

foreach ($s in $Strategies) {
    $t0 = Get-Date
    Write-Step "start $s"
    $code = Invoke-Py @("-m", "src.audit", "strategies/$s.py", "--data", "data/train_1h.parquet", "--holdout", "data/holdout_1h.parquet",
        "--n-random-cuts", "60", "--log", "runs/holdout-2026-11-22/log.jsonl") (Join-Path $Out "reports\$s.txt")
    Write-Step ([string]::Format([Globalization.CultureInfo]::InvariantCulture, "done  {0}  exit {1}  {2:F1} min", $s, $code, ((Get-Date) - $t0).TotalMinutes))
}

$reports = $Strategies | ForEach-Object { "runs/holdout-2026-11-22/reports/$_.txt" }
$code = Invoke-Py (@("-m", "src.contrib.yousof", "--data", "data/train_1h.parquet", "--holdout", "data/holdout_1h.parquet") + $reports) (Join-Path $Out "tables.md")
Write-Step "tables exit $code"
Write-Step "ALL DONE. Wi-Fi can go back on."
[Oqs.Power]::SetThreadExecutionState([uint32]2147483648) | Out-Null
Read-Host "Press Enter to close"
