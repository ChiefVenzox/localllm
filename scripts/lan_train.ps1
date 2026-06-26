param(
    [Parameter(Mandatory = $true)]
    [int]$NodeRank,

    [Parameter(Mandatory = $true)]
    [string]$MasterAddr,

    [int]$Nodes = 3,
    [int]$Processes = 1,
    [int]$LocalRank = 0,
    [int]$MasterPort = 29500,
    [string]$Preset = "small-100m",
    [string]$Data = "data/bin",
    [string]$Device = "auto",
    [int]$MaxSteps = 0,
    [int]$LrDecaySteps = 0,
    [int]$BatchSize = 0,
    [int]$GradAccum = 0,
    [string]$OutDir = "",
    [string]$Resume = "",
    [switch]$ResetBest,
    [switch]$Compile
)

$ErrorActionPreference = "Stop"
if (-not $env:USE_LIBUV) {
    $env:USE_LIBUV = "0"
}

$worldSize = $Nodes * $Processes
$rank = ($NodeRank * $Processes) + $LocalRank

$env:MASTER_ADDR = $MasterAddr
$env:MASTER_PORT = "$MasterPort"
$env:WORLD_SIZE = "$worldSize"
$env:RANK = "$rank"
$env:LOCAL_RANK = "$LocalRank"

if ($Processes -gt 1) {
    Write-Host "[lan_train] Processes > 1 icin her local rank'i ayri terminalde baslat: -LocalRank 0, -LocalRank 1, ..."
}

$runArgs = @(
    "train.py",
    "--preset", $Preset,
    "--data", $Data,
    "--device", $Device,
    "--dist-backend", "gloo"
)

if ($MaxSteps -gt 0) {
    $runArgs += @("--max-steps", "$MaxSteps")
}
if ($LrDecaySteps -gt 0) {
    $runArgs += @("--lr-decay-steps", "$LrDecaySteps")
}
if ($BatchSize -gt 0) {
    $runArgs += @("--batch-size", "$BatchSize")
}
if ($GradAccum -gt 0) {
    $runArgs += @("--grad-accum", "$GradAccum")
}
if ($OutDir) {
    $runArgs += @("--out", $OutDir)
}
if ($Resume) {
    $runArgs += @("--resume", $Resume)
}
if ($ResetBest) {
    $runArgs += "--reset-best"
}
if ($Compile) {
    $runArgs += "--compile"
}

python @runArgs
