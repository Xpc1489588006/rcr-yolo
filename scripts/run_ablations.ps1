# RCR-YOLO ablation + baseline matrix (single-GPU serial queue).
#
# Usage (from the rcr_yolo directory):
#   powershell -ExecutionPolicy Bypass -File scripts\run_ablations.ps1 `
#       -Data datasets\coco_indoor\coco_indoor.yaml -Device 0
#
# Each run is independent; if a run crashes, re-launch this script with
# -SkipExisting to resume from the first missing run.

param(
    [string]$Data = "datasets\coco_indoor\coco_indoor.yaml",
    [string]$Device = "0",
    [int]$Epochs = 150,
    [int]$Batch = 32,
    [switch]$SkipExisting
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# ---- ablation matrix: name -> model yaml ------------------------------------
$runs = [ordered]@{
    "baseline-yolo11n"  = "yolo11n.yaml"        # ultralytics stock baseline
    "ab1-orb"           = "cfg\yolo11n-orb.yaml"
    "ab2-mrfe"          = "cfg\yolo11n-mrfe.yaml"
    "ab3-lcr"           = "cfg\yolo11n-lcr.yaml"
    "ab3b-lcrbase"      = "cfg\yolo11n-lcrbase.yaml"  # fallback LCR variant
    "ab4-rcr-fb"        = "cfg\yolo11n-rcrfb.yaml"    # full w/o ORB (fallback story)
    "rcr-full"          = "cfg\yolo11n-rcr.yaml"      # full RCR-YOLO
}

foreach ($name in $runs.Keys) {
    $done = Join-Path "runs\rcr\$name\weights" "best.pt"
    if ($SkipExisting -and (Test-Path $done)) {
        Write-Host "[skip] $name (weights exist)"
        continue
    }
    Write-Host "===== RUN $name -> $($runs[$name]) =====" -ForegroundColor Cyan
    python train.py --model $runs[$name] --data $Data --name $name `
        --epochs $Epochs --batch $Batch --device $Device --project runs\rcr
}

# ---- PK two-stage cross-domain transfer (needs target-domain data) ----------
# Uncomment once datasets\sunrgbd\sunrgbd.yaml is converted (see data\convert_robot_datasets.py)
# python train_pk.py --model cfg\yolo11n-rcr.yaml `
#     --stage1-data $Data --stage2-data datasets\sunrgbd\sunrgbd.yaml --device $Device

# ---- hard-object metrics on every finished run ------------------------------
Get-ChildItem runs\rcr -Directory | ForEach-Object {
    $w = Join-Path $_.FullName "weights\best.pt"
    if (Test-Path $w) {
        Write-Host "===== EVAL-HARD $($_.Name) =====" -ForegroundColor Yellow
        python eval_hard.py --weights $w --data $Data --device $Device
    }
}

Write-Host "All runs finished. Results under runs\rcr\<name>\results.csv" -ForegroundColor Green
