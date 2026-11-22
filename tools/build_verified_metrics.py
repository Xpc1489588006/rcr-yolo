"""Build results/verified_metrics.json: single source of truth for paper numbers.

Every value is read directly from raw experiment records; nothing is typed by hand.

Sources
  - Standard Val mAP  : runs/rcr300/<run>/results.csv
      best epoch chosen by Ultralytics fitness = 0.1*mAP50 + 0.9*mAP50-95
      (the same rule that selects best.pt); last epoch also recorded.
  - Hard-v2 / COCOeval: runs/rcr300/eval_hard_results*.txt (one block per run)
  - Efficiency        : tools/model_stats.csv (RTX 4090, FP32, batch=1)

Statistics (three seeds s0=42, s1=123, s2=456)
  - mean and sample std (ddof=1)
  - paired t-test over seeds, df=2, exact two-sided p = 1 - |t|/sqrt(t^2+2)

Usage: python tools/build_verified_metrics.py
"""
import csv
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "rcr300"
OUT_JSON = ROOT / "results" / "verified_metrics.json"
OUT_MD = ROOT / "results" / "verified_metrics.md"

EVAL_FILES = [
    "eval_hard_results.txt",
    "eval_hard_results_extra.txt",
    "eval_hard_results (1).txt",
]
SEEDS = {"s0": 42, "s1": 123, "s2": 456}
MULTI_SEED_MODELS = ["baseline-yolo11n", "ab4-rcr-fb", "rcr-full"]
ABLATION_MODELS = [
    "baseline-yolo11n", "ab1-orb", "ab2-mrfe", "ab3-lcr", "ab3b-lcrbase",
    "ab5-orbmrfe", "orblcrb", "mrfelcrb", "ab4-rcr-fb", "rcr-full",
]
COMPARISON_MODELS = ["yolov8n", "yolov10n", "yolo26n"]
PAIRED_TESTS = [
    ("ab4-rcr-fb", "baseline-yolo11n"),
    ("rcr-full", "baseline-yolo11n"),
    ("rcr-full", "ab4-rcr-fb"),
]


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


# ---------------------------------------------------------------- standard val
def read_val(run_dir: Path) -> dict:
    path = run_dir / "results.csv"
    with path.open(newline="") as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    by_epoch = {}
    for line_no, r in enumerate(rows, start=2):  # header is line 1
        by_epoch[int(float(r["epoch"]))] = (line_no, r)  # keep last on duplicates

    def pack(epoch):
        line_no, r = by_epoch[epoch]
        m50 = float(r["metrics/mAP50(B)"])
        m5095 = float(r["metrics/mAP50-95(B)"])
        return {
            "epoch": epoch,
            "mAP50": m50,
            "mAP50-95": m5095,
            "precision": float(r["metrics/precision(B)"]),
            "recall": float(r["metrics/recall(B)"]),
            "fitness": round(0.1 * m50 + 0.9 * m5095, 6),
            "source": f"{rel(path)}:L{line_no}",
        }

    best_epoch = max(by_epoch, key=lambda e: pack(e)["fitness"])
    return {
        "n_epochs": len(by_epoch),
        "best": pack(best_epoch),
        "last": pack(max(by_epoch)),
    }


# ---------------------------------------------------------------- hard-v2
NUM = r"([0-9.]+)"
PATTERNS = {
    "hard": re.compile(rf"^AP50={NUM} AP75={NUM} AP50-95={NUM}"),
    "size": re.compile(
        rf"^AP50-small={NUM}\((\d+)\) AP50-medium={NUM}\((\d+)\) AP50-large={NUM}\((\d+)\)"),
    "crowd": re.compile(
        rf"^crowd-free={NUM}\((\d+)\) crowd-partial={NUM}\((\d+)\) crowd-heavy={NUM}\((\d+)\)"),
    "coco": re.compile(
        rf"^COCOeval AP={NUM} AP50={NUM} AP75={NUM} AP_S={NUM} AP_M={NUM} AP_L={NUM}"),
    "header": re.compile(r"^GT boxes=(\d+) images=(\d+) detections=(\d+) protocol=(\S+)"),
}


def read_hard() -> dict:
    out = {}
    for name in EVAL_FILES:
        path = RUNS / name
        current = None
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            m = re.match(r"^=+ (\S+) =+$", line)
            if m:
                current = m.group(1)
                if current in out:
                    raise ValueError(f"duplicate eval block {current} in {name}")
                out[current] = {"source": f"{rel(path)}:L{line_no}"}
                continue
            if current is None:
                continue
            blk = out[current]
            for key, pat in PATTERNS.items():
                mm = pat.match(line)
                if not mm:
                    continue
                g = mm.groups()
                if key == "header":
                    blk.update(gt_boxes=int(g[0]), images=int(g[1]),
                               detections=int(g[2]), protocol=g[3])
                elif key == "hard":
                    blk.update({"Hard AP50": float(g[0]), "Hard AP75": float(g[1]),
                                "Hard AP50-95": float(g[2])})
                elif key == "size":
                    blk.update({"Hard AP50-small": float(g[0]), "n_small": int(g[1]),
                                "Hard AP50-medium": float(g[2]), "n_medium": int(g[3]),
                                "Hard AP50-large": float(g[4]), "n_large": int(g[5])})
                elif key == "crowd":
                    blk.update({"crowd-free": float(g[0]), "n_crowd_free": int(g[1]),
                                "crowd-partial": float(g[2]), "n_crowd_partial": int(g[3]),
                                "crowd-heavy": float(g[4]), "n_crowd_heavy": int(g[5])})
                elif key == "coco":
                    blk.update({"COCO AP": float(g[0]), "COCO AP50": float(g[1]),
                                "COCO AP75": float(g[2]), "COCO AP_S": float(g[3]),
                                "COCO AP_M": float(g[4]), "COCO AP_L": float(g[5])})
    required = ["Hard AP50-95", "Hard AP50-medium", "crowd-heavy", "COCO AP_S"]
    for run, blk in out.items():
        missing = [k for k in required if k not in blk]
        if missing:
            raise ValueError(f"{run} block incomplete, missing {missing}")
    return out


# ---------------------------------------------------------------- efficiency
def read_eff() -> dict:
    out = {}
    path = ROOT / "tools" / "model_stats.csv"  # RTX 4090 run, ablation models
    with path.open(newline="") as f:
        for line_no, r in enumerate(csv.DictReader(f), start=2):
            out[r["name"]] = {
                "params_ckpt_M": float(r["params_ckpt_M"]),
                "params_deploy_M": float(r["params_deploy_M"]),
                "params_orb_M": float(r["params_orb_M"]),
                "GFLOPs": float(r["gflops"]),
                "latency_ms": float(r["latency_ms"]),
                "source": f"{rel(path)}:L{line_no}",
            }
    # comparison models: tools/model_stats_comparison.py (same recipe,
    # one2many head removed for end2end models; no latency measured)
    path = ROOT / "tools" / "model_stats_comparison.csv"
    with path.open(newline="") as f:
        for line_no, r in enumerate(csv.DictReader(f), start=2):
            out[r["name"]] = {
                "params_ckpt_M": float(r["params_ckpt_M"]),
                "params_deploy_M": float(r["params_deploy_M"]),
                "params_orb_M": 0.0,
                "GFLOPs": float(r["gflops"]),
                "latency_ms": float(r["latency_ms"]) if r["latency_ms"] else None,
                "end2end": r["end2end"] == "True",
                "source": f"{rel(path)}:L{line_no}",
            }
    return out


# ---------------------------------------------------------------- statistics
def paired_t(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    md = statistics.mean(d)
    sd = statistics.stdev(d)
    if sd == 0:
        t = math.inf if md != 0 else 0.0
    else:
        t = md / (sd / math.sqrt(n))
    if n != 3:
        raise ValueError("closed-form p only valid for df=2")
    p = 0.0 if math.isinf(t) else 1 - abs(t) / math.sqrt(t * t + 2)
    return {"mean_diff": md, "sd_diff": sd, "t": t, "df": n - 1, "p_two_sided": p}


def summarize(values):
    return {
        "per_seed": values,
        "mean": statistics.mean(values),
        "std_sample": statistics.stdev(values),
    }


def main():
    hard = read_hard()
    eff = read_eff()

    runs = {}
    for d in sorted(RUNS.iterdir()):
        if not (d / "results.csv").is_file():
            continue
        entry = {"val": read_val(d)}
        if d.name in hard:
            entry["hard_v2"] = hard[d.name]
        runs[d.name] = entry
    for name in hard:
        if name not in runs:
            raise ValueError(f"eval block {name} has no matching results.csv")

    # metrics used for multi-seed statistics
    def metric_vector(run_entry):
        v = {"Val mAP50": run_entry["val"]["best"]["mAP50"],
             "Val mAP50-95": run_entry["val"]["best"]["mAP50-95"]}
        h = run_entry.get("hard_v2", {})
        for k in ["Hard AP50", "Hard AP75", "Hard AP50-95", "Hard AP50-small",
                  "Hard AP50-medium", "Hard AP50-large", "crowd-free",
                  "crowd-partial", "crowd-heavy", "COCO AP_S", "COCO AP_M", "COCO AP_L"]:
            if k in h:
                v[k] = h[k]
        return v

    vectors = {name: metric_vector(e) for name, e in runs.items()}

    three_seed = {}
    for model in MULTI_SEED_MODELS:
        names = [f"{model}-{s}" for s in SEEDS]
        metrics = vectors[names[0]].keys()
        three_seed[model] = {
            m: summarize([vectors[n][m] for n in names]) for m in metrics
        }

    tests = {}
    for a, b in PAIRED_TESTS:
        key = f"{a} vs {b}"
        tests[key] = {}
        for m in three_seed[a]:
            tests[key][m] = paired_t(three_seed[a][m]["per_seed"],
                                     three_seed[b][m]["per_seed"])

    std_ratio = {}
    for m in three_seed["baseline-yolo11n"]:
        base = three_seed["baseline-yolo11n"][m]["std_sample"]
        std_ratio[m] = {
            other: (base / three_seed[other][m]["std_sample"]
                    if three_seed[other][m]["std_sample"] else None)
            for other in ["ab4-rcr-fb", "rcr-full"]
        }

    efficiency = {}
    for model in ABLATION_MODELS + COMPARISON_MODELS:
        if model in eff:
            efficiency[model] = eff[model]

    result = {
        "_meta": {
            "generator": "tools/build_verified_metrics.py",
            "val_checkpoint_rule": "best epoch by fitness = 0.1*mAP50 + 0.9*mAP50-95",
            "std": "sample std, ddof=1",
            "t_test": "paired over seeds, df=2, p = 1 - |t|/sqrt(t^2+2)",
            "seeds": SEEDS,
            "warning": "Val mAP (results.csv) and Hard-v2 AP (eval_hard) are different "
                       "protocols; never mix them in one column.",
            "missing": {
                "efficiency": [m for m in ABLATION_MODELS + COMPARISON_MODELS
                               if m not in eff],
                "latency": [m for m in ABLATION_MODELS + COMPARISON_MODELS
                            if m in eff and eff[m]["latency_ms"] is None],
                "multi_seed": [m for m in ABLATION_MODELS + COMPARISON_MODELS
                               if m not in MULTI_SEED_MODELS],
            },
        },
        "runs": runs,
        "three_seed": three_seed,
        "paired_t_tests": tests,
        "std_ratio_baseline_over_model": std_ratio,
        "efficiency": efficiency,
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render_md(result, vectors), encoding="utf-8")
    print(f"wrote {rel(OUT_JSON)} and {rel(OUT_MD)}")


# ---------------------------------------------------------------- markdown
def f4(x):
    return "—" if x is None else f"{x:.4f}"


def render_md(res, vectors):
    L = ["# Verified metrics (auto-generated, do not edit by hand)", "",
         f"Generated by `{res['_meta']['generator']}`. "
         "Val = results.csv best epoch; Hard/crowd/COCO = eval_hard v2. "
         "Never mix Val and Hard in one column.", ""]

    def table(title, models, cols):
        L.extend([f"## {title}", "",
                  "| model | " + " | ".join(cols) + " |",
                  "|---|" + "---:|" * len(cols)])
        for m in models:
            v = vectors.get(m, {})
            L.append(f"| {m} | " + " | ".join(f4(v.get(c)) for c in cols) + " |")
        L.append("")

    s0 = lambda ms: [f"{m}-s0" for m in ms]
    table("Ablation (s0)", s0(ABLATION_MODELS),
          ["Val mAP50", "Val mAP50-95", "Hard AP50-95", "Hard AP50-small",
           "Hard AP50-medium", "Hard AP50-large", "crowd-heavy", "COCO AP_S"])
    table("Modern YOLO comparison (s0)", s0(COMPARISON_MODELS + ["baseline-yolo11n",
                                                                "ab4-rcr-fb", "rcr-full"]),
          ["Val mAP50", "Val mAP50-95", "Hard AP50", "Hard AP75", "Hard AP50-95",
           "Hard AP50-small", "Hard AP50-medium", "Hard AP50-large", "crowd-heavy",
           "COCO AP_S", "COCO AP_M", "COCO AP_L"])

    L.extend(["## Three seeds (mean ± sample std)", "",
              "| metric | " + " | ".join(MULTI_SEED_MODELS) + " |",
              "|---|" + "---:|" * len(MULTI_SEED_MODELS)])
    for m in res["three_seed"]["baseline-yolo11n"]:
        cells = [f"{res['three_seed'][mm][m]['mean']:.4f} ± "
                 f"{res['three_seed'][mm][m]['std_sample']:.4f}" for mm in MULTI_SEED_MODELS]
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L.append("")

    L.extend(["## Paired t-tests (n=3, df=2, exact p)", "",
              "| comparison | metric | mean Δ | t | p |", "|---|---|---:|---:|---:|"])
    for comp, ms in res["paired_t_tests"].items():
        for m, t in ms.items():
            L.append(f"| {comp} | {m} | {t['mean_diff']:+.4f} | {t['t']:.3f} | "
                     f"{t['p_two_sided']:.4f} |")
    L.append("")

    L.extend(["## Std ratio (baseline std / model std)", "",
              "| metric | ab4-rcr-fb | rcr-full |", "|---|---:|---:|"])
    for m, r in res["std_ratio_baseline_over_model"].items():
        L.append(f"| {m} | " + " | ".join(
            "—" if r[k] is None else f"{r[k]:.2f}×" for k in ["ab4-rcr-fb", "rcr-full"]) + " |")
    L.append("")

    L.extend(["## Efficiency (RTX 4090, FP32, batch=1)", "",
              "| model | deploy params M | GFLOPs | latency ms |", "|---|---:|---:|---:|"])
    for m, e in res["efficiency"].items():
        lat = "not measured" if e["latency_ms"] is None else f"{e['latency_ms']:.2f}"
        L.append(f"| {m} | {e['params_deploy_M']:.3f} | {e['GFLOPs']:.2f} | {lat} |")
    L.append("")
    L.append("Comparison models (yolov8n/yolov10n/yolo26n): params/GFLOPs from "
             "tools/model_stats_comparison.py, same recipe as tools/model_stats.csv; "
             "end2end models have the training-only one2many head removed. "
             "Latency missing: " + (", ".join(res["_meta"]["missing"]["latency"]) or "none"))
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
