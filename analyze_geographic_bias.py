#!/usr/bin/env python3
"""
Post-hoc analysis of existing blind-earth runs — no model inference required.

Produces two thesis-ready figures per selected model:

  1. *Per-region MSE* — bar chart of MSE broken down by latitude band
     (Polar N, Mid-lat N, Tropics, Mid-lat S, Polar S) and by hemisphere
     (Western vs Eastern). Reveals geographic bias: are tropics worse?
     Is the Western hemisphere (English-language-heavy) better?

  2. *Calibration curve* — bins all pixels by predicted P(Land) and plots
     the actual land-rate inside each bin against the bin center. A
     perfectly calibrated model lies on the y = x diagonal. Also reports
     Brier score (mean squared error, identical to the run's `mse` field
     but kept here for completeness) and the ECE (expected calibration
     error).

Also writes a CSV summary across models so you can drop the table straight
into the thesis.

Usage:
    .venv/bin/python analyze_geographic_bias.py
    .venv/bin/python analyze_geographic_bias.py --models Qwen_Qwen2.5-3B-Instruct Qwen_Qwen2.5-3B-Instruct_memory3
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from global_land_mask import globe
except ImportError as exc:
    raise SystemExit(
        "global-land-mask is required: pip install global-land-mask"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Generated models"
ANALYSIS_DIR = SCRIPT_DIR / "analysis"

# Default model selection: 3B (best baseline), 3B-memory (this week's run),
# 7B (next size up), and 32B-AWQ (top-end) — gives a good cross-section.
DEFAULT_MODELS = [
    "Qwen_Qwen2.5-0.5B-Instruct",
    "Qwen_Qwen2.5-1.5B-Instruct",
    "Qwen_Qwen2.5-3B-Instruct",
    "Qwen_Qwen2.5-3B-Instruct_memory3",
    "Qwen_Qwen2.5-3B-Instruct_groundtruth3",
    "Qwen_Qwen2.5-7B-Instruct",
    "Qwen_Qwen2.5-7B-Instruct_memory3",
    "Qwen_Qwen2.5-14B-Instruct",
    "Qwen_Qwen2.5-32B-Instruct-AWQ",
]

# Latitude-band definitions (south-edge inclusive, north-edge exclusive).
LAT_BANDS = [
    ("Polar S",      -90.0, -66.5),
    ("Mid-lat S",    -66.5, -23.5),
    ("Tropics",      -23.5,  23.5),
    ("Mid-lat N",     23.5,  66.5),
    ("Polar N",       66.5,  90.0),
]


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _lat_lon_arrays(n_rows: int, n_cols: int, resolution: float):
    """Reconstruct the same lat/lon vectors used by generate_coordinates()."""
    lats = np.arange(90, -90, -resolution)
    lons = np.arange(-180, 180, resolution)
    assert lats.shape[0] == n_rows, (lats.shape, n_rows)
    assert lons.shape[0] == n_cols, (lons.shape, n_cols)
    return lats, lons


def _ground_truth(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """1.0 for land pixels, 0.0 for water, shape (len(lats), len(lons))."""
    gt = np.zeros((lats.size, lons.size), dtype=float)
    for r, lat in enumerate(lats):
        for c, lon in enumerate(lons):
            gt[r, c] = 1.0 if globe.is_land(float(lat), float(lon)) else 0.0
    return gt


def _load_run(json_path: Path):
    d = json.loads(json_path.read_text())
    grid = np.asarray(d["grid"], dtype=float)
    res = float(d["resolution_deg"])
    n_rows, n_cols = grid.shape
    lats, lons = _lat_lon_arrays(n_rows, n_cols, res)
    gt = _ground_truth(lats, lons)
    return {
        "model": d["model"],
        "grid": grid,
        "ground_truth": gt,
        "lats": lats,
        "lons": lons,
        "resolution": res,
        "overall_mse": float(d["mse"]),
    }


# ---------------------------------------------------------------------------
# Regional MSE
# ---------------------------------------------------------------------------

def regional_mse(run) -> Dict[str, float]:
    """MSE per latitude band + per hemisphere (W/E, N/S)."""
    grid, gt = run["grid"], run["ground_truth"]
    lats, lons = run["lats"], run["lons"]
    diff_sq = (grid - gt) ** 2

    results: Dict[str, float] = {"Global": float(diff_sq.mean())}

    for name, south, north in LAT_BANDS:
        mask = (lats >= south) & (lats < north)
        if mask.any():
            results[name] = float(diff_sq[mask, :].mean())

    # Hemispheres
    results["Northern (≥0°)"] = float(diff_sq[lats >= 0, :].mean())
    results["Southern (<0°)"] = float(diff_sq[lats < 0, :].mean())
    results["Western (lon<0)"] = float(diff_sq[:, lons < 0].mean())
    results["Eastern (lon≥0)"] = float(diff_sq[:, lons >= 0].mean())
    return results


def plot_regional_mse(per_model_regional: Dict[str, Dict[str, float]], out: Path):
    region_order = [
        "Polar N", "Mid-lat N", "Tropics", "Mid-lat S", "Polar S",
        "Northern (≥0°)", "Southern (<0°)",
        "Western (lon<0)", "Eastern (lon≥0)",
        "Global",
    ]
    models = list(per_model_regional.keys())
    n_models = len(models)
    n_regions = len(region_order)

    fig, ax = plt.subplots(figsize=(max(11, 1.0 * n_regions + 2), 5.5))
    width = 0.8 / max(1, n_models)
    x = np.arange(n_regions)

    cmap = plt.get_cmap("tab10")
    for i, model in enumerate(models):
        regional = per_model_regional[model]
        vals = [regional.get(r, np.nan) for r in region_order]
        ax.bar(x + i * width, vals, width=width, label=model, color=cmap(i % 10))

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(region_order, rotation=30, ha="right")
    ax.set_ylabel("MSE vs. ground-truth land mask")
    ax.set_title("Per-region MSE (lower = better geographic accuracy)")
    ax.axhline(0.25, color="grey", linestyle="--", linewidth=0.7,
               label="uniform-0.5 baseline (MSE=0.25)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibration(run, n_bins: int = 10):
    """Return (bin_centers, frac_land_per_bin, count_per_bin, ECE)."""
    p = run["grid"].ravel()
    y = run["ground_truth"].ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    bin_id = np.minimum(np.digitize(p, edges) - 1, n_bins - 1)
    bin_id = np.maximum(bin_id, 0)

    frac_land = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    pred_mean = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bin_id == b
        counts[b] = int(mask.sum())
        if counts[b] > 0:
            frac_land[b] = float(y[mask].mean())
            pred_mean[b] = float(p[mask].mean())
        else:
            frac_land[b] = np.nan
            pred_mean[b] = centers[b]

    total = counts.sum()
    valid = counts > 0
    ece = float(
        np.sum(counts[valid] / total * np.abs(frac_land[valid] - pred_mean[valid]))
    )
    return centers, frac_land, counts, ece


def plot_calibration(per_model_cal, out: Path):
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.5, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    cmap = plt.get_cmap("tab10")

    ax_top.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect calibration")
    for i, (model, cal) in enumerate(per_model_cal.items()):
        centers, frac, counts, ece = cal
        valid = counts > 0
        ax_top.plot(centers[valid], frac[valid], "o-",
                    color=cmap(i % 10), label=f"{model}  (ECE={ece:.3f})")

    ax_top.set_ylabel("Actual fraction of pixels that are land")
    ax_top.set_title("Calibration: predicted P(Land) vs. observed land rate")
    ax_top.grid(True, alpha=0.3)
    ax_top.set_axisbelow(True)
    ax_top.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1)

    # Bin populations underneath so the reader sees where the mass is.
    width = 1.0 / len(next(iter(per_model_cal.values()))[0]) * 0.9 / len(per_model_cal)
    for i, (model, cal) in enumerate(per_model_cal.items()):
        centers, _, counts, _ = cal
        frac_total = counts / max(1, counts.sum())
        ax_bot.bar(centers + (i - len(per_model_cal) / 2) * width, frac_total,
                   width=width, color=cmap(i % 10), label=model)
    ax_bot.set_xlabel("Predicted P(Land) (bin center)")
    ax_bot.set_ylabel("Pixel mass")
    ax_bot.grid(True, alpha=0.3, axis="y")
    ax_bot.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model slugs (without _data.json) to include in the analysis.",
    )
    parser.add_argument("--out", type=Path, default=ANALYSIS_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = {}
    for slug in args.models:
        p = DATA_DIR / f"{slug}_data.json"
        if not p.exists():
            print(f"  [skip] {slug} -> {p} not found")
            continue
        print(f"  [load] {slug}")
        runs[slug] = _load_run(p)

    if not runs:
        print("No runs loaded. Exiting.")
        return

    # Per-region MSE
    per_model_regional = {slug: regional_mse(run) for slug, run in runs.items()}
    plot_regional_mse(per_model_regional, args.out / "regional_mse")
    print(f"[wrote] {args.out / 'regional_mse.png'}")

    # Calibration
    per_model_cal = {slug: calibration(run, n_bins=10) for slug, run in runs.items()}
    plot_calibration(per_model_cal, args.out / "calibration")
    print(f"[wrote] {args.out / 'calibration.png'}")

    # CSV summary
    csv_path = args.out / "regional_mse_summary.csv"
    region_cols = [
        "Global", "Polar N", "Mid-lat N", "Tropics", "Mid-lat S", "Polar S",
        "Northern (≥0°)", "Southern (<0°)",
        "Western (lon<0)", "Eastern (lon≥0)",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model"] + region_cols + ["ECE"])
        for slug in runs:
            regional = per_model_regional[slug]
            _, _, _, ece = per_model_cal[slug]
            row = [slug] + [f"{regional.get(c, ''):.4f}" if c in regional else ""
                            for c in region_cols] + [f"{ece:.4f}"]
            w.writerow(row)
    print(f"[wrote] {csv_path}")

    # Console table for quick eyeball
    print()
    print(f"{'model':45s} | " + " | ".join(f"{c:14s}" for c in region_cols) + " | ECE")
    for slug in runs:
        regional = per_model_regional[slug]
        _, _, _, ece = per_model_cal[slug]
        row = " | ".join(f"{regional.get(c, float('nan')):14.4f}" for c in region_cols)
        print(f"{slug:45s} | {row} | {ece:.4f}")


if __name__ == "__main__":
    main()
