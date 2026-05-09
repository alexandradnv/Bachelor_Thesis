#!/usr/bin/env python3
"""Recompute MSE vs ground truth for every saved blind-earth map.

Scans the "Generated models" directory for `*_data.json` files (each holds the
P(Land) grid produced by one experiment run), rebuilds the land/water ground
truth for whichever resolution(s) it finds, and writes a consolidated CSV +
JSON ranking of all runs by MSE.

Output: MSEs/all_mses.{csv,json}
"""

import csv
import json
from pathlib import Path

import numpy as np
from global_land_mask import globe

SCRIPT_DIR = Path(__file__).resolve().parent
GENERATED_DIR = SCRIPT_DIR / "Generated models"
MSE_DIR = SCRIPT_DIR / "MSEs"


def build_ground_truth(resolution: float) -> np.ndarray:
    """Generate the same lat/lon grid the experiment uses, then mark land=1, water=0."""
    lats = np.arange(90, -90, -resolution)
    lons = np.arange(-180, 180, resolution)
    gt = np.zeros((len(lats), len(lons)), dtype=float)
    for r, lat in enumerate(lats):
        for c, lon in enumerate(lons):
            gt[r, c] = 1.0 if globe.is_land(float(lat), float(lon)) else 0.0
    return gt


def main():
    data_files = sorted(p for p in GENERATED_DIR.glob("*_data.json") if "ground_truth" not in p.name)
    print(f"Found {len(data_files)} data files to score")

    gt_cache: dict[float, np.ndarray] = {}
    rows = []

    for path in data_files:
        try:
            with path.open() as f:
                payload = json.load(f)
        except Exception as e:
            print(f"  [skip] {path.name}: load failed ({e})")
            continue

        resolution = float(payload.get("resolution_deg", 2.0))
        grid = np.asarray(payload.get("grid"), dtype=float)
        if grid.ndim != 2:
            print(f"  [skip] {path.name}: bad grid shape {grid.shape}")
            continue

        if resolution not in gt_cache:
            print(f"  Building ground truth for resolution {resolution}°...")
            gt_cache[resolution] = build_ground_truth(resolution)
        gt = gt_cache[resolution]

        if grid.shape != gt.shape:
            print(f"  [skip] {path.name}: grid {grid.shape} != gt {gt.shape}")
            continue

        mse = float(np.mean((grid - gt) ** 2))
        slug = path.name.removesuffix("_data.json")
        map_path = path.with_name(f"{slug}.png")

        rows.append({
            "slug": slug,
            "model": payload.get("model", slug),
            "mse": mse,
            "stored_mse": payload.get("mse", ""),
            "resolution_deg": resolution,
            "n_rows": int(grid.shape[0]),
            "n_cols": int(grid.shape[1]),
            "elapsed_sec": payload.get("elapsed_sec", ""),
            "map_path": str(map_path) if map_path.exists() else "",
            "data_path": str(path),
        })
        print(f"  {slug}: MSE={mse:.6f}")

    rows.sort(key=lambda r: r["mse"])

    MSE_DIR.mkdir(parents=True, exist_ok=True)
    json_path = MSE_DIR / "all_mses.json"
    csv_path = MSE_DIR / "all_mses.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"count": len(rows), "results": rows}, f, indent=2)

    fieldnames = ["slug", "model", "mse", "stored_mse", "resolution_deg",
                  "n_rows", "n_cols", "elapsed_sec", "map_path", "data_path"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print()
    print(f"Wrote {len(rows)} rows to:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print()
    print("Top 10 (lowest MSE):")
    for r in rows[:10]:
        print(f"  {r['mse']:.6f}  {r['slug']}")
    print()
    print("Bottom 5 (highest MSE):")
    for r in rows[-5:]:
        print(f"  {r['mse']:.6f}  {r['slug']}")


if __name__ == "__main__":
    main()
