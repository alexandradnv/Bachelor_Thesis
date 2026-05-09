#!/usr/bin/env python3
"""Generate an MSE comparison graph across all models."""

import json
import os
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GENERATED_DIR = SCRIPT_DIR / "Generated models"
OUTPUT_PATH = SCRIPT_DIR / "graph.png"

# Collect MSE from all data files
olmo3_stage1 = []  # (step, mse)
olmo3_stage3 = []  # (step, mse)
olmo3_main_mse = None
other_models = []  # (label, mse)

for f in sorted(os.listdir(GENERATED_DIR)):
    if not f.endswith("_data.json") or f.startswith("ground_truth"):
        continue
    with open(GENERATED_DIR / f) as fh:
        d = json.load(fh)

    model = d["model"]
    mse = d.get("mse")
    if mse is None:
        continue

    # Skip stale data from broken runs (all-0.5 maps produce MSE=0.25 exactly)
    if abs(mse - 0.25) < 0.001 and "OLMo" in model:
        continue

    if "OLMo-3-1025-7B@stage1-step" in model:
        m = re.search(r"step(\d+)", model)
        if m:
            olmo3_stage1.append((int(m.group(1)), mse))
    elif "OLMo-3-1025-7B@stage3-step" in model:
        m = re.search(r"step(\d+)", model)
        if m:
            olmo3_stage3.append((int(m.group(1)), mse))
    elif "OLMo-3-1025-7B@main" in model:
        olmo3_main_mse = mse
    else:
        # Clean up model name for display
        label = model.split("/")[-1]
        other_models.append((label, mse))

olmo3_stage1.sort()
olmo3_stage3.sort()
other_models.sort(key=lambda x: x[1])

# --- Create figure with two subplots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [2, 1]})

# === Left panel: OLMo3 MSE over training steps ===
if olmo3_stage1:
    steps = [s for s, _ in olmo3_stage1]
    mses = [m for _, m in olmo3_stage1]
    ax1.plot(steps, mses, "o-", color="#2e86c1", linewidth=2, markersize=6,
             label="OLMo3-7B Stage 1 (pretraining)", zorder=5)

# Mark stage 3 checkpoints (offset x-axis beyond stage1 end)
if olmo3_stage3:
    stage1_end = olmo3_stage1[-1][0] if olmo3_stage1 else 1413814
    for step, mse in olmo3_stage3:
        x = stage1_end + step
        ax1.plot(x, mse, "s", color="#27ae60", markersize=8, zorder=5)
    # Connect last stage1 to first stage3
    ax1.plot([stage1_end, stage1_end + olmo3_stage3[0][0]],
             [olmo3_stage1[-1][1], olmo3_stage3[0][1]],
             "--", color="#27ae60", linewidth=1.5)
    stage3_steps = [stage1_end + s for s, _ in olmo3_stage3]
    stage3_mses = [m for _, m in olmo3_stage3]
    ax1.plot(stage3_steps, stage3_mses, "s-", color="#27ae60", linewidth=1.5,
             markersize=8, label="OLMo3-7B Stage 3 (fine-tuning)", zorder=5)

# Mark main checkpoint
if olmo3_main_mse is not None:
    main_x = (stage1_end + olmo3_stage3[-1][0] + 5000) if olmo3_stage3 else stage1_end + 20000
    ax1.plot(main_x, olmo3_main_mse, "*", color="#e74c3c", markersize=15,
             label=f"OLMo3-7B main (MSE={olmo3_main_mse:.3f})", zorder=6)

# Add horizontal reference lines for other models
colors_ref = plt.cm.Set2(np.linspace(0, 1, len(other_models)))
for i, (label, mse) in enumerate(other_models):
    ax1.axhline(y=mse, color=colors_ref[i], linestyle=":", linewidth=1.5, alpha=0.7,
                label=f"{label} (MSE={mse:.3f})")

ax1.set_xlabel("Training Step", fontsize=12)
ax1.set_ylabel("MSE vs Ground Truth", fontsize=12)
ax1.set_title("OLMo3-7B: MSE During Training vs Other Models", fontsize=13, fontweight="bold")
ax1.legend(fontsize=8, loc="upper right")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 0.75)
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k"))

# === Right panel: Bar chart of all models ===
all_models_bar = list(other_models)
if olmo3_main_mse is not None:
    all_models_bar.append(("OLMo3-7B (main)", olmo3_main_mse))
if olmo3_stage1:
    # Add best OLMo3 checkpoint
    best_step, best_mse = min(olmo3_stage1, key=lambda x: x[1])
    all_models_bar.append((f"OLMo3-7B (step {best_step//1000}k)", best_mse))

all_models_bar.sort(key=lambda x: x[1])

labels = [m for m, _ in all_models_bar]
mse_vals = [m for _, m in all_models_bar]
colors_bar = ["#27ae60" if "OLMo" in l else "#2e86c1" for l in labels]

bars = ax2.barh(labels, mse_vals, color=colors_bar, edgecolor="white", height=0.6)
for bar, val in zip(bars, mse_vals):
    ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
             va="center", fontsize=9)

ax2.set_xlabel("MSE vs Ground Truth", fontsize=12)
ax2.set_title("All Models Compared", fontsize=13, fontweight="bold")
ax2.set_xlim(0, max(mse_vals) * 1.2)
ax2.grid(True, axis="x", alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Graph saved to {OUTPUT_PATH}")
