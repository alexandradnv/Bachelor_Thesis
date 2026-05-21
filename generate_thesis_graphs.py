#!/usr/bin/env python3
"""Generate thesis-quality MSE comparison graphs for the water/land experiment.

Produces five figures (PNG + PDF):
  1.  normal_experiment            - all models tested with default settings
  1b. model_size_experiment        - MSE vs. model parameter count (scatter)
  1c. model_size_experiment_bars   - same as 1b but as a sorted bar chart
  2.  anchor_experiment            - Qwen2.5-32B-Instruct-AWQ vs 5/10/15/20 anchors
  3.  language_experiment          - Qwen + BgGPT models in different languages
"""

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "MSEs" / "all_mses.csv"
OUT_DIR = SCRIPT_DIR / "MSEs" / "thesis_graphs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- thesis-friendly defaults ----------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

PRIMARY = "#2E5A87"   # deep blue
ACCENT = "#C0392B"    # red - highlight (worst / target)
HIGHLIGHT = "#27AE60" # green - highlight (best)
NEUTRAL = "#7f8c8d"

# ---------- load CSV ----------
rows = []
with open(CSV_PATH, newline="") as fh:
    reader = csv.DictReader(fh)
    for r in reader:
        try:
            r["mse"] = float(r["mse"])
        except (TypeError, ValueError):
            continue
        rows.append(r)


def short_name(model: str) -> str:
    """Shorten 'org/Model-Name' for display."""
    name = model.split("/")[-1]
    # strip quantization suffixes for readability
    name = name.replace("-GPTQ-W4A16", " (Q4)")
    name = name.replace("-AWQ", " (AWQ)")
    return name


# ============================================================
#  1. NORMAL EXPERIMENT
# ============================================================
# Keep only rows whose slug equals the slugified model name (no suffix).
# Drop training-checkpoint runs and stale 0.25 (broken) MSEs.
normal = []
for r in rows:
    slug = r["slug"]
    model = r["model"]
    if "@" in model:                      # training checkpoints
        continue
    if abs(r["mse"] - 0.25) < 1e-3:        # broken/all-0.5 maps
        continue
    expected_slug = model.replace("/", "_")
    if slug != expected_slug:              # has a suffix (anchor/lang/from)
        continue
    normal.append((short_name(model), r["mse"]))

normal.sort(key=lambda x: x[1])
labels = [n for n, _ in normal]
values = [m for _, m in normal]

fig, ax = plt.subplots(figsize=(10, max(5, 0.32 * len(labels) + 1.5)))

colors = [NEUTRAL] * len(values)
colors[0] = HIGHLIGHT       # best
colors[-1] = ACCENT         # worst

bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.72)

for bar, v in zip(bars, values):
    ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}", va="center", ha="left", fontsize=9, color="#333333")

ax.invert_yaxis()  # best at top
ax.set_xlabel("MSE vs. ground-truth water/land map")
ax.set_title("Water/Land Reconstruction Error per Model")
ax.set_xlim(0, max(values) * 1.12)
ax.grid(True, axis="x", alpha=0.45)
ax.set_axisbelow(True)

# subtle reference line at MSE = 0.25 (uniform-50% map)
ax.axvline(0.25, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
ax.text(0.25, -0.6, "uniform 0.5 baseline (MSE = 0.25)",
        fontsize=8, color="#666666", ha="center", va="top")

fig.tight_layout()
out = OUT_DIR / "normal_experiment"
fig.savefig(out.with_suffix(".png"))
fig.savefig(out.with_suffix(".pdf"))
plt.close(fig)
print(f"[1/5] {len(labels)} models -> {out}.png/.pdf")


# ============================================================
#  1b. MODEL-SIZE EXPERIMENT  (MSE vs. parameter count)
# ============================================================
# Manual size overrides (in billions of parameters) for models whose
# names don't contain an explicit "<N>B" token, or where the on-disk
# parameter count differs from the marketing name (e.g. MoE totals).
SIZE_OVERRIDES_B = {
    "microsoft/Phi-3-medium-4k-instruct": 14.0,
    "microsoft/phi-4": 14.0,
    "mistralai/Mistral-Nemo-Instruct-2407": 12.0,
    "casperhansen/mixtral-instruct-awq": 46.7,  # Mixtral-8x7B total params
}
SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)[Bb](?![a-zA-Z])")


def parse_param_count_b(model: str) -> float | None:
    """Return parameter count in billions, or None if unknown."""
    if model in SIZE_OVERRIDES_B:
        return SIZE_OVERRIDES_B[model]
    matches = SIZE_RE.findall(model.split("/")[-1])
    if not matches:
        return None
    # Prefer the LAST match: names like "Qwen2.5-32B" or "Yi-1.5-9B" put
    # the version number first and the actual size last.
    return float(matches[-1])


size_points = []  # (params_B, mse, pretty_name)
for r in rows:
    slug = r["slug"]
    model = r["model"]
    if "@" in model:
        continue
    if abs(r["mse"] - 0.25) < 1e-3:
        continue
    if slug != model.replace("/", "_"):
        continue
    params = parse_param_count_b(model)
    if params is None:
        continue
    size_points.append((params, r["mse"], short_name(model)))

size_points.sort(key=lambda p: p[0])

fig, ax = plt.subplots(figsize=(10, 6))

xs = np.array([p[0] for p in size_points])
ys = np.array([p[1] for p in size_points])

ax.scatter(xs, ys, s=55, color=PRIMARY, edgecolor="white",
           linewidth=0.8, zorder=3)

# Best (lowest MSE) and worst (highest MSE) get coloured markers
best_i = int(np.argmin(ys))
worst_i = int(np.argmax(ys))
ax.scatter(xs[best_i], ys[best_i], s=90, color=HIGHLIGHT,
           edgecolor="white", linewidth=0.8, zorder=4)
ax.scatter(xs[worst_i], ys[worst_i], s=90, color=ACCENT,
           edgecolor="white", linewidth=0.8, zorder=4)

# Label every point; small offset to keep things readable
for x, y, name in size_points:
    ax.annotate(name, (x, y), xytext=(5, 4), textcoords="offset points",
                fontsize=8, color="#333333")

# Reference line: uniform-50% baseline
ax.axhline(0.25, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
ax.text(ax.get_xlim()[1] if False else xs.max() * 1.02, 0.25,
        "uniform 0.5 baseline", fontsize=8, color="#666666",
        va="center", ha="left")

ax.set_xscale("log")
ax.set_xlabel("Model size (billions of parameters, log scale)")
ax.set_ylabel("MSE vs. ground-truth water/land map")
ax.set_title("Reconstruction Error vs. Model Size")
ax.grid(True, which="both", alpha=0.35)
ax.set_axisbelow(True)

fig.tight_layout()
out = OUT_DIR / "model_size_experiment"
fig.savefig(out.with_suffix(".png"))
fig.savefig(out.with_suffix(".pdf"))
plt.close(fig)
print(f"[2/5] {len(size_points)} models -> {out}.png/.pdf")


# ------------------------------------------------------------
#  1c. MODEL-SIZE EXPERIMENT - bar version
# ------------------------------------------------------------
# Explicit slug-based allowlist (see thesis Section 1c). Slug-based (not
# model-name-based) so we can include specific suffix variants — e.g. the
# English-language run that stands in for the missing default gemma-3-27b-it,
# and the Qwen3-4B thinking-mode run alongside the non-thinking run.
#
# Value = display-name override; None means use short_name(model).
ALLOWLIST_1C_SLUGS = {
    "mistralai_Mistral-Nemo-Instruct-2407": None,
    "Qwen_Qwen2.5-0.5B-Instruct": None,
    "Qwen_Qwen2.5-1.5B-Instruct": None,
    "Qwen_Qwen2.5-3B-Instruct": None,
    "Qwen_Qwen2.5-7B-Instruct": None,
    "Qwen_Qwen2.5-14B-Instruct": None,
    "Qwen_Qwen2.5-32B-Instruct": None,
    "Qwen_Qwen2.5-72B-Instruct": None,
    "Qwen_Qwen2.5-Coder-32B-Instruct": None,
    "google_gemma-3-12b-it": None,
    # No bare normal run for gemma-3-27b-it yet; use the English-language run
    # (same prompt as default) as the stand-in.
    "google_gemma-3-27b-it_lang_en": "gemma-3-27b-it",
    "Qwen_Qwen2.5-32B-Instruct-AWQ": None,
    "Qwen_Qwen2.5-72B-Instruct-AWQ": None,
    "mistralai_Mistral-Small-24B-Instruct-2501": None,
    "casperhansen_mixtral-instruct-awq": None,
    "INSAIT-Institute_BgGPT-Gemma-3-12B-IT": None,
    "INSAIT-Institute_BgGPT-Gemma-3-27B-IT-GPTQ-W4A16": None,
    "Qwen_Qwen2.5-Coder-14B-Instruct": None,
    "Qwen_Qwen2.5-Coder-32B-Instruct-AWQ": None,
    "Qwen_Qwen3-4B": "Qwen3-4B (non-thinking)",
    "Qwen_Qwen3-4B_reasoning_n20": "Qwen3-4B (thinking, n=20)",
    "allenai_Olmo-3-7B-Instruct": None,
    "allenai_Olmo-3.1-32B-Instruct": None,
    "allenai_Olmo-3-7B-Think": None,
    "01-ai_Yi-6B-Chat": None,
    "01-ai_Yi-1.5-9B-Chat": None,
    "microsoft_Phi-3-medium-4k-instruct": None,
    "microsoft_phi-4": None,
    "upstage_SOLAR-10.7B-Instruct-v1.0": None,
    "HuggingFaceH4_zephyr-7b-beta": None,
    "internlm_internlm2_5-7b-chat": None,
    "ibm-granite_granite-3.1-8b-instruct": None,
    "tiiuae_Falcon3-10B-Instruct": None,
    "TinyLlama_TinyLlama-1.1B-Chat-v1.0": None,
}

size_points_1c = []  # (params_B, mse, display_name)
seen_slugs = set()
for r in rows:
    slug = r["slug"]
    if slug not in ALLOWLIST_1C_SLUGS or slug in seen_slugs:
        continue
    if "@" in r["model"]:
        continue
    if abs(r["mse"] - 0.25) < 1e-3:
        continue
    params = parse_param_count_b(r["model"])
    if params is None:
        continue
    override = ALLOWLIST_1C_SLUGS[slug]
    display = override if override else short_name(r["model"])
    size_points_1c.append((params, r["mse"], display))
    seen_slugs.add(slug)

# Sort by (params asc, display name) so 4B Qwen3 thinking/non-thinking sit
# next to each other deterministically.
size_points_1c.sort(key=lambda p: (p[0], p[2]))

missing_1c = sorted(set(ALLOWLIST_1C_SLUGS) - {r["slug"] for r in rows})
if missing_1c:
    print(f"     (1c: allowlisted but not in CSV, skipped: {len(missing_1c)} -> "
          + ", ".join(missing_1c) + ")")

fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(size_points_1c) + 2), 6))

bar_labels = [f"{name}\n({params:g}B)" for params, _, name in size_points_1c]
bar_values = [mse for _, mse, _ in size_points_1c]

best_i_1c = int(np.argmin(bar_values))
worst_i_1c = int(np.argmax(bar_values))

bar_colors = [PRIMARY] * len(size_points_1c)
bar_colors[best_i_1c] = HIGHLIGHT
bar_colors[worst_i_1c] = ACCENT

bars = ax.bar(np.arange(len(size_points_1c)), bar_values,
              color=bar_colors, edgecolor="white", width=0.78)

for bar, v in zip(bars, bar_values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.006,
            f"{v:.3f}", ha="center", va="bottom",
            fontsize=8, color="#333333")

ax.set_xticks(np.arange(len(size_points_1c)))
ax.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("MSE vs. ground-truth water/land map")
ax.set_title("Reconstruction Error by Model (sorted by parameter count)")
ax.set_ylim(0, max(bar_values) * 1.12)
ax.axhline(0.25, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
ax.text(len(size_points_1c) - 0.5, 0.25, " uniform 0.5 baseline",
        fontsize=8, color="#666666", va="center", ha="left")
ax.grid(True, axis="y", alpha=0.45)
ax.set_axisbelow(True)

fig.tight_layout()
out = OUT_DIR / "model_size_experiment_bars"
fig.savefig(out.with_suffix(".png"))
fig.savefig(out.with_suffix(".pdf"))
plt.close(fig)
print(f"[3/5] {len(size_points_1c)} models -> {out}.png/.pdf")


# ============================================================
#  2. ANCHOR EXPERIMENT
# ============================================================
# Qwen2.5-32B-Instruct-AWQ with 0 (baseline) / 5 / 10 / 15 / 20 anchors.
TARGET = "Qwen/Qwen2.5-32B-Instruct-AWQ"

anchor_data = {}  # n_anchors -> mse
for r in rows:
    if r["model"] != TARGET:
        continue
    slug = r["slug"]
    if slug == TARGET.replace("/", "_"):
        anchor_data[0] = r["mse"]
        continue
    m = re.search(r"_anchors(\d+)$", slug)
    if m:
        anchor_data[int(m.group(1))] = r["mse"]

ns = sorted(anchor_data.keys())
mses = [anchor_data[n] for n in ns]

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(ns))
bar_colors = [PRIMARY if n == 0 else "#5DADE2" for n in ns]
bars = ax.bar(x, mses, color=bar_colors, edgecolor="white", width=0.6)

for bar, v in zip(bars, mses):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004,
            f"{v:.3f}", ha="center", va="bottom", fontsize=10, color="#333333")

ax.set_xticks(x)
ax.set_xticklabels([("baseline\n(no anchors)" if n == 0 else f"{n} anchors") for n in ns])
ax.set_ylabel("MSE vs. ground-truth water/land map")
ax.set_title("Effect of Geographic Anchors on Reconstruction Error\n"
             "(Qwen2.5-32B-Instruct-AWQ)")
ax.set_ylim(0, max(mses) * 1.18)
ax.grid(True, axis="y", alpha=0.45)
ax.set_axisbelow(True)

fig.tight_layout()
out = OUT_DIR / "anchor_experiment"
fig.savefig(out.with_suffix(".png"))
fig.savefig(out.with_suffix(".pdf"))
plt.close(fig)
print(f"[4/5] {len(ns)} anchor settings -> {out}.png/.pdf")


# ============================================================
#  3. LANGUAGE EXPERIMENT
# ============================================================
# Group by model, plot one bar per language. Include 'default' (no _lang_)
# for models that also have language variants.
LANG_RE = re.compile(r"_lang_([a-z]{2})$")

# {(model_pretty): {lang_code: mse}}
lang_data: dict[str, dict[str, float]] = {}
for r in rows:
    slug = r["slug"]
    m = LANG_RE.search(slug)
    pretty = short_name(r["model"])
    if m:
        lang_data.setdefault(pretty, {})[m.group(1)] = r["mse"]

# stable ordering of models (by mean MSE) and languages
LANG_ORDER_PRIORITY = ["en", "de", "es", "ru", "zh", "bg"]
LANG_LABELS = {
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "zh": "Chinese",
    "bg": "Bulgarian",
}
LANG_COLORS = {
    "en":      "#2E5A87",
    "de":      "#F39C12",
    "es":      "#E67E22",
    "ru":      "#8E44AD",
    "zh":      "#16A085",
    "bg":      "#C0392B",
}

models_sorted = sorted(lang_data.keys(),
                       key=lambda m: np.mean(list(lang_data[m].values())))
all_langs = []
for lang in LANG_ORDER_PRIORITY:
    if any(lang in lang_data[m] for m in models_sorted):
        all_langs.append(lang)

n_models = len(models_sorted)
n_langs = len(all_langs)
group_w = 0.82
bar_w = group_w / n_langs

fig, ax = plt.subplots(figsize=(max(8, 1.6 * n_models + 2.5), 5.5))

x = np.arange(n_models)
for i, lang in enumerate(all_langs):
    vals = [lang_data[m].get(lang, np.nan) for m in models_sorted]
    offsets = x - group_w / 2 + (i + 0.5) * bar_w
    bars = ax.bar(offsets, vals, width=bar_w * 0.95,
                  label=LANG_LABELS[lang], color=LANG_COLORS[lang],
                  edgecolor="white")
    for bar, v in zip(bars, vals):
        if np.isnan(v):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="#333333")

ax.set_xticks(x)
ax.set_xticklabels(models_sorted, rotation=0)
ax.set_ylabel("MSE vs. ground-truth water/land map")
ax.set_title("Reconstruction Error per Prompt Language")
ax.set_ylim(0, max(max(d.values()) for d in lang_data.values()) * 1.20)
ax.grid(True, axis="y", alpha=0.45)
ax.set_axisbelow(True)
ax.legend(title="Prompt language", loc="upper left",
          frameon=False, ncol=min(n_langs, 4))

fig.tight_layout()
out = OUT_DIR / "language_experiment"
fig.savefig(out.with_suffix(".png"))
fig.savefig(out.with_suffix(".pdf"))
plt.close(fig)
print(f"[5/5] {n_models} models x {n_langs} languages -> {out}.png/.pdf")
