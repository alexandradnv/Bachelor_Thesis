#!/usr/bin/env python3
"""
"Memory" experiment: does giving the model memory of its own past predictions
change its land/water map?

For each lat/lon coordinate (visited in scan order: top-left -> bottom-right),
include the previous N prompts and the model's P(Land) answers inside the
current user message.  The model then has explicit recall of its own
confidence on the most recent coordinates.

This is the same task as `blind_model_experiment.py`, but:
  * fully sequential (single chain, no thread pool), so the memory really is
    a chain of consecutive queries;
  * the prompt grows to include the last `--memory-size` (default 3)
    (lat, lon, P(Land), Land/Water-verdict) tuples.

Outputs `<slug>_memory<N>.png` and `<slug>_memory<N>_data.json` in the
"Generated models" folder, plus an MSE summary in MSEs/memory<N>/.
"""

import argparse
import random
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from blind_model_experiment import (
    API_KEY,
    DEFAULT_GENERATED_MODELS_DIR,
    DEFAULT_MSE_DIR,
    DEFAULT_RESOLUTION,
    DEFAULT_SERVER_SCRIPT,
    DEFAULT_SERVER_STARTUP_TIMEOUT,
    VLLM_BASE_URL,
    _extract_p_land_from_logprobs,
    build_ground_truth_grid,
    compute_mse,
    format_lat,
    format_lon,
    generate_coordinates,
    render_map,
    save_ground_truth_assets,
    slugify_model_name,
    start_vllm_server_for_model,
    stop_vllm_server,
    write_json,
    write_mse_summaries,
)

try:
    from global_land_mask import globe
except ImportError as exc:
    raise SystemExit(
        "global-land-mask is required: pip install global-land-mask"
    ) from exc


# Fixed seed coordinates injected at the start of every row in `perrowseed`
# mode. Mix of two land + one water spread across hemispheres so the model
# sees both verdicts and is not anchored toward either Land or Water.
PER_ROW_SEED_COORDS = [
    (49.5, 8.5),       # Mannheim, Germany (Land)
    (0.0, -150.0),     # Central Pacific  (Water)
    (35.7, 139.7),     # Tokyo, Japan     (Land)
]


def _build_seed_history() -> List[Tuple[float, float, float]]:
    """Ground-truth-labelled seed entries for `perrowseed` mode."""
    return [
        (lat, lon, 1.0 if globe.is_land(lat, lon) else 0.0)
        for lat, lon in PER_ROW_SEED_COORDS
    ]


VALID_MODES = ("sequential", "shuffle", "perrowseed")


# Prompt used on the very first queries (memory window still empty).
EMPTY_MEMORY_PROMPT = (
    "If this location is over land, say 'Land'. "
    "If this location is over water, say 'Water'. "
    "Do not say anything else.\n"
    "{lat}, {lon}"
)

# Prompt used once at least one previous answer exists.
# `{history}` is filled with a numbered list of past (coord, verdict, P(Land))
# entries; oldest first, most recent last.
MEMORY_PROMPT = (
    "Below are your previous answers to land/water questions, with the "
    "confidence you gave for each (P(Land), where 1.0 means certainly land "
    "and 0.0 means certainly water):\n"
    "{history}\n"
    "Now answer the next question.\n"
    "If this location is over land, say 'Land'. "
    "If this location is over water, say 'Water'. "
    "Do not say anything else.\n"
    "{lat}, {lon}"
)


def _format_history(history: List[Tuple[float, float, float]]) -> str:
    """Render the memory window as a numbered list (oldest -> newest)."""
    lines = []
    for i, (hlat, hlon, hp) in enumerate(history, start=1):
        verdict = "Land" if hp >= 0.5 else "Water"
        lines.append(
            f"{i}. {format_lat(hlat)}, {format_lon(hlon)} "
            f"-> {verdict} (P(Land)={hp:.2f})"
        )
    return "\n".join(lines)


def build_memory_prompt(
    lat: float, lon: float, history: List[Tuple[float, float, float]]
) -> str:
    if not history:
        return EMPTY_MEMORY_PROMPT.format(lat=format_lat(lat), lon=format_lon(lon))
    return MEMORY_PROMPT.format(
        history=_format_history(history),
        lat=format_lat(lat),
        lon=format_lon(lon),
    )


def query_with_memory(
    client: OpenAI,
    model: str,
    lat: float,
    lon: float,
    history: List[Tuple[float, float, float]],
) -> float:
    """One memory-conditioned query.  Returns P(Land) via top-logprobs."""
    prompt = build_memory_prompt(lat, lon, history)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
        )
        choice = response.choices[0]
        if not choice.logprobs or not choice.logprobs.content:
            text = (choice.message.content or "").strip().lower()
            if "land" in text:
                return 1.0
            if "water" in text:
                return 0.0
            return 0.5
        top = choice.logprobs.content[0].top_logprobs
        return _extract_p_land_from_logprobs(top, choice.message.content or "")
    except Exception as e:
        print(f"  [warn] query failed for ({lat}, {lon}): {e}")
        return 0.5


def _mode_suffix(mode: str, memory_size: int) -> str:
    """File-name suffix per mode (sequential keeps the legacy name)."""
    base = f"_memory{memory_size}"
    if mode == "sequential":
        return base
    if mode == "shuffle":
        return f"{base}_shuffled"
    if mode == "perrowseed":
        return f"{base}_perrowseed"
    raise ValueError(f"Unknown mode: {mode}")


def run_memory_experiment(
    model: str,
    resolution: float,
    base_url: str,
    output_dir: Path,
    memory_size: int = 3,
    mode: str = "sequential",
    seed: int = 42,
    coords=None,
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
    ground_truth=None,
):
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if coords is None or n_rows is None or n_cols is None:
        coords, n_rows, n_cols = generate_coordinates(resolution)
    if ground_truth is None:
        ground_truth = build_ground_truth_grid(coords, n_rows, n_cols)
    save_ground_truth_assets(
        ground_truth=ground_truth, resolution=resolution, output_dir=output_dir
    )

    # Visit-order depends on mode.  Scan order is the iteration order returned
    # by generate_coordinates(); shuffle randomises it with a fixed seed.
    visit_order = list(coords)
    if mode == "shuffle":
        rng = random.Random(seed)
        rng.shuffle(visit_order)

    client = OpenAI(api_key=API_KEY, base_url=base_url)

    try:
        models_list = client.models.list()
        available = [m.id for m in models_list.data]
        print(f"Server models: {available}")
        if model not in available:
            print(f"WARNING: requested model '{model}' not in server model list!")
    except Exception as e:
        raise RuntimeError(f"Could not reach vLLM server at {base_url}: {e}") from e

    total = len(visit_order)
    print(f"Model:       {model}")
    print(f"Resolution:  {resolution}° -> {n_cols}×{n_rows} = {total} queries")
    print(f"Memory:      {memory_size} previous (coord, P(Land)) entries")
    mode_desc = {
        "sequential": "fully sequential single chain (scan order)",
        "shuffle":    f"fully sequential single chain (SHUFFLED order, seed={seed})",
        "perrowseed": "scan order; memory reset to 3 ground-truth anchors at every row start",
    }[mode]
    print(f"Mode:        {mode_desc}")
    if mode == "perrowseed":
        print(f"Seed coords: {_build_seed_history()}")
    print()

    # Print a sample full prompt so we can sanity-check the memory text.
    if total > memory_size + 1:
        sample_history = [(0.0, 0.0, 0.5)] * memory_size
        r, c, lat, lon = visit_order[memory_size + 1]
        sample_prompt = build_memory_prompt(lat, lon, sample_history)
        print("---- Sample prompt (with dummy history) ----")
        print(sample_prompt)
        print("--------------------------------------------\n")

    grid = np.full((n_rows, n_cols), 0.5)
    history: "deque[Tuple[float, float, float]]" = deque(maxlen=memory_size)
    current_row: Optional[int] = None

    t0 = time.time()
    for r, c, lat, lon in tqdm(visit_order, desc=f"Memory chain ({mode})", unit="px"):
        if mode == "perrowseed" and r != current_row:
            history.clear()
            for entry in _build_seed_history():
                history.append(entry)
            current_row = r

        p_land = query_with_memory(client, model, lat, lon, list(history))
        grid[r, c] = p_land
        history.append((lat, lon, p_land))
    elapsed = time.time() - t0
    qps = total / elapsed if elapsed > 0 else 0.0

    mse = compute_mse(grid, ground_truth)
    print(f"\nDone in {elapsed:.1f}s ({qps:.1f} queries/sec)")
    print(f"MSE vs ground truth: {mse:.6f}")

    slug = slugify_model_name(model)
    suffix = _mode_suffix(mode, memory_size)
    map_path = output_dir / f"{slug}{suffix}.png"
    data_path = output_dir / f"{slug}{suffix}_data.json"

    write_json(
        data_path,
        {
            "model": model,
            "memory_size": memory_size,
            "mode": mode,
            "seed": seed if mode == "shuffle" else None,
            "per_row_seed_coords": (
                PER_ROW_SEED_COORDS if mode == "perrowseed" else None
            ),
            "resolution_deg": resolution,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "elapsed_sec": round(elapsed, 2),
            "mse": mse,
            "grid": grid.tolist(),
        },
    )
    title = f"{model} (memory={memory_size}, mode={mode})"
    render_map(grid, map_path, title, resolution)
    print(f"Raw data saved to {data_path}")

    return {
        "model": model,
        "slug": f"{slug}{suffix}",
        "memory_size": memory_size,
        "mode": mode,
        "status": "completed",
        "mse": mse,
        "elapsed_sec": round(elapsed, 2),
        "resolution_deg": resolution,
        "map_path": str(map_path),
        "data_path": str(data_path),
        "error": "",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Blind-earth with memory: chain of consecutive queries "
        "that include the model's last N (coord, P(Land)) answers."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION)
    parser.add_argument(
        "--memory-size",
        type=int,
        default=3,
        help="How many previous answers to include in each prompt (default: 3).",
    )
    parser.add_argument(
        "--mode",
        default="sequential",
        choices=VALID_MODES,
        help=(
            "Memory-chain protocol:\n"
            "  sequential  — scan order, model's own past P(Land) fills memory (default).\n"
            "  shuffle     — randomised visit order, otherwise same as sequential.\n"
            "  perrowseed  — scan order, but memory is reset to 3 fixed ground-truth\n"
            "                anchor coords at the start of every row."
        ),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=VALID_MODES,
        default=None,
        help=(
            "Run multiple modes back-to-back in one Python process "
            "(re-uses the same vLLM server). Overrides --mode."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for `--mode shuffle` (ignored otherwise).",
    )
    parser.add_argument("--base-url", default=VLLM_BASE_URL)
    parser.add_argument(
        "--server-script", type=Path, default=DEFAULT_SERVER_SCRIPT
    )
    parser.add_argument(
        "--server-startup-timeout",
        type=int,
        default=DEFAULT_SERVER_STARTUP_TIMEOUT,
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--reuse-existing-server",
        action="store_true",
        help="Skip starting vLLM; assume a server is already running.",
    )
    parser.add_argument(
        "--generated-models-dir",
        type=Path,
        default=DEFAULT_GENERATED_MODELS_DIR,
    )
    parser.add_argument("--mses-dir", type=Path, default=DEFAULT_MSE_DIR)
    args = parser.parse_args()

    server_process: Optional[object] = None
    try:
        if not args.reuse_existing_server:
            server_process = start_vllm_server_for_model(
                model=args.model,
                base_url=args.base_url,
                server_script=args.server_script,
                timeout_sec=args.server_startup_timeout,
                tensor_parallel_size=args.tensor_parallel_size,
            )

        modes_to_run = args.modes if args.modes else [args.mode]

        # Pre-compute coords + ground truth once and reuse for every mode in
        # this Python process so we don't pay the global_land_mask cost twice.
        from blind_model_experiment import generate_coordinates as _gc
        coords, n_rows, n_cols = _gc(args.resolution)
        gt = build_ground_truth_grid(coords, n_rows, n_cols)

        results = []
        for mode in modes_to_run:
            result = run_memory_experiment(
                model=args.model,
                resolution=args.resolution,
                base_url=args.base_url,
                output_dir=args.generated_models_dir,
                memory_size=args.memory_size,
                mode=mode,
                seed=args.seed,
                coords=coords,
                n_rows=n_rows,
                n_cols=n_cols,
                ground_truth=gt,
            )
            results.append(result)

        # Each mode gets its own MSE sub-directory so summaries don't overwrite.
        for result in results:
            mode = result["mode"]
            if mode == "sequential":
                sub = f"memory{args.memory_size}"
            else:
                sub = f"memory{args.memory_size}_{mode}"
            summary = {
                "resolution_deg": args.resolution,
                "memory_size": args.memory_size,
                "mode": mode,
                "results": [result],
            }
            write_mse_summaries(summary, args.mses_dir / sub)
    finally:
        stop_vllm_server(server_process)


if __name__ == "__main__":
    main()
