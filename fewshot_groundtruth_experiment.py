#!/usr/bin/env python3
"""
"Few-shot ground-truth" experiment — the controlled counterpart to
`memory_experiment.py`.

For each coordinate visited in scan order, the prompt includes the same
three previous coordinates that the memory experiment would have shown,
but labelled with the ground-truth verdict (looked up from
global_land_mask), not the model's own prediction.

This isolates the source of context-quality: identical prompt format,
identical sequence of in-context (lat, lon) pairs as memory_experiment.py
— only the labels differ.  If accuracy goes UP here and DOWN with
self-memory, the memory failure is about prediction quality, not about
the in-context format.

Outputs:
  Generated models/<slug>_groundtruth3.png
  Generated models/<slug>_groundtruth3_data.json
  MSEs/groundtruth3/mse_summary.{json,csv}
"""

import argparse
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from blind_model_experiment import (
    ANCHOR_POINTS,
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


EMPTY_PROMPT = (
    "If this location is over land, say 'Land'. "
    "If this location is over water, say 'Water'. "
    "Do not say anything else.\n"
    "{lat}, {lon}"
)

# Same format as memory_experiment.py so the in-prompt layout is identical;
# only the underlying labels differ.  We emit P(Land)=1.00 for land examples
# and P(Land)=0.00 for water examples — i.e. the ground truth presented as
# perfectly-confident "past answers".
GROUNDTRUTH_PROMPT = (
    "Below are previous land/water questions with the correct answers, "
    "given as confidence values (P(Land), where 1.0 means certainly land "
    "and 0.0 means certainly water):\n"
    "{history}\n"
    "Now answer the next question.\n"
    "If this location is over land, say 'Land'. "
    "If this location is over water, say 'Water'. "
    "Do not say anything else.\n"
    "{lat}, {lon}"
)

# Same templates as above but with the anchored-experiment prefix prepended.
# The prefix matches the original ANCHORED_PROMPT_TEMPLATE in
# blind_model_experiment.py so the spatial-context wording is identical.
ANCHORED_EMPTY_PROMPT = (
    "You are starting from {anchor_name} ({anchor_lat}, {anchor_lon}). "
    "If the following location is over land, say 'Land'. "
    "If it is over water, say 'Water'. "
    "Do not say anything else.\n"
    "{lat}, {lon}"
)

ANCHORED_GROUNDTRUTH_PROMPT = (
    "You are starting from {anchor_name} ({anchor_lat}, {anchor_lon}). "
    "Below are previous land/water questions with the correct answers, "
    "given as confidence values (P(Land), where 1.0 means certainly land "
    "and 0.0 means certainly water):\n"
    "{history}\n"
    "Now answer the next question.\n"
    "If the following location is over land, say 'Land'. "
    "If it is over water, say 'Water'. "
    "Do not say anything else.\n"
    "{lat}, {lon}"
)


def _format_history(history: List[Tuple[float, float, float]]) -> str:
    lines = []
    for i, (hlat, hlon, hp) in enumerate(history, start=1):
        verdict = "Land" if hp >= 0.5 else "Water"
        lines.append(
            f"{i}. {format_lat(hlat)}, {format_lon(hlon)} "
            f"-> {verdict} (P(Land)={hp:.2f})"
        )
    return "\n".join(lines)


def build_prompt(
    lat: float,
    lon: float,
    history: List[Tuple[float, float, float]],
    anchor: Optional[Tuple[str, str, str]] = None,
) -> str:
    """Build the user prompt; if `anchor` is given, prepend the
    anchored-experiment preamble in front of either the empty or
    ground-truth-history template."""
    lat_s, lon_s = format_lat(lat), format_lon(lon)
    if anchor is not None:
        anchor_name, anchor_lat, anchor_lon = anchor
        if not history:
            return ANCHORED_EMPTY_PROMPT.format(
                anchor_name=anchor_name, anchor_lat=anchor_lat,
                anchor_lon=anchor_lon, lat=lat_s, lon=lon_s,
            )
        return ANCHORED_GROUNDTRUTH_PROMPT.format(
            anchor_name=anchor_name, anchor_lat=anchor_lat,
            anchor_lon=anchor_lon, history=_format_history(history),
            lat=lat_s, lon=lon_s,
        )
    if not history:
        return EMPTY_PROMPT.format(lat=lat_s, lon=lon_s)
    return GROUNDTRUTH_PROMPT.format(
        history=_format_history(history), lat=lat_s, lon=lon_s,
    )


def query_one(
    client: OpenAI,
    model: str,
    lat: float,
    lon: float,
    history: List[Tuple[float, float, float]],
    anchor: Optional[Tuple[str, str, str]] = None,
) -> float:
    prompt = build_prompt(lat, lon, history, anchor=anchor)
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


def run_groundtruth_experiment(
    model: str,
    resolution: float,
    base_url: str,
    output_dir: Path,
    memory_size: int = 3,
    anchor: Optional[Tuple[str, str, str]] = None,
    anchor_key: Optional[str] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    coords, n_rows, n_cols = generate_coordinates(resolution)
    ground_truth = build_ground_truth_grid(coords, n_rows, n_cols)
    save_ground_truth_assets(
        ground_truth=ground_truth, resolution=resolution, output_dir=output_dir
    )

    client = OpenAI(api_key=API_KEY, base_url=base_url)

    try:
        models_list = client.models.list()
        available = [m.id for m in models_list.data]
        print(f"Server models: {available}")
        if model not in available:
            print(f"WARNING: requested model '{model}' not in server model list!")
    except Exception as e:
        raise RuntimeError(f"Could not reach vLLM server at {base_url}: {e}") from e

    total = len(coords)
    print(f"Model:       {model}")
    print(f"Resolution:  {resolution}° -> {n_cols}×{n_rows} = {total} queries")
    print(f"Context:     {memory_size} previous coords with GROUND-TRUTH labels")
    if anchor is not None:
        print(f"Anchor:      {anchor[0]} ({anchor[1]}, {anchor[2]})")
    print(f"Mode:        fully sequential single chain (scan order)")
    print()

    # Print one sample prompt with real ground-truth labels for the first
    # `memory_size` coords so we can eyeball the structure.
    if total > memory_size + 1:
        seed_hist = []
        for r, c, slat, slon in coords[:memory_size]:
            seed_hist.append(
                (slat, slon, 1.0 if globe.is_land(float(slat), float(slon)) else 0.0)
            )
        r, c, lat, lon = coords[memory_size]
        sample_prompt = build_prompt(lat, lon, seed_hist, anchor=anchor)
        print("---- Sample prompt (real ground-truth history) ----")
        print(sample_prompt)
        print("---------------------------------------------------\n")

    grid = np.full((n_rows, n_cols), 0.5)
    history: "deque[Tuple[float, float, float]]" = deque(maxlen=memory_size)

    t0 = time.time()
    for r, c, lat, lon in tqdm(coords, desc="GT chain", unit="px"):
        p_land = query_one(client, model, lat, lon, list(history), anchor=anchor)
        grid[r, c] = p_land
        # Append the *ground truth* for this coord, not the model's prediction.
        # The next query's context will see this entry as P(Land)=1.00 or 0.00.
        is_land_truth = 1.0 if globe.is_land(float(lat), float(lon)) else 0.0
        history.append((lat, lon, is_land_truth))
    elapsed = time.time() - t0
    qps = total / elapsed if elapsed > 0 else 0.0

    mse = compute_mse(grid, ground_truth)
    print(f"\nDone in {elapsed:.1f}s ({qps:.1f} queries/sec)")
    print(f"MSE vs ground truth: {mse:.6f}")

    slug = slugify_model_name(model)
    suffix = f"_groundtruth{memory_size}"
    if anchor is not None:
        # File-name suffix matches the existing anchored-experiment convention:
        # `_from_<lowercase first word of anchor name>`.
        anchor_tag = (anchor_key or anchor[0].split(",")[0]
                      ).lower().replace(" ", "")
        suffix = f"{suffix}_from_{anchor_tag}"
    map_path = output_dir / f"{slug}{suffix}.png"
    data_path = output_dir / f"{slug}{suffix}_data.json"

    write_json(
        data_path,
        {
            "model": model,
            "context_source": "groundtruth",
            "memory_size": memory_size,
            "anchor": (
                {"name": anchor[0], "lat": anchor[1], "lon": anchor[2],
                 "key": anchor_key} if anchor is not None else None
            ),
            "resolution_deg": resolution,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "elapsed_sec": round(elapsed, 2),
            "mse": mse,
            "grid": grid.tolist(),
        },
    )
    title = f"{model} (groundtruth={memory_size}"
    if anchor is not None:
        title += f", from={anchor[0]}"
    title += ")"
    render_map(grid, map_path, title, resolution)
    print(f"Raw data saved to {data_path}")

    return {
        "model": model,
        "slug": f"{slug}{suffix}",
        "context_source": "groundtruth",
        "memory_size": memory_size,
        "anchor_key": anchor_key,
        "anchor_name": anchor[0] if anchor is not None else None,
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
        description="Few-shot ground-truth blind-earth (control for memory experiment)."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION)
    parser.add_argument(
        "--memory-size",
        type=int,
        default=3,
        help="How many previous coords to include with ground-truth labels (default: 3).",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default=None,
        help=(
            "Optional anchor name (e.g. 'mannheim', 'tokyo', 'nyc'). When set, "
            "each prompt is prefixed with 'You are starting from <anchor> ...' "
            "so the experiment combines few-shot ground-truth context with the "
            "anchored-prompt setup. "
            f"Built-in anchors: {', '.join(ANCHOR_POINTS.keys())}."
        ),
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

    # Resolve --anchor key to (name, lat, lon) tuple.
    anchor_tuple: Optional[Tuple[str, str, str]] = None
    anchor_key: Optional[str] = None
    if args.anchor:
        key = args.anchor.lower().strip()
        if key not in ANCHOR_POINTS:
            raise SystemExit(
                f"Unknown anchor '{args.anchor}'. "
                f"Available: {', '.join(ANCHOR_POINTS.keys())}"
            )
        anchor_tuple = ANCHOR_POINTS[key]
        anchor_key = key
        print(f"Using anchor: {anchor_tuple[0]} "
              f"({anchor_tuple[1]}, {anchor_tuple[2]})")

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

        result = run_groundtruth_experiment(
            model=args.model,
            resolution=args.resolution,
            base_url=args.base_url,
            output_dir=args.generated_models_dir,
            memory_size=args.memory_size,
            anchor=anchor_tuple,
            anchor_key=anchor_key,
        )

        # Sub-directory: groundtruth<N> for the no-anchor variant,
        # groundtruth<N>_from_<anchor_key> when anchored, so summaries
        # don't collide across runs.
        sub = f"groundtruth{args.memory_size}"
        if anchor_key:
            sub = f"{sub}_from_{anchor_key}"
        mse_subdir = args.mses_dir / sub
        summary = {
            "resolution_deg": args.resolution,
            "memory_size": args.memory_size,
            "context_source": "groundtruth",
            "anchor": anchor_key,
            "results": [result],
        }
        write_mse_summaries(summary, mse_subdir)
    finally:
        stop_vllm_server(server_process)


if __name__ == "__main__":
    main()
