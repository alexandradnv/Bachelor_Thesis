#!/usr/bin/env python3
"""
Sort the flat `Generated models/` directory into per-experiment sub-folders.

Folder scheme (each holds the matching .png + _data.json pairs):
  normal_experiment/                   plain baseline runs (no extra suffix)
  anchored_experiment/                 *_from_<anchor> or *_anchors<N>
  language_experiment/                 *_lang_<code>
  reasoning_experiment/                *_reasoning_*
  memory_experiment/                   *_memory<N>[_shuffled|_perrowseed]
  ground_truth_memory_experiment/      *_groundtruth<N>[_from_<anchor>]

Shared assets `ground_truth_map.{png,json}` stay at the root.

Idempotent: re-running picks up newly-arrived files and leaves already-sorted
ones in place.

Usage:
    .venv/bin/python organize_generated_models.py
    .venv/bin/python organize_generated_models.py --dry-run   # show plan only
"""

import argparse
import re
import shutil
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR / "Generated models"

CATEGORIES = (
    "normal_experiment",
    "anchored_experiment",
    "language_experiment",
    "reasoning_experiment",
    "memory_experiment",
    "ground_truth_memory_experiment",
)

# Files that should never be moved (shared reference assets).
KEEP_AT_ROOT = {"ground_truth_map.png", "ground_truth_map_data.json"}

# Pre-compiled regexes for category detection.  Order matters — the first
# match wins, so the more specific categories must come first.
_RE_GROUNDTRUTH = re.compile(r"_groundtruth\d+")
_RE_MEMORY      = re.compile(r"_memory\d+")
_RE_REASONING   = re.compile(r"_reasoning")
_RE_LANG        = re.compile(r"_lang_[a-z]{2,3}")
_RE_FROM        = re.compile(r"_from_")
_RE_ANCHORS     = re.compile(r"_anchors\d+")


def categorize(stem: str) -> Optional[str]:
    """Return the sub-folder name for a file stem, or None if it should
    stay at the root."""
    if _RE_GROUNDTRUTH.search(stem):
        return "ground_truth_memory_experiment"
    if _RE_MEMORY.search(stem):
        return "memory_experiment"
    if _RE_REASONING.search(stem):
        return "reasoning_experiment"
    if _RE_LANG.search(stem):
        return "language_experiment"
    if _RE_FROM.search(stem) or _RE_ANCHORS.search(stem):
        return "anchored_experiment"
    return "normal_experiment"


def _pair_stem(path: Path) -> str:
    """The 'base' identifier shared by <stem>.png and <stem>_data.json."""
    name = path.name
    if name.endswith("_data.json"):
        return name[:-len("_data.json")]
    return path.stem  # strips .png


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Folder to organize (default: ./Generated models)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned moves but don't touch the filesystem.")
    args = parser.parse_args()

    if not args.root.exists():
        raise SystemExit(f"Root not found: {args.root}")

    # Collect file stems at the root (not in sub-folders).
    files = [p for p in args.root.iterdir() if p.is_file()]

    # Pre-create category sub-dirs (no-op if dry-run).
    if not args.dry_run:
        for cat in CATEGORIES:
            (args.root / cat).mkdir(exist_ok=True)

    moved = {cat: 0 for cat in CATEGORIES}
    kept = 0
    skipped = 0

    for path in sorted(files):
        if path.name in KEEP_AT_ROOT:
            kept += 1
            continue
        stem = _pair_stem(path)
        if not stem:
            skipped += 1
            continue
        cat = categorize(stem)
        if cat is None:
            kept += 1
            continue
        dest_dir = args.root / cat
        dest = dest_dir / path.name
        if dest.exists():
            # Already sorted; skip silently.
            skipped += 1
            continue
        action = "would move" if args.dry_run else "moving"
        print(f"  {action}  {path.name:70s} -> {cat}/")
        if not args.dry_run:
            shutil.move(str(path), str(dest))
        moved[cat] += 1

    print()
    print("Summary:")
    for cat in CATEGORIES:
        print(f"  {cat:38s}  {moved[cat]:4d} file(s) {'planned' if args.dry_run else 'moved'}")
    print(f"  kept-at-root (ground_truth_map etc.)    {kept:4d}")
    print(f"  skipped (already sorted or unrecognized) {skipped:4d}")


if __name__ == "__main__":
    main()
