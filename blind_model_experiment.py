#!/usr/bin/env python3
"""
"How Does A Blind Model See The Earth?" — Experiment Recreation

Based on: https://www.lesswrong.com/posts/xwdRzJxyqFqgXTWbH

For each lat/lon coordinate on a grid, asks the LLM whether that location is
over land or water. Uses logprobs to extract the model's confidence, then
renders the result as an equirectangular world map image.

This version can also:
    1. Start one vLLM model after another.
    2. Render every generated map into the "Generated models" folder.
    3. Create a real land/water ground-truth map.
    4. Compute mean squared error (MSE) against that ground truth.
    5. Save MSE summaries into the "MSEs" folder.

Requirements:
    pip install openai numpy matplotlib tqdm global-land-mask
"""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VLLM_BASE_URL = "http://localhost:8000/v1"
API_KEY = "not-needed"

# 2° --> 180×90 pixels = 16,200 queries
DEFAULT_RESOLUTION = 2

# How many requests to send in parallel 
DEFAULT_WORKERS = 32
DEFAULT_SERVER_STARTUP_TIMEOUT = 7200

# Directories + filenames
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GENERATED_MODELS_DIR = SCRIPT_DIR / "Generated models"
DEFAULT_MSE_DIR = SCRIPT_DIR / "MSEs"
DEFAULT_SERVER_SCRIPT = SCRIPT_DIR / "vllm_server.py"
GROUND_TRUTH_MAP_NAME = "ground_truth_map.png"
GROUND_TRUTH_DATA_NAME = "ground_truth_map_data.json"


# ! Gemma models are gated on Hugging Face and require authentication --> will fail
TESTED_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen3-32B-Instruct",
]

# Student-friendly model list: models that fit on 48GB GPUs (<7B parameters) +
# are publicly available
STUDENT_TESTED_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

# Alternative lightweight open-source models (no authentication required)
# Mix of providers to test different architectures and training approaches
ALT_STUDENT_TESTED_MODELS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B - very lightweight
    "microsoft/Phi-3-mini-4k-instruct",     # 3.8B - Microsoft's efficient model
    "mistralai/Mistral-7B-Instruct-v0.3",   # 7B - high quality
    "01-ai/Yi-6B-Chat",                     # 6B - good geographic knowledge
]

# Diverse well-known open-source models from 7 different families, 7B–32B.
# All fit on 2×48 GB GPUs (tensor-parallel-size=2).
# Models marked [gated] require accepting a licence on HuggingFace first
# (set HF_TOKEN in your environment before running).
DIVERSE_OS_MODELS = [
    "mistralai/Mistral-Nemo-Instruct-2407",    # 12B - Mistral Nemo
    "mistralai/Mistral-Small-24B-Instruct-2501", # 24B - Mistral Small
    "microsoft/phi-4",                         # 14B - Microsoft Phi-4
    "Qwen/Qwen2.5-32B-Instruct",              # 32B - Qwen 2.5 (general, not coder)
    "allenai/OLMo-3-1025-7B-Instruct",        # 7B  - AllenAI OLMo-3
    "tiiuae/Falcon3-10B-Instruct",             # 10B - TII Falcon 3
    "ibm-granite/granite-3.1-8b-instruct",    # 8B  - IBM Granite 3.1
]

# Open-source reasoning models (second batch). No API keys or gated access required.
# All fit on 2×48 GB GPUs (tensor-parallel-size=2).
DIVERSE_OS_MODELS_V2 = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",    # 14B - DeepSeek R1 reasoning distill
    "allenai/Olmo-3-7B-Think",                      # 7B  - AllenAI OLMo-3 Think (reasoning)
    "allenai/Olmo-3-7B-Instruct",                   # 7B  - AllenAI OLMo-3 Instruct (final)
]

# Checkpoints to test how geographic knowledge evolves over training.
# AllenAI hosts intermediate OLMo checkpoints as HuggingFace revision branches.
# We append `@branch_name` which gets split into --model + --revision for vLLM.
#
# Stage 1: main pretraining (~5.9T tokens, steps 2000..1413814)
# Stage 3: long-context fine-tuning (~50B tokens, steps 1000..11921)
# `main`: the final released checkpoint
#
# To list all available branches programmatically:
#   from huggingface_hub import list_repo_refs
#   refs = list_repo_refs("allenai/OLMo-3-1025-7B")
#   print([b.name for b in refs.branches])
OLMO3_CHECKPOINTS = [
    # Logarithmic spacing: dense early (knowledge emerging), sparse late (refinement)
    # Very early — random baseline, no knowledge yet
    "allenai/OLMo-3-1025-7B@stage1-step1000",
    "allenai/OLMo-3-1025-7B@stage1-step5000",
    # Early — language patterns forming
    "allenai/OLMo-3-1025-7B@stage1-step10000",
    "allenai/OLMo-3-1025-7B@stage1-step25000",
    "allenai/OLMo-3-1025-7B@stage1-step50000",
    # Mid — geographic knowledge likely emerging (dense sampling here)
    "allenai/OLMo-3-1025-7B@stage1-step75000",
    "allenai/OLMo-3-1025-7B@stage1-step100000",
    "allenai/OLMo-3-1025-7B@stage1-step150000",
    "allenai/OLMo-3-1025-7B@stage1-step200000",
    "allenai/OLMo-3-1025-7B@stage1-step300000",
    # Late — refinement (sparser)
    "allenai/OLMo-3-1025-7B@stage1-step500000",
    "allenai/OLMo-3-1025-7B@stage1-step750000",
    "allenai/OLMo-3-1025-7B@stage1-step1000000",
    "allenai/OLMo-3-1025-7B@stage1-step1413814",
    # Long-context fine-tuning (stage 3) — does it change geographic knowledge?
    "allenai/OLMo-3-1025-7B@stage3-step1000",
    "allenai/OLMo-3-1025-7B@stage3-step6000",
    "allenai/OLMo-3-1025-7B@stage3-step11921",
    # Final released checkpoint
    "allenai/OLMo-3-1025-7B@main",
]

# OLMo-3 32B (allenai/Olmo-3-1125-32B) checkpoint evolution.
# Same logarithmic spacing strategy as the 7B model.
# Stage1 runs to step ~656000; stage3 to step 11921.
OLMO3_32B_CHECKPOINTS = [
    # Very early — random baseline, no knowledge yet
    "allenai/Olmo-3-1125-32B@stage1-step1000",
    "allenai/Olmo-3-1125-32B@stage1-step5000",
    # Early — language patterns forming
    "allenai/Olmo-3-1125-32B@stage1-step10000",
    "allenai/Olmo-3-1125-32B@stage1-step25000",
    "allenai/Olmo-3-1125-32B@stage1-step50000",
    # Mid — geographic knowledge likely emerging (dense sampling here)
    "allenai/Olmo-3-1125-32B@stage1-step75000",
    "allenai/Olmo-3-1125-32B@stage1-step100000",
    "allenai/Olmo-3-1125-32B@stage1-step150000",
    "allenai/Olmo-3-1125-32B@stage1-step200000",
    "allenai/Olmo-3-1125-32B@stage1-step300000",
    # Late — refinement (sparser)
    "allenai/Olmo-3-1125-32B@stage1-step500000",
    "allenai/Olmo-3-1125-32B@stage1-step656000",
    # Long-context fine-tuning (stage 3) — does it change geographic knowledge?
    "allenai/Olmo-3-1125-32B@stage3-step1000",
    "allenai/Olmo-3-1125-32B@stage3-step6000",
    "allenai/Olmo-3-1125-32B@stage3-step11921",
    # Final released checkpoint
    "allenai/Olmo-3-1125-32B@main",
]

# Minimal Jinja2 chat template for base (non-instruct) models.
# Just concatenates message content so the /chat/completions endpoint works.
BASE_MODEL_CHAT_TEMPLATE = (
    "{% for message in messages %}{{ message['content'] }}{% endfor %}"
)

_CHAT_MODEL_KEYWORDS = {"instruct", "chat", "-it", "think", "qwen3"}

"""Heuristic: return True if the model name suggests it is instruction-tuned."""
def _looks_like_chat_model(model_id: str) -> bool:
    lower = model_id.lower()
    return any(kw in lower for kw in _CHAT_MODEL_KEYWORDS)

"""Heuristic: return True if the model is a 'Think' / reasoning model."""
def _is_thinking_model(model_id: str) -> bool:
    return "think" in model_id.lower()

# ChatML template for Think models that strips the forced <think> prefix.
# This lets the model answer directly so max_tokens=1 + logprobs still works.
THINK_MODEL_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'user' %}"
    "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<|im_start|>assistant\n{{ message['content'] }}"
    "{% if not loop.last %}<|im_end|>\n{% endif %}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


# The prompt template:  {lat} and {lon} will be filled in
PROMPT_TEMPLATE = (
    "If this location is over land, say 'Land'. "
    "If this location is over water, say 'Water'. "
    "Do not say anything else.\n"
    "{lat}, {lon}"
)

# Language & cultural-bias experiment: identical prompt translated into
# several languages.  The expected response token is also translated, so the
# model must answer in-language for the logprob extractor to score it as
# confident.  Latitude/longitude formatting stays in ASCII (degrees + N/S/E/W)
# so the only variable is the natural-language wrapper.
LANGUAGE_CONFIGS = {
    "en": {
        "name": "English",
        "prompt": (
            "If this location is over land, say 'Land'. "
            "If this location is over water, say 'Water'. "
            "Do not say anything else.\n"
            "{lat}, {lon}"
        ),
        "land_word": "Land",
        "water_word": "Water",
    },
    "de": {
        "name": "German",
        "prompt": (
            "Ist dieser Ort an Land oder im Wasser? "
            "Antworte nur mit 'Land' oder 'Wasser'.\n"
            "{lat}, {lon}"
        ),
        "land_word": "Land",
        "water_word": "Wasser",
    },
    "es": {
        "name": "Spanish",
        "prompt": (
            "Si esta ubicación está sobre tierra, responde 'Tierra'. "
            "Si está sobre agua, responde 'Agua'. "
            "No digas nada más.\n"
            "{lat}, {lon}"
        ),
        "land_word": "Tierra",
        "water_word": "Agua",
    },
    "zh": {
        "name": "Mandarin",
        "prompt": (
            "如果这个位置在陆地上，请回答'陆地'。"
            "如果这个位置在水域中，请回答'水域'。"
            "不要回答其他任何内容。\n"
            "{lat}, {lon}"
        ),
        "land_word": "陆",
        "water_word": "水",
    },
    "ru": {
        "name": "Russian",
        "prompt": (
            "Если это место находится на суше, ответь 'Суша'. "
            "Если оно находится в воде, ответь 'Вода'. "
            "Не говори ничего другого.\n"
            "{lat}, {lon}"
        ),
        "land_word": "Суша",
        "water_word": "Вода",
    },
    "bg": {
        "name": "Bulgarian",
        "prompt": (
            "Ако това място се намира на сушата, отговори 'Земя'. "
            "Ако се намира във вода, отговори 'Вода'. "
            "Не казвай нищо друго.\n"
            "{lat}, {lon}"
        ),
        "land_word": "Земя",
        "water_word": "Вода",
    },
    "th": {
        "name": "Thai",
        "prompt": (
            "หากตำแหน่งนี้อยู่บนบก ให้ตอบ 'บก' "
            "หากตำแหน่งนี้อยู่ในน้ำ ให้ตอบ 'น้ำ' "
            "ห้ามตอบอย่างอื่น\n"
            "{lat}, {lon}"
        ),
        "land_word": "บก",
        "water_word": "น้ำ",
    },
}

# Reference-primed prompt: anchors the model to a starting location.
# Tests whether the model's geographic accuracy changes when given spatial context.
ANCHORED_PROMPT_TEMPLATE = (
    "You are starting from {anchor_name} ({anchor_lat}, {anchor_lon}). "
    "If the following location is over land, say 'Land'. "
    "If it is over water, say 'Water'. "
    "Do not say anything else.\n"
    "{lat}, {lon}"
)

# Well-known anchor points for the reference-primed experiment
ANCHOR_POINTS = {
    "mannheim": ("Mannheim, Germany", "49.5° N", "8.5° E"),
    "tokyo": ("Tokyo, Japan", "35.7° N", "139.7° E"),
    "nyc": ("New York City, USA", "40.7° N", "74.0° W"),
    "sydney": ("Sydney, Australia", "33.9° S", "151.2° E"),
    "nairobi": ("Nairobi, Kenya", "1.3° S", "36.8° E"),
    "saopaulo": ("São Paulo, Brazil", "23.6° S", "46.6° W"),
}

# 20 geographically diverse cities for the multi-anchor experiment.
# Ordered so that every prefix of size 5/10/15/20 gives a good global spread.
MULTI_ANCHOR_CITIES = [
    # --- first 5: one per inhabited continent ---
    ("Mannheim, Germany",        "49.5° N",  "8.5° E"),
    ("Tokyo, Japan",             "35.7° N",  "139.7° E"),
    ("New York City, USA",       "40.7° N",  "74.0° W"),
    ("São Paulo, Brazil",        "23.5° S",  "46.6° W"),
    ("Sydney, Australia",        "33.9° S",  "151.2° E"),
    # --- next 5: fill Africa, South/Central Asia, Russia, Central America ---
    ("Cairo, Egypt",             "30.1° N",  "31.2° E"),
    ("Nairobi, Kenya",           "1.3° S",   "36.8° E"),
    ("Mumbai, India",            "19.1° N",  "72.9° E"),
    ("Moscow, Russia",           "55.8° N",  "37.6° E"),
    ("Mexico City, Mexico",      "19.4° N",  "99.1° W"),
    # --- next 5: East Asia, West Africa, southern S. America, NW Europe, Middle East ---
    ("Beijing, China",           "39.9° N",  "116.4° E"),
    ("Lagos, Nigeria",           "6.5° N",   "3.4° E"),
    ("Buenos Aires, Argentina",  "34.6° S",  "58.4° W"),
    ("London, United Kingdom",   "51.5° N",  "0.1° W"),
    ("Tehran, Iran",             "35.7° N",  "51.4° E"),
    # --- final 5: Canada, Southern Africa, Korea, SE Asia, Andean S. America ---
    ("Toronto, Canada",          "43.7° N",  "79.4° W"),
    ("Johannesburg, South Africa","26.2° S", "28.0° E"),
    ("Seoul, South Korea",       "37.6° N",  "127.0° E"),
    ("Jakarta, Indonesia",       "6.2° S",   "106.8° E"),
    ("Lima, Peru",               "12.1° S",  "77.0° W"),
]

# Prompt template for the multi-anchor experiment.
# Lists N known land locations as geographic context, then asks about the query point.
MULTI_ANCHORED_PROMPT_TEMPLATE = (
    "The following cities are all located on land:\n"
    "{anchor_lines}"
    "If the following location is over land, say 'Land'. "
    "If it is over water, say 'Water'. "
    "Do not say anything else.\n"
    "{lat}, {lon}"
)

# Few-shot prompt for base (non-instruct) models --> Base models are not that good with instructions
# The model just needs to continue the pattern with "Land" or "Water".
BASE_MODEL_PROMPT_TEMPLATE = (
    "For each coordinate, answer only Land or Water.\n\n"
    "45.0° N, 90.0° E\nLand\n\n"
    "0.0° N, 150.0° W\nWater\n\n"
    "30.0° S, 25.0° E\nLand\n\n"
    "60.0° N, 10.0° E\nWater\n\n"
    "35.0° N, 139.0° E\nLand\n\n"
    "0.0° N, 0.0° E\nWater\n\n"
    "{lat}, {lon}\n"
)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def format_lat(lat: float) -> str:
    """Format latitude as e.g. '45° N' or '30° S'."""
    if lat >= 0:
        return f"{abs(lat):.1f}° N"
    return f"{abs(lat):.1f}° S"


def format_lon(lon: float) -> str:
    """Format longitude as e.g. '90° E' or '120° W'."""
    if lon >= 0:
        return f"{abs(lon):.1f}° E"
    return f"{abs(lon):.1f}° W"


def generate_coordinates(resolution: float):
    """
    Generate (lat, lon) pairs covering the globe in a rectangular grid.

    Latitude:  90 --> -90  (top to bottom, so the image is right-side-up)
    Longitude: -180 --> 180  (left to right)

    Returns list of (row, col, lat, lon).
    """
    lats = np.arange(90, -90, -resolution)
    lons = np.arange(-180, 180, resolution)
    coords = []
    for r, lat in enumerate(lats):
        for c, lon in enumerate(lons):
            coords.append((r, c, float(lat), float(lon)))
    return coords, len(lats), len(lons)

"""Build the real land/water map to validate the sampled coordinates"""
def build_ground_truth_grid(coords, n_rows: int, n_cols: int) -> np.ndarray:   
    try:
        from global_land_mask import globe
    except ImportError as exc:
        raise ImportError(
        ) from exc

    ground_truth = np.zeros((n_rows, n_cols), dtype=float)
    for r, c, lat, lon in coords:
        ground_truth[r, c] = 1.0 if globe.is_land(lat, lon) else 0.0
    return ground_truth

"""Convert a model name into a filesystem-safe stem"""
def slugify_model_name(model: str) -> str:    
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._-")
    return slug or "model"

# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

"""Build the prompt for a given latitude and longitude"""
def build_prompt(lat: float, lon: float, is_base_model: bool = False,
                 anchor: Optional[tuple] = None,
                 anchor_list: Optional[list] = None,
                 language: Optional[Dict[str, str]] = None) -> str:
    if anchor_list is not None:
        lines = "".join(
            f"- {name} ({alat}, {alon})\n"
            for name, alat, alon in anchor_list
        )
        return MULTI_ANCHORED_PROMPT_TEMPLATE.format(
            anchor_lines=lines,
            lat=format_lat(lat), lon=format_lon(lon),
        )
    if anchor is not None:
        anchor_name, anchor_lat, anchor_lon = anchor
        return ANCHORED_PROMPT_TEMPLATE.format(
            anchor_name=anchor_name, anchor_lat=anchor_lat, anchor_lon=anchor_lon,
            lat=format_lat(lat), lon=format_lon(lon),
        )
    if language is not None and not is_base_model:
        template = language["prompt"]
        return template.format(lat=format_lat(lat), lon=format_lon(lon))
    if is_base_model:
        template = BASE_MODEL_PROMPT_TEMPLATE
    else:
        template = PROMPT_TEMPLATE
    return template.format(lat=format_lat(lat), lon=format_lon(lon))

def _token_matches(tok: str, target: str) -> bool:
    """Robust prefix match for one-token responses across languages.

    Tokenizers split words differently (e.g. 'Wasser' might come back as 'W',
    'Was', or 'Wasser'; 'Суша' as 'С' or 'Су'). We accept the token if either
    side is a non-empty prefix of the other after lower-casing.
    """
    t = tok.strip().lower()
    if not t:
        return False
    w = target.lower()
    return w.startswith(t) or t.startswith(w)


"""Extract P(Land) from a list of logprob token entries."""
def _extract_p_land_from_logprobs(logprob_entries, fallback_text: str,
                                  land_word: str = "Land",
                                  water_word: str = "Water") -> float:

    land_logprob = None
    water_logprob = None

    for entry in logprob_entries:

        if hasattr(entry, 'token'):
            tok = entry.token.strip()
        else:
            tok = str(entry).strip()

        if hasattr(entry, 'logprob'):
            lp = entry.logprob
        else:
            lp = 0.0

        if land_logprob is None and _token_matches(tok, land_word):
            land_logprob = lp
        if water_logprob is None and _token_matches(tok, water_word):
            water_logprob = lp

    if land_logprob is None and water_logprob is None:
        text = fallback_text.strip().lower()
        if land_word.lower() in text:
            return 1.0
        if water_word.lower() in text:
            return 0.0
        return 0.5

    if land_logprob is None:
        land_logprob = water_logprob - 20
    if water_logprob is None:
        water_logprob = land_logprob - 20

    # softmax with logprobs: exp(lp - max_lp) to prevent underflow, then normalize

    max_lp = max(land_logprob, water_logprob)
    exp_land = math.exp(land_logprob - max_lp)
    exp_water = math.exp(water_logprob - max_lp)
    return exp_land / (exp_land + exp_water)


# How many top logprobs to request from the completions endpoint
# vLLM caps this at 20 for both endpoints
COMPLETIONS_LOGPROBS = 20


def query_logprobs(client: OpenAI, model: str, lat: float, lon: float,
                   is_base_model: bool = False, anchor: Optional[tuple] = None,
                   anchor_list: Optional[list] = None,
                   language: Optional[Dict[str, str]] = None,
                   no_thinking: bool = False):
    """
    Send one coordinate to the model and return P(Land).

    For instruct/chat models: uses /v1/chat/completions with top_logprobs=20.
    For base models: uses /v1/completions with logprobs=20 and a few-shot
    prompt, which gives far better coverage of the token vocabulary.
    """
    prompt = build_prompt(lat, lon, is_base_model=is_base_model,
                          anchor=anchor, anchor_list=anchor_list,
                          language=language)
    if language is not None and not is_base_model:
        land_word = language["land_word"]
        water_word = language["water_word"]
    else:
        land_word, water_word = "Land", "Water"

    try:
        # Base models --> we use a few-shot example first
        if is_base_model:
            # Use the completions endpoint —> supports higher logprobs and
            # works much better for base models with the few-shot prompt
            response = client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=1, # only 1 token is needed (land or water)
                temperature=0, # deterministic 
                logprobs=COMPLETIONS_LOGPROBS,
            )
            choice = response.choices[0]
            fallback_text = choice.text or ""

            if choice.logprobs and choice.logprobs.top_logprobs:
                # top_logprobs is a list of dicts: [{token: logprob, ...}, ...]
                token_dict = choice.logprobs.top_logprobs[0]

                # Convert dict to list of objects for the shared extractor
                # need this for the _extract_p_land_from_logprobs 
                class _Entry:
                    __slots__ = ("token", "logprob")
                    def __init__(self, t, lp):
                        self.token = t
                        self.logprob = lp
                entries = [_Entry(t, lp) for t, lp in token_dict.items()]
                return _extract_p_land_from_logprobs(
                    entries, fallback_text,
                    land_word=land_word, water_word=water_word,
                )

            # No logprobs —-> fall back to text
            text = fallback_text.strip().lower()
            if land_word.lower() in text:
                return 1.0
            if water_word.lower() in text:
                return 0.0
            return 0.5

        else:
            # Instruct model: use chat completions (top_logprobs capped at 20)
            extra_body = None
            if no_thinking:
                # Qwen3 chat-template kwarg: skip the <think>...</think> block
                # so the very first generated token is already the final answer.
                extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1,
                temperature=0,
                logprobs=True,
                top_logprobs=20,
                extra_body=extra_body,
            )
            choice = response.choices[0]

            # No logprobs —> fall back to text
            if not choice.logprobs or not choice.logprobs.content:
                text = choice.message.content.strip().lower()
                if land_word.lower() in text:
                    return 1.0
                if water_word.lower() in text:
                    return 0.0
                return 0.5

            top = choice.logprobs.content[0].top_logprobs
            return _extract_p_land_from_logprobs(
                top, choice.message.content or "",
                land_word=land_word, water_word=water_word,
            )

    except Exception as e:
        print(f"  [warn] query failed for ({lat}, {lon}): {e}")
        return 0.5


# Strips a leading <think>...</think> block in case the vLLM reasoning parser
# is unavailable / not configured. With --reasoning-parser qwen3, the server
# already strips it, but the regex is a cheap safety net.
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Matches the explicit answer-tag we ask the model to emit.
_ANSWER_TAG_RE = re.compile(r"<answer>\s*(\S+)", re.IGNORECASE)

# Reasoning-mode prompt: instructs the model to wrap its final verdict in
# <answer>...</answer> tags so we can pair it with a stop=["</answer>"]
# sampling param. That stop sequence kills generation the moment the model
# commits, capping token cost regardless of how long it wanted to think.
REASONING_PROMPT_TEMPLATE = (
    "Decide whether the following coordinate is over land or over water. "
    "Think briefly. After thinking, give your final answer wrapped in tags, "
    "exactly like <answer>{land_word}</answer> or <answer>{water_word}</answer>.\n"
    "{lat}, {lon}"
)


def query_reasoning_samples(
    client: OpenAI,
    model: str,
    lat: float,
    lon: float,
    language: Optional[Dict[str, str]] = None,
    num_samples: int = 5,
    max_tokens: int = 1024,
    temperature: float = 0.7,
):
    """Multi-sample voting for reasoning models.

    Asks the model n=num_samples times to wrap its final answer in
    <answer>...</answer> tags. Generation stops at "</answer>", so the
    request finishes as soon as the model commits even if it would have
    kept thinking. Counts how many samples answered land vs water and
    returns P(Land). Falls back to 0.5 if no sample produced a parseable
    answer.
    """
    if language is not None:
        land_word = language["land_word"]
        water_word = language["water_word"]
    else:
        land_word, water_word = "Land", "Water"
    lw, ww = land_word.lower(), water_word.lower()

    prompt = REASONING_PROMPT_TEMPLATE.format(
        lat=format_lat(lat), lon=format_lon(lon),
        land_word=land_word, water_word=water_word,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            n=num_samples,
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
            stop=["</answer>"],
        )
        land_count = 0
        water_count = 0
        for choice in response.choices:
            text = choice.message.content or ""
            # Reasoning parser strips <think>...</think> server-side; this is a
            # belt-and-braces strip in case the parser is off.
            text = _THINK_TAG_RE.sub("", text)
            ans_token = None
            m = _ANSWER_TAG_RE.search(text)
            if m:
                ans_token = m.group(1).strip().lower()
            else:
                # Fallback: model didn't use the tag — search the post-think
                # text for the bare word.
                lower_text = text.strip().lower()
                li = lower_text.find(lw)
                wi = lower_text.find(ww)
                if li == -1 and wi == -1:
                    continue
                ans_token = lw if (li != -1 and (wi == -1 or li < wi)) else ww
            if _token_matches(ans_token, land_word):
                land_count += 1
            elif _token_matches(ans_token, water_word):
                water_count += 1
        total = land_count + water_count
        if total == 0:
            return 0.5
        return land_count / total
    except Exception as e:
        print(f"  [warn] reasoning query failed for ({lat}, {lon}): {e}")
        return 0.5


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

"""Write JSON with a stable indentation."""
def write_json(path: Path, payload: Dict[str, Any]):    
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

"""Compute the mean squared error against the ground-truth land mask."""
def compute_mse(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    return float(np.mean((predicted - ground_truth) ** 2))

"""Convert an OpenAI-compatible `/v1` URL into the server root URL."""
def get_server_root(base_url: str) -> str:   
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized[:-3]
    return normalized

"""Save the real land/water map once for the whole run."""
def save_ground_truth_assets(
    ground_truth: np.ndarray,
    resolution: float,
    output_dir: Path,
):
   
    output_dir.mkdir(parents=True, exist_ok=True)
    map_path = output_dir / GROUND_TRUTH_MAP_NAME
    data_path = output_dir / GROUND_TRUTH_DATA_NAME

    render_map(ground_truth, map_path, "Ground truth land / water", resolution)
    write_json(
        data_path,
        {
            "model": "ground_truth",
            "resolution_deg": resolution,
            "n_rows": int(ground_truth.shape[0]),
            "n_cols": int(ground_truth.shape[1]),
            "grid": ground_truth.tolist(),
        },
    )
    return map_path, data_path

"""Save JSON and CSV summaries of the MSE results."""
def write_mse_summaries(summary: Dict[str, Any], output_dir: Path):
   
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "mse_summary.json"
    csv_path = output_dir / "mse_summary.csv"

    write_json(json_path, summary)

    rows = summary.get("results", [])
    fieldnames = [
        "model",
        "slug",
        "status",
        "mse",
        "elapsed_sec",
        "resolution_deg",
        "map_path",
        "data_path",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"MSE summary saved to {json_path}")
    print(f"MSE CSV saved to {csv_path}")


# ---------------------------------------------------------------------------
# vLLM server helpers
# ---------------------------------------------------------------------------

def wait_for_server_ready(
    base_url: str,
    timeout_sec: int,
    process: Optional[subprocess.Popen] = None,
):
    """Wait until the OpenAI-compatible `/v1/models` endpoint responds."""
    models_url = base_url.rstrip("/") + "/models"
    deadline = time.time() + timeout_sec
    last_error = None

    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"vLLM server exited early with code {process.returncode}."
            )

        try:
            with urllib.request.urlopen(models_url, timeout=15) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc

        time.sleep(5)

    raise TimeoutError(
        f"Timed out waiting for vLLM server at {models_url}. Last error: {last_error}"
    )

"""Start a fresh vLLM server process for one model and wait until it is ready."""
def start_vllm_server_for_model(
    model: str,
    base_url: str,
    server_script: Path,
    timeout_sec: int,
    tensor_parallel_size: int = 1,
    reasoning_parser: Optional[str] = None,
    max_model_len: Optional[int] = None,
) -> subprocess.Popen:
    
    server_root = get_server_root(base_url)
    parsed = urllib.parse.urlparse(server_root)
    host = parsed.hostname or "0.0.0.0"
    port = parsed.port or 8000

    # Kill lingering vLLM processes from previous models
    print("Cleaning up any previous vLLM processes...")
    try:
        subprocess.run(
            ["pkill", "-f", "vllm.entrypoints.openai.api_server"],
            check=False,
            timeout=10,
        )
        time.sleep(3)  # Let GPU memory fully release
    except Exception as e:
        print(f"  (cleanup warning: {e})")

    # Short max sequence length since we only query "Land"/"Water" and coordinates.
    # 1024 is enough for the 20-anchor prompt (~400 tokens) plus output,
    # while still preventing vLLM from pre-allocating gigabytes of unused VRAM.
    # Reasoning models need much more headroom for the <think>...</think> block.
    if max_model_len is not None:
        max_len = max_model_len
    elif reasoning_parser is not None:
        max_len = 4096
    else:
        max_len = 1024
    
    # Set environment variables to reduce memory fragmentation and improve GPU efficiency
    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["VLLM_DISABLE_TRITON"] = "1"
    
    # Increase the timeout significantly for large models on network drives.
    # 7200s (2 h) gives enough headroom even when loading ~65 GB from ceph.
    env["VLLM_ENGINE_READY_TIMEOUT_S"] = "7200"
    
    command = [
        sys.executable,
        str(server_script),
        "--host",
        host,
        "--port",
        str(port),
        "--max-model-len",
        str(max_len),
    ]

    # Handle model revisions separated by '@'
    if "@" in model:
        model_name, revision = model.split("@", 1)
        command.extend(["--model", model_name, "--revision", revision])
    else:
        model_name = model
        command.extend(["--model", model])

    # Base models (no "Instruct"/"Chat" suffix) lack a chat template.
    # Supply a minimal one so vLLM's /chat/completions endpoint still works.
    if not _looks_like_chat_model(model_name):
        command.extend([
            "--chat-template",
            BASE_MODEL_CHAT_TEMPLATE,
        ])
    # Think/reasoning models inject <think> at the start of the assistant turn,
    # which breaks the max_tokens=1 logprobs approach.  Override with a plain
    # ChatML template so the model answers directly.
    elif _is_thinking_model(model_name):
        command.extend([
            "--chat-template",
            THINK_MODEL_CHAT_TEMPLATE,
        ])
    
    # Always pass tensor-parallel-size to override vllm_server.py default
    command.extend(["--tensor-parallel-size", str(tensor_parallel_size)])

    # Forward reasoning-parser to vLLM (after `--`, per vllm_server.py REMAINDER).
    # Splits <think>...</think> into reasoning_content and leaves the final
    # answer in choice.message.content, which is what we score.
    if reasoning_parser is not None:
        command.extend(["--", "--reasoning-parser", reasoning_parser])

    print("\n" + "=" * 80)
    print(f"Starting vLLM server for model: {model}")
    print(f"  Max sequence length: {max_len}")
    if tensor_parallel_size > 1:
        print(f"  Tensor parallelism: {tensor_parallel_size} GPUs")
    if reasoning_parser is not None:
        print(f"  Reasoning parser: {reasoning_parser}")
    print("=" * 80)
    process = subprocess.Popen(command, cwd=SCRIPT_DIR, env=env)

    try:
        wait_for_server_ready(base_url=base_url, timeout_sec=timeout_sec, process=process)
    except Exception:
        stop_vllm_server(process)
        raise

    return process


def stop_vllm_server(process: Optional[subprocess.Popen]):
    """Terminate a vLLM server process cleanly."""
    if process is None or process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    
    # Give GPU time to fully release memory
    time.sleep(5)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

"""Run the full experiment for one model and return its metrics."""
def run_experiment(
    resolution: float = DEFAULT_RESOLUTION,
    workers: int = DEFAULT_WORKERS,
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    output: str = "blind_earth_map.png",
    save_data: bool = True,
    base_url: str = VLLM_BASE_URL,
    coords=None,
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
    ground_truth: Optional[np.ndarray] = None,
    anchor: Optional[tuple] = None,
    anchor_list: Optional[list] = None,
    language: Optional[Dict[str, str]] = None,
    is_reasoning: bool = False,
    num_samples: int = 5,
    no_thinking: bool = False,
):

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if coords is None or n_rows is None or n_cols is None:
        coords, n_rows, n_cols = generate_coordinates(resolution)
    if ground_truth is None:
        ground_truth = build_ground_truth_grid(coords, n_rows, n_cols)

    total = len(coords)

    # Detect whether this is a base model (needs completions endpoint)
    # Strip revision suffix for the check
    if "@" in model:
        model_base = model.split("@")[0]
    else:
        model_base = model

    # Reasoning models are always treated as chat models (the thinking-mode
    # chat template is applied by vLLM via --reasoning-parser).
    is_base = not _looks_like_chat_model(model_base) and not is_reasoning

    print(f"Model:      {model}")
    if is_reasoning:
        endpoint_label = f"chat/completions (reasoning, n={num_samples} samples per query)"
    elif is_base:
        endpoint_label = f"completions (base model, logprobs={COMPLETIONS_LOGPROBS})"
    else:
        endpoint_label = "chat/completions (instruct, top_logprobs=20)"
    print(f"Endpoint:   {endpoint_label}")
    if language is not None:
        print(f"Language:   {language['name']} (land='{language['land_word']}', water='{language['water_word']}')")
    if anchor_list is not None:
        print(f"Anchors:    {len(anchor_list)} cities ({', '.join(a[0] for a in anchor_list[:3])}...)")
    elif anchor:
        print(f"Anchor:     {anchor[0]} ({anchor[1]}, {anchor[2]})")
    print(f"Resolution: {resolution}° per pixel  --> {n_cols}×{n_rows} image  ({total} queries)")
    print(f"Workers:    {workers}")
    print()

    grid = np.full((n_rows, n_cols), 0.5)
    client = OpenAI(api_key=API_KEY, base_url=base_url)

    try:
        models_list = client.models.list()
        available = [m.id for m in models_list.data]
        print(f"Server models: {available}")
        if model not in available:
            print(f"WARNING: requested model '{model}' not in server model list!")
    except Exception as e:
        raise RuntimeError(f"Could not reach vLLM server at {base_url}: {e}") from e

    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for r, c, lat, lon in coords:
            if is_reasoning:
                fut = pool.submit(query_reasoning_samples, client, model,
                                  lat, lon, language, num_samples)
            else:
                fut = pool.submit(query_logprobs, client, model, lat, lon,
                                  is_base_model=is_base, anchor=anchor,
                                  anchor_list=anchor_list, language=language,
                                  no_thinking=no_thinking)
            futures[fut] = (r, c)

        with tqdm(total=total, desc="Querying model", unit="px") as pbar:
            for fut in as_completed(futures):
                r, c = futures[fut]
                try:
                    grid[r, c] = fut.result()
                except Exception as e:
                    print(f"  [error] pixel ({r},{c}): {e}")
                pbar.update(1)

    elapsed = time.time() - t0
    if elapsed > 0:
        qps = total / elapsed 
    else: 
        qps = 0
        
    mse = compute_mse(grid, ground_truth)
    print(f"\nDone in {elapsed:.1f}s  ({qps:.1f} queries/sec)")
    print(f"MSE vs ground truth: {mse:.6f}")

    data_path = output_path.with_name(f"{output_path.stem}_data.json")
    if save_data:
        write_json(
            data_path,
            {
                "model": model,
                "resolution_deg": resolution,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "elapsed_sec": round(elapsed, 2),
                "mse": mse,
                "grid": grid.tolist(),
            },
        )
        print(f"Raw data saved to {data_path}")

    title_model = f"{model} [{language['name']}]" if language is not None else model
    render_map(grid, output_path, title_model, resolution)

    return {
        "model": model,
        "slug": slugify_model_name(model),
        "status": "completed",
        "mse": mse,
        "elapsed_sec": round(elapsed, 2),
        "resolution_deg": resolution,
        "map_path": str(output_path),
        "data_path": str(data_path) if save_data else "",
        "error": "",
    }


def run_models_sequentially(
    models: Sequence[str],
    resolution: float,
    workers: int,
    base_url: str,
    save_data: bool,
    generated_models_dir: Path,
    mse_dir: Path,
    server_script: Path,
    server_startup_timeout: int,
    start_server_per_model: bool,
    tensor_parallel_size: int = 1,
    anchor: Optional[tuple] = None,
    anchor_list: Optional[list] = None,
    languages: Optional[Sequence[Dict[str, str]]] = None,
    language_codes: Optional[Sequence[str]] = None,
    is_reasoning: bool = False,
    num_samples: int = 5,
    reasoning_parser: Optional[str] = None,
    max_model_len: Optional[int] = None,
    no_thinking: bool = False,
):
    """Run multiple models one after another and write map/MSE artifacts."""
    coords, n_rows, n_cols = generate_coordinates(resolution)
    ground_truth = build_ground_truth_grid(coords, n_rows, n_cols)
    ground_truth_map_path, ground_truth_data_path = save_ground_truth_assets(
        ground_truth=ground_truth,
        resolution=resolution,
        output_dir=generated_models_dir,
    )

    # Build the per-model language list. If no languages are given, we run
    # once with language=None (back-compat with the original English prompt).
    if languages:
        language_runs = list(zip(language_codes or [None] * len(languages), languages))
    else:
        language_runs = [(None, None)]

    results = []
    for model in models:
        server_process = None
        slug = slugify_model_name(model)
        if anchor_list is not None:
            anchor_suffix = f"_anchors{len(anchor_list)}"
        elif anchor:
            anchor_suffix = f"_from_{anchor[0].split(',')[0].lower().replace(' ', '')}"
        else:
            anchor_suffix = ""
        reasoning_suffix = f"_reasoning_n{num_samples}" if is_reasoning else ""

        try:
            if start_server_per_model:
                server_process = start_vllm_server_for_model(
                    model=model,
                    base_url=base_url,
                    server_script=server_script,
                    timeout_sec=server_startup_timeout,
                    tensor_parallel_size=tensor_parallel_size,
                    reasoning_parser=reasoning_parser,
                    max_model_len=max_model_len,
                )

            for lang_code, language in language_runs:
                lang_suffix = f"_lang_{lang_code}" if lang_code else ""
                output_path = generated_models_dir / f"{slug}{anchor_suffix}{reasoning_suffix}{lang_suffix}.png"
                try:
                    result = run_experiment(
                        resolution=resolution,
                        workers=workers,
                        model=model,
                        output=str(output_path),
                        save_data=save_data,
                        base_url=base_url,
                        coords=coords,
                        n_rows=n_rows,
                        n_cols=n_cols,
                        ground_truth=ground_truth,
                        anchor=anchor,
                        anchor_list=anchor_list,
                        language=language,
                        is_reasoning=is_reasoning,
                        num_samples=num_samples,
                        no_thinking=no_thinking,
                    )
                    if lang_code:
                        result["language"] = lang_code
                        result["slug"] = f"{slug}{reasoning_suffix}_lang_{lang_code}"
                    elif reasoning_suffix:
                        result["slug"] = f"{slug}{reasoning_suffix}"
                    if is_reasoning:
                        result["reasoning"] = True
                        result["num_samples"] = num_samples
                except Exception as exc:
                    print(f"[error] Failed to evaluate {model} ({lang_code}): {exc}")
                    result = {
                        "model": model,
                        "slug": f"{slug}{reasoning_suffix}{lang_suffix}",
                        "status": "failed",
                        "mse": "",
                        "elapsed_sec": "",
                        "resolution_deg": resolution,
                        "map_path": "",
                        "data_path": "",
                        "error": str(exc),
                        "language": lang_code or "",
                    }
                results.append(result)
        except KeyboardInterrupt:
            stop_vllm_server(server_process)
            raise
        except Exception as exc:
            print(f"[error] Failed to start server for {model}: {exc}")
            results.append({
                "model": model,
                "slug": slug,
                "status": "failed",
                "mse": "",
                "elapsed_sec": "",
                "resolution_deg": resolution,
                "map_path": "",
                "data_path": "",
                "error": str(exc),
            })
        finally:
            stop_vllm_server(server_process)

    summary = {
        "resolution_deg": resolution,
        "ground_truth_map_path": str(ground_truth_map_path),
        "ground_truth_data_path": str(ground_truth_data_path),
        "results": results,
    }
    write_mse_summaries(summary, mse_dir)
    return summary


def render_map(grid: np.ndarray, output: Path, model: str, resolution: float):
    """Render the P(Land) grid as a color map and save to disk."""
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "land_water",
        ["#1a5276", "#2e86c1", "#aed6f1", "#a9dfbf", "#27ae60", "#1e8449"],
    )

    n_rows, n_cols = grid.shape
    fig_w = max(12, n_cols / 15)
    fig_h = max(6, n_rows / 15)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(
        grid,
        cmap=cmap,
        vmin=0,
        vmax=1,
        aspect="equal",
        interpolation="nearest",
        extent=[-180, 180, -90, 90],
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Blind Earth Map — {model}  ({resolution}° resolution)")
    fig.colorbar(im, ax=ax, label="P(Land)", shrink=0.7)

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Map saved to {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="How Does A Blind Model See The Earth? — LLM geographic knowledge probe"
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=DEFAULT_RESOLUTION,
        help=(
            f"Grid resolution in degrees (default: {DEFAULT_RESOLUTION}). "
            "Lower = higher fidelity but more queries (quadratic)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel request workers (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Single model name served by vLLM.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Explicit list of model names to run sequentially.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run the built-in tested-model list one after another.",
    )
    parser.add_argument(
        "--student-models",
        action="store_true",
        help="Run only models that fit on 48GB GPUs (≤8B) and don't require authentication.",
    )
    parser.add_argument(
        "--alt-models",
        action="store_true",
        help="Run alternative lightweight open-source models (mix of providers: TinyLlama, Phi, Mistral, Yi).",
    )
    parser.add_argument(
        "--diverse-models",
        action="store_true",
        help=(
            "Run 8 diverse open-source models from different families (7B–32B): "
            "Gemma-2 9B/27B, Phi-4 14B, Mistral-Nemo 12B, Qwen2.5 32B, OLMo-3 7B, "
            "Falcon3 10B, Granite 8B. All fit on 2×48 GB GPUs. "
            "Gated models (Gemma) require HF_TOKEN."
        ),
    )
    parser.add_argument(
        "--diverse-models-v2",
        action="store_true",
        help=(
            "Run second batch of open-source reasoning models "
            "(DeepSeek-R1-Distill 32B/14B, QwQ-32B). All fit on 2×48 GB GPUs. "
            "No API keys or gated access required."
        ),
    )
    parser.add_argument(
        "--test-olmo3-evolution",
        action="store_true",
        help="Run different training checkpoints of the olmo3 7B model to see how map changes.",
    )
    parser.add_argument(
        "--test-olmo3-32b-evolution",
        action="store_true",
        help="Run different training checkpoints of the olmo3 32B model to see how map changes.",
    )
    parser.add_argument(
        "--list-tested-models",
        action="store_true",
        help="Print the built-in tested-model list and exit.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="blind_earth_map.png",
        help="Output image filename for a single-model run.",
    )
    parser.add_argument(
        "--generated-models-dir",
        type=Path,
        default=DEFAULT_GENERATED_MODELS_DIR,
        help="Directory where generated maps are saved.",
    )
    parser.add_argument(
        "--mses-dir",
        type=Path,
        default=DEFAULT_MSE_DIR,
        help="Directory where MSE summary files are saved.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=VLLM_BASE_URL,
        help="vLLM server base URL.",
    )
    parser.add_argument(
        "--server-script",
        type=Path,
        default=DEFAULT_SERVER_SCRIPT,
        help="Path to the vLLM server launcher script.",
    )
    parser.add_argument(
        "--server-startup-timeout",
        type=int,
        default=DEFAULT_SERVER_STARTUP_TIMEOUT,
        help="Seconds to wait for each model server to come up.",
    )
    parser.add_argument(
        "--reuse-existing-server",
        action="store_true",
        help="Do not restart vLLM for each model; use the already running server.",
    )
    parser.add_argument(
        "--no-save-data",
        action="store_true",
        help="Skip saving the raw JSON data.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Number of GPUs to use for tensor parallelism (splits large models across multiple GPUs). Default: 2.",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default=None,
        help=(
            "Anchor the prompt to a starting location. This adds spatial context "
            "to test if the model's accuracy changes relative to a reference point. "
            f"Built-in anchors: {', '.join(ANCHOR_POINTS.keys())}. "
            "Or use custom format: 'Name,lat,lon' e.g. 'Berlin,52.5° N,13.4° E'"
        ),
    )
    parser.add_argument(
        "--anchor-count",
        type=int,
        default=None,
        choices=[5, 10, 15, 20],
        help=(
            "Number of geographic anchor cities to include in the prompt (5, 10, 15, or 20). "
            "Uses the first N cities from MULTI_ANCHOR_CITIES. "
            "Output file will be named <model>_anchorsN.png."
        ),
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=sorted(LANGUAGE_CONFIGS.keys()),
        help=(
            "Run the prompt in a single non-English language. "
            "Translates both the question and the expected answer token. "
            "Output file gets a _lang_<code> suffix."
        ),
    )
    parser.add_argument(
        "--all-languages",
        action="store_true",
        help=(
            "Run the prompt once per language in LANGUAGE_CONFIGS "
            "(en/de/es/zh/ru) for each selected model, reusing the same vLLM "
            "server. Use this to compare geographic accuracy across languages."
        ),
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help=(
            "Treat the model as a reasoning model (e.g. Qwen3, DeepSeek-R1). "
            "Skips logprobs; instead samples n=--num-samples completions per "
            "coordinate and computes P(Land) = land_count / (land + water). "
            "Requires --reasoning-parser to strip <think>...</think> blocks "
            "server-side."
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples per coordinate when --reasoning is set (default: 5).",
    )
    parser.add_argument(
        "--reasoning-parser",
        type=str,
        default=None,
        help=(
            "vLLM reasoning parser name to forward to the server (e.g. 'qwen3', "
            "'deepseek_r1'). Splits <think>...</think> from the final answer."
        ),
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override the vLLM max-model-len. Defaults to 1024, or 4096 when --reasoning-parser is set.",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help=(
            "For Qwen3 chat models: disable thinking via "
            "chat_template_kwargs={'enable_thinking': False}, so the very "
            "first generated token is the final answer (compatible with the "
            "max_tokens=1 + logprobs path used by all non-reasoning models)."
        ),
    )
    args = parser.parse_args()

    if args.list_tested_models:
        for model_name in TESTED_MODELS:
            print(model_name)
        return

    # Parse multi-anchor-count argument
    anchor_list = None
    if args.anchor_count is not None:
        anchor_list = MULTI_ANCHOR_CITIES[:args.anchor_count]
        print(f"Using {args.anchor_count} geographic anchors: {', '.join(a[0] for a in anchor_list)}")

    # Parse anchor argument
    anchor = None
    if args.anchor:
        key = args.anchor.lower().strip()
        if key in ANCHOR_POINTS:
            anchor = ANCHOR_POINTS[key]
            print(f"Using built-in anchor: {anchor[0]} ({anchor[1]}, {anchor[2]})")
        elif "," in args.anchor:
            parts = [p.strip() for p in args.anchor.split(",", 2)]
            if len(parts) == 3:
                anchor = (parts[0], parts[1], parts[2])
                print(f"Using custom anchor: {anchor[0]} ({anchor[1]}, {anchor[2]})")
            else:
                print(f"ERROR: Custom anchor must be 'Name,lat,lon'. Got: {args.anchor}")
                return
        else:
            print(f"ERROR: Unknown anchor '{args.anchor}'. Available: {', '.join(ANCHOR_POINTS.keys())}")
            return

    selected_models = args.models
    if args.all_models:
        selected_models = TESTED_MODELS
    elif args.student_models:
        selected_models = STUDENT_TESTED_MODELS
    elif args.alt_models:
        selected_models = ALT_STUDENT_TESTED_MODELS
    elif args.diverse_models:
        selected_models = DIVERSE_OS_MODELS
    elif args.diverse_models_v2:
        selected_models = DIVERSE_OS_MODELS_V2
    elif args.test_olmo3_evolution:
        print("Testing geographic knowledge evolution across olmo3 checkpoints...")
        selected_models = OLMO3_CHECKPOINTS
    elif args.test_olmo3_32b_evolution:
        print("Testing geographic knowledge evolution across olmo3 32B checkpoints...")
        selected_models = OLMO3_32B_CHECKPOINTS
    elif not selected_models:
        # Single model mode: treat as a list with one model
        selected_models = [args.model]

    # Resolve language selection
    language_codes = None
    languages = None
    if args.all_languages:
        language_codes = list(LANGUAGE_CONFIGS.keys())
        languages = [LANGUAGE_CONFIGS[c] for c in language_codes]
        print(f"Running across {len(languages)} languages: {', '.join(language_codes)}")
    elif args.language:
        language_codes = [args.language]
        languages = [LANGUAGE_CONFIGS[args.language]]
        print(f"Using language: {args.language} ({LANGUAGE_CONFIGS[args.language]['name']})")

    run_models_sequentially(
        models=selected_models,
        resolution=args.resolution,
        workers=args.workers,
        base_url=args.base_url,
        save_data=not args.no_save_data,
        generated_models_dir=args.generated_models_dir,
        mse_dir=args.mses_dir,
        server_script=args.server_script,
        server_startup_timeout=args.server_startup_timeout,
        start_server_per_model=not args.reuse_existing_server,
        tensor_parallel_size=args.tensor_parallel_size,
        anchor=anchor,
        anchor_list=anchor_list,
        languages=languages,
        language_codes=language_codes,
        is_reasoning=args.reasoning,
        num_samples=args.num_samples,
        reasoning_parser=args.reasoning_parser,
        max_model_len=args.max_model_len,
        no_thinking=args.no_thinking,
    )


if __name__ == "__main__":
    main()
