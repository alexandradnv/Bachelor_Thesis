# How Does a Blind Model See the Earth?

Code and results for my Bachelor's thesis. The project recreates and extends the
["How Does A Blind Model See The Earth?"](https://www.lesswrong.com/posts/xwdRzJxyqFqgXTWbH)
experiment: for every point on a latitude/longitude grid, a large language model
(LLM) is asked whether that coordinate is over **land** or **water**. The model's
confidence is read from token log-probabilities and rendered as an
equirectangular world map. Comparing that map against a real land/water ground
truth gives a single quality score — the mean squared error (MSE) — that lets us
rank models and experimental conditions.

Models are served locally with [vLLM](https://github.com/vllm-project/vllm)
through an OpenAI-compatible API, so the same code works across dozens of open
models (Qwen, Gemma/BgGPT, Mistral, Yi, OLMo, Mixtral, Phi, TinyLlama, …).

## Experiments

| Experiment | Script | Question |
|---|---|---|
| **Baseline / model size** | `blind_model_experiment.py` | How well does each model reconstruct the map, and does accuracy scale with parameter count? |
| **Anchored few-shot** | `blind_model_experiment.py` (anchor options) | Does giving the model a few known reference coordinates ("anchors") improve the map? |
| **Language** | `blind_model_experiment.py` (language options) | Does the prompt language (en/de/bg/ru/zh/…) change the geography the model "sees"? |
| **Reasoning** | `blind_model_experiment.py` (reasoning models) | Do reasoning models (e.g. Qwen3) draw better maps, and how does it scale with the number of anchors? |
| **Self-memory** | `memory_experiment.py` | If the model is shown its own previous *N* predictions, does the map get better or worse? |
| **Ground-truth memory** | `fewshot_groundtruth_experiment.py` | Controlled counterpart to self-memory: same prompt format, but the in-context examples are *correctly* labelled. Isolates whether the memory effect is about prediction quality or prompt format. |

Post-hoc analysis (no model inference required):

- **`analyze_geographic_bias.py`** — per-region MSE (latitude bands & hemispheres)
  and calibration curves (with Brier score and expected calibration error) for
  selected runs.
- **`compute_all_mses.py`** — rescores every saved map against the ground truth
  and writes a consolidated ranking to `MSEs/all_mses.{csv,json}`.
- **`generate_thesis_graphs.py`** — produces the publication-quality figures
  (PNG + PDF) used in the thesis from the MSE summaries.

## Repository layout

```
blind_model_experiment.py       Main experiment: map generation, ground truth, MSE
memory_experiment.py            Self-memory experiment
fewshot_groundtruth_experiment.py  Ground-truth-memory control experiment
analyze_geographic_bias.py      Per-region MSE + calibration analysis
compute_all_mses.py             Rescore all runs -> MSEs/all_mses.*
generate_thesis_graphs.py       Build thesis figures from MSE data
organize_generated_models.py    Sort raw outputs into per-experiment folders
vllm_server.py                  Start a vLLM OpenAI-compatible server for one model

Batch Jobs/                     Shell scripts (Slurm-style) for each model/run
Generated models/               Output maps (.png) + raw P(Land) grids (_data.json),
                                sorted into per-experiment sub-folders
MSEs/                           MSE summaries per experiment + thesis_graphs/
analysis/                       Geographic-bias and calibration figures
```

The `Generated models/` outputs are grouped into `normal_experiment/`,
`anchored_experiment/`, `language_experiment/`, `reasoning_experiment/`,
`ground_truth_memory_experiment/`, and `inconclusive_experiments/`. The shared
`ground_truth_map.{png,json}` stays at the root. Run `organize_generated_models.py`
(it is idempotent) to re-sort after new runs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install vllm openai numpy matplotlib tqdm global-land-mask
```

A CUDA-capable GPU is required to serve models with vLLM. Most experiments here
were run on the larger AWQ/GPTQ-quantised models on a single high-memory GPU.

## Running an experiment

1. **Serve a model** (in one terminal):

   ```bash
   python vllm_server.py --model Qwen/Qwen2.5-7B-Instruct
   ```

2. **Run the experiment** against the local server (in another terminal):

   ```bash
   python blind_model_experiment.py --model Qwen/Qwen2.5-7B-Instruct
   ```

   This writes `<model>.png` and `<model>_data.json` into `Generated models/`
   and an MSE summary into `MSEs/`. See `--help` on each script for anchor,
   language, reasoning, and resolution options.

The `Batch Jobs/` folder contains ready-to-submit scripts for every model and
condition used in the thesis (e.g. `run_qwen32b_awq_anchors_experiment.sh`,
`run_gemma3_27b_languages_experiment.sh`, `run_memory_experiment.sh`).

## Reproducing the thesis figures

```bash
python compute_all_mses.py        # rescore every saved run
python generate_thesis_graphs.py  # build PNG + PDF figures in MSEs/thesis_graphs/
python analyze_geographic_bias.py # per-region MSE + calibration in analysis/
```

## Credit

The original "blind model" idea and methodology come from Henry / the LessWrong
post linked above. This repository adapts and extends it for the thesis with
additional models, anchoring, multilingual, reasoning, and memory experiments.
