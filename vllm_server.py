#!/usr/bin/env python3
"""
vLLM OpenAI-compatible API Server
Serves a language model via http://localhost:8000

Examples:
  python vllm_server.py --model Qwen/Qwen2.5-7B-Instruct
  python vllm_server.py --model meta-llama/Llama-3.1-8B-Instruct --port 8001
"""

import argparse
import os
import subprocess
import sys

try:
    import vllm  
except ImportError:
    sys.exit(
        "ERROR: vllm is not installed in this Python interpreter. "
        "Activate the correct environment or install it with "
        "`pip install vllm` before running this script."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Start a vLLM OpenAI-compatible API server for one model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model identifier or local path to serve.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Specific model revision/branch to load from HuggingFace.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host interface to bind.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="vLLM dtype value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed passed to vLLM.",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="Optional served model name exposed by the OpenAI API.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional maximum model length override.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="vLLM GPU memory utilization setting (default: 0.95).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Number of GPUs for tensor parallelism (splits large models across multiple GPUs).",
    )
    parser.add_argument(
        "--swap-space",
        type=int,
        default=0,
        help="CPU swap space size (GiB) per GPU. Set to 0 to save CPU RAM.",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help="Jinja2 chat template string or path for base models without one.",
    )
    parser.add_argument(
        "vllm_args",
        nargs=argparse.REMAINDER,
        help="Extra raw arguments forwarded to vLLM after '--'.",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1" # massivly speed up weight downloads
    env["VLLM_DISABLE_TRITON"] = "1"
    env["TORCH_COMPILE_DEBUG"] = "0"
    env["TORCH_COMPILE_DISABLE"] = "1"
    env["TORCH_DYNAMO_DISABLE"] = "1"
    env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    # Set a much longer timeout for the engine processes to start because loading 32B weights is extremely slow on network drives.
    env["VLLM_ENGINE_READY_TIMEOUT_S"] = "7200" # 2 hours for slow ceph network drives
    # Optimize memory for many short independent queries
    env["VLLM_ALLOW_LONG_MAX_POSITION_EMBEDDINGS"] = "1"
    env["VLLM_ENABLE_PREFIX_CACHING"] = "1"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn" # prevent memory duplication in workers

    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--enable-prefix-caching",
        "--enable-chunked-prefill",        # Allows better memory packing during prompt processing
        "--distributed-executor-backend", "mp", # avoids heavy ray overhead on 1 node
        "--max-num-seqs", "128",            # Prevent excessive batching overhead
        "--max-num-batched-tokens", "4096", # Reduce processing spike memory
        "--seed",
        str(args.seed),
        "--trust-remote-code",
    ]

    if args.revision:
        command.extend(["--revision", args.revision])
        # If loading a specific revision, prepend it to the served model name 
        # so it doesn't conflict with other runs of the same base repo
        served_name = args.served_model_name or f"{args.model}@{args.revision}"
        command.extend(["--served-model-name", served_name])
    elif args.served_model_name:
        command.extend(["--served-model-name", args.served_model_name])
        
    if args.max_model_len is not None:
        command.extend(["--max-model-len", str(args.max_model_len)])
    # Always let vLLM manage its own gpu-memory-utilization
    command.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    # Offload KV cache block to CPU memory to allow larger model weights natively in GPU
    command.extend(["--swap-space", str(args.swap_space)])
    # Add tensor-parallel-size if > 1. Custom all-reduce is disabled because
    # CUDA graphs + custom all-reduce crashes on this cluster's GPU pair
    # (`Failed: Cuda error custom_all_reduce.cuh:455 'invalid argument'`).
    # --enforce-eager is forced on for TP because torch.compile + multi-graph
    # capture takes >2 h on a 24 B+ model split across 2 GPUs and the server
    # startup wait times out before the engine reports ready. For TP=1 we
    # leave CUDA graphs on (compilation is cheap on small models).
    if args.tensor_parallel_size > 1:
        command.extend([
            "--tensor-parallel-size", str(args.tensor_parallel_size),
            "--disable-custom-all-reduce",
            "--enforce-eager",
        ])

    if args.chat_template:
        command.extend(["--chat-template", args.chat_template])

    extra_args = list(args.vllm_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    command.extend(extra_args)

    print("Starting vLLM OpenAI-compatible API server...")
    print(f"Model: {args.model}")
    print(f"URL: http://{args.host}:{args.port}")
    print("\nPress Ctrl+C to stop the server.")

    subprocess.run(command, env=env, check=False)


if __name__ == "__main__":
    main()



