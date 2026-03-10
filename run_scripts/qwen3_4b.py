#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vLLM inference for Qwen3-2.5B

Usage:
  python infer_qwen3_2_5b_vllm.py --prompt "你好，介绍一下你自己"
  python infer_qwen3_2_5b_vllm.py --prompt "写一段C++多线程示例" --max-tokens 256
"""

import argparse
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="HF model id or local path",
    )
    parser.add_argument("--prompt", type=str, default="你好，介绍一下你自己")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Multi-GPU tensor parallel size",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "half", "bfloat16", "float16", "float32"],
        help="Model dtype",
    )
    args = parser.parse_args()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        trust_remote_code=True,
    )

    outputs = llm.generate([args.prompt], sampling_params)
    text = outputs[0].outputs[0].text

    print("=== PROMPT ===")
    print(args.prompt)
    print("\n=== OUTPUT ===")
    print(text)


if __name__ == "__main__":
    main()