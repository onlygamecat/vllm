#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", type=str, default="你好，介绍一下你自己")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=None)  # 改为 None：不固定随机性
    parser.add_argument("--n", type=int, default=5, help="number of inferences")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "half", "bfloat16", "float16", "float32"],
    )
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        trust_remote_code=True,
    )

    print("=== PROMPT ===")
    print(args.prompt)

    for i in range(args.n):
        # 如果你想“每次不同但可复现”，可以用 seed=(args.seed + i) 且 args.seed 不为 None
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=None if args.seed is None else (args.seed + i),
        )

        outputs = llm.generate([args.prompt], sampling_params)
        text = outputs[0].outputs[0].text

        print(f"\n=== OUTPUT #{i+1} ===")
        print(text)


if __name__ == "__main__":
    main()