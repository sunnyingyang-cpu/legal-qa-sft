#!/usr/bin/env python3
"""GGUF 推理压测（阶段 5）：TTFT / 解码吞吐 / P95 端到端延迟（对应文章第 10 节）。

依赖：pip install llama-cpp-python
用法：
    python deploy/benchmark_latency.py --model models/qwen2.5-1.5b-legal-q8_0.gguf \
        --data data/legal_qa_test.json -n 50 --n-gpu-layers -1

文章要求 P95 端到端 < 2s，1.5B 的 Q8_0 在单卡/中端 CPU 上均可轻松达标。
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

SYSTEM = "你是一名专业的中国法律咨询助手，回答需引用具体法条。"


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def main() -> None:
    ap = argparse.ArgumentParser(description="GGUF 推理压测")
    ap.add_argument("--model", required=True, help="GGUF 文件路径")
    ap.add_argument("--data", default="data/legal_qa_test.json")
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--n-gpu-layers", type=int, default=-1,
                    help="-1 全部层放 GPU，0 仅 CPU")
    args = ap.parse_args()

    from llama_cpp import Llama

    with open(args.data, encoding="utf-8") as f:
        questions = [s["instruction"] for s in json.load(f)[: args.n]]

    print(f"加载 {args.model} ...")
    llm = Llama(model_path=args.model, n_ctx=2048,
                n_gpu_layers=args.n_gpu_layers, verbose=False)

    ttfts, e2es, tps = [], [], []
    for i, q in enumerate(questions, 1):
        t0 = time.perf_counter()
        ttft, n_tok = None, 0
        stream = llm.create_chat_completion(
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": q}],
            max_tokens=args.max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n_tok += 1
        total = time.perf_counter() - t0
        if ttft is None or n_tok == 0:
            continue
        ttfts.append(ttft)
        e2es.append(total)
        tps.append(n_tok / max(total - ttft, 1e-6))
        print(f"[{i}/{len(questions)}] ttft={ttft:.2f}s  e2e={total:.2f}s  {tps[-1]:.1f} tok/s")

    summary = {
        "n": len(ttfts),
        "ttft_mean_s": round(statistics.mean(ttfts), 3),
        "ttft_p95_s": round(pct(ttfts, 0.95), 3),
        "e2e_mean_s": round(statistics.mean(e2es), 3),
        "e2e_p95_s": round(pct(e2es, 0.95), 3),
        "decode_tok_s_mean": round(statistics.mean(tps), 1),
    }
    print("\n===== 汇总 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    verdict = "达标" if summary["e2e_p95_s"] < 2 else "未达标"
    print(f"文章要求 P95 < 2s：{verdict}")


if __name__ == "__main__":
    main()
