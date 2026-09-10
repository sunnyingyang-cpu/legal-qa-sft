#!/usr/bin/env python3
"""评测脚本（阶段 4）：ROUGE-L / BLEU / （可选）BERTScore / 引用准确率。

输入：LLaMA-Factory do_predict 产出的 generated_predictions.jsonl，
     每行形如 {"predict": "...", "label": "..."}。

引用准确率把原文的单一 Citation Accuracy 拆成两层：
    format_rate —— 参考答案带引用时，预测也给出引用的比例（格式合规）
    precision / recall / exact_match —— 预测引用与参考引用（按法条集合）的重合，事实正确

用法：
    python eval/metrics.py --pred results/ft-1.5b/generated_predictions.jsonl \
        --bertscore --out eval_results/ft-1.5b.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

CITE_RE = re.compile(r"《[^》]{2,30}》第[零一二三四五六七八九十百千万〇0-9]+条")


def char_split(text: str) -> str:
    """中文按字切分，供 rouge/sacrebleu 使用。"""
    return " ".join(list(text.strip()))


def extract_cites(text: str) -> set:
    return {m.group(0) for m in CITE_RE.finditer(text)}


def load_pairs(path: str):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            p, r = (d.get("predict") or "").strip(), (d.get("label") or "").strip()
            if r:
                pairs.append((p, r))
    return pairs


def rouge_l(preds, refs) -> float:
    from rouge_chinese import Rouge

    rouge = Rouge()
    scores = []
    for p, r in zip(preds, refs):
        try:
            scores.append(rouge.get_scores([char_split(p)], [char_split(r)])[0]["rouge-l"]["f"])
        except Exception:  # 空预测等边界情况记 0
            scores.append(0.0)
    return statistics.mean(scores)


def bleu(preds, refs) -> float:
    from sacrebleu.metrics import BLEU

    return BLEU(tokenize="char").corpus_score(list(preds), [list(refs)]).score


def bert_score(preds, refs) -> float:
    from bert_score import score

    _, _, f1 = score(list(preds), list(refs), lang="zh", verbose=False)
    return f1.mean().item()


def citation_metrics(preds, refs) -> dict:
    n_ref_has, n_fmt, exact = 0, 0, 0
    precs, recs = [], []
    for p, r in zip(preds, refs):
        pc, rc = extract_cites(p), extract_cites(r)
        if not rc:
            continue
        n_ref_has += 1
        if pc:
            n_fmt += 1
            precs.append(len(pc & rc) / len(pc))
        else:
            precs.append(0.0)
        recs.append(len(pc & rc) / len(rc))
        if pc == rc:
            exact += 1
    n = max(n_ref_has, 1)
    return {
        "n_ref_with_citation": n_ref_has,
        "format_rate": round(n_fmt / n, 4),
        "precision": round(statistics.mean(precs), 4) if precs else 0.0,
        "recall": round(statistics.mean(recs), 4) if recs else 0.0,
        "exact_match": round(exact / n, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="法律问答评测")
    ap.add_argument("--pred", required=True, help="generated_predictions.jsonl 路径")
    ap.add_argument("--out", default=None, help="结果 JSON 输出路径")
    ap.add_argument("--bertscore", action="store_true",
                    help="启用 BERTScore（首次运行会下载中文 BERT，约 400MB）")
    args = ap.parse_args()

    pairs = load_pairs(args.pred)
    if not pairs:
        raise SystemExit(f"没有可用样本：{args.pred}")
    preds, refs = zip(*pairs)
    print(f"样本数：{len(pairs)}，空预测：{sum(1 for p in preds if not p)}")

    result = {
        "n": len(pairs),
        "rouge_l": round(rouge_l(preds, refs), 4),
        "bleu_char": round(bleu(preds, refs), 2),
        "citation": citation_metrics(preds, refs),
    }
    if args.bertscore:
        result["bertscore_f1"] = round(bert_score(preds, refs), 4)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"已写入 {out}")


if __name__ == "__main__":
    main()
