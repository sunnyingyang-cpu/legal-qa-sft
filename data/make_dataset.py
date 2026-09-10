#!/usr/bin/env python3
"""构建法律问答 SFT 数据包（阶段 1）。

流程：下载 DISC-Law-SFT（HuggingFace）→ 字段归一化 → 清洗去重 → 分层抽样
     → 切分 train/val/test。验证与测试集优先保留答案带法条引用的样本，
     保证"引用准确率"指标可计算。

产出（写入 data/ 目录）：
    legal_qa_train.json / legal_qa_val.json / legal_qa_test.json
    （LLaMA-Factory alpaca 格式：{"instruction", "input", "output"}）
    data_stats.json（数据统计）

用法：python data/make_dataset.py --train 5000 --val 300 --test 300
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE

REPO_ID = "ShengbinYue/DISC-Law-SFT"
RAW_FILE = "DISC-Law-SFT-Pair.jsonl"

QUESTION_KEYS = ("question", "instruction", "input", "q")
ANSWER_KEYS = ("answer", "output", "response")
CATEGORY_KEYS = ("category", "task", "type", "source")

CITE_RE = re.compile(r"《[^》]{2,30}》第[零一二三四五六七八九十百千万〇0-9]+条")
Q_MIN, Q_MAX, A_MIN, A_MAX = 10, 512, 20, 1200


def normalize(text: str) -> str:
    """NFKC 归一化并去标点空白，用于去重比对。"""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"[\s，。；：、！？（）《》“”‘’\"'…·—\-]", "", text).lower()


def pick(row: dict, keys) -> str:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def load_raw() -> list:
    """下载并读取 DISC-Law-SFT-Pair jsonl。"""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=REPO_ID, filename=RAW_FILE, repo_type="dataset")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clean(rows: list) -> list:
    seen = set()
    cleaned = []
    for row in rows:
        q, a = pick(row, QUESTION_KEYS), pick(row, ANSWER_KEYS)
        if not (Q_MIN <= len(q) <= Q_MAX and A_MIN <= len(a) <= A_MAX):
            continue
        key = hashlib.md5(normalize(q).encode("utf-8")).hexdigest()
        if key in seen:  # 问题级去重，同时防训练/测试集泄漏
            continue
        seen.add(key)
        cat = pick(row, CATEGORY_KEYS) or "unknown"
        cleaned.append({"instruction": q, "input": "", "output": a, "_cat": cat})
    return cleaned


def split(cleaned: list, n_train: int, n_val: int, n_test: int, seed: int):
    rng = random.Random(seed)
    with_cite = [s for s in cleaned if CITE_RE.search(s["output"])]
    rest = [s for s in cleaned if not CITE_RE.search(s["output"])]
    rng.shuffle(with_cite)
    rng.shuffle(rest)

    # val/test 优先取带引用样本
    val = with_cite[:n_val]
    test = with_cite[n_val:n_val + n_test]
    if len(val) < n_val:  # 引用样本不足时从普通样本回补
        need = n_val - len(val)
        val += rest[:need]
        rest = rest[need:]
    if len(test) < n_test:
        need = n_test - len(test)
        test += rest[:need]
        rest = rest[need:]

    used = {id(s) for s in val} | {id(s) for s in test}

    # 训练集：单类别占比不超过 40%，防止单一任务类型淹没其他类别
    pool = [s for s in with_cite + rest if id(s) not in used]
    by_cat = {}
    for s in pool:
        by_cat.setdefault(s["_cat"], []).append(s)
    cap = max(1, int(n_train * 0.4))
    train = []
    for items in by_cat.values():
        rng.shuffle(items)
        train.extend(items[:cap])
    rng.shuffle(train)
    if len(train) < n_train:  # 被类别上限截掉的样本回补
        chosen = {id(s) for s in train}
        train += [s for s in pool if id(s) not in chosen][:n_train - len(train)]
    return train[:n_train], val[:n_val], test[:n_test]


def dump(samples: list, path: Path) -> None:
    rows = [{"instruction": s["instruction"], "input": "", "output": s["output"]} for s in samples]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)


def main() -> None:
    ap = argparse.ArgumentParser(description="构建法律问答 SFT 数据包")
    ap.add_argument("--train", type=int, default=5000)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[1/4] 下载 {REPO_ID}/{RAW_FILE} ...")
    raw = load_raw()
    print(f"      原始样本：{len(raw)}")

    print("[2/4] 清洗（长度过滤 + 问题级去重）...")
    cleaned = clean(raw)
    n_cite = sum(1 for s in cleaned if CITE_RE.search(s["output"]))
    print(f"      清洗后：{len(cleaned)}（含法条引用 {n_cite}，{n_cite / max(len(cleaned), 1):.1%}）")

    print("[3/4] 分层抽样与切分 ...")
    train, val, test = split(cleaned, args.train, args.val, args.test, args.seed)

    print("[4/4] 写出数据文件 ...")
    dump(train, OUT_DIR / "legal_qa_train.json")
    dump(val, OUT_DIR / "legal_qa_val.json")
    dump(test, OUT_DIR / "legal_qa_test.json")
    stats = {
        "raw": len(raw),
        "cleaned": len(cleaned),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "test_citation_rate": round(
            sum(1 for s in test if CITE_RE.search(s["output"])) / max(len(test), 1), 4
        ),
        "train_top_categories": dict(Counter(s["_cat"] for s in train).most_common(10)),
        "seed": args.seed,
    }
    with open(OUT_DIR / "data_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
