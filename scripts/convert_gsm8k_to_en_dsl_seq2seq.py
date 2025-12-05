#!/usr/bin/env python
"""
Convert generated GSM8K DSL data (with en_token_names) to EN-DSL seq2seq format.

Input:  data/gsm8k_en_dsl_full/train.jsonl (from dev.gen_gsm8k)
Output: data/en_dsl_seq2seq/train.jsonl (for dataset/build_english_trm_dataset.py)

Usage:
    python scripts/convert_gsm8k_to_en_dsl_seq2seq.py
"""

import json
from pathlib import Path

SRC_DIR = Path("data/gsm8k_en_dsl_full")
DST_DIR = Path("data/en_dsl_seq2seq")

SPLITS = [
    ("train", "train"),
    ("valid", "test_id"),
    ("test", "test_ood"),
]


def convert_split(dst_name: str, src_name: str) -> int:
    """Convert one split, keeping only examples with en_token_names."""
    src_file = SRC_DIR / f"{src_name}.jsonl"
    dst_file = DST_DIR / f"{dst_name}.jsonl"
    
    if not src_file.exists():
        print(f"Warning: {src_file} not found, skipping")
        return 0
    
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    
    n = 0
    with src_file.open() as fin, dst_file.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            ex = json.loads(line)
            en_tokens = ex.get("en_token_names")
            question = ex.get("question")
            
            # Skip if no EN-DSL tokens
            if not en_tokens or not question:
                continue
            
            record = {
                "id": ex.get("id", f"gsm8k/{dst_name}/{n:06d}"),
                "module": "gsm8k",
                "question": question,
                "en_tokens": en_tokens,
            }
            fout.write(json.dumps(record) + "\n")
            n += 1
    
    print(f"[{dst_name}] wrote {n} examples to {dst_file}")
    return n


def main():
    total = 0
    for dst_name, src_name in SPLITS:
        total += convert_split(dst_name, src_name)
    print(f"\nTotal: {total} examples")


if __name__ == "__main__":
    main()

