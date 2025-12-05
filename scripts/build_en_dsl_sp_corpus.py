#!/usr/bin/env python
"""
Build a SentencePiece-style training corpus from EN-DSL seq2seq data.

Each line of the output is a space-separated sequence of EN-DSL token names
from the `en_tokens` field, e.g.:

    BOS EN_EVT_INIT EN_AMOUNT INT_24 EN_EVT_GAIN EN_AMOUNT INT_72 EN_QUERY ...

This file can be used as the `--input` to `spm_train` with our new
`--model_type=dsl` alias (internally treated as a WORD model).
"""

from pathlib import Path
import json


SRC = Path("data/en_dsl_seq2seq/train.jsonl")
OUT = Path("data/en_dsl_seq2seq/en_dsl_sp_corpus.txt")


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"Source EN-DSL seq2seq file not found: {SRC}")

    OUT.parent.mkdir(parents=True, exist_ok=True)

    num_lines = 0
    with SRC.open() as f_in, OUT.open("w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            tokens = ex.get("en_tokens", [])
            if not tokens:
                continue
            f_out.write(" ".join(tokens) + "\n")
            num_lines += 1

    print(f"Wrote {num_lines} DSL sequences to {OUT}")


if __name__ == "__main__":
    main()


