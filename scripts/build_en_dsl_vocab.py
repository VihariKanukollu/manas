#!/usr/bin/env python
"""
Scan EN-DSL seq2seq training data and build a flat vocabulary file of
all EN-DSL token names that appear in `en_tokens`.

Output format (one token per line), e.g.:

    BOS
    EOS
    EN_QUERY
    EN_Q_ATTR
    EN_GROUP
    EN_MEMBER
    EN_AMOUNT
    INT_0
    INT_1
    ...
"""

from pathlib import Path
import json


SRC = Path("data/en_dsl_seq2seq/train.jsonl")
OUT = Path("data/en_dsl_seq2seq/en_dsl_vocab.txt")


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"Source EN-DSL seq2seq file not found: {SRC}")

    vocab = set()
    with SRC.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            for t in ex.get("en_tokens", []):
                vocab.add(t)

    tokens = sorted(vocab)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as out:
        for t in tokens:
            out.write(t + "\n")

    print(f"Found {len(tokens)} EN-DSL tokens, wrote {OUT}")


if __name__ == "__main__":
    main()


