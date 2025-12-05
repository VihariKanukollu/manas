#!/usr/bin/env python
"""
Build a SentencePiece training corpus for the full Gyan DSL vocabulary.

We combine:
- Real EN-DSL sequences from `data/en_dsl_seq2seq/train.jsonl` (the `en_tokens` field)
- Synthetic lines that ensure *every* `GyanDSLToken` name appears at least once

Output:
- `data/en_dsl_seq2seq/en_dsl_sp_corpus_full.txt`

This file can be passed to SentencePiece with our `dsl` model_type alias
(`model_type=dsl`, which is wired to the WORD model) so that every DSL token
becomes an atomic piece.
"""

from pathlib import Path
import json
import sys
from typing import Set, List

# Ensure project root (containing `dsl/`) is on sys.path when this script
# is executed from the `scripts/` directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dsl.tokens import GyanDSLToken  # source of truth for the DSL vocab


SRC = Path("data/en_dsl_seq2seq/train.jsonl")
OUT = Path("data/en_dsl_seq2seq/en_dsl_sp_corpus_full.txt")


def iter_all_dsl_token_names() -> List[str]:
    """Return all DSL token names from GyanDSLToken."""
    return [tok.name for tok in GyanDSLToken]


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"Source EN-DSL seq2seq file not found: {SRC}")

    OUT.parent.mkdir(parents=True, exist_ok=True)

    all_dsl_tokens = set(iter_all_dsl_token_names())
    seen_tokens: Set[str] = set()
    total_real_lines = 0

    with SRC.open() as f_in, OUT.open("w") as f_out:
        # 1) Real sequences from training data
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            tokens = ex.get("en_tokens", [])
            if not tokens:
                continue
            # Record which tokens we've seen in real data.
            for t in tokens:
                if isinstance(t, str):
                    seen_tokens.add(t)
            f_out.write(" ".join(tokens) + "\n")
            total_real_lines += 1

        # 2) Synthetic coverage lines for any unseen DSL tokens
        unseen_tokens = sorted(all_dsl_tokens - seen_tokens)

        synthetic_lines = 0
        if unseen_tokens:
            # Group unseen tokens into reasonably sized lines so SentencePiece
            # can see them in simple contexts.
            chunk_size = 50
            for i in range(0, len(unseen_tokens), chunk_size):
                chunk = unseen_tokens[i : i + chunk_size]
                f_out.write(" ".join(chunk) + "\n")
                synthetic_lines += 1

    print(f"Wrote SentencePiece corpus to: {OUT}")
    print(f"  Real EN-DSL sequences : {total_real_lines}")
    print(f"  Unique DSL tokens     : {len(all_dsl_tokens)}")
    print(f"  Seen in real data     : {len(seen_tokens)}")
    print(f"  Unseen (synthetic)    : {len(all_dsl_tokens - seen_tokens)}")
    print(f"  Synthetic lines added : {synthetic_lines}")


if __name__ == "__main__":
    main()


