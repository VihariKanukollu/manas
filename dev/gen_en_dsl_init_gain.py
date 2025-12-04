"""
Generate synthetic (English, EN-DSL, math-expression, answer) data for the
INIT+GAIN+HOW_MANY pattern.

Each example is a tiny inventory story of the form:

    "<Name> had A apples. <Name> bought B more apples.
     How many apples does <Name> have now?"

We rely on the existing EN tooling:
  - `dsl.en_pipeline.solve_init_gain_problem` to:
      * parse English → EN-DSL tokens
      * run the EN-DSL interpreter
      * derive a math-style postfix expression and numeric answer

Output format: JSONL with records like:

    {
      "id": "en_init_gain/train/000001",
      "module": "en_dsl_init_gain",
      "split": "train",
      "english": "...",
      "answer": 8,
      "en_token_ids": [...],
      "en_token_names": [...],
      "expr_token_ids": [...],
      "expr_token_names": [...]
    }

Later, a separate dataset builder will convert this JSONL into the
`PuzzleDataset` format expected by `pretrain.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import List, Dict, Any, Tuple

from dsl.en_pipeline import solve_init_gain_problem


NAMES = ["Alice", "Bob", "Carol", "David", "Eve"]
ITEMS = ["apples", "marbles", "stickers", "coins"]


def _sample_ab(rng: random.Random, max_sum: int = 40) -> Tuple[int, int]:
    """
    Sample small positive integers A, B such that A + B stays safely within
    the INT_* range used in EN-DSL (0..99) and expression values remain small.
    """

    while True:
        a = rng.randint(1, max_sum // 2)
        b = rng.randint(1, max_sum // 2)
        if a + b < 100:
            return a, b


def _make_problem(name: str, item: str, a: int, b: int) -> str:
    """
    Construct the canonical INIT+GAIN+HOW_MANY story for a given (name, item, A, B).
    """

    # Use the exact surface form expected by the current rule-based parser.
    return (
        f"{name} had {a} {item}. "
        f"{name} bought {b} more {item}. "
        f"How many {item} does {name} have now?"
    )


def generate_split(
    split_name: str,
    num_examples: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    Generate a list of synthetic examples for a given split.
    """

    examples: List[Dict[str, Any]] = []

    for idx in range(num_examples):
        name = rng.choice(NAMES)
        item = rng.choice(ITEMS)
        a, b = _sample_ab(rng)

        problem = _make_problem(name, item, a, b)

        # Use the existing EN pipeline to ensure EN tokens and math expression
        # are consistent with the interpreter.
        res = solve_init_gain_problem(problem)

        examples.append(
            {
                "id": f"en_init_gain/{split_name}/{idx:06d}",
                "module": "en_dsl_init_gain",
                "split": split_name,
                "english": res["problem"],
                "answer": res["answer"],
                "en_token_ids": res["en_token_ids"],
                "en_token_names": res["en_token_names"],
                "expr_token_ids": res["expr_token_ids"],
                "expr_token_names": res["expr_token_names"],
            }
        )

    return examples


def write_jsonl(path: str, examples: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic (English, EN-DSL, math-expression, answer) data "
        "for the INIT+GAIN+HOW_MANY pattern."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/en_dsl_init_gain",
        help="Directory to write JSONL files (train.jsonl, test.jsonl).",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=10000,
        help="Number of training examples to generate.",
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=2000,
        help="Number of test examples to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    rng = random.Random(args.seed)

    train_examples = generate_split("train", args.num_train, rng)
    test_examples = generate_split("test", args.num_test, rng)

    train_path = os.path.join(args.output_dir, "train.jsonl")
    test_path = os.path.join(args.output_dir, "test.jsonl")

    write_jsonl(train_path, train_examples)
    write_jsonl(test_path, test_examples)

    print(f"Wrote {len(train_examples)} train examples to {train_path}")
    print(f"Wrote {len(test_examples)} test examples to {test_path}")


if __name__ == "__main__":
    main()


