"""
Build a PuzzleDataset-compatible dataset for the EN-DSL INIT+GAIN+HOW_MANY task.

Input:
    JSONL files produced by `dev/gen_en_dsl_init_gain.py`, containing records:

        {
          "id": "...",
          "module": "en_dsl_init_gain",
          "split": "train" | "test",
          "english": "...",
          "answer": 8,
          "en_token_ids": [...],       # GyanDSLToken IDs (including BOS/EOS)
          "en_token_names": [...],
          "expr_token_ids": [...],
          "expr_token_names": [...]
        }

Output:
    A directory in the format expected by `PuzzleDataset`:

        output_dir/
          train/
            all__inputs.npy
            all__labels.npy
            all__puzzle_identifiers.npy
            all__puzzle_indices.npy
            all__group_indices.npy
            dataset.json
          test/
            (same fields for test split)
          module_mapping.json
          identifiers.json

Design:
    - We treat each example as its own puzzle and group.
    - Inputs are a single sequence:
          [ENGLISH_TOKENS..., SEP, EN_DSL_TOKENS...]
      where:
        * ENGLISH_TOKENS are simple word-level tokens mapped to integer IDs
          that live *above* the Gyan DSL vocabulary.
        * EN_DSL_TOKENS are the provided `en_token_ids`, which are valid
          `GyanDSLToken` IDs.
    - Labels follow the standard TRM objective:
        * labels = IGNORE_LABEL_ID everywhere except the EN-DSL suffix
          (after SEP), where labels equal the true EN-DSL token IDs.
        * Inputs are masked over the EN-DSL suffix (answer span) by setting
          those positions to PAD_ID, so the model must reconstruct EN-DSL
          from English alone.
"""

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel

from dataset.common import PuzzleDatasetMetadata
from dsl.tokens import GyanDSLToken, get_vocab_size


PAD_ID = GyanDSLToken.PAD.value
IGNORE_LABEL_ID = -100

cli = ArgParser()


class ENDSLInitGainDatasetConfig(BaseModel):
    input_dir: str
    output_dir: str
    seq_len: int = 128
    seed: int = 42


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def tokenize_english(text: str) -> List[str]:
    """
    Very simple word/char tokenizer for the narrow INIT+GAIN+HOW_MANY stories.

    Splits into:
      - alphabetic spans
      - numeric spans
      - individual punctuation
    """

    return re.findall(r"[A-Za-z]+|[0-9]+|[^A-Za-z0-9\\s]", text)


@dataclass
class ENVocab:
    base_vocab_size: int
    word_to_id: Dict[str, int]

    @property
    def offset(self) -> int:
        return self.base_vocab_size

    def get_id(self, token: str) -> int:
        """
        Map an English token to a global token ID in the unified vocabulary.

        English tokens occupy the range [base_vocab_size, base_vocab_size + N).
        """

        if token not in self.word_to_id:
            self.word_to_id[token] = len(self.word_to_id)
        return self.offset + self.word_to_id[token]

    @property
    def vocab_size(self) -> int:
        return self.base_vocab_size + len(self.word_to_id)


def build_english_vocab(examples: List[Dict[str, Any]]) -> ENVocab:
    base_vocab_size = get_vocab_size()
    word_to_id: Dict[str, int] = {}

    for ex in examples:
        text = str(ex["english"])
        for tok in tokenize_english(text):
            if tok not in word_to_id:
                word_to_id[tok] = len(word_to_id)

    # Reserve a special [SEP] token in the English vocab.
    if "[SEP]" not in word_to_id:
        word_to_id["[SEP]"] = len(word_to_id)

    return ENVocab(base_vocab_size=base_vocab_size, word_to_id=word_to_id)


@dataclass
class ExampleTensors:
    inputs: np.ndarray
    labels: np.ndarray
    puzzle_id: int


def make_example_tensors(
    english: str,
    en_token_ids: List[int],
    vocab: ENVocab,
    module_id: int,
    seq_len: int,
) -> ExampleTensors | None:
    """
    Build (inputs, labels, puzzle_id) for a single example.

    - inputs: [ENGLISH_TOKENS..., SEP, EN_DSL_TOKENS...] with EN-DSL suffix
      masked to PAD_ID.
    - labels: IGNORE_LABEL_ID except on the EN-DSL suffix, where labels equal
      the true EN-DSL token IDs.
    """

    english_tokens = tokenize_english(english)
    english_ids = [vocab.get_id(tok) for tok in english_tokens]

    sep_id = vocab.get_id("[SEP]")

    # EN-DSL tokens are already valid GyanDSLToken IDs.
    dsl_ids = list(map(int, en_token_ids))

    full_seq: List[int] = english_ids + [sep_id] + dsl_ids
    if len(full_seq) > seq_len:
        return None

    ans_start = len(english_ids) + 1  # first EN-DSL token
    ans_end = len(full_seq)
    if ans_start >= ans_end:
        return None

    inputs = np.full(seq_len, PAD_ID, dtype=np.int32)
    inputs[: len(full_seq)] = np.array(full_seq, dtype=np.int32)

    labels = np.full(seq_len, IGNORE_LABEL_ID, dtype=np.int32)
    labels[ans_start:ans_end] = inputs[ans_start:ans_end]

    # Mask answer tokens in the input.
    inputs[ans_start:ans_end] = PAD_ID

    return ExampleTensors(inputs=inputs, labels=labels, puzzle_id=module_id)


def convert_split(
    examples: List[Dict[str, Any]],
    vocab: ENVocab,
    module_to_id: Dict[str, int],
    seq_len: int,
    split_name: str,
) -> Dict[str, np.ndarray]:
    all_inputs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_puzzle_ids: List[int] = []

    skipped = 0

    for ex in examples:
        module = ex.get("module", "en_dsl_init_gain")
        if module not in module_to_id:
            continue
        module_id = module_to_id[module]

        tensors = make_example_tensors(
            english=str(ex["english"]),
            en_token_ids=ex["en_token_ids"],
            vocab=vocab,
            module_id=module_id,
            seq_len=seq_len,
        )
        if tensors is None:
            skipped += 1
            continue

        all_inputs.append(tensors.inputs)
        all_labels.append(tensors.labels)
        all_puzzle_ids.append(tensors.puzzle_id)

    num_examples = len(all_inputs)
    if num_examples == 0:
        print(f"[{split_name}] WARNING: no usable examples produced (skipped={skipped}).")
        return {
            "inputs": np.zeros((0, seq_len), dtype=np.int32),
            "labels": np.zeros((0, seq_len), dtype=np.int32),
            "puzzle_identifiers": np.zeros((0,), dtype=np.int32),
            "puzzle_indices": np.zeros((1,), dtype=np.int32),
            "group_indices": np.zeros((1,), dtype=np.int32),
        }

    inputs_arr = np.stack(all_inputs, axis=0)
    labels_arr = np.stack(all_labels, axis=0)
    puzzle_ids_arr = np.array(all_puzzle_ids, dtype=np.int32)

    # One example per puzzle and per group.
    puzzle_indices = np.arange(0, num_examples + 1, dtype=np.int32)
    group_indices = np.arange(0, num_examples + 1, dtype=np.int32)

    print(f"[{split_name}] examples: {num_examples}, skipped: {skipped}")

    return {
        "inputs": inputs_arr,
        "labels": labels_arr,
        "puzzle_identifiers": puzzle_ids_arr,
        "puzzle_indices": puzzle_indices,
        "group_indices": group_indices,
    }


@cli.command(singleton=True)
def main(config: ENDSLInitGainDatasetConfig) -> None:  # pragma: no cover - CLI entry
    np.random.seed(config.seed)

    train_path = os.path.join(config.input_dir, "train.jsonl")
    test_path = os.path.join(config.input_dir, "test.jsonl")

    train_examples = load_jsonl(train_path)
    test_examples = load_jsonl(test_path)
    all_examples = train_examples + test_examples

    if not all_examples:
        raise ValueError(f"No examples found under {config.input_dir}")

    # Build English-vocabulary overlay on top of the existing DSL vocab.
    vocab = build_english_vocab(all_examples)
    print(f"Gyan DSL base vocab size: {vocab.base_vocab_size}")
    print(f"English vocab size: {len(vocab.word_to_id)}")
    print(f"Total vocab size (DSL + English): {vocab.vocab_size}")
    print(f"Sequence length: {config.seq_len}")

    # Single module for this task.
    module_to_id = {"en_dsl_init_gain": 1}
    num_modules = len(module_to_id) + 1  # +1 for <blank>

    os.makedirs(config.output_dir, exist_ok=True)

    splits = {
        "train": train_examples,
        "test": test_examples,
    }

    for split_name, exs in splits.items():
        split_dir = os.path.join(config.output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        data = convert_split(
            examples=exs,
            vocab=vocab,
            module_to_id=module_to_id,
            seq_len=config.seq_len,
            split_name=split_name,
        )

        for name, arr in data.items():
            path = os.path.join(split_dir, f"all__{name}.npy")
            np.save(path, arr)
            print(f"  [{split_name}] saved {name}: shape={arr.shape}")

        num_examples = data["inputs"].shape[0]
        num_groups = data["group_indices"].size - 1

        metadata = PuzzleDatasetMetadata(
            seq_len=config.seq_len,
            vocab_size=vocab.vocab_size,
            pad_id=PAD_ID,
            ignore_label_id=None,  # labels already use -100 for ignore
            blank_identifier_id=0,
            num_puzzle_identifiers=num_modules,
            total_groups=num_groups,
            mean_puzzle_examples=1.0,
            total_puzzles=num_examples,
            sets=["all"],
        )

        with open(os.path.join(split_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)

    # Save module mapping for reference.
    with open(os.path.join(config.output_dir, "module_mapping.json"), "w") as f:
        json.dump(module_to_id, f, indent=2)

    id_to_module = {0: "<blank>"}
    id_to_module.update({v: k for k, v in module_to_id.items()})
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump([id_to_module.get(i, "<unknown>") for i in range(num_modules)], f, indent=2)

    # Persist English vocabulary for inspection.
    with open(os.path.join(config.output_dir, "english_vocab.json"), "w") as f:
        json.dump(vocab.word_to_id, f, indent=2)

    print(f"\nDataset saved to {config.output_dir}")
    print(
        "Task: Given an input sequence with English prefix and masked EN-DSL "
        "suffix, predict the EN-DSL program tokens."
    )


if __name__ == "__main__":
    # Use Argdantic's CLI entry point (parses args and calls `main`). 
    cli()


