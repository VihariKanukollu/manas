"""
Build a DSL token dataset in the format expected by TRM's `PuzzleDataset`.

Design goals
------------
- **No heuristics.** We rely only on the known DSL structure produced by
  `dev.gen_full_math.py`, not on guessing from the answer string.
- **Module‑aware.** For now we support the equation‑solving modules that use
  the pattern:

    BOS <lhs_tokens> <rhs_tokens> EQ REAL_VAR_i <answer_tokens> IS_SOLUTION EOS

  This includes:
    - algebra__linear_1d
    - algebra__linear_1d_composed
    - algebra__linear_2d
    - algebra__linear_2d_composed

- **TRM objective.**
  - Inputs: full token sequence, but answer tokens are replaced by PAD.
  - Labels: IGNORE_LABEL_ID (-100) everywhere except answer positions.

This keeps the data generation *structurally correct* for the modules we care
about now (especially `algebra__linear_1d`) and is easy to extend with more
per‑module span rules later.

Usage
-----
    python -m dataset.build_dsl_dataset \\
        --input-dir data/math_dsl_small \\
        --output-dir data/dsl_trm_small
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel

from dataset.common import PuzzleDatasetMetadata

# Import DSL tokens
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsl.tokens import get_vocab_size, GyanDSLToken  # type: ignore


PAD_ID = GyanDSLToken.PAD.value
EOS_ID = GyanDSLToken.EOS.value
IGNORE_LABEL_ID = -100


# Modules whose answers follow the EQ / REAL_VAR / IS_SOLUTION pattern.
EQ_SOLUTION_MODULES = {
    "algebra__linear_1d",
    "algebra__linear_1d_composed",
    "algebra__linear_2d",
    "algebra__linear_2d_composed",
}


cli = ArgParser()


class DSLDatasetConfig(BaseModel):
    """CLI configuration."""

    input_dir: str
    output_dir: str
    seq_len: int = 128
    seed: int = 42


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    examples: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def build_module_id_map(examples: List[Dict[str, Any]]) -> Dict[str, int]:
    """Assign a numeric ID to each module name (for puzzle embeddings)."""
    modules = sorted({ex["module"] for ex in examples})
    # 0 is reserved for <blank> in metadata
    return {m: i + 1 for i, m in enumerate(modules)}


def locate_eq_solution_span(token_names: List[str]) -> Tuple[int, int]:
    """
    Locate the answer span for EQ/IS_SOLUTION style modules.

    Expected tail structure (see `build_equation_solution_tokens` and
    `build_system_solution_tokens` in `dev/gen_full_math.py`):

        ... EQ REAL_VAR_k <answer_tokens...> IS_SOLUTION EOS

    We return (start, end) indices for `<answer_tokens...>`.
    """
    try:
        is_idx = token_names.index("IS_SOLUTION")
    except ValueError:
        return 0, 0

    # Find the EQ that immediately precedes this IS_SOLUTION.
    eq_idx = -1
    for i in range(is_idx - 1, -1, -1):
        if token_names[i] == "EQ":
            eq_idx = i
            break

    if eq_idx < 0:
        return 0, 0

    # Sanity check: token right after EQ should be REAL_VAR_*
    if eq_idx + 1 >= len(token_names):
        return 0, 0
    if not token_names[eq_idx + 1].startswith("REAL_VAR_"):
        return 0, 0

    answer_start = eq_idx + 2
    answer_end = is_idx
    if answer_start >= answer_end:
        return 0, 0
    return answer_start, answer_end


@dataclass
class ExampleTensors:
    inputs: np.ndarray
    labels: np.ndarray
    puzzle_id: int


def make_example_tensors(
    module: str,
    token_ids: List[int],
    token_names: List[str],
    module_id: int,
    seq_len: int,
) -> ExampleTensors | None:
    """
    Convert a single JSONL example into (inputs, labels, puzzle_id).

    - For unsupported modules, returns None.
    - For supported EQ_SOLUTION_MODULES, uses exact structural rules.
    """
    if module not in EQ_SOLUTION_MODULES:
        # Unsupported module for now
        return None

    # Sanity: lengths should match
    if len(token_ids) != len(token_names):
        return None

    # Skip overly long sequences (should not happen for linear_1d, but be safe).
    if len(token_ids) > seq_len:
        return None

    # Locate answer span
    ans_start, ans_end = locate_eq_solution_span(token_names)
    if ans_start == ans_end:
        return None

    # Build padded input
    inputs = np.full(seq_len, PAD_ID, dtype=np.int32)
    inputs[: len(token_ids)] = np.array(token_ids, dtype=np.int32)

    # Build labels: only answer span is supervised
    labels = np.full(seq_len, IGNORE_LABEL_ID, dtype=np.int32)
    labels[ans_start:ans_end] = inputs[ans_start:ans_end]

    # Mask answer tokens in the input
    inputs[ans_start:ans_end] = PAD_ID

    return ExampleTensors(inputs=inputs, labels=labels, puzzle_id=module_id)


def convert_split(
    examples: List[Dict[str, Any]],
    module_to_id: Dict[str, int],
    seq_len: int,
    split_name: str,
) -> Dict[str, np.ndarray]:
    """
    Convert a list of JSONL examples into numpy arrays for TRM.

    Each example becomes its own "puzzle" and its own "group" for simplicity:
      - puzzle_indices: [0, 1, 2, ..., N]
      - group_indices:  [0, 1, 2, ..., N]
    """
    all_inputs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_puzzle_ids: List[int] = []

    skipped_by_reason: Dict[str, int] = {}

    for ex in examples:
        module = ex["module"]
        if module not in module_to_id:
            skipped_by_reason["unknown_module"] = skipped_by_reason.get("unknown_module", 0) + 1
            continue
        module_id = module_to_id[module]

        tensors = make_example_tensors(
            module=module,
            token_ids=ex["token_ids"],
            token_names=ex.get("token_names", []),
            module_id=module_id,
            seq_len=seq_len,
        )
        if tensors is None:
            skipped_by_reason["unsupported_or_bad_example"] = (
                skipped_by_reason.get("unsupported_or_bad_example", 0) + 1
            )
            continue

        all_inputs.append(tensors.inputs)
        all_labels.append(tensors.labels)
        all_puzzle_ids.append(tensors.puzzle_id)

    num_examples = len(all_inputs)
    if num_examples == 0:
        print(f"[{split_name}] WARNING: no usable examples produced.")
        for reason, count in skipped_by_reason.items():
            print(f"  Skipped ({reason}): {count}")
        # Return empty tensors with the right shapes.
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

    # One example per puzzle and per group
    puzzle_indices = np.arange(0, num_examples + 1, dtype=np.int32)
    group_indices = np.arange(0, num_examples + 1, dtype=np.int32)

    print(f"[{split_name}] examples: {num_examples}")
    for reason, count in skipped_by_reason.items():
        print(f"  Skipped ({reason}): {count}")

    return {
        "inputs": inputs_arr,
        "labels": labels_arr,
        "puzzle_identifiers": puzzle_ids_arr,
        "puzzle_indices": puzzle_indices,
        "group_indices": group_indices,
    }


@cli.command(singleton=True)
def main(config: DSLDatasetConfig) -> None:  # pragma: no cover - CLI entry
    np.random.seed(config.seed)

    vocab_size = get_vocab_size()
    print(f"DSL vocab size: {vocab_size}")
    print(f"Sequence length: {config.seq_len}")

    # Load JSONL splits
    train_path = os.path.join(config.input_dir, "train.jsonl")
    test_id_path = os.path.join(config.input_dir, "test_id.jsonl")
    test_ood_path = os.path.join(config.input_dir, "test_ood.jsonl")

    train_examples = load_jsonl(train_path)
    test_id_examples = load_jsonl(test_id_path)
    test_ood_examples = load_jsonl(test_ood_path)

    all_examples = train_examples + test_id_examples + test_ood_examples
    module_to_id = build_module_id_map(all_examples)
    num_modules = len(module_to_id) + 1  # +1 for <blank>

    print(f"Modules found: {sorted(module_to_id.keys())}")
    print(f"Supported EQ_SOLUTION modules: {sorted(EQ_SOLUTION_MODULES)}")

    os.makedirs(config.output_dir, exist_ok=True)

    splits = {
        "train": train_examples,
        "test": test_id_examples + test_ood_examples,
    }

    for split_name, exs in splits.items():
        split_dir = os.path.join(config.output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        data = convert_split(
            examples=exs,
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
            vocab_size=vocab_size,
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

    # Save module mapping for reference
    with open(os.path.join(config.output_dir, "module_mapping.json"), "w") as f:
        json.dump(module_to_id, f, indent=2)

    id_to_module = {0: "<blank>"}
    id_to_module.update({v: k for k, v in module_to_id.items()})
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump([id_to_module.get(i, "<unknown>") for i in range(num_modules)], f, indent=2)

    print(f"\nDataset saved to {config.output_dir}")
    print("Task: Given DSL sequence with answer masked, predict the answer tokens.")


if __name__ == "__main__":
    # Use Argdantic's CLI entry point (parses args and calls `main`).
    cli()


