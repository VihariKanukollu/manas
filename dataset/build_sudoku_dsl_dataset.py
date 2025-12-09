from __future__ import annotations

"""
Build a Sudoku DSL *program* dataset for TRM training.

Each example consists of:
  - An unsolved 9x9 Sudoku grid (puzzle) encoded as DSL INT_* tokens.
  - A DSL program that, when executed, fills in the missing cells.

Program format (postfix-style, but interpreted procedurally):
  BOS,
    [ INT_row, INT_col, INT_val, SET_CELL ] * K,
    CHECK_SUDOKU,
  EOS

Where:
  - (row, col) are 0-based indices in [0..8]
  - val is the solution digit in [1..9]
  - K is the number of blanks in the puzzle

We store:
  - english: [N, english_seq_len]  (flattened puzzle grid as INT_* tokens)
  - program: [N, program_seq_len]  (padded program tokens)
  - inputs / labels: placeholders, rebuilt in PuzzleDataset._collate_batch as:
        inputs = [english, PAD(program)]
        labels = [IGNORE, program]

This mirrors the English→DSL and ARC-DSL program datasets and avoids label leak.
"""

import os
import csv
import json
from typing import Optional, List, Tuple

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from dataset.common import PuzzleDatasetMetadata
from dsl.tokens import (
    GyanDSLToken,
    get_vocab_size,
    get_int_const_token,
)


cli = ArgParser()

PAD_ID = GyanDSLToken.PAD.value
IGNORE_LABEL_ID = -100


class SudokuDSLConfig(BaseModel):
    """CLI configuration for Sudoku DSL-program dataset."""

    source_repo: str = "sapientinc/sudoku-extreme"
    output_dir: str = "data/sudoku_dsl_trm"

    subsample_size: Optional[int] = None
    min_difficulty: Optional[int] = None
    num_aug: int = 0  # number of augmentations per puzzle in train split

    # Segment lengths
    english_seq_len: int = 81          # 9x9 puzzle grid
    program_seq_len: int = 400         # sufficient for 81 moves * 4 + BOS/EOS/CHECK


def _load_sudoku_split(
    set_name: str,
    config: SudokuDSLConfig,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load Sudoku puzzles and solutions from CSV.

    Returns a list of (puzzle, solution) pairs, each 9x9 int array with values 0-9.
    """
    puzzles: List[np.ndarray] = []
    solutions: List[np.ndarray] = []

    path = hf_hub_download(config.source_repo, f"{set_name}.csv", repo_type="dataset")  # type: ignore
    with open(path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for _source, q, a, rating in reader:
            if (config.min_difficulty is not None) and (int(rating) < config.min_difficulty):
                continue

            assert len(q) == 81 and len(a) == 81

            # Puzzle: '.' indicates blank -> treat as '0'
            puzzle = (
                np.frombuffer(q.replace(".", "0").encode(), dtype=np.uint8)
                .reshape(9, 9)
                - ord("0")
            )
            # Solution: fully filled 9x9 grid
            solution = (
                np.frombuffer(a.encode(), dtype=np.uint8)
                .reshape(9, 9)
                - ord("0")
            )

            puzzles.append(puzzle.astype(np.int32))
            solutions.append(solution.astype(np.int32))

    # Optional subsampling (train only)
    if set_name == "train" and config.subsample_size is not None:
        total_samples = len(puzzles)
        if config.subsample_size < total_samples:
            indices = np.random.choice(total_samples, size=config.subsample_size, replace=False)
            puzzles = [puzzles[i] for i in indices]
            solutions = [solutions[i] for i in indices]

    return list(zip(puzzles, solutions))


def _shuffle_sudoku(board: np.ndarray, solution: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a structure-preserving random transformation to a Sudoku puzzle
    and its solution.

    This mirrors the augmentation used in `build_sudoku_dataset.py`:
      - Random permutation of digits 1..9 (0 stays 0).
      - Optional transpose.
      - Random permutation of row bands and column stacks.
    """
    # Create a random digit mapping: a permutation of 1..9, with zero unchanged
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))

    # Randomly decide whether to transpose.
    transpose_flag = np.random.rand() < 0.5

    # Generate a valid row permutation:
    # - Shuffle the 3 bands (each band = 3 rows) and for each band, shuffle its 3 rows.
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # Similarly for columns (stacks).
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    # Build an 81->81 mapping. For each new cell at (i, j)
    # (row index = i // 9, col index = i % 9),
    # its value comes from old row = row_perm[i//9] and old col = col_perm[i%9].
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        g = x
        # Apply transpose flag
        if transpose_flag:
            g = g.T
        # Apply the position mapping.
        new_board = g.flatten()[mapping].reshape(9, 9).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)


def _encode_english_grid(grid: np.ndarray, english_len: int) -> np.ndarray:
    """
    Encode a 9x9 puzzle grid (values 0-9) into a length-english_len vector
    of DSL INT_* token IDs (row-major order, padded with PAD).
    """
    flat = grid.reshape(-1)
    if flat.size > english_len:
        raise ValueError(f"Puzzle grid has {flat.size} cells, exceeds english_len={english_len}")

    out = np.full(english_len, PAD_ID, dtype=np.int32)
    for i, v in enumerate(flat):
        # Map digit 0..9 -> INT_0..INT_9
        out[i] = get_int_const_token(int(v)).value
    return out


def _build_sudoku_program(
    puzzle: np.ndarray,
    solution: np.ndarray,
) -> List[int]:
    """
    Build a DSL program that fills in the missing cells of `puzzle`
    to reach `solution`.

    Program:
        BOS,
          [ INT_row, INT_col, INT_val, SET_CELL ] * K,
          CHECK_SUDOKU,
        EOS
    """
    tokens: List[int] = []
    tokens.append(GyanDSLToken.BOS.value)

    for r in range(9):
        for c in range(9):
            v_in = int(puzzle[r, c])
            v_sol = int(solution[r, c])
            if v_in == 0 and v_sol != 0:
                # Encode (row, col, value) triple as INT_* constants
                tokens.append(get_int_const_token(r).value)
                tokens.append(get_int_const_token(c).value)
                tokens.append(get_int_const_token(v_sol).value)
                tokens.append(GyanDSLToken.SET_CELL.value)

    # Optional final validation
    tokens.append(GyanDSLToken.CHECK_SUDOKU.value)
    tokens.append(GyanDSLToken.EOS.value)

    return tokens


def _convert_to_numpy_arrays(
    examples: List[Tuple[np.ndarray, np.ndarray]],
    config: SudokuDSLConfig,
    split: str,
) -> dict:
    """
    Convert (puzzle, solution) pairs into numpy arrays compatible with PuzzleDataset.
    """
    english_len = config.english_seq_len
    program_len = config.program_seq_len

    english_list: List[np.ndarray] = []
    program_list: List[np.ndarray] = []

    num_aug = config.num_aug if split == "train" else 0

    for puzzle, solution in tqdm(examples, desc="Encoding Sudoku DSL examples"):
        # Original + augmentations
        for aug_idx in range(1 + num_aug):
            if aug_idx == 0:
                p, s = puzzle, solution
            else:
                p, s = _shuffle_sudoku(puzzle, solution)

            # english segment: puzzle grid as INT_* tokens
            eng = _encode_english_grid(p, english_len)
            english_list.append(eng)

            # program segment: DSL program tokens padded to program_len
            prog_tokens = np.array(_build_sudoku_program(p, s), dtype=np.int32)
            if prog_tokens.size > program_len:
                raise ValueError(
                    f"Program length {prog_tokens.size} exceeds configured program_len={program_len}"
                )
            prog = np.full(program_len, PAD_ID, dtype=np.int32)
            prog[: prog_tokens.size] = prog_tokens
            program_list.append(prog)

    english = np.stack(english_list, axis=0)
    program = np.stack(program_list, axis=0)

    n = english.shape[0]

    # Placeholder inputs/labels; rebuilt in PuzzleDataset._collate_batch
    total_seq_len = english_len + program_len
    inputs = np.full((n, total_seq_len), PAD_ID, dtype=np.int32)
    labels = np.full((n, total_seq_len), IGNORE_LABEL_ID, dtype=np.int32)

    # Simple identifiers: one shared puzzle family id
    puzzle_identifiers = np.zeros(n, dtype=np.int32)
    puzzle_indices = np.arange(n + 1, dtype=np.int32)
    group_indices = np.arange(n + 1, dtype=np.int32)

    return {
        "inputs": inputs,
        "labels": labels,
        "english": english,
        "program": program,
        "puzzle_identifiers": puzzle_identifiers,
        "puzzle_indices": puzzle_indices,
        "group_indices": group_indices,
    }


@cli.command(singleton=True)
def main(config: SudokuDSLConfig) -> None:
    """Build Sudoku DSL-program dataset."""
    np.random.seed(42)

    os.makedirs(config.output_dir, exist_ok=True)

    vocab_size = get_vocab_size()
    num_identifiers = 1  # single shared puzzle family

    for split in ["train", "test"]:
        print(f"Loading {split} split from {config.source_repo}...")
        examples = _load_sudoku_split(split, config)
        print(f"  Loaded {len(examples)} puzzles")

        if not examples:
            continue

        print("Encoding to numpy arrays...")
        data = _convert_to_numpy_arrays(examples, config, split)

        split_dir = os.path.join(config.output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        # Save arrays
        for name, arr in data.items():
            path = os.path.join(split_dir, f"all__{name}.npy")
            np.save(path, arr)
            print(f"  [{split}] saved {name}: shape={arr.shape}")

        # Metadata: seq_len is combined english + program
        seq_len_meta = config.english_seq_len + config.program_seq_len
        metadata = PuzzleDatasetMetadata(
            seq_len=seq_len_meta,
            vocab_size=vocab_size,
            pad_id=PAD_ID,
            ignore_label_id=None,  # labels use -100 for ignore
            blank_identifier_id=0,
            num_puzzle_identifiers=num_identifiers,
            total_groups=len(examples),
            mean_puzzle_examples=1.0,
            total_puzzles=len(examples),
            sets=["all"],
        )

        with open(os.path.join(split_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)

    # Save identifier mapping
    id_to_name = {0: "<sudoku>"}  # single family
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump([id_to_name.get(i, "<unknown>") for i in range(num_identifiers)], f, indent=2)

    print(f"\nSudoku DSL-program dataset saved to {config.output_dir}")


if __name__ == "__main__":
    cli()


