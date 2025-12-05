#!/usr/bin/env python
"""
Build English→DSL TRM dataset.

This creates a TRM-format dataset where:
- Input: English text encoded as CHAR_* tokens + PAD + [MASK for DSL]
- Labels: -100 for English positions, DSL tokens for output positions

The model learns to "solve the puzzle": given English chars, produce DSL tokens.

Usage:
    python -m dataset.build_english_trm_dataset \
        --src data/en_dsl_seq2seq/train.jsonl \
        --dst data/english_to_dsl_trm/train \
        --seq_len 256
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dsl.tokens import GyanDSLToken, text_to_token_ids, get_vocab_size

# Special token IDs from our vocabulary
PAD_ID = GyanDSLToken.PAD.value
BOS_ID = GyanDSLToken.BOS.value
EOS_ID = GyanDSLToken.EOS.value
IGNORE_LABEL_ID = -100


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load examples from JSONL file."""
    examples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def encode_english(text: str, max_len: int) -> Tuple[List[int], int]:
    """
    Encode English text as CHAR_* token IDs.
    
    Returns:
        (token_ids, actual_length)
    """
    token_ids = text_to_token_ids(text)
    actual_len = len(token_ids)
    
    # Truncate if needed
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
        actual_len = max_len
    
    return token_ids, actual_len


def encode_dsl(token_names: List[str]) -> List[int]:
    """
    Encode DSL token names to IDs.
    
    Args:
        token_names: List like ["BOS", "EN_EVT_INIT", "EN_AMOUNT", "INT_24", ...]
    
    Returns:
        List of token IDs
    """
    ids = []
    for name in token_names:
        try:
            tok = GyanDSLToken[name]
            ids.append(tok.value)
        except KeyError:
            print(f"Warning: Unknown DSL token '{name}', skipping")
            continue
    return ids


def build_example(
    question: str,
    en_tokens: List[str],
    seq_len: int,
    english_max_len: int,
) -> Tuple[np.ndarray, np.ndarray] | None:
    """
    Build a single training example.
    
    Format:
        Input:  [CHAR_...] [PAD] [PAD...for DSL output]
        Labels: [-100...]  [-100] [DSL tokens...]
    
    Returns:
        (inputs, labels) arrays of shape (seq_len,), or None if invalid
    """
    # Encode English as CHAR tokens
    eng_ids, eng_len = encode_english(question, english_max_len)
    
    # Encode DSL output
    dsl_ids = encode_dsl(en_tokens)
    dsl_len = len(dsl_ids)
    
    # Check if it fits
    # Layout: [English] [SEP/PAD] [DSL output] [PAD...]
    # We use one PAD as separator between English and DSL
    total_needed = eng_len + 1 + dsl_len
    if total_needed > seq_len:
        print(f"Warning: Example too long ({total_needed} > {seq_len}), skipping")
        return None
    
    # Build input array
    inputs = np.full(seq_len, PAD_ID, dtype=np.int32)
    inputs[:eng_len] = eng_ids
    # Position eng_len is PAD (separator)
    # DSL output positions are also PAD (masked - model must predict)
    
    # Build labels array
    labels = np.full(seq_len, IGNORE_LABEL_ID, dtype=np.int32)
    # English positions: ignore
    # Separator position: ignore
    # DSL output positions: actual DSL tokens
    dsl_start = eng_len + 1
    dsl_end = dsl_start + dsl_len
    labels[dsl_start:dsl_end] = dsl_ids
    
    return inputs, labels


def build_dataset(
    src_path: Path,
    dst_dir: Path,
    seq_len: int = 256,
    english_max_len: int = 150,
) -> None:
    """
    Build the full dataset.
    
    Args:
        src_path: Path to source JSONL file
        dst_dir: Output directory for numpy arrays
        seq_len: Total sequence length
        english_max_len: Max tokens for English input
    """
    examples = load_jsonl(src_path)
    print(f"Loaded {len(examples)} examples from {src_path}")
    
    all_inputs = []
    all_labels = []
    all_puzzle_ids = []
    all_puzzle_indices = []
    all_group_indices = []
    
    # Build module -> puzzle_id mapping
    modules = sorted(set(ex.get("module", "unknown") for ex in examples))
    module_to_id = {m: i + 1 for i, m in enumerate(modules)}  # 0 reserved for blank
    
    for idx, ex in enumerate(examples):
        question = ex.get("question", "")
        en_tokens = ex.get("en_tokens", [])
        module = ex.get("module", "unknown")
        
        if not question or not en_tokens:
            print(f"Skipping example {idx}: missing question or en_tokens")
            continue
        
        result = build_example(question, en_tokens, seq_len, english_max_len)
        if result is None:
            continue
        
        inputs, labels = result
        all_inputs.append(inputs)
        all_labels.append(labels)
        all_puzzle_ids.append(module_to_id.get(module, 0))
        all_puzzle_indices.append(len(all_inputs) - 1)
        all_group_indices.append(len(all_inputs) - 1)
    
    if not all_inputs:
        print("No valid examples!")
        return
    
    # Convert to numpy arrays
    inputs_arr = np.stack(all_inputs)
    labels_arr = np.stack(all_labels)
    puzzle_ids_arr = np.array(all_puzzle_ids, dtype=np.int32)
    puzzle_indices_arr = np.array(all_puzzle_indices, dtype=np.int32)
    group_indices_arr = np.array(all_group_indices, dtype=np.int32)
    
    # Save
    dst_dir.mkdir(parents=True, exist_ok=True)
    np.save(dst_dir / "all__inputs.npy", inputs_arr)
    np.save(dst_dir / "all__labels.npy", labels_arr)
    np.save(dst_dir / "all__puzzle_identifiers.npy", puzzle_ids_arr)
    np.save(dst_dir / "all__puzzle_indices.npy", puzzle_indices_arr)
    np.save(dst_dir / "all__group_indices.npy", group_indices_arr)
    
    # Save metadata
    metadata = {
        "pad_id": PAD_ID,
        "ignore_label_id": IGNORE_LABEL_ID,
        "blank_identifier_id": 0,
        "vocab_size": get_vocab_size(),
        "seq_len": seq_len,
        "num_puzzle_identifiers": len(modules) + 1,  # +1 for blank
        "total_groups": len(all_inputs),
        "total_puzzles": len(all_inputs),
        "mean_puzzle_examples": 1.0,
        "sets": ["all"],
    }
    with open(dst_dir / "dataset.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save module mapping
    with open(dst_dir.parent / "module_mapping.json", "w") as f:
        json.dump(module_to_id, f, indent=2)
    
    # Save identifiers
    id_to_module = {v: k for k, v in module_to_id.items()}
    id_to_module[0] = "blank"
    with open(dst_dir.parent / "identifiers.json", "w") as f:
        json.dump(id_to_module, f, indent=2)
    
    print(f"\nDataset built successfully!")
    print(f"  Examples: {len(all_inputs)}")
    print(f"  Seq length: {seq_len}")
    print(f"  Vocab size: {get_vocab_size()}")
    print(f"  Modules: {modules}")
    print(f"  Output: {dst_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build English→DSL TRM dataset")
    parser.add_argument(
        "--src",
        type=str,
        default="data/en_dsl_seq2seq/train.jsonl",
        help="Source JSONL file with question + en_tokens",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="data/english_to_dsl_trm/train",
        help="Output directory for numpy arrays",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=256,
        help="Total sequence length",
    )
    parser.add_argument(
        "--english_max_len",
        type=int,
        default=150,
        help="Max tokens for English input",
    )
    args = parser.parse_args()
    
    build_dataset(
        src_path=Path(args.src),
        dst_dir=Path(args.dst),
        seq_len=args.seq_len,
        english_max_len=args.english_max_len,
    )


if __name__ == "__main__":
    main()

