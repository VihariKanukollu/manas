"""
Build DSL token dataset in the format expected by TRM's PuzzleDataset.

Converts JSONL files (from dev/gen_full_math.py) to numpy arrays:
- inputs: (N, seq_len) - DSL token sequences
- labels: (N, seq_len) - Same as inputs (TRM learns to refine/reproduce)
- puzzle_identifiers: (num_puzzles,) - Module ID for each example
- puzzle_indices: Cumulative indices into inputs/labels
- group_indices: Cumulative indices for groups (used for batching related examples)

Usage:
    python -m dataset.build_dsl_dataset \
        --input_dir data/math_dsl_test \
        --output_dir data/dsl_trm
"""

import os
import json
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict

from argdantic import ArgParser
from pydantic import BaseModel

from dataset.common import PuzzleDatasetMetadata

# Import DSL vocab size
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsl.tokens import get_vocab_size, GyanDSLToken


cli = ArgParser()


class DSLDatasetConfig(BaseModel):
    input_dir: str  # Directory with train.jsonl, test_id.jsonl, test_ood.jsonl
    output_dir: str
    seq_len: int = 128  # Max sequence length (pad/truncate to this)
    seed: int = 42
    # Group size for training batching - examples from same module grouped together
    group_size: int = 50


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    examples = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def build_module_mapping(examples: List[Dict[str, Any]]) -> Dict[str, int]:
    """Build mapping from module name to integer ID."""
    modules = sorted(set(ex["module"] for ex in examples))
    # ID 0 is reserved for blank/padding
    return {mod: i + 1 for i, mod in enumerate(modules)}


def convert_split(
    examples: List[Dict[str, Any]],
    module_to_id: Dict[str, int],
    seq_len: int,
    group_size: int,
    is_train: bool,
) -> Dict[str, np.ndarray]:
    """Convert examples to numpy arrays for a single split."""
    
    pad_id = GyanDSLToken.PAD.value
    
    # Group examples by module for better batching
    by_module: Dict[str, List[Dict]] = defaultdict(list)
    for ex in examples:
        by_module[ex["module"]].append(ex)
    
    # Build arrays
    all_inputs = []
    all_labels = []
    all_puzzle_ids = []
    puzzle_indices = [0]
    group_indices = [0]
    
    puzzle_count = 0
    
    # Process each module
    for module_name in sorted(by_module.keys()):
        module_examples = by_module[module_name]
        module_id = module_to_id[module_name]
        
        # Split into groups for training
        if is_train:
            # Shuffle within module
            np.random.shuffle(module_examples)
        
        # Process examples
        for i, ex in enumerate(module_examples):
            token_ids = ex["token_ids"]
            
            # Pad or truncate to seq_len
            if len(token_ids) > seq_len:
                # Truncate (keep BOS at start, try to keep EOS at end)
                token_ids = token_ids[:seq_len-1] + [GyanDSLToken.EOS.value]
            elif len(token_ids) < seq_len:
                # Pad with PAD token
                token_ids = token_ids + [pad_id] * (seq_len - len(token_ids))
            
            all_inputs.append(token_ids)
            all_labels.append(token_ids)  # Labels same as inputs for TRM
            all_puzzle_ids.append(module_id)
            
            puzzle_count += 1
            puzzle_indices.append(puzzle_count)
            
            # Create groups every group_size examples (or at module boundary)
            if is_train and (i + 1) % group_size == 0:
                group_indices.append(puzzle_count)
        
        # End of module - create group boundary if not already
        if is_train and (group_indices[-1] != puzzle_count):
            group_indices.append(puzzle_count)
    
    # For test, each example is its own group
    if not is_train:
        group_indices = list(range(puzzle_count + 1))
    
    return {
        "inputs": np.array(all_inputs, dtype=np.int32),
        "labels": np.array(all_labels, dtype=np.int32),
        "puzzle_identifiers": np.array(all_puzzle_ids, dtype=np.int32),
        "puzzle_indices": np.array(puzzle_indices, dtype=np.int32),
        "group_indices": np.array(group_indices, dtype=np.int32),
    }


def build_dataset(config: DSLDatasetConfig):
    """Build the full dataset."""
    np.random.seed(config.seed)
    
    vocab_size = get_vocab_size()
    pad_id = GyanDSLToken.PAD.value
    
    print(f"DSL vocab size: {vocab_size}")
    print(f"Sequence length: {config.seq_len}")
    
    # Load all data to build module mapping
    train_examples = load_jsonl(os.path.join(config.input_dir, "train.jsonl"))
    test_id_examples = load_jsonl(os.path.join(config.input_dir, "test_id.jsonl"))
    test_ood_examples = load_jsonl(os.path.join(config.input_dir, "test_ood.jsonl"))
    
    all_examples = train_examples + test_id_examples + test_ood_examples
    module_to_id = build_module_mapping(all_examples)
    num_modules = len(module_to_id) + 1  # +1 for blank ID 0
    
    print(f"Number of modules: {len(module_to_id)}")
    print(f"Train examples: {len(train_examples)}")
    print(f"Test ID examples: {len(test_id_examples)}")
    print(f"Test OOD examples: {len(test_ood_examples)}")
    
    # Convert each split
    splits = {
        "train": (train_examples, True),
        "test": (test_id_examples + test_ood_examples, False),  # Combine test sets
    }
    
    for split_name, (examples, is_train) in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        os.makedirs(os.path.join(config.output_dir, split_name), exist_ok=True)
        
        data = convert_split(
            examples, 
            module_to_id, 
            config.seq_len, 
            config.group_size,
            is_train
        )
        
        # Save numpy arrays
        for name, arr in data.items():
            filepath = os.path.join(config.output_dir, split_name, f"all__{name}.npy")
            np.save(filepath, arr)
            print(f"  Saved {name}: shape={arr.shape}, dtype={arr.dtype}")
        
        # Compute statistics
        num_examples = len(data["inputs"])
        num_puzzles = len(data["puzzle_identifiers"])
        num_groups = len(data["group_indices"]) - 1
        
        # Save metadata
        metadata = PuzzleDatasetMetadata(
            seq_len=config.seq_len,
            vocab_size=vocab_size,
            pad_id=pad_id,
            ignore_label_id=pad_id,  # Ignore PAD tokens in loss
            blank_identifier_id=0,
            num_puzzle_identifiers=num_modules,
            total_groups=num_groups,
            mean_puzzle_examples=num_examples / max(num_puzzles, 1),
            total_puzzles=num_puzzles,
            sets=["all"],
        )
        
        with open(os.path.join(config.output_dir, split_name, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)
        
        print(f"  Examples: {num_examples}, Puzzles: {num_puzzles}, Groups: {num_groups}")
    
    # Save module mapping
    with open(os.path.join(config.output_dir, "module_mapping.json"), "w") as f:
        json.dump(module_to_id, f, indent=2)
    
    # Save identifiers list (for compatibility with existing code)
    id_to_module = {0: "<blank>"}
    id_to_module.update({v: k for k, v in module_to_id.items()})
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump([id_to_module.get(i, "<unknown>") for i in range(num_modules)], f)
    
    print(f"\nDataset saved to {config.output_dir}")


@cli.command(singleton=True)
def main(config: DSLDatasetConfig):
    build_dataset(config)


if __name__ == "__main__":
    cli()

