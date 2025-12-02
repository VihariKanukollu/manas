"""
Build DSL token dataset in the format expected by TRM's PuzzleDataset.

KEY INSIGHT: TRM is non-autoregressive. It predicts ALL positions simultaneously.
- For Sudoku: input has empty cells, model predicts filled cells
- For DSL: input has answer MASKED, model predicts the answer tokens

The answer is identified using the `answer` field in the JSONL:
- Parse the answer to expected token pattern
- Find those tokens at the end of the sequence (before EOS)
- Mask those positions in input, only compute loss on those positions

Usage:
    python -m dataset.build_dsl_dataset \
        --input-dir data/math_dsl_test \
        --output-dir data/dsl_trm
"""

import os
import json
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict

from argdantic import ArgParser
from pydantic import BaseModel

from dataset.common import PuzzleDatasetMetadata

# Import DSL tokens
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsl.tokens import get_vocab_size, GyanDSLToken

# Special token IDs
PAD_ID = GyanDSLToken.PAD.value
BOS_ID = GyanDSLToken.BOS.value
EOS_ID = GyanDSLToken.EOS.value

# Labels to ignore in loss computation
IGNORE_LABEL_ID = -100

cli = ArgParser()


class DSLDatasetConfig(BaseModel):
    input_dir: str
    output_dir: str
    seq_len: int = 128
    seed: int = 42
    group_size: int = 50
    # How many tokens at end to treat as answer (before EOS)
    answer_window: int = 5


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
    return {mod: i + 1 for i, mod in enumerate(modules)}


def parse_answer_to_tokens(answer: str) -> List[str]:
    """
    Parse answer string to expected token names.
    
    Examples:
        "37" -> ["INT_37"]
        "-21" -> ["INT_0", "INT_21", "SUB"] (postfix for 0 - 21)
        "2" -> ["INT_2"]
        "True" -> ["BOOL_TRUE"]
        "False" -> ["BOOL_FALSE"]
    """
    answer = answer.strip()
    
    # Boolean answers
    if answer.lower() == "true":
        return ["BOOL_TRUE"]
    if answer.lower() == "false":
        return ["BOOL_FALSE"]
    
    # Try to parse as integer
    try:
        val = int(answer)
        if 0 <= val < 100:
            return [f"INT_{val}"]
        elif val == -1:
            return ["INT_NEG1"]
        elif val == -2:
            return ["INT_NEG2"]
        elif val == -10:
            return ["INT_NEG10"]
        elif val == -100:
            return ["INT_NEG100"]
        elif val < 0:
            # Negative: represented as 0 - |val| in postfix
            abs_val = abs(val)
            if abs_val < 100:
                return ["INT_0", f"INT_{abs_val}", "SUB"]
            else:
                # Complex negative, skip for now
                return []
        else:
            # Large positive, might be factored
            return []
    except ValueError:
        pass
    
    # Fraction like "2/5"
    if "/" in answer:
        parts = answer.split("/")
        if len(parts) == 2:
            try:
                num, den = int(parts[0]), int(parts[1])
                if 0 <= num < 100 and 0 < den < 100:
                    return [f"INT_{num}", f"INT_{den}", "DIV"]
            except ValueError:
                pass
    
    # Can't parse - return empty
    return []


def find_answer_tokens_in_sequence(
    token_names: List[str],
    answer_tokens: List[str],
) -> Tuple[Optional[int], Optional[int]]:
    """
    Find where answer tokens appear in the sequence.
    
    Returns (start, end) indices or (None, None) if not found.
    """
    if not answer_tokens:
        return None, None
    
    # Search from end (answer is typically near the end)
    seq_len = len(token_names)
    pattern_len = len(answer_tokens)
    
    # Search in the last portion of sequence (before EOS)
    search_end = seq_len - 1 if token_names[-1] == "EOS" else seq_len
    search_start = max(0, search_end - 20)  # Look in last 20 tokens
    
    for i in range(search_end - pattern_len, search_start - 1, -1):
        if token_names[i:i + pattern_len] == answer_tokens:
            return i, i + pattern_len
    
    return None, None


def find_answer_by_heuristic(
    token_ids: List[int],
    token_names: List[str],
    answer_window: int,
) -> Tuple[int, int]:
    """
    Heuristic: answer is the last `answer_window` tokens before EOS,
    excluding common structural tokens.
    
    Returns (start, end) indices.
    """
    # Find EOS position
    try:
        eos_pos = token_names.index("EOS") if "EOS" in token_names else len(token_names)
    except ValueError:
        eos_pos = len(token_names)
    
    # Structural tokens that are NOT part of the answer value
    structural = {"EOS", "BOS", "IS_SOLUTION", "EQ", "EQ_CMP"}
    
    # Work backwards from EOS to find answer span
    end = eos_pos
    
    # Skip structural tokens at the end
    while end > 0 and token_names[end - 1] in structural:
        end -= 1
    
    # Take up to answer_window tokens
    start = max(0, end - answer_window)
    
    # But don't go past structural tokens going backwards
    while start < end and token_names[start] in structural:
        start += 1
    
    return start, end


def create_input_label_pair(
    token_ids: List[int],
    token_names: List[str],
    answer: str,
    seq_len: int,
    answer_window: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Create (input, label) pair for a single example.
    
    Returns (input_seq, label_seq, num_answer_tokens)
    """
    # Try to find answer using parsed tokens first
    answer_tokens = parse_answer_to_tokens(answer)
    ans_start, ans_end = find_answer_tokens_in_sequence(token_names, answer_tokens)
    
    # If not found, use heuristic
    if ans_start is None:
        ans_start, ans_end = find_answer_by_heuristic(
            token_ids, token_names, answer_window
        )
    
    # Ensure valid range
    if ans_start >= ans_end:
        ans_start, ans_end = 0, 0
    
    # Pad or truncate
    if len(token_ids) > seq_len:
        token_ids = token_ids[:seq_len-1] + [EOS_ID]
        token_names = token_names[:seq_len-1] + ["EOS"]
        # Recompute if truncated
        if ans_end > seq_len:
            ans_start, ans_end = find_answer_by_heuristic(
                token_ids, token_names, answer_window
            )
    
    # Create padded sequences
    padded_ids = [PAD_ID] * seq_len
    padded_ids[:len(token_ids)] = token_ids
    
    # Input: mask answer positions
    input_seq = padded_ids.copy()
    for i in range(ans_start, min(ans_end, seq_len)):
        input_seq[i] = PAD_ID
    
    # Label: only answer positions
    label_seq = [IGNORE_LABEL_ID] * seq_len
    for i in range(ans_start, min(ans_end, seq_len)):
        label_seq[i] = padded_ids[i]
    
    num_answer = min(ans_end, seq_len) - ans_start
    return (
        np.array(input_seq, dtype=np.int32),
        np.array(label_seq, dtype=np.int32),
        num_answer
    )


def convert_split(
    examples: List[Dict[str, Any]],
    module_to_id: Dict[str, int],
    seq_len: int,
    group_size: int,
    answer_window: int,
    is_train: bool,
) -> Dict[str, np.ndarray]:
    """Convert examples to numpy arrays for a single split."""
    
    by_module: Dict[str, List[Dict]] = defaultdict(list)
    for ex in examples:
        by_module[ex["module"]].append(ex)
    
    all_inputs = []
    all_labels = []
    all_puzzle_ids = []
    puzzle_indices = [0]
    group_indices = [0]
    
    puzzle_count = 0
    skipped = 0
    total_answer_tokens = 0
    
    for module_name in sorted(by_module.keys()):
        module_examples = by_module[module_name]
        module_id = module_to_id[module_name]
        
        if is_train:
            np.random.shuffle(module_examples)
        
        for i, ex in enumerate(module_examples):
            token_ids = ex["token_ids"]
            token_names = ex.get("token_names", [])
            answer = ex.get("answer", "")
            
            input_seq, label_seq, num_answer = create_input_label_pair(
                token_ids, token_names, answer, seq_len, answer_window
            )
            
            # Skip if no answer found
            if num_answer == 0:
                skipped += 1
                continue
            
            all_inputs.append(input_seq)
            all_labels.append(label_seq)
            all_puzzle_ids.append(module_id)
            total_answer_tokens += num_answer
            
            puzzle_count += 1
            puzzle_indices.append(puzzle_count)
            
            if is_train and (i + 1) % group_size == 0:
                group_indices.append(puzzle_count)
        
        if is_train and (len(group_indices) == 0 or group_indices[-1] != puzzle_count):
            group_indices.append(puzzle_count)
    
    if not is_train:
        group_indices = list(range(puzzle_count + 1))
    
    if skipped > 0:
        print(f"  Skipped {skipped} examples")
    
    avg_answer_len = total_answer_tokens / max(puzzle_count, 1)
    print(f"  Avg answer length: {avg_answer_len:.1f} tokens")
    
    return {
        "inputs": np.stack(all_inputs, axis=0) if all_inputs else np.zeros((0, seq_len), dtype=np.int32),
        "labels": np.stack(all_labels, axis=0) if all_labels else np.zeros((0, seq_len), dtype=np.int32),
        "puzzle_identifiers": np.array(all_puzzle_ids, dtype=np.int32),
        "puzzle_indices": np.array(puzzle_indices, dtype=np.int32),
        "group_indices": np.array(group_indices, dtype=np.int32),
    }


def build_dataset(config: DSLDatasetConfig):
    """Build the full dataset."""
    np.random.seed(config.seed)
    
    vocab_size = get_vocab_size()
    
    print(f"DSL vocab size: {vocab_size}")
    print(f"Sequence length: {config.seq_len}")
    print(f"Answer window: {config.answer_window}")
    
    # Load data
    train_examples = load_jsonl(os.path.join(config.input_dir, "train.jsonl"))
    test_id_examples = load_jsonl(os.path.join(config.input_dir, "test_id.jsonl"))
    test_ood_examples = load_jsonl(os.path.join(config.input_dir, "test_ood.jsonl"))
    
    all_examples = train_examples + test_id_examples + test_ood_examples
    module_to_id = build_module_mapping(all_examples)
    num_modules = len(module_to_id) + 1
    
    print(f"Number of modules: {len(module_to_id)}")
    print(f"Train examples: {len(train_examples)}")
    print(f"Test examples: {len(test_id_examples) + len(test_ood_examples)}")
    
    # Test answer detection
    print("\nAnswer detection samples:")
    for ex in train_examples[:5]:
        answer = ex.get("answer", "")
        answer_tokens = parse_answer_to_tokens(answer)
        token_names = ex.get("token_names", [])
        start, end = find_answer_tokens_in_sequence(token_names, answer_tokens)
        if start is None:
            start, end = find_answer_by_heuristic(ex["token_ids"], token_names, config.answer_window)
        found_names = token_names[start:end] if token_names else []
        print(f"  {ex['module'][:25]:25} answer='{answer:8}' -> tokens {found_names}")
    
    # Convert splits
    splits = {
        "train": (train_examples, True),
        "test": (test_id_examples + test_ood_examples, False),
    }
    
    for split_name, (examples, is_train) in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        os.makedirs(os.path.join(config.output_dir, split_name), exist_ok=True)
        
        data = convert_split(
            examples,
            module_to_id,
            config.seq_len,
            config.group_size,
            config.answer_window,
            is_train
        )
        
        for name, arr in data.items():
            filepath = os.path.join(config.output_dir, split_name, f"all__{name}.npy")
            np.save(filepath, arr)
            print(f"  Saved {name}: shape={arr.shape}")
        
        num_examples = len(data["inputs"])
        num_groups = len(data["group_indices"]) - 1
        
        print(f"  Examples: {num_examples}, Groups: {num_groups}")
        
        metadata = PuzzleDatasetMetadata(
            seq_len=config.seq_len,
            vocab_size=vocab_size,
            pad_id=PAD_ID,
            ignore_label_id=None,
            blank_identifier_id=0,
            num_puzzle_identifiers=num_modules,
            total_groups=num_groups,
            mean_puzzle_examples=1.0,
            total_puzzles=num_examples,
            sets=["all"],
        )
        
        with open(os.path.join(config.output_dir, split_name, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)
    
    # Save mappings
    with open(os.path.join(config.output_dir, "module_mapping.json"), "w") as f:
        json.dump(module_to_id, f, indent=2)
    
    id_to_module = {0: "<blank>"}
    id_to_module.update({v: k for k, v in module_to_id.items()})
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump([id_to_module.get(i, "<unknown>") for i in range(num_modules)], f)
    
    print(f"\nDataset saved to {config.output_dir}")
    print("Task: Given DSL sequence with answer masked, predict the answer tokens")


@cli.command(singleton=True)
def main(config: DSLDatasetConfig):
    build_dataset(config)


if __name__ == "__main__":
    cli()
