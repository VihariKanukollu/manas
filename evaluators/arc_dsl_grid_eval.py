#!/usr/bin/env python3
"""
ARC-DSL Grid-Level Evaluator

This script loads a trained checkpoint, runs inference on ARC-DSL examples,
executes the predicted DSL programs via a postfix interpreter, and compares
the output grids to ground truth.

Reports:
  - Program EM (exact token match)
  - Grid accuracy (output grid matches target)

Usage:
    python -m evaluators.arc_dsl_grid_eval \
        --checkpoint path/to/step_XXXXX \
        --data_path data/arc_dsl_trm \
        --num_examples 100
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dsl import dsl as dsl_funcs
from dsl.tokens import GyanDSLToken, get_vocab_size
from dataset.common import PuzzleDatasetMetadata
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


# ============================================================================
# Postfix Interpreter for ARC DSL
# ============================================================================

# Map token names to DSL functions
DSL_FUNC_MAP: Dict[str, Callable] = {}

def _build_func_map():
    """Build mapping from token names to dsl.py functions."""
    global DSL_FUNC_MAP
    if DSL_FUNC_MAP:
        return
    
    # Get all functions from dsl module
    for name in dir(dsl_funcs):
        if not name.startswith('_'):
            obj = getattr(dsl_funcs, name)
            if callable(obj):
                DSL_FUNC_MAP[name.upper()] = obj
    
    # Add some aliases
    DSL_FUNC_MAP['HMIRROR'] = dsl_funcs.hmirror
    DSL_FUNC_MAP['VMIRROR'] = dsl_funcs.vmirror
    DSL_FUNC_MAP['DMIRROR'] = dsl_funcs.dmirror
    DSL_FUNC_MAP['CMIRROR'] = dsl_funcs.cmirror
    DSL_FUNC_MAP['ROT90'] = dsl_funcs.rot90
    DSL_FUNC_MAP['ROT180'] = dsl_funcs.rot180
    DSL_FUNC_MAP['ROT270'] = dsl_funcs.rot270

_build_func_map()


class PostfixInterpreterError(Exception):
    pass


def execute_postfix_program(
    tokens: List[int],
    input_grid: Tuple[Tuple[int, ...], ...],
) -> Tuple[Tuple[int, ...], ...]:
    """
    Execute a postfix (RPN) DSL program on an input grid.
    
    Args:
        tokens: List of token IDs (from model prediction or ground truth)
        input_grid: The ARC input grid as tuple of tuples
        
    Returns:
        The output grid as tuple of tuples
        
    Raises:
        PostfixInterpreterError if execution fails
    """
    id_to_token = {t.value: t for t in GyanDSLToken}
    stack: List[Any] = []
    variables: Dict[int, Any] = {0: input_grid}  # REAL_VAR_0 = input
    next_var = 1
    
    PAD_ID = GyanDSLToken.PAD.value
    BOS_ID = GyanDSLToken.BOS.value
    EOS_ID = GyanDSLToken.EOS.value
    
    for tid in tokens:
        if tid in (PAD_ID, BOS_ID, EOS_ID):
            continue
            
        token = id_to_token.get(tid)
        if token is None:
            raise PostfixInterpreterError(f"Unknown token ID: {tid}")
        
        name = token.name
        
        # Handle integer constants
        if name.startswith("INT_"):
            if name == "INT_NEG1":
                stack.append(-1)
            elif name == "INT_NEG2":
                stack.append(-2)
            elif name == "INT_NEG10":
                stack.append(-10)
            elif name == "INT_NEG100":
                stack.append(-100)
            else:
                val = int(name.split("_", 1)[1])
                stack.append(val)
            continue
        
        # Handle booleans
        if name == "BOOL_TRUE":
            stack.append(True)
            continue
        if name == "BOOL_FALSE":
            stack.append(False)
            continue
        
        # Handle variable references
        if name.startswith("REAL_VAR_"):
            var_id = int(name.split("_")[-1])
            if var_id in variables:
                stack.append(variables[var_id])
            else:
                raise PostfixInterpreterError(f"Undefined variable: {name}")
            continue
        
        # Handle DSL functions
        func_name = name.lower()
        if func_name in DSL_FUNC_MAP or name in DSL_FUNC_MAP:
            func = DSL_FUNC_MAP.get(func_name) or DSL_FUNC_MAP.get(name)
            
            # Determine arity by inspection or trial
            import inspect
            try:
                sig = inspect.signature(func)
                arity = len([p for p in sig.parameters.values() 
                            if p.default == inspect.Parameter.empty])
            except:
                arity = 1  # Default guess
            
            if len(stack) < arity:
                raise PostfixInterpreterError(
                    f"Not enough args for {name}: need {arity}, have {len(stack)}"
                )
            
            args = [stack.pop() for _ in range(arity)][::-1]
            
            try:
                result = func(*args)
                stack.append(result)
                # Also assign to next variable slot
                variables[next_var] = result
                next_var += 1
            except Exception as e:
                raise PostfixInterpreterError(f"Error executing {name}: {e}")
            continue
        
        # Unknown token - skip or error
        # For robustness, we skip unknown tokens
        pass
    
    if not stack:
        raise PostfixInterpreterError("Empty stack after execution")
    
    result = stack[-1]
    
    # Convert to tuple of tuples if needed
    if isinstance(result, (list, tuple)):
        return tuple(tuple(row) for row in result)
    
    raise PostfixInterpreterError(f"Result is not a grid: {type(result)}")


# ============================================================================
# Model Loading and Inference
# ============================================================================

def load_model_for_eval(checkpoint_path: str, metadata: PuzzleDatasetMetadata, batch_size: int = 16):
    """Load model from checkpoint for evaluation."""
    from omegaconf import OmegaConf
    import hydra
    
    # Load config from checkpoint directory (config is at run level, not step level)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, "all_config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    with open(config_path) as f:
        import yaml
        config = yaml.safe_load(f)
    
    # Build model
    arch_config = config['arch']
    arch_name, arch_class = arch_config['name'].split('@')
    
    # Import model class dynamically
    module_path = arch_name.replace('/', '.')
    import importlib
    module = importlib.import_module(f"models.{module_path}")
    model_class = getattr(module, arch_class)
    
    # Instantiate model - build config dict
    config_dict = {k: v for k, v in arch_config.items() if k not in ('name', 'loss')}
    config_dict['vocab_size'] = metadata.vocab_size
    config_dict['num_puzzle_identifiers'] = metadata.num_puzzle_identifiers
    config_dict['seq_len'] = metadata.seq_len
    config_dict['batch_size'] = batch_size
    
    model = model_class(config_dict=config_dict)
    
    # Load weights
    state_dict = torch.load(checkpoint_path, map_location="cuda")
    
    # Handle compiled model prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("_orig_mod.", "")
        new_state_dict[new_k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model = model.cuda().eval()
    
    return model


def move_carry_to_device(carry, device):
    """Recursively move carry dataclass to device."""
    from dataclasses import fields, is_dataclass
    
    new_values = {}
    for field in fields(carry):
        val = getattr(carry, field.name)
        if isinstance(val, torch.Tensor):
            new_values[field.name] = val.to(device)
        elif isinstance(val, dict):
            new_values[field.name] = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                       for k, v in val.items()}
        elif is_dataclass(val):
            new_values[field.name] = move_carry_to_device(val, device)
        else:
            new_values[field.name] = val
    
    return type(carry)(**new_values)


def run_inference(model, batch: Dict[str, torch.Tensor], max_steps: int = 16) -> torch.Tensor:
    """Run inference and return predicted token IDs.
    
    The model uses a stateful approach with carry objects.
    We run for max_steps iterations and return final predictions.
    """
    with torch.no_grad():
        # Move batch to GPU
        device = next(model.parameters()).device
        batch_gpu = {k: v.to(device) for k, v in batch.items()}
        
        # Initialize carry and move to GPU
        carry = model.initial_carry(batch_gpu)
        carry = move_carry_to_device(carry, device)
        
        # Run for max_steps
        for step in range(max_steps):
            carry, outputs = model(carry, batch_gpu)
        
        # Get predictions (argmax over vocab)
        logits = outputs['logits']
        preds = logits.argmax(dim=-1)
        
    return preds.cpu()


# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ARC-DSL Grid-Level Evaluator")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (e.g., checkpoints/.../step_XXXXX)")
    parser.add_argument("--data_path", type=str, default="data/arc_dsl_trm",
                        help="Path to ARC-DSL dataset")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to evaluate (train or test)")
    parser.add_argument("--num_examples", type=int, default=None,
                        help="Max number of examples to evaluate (None = all)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for inference")
    parser.add_argument("--english_len", type=int, default=900,
                        help="Length of grid/english segment")
    parser.add_argument("--program_len", type=int, default=256,
                        help="Length of program segment")
    args = parser.parse_args()
    
    print(f"Loading dataset from {args.data_path}...")
    
    # Load metadata
    meta_path = os.path.join(args.data_path, args.split, "dataset.json")
    with open(meta_path) as f:
        meta_dict = json.load(f)
    metadata = PuzzleDatasetMetadata(**meta_dict)
    
    print(f"  seq_len: {metadata.seq_len}")
    print(f"  vocab_size: {metadata.vocab_size}")
    print(f"  num_puzzles: {metadata.total_puzzles}")
    
    # Load ARC puzzles for grid comparison
    from dataset.build_arc_dsl_dataset import load_arc_puzzles, get_all_dsl_solvers
    puzzles = load_arc_puzzles("kaggle/combined/arc-agi", ["training", "evaluation"])
    dsl_programs = get_all_dsl_solvers()
    
    # Load identifier mapping
    id_path = os.path.join(args.data_path, "identifiers.json")
    with open(id_path) as f:
        id_to_name = json.load(f)
    
    print(f"\nLoading model from {args.checkpoint}...")
    model = load_model_for_eval(args.checkpoint, metadata, batch_size=args.batch_size)
    print("  Model loaded.")
    
    # Create dataloader
    cfg = PuzzleDatasetConfig(
        seed=42,
        dataset_paths=[args.data_path],
        global_batch_size=args.batch_size,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )
    dataset = PuzzleDataset(cfg, split=args.split)
    
    # Evaluation stats
    total = 0
    program_em_correct = 0
    grid_correct = 0
    exec_errors = 0
    
    PAD_ID = GyanDSLToken.PAD.value
    IGNORE_ID = -100
    
    print(f"\nEvaluating on {args.split} split...")
    
    for set_name, batch, _ in dataset:
        if args.num_examples and total >= args.num_examples:
            break
        
        # Run inference
        preds = run_inference(model, batch)
        
        labels = batch['labels']
        puzzle_ids = batch['puzzle_identifiers']
        inputs = batch['inputs']
        
        # Process each example in batch
        for i in range(preds.shape[0]):
            if args.num_examples and total >= args.num_examples:
                break
            
            # Extract program segment
            prog_preds = preds[i, args.english_len:].numpy()
            prog_labels = labels[i, args.english_len:].numpy()
            grid_tokens = inputs[i, :args.english_len].numpy()
            
            # Mask out PAD and IGNORE
            mask = (prog_labels != IGNORE_ID) & (prog_labels != PAD_ID)
            
            if not mask.any():
                continue
            
            total += 1
            
            # Program EM
            pred_masked = prog_preds[mask]
            label_masked = prog_labels[mask]
            em = (pred_masked == label_masked).all()
            if em:
                program_em_correct += 1
            
            # Grid accuracy - execute predicted program
            puzzle_id = puzzle_ids[i].item()
            puzzle_name = id_to_name[puzzle_id] if puzzle_id < len(id_to_name) else None
            
            if puzzle_name and puzzle_name in puzzles:
                puzzle = puzzles[puzzle_name]
                # Get first train example's input/output
                train_ex = puzzle["train"][0]
                input_grid = tuple(tuple(row) for row in train_ex["input"])
                expected_grid = tuple(tuple(row) for row in train_ex["output"])
                
                try:
                    # Execute predicted program
                    pred_grid = execute_postfix_program(
                        list(prog_preds[:mask.sum() + 10]),  # include a bit extra
                        input_grid
                    )
                    
                    if pred_grid == expected_grid:
                        grid_correct += 1
                        
                except PostfixInterpreterError as e:
                    exec_errors += 1
            
            # Progress
            if total % 50 == 0:
                print(f"  Processed {total} examples...")
    
    # Final results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total examples: {total}")
    print(f"Program EM:     {program_em_correct}/{total} = {100*program_em_correct/max(1,total):.2f}%")
    print(f"Grid Accuracy:  {grid_correct}/{total} = {100*grid_correct/max(1,total):.2f}%")
    print(f"Exec Errors:    {exec_errors}/{total} = {100*exec_errors/max(1,total):.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()

