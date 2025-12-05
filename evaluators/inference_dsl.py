"""
Inference script for DSL equation solving models.

Shows how the model solves equations step-by-step with ACT (Adaptive Computation Time).

Usage:
    python inference_dsl.py --checkpoint <path> [--num_examples 10] [--show_steps]
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from dsl.tokens import GyanDSLToken, id_to_token, get_vocab_size


# Constants
IGNORE_LABEL_ID = -100
PAD_ID = 0


def load_vocab() -> Dict[int, str]:
    """Build id -> token name mapping."""
    vocab = {}
    for tok in GyanDSLToken:
        vocab[tok.value] = tok.name
    return vocab


def decode_tokens(token_ids: np.ndarray, vocab: Dict[int, str], skip_special: bool = True) -> str:
    """Decode token IDs to readable string."""
    tokens = []
    for tid in token_ids:
        if tid == PAD_ID and skip_special:
            continue
        name = vocab.get(int(tid), f"UNK_{tid}")
        if name in ("PAD", "BOS", "EOS") and skip_special:
            continue
        tokens.append(name)
    return " ".join(tokens)


def decode_tokens_highlighted(
    input_ids: np.ndarray, 
    label_ids: np.ndarray, 
    pred_ids: np.ndarray, 
    vocab: Dict[int, str]
) -> Tuple[str, str, str]:
    """Decode with highlighting of answer region."""
    input_parts = []
    label_parts = []
    pred_parts = []
    
    for i, (inp, lbl, prd) in enumerate(zip(input_ids, label_ids, pred_ids)):
        inp_name = vocab.get(int(inp), f"UNK_{inp}")
        
        if inp == PAD_ID:
            continue
            
        if lbl != IGNORE_LABEL_ID:
            # This is an answer token
            lbl_name = vocab.get(int(lbl), f"UNK_{lbl}")
            prd_name = vocab.get(int(prd), f"UNK_{prd}")
            
            input_parts.append(f"[{inp_name}]")
            label_parts.append(lbl_name)
            pred_parts.append(prd_name)
        else:
            input_parts.append(inp_name)
    
    return " ".join(input_parts), " ".join(label_parts), " ".join(pred_parts)


def load_test_data(data_dir: str, num_examples: int = 10, per_module: bool = False) -> Dict[str, np.ndarray]:
    """Load test examples from dataset.
    
    If per_module=True, samples num_examples from each module.
    """
    test_dir = os.path.join(data_dir, "test")
    
    inputs = np.load(os.path.join(test_dir, "all__inputs.npy"))
    labels = np.load(os.path.join(test_dir, "all__labels.npy"))
    puzzle_ids = np.load(os.path.join(test_dir, "all__puzzle_identifiers.npy"))
    
    # Load identifiers mapping
    with open(os.path.join(data_dir, "identifiers.json")) as f:
        id_to_name = json.load(f)
    
    if per_module:
        # Sample num_examples from each module (skip blank id=0)
        all_indices = []
        for module_id in range(1, len(id_to_name)):
            module_indices = np.where(puzzle_ids == module_id)[0]
            if len(module_indices) > 0:
                sampled = np.random.choice(
                    module_indices, 
                    min(num_examples, len(module_indices)), 
                    replace=False
                )
                all_indices.extend(sampled)
        indices = np.array(all_indices)
    else:
        # Sample random indices
        indices = np.random.choice(len(inputs), min(num_examples, len(inputs)), replace=False)
    
    return {
        "inputs": inputs[indices],
        "labels": labels[indices],
        "puzzle_identifiers": puzzle_ids[indices],
        "id_to_name": id_to_name,
        "indices": indices,
    }


def load_model(checkpoint_path: str, data_dir: str):
    """Load model from checkpoint."""
    # Load dataset metadata
    with open(os.path.join(data_dir, "test", "dataset.json")) as f:
        meta = json.load(f)
    
    # Find the latest checkpoint
    ckpt_dir = Path(checkpoint_path)
    if ckpt_dir.is_dir():
        steps = [d.name for d in ckpt_dir.iterdir() if d.name.startswith("step_")]
        if steps:
            latest = max(steps, key=lambda x: int(x.split("_")[1]))
            checkpoint_file = ckpt_dir / latest
        else:
            raise ValueError(f"No checkpoints found in {checkpoint_path}")
    else:
        checkpoint_file = ckpt_dir
    
    print(f"Loading checkpoint: {checkpoint_file}")
    
    # Load config from checkpoint directory
    config_path = ckpt_dir / "all_config.yaml" if ckpt_dir.is_dir() else ckpt_dir.parent / "all_config.yaml"
    
    # Import model classes
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
    from models.losses import ACTLossHead
    
    # Build config dict - use values from the actual training
    model_cfg = {
        "batch_size": 1,
        "vocab_size": meta["vocab_size"],
        "seq_len": meta["seq_len"],
        "num_puzzle_identifiers": meta["num_puzzle_identifiers"],
        "hidden_size": 256,  # Small model
        "num_heads": 8,
        "expansion": 4,
        "H_cycles": 3,
        "L_cycles": 4,
        "H_layers": 0,
        "L_layers": 2,
        "halt_exploration_prob": 0.1,
        "halt_max_steps": 16,
        "puzzle_emb_ndim": 256,
        "puzzle_emb_len": 8,
        "pos_encodings": "rope",
        "forward_dtype": "bfloat16",
        "mlp_t": False,
        "no_ACT_continue": True,
        "causal": False,
    }
    
    # Create model
    with torch.device("cuda"):
        model = TinyRecursiveReasoningModel_ACTV1(model_cfg)
        model = ACTLossHead(model, loss_type="stablemax_cross_entropy")
    
    # Load state dict
    state_dict = torch.load(checkpoint_file, map_location="cuda")
    
    # Handle compiled model prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove _orig_mod. prefix if present
        new_k = k.replace("_orig_mod.", "")
        new_state_dict[new_k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    return model, meta


@torch.inference_mode()
def run_inference(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    max_steps: int = 16,
    show_steps: bool = False,
    vocab: Optional[Dict[int, str]] = None,
) -> Tuple[np.ndarray, int, List[float]]:
    """
    Run inference with ACT, optionally showing intermediate steps.
    
    Returns:
        predictions: Final predicted token IDs
        num_steps: Number of ACT steps taken
        q_values: Q-halt values at each step
    """
    with torch.device("cuda"):
        carry = model.initial_carry(batch)
    
    q_values = []
    step = 0
    
    while step < max_steps:
        carry, loss, metrics, preds, all_finish = model(
            carry=carry, batch=batch, return_keys={"preds", "q_halt_logits"}
        )
        
        # Get Q-halt probability
        q_halt = torch.sigmoid(preds["q_halt_logits"]).item()
        q_values.append(q_halt)
        
        if show_steps and vocab is not None:
            pred_str = decode_tokens(preds["preds"][0].cpu().numpy(), vocab)
            print(f"  Step {step+1}: q_halt={q_halt:.3f}")
            if step < 3 or all_finish:  # Show first few and last
                print(f"    Pred: {pred_str[:100]}...")
        
        step += 1
        
        if all_finish:
            break
    
    return preds["preds"][0].cpu().numpy(), step, q_values


def main():
    parser = argparse.ArgumentParser(description="DSL Model Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--data_dir", type=str, default="data/dsl_trm_eq_1m", help="Path to dataset")
    parser.add_argument("--num_examples", type=int, default=10, help="Number of examples to run (per module if --per_module)")
    parser.add_argument("--per_module", action="store_true", help="Sample num_examples from each module")
    parser.add_argument("--show_steps", action="store_true", help="Show ACT steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load vocab
    vocab = load_vocab()
    print(f"Vocabulary size: {len(vocab)}")
    
    # Load test data
    print(f"\nLoading test data from {args.data_dir}...")
    data = load_test_data(args.data_dir, args.num_examples, per_module=args.per_module)
    print(f"Loaded {len(data['inputs'])} examples")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model, meta = load_model(args.checkpoint, args.data_dir)
    print("Model loaded successfully")
    
    # Run inference
    print("\n" + "=" * 80)
    print("INFERENCE RESULTS")
    print("=" * 80)
    
    correct = 0
    total = 0
    module_stats = {}  # {module_name: {"correct": 0, "total": 0}}
    current_module = None
    
    for i in range(len(data["inputs"])):
        inputs = data["inputs"][i]
        labels = data["labels"][i]
        puzzle_id = data["puzzle_identifiers"][i]
        puzzle_name = data["id_to_name"][puzzle_id]
        
        # Track module stats
        if puzzle_name not in module_stats:
            module_stats[puzzle_name] = {"correct": 0, "total": 0}
        
        # Print module header when module changes
        if args.per_module and puzzle_name != current_module:
            current_module = puzzle_name
            print(f"\n{'='*80}")
            print(f"MODULE: {puzzle_name}")
            print(f"{'='*80}")
        
        module_example_num = module_stats[puzzle_name]["total"] + 1
        print(f"\n--- Example {module_example_num} [{puzzle_name}] ---")
        
        # Prepare batch
        batch = {
            "inputs": torch.tensor(inputs).unsqueeze(0).cuda(),
            "labels": torch.tensor(labels).unsqueeze(0).cuda(),
            "puzzle_identifiers": torch.tensor([puzzle_id]).cuda(),
        }
        
        # Show input
        input_str, label_str, _ = decode_tokens_highlighted(inputs, labels, inputs, vocab)
        print(f"Input: {input_str[:200]}...")
        print(f"Target answer: {label_str}")
        
        # Run inference
        if args.show_steps:
            print("\nACT Steps:")
        
        preds, num_steps, q_values = run_inference(
            model, batch, show_steps=args.show_steps, vocab=vocab
        )
        
        # Check correctness
        mask = labels != IGNORE_LABEL_ID
        pred_answer = preds[mask]
        true_answer = labels[mask]
        
        is_correct = np.all(pred_answer == true_answer)
        correct += int(is_correct)
        total += 1
        module_stats[puzzle_name]["correct"] += int(is_correct)
        module_stats[puzzle_name]["total"] += 1
        
        # Decode prediction
        pred_str = " ".join([vocab.get(int(t), f"UNK_{t}") for t in pred_answer])
        
        print(f"Predicted answer: {pred_str}")
        print(f"Correct: {'YES' if is_correct else 'NO'}")
        print(f"ACT steps: {num_steps}, final q_halt: {q_values[-1]:.3f}")
        
        # Show token-by-token comparison if wrong
        if not is_correct:
            print("Token comparison:")
            for j, (t, p) in enumerate(zip(true_answer, pred_answer)):
                t_name = vocab.get(int(t), f"UNK_{t}")
                p_name = vocab.get(int(p), f"UNK_{p}")
                match = "OK" if t == p else "WRONG"
                print(f"  [{j}] target={t_name}, pred={p_name} [{match}]")
    
    print("\n" + "=" * 80)
    print("SUMMARY BY MODULE")
    print("=" * 80)
    for module_name, stats in sorted(module_stats.items()):
        acc = 100 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {module_name}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    print("-" * 80)
    print(f"OVERALL: {correct}/{total} correct ({100*correct/total:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()

