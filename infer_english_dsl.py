"""
Inference script for English -> DSL compiler model.

Tests the checkpoint on generating DSL programs from English math questions.
"""

import torch
import json
import os
import sys
import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional

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


def decode_char_tokens(token_ids: np.ndarray, vocab: Dict[int, str]) -> str:
    """Decode CHAR_* tokens back to readable text."""
    result = []
    for tid in token_ids:
        if tid == PAD_ID:
            continue
        name = vocab.get(int(tid), f"UNK_{tid}")
        if name == "PAD" or name == "BOS" or name == "EOS":
            continue
        # Handle CHAR_* tokens
        if name.startswith("CHAR_"):
            char_part = name[5:]  # Remove "CHAR_" prefix
            if char_part == "SPACE":
                result.append(" ")
            elif char_part == "NEWLINE":
                result.append("\n")
            elif char_part == "DOT":
                result.append(".")
            elif char_part == "COMMA":
                result.append(",")
            elif char_part == "QUESTION":
                result.append("?")
            elif char_part == "DASH":
                result.append("-")
            elif char_part == "PLUS":
                result.append("+")
            elif char_part == "STAR":
                result.append("*")
            elif char_part == "SLASH":
                result.append("/")
            elif char_part == "EQUALS":
                result.append("=")
            elif char_part == "LPAREN":
                result.append("(")
            elif char_part == "RPAREN":
                result.append(")")
            elif char_part.endswith("_UP"):
                result.append(char_part[:-3].upper())  # e.g., "A_UP" -> "A"
            elif len(char_part) == 1:
                result.append(char_part)
            else:
                result.append(f"[{name}]")
        else:
            result.append(f"[{name}]")
    return "".join(result)


def decode_dsl_tokens(token_ids: np.ndarray, vocab: Dict[int, str], skip_special: bool = True) -> str:
    """Decode DSL tokens to readable string."""
    tokens = []
    for tid in token_ids:
        if tid == PAD_ID and skip_special:
            continue
        name = vocab.get(int(tid), f"UNK_{tid}")
        if name in ("PAD", "BOS", "EOS") and skip_special:
            continue
        tokens.append(name)
    return " ".join(tokens)


def load_model_and_config(checkpoint_dir: str, data_dir: str):
    """Load model from checkpoint directory."""
    # Load config
    config_path = os.path.join(checkpoint_dir, "all_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    arch_cfg = config["arch"]
    print(f"Model config:")
    print(f"  hidden_size: {arch_cfg['hidden_size']}")
    print(f"  H_cycles: {arch_cfg['H_cycles']}")
    print(f"  L_cycles: {arch_cfg['L_cycles']}")
    print(f"  L_layers: {arch_cfg['L_layers']}")
    print(f"  halt_max_steps: {arch_cfg['halt_max_steps']}")
    
    # Load dataset metadata
    test_meta_path = os.path.join(data_dir, "test", "dataset.json")
    with open(test_meta_path) as f:
        meta = json.load(f)
    print(f"\nDataset metadata:")
    print(f"  vocab_size: {meta['vocab_size']}")
    print(f"  seq_len: {meta['seq_len']}")
    print(f"  num_puzzle_identifiers: {meta['num_puzzle_identifiers']}")
    
    # Find latest checkpoint
    ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("step_")]
    if not ckpt_files:
        raise ValueError(f"No step_* checkpoints found in {checkpoint_dir}")
    latest = max(ckpt_files, key=lambda x: int(x.split("_")[1]))
    ckpt_path = os.path.join(checkpoint_dir, latest)
    print(f"\nLoading checkpoint: {ckpt_path}")
    
    # Import model
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
    from models.losses import ACTLossHead
    
    # Build model config
    model_cfg = {
        "batch_size": 1,
        "vocab_size": meta["vocab_size"],
        "seq_len": meta["seq_len"],
        "num_puzzle_identifiers": meta["num_puzzle_identifiers"],
        "hidden_size": arch_cfg["hidden_size"],
        "num_heads": arch_cfg["num_heads"],
        "expansion": arch_cfg["expansion"],
        "H_cycles": arch_cfg["H_cycles"],
        "L_cycles": arch_cfg["L_cycles"],
        "H_layers": arch_cfg.get("H_layers", 0),
        "L_layers": arch_cfg["L_layers"],
        "halt_exploration_prob": arch_cfg["halt_exploration_prob"],
        "halt_max_steps": arch_cfg["halt_max_steps"],
        "puzzle_emb_ndim": arch_cfg.get("puzzle_emb_ndim", 512),
        "puzzle_emb_len": arch_cfg.get("puzzle_emb_len", 8),
        "pos_encodings": arch_cfg.get("pos_encodings", "rope"),
        "forward_dtype": arch_cfg.get("forward_dtype", "bfloat16"),
        "mlp_t": arch_cfg.get("mlp_t", False),
        "no_ACT_continue": arch_cfg.get("no_ACT_continue", True),
    }
    
    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    with torch.device(device):
        model = TinyRecursiveReasoningModel_ACTV1(model_cfg)
        model = ACTLossHead(model, loss_type=arch_cfg["loss"]["loss_type"])
    
    # Load state dict
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Handle compiled model prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("_orig_mod.", "")
        new_state_dict[new_k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    model.to(device)
    
    return model, meta, device


def load_test_data(data_dir: str, num_examples: int = 10, per_module: bool = False) -> Dict:
    """Load test examples.
    
    If per_module=True, samples num_examples from each module.
    """
    test_dir = os.path.join(data_dir, "test")
    
    # Load arrays
    english = np.load(os.path.join(test_dir, "all__english.npy"))
    program = np.load(os.path.join(test_dir, "all__program.npy"))
    puzzle_ids = np.load(os.path.join(test_dir, "all__puzzle_identifiers.npy"))
    
    # Load identifiers
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
        indices = np.random.choice(len(english), min(num_examples, len(english)), replace=False)
    
    return {
        "english": english[indices],
        "program": program[indices],
        "puzzle_identifiers": puzzle_ids[indices],
        "id_to_name": id_to_name,
        "indices": indices,
    }


@torch.inference_mode()
def run_inference(
    model,
    batch: Dict[str, torch.Tensor],
    device: str,
    max_steps: int = 16,
) -> Tuple[np.ndarray, int, List[float]]:
    """Run inference with ACT."""
    carry = model.initial_carry(batch)
    
    q_values = []
    step = 0
    
    while step < max_steps:
        carry, loss, metrics, outputs, all_finish = model(
            carry=carry, batch=batch, return_keys=["preds", "q_halt_logits"]
        )
        
        q_halt = torch.sigmoid(outputs["q_halt_logits"]).item()
        q_values.append(q_halt)
        
        step += 1
        
        if all_finish:
            break
    
    return outputs["preds"][0].cpu().numpy(), step, q_values


def main():
    import argparse
    parser = argparse.ArgumentParser(description="English -> DSL Inference")
    parser.add_argument("--checkpoint", type=str, 
                        default="checkpoints/English_dsl_compiler_1m_trm-ACT-torch",
                        help="Path to checkpoint directory")
    parser.add_argument("--data_dir", type=str,
                        default="data/english_dsl_compiler_1m_trm",
                        help="Path to dataset (must have english/program arrays)")
    parser.add_argument("--num_examples", type=int, default=5, help="Number of examples (per module if --per_module)")
    parser.add_argument("--per_module", action="store_true", help="Sample num_examples from each module")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load vocab
    vocab = load_vocab()
    print(f"Vocabulary size: {len(vocab)}")
    
    # Load model
    print(f"\n{'='*80}")
    print("LOADING MODEL")
    print(f"{'='*80}")
    model, meta, device = load_model_and_config(args.checkpoint, args.data_dir)
    print("Model loaded successfully")
    
    # Load test data
    print(f"\n{'='*80}")
    print("LOADING TEST DATA")
    print(f"{'='*80}")
    data = load_test_data(args.data_dir, args.num_examples, per_module=args.per_module)
    print(f"Loaded {len(data['english'])} examples")
    
    # Run inference
    print(f"\n{'='*80}")
    print("INFERENCE RESULTS")
    print(f"{'='*80}")
    
    correct = 0
    total = 0
    module_stats = {}  # {module_name: {"correct": 0, "total": 0}}
    current_module = None
    
    for i in range(len(data["english"])):
        english = data["english"][i]
        program = data["program"][i]
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
        
        # Decode English input
        english_text = decode_char_tokens(english, vocab)
        print(f"English: {english_text}")
        
        # Decode target program
        target_program = decode_dsl_tokens(program, vocab)
        print(f"Target DSL: {target_program}")
        
        # Prepare batch (decoder-only: concat english + program_pad for inputs)
        english_tensor = torch.tensor(english).unsqueeze(0).to(device)
        program_pad = torch.full_like(english_tensor, PAD_ID)
        inputs = torch.cat([english_tensor, program_pad], dim=1)
        
        # Labels: ignore for english, supervise program
        ignore = torch.full_like(english_tensor, IGNORE_LABEL_ID)
        program_tensor = torch.tensor(program).unsqueeze(0).to(device)
        labels = torch.cat([ignore, program_tensor], dim=1)
        
        batch = {
            "inputs": inputs,
            "labels": labels,
            "puzzle_identifiers": torch.tensor([puzzle_id]).to(device),
        }
        
        # Run inference
        preds, num_steps, q_values = run_inference(model, batch, device)
        
        # Extract program predictions (second half of seq)
        english_len = english.shape[0]
        pred_program = preds[english_len:]
        
        # Decode prediction
        pred_program_str = decode_dsl_tokens(pred_program, vocab)
        print(f"Predicted DSL: {pred_program_str}")
        
        # Check correctness (compare non-pad tokens)
        target_mask = program != PAD_ID
        pred_masked = pred_program[target_mask]
        target_masked = program[target_mask]
        
        is_correct = np.all(pred_masked == target_masked)
        correct += int(is_correct)
        total += 1
        module_stats[puzzle_name]["correct"] += int(is_correct)
        module_stats[puzzle_name]["total"] += 1
        
        print(f"Correct: {'YES' if is_correct else 'NO'}")
        print(f"ACT steps: {num_steps}, final q_halt: {q_values[-1]:.3f}")
        
        if not is_correct:
            # Show token-by-token comparison for first few tokens
            print("Token comparison (first 10 non-pad):")
            count = 0
            for j, (t, p) in enumerate(zip(program, pred_program)):
                if t == PAD_ID:
                    continue
                t_name = vocab.get(int(t), f"UNK_{t}")
                p_name = vocab.get(int(p), f"UNK_{p}")
                match = "OK" if t == p else "WRONG"
                print(f"  [{j}] target={t_name}, pred={p_name} [{match}]")
                count += 1
                if count >= 10:
                    break
    
    print(f"\n{'='*80}")
    print("SUMMARY BY MODULE")
    print(f"{'='*80}")
    for module_name, stats in sorted(module_stats.items()):
        acc = 100 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {module_name}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    print(f"{'-'*80}")
    print(f"OVERALL: {correct}/{total} correct ({100*correct/total:.1f}%)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

