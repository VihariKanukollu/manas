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

# Re‑use a few helper utilities from the generator so that answer‑span
# detection exactly mirrors how tokens were built.
from dev.gen_full_math import (  # type: ignore
    sympify_answer,
    expr_to_tokens,
    int_to_tokens,
)
import sympy  # type: ignore


PAD_ID = GyanDSLToken.PAD.value
EOS_ID = GyanDSLToken.EOS.value
IGNORE_LABEL_ID = -100


# ---------------------------------------------------------------------------
# Module families / answer‑span patterns
# ---------------------------------------------------------------------------

# Modules whose answers follow the EQ / REAL_VAR / IS_SOLUTION pattern.
EQ_SOLUTION_MODULES = {
    "algebra__linear_1d",
    "algebra__linear_1d_composed",
    "algebra__linear_2d",
    "algebra__linear_2d_composed",
}

# Arithmetic expression‑evaluation modules using build_expression_eval_tokens:
#   BOS <expr> <answer> EQ_CMP EOS
EXPR_EVAL_MODULES = {
    "arithmetic__add_or_sub",
    "arithmetic__mul",
    "arithmetic__div",
    "arithmetic__mixed",
    "arithmetic__add_sub_multiple",
    "arithmetic__mul_div_multiple",
}

# Unary numeric answer‑only programs (no EQ_CMP):
#   BOS [OP] <answer_expr_tokens> EOS
UNARY_NUMERIC_MODULES = {
    "algebra__sequence_next_term",
    "algebra__sequence_nth_term",
    "arithmetic__simplify_surd",
    "numbers__gcd_composed",
    "numbers__lcm_composed",
    "numbers__div_remainder_composed",
    "numbers__round_number_composed",
    "numbers__place_value_composed",
    "numbers__list_prime_factors_composed",
    "polynomials__evaluate",
    "polynomials__evaluate_composed",
    "polynomials__compose",
}

# Unary boolean answer‑only programs:
#   BOS OP BOOL_TRUE/FALSE EOS   or   BOS BOOL_TRUE/FALSE EOS
UNARY_BOOL_MODULES = {
    "numbers__is_prime_composed",
    "comparison__pair_composed",
}

# Answer‑only expression/value modules with custom operators:
ANSWER_ONLY_EXPR_MODULES = {
    "algebra__polynomial_roots_composed",  # root‑list branch
    "comparison__kth_biggest_composed",
    "comparison__closest_composed",
    "comparison__sort_composed",
    "comparison__sort",
    "polynomials__add",
    "polynomials__simplify_power",
    "comparison__closest",  # structured but EOS‑terminated
    "comparison__kth_biggest",
    "numbers__list_prime_factors",
    "numbers__list_prime_factors_composed",
}

# EQ_CMP‑based structured modules where the answer is always the suffix
# before EQ_CMP, optionally after a marker token.
GCD_LCM_MODULES = {"numbers__gcd", "numbers__lcm"}
DIV_REMAINDER_MODULES = {"numbers__div_remainder"}
PLACE_VALUE_MODULES = {"numbers__place_value"}
IS_PRIME_EQCMP_MODULES = {"numbers__is_prime"}
MEASUREMENT_MODULES = {"measurement__conversion", "measurement__time"}
PROBABILITY_MODULES = {
    "probability__swr_p_sequence",
    "probability__swr_p_level_set",
}
POLY_EQCMP_OP_MODULES = {
    "algebra__polynomial_roots",
    "polynomials__collect",
    "polynomials__expand",
    "polynomials__coefficient_named",
}
DIFFERENTIATE_MODULES = {
    "calculus__differentiate",
    "calculus__differentiate_composed",
}
BASE_CONVERSION_MODULES = {
    "numbers__base_conversion",
    "arithmetic__add_or_sub_in_base",
}


def _bool_from_answer(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    return s in ("true", "t", "1")


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


def _last_index(token_names: List[str], target: str, before: int | None = None) -> int:
    """Find last index of `target` in token_names (optionally before `before`)."""
    hi = len(token_names) if before is None else before
    for i in range(hi - 1, -1, -1):
        if token_names[i] == target:
            return i
    return -1


def locate_generic_answer_span(
    module: str,
    token_names: List[str],
    answer: Any,
) -> Tuple[int, int]:
    """
    Locate answer span for non‑EQ_SOLUTION modules.

    We rely on the exact DSL patterns used in `dev.gen_full_math.py`:
      - EQ_CMP‑based structured programs: answer is a suffix before EQ_CMP,
        often after a marker token like GCD/LCM/PLACE_VALUE/PROBABILITY/etc.
      - Unary / answer‑only programs: BOS [OP] <answer_tokens> EOS.
    """
    n = len(token_names)
    if n == 0:
        return 0, 0

    # ---------- EQ_CMP‑based modules ----------
    if "EQ_CMP" in token_names:
        eq_idx = _last_index(token_names, "EQ_CMP")
        if eq_idx <= 0:
            return 0, 0

        # Arithmetic expression evaluation: BOS <expr> <answer> EQ_CMP EOS
        if module in EXPR_EVAL_MODULES:
            try:
                ans_expr = sympify_answer(answer)
                ans_tokens = expr_to_tokens(ans_expr, {})
                ans_len = len(ans_tokens)
                ans_start = eq_idx - ans_len
                if ans_start < 1:
                    return 0, 0
                return ans_start, eq_idx
            except Exception:
                return 0, 0

        # GCD / LCM: BOS <a> <b> GCD/LCM <answer> EQ_CMP EOS
        if module in GCD_LCM_MODULES:
            try:
                ans_int = int(answer)
                ans_tokens = int_to_tokens(ans_int)
                ans_len = len(ans_tokens)
                ans_start = eq_idx - ans_len
                return max(1, ans_start), eq_idx
            except Exception:
                return 0, 0

        # Div remainder: BOS p q DIV_REMAINDER <answer> EQ_CMP EOS
        if module in DIV_REMAINDER_MODULES:
            try:
                ans_int = int(answer)
                ans_tokens = int_to_tokens(ans_int)
                ans_len = len(ans_tokens)
                ans_start = eq_idx - ans_len
                return max(1, ans_start), eq_idx
            except Exception:
                return 0, 0

        # Factorials: BOS n FACTORIAL <answer> EQ_CMP EOS  (not present in current meta,
        # but safe to support if added later).
        if module == "numbers__factorial":
            try:
                ans_int = int(answer)
                ans_tokens = int_to_tokens(ans_int)
                ans_len = len(ans_tokens)
                ans_start = eq_idx - ans_len
                return max(1, ans_start), eq_idx
            except Exception:
                return 0, 0

        # Place value: BOS <number> <position> PLACE_VALUE <digit> EQ_CMP EOS
        if module in PLACE_VALUE_MODULES:
            op_idx = _last_index(token_names, "PLACE_VALUE", before=eq_idx)
            if op_idx >= 0:
                return op_idx + 1, eq_idx
            return 0, 0

        # is_prime: BOS <n> IS_PRIME BOOL EQ_CMP EOS
        if module in IS_PRIME_EQCMP_MODULES:
            op_idx = _last_index(token_names, "IS_PRIME", before=eq_idx)
            if op_idx >= 0:
                return op_idx + 1, eq_idx
            return 0, 0

        # Measurement conversion/time:
        #   conversion: BOS <value> ... EVAL_EXPR <answer> EQ_CMP EOS
        #   time:       BOS <t1/t2/...> ... EVAL_EXPR <answer> EQ_CMP EOS
        if module in MEASUREMENT_MODULES:
            op_idx = _last_index(token_names, "EVAL_EXPR", before=eq_idx)
            if op_idx >= 0:
                return op_idx + 1, eq_idx
            return 0, 0

        # Probability modules:
        #   BOS ... PROBABILITY <num_tokens> <den_tokens> DIV EQ_CMP EOS
        if module in PROBABILITY_MODULES:
            prob_idx = _last_index(token_names, "PROBABILITY", before=eq_idx)
            if prob_idx >= 0:
                return prob_idx + 1, eq_idx
            return 0, 0

        # Base‑conversion / add_or_sub_in_base:
        #   ... TO_BASE <answer> EQ_CMP EOS
        if module in BASE_CONVERSION_MODULES:
            marker = "TO_BASE"
            op_idx = _last_index(token_names, marker, before=eq_idx)
            if op_idx >= 0:
                return op_idx + 1, eq_idx
            return 0, 0

        # Polynomial factor / collect / expand / coefficient_named
        if module in POLY_EQCMP_OP_MODULES:
            if module == "algebra__polynomial_roots":
                marker = "FACTOR"
            elif module == "polynomials__collect":
                marker = "COLLECT"
            elif module == "polynomials__expand":
                marker = "EXPAND"
            else:  # coefficient_named
                marker = "COEFF_AT_POWER"
            op_idx = _last_index(token_names, marker, before=eq_idx)
            if op_idx >= 0:
                return op_idx + 1, eq_idx
            return 0, 0

        # Differentiation: BOS <expr> <var> DIFF...DIFF <answer> EQ_CMP EOS
        if module in DIFFERENTIATE_MODULES:
            diff_idx = _last_index(token_names, "DIFF", before=eq_idx)
            if diff_idx >= 0:
                return diff_idx + 1, eq_idx
            return 0, 0

        # Fallback for any remaining EQ_CMP‑based module: treat the tokens
        # immediately before EQ_CMP as the answer span. This will typically
        # still cover the intended answer expression.
        return max(1, eq_idx - 1), eq_idx

    # ---------- Answer‑only / unary programs (no EQ_CMP) ----------
    eos_idx = n - 1 if token_names[-1] == "EOS" else n

    # Simple boolean answer only: BOS BOOL_TRUE/FALSE EOS
    if module in UNARY_BOOL_MODULES or module in {"numbers__is_factor", "numbers__is_factor_composed"}:
        if eos_idx - 1 >= 1:
            return eos_idx - 1, eos_idx
        return 0, 0

    # Unary numeric programs: BOS [OP] <answer_expr_tokens> EOS
    if module in UNARY_NUMERIC_MODULES:
        try:
            ans_expr = sympify_answer(answer)
            # Mirroring build_unary_numeric_answer_tokens: var_map from free symbols.
            var_map: Dict[sympy.Symbol, int] = {}
            if isinstance(ans_expr, sympy.Expr):
                for sym in sorted(list(ans_expr.free_symbols), key=lambda s: s.name):
                    if sym not in var_map:
                        var_map[sym] = len(var_map)
            ans_tokens = expr_to_tokens(ans_expr, var_map)
            ans_len = len(ans_tokens)
            ans_start = eos_idx - ans_len
            return max(1, ans_start), eos_idx
        except Exception:
            return 0, 0

    # List‑prime‑factors modules: answer is a comma‑separated list of primes.
    if module.startswith("numbers__list_prime_factors"):
        try:
            prime_strs = [s.strip() for s in str(answer).split(",") if s.strip()]
            ans_tokens: List[int] = []
            for p_str in prime_strs:
                p = int(p_str)
                ans_tokens.extend([t.value for t in int_to_tokens(p)])
            ans_len = len(ans_tokens)
            ans_start = eos_idx - ans_len
            return max(1, ans_start), eos_idx
        except Exception:
            return 0, 0

    # Kth_biggest / closest / sort / add_polynomials / simplify_power /
    # polynomial_roots_composed (root‑list branch) – all encode the answer as
    # the suffix expression before EOS.
    if module in ANSWER_ONLY_EXPR_MODULES or module in {
        "comparison__pair",  # value‑answer branch
    }:
        try:
            # For many of these, the answer is an expression or scalar that
            # `sympify_answer` can handle.
            ans_expr = sympify_answer(answer)
            # For sort and list‑style answers, fall back to string if needed.
            if isinstance(ans_expr, sympy.Basic):
                ans_tokens = expr_to_tokens(ans_expr, {})
            else:
                # Try a best‑effort sympify; if that still fails, skip.
                ans_tokens = expr_to_tokens(sympy.sympify(str(answer)), {})
            ans_len = len(ans_tokens)
            ans_start = eos_idx - ans_len
            return max(1, ans_start), eos_idx
        except Exception:
            return 0, 0

    # Absolute fallback – if we reach here, treat the last non‑special token
    # before EOS as a single‑token answer.
    if eos_idx > 1:
        return eos_idx - 1, eos_idx
    return 0, 0


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
    answer: Any,
) -> ExampleTensors | None:
    """
    Convert a single JSONL example into (inputs, labels, puzzle_id).

    - For unsupported modules, returns None.
    - For supported modules, uses module‑aware structural rules to locate the
      answer span and builds a masked‑LM style objective where:
          * inputs: answer span tokens are replaced with PAD
          * labels: IGNORE everywhere except the answer span
    """
    # Sanity: lengths should match
    if len(token_ids) != len(token_names):
        return None

    # Skip overly long sequences (should not happen for linear_1d, but be safe).
    if len(token_ids) > seq_len:
        return None

    # Locate answer span depending on module family.
    if module in EQ_SOLUTION_MODULES or "IS_SOLUTION" in token_names:
        ans_start, ans_end = locate_eq_solution_span(token_names)
    else:
        ans_start, ans_end = locate_generic_answer_span(module, token_names, answer)

    if ans_start == ans_end or ans_start < 0 or ans_end > len(token_ids):
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
            answer=ex.get("answer"),
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


