"""
Generate DSL token-based training data from the GSM8K dataset.

We treat GSM8K as a single module, `gsm8k`, and encode the *chain-of-thought*
steps found in the annotated answers:

    Natalia sold 48/2 = <<48/2=24>>24 clips in May.
    Natalia sold 48+24 = <<48+24=72>>72 clips altogether.
    #### 72

For each example we build a single DSL sequence:

    BOS
      <tokens(expr_1)> <tokens(result_1)> EQ_CMP
      <tokens(expr_2)> <tokens(result_2)> EQ_CMP
      ...
      <tokens(final_answer)>
    EOS

The dataset builder then treats the **final answer span** as the supervised
target: everything after the last EQ_CMP up to EOS.

EN-DSL Generation:
------------------
We also generate EN-DSL tokens that capture the semantic structure of the
word problem. GSM8K word problems typically involve:
- Entities (people, objects)
- Events (INIT, GAIN, LOSS, TRANSFER)
- Quantities (amounts, rates)
- A final HOW_MANY query

The EN-DSL sequence encodes the steps as:
    BOS
      EN_ENTITY <expr_1> EN_UNIT
      EN_EVT_* EN_AMOUNT <expr_2>
      ...
      EN_QUERY EN_Q_HOW_MANY EN_TOTAL
    EOS

Usage (small smoke test):

    python -m dev.gen_gsm8k \\
        --source-dir data/gsm8k \\
        --output-dir data/math_gsm8k_dsl_small \\
        --max-train 500 \\
        --max-test 200
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import sympy
from datasets import load_from_disk

from dsl.tokens import GyanDSLToken  # type: ignore
from dev.gen_full_math import (  # type: ignore
    expr_to_tokens,
    int_to_tokens,
    sympify_answer,
)


STEP_PATTERN = re.compile(r"<<([^=<>]+)=([^<>]+)>>")

# Patterns for semantic parsing of GSM8K word problems
# These are intentionally simple and cover common GSM8K patterns
GAIN_PATTERNS = [
    r"(?:sold|earned|received|got|found|bought|collected|made|gained|picked|harvested)\s+(\d+)",
    r"(\d+)\s+(?:more|additional|extra)",
]
LOSS_PATTERNS = [
    r"(?:spent|used|gave|lost|ate|sold|removed|took)\s+(\d+)",
    r"(\d+)\s+(?:less|fewer)",
]
INIT_PATTERNS = [
    r"(?:has|have|had|owns|owned|starts? with|began with)\s+(\d+)",
    r"(\d+)\s+(?:in total|altogether|initially|at first|to begin)",
]
RATE_PATTERNS = [
    r"\$(\d+(?:\.\d+)?)\s+(?:per|an|a|each)\s+(\w+)",
    r"(\d+(?:\.\d+)?)\s+(?:per|an|a|each)\s+(\w+)",
]


def build_gsm8k_en_tokens(
    question: str,
    steps: List[Tuple[sympy.Expr, sympy.Expr]],
    final_answer: sympy.Expr,
) -> Optional[List[GyanDSLToken]]:
    """
    Build EN-DSL tokens for a GSM8K word problem.
    
    The EN-DSL encodes the semantic structure only:
    - Initial states (EN_EVT_INIT)
    - Gains/increases (EN_EVT_GAIN)
    - Losses/decreases (EN_EVT_LOSS)
    - The final query (EN_QUERY EN_Q_HOW_MANY EN_TOTAL)

    IMPORTANT DESIGN CHOICE
    -----------------------
    We intentionally **do not encode solved numeric results** here.

    - For each CoT step, we encode the **expression** (e.g. 48/2, 48+24)
      using the generic math DSL via `expr_to_tokens`, not the evaluated
      result (24, 72, ...).
    - For the final answer, we **do not** emit EN_DIGIT_* tokens or INT_*
      tokens. The answer is left for the math-DSL/TRM solver to compute.

    This keeps EN-DSL as a *puzzle description*; the reasoning model is
    responsible for deriving numeric answers.
    """
    from dev.gen_full_math import int_to_tokens as gen_int_to_tokens

    def _encode_int_as_en_digits(n: int) -> List[GyanDSLToken]:
        """Encode an integer as EN_DIGIT_* tokens (with EN_DIGIT_NEG if needed)."""
        s = str(n)
        out: List[GyanDSLToken] = []
        for ch in s:
            if ch == "-":
                out.append(GyanDSLToken.EN_DIGIT_NEG)
            elif ch == ".":
                out.append(GyanDSLToken.EN_DIGIT_DOT)
            elif ch.isdigit():
                out.append(GyanDSLToken[f"EN_DIGIT_{ch}"])
            else:
                # Unexpected character in numeric answer; fail so caller can skip.
                raise ValueError(f"Unsupported character in GSM8K answer for EN-DSL digits: {ch!r}")
        return out

    tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]

    # Encode each CoT step as an event whose amount is the underlying
    # expression (unsolved), not the numeric result.
    for i, (expr, result) in enumerate(steps):
        # Determine event type based on the expression structure
        if expr.is_Add:
            # Addition -> GAIN
            tokens.append(GyanDSLToken.EN_EVT_GAIN)
        elif expr.is_Mul and any(arg.is_Rational and arg < 1 for arg in expr.args if not arg.is_Symbol):
            # Multiplication by fraction < 1 -> LOSS
            tokens.append(GyanDSLToken.EN_EVT_LOSS)
        elif expr.is_Mul:
            # Other multiplication -> RATE
            tokens.append(GyanDSLToken.EN_EVT_RATE)
        elif str(expr).startswith('-') or (hasattr(expr, 'could_extract_minus_sign') and expr.could_extract_minus_sign()):
            # Subtraction -> LOSS
            tokens.append(GyanDSLToken.EN_EVT_LOSS)
        else:
            # Default: treat first step as INIT, others as GAIN
            if i == 0:
                tokens.append(GyanDSLToken.EN_EVT_INIT)
            else:
                tokens.append(GyanDSLToken.EN_EVT_GAIN)

        # Encode the amount: use the CoT expression itself, not the result.
        tokens.append(GyanDSLToken.EN_AMOUNT)
        try:
            tokens += expr_to_tokens(expr, {})
        except Exception:
            # If expression-to-tokens fails, skip this example entirely.
            return None

    # Encode the query: asking for the total/final amount.
    # We deliberately do NOT encode the numeric answer here; EN-DSL stays as
    # a pure semantic + structural description of the puzzle.
    tokens.append(GyanDSLToken.EN_QUERY)
    tokens.append(GyanDSLToken.EN_Q_HOW_MANY)
    tokens.append(GyanDSLToken.EN_TOTAL)
    tokens.append(GyanDSLToken.EN_AMOUNT)

    tokens.append(GyanDSLToken.EOS)
    return tokens


def _parse_final_answer(answer_str: str) -> Optional[sympy.Expr]:
    """
    Extract the final numeric answer from a GSM8K answer string.

    GSM8K uses a line like:

        #### 72
    """
    lines = [ln.strip() for ln in answer_str.strip().splitlines()]
    for line in reversed(lines):
        if line.startswith("####"):
            raw = line.split("####", 1)[1].strip()
            # Strip simple trailing punctuation ('.', '$', etc.)
            raw = raw.rstrip().rstrip(".").strip()
            if not raw:
                return None
            try:
                return sympy.sympify(raw)
            except Exception:
                return None
    return None


def _parse_steps(answer_str: str) -> List[Tuple[sympy.Expr, sympy.Expr]]:
    """
    Parse all `<<expr=result>>` annotated steps in the GSM8K answer.

    Returns a list of (expr, result) SymPy pairs. If any step fails to parse,
    the entire list is returned empty so that the caller can decide to skip
    the example.
    """
    steps: List[Tuple[sympy.Expr, sympy.Expr]] = []
    try:
        for m in STEP_PATTERN.finditer(answer_str):
            expr_str = m.group(1).strip()
            res_str = m.group(2).strip()
            expr = sympy.sympify(expr_str)
            res = sympy.sympify(res_str)
            steps.append((expr, res))
    except Exception:
        return []
    return steps


def build_gsm8k_tokens(
    answer_str: str,
    drop_step_results: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Convert a GSM8K chain-of-thought answer string into a DSL token sequence.

    Returns a dict with:
        - token_ids: List[int]
        - token_names: List[str]
        - final_answer: str  (string form of the final numeric answer)

    or None if parsing/conversion fails.
    """
    # Parse steps and final answer.
    final_expr = _parse_final_answer(answer_str)
    if final_expr is None:
        return None

    steps = _parse_steps(answer_str)

    # If there are annotated steps, sanity-check that the last step's result
    # matches the final answer numerically. We still perform this check even
    # when `drop_step_results=True` to avoid keeping inconsistent CoT traces.
    if steps:
        try:
            last_res = steps[-1][1]
            if not sympy.simplify(last_res - final_expr) == 0:
                # Mismatch between last annotated result and final answer –
                # to keep the DSL structurally clean, skip such examples.
                return None
        except Exception:
            return None

    tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]

    # Encode each intermediate step.
    #
    # Default (drop_step_results == False):
    #   <expr_tokens> <result_tokens> EQ_CMP
    #
    # Expression-only variant (drop_step_results == True):
    #   <expr_tokens> EQ_CMP
    #
    # In both cases, the *final* answer appears only in the suffix after the
    # last EQ_CMP, which the dataset builder uses as the supervised span.
    for expr, res in steps:
        try:
            expr_tokens = expr_to_tokens(expr, {})
            if drop_step_results:
                res_tokens = []
            else:
                res_tokens = expr_to_tokens(res, {})
        except Exception:
            return None
        tokens.extend(expr_tokens)
        tokens.extend(res_tokens)
        tokens.append(GyanDSLToken.EQ_CMP)

    # Encode final answer as the suffix after the last EQ_CMP.
    try:
        # Use sympify_answer to handle integers/decimals consistently.
        ans_expr = sympify_answer(str(final_expr))
        if isinstance(ans_expr, sympy.Expr):
            ans_tokens = expr_to_tokens(ans_expr, {})
        else:
            ans_tokens = int_to_tokens(int(final_expr))
    except Exception:
        return None

    tokens.extend(ans_tokens)
    tokens.append(GyanDSLToken.EOS)

    result = {
        "token_ids": [t.value for t in tokens],
        "token_names": [t.name for t in tokens],
        "final_answer": str(final_expr),
    }
    
    # Generate EN-DSL tokens
    en_tokens = build_gsm8k_en_tokens(
        question="",  # Not used currently
        steps=steps,
        final_answer=final_expr,
    )
    if en_tokens is not None:
        result["en_token_ids"] = [t.value for t in en_tokens]
        result["en_token_names"] = [t.name for t in en_tokens]
    
    return result


def process_split(
    examples,
    split_name: str,
    output_path: str,
    max_examples: int,
    drop_step_results: bool,
) -> int:
    """
    Process a GSM8K split (`train` or `test`) and write JSONL.

    Each output record has:
        - id: gsm8k/<split>/<idx>
        - module: "gsm8k"
        - split: "train" | "test_id" | "test_ood"
        - question: original question text
        - answer: final numeric answer string
        - cot: original chain-of-thought answer (for debugging)
        - token_ids, token_names: DSL sequence
    """
    n_total = len(examples)
    if max_examples > 0:
        n_total = min(n_total, max_examples)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    num_kept = 0
    num_skipped_parse = 0

    with open(output_path, "w") as f:
        for idx in range(n_total):
            ex = examples[idx]
            question = str(ex["question"])
            answer_str = str(ex["answer"])

            dsl = build_gsm8k_tokens(answer_str, drop_step_results=drop_step_results)
            if dsl is None:
                num_skipped_parse += 1
                continue

            rec = {
                "id": f"gsm8k/{split_name}/{idx:06d}",
                "module": "gsm8k",
                "split": split_name,
                "question": question,
                "answer": dsl["final_answer"],
                "cot": answer_str,
                "token_ids": dsl["token_ids"],
                "token_names": dsl["token_names"],
            }
            # Add EN-DSL tokens if available
            if "en_token_ids" in dsl:
                rec["en_token_ids"] = dsl["en_token_ids"]
                rec["en_token_names"] = dsl["en_token_names"]
            f.write(json.dumps(rec) + "\n")
            num_kept += 1

    print(
        f"[{split_name}] kept {num_kept} examples, "
        f"skipped {num_skipped_parse} (parse/tokenization failures)"
    )
    return num_kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GSM8K DSL data with CoT steps.")
    parser.add_argument(
        "--source-dir",
        type=str,
        default="data/gsm8k",
        help="Directory with GSM8K dataset saved via `load_from_disk`.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/math_gsm8k_dsl",
        help="Output directory for DSL JSONL files.",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=-1,
        help="Optional cap on number of train examples (-1 = all).",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=-1,
        help="Optional cap on number of test examples (-1 = all).",
    )
    parser.add_argument(
        "--drop-step-results",
        action="store_true",
        help="If set, omit numeric results from <<expr=result>> steps (expression-only CoT).",
    )
    args = parser.parse_args()

    print(f"Loading GSM8K from: {args.source_dir}")
    ds = load_from_disk(args.source_dir)

    train = ds["train"]
    test = ds["test"]

    print(f"Train examples: {len(train)}, Test examples: {len(test)}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Train → train.jsonl
    train_path = os.path.join(args.output_dir, "train.jsonl")
    n_train = process_split(
        examples=train,
        split_name="train",
        output_path=train_path,
        max_examples=args.max_train,
        drop_step_results=args.drop_step_results,
    )

    # For compatibility with `build_dsl_dataset.py`, we need test_id.jsonl and
    # test_ood.jsonl. We split the original GSM8K test set in half
    # deterministically (by index).
    n_test_total = len(test)
    if args.max_test > 0:
        n_test_total = min(n_test_total, args.max_test)

    mid = n_test_total // 2
    test_id = test.select(range(0, mid))
    test_ood = test.select(range(mid, n_test_total))

    test_id_path = os.path.join(args.output_dir, "test_id.jsonl")
    test_ood_path = os.path.join(args.output_dir, "test_ood.jsonl")

    n_test_id = process_split(
        examples=test_id,
        split_name="test_id",
        output_path=test_id_path,
        max_examples=-1,
        drop_step_results=args.drop_step_results,
    )
    n_test_ood = process_split(
        examples=test_ood,
        split_name="test_ood",
        output_path=test_ood_path,
        max_examples=-1,
        drop_step_results=args.drop_step_results,
    )

    meta = {
        "source_dir": args.source_dir,
        "output_dir": args.output_dir,
        "train_examples_kept": n_train,
        "test_id_examples_kept": n_test_id,
        "test_ood_examples_kept": n_test_ood,
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print("\nGSM8K DSL generation complete.")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()


