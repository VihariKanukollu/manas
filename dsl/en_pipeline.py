"""
End-to-end helper pipeline for a tiny class of English word problems.

This module wires together:

    English text
      → EN-DSL tokens (rule-based)
      → EN-DSL interpreter / world simulation
      → math-style postfix expression
      → numeric answer (via a tiny deterministic evaluator)

It is intentionally narrow and self-contained. Nothing in the existing math
dataset builders or model inference stack is modified.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .tokens import GyanDSLToken
from .en_rules import parse_init_gain_how_many
from .en_runtime import (
    ENWorldState,
    ENQuery,
    build_expression_tokens,
    evaluate_rpn_expression,
    parse_en_program,
    simulate_en_program,
    answer_query,
)


def solve_init_gain_problem(problem: str) -> Dict[str, Any]:
    """
    Solve a very narrow INIT+GAIN+HOW_MANY word problem end-to-end.

    Steps:
      1. English → EN-DSL tokens (rule-based).
      2. EN-DSL tokens → ENEvent / ENQuery lists.
      3. Simulate EN events to build world state.
      4. Answer the last HOW_MANY query numerically.
      5. Build the corresponding postfix arithmetic expression (INT_*/ADD/SUB).
      6. Evaluate the expression deterministically and assert consistency.
    """

    en_tokens, meta = parse_init_gain_how_many(problem)

    events, queries = parse_en_program(en_tokens)
    if not queries:
        raise ValueError("No EN queries parsed from token sequence")

    world: ENWorldState = simulate_en_program(events)

    # For v0 we simply answer the last query.
    query: ENQuery = queries[-1]
    numeric_answer, expr_tokens = answer_query(  # type: ignore[misc]
        world,
        query,
        return_expr=True,
    )

    # Deterministic check: evaluate the expression and compare to world answer.
    eval_answer = evaluate_rpn_expression(expr_tokens)
    if eval_answer != numeric_answer:
        raise AssertionError(
            f"Expression evaluation mismatch: world={numeric_answer}, expr={eval_answer}"
        )

    result: Dict[str, Any] = {
        "problem": problem,
        "answer": int(numeric_answer),
        "en_token_ids": [int(t.value) for t in en_tokens],
        "en_token_names": [t.name for t in en_tokens],
        "expr_token_ids": [int(t.value) for t in expr_tokens],
        "expr_token_names": [t.name for t in expr_tokens],
        "meta": meta,
    }
    return result


__all__ = [
    "solve_init_gain_problem",
]


