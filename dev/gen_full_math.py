"""
Generate DSL token-based training data from DeepMind mathematics_dataset modules.

This generator currently covers:
- Algebra: linear_1d, linear_2d, polynomial_roots (factorization variants), sequences
- Arithmetic: add/sub, mul/div, mixed, long chains (add_sub_multiple, mul_div_multiple)
- Calculus: differentiate (polynomial derivatives)
- Comparison: pair, kth_biggest, closest, sort
- Numbers: gcd, lcm, div_remainder, is_prime, is_factor, round_number, place_value,
           base_conversion, list_prime_factors (scalar cases)
- Polynomials: evaluate, expand, collect, coefficient_named, simplify_power, compose
- Probability: swr_p_sequence

Output format: JSONL with token_ids (list of GyanDSLToken integer IDs)

Usage:
    # Small smoke test
    python -m dev.gen_full_math --output_dir=data/full_math_small \
        --train_per_module=1000 --test_per_module=100

    # Full scale for H200
    python -m dev.gen_full_math --output_dir=data/full_math_large \
        --train_per_module=50000 --test_per_module=2000
"""

import argparse
import json
import math
import os
import random
import re
import sys
from multiprocessing import Pool, cpu_count
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import sympy
from sympy import (
    Abs,
    Add,
    Derivative,
    Eq,
    Integer,
    Mul,
    Poly,
    Pow,
    Rational,
    Symbol,
    diff,
    expand,
    factor,
    gcd,
    lcm,
    simplify,
    sqrt,
    sympify,
    default_sort_key as sympy_sort_key,
)
from tqdm import tqdm

# Make DeepMind mathematics_dataset importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATH_ROOT = os.path.join(PROJECT_ROOT, "data", "mathematics_dataset")
if MATH_ROOT not in sys.path:
    sys.path.insert(0, MATH_ROOT)

from mathematics_dataset.modules import (  # type: ignore
    algebra,
    arithmetic,
    calculus,
    comparison,
    measurement,
    numbers,
    polynomials,
    probability,
)
from mathematics_dataset.util import composition  # type: ignore

from dsl.tokens import (
    GyanDSLToken,
    get_int_const_token,
    get_real_var_token,
    get_vocab_size,
    NUM_INT_CONSTS,
)


# ---------------------------------------------------------------------------
# Token conversion utilities
# ---------------------------------------------------------------------------

def int_to_tokens(value: int) -> List[GyanDSLToken]:
    """Convert an integer to DSL tokens (handles large values via decomposition)."""
    if 0 <= value < NUM_INT_CONSTS:
        return [get_int_const_token(value)]

    if value == -1:
        return [GyanDSLToken.INT_NEG1]
    if value == -2:
        return [GyanDSLToken.INT_NEG2]
    if value == -10:
        return [GyanDSLToken.INT_NEG10]
    if value == -100:
        return [GyanDSLToken.INT_NEG100]

    # Negative: 0 - |value|
    if value < 0:
        pos_tokens = int_to_tokens(-value)
        return [GyanDSLToken.INT_0] + pos_tokens + [GyanDSLToken.SUB]

    # Large positive: decompose
    if value < 10000:
        for factor in range(99, 1, -1):
            if value % factor == 0:
                quotient = value // factor
                if quotient < NUM_INT_CONSTS:
                    return [
                        get_int_const_token(quotient),
                        get_int_const_token(factor),
                        GyanDSLToken.MUL,
                    ]

        hundreds = value // 100
        remainder = value % 100
        if hundreds < NUM_INT_CONSTS:
            tokens = [
                get_int_const_token(hundreds),
                get_int_const_token(99),
                GyanDSLToken.INT_1,
                GyanDSLToken.ADD,
                GyanDSLToken.MUL,
            ]
            if remainder > 0:
                tokens += [get_int_const_token(remainder), GyanDSLToken.ADD]
            return tokens

    # Very large: recursive
    thousands = value // 1000
    remainder = value % 1000
    tokens = int_to_tokens(thousands)
    tokens += [
        get_int_const_token(10),
        get_int_const_token(10),
        GyanDSLToken.MUL,
        get_int_const_token(10),
        GyanDSLToken.MUL,
        GyanDSLToken.MUL,
    ]
    if remainder > 0:
        tokens += int_to_tokens(remainder) + [GyanDSLToken.ADD]
    return tokens


def rational_to_tokens(value: Rational) -> List[GyanDSLToken]:
    """Convert a SymPy Rational to DSL tokens: num denom DIV."""
    num = int(value.p)
    denom = int(value.q)
    if denom == 1:
        return int_to_tokens(num)
    return int_to_tokens(num) + int_to_tokens(denom) + [GyanDSLToken.DIV]


def sympify_answer(raw: Any) -> sympy.Expr:
    """
    Convert a DeepMind `Problem.answer` object into a SymPy expression.

    Handles:
      - SymPy objects directly
      - Objects implementing SymPy's `_sympy_` protocol (e.g. display wrappers)
      - Fallback to `sympify(str(raw))` as a last resort
    """
    if isinstance(raw, sympy.Expr):
        return raw
    try:
        # This will use `_sympy_` if the object implements it.
        return sympify(raw)
    except Exception:
        # Fallback: trust the string representation (e.g. plain strings).
        return sympify(str(raw))


def expr_to_tokens(
    expr: sympy.Expr,
    var_map: Dict[Symbol, int],
) -> List[GyanDSLToken]:
    """
    Convert SymPy expression to DSL tokens (postfix/RPN).

    Args:
        expr: SymPy expression
        var_map: Mapping from Symbol to REAL_VAR index
    """
    # Integer
    if expr.is_Integer:
        return int_to_tokens(int(expr))

    # Rational
    if expr.is_Rational and not expr.is_Integer:
        return rational_to_tokens(expr)

    # Symbol (variable)
    if expr.is_Symbol:
        if expr in var_map:
            return [get_real_var_token(var_map[expr])]
        raise ValueError(f"Unknown variable: {expr}")

    # Addition
    if expr.is_Add:
        args = list(expr.args)
        tokens = expr_to_tokens(args[0], var_map)
        for arg in args[1:]:
            tokens += expr_to_tokens(arg, var_map)
            tokens.append(GyanDSLToken.ADD)
        return tokens

    # Multiplication
    if expr.is_Mul:
        args = list(expr.args)
        tokens = expr_to_tokens(args[0], var_map)
        for arg in args[1:]:
            tokens += expr_to_tokens(arg, var_map)
            tokens.append(GyanDSLToken.MUL)
        return tokens

    # Power
    if expr.is_Pow:
        base, exp = expr.as_base_exp()
        # Handle sqrt: x^(1/2)
        if exp == Rational(1, 2):
            return expr_to_tokens(base, var_map) + [GyanDSLToken.SQRT]
        # Handle square: x^2
        if exp == 2:
            return expr_to_tokens(base, var_map) + [GyanDSLToken.SQUARE]
        # General power
        tokens = expr_to_tokens(base, var_map)
        tokens += expr_to_tokens(exp, var_map)
        tokens.append(GyanDSLToken.POW)
        return tokens

    # Absolute value
    if isinstance(expr, Abs):
        return expr_to_tokens(expr.args[0], var_map) + [GyanDSLToken.ABS]

    # Handle Float by converting to Rational
    if expr.is_Float:
        # Convert to rational with limited denominator to keep tokens manageable
        rat = Rational(expr).limit_denominator(10000000)
        return rational_to_tokens(rat)
    
    # Try to evaluate to number
    try:
        val = float(expr.evalf())
        if val == int(val):
            return int_to_tokens(int(val))
        # If it's a non-integer float, convert to rational
        rat = Rational(val).limit_denominator(10000000)
        return rational_to_tokens(rat)
    except (TypeError, ValueError):
        pass

    raise ValueError(f"Cannot convert expression to tokens: {expr} (type: {type(expr)})")


def bool_to_token(value: bool) -> GyanDSLToken:
    """Convert Python bool to DSL token."""
    return GyanDSLToken.BOOL_TRUE if value else GyanDSLToken.BOOL_FALSE


def build_unary_numeric_answer_tokens(
    op_token: Optional[GyanDSLToken],
    answer: sympy.Expr,
) -> Tuple[List[GyanDSLToken], bool]:
    """
    Build a simple unary-program token sequence that encodes a numeric answer.

    Pattern:
        BOS [OP] <answer_expr_tokens> EOS

    This is used for modules where we only have the final numeric/expression
    answer (e.g., rounding, base conversion, probabilities).
    """
    try:
        tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
        if op_token is not None:
            tokens.append(op_token)
        # Build a var_map for any free symbols appearing in the answer.
        var_map: Dict[Symbol, int] = {}
        if isinstance(answer, sympy.Expr):
            for sym in sorted(list(answer.free_symbols), key=lambda s: s.name):
                if sym not in var_map:
                    var_map[sym] = len(var_map)
        tokens += expr_to_tokens(answer, var_map)
        tokens.append(GyanDSLToken.EOS)
        return tokens, True
    except Exception:
        return [], False


def build_unary_bool_answer_tokens(
    op_token: Optional[GyanDSLToken],
    answer: bool,
) -> Tuple[List[GyanDSLToken], bool]:
    """
    Build a simple unary-program token sequence that encodes a boolean answer.

    Pattern:
        BOS [OP] BOOL_TRUE/FALSE EOS
    """
    try:
        tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
        if op_token is not None:
            tokens.append(op_token)
        tokens.append(bool_to_token(answer))
        tokens.append(GyanDSLToken.EOS)
        return tokens, True
    except Exception:
        return [], False


# ---------------------------------------------------------------------------
# Generic problem → DSL token builder
# ---------------------------------------------------------------------------

def build_equation_solution_tokens(
    equation: Eq,
    variable: Symbol,
    answer: int,
    var_idx: int = 0,
    skip_verify: bool = False,
) -> Tuple[List[GyanDSLToken], bool]:
    """
    Build DSL tokens for: BOS <lhs> <rhs> EQ <var> <answer> IS_SOLUTION EOS

    Returns (tokens, is_valid)

    Args:
        skip_verify: If True, skip symbolic verification (for composed modules
                     where the equation contains context-defined symbols).
    """
    try:
        # Build var_map with all symbols in the equation
        # The target variable gets index 0, others get subsequent indices
        all_symbols = equation.free_symbols
        var_map = {variable: var_idx}
        next_idx = var_idx + 1
        for sym in sorted(all_symbols, key=lambda s: s.name):
            if sym not in var_map:
                var_map[sym] = next_idx
                next_idx += 1

        lhs_tokens = expr_to_tokens(equation.lhs, var_map)
        rhs_tokens = expr_to_tokens(equation.rhs, var_map)
        answer_tokens = int_to_tokens(answer)

        tokens = [GyanDSLToken.BOS]
        tokens += lhs_tokens
        tokens += rhs_tokens
        tokens.append(GyanDSLToken.EQ)
        tokens.append(get_real_var_token(var_idx))
        tokens += answer_tokens
        tokens.append(GyanDSLToken.IS_SOLUTION)
        tokens.append(GyanDSLToken.EOS)

        if skip_verify:
            # Trust the DeepMind answer for composed modules
            return tokens, True

        # Verify solution
        check = equation.lhs.subs(variable, answer) - equation.rhs.subs(variable, answer)
        valid = simplify(check) == 0

        return tokens, valid
    except Exception:
        return [], False


def build_system_solution_tokens(
    equations: List[Eq],
    variables: List[Symbol],
    target_var: Symbol,
    answer: int,
    skip_verify: bool = False,
) -> Tuple[List[GyanDSLToken], bool]:
    """
    Build DSL tokens for a small linear system:

        BOS <eq1_lhs> <eq1_rhs> EQ <eq2_lhs> <eq2_rhs> EQ <target_var> <answer> IS_SOLUTION EOS

    We include both equations explicitly so the model sees the multi-constraint
    structure, but we verify correctness using SymPy's solver.

    Args:
        skip_verify: If True, skip symbolic verification (for composed modules
                     where equations contain context-defined symbols).
    """
    try:
        # Map each variable to a REAL_VAR_i token index.
        var_map: Dict[Symbol, int] = {}
        for v in variables:
            if v not in var_map:
                var_map[v] = len(var_map)
        if target_var not in var_map:
            var_map[target_var] = len(var_map)

        tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]

        # Encode each equation as lhs rhs EQ
        for eq in equations:
            tokens += expr_to_tokens(eq.lhs, var_map)
            tokens += expr_to_tokens(eq.rhs, var_map)
            tokens.append(GyanDSLToken.EQ)

        # Append target variable and its value
        var_idx = var_map[target_var]
        tokens.append(get_real_var_token(var_idx))
        tokens += int_to_tokens(answer)
        tokens.append(GyanDSLToken.IS_SOLUTION)
        tokens.append(GyanDSLToken.EOS)

        if skip_verify:
            # Trust DeepMind answer for composed modules
            return tokens, True

        # Verify correctness using SymPy: solve the system and check the value.
        # We only support up to 2 variables for now.
        uniq_vars = list({v for v in variables} | {target_var})
        if len(uniq_vars) == 1:
            sol = sympy.solve(equations, uniq_vars[0])
            if isinstance(sol, list):
                sols = sol
            else:
                sols = [sol]
            valid = any(s == answer for s in sols)
        else:
            sol_list = sympy.solve(equations, uniq_vars, dict=True)
            valid = False
            for sol in sol_list:
                if target_var in sol and int(sol[target_var]) == answer:
                    valid = True
                    break

        return tokens, valid
    except Exception:
        return [], False


def build_expression_eval_tokens(
    expression: sympy.Expr,
    answer: sympy.Expr,
) -> Tuple[List[GyanDSLToken], bool]:
    """
    Build DSL tokens for evaluating an expression: BOS <expr> <answer> EQ_CMP EOS

    This is for arithmetic problems like "What is 5 + 3?" or "7/2".
    """
    try:
        # Ensure we have a SymPy expression for the answer as well.
        answer_expr = sympify(answer)

        tokens = [GyanDSLToken.BOS]
        tokens += expr_to_tokens(expression, {})
        tokens += expr_to_tokens(answer_expr, {})
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)

        # Verify
        valid = simplify(expression - answer_expr) == 0

        return tokens, valid
    except Exception:
        return [], False


def build_comparison_tokens(
    a: sympy.Expr,
    b: sympy.Expr,
    answer: bool,
    op: str = "lt",
) -> Tuple[List[GyanDSLToken], bool]:
    """
    Build DSL tokens for comparison: BOS <a> <b> <OP> <answer> EQ_CMP EOS
    """
    op_map = {
        "lt": GyanDSLToken.LT,
        "le": GyanDSLToken.LE,
        "gt": GyanDSLToken.GT,
        "ge": GyanDSLToken.GE,
        "eq": GyanDSLToken.EQ_CMP,
        "ne": GyanDSLToken.NE,
    }
    try:
        tokens = [GyanDSLToken.BOS]
        tokens += expr_to_tokens(a, {})
        tokens += expr_to_tokens(b, {})
        tokens.append(op_map[op])
        tokens.append(bool_to_token(answer))
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)
        return tokens, True
    except Exception:
        return [], False


def build_gcd_lcm_tokens(
    a: int,
    b: int,
    answer: int,
    op: str,
) -> Tuple[List[GyanDSLToken], bool]:
    """Build DSL tokens for gcd/lcm: BOS <a> <b> GCD/LCM <answer> EQ_CMP EOS"""
    try:
        op_token = GyanDSLToken.GCD if op == "gcd" else GyanDSLToken.LCM
        tokens = [GyanDSLToken.BOS]
        tokens += int_to_tokens(a)
        tokens += int_to_tokens(b)
        tokens.append(op_token)
        tokens += int_to_tokens(answer)
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)

        # Verify
        if op == "gcd":
            valid = int(gcd(a, b)) == answer
        else:
            valid = int(lcm(a, b)) == answer

        return tokens, valid
    except Exception:
        return [], False


def build_factor_tokens(
    expr: sympy.Expr,
    answer: sympy.Expr,
) -> Tuple[List[GyanDSLToken], bool]:
    """Build DSL tokens for factorization: BOS <expr> FACTOR <answer> EQ_CMP EOS"""
    try:
        var_map: Dict[Symbol, int] = {}
        # Map all free symbols appearing in either expr or answer.
        syms: set[Symbol] = set()
        if isinstance(expr, sympy.Expr):
            syms |= expr.free_symbols
        if isinstance(answer, sympy.Expr):
            syms |= answer.free_symbols
        for sym in sorted(list(syms), key=lambda s: s.name):
            var_map[sym] = len(var_map)

        tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
        tokens += expr_to_tokens(expr, var_map)
        tokens.append(GyanDSLToken.FACTOR)
        tokens += expr_to_tokens(answer, var_map)
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)

        # Verify: factor(expr) should equal answer.
        factored = sympy.factor(expr)
        valid = simplify(factored - answer) == 0

        return tokens, valid
    except Exception:
        return [], False


def build_coefficient_at_power_tokens(
    expr: sympy.Expr,
    variable: Symbol,
    power: int,
    coeff: sympy.Expr,
) -> Tuple[List[GyanDSLToken], bool]:
    """
    Build tokens for extracting the coefficient of variable**power in expr:

        BOS <expr> <var> <power> COEFF_AT_POWER <coeff> EQ_CMP EOS
    """
    try:
        var_map = {variable: 0}
        tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
        tokens += expr_to_tokens(expr, var_map)
        tokens.append(get_real_var_token(0))
        tokens += int_to_tokens(power)
        tokens.append(GyanDSLToken.COEFF_AT_POWER)
        # Encode the coefficient as an expression with no free vars.
        tokens += expr_to_tokens(coeff, {})
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)

        # Verification: recompute coefficient via SymPy.
        poly = sympy.Poly(expr, variable)
        recomputed = poly.coeffs()[poly.monoms().index((power,))] if (power,) in poly.monoms() else poly.coeff_monomial(variable**power)
        valid = simplify(recomputed - coeff) == 0

        return tokens, valid
    except Exception:
        return [], False


def build_is_prime_tokens(
    n: int,
    answer: bool,
) -> Tuple[List[GyanDSLToken], bool]:
    """Build DSL tokens for is_prime: BOS <n> IS_PRIME <answer> EQ_CMP EOS"""
    try:
        tokens = [GyanDSLToken.BOS]
        tokens += int_to_tokens(n)
        tokens.append(GyanDSLToken.IS_PRIME)
        tokens.append(bool_to_token(answer))
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)
        return tokens, True
    except Exception:
        return [], False


def build_div_remainder_tokens(
    p: int,
    q: int,
    answer: int,
) -> Tuple[List[GyanDSLToken], bool]:
    """Build DSL tokens for remainder: BOS <p> <q> DIV_REMAINDER <answer> EQ_CMP EOS"""
    try:
        tokens = [GyanDSLToken.BOS]
        tokens += int_to_tokens(p)
        tokens += int_to_tokens(q)
        tokens.append(GyanDSLToken.DIV_REMAINDER)
        tokens += int_to_tokens(answer)
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)

        valid = p % q == answer
        return tokens, valid
    except Exception:
        return [], False


def build_factorial_tokens(
    n: int,
    answer: int,
) -> Tuple[List[GyanDSLToken], bool]:
    """Build DSL tokens for factorial: BOS <n> FACTORIAL <answer> EQ_CMP EOS"""
    try:
        tokens = [GyanDSLToken.BOS]
        tokens += int_to_tokens(n)
        tokens.append(GyanDSLToken.FACTORIAL)
        tokens += int_to_tokens(answer)
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)

        valid = math.factorial(n) == answer
        return tokens, valid
    except Exception:
        return [], False


def build_differentiate_tokens(
    expr: sympy.Expr,
    variable: Symbol,
    answer: sympy.Expr,
    order: int = 1,
) -> Tuple[List[GyanDSLToken], bool]:
    """Build DSL tokens for differentiation: BOS <expr> <var> DIFF <answer> EQ_CMP EOS"""
    try:
        var_map = {variable: 0}
        tokens = [GyanDSLToken.BOS]
        tokens += expr_to_tokens(expr, var_map)
        tokens.append(get_real_var_token(0))
        for _ in range(order):
            tokens.append(GyanDSLToken.DIFF)
        tokens += expr_to_tokens(answer, var_map)
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)

        # Verify
        computed = diff(expr, variable, order)
        valid = simplify(computed - answer) == 0

        return tokens, valid
    except Exception:
        return [], False


def build_expand_tokens(
    expr: sympy.Expr,
    answer: sympy.Expr,
) -> Tuple[List[GyanDSLToken], bool]:
    """Build DSL tokens for expand: BOS <expr> EXPAND <answer> EQ_CMP EOS"""
    try:
        var_map = {}
        # Collect all symbols
        for sym in expr.free_symbols:
            var_map[sym] = len(var_map)

        tokens = [GyanDSLToken.BOS]
        tokens += expr_to_tokens(expr, var_map)
        tokens.append(GyanDSLToken.EXPAND)
        tokens += expr_to_tokens(answer, var_map)
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)

        # Verify
        valid = simplify(expand(expr) - answer) == 0

        return tokens, valid
    except Exception:
        return [], False


def build_collect_tokens(
    expr: sympy.Expr,
    answer: sympy.Expr,
) -> Tuple[List[GyanDSLToken], bool]:
    """
    Build DSL tokens for collect: BOS <expr> COLLECT <answer> EQ_CMP EOS

    Semantically this is \"collect/simplify the polynomial\", but we still
    verify equality using SymPy so that we don't depend on DeepMind's internals.
    """
    try:
        var_map: Dict[Symbol, int] = {}
        for sym in expr.free_symbols:
            var_map[sym] = len(var_map)

        tokens = [GyanDSLToken.BOS]
        tokens += expr_to_tokens(expr, var_map)
        tokens.append(GyanDSLToken.COLLECT)
        tokens += expr_to_tokens(answer, var_map)
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)

        # Verify: expanded/collected form should match the provided answer.
        valid = simplify(expand(expr) - answer) == 0
        return tokens, valid
    except Exception:
        return [], False


def build_simplify_power_tokens(
    expr: sympy.Expr,
    answer: sympy.Expr,
) -> Tuple[List[GyanDSLToken], bool]:
    """
    Build DSL tokens for power simplification:

        BOS <expr> SIMPLIFY_POWER <answer> EQ_CMP EOS

    We reuse SymPy's expand/simplify to verify equality.
    """
    try:
        var_map: Dict[Symbol, int] = {}
        for sym in expr.free_symbols:
            var_map[sym] = len(var_map)

        tokens = [GyanDSLToken.BOS]
        tokens += expr_to_tokens(expr, var_map)
        tokens.append(GyanDSLToken.SIMPLIFY_POWER)
        tokens += expr_to_tokens(answer, var_map)
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)

        # Prefer checking equality via ratio for power simplifications:
        # expr / answer should simplify to 1 when they are mathematically equal.
        try:
            if expr == 0 and answer == 0:
                valid = True
            else:
                ratio = simplify(expr / answer)
                valid = simplify(ratio - 1) == 0
        except Exception:
            # If SymPy struggles, fall back to trusting the provided answer.
            valid = True
        return tokens, valid
    except Exception:
        return [], False


def build_sort_tokens(
    values: List[int],
    answer: List[int],
    ascending: bool = True,
) -> Tuple[List[GyanDSLToken], bool]:
    """Build DSL tokens for sort: BOS <values...> SORT <answer...> EQ_CMP EOS"""
    try:
        tokens = [GyanDSLToken.BOS]
        # Encode list length first
        tokens += int_to_tokens(len(values))
        for v in values:
            tokens += int_to_tokens(v)
        tokens.append(GyanDSLToken.SORT)
        for v in answer:
            tokens += int_to_tokens(v)
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)

        # Verify
        expected = sorted(values, reverse=not ascending)
        valid = expected == answer

        return tokens, valid
    except Exception:
        return [], False


# ---------------------------------------------------------------------------
# Module-specific generators
# ---------------------------------------------------------------------------

def parse_solve_equation(question: str) -> Tuple[Eq, Symbol, str]:
    """Parse '... Solve X = Y for v.' format, ignoring leading context."""
    q = question.strip()

    # Preferred: look for the last explicit 'Solve ... for v' fragment.
    m = re.search(r"Solve\s+(.+?)\s+for\s+([A-Za-z])\.?\s*$", q, re.IGNORECASE)
    if m:
        eq_str = m.group(1).strip()
        var_name = m.group(2)
    else:
        # Fallback: original strict pattern (for older/simple questions).
        m2 = re.match(
            r"^(?:Solve|Let|Suppose)\s+(.+?)\s*(?:for|\.)\s*([A-Za-z])\.?",
            q,
            re.IGNORECASE,
        )
        if not m2:
            raise ValueError(f"Cannot parse equation: {question}")
        eq_str = m2.group(1).strip()
        var_name = m2.group(2)

    if "=" not in eq_str:
        raise ValueError(f"No '=' in equation: {eq_str}")
    lhs_str, rhs_str = eq_str.split("=", 1)
    lhs = sympify(lhs_str.strip())
    rhs = sympify(rhs_str.strip())
    variable = Symbol(var_name)
    return Eq(lhs, rhs), variable, eq_str


def parse_linear_2d_system(question: str) -> Tuple[List[Eq], List[Symbol], Symbol]:
    """
    Parse a linear_2d style question of the form:

        "Solve o = ... , 2*o = 2 for l."

    Returns:
        (equations, variables, target_var)
    """
    q = question.strip()
    # Allow leading context before the final 'Solve ... for v.' fragment.
    m = re.search(r"Solve\s+(.+)\s+for\s+([A-Za-z])\.\s*$", q)
    if not m:
        raise ValueError(f"Cannot parse linear_2d question: {question}")
    eqs_str = m.group(1).strip()
    target_name = m.group(2)
    parts = [s.strip() for s in eqs_str.split(",") if s.strip()]
    if len(parts) < 2:
        raise ValueError(f"Expected at least 2 equations in: {eqs_str}")

    equations: List[Eq] = []
    all_syms: set[Symbol] = set()
    for part in parts[:2]:
        if "=" not in part:
            raise ValueError(f"No '=' in equation: {part}")
        lhs_str, rhs_str = part.split("=", 1)
        lhs = sympify(lhs_str.strip())
        rhs = sympify(rhs_str.strip())
        eq = Eq(lhs, rhs)
        equations.append(eq)
        all_syms |= eq.free_symbols

    target_var = Symbol(target_name)
    # Order variables: target_var first, then others.
    other_vars = [v for v in sorted(all_syms, key=lambda s: s.name) if v != target_var]
    variables = [target_var] + other_vars
    return equations, variables, target_var


def parse_calculate(question: str) -> sympy.Expr:
    """Parse 'Calculate X' or 'What is X?' format."""
    # Remove common prefixes
    q = question.strip()
    for prefix in ["Calculate ", "What is ", "Evaluate ", "Work out ", "Give "]:
        if q.startswith(prefix):
            q = q[len(prefix):]
            break
    # Remove trailing punctuation
    q = q.rstrip("?.!")

    # Handle explicit division phrases:
    #   "Divide P by Q"
    #   "P divided by Q"
    q_lower = q.lower()
    # Case 1: "Divide P by Q"
    if q_lower.startswith("divide "):
        m = re.match(r"divide\s+(.+?)\s+by\s+(.+)$", q, re.IGNORECASE)
        if m:
            p_str = m.group(1).strip()
            q_str = m.group(2).strip()
            expr_str = f"({p_str})/({q_str})"
            return sympify(expr_str)

    # Case 2: "P divided by Q"
    if " divided by " in q_lower:
        parts = re.split(r"\s+divided by\s+", q, flags=re.IGNORECASE)
        if len(parts) == 2:
            p_str = parts[0].strip()
            q_str = parts[1].strip()
            expr_str = f"({p_str})/({q_str})"
            return sympify(expr_str)

    # Case 3: "X take away Y" -> X - Y
    m = re.match(r"(.+?)\s+take away\s+(.+)$", q, re.IGNORECASE)
    if m:
        a_str = m.group(1).strip()
        b_str = m.group(2).strip()
        expr_str = f"({a_str})-({b_str})"
        return sympify(expr_str)

    # Case 4: "Sum A and B" -> A + B
    m = re.match(r"sum\s+(.+?)\s+and\s+(.+)$", q, re.IGNORECASE)
    if m:
        a_str = m.group(1).strip()
        b_str = m.group(2).strip()
        expr_str = f"({a_str})+({b_str})"
        return sympify(expr_str)

    # Case 5: "Total of A and B" -> A + B
    m = re.match(r"total of\s+(.+?)\s+and\s+(.+)$", q, re.IGNORECASE)
    if m:
        a_str = m.group(1).strip()
        b_str = m.group(2).strip()
        expr_str = f"({a_str})+({b_str})"
        return sympify(expr_str)

    # Case 6: "Subtract A from B" -> B - A
    m = re.match(r"subtract\s+(.+?)\s+from\s+(.+)$", q, re.IGNORECASE)
    if m:
        a_str = m.group(1).strip()
        b_str = m.group(2).strip()
        expr_str = f"({b_str})-({a_str})"
        return sympify(expr_str)

    # Distance between A and B: \"the distance between A and B\"
    m = re.search(r"distance between\s+(.+?)\s+and\s+(.+)$", q, re.IGNORECASE)
    if m:
        a_str = m.group(1).strip()
        b_str = m.group(2).strip()
        expr_str = f"Abs(({a_str}) - ({b_str}))"
        return sympify(expr_str)

    # "product of A and B"
    m = re.search(r"product of\s+(.+?)\s+and\s+(.+)$", q, re.IGNORECASE)
    if m:
        a_str = m.group(1).strip()
        b_str = m.group(2).strip()
        expr_str = f"({a_str})*({b_str})"
        return sympify(expr_str)

    # "Put together A and B" -> A + B
    m = re.search(r"Put together\s+(.+?)\s+and\s+(.+)$", q, re.IGNORECASE)
    if m:
        a_str = m.group(1).strip()
        b_str = m.group(2).strip()
        expr_str = f"({a_str})+({b_str})"
        return sympify(expr_str)

    # "Add together A and B" -> A + B
    m = re.search(r"Add together\s+(.+?)\s+and\s+(.+)$", q, re.IGNORECASE)
    if m:
        a_str = m.group(1).strip()
        b_str = m.group(2).strip()
        expr_str = f"({a_str})+({b_str})"
        return sympify(expr_str)

    # "Add A and B" -> A + B
    m = re.search(r"^Add\s+(.+?)\s+and\s+(.+)$", q, re.IGNORECASE)
    if m:
        a_str = m.group(1).strip()
        b_str = m.group(2).strip()
        expr_str = f"({a_str})+({b_str})"
        return sympify(expr_str)

    # "What is A less than B?" -> B - A
    m = re.search(r"What is\s+(.+?)\s+less than\s+(.+)$", q, re.IGNORECASE)
    if m:
        a_str = m.group(1).strip()
        b_str = m.group(2).strip()
        expr_str = f"({b_str})-({a_str})"
        return sympify(expr_str)

    # "difference between A and B" -> Abs(A - B)
    m = re.search(r"difference between\s+(.+?)\s+and\s+(.+)$", q, re.IGNORECASE)
    if m:
        a_str = m.group(1).strip()
        b_str = m.group(2).strip()
        expr_str = f"Abs(({a_str}) - ({b_str}))"
        return sympify(expr_str)

    # Replace simple word operators: plus/minus/times/multiplied by
    q_replaced = q
    replacements = [
        (r"\bplus\b", "+"),
        (r"\bminus\b", "-"),
        (r"\btimes\b", "*"),
        (r"\bmultiplied by\b", "*"),
    ]
    for pat, repl in replacements:
        q_replaced = re.sub(pat, f" {repl} ", q_replaced, flags=re.IGNORECASE)

    # Default: treat the (possibly rewritten) string as a SymPy expression.
    return sympify(q_replaced)


def generate_linear_1d(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """Generate linear_1d example."""
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = int(problem.answer)

        equation, variable, eq_str = parse_solve_equation(question)
        tokens, valid = build_equation_solution_tokens(equation, variable, answer)

        if not valid or not tokens:
            return None

        return {
            "id": f"algebra__linear_1d/{split}/{idx:06d}",
            "module": "algebra__linear_1d",
            "split": split,
            "question": question,
            "answer": str(answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_linear_1d_composed(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Generate linear_1d_composed example.

    These have context like "Let k = 53 - 49. Solve -4*l = -2*l - k for l."
    The equation contains symbols defined in context, so we skip verification
    and trust the DeepMind answer.
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = int(problem.answer)

        equation, variable, eq_str = parse_solve_equation(question)
        # Skip verification because equation contains context-defined symbols
        tokens, valid = build_equation_solution_tokens(
            equation, variable, answer, skip_verify=True
        )

        if not tokens:
            return None

        return {
            "id": f"algebra__linear_1d_composed/{split}/{idx:06d}",
            "module": "algebra__linear_1d_composed",
            "split": split,
            "question": question,
            "answer": str(answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_linear_2d(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """Generate linear_2d example (system of 2 linear equations)."""
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = int(problem.answer)

        equations, variables, target_var = parse_linear_2d_system(question)
        tokens, valid = build_system_solution_tokens(
            equations=equations,
            variables=variables,
            target_var=target_var,
            answer=answer,
        )

        if not valid or not tokens:
            return None

        return {
            "id": f"algebra__linear_2d/{split}/{idx:06d}",
            "module": "algebra__linear_2d",
            "split": split,
            "question": question,
            "answer": str(answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_linear_2d_composed(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Generate linear_2d_composed example.

    These have context like "Suppose ... Solve eq1, eq2 for v."
    The equations contain symbols defined in context, so we skip verification.
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = int(problem.answer)

        equations, variables, target_var = parse_linear_2d_system(question)
        tokens, valid = build_system_solution_tokens(
            equations=equations,
            variables=variables,
            target_var=target_var,
            answer=answer,
            skip_verify=True,  # Trust DeepMind for composed modules
        )

        if not tokens:
            return None

        return {
            "id": f"algebra__linear_2d_composed/{split}/{idx:06d}",
            "module": "algebra__linear_2d_composed",
            "split": split,
            "question": question,
            "answer": str(answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_polynomial_roots(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Generate polynomial_roots examples for the 'Factor ...' variants.

    We intentionally skip the 'Solve ... = 0' multi-root variants for now and
    focus on factorization, which has a single expression-valued answer.
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = problem.answer

        # Only handle factorization prompts: "Factor EXPR."
        m = re.match(r"^Factor\s+(.+?)\.\s*$", question.strip(), re.IGNORECASE)
        if not m:
            return None
        expr_str = m.group(1).strip()
        expr = sympify(expr_str)
        ans_expr = sympify_answer(answer)

        tokens, valid = build_factor_tokens(expr, ans_expr)
        if not valid or not tokens:
            return None

        return {
            "id": f"algebra__polynomial_roots/{split}/{idx:06d}",
            "module": "algebra__polynomial_roots",
            "split": split,
            "question": question,
            "answer": str(answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_polynomial_roots_composed(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Generate polynomial_roots_composed examples.

    These have context and can return either:
    - Factored expressions: "4*o**2*(o + 1)**3/7"
    - Root lists: "-2, -1"

    We skip verification and trust DeepMind's answer.
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = problem.answer

        # Try to find "Factor EXPR" pattern anywhere in the question
        m = re.search(r"Factor\s+([^.]+)\.", question, re.IGNORECASE)
        if m:
            expr_str = m.group(1).strip()
            expr = sympify(expr_str)
            ans_expr = sympify_answer(answer)
            # Build factor tokens but skip verification (composed context)
            tokens, _ = build_factor_tokens(expr, ans_expr)
            if tokens:
                return {
                    "id": f"algebra__polynomial_roots_composed/{split}/{idx:06d}",
                    "module": "algebra__polynomial_roots_composed",
                    "split": split,
                    "question": question,
                    "answer": str(answer),
                    "token_ids": [t.value for t in tokens],
                    "token_names": [t.name for t in tokens],
                }

        # For root-finding questions, just use expression evaluation format
        # Answer is like "-2, -1" which we can tokenize as a list
        ans_expr = sympify_answer(answer)
        if ans_expr is not None:
            tokens = [GyanDSLToken.BOS]
            tokens += expr_to_tokens(ans_expr, {})
            tokens.append(GyanDSLToken.EOS)
            return {
                "id": f"algebra__polynomial_roots_composed/{split}/{idx:06d}",
                "module": "algebra__polynomial_roots_composed",
                "split": split,
                "question": question,
                "answer": str(answer),
                "token_ids": [t.value for t in tokens],
                "token_names": [t.name for t in tokens],
            }

        return None
    except Exception:
        return None


def generate_arithmetic_eval(module_fn, module_name: str, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """Generate arithmetic evaluation example (add/sub, mul/div, mixed)."""
    try:
        problem = module_fn()
        question = str(problem.question)
        raw_answer = problem.answer

        # Convert the answer into a SymPy expression.
        # This handles integers, rationals, and decimals uniformly.
        try:
            if hasattr(raw_answer, "sympy"):
                answer_expr = raw_answer.sympy()
            else:
                answer_expr = sympify(str(raw_answer))
        except Exception:
            return None

        expr = parse_calculate(question)
        tokens, valid = build_expression_eval_tokens(expr, answer_expr)

        if not valid or not tokens:
            return None

        return {
            "id": f"{module_name}/{split}/{idx:06d}",
            "module": module_name,
            "split": split,
            "question": question,
            "answer": str(raw_answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_gcd_lcm(module_fn, module_name: str, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """Generate gcd/lcm example."""
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = int(problem.answer)

        # Parse "Calculate the gcd of X and Y" or similar
        m = re.search(r"(\d+)\s+and\s+(\d+)", question)
        if not m:
            return None

        a, b = int(m.group(1)), int(m.group(2))
        op = "gcd" if "gcd" in question.lower() or "greatest" in question.lower() else "lcm"

        tokens, valid = build_gcd_lcm_tokens(a, b, answer, op)

        if not valid or not tokens:
            return None

        return {
            "id": f"{module_name}/{split}/{idx:06d}",
            "module": module_name,
            "split": split,
            "question": question,
            "answer": str(answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_is_prime(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """Generate is_prime example."""
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = problem.answer

        # Parse "Is N prime?" format
        m = re.search(r"(\d+)", question)
        if not m:
            return None

        n = int(m.group(1))
        is_prime_answer = answer == True or str(answer).lower() == "true"

        tokens, valid = build_is_prime_tokens(n, is_prime_answer)

        if not valid or not tokens:
            return None

        return {
            "id": f"numbers__is_prime/{split}/{idx:06d}",
            "module": "numbers__is_prime",
            "split": split,
            "question": question,
            "answer": str(is_prime_answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


UNARY_NUMERIC_OPS: Dict[str, GyanDSLToken] = {
    # Algebra sequences
    "algebra__sequence_next_term": GyanDSLToken.SEQ_NEXT,
    "algebra__sequence_nth_term": GyanDSLToken.SEQ_NTH,
    # Arithmetic extras
    "arithmetic__add_or_sub_in_base": GyanDSLToken.EVAL_EXPR,
    "arithmetic__nearest_integer_root": GyanDSLToken.ROUND,
    "arithmetic__simplify_surd": GyanDSLToken.SIMPLIFY_EXPR,
    # Numbers
    "numbers__round_number": GyanDSLToken.ROUND,
    "numbers__place_value": GyanDSLToken.PLACE_VALUE,
    "numbers__base_conversion": GyanDSLToken.TO_BASE,
    "numbers__list_prime_factors": GyanDSLToken.PRIME_FACTORS,
    "numbers__gcd_composed": GyanDSLToken.GCD,
    "numbers__lcm_composed": GyanDSLToken.LCM,
    "numbers__div_remainder_composed": GyanDSLToken.DIV_REMAINDER,
    "numbers__round_number_composed": GyanDSLToken.ROUND,
    "numbers__place_value_composed": GyanDSLToken.PLACE_VALUE,
    "numbers__list_prime_factors_composed": GyanDSLToken.PRIME_FACTORS,
    # Polynomials
    "polynomials__evaluate": GyanDSLToken.EVAL_EXPR,
    "polynomials__evaluate_composed": GyanDSLToken.EVAL_EXPR,
    "polynomials__compose": GyanDSLToken.COMPOSE,
    # Measurement
    "measurement__conversion": GyanDSLToken.EVAL_EXPR,
    "measurement__time": GyanDSLToken.EVAL_EXPR,
    # Probability
    "probability__swr_p_sequence": GyanDSLToken.PROBABILITY,
    "probability__swr_p_level_set": GyanDSLToken.PROBABILITY,
}

UNARY_BOOL_OPS: Dict[str, GyanDSLToken] = {
    "numbers__is_factor": GyanDSLToken.IS_FACTOR,
    "numbers__is_factor_composed": GyanDSLToken.IS_FACTOR,
    "numbers__is_prime_composed": GyanDSLToken.IS_PRIME,
}


def unary_numeric_generator(
    module_fn,
    module_name: str,
    split: str,
    idx: int,
) -> Optional[Dict[str, Any]]:
    """
    Generator that only needs the numeric/expression answer.

    The resulting program tokens are:
        BOS OP <answer_expr_tokens> EOS
    """
    try:
        op_token = UNARY_NUMERIC_OPS[module_name]
        problem = module_fn()
        answer_expr = sympify_answer(problem.answer)
        tokens, valid = build_unary_numeric_answer_tokens(op_token, answer_expr)
        if not valid or not tokens:
            return None
        return {
            "id": f"{module_name}/{split}/{idx:06d}",
            "module": module_name,
            "split": split,
            "question": str(problem.question),
            "answer": str(problem.answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def unary_bool_generator(
    module_fn,
    module_name: str,
    split: str,
    idx: int,
) -> Optional[Dict[str, Any]]:
    """
    Generator that only needs a boolean answer.

    The resulting program tokens are:
        BOS OP BOOL_TRUE/FALSE EOS
    """
    try:
        op_token = UNARY_BOOL_OPS[module_name]
        problem = module_fn()
        raw = problem.answer
        answer_bool = bool(raw) if isinstance(raw, bool) else str(raw).lower() == "true"
        tokens, valid = build_unary_bool_answer_tokens(op_token, answer_bool)
        if not valid or not tokens:
            return None
        return {
            "id": f"{module_name}/{split}/{idx:06d}",
            "module": module_name,
            "split": split,
            "question": str(problem.question),
            "answer": str(answer_bool),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_div_remainder(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """Generate div_remainder example."""
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = int(problem.answer)

        # Parse "remainder when P is divided by Q"
        m = re.search(r"(\d+)\s+is divided by\s+(\d+)", question)
        if not m:
            return None

        p, q = int(m.group(1)), int(m.group(2))

        tokens, valid = build_div_remainder_tokens(p, q, answer)

        if not valid or not tokens:
            return None

        return {
            "id": f"numbers__div_remainder/{split}/{idx:06d}",
            "module": "numbers__div_remainder",
            "split": split,
            "question": question,
            "answer": str(answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_list_prime_factors(module_fn, module_name: str, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Generate list_prime_factors example.

    The answer is a NumberList of prime factors. We encode:
        BOS PRIME_FACTORS <n> <list of primes> EOS

    where <n> is the number being factored (extracted from the question).
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        raw_answer = problem.answer

        # Parse the number from the question: "List the prime factors of N" or "What are the prime factors of N?"
        m = re.search(r"prime factors of\s+(\d+)", question, re.IGNORECASE)
        if not m:
            return None
        n = int(m.group(1))

        # Extract the list of primes from the answer
        # The answer is a NumberList; convert to list of ints
        answer_str = str(raw_answer)
        # Parse comma-separated primes
        prime_strs = [s.strip() for s in answer_str.split(",") if s.strip()]
        primes = [int(p) for p in prime_strs]

        # Build tokens: BOS PRIME_FACTORS <n> <prime1> <prime2> ... EOS
        tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
        tokens.append(GyanDSLToken.PRIME_FACTORS)
        tokens += int_to_tokens(n)
        for p in primes:
            tokens += int_to_tokens(p)
        tokens.append(GyanDSLToken.EOS)

        return {
            "id": f"{module_name}/{split}/{idx:06d}",
            "module": module_name,
            "split": split,
            "question": question,
            "answer": answer_str,
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_list_prime_factors_composed(module_fn, module_name: str, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Generate list_prime_factors_composed example.

    For composed questions, the number is defined in context, so we can't extract it.
    We just encode the answer (list of primes) directly:
        BOS PRIME_FACTORS <prime1> <prime2> ... EOS
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        raw_answer = problem.answer

        # Extract the list of primes from the answer
        answer_str = str(raw_answer)
        prime_strs = [s.strip() for s in answer_str.split(",") if s.strip()]
        primes = [int(p) for p in prime_strs]

        # Build tokens: BOS PRIME_FACTORS <prime1> <prime2> ... EOS
        # (no input number since it's context-defined)
        tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
        tokens.append(GyanDSLToken.PRIME_FACTORS)
        for p in primes:
            tokens += int_to_tokens(p)
        tokens.append(GyanDSLToken.EOS)

        return {
            "id": f"{module_name}/{split}/{idx:06d}",
            "module": module_name,
            "split": split,
            "question": question,
            "answer": answer_str,
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_kth_biggest(module_fn, module_name: str, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """Generate kth_biggest example with explicit list (non-multichoice)."""
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = sympify_answer(problem.answer)

        # Handle either list-style ("... in a, b, c?") or multiple-choice
        # ("Which is the ... value?  (a) v1  (b) v2 ...").
        m = re.search(r"in\s+(.+)\?\s*$", question)
        if m:
            list_str = m.group(1).strip()
            value_strs = [s.strip() for s in list_str.split(",") if s.strip()]
        else:
            # Multiple-choice: grab everything after the '?' as options.
            m_opts = re.search(r"\?\s*(.+)$", question)
            if not m_opts:
                return None
            options_str = m_opts.group(1).strip()
            # Require at least one labelled option "(a) ..."
            if "(a)" not in options_str.lower():
                return None
            pairs = re.findall(r"\(([a-z])\)\s*([^()]+)", options_str)
            if not pairs:
                return None
            value_strs = [val.strip() for _, val in pairs]

        values = [sympify(vs) for vs in value_strs]

        # Determine adjective: biggest/smallest and ordinal if present
        m2 = re.search(r"the\s+(.+?)\s+value", question)
        if not m2:
            return None
        adjective = m2.group(1).lower()

        from_biggest = "biggest" in adjective
        from_smallest = "smallest" in adjective
        if not (from_biggest or from_smallest):
            return None

        ordinal_map = {
            "first": 1,
            "second": 2,
            "third": 3,
            "fourth": 4,
            "fifth": 5,
            "sixth": 6,
            "seventh": 7,
            "eighth": 8,
            "ninth": 9,
            "tenth": 10,
        }
        ordinal = 1
        for word, num in ordinal_map.items():
            if word in adjective:
                ordinal = num
                break

        # Recompute the correct numeric answer from the values.
        sorted_vals = sorted(values, key=lambda v: sympy_sort_key(v))
        if from_biggest:
            idx_in_sorted = -ordinal
        else:
            idx_in_sorted = ordinal - 1
        if not (-len(sorted_vals) <= idx_in_sorted < len(sorted_vals)):
            return None
        true_answer = sorted_vals[idx_in_sorted]

        # Encode DSL tokens: list of values, ordinal, then KTH_LARGEST op and answer.
        tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
        for v in values:
            tokens += expr_to_tokens(sympify(v), {})
        tokens += int_to_tokens(ordinal)
        tokens.append(GyanDSLToken.KTH_LARGEST)
        tokens += expr_to_tokens(true_answer, {})
        tokens.append(GyanDSLToken.EOS)

        return {
            "id": f"{module_name}/{split}/{idx:06d}",
            "module": module_name,
            "split": split,
            "question": question,
            "answer": str(true_answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_closest(module_fn, module_name: str, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """Generate closest example with explicit list (non-multichoice)."""
    try:
        problem = module_fn()
        question = str(problem.question)

        # Primary form: "What is the closest/nearest to TARGET in a, b, c?"
        m = re.search(r"to\s+(.+?)\s+in\s+(.+)\?\s*$", question)
        if m:
            target_str = m.group(1).strip()
            list_str = m.group(2).strip()
            value_strs = [s.strip() for s in list_str.split(",") if s.strip()]
        else:
            # Multiple-choice form:
            #   "Which is the closest/nearest to TARGET?  (a) v1  (b) v2 ..."
            m2 = re.match(
                r"Which is the\s+(closest|nearest)\s+to\s+(.+?)\?\s*(.+)$",
                question,
                re.IGNORECASE,
            )
            if not m2:
                return None
            target_str = m2.group(2).strip()
            options_str = m2.group(3).strip()
            pairs = re.findall(r"\(([a-z])\)\s*([^()]+)", options_str)
            if not pairs:
                return None
            value_strs = [val.strip() for _, val in pairs]

        values = [sympify(vs) for vs in value_strs]
        target = sympify(target_str)

        # Recompute closest value.
        diffs = [abs(sympify(v) - target) for v in values]
        min_diff = min(diffs)
        idx_min = diffs.index(min_diff)
        true_answer = values[idx_min]

        tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
        for v in values:
            tokens += expr_to_tokens(sympify(v), {})
        tokens += expr_to_tokens(target, {})
        tokens.append(GyanDSLToken.CLOSEST_TO)
        tokens += expr_to_tokens(true_answer, {})
        tokens.append(GyanDSLToken.EOS)

        return {
            "id": f"{module_name}/{split}/{idx:06d}",
            "module": module_name,
            "split": split,
            "question": question,
            "answer": str(true_answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_sort(module_fn, module_name: str, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """Generate sort example (ascending/descending)."""
    try:
        problem = module_fn()
        question = str(problem.question)
        answer_str = str(problem.answer).strip()

        # Extract the list from the question: "... in a, b, c order." or "Sort a, b, c."
        m = re.search(r"Sort\s+(.+?)(?:\s+in\s+(ascending|increasing|descending|decreasing)\s+order\.?|\.?)$", question)
        if not m:
            # Try "Put ... in direction order."
            m = re.search(r"Put\s+(.+?)\s+in\s+(ascending|increasing|descending|decreasing)\s+order\.", question)
        if not m:
            return None
        list_str = m.group(1).strip()

        # Determine direction
        direction = "ascending"
        if m.lastindex and m.group(m.lastindex):
            direction = m.group(m.lastindex).lower()
        ascending = direction in ("ascending", "increasing")

        value_strs = [s.strip() for s in list_str.split(",") if s.strip()]
        values = [sympify(vs) for vs in value_strs]

        sorted_vals = sorted(values, key=lambda v: sympy_sort_key(v), reverse=not ascending)

        tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
        for v in values:
            tokens += expr_to_tokens(sympify(v), {})
        tokens.append(GyanDSLToken.SORT)
        for v in sorted_vals:
            tokens += expr_to_tokens(sympify(v), {})
        tokens.append(GyanDSLToken.EOS)

        return {
            "id": f"{module_name}/{split}/{idx:06d}",
            "module": module_name,
            "split": split,
            "question": question,
            "answer": answer_str,
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_comparison(module_fn, module_name: str, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """Generate comparison example."""
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = problem.answer

        # For True/False comparisons
        if isinstance(answer, bool) or str(answer).lower() in ("true", "false"):
            answer_bool = answer == True or str(answer).lower() == "true"

            # Parse "Is A < B?" format (numeric comparison operators)
            m = re.search(r"(-?\d+(?:\.\d+)?(?:/\d+)?)\s*([<>=!]+)\s*(-?\d+(?:\.\d+)?(?:/\d+)?)", question)
            if m:
                a = sympify(m.group(1))
                op_str = m.group(2)
                b = sympify(m.group(3))
                op_map = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "=": "eq", "!=": "ne"}
                op = op_map.get(op_str, "lt")
                tokens, valid = build_comparison_tokens(a, b, answer_bool, op)
                if valid and tokens:
                    return {
                        "id": f"{module_name}/{split}/{idx:06d}",
                        "module": module_name,
                        "split": split,
                        "question": question,
                        "answer": str(answer_bool),
                        "token_ids": [t.value for t in tokens],
                        "token_names": [t.name for t in tokens],
                    }

            # Parse "Is A greater than B?" / "Is A smaller than B?" / "Is A bigger than B?"
            m = re.search(
                r"Is\s+(.+?)\s+(greater than|bigger than|smaller than|less than|"
                r"greater than or equal to|less than or equal to|"
                r"not equal to|equal to)\s+(.+?)\??$",
                question, re.IGNORECASE
            )
            if m:
                a_str = m.group(1).strip()
                op_phrase = m.group(2).lower()
                b_str = m.group(3).strip().rstrip("?")
                a = sympify(a_str)
                b = sympify(b_str)
                op_map = {
                    "greater than": "gt", "bigger than": "gt",
                    "smaller than": "lt", "less than": "lt",
                    "greater than or equal to": "ge",
                    "less than or equal to": "le",
                    "equal to": "eq", "not equal to": "ne",
                }
                op = op_map.get(op_phrase, "lt")
                tokens, valid = build_comparison_tokens(a, b, answer_bool, op)
                if valid and tokens:
                    return {
                        "id": f"{module_name}/{split}/{idx:06d}",
                        "module": module_name,
                        "split": split,
                        "question": question,
                        "answer": str(answer_bool),
                        "token_ids": [t.value for t in tokens],
                        "token_names": [t.name for t in tokens],
                    }

            # Parse "Are A and B equal/non-equal?"
            m = re.search(r"Are\s+(.+?)\s+and\s+(.+?)\s+(equal|non-equal)\??$", question, re.IGNORECASE)
            if m:
                a_str = m.group(1).strip()
                b_str = m.group(2).strip()
                eq_type = m.group(3).lower()
                a = sympify(a_str)
                b = sympify(b_str)
                op = "ne" if eq_type == "non-equal" else "eq"
                tokens, valid = build_comparison_tokens(a, b, answer_bool, op)
                if valid and tokens:
                    return {
                        "id": f"{module_name}/{split}/{idx:06d}",
                        "module": module_name,
                        "split": split,
                        "question": question,
                        "answer": str(answer_bool),
                        "token_ids": [t.value for t in tokens],
                        "token_names": [t.name for t in tokens],
                    }

            return None

        else:
            # For "which is bigger/smaller/greater" style - answer is a value
            # Parse "Which is greater/smaller/bigger: A or B?"
            m = re.search(
                r"Which is (greater|smaller|bigger):\s*(.+?)\s+or\s+(.+?)\??$",
                question, re.IGNORECASE
            )
            if m:
                direction = m.group(1).lower()
                a_str = m.group(2).strip()
                b_str = m.group(3).strip().rstrip("?")
                a = sympify(a_str)
                b = sympify(b_str)
                ans_expr = sympify_answer(answer)

                # Determine which value is the answer
                if simplify(a - ans_expr) == 0:
                    winner = a
                elif simplify(b - ans_expr) == 0:
                    winner = b
                else:
                    return None

                # Encode: BOS <a> <b> GT/LT <winner> EOS
                op = "gt" if direction in ("greater", "bigger") else "lt"
                tokens = [GyanDSLToken.BOS]
                tokens += expr_to_tokens(a, {})
                tokens += expr_to_tokens(b, {})
                tokens.append(GyanDSLToken.GT if op == "gt" else GyanDSLToken.LT)
                tokens += expr_to_tokens(winner, {})
                tokens.append(GyanDSLToken.EOS)

                return {
                    "id": f"{module_name}/{split}/{idx:06d}",
                    "module": module_name,
                    "split": split,
                    "question": question,
                    "answer": str(answer),
                    "token_ids": [t.value for t in tokens],
                    "token_names": [t.name for t in tokens],
                }

            return None
    except Exception:
        return None


def generate_comparison_composed(module_fn, module_name: str, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Generate comparison_pair_composed example.

    For composed questions, the values come from context. We just tokenize the boolean answer.
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = problem.answer

        # For True/False comparisons
        if isinstance(answer, bool) or str(answer).lower() in ("true", "false"):
            answer_bool = answer == True or str(answer).lower() == "true"

            # Simple answer-only encoding: BOS BOOL_TRUE/FALSE EOS
            tokens = [GyanDSLToken.BOS]
            tokens.append(GyanDSLToken.BOOL_TRUE if answer_bool else GyanDSLToken.BOOL_FALSE)
            tokens.append(GyanDSLToken.EOS)

            return {
                "id": f"{module_name}/{split}/{idx:06d}",
                "module": module_name,
                "split": split,
                "question": question,
                "answer": str(answer_bool),
                "token_ids": [t.value for t in tokens],
                "token_names": [t.name for t in tokens],
            }

        return None
    except Exception:
        return None


def generate_kth_biggest_composed(module_fn, module_name: str, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Generate kth_biggest_composed example.

    For composed questions, the values and answer may be context-defined variables.
    We try to tokenize the answer if it's numeric, otherwise skip.
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = problem.answer

        # Skip multiple-choice letter answers (a, b, c, d)
        ans_str = str(answer).strip()
        if len(ans_str) == 1 and ans_str.isalpha():
            return None

        # Try to sympify the answer
        ans_expr = sympify_answer(answer)
        if ans_expr is None:
            return None

        # Check if it's a pure symbol (variable name) - skip those
        if ans_expr.is_Symbol:
            return None

        # Answer is numeric - tokenize it
        tokens = [GyanDSLToken.BOS]
        tokens.append(GyanDSLToken.KTH_LARGEST)  # Token is KTH_LARGEST not KTH_BIGGEST
        tokens += expr_to_tokens(ans_expr, {})
        tokens.append(GyanDSLToken.EOS)

        return {
            "id": f"{module_name}/{split}/{idx:06d}",
            "module": module_name,
            "split": split,
            "question": question,
            "answer": str(answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_closest_composed(module_fn, module_name: str, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Generate closest_composed example.

    For composed questions, the answer may be a variable name or numeric.
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = problem.answer

        # Handle multiple-choice answers like 'b' -> need to map to value
        ans_str = str(answer).strip()
        if len(ans_str) == 1 and ans_str.isalpha():
            # Multiple-choice answer like 'a', 'b', 'c'
            # Try to extract the value from the question
            m = re.search(rf"\({ans_str}\)\s*([^()]+?)(?:\s*\(|$)", question)
            if m:
                ans_expr = sympify(m.group(1).strip())
            else:
                return None
        else:
            ans_expr = sympify_answer(answer)
            if ans_expr is None:
                return None

        tokens = [GyanDSLToken.BOS]
        tokens.append(GyanDSLToken.CLOSEST_TO)
        tokens += expr_to_tokens(ans_expr, {})
        tokens.append(GyanDSLToken.EOS)

        return {
            "id": f"{module_name}/{split}/{idx:06d}",
            "module": module_name,
            "split": split,
            "question": question,
            "answer": str(answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_sort_composed(module_fn, module_name: str, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Generate sort_composed example.

    For composed questions, the sorted list may contain variable names.
    We only tokenize if ALL values are numeric (no symbols).
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        answer_str = str(problem.answer).strip()

        # Parse the sorted answer list
        value_strs = [s.strip() for s in answer_str.split(",") if s.strip()]

        # Try to sympify each value - skip if any is a symbol
        values = []
        for vs in value_strs:
            try:
                v = sympify(vs)
                # Check if it's a pure symbol (variable name) - skip entire example
                if v.is_Symbol:
                    return None
                # Also skip if it contains free symbols (like 2*x)
                if v.free_symbols:
                    return None
                values.append(v)
            except Exception:
                return None

        if not values:
            return None

        # Build tokens: BOS SORT <sorted values> EOS
        tokens = [GyanDSLToken.BOS]
        tokens.append(GyanDSLToken.SORT)
        for v in values:
            tokens += expr_to_tokens(v, {})
        tokens.append(GyanDSLToken.EOS)

        return {
            "id": f"{module_name}/{split}/{idx:06d}",
            "module": module_name,
            "split": split,
            "question": question,
            "answer": answer_str,
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_differentiate(
    module_fn,
    module_name: str,
    split: str,
    idx: int,
) -> Optional[Dict[str, Any]]:
    """Generate differentiation example."""
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = problem.answer

        # Determine differentiation order from question
        m_order = re.search(r"(first|second|third|fourth|fifth|\d+)(?:st|nd|rd|th)?\s+derivative", question, re.IGNORECASE)
        if m_order:
            order_str = m_order.group(1).lower()
            order_map = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
            order = order_map.get(order_str, None)
            if order is None:
                order = int(order_str)
        else:
            order = 1  # Default to first derivative

        expr_str = None
        var_name = None

        # Pattern 1: "... derivative of EXPR wrt VAR"
        m = re.search(r"derivative of\s+(.+?)\s+(?:wrt|with respect to)\s+([a-z])", question, re.IGNORECASE)
        if m:
            expr_str = m.group(1).strip()
            var_name = m.group(2)

        # Pattern 2: "Find the Nth derivative of EXPR wrt VAR." (same as above but with "Find")
        if not expr_str:
            m = re.search(r"Find the .+ derivative of\s+(.+?)\s+(?:wrt|with respect to)\s+([a-z])", question, re.IGNORECASE)
            if m:
                expr_str = m.group(1).strip()
                var_name = m.group(2)

        # Pattern 3: "Differentiate EXPR wrt VAR" or "Differentiate EXPR with respect to VAR"
        if not expr_str:
            m = re.search(r"Differentiate\s+(.+?)\s+(?:wrt|with respect to)\s+([a-z])", question, re.IGNORECASE)
            if m:
                expr_str = m.group(1).strip()
                var_name = m.group(2)

        # Pattern 4: "What is the Nth derivative of EXPR?" (no explicit variable)
        if not expr_str:
            m = re.search(r"derivative of\s+(.+?)(?:\s+wrt|\s+with respect to|\?|$)", question, re.IGNORECASE)
            if m:
                expr_str = m.group(1).strip().rstrip("?")
                # Infer variable from expression
                tmp_expr = sympify(expr_str)
                free_syms = sorted(list(tmp_expr.free_symbols), key=lambda s: s.name)
                var_name = free_syms[0].name if free_syms else "x"

        # Pattern 5: "Differentiate EXPR." (simple, no wrt)
        if not expr_str:
            m = re.search(r"Differentiate\s+(.+?)\.?$", question, re.IGNORECASE)
            if m:
                expr_str = m.group(1).strip()
                tmp_expr = sympify(expr_str)
                free_syms = sorted(list(tmp_expr.free_symbols), key=lambda s: s.name)
                var_name = free_syms[0].name if free_syms else "x"

        if not expr_str or not var_name:
            return None

        expr = sympify(expr_str)
        variable = Symbol(var_name)

        tokens, valid = build_differentiate_tokens(expr, variable, answer, order)

        if not valid or not tokens:
            return None

        return {
            "id": f"{module_name}/{split}/{idx:06d}",
            "module": module_name,
            "split": split,
            "question": question,
            "answer": str(answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_differentiate_composed(
    module_fn,
    module_name: str,
    split: str,
    idx: int,
) -> Optional[Dict[str, Any]]:
    """
    Generate differentiate_composed example.

    For composed questions, the expression may be defined in context.
    We try to parse the final differentiation request and skip verification.
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = problem.answer

        # Determine differentiation order from question
        m_order = re.search(r"(first|second|third|fourth|fifth|\d+)(?:st|nd|rd|th)?\s+derivative", question, re.IGNORECASE)
        if m_order:
            order_str = m_order.group(1).lower()
            order_map = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
            order = order_map.get(order_str, None)
            if order is None:
                order = int(order_str)
        else:
            order = 1

        # Try to extract expression and variable from question
        # Common patterns:
        #   "Find the Nth derivative of EXPR wrt VAR."
        #   "What is the Nth derivative of EXPR wrt VAR?"
        #   "Differentiate EXPR wrt VAR."
        expr_str = None
        var_name = None

        # Pattern 1: "derivative of EXPR wrt VAR"
        m = re.search(r"derivative of\s+(.+?)\s+(?:wrt|with respect to)\s+([a-z])", question, re.IGNORECASE)
        if m:
            expr_str = m.group(1).strip()
            var_name = m.group(2)

        # Pattern 2: "Differentiate EXPR wrt VAR"
        if not expr_str:
            m = re.search(r"Differentiate\s+(.+?)\s+(?:wrt|with respect to)\s+([a-z])", question, re.IGNORECASE)
            if m:
                expr_str = m.group(1).strip()
                var_name = m.group(2)

        # Pattern 3: "Differentiate EXPR." (infer variable)
        if not expr_str:
            m = re.search(r"Differentiate\s+(.+?)\.?$", question, re.IGNORECASE)
            if m:
                expr_str = m.group(1).strip()
                try:
                    temp_expr = sympify(expr_str)
                    free_syms = list(temp_expr.free_symbols)
                    var_name = free_syms[0].name if free_syms else "x"
                except:
                    return None

        if not expr_str or not var_name:
            return None

        expr = sympify(expr_str)
        variable = Symbol(var_name)
        ans_expr = sympify_answer(answer)

        # Build tokens but skip verification (composed context may have undefined symbols)
        var_map = {variable: 0}
        tokens = [GyanDSLToken.BOS]
        tokens += expr_to_tokens(expr, var_map)
        tokens.append(get_real_var_token(0))
        for _ in range(order):
            tokens.append(GyanDSLToken.DIFF)
        tokens += expr_to_tokens(ans_expr, var_map)
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)

        return {
            "id": f"{module_name}/{split}/{idx:06d}",
            "module": module_name,
            "split": split,
            "question": question,
            "answer": str(answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_expand(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """Generate polynomial expand example."""
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = problem.answer

        # Parse "Expand (X)"
        m = re.search(r"Expand\s+(.+?)\.?$", question, re.IGNORECASE)
        if not m:
            return None

        expr_str = m.group(1).strip()
        expr = sympify(expr_str)

        tokens, valid = build_expand_tokens(expr, answer)

        if not valid or not tokens:
            return None

        return {
            "id": f"polynomials__expand/{split}/{idx:06d}",
            "module": "polynomials__expand",
            "split": split,
            "question": question,
            "answer": str(answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_add_polynomials(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Generate polynomials__add example.

    Questions like: "Let y(x) = 1 - x. Let a(b) = 3*b - 6. Determine 2*a(f) + 4*y(f)."
    Answer: "2*f - 8"

    We encode the answer polynomial: BOS ADD_POLY <answer> EOS
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = problem.answer

        ans_expr = sympify_answer(answer)
        if ans_expr is None:
            return None

        # Build var_map from free symbols in the answer
        var_map: Dict[Symbol, int] = {}
        for sym in sorted(ans_expr.free_symbols, key=lambda s: s.name):
            var_map[sym] = len(var_map)

        # Build tokens: BOS ADD_POLY <answer> EOS
        tokens = [GyanDSLToken.BOS]
        tokens.append(GyanDSLToken.ADD_POLY)
        tokens += expr_to_tokens(ans_expr, var_map)
        tokens.append(GyanDSLToken.EOS)

        return {
            "id": f"polynomials__add/{split}/{idx:06d}",
            "module": "polynomials__add",
            "split": split,
            "question": question,
            "answer": str(answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_collect(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """Generate polynomial collect example."""
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = problem.answer

        # Parse "Collect the terms in EXPR."
        m = re.search(r"Collect the terms in\s+(.+?)\.\s*$", question, re.IGNORECASE)
        if not m:
            return None
        expr_str = m.group(1).strip()
        expr = sympify(expr_str)
        ans_expr = sympify_answer(answer)

        tokens, valid = build_collect_tokens(expr, ans_expr)

        if not valid or not tokens:
            return None

        return {
            "id": f"polynomials__collect/{split}/{idx:06d}",
            "module": "polynomials__collect",
            "split": split,
            "question": question,
            "answer": str(answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_simplify_power(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Generate simplify_power example.

    These questions simplify complex power expressions like:
        (v**12*v)/v*v*v**(-13) → 1
        ((v**0*v)/(v**(-1/13)/v))**(-50))**(-23) → v**(31050/13)

    The answer is typically a power (v^rational) or 1.
    We encode: BOS SIMPLIFY_POWER <answer> EOS
    (We skip encoding the full input expression since it's often very complex
    and would explode the token sequence. The question text provides context.)
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = problem.answer

        # Parse "Simplify EXPR assuming VAR is positive."
        m = re.search(r"Simplify\s+(.+?)\s+assuming\s+([A-Za-z])\s+is\s+positive\.\s*$", question, re.IGNORECASE)
        if not m:
            return None
        var_name = m.group(2)
        var = Symbol(var_name)
        ans_expr = sympify_answer(answer)

        # Build simplified tokens: BOS SIMPLIFY_POWER <var> <answer> EOS
        # The answer is typically var^(rational) or 1.
        tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
        tokens.append(GyanDSLToken.SIMPLIFY_POWER)

        var_map = {var: 0}
        try:
            tokens += expr_to_tokens(ans_expr, var_map)
        except ValueError:
            # Answer might be complex (e.g., v^(31050/13)), try to encode it
            # If it's a power, encode base^exp
            if ans_expr.is_Pow:
                base, exp = ans_expr.as_base_exp()
                tokens += expr_to_tokens(base, var_map)
                tokens += expr_to_tokens(exp, {})
                tokens.append(GyanDSLToken.POW)
            elif ans_expr == 1:
                tokens += int_to_tokens(1)
            else:
                return None

        tokens.append(GyanDSLToken.EOS)

        return {
            "id": f"polynomials__simplify_power/{split}/{idx:06d}",
            "module": "polynomials__simplify_power",
            "split": split,
            "question": question,
            "answer": str(answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_coefficient_named(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """Generate coefficient_named examples."""
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = sympify_answer(problem.answer)

        # Match: "Express EXP as CANONICAL and give T." or "Rearrange EXP to ... and give T."
        m = re.match(
            r"^(?:Express|Rearrange)\s+(.+?)\s+(?:as|to|in the form|to the form)\s+(.+?)\s+and give\s+([A-Za-z])\.\s*$",
            question.strip(),
            flags=re.IGNORECASE,
        )
        if not m:
            return None
        expr_str = m.group(1).strip()
        canonical_str = m.group(2).strip()
        target_name = m.group(3)

        expr = sympify(expr_str)
        canonical = sympify(canonical_str)

        # Identify the main variable (assume single variable polynomial).
        vars_expr = sorted(list(expr.free_symbols), key=lambda s: s.name)
        if not vars_expr:
            return None
        var = vars_expr[0]

        target_sym = sympy.Symbol(target_name)

        poly_can = sympy.Poly(canonical, var)
        deg = poly_can.degree()
        target_power = None
        for i, coeff in enumerate(poly_can.all_coeffs()):
            power = deg - i
            if coeff == target_sym:
                target_power = power
                break
        if target_power is None:
            return None

        poly_expr = sympy.Poly(expr, var)
        coeff_val = poly_expr.coeff_monomial(var ** target_power)

        # Verify correctness.
        if simplify(coeff_val - answer) != 0:
            return None

        tokens, valid = build_coefficient_at_power_tokens(expr, var, target_power, coeff_val)
        if not valid or not tokens:
            return None

        return {
            "id": f"polynomials__coefficient_named/{split}/{idx:06d}",
            "module": "polynomials__coefficient_named",
            "split": split,
            "question": question,
            "answer": str(answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Full-structured generators for previously answer-only modules
# ---------------------------------------------------------------------------

def generate_is_factor_structured(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Generate numbers__is_factor with full structure.
    
    Question: "Is 42 a factor of 4532047?" or "Does 115 divide 34155?"
    
    Full structured: BOS <divisor> <number> IS_FACTOR BOOL EOS
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        raw_answer = problem.answer
        answer_bool = bool(raw_answer) if isinstance(raw_answer, bool) else str(raw_answer).lower() == "true"
        
        # Parse: "Is X a factor of Y?" or "Does X divide Y?" or "Is Y a multiple of X?"
        m = re.search(r"Is\s+(\d+)\s+a factor of\s+(\d+)", question, re.IGNORECASE)
        if m:
            divisor, number = int(m.group(1)), int(m.group(2))
        else:
            m = re.search(r"Does\s+(\d+)\s+divide\s+(\d+)", question, re.IGNORECASE)
            if m:
                divisor, number = int(m.group(1)), int(m.group(2))
            else:
                m = re.search(r"Is\s+(\d+)\s+a multiple of\s+(\d+)", question, re.IGNORECASE)
                if m:
                    number, divisor = int(m.group(1)), int(m.group(2))
                else:
                    return None
        
        # Verify
        expected = (number % divisor == 0)
        if expected != answer_bool:
            return None
        
        # Build tokens: BOS <divisor> <number> IS_FACTOR BOOL EOS
        tokens = [GyanDSLToken.BOS]
        tokens += int_to_tokens(divisor)
        tokens += int_to_tokens(number)
        tokens.append(GyanDSLToken.IS_FACTOR)
        tokens.append(GyanDSLToken.BOOL_TRUE if answer_bool else GyanDSLToken.BOOL_FALSE)
        tokens.append(GyanDSLToken.EOS)
        
        return {
            "id": f"numbers__is_factor/{split}/{idx:06d}",
            "module": "numbers__is_factor",
            "split": split,
            "question": question,
            "answer": str(answer_bool),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_place_value_structured(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Generate numbers__place_value with full structure.
    
    Question: "What is the thousands digit of 228221?"
    
    Full structured: BOS <number> <position> PLACE_VALUE <digit> EQ_CMP EOS
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = int(sympify_answer(problem.answer))
        
        # Parse: "What is the X digit of Y?"
        positions = {
            "units": 0, "ones": 0,
            "tens": 1,
            "hundreds": 2,
            "thousands": 3,
            "ten thousands": 4, "ten-thousands": 4,
            "hundred thousands": 5, "hundred-thousands": 5,
            "millions": 6,
            "ten millions": 7, "ten-millions": 7,
            "hundred millions": 8, "hundred-millions": 8,
            "billions": 9,
        }
        
        m = re.search(r"What is the\s+([a-z\s-]+)\s+digit of\s+(-?\d+)", question, re.IGNORECASE)
        if not m:
            return None
        
        pos_str = m.group(1).strip().lower()
        number = int(m.group(2))
        
        if pos_str not in positions:
            return None
        position = positions[pos_str]
        
        # Verify: extract digit at position
        abs_num = abs(number)
        digit = (abs_num // (10 ** position)) % 10
        if digit != answer:
            return None
        
        # Build tokens: BOS <number> <position> PLACE_VALUE <digit> EQ_CMP EOS
        tokens = [GyanDSLToken.BOS]
        tokens += int_to_tokens(number)
        tokens += int_to_tokens(position)
        tokens.append(GyanDSLToken.PLACE_VALUE)
        tokens += int_to_tokens(answer)
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)
        
        return {
            "id": f"numbers__place_value/{split}/{idx:06d}",
            "module": "numbers__place_value",
            "split": split,
            "question": question,
            "answer": str(answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_round_number_structured(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Generate numbers__round_number with full structure.
    
    Question: "Round 20.2 to the nearest ten." or "What is -86729.048 rounded to the nearest ten thousand?"
              "Round -19.8273477 to 2 dps." or "What is 0.034067 rounded to three decimal places?"
    
    Full structured: BOS <number> <precision> ROUND <answer> EQ_CMP EOS
    
    precision > 0 means round to 10^precision (e.g., 2 = hundreds)
    precision < 0 means round to N decimal places (e.g., -2 = 2 decimal places)
    precision = 0 means round to integer
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        answer_raw = problem.answer
        
        # Parse precision - word forms for "nearest X"
        precisions = {
            "integer": 0, "whole number": 0, "one": 0,
            "ten": 1, "tens": 1,
            "hundred": 2, "hundreds": 2,
            "thousand": 3, "thousands": 3, "one thousand": 3,
            "ten thousand": 4, "ten thousands": 4,
            "hundred thousand": 5, "hundred thousands": 5, "one hundred thousand": 5,
            "million": 6, "millions": 6, "one million": 6,
            "ten million": 7, "ten millions": 7,
            "hundred million": 8, "hundred millions": 8,
            "billion": 9, "billions": 9, "one billion": 9,
        }
        
        # Numeric forms like "nearest 1000000"
        numeric_precisions = {
            "10": 1, "100": 2, "1000": 3, "10000": 4, "100000": 5,
            "1000000": 6, "10000000": 7, "100000000": 8, "1000000000": 9,
        }
        
        # Decimal places word forms
        dp_words = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
        }
        
        number_str = None
        precision_power = None
        
        # Pattern 1: "Round X to N dps" or "Round X to N decimal places"
        m = re.search(r"(?:Round|What is)\s+(-?[\d.]+)\s+(?:to|rounded to)\s+(\w+)\s+(?:dps?|decimal places?)", 
                      question, re.IGNORECASE)
        if m:
            number_str = m.group(1)
            dp_str = m.group(2).lower()
            if dp_str in dp_words:
                precision_power = -dp_words[dp_str]  # Negative for decimal places
            else:
                return None
        
        # Pattern 2: "Round X to the nearest Y" or "What is X rounded to the nearest Y?"
        if number_str is None:
            m = re.search(r"(?:Round|What is)\s+(-?[\d.]+)\s+(?:to the nearest|rounded to the nearest)\s+([a-z0-9\s]+)", 
                          question, re.IGNORECASE)
            if m:
                number_str = m.group(1)
                prec_str = m.group(2).strip().lower().rstrip('?.')
                
                if prec_str in precisions:
                    precision_power = precisions[prec_str]
                elif prec_str in numeric_precisions:
                    precision_power = numeric_precisions[prec_str]
                else:
                    return None
        
        if number_str is None or precision_power is None:
            return None
        
        number = float(number_str)
        
        # Compute expected answer
        if precision_power >= 0:
            # Round to nearest 10^precision_power
            scale = 10 ** precision_power
            expected = round(number / scale) * scale
            answer = int(expected)
        else:
            # Round to N decimal places
            dp = -precision_power
            expected = round(number, dp)
            answer = expected
        
        # Verify (allow small tolerance for floats)
        # answer_raw may be Decimal, int, or float
        answer_float = float(str(answer_raw))
        if abs(answer_float - expected) > 0.0001:
            return None
        
        # Build tokens: BOS <number> <precision_power> ROUND <answer> EQ_CMP EOS
        tokens = [GyanDSLToken.BOS]
        # Encode number
        if number == int(number):
            tokens += int_to_tokens(int(number))
        else:
            tokens += expr_to_tokens(sympify(number_str), {})
        tokens += int_to_tokens(precision_power)
        tokens.append(GyanDSLToken.ROUND)
        # Encode answer
        if precision_power >= 0:
            tokens += int_to_tokens(int(answer))
        else:
            tokens += expr_to_tokens(sympify(str(answer_raw)), {})
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)
        
        return {
            "id": f"numbers__round_number/{split}/{idx:06d}",
            "module": "numbers__round_number",
            "split": split,
            "question": question,
            "answer": str(answer_raw),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_polynomial_evaluate_structured(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Generate polynomials__evaluate with full structure.
    
    Question: "Let u(m) = 17*m**2 - 69*m + 5. Give u(3)."
    
    Full structured: BOS <poly_expr> <x_value> EVAL_EXPR <answer> EQ_CMP EOS
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = sympify_answer(problem.answer)
        
        # Parse: "Let f(x) = EXPR. Give/Determine/Calculate f(VALUE)."
        m = re.search(
            r"Let\s+([a-z])\(([a-z])\)\s*=\s*(.+?)\.\s*(?:Give|Determine|Calculate|What is)\s+\1\((-?\d+)\)",
            question, re.IGNORECASE
        )
        if not m:
            return None
        
        func_name = m.group(1)
        var_name = m.group(2)
        expr_str = m.group(3).strip()
        eval_value = int(m.group(4))
        
        var = sympy.Symbol(var_name)
        expr = sympify(expr_str)
        
        # Verify
        computed = expr.subs(var, eval_value)
        if simplify(computed - answer) != 0:
            return None
        
        var_map = {var: 0}
        
        # Build tokens: BOS <poly_expr> <eval_value> EVAL_EXPR <answer> EQ_CMP EOS
        tokens = [GyanDSLToken.BOS]
        tokens += expr_to_tokens(expr, var_map)
        tokens += int_to_tokens(eval_value)
        tokens.append(GyanDSLToken.EVAL_EXPR)
        tokens += int_to_tokens(int(answer))
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)
        
        return {
            "id": f"polynomials__evaluate/{split}/{idx:06d}",
            "module": "polynomials__evaluate",
            "split": split,
            "question": question,
            "answer": str(answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_sequence_next_structured(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Generate algebra__sequence_next_term with full structure.
    
    Question: "What is next in 2449, 4897, 7345?"
    
    Full structured: BOS <n1> <n2> <n3> ... SEQ_NEXT <answer> EQ_CMP EOS
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = int(sympify_answer(problem.answer))
        
        # Parse: "What is next in X, Y, Z?" or "What comes next: X, Y, Z?" or "What is the next term in X, Y, Z?"
        m = re.search(r"(?:What is (?:the )?next (?:term )?in|What comes next:?)\s*(-?[\d,\s-]+)", question, re.IGNORECASE)
        if not m:
            return None
        
        seq_str = m.group(1).strip().rstrip('?')
        # Split by comma
        parts = [p.strip() for p in seq_str.split(',') if p.strip()]
        if len(parts) < 2:
            return None
        
        sequence = [int(p) for p in parts]
        
        # Build tokens: BOS <seq_elements> SEQ_NEXT <answer> EQ_CMP EOS
        tokens = [GyanDSLToken.BOS]
        for val in sequence:
            tokens += int_to_tokens(val)
        tokens.append(GyanDSLToken.SEQ_NEXT)
        tokens += int_to_tokens(answer)
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)
        
        return {
            "id": f"algebra__sequence_next_term/{split}/{idx:06d}",
            "module": "algebra__sequence_next_term",
            "split": split,
            "question": question,
            "answer": str(answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_nearest_root_structured(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Generate arithmetic__nearest_integer_root with full structure.
    
    Question: "What is 265305175 to the power of 1/3, to the nearest integer?"
              "What is the cube root of 55652 to the nearest integer?"
    
    Full structured: BOS <number> <root> NEAREST_ROOT <answer> EQ_CMP EOS
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = int(sympify_answer(problem.answer))
        
        # Parse: "What is X to the power of 1/N, to the nearest integer?"
        m = re.search(r"What is\s+(-?\d+)\s+to the power of\s+1/(\d+)", question, re.IGNORECASE)
        if m:
            number = int(m.group(1))
            root = int(m.group(2))
        else:
            # Parse: "What is the cube/square/Nth root of X to the nearest integer?"
            root_names = {
                "square": 2, "second": 2,
                "cube": 3, "third": 3,
                "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
                "eighth": 8, "ninth": 9, "tenth": 10
            }
            m = re.search(r"What is the\s+(\w+)\s+root of\s+(-?\d+)", question, re.IGNORECASE)
            if m:
                root_name = m.group(1).lower()
                number = int(m.group(2))
                if root_name in root_names:
                    root = root_names[root_name]
                else:
                    return None
            else:
                return None
        
        # Verify
        computed = round(number ** (1.0 / root))
        if computed != answer:
            return None
        
        # Build tokens: BOS <number> <root> NEAREST_ROOT <answer> EQ_CMP EOS
        tokens = [GyanDSLToken.BOS]
        tokens += int_to_tokens(number)
        tokens += int_to_tokens(root)
        tokens.append(GyanDSLToken.NEAREST_ROOT)
        tokens += int_to_tokens(answer)
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)
        
        return {
            "id": f"arithmetic__nearest_integer_root/{split}/{idx:06d}",
            "module": "arithmetic__nearest_integer_root",
            "split": split,
            "question": question,
            "answer": str(answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_base_conversion_structured(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Generate numbers__base_conversion with full structure.
    
    Question: "-1001110110100 (base 2) to base 16" or "What is -5 (base 9) in base 13?"
    
    Full structured: BOS <number_in_base10> <from_base> <to_base> CONVERT_BASE <answer_digits> EOS
    
    Note: We encode the decimal value of the input number, not the raw digits.
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        answer_str = str(problem.answer)
        
        # Parse: "X (base N) to base M" or "What is X (base N) in base M?"
        m = re.search(r"(-?[0-9a-fA-F]+)\s*\(base\s*(\d+)\)\s*(?:to base|in base)\s*(\d+)", question, re.IGNORECASE)
        if not m:
            return None
        
        number_str = m.group(1)
        from_base = int(m.group(2))
        to_base = int(m.group(3))
        
        # Convert input to decimal
        is_negative = number_str.startswith('-')
        abs_str = number_str[1:] if is_negative else number_str
        decimal_value = int(abs_str, from_base)
        if is_negative:
            decimal_value = -decimal_value
        
        # Build tokens: BOS <decimal_value> <from_base> <to_base> TO_BASE <answer_as_list> EOS
        # We encode the answer as the decimal value for simplicity
        # The model learns: given decimal value and target base, produce the answer
        
        # For verification, convert answer back to decimal
        ans_negative = answer_str.startswith('-')
        ans_abs = answer_str[1:] if ans_negative else answer_str
        ans_decimal = int(ans_abs, to_base)
        if ans_negative:
            ans_decimal = -ans_decimal
        
        if ans_decimal != decimal_value:
            return None
        
        tokens = [GyanDSLToken.BOS]
        tokens += int_to_tokens(decimal_value)
        tokens += int_to_tokens(from_base)
        tokens += int_to_tokens(to_base)
        tokens.append(GyanDSLToken.TO_BASE)
        # Encode answer as decimal (model learns base conversion)
        tokens += int_to_tokens(ans_decimal)
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)
        
        return {
            "id": f"numbers__base_conversion/{split}/{idx:06d}",
            "module": "numbers__base_conversion",
            "split": split,
            "question": question,
            "answer": answer_str,
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_add_sub_in_base_structured(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Generate arithmetic__add_or_sub_in_base with full structure.
    
    Question: "In base 4, what is 3 - -2233?" or "In base 8, what is -2705644315 + 3?"
    
    Full structured: BOS <a_decimal> <b_decimal> <base> ADD/SUB <answer_decimal> EQ_CMP EOS
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        answer_str = str(problem.answer)
        
        # Parse: "In base N, what is A + B?" or "In base N, what is A - B?"
        m = re.search(r"In base\s*(\d+),?\s*what is\s*(-?[0-9a-fA-F]+)\s*([+-])\s*(-?[0-9a-fA-F]+)", 
                      question, re.IGNORECASE)
        if not m:
            return None
        
        base = int(m.group(1))
        a_str = m.group(2)
        op = m.group(3)
        b_str = m.group(4)
        
        # Convert to decimal
        def to_decimal(s, b):
            neg = s.startswith('-')
            abs_s = s[1:] if neg else s
            val = int(abs_s, b)
            return -val if neg else val
        
        a_dec = to_decimal(a_str, base)
        b_dec = to_decimal(b_str, base)
        
        # Compute result
        if op == '+':
            result_dec = a_dec + b_dec
            op_token = GyanDSLToken.ADD
        else:
            result_dec = a_dec - b_dec
            op_token = GyanDSLToken.SUB
        
        # Verify against answer
        ans_dec = to_decimal(answer_str, base)
        if ans_dec != result_dec:
            return None
        
        # Build tokens: BOS <a> <b> OP <base> IN_BASE <answer> EQ_CMP EOS
        tokens = [GyanDSLToken.BOS]
        tokens += int_to_tokens(a_dec)
        tokens += int_to_tokens(b_dec)
        tokens.append(op_token)
        tokens += int_to_tokens(base)
        tokens.append(GyanDSLToken.TO_BASE)  # Reuse TO_BASE to indicate "in base"
        tokens += int_to_tokens(result_dec)
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)
        
        return {
            "id": f"arithmetic__add_or_sub_in_base/{split}/{idx:06d}",
            "module": "arithmetic__add_or_sub_in_base",
            "split": split,
            "question": question,
            "answer": answer_str,
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Measurement generators (full structured)
# ---------------------------------------------------------------------------

# Unit conversion factors to a common base
UNIT_CONVERSIONS = {
    # Length (base: meters)
    "meter": 1, "meters": 1, "metre": 1, "metres": 1, "m": 1,
    "centimeter": 0.01, "centimeters": 0.01, "centimetre": 0.01, "centimetres": 0.01, "cm": 0.01,
    "millimeter": 0.001, "millimeters": 0.001, "millimetre": 0.001, "millimetres": 0.001, "mm": 0.001,
    "micrometer": 1e-6, "micrometers": 1e-6, "micrometre": 1e-6, "micrometres": 1e-6,
    "nanometer": 1e-9, "nanometers": 1e-9, "nanometre": 1e-9, "nanometres": 1e-9,
    "kilometer": 1000, "kilometers": 1000, "kilometre": 1000, "kilometres": 1000, "km": 1000,
    # Volume (base: liters)
    "liter": 1, "liters": 1, "litre": 1, "litres": 1, "l": 1,
    "milliliter": 0.001, "milliliters": 0.001, "millilitre": 0.001, "millilitres": 0.001, "ml": 0.001,
    "microliter": 1e-6, "microliters": 1e-6, "microlitre": 1e-6, "microlitres": 1e-6,
    # Mass (base: grams)
    "gram": 1, "grams": 1, "g": 1,
    "kilogram": 1000, "kilograms": 1000, "kg": 1000,
    "milligram": 0.001, "milligrams": 0.001, "mg": 0.001,
    "microgram": 1e-6, "micrograms": 1e-6,
    "tonne": 1000000, "tonnes": 1000000,
    # Time (base: seconds)
    "second": 1, "seconds": 1, "s": 1,
    "millisecond": 0.001, "milliseconds": 0.001, "ms": 0.001,
    "microsecond": 1e-6, "microseconds": 1e-6,
    "nanosecond": 1e-9, "nanoseconds": 1e-9,
    "minute": 60, "minutes": 60,
    "hour": 3600, "hours": 3600,
    "day": 86400, "days": 86400,
    "week": 604800, "weeks": 604800,
    "month": 2628000, "months": 2628000,  # avg month
    "year": 31536000, "years": 31536000,
    "decade": 315360000, "decades": 315360000,
    "century": 3153600000, "centuries": 3153600000,
    "millennium": 31536000000, "millennia": 31536000000,
}


def generate_measurement_conversion_structured(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Generate measurement__conversion with full structure.
    
    Questions like:
    - "How many millilitres are there in 89/2 of a litre?"
    - "Convert 509.2872 months to centuries."
    - "What is 15/4 of a meter in centimeters?"
    
    Full structured: BOS <value> <from_unit_factor> <to_unit_factor> CONVERT <answer> EQ_CMP EOS
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = sympify_answer(problem.answer)
        
        # Pattern 1: "How many X are there in Y of a Z?"
        m = re.search(r"How many\s+(\w+)\s+are there in\s+(.+?)\s+of a\s+(\w+)", question, re.IGNORECASE)
        if m:
            to_unit = m.group(1).lower()
            value_str = m.group(2).strip()
            from_unit = m.group(3).lower()
        else:
            # Pattern 2: "Convert X Y to Z"
            m = re.search(r"Convert\s+([\d./]+)\s*(\w+)\s+to\s+(\w+)", question, re.IGNORECASE)
            if m:
                value_str = m.group(1)
                from_unit = m.group(2).lower()
                to_unit = m.group(3).lower()
            else:
                # Pattern 3: "What is X of a Y in Z?"
                m = re.search(r"What is\s+(.+?)\s+of a\s+(\w+)\s+in\s+(\w+)", question, re.IGNORECASE)
                if m:
                    value_str = m.group(1).strip()
                    from_unit = m.group(2).lower()
                    to_unit = m.group(3).lower()
                else:
                    # Pattern 4: "What is X<unit> in Y?" e.g. "What is 8195.933kg in grams?"
                    m = re.search(r"What is\s+([\d.]+)\s*(\w+)\s+in\s+(\w+)", question, re.IGNORECASE)
                    if m:
                        value_str = m.group(1)
                        from_unit = m.group(2).lower()
                        to_unit = m.group(3).lower()
                    else:
                        # Pattern 5: "How many X are there in Y Z?" e.g. "How many months are there in 253.3164 millennia?"
                        # Also handles "How many centimeters are there in 9.260309mm?" (no space before unit)
                        m = re.search(r"How many\s+(\w+)\s+are there in\s+([\d.]+)\s*(\w+)", question, re.IGNORECASE)
                        if m:
                            to_unit = m.group(1).lower()
                            value_str = m.group(2)
                            from_unit = m.group(3).lower()
                        else:
                            return None
        
        # Parse value (handle fractions like "89/2", "three tenths", etc.)
        value_str = value_str.replace("thirty-four fifths", "34/5").replace("three tenths", "3/10")
        value_str = value_str.replace("one", "1").replace("two", "2").replace("three", "3")
        value_str = value_str.replace("four", "4").replace("five", "5")
        
        try:
            value = float(sympify(value_str))
        except:
            return None
        
        # Get conversion factors
        if from_unit not in UNIT_CONVERSIONS or to_unit not in UNIT_CONVERSIONS:
            return None
        
        from_factor = UNIT_CONVERSIONS[from_unit]
        to_factor = UNIT_CONVERSIONS[to_unit]
        
        # Compute expected answer
        base_value = value * from_factor
        expected = base_value / to_factor
        
        # Verify (allow some floating point tolerance)
        if abs(float(answer) - expected) > 0.0001 * max(abs(expected), 1):
            return None
        
        # Build tokens: BOS <value> <from_factor> MUL <to_factor> DIV CONVERT <answer> EQ_CMP EOS
        tokens = [GyanDSLToken.BOS]
        tokens += expr_to_tokens(sympify(value_str), {})
        # We encode the conversion as: value * from_factor / to_factor = answer
        # Simplified: just encode value, conversion_ratio, answer
        ratio = Rational(from_factor / to_factor).limit_denominator(10000)
        tokens += rational_to_tokens(ratio)
        tokens.append(GyanDSLToken.MUL)
        tokens.append(GyanDSLToken.EVAL_EXPR)  # CONVERT marker
        tokens += expr_to_tokens(answer, {})
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)
        
        return {
            "id": f"measurement__conversion/{split}/{idx:06d}",
            "module": "measurement__conversion",
            "split": split,
            "question": question,
            "answer": str(problem.answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_measurement_time_structured(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Generate measurement__time with full structure.
    
    Questions like:
    - "How many minutes are there between 6:49 PM and 8:36 PM?"
    - "What is 38 minutes after 7:39 AM?"
    - "What is 387 minutes before 10:39 PM?"
    
    Full structured: BOS <time1_minutes> <time2_minutes_or_delta> TIME_OP <answer_minutes> EQ_CMP EOS
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        answer_str = str(problem.answer)
        
        def parse_time_to_minutes(time_str: str) -> int:
            """Convert "7:39 AM" to minutes from midnight."""
            m = re.match(r"(\d+):(\d+)\s*(AM|PM)", time_str.strip(), re.IGNORECASE)
            if not m:
                return None
            hour = int(m.group(1))
            minute = int(m.group(2))
            period = m.group(3).upper()
            
            if period == "PM" and hour != 12:
                hour += 12
            elif period == "AM" and hour == 12:
                hour = 0
            
            return hour * 60 + minute
        
        # Pattern 1: "How many minutes are there between X and Y?"
        m = re.search(r"How many minutes are there between\s+(\d+:\d+\s*[AP]M)\s+and\s+(\d+:\d+\s*[AP]M)", 
                      question, re.IGNORECASE)
        if m:
            time1 = parse_time_to_minutes(m.group(1))
            time2 = parse_time_to_minutes(m.group(2))
            if time1 is None or time2 is None:
                return None
            
            # Answer is the difference
            expected = time2 - time1
            if expected < 0:
                expected += 24 * 60  # Next day
            
            answer_int = int(answer_str)
            if expected != answer_int:
                return None
            
            # Build tokens: BOS <time1> <time2> SUB TIME <answer> EQ_CMP EOS
            tokens = [GyanDSLToken.BOS]
            tokens += int_to_tokens(time1)
            tokens += int_to_tokens(time2)
            tokens.append(GyanDSLToken.SUB)
            tokens.append(GyanDSLToken.EVAL_EXPR)  # TIME marker
            tokens += int_to_tokens(answer_int)
            tokens.append(GyanDSLToken.EQ_CMP)
            tokens.append(GyanDSLToken.EOS)
            
            return {
                "id": f"measurement__time/{split}/{idx:06d}",
                "module": "measurement__time",
                "split": split,
                "question": question,
                "answer": answer_str,
                "token_ids": [t.value for t in tokens],
                "token_names": [t.name for t in tokens],
            }
        
        # Pattern 2: "What is X minutes after/before Y?"
        m = re.search(r"What is\s+(\d+)\s+minutes\s+(after|before)\s+(\d+:\d+\s*[AP]M)", 
                      question, re.IGNORECASE)
        if m:
            delta = int(m.group(1))
            direction = m.group(2).lower()
            base_time = parse_time_to_minutes(m.group(3))
            if base_time is None:
                return None
            
            if direction == "after":
                result_minutes = (base_time + delta) % (24 * 60)
            else:
                result_minutes = (base_time - delta) % (24 * 60)
            
            # Parse answer time
            answer_time = parse_time_to_minutes(answer_str)
            if answer_time is None or answer_time != result_minutes:
                return None
            
            # Build tokens: BOS <base_time> <delta> ADD/SUB TIME <result> EQ_CMP EOS
            tokens = [GyanDSLToken.BOS]
            tokens += int_to_tokens(base_time)
            tokens += int_to_tokens(delta)
            tokens.append(GyanDSLToken.ADD if direction == "after" else GyanDSLToken.SUB)
            tokens.append(GyanDSLToken.EVAL_EXPR)  # TIME marker
            tokens += int_to_tokens(result_minutes)
            tokens.append(GyanDSLToken.EQ_CMP)
            tokens.append(GyanDSLToken.EOS)
            
            return {
                "id": f"measurement__time/{split}/{idx:06d}",
                "module": "measurement__time",
                "split": split,
                "question": question,
                "answer": answer_str,
                "token_ids": [t.value for t in tokens],
                "token_names": [t.name for t in tokens],
            }
        
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Probability generators (full structured)
# ---------------------------------------------------------------------------

def generate_probability_sequence_structured(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Generate probability__swr_p_sequence with full structure.
    
    Questions like:
    - "Calculate prob of sequence fjbb when four letters picked without replacement from bfjbjjbbjcbjbb."
    - "Two letters picked without replacement from {o: 13, s: 5}. What is prob of sequence sos?"
    
    Full structured: BOS <total> <counts...> <sequence_indices> PROBABILITY <numerator> <denominator> DIV EQ_CMP EOS
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = sympify_answer(problem.answer)
        
        # Extract the sequence to pick
        m = re.search(r"(?:prob of sequence|sequence)\s+([a-z]+)", question, re.IGNORECASE)
        if not m:
            return None
        sequence = m.group(1).lower()
        
        # Extract the source - either a string or a dict
        # Pattern 1: "from {a: 2, b: 3}"
        m_dict = re.search(r"from\s*\{([^}]+)\}", question)
        # Pattern 2: "from abcabc" (may end with ?, ., or whitespace)
        m_str = re.search(r"from\s+([a-z]+)(?:[?.\s]|$)", question, re.IGNORECASE)
        
        if m_dict:
            # Parse dict format
            counts = {}
            for part in m_dict.group(1).split(','):
                k, v = part.strip().split(':')
                counts[k.strip()] = int(v.strip())
        elif m_str:
            # Count letters in string
            source = m_str.group(1).lower()
            counts = {}
            for c in source:
                counts[c] = counts.get(c, 0) + 1
        else:
            return None
        
        total = sum(counts.values())
        
        # Compute probability of the sequence
        # P(seq) = product of (count_i / remaining) for each letter in sequence
        numerator = 1
        denominator = 1
        remaining = total
        temp_counts = counts.copy()
        
        for letter in sequence:
            if letter not in temp_counts or temp_counts[letter] == 0:
                return None  # Impossible sequence
            numerator *= temp_counts[letter]
            denominator *= remaining
            temp_counts[letter] -= 1
            remaining -= 1
        
        # Simplify fraction
        from math import gcd
        g = gcd(numerator, denominator)
        num_simplified = numerator // g
        den_simplified = denominator // g
        
        # Verify against answer
        expected = Rational(num_simplified, den_simplified)
        if simplify(answer - expected) != 0:
            return None
        
        # Build tokens: BOS <total> <seq_len> PROBABILITY <num> <den> DIV EQ_CMP EOS
        tokens = [GyanDSLToken.BOS]
        tokens += int_to_tokens(total)
        tokens += int_to_tokens(len(sequence))
        tokens.append(GyanDSLToken.PROBABILITY)
        tokens += int_to_tokens(num_simplified)
        tokens += int_to_tokens(den_simplified)
        tokens.append(GyanDSLToken.DIV)
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)
        
        return {
            "id": f"probability__swr_p_sequence/{split}/{idx:06d}",
            "module": "probability__swr_p_sequence",
            "split": split,
            "question": question,
            "answer": str(problem.answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_probability_level_set_structured(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Generate probability__swr_p_level_set with full structure.
    
    Questions like:
    - "What is prob of picking 2 t when two letters picked without replacement from iititttittti?"
    - "What is prob of picking 1 f and 3 d when four letters picked without replacement from {h: 3, f: 1, ...}?"
    
    Full structured: BOS <total> <picks> PROBABILITY <num> <den> DIV EQ_CMP EOS
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = sympify_answer(problem.answer)
        
        # Extract number of picks
        m = re.search(r"(\w+)\s+letters?\s+picked", question, re.IGNORECASE)
        if not m:
            return None
        num_words = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
        picks_str = m.group(1).lower()
        if picks_str in num_words:
            num_picks = num_words[picks_str]
        elif picks_str.isdigit():
            num_picks = int(picks_str)
        else:
            return None
        
        # Extract the source
        m_dict = re.search(r"from\s*\{([^}]+)\}", question)
        m_str = re.search(r"from\s+([a-z]+)(?:[?.\s]|$)", question, re.IGNORECASE)
        
        if m_dict:
            counts = {}
            for part in m_dict.group(1).split(','):
                k, v = part.strip().split(':')
                counts[k.strip()] = int(v.strip())
        elif m_str:
            source = m_str.group(1).lower()
            counts = {}
            for c in source:
                counts[c] = counts.get(c, 0) + 1
        else:
            return None
        
        total = sum(counts.values())
        
        # The answer is already computed by DeepMind - we trust it
        # Just encode the structure
        
        # Build tokens: BOS <total> <picks> PROBABILITY <answer_fraction> EQ_CMP EOS
        tokens = [GyanDSLToken.BOS]
        tokens += int_to_tokens(total)
        tokens += int_to_tokens(num_picks)
        tokens.append(GyanDSLToken.PROBABILITY)
        tokens += expr_to_tokens(answer, {})
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)
        
        return {
            "id": f"probability__swr_p_level_set/{split}/{idx:06d}",
            "module": "probability__swr_p_level_set",
            "split": split,
            "question": question,
            "answer": str(problem.answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Additional structured generators
# ---------------------------------------------------------------------------

def generate_polynomials_compose_structured(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Generate polynomials__compose with full structure.
    
    Questions define nested functions and ask for composition like:
    "Let f(x) = ..., g(x) = ... Give f(g(t))."
    
    Answer is a polynomial expression like "30*t" or "-48*j".
    
    Full structured: BOS COMPOSE_FN <answer_polynomial> EOS
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = problem.answer
        
        ans_expr = sympify_answer(answer)
        if ans_expr is None:
            return None
        
        # Build var_map from free symbols in the answer
        var_map: Dict[Symbol, int] = {}
        for sym in sorted(ans_expr.free_symbols, key=lambda s: s.name):
            var_map[sym] = len(var_map)
        
        # Build tokens: BOS COMPOSE <answer_polynomial> EOS
        tokens = [GyanDSLToken.BOS]
        tokens.append(GyanDSLToken.COMPOSE)
        tokens += expr_to_tokens(ans_expr, var_map)
        tokens.append(GyanDSLToken.EOS)
        
        return {
            "id": f"polynomials__compose/{split}/{idx:06d}",
            "module": "polynomials__compose",
            "split": split,
            "question": question,
            "answer": str(answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_polynomial_evaluate_composed_structured(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Generate polynomials__evaluate_composed with semi-structured format.
    
    Questions like: "Let f(x) = ... Let g be ... Determine f(g)."
    Answer is a simple integer.
    
    We extract the evaluation point from the question and encode:
    BOS <eval_point> EVAL_EXPR <answer> EQ_CMP EOS
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        answer = int(sympify_answer(problem.answer))
        
        # Try to extract the evaluation point from patterns like:
        # "Determine f(3)." or "Give f(-4)." or "What is f(2)?"
        m = re.search(r"(?:Determine|Give|What is|Calculate)\s+\w+\((-?\d+)\)", question)
        if m:
            eval_point = int(m.group(1))
        else:
            # Fallback: use 0 as placeholder
            eval_point = 0
        
        # Build tokens: BOS <eval_point> EVAL_EXPR <answer> EQ_CMP EOS
        tokens = [GyanDSLToken.BOS]
        tokens += int_to_tokens(eval_point)
        tokens.append(GyanDSLToken.EVAL_EXPR)
        tokens += int_to_tokens(answer)
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)
        
        return {
            "id": f"polynomials__evaluate_composed/{split}/{idx:06d}",
            "module": "polynomials__evaluate_composed",
            "split": split,
            "question": question,
            "answer": str(answer),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


def generate_is_factor_composed_structured(module_fn, split: str, idx: int) -> Optional[Dict[str, Any]]:
    """
    Generate numbers__is_factor_composed with semi-structured format.
    
    Questions like: "Let a be the remainder when 142525 is divided by 70378. Does 3 divide a?"
    
    We extract the divisor being tested and encode:
    BOS <divisor> IS_FACTOR BOOL EOS
    """
    try:
        problem = module_fn()
        question = str(problem.question)
        raw_answer = problem.answer
        answer_bool = bool(raw_answer) if isinstance(raw_answer, bool) else str(raw_answer).lower() == "true"
        
        # Extract the divisor from patterns like:
        # "Does 3 divide X?" or "Is 10 a factor of X?" or "Is X a multiple of 5?"
        m = re.search(r"Does\s+(\d+)\s+divide", question, re.IGNORECASE)
        if m:
            divisor = int(m.group(1))
        else:
            m = re.search(r"Is\s+(\d+)\s+a factor of", question, re.IGNORECASE)
            if m:
                divisor = int(m.group(1))
            else:
                m = re.search(r"Is\s+\w+\s+a multiple of\s+(\d+)", question, re.IGNORECASE)
                if m:
                    divisor = int(m.group(1))
                else:
                    return None
        
        # Build tokens: BOS <divisor> IS_FACTOR BOOL EOS
        tokens = [GyanDSLToken.BOS]
        tokens += int_to_tokens(divisor)
        tokens.append(GyanDSLToken.IS_FACTOR)
        tokens.append(GyanDSLToken.BOOL_TRUE if answer_bool else GyanDSLToken.BOOL_FALSE)
        tokens.append(GyanDSLToken.EOS)
        
        return {
            "id": f"numbers__is_factor_composed/{split}/{idx:06d}",
            "module": "numbers__is_factor_composed",
            "split": split,
            "question": question,
            "answer": str(answer_bool),
            "token_ids": [t.value for t in tokens],
            "token_names": [t.name for t in tokens],
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Batch worker functions
# ---------------------------------------------------------------------------

def worker_batch(args: Tuple) -> List[Dict[str, Any]]:
    """Generic worker for generating batches."""
    generator_fn, module_fn, module_name, split, start_idx, batch_size = args
    results = []
    attempts = 0
    max_attempts = batch_size * 5

    while len(results) < batch_size and attempts < max_attempts:
        attempts += 1
        ex = generator_fn(module_fn, split, start_idx + len(results))
        if ex is not None:
            results.append(ex)

    return results


def worker_batch_named(args: Tuple) -> List[Dict[str, Any]]:
    """Worker for generators that need module_name."""
    generator_fn, module_fn, module_name, split, start_idx, batch_size = args
    results = []
    attempts = 0
    max_attempts = batch_size * 5

    while len(results) < batch_size and attempts < max_attempts:
        attempts += 1
        ex = generator_fn(module_fn, module_name, split, start_idx + len(results))
        if ex is not None:
            results.append(ex)

    return results


# ---------------------------------------------------------------------------
# Main generation orchestrator
# ---------------------------------------------------------------------------

def generate_module_split(
    generator_fn: Callable,
    module_fn: Callable,
    module_name: str,
    split: str,
    num_examples: int,
    output_path: str,
    num_workers: int = None,
    needs_module_name: bool = False,
) -> int:
    """Generate examples for one module/split and append to JSONL."""
    if num_examples <= 0:
        return 0

    if num_workers is None:
        num_workers = min(cpu_count(), 16)

    batch_size = max(50, num_examples // (num_workers * 10))
    num_batches = (num_examples + batch_size - 1) // batch_size

    examples = []
    pbar = tqdm(total=num_examples, desc=f"{module_name}:{split}", leave=False)

    work_items = []
    for i in range(num_batches):
        start_idx = i * batch_size
        this_batch = min(batch_size, num_examples - start_idx)
        if this_batch > 0:
            work_items.append((generator_fn, module_fn, module_name, split, start_idx, this_batch))

    worker_fn = worker_batch_named if needs_module_name else worker_batch

    with Pool(processes=num_workers) as pool:
        for batch_results in pool.imap_unordered(worker_fn, work_items):
            examples.extend(batch_results)
            pbar.update(len(batch_results))
            if len(examples) >= num_examples:
                break

    pbar.close()

    # Trim and re-index
    examples = examples[:num_examples]
    for i, ex in enumerate(examples):
        ex["id"] = f"{module_name}/{split}/{i:06d}"

    # Append to JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mode = "a" if os.path.exists(output_path) else "w"
    with open(output_path, mode) as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    return len(examples)


def main():
    parser = argparse.ArgumentParser(description="Generate full DeepMind math DSL data")
    parser.add_argument("--output_dir", type=str, default="data/full_math_small")
    parser.add_argument("--train_per_module", type=int, default=1000)
    parser.add_argument("--test_per_module", type=int, default=100)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation (applies to random and numpy.random).",
    )
    parser.add_argument(
        "--modules",
        type=str,
        nargs="+",
        default=None,
        help="List of module names to generate. If not provided, generates all modules.",
    )
    args = parser.parse_args()

    # Set global seeds for reproducibility, if requested.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    print(f"Gyan DSL vocab size: {get_vocab_size()}")
    print(f"Output: {args.output_dir}")
    print(f"Train per module: {args.train_per_module}")
    print(f"Test per module: {args.test_per_module}")
    print(f"Seed: {args.seed}")
    print(f"Modules filter: {args.modules if args.modules else 'all'}")
    print()

    # Get entropy function
    entropy_fn = lambda r: r

    # Define all modules to generate
    modules_config = [
        # (module_name, train_fn_getter, test_fn_getter, generator_fn, needs_module_name)

        # Algebra - using train(entropy_fn) for BOTH train and test to get i.i.d. distributions
        ("algebra__linear_1d",
         lambda: algebra.train(entropy_fn)["linear_1d"],
         lambda: algebra.train(entropy_fn)["linear_1d"],  # Use train() for test too (i.i.d.)
         generate_linear_1d, False),

        ("algebra__linear_1d_composed",
         lambda: algebra.train(entropy_fn)["linear_1d_composed"],
         lambda: algebra.train(entropy_fn)["linear_1d_composed"],  # Use train() for test too (i.i.d.)
         generate_linear_1d_composed, False),

        ("algebra__linear_2d",
         lambda: algebra.train(entropy_fn)["linear_2d"],
         lambda: algebra.train(entropy_fn)["linear_2d"],  # Use train() for test too (i.i.d.)
         generate_linear_2d, False),

        ("algebra__linear_2d_composed",
         lambda: algebra.train(entropy_fn)["linear_2d_composed"],
         lambda: algebra.train(entropy_fn)["linear_2d_composed"],  # Use train() for test too (i.i.d.)
         generate_linear_2d_composed, False),

        ("algebra__polynomial_roots",
         lambda: algebra.train(entropy_fn)["polynomial_roots"],
         lambda: algebra.test()["polynomial_roots"],
         generate_polynomial_roots, False),

        ("algebra__polynomial_roots_composed",
         lambda: algebra.train(entropy_fn)["polynomial_roots_composed"],
         lambda: algebra.test()["polynomial_roots_composed"],
         generate_polynomial_roots_composed, False),

        # Algebra sequences - now full structured
        ("algebra__sequence_next_term",
         lambda: algebra.train(entropy_fn)["sequence_next_term"],
         lambda: algebra.test()["sequence_next_term"],
         generate_sequence_next_structured,
         False),

        ("algebra__sequence_nth_term",
         lambda: algebra.train(entropy_fn)["sequence_nth_term"],
         lambda: algebra.test()["sequence_nth_term"],
         unary_numeric_generator,
         True),

        # Arithmetic
        ("arithmetic__add_or_sub",
         lambda: arithmetic.train(entropy_fn)["add_or_sub"],
         lambda: arithmetic.test()["add_or_sub"],
         generate_arithmetic_eval, True),

        ("arithmetic__mul",
         lambda: arithmetic.train(entropy_fn)["mul"],
         lambda: arithmetic.test()["mul"],
         generate_arithmetic_eval, True),

        ("arithmetic__div",
         lambda: arithmetic.train(entropy_fn)["div"],
         lambda: arithmetic.test()["div"],
         generate_arithmetic_eval, True),

        ("arithmetic__mixed",
         lambda: arithmetic.train(entropy_fn)["mixed"],
         lambda: arithmetic.test()["mixed"],
         generate_arithmetic_eval, True),

        # Additional arithmetic expression chains
        ("arithmetic__add_sub_multiple",
         lambda: arithmetic.train(entropy_fn)["add_sub_multiple"],
         lambda: arithmetic.test()["add_sub_multiple"],
         generate_arithmetic_eval, True),

        ("arithmetic__mul_div_multiple",
         lambda: arithmetic.train(entropy_fn)["mul_div_multiple"],
         lambda: arithmetic.test()["mul_div_multiple"],
         generate_arithmetic_eval, True),

        ("arithmetic__add_or_sub_in_base",
         lambda: arithmetic.train(entropy_fn)["add_or_sub_in_base"],
         lambda: arithmetic.test()["add_or_sub_in_base"],
         generate_add_sub_in_base_structured,
         False),

        ("arithmetic__nearest_integer_root",
         lambda: arithmetic.train(entropy_fn)["nearest_integer_root"],
         lambda: arithmetic.test()["nearest_integer_root"],
         generate_nearest_root_structured,
         False),

        ("arithmetic__simplify_surd",
         lambda: arithmetic.train(entropy_fn)["simplify_surd"],
         lambda: arithmetic.test()["simplify_surd"],
         unary_numeric_generator,
         True),

        # Numbers
        ("numbers__gcd",
         lambda: numbers.train(entropy_fn)["gcd"],
         lambda: numbers.test()["gcd"],
         generate_gcd_lcm, True),

        ("numbers__gcd_composed",
         lambda: numbers.train(entropy_fn)["gcd_composed"],
         lambda: numbers.test()["gcd_composed"],
         unary_numeric_generator,
         True),

        ("numbers__lcm",
         lambda: numbers.train(entropy_fn)["lcm"],
         lambda: numbers.test()["lcm"],
         generate_gcd_lcm, True),

        ("numbers__lcm_composed",
         lambda: numbers.train(entropy_fn)["lcm_composed"],
         lambda: numbers.test()["lcm_composed"],
         unary_numeric_generator,
         True),

        ("numbers__is_prime",
            lambda: numbers.train(entropy_fn)["is_prime"],
            lambda: numbers.test()["is_prime"],
            generate_is_prime, False),

        ("numbers__is_prime_composed",
         lambda: numbers.train(entropy_fn)["is_prime_composed"],
         lambda: numbers.test()["is_prime_composed"],
         unary_bool_generator,
         True),

        ("numbers__div_remainder",
         lambda: numbers.train(entropy_fn)["div_remainder"],
         lambda: numbers.test()["div_remainder"],
         generate_div_remainder, False),

        ("numbers__div_remainder_composed",
         lambda: numbers.train(entropy_fn)["div_remainder_composed"],
         lambda: numbers.test()["div_remainder_composed"],
         unary_numeric_generator,
         True),

        ("numbers__is_factor",
         lambda: numbers.train(entropy_fn)["is_factor"],
         lambda: numbers.test()["is_factor"],
         generate_is_factor_structured,
         False),

        ("numbers__is_factor_composed",
         lambda: numbers.train(entropy_fn)["is_factor_composed"],
         lambda: numbers.test()["is_factor_composed"],
         generate_is_factor_composed_structured,
         False),

        ("numbers__round_number",
         lambda: numbers.train(entropy_fn)["round_number"],
         lambda: numbers.test()["round_number"],
         generate_round_number_structured,
         False),

        ("numbers__round_number_composed",
         lambda: numbers.train(entropy_fn)["round_number_composed"],
         lambda: numbers.test()["round_number_composed"],
         unary_numeric_generator,
         True),

        ("numbers__place_value",
         lambda: numbers.train(entropy_fn)["place_value"],
         lambda: numbers.test()["place_value"],
         generate_place_value_structured,
         False),

        ("numbers__place_value_composed",
         lambda: numbers.train(entropy_fn)["place_value_composed"],
         lambda: numbers.test()["place_value_composed"],
         unary_numeric_generator,
         True),

        ("numbers__base_conversion",
         lambda: numbers.train(entropy_fn)["base_conversion"],
         lambda: numbers.test()["base_conversion"],
         generate_base_conversion_structured,
         False),

        ("numbers__list_prime_factors",
         lambda: numbers.train(entropy_fn)["list_prime_factors"],
         lambda: numbers.test()["list_prime_factors"],
         generate_list_prime_factors,
         True),

        ("numbers__list_prime_factors_composed",
         lambda: numbers.train(entropy_fn)["list_prime_factors_composed"],
         lambda: numbers.test()["list_prime_factors_composed"],
         generate_list_prime_factors_composed,
         True),

        # Measurement - now full structured
        ("measurement__conversion",
         lambda: measurement.train(entropy_fn)["conversion"],
         lambda: measurement.test()["conversion"],
         generate_measurement_conversion_structured,
         False),

        ("measurement__time",
         lambda: measurement.train(entropy_fn)["time"],
         lambda: measurement.test()["time"],
         generate_measurement_time_structured,
         False),

        # Comparison
        ("comparison__pair",
         lambda: comparison.train(entropy_fn)["pair"],
         lambda: comparison.test()["pair"],
         generate_comparison, True),

        ("comparison__pair_composed",
         lambda: comparison.train(entropy_fn)["pair_composed"],
         lambda: comparison.test()["pair_composed"],
         generate_comparison_composed, True),

        ("comparison__kth_biggest",
         lambda: comparison.train(entropy_fn)["kth_biggest"],
         lambda: comparison.test()["kth_biggest"],
         generate_kth_biggest, True),

        ("comparison__kth_biggest_composed",
         lambda: comparison.train(entropy_fn)["kth_biggest_composed"],
         lambda: comparison.test()["kth_biggest_composed"],
         generate_kth_biggest_composed, True),

        ("comparison__closest",
         lambda: comparison.train(entropy_fn)["closest"],
         lambda: comparison.test()["closest"],
         generate_closest, True),

        ("comparison__closest_composed",
         lambda: comparison.train(entropy_fn)["closest_composed"],
         lambda: comparison.test()["closest_composed"],
         generate_closest_composed, True),

        ("comparison__sort",
         lambda: comparison.train(entropy_fn)["sort"],
         lambda: comparison.test()["sort"],
         generate_sort, True),

        ("comparison__sort_composed",
         lambda: comparison.train(entropy_fn)["sort_composed"],
         lambda: comparison.test()["sort_composed"],
         generate_sort_composed, True),

        # Calculus
        ("calculus__differentiate",
         lambda: calculus.train(entropy_fn)["differentiate"],
         lambda: calculus.test()["differentiate"],
         generate_differentiate, True),

        ("calculus__differentiate_composed",
         lambda: calculus.train(entropy_fn)["differentiate_composed"],
         lambda: calculus.test()["differentiate_composed"],
         generate_differentiate_composed, True),

        # Polynomials
        ("polynomials__expand",
         lambda: polynomials.train(entropy_fn)["expand"],
         lambda: polynomials.test()["expand"],
         generate_expand, False),

        ("polynomials__evaluate",
         lambda: polynomials.train(entropy_fn)["evaluate"],
         lambda: polynomials.test()["evaluate"],
         generate_polynomial_evaluate_structured,
         False),

        ("polynomials__evaluate_composed",
         lambda: polynomials.train(entropy_fn)["evaluate_composed"],
         lambda: polynomials.test()["evaluate_composed"],
         generate_polynomial_evaluate_composed_structured,
         False),

        ("polynomials__add",
         lambda: polynomials.train(entropy_fn)["add"],
         lambda: polynomials.test()["add"],
         generate_add_polynomials,
         False),

        ("polynomials__collect",
         lambda: polynomials.train(entropy_fn)["collect"],
         lambda: polynomials.test()["collect"],
         generate_collect,
         False),

        ("polynomials__simplify_power",
         lambda: polynomials.train(entropy_fn)["simplify_power"],
         lambda: polynomials.test()["simplify_power"],
         generate_simplify_power,
         False),

        ("polynomials__coefficient_named",
         lambda: polynomials.train(entropy_fn)["coefficient_named"],
         lambda: polynomials.test()["coefficient_named"],
         generate_coefficient_named,
         False),

        ("polynomials__compose",
         lambda: polynomials.train(entropy_fn)["compose"],
         lambda: polynomials.test()["compose"],
         generate_polynomials_compose_structured,
         False),

        # Polynomials: we now include expand, evaluate, add, collect, simplify_power,
        # coefficient_named, and compose. Additional tasks can be added later if needed.

        # Probability - now full structured
        ("probability__swr_p_sequence",
         lambda: probability.train(entropy_fn)["swr_p_sequence"],
         lambda: probability.test()["swr_p_sequence"],
         generate_probability_sequence_structured,
         False),

        ("probability__swr_p_level_set",
         lambda: probability.train(entropy_fn)["swr_p_level_set"],
         lambda: probability.test()["swr_p_level_set"],
         generate_probability_level_set_structured,
         False),
    ]

    total_train = 0
    total_test_id = 0
    total_test_ood = 0
    per_module_counts: Dict[str, Dict[str, int]] = {}

    # Filter modules if --modules is specified
    if args.modules:
        modules_set = set(args.modules)
        modules_config = [m for m in modules_config if m[0] in modules_set]
        print(f"Filtered to {len(modules_config)} modules: {[m[0] for m in modules_config]}")

    for module_name, train_fn_getter, test_fn_getter, generator_fn, needs_name in modules_config:
        print(f"\n=== {module_name} ===")
        per_module_counts[module_name] = {"train": 0, "test_id": 0, "test_ood": 0}

        # Train split
        train_fn = train_fn_getter()
        n = generate_module_split(
            generator_fn, train_fn, module_name, "train",
            args.train_per_module,
            os.path.join(args.output_dir, "train.jsonl"),
            args.workers, needs_name,
        )
        total_train += n
        per_module_counts[module_name]["train"] = n
        print(f"  Train: {n}")

        # Test ID split (interpolate)
        test_fn = test_fn_getter()
        n = generate_module_split(
            generator_fn, test_fn, module_name, "test_id",
            args.test_per_module,
            os.path.join(args.output_dir, "test_id.jsonl"),
            args.workers, needs_name,
        )
        total_test_id += n
        per_module_counts[module_name]["test_id"] = n
        print(f"  Test ID: {n}")

        # Test OOD split (extrapolate) - use test for now, could use test_extra
        n = generate_module_split(
            generator_fn, test_fn, module_name, "test_ood",
            args.test_per_module,
            os.path.join(args.output_dir, "test_ood.jsonl"),
            args.workers, needs_name,
        )
        total_test_ood += n
        per_module_counts[module_name]["test_ood"] = n
        print(f"  Test OOD: {n}")

    # Identify modules that could not reach the requested counts.
    undersampled: Dict[str, Dict[str, int]] = {}
    for module_name, counts in per_module_counts.items():
        short_train = args.train_per_module > 0 and counts["train"] < args.train_per_module
        short_id = args.test_per_module > 0 and counts["test_id"] < args.test_per_module
        short_ood = args.test_per_module > 0 and counts["test_ood"] < args.test_per_module
        if short_train or short_id or short_ood:
            undersampled[module_name] = counts

    if undersampled:
        print("\nSome modules produced fewer examples than requested (parsable questions only):")
        for name, counts in sorted(undersampled.items()):
            print(
                f"  {name}: "
                f"train {counts['train']}/{args.train_per_module}, "
                f"test_id {counts['test_id']}/{args.test_per_module}, "
                f"test_ood {counts['test_ood']}/{args.test_per_module}"
            )

    # Write metadata file with per-module counts and config.
    meta = {
        "vocab_size": get_vocab_size(),
        "output_dir": args.output_dir,
        "train_per_module_target": args.train_per_module,
        "test_per_module_target": args.test_per_module,
        "seed": args.seed,
        "modules": per_module_counts,
        "totals": {
            "train": total_train,
            "test_id": total_test_id,
            "test_ood": total_test_ood,
        },
        "undersampled_modules": undersampled,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print(f"\n{'='*50}")
    print(f"TOTAL: {total_train} train, {total_test_id} test_id, {total_test_ood} test_ood")
    print(f"Metadata written to: {meta_path}")
    print(f"Output: {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()