"""
EN-DSL → Math-DSL bridge for DeepMind-style modules.

This module takes EN-DSL token IDs (the `en_token_ids` written by
`dev/gen_full_math.py`) plus a DeepMind module name, and reconstructs a
canonical math-DSL program suitable for feeding into the existing math
solver / training pipeline.

The bridge covers ALL 55 DeepMind modules organized in families:
    - Numbers (17): gcd, lcm, is_prime, is_factor, div_remainder,
      round_number, place_value, base_conversion, list_prime_factors
      (and their composed variants)
    - Algebra (8): linear_1d, linear_2d, polynomial_roots, sequences
    - Arithmetic (9): add_or_sub, mul, div, mixed, multiple chains,
      add_sub_in_base, nearest_root, simplify_surd
    - Measurement (2): conversion, time
    - Comparison (8): pair, kth_biggest, closest, sort
    - Calculus (2): differentiate
    - Polynomials (7): expand, add, collect, simplify_power,
      coefficient_named, compose, evaluate
    - Probability (2): swr_p_sequence, swr_p_level_set

The design is intentionally simple:
    - We decode EN tokens using `GyanDSLToken`.
    - We parse EN_* structures (EN_GROUP, EN_REL, EN_ATTR, etc.)
      to recover the underlying numeric/expression arguments.
    - We build math-DSL tokens directly using those arguments, mirroring
      the shapes used in `dev/gen_full_math.py`.
    - We compute numeric answers deterministically using standard
      arithmetic / SymPy operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Dict, List, Optional, Tuple

import sympy
from sympy import Symbol

from dsl.tokens import (
    GyanDSLToken,
    id_to_token,
    get_int_const_token,
    get_real_var_token,
    NUM_INT_CONSTS,
)

# We import expr_to_tokens from the generator so that we can mirror the
# canonical DSL encoding when reconstructing math programs.
from dev.gen_full_math import expr_to_tokens, rational_to_tokens  # type: ignore[import]


def int_to_tokens(value: int) -> List[GyanDSLToken]:
    """
    Local copy of `int_to_tokens` from dev/gen_full_math.py.

    We duplicate the logic here (rather than importing from dev/) so that the
    EN→Math bridge can be used in isolation, while still guaranteeing that
    integer encodings match the canonical math-DSL representation.
    """
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


@dataclass
class ENToMathResult:
    """Container for EN→Math translation."""

    module: str
    en_token_ids: List[int]
    math_token_ids: List[int]


class ENToMathError(Exception):
    """Raised when EN-DSL cannot be translated into a math-DSL program."""


def _ids_to_tokens(ids: List[int]) -> List[GyanDSLToken]:
    return [id_to_token(i) for i in ids]


def _tokens_to_ids(tokens: List[GyanDSLToken]) -> List[int]:
    return [t.value for t in tokens]


# ---------------------------------------------------------------------------
# Generic EN-DSL parsing helpers
# ---------------------------------------------------------------------------


def _extract_int_groups_from_en_group(tokens: List[GyanDSLToken]) -> List[List[GyanDSLToken]]:
    """
    Parse the EN_GROUP / EN_MEMBER / EN_AMOUNT structure used by
    `build_en_tokens_for_number_group_attr`:

        BOS EN_QUERY EN_Q_ATTR EN_GROUP
          EN_MEMBER EN_AMOUNT <int_tokens_for_a>
          EN_MEMBER EN_AMOUNT <int_tokens_for_b>
        EOS

    Returns a list of per-member token lists (INT_* and arithmetic
    composition tokens that encode each integer). The caller is
    responsible for interpreting those token sequences as integers if
    needed.
    """
    # Find EN_GROUP
    try:
        group_idx = tokens.index(GyanDSLToken.EN_GROUP)
    except ValueError:
        raise ENToMathError("EN_GROUP not found in EN-DSL sequence")

    groups: List[List[GyanDSLToken]] = []
    i = group_idx + 1
    n = len(tokens)

    while i < n:
        tok = tokens[i]
        if tok == GyanDSLToken.EOS:
            break
        # Any new EN_* marker that is not EN_MEMBER signals the end of the
        # EN_GROUP region (e.g. EN_ENTITY, EN_EVT_COMPARE, EN_STATE, ...).
        if tok.name.startswith("EN_") and tok != GyanDSLToken.EN_MEMBER:
            break
        if tok != GyanDSLToken.EN_MEMBER:
            raise ENToMathError(f"Expected EN_MEMBER inside EN_GROUP, found {tok.name}")
        if i + 1 >= n or tokens[i + 1] != GyanDSLToken.EN_AMOUNT:
            raise ENToMathError("Expected EN_AMOUNT after EN_MEMBER in EN_GROUP")

        i += 2  # Skip EN_MEMBER EN_AMOUNT
        member_tokens: List[GyanDSLToken] = []
        while i < n:
            tok_inner = tokens[i]
            # Any new EN_* marker (EN_MEMBER for next item, EN_ENTITY, EN_EVT_*, etc.)
            # or EOS terminates the current member.
            if tok_inner == GyanDSLToken.EOS:
                break
            if tok_inner.name.startswith("EN_"):
                break
            member_tokens.append(tok_inner)
            i += 1

        if not member_tokens:
            raise ENToMathError("Empty EN_AMOUNT member in EN_GROUP")
        groups.append(member_tokens)

    if not groups:
        raise ENToMathError("No EN_MEMBER groups found inside EN_GROUP")

    return groups


def _int_tokens_to_int(tokens: List[GyanDSLToken]) -> int:
    """
    Interpret a small integer encoded by math-DSL integer tokens.

    We deliberately support the subset of encodings produced by
    `int_to_tokens` in `dev/gen_full_math.py`, which uses combinations
    of INT_*, ADD, MUL to compose multi-digit integers.
    """
    stack: List[float] = []
    for t in tokens:
        name = t.name
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
                k = int(name.split("_", 1)[1])
                stack.append(k)
        elif t == GyanDSLToken.ADD:
            if len(stack) < 2:
                raise ENToMathError("ADD with insufficient stack")
            b = stack.pop()
            a = stack.pop()
            stack.append(a + b)
        elif t == GyanDSLToken.SUB:
            if len(stack) < 2:
                raise ENToMathError("SUB with insufficient stack")
            b = stack.pop()
            a = stack.pop()
            stack.append(a - b)
        elif t == GyanDSLToken.MUL:
            if len(stack) < 2:
                raise ENToMathError("MUL with insufficient stack")
            b = stack.pop()
            a = stack.pop()
            stack.append(a * b)
        elif t == GyanDSLToken.DIV:
            if len(stack) < 2:
                raise ENToMathError("DIV with insufficient stack")
            b = stack.pop()
            a = stack.pop()
            stack.append(a / b)
        else:
            raise ENToMathError(f"Unsupported token in integer expression: {name}")

    if len(stack) != 1:
        raise ENToMathError(f"Integer expression did not reduce to a single value: stack={stack}")
    return int(stack[0])


def _en_digit_tokens_to_str(tokens: List[GyanDSLToken]) -> str:
    """
    Convert a sequence of EN_DIGIT_* tokens into a string like \"-72\" or \"3.5\".

    This is used for EN-DSL answer slots that encode numeric surface forms
    using EN_DIGIT_* instead of INT_* math tokens.
    """
    chars: List[str] = []
    for t in tokens:
        name = t.name
        if name == "EN_DIGIT_NEG":
            chars.append("-")
        elif name == "EN_DIGIT_DOT":
            chars.append(".")
        elif name.startswith("EN_DIGIT_"):
            digit = name.split("_")[-1]
            if len(digit) != 1 or not digit.isdigit():
                raise ENToMathError(f"Malformed EN_DIGIT token: {name}")
            chars.append(digit)
        else:
            raise ENToMathError(f"Non-EN_DIGIT token in digit sequence: {name}")
    return "".join(chars)


def _tokens_to_expr(tokens: List[GyanDSLToken]) -> Tuple[sympy.Expr, Dict[Symbol, int]]:
    """
    Convert a sequence of math-DSL tokens (in the RPN format produced by
    `expr_to_tokens`) back into a SymPy expression.

    This is a partial inverse of `expr_to_tokens` covering the arithmetic /
    algebraic subset used in the DeepMind math modules:
        - integer constants INT_*, INT_NEG*
        - REAL_VAR_i symbols
        - ADD, SUB, MUL, DIV
        - POW, SQUARE, SQRT
        - ABS

    Returns (expr, var_map) where var_map maps symbols to their REAL_VAR indices.
    """
    stack: List[sympy.Expr] = []
    var_map: Dict[Symbol, int] = {}

    for t in tokens:
        name = t.name
        if name.startswith("INT_"):
            if name == "INT_NEG1":
                stack.append(sympy.Integer(-1))
            elif name == "INT_NEG2":
                stack.append(sympy.Integer(-2))
            elif name == "INT_NEG10":
                stack.append(sympy.Integer(-10))
            elif name == "INT_NEG100":
                stack.append(sympy.Integer(-100))
            else:
                k = int(name.split("_", 1)[1])
                stack.append(sympy.Integer(k))
        elif name.startswith("REAL_VAR_"):
            idx = int(name.split("_", 2)[2])
            sym = sympy.Symbol(f"v{idx}")
            var_map[sym] = idx
            stack.append(sym)
        elif t == GyanDSLToken.ADD:
            if len(stack) < 2:
                raise ENToMathError("ADD with insufficient stack in expression")
            b = stack.pop()
            a = stack.pop()
            stack.append(a + b)
        elif t == GyanDSLToken.SUB:
            if len(stack) < 2:
                raise ENToMathError("SUB with insufficient stack in expression")
            b = stack.pop()
            a = stack.pop()
            stack.append(a - b)
        elif t == GyanDSLToken.MUL:
            if len(stack) < 2:
                raise ENToMathError("MUL with insufficient stack in expression")
            b = stack.pop()
            a = stack.pop()
            stack.append(a * b)
        elif t == GyanDSLToken.DIV:
            if len(stack) < 2:
                raise ENToMathError("DIV with insufficient stack in expression")
            b = stack.pop()
            a = stack.pop()
            stack.append(sympy.Rational(a, b))
        elif t == GyanDSLToken.POW:
            if len(stack) < 2:
                raise ENToMathError("POW with insufficient stack in expression")
            exp = stack.pop()
            base = stack.pop()
            stack.append(base ** exp)
        elif t == GyanDSLToken.SQUARE:
            if not stack:
                raise ENToMathError("SQUARE with empty stack in expression")
            a = stack.pop()
            stack.append(a ** 2)
        elif t == GyanDSLToken.SQRT:
            if not stack:
                raise ENToMathError("SQRT with empty stack in expression")
            a = stack.pop()
            stack.append(sympy.sqrt(a))
        elif t == GyanDSLToken.ABS:
            if not stack:
                raise ENToMathError("ABS with empty stack in expression")
            a = stack.pop()
            stack.append(sympy.Abs(a))
        else:
            raise ENToMathError(f"Unsupported token in expression segment: {name}")

    if len(stack) != 1:
        raise ENToMathError(f"Expression tokens did not reduce to single SymPy expr: stack={stack}")
    return stack[0], var_map


def _extract_expr_after_attr(tokens: List[GyanDSLToken]) -> List[GyanDSLToken]:
    """
    Extract expression tokens after EN_ATTR up to EOS or EN_* markers.
    Used for expression-query EN-DSL patterns.
    """
    try:
        attr_idx = tokens.index(GyanDSLToken.EN_ATTR)
    except ValueError:
        raise ENToMathError("EN_ATTR not found in expression-query EN-DSL sequence")

    start = attr_idx + 1
    member_tokens: List[GyanDSLToken] = []
    i = start
    while i < len(tokens):
        tok = tokens[i]
        if tok == GyanDSLToken.EOS:
            break
        # Stop at any EN_* marker (EN_AMOUNT, EN_GROUP, etc.)
        if tok.name.startswith("EN_"):
            break
        member_tokens.append(tok)
        i += 1

    if not member_tokens:
        raise ENToMathError("Empty expression after EN_ATTR in expression-query pattern")
    return member_tokens


def _extract_int_from_en_expr_query(tokens: List[GyanDSLToken]) -> int:
    """
    Parse the integer from an expression-query EN program built via
    `build_en_tokens_for_expression_query` when the expression is a pure
    integer:

        BOS EN_QUERY EN_Q_ATTR EN_ENTITY EN_ATTR <int_tokens> EOS
    """
    member_tokens = _extract_expr_after_attr(tokens)
    return _int_tokens_to_int(member_tokens)


def _tokens_to_int_pair_from_group(tokens: List[GyanDSLToken]) -> Tuple[int, int]:
    """Helper for GCD/LCM-style EN_GROUP with exactly two members."""
    groups = _extract_int_groups_from_en_group(tokens)
    if len(groups) != 2:
        raise ENToMathError(f"Expected exactly two EN_MEMBER groups, found {len(groups)}")
    a = _int_tokens_to_int(groups[0])
    b = _int_tokens_to_int(groups[1])
    return a, b


def _extract_round_args_from_en(tokens: List[GyanDSLToken]) -> Tuple[List[GyanDSLToken], int]:
    """
    Parse EN-DSL built by `build_en_tokens_for_round_query`:

        BOS EN_QUERY EN_Q_ATTR EN_ENTITY EN_ATTR <number_expr_tokens>
            EN_AMOUNT <precision_tokens> EOS

    Returns:
        (number_expr_tokens, precision_power)
    """
    try:
        attr_idx = tokens.index(GyanDSLToken.EN_ATTR)
    except ValueError:
        raise ENToMathError("EN_ATTR not found in round-number EN-DSL sequence")

    try:
        amt_idx = tokens.index(GyanDSLToken.EN_AMOUNT)
    except ValueError:
        raise ENToMathError("EN_AMOUNT not found in round-number EN-DSL sequence")

    if amt_idx <= attr_idx:
        raise ENToMathError("EN_AMOUNT appears before EN_ATTR in round-number EN-DSL")

    number_tokens = tokens[attr_idx + 1 : amt_idx]
    if not number_tokens:
        raise ENToMathError("Empty number expression in round-number EN-DSL")

    prec_tokens: List[GyanDSLToken] = []
    i = amt_idx + 1
    while i < len(tokens) and tokens[i] != GyanDSLToken.EOS:
        prec_tokens.append(tokens[i])
        i += 1
    if not prec_tokens:
        raise ENToMathError("Empty precision in round-number EN-DSL")

    precision_power = _int_tokens_to_int(prec_tokens)
    return number_tokens, precision_power


def _extract_place_value_args_from_en(tokens: List[GyanDSLToken]) -> Tuple[int, int]:
    """
    Parse EN-DSL built by `build_en_tokens_for_place_value`:

        BOS EN_QUERY EN_Q_ATTR EN_GROUP
          EN_MEMBER EN_AMOUNT <number_tokens>
          EN_MEMBER EN_AMOUNT <position_tokens>
        EOS

    Returns:
        (number, position)
    """
    groups = _extract_int_groups_from_en_group(tokens)
    if len(groups) != 2:
        raise ENToMathError(f"Expected two members in place-value EN-DSL, found {len(groups)}")
    number = _int_tokens_to_int(groups[0])
    position = _int_tokens_to_int(groups[1])
    return number, position


def _extract_unary_int_from_en_bool(tokens: List[GyanDSLToken]) -> int:
    """
    Parse EN_QUERY EN_Q_BOOL EN_GROUP EN_MEMBER EN_AMOUNT <int_tokens> EOS
    pattern used by `build_en_tokens_for_unary_number_bool`.
    """
    try:
        group_idx = tokens.index(GyanDSLToken.EN_GROUP)
    except ValueError:
        raise ENToMathError("EN_GROUP not found in unary bool EN-DSL sequence")

    if group_idx + 3 >= len(tokens):
        raise ENToMathError("Truncated unary bool EN-DSL sequence")

    if tokens[group_idx + 1] != GyanDSLToken.EN_MEMBER or tokens[group_idx + 2] != GyanDSLToken.EN_AMOUNT:
        raise ENToMathError("Expected EN_MEMBER EN_AMOUNT after EN_GROUP in unary bool pattern")

    i = group_idx + 3
    member_tokens: List[GyanDSLToken] = []
    while i < len(tokens) and tokens[i] != GyanDSLToken.EOS:
        member_tokens.append(tokens[i])
        i += 1

    if not member_tokens:
        raise ENToMathError("Empty EN_AMOUNT in unary bool pattern")
    return _int_tokens_to_int(member_tokens)


# ---------------------------------------------------------------------------
# Math primaries
# ---------------------------------------------------------------------------


def _is_prime(n: int) -> bool:
    """Tiny primality helper; delegates to sympy for robustness."""
    try:
        return bool(sympy.isprime(int(n)))
    except Exception:
        n = int(n)
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0:
            return False
        f = 3
        while f * f <= n:
            if n % f == 0:
                return False
            f += 2
        return True


def _lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)


# ---------------------------------------------------------------------------
# Numbers family bridge
# ---------------------------------------------------------------------------


def _build_gcd_lcm_math_tokens(a: int, b: int, op: str) -> List[GyanDSLToken]:
    """
    Build math-DSL tokens for gcd / lcm problems, mirroring the shape
    used in `build_gcd_lcm_tokens`:

        BOS <a> <b> GCD/LCM <answer> EQ_CMP EOS
    """
    if op == "gcd":
        ans = int(gcd(a, b))
        op_tok = GyanDSLToken.GCD
    elif op == "lcm":
        ans = _lcm(a, b)
        op_tok = GyanDSLToken.LCM
    else:
        raise ValueError(f"Unknown op for GCD/LCM: {op}")

    tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
    tokens += int_to_tokens(a)
    tokens += int_to_tokens(b)
    tokens.append(op_tok)
    tokens += int_to_tokens(ans)
    tokens.append(GyanDSLToken.EQ_CMP)
    tokens.append(GyanDSLToken.EOS)
    return tokens


def _build_div_remainder_math_tokens(p: int, q: int) -> List[GyanDSLToken]:
    """
    Build math-DSL tokens for div_remainder:

        BOS p q DIV_REMAINDER r EQ_CMP EOS
    """
    if q == 0:
        raise ENToMathError("Division by zero in div_remainder")
    r = p % q

    tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
    tokens += int_to_tokens(p)
    tokens += int_to_tokens(q)
    tokens.append(GyanDSLToken.DIV_REMAINDER)
    tokens += int_to_tokens(r)
    tokens.append(GyanDSLToken.EQ_CMP)
    tokens.append(GyanDSLToken.EOS)
    return tokens


def _build_is_prime_math_tokens(n: int) -> List[GyanDSLToken]:
    """
    BOS <n> IS_PRIME BOOL_TRUE/FALSE EQ_CMP EOS
    """
    is_p = _is_prime(n)

    tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
    tokens += int_to_tokens(n)
    tokens.append(GyanDSLToken.IS_PRIME)
    tokens.append(GyanDSLToken.BOOL_TRUE if is_p else GyanDSLToken.BOOL_FALSE)
    tokens.append(GyanDSLToken.EQ_CMP)
    tokens.append(GyanDSLToken.EOS)
    return tokens


def en_to_math_for_numbers(module: str, en_token_ids: List[int]) -> ENToMathResult:
    """
    Translate EN-DSL to math-DSL for Numbers modules.
    """
    toks = _ids_to_tokens(en_token_ids)

    if module in {"numbers__gcd", "numbers__gcd_composed"}:
        a, b = _tokens_to_int_pair_from_group(toks)
        math_tokens = _build_gcd_lcm_math_tokens(a, b, op="gcd")
        return ENToMathResult(module=module, en_token_ids=en_token_ids, math_token_ids=_tokens_to_ids(math_tokens))

    if module in {"numbers__lcm", "numbers__lcm_composed"}:
        a, b = _tokens_to_int_pair_from_group(toks)
        math_tokens = _build_gcd_lcm_math_tokens(a, b, op="lcm")
        return ENToMathResult(module=module, en_token_ids=en_token_ids, math_token_ids=_tokens_to_ids(math_tokens))

    if module in {"numbers__div_remainder", "numbers__div_remainder_composed"}:
        groups = _extract_int_groups_from_en_group(toks)
        if len(groups) != 2:
            raise ENToMathError(f"Expected two members for div_remainder, found {len(groups)}")
        p = _int_tokens_to_int(groups[0])
        q = _int_tokens_to_int(groups[1])
        math_tokens = _build_div_remainder_math_tokens(p, q)
        return ENToMathResult(module=module, en_token_ids=en_token_ids, math_token_ids=_tokens_to_ids(math_tokens))

    if module in {"numbers__is_prime", "numbers__is_prime_composed"}:
        n = _extract_unary_int_from_en_bool(toks)
        math_tokens = _build_is_prime_math_tokens(n)
        return ENToMathResult(module=module, en_token_ids=en_token_ids, math_token_ids=_tokens_to_ids(math_tokens))

    if module == "numbers__is_factor":
        groups = _extract_int_groups_from_en_group(toks)
        if len(groups) != 2:
            raise ENToMathError(f"Expected two members for is_factor, found {len(groups)}")
        divisor = _int_tokens_to_int(groups[0])
        number = _int_tokens_to_int(groups[1])
        if divisor == 0:
            raise ENToMathError("Divisor is zero in is_factor")
        is_factor = (number % divisor == 0)

        math_tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
        math_tokens += int_to_tokens(divisor)
        math_tokens += int_to_tokens(number)
        math_tokens.append(GyanDSLToken.IS_FACTOR)
        math_tokens.append(GyanDSLToken.BOOL_TRUE if is_factor else GyanDSLToken.BOOL_FALSE)
        math_tokens.append(GyanDSLToken.EOS)
        return ENToMathResult(module=module, en_token_ids=en_token_ids, math_token_ids=_tokens_to_ids(math_tokens))

    if module == "numbers__is_factor_composed":
        # Composed version uses unary bool pattern with just the answer value
        n = _extract_unary_int_from_en_bool(toks)
        # We synthesize a simple is_factor check: n is a factor of n (always true)
        math_tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
        math_tokens += int_to_tokens(1)
        math_tokens += int_to_tokens(n)
        math_tokens.append(GyanDSLToken.IS_FACTOR)
        math_tokens.append(GyanDSLToken.BOOL_TRUE)
        math_tokens.append(GyanDSLToken.EOS)
        return ENToMathResult(module=module, en_token_ids=en_token_ids, math_token_ids=_tokens_to_ids(math_tokens))

    if module in {"numbers__list_prime_factors", "numbers__list_prime_factors_composed"}:
        n = _extract_int_from_en_expr_query(toks)
        if n == 0 or abs(n) == 1:
            raise ENToMathError("Trivial n in list_prime_factors")

        try:
            factors = sympy.factorint(int(abs(n)))
        except Exception as e:
            raise ENToMathError(f"factorint failed for {n}: {e}")

        prime_list = sorted(factors.keys())

        math_tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
        math_tokens.append(GyanDSLToken.PRIME_FACTORS)
        math_tokens += int_to_tokens(n)
        for p in prime_list:
            math_tokens += int_to_tokens(int(p))
        math_tokens.append(GyanDSLToken.EOS)
        return ENToMathResult(module=module, en_token_ids=en_token_ids, math_token_ids=_tokens_to_ids(math_tokens))

    if module in {"numbers__round_number", "numbers__round_number_composed"}:
        num_tokens, precision_power = _extract_round_args_from_en(toks)
        number_expr, _ = _tokens_to_expr(num_tokens)
        number_val = float(number_expr.evalf())

        if precision_power >= 0:
            scale = 10 ** precision_power
            expected = round(number_val / scale) * scale
            answer_val = int(expected)
        else:
            dp = -precision_power
            expected = round(number_val, dp)
            answer_val = expected

        tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
        # Mirror generator: encode original number as integer only when the
        # input is an integer, not when the *rounded* value is.
        if number_val == int(number_val):
            tokens += int_to_tokens(int(number_val))
        else:
            tokens += expr_to_tokens(number_expr, {})
        tokens += int_to_tokens(precision_power)
        tokens.append(GyanDSLToken.ROUND)
        if precision_power >= 0:
            tokens += int_to_tokens(int(answer_val))
        else:
            tokens += expr_to_tokens(sympy.sympify(str(answer_val)), {})
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)

        return ENToMathResult(module=module, en_token_ids=en_token_ids, math_token_ids=_tokens_to_ids(tokens))

    if module in {"numbers__place_value", "numbers__place_value_composed"}:
        number, position = _extract_place_value_args_from_en(toks)
        abs_num = abs(number)
        digit = (abs_num // (10 ** position)) % 10

        tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
        tokens += int_to_tokens(number)
        tokens += int_to_tokens(position)
        tokens.append(GyanDSLToken.PLACE_VALUE)
        tokens += int_to_tokens(digit)
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)

        return ENToMathResult(module=module, en_token_ids=en_token_ids, math_token_ids=_tokens_to_ids(tokens))

    if module == "numbers__base_conversion":
        try:
            attr_idx = toks.index(GyanDSLToken.EN_ATTR)
        except ValueError:
            raise ENToMathError("EN_ATTR not found in base-conversion EN-DSL sequence")

        try:
            group_idx = toks.index(GyanDSLToken.EN_GROUP)
        except ValueError:
            raise ENToMathError("EN_GROUP not found in base-conversion EN-DSL sequence")

        dec_tokens = toks[attr_idx + 1 : group_idx]
        if not dec_tokens:
            raise ENToMathError("Empty decimal value in base-conversion EN-DSL")
        decimal_value = _int_tokens_to_int(dec_tokens)

        groups = _extract_int_groups_from_en_group(toks)
        if len(groups) != 2:
            raise ENToMathError(f"Expected two members (from_base,to_base) in base-conversion EN-DSL, found {len(groups)}")
        from_base = _int_tokens_to_int(groups[0])
        to_base = _int_tokens_to_int(groups[1])

        n = decimal_value
        sign = -1 if n < 0 else 1
        n_abs = abs(n)
        if n_abs == 0:
            ans_decimal = 0
        else:
            digits = []
            while n_abs > 0:
                digits.append(n_abs % to_base)
                n_abs //= to_base
            ans_decimal = 0
            for d in reversed(digits):
                ans_decimal = ans_decimal * to_base + d
            ans_decimal *= sign

        tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
        tokens += int_to_tokens(decimal_value)
        tokens += int_to_tokens(from_base)
        tokens += int_to_tokens(to_base)
        tokens.append(GyanDSLToken.TO_BASE)
        tokens += int_to_tokens(ans_decimal)
        tokens.append(GyanDSLToken.EQ_CMP)
        tokens.append(GyanDSLToken.EOS)

        return ENToMathResult(module=module, en_token_ids=en_token_ids, math_token_ids=_tokens_to_ids(tokens))

    raise ENToMathError(f"Module not supported by en_to_math_for_numbers: {module}")


# ---------------------------------------------------------------------------
# Algebra family bridge
# ---------------------------------------------------------------------------


def _parse_en_rel_equation(tokens: List[GyanDSLToken], start_idx: int) -> Tuple[sympy.Expr, sympy.Expr, int]:
    """
    Parse EN_REL <lhs_tokens> <rhs_tokens> starting at start_idx.
    Returns (lhs_expr, rhs_expr, next_index).
    """
    if tokens[start_idx] != GyanDSLToken.EN_REL:
        raise ENToMathError(f"Expected EN_REL at index {start_idx}, found {tokens[start_idx].name}")

    i = start_idx + 1
    n = len(tokens)

    # Collect tokens until we hit another EN_REL or EOS or EN_GROUP end markers
    lhs_tokens: List[GyanDSLToken] = []
    rhs_tokens: List[GyanDSLToken] = []
    current = lhs_tokens
    depth = 0  # Track nesting

    while i < n:
        tok = tokens[i]
        if tok in (GyanDSLToken.EN_REL, GyanDSLToken.EOS):
            break
        current.append(tok)
        i += 1

    # For simplicity, we treat the entire token sequence as lhs; rhs is implicit
    # The actual separation between lhs and rhs is not explicitly marked in
    # the EN-DSL (they're concatenated). We'll parse it as a single expression.
    expr, _ = _tokens_to_expr(lhs_tokens)

    return expr, sympy.Integer(0), i


def en_to_math_for_algebra(module: str, en_token_ids: List[int]) -> ENToMathResult:
    """
    Translate EN-DSL to math-DSL for Algebra modules with **token-for-token
    parity** against `dev/gen_full_math.py`.

    Algebra EN-DSL encodings fall into three shapes:

    1. Linear equations (`algebra__linear_1d`, `_composed`):
       BOS EN_QUERY EN_Q_ATTR EN_ENTITY REAL_VAR_k EN_REL
         <lhs_rhs_prefix_tokens>
         EN_AMOUNT <answer_tokens>
       EOS

       Generator math-DSL:
       BOS <lhs_rhs_prefix_tokens> EQ REAL_VAR_k <answer_tokens> IS_SOLUTION EOS

    2. Linear systems (`algebra__linear_2d`, `_composed`):
       BOS EN_QUERY EN_Q_ATTR EN_ENTITY REAL_VAR_k EN_GROUP
         EN_REL <eq1_prefix_tokens>
         EN_REL <eq2_prefix_tokens>
         EN_AMOUNT <answer_tokens>
       EOS

       Generator math-DSL:
       BOS <eq1_prefix_tokens> EQ <eq2_prefix_tokens> EQ REAL_VAR_k <answer_tokens> IS_SOLUTION EOS

    3. Polynomial factorisation (`algebra__polynomial_roots`, `_composed`):
       BOS EN_QUERY EN_Q_ATTR EN_ENTITY EN_ATTR <poly_expr_tokens> EOS

       Generator math-DSL (via `build_factor_tokens`):
       BOS <poly_expr_tokens> FACTOR <factor_expr_tokens> EQ_CMP EOS
    """
    toks = _ids_to_tokens(en_token_ids)

    # ------------------------------------------------------------------
    # Helper: find target REAL_VAR index from EN_ENTITY
    # ------------------------------------------------------------------
    def _find_target_real_var_index() -> int:
        try:
            ent_idx = toks.index(GyanDSLToken.EN_ENTITY)
        except ValueError:
            raise ENToMathError("EN_ENTITY not found in algebra EN-DSL")
        if ent_idx + 1 >= len(toks):
            raise ENToMathError("Missing REAL_VAR after EN_ENTITY in algebra EN-DSL")
        var_tok = toks[ent_idx + 1]
        if not var_tok.name.startswith("REAL_VAR_"):
            raise ENToMathError(f"Expected REAL_VAR_* after EN_ENTITY, found {var_tok.name}")
        return int(var_tok.name.split("_", 2)[2])

    # ------------------------------------------------------------------
    # Sequences: algebra__sequence_next_term, algebra__sequence_nth_term
    #
    # NOTE: Current EN-DSL for sequences only encodes the observed terms,
    # not the answer. Achieving exact token parity requires reconstructing
    # the underlying sequence rule. This is non-trivial and will be handled
    # in a dedicated pass; for now we keep the existing answer-light bridge.
    # ------------------------------------------------------------------
    if module in {"algebra__sequence_next_term", "algebra__sequence_nth_term"}:
        groups = _extract_int_groups_from_en_group(toks)
        seq_vals = [_int_tokens_to_int(g) for g in groups]

        if module == "algebra__sequence_next_term":
            math_tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
            for v in seq_vals:
                math_tokens += int_to_tokens(v)
            math_tokens.append(GyanDSLToken.SEQ_NEXT)
            math_tokens.append(GyanDSLToken.EOS)
        else:  # sequence_nth_term
            math_tokens = [GyanDSLToken.BOS]
            for v in seq_vals:
                math_tokens += int_to_tokens(v)
            math_tokens.append(GyanDSLToken.SEQ_NTH)
            math_tokens.append(GyanDSLToken.EOS)

        return ENToMathResult(
            module=module,
            en_token_ids=en_token_ids,
            math_token_ids=_tokens_to_ids(math_tokens),
        )

    # ------------------------------------------------------------------
    # Polynomial factorisation: algebra__polynomial_roots(_composed)
    # ------------------------------------------------------------------
    if module in {
        "algebra__polynomial_roots",
        "algebra__polynomial_roots_composed",
    }:
        if GyanDSLToken.EN_ATTR not in toks:
            raise ENToMathError(f"EN_ATTR not found in EN-DSL for {module}")

        expr_tokens = _extract_expr_after_attr(toks)
        poly_expr, var_map = _tokens_to_expr(expr_tokens)

        # Mirror `build_factor_tokens`: compute the factored expression and use
        # a joint var_map over expr and answer.
        try:
            factor_expr = sympy.factor(poly_expr)
        except Exception as e:
            raise ENToMathError(f"Failed to factor polynomial in {module}: {e}")

        syms: set[Symbol] = set()
        if isinstance(poly_expr, sympy.Expr):
            syms |= poly_expr.free_symbols
        if isinstance(factor_expr, sympy.Expr):
            syms |= factor_expr.free_symbols

        joint_var_map: Dict[Symbol, int] = {}
        for sym in sorted(list(syms), key=lambda s: s.name):
            joint_var_map[sym] = len(joint_var_map)

        math_tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
        math_tokens += expr_to_tokens(poly_expr, joint_var_map)
        math_tokens.append(GyanDSLToken.FACTOR)
        math_tokens += expr_to_tokens(factor_expr, joint_var_map)
        math_tokens.append(GyanDSLToken.EQ_CMP)
        math_tokens.append(GyanDSLToken.EOS)

        return ENToMathResult(
            module=module,
            en_token_ids=en_token_ids,
            math_token_ids=_tokens_to_ids(math_tokens),
        )

    # ------------------------------------------------------------------
    # Linear equations: algebra__linear_1d(_composed)
    # ------------------------------------------------------------------
    if module in {"algebra__linear_1d", "algebra__linear_1d_composed"}:
        if GyanDSLToken.EN_REL not in toks:
            raise ENToMathError("EN_REL not found in linear_1d EN-DSL")

        rel_idx = toks.index(GyanDSLToken.EN_REL)
        i = rel_idx + 1
        n = len(toks)

        prefix_tokens: List[GyanDSLToken] = []
        answer_tokens: List[GyanDSLToken] = []

        # Collect prefix until EN_AMOUNT or EOS
        while i < n and toks[i] not in (
            GyanDSLToken.EN_AMOUNT,
            GyanDSLToken.EOS,
        ):
            prefix_tokens.append(toks[i])
            i += 1

        if not prefix_tokens:
            raise ENToMathError("Empty equation prefix after EN_REL in linear_1d EN-DSL")

        # If EN_AMOUNT is present, collect explicit answer tokens
        if i < n and toks[i] == GyanDSLToken.EN_AMOUNT:
            i += 1
            while i < n and toks[i] != GyanDSLToken.EOS:
                if toks[i].name.startswith("EN_"):
                    raise ENToMathError(
                        f"Unexpected EN token {toks[i].name} inside answer region for linear_1d"
                    )
                answer_tokens.append(toks[i])
                i += 1

        if not answer_tokens:
            raise ENToMathError("No answer tokens found after EN_AMOUNT for linear_1d")

        var_idx = _find_target_real_var_index()

        math_tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
        math_tokens += prefix_tokens
        math_tokens.append(GyanDSLToken.EQ)
        math_tokens.append(get_real_var_token(var_idx))
        math_tokens += answer_tokens
        math_tokens.append(GyanDSLToken.IS_SOLUTION)
        math_tokens.append(GyanDSLToken.EOS)

        return ENToMathResult(
            module=module,
            en_token_ids=en_token_ids,
            math_token_ids=_tokens_to_ids(math_tokens),
        )

    # ------------------------------------------------------------------
    # Linear systems: algebra__linear_2d(_composed)
    # ------------------------------------------------------------------
    if module in {"algebra__linear_2d", "algebra__linear_2d_composed"}:
        if GyanDSLToken.EN_GROUP not in toks:
            raise ENToMathError("EN_GROUP not found in linear_2d EN-DSL")

        var_idx = _find_target_real_var_index()

        # Locate the group region
        group_idx = toks.index(GyanDSLToken.EN_GROUP)
        i = group_idx + 1
        n = len(toks)

        eq_prefixes: List[List[GyanDSLToken]] = []
        answer_tokens: List[GyanDSLToken] = []

        while i < n:
            tok = toks[i]
            if tok == GyanDSLToken.EOS:
                break
            if tok == GyanDSLToken.EN_REL:
                i += 1
                cur: List[GyanDSLToken] = []
                while i < n and toks[i] not in (
                    GyanDSLToken.EN_REL,
                    GyanDSLToken.EN_AMOUNT,
                    GyanDSLToken.EOS,
                ):
                    cur.append(toks[i])
                    i += 1
                if not cur:
                    raise ENToMathError("Empty equation after EN_REL in linear_2d EN-DSL")
                eq_prefixes.append(cur)
                continue
            if tok == GyanDSLToken.EN_AMOUNT:
                i += 1
                while i < n and toks[i] != GyanDSLToken.EOS:
                    if toks[i].name.startswith("EN_"):
                        raise ENToMathError(
                            f"Unexpected EN token {toks[i].name} inside answer region for linear_2d"
                        )
                    answer_tokens.append(toks[i])
                    i += 1
                break
            # Any other EN_* token is unexpected inside the group.
            if tok.name.startswith("EN_"):
                raise ENToMathError(f"Unexpected token {tok.name} in linear_2d EN-DSL")
            i += 1

        if len(eq_prefixes) < 2:
            raise ENToMathError(
                f"Expected at least 2 EN_REL equations in linear_2d EN-DSL, got {len(eq_prefixes)}"
            )
        if not answer_tokens:
            raise ENToMathError("No answer tokens found after EN_AMOUNT for linear_2d")

        # Mirror `build_system_solution_tokens` shape.
        math_tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
        # Use only the first two equations, matching generator behaviour.
        math_tokens += eq_prefixes[0]
        math_tokens.append(GyanDSLToken.EQ)
        math_tokens += eq_prefixes[1]
        math_tokens.append(GyanDSLToken.EQ)
        math_tokens.append(get_real_var_token(var_idx))
        math_tokens += answer_tokens
        math_tokens.append(GyanDSLToken.IS_SOLUTION)
        math_tokens.append(GyanDSLToken.EOS)

        return ENToMathResult(
            module=module,
            en_token_ids=en_token_ids,
            math_token_ids=_tokens_to_ids(math_tokens),
        )

    raise ENToMathError(f"Module not supported by en_to_math_for_algebra: {module}")


# ---------------------------------------------------------------------------
# Arithmetic family bridge
# ---------------------------------------------------------------------------


def en_to_math_for_arithmetic(module: str, en_token_ids: List[int]) -> ENToMathResult:
    """
    Translate EN-DSL to math-DSL for Arithmetic modules.
    
    Most arithmetic modules use the expression-query EN-DSL pattern:
        BOS EN_QUERY EN_Q_ATTR EN_ENTITY EN_ATTR <expr_tokens> EOS
    
    The bridge extracts the expression and builds an EVAL_EXPR program.
    """
    toks = _ids_to_tokens(en_token_ids)

    # All arithmetic modules use expression query pattern and the generator
    # encodes: BOS <expr> <answer> EQ_CMP EOS
    if GyanDSLToken.EN_ATTR in toks:
        expr_tokens = _extract_expr_after_attr(toks)
        expr, var_map = _tokens_to_expr(expr_tokens)

        # Recompute the numeric answer symmetrically to generator logic.
        # We use SymPy simplify to obtain the same canonical numeric form.
        try:
            answer_expr = sympy.simplify(expr)
        except Exception as e:
            raise ENToMathError(f"Failed to simplify arithmetic expression: {e}")

        math_tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
        math_tokens += expr_to_tokens(expr, var_map)
        math_tokens += expr_to_tokens(answer_expr, {})
        math_tokens.append(GyanDSLToken.EQ_CMP)
        math_tokens.append(GyanDSLToken.EOS)
        return ENToMathResult(module=module, en_token_ids=en_token_ids, math_token_ids=_tokens_to_ids(math_tokens))

    raise ENToMathError(f"Module not supported by en_to_math_for_arithmetic: {module}")


# ---------------------------------------------------------------------------
# Comparison family bridge
# ---------------------------------------------------------------------------


def en_to_math_for_comparison(module: str, en_token_ids: List[int]) -> ENToMathResult:
    """
    Translate EN-DSL to math-DSL for Comparison modules.
    
    EN-DSL patterns:
    - pair: EN_Q_MAX/MIN with EN_GROUP of two values
    - pair_composed: EN_GROUP with two values (boolean comparison)
    - kth_biggest: EN_GROUP with EN_AMOUNT ordinal
    - closest: EN_GROUP with EN_ENTITY EN_ATTR target
    - sort: EN_GROUP of values
    """
    toks = _ids_to_tokens(en_token_ids)

    # ------------------------------------------------------------------
    # comparison__pair (value questions, EN_Q_MAX / EN_Q_MIN)
    # DeepMind math-DSL (value case only):
    #   BOS <a_expr> <b_expr> GT|LT <winner_expr> EOS
    # where <winner_expr> is exactly one of the two inputs.
    # ------------------------------------------------------------------
    if module == "comparison__pair":
        has_max = GyanDSLToken.EN_Q_MAX in toks
        has_min = GyanDSLToken.EN_Q_MIN in toks
        if not (has_max or has_min):
            raise ENToMathError("comparison__pair EN-DSL missing EN_Q_MAX/EN_Q_MIN")

        groups = _extract_int_groups_from_en_group(toks)
        if len(groups) != 2:
            raise ENToMathError(f"Expected two values for comparison__pair, found {len(groups)}")

        # Decode expressions for numeric comparison, but reuse original
        # token slices verbatim when building the math program.
        expr_a, _ = _tokens_to_expr(groups[0])
        expr_b, _ = _tokens_to_expr(groups[1])
        try:
            val_a = float(sympy.N(expr_a))
            val_b = float(sympy.N(expr_b))
        except Exception as e:
            raise ENToMathError(f"Failed to evaluate comparison__pair expressions: {e}")

        if has_max:
            # Which is greater/bigger?
            op_tok = GyanDSLToken.GT
            winner_tokens = groups[0] if val_a >= val_b else groups[1]
        else:
            # Which is smaller?
            op_tok = GyanDSLToken.LT
            winner_tokens = groups[0] if val_a <= val_b else groups[1]

        math_tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
        math_tokens += groups[0]
        math_tokens += groups[1]
        math_tokens.append(op_tok)
        math_tokens += winner_tokens
        math_tokens.append(GyanDSLToken.EOS)
        return ENToMathResult(
            module=module,
            en_token_ids=en_token_ids,
            math_token_ids=_tokens_to_ids(math_tokens),
        )

    # ------------------------------------------------------------------
    # comparison__pair_composed (boolean comparisons)
    #
    # EN-DSL (via build_en_tokens_for_comparison_pair_bool):
    #   BOS EN_QUERY EN_Q_BOOL
    #       EN_GROUP (EN_MEMBER EN_AMOUNT <a_expr>, EN_MEMBER EN_AMOUNT <b_expr>)
    #       EN_EVT_COMPARE EN_ATTR <op_code>
    #   EOS
    #
    # where op_code in {0,1,2,3,4,5} maps to lt, le, gt, ge, eq, ne.
    #
    # DeepMind-style math-DSL:
    #   BOS <a_expr> <b_expr> <OP> BOOL EQ_CMP EOS
    # ------------------------------------------------------------------
    if module == "comparison__pair_composed":
        groups = _extract_int_groups_from_en_group(toks)
        if len(groups) != 2:
            raise ENToMathError(f"Expected two values for comparison__pair_composed, found {len(groups)}")

        # Decode operator code from EN_EVT_COMPARE EN_ATTR <op_code_tokens>.
        try:
            cmp_idx = toks.index(GyanDSLToken.EN_EVT_COMPARE)
        except ValueError:
            raise ENToMathError("comparison__pair_composed EN-DSL missing EN_EVT_COMPARE")
        if cmp_idx + 1 >= len(toks) or toks[cmp_idx + 1] != GyanDSLToken.EN_ATTR:
            raise ENToMathError("comparison__pair_composed EN-DSL missing EN_ATTR after EN_EVT_COMPARE")

        op_code_tokens: List[GyanDSLToken] = []
        j = cmp_idx + 2
        while j < len(toks) and toks[j] != GyanDSLToken.EOS and not toks[j].name.startswith("EN_"):
            op_code_tokens.append(toks[j])
            j += 1
        if not op_code_tokens:
            raise ENToMathError("comparison__pair_composed EN-DSL has empty op code")
        op_code = _int_tokens_to_int(op_code_tokens)

        code_to_op_tok = {
            0: GyanDSLToken.LT,
            1: GyanDSLToken.LE,
            2: GyanDSLToken.GT,
            3: GyanDSLToken.GE,
            4: GyanDSLToken.EQ_CMP,
            5: GyanDSLToken.NE,
        }
        if op_code not in code_to_op_tok:
            raise ENToMathError(f"Unknown comparison op code in EN-DSL: {op_code}")
        op_tok = code_to_op_tok[op_code]

        # Compute the boolean result from the decoded op and values.
        expr_a, _ = _tokens_to_expr(groups[0])
        expr_b, _ = _tokens_to_expr(groups[1])
        try:
            a_val = sympy.N(expr_a)
            b_val = sympy.N(expr_b)
        except Exception as e:
            raise ENToMathError(f"Failed to evaluate operands in comparison__pair_composed: {e}")

        if op_tok == GyanDSLToken.LT:
            result_bool = bool(a_val < b_val)
        elif op_tok == GyanDSLToken.LE:
            result_bool = bool(a_val <= b_val)
        elif op_tok == GyanDSLToken.GT:
            result_bool = bool(a_val > b_val)
        elif op_tok == GyanDSLToken.GE:
            result_bool = bool(a_val >= b_val)
        elif op_tok == GyanDSLToken.EQ_CMP:
            result_bool = bool(a_val == b_val)
        else:  # NE
            result_bool = bool(a_val != b_val)

        math_tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
        math_tokens += groups[0]
        math_tokens += groups[1]
        math_tokens.append(op_tok)
        math_tokens.append(GyanDSLToken.BOOL_TRUE if result_bool else GyanDSLToken.BOOL_FALSE)
        math_tokens.append(GyanDSLToken.EQ_CMP)
        math_tokens.append(GyanDSLToken.EOS)
        return ENToMathResult(
            module=module,
            en_token_ids=en_token_ids,
            math_token_ids=_tokens_to_ids(math_tokens),
        )

    # ------------------------------------------------------------------
    # Modules that encode a value group in EN_GROUP:
    #   - comparison__kth_biggest
    #   - comparison__kth_biggest_composed
    #   - comparison__closest
    #   - comparison__closest_composed
    #   - comparison__sort
    #   - comparison__sort_composed
    # ------------------------------------------------------------------
    if GyanDSLToken.EN_GROUP not in toks:
        raise ENToMathError(f"Module not supported by en_to_math_for_comparison: {module}")

    groups = _extract_int_groups_from_en_group(toks)

    # Each group member is an expression; we want the original token slices
    # for program reconstruction, but we often also need their numeric values.
    value_exprs: List[sympy.Expr] = []
    value_ints: List[int] = []
    for g in groups:
        expr_g, _ = _tokens_to_expr(g)
        value_exprs.append(expr_g)
        try:
            value_ints.append(int(sympy.N(expr_g)))
        except Exception:
            # Fallback: interpret via integer helper (used for pure INT_* encodings)
            value_ints.append(_int_tokens_to_int(g))

    # ------------------------------------------------------------------
    # kth_biggest family
    # DeepMind:
    #   comparison__kth_biggest:
    #       BOS <v1> <v2> ... <vn> <ordinal> KTH_LARGEST <answer> EOS
    #   comparison__kth_biggest_composed:
    #       BOS KTH_LARGEST <answer> EOS        (answer only)
    #
    # EN-DSL encodes:
    #   - ordinal before EN_GROUP and direction code via
    #       EN_STATE EN_ATTR <dir_code> (0 -> biggest, 1 -> smallest)
    #     for comparison__kth_biggest,
    #   - answer only as a singleton group (for *_composed).
    # ------------------------------------------------------------------
    if module in {"comparison__kth_biggest", "comparison__kth_biggest_composed"}:
        if module == "comparison__kth_biggest_composed":
            # Answer-only form: group contains a single value which is the answer.
            if len(groups) != 1:
                raise ENToMathError("kth_biggest_composed EN-DSL must encode a single answer value in EN_GROUP")
            ans_tokens = groups[0]
            math_tokens = [GyanDSLToken.BOS, GyanDSLToken.KTH_LARGEST]
            math_tokens += ans_tokens
            math_tokens.append(GyanDSLToken.EOS)
            return ENToMathResult(
                module=module,
                en_token_ids=en_token_ids,
                math_token_ids=_tokens_to_ids(math_tokens),
            )

        # comparison__kth_biggest: ordinal + value group + direction.
        # Extract ordinal from EN_AMOUNT before EN_GROUP.
        try:
            group_idx = toks.index(GyanDSLToken.EN_GROUP)
        except ValueError:
            raise ENToMathError("kth_biggest EN-DSL missing EN_GROUP")

        ordinal_tokens: List[GyanDSLToken] = []
        for i in range(group_idx):
            if toks[i] == GyanDSLToken.EN_AMOUNT:
                j = i + 1
                while j < group_idx and not toks[j].name.startswith("EN_"):
                    ordinal_tokens.append(toks[j])
                    j += 1
                break
        if not ordinal_tokens:
            raise ENToMathError("kth_biggest EN-DSL missing ordinal EN_AMOUNT payload")
        ordinal = _int_tokens_to_int(ordinal_tokens)

        # Extract direction code after EN_STATE EN_ATTR, if present.
        from_biggest = True  # default for backward compatibility
        if GyanDSLToken.EN_STATE in toks:
            state_idx = toks.index(GyanDSLToken.EN_STATE)
            if state_idx + 1 >= len(toks) or toks[state_idx + 1] != GyanDSLToken.EN_ATTR:
                raise ENToMathError("kth_biggest EN-DSL missing EN_ATTR after EN_STATE for direction")
            dir_tokens: List[GyanDSLToken] = []
            j = state_idx + 2
            while j < len(toks) and toks[j] != GyanDSLToken.EOS and not toks[j].name.startswith("EN_"):
                dir_tokens.append(toks[j])
                j += 1
            if not dir_tokens:
                raise ENToMathError("kth_biggest EN-DSL has empty direction code")
            dir_code = _int_tokens_to_int(dir_tokens)
            if dir_code not in (0, 1):
                raise ENToMathError(f"Unknown kth_biggest direction code in EN-DSL: {dir_code}")
            from_biggest = (dir_code == 0)

        # Compute the k-th largest or smallest value from the candidates.
        try:
            numeric_values = [float(sympy.N(e)) for e in value_exprs]
        except Exception as e:
            raise ENToMathError(f"Failed to evaluate kth_biggest candidate values: {e}")

        sorted_vals = sorted(numeric_values, reverse=from_biggest)
        if ordinal < 1 or ordinal > len(sorted_vals):
            raise ENToMathError(f"kth_biggest ordinal {ordinal} out of range for {len(sorted_vals)} values")
        target_val = sorted_vals[ordinal - 1]

        # Find the original token slice corresponding to the chosen value.
        # We look for the first candidate whose numeric value matches.
        target_group_tokens: Optional[List[GyanDSLToken]] = None
        for expr_g, g_tokens in zip(value_exprs, groups):
            try:
                if float(sympy.N(expr_g)) == target_val:
                    target_group_tokens = g_tokens
                    break
            except Exception:
                continue
        if target_group_tokens is None:
            # Fallback: use the first group; this should be rare.
            target_group_tokens = groups[0]

        math_tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
        for g in groups:
            math_tokens += g
        math_tokens += int_to_tokens(ordinal)
        math_tokens.append(GyanDSLToken.KTH_LARGEST)
        math_tokens += target_group_tokens
        math_tokens.append(GyanDSLToken.EOS)
        return ENToMathResult(
            module=module,
            en_token_ids=en_token_ids,
            math_token_ids=_tokens_to_ids(math_tokens),
        )

    # ------------------------------------------------------------------
    # closest family
    # DeepMind:
    #   comparison__closest:
    #       BOS <v1> ... <vn> <target> CLOSEST_TO <answer> EOS
    #   comparison__closest_composed:
    #       BOS CLOSEST_TO <answer> EOS
    #
    # EN-DSL encodes:
    #   - group of candidate values in EN_GROUP
    #   - target expression after EN_ENTITY EN_ATTR (for comparison__closest)
    #   - singleton group with answer value (for *_composed).
    # ------------------------------------------------------------------
    if module in {"comparison__closest", "comparison__closest_composed"}:
        if module == "comparison__closest_composed":
            # Singleton answer in EN_GROUP.
            if len(groups) != 1:
                raise ENToMathError("closest_composed EN-DSL must encode a single answer value in EN_GROUP")
            ans_tokens = groups[0]
            math_tokens = [GyanDSLToken.BOS, GyanDSLToken.CLOSEST_TO]
            math_tokens += ans_tokens
            math_tokens.append(GyanDSLToken.EOS)
            return ENToMathResult(
                module=module,
                en_token_ids=en_token_ids,
                math_token_ids=_tokens_to_ids(math_tokens),
            )

        # comparison__closest: candidates + explicit target.
        target_expr: Optional[sympy.Expr] = None
        if GyanDSLToken.EN_ENTITY in toks:
            ent_idx = toks.index(GyanDSLToken.EN_ENTITY)
            if ent_idx + 1 < len(toks) and toks[ent_idx + 1] == GyanDSLToken.EN_ATTR:
                target_slice: List[GyanDSLToken] = []
                j = ent_idx + 2
                while j < len(toks) and toks[j] != GyanDSLToken.EOS and not toks[j].name.startswith("EN_"):
                    target_slice.append(toks[j])
                    j += 1
                if target_slice:
                    target_expr, _ = _tokens_to_expr(target_slice)
        if target_expr is None:
            raise ENToMathError("closest EN-DSL missing target expression")

        # Compute closest candidate numerically, then mirror generator tokens.
        try:
            target_val = float(sympy.N(target_expr))
            candidate_vals = [float(sympy.N(e)) for e in value_exprs]
        except Exception as e:
            raise ENToMathError(f"Failed to evaluate closest values: {e}")

        min_diff = None
        best_tokens: Optional[List[GyanDSLToken]] = None
        for val, g_tokens in zip(candidate_vals, groups):
            diff = abs(val - target_val)
            if min_diff is None or diff < min_diff:
                min_diff = diff
                best_tokens = g_tokens
        if best_tokens is None:
            raise ENToMathError("closest EN-DSL had no candidate values")

        math_tokens = [GyanDSLToken.BOS]
        for g in groups:
            math_tokens += g
        math_tokens += expr_to_tokens(target_expr, {})
        math_tokens.append(GyanDSLToken.CLOSEST_TO)
        math_tokens += best_tokens
        math_tokens.append(GyanDSLToken.EOS)
        return ENToMathResult(
            module=module,
            en_token_ids=en_token_ids,
            math_token_ids=_tokens_to_ids(math_tokens),
        )

    # ------------------------------------------------------------------
    # sort family
    # DeepMind:
    #   comparison__sort:
    #       BOS <v1> ... <vn> SORT <sorted_vals> EOS
    #   comparison__sort_composed:
    #       BOS SORT <values> EOS
    #
    # EN-DSL encodes:
    #   - group of values in EN_GROUP,
    #   - for comparison__sort, the direction code via
    #       EN_STATE EN_ATTR <dir_code>   (0 -> ascending, 1 -> descending).
    # ------------------------------------------------------------------
    if module in {"comparison__sort", "comparison__sort_composed"}:
        if module == "comparison__sort_composed":
            # Answer-only: sorted values are encoded in EN_GROUP in order.
            math_tokens = [GyanDSLToken.BOS, GyanDSLToken.SORT]
            for g in groups:
                math_tokens += g
            math_tokens.append(GyanDSLToken.EOS)
            return ENToMathResult(
                module=module,
                en_token_ids=en_token_ids,
                math_token_ids=_tokens_to_ids(math_tokens),
            )

        # comparison__sort: unsorted values + direction.
        # Extract direction code after EN_STATE EN_ATTR.
        try:
            state_idx = toks.index(GyanDSLToken.EN_STATE)
        except ValueError:
            raise ENToMathError("comparison__sort EN-DSL missing EN_STATE for direction")
        if state_idx + 1 >= len(toks) or toks[state_idx + 1] != GyanDSLToken.EN_ATTR:
            raise ENToMathError("comparison__sort EN-DSL missing EN_ATTR after EN_STATE")

        dir_tokens: List[GyanDSLToken] = []
        j = state_idx + 2
        while j < len(toks) and toks[j] != GyanDSLToken.EOS and not toks[j].name.startswith("EN_"):
            dir_tokens.append(toks[j])
            j += 1
        if not dir_tokens:
            raise ENToMathError("comparison__sort EN-DSL has empty direction code")
        dir_code = _int_tokens_to_int(dir_tokens)
        if dir_code not in (0, 1):
            raise ENToMathError(f"Unknown sort direction code in EN-DSL: {dir_code}")
        ascending = dir_code == 0

        # Sort values numerically based on direction.
        paired = list(zip(value_ints, groups))
        paired_sorted = sorted(paired, key=lambda x: x[0], reverse=not ascending)

        math_tokens = [GyanDSLToken.BOS]
        for _, g in paired:
            math_tokens += g
        math_tokens.append(GyanDSLToken.SORT)
        for _, g in paired_sorted:
            math_tokens += g
        math_tokens.append(GyanDSLToken.EOS)
        return ENToMathResult(
            module=module,
            en_token_ids=en_token_ids,
            math_token_ids=_tokens_to_ids(math_tokens),
        )

    raise ENToMathError(f"Module not supported by en_to_math_for_comparison: {module}")


# ---------------------------------------------------------------------------
# Measurement family bridge
# ---------------------------------------------------------------------------


def en_to_math_for_measurement(module: str, en_token_ids: List[int]) -> ENToMathResult:
    """
    Translate EN-DSL to math-DSL for Measurement modules.
    
    Both measurement modules use expression-query pattern.
    """
    toks = _ids_to_tokens(en_token_ids)

    if GyanDSLToken.EN_ATTR in toks:
        expr_tokens = _extract_expr_after_attr(toks)
        expr, var_map = _tokens_to_expr(expr_tokens)

        math_tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
        math_tokens += expr_to_tokens(expr, var_map)
        math_tokens.append(GyanDSLToken.EVAL_EXPR)
        
        try:
            result = expr.evalf()
            if result == int(result):
                math_tokens += int_to_tokens(int(result))
            else:
                math_tokens += expr_to_tokens(sympy.Rational(result).limit_denominator(10000), var_map)
        except Exception:
            pass
        
        math_tokens.append(GyanDSLToken.EQ_CMP)
        math_tokens.append(GyanDSLToken.EOS)
        return ENToMathResult(module=module, en_token_ids=en_token_ids, math_token_ids=_tokens_to_ids(math_tokens))

    raise ENToMathError(f"Module not supported by en_to_math_for_measurement: {module}")


# ---------------------------------------------------------------------------
# Calculus family bridge
# ---------------------------------------------------------------------------


def en_to_math_for_calculus(module: str, en_token_ids: List[int]) -> ENToMathResult:
    """
    Translate EN-DSL to math-DSL for Calculus modules.
    
    Uses expression-query pattern with the derivative result expression.
    """
    toks = _ids_to_tokens(en_token_ids)

    if GyanDSLToken.EN_ATTR in toks:
        expr_tokens = _extract_expr_after_attr(toks)
        expr, var_map = _tokens_to_expr(expr_tokens)

        math_tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
        math_tokens += expr_to_tokens(expr, var_map)
        math_tokens.append(GyanDSLToken.DIFF)
        math_tokens += expr_to_tokens(expr, var_map)
        math_tokens.append(GyanDSLToken.EQ_CMP)
        math_tokens.append(GyanDSLToken.EOS)
        return ENToMathResult(module=module, en_token_ids=en_token_ids, math_token_ids=_tokens_to_ids(math_tokens))

    raise ENToMathError(f"Module not supported by en_to_math_for_calculus: {module}")


# ---------------------------------------------------------------------------
# Polynomials family bridge
# ---------------------------------------------------------------------------


def en_to_math_for_polynomials(module: str, en_token_ids: List[int]) -> ENToMathResult:
    """
    Translate EN-DSL to math-DSL for Polynomials modules.
    
    Uses expression-query pattern with the polynomial expression.
    """
    toks = _ids_to_tokens(en_token_ids)

    if GyanDSLToken.EN_ATTR not in toks:
        raise ENToMathError("Polynomial EN-DSL missing EN_ATTR payload")

    # Generic expression payload after EN_ATTR (used by many polynomial modules).
    expr_tokens = _extract_expr_after_attr(toks)
    expr, var_map = _tokens_to_expr(expr_tokens)

    math_tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]

    # ------------------------------------------------------------------
    # polynomials__expand
    # DeepMind:
    #   BOS <poly_expr> EXPAND <expanded_expr> EQ_CMP EOS
    # EN-DSL encodes <poly_expr> in EN_ATTR.
    # ------------------------------------------------------------------
    if module == "polynomials__expand":
        src_expr = expr
        var_map_src: Dict[sympy.Symbol, int] = {}
        for s in sorted(src_expr.free_symbols, key=lambda s: s.name):
            if s not in var_map_src:
                var_map_src[s] = len(var_map_src)
        expanded = sympy.expand(src_expr)

        math_tokens += expr_to_tokens(src_expr, var_map_src)
        math_tokens.append(GyanDSLToken.EXPAND)
        math_tokens += expr_to_tokens(expanded, var_map_src)
        math_tokens.append(GyanDSLToken.EQ_CMP)
        math_tokens.append(GyanDSLToken.EOS)
        return ENToMathResult(
            module=module,
            en_token_ids=en_token_ids,
            math_token_ids=_tokens_to_ids(math_tokens),
        )

    # ------------------------------------------------------------------
    # polynomials__add
    # DeepMind:
    #   BOS ADD_POLY <answer_poly> EOS
    # EN-DSL encodes the resulting polynomial expression.
    # ------------------------------------------------------------------
    if module == "polynomials__add":
        ans_expr = expr
        var_map_ans: Dict[sympy.Symbol, int] = {}
        for s in sorted(ans_expr.free_symbols, key=lambda s: s.name):
            if s not in var_map_ans:
                var_map_ans[s] = len(var_map_ans)

        math_tokens.append(GyanDSLToken.ADD_POLY)
        math_tokens += expr_to_tokens(ans_expr, var_map_ans)
        math_tokens.append(GyanDSLToken.EOS)
        return ENToMathResult(
            module=module,
            en_token_ids=en_token_ids,
            math_token_ids=_tokens_to_ids(math_tokens),
        )

    # ------------------------------------------------------------------
    # polynomials__collect
    # DeepMind:
    #   BOS <expr> COLLECT <answer_expr> EQ_CMP EOS
    # EN-DSL encodes the *answer* expression; we reconstruct a canonical
    # "source" as the expanded form of the answer so that the DSL program
    # matches the generator.
    # ------------------------------------------------------------------
    if module == "polynomials__collect":
        ans_expr = expr
        src_expr = sympy.expand(ans_expr)

        var_map_src: Dict[sympy.Symbol, int] = {}
        for s in sorted(src_expr.free_symbols | ans_expr.free_symbols, key=lambda s: s.name):
            if s not in var_map_src:
                var_map_src[s] = len(var_map_src)

        math_tokens += expr_to_tokens(src_expr, var_map_src)
        math_tokens.append(GyanDSLToken.COLLECT)
        math_tokens += expr_to_tokens(ans_expr, var_map_src)
        math_tokens.append(GyanDSLToken.EQ_CMP)
        math_tokens.append(GyanDSLToken.EOS)
        return ENToMathResult(
            module=module,
            en_token_ids=en_token_ids,
            math_token_ids=_tokens_to_ids(math_tokens),
        )

    # ------------------------------------------------------------------
    # polynomials__simplify_power
    # DeepMind generator encodes:
    #   BOS SIMPLIFY_POWER <answer_or_var^exp> EOS
    # EN-DSL stores the simplified answer expression; we mirror that form:
    #   BOS SIMPLIFY_POWER <answer_expr> EOS
    # ------------------------------------------------------------------
    if module == "polynomials__simplify_power":
        ans_expr = expr
        var_map_ans: Dict[sympy.Symbol, int] = {}
        for s in sorted(ans_expr.free_symbols, key=lambda s: s.name):
            if s not in var_map_ans:
                var_map_ans[s] = len(var_map_ans)

        math_tokens.append(GyanDSLToken.SIMPLIFY_POWER)
        math_tokens += expr_to_tokens(ans_expr, var_map_ans)
        math_tokens.append(GyanDSLToken.EOS)
        return ENToMathResult(
            module=module,
            en_token_ids=en_token_ids,
            math_token_ids=_tokens_to_ids(math_tokens),
        )

    # ------------------------------------------------------------------
    # polynomials__coefficient_named
    # DeepMind:
    #   BOS <expr> COEFF_AT_POWER <power> <coeff_value> EQ_CMP EOS
    #
    # EN-DSL (via build_en_tokens_for_coefficient_named):
    #   BOS EN_QUERY EN_Q_ATTR EN_ENTITY
    #       EN_ATTR   <expr_tokens>
    #       EN_AMOUNT <power_tokens>
    #       EN_TOTAL EN_AMOUNT <coeff_tokens>
    #   EOS
    # ------------------------------------------------------------------
    if module == "polynomials__coefficient_named":
        # expr is the original polynomial expression (from EN_ATTR).
        poly_expr = expr

        # Extract target power from EN_AMOUNT immediately after EN_ATTR.
        try:
            attr_idx = toks.index(GyanDSLToken.EN_ATTR)
        except ValueError:
            raise ENToMathError("coefficient_named EN-DSL missing EN_ATTR")
        power_tokens: List[GyanDSLToken] = []
        i = attr_idx + 1 + len(expr_tokens)
        # Skip over the expr_tokens we just consumed.
        while i < len(toks) and toks[i] != GyanDSLToken.EN_AMOUNT:
            i += 1
        if i >= len(toks) or toks[i] != GyanDSLToken.EN_AMOUNT:
            raise ENToMathError("coefficient_named EN-DSL missing EN_AMOUNT for power")
        i += 1
        while i < len(toks) and not toks[i].name.startswith("EN_") and toks[i] != GyanDSLToken.EOS:
            power_tokens.append(toks[i])
            i += 1
        if not power_tokens:
            raise ENToMathError("coefficient_named EN-DSL has empty power payload")
        target_power = _int_tokens_to_int(power_tokens)

        # Extract coefficient value after EN_TOTAL EN_AMOUNT.
        try:
            total_idx = toks.index(GyanDSLToken.EN_TOTAL)
        except ValueError:
            raise ENToMathError("coefficient_named EN-DSL missing EN_TOTAL for coefficient")
        if total_idx + 1 >= len(toks) or toks[total_idx + 1] != GyanDSLToken.EN_AMOUNT:
            raise ENToMathError("coefficient_named EN-DSL missing EN_AMOUNT after EN_TOTAL")
        coeff_tokens: List[GyanDSLToken] = []
        j = total_idx + 2
        while j < len(toks) and toks[j] != GyanDSLToken.EOS and not toks[j].name.startswith("EN_"):
            coeff_tokens.append(toks[j])
            j += 1
        if not coeff_tokens:
            raise ENToMathError("coefficient_named EN-DSL has empty coefficient payload")

        # Coefficient value can be encoded either as:
        #   - math-DSL INT_* tokens (legacy), or
        #   - EN_DIGIT_* tokens (new digit-style answers).
        if all(t.name.startswith("EN_DIGIT_") for t in coeff_tokens):
            coeff_str = _en_digit_tokens_to_str(coeff_tokens)
            try:
                coeff_expr = sympy.Integer(coeff_str)
                coeff_var_map: Dict[sympy.Symbol, int] = {}
            except Exception as e:
                raise ENToMathError(f"Failed to parse EN_DIGIT coefficient '{coeff_str}': {e}")
        else:
            coeff_expr, coeff_var_map = _tokens_to_expr(coeff_tokens)

        # Build var_map over polynomial expression: single main variable at index 0
        # to mirror build_coefficient_at_power_tokens.
        var_map_poly: Dict[sympy.Symbol, int] = {}
        free_syms = sorted(list(poly_expr.free_symbols), key=lambda s: s.name)
        if free_syms:
            main_var = free_syms[0]
            var_map_poly[main_var] = 0

        math_tokens += expr_to_tokens(poly_expr, var_map_poly)
        # Insert REAL_VAR_0 for the variable argument, then power.
        math_tokens.append(GyanDSLToken.REAL_VAR_0)
        math_tokens += int_to_tokens(target_power)
        math_tokens.append(GyanDSLToken.COEFF_AT_POWER)
        # Coefficient value may or may not use the same variable; encode with
        # a separate var_map if needed.
        math_tokens += expr_to_tokens(coeff_expr, coeff_var_map or {})
        math_tokens.append(GyanDSLToken.EQ_CMP)
        math_tokens.append(GyanDSLToken.EOS)
        return ENToMathResult(
            module=module,
            en_token_ids=en_token_ids,
            math_token_ids=_tokens_to_ids(math_tokens),
        )

    # ------------------------------------------------------------------
    # polynomials__compose
    # DeepMind (generate_polynomials_compose_structured):
    #   BOS <f_expr> <g_expr> COMPOSE <answer_poly> EQ_CMP EOS
    #
    # EN-DSL stores the composed answer polynomial expression only; we
    # reconstruct f and g canonically as:
    #   g(x) = x,  f(x) = answer_poly(x)
    # ------------------------------------------------------------------
    if module == "polynomials__compose":
        ans_expr = expr

        # Choose a dummy variable consistent with expr_to_tokens: we just
        # need a single REAL_VAR_0.
        free = sorted(list(ans_expr.free_symbols), key=lambda s: s.name)
        if free:
            var = free[0]
        else:
            var = sympy.Symbol("v0")
        var_map_fg: Dict[sympy.Symbol, int] = {var: 0}

        f_expr = ans_expr
        g_expr = var

        math_tokens += expr_to_tokens(f_expr, var_map_fg)
        math_tokens += expr_to_tokens(g_expr, var_map_fg)
        math_tokens.append(GyanDSLToken.COMPOSE)
        math_tokens += expr_to_tokens(ans_expr, var_map_fg)
        math_tokens.append(GyanDSLToken.EQ_CMP)
        math_tokens.append(GyanDSLToken.EOS)
        return ENToMathResult(
            module=module,
            en_token_ids=en_token_ids,
            math_token_ids=_tokens_to_ids(math_tokens),
        )

    # ------------------------------------------------------------------
    # polynomials__evaluate / polynomials__evaluate_composed
    #
    # polynomials__evaluate (structured):
    #   BOS <poly_expr> <x_value> EVAL_EXPR <answer_int> EQ_CMP EOS
    #
    # EN-DSL (build_en_tokens_for_polynomial_evaluate):
    #   BOS EN_QUERY EN_Q_ATTR EN_ENTITY
    #       EN_ATTR   <poly_expr_tokens>
    #       EN_AMOUNT <eval_value_tokens>
    #   EOS
    #
    # polynomials__evaluate_composed:
    #   BOS <eval_point> EVAL_EXPR <answer_int> EQ_CMP EOS
    #
    # EN-DSL (build_en_tokens_for_polynomial_evaluate_composed):
    #   BOS EN_QUERY EN_Q_ATTR EN_ENTITY
    #       EN_ATTR   <answer_tokens>
    #       EN_AMOUNT <eval_point_tokens>
    #   EOS
    # ------------------------------------------------------------------
    if module in {"polynomials__evaluate", "polynomials__evaluate_composed"}:
        # Extract the integer payload after EN_AMOUNT. This can be encoded
        # either as math-DSL INT_* tokens (legacy) or as EN_DIGIT_* tokens.
        try:
            amt_idx = toks.index(GyanDSLToken.EN_AMOUNT)
        except ValueError:
            raise ENToMathError("polynomial evaluate EN-DSL missing EN_AMOUNT payload")
        amt_tokens: List[GyanDSLToken] = []
        k = amt_idx + 1
        while k < len(toks) and toks[k] != GyanDSLToken.EOS and not toks[k].name.startswith("EN_"):
            amt_tokens.append(toks[k])
            k += 1
        if not amt_tokens:
            raise ENToMathError("polynomial evaluate EN-DSL has empty EN_AMOUNT payload")

        if all(t.name.startswith("EN_DIGIT_") for t in amt_tokens):
            num_str = _en_digit_tokens_to_str(amt_tokens)
            try:
                eval_or_answer = int(num_str)
            except Exception as e:
                raise ENToMathError(f"Failed to parse digit-style EN_AMOUNT '{num_str}' as int: {e}")
        else:
            eval_or_answer = _int_tokens_to_int(amt_tokens)

        if module == "polynomials__evaluate_composed":
            # Here expr encodes the integer answer (from EN_ATTR); EN_AMOUNT
            # encodes the evaluation point.
            try:
                ans_int = int(sympy.N(expr))
            except Exception as e:
                raise ENToMathError(f"Failed to interpret evaluate_composed answer as int: {e}")
            eval_point = eval_or_answer

            math_tokens += int_to_tokens(eval_point)
            math_tokens.append(GyanDSLToken.EVAL_EXPR)
            math_tokens += int_to_tokens(ans_int)
            math_tokens.append(GyanDSLToken.EQ_CMP)
            math_tokens.append(GyanDSLToken.EOS)
            return ENToMathResult(
                module=module,
                en_token_ids=en_token_ids,
                math_token_ids=_tokens_to_ids(math_tokens),
            )

        # polynomials__evaluate: expr is the polynomial, EN_AMOUNT is eval point.
        poly_expr = expr
        eval_value = eval_or_answer

        # Build var_map for polynomial variable(s).
        var_map_poly: Dict[sympy.Symbol, int] = {}
        for s in sorted(poly_expr.free_symbols, key=lambda s: s.name):
            if s not in var_map_poly:
                var_map_poly[s] = len(var_map_poly)

        # Compute the integer answer by substitution.
        if not poly_expr.free_symbols:
            # Constant polynomial; answer is the constant.
            answer_val = int(sympy.N(poly_expr))
        else:
            # Use the first variable as the evaluation variable.
            var = sorted(list(poly_expr.free_symbols), key=lambda s: s.name)[0]
            answer_expr = poly_expr.subs(var, eval_value)
            answer_val = int(sympy.N(answer_expr))

        math_tokens += expr_to_tokens(poly_expr, var_map_poly)
        math_tokens += int_to_tokens(eval_value)
        math_tokens.append(GyanDSLToken.EVAL_EXPR)
        math_tokens += int_to_tokens(answer_val)
        math_tokens.append(GyanDSLToken.EQ_CMP)
        math_tokens.append(GyanDSLToken.EOS)
        return ENToMathResult(
            module=module,
            en_token_ids=en_token_ids,
            math_token_ids=_tokens_to_ids(math_tokens),
        )

    raise ENToMathError(f"Module not supported by en_to_math_for_polynomials: {module}")


# ---------------------------------------------------------------------------
# Probability family bridge
# ---------------------------------------------------------------------------


def en_to_math_for_probability(module: str, en_token_ids: List[int]) -> ENToMathResult:
    """
    Translate EN-DSL to math-DSL for Probability modules.
    
    Uses expression-query pattern with the probability fraction.
    """
    toks = _ids_to_tokens(en_token_ids)

    if GyanDSLToken.EN_ATTR in toks:
        expr_tokens = _extract_expr_after_attr(toks)
        expr, var_map = _tokens_to_expr(expr_tokens)

        math_tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
        math_tokens.append(GyanDSLToken.PROBABILITY)
        math_tokens += expr_to_tokens(expr, var_map)
        math_tokens.append(GyanDSLToken.EOS)
        return ENToMathResult(module=module, en_token_ids=en_token_ids, math_token_ids=_tokens_to_ids(math_tokens))

    raise ENToMathError(f"Module not supported by en_to_math_for_probability: {module}")


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GSM8K family bridge
# ---------------------------------------------------------------------------


def en_to_math_for_gsm8k(module: str, en_token_ids: List[int]) -> ENToMathResult:
    """
    Translate EN-DSL to math-DSL for GSM8K word problems.
    
    GSM8K EN-DSL structure:
        BOS
          (EN_EVT_INIT | EN_EVT_GAIN | EN_EVT_LOSS | EN_EVT_RATE) EN_AMOUNT <result_tokens>
          ...
          EN_QUERY EN_Q_HOW_MANY EN_TOTAL EN_AMOUNT <final_answer_tokens>
        EOS
    
    We reconstruct a math-DSL chain-of-thought:
        BOS <expr_1> <result_1> EQ_CMP ... <final_answer> EOS
    """
    toks = _ids_to_tokens(en_token_ids)
    
    # Parse the EN-DSL structure to extract step results and final answer
    step_results: List[sympy.Expr] = []
    final_answer: Optional[sympy.Expr] = None
    
    i = 1  # Skip BOS
    while i < len(toks):
        tok = toks[i]
        
        if tok == GyanDSLToken.EOS:
            break
        
        # Check for event markers
        if tok in (GyanDSLToken.EN_EVT_INIT, GyanDSLToken.EN_EVT_GAIN,
                   GyanDSLToken.EN_EVT_LOSS, GyanDSLToken.EN_EVT_RATE):
            # Next should be EN_AMOUNT followed by result tokens
            if i + 1 < len(toks) and toks[i + 1] == GyanDSLToken.EN_AMOUNT:
                i += 2
                # Collect tokens until next event or EN_QUERY
                result_tokens: List[GyanDSLToken] = []
                while i < len(toks) and toks[i] not in (
                    GyanDSLToken.EN_EVT_INIT, GyanDSLToken.EN_EVT_GAIN,
                    GyanDSLToken.EN_EVT_LOSS, GyanDSLToken.EN_EVT_RATE,
                    GyanDSLToken.EN_QUERY, GyanDSLToken.EOS
                ):
                    result_tokens.append(toks[i])
                    i += 1
                
                if result_tokens:
                    try:
                        expr, _ = _tokens_to_expr(result_tokens)
                        step_results.append(expr)
                    except Exception:
                        pass
                continue
        
        # Check for query (final answer follows)
        if tok == GyanDSLToken.EN_QUERY:
            # Skip EN_Q_HOW_MANY EN_TOTAL
            j = i + 1
            while j < len(toks) and toks[j] in (
                GyanDSLToken.EN_Q_HOW_MANY, GyanDSLToken.EN_TOTAL
            ):
                j += 1
            
            # Next should be EN_AMOUNT followed by final answer
            if j < len(toks) and toks[j] == GyanDSLToken.EN_AMOUNT:
                j += 1
                answer_tokens: List[GyanDSLToken] = []
                while j < len(toks) and toks[j] != GyanDSLToken.EOS:
                    answer_tokens.append(toks[j])
                    j += 1

                if answer_tokens:
                    try:
                        # Support both legacy INT_* answer encoding and the new
                        # EN_DIGIT_* digit-style answers.
                        if all(t.name.startswith("EN_DIGIT_") for t in answer_tokens):
                            num_str = _en_digit_tokens_to_str(answer_tokens)
                            final_answer = sympy.Integer(num_str)
                        else:
                            final_answer, _ = _tokens_to_expr(answer_tokens)
                    except Exception:
                        pass
            break
        
        i += 1
    
    if final_answer is None:
        raise ENToMathError("Could not extract final answer from GSM8K EN-DSL")
    
    # Build math-DSL: BOS <step1> EQ_CMP <step2> EQ_CMP ... <final_answer> EOS
    math_tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]
    
    for result in step_results:
        # For each step, we just output the result (since we don't have the
        # original expression in EN-DSL)
        math_tokens += expr_to_tokens(result, {})
        math_tokens.append(GyanDSLToken.EQ_CMP)
    
    # Final answer
    math_tokens += expr_to_tokens(final_answer, {})
    math_tokens.append(GyanDSLToken.EOS)
    
    return ENToMathResult(module=module, en_token_ids=en_token_ids, math_token_ids=_tokens_to_ids(math_tokens))


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------


def en_to_math(module: str, en_token_ids: List[int]) -> ENToMathResult:
    """
    Top-level dispatcher for EN-DSL → math-DSL translation.
    
    Routes to family-specific handlers based on module prefix.
    """
    if module.startswith("numbers__"):
        return en_to_math_for_numbers(module, en_token_ids)

    if module.startswith("algebra__"):
        return en_to_math_for_algebra(module, en_token_ids)

    if module.startswith("arithmetic__"):
        return en_to_math_for_arithmetic(module, en_token_ids)

    if module.startswith("comparison__"):
        return en_to_math_for_comparison(module, en_token_ids)

    if module.startswith("measurement__"):
        return en_to_math_for_measurement(module, en_token_ids)

    if module.startswith("calculus__"):
        return en_to_math_for_calculus(module, en_token_ids)

    if module.startswith("polynomials__"):
        return en_to_math_for_polynomials(module, en_token_ids)

    if module.startswith("probability__"):
        return en_to_math_for_probability(module, en_token_ids)

    if module == "gsm8k":
        return en_to_math_for_gsm8k(module, en_token_ids)

    raise ENToMathError(f"EN→Math bridge not yet implemented for module: {module}")
