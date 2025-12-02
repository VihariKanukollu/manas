"""
Gyan DSL token definitions.

This file is the single source of truth for:
- All DSL tokens in the Gyan vocabulary
- Their integer IDs (dense in [0, vocab_size))
- Metadata for type-based masking

The vocabulary covers:
- Structural tokens (PAD, BOS, EOS)
- Variables (REAL_VAR_0..N, BOOL_VAR_0..M)
- Constants (integers, special values)
- Arithmetic operations (ADD, SUB, MUL, DIV, etc.)
- Comparison operations (LT, LE, GT, GE, EQ, NE)
- Boolean logic (AND, OR, NOT, IMPLIES, IFF)
- Math DSL (eq, is_solution, simplify, expand, factor, etc.)
- Logic DSL (propositional, inference rules)
- CSP primitives (DOMAIN, ASSIGN, PROPAGATE, etc.)
- ARC primitives (grid ops, object ops, transformations)
"""

from enum import Enum
from typing import Dict


# ---------------------------------------------------------------------------
# Configurable variable counts
# ---------------------------------------------------------------------------

NUM_REAL_VARS: int = 32   # REAL_VAR_0 .. REAL_VAR_31
NUM_BOOL_VARS: int = 16   # BOOL_VAR_0 .. BOOL_VAR_15
NUM_INT_CONSTS: int = 100 # INT_0 .. INT_99 (covers most math problems)


def _build_gyan_dsl_token_enum() -> type[Enum]:
    """
    Construct the GyanDSLToken Enum programmatically.

    We assign explicit integer IDs so that:
      - token.value is always the stable ID
      - IDs are dense in [0, vocab_size)
      - Expanding variable counts just increases vocab_size
    """

    members: Dict[str, int] = {}
    idx = 0

    # -------------------------------------------------------------------------
    # 1. Structural tokens
    # -------------------------------------------------------------------------
    for name in ["PAD", "BOS", "EOS"]:
        members[name] = idx
        idx += 1

    # -------------------------------------------------------------------------
    # 2. Variables
    # -------------------------------------------------------------------------
    # REAL variables (for numeric values)
    for i in range(NUM_REAL_VARS):
        members[f"REAL_VAR_{i}"] = idx
        idx += 1

    # BOOL variables (for boolean values)
    for i in range(NUM_BOOL_VARS):
        members[f"BOOL_VAR_{i}"] = idx
        idx += 1

    # -------------------------------------------------------------------------
    # 3. Constants
    # -------------------------------------------------------------------------
    # Integer constants 0..99
    for i in range(NUM_INT_CONSTS):
        members[f"INT_{i}"] = idx
        idx += 1

    # Special numeric constants
    for name in [
        "INT_NEG1",      # -1
        "INT_NEG2",      # -2
        "INT_NEG10",     # -10
        "INT_NEG100",    # -100
        "REAL_HALF",     # 0.5
        "REAL_PI",       # π
        "REAL_E",        # e
    ]:
        members[name] = idx
        idx += 1

    # Boolean constants
    for name in ["BOOL_TRUE", "BOOL_FALSE"]:
        members[name] = idx
        idx += 1

    # -------------------------------------------------------------------------
    # 4. Arithmetic operations (REAL, REAL -> REAL) or (REAL -> REAL)
    # -------------------------------------------------------------------------
    for name in [
        "ADD",      # a + b
        "SUB",      # a - b
        "MUL",      # a * b
        "DIV",      # a / b
        "MOD",      # a % b
        "POW",      # a ^ b
        "NEG",      # -a (unary)
        "ABS",      # |a| (unary)
        "SQRT",     # √a (unary)
        "SQUARE",   # a² (unary)
    ]:
        members[name] = idx
        idx += 1

    # -------------------------------------------------------------------------
    # 5. Comparison operations (REAL, REAL -> BOOL)
    # -------------------------------------------------------------------------
    for name in ["LT", "LE", "GT", "GE", "EQ_CMP", "NE"]:
        members[name] = idx
        idx += 1

    # -------------------------------------------------------------------------
    # 6. Boolean logic (BOOL, BOOL -> BOOL) or (BOOL -> BOOL)
    # -------------------------------------------------------------------------
    for name in ["AND", "OR", "NOT", "IMPLIES", "IFF"]:
        members[name] = idx
        idx += 1

    # -------------------------------------------------------------------------
    # 7. Math DSL - Equations and expressions
    # -------------------------------------------------------------------------
    for name in [
        # Equation construction
        "EQ",                 # eq(lhs, rhs) -> Equation
        "INEQ_LT",            # ineq("<", lhs, rhs) -> Inequality
        "INEQ_LE",            # ineq("<=", lhs, rhs) -> Inequality
        "INEQ_GT",            # ineq(">", lhs, rhs) -> Inequality
        "INEQ_GE",            # ineq(">=", lhs, rhs) -> Inequality

        # Equation manipulation
        "ADD_BOTH_SIDES",     # add_both_sides(eq, delta) -> Equation
        "SUB_BOTH_SIDES",     # sub_both_sides(eq, delta) -> Equation
        "MUL_BOTH_SIDES",     # mul_both_sides(eq, factor) -> Equation
        "DIV_BOTH_SIDES",     # div_both_sides(eq, factor) -> Equation
        "SUBSTITUTE",         # substitute(eq, var, expr) -> Equation

        # Expression / polynomial manipulation
        "EXPAND",           # expand_expr(expr) -> Expr
        "FACTOR",           # factor_expr(expr) -> Expr
        "SIMPLIFY",         # simplify_expr(expr) -> Expr
        "COLLECT",          # collect(expr) -> Expr (polynomial collect)
        "SIMPLIFY_POWER",   # simplify_power(expr) -> Expr (power simplification)
        "ADD_POLY",         # add polynomials -> Expr

        # Verification
        "IS_SOLUTION",  # is_solution(eq, var, value) -> Bool

        # Numeric operations
        "GCD",          # gcd(a, b) -> Int
        "LCM",          # lcm(a, b) -> Int
        "FACTORIAL",    # factorial(n) -> Int
        "FACTORINT",    # factorint(n) -> Dict (returns prime factors)

        # Polynomial structure
        "COEFF_AT_POWER",     # coefficient of x^k in a polynomial

        # Number theory (for DeepMind numbers module)
        "IS_PRIME",           # is_prime(n) -> Bool
        "IS_FACTOR",          # is_factor(a, b) -> Bool (does a divide b?)
        "DIV_REMAINDER",      # div_remainder(a, b) -> Int (a % b)
        "PRIME_FACTORS",      # prime_factors(n) -> List[Int]
        "NEXT_PRIME",         # next_prime(n) -> Int
        "NTH_PRIME",          # nth_prime(n) -> Int

        # Rounding / place value / roots
        "ROUND",              # round(x, decimals) -> Real
        "FLOOR",              # floor(x) -> Int
        "CEIL",               # ceil(x) -> Int
        "PLACE_VALUE",        # place_value(n, place) -> Int (digit at place)
        "TRUNC",              # truncate towards zero
        "NEAREST_ROOT",       # nearest_integer_root(n, k) -> Int (round(n^(1/k)))

        # Base conversion
        "TO_BASE",            # to_base(n, base) -> List[Int]
        "FROM_BASE",          # from_base(digits, base) -> Int

        # Sequences (for DeepMind algebra sequences)
        "SEQ_NTH",            # seq_nth(seq, n) -> Int
        "SEQ_NEXT",           # seq_next(seq) -> Int
        "ARITH_SEQ",          # arithmetic_sequence(start, diff) -> Seq
        "GEOM_SEQ",           # geometric_sequence(start, ratio) -> Seq
        "POLY_SEQ",           # polynomial_sequence(coeffs) -> Seq

        # Calculus
        "DIFF",               # differentiate(expr, var) -> Expr
        "DIFF_N",             # nth derivative
        "INTEGRATE",          # integrate(expr, var) -> Expr

        # Systems of equations (for linear_2d)
        "SOLVE_SYSTEM",       # solve_system([eq1, eq2], [var1, var2]) -> Dict

        # Probability / combinatorics
        "PERMUTATION",        # P(n, k) = n! / (n-k)!
        "COMBINATION",        # C(n, k) = n! / (k!(n-k)!)
        "PROBABILITY",        # probability(event, space) -> Rational
    ]:
        members[name] = idx
        idx += 1

    # -------------------------------------------------------------------------
    # 8. Logic DSL - Propositional logic
    # -------------------------------------------------------------------------
    for name in [
        # Propositional constructors
        "PROP_VAR",     # PROP_VAR(name) -> Prop
        "PROP_TRUE",    # constant True
        "PROP_FALSE",   # constant False
        "PROP_AND",     # AND(a, b) -> Prop
        "PROP_OR",      # OR(a, b) -> Prop
        "PROP_NOT",     # NOT(a) -> Prop
        "PROP_IMPLIES", # IMPLIES(a, b) -> Prop
        "PROP_IFF",     # IFF(a, b) -> Prop

        # Evaluation and simplification
        "EVAL_PROP",    # EVAL_PROP(prop, env) -> Bool
        "SIMPLIFY_PROP",# SIMPLIFY_PROP(prop) -> Prop
    ]:
        members[name] = idx
        idx += 1

    # -------------------------------------------------------------------------
    # 9. Logic DSL - Inference rules
    # -------------------------------------------------------------------------
    for name in [
        "MODUS_PONENS",              # From p and (p → q), derive q
        "MODUS_TOLLENS",             # From ¬q and (p → q), derive ¬p
        "HYPOTHETICAL_SYLLOGISM",    # From (p → q) and (q → r), derive (p → r)
        "DISJUNCTIVE_SYLLOGISM",     # From (p ∨ q) and ¬p, derive q
        "CONJUNCTION_INTRO",         # From p and q, derive (p ∧ q)
        "CONJUNCTION_ELIM_L",        # From (p ∧ q), derive p
        "CONJUNCTION_ELIM_R",        # From (p ∧ q), derive q
        "DOUBLE_NEG_ELIM",           # From ¬¬p, derive p
    ]:
        members[name] = idx
        idx += 1

    # -------------------------------------------------------------------------
    # 10. CSP primitives
    # -------------------------------------------------------------------------
    for name in [
        "DOMAIN",       # DOMAIN(var, state) -> Set
        "ASSIGN",       # ASSIGN(var, val, state) -> State
        "PROPAGATE",    # PROPAGATE(constraint, state) -> State
        "ELIMINATE",    # ELIMINATE(var, val, state) -> State
        "IS_CONSISTENT",# IS_CONSISTENT(state) -> Bool
        "BACKTRACK",    # BACKTRACK(state, checkpoint) -> State
    ]:
        members[name] = idx
        idx += 1

    # -------------------------------------------------------------------------
    # 10b. Sudoku / Grid CSP primitives
    # -------------------------------------------------------------------------
    for name in [
        "SUDOKU_GRID",      # Separator: marks end of puzzle, start of solution
        "CHECK_SUDOKU",     # CHECK_SUDOKU(grid) -> Bool (verify 9x9 Sudoku validity)
        "CHECK_LATIN",      # CHECK_LATIN(grid) -> Bool (verify Latin square)
        "ALL_DIFFERENT",    # ALL_DIFFERENT(cells) -> Bool (all values distinct)
        "ROW_VALID",        # ROW_VALID(grid, row_idx) -> Bool
        "COL_VALID",        # COL_VALID(grid, col_idx) -> Bool
        "BOX_VALID",        # BOX_VALID(grid, box_idx) -> Bool (3x3 box)
        "CELL",             # CELL(grid, row, col) -> Int (get cell value)
        "SET_CELL",         # SET_CELL(grid, row, col, val) -> Grid
        "EMPTY_CELLS",      # EMPTY_CELLS(grid) -> List[(row, col)]
        "CANDIDATES",       # CANDIDATES(grid, row, col) -> Set[Int]
    ]:
        members[name] = idx
        idx += 1

    # -------------------------------------------------------------------------
    # 11. First-Order Logic primitives
    # -------------------------------------------------------------------------
    for name in [
        "FORALL",       # FORALL(var, domain, prop) -> Prop
        "EXISTS",       # EXISTS(var, domain, prop) -> Prop
        "PREDICATE",        # PREDICATE(name, args) -> Prop
        "SUBSTITUTE_TERM",  # SUBSTITUTE_TERM(prop, var, term) -> Prop
        "UNIFY",            # UNIFY(t1, t2) -> Substitution
        "APPLY_SUBST",      # APPLY_SUBST(subst, prop) -> Prop
    ]:
        members[name] = idx
        idx += 1

    # -------------------------------------------------------------------------
    # 12. ARC primitives - Grid operations
    # -------------------------------------------------------------------------
    for name in [
        # Grid creation / access
        "CANVAS",       # canvas(value, dims) -> Grid
        "CROP",         # crop(grid, start, dims) -> Grid
        "INDEX",        # index(grid, loc) -> Int

        # Grid transformations
        "ROT90",        # rot90(grid) -> Grid
        "ROT180",       # rot180(grid) -> Grid
        "ROT270",       # rot270(grid) -> Grid
        "HMIRROR",      # hmirror(grid) -> Grid
        "VMIRROR",      # vmirror(grid) -> Grid
        "DMIRROR",      # dmirror(grid) -> Grid

        # Grid modifications
        "FILL",         # fill(grid, value, patch) -> Grid
        "PAINT",        # paint(grid, obj) -> Grid
        "REPLACE",      # replace(grid, old, new) -> Grid

        # Grid concatenation
        "HCONCAT",      # hconcat(a, b) -> Grid
        "VCONCAT",      # vconcat(a, b) -> Grid

        # Grid scaling
        "UPSCALE",      # upscale(grid, factor) -> Grid
        "DOWNSCALE",    # downscale(grid, factor) -> Grid

        # Grid splitting
        "HSPLIT",       # hsplit(grid, n) -> Tuple[Grid]
        "VSPLIT",       # vsplit(grid, n) -> Tuple[Grid]
    ]:
        members[name] = idx
        idx += 1

    # -------------------------------------------------------------------------
    # 13. ARC primitives - Object operations
    # -------------------------------------------------------------------------
    for name in [
        # Object extraction
        "OBJECTS",      # objects(grid, univalued, diagonal, without_bg) -> Objects
        "PARTITION",    # partition(grid) -> Objects
        "OFCOLOR",      # ofcolor(grid, value) -> Indices

        # Object properties
        "HEIGHT",       # height(obj) -> Int
        "WIDTH",        # width(obj) -> Int
        "SHAPE",        # shape(obj) -> (Int, Int)
        "COLOR",        # color(obj) -> Int
        "SIZE",         # size(obj) -> Int

        # Object transformations
        "SHIFT",        # shift(obj, offset) -> Object
        "NORMALIZE",    # normalize(obj) -> Object
        "RECOLOR",      # recolor(value, obj) -> Object

        # Object positions
        "ULCORNER",     # ulcorner(obj) -> (Int, Int)
        "LRCORNER",     # lrcorner(obj) -> (Int, Int)
        "CENTER",       # center(obj) -> (Int, Int)
        "CENTEROFMASS", # centerofmass(obj) -> (Int, Int)
    ]:
        members[name] = idx
        idx += 1

    # -------------------------------------------------------------------------
    # 14. ARC primitives - Set/container operations
    # -------------------------------------------------------------------------
    for name in [
        "COMBINE",      # combine(a, b) -> Container
        "INTERSECTION", # intersection(a, b) -> FrozenSet
        "DIFFERENCE",   # difference(a, b) -> FrozenSet
        "INSERT",       # insert(value, container) -> FrozenSet
        "REMOVE",       # remove(value, container) -> Container
        "SFILTER",      # sfilter(container, condition) -> Container
        "APPLY",        # apply(function, container) -> Container
        "MERGE",        # merge(containers) -> Container
    ]:
        members[name] = idx
        idx += 1

    # -------------------------------------------------------------------------
    # 15. ARC primitives - Comparison and selection
    # -------------------------------------------------------------------------
    for name in [
        "ARGMAX",       # argmax(container, compfunc) -> Any
        "ARGMIN",       # argmin(container, compfunc) -> Any
        "MAXIMUM",      # maximum(container) -> Int
        "MINIMUM",      # minimum(container) -> Int
        "MOSTCOMMON",   # mostcommon(container) -> Any
        "LEASTCOMMON",  # leastcommon(container) -> Any
        "FIRST",        # first(container) -> Any
        "LAST",         # last(container) -> Any
    ]:
        members[name] = idx
        idx += 1

    # -------------------------------------------------------------------------
    # 16. Control flow / composition
    # -------------------------------------------------------------------------
    for name in [
        "BRANCH",       # branch(condition, a, b) -> Any
        "COMPOSE",      # compose(outer, inner) -> Callable
        "CHAIN",        # chain(h, g, f) -> Callable
        "LBIND",        # lbind(func, fixed) -> Callable
        "RBIND",        # rbind(func, fixed) -> Callable
    ]:
        members[name] = idx
        idx += 1

    # -------------------------------------------------------------------------
    # 17. Code IR primitives (program structure / data-flow)
    # -------------------------------------------------------------------------
    # These tokens are used to serialize code-level intermediate
    # representations (IR) such as AST/DFG information into the unified
    # Gyan DSL token space. They are intentionally coarse: we rely on
    # surrounding INT_*, REAL_VAR_* and math/logic tokens to express fine
    # details like indices, literals, and predicates.
    for name in [
        # Basic program structure
        "CODE_FUNC_DEF",       # function definition header
        "CODE_PARAM",          # function parameter
        "CODE_RETURN",         # return statement
        "CODE_CALL",           # function/method call

        # Control-flow skeleton
        "CODE_IF",             # if / conditional start
        "CODE_ELSE",           # else / alternative branch
        "CODE_FOR",            # for-loop header
        "CODE_WHILE",          # while-loop header
        "CODE_BLOCK_START",    # begin lexical block/scope
        "CODE_BLOCK_END",      # end lexical block/scope

        # Variable and assignment events
        "CODE_DEF_VAR",        # variable declaration
        "CODE_ASSIGN",         # assignment of an expression to a variable

        # Data-flow relations (aligned with GraphCodeBERT DFG edge labels)
        "CODE_COMES_FROM",     # target comesFrom source definition(s)
        "CODE_COMPUTED_FROM",  # target computedFrom source(s)
    ]:
        members[name] = idx
        idx += 1

    # -------------------------------------------------------------------------
    # 18. Additional DSL primitives (auto-derived from dsl.dsl)
    # -------------------------------------------------------------------------
    # These are functions defined in dsl/dsl.py that don't yet have explicit
    # tokens above. We expose them as individual tokens so the full DSL
    # surface area is available to the model. Their precise type signatures
    # can be refined in GyanDSLMetadata over time.
    for name in [
        "ADJACENT",
        "ASINDICES",
        "ASOBJECT",
        "ASTUPLE",
        "BACKDROP",
        "BORDERING",
        "BOTH",
        "BOTTOMHALF",
        "BOX",
        "CELLWISE",
        "CLOSEST_TO",
        "CMIRROR",
        "COLORCOUNT",
        "COLORFILTER",
        "COMPRESS",
        "CONNECT",
        "CONST",
        "CONTAINED",
        "CORNERS",
        "COVER",
        "CREMENT",
        "DECREMENT",
        "DEDUPE",
        "DELTA",
        "DIFFERENTIATE",
        "DIVIDE",
        "DNEIGHBORS",
        "DOUBLE",
        "EITHER",
        "EQUALITY",
        "EVAL_EXPR",
        "EVEN",
        "EXPAND_EXPR",
        "EXTRACT",
        "FACTOR_EXPR",
        "FGPARTITION",
        "FLIP",
        "FORK",
        "FRONTIERS",
        "GRAVITATE",
        "GREATER",
        "HALVE",
        "HFRONTIER",
        "HLINE",
        "HMATCHING",
        "HPERIOD",
        "HUPSCALE",
        "IDENTITY",
        "INBOX",
        "INCREMENT",
        "INEIGHBORS",
        "INEQ",
        "INITSET",
        "INTEGRATE_POLY",
        "INTERVAL",
        "INVERT",
        "IS_INTEGER",
        "KTH_LARGEST",
        "LEASTCOLOR",
        "LEFTHALF",
        "LEFTMOST",
        "LLCORNER",
        "LOWERMOST",
        "MAKE_POLY",
        "MANHATTAN",
        "MAPPLY",
        "MATCHER",
        "MFILTER",
        "MOSTCOLOR",
        "MOVE",
        "MPAPPLY",
        "MULTIPLY",
        "NEIGHBORS",
        "NUMCOLORS",
        "OCCURRENCES",
        "ORDER",
        "OTHER",
        "OUTBOX",
        "PAIR",
        "PALETTE",
        "PAPPLY",
        "PORTRAIT",
        "POSITION",
        "POSITIVE",
        "POWER",
        "PRAPPLY",
        "PROD",
        "PRODUCT",
        "RAPPLY",
        "REPEAT",
        "RIGHTHALF",
        "RIGHTMOST",
        "SHOOT",
        "SIGN",
        "SIMPLIFY_EXPR",
        "SIZEFILTER",
        "SORT",
        "SUBGRID",
        "SUBTRACT",
        "SWITCH",
        "TOINDICES",
        "TOIVEC",
        "TOJVEC",
        "TOOBJECT",
        "TOPHALF",
        "TOTUPLE",
        "TRIM",
        "UNDERFILL",
        "UNDERPAINT",
        "UPPERMOST",
        "URCORNER",
        "VALMAX",
        "VALMIN",
        "VAR",
        "VFRONTIER",
        "VLINE",
        "VMATCHING",
        "VPERIOD",
        "VUPSCALE",
    ]:
        members[name] = idx
        idx += 1

    # Build the enum
    enum_cls = Enum("GyanDSLToken", members)

    # Attach metadata for convenience
    enum_cls.NUM_REAL_VARS = NUM_REAL_VARS  # type: ignore[attr-defined]
    enum_cls.NUM_BOOL_VARS = NUM_BOOL_VARS  # type: ignore[attr-defined]
    enum_cls.NUM_INT_CONSTS = NUM_INT_CONSTS  # type: ignore[attr-defined]

    return enum_cls


GyanDSLToken = _build_gyan_dsl_token_enum()


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

def get_vocab_size() -> int:
    """Return the total vocabulary size."""
    return len(GyanDSLToken)


def token_to_id(token: GyanDSLToken) -> int:
    """Get the integer ID for a token."""
    return token.value


def id_to_token(token_id: int) -> GyanDSLToken:
    """Get the token for an integer ID."""
    for tok in GyanDSLToken:
        if tok.value == token_id:
            return tok
    raise KeyError(f"Unknown token ID: {token_id}")


def get_int_const_token(value: int) -> GyanDSLToken:
    """
    Get the token for an integer constant.

    Supports:
      - 0..99 via INT_0..INT_99
      - -1, -2, -10, -100 via special tokens
    """
    if 0 <= value < NUM_INT_CONSTS:
        return GyanDSLToken[f"INT_{value}"]
    if value == -1:
        return GyanDSLToken.INT_NEG1
    if value == -2:
        return GyanDSLToken.INT_NEG2
    if value == -10:
        return GyanDSLToken.INT_NEG10
    if value == -100:
        return GyanDSLToken.INT_NEG100
    raise ValueError(f"No token for integer constant: {value}")


def get_real_var_token(index: int) -> GyanDSLToken:
    """Get REAL_VAR_i token."""
    if 0 <= index < NUM_REAL_VARS:
        return GyanDSLToken[f"REAL_VAR_{index}"]
    raise ValueError(f"Real variable index out of range: {index}")


def get_bool_var_token(index: int) -> GyanDSLToken:
    """Get BOOL_VAR_i token."""
    if 0 <= index < NUM_BOOL_VARS:
        return GyanDSLToken[f"BOOL_VAR_{index}"]
    raise ValueError(f"Bool variable index out of range: {index}")


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Gyan DSL vocabulary size: {get_vocab_size()}")
    print(f"Real variables: {NUM_REAL_VARS}")
    print(f"Bool variables: {NUM_BOOL_VARS}")
    print(f"Int constants: {NUM_INT_CONSTS}")
    print()
    print("Sample tokens:")
    for tok in list(GyanDSLToken)[:20]:
        print(f"  {tok.name} = {tok.value}")
    print("  ...")
    for tok in list(GyanDSLToken)[-10:]:
        print(f"  {tok.name} = {tok.value}")

