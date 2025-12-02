"""
Gyan DSL metadata and type system.

This file provides:
- ValueType enum (REAL, BOOL, EXPR, EQUATION, PROP, GRID, OBJECT, etc.)
- TokenInfo dataclass (arity, input types, output type)
- GyanDSLMetadata class with:
  - token_to_id / id_to_token mappings
  - is_token_valid_in_context() for type-based masking
  - apply_token_to_stack() for stack simulation
- ProgramState for tracking partial programs during generation
- compute_valid_token_mask() for constrained decoding

This mirrors dhyana2's DSLMetadata but covers the full Gyan DSL.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import torch

from .tokens import GyanDSLToken, get_vocab_size


# ---------------------------------------------------------------------------
# Type system
# ---------------------------------------------------------------------------

class ValueType(Enum):
    """Base value types in the Gyan DSL."""
    # Numeric types
    INT = auto()        # Integer value
    REAL = auto()       # Real number
    BOOL = auto()       # Boolean value

    # Symbolic math types
    EXPR = auto()       # SymPy expression
    EQUATION = auto()   # SymPy Eq or Relational
    VAR = auto()        # SymPy Symbol (variable)

    # Logic types
    PROP = auto()       # Propositional formula
    TERM = auto()       # FOL term
    SUBST = auto()      # Substitution dict

    # CSP types
    CSP_STATE = auto()  # CSP state dict
    CONSTRAINT = auto() # CSP constraint callable

    # ARC types
    GRID = auto()       # 2D grid (tuple of tuples)
    OBJECT = auto()     # Object (frozenset of (value, (i, j)))
    INDICES = auto()    # Set of (i, j) indices
    PATCH = auto()      # Patch (indices or object)

    # Container types
    CONTAINER = auto()  # Generic container (tuple/frozenset)
    CALLABLE = auto()   # Function/callable

    # Special
    ANY = auto()        # Accepts any type


@dataclass
class TypeInfo:
    """Represents a type on the evaluation stack."""
    kind: ValueType


# Convenience type singletons
INT_T = TypeInfo(ValueType.INT)
REAL_T = TypeInfo(ValueType.REAL)
BOOL_T = TypeInfo(ValueType.BOOL)
EXPR_T = TypeInfo(ValueType.EXPR)
EQUATION_T = TypeInfo(ValueType.EQUATION)
VAR_T = TypeInfo(ValueType.VAR)
PROP_T = TypeInfo(ValueType.PROP)
TERM_T = TypeInfo(ValueType.TERM)
SUBST_T = TypeInfo(ValueType.SUBST)
CSP_STATE_T = TypeInfo(ValueType.CSP_STATE)
CONSTRAINT_T = TypeInfo(ValueType.CONSTRAINT)
GRID_T = TypeInfo(ValueType.GRID)
OBJECT_T = TypeInfo(ValueType.OBJECT)
INDICES_T = TypeInfo(ValueType.INDICES)
PATCH_T = TypeInfo(ValueType.PATCH)
CONTAINER_T = TypeInfo(ValueType.CONTAINER)
CALLABLE_T = TypeInfo(ValueType.CALLABLE)
ANY_T = TypeInfo(ValueType.ANY)


# ---------------------------------------------------------------------------
# Token metadata
# ---------------------------------------------------------------------------

@dataclass
class TokenMeta:
    """
    Metadata for a single DSL token.

    Attributes:
        token: the GyanDSLToken enum value
        arity: how many values are popped from the stack
        pushes_value: whether this token pushes a value onto the stack
        output_type: type of the pushed value (if pushes_value=True)
        input_types: expected types for operands, from top of stack
                     e.g., [REAL_T, REAL_T] for ADD; [BOOL_T] for NOT
    """
    token: GyanDSLToken
    arity: int
    pushes_value: bool
    output_type: Optional[TypeInfo] = None
    input_types: Optional[List[TypeInfo]] = None


# ---------------------------------------------------------------------------
# GyanDSLMetadata
# ---------------------------------------------------------------------------

class GyanDSLMetadata:
    """
    Holds DSL token metadata and helper methods for type checking.

    This is the core class for:
    - Looking up token semantics
    - Validating token sequences
    - Computing valid-token masks for constrained generation
    """

    def __init__(self):
        self.token_to_id: Dict[GyanDSLToken, int] = {}
        self.id_to_token: Dict[int, GyanDSLToken] = {}
        self.token_meta: Dict[GyanDSLToken, TokenMeta] = {}
        self._build()

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def _build(self):
        """Build the token metadata tables."""

        def add(
            tok: GyanDSLToken,
            arity: int,
            pushes: bool,
            output_type: Optional[TypeInfo] = None,
            input_types: Optional[List[TypeInfo]] = None,
        ):
            idx = tok.value
            self.token_to_id[tok] = idx
            self.id_to_token[idx] = tok
            self.token_meta[tok] = TokenMeta(
                token=tok,
                arity=arity,
                pushes_value=pushes,
                output_type=output_type,
                input_types=input_types,
            )

        # Process all tokens
        for tok in GyanDSLToken:
            name = tok.name

            # -----------------------------------------------------------------
            # Structural tokens
            # -----------------------------------------------------------------
            if name in ("PAD", "BOS", "EOS"):
                add(tok, 0, False)

            # -----------------------------------------------------------------
            # Variables
            # -----------------------------------------------------------------
            elif name.startswith("REAL_VAR_"):
                add(tok, 0, True, EXPR_T)  # Variables push symbolic expressions
            elif name.startswith("BOOL_VAR_"):
                add(tok, 0, True, BOOL_T)

            # -----------------------------------------------------------------
            # Constants
            # -----------------------------------------------------------------
            elif name.startswith("INT_"):
                add(tok, 0, True, INT_T)
            elif name.startswith("REAL_"):
                add(tok, 0, True, REAL_T)
            elif name in ("BOOL_TRUE", "BOOL_FALSE"):
                add(tok, 0, True, BOOL_T)

            # -----------------------------------------------------------------
            # Arithmetic (binary: EXPR, EXPR -> EXPR)
            # -----------------------------------------------------------------
            elif name in ("ADD", "SUB", "MUL", "DIV", "MOD", "POW"):
                add(tok, 2, True, EXPR_T, [EXPR_T, EXPR_T])

            # Arithmetic (unary: EXPR -> EXPR)
            elif name in ("NEG", "ABS", "SQRT", "SQUARE"):
                add(tok, 1, True, EXPR_T, [EXPR_T])

            # -----------------------------------------------------------------
            # Comparison (EXPR, EXPR -> BOOL)
            # -----------------------------------------------------------------
            elif name in ("LT", "LE", "GT", "GE", "EQ_CMP", "NE"):
                add(tok, 2, True, BOOL_T, [EXPR_T, EXPR_T])

            # -----------------------------------------------------------------
            # Boolean logic
            # -----------------------------------------------------------------
            elif name in ("AND", "OR", "IMPLIES", "IFF"):
                add(tok, 2, True, BOOL_T, [BOOL_T, BOOL_T])
            elif name == "NOT":
                add(tok, 1, True, BOOL_T, [BOOL_T])

            # -----------------------------------------------------------------
            # Math DSL - Equations
            # -----------------------------------------------------------------
            elif name == "EQ":
                add(tok, 2, True, EQUATION_T, [EXPR_T, EXPR_T])
            elif name in ("INEQ_LT", "INEQ_LE", "INEQ_GT", "INEQ_GE"):
                add(tok, 2, True, EQUATION_T, [EXPR_T, EXPR_T])
            elif name in ("ADD_BOTH_SIDES", "SUB_BOTH_SIDES", "MUL_BOTH_SIDES", "DIV_BOTH_SIDES"):
                add(tok, 2, True, EQUATION_T, [EQUATION_T, EXPR_T])
            elif name == "SUBSTITUTE":
                add(tok, 3, True, EQUATION_T, [EQUATION_T, VAR_T, EXPR_T])
            elif name in ("EXPAND", "FACTOR", "SIMPLIFY"):
                add(tok, 1, True, EXPR_T, [EXPR_T])
            elif name == "IS_SOLUTION":
                add(tok, 3, True, BOOL_T, [EQUATION_T, VAR_T, EXPR_T])
            elif name in ("GCD", "LCM"):
                add(tok, 2, True, INT_T, [INT_T, INT_T])
            elif name == "FACTORIAL":
                add(tok, 1, True, INT_T, [INT_T])
            elif name == "FACTORINT":
                add(tok, 1, True, CONTAINER_T, [INT_T])

            # -----------------------------------------------------------------
            # Logic DSL - Propositional
            # -----------------------------------------------------------------
            elif name == "PROP_VAR":
                add(tok, 1, True, PROP_T, [INT_T])  # Takes name index
            elif name in ("PROP_TRUE", "PROP_FALSE"):
                add(tok, 0, True, PROP_T)
            elif name in ("PROP_AND", "PROP_OR", "PROP_IMPLIES", "PROP_IFF"):
                add(tok, 2, True, PROP_T, [PROP_T, PROP_T])
            elif name == "PROP_NOT":
                add(tok, 1, True, PROP_T, [PROP_T])
            elif name == "EVAL_PROP":
                add(tok, 2, True, BOOL_T, [PROP_T, CONTAINER_T])
            elif name == "SIMPLIFY_PROP":
                add(tok, 1, True, PROP_T, [PROP_T])

            # -----------------------------------------------------------------
            # Logic DSL - Inference rules
            # -----------------------------------------------------------------
            elif name == "MODUS_PONENS":
                add(tok, 2, True, PROP_T, [PROP_T, PROP_T])
            elif name == "MODUS_TOLLENS":
                add(tok, 2, True, PROP_T, [PROP_T, PROP_T])
            elif name == "HYPOTHETICAL_SYLLOGISM":
                add(tok, 2, True, PROP_T, [PROP_T, PROP_T])
            elif name == "DISJUNCTIVE_SYLLOGISM":
                add(tok, 2, True, PROP_T, [PROP_T, PROP_T])
            elif name == "CONJUNCTION_INTRO":
                add(tok, 2, True, PROP_T, [PROP_T, PROP_T])
            elif name in ("CONJUNCTION_ELIM_L", "CONJUNCTION_ELIM_R"):
                add(tok, 1, True, PROP_T, [PROP_T])
            elif name == "DOUBLE_NEG_ELIM":
                add(tok, 1, True, PROP_T, [PROP_T])

            # -----------------------------------------------------------------
            # CSP primitives
            # -----------------------------------------------------------------
            elif name == "DOMAIN":
                add(tok, 2, True, CONTAINER_T, [INT_T, CSP_STATE_T])
            elif name == "ASSIGN":
                add(tok, 3, True, CSP_STATE_T, [INT_T, ANY_T, CSP_STATE_T])
            elif name == "PROPAGATE":
                add(tok, 2, True, CSP_STATE_T, [CONSTRAINT_T, CSP_STATE_T])
            elif name == "ELIMINATE":
                add(tok, 3, True, CSP_STATE_T, [INT_T, ANY_T, CSP_STATE_T])
            elif name == "IS_CONSISTENT":
                add(tok, 1, True, BOOL_T, [CSP_STATE_T])
            elif name == "BACKTRACK":
                add(tok, 2, True, CSP_STATE_T, [CSP_STATE_T, CSP_STATE_T])

            # -----------------------------------------------------------------
            # Sudoku / Grid CSP primitives
            # -----------------------------------------------------------------
            elif name == "SUDOKU_GRID":
                # Separator token (no stack effect, like structural)
                add(tok, 0, False)
            elif name == "CHECK_SUDOKU":
                # Consumes 81 INT values from stack, pushes BOOL
                # For simplicity, treat as unary on GRID type
                add(tok, 81, True, BOOL_T, [INT_T] * 81)
            elif name == "CHECK_LATIN":
                add(tok, 81, True, BOOL_T, [INT_T] * 81)
            elif name == "ALL_DIFFERENT":
                add(tok, 1, True, BOOL_T, [CONTAINER_T])
            elif name == "ROW_VALID":
                add(tok, 2, True, BOOL_T, [GRID_T, INT_T])
            elif name == "COL_VALID":
                add(tok, 2, True, BOOL_T, [GRID_T, INT_T])
            elif name == "BOX_VALID":
                add(tok, 2, True, BOOL_T, [GRID_T, INT_T])
            elif name == "CELL":
                add(tok, 3, True, INT_T, [GRID_T, INT_T, INT_T])
            elif name == "SET_CELL":
                add(tok, 4, True, GRID_T, [GRID_T, INT_T, INT_T, INT_T])
            elif name == "EMPTY_CELLS":
                add(tok, 1, True, CONTAINER_T, [GRID_T])
            elif name == "CANDIDATES":
                add(tok, 3, True, CONTAINER_T, [GRID_T, INT_T, INT_T])

            # -----------------------------------------------------------------
            # FOL primitives
            # -----------------------------------------------------------------
            elif name == "FORALL":
                add(tok, 3, True, PROP_T, [INT_T, CONTAINER_T, PROP_T])
            elif name == "EXISTS":
                add(tok, 3, True, PROP_T, [INT_T, CONTAINER_T, PROP_T])
            elif name == "PREDICATE":
                add(tok, 2, True, PROP_T, [INT_T, CONTAINER_T])
            elif name == "SUBSTITUTE_TERM":
                add(tok, 3, True, PROP_T, [PROP_T, INT_T, TERM_T])
            elif name == "UNIFY":
                add(tok, 2, True, SUBST_T, [TERM_T, TERM_T])
            elif name == "APPLY_SUBST":
                add(tok, 2, True, PROP_T, [SUBST_T, PROP_T])

            # -----------------------------------------------------------------
            # ARC - Grid operations
            # -----------------------------------------------------------------
            elif name == "CANVAS":
                add(tok, 2, True, GRID_T, [INT_T, CONTAINER_T])
            elif name == "CROP":
                add(tok, 3, True, GRID_T, [GRID_T, CONTAINER_T, CONTAINER_T])
            elif name == "INDEX":
                add(tok, 2, True, INT_T, [GRID_T, CONTAINER_T])
            elif name in ("ROT90", "ROT180", "ROT270", "HMIRROR", "VMIRROR", "DMIRROR"):
                add(tok, 1, True, GRID_T, [GRID_T])
            elif name == "FILL":
                add(tok, 3, True, GRID_T, [GRID_T, INT_T, PATCH_T])
            elif name == "PAINT":
                add(tok, 2, True, GRID_T, [GRID_T, OBJECT_T])
            elif name == "REPLACE":
                add(tok, 3, True, GRID_T, [GRID_T, INT_T, INT_T])
            elif name in ("HCONCAT", "VCONCAT"):
                add(tok, 2, True, GRID_T, [GRID_T, GRID_T])
            elif name in ("UPSCALE", "DOWNSCALE"):
                add(tok, 2, True, GRID_T, [GRID_T, INT_T])
            elif name in ("HSPLIT", "VSPLIT"):
                add(tok, 2, True, CONTAINER_T, [GRID_T, INT_T])

            # -----------------------------------------------------------------
            # ARC - Object operations
            # -----------------------------------------------------------------
            elif name == "OBJECTS":
                add(tok, 4, True, CONTAINER_T, [GRID_T, BOOL_T, BOOL_T, BOOL_T])
            elif name == "PARTITION":
                add(tok, 1, True, CONTAINER_T, [GRID_T])
            elif name == "OFCOLOR":
                add(tok, 2, True, INDICES_T, [GRID_T, INT_T])
            elif name in ("HEIGHT", "WIDTH"):
                add(tok, 1, True, INT_T, [PATCH_T])
            elif name == "SHAPE":
                add(tok, 1, True, CONTAINER_T, [PATCH_T])
            elif name == "COLOR":
                add(tok, 1, True, INT_T, [OBJECT_T])
            elif name == "SIZE":
                add(tok, 1, True, INT_T, [CONTAINER_T])
            elif name == "SHIFT":
                add(tok, 2, True, PATCH_T, [PATCH_T, CONTAINER_T])
            elif name == "NORMALIZE":
                add(tok, 1, True, PATCH_T, [PATCH_T])
            elif name == "RECOLOR":
                add(tok, 2, True, OBJECT_T, [INT_T, PATCH_T])
            elif name in ("ULCORNER", "LRCORNER", "CENTER", "CENTEROFMASS"):
                add(tok, 1, True, CONTAINER_T, [PATCH_T])

            # -----------------------------------------------------------------
            # ARC - Container operations
            # -----------------------------------------------------------------
            elif name in ("COMBINE", "INTERSECTION", "DIFFERENCE"):
                add(tok, 2, True, CONTAINER_T, [CONTAINER_T, CONTAINER_T])
            elif name == "INSERT":
                add(tok, 2, True, CONTAINER_T, [ANY_T, CONTAINER_T])
            elif name == "REMOVE":
                add(tok, 2, True, CONTAINER_T, [ANY_T, CONTAINER_T])
            elif name == "SFILTER":
                add(tok, 2, True, CONTAINER_T, [CONTAINER_T, CALLABLE_T])
            elif name == "APPLY":
                add(tok, 2, True, CONTAINER_T, [CALLABLE_T, CONTAINER_T])
            elif name == "MERGE":
                add(tok, 1, True, CONTAINER_T, [CONTAINER_T])

            # -----------------------------------------------------------------
            # ARC - Selection operations
            # -----------------------------------------------------------------
            elif name in ("ARGMAX", "ARGMIN"):
                add(tok, 2, True, ANY_T, [CONTAINER_T, CALLABLE_T])
            elif name in ("MAXIMUM", "MINIMUM"):
                add(tok, 1, True, INT_T, [CONTAINER_T])
            elif name in ("MOSTCOMMON", "LEASTCOMMON"):
                add(tok, 1, True, ANY_T, [CONTAINER_T])
            elif name in ("FIRST", "LAST"):
                add(tok, 1, True, ANY_T, [CONTAINER_T])

            # -----------------------------------------------------------------
            # Control flow
            # -----------------------------------------------------------------
            elif name == "BRANCH":
                add(tok, 3, True, ANY_T, [BOOL_T, ANY_T, ANY_T])
            elif name == "COMPOSE":
                add(tok, 2, True, CALLABLE_T, [CALLABLE_T, CALLABLE_T])
            elif name == "CHAIN":
                add(tok, 3, True, CALLABLE_T, [CALLABLE_T, CALLABLE_T, CALLABLE_T])
            elif name in ("LBIND", "RBIND"):
                add(tok, 2, True, CALLABLE_T, [CALLABLE_T, ANY_T])
            else:
                # Fallback: generic unary operator on ANY -> ANY.
                # This keeps newly added DSL primitives usable even if we
                # haven't yet written precise type rules for them.
                add(tok, 1, True, ANY_T, [ANY_T])

    # -------------------------------------------------------------------------
    # Type checking
    # -------------------------------------------------------------------------

    def is_token_valid_in_context(
        self,
        token: GyanDSLToken,
        stack_types: List[TypeInfo]
    ) -> bool:
        """
        Return True if token is type-valid given the current stack.

        This is used for constrained generation: we mask out tokens
        that would be invalid given the current program state.
        """
        meta = self.token_meta[token]

        # Structural tokens are always allowed
        if token in (GyanDSLToken.PAD, GyanDSLToken.BOS, GyanDSLToken.EOS):
            return True

        # Need enough operands on the stack
        if len(stack_types) < meta.arity:
            return False

        # Leaf value tokens (constants, variables) are always OK
        if meta.pushes_value and meta.arity == 0:
            return True

        # For operators, check operand types
        if meta.input_types is not None:
            for i, expected_type in enumerate(reversed(meta.input_types), start=1):
                actual_type = stack_types[-i]
                if not self._types_compatible(actual_type, expected_type):
                    return False

        return True

    def _types_compatible(self, actual: TypeInfo, expected: TypeInfo) -> bool:
        """Check if actual type is compatible with expected type."""
        # ANY accepts anything
        if expected.kind == ValueType.ANY:
            return True
        if actual.kind == ValueType.ANY:
            return True

        # Exact match
        if actual.kind == expected.kind:
            return True

        # INT and REAL are compatible with EXPR (numeric expressions)
        if expected.kind == ValueType.EXPR:
            if actual.kind in (ValueType.INT, ValueType.REAL, ValueType.EXPR):
                return True

        # OBJECT, INDICES are compatible with PATCH
        if expected.kind == ValueType.PATCH:
            if actual.kind in (ValueType.OBJECT, ValueType.INDICES, ValueType.PATCH):
                return True

        return False

    def apply_token_to_stack(
        self,
        token: GyanDSLToken,
        stack_types: List[TypeInfo]
    ) -> List[TypeInfo]:
        """
        Return new stack types after applying token.

        Assumes is_token_valid_in_context(token, stack_types) is True.
        """
        meta = self.token_meta[token]
        new_stack = list(stack_types)

        # Structural tokens have no effect
        if token in (GyanDSLToken.PAD, GyanDSLToken.BOS, GyanDSLToken.EOS):
            return new_stack

        # Pop operands
        for _ in range(meta.arity):
            new_stack.pop()

        # Push result
        if meta.pushes_value and meta.output_type is not None:
            new_stack.append(meta.output_type)

        return new_stack


# ---------------------------------------------------------------------------
# Program state for generation
# ---------------------------------------------------------------------------

@dataclass
class ProgramState:
    """Tracks the partial program and type stack for one sequence."""
    tokens: List[int]
    stack_types: List[TypeInfo]
    metadata: GyanDSLMetadata
    extra: Optional[Dict[str, Any]] = None


def init_program_state(metadata: GyanDSLMetadata) -> ProgramState:
    """Initialize an empty program state."""
    return ProgramState(
        tokens=[],
        stack_types=[],
        metadata=metadata,
        extra={},
    )


def update_program_state(state: ProgramState, new_token_id: int) -> ProgramState:
    """
    Return a new ProgramState after appending new_token_id.

    Caller is responsible for ensuring the token is type-valid.
    """
    metadata = state.metadata
    token_enum = metadata.id_to_token.get(new_token_id)
    if token_enum is None:
        raise KeyError(f"Unknown token ID: {new_token_id}")

    new_tokens = list(state.tokens)
    new_tokens.append(new_token_id)

    new_stack = metadata.apply_token_to_stack(token_enum, state.stack_types)

    return ProgramState(
        tokens=new_tokens,
        stack_types=new_stack,
        metadata=metadata,
        extra=dict(state.extra or {}),
    )


def compute_valid_token_mask(
    state: ProgramState,
    vocab_size: int
) -> torch.BoolTensor:
    """
    Compute a boolean mask [vocab_size] of type-valid next tokens.

    True means token is allowed given the current stack.
    """
    metadata = state.metadata
    mask = torch.zeros(vocab_size, dtype=torch.bool)

    for tid in range(vocab_size):
        token_enum = metadata.id_to_token.get(tid)
        if token_enum is None:
            continue
        if metadata.is_token_valid_in_context(token_enum, state.stack_types):
            mask[tid] = True

    return mask


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_METADATA: Optional[GyanDSLMetadata] = None


def get_metadata() -> GyanDSLMetadata:
    """Return a shared GyanDSLMetadata instance."""
    global _METADATA
    if _METADATA is None:
        _METADATA = GyanDSLMetadata()
    return _METADATA


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    metadata = get_metadata()
    print(f"Gyan DSL vocab size: {metadata.vocab_size}")
    print()

    # Test type checking
    state = init_program_state(metadata)

    # Push two integers
    state = update_program_state(state, GyanDSLToken.INT_5.value)
    print(f"After INT_5: stack = {[t.kind.name for t in state.stack_types]}")

    state = update_program_state(state, GyanDSLToken.INT_3.value)
    print(f"After INT_3: stack = {[t.kind.name for t in state.stack_types]}")

    # Check what's valid
    print(f"ADD valid? {metadata.is_token_valid_in_context(GyanDSLToken.ADD, state.stack_types)}")
    print(f"NOT valid? {metadata.is_token_valid_in_context(GyanDSLToken.NOT, state.stack_types)}")

    # Apply ADD
    state = update_program_state(state, GyanDSLToken.ADD.value)
    print(f"After ADD: stack = {[t.kind.name for t in state.stack_types]}")

