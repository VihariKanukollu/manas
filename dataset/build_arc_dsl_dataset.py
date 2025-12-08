"""
Build an ARC DSL dataset for TRM training.

This converts ARC puzzles with their DSL solutions (from dsl/solvers.py) into
the token format expected by TRM's PuzzleDataset.

Architecture:
    Input:  ARC grid (flattened or tokenized representation)
    Target: DSL program tokens that solve the puzzle

Two modes are supported:
    1. Program synthesis mode: Given input grid, predict the DSL program
    2. Compiler mode: Given English description, predict DSL program

Usage:
    python -m dataset.build_arc_dsl_dataset \
        --arc_data_prefix=kaggle/combined/arc-agi \
        --output_dir=data/arc_dsl_trm \
        --mode=program_synthesis
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel

from dataset.common import PuzzleDatasetMetadata
from dsl import solvers
from dsl.tokens import GyanDSLToken, get_vocab_size, get_int_const_token

cli = ArgParser()

# Token constants (reuse unified DSL vocab)
PAD_ID = GyanDSLToken.PAD.value
BOS_ID = GyanDSLToken.BOS.value
EOS_ID = GyanDSLToken.EOS.value
IGNORE_LABEL_ID = -100

# Grid constants
ARC_MAX_GRID_SIZE = 30

# Precompute color -> INT_* token mapping for ARC grids (colors 0..9)
COLOR_TOKEN_IDS = np.array(
    [get_int_const_token(c).value for c in range(10)], dtype=np.int32
)


class ARCDSLConfig(BaseModel):
    """CLI configuration."""
    arc_data_prefix: str  # e.g., "kaggle/combined/arc-agi"
    output_dir: str
    subsets: List[str] = ["training", "evaluation"]
    mode: str = "program_synthesis"  # Reserved for future extensions

    # Program (DSL) sequence length
    seq_len: int = 256

    # Flattened ARC grid length (fixed at 30x30 for now)
    grid_seq_len: int = ARC_MAX_GRID_SIZE * ARC_MAX_GRID_SIZE  # 900 for 30x30

    # Optional split of input vs. program segments for decoder-only training.
    # If left as None, we default to:
    #   english_seq_len = grid_seq_len
    #   program_seq_len = seq_len
    english_seq_len: Optional[int] = None
    program_seq_len: Optional[int] = None

    seed: int = 42
    include_unsolved: bool = False  # Include puzzles without DSL solutions


# ---------------------------------------------------------------------------
# DSL Program Parsing
# ---------------------------------------------------------------------------

# Map DSL function names to their token enum values
DSL_FUNC_TO_TOKEN: Dict[str, GyanDSLToken] = {}

def _build_dsl_func_map():
    """Build mapping from DSL function names to tokens."""
    global DSL_FUNC_TO_TOKEN
    if DSL_FUNC_TO_TOKEN:
        return
    
    # Get all token names
    for token in GyanDSLToken:
        # Match function names (case-insensitive)
        name_lower = token.name.lower()
        DSL_FUNC_TO_TOKEN[name_lower] = token
    
    # Add common aliases
    aliases = {
        "t": GyanDSLToken.BOOL_TRUE,
        "f": GyanDSLToken.BOOL_FALSE,
        "true": GyanDSLToken.BOOL_TRUE,
        "false": GyanDSLToken.BOOL_FALSE,
    }
    DSL_FUNC_TO_TOKEN.update(aliases)

_build_dsl_func_map()


# Map constants from dsl/constants.py
CONSTANT_MAP = {
    "ZERO": 0, "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4,
    "FIVE": 5, "SIX": 6, "SEVEN": 7, "EIGHT": 8, "NINE": 9,
    "TEN": 10, 
    "NEG_ONE": -1, "NEG_TWO": -2,
    "ORIGIN": (0, 0), "UNITY": (1, 1), "NEG_UNITY": (-1, -1),
    "UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1),
    "UP_RIGHT": (-1, 1), "DOWN_LEFT": (1, -1),
    "ZERO_BY_TWO": (0, 2), "TWO_BY_ZERO": (2, 0),
    "TWO_BY_TWO": (2, 2), "THREE_BY_THREE": (3, 3),
}


@dataclass
class ParsedDSLProgram:
    """Parsed DSL program with token sequence."""
    puzzle_id: str
    tokens: List[GyanDSLToken]
    token_ids: List[int]
    source_code: str
    num_variables: int


class DSLASTVisitor(ast.NodeVisitor):
    """
    AST visitor that converts a DSL solver function to postfix token sequence.
    
    The DSL programs are purely functional with assignments, so we convert
    to a postfix (reverse Polish) notation where:
        x1 = vmirror(I)    -> I VMIRROR
        O = hconcat(I, x1) -> I x1 HCONCAT
    """
    
    def __init__(self):
        self.tokens: List[GyanDSLToken] = []
        self.var_stack: Dict[str, int] = {}  # variable name -> stack position
        self.next_var_id = 0
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Process function definition."""
        # Mark input as var 0
        if node.args.args:
            input_name = node.args.args[0].arg
            self.var_stack[input_name] = self.next_var_id
            self.next_var_id += 1
        
        # Process body
        for stmt in node.body:
            self.visit(stmt)
            
    def visit_Assign(self, node: ast.Assign):
        """Process assignment: x1 = expr"""
        # Evaluate RHS
        self._visit_expr(node.value)
        
        # Assign to variable
        if node.targets:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                self.var_stack[target.id] = self.next_var_id
                self.next_var_id += 1
                
    def visit_Return(self, node: ast.Return):
        """Process return statement."""
        if node.value:
            self._visit_expr(node.value)
            
    def _visit_expr(self, node: ast.expr):
        """Visit an expression node and emit tokens."""
        if isinstance(node, ast.Call):
            self._visit_call(node)
        elif isinstance(node, ast.Name):
            self._visit_name(node)
        elif isinstance(node, ast.Constant):
            self._visit_constant(node)
        elif isinstance(node, ast.Tuple):
            self._visit_tuple(node)
        elif isinstance(node, ast.UnaryOp):
            self._visit_unary(node)
        elif isinstance(node, ast.BinOp):
            self._visit_binop(node)
        else:
            # Fallback: try to handle
            pass
            
    def _visit_call(self, node: ast.Call):
        """Visit function call: func(arg1, arg2, ...)"""
        # First emit all arguments
        for arg in node.args:
            self._visit_expr(arg)
            
        # Then emit the function token
        if isinstance(node.func, ast.Name):
            func_name = node.func.id.lower()
            if func_name in DSL_FUNC_TO_TOKEN:
                self.tokens.append(DSL_FUNC_TO_TOKEN[func_name])
            elif func_name in self.var_stack or func_name.startswith('x'):
                # Variable being used as a function (partial application)
                # Emit the variable reference followed by APPLY
                if func_name in self.var_stack:
                    var_id = self.var_stack[func_name]
                    try:
                        self.tokens.append(GyanDSLToken[f"REAL_VAR_{var_id}"])
                        self.tokens.append(GyanDSLToken.APPLY)
                    except KeyError:
                        pass
            else:
                # Unknown function - emit as special token (debug only)
                pass  # Silently skip to reduce noise
                
    def _visit_name(self, node: ast.Name):
        """Visit variable reference."""
        name = node.id
        
        # Check if it's a constant
        if name in CONSTANT_MAP:
            val = CONSTANT_MAP[name]
            if isinstance(val, int):
                if val >= 0:
                    self.tokens.append(get_int_const_token(val))
                else:
                    # Negative constant: emit as 0 - abs(val)
                    self.tokens.append(get_int_const_token(0))
                    self.tokens.append(get_int_const_token(abs(val)))
                    self.tokens.append(GyanDSLToken.SUB)
            elif isinstance(val, tuple):
                # Emit tuple as two ints (handle negatives)
                for v in val:
                    if v >= 0:
                        self.tokens.append(get_int_const_token(v))
                    else:
                        self.tokens.append(get_int_const_token(0))
                        self.tokens.append(get_int_const_token(abs(v)))
                        self.tokens.append(GyanDSLToken.SUB)
            return
            
        # Check if it's T/F for booleans
        if name == "T":
            self.tokens.append(GyanDSLToken.BOOL_TRUE)
            return
        if name == "F":
            self.tokens.append(GyanDSLToken.BOOL_FALSE)
            return
            
        # It's a variable reference - emit VAR token
        if name in self.var_stack:
            var_id = self.var_stack[name]
            try:
                self.tokens.append(GyanDSLToken[f"REAL_VAR_{var_id}"])
            except KeyError:
                # Too many variables
                pass
                
    def _visit_constant(self, node: ast.Constant):
        """Visit constant value."""
        val = node.value
        if isinstance(val, bool):
            self.tokens.append(GyanDSLToken.BOOL_TRUE if val else GyanDSLToken.BOOL_FALSE)
        elif isinstance(val, int):
            self.tokens.append(get_int_const_token(val))
            
    def _visit_tuple(self, node: ast.Tuple):
        """Visit tuple literal."""
        for elt in node.elts:
            self._visit_expr(elt)
            
    def _visit_unary(self, node: ast.UnaryOp):
        """Visit unary operation."""
        self._visit_expr(node.operand)
        if isinstance(node.op, ast.USub):
            self.tokens.append(GyanDSLToken.NEG)
            
    def _visit_binop(self, node: ast.BinOp):
        """Visit binary operation."""
        self._visit_expr(node.left)
        self._visit_expr(node.right)
        if isinstance(node.op, ast.Add):
            self.tokens.append(GyanDSLToken.ADD)
        elif isinstance(node.op, ast.Sub):
            self.tokens.append(GyanDSLToken.SUB)
        elif isinstance(node.op, ast.Mult):
            self.tokens.append(GyanDSLToken.MUL)
        elif isinstance(node.op, ast.Div):
            self.tokens.append(GyanDSLToken.DIV)


def parse_dsl_solver(func: Callable) -> Optional[ParsedDSLProgram]:
    """
    Parse a DSL solver function into a token sequence.
    
    Args:
        func: A solve_XXXXX function from dsl/solvers.py
        
    Returns:
        ParsedDSLProgram or None if parsing fails
    """
    # Extract puzzle ID from function name
    func_name = func.__name__
    if not func_name.startswith("solve_"):
        return None
    puzzle_id = func_name[6:]  # Remove "solve_" prefix
    
    # Get source code
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return None
        
    # Parse AST
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
        
    # Visit and extract tokens
    visitor = DSLASTVisitor()
    visitor.visit(tree)
    
    # Build final token sequence with BOS/EOS
    tokens = [GyanDSLToken.BOS] + visitor.tokens + [GyanDSLToken.EOS]
    token_ids = [t.value for t in tokens]
    
    return ParsedDSLProgram(
        puzzle_id=puzzle_id,
        tokens=tokens,
        token_ids=token_ids,
        source_code=source,
        num_variables=visitor.next_var_id
    )


def get_all_dsl_solvers() -> Dict[str, ParsedDSLProgram]:
    """
    Extract all DSL solver programs from dsl/solvers.py.
    
    Returns:
        Dict mapping puzzle_id -> ParsedDSLProgram
    """
    result = {}
    
    # Get all solve_* functions from solvers module
    for name in dir(solvers):
        if name.startswith("solve_"):
            func = getattr(solvers, name)
            if callable(func):
                parsed = parse_dsl_solver(func)
                if parsed:
                    result[parsed.puzzle_id] = parsed
                    
    return result


# ---------------------------------------------------------------------------
# Grid Tokenization (for input representation)
# ---------------------------------------------------------------------------

def tokenize_grid(grid: List[List[int]], max_size: int = ARC_MAX_GRID_SIZE) -> np.ndarray:
    """
    Tokenize an ARC grid into a flattened sequence.
    
    Format:
      - Each cell value (0-9) is mapped to the corresponding INT_* DSL token.
      - Grid is padded to max_size x max_size with PAD tokens.
    """
    arr = np.array(grid, dtype=np.uint8)
    nrow, ncol = arr.shape
    
    # Create padded result filled with PAD
    result = np.full((max_size, max_size), PAD_ID, dtype=np.int32)

    if nrow > max_size or ncol > max_size:
        raise ValueError(f"Grid shape {arr.shape} exceeds max_size={max_size}")

    # Map colors 0..9 -> INT_* tokens
    if arr.size > 0:
        if np.any((arr < 0) | (arr >= COLOR_TOKEN_IDS.shape[0])):
            raise ValueError(f"Grid contains color outside 0..9: {arr.min()}..{arr.max()}")
        result[:nrow, :ncol] = COLOR_TOKEN_IDS[arr]

    return result.flatten()


# ---------------------------------------------------------------------------
# Dataset Construction
# ---------------------------------------------------------------------------

@dataclass
class ARCDSLExample:
    """Single training example."""
    puzzle_id: str
    input_grid: np.ndarray       # Flattened input grid tokens
    output_grid: np.ndarray      # Flattened output grid tokens
    program_tokens: np.ndarray   # DSL program tokens (padded)
    is_train_example: bool       # True for training examples, False for test


def load_arc_puzzles(prefix: str, subsets: List[str]) -> Dict[str, dict]:
    """Load ARC puzzles from JSON files."""
    puzzles = {}
    
    for subset in subsets:
        challenges_path = f"{prefix}_{subset}_challenges.json"
        solutions_path = f"{prefix}_{subset}_solutions.json"
        
        if not os.path.exists(challenges_path):
            print(f"Warning: {challenges_path} not found, skipping")
            continue
            
        with open(challenges_path, "r") as f:
            challenges = json.load(f)
            
        solutions = {}
        if os.path.exists(solutions_path):
            with open(solutions_path, "r") as f:
                solutions = json.load(f)
                
        for puzzle_id, puzzle in challenges.items():
            # Add solutions if available
            if puzzle_id in solutions:
                for i, sol in enumerate(solutions[puzzle_id]):
                    puzzle["test"][i]["output"] = sol
            else:
                # Fill with dummy for puzzles without solutions
                for test_ex in puzzle["test"]:
                    if "output" not in test_ex:
                        test_ex["output"] = [[0]]
                        
            puzzles[puzzle_id] = puzzle
            
    return puzzles


def build_examples(
    puzzles: Dict[str, dict],
    dsl_programs: Dict[str, ParsedDSLProgram],
    config: ARCDSLConfig,
) -> Tuple[List[ARCDSLExample], Dict[str, int]]:
    """
    Build training examples from puzzles and DSL programs.
    
    Returns:
        - List of ARCDSLExample
        - Dict mapping puzzle_id to numeric identifier
    """
    examples = []
    puzzle_id_map = {"<blank>": 0}
    next_id = 1
    
    for puzzle_id, puzzle in puzzles.items():
        # Check if we have a DSL solution
        has_solution = puzzle_id in dsl_programs
        if not has_solution and not config.include_unsolved:
            continue
            
        # Assign numeric ID
        if puzzle_id not in puzzle_id_map:
            puzzle_id_map[puzzle_id] = next_id
            next_id += 1
            
        # Get program tokens (or empty if no solution)
        if has_solution:
            program = dsl_programs[puzzle_id]
            program_tokens = np.array(program.token_ids, dtype=np.int32)
        else:
            program_tokens = np.array([BOS_ID, EOS_ID], dtype=np.int32)
            
        # Pad program to seq_len
        if len(program_tokens) > config.seq_len:
            print(f"Warning: Program for {puzzle_id} exceeds seq_len ({len(program_tokens)} > {config.seq_len})")
            continue
        program_padded = np.full(config.seq_len, PAD_ID, dtype=np.int32)
        program_padded[:len(program_tokens)] = program_tokens
        
        # Create example for each train+test pair
        for ex in puzzle["train"]:
            input_grid = tokenize_grid(ex["input"])
            output_grid = tokenize_grid(ex["output"])
            examples.append(ARCDSLExample(
                puzzle_id=puzzle_id,
                input_grid=input_grid,
                output_grid=output_grid,
                program_tokens=program_padded,
                is_train_example=True,
            ))
            
        for ex in puzzle["test"]:
            input_grid = tokenize_grid(ex["input"])
            output_grid = tokenize_grid(ex["output"])
            examples.append(ARCDSLExample(
                puzzle_id=puzzle_id,
                input_grid=input_grid,
                output_grid=output_grid,
                program_tokens=program_padded,
                is_train_example=False,
            ))
            
    return examples, puzzle_id_map


def convert_to_numpy_arrays(
    examples: List[ARCDSLExample],
    puzzle_id_map: Dict[str, int],
    config: ARCDSLConfig,
) -> Dict[str, np.ndarray]:
    """
    Convert examples to numpy arrays for PuzzleDataset.

    We follow the same pattern as `build_dsl_dataset.py` in *compiler-mode*:
      - Store ARC grids in an `english` tensor (encoder segment).
      - Store DSL programs in a `program` tensor (decoder segment).
      - Provide placeholder `inputs` / `labels` which will be overridden
        in `PuzzleDataset._collate_batch` when both `english` and `program`
        are present.
    """
    n = len(examples)

    # Resolve effective segment lengths
    english_len = config.english_seq_len or config.grid_seq_len
    program_len = config.program_seq_len or config.seq_len

    # ARC grids -> "english" segment
    english_list: List[np.ndarray] = []
    program_list: List[np.ndarray] = []
    puzzle_ids_list: List[int] = []

    for ex in examples:
        # English: flattened grid tokens padded/truncated to english_len
        grid_tokens = ex.input_grid
        if grid_tokens.shape[0] > english_len:
            raise ValueError(
                f"Grid sequence length {grid_tokens.shape[0]} exceeds english_len={english_len}"
            )
        eng = np.full(english_len, PAD_ID, dtype=np.int32)
        eng[: grid_tokens.shape[0]] = grid_tokens
        english_list.append(eng)

        # Program tokens are already padded to config.seq_len/program_len
        if ex.program_tokens.shape[0] != program_len:
            raise ValueError(
                f"Program length {ex.program_tokens.shape[0]} != configured program_len={program_len}"
            )
        program_list.append(ex.program_tokens)
        puzzle_ids_list.append(puzzle_id_map[ex.puzzle_id])

    english = np.stack(english_list, axis=0)
    programs = np.stack(program_list, axis=0)

    # Placeholder inputs/labels (overridden in PuzzleDataset._collate_batch)
    total_seq_len = english_len + program_len
    inputs = np.full((n, total_seq_len), PAD_ID, dtype=np.int32)
    labels = np.full((n, total_seq_len), IGNORE_LABEL_ID, dtype=np.int32)

    # Puzzle identifiers
    puzzle_ids = np.array(puzzle_ids_list, dtype=np.int32)
    
    # For simplicity: one example per puzzle, one puzzle per group
    puzzle_indices = np.arange(n + 1, dtype=np.int32)
    group_indices = np.arange(n + 1, dtype=np.int32)
    
    return {
        "inputs": inputs,
        "labels": labels,
        "english": english,
        "program": programs,
        "puzzle_identifiers": puzzle_ids,
        "puzzle_indices": puzzle_indices,
        "group_indices": group_indices,
    }


@cli.command(singleton=True)
def main(config: ARCDSLConfig) -> None:
    """Build ARC DSL dataset."""
    np.random.seed(config.seed)

    # Resolve segment lengths if not explicitly provided
    if config.english_seq_len is None:
        config.english_seq_len = config.grid_seq_len
    if config.program_seq_len is None:
        config.program_seq_len = config.seq_len

    print("Loading DSL solvers...")
    dsl_programs = get_all_dsl_solvers()
    print(f"Found {len(dsl_programs)} DSL solver programs")
    
    print("Loading ARC puzzles...")
    puzzles = load_arc_puzzles(config.arc_data_prefix, config.subsets)
    print(f"Found {len(puzzles)} ARC puzzles")
    
    # Count coverage
    covered = sum(1 for pid in puzzles if pid in dsl_programs)
    print(f"DSL coverage: {covered}/{len(puzzles)} puzzles ({100*covered/len(puzzles):.1f}%)")
    
    print("Building examples...")
    examples, puzzle_id_map = build_examples(puzzles, dsl_programs, config)
    print(f"Created {len(examples)} examples")
    
    # Split into train/test
    train_examples = [ex for ex in examples if ex.is_train_example]
    test_examples = [ex for ex in examples if not ex.is_train_example]
    
    print(f"Train examples: {len(train_examples)}")
    print(f"Test examples: {len(test_examples)}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    vocab_size = get_vocab_size()
    num_identifiers = len(puzzle_id_map)
    
    # Process each split
    for split_name, split_examples in [("train", train_examples), ("test", test_examples)]:
        if not split_examples:
            print(f"Warning: No examples for {split_name} split")
            continue
            
        split_dir = os.path.join(config.output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        data = convert_to_numpy_arrays(split_examples, puzzle_id_map, config)
        
        # Save arrays
        for name, arr in data.items():
            path = os.path.join(split_dir, f"all__{name}.npy")
            np.save(path, arr)
            print(f"  [{split_name}] saved {name}: shape={arr.shape}")
            
        # Save metadata.
        # IMPORTANT: seq_len here should match the *combined* sequence length
        # that the model will see after `PuzzleDataset._collate_batch`, i.e.
        # english_len + program_len.
        seq_len_meta = (config.english_seq_len or config.grid_seq_len) + (
            config.program_seq_len or config.seq_len
        )
        metadata = PuzzleDatasetMetadata(
            seq_len=seq_len_meta,
            vocab_size=vocab_size,
            pad_id=PAD_ID,
            ignore_label_id=None,  # labels already use -100 for ignore
            blank_identifier_id=0,
            num_puzzle_identifiers=num_identifiers,
            total_groups=len(split_examples),
            mean_puzzle_examples=1.0,
            total_puzzles=len(split_examples),
            sets=["all"],
        )
        
        with open(os.path.join(split_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)
            
    # Save identifier mapping
    id_to_name = {v: k for k, v in puzzle_id_map.items()}
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump([id_to_name.get(i, "<unknown>") for i in range(num_identifiers)], f, indent=2)
        
    # Save program mapping (puzzle_id -> program source)
    program_sources = {pid: prog.source_code for pid, prog in dsl_programs.items()}
    with open(os.path.join(config.output_dir, "program_sources.json"), "w") as f:
        json.dump(program_sources, f, indent=2)
        
    print(f"\nDataset saved to {config.output_dir}")


if __name__ == "__main__":
    cli()

