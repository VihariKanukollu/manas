#!/usr/bin/env python
"""
Build English→DSL TRM dataset using SentencePiece tokenizer.

This creates a TRM-format dataset where:
- Input: English text encoded via SentencePiece (CHAR tokens) + PAD + [MASK for DSL]
- Labels: -100 for English positions, DSL tokens for output positions

The model learns to "solve the puzzle": given English chars, produce DSL tokens.

Usage:
    python -m dataset.build_english_trm_dataset \
        --src data/en_dsl_seq2seq/train.jsonl \
        --dst data/english_to_dsl_trm/train \
        --sp_model data/en_dsl_seq2seq/en_dsl_spm.model \
        --seq_len 256
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

try:
    import sentencepiece as spm
except ImportError:
    print("ERROR: sentencepiece not installed. Run: pip install sentencepiece")
    sys.exit(1)

# Constants
IGNORE_LABEL_ID = -100


class DSLSentencePieceTokenizer:
    """Wrapper around SentencePiece for DSL tokenization.

    Important design choice:
    ------------------------
    We treat every `GyanDSLToken` name (including `CHAR_*` tokens) as a
    **single atomic piece**. Instead of letting SentencePiece segment strings,
    we build an explicit mapping from token-name → piece-id and look up IDs
    directly. This guarantees:

    - No accidental `<unk>` pieces.
    - One ID per DSL token name.
    - Robustness if we ever change how we join tokens into strings.
    """

    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

        # Cache special token IDs
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.unk_id = self.sp.unk_id()

        # Build mapping from DSL token name -> SentencePiece ID.
        # With WORD model, pieces are stored as "▁TOKEN".
        self.name_to_id: Dict[str, int] = {}
        pad_id: Optional[int] = None
        for i in range(self.sp.GetPieceSize()):
            piece = self.sp.IdToPiece(i)
            # Strip leading word-boundary marker.
            name = piece[1:] if piece.startswith("▁") else piece

            # Skip meta tokens like <unk>, <s>, </s>.
            if name and not name.startswith("<"):
                # First occurrence wins; DSL tokens are unique anyway.
                if name not in self.name_to_id:
                    self.name_to_id[name] = i

            if name == "PAD":
                pad_id = i

        # PAD should be a real DSL token; fall back to 0 if missing.
        self.pad_id = int(pad_id) if pad_id is not None else 0

        print(f"Loaded SentencePiece model: {model_path}")
        print(f"  Vocab size: {self.sp.GetPieceSize()}")
        print(f"  PAD ID: {self.pad_id}")
        print(f"  BOS ID: {self.bos_id}")
        print(f"  EOS ID: {self.eos_id}")

    @property
    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()

    def encode_text(self, text: str) -> List[int]:
        """Encode raw text to token IDs (unused in TRM pipeline, kept for debug)."""
        return self.sp.EncodeAsIds(text)

    def encode_dsl_tokens(self, token_names: List[str]) -> List[int]:
        """
        Encode DSL token names to IDs via a direct lookup.

        Args:
            token_names: List like ["BOS", "EN_EVT_INIT", "EN_AMOUNT", "INT_24", ...]

        Returns:
            List of token IDs
        """
        ids: List[int] = []
        missing: List[str] = []

        for name in token_names:
            idx = self.name_to_id.get(name)
            if idx is None:
                missing.append(name)
            else:
                ids.append(idx)

        if missing:
            # Fail fast: this should never happen if the SP model was trained
            # from `GyanDSLToken` names.
            raise ValueError(
                f"SentencePiece model is missing DSL tokens (showing up to 10): "
                f"{sorted(set(missing))[:10]}"
            )

        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.sp.DecodeIds(ids)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load examples from JSONL file."""
    examples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def text_to_char_tokens(text: str) -> List[str]:
    """
    Convert English text to CHAR_* token names.
    
    E.g., "Hi!" -> ["CHAR_H_UP", "CHAR_i", "CHAR_EXCLAIM"]
    
    Only ASCII letters a-z, A-Z and digits 0-9 are supported. Non-ASCII
    characters (e.g., ñ, é) are skipped to avoid missing token errors.
    """
    char_tokens = []
    for char in text:
        if char == ' ':
            char_tokens.append("CHAR_SPACE")
        elif char == '.':
            char_tokens.append("CHAR_DOT")
        elif char == ',':
            char_tokens.append("CHAR_COMMA")
        elif char == '?':
            char_tokens.append("CHAR_QUESTION")
        elif char == '!':
            char_tokens.append("CHAR_EXCLAIM")
        elif char == '$':
            char_tokens.append("CHAR_DOLLAR")
        elif char == '-':
            char_tokens.append("CHAR_DASH")
        elif char == "'":
            char_tokens.append("CHAR_APOSTROPHE")
        elif char == '/':
            char_tokens.append("CHAR_SLASH")
        elif char == ':':
            char_tokens.append("CHAR_COLON")
        elif char == ';':
            char_tokens.append("CHAR_SEMICOLON")
        elif char == '(':
            char_tokens.append("CHAR_LPAREN")
        elif char == ')':
            char_tokens.append("CHAR_RPAREN")
        elif char == '\n':
            char_tokens.append("CHAR_NEWLINE")
        elif char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            char_tokens.append(f"CHAR_{char}_UP")
        elif char in 'abcdefghijklmnopqrstuvwxyz':
            char_tokens.append(f"CHAR_{char}")
        elif char in '0123456789':
            char_tokens.append(f"CHAR_{char}")
        else:
            # Unknown char (non-ASCII, special symbols) - skip
            pass
    return char_tokens


def build_example(
    tokenizer: DSLSentencePieceTokenizer,
    question: str,
    en_tokens: List[str],
    seq_len: int,
    english_max_len: int,
) -> Tuple[np.ndarray, np.ndarray] | None:
    """
    Build a single training example.
    
    Format:
        Input:  [English CHAR tokens] [PAD] [PAD...for DSL output]
        Labels: [-100...]             [-100] [DSL tokens...]
    
    Returns:
        (inputs, labels) arrays of shape (seq_len,), or None if invalid
    """
    # Convert English to CHAR_* tokens, then encode with SentencePiece
    char_tokens = text_to_char_tokens(question)
    eng_ids = tokenizer.encode_dsl_tokens(char_tokens)
    
    # Truncate if needed
    if len(eng_ids) > english_max_len:
        eng_ids = eng_ids[:english_max_len]
    eng_len = len(eng_ids)
    
    # Encode DSL output
    dsl_ids = tokenizer.encode_dsl_tokens(en_tokens)
    dsl_len = len(dsl_ids)
    
    # Check if it fits
    # Layout: [English] [SEP/PAD] [DSL output] [PAD...]
    total_needed = eng_len + 1 + dsl_len
    if total_needed > seq_len:
        print(f"Warning: Example too long ({total_needed} > {seq_len}), skipping")
        return None
    
    # Build input array
    inputs = np.full(seq_len, tokenizer.pad_id, dtype=np.int32)
    inputs[:eng_len] = eng_ids
    # Position eng_len is PAD (separator)
    # DSL output positions are also PAD (masked - model must predict)
    
    # Build labels array
    labels = np.full(seq_len, IGNORE_LABEL_ID, dtype=np.int32)
    # English positions: ignore
    # Separator position: ignore
    # DSL output positions: actual DSL tokens
    dsl_start = eng_len + 1
    dsl_end = dsl_start + dsl_len
    labels[dsl_start:dsl_end] = dsl_ids
    
    return inputs, labels


def build_dataset(
    src_path: Path,
    dst_dir: Path,
    sp_model_path: str,
    seq_len: int = 256,
    english_max_len: int = 150,
) -> None:
    """
    Build the full dataset.
    
    Args:
        src_path: Path to source JSONL file
        dst_dir: Output directory for numpy arrays
        sp_model_path: Path to SentencePiece model
        seq_len: Total sequence length
        english_max_len: Max tokens for English input
    """
    # Load tokenizer
    tokenizer = DSLSentencePieceTokenizer(sp_model_path)
    
    # Load examples
    examples = load_jsonl(src_path)
    print(f"Loaded {len(examples)} examples from {src_path}")

    all_inputs = []
    all_labels = []
    all_puzzle_ids = []

    # Build module -> puzzle_id mapping
    modules = sorted(set(ex.get("module", "unknown") for ex in examples))
    module_to_id = {m: i + 1 for i, m in enumerate(modules)}  # 0 reserved for blank
    
    for idx, ex in enumerate(examples):
        question = ex.get("question", "")
        en_tokens = ex.get("en_tokens", [])
        module = ex.get("module", "unknown")
        
        if not question or not en_tokens:
            print(f"Skipping example {idx}: missing question or en_tokens")
            continue

        result = build_example(tokenizer, question, en_tokens, seq_len, english_max_len)
        if result is None:
            continue
        
        inputs, labels = result
        all_inputs.append(inputs)
        all_labels.append(labels)
        all_puzzle_ids.append(module_to_id.get(module, 0))
    
    if not all_inputs:
        print("No valid examples!")
        return
    
    # Convert to numpy arrays
    num_examples = len(all_inputs)
    inputs_arr = np.stack(all_inputs)
    labels_arr = np.stack(all_labels)
    puzzle_ids_arr = np.array(all_puzzle_ids, dtype=np.int32)
    # One example per puzzle and per group (like DSL dataset builder).
    puzzle_indices_arr = np.arange(0, num_examples + 1, dtype=np.int32)
    group_indices_arr = np.arange(0, num_examples + 1, dtype=np.int32)
    
    # Save
    dst_dir.mkdir(parents=True, exist_ok=True)
    np.save(dst_dir / "all__inputs.npy", inputs_arr)
    np.save(dst_dir / "all__labels.npy", labels_arr)
    np.save(dst_dir / "all__puzzle_identifiers.npy", puzzle_ids_arr)
    np.save(dst_dir / "all__puzzle_indices.npy", puzzle_indices_arr)
    np.save(dst_dir / "all__group_indices.npy", group_indices_arr)
    
    # Save metadata
    metadata = {
        "pad_id": int(tokenizer.pad_id),
        "ignore_label_id": IGNORE_LABEL_ID,
        "blank_identifier_id": 0,
        "vocab_size": tokenizer.vocab_size,
        "seq_len": seq_len,
        "num_puzzle_identifiers": len(modules) + 1,  # +1 for blank
        "total_groups": num_examples,
        "total_puzzles": num_examples,
        "mean_puzzle_examples": 1.0,
        "sets": ["all"],
    }
    with open(dst_dir / "dataset.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save module mapping
    with open(dst_dir.parent / "module_mapping.json", "w") as f:
        json.dump(module_to_id, f, indent=2)
    
    # Save identifiers
    id_to_module = {v: k for k, v in module_to_id.items()}
    id_to_module[0] = "blank"
    with open(dst_dir.parent / "identifiers.json", "w") as f:
        json.dump(id_to_module, f, indent=2)
    
    print(f"\nDataset built successfully!")
    print(f"  Examples: {len(all_inputs)}")
    print(f"  Seq length: {seq_len}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Modules: {modules}")
    print(f"  Output: {dst_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build English→DSL TRM dataset")
    parser.add_argument(
        "--src",
        type=str,
        default="data/en_dsl_seq2seq/train.jsonl",
        help="Source JSONL file with question + en_tokens",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="data/english_to_dsl_trm/train",
        help="Output directory for numpy arrays",
    )
    parser.add_argument(
        "--sp_model",
        type=str,
        default="data/en_dsl_seq2seq/en_dsl_spm.model",
        help="Path to SentencePiece model",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=256,
        help="Total sequence length",
    )
    parser.add_argument(
        "--english_max_len",
        type=int,
        default=150,
        help="Max tokens for English input",
    )
    args = parser.parse_args()
    
    build_dataset(
        src_path=Path(args.src),
        dst_dir=Path(args.dst),
        sp_model_path=args.sp_model,
        seq_len=args.seq_len,
        english_max_len=args.english_max_len,
    )


if __name__ == "__main__":
    main()
