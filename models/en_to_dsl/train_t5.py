#!/usr/bin/env python
"""
Train T5 for English -> DSL translation.

Uses:
- T5's BPE tokenizer for English input
- DSL tokens added as special tokens (from our SentencePiece vocab)
- This ensures DSL tokens like INT_24, EN_EVT_GAIN are atomic, never split
"""
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    get_cosine_schedule_with_warmup,
)

from models.en_to_dsl.dataset import EnToDslSeq2SeqDataset


def load_dsl_vocab_from_sp(sp_vocab_path: str) -> list[str]:
    """
    Load DSL token names from SentencePiece vocab file.
    
    The SP vocab file has lines like:
        <unk>	0
        <s>	0
        </s>	0
        ▁BOS	-2.89289
        ▁EN_EVT_INIT	-4.3745
        ...
    
    We extract the token names (stripping the ▁ prefix for user-defined pieces).
    """
    tokens: list[str] = []
    path = Path(sp_vocab_path)
    
    # Skip SP special tokens that T5 already has
    skip_tokens = {"<unk>", "<s>", "</s>", "<pad>"}
    
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: piece<tab>score
            parts = line.split("\t")
            if len(parts) < 1:
                continue
            piece = parts[0]
            
            # Skip SP meta tokens
            if piece in skip_tokens:
                continue
            
            # Remove the ▁ (word boundary) prefix that SP adds
            if piece.startswith("▁"):
                piece = piece[1:]
            
            if piece:
                tokens.append(piece)
    
    return tokens


def load_en_dsl_tokens(vocab_path: str) -> list[str]:
    """Load EN-DSL token names (one per line) from a vocab text file."""
    tokens: list[str] = []
    path = Path(vocab_path)
    with path.open() as f:
        for line in f:
            t = line.strip()
            if t:
                tokens.append(t)
    return tokens


def train(
    model_name: str = "t5-small",
    train_path: str = "data/en_dsl_seq2seq/train.jsonl",
    val_path: str = "data/en_dsl_seq2seq/valid.jsonl",
    out_dir: str = "checkpoints/en_to_dsl_t5_small",
    batch_size: int = 32,
    num_epochs: int = 3,
    lr: float = 3e-4,
    max_input_len: int = 128,
    max_target_len: int = 128,
    warmup_steps: int = 1000,
    grad_accum_steps: int = 1,
    num_workers: int = 4,
    use_sp_vocab: bool = True,  # Use SentencePiece vocab (recommended)
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load base T5 tokenizer
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    
    # Load DSL vocabulary - prefer SentencePiece vocab for full coverage
    if use_sp_vocab:
        sp_vocab_path = "data/en_dsl_seq2seq/en_dsl_spm.vocab"
        if Path(sp_vocab_path).exists():
            dsl_tokens = load_dsl_vocab_from_sp(sp_vocab_path)
            print(f"Loaded {len(dsl_tokens)} DSL tokens from SentencePiece vocab")
        else:
            print(f"Warning: SP vocab not found at {sp_vocab_path}, falling back to text vocab")
            dsl_tokens = load_en_dsl_tokens("data/en_dsl_seq2seq/en_dsl_vocab.txt")
    else:
        vocab_path = "data/en_dsl_seq2seq/en_dsl_vocab.txt"
        dsl_tokens = load_en_dsl_tokens(vocab_path)
        print(f"Loaded {len(dsl_tokens)} DSL tokens from text vocab")
    
    # Add DSL tokens as special tokens - this ensures they're NEVER split
    specials = {"additional_special_tokens": dsl_tokens}
    num_added = tokenizer.add_special_tokens(specials)
    print(f"Added {num_added} DSL tokens to T5 tokenizer")

    # Load model and resize embeddings for new tokens
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    print(f"Model vocab size: {len(tokenizer)}")

    # Create datasets
    train_ds = EnToDslSeq2SeqDataset(train_path, tokenizer, max_input_len, max_target_len)
    val_ds = EnToDslSeq2SeqDataset(val_path, tokenizer, max_input_len, max_target_len)
    print(f"Train examples: {len(train_ds)}, Val examples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_steps = (len(train_loader) * num_epochs) // grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    running_loss = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            running_loss += loss.item() * grad_accum_steps

            if global_step > 0 and global_step % 100 == 0:
                avg_loss = running_loss / 100
                print(f"Epoch {epoch} | step {global_step} | loss {avg_loss:.4f}")
                running_loss = 0.0

        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} finished. Val loss = {val_loss:.4f}")

        ckpt_path = out_dir / f"epoch_{epoch}"
        ckpt_path.mkdir(exist_ok=True)
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Training complete. Saved to {out_dir}")


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += outputs.loss.item()
        total_batches += 1
    return total_loss / max(1, total_batches)


if __name__ == "__main__":
    train()
