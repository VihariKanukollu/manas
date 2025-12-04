#!/usr/bin/env python
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
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load base tokenizer and extend it with EN-DSL symbols as special tokens.
    vocab_path = "data/en_dsl_seq2seq/en_dsl_vocab.txt"
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    en_dsl_tokens = load_en_dsl_tokens(vocab_path)
    specials = {"additional_special_tokens": en_dsl_tokens}
    num_added = tokenizer.add_special_tokens(specials)
    print(f"Added {num_added} EN-DSL tokens to tokenizer")

    model = T5ForConditionalGeneration.from_pretrained(model_name)
    # Resize embeddings so newly added tokens are usable.
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    train_ds = EnToDslSeq2SeqDataset(train_path, tokenizer, max_input_len, max_target_len)
    val_ds = EnToDslSeq2SeqDataset(val_path, tokenizer, max_input_len, max_target_len)

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


