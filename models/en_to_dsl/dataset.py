from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import json
from torch.utils.data import Dataset


@dataclass
class EnDslExample:
    id: str
    module: str
    question: str
    en_tokens: List[str]


def load_jsonl(path: Path) -> List[EnDslExample]:
    items: List[EnDslExample] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            items.append(
                EnDslExample(
                    id=str(ex.get("id")),
                    module=ex.get("module", ""),
                    question=ex.get("question", ""),
                    en_tokens=ex.get("en_tokens", []),
                )
            )
    return items


class EnToDslSeq2SeqDataset(Dataset):
    """
    Wraps (question -> en_tokens) for a seq2seq model, e.g. T5.

    Input: text question
    Output: space-separated EN token names "BOS EN_QUERY ..."
    """

    def __init__(self, path: str, tokenizer, max_input_len: int = 128, max_target_len: int = 128):
        self.path = Path(path)
        self.examples = load_jsonl(self.path)
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]

        # Input: just the question
        input_enc = self.tokenizer(
            ex.question,
            padding="max_length",
            truncation=True,
            max_length=self.max_input_len,
            return_tensors="pt",
        )

        # Target: EN token names joined by spaces
        target_str = " ".join(ex.en_tokens)
        target_enc = self.tokenizer(
            target_str,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_len,
            return_tensors="pt",
        )

        labels = target_enc["input_ids"].squeeze(0)
        # Ignore loss on padding positions (HuggingFace convention: -100).
        pad_token_id = self.tokenizer.pad_token_id
        labels[labels == pad_token_id] = -100

        item = {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels": labels,
        }
        return item


