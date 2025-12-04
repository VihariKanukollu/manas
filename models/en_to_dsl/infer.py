#!/usr/bin/env python
from typing import List, Optional

import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast


class EnToDslModel:
    def __init__(self, ckpt_dir: str, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = T5TokenizerFast.from_pretrained(ckpt_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(ckpt_dir).to(self.device)
        self.model.eval()

        # Set of valid EN-DSL tokens (added as additional_special_tokens).
        self.en_dsl_tokens = set(self.tokenizer.additional_special_tokens)

    @torch.no_grad()
    def encode_question(self, question: str, max_new_tokens: int = 128) -> List[str]:
        enc = self.tokenizer(question, return_tensors="pt").to(self.device)
        gen_ids = self.model.generate(
            **enc,
            max_length=max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )
        # Do NOT skip special tokens here, because EN-DSL symbols are registered
        # as special tokens. Instead, decode everything and then filter to the
        # EN-DSL vocabulary.
        out_str = self.tokenizer.decode(gen_ids[0], skip_special_tokens=False)
        raw_tokens = out_str.strip().split()
        token_names = [t for t in raw_tokens if t in self.en_dsl_tokens]
        return token_names


if __name__ == "__main__":
    model = EnToDslModel("checkpoints/en_to_dsl_t5_small")

    q = "Calculate the greatest common factor of 6 and 426."
    en_tokens = model.encode_question(q)
    print("EN-DSL:", en_tokens)


