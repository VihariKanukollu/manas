#!/usr/bin/env python
"""
Inference script for English -> DSL translation.

Usage:
    python -m models.en_to_dsl.infer --checkpoint checkpoints/en_to_dsl_t5_small
    
    # Or with a specific question:
    python -m models.en_to_dsl.infer --checkpoint checkpoints/en_to_dsl_t5_small \
        --question "Natalia sold clips to 48 of her friends..."
"""
import argparse
from typing import List, Optional

import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast


class EnToDslModel:
    """English to DSL translation model using fine-tuned T5."""
    
    def __init__(self, ckpt_dir: str, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = T5TokenizerFast.from_pretrained(ckpt_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(ckpt_dir).to(self.device)
        self.model.eval()

        # Set of valid EN-DSL tokens (added as additional_special_tokens).
        self.en_dsl_tokens = set(self.tokenizer.additional_special_tokens)
        # Build token_id -> token_name mapping for EN-DSL tokens
        self.id_to_token = {
            self.tokenizer.convert_tokens_to_ids(t): t
            for t in self.en_dsl_tokens
        }
        
        print(f"Loaded model from {ckpt_dir}")
        print(f"  Device: {self.device}")
        print(f"  Vocab size: {len(self.tokenizer)}")
        print(f"  DSL tokens: {len(self.en_dsl_tokens)}")

    @torch.no_grad()
    def translate(
        self, 
        question: str, 
        max_new_tokens: int = 128, 
        num_beams: int = 1,
        return_raw: bool = False,
    ) -> List[str]:
        """
        Translate an English question to DSL tokens.
        
        Args:
            question: The English question text
            max_new_tokens: Maximum output length
            num_beams: Beam search width (1 = greedy)
            return_raw: If True, return raw decoded string instead of token list
            
        Returns:
            List of DSL token names (e.g., ["BOS", "EN_EVT_INIT", "EN_AMOUNT", "INT_24", ...])
        """
        enc = self.tokenizer(question, return_tensors="pt").to(self.device)
        
        gen_ids = self.model.generate(
            **enc,
            max_length=max_new_tokens,
            num_beams=num_beams,
            early_stopping=num_beams > 1,
        )
        
        if return_raw:
            return self.tokenizer.decode(gen_ids[0], skip_special_tokens=False)
        
        # Decode token-by-token to extract EN-DSL tokens.
        # String-based decode+split fails because special tokens get concatenated.
        token_names = []
        for tid in gen_ids[0].tolist():
            if tid in self.id_to_token:
                token_names.append(self.id_to_token[tid])
        return token_names

    # Alias for backwards compatibility
    encode_question = translate


def main():
    parser = argparse.ArgumentParser(description="English to DSL inference")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="checkpoints/en_to_dsl_t5_small",
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question to translate (if not provided, uses test examples)"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Beam search width (1 = greedy decoding)"
    )
    args = parser.parse_args()
    
    # Load model
    model = EnToDslModel(args.checkpoint)
    
    # Test questions
    if args.question:
        questions = [args.question]
    else:
        questions = [
            "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            "Calculate the greatest common factor of 6 and 426.",
        ]
    
    print("\n" + "="*80)
    print("INFERENCE RESULTS")
    print("="*80)
    
    for i, q in enumerate(questions):
        print(f"\n[{i+1}] Question:")
        print(f"    {q[:100]}{'...' if len(q) > 100 else ''}")
        
        # Get DSL tokens
        dsl_tokens = model.translate(q, num_beams=args.num_beams)
        
        print(f"\n    DSL Output ({len(dsl_tokens)} tokens):")
        print(f"    {' '.join(dsl_tokens)}")
        
        # Also show raw decoded output for comparison
        raw_output = model.translate(q, num_beams=args.num_beams, return_raw=True)
        print(f"\n    Raw decoded:")
        print(f"    {raw_output[:200]}{'...' if len(raw_output) > 200 else ''}")
        
        print("-"*80)


if __name__ == "__main__":
    main()
