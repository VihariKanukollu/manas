"""
ARC-DSL program-level evaluator.

This evaluator assumes a dataset built by `dataset/build_arc_dsl_dataset.py`
where each example contains:
  - an ARC input grid encoded as DSL tokens (english segment)
  - a ground-truth DSL program encoded as token IDs (program segment)

The TRM+ACTLossHead model is trained with:
  inputs = [english, PAD(program)]
  labels = [IGNORE, program_tokens]

So at evaluation time we:
  - request `preds` from the model (full sequence [B, seq_len])
  - slice off the program segment (last program_len tokens)
  - compare to ground-truth program tokens (from `labels` / dataset)

This evaluator reports:
  - token-level accuracy over the program segment
  - exact whole-program accuracy (all tokens match, ignoring PAD)
"""

from typing import Dict, Optional
import os
import json

import numpy as np
import torch

from dataset.common import PuzzleDatasetMetadata
from dsl.tokens import GyanDSLToken


class ARC_DSL_ProgramEvaluator:
    """
    Program-level accuracy evaluator for ARC-DSL models.

    Expected `batch` / `preds` inputs from `pretrain.evaluate`:
      - batch["labels"]: [B, seq_len]
      - preds["preds"]:  [B, seq_len]  (argmax over vocab)

    We assume:
      - The first `english_len` tokens are the ARC grid (labels == IGNORE_LABEL_ID)
      - The last  `program_len` tokens are supervised (labels != IGNORE_LABEL_ID)
    """

    # We need logits->preds; tell `evaluate` to request them.
    required_outputs = {"preds"}

    def __init__(
        self,
        data_path: str,
        eval_metadata: PuzzleDatasetMetadata,
        english_len: int = 900,
        program_len: int = 256,
        decode_examples: int = 0,
    ) -> None:
        """
        Args:
            data_path: Root of the ARC-DSL dataset (for optional decoding).
            eval_metadata: Dataset metadata (seq_len, vocab_size, etc.).
            english_len: Length of the ARC grid segment.
            program_len: Length of the program segment.
            decode_examples: If >0, save a small JSON file with a few
                decoded (target, pred) program pairs for inspection.
        """
        super().__init__()
        self.data_path = data_path
        self.eval_metadata = eval_metadata
        self.english_len = english_len
        self.program_len = program_len
        self.decode_examples = decode_examples

        # Sanity check against metadata
        assert (
            self.english_len + self.program_len == eval_metadata.seq_len
        ), f"english_len+program_len={self.english_len + self.program_len} != seq_len={eval_metadata.seq_len}"

        # Stats
        self._total_tokens = 0
        self._correct_tokens = 0
        self._total_programs = 0
        self._correct_programs = 0

        # Optional storage for decoded examples
        self._decoded_examples = []

    def begin_eval(self) -> None:
        """Reset counters before a new evaluation run."""
        self._total_tokens = 0
        self._correct_tokens = 0
        self._total_programs = 0
        self._correct_programs = 0
        self._decoded_examples = []

    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]) -> None:
        """
        Update metrics from a single evaluation batch.

        Args:
            batch: Dict with at least "labels" and "puzzle_identifiers".
            preds: Dict with "preds" tensor from the model.
        """
        labels = batch["labels"].cpu().numpy()  # [B, seq_len]
        predictions = preds["preds"].cpu().numpy()  # [B, seq_len]

        # Program segment is the last `program_len` tokens
        prog_labels = labels[:, self.english_len :]
        prog_preds = predictions[:, self.english_len :]

        # Mask: positions where we have a real (non-IGNORE, non-PAD) label.
        # In our ARC-DSL dataset, labels are:
        #   - -100 outside the program segment
        #   - PAD_ID on padded program positions
        #   - true token IDs on real program positions
        IGNORE = -100
        PAD_ID = GyanDSLToken.PAD.value
        mask = (prog_labels != IGNORE) & (prog_labels != PAD_ID)

        # Token-level accuracy
        correct = (prog_labels == prog_preds) & mask
        num_correct = int(correct.sum())
        num_tokens = int(mask.sum())

        self._correct_tokens += num_correct
        self._total_tokens += num_tokens

        # Whole-program accuracy (all supervised positions match)
        # We ignore PAD positions (labels == IGNORE or PAD_ID)
        # Treat sequences with no supervised tokens as skipped.
        if num_tokens > 0:
            # For each sequence, check if all supervised tokens are correct.
            per_seq_correct = correct.sum(axis=1) == mask.sum(axis=1)

            self._total_programs += int(mask.any(axis=1).sum())
            self._correct_programs += int(per_seq_correct.sum())

        # Optionally record a few decoded examples for inspection
        if self.decode_examples > 0 and len(self._decoded_examples) < self.decode_examples:
            # Build reverse vocab
            id_to_name = {tok.value: tok.name for tok in GyanDSLToken}

            for lbl_seq, pred_seq, m in zip(prog_labels, prog_preds, mask):
                if len(self._decoded_examples) >= self.decode_examples:
                    break

                # Extract only supervised tokens
                target_tokens = [id_to_name.get(int(t), f"UNK_{int(t)}") for t, mm in zip(lbl_seq, m) if mm]
                pred_tokens = [id_to_name.get(int(t), f"UNK_{int(t)}") for t, mm in zip(pred_seq, m) if mm]

                self._decoded_examples.append(
                    {
                        "target_program": target_tokens,
                        "pred_program": pred_tokens,
                    }
                )

    def result(
        self,
        save_path: Optional[str],
        rank: int,
        world_size: int,
        group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Finalize and return metrics.

        Returns:
            dict with:
              - "ARC_DSL/token_accuracy"
              - "ARC_DSL/program_accuracy"
        """
        if rank != 0:
            return None

        if self._total_tokens == 0 or self._total_programs == 0:
            return None

        token_acc = self._correct_tokens / max(1, self._total_tokens)
        prog_acc = self._correct_programs / max(1, self._total_programs)

        metrics = {
            "ARC_DSL/token_accuracy": float(token_acc),
            "ARC_DSL/program_accuracy": float(prog_acc),
        }

        # Optionally save decoded programs
        if save_path is not None and self._decoded_examples:
            os.makedirs(save_path, exist_ok=True)
            out_file = os.path.join(save_path, "arc_dsl_programs.json")
            with open(out_file, "w") as f:
                json.dump(self._decoded_examples, f, indent=2)

        return metrics


