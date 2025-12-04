"""
Rule-based English → EN-DSL parser for very narrow word-problem patterns.

This module deliberately handles only a tiny canonical pattern for v0:

    "<Name> had A apples. <Name> bought B more apples. "
    "How many apples does <Name> have now?"

All names and item types must match (case-insensitively), and A, B must be
non-negative integers within the existing INT_* range.

The parser produces a sequence of GyanDSLToken values that encode:

    BOS
      EN_EVENT EN_EVT_INIT
        EN_ROLE_AGENT EN_ENTITY INT_0
        EN_ROLE_THEME EN_UNIT   INT_0
        EN_AMOUNT      INT_A
      EN_EVENT EN_EVT_GAIN
        EN_ROLE_AGENT EN_ENTITY INT_0
        EN_ROLE_THEME EN_UNIT   INT_0
        EN_AMOUNT      INT_B
      EN_QUERY EN_Q_HOW_MANY
        EN_ROLE_AGENT EN_ENTITY INT_0
        EN_ROLE_THEME EN_UNIT   INT_0
    EOS

Where INT_A / INT_B are standard integer-constant tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .tokens import (
    GyanDSLToken,
    NUM_INT_CONSTS,
    get_int_const_token,
)


_INIT_GAIN_PATTERN = re.compile(
    r"""
    ^\s*
    (?P<name>[A-Z][a-zA-Z]*)\s+had\s+
    (?P<a>\d+)\s+
    (?P<item>[a-zA-Z]+)\.\s+
    (?P<name2>[A-Z][a-zA-Z]*)\s+
    (?:bought|buys|purchased|got)\s+
    (?P<b>\d+)\s+more\s+
    (?P<item2>[a-zA-Z]+)\.\s+
    How\s+many\s+
    (?P<item3>[a-zA-Z]+)\s+
    does\s+
    (?P<name3>[A-Z][a-zA-Z]*)\s+
    have\s+now\?
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass
class ParsedInitGainProblem:
    """Structured metadata for the narrow INIT+GAIN+HOW_MANY pattern."""

    entity_name: str
    entity_id: int
    unit_name: str
    unit_id: int
    init_amount: int
    gain_amount: int


def _parse_canonical_init_gain(problem: str) -> ParsedInitGainProblem:
    """
    Parse the canonical "X had A apples, X bought B more apples, how many now?"
    pattern. Raises ValueError on any mismatch.
    """

    m = _INIT_GAIN_PATTERN.match(problem)
    if not m:
        raise ValueError("Problem does not match INIT+GAIN+HOW_MANY canonical pattern")

    name = m.group("name")
    name2 = m.group("name2")
    name3 = m.group("name3")
    item = m.group("item")
    item2 = m.group("item2")
    item3 = m.group("item3")

    # Require consistent entity / item references (case-insensitive).
    if not (
        name.lower() == name2.lower() == name3.lower()
        and item.lower() == item2.lower() == item3.lower()
    ):
        raise ValueError("Entity or item names are inconsistent across sentences")

    try:
        init_amount = int(m.group("a"))
        gain_amount = int(m.group("b"))
    except ValueError as exc:
        raise ValueError("Failed to parse integer quantities from problem") from exc

    if not (0 <= init_amount < NUM_INT_CONSTS and 0 <= gain_amount < NUM_INT_CONSTS):
        raise ValueError(
            f"Quantities {init_amount}, {gain_amount} out of supported INT_* range "
            f"[0, {NUM_INT_CONSTS})"
        )

    # For v0 we simply assign entity_id = 0, unit_id = 0.
    return ParsedInitGainProblem(
        entity_name=name,
        entity_id=0,
        unit_name=item,
        unit_id=0,
        init_amount=init_amount,
        gain_amount=gain_amount,
    )


def parse_init_gain_how_many(
    problem: str,
) -> Tuple[List[GyanDSLToken], Dict[str, Any]]:
    """
    Parse a narrow English word problem into EN-DSL tokens for INIT+GAIN+HOW_MANY.

    Returns:
        tokens: List[GyanDSLToken]  – full sequence including BOS/EOS
        meta:   Dict[str, Any]      – structured metadata for debugging / logging

    Raises:
        ValueError if the input does not match the supported pattern or if any
        quantity lies outside the small INT_* range.
    """

    parsed = _parse_canonical_init_gain(problem)

    ent_id = parsed.entity_id
    unit_id = parsed.unit_id

    ent_tok = get_int_const_token(ent_id)
    unit_tok = get_int_const_token(unit_id)

    tokens: List[GyanDSLToken] = [GyanDSLToken.BOS]

    # INIT event: X has A apples initially.
    tokens.extend(
        [
            GyanDSLToken.EN_EVENT,
            GyanDSLToken.EN_EVT_INIT,
            GyanDSLToken.EN_ROLE_AGENT,
            GyanDSLToken.EN_ENTITY,
            ent_tok,
            GyanDSLToken.EN_ROLE_THEME,
            GyanDSLToken.EN_UNIT,
            unit_tok,
            GyanDSLToken.EN_AMOUNT,
            get_int_const_token(parsed.init_amount),
        ]
    )

    # GAIN event: X buys B more apples.
    tokens.extend(
        [
            GyanDSLToken.EN_EVENT,
            GyanDSLToken.EN_EVT_GAIN,
            GyanDSLToken.EN_ROLE_AGENT,
            GyanDSLToken.EN_ENTITY,
            ent_tok,
            GyanDSLToken.EN_ROLE_THEME,
            GyanDSLToken.EN_UNIT,
            unit_tok,
            GyanDSLToken.EN_AMOUNT,
            get_int_const_token(parsed.gain_amount),
        ]
    )

    # HOW_MANY query: how many apples does X have now?
    tokens.extend(
        [
            GyanDSLToken.EN_QUERY,
            GyanDSLToken.EN_Q_HOW_MANY,
            GyanDSLToken.EN_ROLE_AGENT,
            GyanDSLToken.EN_ENTITY,
            ent_tok,
            GyanDSLToken.EN_ROLE_THEME,
            GyanDSLToken.EN_UNIT,
            unit_tok,
        ]
    )

    tokens.append(GyanDSLToken.EOS)

    meta: Dict[str, Any] = {
        "entity_name": parsed.entity_name,
        "entity_id": parsed.entity_id,
        "unit_name": parsed.unit_name,
        "unit_id": parsed.unit_id,
        "init_amount": parsed.init_amount,
        "gain_amount": parsed.gain_amount,
    }

    return tokens, meta


__all__ = [
    "ParsedInitGainProblem",
    "parse_init_gain_how_many",
]


