"""
EN-DSL runtime / interpreter utilities.

This module provides a tiny, deterministic interpreter for a *subset* of the
English/world semantics tokens defined in `dsl.tokens.GyanDSLToken`.

The current scope is intentionally narrow and surgical:

- World state tracks simple inventory-style quantities for (entity, unit) pairs.
- Supported event types:
    - EN_EVT_INIT      – initialise a quantity
    - EN_EVT_GAIN      – gain / receive items
    - EN_EVT_LOSS      – lose / use items
    - EN_EVT_TRANSFER  – transfer items between entities
- Supported query type:
    - EN_Q_HOW_MANY    – “how many UNIT does ENTITY have?”

Events and queries are encoded as sequences of EN_* tokens plus small integer
IDs (INT_k) for entity / unit indices and amounts, e.g.:

    BOS
      EN_EVENT EN_EVT_INIT
        EN_ROLE_AGENT EN_ENTITY INT_0
        EN_ROLE_THEME EN_UNIT   INT_0
        EN_AMOUNT      INT_5
      EN_EVENT EN_EVT_GAIN
        EN_ROLE_AGENT EN_ENTITY INT_0
        EN_ROLE_THEME EN_UNIT   INT_0
        EN_AMOUNT      INT_3
      EN_QUERY EN_Q_HOW_MANY
        EN_ROLE_AGENT EN_ENTITY INT_0
        EN_ROLE_THEME EN_UNIT   INT_0
    EOS

The interpreter turns such streams into:

- A world state mapping (entity_id, unit_id) -> integer quantity.
- For each (entity_id, unit_id), a postfix (RPN) arithmetic expression over
  INT_* / ADD / SUB tokens that explains how the final quantity was obtained.

This stays fully compatible with the existing math DSL:
we *reuse* the existing integer and arithmetic tokens, and we do not touch
any of the dataset builders or model code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .tokens import (
    GyanDSLToken,
    NUM_INT_CONSTS,
    get_int_const_token,
)


EntityId = int
UnitId = int
Amount = int
WorldKey = Tuple[EntityId, UnitId]


@dataclass
class ENEvent:
    """
    Structured representation of a single EN event.

    Only the small subset needed for inventory-style word problems is modeled
    here; additional roles/fields can be added later as needed.
    """

    event_type: GyanDSLToken
    agent: Optional[EntityId] = None
    theme_unit: Optional[UnitId] = None
    amount: Optional[Amount] = None
    source: Optional[EntityId] = None
    dest: Optional[EntityId] = None


@dataclass
class ENQuery:
    """
    Structured representation of a single EN query.

    Currently we only support EN_Q_HOW_MANY queries over a single
    (entity, unit) pair.
    """

    query_type: GyanDSLToken
    agent: Optional[EntityId] = None
    theme_unit: Optional[UnitId] = None


@dataclass
class ENWorldState:
    """
    World state tracked by the EN-DSL interpreter.

    - quantities[(entity_id, unit_id)] -> integer amount
    - expr_tokens[(entity_id, unit_id)] -> postfix arithmetic expression
      over existing math DSL tokens (INT_*, ADD, SUB).
    """

    quantities: Dict[WorldKey, Amount] = field(default_factory=dict)
    expr_tokens: Dict[WorldKey, List[GyanDSLToken]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Integer helpers (local to EN tooling)
# ---------------------------------------------------------------------------

def en_int_token_to_value(token: GyanDSLToken) -> int:
    """
    Decode a small integer from an INT_* or INT_NEG* token.

    This is deliberately a *local* helper: it does not change or replace the
    global int_to_tokens logic used in dataset builders.
    """

    name = token.name
    if name.startswith("INT_") and name not in {
        "INT_NEG1",
        "INT_NEG2",
        "INT_NEG10",
        "INT_NEG100",
    }:
        try:
            return int(name.split("_", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Cannot decode integer from token {token}") from exc

    if token is GyanDSLToken.INT_NEG1:
        return -1
    if token is GyanDSLToken.INT_NEG2:
        return -2
    if token is GyanDSLToken.INT_NEG10:
        return -10
    if token is GyanDSLToken.INT_NEG100:
        return -100

    raise ValueError(f"Unsupported integer token for EN runtime: {token}")


def en_value_to_simple_int_tokens(value: int) -> List[GyanDSLToken]:
    """
    Encode a small integer as a minimal list of INT_* / INT_NEG* tokens.

    This is intentionally much simpler than dev.gen_full_math.int_to_tokens:
    it only handles the range actually used in EN examples and tests.
    """

    if 0 <= value < NUM_INT_CONSTS:
        return [get_int_const_token(value)]

    if value in (-1, -2, -10, -100):
        return [get_int_const_token(value)]

    raise ValueError(
        f"Value {value} is out of range for en_value_to_simple_int_tokens; "
        "EN examples should keep quantities within the small INT_* range."
    )


# ---------------------------------------------------------------------------
# Core interpreter
# ---------------------------------------------------------------------------

def _get_expr_list(state: ENWorldState, key: WorldKey) -> List[GyanDSLToken]:
    """Get (and create if needed) the expression token list for a key."""

    if key not in state.expr_tokens:
        state.expr_tokens[key] = []
    return state.expr_tokens[key]


def apply_event(state: ENWorldState, event: ENEvent) -> None:
    """
    Apply a single ENEvent to the world state.

    Semantics (for a key (entity, unit)):
      - EN_EVT_INIT:    amount := value; expr := [INT_amount]
      - EN_EVT_GAIN:    amount += value; expr += [INT_value, ADD]
      - EN_EVT_LOSS:    amount -= value; expr += [INT_value, SUB]
      - EN_EVT_TRANSFER:
            quantities[(source, unit)] -= value; expr += [INT_value, SUB]
            quantities[(dest, unit)]   += value; expr += [INT_value, ADD]
    """

    etype = event.event_type

    if etype is GyanDSLToken.EN_EVT_INIT:
        if event.agent is None or event.theme_unit is None or event.amount is None:
            raise ValueError(f"EN_EVT_INIT requires agent, theme_unit, amount: {event}")
        key: WorldKey = (event.agent, event.theme_unit)
        state.quantities[key] = event.amount
        state.expr_tokens[key] = en_value_to_simple_int_tokens(event.amount)
        return

    if etype is GyanDSLToken.EN_EVT_GAIN:
        if event.agent is None or event.theme_unit is None or event.amount is None:
            raise ValueError(f"EN_EVT_GAIN requires agent, theme_unit, amount: {event}")
        key = (event.agent, event.theme_unit)
        prev = state.quantities.get(key, 0)
        new_val = prev + event.amount
        state.quantities[key] = new_val
        expr = _get_expr_list(state, key)
        expr.extend(en_value_to_simple_int_tokens(event.amount))
        expr.append(GyanDSLToken.ADD)
        return

    if etype is GyanDSLToken.EN_EVT_LOSS:
        if event.agent is None or event.theme_unit is None or event.amount is None:
            raise ValueError(f"EN_EVT_LOSS requires agent, theme_unit, amount: {event}")
        key = (event.agent, event.theme_unit)
        prev = state.quantities.get(key, 0)
        new_val = prev - event.amount
        state.quantities[key] = new_val
        expr = _get_expr_list(state, key)
        expr.extend(en_value_to_simple_int_tokens(event.amount))
        expr.append(GyanDSLToken.SUB)
        return

    if etype is GyanDSLToken.EN_EVT_TRANSFER:
        if (
            event.source is None
            or event.dest is None
            or event.theme_unit is None
            or event.amount is None
        ):
            raise ValueError(
                "EN_EVT_TRANSFER requires source, dest, theme_unit, amount: "
                f"{event}"
            )

        # Source loses
        src_key: WorldKey = (event.source, event.theme_unit)
        src_prev = state.quantities.get(src_key, 0)
        state.quantities[src_key] = src_prev - event.amount
        src_expr = _get_expr_list(state, src_key)
        src_expr.extend(en_value_to_simple_int_tokens(event.amount))
        src_expr.append(GyanDSLToken.SUB)

        # Destination gains
        dst_key: WorldKey = (event.dest, event.theme_unit)
        dst_prev = state.quantities.get(dst_key, 0)
        state.quantities[dst_key] = dst_prev + event.amount
        dst_expr = _get_expr_list(state, dst_key)
        dst_expr.extend(en_value_to_simple_int_tokens(event.amount))
        dst_expr.append(GyanDSLToken.ADD)
        return

    raise NotImplementedError(f"Unsupported EN event type for runtime: {etype}")


def simulate_en_program(events: List[ENEvent]) -> ENWorldState:
    """
    Execute a list of ENEvent objects and return the resulting world state.
    """

    state = ENWorldState()
    for ev in events:
        apply_event(state, ev)
    return state


def answer_query(
    state: ENWorldState,
    query: ENQuery,
    return_expr: bool = False,
) -> Tuple[int, List[GyanDSLToken]] | int:
    """
    Answer a single ENQuery against a world state.

    Currently supports only EN_Q_HOW_MANY over a single (entity, unit) pair.

    If return_expr=True, returns a (numeric_answer, expr_tokens) pair, where
    expr_tokens is the RPN arithmetic expression for the quantity. If the
    expression trace is missing, it falls back to a trivial [INT_answer] trace.
    """

    if query.query_type is not GyanDSLToken.EN_Q_HOW_MANY:
        raise NotImplementedError(f"Unsupported EN query type: {query.query_type}")

    if query.agent is None or query.theme_unit is None:
        raise ValueError(f"EN_Q_HOW_MANY query requires agent and theme_unit: {query}")

    key: WorldKey = (query.agent, query.theme_unit)
    value = state.quantities.get(key, 0)
    expr = state.expr_tokens.get(key)
    if expr is None:
        expr = en_value_to_simple_int_tokens(value)

    if return_expr:
        return value, list(expr)
    return value


def build_expression_tokens(
    state: ENWorldState,
    query: ENQuery,
) -> List[GyanDSLToken]:
    """
    Convenience wrapper: return only the expression tokens for a query.
    """

    _, expr = answer_query(state, query, return_expr=True)  # type: ignore[misc]
    return expr


# ---------------------------------------------------------------------------
# Parsing raw EN token streams into structured events/queries
# ---------------------------------------------------------------------------

_EVENT_TYPE_TOKENS = {
    GyanDSLToken.EN_EVT_INIT,
    GyanDSLToken.EN_EVT_GAIN,
    GyanDSLToken.EN_EVT_LOSS,
    GyanDSLToken.EN_EVT_TRANSFER,
}


def parse_en_program(tokens: List[GyanDSLToken]) -> Tuple[List[ENEvent], List[ENQuery]]:
    """
    Parse a flat sequence of EN tokens into structured events and queries.

    Assumptions (v0):
      - Tokens come from a single short problem, bracketed by BOS/EOS.
      - Only the subset of EN tokens used by our rule-based generator appears.
      - Events use the pattern:
            EN_EVENT <event_type>
              EN_ROLE_AGENT EN_ENTITY INT_k
              EN_ROLE_THEME EN_UNIT   INT_m
              [EN_ROLE_SOURCE EN_ENTITY INT_i]
              [EN_ROLE_DEST   EN_ENTITY INT_j]
              EN_AMOUNT INT_a
      - Queries use the pattern:
            EN_QUERY EN_Q_HOW_MANY
              EN_ROLE_AGENT EN_ENTITY INT_k
              EN_ROLE_THEME EN_UNIT   INT_m
    """

    # Strip structural tokens; they are not part of EN semantics here.
    core: List[GyanDSLToken] = [
        t for t in tokens if t not in (GyanDSLToken.PAD, GyanDSLToken.BOS, GyanDSLToken.EOS)
    ]

    events: List[ENEvent] = []
    queries: List[ENQuery] = []

    i = 0
    n = len(core)

    while i < n:
        tok = core[i]

        if tok is GyanDSLToken.EN_EVENT:
            if i + 1 >= n:
                raise ValueError("EN_EVENT at end of sequence without event type")
            event_type = core[i + 1]
            if event_type not in _EVENT_TYPE_TOKENS:
                raise ValueError(f"Unsupported EN_EVENT type: {event_type}")
            i += 2

            agent: Optional[EntityId] = None
            theme_unit: Optional[UnitId] = None
            amount: Optional[Amount] = None
            source: Optional[EntityId] = None
            dest: Optional[EntityId] = None

            while i < n and core[i] not in (GyanDSLToken.EN_EVENT, GyanDSLToken.EN_QUERY):
                role = core[i]

                if role in (
                    GyanDSLToken.EN_ROLE_AGENT,
                    GyanDSLToken.EN_ROLE_THEME,
                    GyanDSLToken.EN_ROLE_SOURCE,
                    GyanDSLToken.EN_ROLE_DEST,
                ):
                    if i + 3 > n:
                        raise ValueError(
                            f"Incomplete role triple after {role} in EN_EVENT segment"
                        )
                    value_kind = core[i + 1]
                    value_token = core[i + 2]

                    if value_kind is GyanDSLToken.EN_ENTITY:
                        value = en_int_token_to_value(value_token)
                        if role is GyanDSLToken.EN_ROLE_AGENT:
                            agent = value
                        elif role is GyanDSLToken.EN_ROLE_SOURCE:
                            source = value
                        elif role is GyanDSLToken.EN_ROLE_DEST:
                            dest = value
                        else:
                            raise ValueError(
                                f"Unexpected combination of role {role} and EN_ENTITY"
                            )
                    elif value_kind is GyanDSLToken.EN_UNIT:
                        if role is not GyanDSLToken.EN_ROLE_THEME:
                            raise ValueError(
                                f"EN_UNIT should only appear with EN_ROLE_THEME, got {role}"
                            )
                        theme_unit = en_int_token_to_value(value_token)
                    else:
                        raise ValueError(
                            f"Unexpected value kind {value_kind} after role {role}"
                        )

                    i += 3
                    continue

                if role is GyanDSLToken.EN_AMOUNT:
                    if i + 1 >= n:
                        raise ValueError("EN_AMOUNT missing following INT_k token")
                    amount = en_int_token_to_value(core[i + 1])
                    i += 2
                    continue

                # For v0 we fail fast on any other EN_* token inside an event.
                raise ValueError(f"Unexpected token inside EN_EVENT segment: {role}")

            events.append(
                ENEvent(
                    event_type=event_type,
                    agent=agent,
                    theme_unit=theme_unit,
                    amount=amount,
                    source=source,
                    dest=dest,
                )
            )
            continue

        if tok is GyanDSLToken.EN_QUERY:
            if i + 1 >= n:
                raise ValueError("EN_QUERY at end of sequence without query type")
            query_type = core[i + 1]
            if query_type is not GyanDSLToken.EN_Q_HOW_MANY:
                raise NotImplementedError(f"Unsupported EN_QUERY type: {query_type}")
            i += 2

            agent_q: Optional[EntityId] = None
            theme_unit_q: Optional[UnitId] = None

            while i < n and core[i] not in (GyanDSLToken.EN_EVENT, GyanDSLToken.EN_QUERY):
                role = core[i]

                if role in (GyanDSLToken.EN_ROLE_AGENT, GyanDSLToken.EN_ROLE_THEME):
                    if i + 3 > n:
                        raise ValueError(
                            f"Incomplete role triple after {role} in EN_QUERY segment"
                        )
                    value_kind = core[i + 1]
                    value_token = core[i + 2]

                    if value_kind is GyanDSLToken.EN_ENTITY:
                        if role is not GyanDSLToken.EN_ROLE_AGENT:
                            raise ValueError(
                                f"EN_ENTITY should only appear with EN_ROLE_AGENT in queries, got {role}"
                            )
                        agent_q = en_int_token_to_value(value_token)
                    elif value_kind is GyanDSLToken.EN_UNIT:
                        if role is not GyanDSLToken.EN_ROLE_THEME:
                            raise ValueError(
                                f"EN_UNIT should only appear with EN_ROLE_THEME in queries, got {role}"
                            )
                        theme_unit_q = en_int_token_to_value(value_token)
                    else:
                        raise ValueError(
                            f"Unexpected value kind {value_kind} after role {role}"
                        )

                    i += 3
                    continue

                raise ValueError(f"Unexpected token inside EN_QUERY segment: {role}")

            queries.append(
                ENQuery(
                    query_type=query_type,
                    agent=agent_q,
                    theme_unit=theme_unit_q,
                )
            )
            continue

        # Any other token at top level is currently ignored by the EN runtime.
        i += 1

    return events, queries


# ---------------------------------------------------------------------------
# Tiny arithmetic evaluator for postfix INT_*/ADD/SUB expressions
# ---------------------------------------------------------------------------

def evaluate_rpn_expression(tokens: List[GyanDSLToken]) -> int:
    """
    Evaluate a postfix (RPN) arithmetic expression using INT_* / INT_NEG*
    and ADD / SUB tokens.

    This is used as a sanity check that the expression traces produced by
    the EN interpreter agree with the numeric world state.
    """

    stack: List[int] = []

    for tok in tokens:
        if tok.name.startswith("INT_"):
            stack.append(en_int_token_to_value(tok))
            continue

        if tok is GyanDSLToken.ADD:
            if len(stack) < 2:
                raise ValueError("ADD requires at least two values on the stack")
            b = stack.pop()
            a = stack.pop()
            stack.append(a + b)
            continue

        if tok is GyanDSLToken.SUB:
            if len(stack) < 2:
                raise ValueError("SUB requires at least two values on the stack")
            b = stack.pop()
            a = stack.pop()
            stack.append(a - b)
            continue

        raise NotImplementedError(
            f"Unsupported token {tok} in evaluate_rpn_expression; "
            "EN v0 expressions should only use INT_*/ADD/SUB."
        )

    if len(stack) != 1:
        raise ValueError(
            f"RPN evaluation finished with stack size {len(stack)}, expected 1"
        )

    return stack[0]


__all__ = [
    "ENEvent",
    "ENQuery",
    "ENWorldState",
    "apply_event",
    "simulate_en_program",
    "answer_query",
    "build_expression_tokens",
    "parse_en_program",
    "evaluate_rpn_expression",
    "en_int_token_to_value",
    "en_value_to_simple_int_tokens",
]


