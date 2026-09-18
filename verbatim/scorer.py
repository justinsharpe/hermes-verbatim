"""
Vendored typed decision scorer — one upstream call, many typed questions.

Provides a typed verdict surface so the context engine can ask structured
keep/drop/truncate questions and get back confidence-tagged answers
without a purpose-built decision model. Strict enumeration: anything
outside the declared options is a parse error (confidence 0.0), never a
crash; wrong answers are still possible — confidence carries the honesty.
"""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Optional

# --- Public answer types ------------------------------------------------------

Q_CHOICE = "choice"
Q_SCORE = "score"
Q_NOUL = "noul"
VALID_Q_TYPES = {Q_CHOICE, Q_SCORE, Q_NOUL}


@dataclass
class ChoiceAnswer:
    chosen: str
    confidence: float
    parse_error: bool = False


@dataclass
class ScoreAnswer:
    score: int
    label: str
    confidence: float
    parse_error: bool = False


@dataclass
class NoulAnswer:
    verdict: bool
    probability: float
    confidence: float
    parse_error: bool = False


Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer
QuestionSpec = dict[str, Any]

# --- Prompt builder -----------------------------------------------------------

_CONFIDENCE_RE = re.compile(
    r"CONFIDENCE:\s*(?P<value>0?\.\d+|1(?:\.0+)?|0)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _build_typed_prompt(state: str, questions: list[QuestionSpec]) -> str:
    blocks: list[str] = [
        "You are a strict typed-verdict engine. Answer each question exactly",
        "in the format specified. Do not add explanation, preamble, or any text",
        "outside the required format.\n",
    ]

    for q in questions:
        qtype = q["type"]
        key = q["key"]
        prompt = q["prompt"]

        if qtype == Q_CHOICE:
            opts = q["options"]
            if len(opts) > 255:
                raise ValueError(f"choice question '{key}' has {len(opts)} options (max 255)")
            options_block = "\n".join(f"  {i+1}. {o}" for i, o in enumerate(opts))
            blocks.append(textwrap.dedent(f"""\
                ---
                QUESTION KEY: {key}
                TYPE: choice
                PROMPT: {prompt}
                OPTIONS:
                {options_block}
                ANSWER FORMAT (exactly):
                  CHOSEN: <option text>
                  CONFIDENCE: <0.0–1.0>
                """))

        elif qtype == Q_SCORE:
            levels = q["levels"]
            if len(levels) < 2:
                raise ValueError(f"score question '{key}' needs ≥2 levels, got {len(levels)}")
            levels_block = "\n".join(f"  {i+1}. {l}" for i, l in enumerate(levels))
            blocks.append(textwrap.dedent(f"""\
                ---
                QUESTION KEY: {key}
                TYPE: score
                PROMPT: {prompt}
                ORDERED LEVELS (best to worst):
                {levels_block}
                ANSWER FORMAT (exactly):
                  SCORE: <1–{len(levels)}>
                  LABEL: <the descriptive label you selected>
                  CONFIDENCE: <0.0–1.0>
                """))

        elif qtype == Q_NOUL:
            blocks.append(textwrap.dedent(f"""\
                ---
                QUESTION KEY: {key}
                TYPE: noul
                PROMPT: {prompt}
                ANSWER FORMAT (exactly):
                  VERDICT: true|false
                  PROBABILITY: <0.0–1.0>
                  CONFIDENCE: <0.0–1.0>
                """))

        else:
            raise ValueError(f"unknown question type '{qtype}' for key '{key}'")

    if state:
        blocks.insert(1, f"STATE TO EVALUATE:\n{state}\n")

    return "\n".join(blocks)


# --- Answer parsers -----------------------------------------------------------

def _parse_float(s: str) -> Optional[float]:
    try:
        v = float(s.strip())
        return max(0.0, min(1.0, v))
    except (ValueError, TypeError):
        return None


def _parse_choice(raw: str, options: list[str]) -> ChoiceAnswer:
    m = re.search(r"^\s*CHOSEN:\s*(.+)$", raw, re.IGNORECASE | re.MULTILINE)
    if not m:
        return ChoiceAnswer(chosen="", confidence=0.0, parse_error=True)

    chosen = m.group(1).strip()

    conf_m = _CONFIDENCE_RE.search(raw)
    confidence = _parse_float(conf_m.group("value")) if conf_m else None
    if confidence is None:
        return ChoiceAnswer(chosen=chosen, confidence=0.0, parse_error=True)

    if chosen.lower() not in {o.lower() for o in options}:
        return ChoiceAnswer(chosen=chosen, confidence=0.0, parse_error=True)

    return ChoiceAnswer(chosen=chosen, confidence=confidence, parse_error=False)


def _parse_score(raw: str, levels: list[str]) -> ScoreAnswer:
    idx_m = re.search(r"^\s*SCORE:\s*(\d+)\s*$", raw, re.IGNORECASE | re.MULTILINE)
    label_m = re.search(r"^\s*LABEL:\s*(.+)$", raw, re.IGNORECASE | re.MULTILINE)
    if not (idx_m and label_m):
        return ScoreAnswer(score=0, label="", confidence=0.0, parse_error=True)

    try:
        score = int(idx_m.group(1))
    except ValueError:
        return ScoreAnswer(score=0, label="", confidence=0.0, parse_error=True)

    label = label_m.group(1).strip()

    if not (1 <= score <= len(levels)):
        return ScoreAnswer(score=score, label=label, confidence=0.0, parse_error=True)

    if label.lower() != levels[score - 1].lower():
        return ScoreAnswer(score=score, label=label, confidence=0.0, parse_error=True)

    conf_m = _CONFIDENCE_RE.search(raw)
    confidence = _parse_float(conf_m.group("value")) if conf_m else 0.0
    if confidence is None:
        return ScoreAnswer(score=score, label=label, confidence=0.0, parse_error=True)

    return ScoreAnswer(score=score, label=label, confidence=confidence, parse_error=False)


def _parse_noul(raw: str) -> NoulAnswer:
    v_m = re.search(r"^\s*VERDICT:\s*(true|false)\s*$", raw, re.IGNORECASE | re.MULTILINE)
    p_m = re.search(
        r"^\s*PROBABILITY:\s*(-?0?\.\d+|-?1(?:\.\d+)?|-?0)\s*$",
        raw,
        re.IGNORECASE | re.MULTILINE,
    )
    if not (v_m and p_m):
        return NoulAnswer(verdict=False, probability=0.0, confidence=0.0, parse_error=True)

    verdict = v_m.group(1).lower() == "true"

    try:
        prob = max(0.0, min(1.0, float(p_m.group(1))))
    except ValueError:
        return NoulAnswer(verdict=verdict, probability=0.0, confidence=0.0, parse_error=True)

    conf_m = _CONFIDENCE_RE.search(raw)
    confidence = _parse_float(conf_m.group("value")) if conf_m else 0.0
    if confidence is None:
        return NoulAnswer(verdict=verdict, probability=prob, confidence=0.0, parse_error=True)

    return NoulAnswer(verdict=verdict, probability=prob, confidence=confidence, parse_error=False)


def _fallback_answer(qtype: str) -> Answer:
    if qtype == Q_CHOICE:
        return ChoiceAnswer(chosen="", confidence=0.0, parse_error=True)
    if qtype == Q_SCORE:
        return ScoreAnswer(score=0, label="", confidence=0.0, parse_error=True)
    return NoulAnswer(verdict=False, probability=0.0, confidence=0.0, parse_error=True)


# --- Core ask_typed ------------------------------------------------------------
#
# NOTE on the async upstream interface:
#   The Hermes plugin host (agent/context_engine.py) calls compress() synchronously
#   on a pooled daemon thread.  The compress() method here therefore runs its LLM
#   call on that same thread — no new thread pool required.
#
#   upstream.complete(messages, model, reasoning_effort) -> raw response dict
#   Expected shape: {"choices": [{"message": {"content": "..."}}]}
#   This matches the Hermes auxiliary_client.call_llm contract and the OpenAI
#   chat completion wire format.
#
#   model_route: object with .model (str) and optional .reasoning_effort.
#   upstream: object with an async complete() method.
#
#   Scorer timeout MUST be set BELOW the host compression timeout
#   (compression.context_timeout_seconds).  The caller passes timeout_seconds
#   so the engine can enforce the correct budget.

async def ask_typed(
    state: str,
    questions: dict[str, QuestionSpec],
    model_route,
    upstream,
    timeout_seconds: float = 25.0,
) -> dict[str, Answer]:
    """Ask multiple typed questions in ONE upstream call.

    Parameters
    ----------
    state:
        Context string the model should evaluate against.
    questions:
        Dict of QuestionSpec keyed by unique string identifiers.
    model_route:
        Object with .model (str) and optional .reasoning_effort.
    upstream:
        Async client with a ``complete(messages, model, reasoning_effort=None)`` method.
    timeout_seconds:
        Per-call timeout enforced by this wrapper.  Must be below the host
        compression timeout so the host's own retry/cooldown ladder handles
        slow-but-not-failed calls.

    Returns
    -------
    dict[str, Answer]
        One entry per question key.

    Raises
    ------
    ScoringError
        Any upstream failure propagates as ScoringError (caught by the engine's
        fail-open logic).
    """
    spec_list: list[QuestionSpec] = []
    for key, spec in questions.items():
        spec = dict(spec)
        spec["key"] = key
        if spec.get("type") not in VALID_Q_TYPES:
            raise ValueError(f"question '{key}' has unknown type '{spec.get('type')}'")
        spec_list.append(spec)

    prompt = _build_typed_prompt(state, spec_list)

    import asyncio

    try:
        async with asyncio.timeout(timeout_seconds):
            body = await upstream.complete(
                messages=[{"role": "user", "content": prompt}],
                model=model_route.model,
                reasoning_effort=getattr(model_route, "reasoning_effort", None),
            )
    except asyncio.TimeoutError:
        raise ScorerTimeoutError(f"Scorer upstream timed out after {timeout_seconds}s")
    except Exception as exc:
        raise ScoringError(f"Scorer upstream failed: {exc}") from exc

    raw_text = (
        (body.get("choices") or [{}])[0]
        .get("message", {})
        .get("content", "")
    )

    raw_blocks = [b for b in raw_text.split("---") if b.strip()]
    parsed: dict[str, Answer] = {}

    for spec in spec_list:
        key = spec["key"]
        qtype = spec["type"]

        block_text = ""
        for block in raw_blocks:
            if re.search(
                rf"QUESTION KEY:\s*{re.escape(key)}\s*$",
                block,
                re.MULTILINE | re.IGNORECASE,
            ):
                block_text = block
                break

        if not block_text:
            parsed[key] = _fallback_answer(qtype)
            continue

        block_text = block_text.strip()

        if qtype == Q_CHOICE:
            parsed[key] = _parse_choice(block_text, spec["options"])
        elif qtype == Q_SCORE:
            parsed[key] = _parse_score(block_text, spec["levels"])
        else:
            parsed[key] = _parse_noul(block_text)

    return parsed


class ScoringError(Exception):
    """Raised when the upstream scorer call fails for any reason."""
    pass


class ScorerTimeoutError(ScoringError):
    """Raised when the upstream scorer call times out."""
    pass


# --- Question presets for the verbatim context engine -------------------------

KEEP_OPTIONS = ["KEEP_FULL", "KEEP_HEAD_800", "DROP"]

TOOL_QUESTION_SPEC = {
    "type": Q_CHOICE,
    "prompt": (
        "Should this tool interaction be kept verbatim, truncated to its first 800 "
        "characters, or dropped entirely from the conversation context?"
    ),
    "options": KEEP_OPTIONS,
}

TOOL_STILL_NEEDED_SPEC = {
    "type": Q_NOUL,
    "prompt": "Is the information in this tool interaction still needed for the current task?",
}


def build_tool_questions(
    tool_states: list[dict[str, Any]],
    focus_topic: str | None,
) -> tuple[str, dict[str, QuestionSpec]]:
    """Build the state string and question dict for a batch of tool interactions.

    Parameters
    ----------
    tool_states:
        List of dicts, each with keys: idx, call_name, args_head, result_head.
    focus_topic:
        Optional focus topic from /compress to fold into the state.

    Returns
    -------
    (state_text, questions_dict)
        state_text: the state block for ask_typed
        questions_dict: the questions dict for ask_typed
    """
    lines = ["TOOL INTERACTIONS (oldest first):\n"]
    if focus_topic:
        lines.append(f"FOCUS TOPIC: {focus_topic}\n")

    for t in tool_states:
        lines.append(
            f"[idx={t['idx']}] {t['call_name']} | args: {t['args_head']} | result: {t['result_head']}"
        )

    state = "\n".join(lines)
    questions: dict[str, QuestionSpec] = {}

    for t in tool_states:
        idx = t["idx"]
        questions[f"verdict_{idx}"] = dict(TOOL_QUESTION_SPEC, key=f"verdict_{idx}")
        questions[f"needed_{idx}"] = dict(TOOL_STILL_NEEDED_SPEC, key=f"needed_{idx}")

    return state, questions
