"""
Tests for the VerbatimContext engine plugin.

Run with:  pytest test_verbatim_engine.py -v

Tests use a temp HERMES_HOME, mock the upstream scorer, and exercise:
  (a) verbatim guarantee — user/assistant text byte-identical before/after
  (b) orphan-prevention — no tool result ever separated from its call
  (c) single-batch assertion — one scorer invocation per compress
  (d) fail-open on scorer exception AND on timeout returns the ORIGINAL list
  (e) fallback delegation fires when reduction is insufficient
  (f) placeholder + KEEP_HEAD_800 truncation shapes correct
  (g) replay fixture — token-reduction vs built-in prune-only
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ------------------------------------------------------------------
# Patch sys.path so the plugin can import agent.* modules from the live
# hermes-agent installation (required for the real-ABC test below).
# We do this before any local imports.
# ------------------------------------------------------------------
import sys

_HERMES_AGENT_ROOT = Path.home() / ".hermes" / "hermes-agent"
if _HERMES_AGENT_ROOT.exists() and str(_HERMES_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_HERMES_AGENT_ROOT))

# Now import the plugin
from verbatim import (
    KEEP_HEAD_LIMIT,
    MockScorerUpstream,
    VerbatimEngine,
    _build_fallback_summary,
)
from verbatim.scorer import ScorerTimeoutError


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_msg(role: str, content: str, **extra: Any) -> dict[str, Any]:
    """Factory for a message dict with known shape."""
    msg: dict[str, Any] = {"role": role, "content": content}
    msg.update(extra)
    return msg


def make_tool_msg(tool_call_id: str, content: str, **extra: Any) -> dict[str, Any]:
    """Factory for a tool-result message."""
    msg = make_msg("tool", content, tool_call_id=tool_call_id)
    msg.update(extra)
    return msg


def make_assistant_with_calls(
    tool_calls: list[dict[str, Any]], content: str = ""
) -> dict[str, Any]:
    """Factory for an assistant message carrying tool_calls."""
    return {"role": "assistant", "content": content, "tool_calls": tool_calls}


def tokens_approx(msgs: list[dict[str, Any]]) -> int:
    """Very rough token estimate: sum of content lengths / 4."""
    return sum(len(str(m.get("content", ""))) for m in msgs) // 4


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def plugin_dir(tmp_path: Path) -> Path:
    """Temp plugin directory for the ledger."""
    d = tmp_path / "verbatim"
    d.mkdir()
    return d


@pytest.fixture
def engine(plugin_dir: Path) -> VerbatimEngine:
    """Engine with mock scorer and deterministic ledger path."""
    eng = VerbatimEngine(
        plugin_dir=plugin_dir,
        mock_scorer=True,
        scorer_timeout=10.0,
        fallback_summarizer_cls="agent.context_compressor.ContextCompressor",
    )
    eng.threshold_tokens = 1000
    eng.context_length = 8000
    return eng


@pytest.fixture
def long_tool_content() -> str:
    """Large tool result body (> KEEP_HEAD_LIMIT chars)."""
    return "line1\n" * 300  # ~1800 chars


@pytest.fixture
def short_tool_content() -> str:
    """Small tool result body (< KEEP_HEAD_LIMIT chars)."""
    return "short result"


# ------------------------------------------------------------------
# (a) verbatim guarantee — user/assistant text byte-identical before/after
# ------------------------------------------------------------------

def test_verbatim_user_and_assistant_unchanged(
    engine: VerbatimEngine,
    long_tool_content: str,
):
    """
    compress() must never modify user or assistant prose.
    """
    messages = [
        make_msg("system", "You are a helpful assistant."),
        make_msg("user", "Build me a website."),
        make_assistant_with_calls([
            {
                "id": "call_1",
                "function": {"name": "terminal", "arguments": '{"command": "ls"}'},
                "type": "function",
            }
        ]),
        make_tool_msg("call_1", long_tool_content),
        make_msg("user", "Add a login page."),
        make_assistant_with_calls([], content="Here's the updated site."),
    ]

    result = engine.compress(list(messages), current_tokens=3000, force=True)

    # User messages unchanged
    for orig, comp in zip(
        (m for m in messages if m["role"] == "user"),
        (m for m in result if m["role"] == "user"),
    ):
        assert orig["content"] == comp["content"], "user prose was modified"

    # Assistant content/prose unchanged
    for orig, comp in zip(
        (m for m in messages if m["role"] == "assistant" and m.get("content")),
        (m for m in result if m["role"] == "assistant" and m.get("content")),
    ):
        assert orig["content"] == comp["content"], "assistant prose was modified"

    # Assistant tool_calls unchanged
    orig_tc = [
        tc for m in messages if m["role"] == "assistant" for tc in m.get("tool_calls") or []
    ]
    comp_tc = [
        tc for m in result if m["role"] == "assistant" for tc in m.get("tool_calls") or []
    ]
    assert orig_tc == comp_tc, "assistant tool_calls were modified"


# ------------------------------------------------------------------
# (b) orphan-prevention — no tool result ever separated from its call
# ------------------------------------------------------------------

def test_no_tool_result_orphaned(
    engine: VerbatimEngine,
    long_tool_content: str,
):
    """
    After compress(), every tool result must still have a matching tool_call
    in the preceding assistant message.  No tool result is ever dropped or
    separated from its call.
    """
    messages = [
        make_msg("user", "Search the web."),
        make_assistant_with_calls([
            {
                "id": "call_x",
                "function": {"name": "web_search", "arguments": '{"query": "python"}'},
                "type": "function",
            }
        ]),
        make_tool_msg("call_x", long_tool_content),
        make_msg("user", "Now do something else."),
        make_assistant_with_calls([
            {
                "id": "call_y",
                "function": {"name": "terminal", "arguments": '{"command": "pwd"}'},
                "type": "function",
            }
        ]),
        make_tool_msg("call_y", "result of pwd"),
    ]

    result = engine.compress(list(messages), current_tokens=3000, force=True)

    # Collect all tool_call_ids from assistant messages
    call_ids = set()
    for msg in result:
        if msg["role"] == "assistant":
            for tc in msg.get("tool_calls") or []:
                call_ids.add(str(tc.get("id", "")))

    # Every tool result must have a corresponding call_id
    for msg in result:
        if msg["role"] == "tool":
            tcid = str(msg.get("tool_call_id", ""))
            assert tcid in call_ids, f"tool result {tcid!r} has no matching tool_call"


# ------------------------------------------------------------------
# (c) single-batch assertion — exactly one scorer invocation per compress
# ------------------------------------------------------------------

def test_single_scorer_invocation(engine: VerbatimEngine, long_tool_content: str):
    """
    compress() must call the scorer upstream exactly once.
    """
    messages = [
        make_msg("user", "Do work."),
        make_assistant_with_calls([
            {"id": "c1", "function": {"name": "terminal", "arguments": "{}"}, "type": "function"},
            {"id": "c2", "function": {"name": "read_file", "arguments": '{"path": "x"}'}, "type": "function"},
        ]),
        make_tool_msg("c1", long_tool_content),
        make_tool_msg("c2", long_tool_content),
    ]

    # Patch ask_typed in the verbatim namespace (where compress() imports it as a
    # local name) so the patch is effective even across asyncio.run() boundaries.
    # _scorer_module.ask_typed patching alone does NOT work because the local
    # import alias in verbatim/__init__.py is what compress() actually calls.
    import verbatim as _verbatim_module
    from verbatim import scorer as _scorer_module

    call_count = 0
    _orig_ask_typed = _scorer_module.ask_typed

    async def tracking_ask_typed(state, questions, model_route, upstream, timeout_seconds=25.0):
        nonlocal call_count
        call_count += 1
        return await _orig_ask_typed(state, questions, model_route, upstream, timeout_seconds)

    eng = VerbatimEngine(plugin_dir=engine._ledger._path.parent, mock_scorer=True)
    eng.threshold_tokens = engine.threshold_tokens
    eng.context_length = engine.context_length

    with patch.object(_verbatim_module, "ask_typed", side_effect=tracking_ask_typed):
        eng.compress(list(messages), current_tokens=3000, force=True)

    assert call_count == 1, f"expected 1 scorer call, got {call_count}"


# ------------------------------------------------------------------
# (d) fail-open on scorer exception AND on timeout returns the ORIGINAL list
# ------------------------------------------------------------------

def test_fail_open_on_scorer_exception(engine: VerbatimEngine):
    """
    When the scorer raises any exception, compress() returns the
    ORIGINAL (unmodified) message list.
    """
    messages = [
        make_msg("user", "Hello."),
        make_assistant_with_calls([
            {"id": "c1", "function": {"name": "terminal", "arguments": "{}"}, "type": "function"},
        ]),
        make_tool_msg("c1", "big result " * 200),
    ]

    class FailingUpstream:
        async def complete(self, *args, **kwargs):
            raise RuntimeError("scorer exploded")

    with patch.object(engine, "_get_upstream", return_value=FailingUpstream()):
        result = engine.compress(list(messages), current_tokens=3000, force=True)

    assert result == messages, "fail-open did not return original list"


def test_fail_open_on_timeout(engine: VerbatimEngine):
    """
    When the scorer times out, compress() returns the ORIGINAL list.
    """
    messages = [
        make_msg("user", "Hello."),
        make_assistant_with_calls([
            {"id": "c1", "function": {"name": "terminal", "arguments": "{}"}, "type": "function"},
        ]),
        make_tool_msg("c1", "big result " * 200),
    ]

    class SlowUpstream:
        async def complete(self, *args, **kwargs):
            raise ScorerTimeoutError("timed out")

    with patch.object(engine, "_get_upstream", return_value=SlowUpstream()):
        result = engine.compress(list(messages), current_tokens=3000, force=True)

    assert result == messages, "timeout fail-open did not return original list"


# ------------------------------------------------------------------
# (e) fallback delegation fires when reduction is insufficient
# ------------------------------------------------------------------

def test_fallback_triggered_when_still_over_budget(
    plugin_dir: Path,
    long_tool_content: str,
):
    """
    When typed pruning alone does not bring the message below threshold,
    the fallback summarizer is invoked.
    """
    # Build a transcript that is clearly over budget
    many_tools = []
    messages = [
        make_msg("system", "You are a helpful assistant."),
        make_msg("user", "Work on this project."),
    ]
    for i in range(30):
        cid = f"call_{i}"
        messages.append(
            make_assistant_with_calls([
                {
                    "id": cid,
                    "function": {"name": "terminal", "arguments": f'{{"n": {i}}}'},
                    "type": "function",
                }
            ])
        )
        messages.append(make_tool_msg(cid, f"result {i}\n" + "x" * 500))

    eng = VerbatimEngine(plugin_dir=plugin_dir, mock_scorer=True)
    eng.threshold_tokens = 100  # very low — will definitely need fallback
    eng.context_length = 8000

    # Patch _delegate_fallback_or_return to track whether it attempts delegation
    original_delegate = eng._delegate_fallback_or_return

    delegated = False

    def tracking_delegate(msgs, current_tokens):
        nonlocal delegated
        delegated = True
        return original_delegate(msgs, current_tokens)

    eng._delegate_fallback_or_return = tracking_delegate

    result = eng.compress(list(messages), current_tokens=tokens_approx(messages) * 2, force=True)

    assert delegated, "fallback was not triggered when still over budget"


# ------------------------------------------------------------------
# (f) placeholder + KEEP_HEAD_800 truncation shapes correct
# ------------------------------------------------------------------

def test_drop_placeholder_shape(long_tool_content: str):
    """
    A DROP verdict on a large tool result produces the correct stub:
    "[tool_name] (N chars result)"
    """
    messages = [
        make_msg("user", "Hello."),
        make_assistant_with_calls([
            {"id": "call_drop", "function": {"name": "terminal", "arguments": "{}"}, "type": "function"},
        ]),
        make_tool_msg("call_drop", long_tool_content),
    ]

    # Upstream that always returns DROP for every interaction
    class DropAllUpstream:
        async def complete(self, messages, model, reasoning_effort=None):
            await asyncio.sleep(0.001)
            state = next(
                m["content"] for m in messages if isinstance(m, dict) and m.get("role") == "user"
            )
            import re
            indices = re.findall(r"\[idx=(\d+)\]", state)
            lines = []
            for idx in indices:
                lines.append(f"""\
---
QUESTION KEY: verdict_{idx}
TYPE: choice
PROMPT: Should this tool interaction be kept verbatim, truncated to its first 800 characters, or dropped entirely from the conversation context?
OPTIONS:
  1. KEEP_FULL
  2. KEEP_HEAD_800
  3. DROP
ANSWER FORMAT (exactly):
  CHOSEN: DROP
  CONFIDENCE: 0.95
""")
                lines.append(f"""\
---
QUESTION KEY: needed_{idx}
TYPE: noul
PROMPT: Is the information in this tool interaction still needed for the current task?
ANSWER FORMAT (exactly):
  VERDICT: false
  PROBABILITY: 0.95
  CONFIDENCE: 0.95
""")
            return {"choices": [{"message": {"content": "\n".join(lines)}}]}

    eng = VerbatimEngine(mock_scorer=False)
    # Threshold must be high enough that after the DROP stub is applied,
    # estimated tokens are under budget — otherwise the fallback fires and
    # overwrites the verdict stub with _FALLBACK_PRUNE_PLACEHOLDER.
    # After DROP: ~20-char stub vs original ~1800 chars (~450 tokens).
    # A threshold of 600 keeps the result under budget.
    eng.threshold_tokens = 600
    eng.context_length = 8000

    with patch.object(eng, "_get_upstream", return_value=DropAllUpstream()):
        result = eng.compress(list(messages), current_tokens=5000, force=True)

    tool_msgs = [m for m in result if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    stub = tool_msgs[0]["content"]
    assert "[Old tool output cleared" not in stub, f"fallback stub: {stub!r}"
    assert stub.startswith("[terminal]"), f"unexpected stub: {stub!r}"
    assert "chars result)" in stub, f"unexpected stub: {stub!r}"


def test_keep_head_800_truncation_shape(long_tool_content: str):
    """
    A KEEP_HEAD_800 verdict on a large tool result truncates to 800 chars
    with a visible marker.
    """
    messages = [
        make_msg("user", "Hello."),
        make_assistant_with_calls([
            {"id": "call_trunc", "function": {"name": "terminal", "arguments": "{}"}, "type": "function"},
        ]),
        make_tool_msg("call_trunc", long_tool_content),
    ]

    class TruncateUpstream:
        async def complete(self, messages, model, reasoning_effort=None):
            await asyncio.sleep(0.001)
            state = next(
                m["content"] for m in messages if isinstance(m, dict) and m.get("role") == "user"
            )
            import re
            indices = re.findall(r"\[idx=(\d+)\]", state)
            lines = []
            for idx in indices:
                lines.append(f"""\
---
QUESTION KEY: verdict_{idx}
TYPE: choice
OPTIONS:
  1. KEEP_FULL
  2. KEEP_HEAD_800
  3. DROP
ANSWER FORMAT (exactly):
  CHOSEN: KEEP_HEAD_800
  CONFIDENCE: 0.85
""")
                lines.append(f"""\
---
QUESTION KEY: needed_{idx}
TYPE: noul
ANSWER FORMAT (exactly):
  VERDICT: true
  PROBABILITY: 0.85
  CONFIDENCE: 0.85
""")
            return {"choices": [{"message": {"content": "\n".join(lines)}}]}

    eng = VerbatimEngine(mock_scorer=False)
    # Threshold must be high enough that after truncation (~800 + marker chars)
    # the result stays under budget.  ~200 tokens vs ~450 original.  Threshold=600 works.
    eng.threshold_tokens = 600
    eng.context_length = 8000

    with patch.object(eng, "_get_upstream", return_value=TruncateUpstream()):
        result = eng.compress(list(messages), current_tokens=5000, force=True)

    tool_msgs = [m for m in result if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    content = tool_msgs[0]["content"]
    assert "[Old tool output cleared" not in content, f"fallback stub: {content!r}"
    assert len(content) <= KEEP_HEAD_LIMIT + 60, f"truncation exceeded limit: {len(content)}"
    assert "…[truncated" in content or "...[truncated" in content, f"no truncation marker: {content!r}"


# ------------------------------------------------------------------
# (g) replay fixture — token-reduction vs built-in prune-only
# ------------------------------------------------------------------

def test_replay_fixture_token_reduction(plugin_dir: Path):
    """
    Load a recorded long tool-heavy transcript (sanitised), run compress(),
    and report before/after token counts vs the built-in prune-only baseline.

    The baseline is the rough token count of the original messages.
    After compression we expect a meaningful reduction.
    """
    # Simulated "recorded" transcript — tool-heavy, clearly over threshold
    transcript = [
        make_msg("system", "You are a helpful coding assistant."),
        make_msg("user", "I need you to refactor the entire backend. Start with the auth module."),
        make_assistant_with_calls([
            {
                "id": "tc_read_auth",
                "function": {"name": "read_file", "arguments": '{"path": "auth.py"}'},
                "type": "function",
            }
        ]),
        make_tool_msg(
            "tc_read_auth",
            "def authenticate(token):\n" + "    pass\n" * 200,
        ),
        make_assistant_with_calls([
            {
                "id": "tc_search",
                "function": {"name": "search_files", "arguments": '{"pattern": "def .*auth.*", "path": "."}'},
                "type": "function",
            }
        ]),
        make_tool_msg(
            "tc_search",
            "\n".join(f"auth.py:{i}:def helper_{i}()" for i in range(100)),
        ),
        make_assistant_with_calls([
            {
                "id": "tc_terminal",
                "function": {"name": "terminal", "arguments": '{"command": "git status"}'},
                "type": "function",
            }
        ]),
        make_tool_msg(
            "tc_terminal",
            "M  auth.py\n?? new_auth.py\n" + "x" * 1000,
        ),
        make_msg("user", "Good start. Continue with the models."),
        make_assistant_with_calls([
            {
                "id": "tc_read_models",
                "function": {"name": "read_file", "arguments": '{"path": "models.py"}'},
                "type": "function",
            }
        ]),
        make_tool_msg(
            "tc_read_models",
            "class User:\n" * 300,
        ),
    ]

    # ---- built-in prune-only baseline ----
    baseline_tokens = tokens_approx(transcript)

    # ---- compress with verbatim engine ----
    eng = VerbatimEngine(plugin_dir=plugin_dir, mock_scorer=True)
    eng.threshold_tokens = 200
    eng.context_length = 4000

    result = eng.compress(list(transcript), current_tokens=baseline_tokens * 2, force=True)
    after_tokens = tokens_approx(result)

    reduction = baseline_tokens - after_tokens
    reduction_pct = (reduction / baseline_tokens * 100) if baseline_tokens else 0

    # Sanity: we got smaller
    assert after_tokens < baseline_tokens, (
        f"no reduction: before={baseline_tokens}, after={after_tokens}"
    )

    # ---- ledger was written ----
    ledger_path = plugin_dir / "decisions.jsonl"
    assert ledger_path.exists(), "decision ledger not written"

    lines = ledger_path.read_text().strip().split("\n")
    assert len(lines) >= 1, "ledger empty"
    first_entry = json.loads(lines[0])
    assert "verdict" in first_entry, "ledger entry missing verdict"
    assert "confidence" in first_entry, "ledger entry missing confidence"

    # ---- report (printed for human review) ----
    print(f"\n[Replay Fixture Report]")
    print(f"  Baseline tokens (prune-only): {baseline_tokens}")
    print(f"  After verbatim compress:     {after_tokens}")
    print(f"  Reduction:                   {reduction} tokens ({reduction_pct:.1f}%)")
    print(f"  Ledger entries:              {len(lines)}")
    print(f"  Sample ledger entry:         {first_entry}")


# ------------------------------------------------------------------
# ABC load test — verify the engine loads as a ContextEngine subclass
# ------------------------------------------------------------------

def test_engine_is_valid_context_engine_subclass():
    """
    VerbatimEngine must be a valid subclass of ContextEngine with all
    required abstract methods implemented (loads without mocking the ABC).
    """
    from agent.context_engine import ContextEngine as ABC

    assert issubclass(VerbatimEngine, ABC)

    # Required abstract methods present and callable
    eng = VerbatimEngine.__new__(VerbatimEngine)
    for method in ("update_from_response", "should_compress", "compress", "name"):
        assert hasattr(eng, method), f"missing {method}"

    # Required class attributes
    assert eng.name == "verbatim"


# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
