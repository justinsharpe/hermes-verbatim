"""
VerbatimContext — Hermes context engine plugin.

Replaces age-based tool-result pruning with typed keep/drop/truncate decisions
at compaction time.  Relevance-scored retention via a vendored typed-decision scorer;
everything kept stays byte-exact; built-in summary retained as fallback.

Pattern provenance: typed-verdict surface with confidence attached,
adapted 2026-09-18 from public prior art; no third-party runtime imported.

Activation (NOT automatic — separate gated step):
  Set in your profile's config.yaml:
    context:
      engine: verbatim
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine
from agent.model_metadata import estimate_messages_tokens_rough

from .scorer import (
    Q_CHOICE,
    Q_NOUL,
    ScoringError,
    ScorerTimeoutError,
    ask_typed,
    build_tool_questions,
)

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Config constants (must match the docstring contract)
# ----------------------------------------------------------------------
PROTECT_FIRST_N = 3   # non-system head messages kept verbatim
PROTECT_LAST_N = 20   # tail messages kept verbatim
KEEP_HEAD_LIMIT = 800  # chars when KEEP_HEAD_800 is chosen
FALLBACK_SUMMARIZER_PATH = "agent.context_compressor.ContextCompressor"

# ----------------------------------------------------------------------
# Hard-coded fallback for when the host's compressor cannot be imported.
# Produces a single 1-line-per-tool stub summary, preserving OpenAI
# format invariants — no actual LLM call.
# ----------------------------------------------------------------------
_FALLBACK_PRUNE_PLACEHOLDER = "[Old tool output cleared to save context space]"


def _build_fallback_summary(messages: List[Dict[str, Any]], target_tokens: int) -> List[Dict[str, Any]]:
    """Deterministic fallback: drop tool bodies to stubs, keep everything else."""
    result: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 200:
                result.append({**msg, "content": _FALLBACK_PRUNE_PLACEHOLDER})
            else:
                result.append(msg)
        else:
            result.append(msg)
    return result


# ----------------------------------------------------------------------
# Ledger helpers (thread-safe, one ledger per plugin directory)
# ----------------------------------------------------------------------


class _DecisionLedger:
    """Append-only JSONL decision ledger for Phase-4 calibration data."""

    def __init__(self, ledger_path: Path):
        self._path = ledger_path
        self._lock = threading.Lock()

    def append(self, entry: dict) -> None:
        with self._lock:
            try:
                with open(self._path, "a") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception as exc:
                logger.warning("[VerbatimContext] could not write decision ledger: %s", exc)


# ----------------------------------------------------------------------
# Mock upstream used in tests / sandbox (real Hermes uses a live upstream)
# ----------------------------------------------------------------------


class MockScorerUpstream:
    """Returns mock answers for offline testing.  All KEEP_FULL / confidence 0.9."""

    async def complete(
        self,
        messages: list,
        model: str,
        reasoning_effort: Any = None,
    ) -> dict:
        await asyncio.sleep(0.01)
        # Parse the state to count interactions
        state_text = ""
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                state_text = msg.get("content", "")

        # Count [idx=N] markers in state to determine how many interactions
        import re as _re

        indices = _re.findall(r"\[idx=(\d+)\]", state_text)
        lines: list[str] = []
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
  CHOSEN: KEEP_FULL
  CONFIDENCE: 0.9
""")
            lines.append(f"""\
---
QUESTION KEY: needed_{idx}
TYPE: noul
PROMPT: Is the information in this tool interaction still needed for the current task?
ANSWER FORMAT (exactly):
  VERDICT: true
  PROBABILITY: 0.9
  CONFIDENCE: 0.9
""")

        return {
            "choices": [
                {"message": {"content": "\n".join(lines)}}
            ]
        }


# ----------------------------------------------------------------------
# VerbatimEngine — the actual ContextEngine plugin
# ----------------------------------------------------------------------


def _build_tool_states(pairs: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[int]]:
    """
    Convert indexed tool pairs into the state format expected by build_tool_questions.

    Returns (tool_states, interaction_indices) where tool_states is a list of dicts
    with keys: idx, call_name, args_head, result_head; and interaction_indices
    is the list of original pair indices.
    """
    tool_states: List[Dict[str, Any]] = []
    interaction_indices: List[int] = []
    for pair in pairs:
        idx = len(tool_states)
        tool_states.append({
            "idx": idx,
            "call_name": pair["call_name"],
            "args_head": pair["args_head"],
            "result_head": pair["result_head"],
        })
        interaction_indices.append(idx)
    return tool_states, interaction_indices


class VerbatimEngine(ContextEngine):
    """Context engine that makes typed keep/drop/truncate decisions per tool interaction.

    One-shot LLM call scores every tool interaction, then applies the verdicts
    deterministically.  User/assistant text is NEVER rewritten — only tool result
    bodies are eligible for pruning.  If the scorer fails (timeout, exception),
    fail-open: return the original message list unchanged and let the host's
    cooldown ladder handle retry.
    """

    name: str = "verbatim"

    # Current model id — set by the host via update_model(); used to construct
    # the fallback summarizer and never hardcoded to a vendor model.
    model: str = ""

    # Required ABC attributes (initialised to defaults; updated by host via update_from_response / update_model)
    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    last_total_tokens: int = 0
    threshold_tokens: int = 0
    context_length: int = 0
    compression_count: int = 0

    # Keep-first/keep-last from docstring contract
    protect_first_n: int = PROTECT_FIRST_N
    protect_last_n: int = PROTECT_LAST_N

    def __init__(
        self,
        plugin_dir: str | Path | None = None,
        scorer_timeout: float = 25.0,
        mock_scorer: bool = False,
        fallback_summarizer_cls: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        plugin_dir:
            Directory for the decision ledger.  Defaults to the plugin's own directory.
        scorer_timeout:
            Per-call timeout for the typed scorer.  Must be below the host's
            ``compression.context_timeout_seconds`` so the host's cooldown ladder
            handles slow-but-not-failed calls.
        mock_scorer:
            Use MockScorerUpstream instead of a real upstream.  For testing/sandbox only.
        fallback_summarizer_cls:
            Dotted path to the host summarizer class.  When set and importable, the
            compress pass delegates remaining reduction to it if typed pruning alone
            is insufficient.  Defaults to the host's built-in ContextCompressor —
            the same guardrail shape as the summary fallback in the pattern source.
        """
        super().__init__()
        if plugin_dir is None:
            plugin_dir = Path(__file__).parent
        else:
            plugin_dir = Path(plugin_dir)

        self._ledger = _DecisionLedger(plugin_dir / "decisions.jsonl")
        self._scorer_timeout = scorer_timeout
        self._mock_scorer = mock_scorer
        self._fallback_summarizer_cls = (
            fallback_summarizer_cls or FALLBACK_SUMMARIZER_PATH
        )

        # Lazily resolved host summarizer
        self._SummarizerCls: Any = None

    # -- pickling: the ledger lock cannot be deepcopied (general-plugin path
    # -- deepcopies the registered singleton); rebuild it on unpickle.
    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        ledger = state.get("_ledger")
        if isinstance(ledger, _DecisionLedger):
            state["_ledger"] = {"__ledger_path__": str(ledger._path)}
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        ledger_state = state.pop("_ledger", None)
        self.__dict__.update(state)
        if isinstance(ledger_state, dict) and "__ledger_path__" in ledger_state:
            self._ledger = _DecisionLedger(Path(ledger_state["__ledger_path__"]))
        else:
            self._ledger = _DecisionLedger(Path(__file__).parent / "decisions.jsonl")

    # ------------------------------------------------------------------
    # ContextEngine ABC contract
    # ------------------------------------------------------------------

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = int(usage.get("prompt_tokens", 0))
        self.last_completion_tokens = int(usage.get("completion_tokens", 0))
        self.last_total_tokens = int(usage.get("total_tokens", 0))

    def update_model(
        self, model: str, context_length: int, base_url: str = "", api_key: str = "",
        provider: str = "", api_mode: str = "",
    ) -> None:
        """Capture the current model id (for fallback construction), then let
        the ABC compute threshold_tokens from context_length + threshold_percent."""
        self.model = model
        super().update_model(model, context_length, base_url, api_key, provider, api_mode)

    def should_compress(self, prompt_tokens: int | None = None) -> bool:
        pt = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        if not self.threshold_tokens:
            return False
        return pt >= self.threshold_tokens

    # ------------------------------------------------------------------
    # compress — the main entry point
    # ------------------------------------------------------------------

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int | None = None,
        focus_topic: str | None = None,
        force: bool = False,
        memory_context: str = "",
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Typed keep/drop/truncate compaction.

        1. Protect head (first 3 non-system) + tail (last 20) messages verbatim.
        2. Index eligible tool call/result pairs; build state for ONE ask_typed call.
        3. Apply verdicts: DROP → stub, KEEP_HEAD_800 → truncated, KEEP_FULL → untouched.
        4. Re-estimate tokens; if still over budget, delegate to the host summarizer.
        5. Append each decision to the local decision ledger.
        """
        if not force and not self.should_compress():
            return list(messages)

        try:
            return self._compress_impl(messages, current_tokens, focus_topic)
        except Exception as exc:
            logger.warning(
                "[VerbatimContext] compress failed (fail-open): %s — returning original list",
                exc,
            )
            return list(messages)

    # ------------------------------------------------------------------
    # Internal implementation (all public-facing methods above already
    # catch exceptions and return the original list — fail-open)
    # ------------------------------------------------------------------

    def _compress_impl(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int | None,
        focus_topic: str | None,
    ) -> List[Dict[str, Any]]:
        # ------------------------------------------------------------------
        # Step 0: shallow copy so we never mutate the caller's list
        # ------------------------------------------------------------------
        msgs = [dict(m) for m in messages]

        # ------------------------------------------------------------------
        # Step 1: identify protected zones (head + tail verbatim)
        # ------------------------------------------------------------------
        protected_count = self._count_protected(messages)
        protected_start = protected_count
        protected_end = len(messages)

        # ------------------------------------------------------------------
        # Step 2: index eligible tool interactions
        # ------------------------------------------------------------------
        eligible_pairs = self._index_tool_pairs(msgs, protected_count)

        if not eligible_pairs:
            # Nothing to score — delegate to fallback or return unchanged
            return self._delegate_fallback_or_return(msgs, current_tokens)

        # ------------------------------------------------------------------
        # Step 3: build scorer state + questions
        # ------------------------------------------------------------------
        tool_states, interaction_indices = _build_tool_states(eligible_pairs)
        state_text, questions = build_tool_questions(tool_states, focus_topic)

        # Staged-fit to a rough budget (cap at ~60 interactions to keep prompt reasonable)
        MAX_STATE_INTERACTIONS = 60
        if len(tool_states) > MAX_STATE_INTERACTIONS:
            tool_states = tool_states[:MAX_STATE_INTERACTIONS]
            interaction_indices = interaction_indices[:MAX_STATE_INTERACTIONS]
            # Rebuild questions for the trimmed set
            state_text, questions = build_tool_questions(tool_states, focus_topic)

        # ------------------------------------------------------------------
        # Step 4: ONE ask_typed call
        # ------------------------------------------------------------------
        upstream = self._get_upstream()

        # Build a lightweight model route object.  Empty model → the host's
        # aux router resolves the configured model for the task (no hardcoded ids).
        model_route = _ModelRoute(model=getattr(self, "model", "") or "", reasoning_effort=None)

        try:
            answers: dict = asyncio.run(
                ask_typed(
                    state_text,
                    questions,
                    model_route,
                    upstream,
                    timeout_seconds=self._scorer_timeout,
                )
            )
        except (ScorerTimeoutError, ScoringError, Exception) as exc:
            logger.warning("[VerbatimContext] scorer call failed (fail-open): %s", exc)
            return list(messages)

        # ------------------------------------------------------------------
        # Step 5: apply verdicts deterministically
        # ------------------------------------------------------------------
        applied_any = False
        for idx, pair in zip(interaction_indices, eligible_pairs[: len(interaction_indices)]):
            verdict_key = f"verdict_{idx}"
            needed_key = f"needed_{idx}"

            verdict_ans = answers.get(verdict_key)
            needed_ans = answers.get(needed_key)

            # Default: KEEP_FULL (conservative)
            verdict = "KEEP_FULL"
            confidence = 0.0

            if verdict_ans is not None and hasattr(verdict_ans, "chosen"):
                verdict = verdict_ans.chosen
                confidence = getattr(verdict_ans, "confidence", 0.0)

            # If scorer says DROP, respect it.  KEEP_HEAD_800 truncates.
            # Never drop if the result body is short anyway.
            tool_msg = msgs[pair["tool_msg_idx"]]
            body = tool_msg.get("content", "") or ""
            body_len = len(body) if isinstance(body, str) else 0

            if verdict == "DROP" and body_len > KEEP_HEAD_LIMIT:
                # Replace with one-line stub (preserves call/result pairing)
                msgs[pair["tool_msg_idx"]] = {
                    **tool_msg,
                    "content": f"[{pair['call_name']}] ({body_len:,} chars result)",
                }
                applied_any = True
            elif verdict == "KEEP_HEAD_800" and body_len > KEEP_HEAD_LIMIT:
                truncated = body[:KEEP_HEAD_LIMIT] + f"\n…[truncated; original {body_len:,} chars]"
                msgs[pair["tool_msg_idx"]] = {**tool_msg, "content": truncated}
                applied_any = True

            # Log the decision to the ledger
            self._ledger.append({
                "ts": datetime.now(timezone.utc).isoformat(),
                "idx": idx,
                "call_name": pair["call_name"],
                "verdict": verdict,
                "confidence": confidence,
                "noul_verdict": getattr(needed_ans, "verdict", None) if needed_ans else None,
                "noul_probability": getattr(needed_ans, "probability", None) if needed_ans else None,
                "noul_confidence": getattr(needed_ans, "confidence", None) if needed_ans else None,
                "focus_topic": focus_topic,
                "compression_count": self.compression_count,
            })

        self.compression_count += 1

        # ------------------------------------------------------------------
        # Step 6: if still over budget, delegate to host summarizer
        # ------------------------------------------------------------------
        return self._delegate_fallback_or_return(msgs, current_tokens)

    # ------------------------------------------------------------------
    # Tool-pair indexing
    # ------------------------------------------------------------------

    def _index_tool_pairs(
        self,
        messages: List[Dict[str, Any]],
        protected_count: int,
    ) -> List[Dict[str, Any]]:
        """
        Return eligible tool-call/result pairs outside the protected zones.
        Each entry: {call_name, args_head, result_head, assistant_idx, tool_msg_idx}

        FIX (CT-47813a8b): append the pair ONLY when its tool-result arrives —
        never on tool-call arrival.  The previous implementation appended a pair
        immediately when the assistant message was seen, then appended the *same*
        object again (after populating result_head) when the tool result arrived,
        causing every completed pair to appear twice in the list.
        """
        pairs: List[Dict[str, Any]] = []
        pending: Dict[str, Dict[str, Any]] = {}

        for i, msg in enumerate(messages):
            if i < protected_count:
                continue
            if msg.get("role") == "tool":
                tcid = str(msg.get("tool_call_id", ""))
                if tcid in pending:
                    pair = pending.pop(tcid)
                    pair["tool_msg_idx"] = i
                    pair["result_head"] = self._head(msg.get("content", ""), 400)
                    pairs.append(pair)
                # else: orphan tool result with no matching call — skip silently
            elif msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    tcid = str(tc.get("id", ""))
                    call_name = str(fn.get("name", "unknown"))
                    args_str = str(fn.get("arguments", ""))
                    pending[tcid] = {
                        "call_name": call_name,
                        "args_head": self._head(args_str, 400),
                        "assistant_idx": i,
                        "tool_msg_idx": -1,
                        "result_head": "",
                    }
                # NOTE: pairs is NOT modified here — append happens only on result arrival

        return pairs

    # ------------------------------------------------------------------
    # Fallback delegation
    # ------------------------------------------------------------------

    def _delegate_fallback_or_return(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int | None,
    ) -> List[Dict[str, Any]]:
        """
        Re-estimate tokens.  If still over threshold, try to delegate to the
        host's ContextCompressor.  If that is unavailable, fall back to the
        deterministic stub pass.
        """
        est = estimate_messages_tokens_rough(messages)
        threshold = self.threshold_tokens or (self.context_length * 75 // 100)

        if est <= threshold:
            return messages

        # Try host summarizer
        if self._fallback_summarizer_cls:
            try:
                compressor_cls = self._import_dotted(self._fallback_summarizer_cls)
                model = getattr(self, "model", "") or ""
                if model:
                    compressor = compressor_cls(model=model)
                else:
                    # No model known yet — cannot construct the host summarizer
                    # (it requires one); go straight to the deterministic fallback.
                    logger.debug("[VerbatimContext] no model set; skipping host summarizer fallback")
                    compressor = None
                if compressor is not None:
                    return compressor.compress(
                        messages,
                        current_tokens=current_tokens,
                        focus_topic=None,
                        force=True,
                        memory_context="",
                    )
            except Exception as exc:
                logger.debug("[VerbatimContext] host summarizer unavailable (%s); using deterministic fallback", exc)

        # Deterministic fallback: stub large tool results
        return _build_fallback_summary(messages, threshold)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _count_protected(self, messages: List[Dict[str, Any]]) -> int:
        """Count non-tool prose messages in the protected head (up to protect_first_n).

        Only user messages and assistant messages WITHOUT tool_calls count toward
        the protected head.  Assistant messages that carry tool_calls must be
        processed by _index_tool_pairs so their call IDs are available for pairing
        with tool result messages — protecting them would orphan the tool results.
        """
        protected = 0
        for msg in messages:
            if msg.get("role") == "tool":
                continue  # tool messages are eligible for pruning
            if protected >= self.protect_first_n:
                break
            # Only count prose messages toward the head-protection budget.
            # Assistant messages that carry tool_calls must NOT be counted here
            # (they must be processed by _index_tool_pairs so their call IDs
            # populate tool_msg_idx_by_tcid and can be paired with tool results).
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                continue
            protected += 1
        return protected

    @staticmethod
    def _head(text: str, limit: int) -> str:
        """First `limit` chars of text, or whole text if shorter."""
        if not isinstance(text, str):
            text = str(text or "")
        return text[:limit] + ("…" if len(text) > limit else "")

    @staticmethod
    def _import_dotted(path: str) -> Any:
        """Import a dotted.Module.class path at runtime."""
        parts = path.rsplit(".", 1)
        if len(parts) == 1:
            import importlib

            return importlib.import_module(parts[0])
        module_path, class_name = parts
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)

    def _get_upstream(self) -> Any:
        """
        Resolve the scorer upstream: MockScorerUpstream in mock mode,
        otherwise a live call through the host's auxiliary LLM routing.
        Imported lazily so the plugin loads without crashing when the host
        is not fully initialised.

        The host's real API is the module-level ``agent.auxiliary_client.call_llm``
        (sync, task-routed, returns a chat-completion response).  We wrap it in
        the async ``complete()`` shape ask_typed expects.
        """
        if self._mock_scorer:
            return MockScorerUpstream()

        from agent.auxiliary_client import call_llm as _host_call_llm

        return _HermesUpstreamAdapter(_host_call_llm)


class _ModelRoute:
    """Lightweight model route object passed to ask_typed.

    ``model`` stays empty in production: the host's auxiliary router picks the
    configured aux model for the task, so the plugin never hardcodes a vendor
    model id.  An explicit override may be supplied via config for testing.
    """

    def __init__(self, model: str = "", reasoning_effort: str | None = None):
        self.model = model
        self.reasoning_effort = reasoning_effort


class _HermesUpstreamAdapter:
    """
    Thin adapter exposing the host's ``call_llm`` as an ask_typed-compatible
    async upstream.

    ask_typed calls ``upstream.complete(messages, model, reasoning_effort)``.
    The host's ``call_llm`` is sync and task-routed; an empty ``model`` lets the
    aux router resolve the configured model for the task.  The response is a
    standard chat-completion dict — the same shape ask_typed already parses.
    """

    def __init__(self, call_llm):
        self._call_llm = call_llm

    async def complete(
        self,
        messages: list,
        model: str,
        reasoning_effort: str | None = None,
    ) -> dict:
        call_kwargs: Dict[str, Any] = {
            "task": "verbatim",
            "messages": messages,
        }
        if model:
            call_kwargs["model"] = model
        if reasoning_effort:
            call_kwargs["reasoning_config"] = {"reasoning_effort": reasoning_effort}
        response = self._call_llm(**call_kwargs)
        if not isinstance(response, dict):
            # Object-shaped responses (some providers) — normalise to the dict
            # shape ask_typed parses.  Use the host's own extractor.
            from agent.auxiliary_client import extract_content_or_reasoning

            content = extract_content_or_reasoning(response)
            response = {"choices": [{"message": {"content": content}}]}
        return response


# ----------------------------------------------------------------------
# Plugin entry point (user-level install path: ~/.hermes/plugins/<name>/
# requires register(ctx); the repo-bundled path discovers the ContextEngine
# subclass directly).  Supporting both makes the plugin installable either way.
# ----------------------------------------------------------------------

def register(ctx) -> None:
    """General plugin-system registration: expose the engine for context.engine selection."""
    try:
        ctx.register_context_engine(VerbatimEngine())
    except Exception:
        # register_context_engine may be absent (e.g. NoopPluginContext during
        # probing) — the repo-bundled discovery path finds the subclass anyway.
        pass
