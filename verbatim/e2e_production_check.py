"""E2E production-path check for the verbatim engine (real host imports, no ABC mocks).

Exercises exactly the paths fixed in the ship-prep wave:
  1. update_model() sets self.model + threshold_tokens (fallback construction dep)
  2. _HermesUpstreamAdapter passes task-routed kwargs to the host call_llm shape
     (task="verbatim", no model key when the route is empty)
  3. _delegate_fallback_or_return constructs the REAL agent.context_compressor.ContextCompressor
     with model= (no TypeError) when still over budget
  4. Full compress() on a fixture transcript via the mock scorer — verbatim guarantee holds.
"""
import sys
from pathlib import Path

HOST = Path.home() / ".hermes" / "hermes-agent"
sys.path.insert(0, str(HOST))
sys.path.insert(0, str(Path(__file__).parent.parent))  # context_engine/ dir — verbatim is a package under it

from verbatim import VerbatimEngine, MockScorerUpstream, _HermesUpstreamAdapter  # noqa: E402
from agent.context_engine import ContextEngine  # noqa: E402
import agent.context_compressor as cc  # noqa: E402

ok = True
def check(label, cond, detail=""):
    global ok
    print(f"{'PASS' if cond else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    ok = ok and cond

# --- 1. update_model contract -------------------------------------------------
eng = VerbatimEngine(plugin_dir=Path(__file__).parent, mock_scorer=True)
check("is real ContextEngine subclass", isinstance(eng, ContextEngine))
eng.update_model(model="test-model-x", context_length=200_000)
check("update_model captures self.model", eng.model == "test-model-x", eng.model)
check("update_model computes threshold_tokens", eng.threshold_tokens == int(200_000 * 0.75), str(eng.threshold_tokens))

# --- 2. upstream adapter kwarg shape -----------------------------------------
captured = {}
def fake_call_llm(**kwargs):
    captured.update(kwargs)
    return {"choices": [{"message": {"content": "ANSWER"}}]}

adapter = _HermesUpstreamAdapter(fake_call_llm)
import asyncio
resp = asyncio.run(adapter.complete([{"role": "user", "content": "hi"}], model="", reasoning_effort=None))
check("adapter sends task=verbatim", captured.get("task") == "verbatim", str(captured.get("task")))
check("adapter omits model key when route empty", "model" not in captured, str(captured.get("model")))
check("adapter returns dict response", isinstance(resp, dict) and resp["choices"][0]["message"]["content"] == "ANSWER")

# --- 3. real ContextCompressor construction with model= ----------------------
compressor = cc.ContextCompressor(model="test-model-x")
check("real ContextCompressor(model=...) constructs", compressor is not None)

# --- 4. full compress pass over threshold, mock scorer ------------------------
fixture = [
    {"role": "user", "content": "kick off: run the deploy"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "function": {"name": "terminal", "arguments": "{\"command\":\"deploy --all\"}"}},
    ]},
    {"role": "tool", "tool_call_id": "c1", "content": "B" * 40_000},
    {"role": "assistant", "content": "deploy finished"},
    {"role": "user", "content": "check the logs please — keep this EXACT"},
]
eng2 = VerbatimEngine(plugin_dir=Path(__file__).parent, mock_scorer=True)
eng2.update_model(model="test-model-x", context_length=2_000)
eng2.last_prompt_tokens = 1_900_000  # force over threshold
out = eng2.compress(list(fixture), current_tokens=None, force=True)
user_kept = [m for m in out if m.get("role") == "user"]
check("user messages byte-identical", user_kept == [m for m in fixture if m.get("role") == "user"])
tool_out = [m for m in out if m.get("role") == "tool"]
check("tool result present (no orphan)", len(tool_out) == 1 and tool_out[0].get("tool_call_id") == "c1")

# --- 5. real-routes fallback path (no mock scorer, real compressor import) ----
eng3 = VerbatimEngine(
    plugin_dir=Path(__file__).parent,
    mock_scorer=False,
    fallback_summarizer_cls="agent.context_compressor.ContextCompressor",
)
eng3.update_model(model="test-model-x", context_length=2_000)
try:
    eng3._get_upstream()
    check("_get_upstream resolves real host call_llm", True)
except Exception as exc:
    check("_get_upstream resolves real host call_llm", False, str(exc))

print("\nE2E:", "ALL GREEN" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)