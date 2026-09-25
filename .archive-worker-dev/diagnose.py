"""Diagnostic script to verify engine behavior."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".hermes" / "hermes-agent"))
sys.path.insert(0, str(Path.home() / ".hermes" / "plugins" / "context_engine"))

# Run directly
import asyncio
from verbatim import VerbatimEngine, _build_fallback_summary, _build_tool_states
from verbatim.scorer import build_tool_questions


def make_msg(role, content, **extra):
    msg = {"role": role, "content": content}
    msg.update(extra)
    return msg


def make_tool_msg(tool_call_id, content):
    return {"role": "tool", "content": content, "tool_call_id": tool_call_id}


def make_assistant_with_calls(tool_calls, content=""):
    return {"role": "assistant", "content": content, "tool_calls": tool_calls}


long_content = "line1\n" * 300  # ~1800 chars

messages = [
    make_msg("user", "Hello."),
    make_assistant_with_calls(
        [{"id": "call_trunc", "function": {"name": "terminal", "arguments": "{}"}, "type": "function"}]
    ),
    make_tool_msg("call_trunc", long_content),
]

print("=== DIAGNOSIS ===")
print(f"Total messages: {len(messages)}")

eng = VerbatimEngine(mock_scorer=True)
eng.threshold_tokens = 100
eng.context_length = 8000

print(f"should_compress(5000): {eng.should_compress(5000)}")
print(f"force=True bypasses: YES")

protected = eng._count_protected(messages)
print(f"_count_protected: {protected}  (messages at 0..{protected-1} protected)")

pairs = eng._index_tool_pairs([dict(m) for m in messages], protected)
print(f"eligible_pairs found: {len(pairs)}")
for p in pairs:
    print(f"  pair: call_name={p['call_name']}, tool_idx={p['tool_msg_idx']}")

if pairs:
    states, indices = _build_tool_states(pairs)
    print(f"_build_tool_states: {len(states)} states, indices={indices}")
    state_text, questions = build_tool_questions(states, None)
    print(f"build_tool_questions: {len(questions)} questions")
    print(f"Question keys: {list(questions.keys())}")
    print(f"State text (first 200 chars): {state_text[:200]}")
    upstream = eng._get_upstream()
    print(f"_get_upstream type: {type(upstream)}")
    print("=== All good so far ===")

# Now run compress
print("\n=== Running compress ===")
try:
    result = eng.compress(list(messages), current_tokens=5000, force=True)
    print(f"compress returned {len(result)} messages")
    for i, m in enumerate(result):
        print(f"  [{i}] role={m['role']}, content_len={len(str(m.get('content','')))}")
        if m['role'] == 'tool':
            print(f"       content: {m['content'][:100]!r}")
except Exception as e:
    print(f"EXCEPTION: {e}")
    import traceback
    traceback.print_exc()
