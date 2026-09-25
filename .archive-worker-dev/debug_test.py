"""Debug script for failing tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".hermes" / "hermes-agent"))

import asyncio
from verbatim import VerbatimEngine, _build_fallback_summary
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
        [
            {
                "id": "call_trunc",
                "function": {"name": "terminal", "arguments": "{}"},
                "type": "function",
            }
        ]
    ),
    make_tool_msg("call_trunc", long_content),
]

eng = VerbatimEngine(mock_scorer=True)
eng.threshold_tokens = 100
eng.context_length = 8000

print("should_compress:", eng.should_compress(5000))
print("threshold_tokens:", eng.threshold_tokens)
print("context_length:", eng.context_length)
print("protect_first_n:", eng.protect_first_n)
print("protect_last_n:", eng.protect_last_n)

protected = eng._count_protected(messages)
print("protected_count:", protected)

pairs = eng._index_tool_pairs([dict(m) for m in messages], protected)
print("eligible_pairs:", pairs)

states, questions = build_tool_questions(
    [{"idx": 0, "call_name": "terminal", "args_head": "{}", "result_head": "..."}], None
)
print("\nbuild_tool_questions sample state:", states[:300])
print("questions keys:", list(questions.keys()))

# Now test a real compress with the engine
print("\n--- Compress call ---")
result = eng.compress(list(messages), current_tokens=5000, force=True)
tool_msgs = [m for m in result if m["role"] == "tool"]
print("Result tool content:", tool_msgs[0]["content"] if tool_msgs else "NONE")
