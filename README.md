# hermes-verbatim

**Typed keep/drop/truncate context engine for [Hermes Agent](https://hermes-agent.nousresearch.com) — your messages are never rewritten.**

Hermes's built-in compaction summarizes the middle of long conversations, and summaries are lossy: an exact file path, an error string, or a command can vanish exactly when you still need it. `hermes-verbatim` replaces that with **typed retention decisions**: at compaction time, a single fast call to your existing auxiliary model scores every old tool interaction KEEP_FULL / KEEP_HEAD_800 / DROP. Everything kept stays **byte-exact**. Only tool result bodies are ever eligible for pruning — user and assistant prose is never paraphrased, and tool calls are never orphaned from their results.

If the scorer fails or times out, the engine **fails open** (returns the original context unchanged) and the built-in `ContextCompressor` remains the fallback authority when typed pruning alone isn't enough. Every decision is logged with a confidence score to a local JSONL ledger, so you can audit exactly what was kept and dropped — and calibrate over time.

## Why

- **Summaries erase provenance.** Debugging a long agent session often comes down to the one exact string that a summary paraphrased away.
- **Tool results dominate context growth.** Pruning stale tool output is the single cheapest context win in tool-heavy sessions (measured ~52% cost reduction with improved task solve rates on public SWE benchmarks, arXiv 2508.21433).
- **Age is a bad proxy for relevance.** The built-in engine prunes by age; this engine asks "is this still needed?" instead — and keeps the answer's confidence.

## What it does

At each compaction pass:

1. **Protect the head and tail verbatim** (first 3 + last 20 messages by default).
2. **Index eligible tool call/result pairs** outside the protected zones.
3. **One typed call** over the whole transcript state: per interaction, a `choice` question (KEEP_FULL / KEEP_HEAD_800 / DROP) plus a `noul` question (boolean + probability, "still needed for the current task?"), folded with any `/compress <focus>` topic.
4. **Apply verdicts deterministically** — DROP stubs the body with a one-line record, KEEP_HEAD_800 truncates to 800 chars with a marker, KEEP_FULL is untouched. No prose is ever rewritten.
5. **Fail open** on any scorer error or timeout — original list returned unchanged.
6. **Still over budget?** Delegate the remainder to the built-in `ContextCompressor` summary (the same guardrail shape pattern-matched from the compaction-plugin wave). Never loops.
7. **Ledger every decision + confidence** to `decisions.jsonl` for calibration analysis.

## Requirements

- Hermes Agent (any recent build with the `context.engine` config slot and `agent/context_engine.py` ABC)
- Any configured auxiliary text model (`auxiliary:` in `config.yaml`) — **no new API keys, no new dependencies**. The scorer routes through Hermes's own auxiliary LLM router; whatever provider you already use for compression/summaries works here too.

## Install

Copy into your plugins tree:

```bash
git clone https://github.com/justinsharpe/hermes-verbatim.git
mkdir -p ~/.hermes/plugins/context_engine
cp -r hermes-verbatim/verbatim ~/.hermes/plugins/context_engine/verbatim
```

Then activate (opt-in — never auto-activated):

```yaml
# ~/.hermes/config.yaml
context:
  engine: verbatim
```

Restart Hermes, then verify with `/compress --preview` on a long session.

### Configuration

| Option | Default | Meaning |
|---|---|---|
| `scorer_timeout` | `25.0` | Per-call scorer timeout (s). Keep below `compression.context_timeout_seconds`. |
| `fallback_summarizer` | `agent.context_compressor.ContextCompressor` | Dotted path to the summarizer used when typed pruning alone can't reach target. |

Both live in `verbatim/plugin.yaml` (`config_schema`) and are read at engine construction.

## Development

```bash
git clone https://github.com/justinsharpe/hermes-verbatim.git
cd hermes-verbatim/verbatim
pytest test_verbatim_engine.py -v   # 10 tests, fully offline (mocked upstream)
python e2e_production_check.py      # 10 production-path checks against a real Hermes checkout
```

The unit suite runs offline with a mocked upstream. The E2E check exercises the real host-import path (real `ContextEngine` ABC, real `call_llm` resolution, real `ContextCompressor` construction) — point `HOST` at your Hermes checkout and run it after host upgrades.

## Design notes & honest limits

- **Not a purpose-built decision model.** The scorer is your ordinary auxiliary LLM answering strictly-formatted typed questions, not a calibrated decision model. It can be wrong — that's why every verdict carries a confidence score in the ledger, and why the built-in summary remains the fallback authority. Calibrate by reviewing `decisions.jsonl` and tuning your aux model choice.
- **Adds one upstream call per compaction pass** (not per message) — bounded by the scorer timeout, fail-open on any error.
- **Head/tail protection is verbatim by design**; user prompts and recent turns are never eligible for pruning, which puts a floor on how small the middle can get.
- **Thread-safe.** Runs on the host's pooled compression thread; ledger writes are lock-guarded; no durable external writes before the pass commits.
- **Provider-agnostic.** No third-party decision-model API, no new dependencies, no vendor lock-in — it uses the aux-model routing you already configured.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Credits

Typed-verdict retention pattern (keep/drop/truncate with confidence-tagged answers, summary fallback) adapted from the public 2026 compaction-plugin wave under a pattern-adoption posture; implemented fresh against the Hermes `ContextEngine` ABC with no third-party runtime imported.