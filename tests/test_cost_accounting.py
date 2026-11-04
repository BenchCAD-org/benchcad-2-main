"""Cost and token accounting (tools/core_score.py, tools/prices.json; a tooling-only
change, 2026-11-04). Providers price some models per request by prompt length:
Claude Haiku 5.5 over 100,000 prompt tokens pays 5x on every rate, gpt-6-astra
over 272K pays its long-context rates. Replies the harness discarded and asked
again for are not in the token means (they do not count toward the budget since
2.1) and are reported as discarded_*."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import core_score as C                                            # noqa: E402

TABLE = C.prices()
HAIKU, ASTRA, OPUS = "anthropic/claude-haiku-5-5", "openai/gpt-6-astra", "anthropic/claude-opus-5-5"


def test_the_table_cites_its_sources_and_tiers():
    assert "platform.claude.com/docs/en/about-claude/pricing" in TABLE["_sources"]["anthropic"]
    assert "developers.openai.com/api/docs/pricing" in TABLE["_sources"]["openai"]
    assert TABLE[HAIKU]["long_context"] == {"over_input_tokens": 100000, "input": 0.5, "cached": 0.05, "output": 2.5}
    assert TABLE[ASTRA]["long_context"]["over_input_tokens"] == 272000
    assert "long_context" not in TABLE[OPUS]                      # one price to 1M


def test_a_haiku_request_at_150k_input_is_priced_at_the_long_context_rate():
    tok = {"input": 150_000, "cached": 0, "output": 10_000, "calls": 1,
           "per_call": [{"input": 150_000, "cached": 0, "output": 10_000}]}
    cost, est = C.usd_of(HAIKU, tok, TABLE)
    assert est is None
    assert abs(cost - (150_000 * 0.5 + 10_000 * 2.5) / 1e6) < 1e-12        # 0.10 -> 0.50, 0.50 -> 2.50
    small = {"input": 90_000, "cached": 0, "output": 10_000, "calls": 1,
             "per_call": [{"input": 90_000, "cached": 0, "output": 10_000}]}
    assert abs(C.usd(HAIKU, small, TABLE) - (90_000 * 0.1 + 10_000 * 0.5) / 1e6) < 1e-12


def test_each_request_is_priced_on_its_own_tier_cache_reads_counted_in_the_prompt():
    calls = [{"input": 60_000, "cached": 0, "output": 1_000},          # under the threshold
             {"input": 120_000, "cached": 110_000, "output": 2_000}]   # over it, mostly cache reads
    tok = {"input": 180_000, "cached": 110_000, "output": 3_000, "calls": 2, "per_call": calls}
    want = ((60_000 * 0.1 + 1_000 * 0.5) + (10_000 * 0.5 + 110_000 * 0.05 + 2_000 * 2.5)) / 1e6
    assert abs(C.usd(HAIKU, tok, TABLE) - want) < 1e-12


def test_gpt_6_astra_over_272k():
    tok = {"input": 300_000, "cached": 100_000, "output": 5_000, "calls": 1,
           "per_call": [{"input": 300_000, "cached": 100_000, "output": 5_000}]}
    assert abs(C.usd(ASTRA, tok, TABLE) - (200_000 * 25.0 + 100_000 * 2.0 + 5_000 * 75.0) / 1e6) < 1e-12


def test_an_old_record_without_per_call_sizes_is_a_labelled_estimate():
    tok = {"input": 1_500_000, "cached": 1_200_000, "output": 60_000, "calls": 10}   # mean prompt 150k: all over
    cost, est = C.usd_of(HAIKU, tok, TABLE)
    assert est == C.ESTIMATED == "estimated: per-call sizes not recorded"
    assert abs(cost - (300_000 * 0.5 + 1_200_000 * 0.05 + 60_000 * 2.5) / 1e6) < 1e-9
    cost, est = C.usd_of(OPUS, tok, TABLE)                                          # one price: exact
    assert est is None and abs(cost - (300_000 * 5.0 + 1_200_000 * 0.2 + 60_000 * 20.0) / 1e6) < 1e-9


def _run(tmp_path, recs, model=HAIKU):
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"model": model, "effort": "max", "rounds": C.ROUNDS, "token_budget": C.TOKEN_BUDGET,
                             "harness_version": "2.1", "cases": recs}))
    return C.summarize([f])[0]


def test_a_run_with_one_discarded_retry(tmp_path):
    calls = [{"input": 50_000, "cached": 0, "output": 4_000},
             {"input": 80_000, "cached": 50_000, "output": 3_000, "discarded": True},   # thrown away, asked again
             {"input": 80_000, "cached": 50_000, "output": 2_000}]
    tok = {"input": 210_000, "cached": 100_000, "output": 9_000, "calls": 3, "output_discarded": 3_000,
           "calls_discarded": 1, "per_call": calls}
    r = _run(tmp_path, [{"case": "t1/c1", "case_id": "c1", "score": {"score": 0.5}, "tokens": tok}])
    assert r["input_tokens_per_case"] == 130_000 and r["output_tokens_per_case"] == 6_000
    assert (r["discarded_input_tokens_per_case"], r["discarded_output_tokens_per_case"], r["discarded_calls_per_case"]) \
        == (80_000, 3_000, 1)
    billed = sum(C._call_usd(TABLE[HAIKU], c["input"], c["cached"], c["output"]) for c in calls)
    assert abs(r["usd_per_case"] - billed) < 1e-12 and r["usd_estimated"] is None      # the $ is what was billed
    assert "discarded and asked again, per case: 1.00 calls, out 3000 tok, in 80000 tok" in C.report([r])


def test_old_records_still_summarise_with_the_estimate_labelled(tmp_path):
    tok = {"input": 1_500_000, "cached": 1_200_000, "output": 60_000, "calls": 10, "output_discarded": 0, "calls_discarded": 0}
    r = _run(tmp_path, [{"case": "t1/c1", "case_id": "c1", "score": {"score": 0.5}, "tokens": tok}])
    assert r["usd_estimated"] == "1 of 1 records estimated: per-call sizes not recorded"
    assert "estimated: per-call sizes not recorded" in C.report([r])
