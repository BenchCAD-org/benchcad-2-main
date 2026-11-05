"""own_agent/: the Customized-harness track's meter, results collector, benchcad.py wrapper and task builder,
offline: no Docker, no Harbor, no model API and no Core download.

The meter runs as a process in front of the stub model API (own_agent/metertest/stubapi/stub_api.py, every
reply reporting STUB_OUTPUT_TOKENS output tokens in its provider's own usage field), the way it runs in front
of the real APIs in a task. The builder is run on the test fixtures, laid out as a Core download.
"""
from __future__ import annotations

import hashlib
import http.client
import importlib.util
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OWN = ROOT / "own_agent"
METER = OWN / "template/environment/meter/meter.py"
STUB = OWN / "metertest/stubapi/stub_api.py"
STANDARD_UPSTREAMS = {"anthropic": "https://api.anthropic.com", "openai": "https://api.openai.com",
                      "gemini": "https://generativelanguage.googleapis.com"}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(port: int, proc: subprocess.Popen | None = None) -> None:
    for _ in range(200):
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"exited {proc.returncode}: {proc.stderr.read().decode()[-2000:]}")
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"nothing listening on {port}")


class Meter:
    """The meter as a process, its upstreams the given URL (default: a stub API started here)."""

    def __init__(self, tmp: Path, budget: int, upstream: str | None = None, stub_tokens: int = 1000):
        self.dir, self.procs = tmp / "meter", []
        try:
            self._start(tmp, budget, upstream, stub_tokens)
        except Exception:
            self.close()                        # nothing started here outlives a failed start
            raise

    def _start(self, tmp, budget, upstream, stub_tokens) -> None:
        if upstream is None:
            sp = _port()
            self.procs.append(subprocess.Popen(
                [sys.executable, str(STUB)], stderr=subprocess.PIPE,
                env={"PATH": os.environ.get("PATH", ""), "STUB_HOST": "127.0.0.1", "STUB_PORT": str(sp),
                     "STUB_DIR": str(tmp / "stub"), "STUB_OUTPUT_TOKENS": str(stub_tokens)}))
            _wait(sp, self.procs[-1])
            upstream = f"http://127.0.0.1:{sp}"
        self.port = _port()
        self.procs.append(subprocess.Popen(
            [sys.executable, str(METER)], stderr=subprocess.PIPE,
            env={"PATH": os.environ.get("PATH", ""), "METER_HOST": "127.0.0.1", "METER_PORT": str(self.port),
                 "METER_DIR": str(self.dir), "METER_BUDGET": str(budget), "METER_UPSTREAM_ANTHROPIC": upstream,
                 "METER_UPSTREAM_OPENAI": upstream, "METER_UPSTREAM_GEMINI": upstream}))
        _wait(self.port, self.procs[-1])

    def call(self, path: str, body: dict | None = None, headers: dict | None = None, method: str = "POST"):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        data = None if body is None else json.dumps(body).encode()
        c.request(method, path, body=data, headers={"content-type": "application/json", **(headers or {})})
        r = c.getresponse()
        out = r.status, r.getheader("Content-Type") or "", r.read()
        c.close()
        return out

    def usage(self) -> dict:
        return json.loads((self.dir / "usage.json").read_text())

    def requests(self) -> list[dict]:
        return [json.loads(l) for l in (self.dir / "requests.jsonl").read_text().splitlines()]

    def close(self) -> None:
        for p in self.procs:
            p.kill()
            p.wait()


@pytest.fixture
def meter(tmp_path):
    made = []

    def make(**kw):
        made.append(Meter(tmp_path / f"m{len(made)}", **kw))
        return made[-1]
    yield make
    for m in made:
        m.close()


ANTHROPIC = {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}
RESPONSES = {"model": "m", "input": "hi"}
CHAT = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
GEMINI = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}


# -- the meter ------------------------------------------------------------------------------------------------

def test_every_provider_shape_is_counted_streamed_or_not(meter):
    m = meter(budget=10**9)
    calls = [("/anthropic/v1/messages", ANTHROPIC), ("/anthropic/v1/messages", {**ANTHROPIC, "stream": True}),
             ("/openai/v1/responses", RESPONSES), ("/openai/v1/responses", {**RESPONSES, "stream": True}),
             ("/openai/v1/chat/completions", CHAT), ("/openai/v1/chat/completions", {**CHAT, "stream": True}),
             ("/gemini/v1beta/models/m:generateContent", GEMINI),
             ("/gemini/v1beta/models/m:streamGenerateContent?alt=sse", GEMINI),
             ("/gemini/v1beta/models/m:streamGenerateContent", GEMINI)]
    for path, body in calls:
        st, _, raw = m.call(path, body)
        assert st == 200, (path, raw[:300])
    u = m.usage()
    assert u["output_tokens"] == 1000 * len(calls)          # Gemini: candidates 900 + thoughts 100
    assert u["generation_requests"] == len(calls) and u["usage_unread"] == 0 and u["refused"] == 0
    assert [r["output_tokens"] for r in m.requests()] == [1000] * len(calls)
    assert all(r["usage_read"] for r in m.requests())


def test_a_streamed_chat_completion_is_asked_for_its_usage(meter, tmp_path):
    m = meter(budget=10**9)
    st, ctype, raw = m.call("/openai/v1/chat/completions", {**CHAT, "stream": True})
    assert st == 200 and "event-stream" in ctype and b'"completion_tokens": 1000' in raw
    assert m.usage()["include_usage_added"] == 1 and m.usage()["output_tokens"] == 1000
    sent = [json.loads(l) for l in (m.dir.parent / "stub/requests.jsonl").read_text().splitlines()]
    assert sent[-1]["stream_options"] == {"include_usage": True}


def test_the_budget_is_checked_before_a_call_and_then_every_call_is_refused(meter):
    m = meter(budget=2500)
    assert m.call("/anthropic/v1/messages", ANTHROPIC)[0] == 200
    assert m.call("/anthropic/v1/messages", {**ANTHROPIC, "stream": True})[0] == 200
    assert not m.usage()["budget_hit"]
    # 2000 spent, under 2500: this call is served in full and takes the case past the budget
    assert m.call("/openai/v1/responses", {**RESPONSES, "stream": True})[0] == 200
    u = m.usage()
    assert u["output_tokens"] == 3000 and u["budget_hit"] and u["budget_hit_at"]
    for path, body, shape in [("/openai/v1/responses", RESPONSES, "openai"),
                              ("/anthropic/v1/messages", ANTHROPIC, "anthropic"),
                              ("/gemini/v1beta/models/m:generateContent", GEMINI, "gemini")]:
        st, _, raw = m.call(path, body)
        err = json.loads(raw)
        assert st == 400 and "budget of 2,500 output tokens" in json.dumps(err), raw
        if shape == "anthropic":
            assert err["type"] == "error" and err["error"]["type"] == "invalid_request_error"
        elif shape == "openai":
            assert err["error"]["code"] == "budget_exhausted"
        else:
            assert err["error"]["code"] == 400
    u = m.usage()
    assert u["refused"] == 3 and u["generation_requests"] == 3 and u["output_tokens"] == 3000


def test_a_budget_of_zero_refuses_the_first_call(meter):
    m = meter(budget=0)
    assert m.call("/anthropic/v1/messages", ANTHROPIC)[0] == 400
    assert m.usage()["refused"] == 1 and m.usage()["generation_requests"] == 0


def test_websocket_upgrades_are_refused_and_counted(meter):
    m = meter(budget=10**9)
    st, _, raw = m.call("/openai/v1/responses", None, {"Upgrade": "websocket", "Connection": "Upgrade"}, "GET")
    assert st == 400 and b"HTTP only" in raw
    assert m.usage()["websocket_refused"] == 1 and m.usage()["output_tokens"] == 0


def test_status_and_unknown_routes(meter):
    m = meter(budget=1234)
    st, _, raw = m.call("/_meter/status", None, method="GET")
    assert st == 200 and json.loads(raw)["budget"] == 1234
    assert m.call("/elsewhere/v1/messages", ANTHROPIC)[0] == 404


def test_a_reply_whose_usage_cannot_be_read_is_counted_never_guessed(meter):
    class NoUsage(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            body = b'{"content": [{"type": "text", "text": "no usage here"}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    srv = ThreadingHTTPServer(("127.0.0.1", 0), NoUsage)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        m = meter(budget=10**9, upstream=f"http://127.0.0.1:{srv.server_address[1]}")
        assert m.call("/anthropic/v1/messages", ANTHROPIC)[0] == 200
        assert m.usage()["usage_unread"] == 1 and m.usage()["output_tokens"] == 0
        assert m.requests()[-1]["usage_read"] is False
    finally:
        srv.shutdown()


class Upstream:
    """A model API that misbehaves on purpose, in this process: POST /v1/messages answers by the request's
    model name. "slow": a full stream, 1 s late; "cut": a stream that stops after message_start (only the
    preliminary count); "long": a stream of 40 deltas, 50 ms apart, then the final usage (1000); "hold": a
    stream that waits for release() before it answers at all."""

    def __init__(self):
        self.go = threading.Event()
        self_outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                kind = req.get("model")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()

                def ev(d):
                    self.wfile.write(f"data: {json.dumps(d)}\n\n".encode())
                    self.wfile.flush()
                if kind == "slow":
                    time.sleep(1)
                if kind == "hold":
                    self_outer.go.wait(60)
                ev({"type": "message_start", "message": {"model": kind, "usage": {"output_tokens": 1}}})
                if kind == "cut":
                    return
                for _ in range(40 if kind == "long" else 1):
                    ev({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "x" * 2000}})
                    if kind == "long":
                        time.sleep(0.05)
                ev({"type": "message_delta", "usage": {"output_tokens": 1000}})
                ev({"type": "message_stop"})
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def release(self):
        self.go.set()

    def close(self):
        self.go.set()
        self.srv.shutdown()


@pytest.fixture
def upstream():
    u = Upstream()
    yield u
    u.close()


def _settle(m: Meter, n: int) -> None:
    for _ in range(200):
        if (m.dir / "requests.jsonl").exists() and len(m.requests()) >= n:
            return
        time.sleep(0.05)
    raise AssertionError(f"{n} requests never logged")


def test_a_stream_cut_before_its_final_usage_is_unread_not_its_preliminary_count(meter, upstream):
    m = meter(budget=10**9, upstream=upstream.url)
    assert m.call("/anthropic/v1/messages", {**ANTHROPIC, "model": "cut", "stream": True})[0] == 200
    assert m.usage()["usage_unread"] == 1 and m.requests()[-1]["usage_read"] is False


def test_a_reply_the_agent_stops_reading_is_still_counted_in_full(meter, upstream):
    m = meter(budget=10**9, upstream=upstream.url)
    c = socket.create_connection(("127.0.0.1", m.port))
    body = json.dumps({**ANTHROPIC, "model": "long", "stream": True}).encode()
    c.sendall(b"POST /anthropic/v1/messages HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              b"Content-Length: %d\r\n\r\n%s" % (len(body), body))
    c.recv(4096)
    c.close()                                   # gone after the first bytes
    _settle(m, 1)
    assert m.usage()["output_tokens"] == 1000 and m.usage()["usage_unread"] == 0


def _until(f, what, timeout=20):
    for _ in range(int(timeout / 0.05)):
        try:
            if f():
                return
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    raise AssertionError(f"never happened: {what}")


def test_what_requests_in_flight_may_still_generate_is_held_against_the_budget(meter, upstream):
    """A fan-out cannot pass the budget together: each request in flight reserves its own output cap, and the
    total can go past the budget by at most the reply admitted last."""
    m = meter(budget=1500, upstream=upstream.url)
    out = []
    ts = [threading.Thread(target=lambda: out.append(m.call(
        "/anthropic/v1/messages", {**ANTHROPIC, "model": "hold", "max_tokens": 1000, "stream": True})[0]))
        for _ in range(3)]
    for t in ts:
        t.start()
    # two are admitted (0 spent + 1000 reserved < 1500); the third has nothing left to reserve
    _until(lambda: m.usage()["refused"] == 1, "the third request refused")
    assert m.usage()["in_flight"] == 2 and m.usage()["reserved"] == 2000
    upstream.release()
    for t in ts:
        t.join()
    _until(lambda: m.usage()["in_flight"] == 0, "both replies accounted")
    u = m.usage()
    assert sorted(out) == [200, 200, 400] and u["max_in_flight"] == 2
    assert u["output_tokens"] == 2000 and u["over_budget"] == 500 and u["budget_hit"] and u["reserved"] == 0


def test_a_request_naming_no_output_cap_reserves_the_default(meter, upstream):
    m = meter(budget=10**9, upstream=upstream.url)
    t = threading.Thread(target=lambda: m.call("/openai/v1/responses", {**RESPONSES, "model": "hold", "stream": True}))
    t.start()
    _until(lambda: m.usage()["in_flight"] == 1, "the request in flight")
    assert m.usage()["reserved"] == 64_000
    upstream.release()
    t.join()
    _until(lambda: m.usage()["reserved"] == 0, "the reservation reconciled")


def test_only_the_model_apis_generation_paths_are_carried(meter):
    m = meter(budget=10**9)
    for method, path in [("POST", "/openai/v1/completions"), ("POST", "/openai/v1/responses/compact"),
                         ("POST", "/openai/v1/batches"), ("POST", "/openai/v1/files"), ("GET", "/openai/v1/files/x"),
                         ("POST", "/anthropic/v1/messages/batches"), ("POST", "/gemini/v1beta/cachedContents"),
                         ("GET", "/openai/v1/responses"), ("POST", "/gemini/upload/v1beta/files")]:
        st, _, raw = m.call(path, {} if method == "POST" else None, method=method)
        assert st == 403 and b"generation paths only" in raw, (path, raw)
    assert m.usage()["blocked"] == 9 and m.usage()["requests"] == 0
    for method, path in [("GET", "/openai/v1/models"), ("POST", "/anthropic/v1/messages/count_tokens"),
                         ("POST", "/gemini/v1beta/models/m:countTokens"), ("GET", "/gemini/v1beta/models")]:
        assert m.call(path, {} if method == "POST" else None, method=method)[0] == 200, path


def test_bad_framing_and_oversized_bodies_are_refused_not_buffered(meter):
    m = meter(budget=10**9)

    def raw(head: bytes):
        c = socket.create_connection(("127.0.0.1", m.port), timeout=10)
        c.sendall(head)
        out = c.recv(200)
        c.close()
        return out
    body = json.dumps(ANTHROPIC).encode()
    assert b"400" in raw(b"POST /anthropic/v1/messages HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n")
    assert b"400" in raw(b"POST /anthropic/v1/messages HTTP/1.1\r\nHost: x\r\nContent-Length: %d\r\n"
                         b"Content-Length: 1\r\n\r\n%s" % (len(body), body))
    assert b"400" in raw(b"POST /anthropic/v1/messages HTTP/1.1\r\nHost: x\r\nContent-Length: -5\r\n\r\n")
    assert b"413" in raw(b"POST /anthropic/v1/messages HTTP/1.1\r\nHost: x\r\nContent-Length: 999999999\r\n\r\n")
    assert m.usage()["output_tokens"] == 0 and m.usage()["generation_requests"] == 0
    assert m.call("/anthropic/v1/messages", ANTHROPIC)[0] == 200          # still serving


def test_usage_parsing_takes_each_calls_last_value():
    mt = _load("own_agent_meter", METER)
    u = mt.Usage("anthropic")
    u.feed_sse(b'event: message_start\ndata: {"type": "message_start", "message": {"model": "x", "usage": '
               b'{"output_tokens": 1}}}\n\ndata: {"type": "message_delta", "usage": {"output_tokens": 77}}\n')
    assert u.total() == 77 and u.found and u.model == "x"
    u = mt.Usage("gemini")
    u.feed_json(b'[{"usageMetadata": {"candidatesTokenCount": 3, "thoughtsTokenCount": 4}}, '
                b'{"usageMetadata": {"candidatesTokenCount": 10, "thoughtsTokenCount": 20}}]')
    assert u.total() == 30
    u = mt.Usage("openai")
    u.feed_sse(b'data: {"type": "response.created", "response": {"usage": null}}\n'
               b'data: {"type": "response.completed", "response": {"usage": {"output_tokens": 9}}}\ndata: [DONE]\n')
    assert u.total() == 9
    u = mt.Usage("openai")
    u.feed_sse(b'data: {"type": "response.incomplete", "response": {"usage": {"output_tokens": 4096}}}\n')
    assert u.total() == 4096 and u.found
    u = mt.Usage("anthropic")
    u.feed_sse(b'data: {"type": "message_start", "message": {"usage": {"output_tokens": 3}}}\n')
    assert u.total() == 3 and not u.found              # preliminary only
    u = mt.Usage("anthropic")
    u.feed_json(b'{"type": "message", "usage": {"output_tokens": 12}}')
    assert u.total() == 12 and u.found
    u = mt.Usage("openai")
    u.feed_json(b"not json")
    assert not u.found and u.total() == 0


# -- collect.py -----------------------------------------------------------------------------------------------

PINS = {"data_commit": "4b6fe17f63ef46090e372120d5863bcb91bef9de", "harness_commit": "a" * 40, "harness_version": "2.1",
        "scorer_digest": "c4799cc240612b038362efc5f649fc05e4bf1ce33babec46f1237092a9a7137d",
        "scorer_tree_sha256": "b" * 64, "case_tree_sha256": "c" * 64}


def _trial(job: Path, name: str, case: str, *, reward=None, no_answer=0, error=0, meter: dict | None = None,
           work_ok=True, started="2026-11-01T00:00:00Z", effort="high", pins=None, exception=None, statuses=(),
           detail=None):
    t = job / name
    (t / "verifier").mkdir(parents=True)
    (t / "artifacts").mkdir()
    (t / "agent").mkdir()
    (t / "result.json").write_text(json.dumps({
        "task_name": f"benchcad-2.1-core/{case}", "trial_name": name, "started_at": started,
        "finished_at": "2026-11-01T00:10:00Z",
        "config": {"agent": {"name": "claude-code", "model_name": "m-1", "kwargs": {"reasoning_effort": effort}}},
        "agent_info": {"name": "claude-code", "version": "9.9", "model_info": {"name": "m-1", "provider": "anthropic"}},
        "agent_result": {"n_input_tokens": 10, "n_output_tokens": 20, "n_cache_tokens": 5, "cost_usd": 0.5},
        "agent_execution": {"started_at": started, "finished_at": "2026-11-01T00:05:00Z"},
        "exception_info": {"exception_type": exception} if exception else None}))
    (t / "artifacts/manifest.json").write_text(json.dumps([{"source": "/work", "status": "ok" if work_ok else "failed"}]))
    if meter is not False:
        (t / "artifacts/meter").mkdir()
        (t / "artifacts/meter/usage.json").write_text(json.dumps({
            "budget": 500000, "output_tokens": 1000, "budget_hit": False, "refused": 0, "generation_requests": 2,
            "usage_unread": 0, "websocket_refused": 0, "upstreams": STANDARD_UPSTREAMS, **(meter or {})}))
        (t / "artifacts/meter/requests.jsonl").write_text("".join(
            json.dumps({"provider": "anthropic", "path": "/v1/messages", "status": st}) + "\n" for st in statuses))
    if reward is not None or no_answer or error:
        (t / "verifier/reward.json").write_text(json.dumps({"reward": reward or 0.0, "error": error,
                                                            "no_answer": no_answer}))
    (t / "verifier/detail.json").write_text(json.dumps({"pins": {"case": case, "task": case[:2].replace("t", "task"),
                                                                 "kind": "part", **PINS, **(pins or {})},
                                                        **(detail or {})}))
    # ATIF: embedded subagent trajectories hang off the trajectory, nested
    (t / "agent/trajectory.json").write_text(json.dumps({"steps": [{"tool_calls": [{}, {}]}, {"tool_calls": [{}]}],
        "subagent_trajectories": [{"steps": [{"tool_calls": [{}]}],
                                   "subagent_trajectories": [{"steps": [{"tool_calls": [{}, {}]}]}]}]}))


def _collect(*jobs: Path, tmp_path: Path):
    out, summ = tmp_path / "results.jsonl", tmp_path / "summary.json"
    r = subprocess.run([sys.executable, str(OWN / "collect.py"), *map(str, jobs), "--out", str(out),
                        "--summary", str(summ)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return [json.loads(l) for l in out.read_text().splitlines()], json.loads(summ.read_text())


def test_collect_averages_passes_per_case_then_means_over_cases(tmp_path):
    job = tmp_path / "job"
    _trial(job, "a1", "t1-case001", reward=0.5)
    _trial(job, "a2", "t1-case001", reward=0.7, started="2026-11-01T01:00:00Z")
    _trial(job, "b1", "t3-case001", no_answer=1)
    (job / "run.json").write_text(json.dumps({"agent": "claude-code-bare", "transport": "http", "budget": 500000,
                                              "cases": "task1/case001,task3/case001", "passes": 2}))
    _trial(job, "b2", "t3-case001", no_answer=1, started="2026-11-01T01:00:00Z")
    rows, summary = _collect(job, tmp_path=tmp_path)
    by = {(r["case"], r["pass_index"]): r for r in rows}
    assert by[("t1-case001", 0)]["score"] == 0.5 and by[("t1-case001", 1)]["score"] == 0.7
    assert by[("t3-case001", 0)]["status"] == "no_submission" and by[("t3-case001", 0)]["score"] == 0.0
    r = by[("t1-case001", 0)]
    assert r["agent"] == "claude-code-bare" and r["model"] == "anthropic/m-1" and r["effort"] == "high"
    assert r["tool_calls"] == 6 and r["output_tokens"] == 1000 and r["reportable"] and r["wall_time_s"] == 300.0
    (line,) = summary
    assert line["track"] == "customized-harness" and line["over_budget"] == []
    assert line["mean"] == pytest.approx((0.6 + 0.0) / 2) and line["label"] == "SUBSET" and line["passes"] == [2]
    schema = json.loads((OWN / "result.schema.json").read_text())
    for x in rows:
        assert set(schema["required"]) <= set(x), set(schema["required"]) - set(x)


def test_collect_gives_no_line_while_a_cell_has_no_score(tmp_path):
    job = tmp_path / "job"
    _trial(job, "a", "t1-case001", reward=0.9)
    _trial(job, "b", "t1-case002", error=1)                             # the verifier could not finish
    _trial(job, "e", "t1-case005", no_answer=1, work_ok=False)           # /work never reached the verifier
    rows, (line,) = _collect(job, tmp_path=tmp_path)
    assert [r["status"] for r in rows] == ["scored", "pending", "pending"]
    assert line["mean"] is None and line["label"] == "INCOMPLETE" and len(line["pending"]) == 2
    # no meter record: the budget cannot be shown to have held
    _trial(job, "c", "t1-case003", reward=0.9, meter=False)
    # the agent ended on a provider error it did not get past, before writing an answer: not a zero
    _trial(job, "f", "t1-case006", no_answer=1, exception="NonZeroAgentExitCodeError", statuses=(200, 529))
    # ... but one that ended on its own after a good reply keeps its score, and the budget's refusal is the end
    _trial(job, "g", "t1-case007", no_answer=1, exception="NonZeroAgentExitCodeError", statuses=(200, 200))
    _trial(job, "h", "t1-case008", reward=0.4, exception="NonZeroAgentExitCodeError", statuses=(529,),
           meter={"budget_hit": True})
    (job / "t1-case009__x").mkdir()                                      # a trial Harbor never finished
    rows, (line,) = _collect(job, tmp_path=tmp_path)
    by = {r["case"]: r for r in rows}
    assert by["t1-case003"]["status"] == by["t1-case006"]["status"] == by["t1-case009"]["status"] == "pending"
    assert "HTTP 529" in by["t1-case006"]["why"]
    assert by["t1-case007"]["status"] == "no_submission" and by["t1-case008"]["score"] == 0.4
    assert line["mean"] is None and line["label"] == "INCOMPLETE"


def test_collect_scores_a_trial_whose_usage_could_not_be_read_but_never_reports_it(tmp_path):
    """A reply the meter could not read would otherwise be a free re-run: the cell keeps its score and the
    line is not reportable."""
    job = tmp_path / "job"
    _trial(job, "a", "t1-case001", reward=0.9, meter={"usage_unread": 2})
    rows, (line,) = _collect(job, tmp_path=tmp_path)
    assert rows[0]["status"] == "scored" and rows[0]["score"] == 0.9 and rows[0]["usage_unread"] == 2
    assert not rows[0]["reportable"] and line["mean"] == 0.9 and line["label"] == "NOT REPORTABLE"


def test_collect_allows_one_rerun_a_pass_then_counts_it_zero(tmp_path):
    job = tmp_path / "job"
    _trial(job, "a1", "t1-case001", error=1)
    (job / "run.json").write_text(json.dumps({"agent": "claude-code", "model": "anthropic/m-1", "effort": "high",
                                              "cases": "task1/case001", "passes": 1}))                             # the verifier could not finish
    _, (line,) = _collect(job, tmp_path=tmp_path)
    assert line["pending"] == ["t1-case001"] and line["label"] == "INCOMPLETE" and line["reruns"] == 0
    _trial(job, "a2", "t1-case001", error=1, started="2026-11-01T02:00:00Z")   # the re-run fails too
    _, (line,) = _collect(job, tmp_path=tmp_path)
    assert line["pending"] == [] and line["zero_after_rerun"] == ["t1-case001"]
    assert line["mean"] == 0.0 and line["reruns"] == 1 and line["label"] == "SUBSET"


def test_collect_takes_the_rerun_that_scored(tmp_path):
    job = tmp_path / "job"
    _trial(job, "a1", "t1-case001", error=1)
    (job / "run.json").write_text(json.dumps({"agent": "claude-code", "model": "anthropic/m-1", "effort": "high",
                                              "cases": "task1/case001", "passes": 1}))
    _trial(job, "a2", "t1-case001", reward=0.8, started="2026-11-01T02:00:00Z")
    _, (line,) = _collect(job, tmp_path=tmp_path)
    assert line["mean"] == 0.8 and line["pending"] == [] and line["zero_after_rerun"] == []
    assert line["reruns"] == 1 and line["label"] == "SUBSET"


def test_collect_lists_a_case_the_scorer_could_not_score(tmp_path):
    job = tmp_path / "job"
    _trial(job, "a", "t1-case001", no_answer=1, detail={"scorer_failed": "RecursionError"})
    rows, (line,) = _collect(job, tmp_path=tmp_path)
    assert rows[0]["status"] == "no_submission" and rows[0]["score"] == 0.0
    assert rows[0]["scorer_failed"] == "RecursionError" and line["scorer_failed"] == ["t1-case001"]


def test_collect_counts_a_missing_pass_as_pending(tmp_path):
    job = tmp_path / "job"
    _trial(job, "a1", "t1-case001", reward=0.9)
    (job / "run.json").write_text(json.dumps({"agent": "claude-code", "model": "anthropic/m-1", "effort": "high",
                                              "cases": "task1/case001,task2/case001", "passes": 2}))
    rows, (line,) = _collect(job, tmp_path=tmp_path)
    assert sorted((r["case"], r["status"]) for r in rows) == [
        ("t1-case001", "pending"), ("t1-case001", "scored"), ("t2-case001", "pending"), ("t2-case001", "pending")]
    assert line["label"] == "INCOMPLETE" and line["mean"] is None


def test_collect_labels_a_run_off_the_standard_meter_not_reportable(tmp_path):
    job = tmp_path / "job"
    _trial(job, "a", "t1-case001", reward=0.9, meter={"budget": 5000})
    _trial(job, "b", "t1-case002", reward=0.9, meter={"upstreams": {**STANDARD_UPSTREAMS, "openai": "http://stub"}})
    rows, (line,) = _collect(job, tmp_path=tmp_path)
    assert not any(r["reportable"] for r in rows) and line["label"] == "NOT REPORTABLE"


def test_collect_shows_every_trial_that_went_past_the_budget_and_still_reports_it(tmp_path):
    job = tmp_path / "job"
    _trial(job, "a", "t1-case001", reward=0.9, meter={"budget_hit": True, "output_tokens": 512000, "over_budget": 12000,
                                                    "max_in_flight": 3})
    _trial(job, "b", "t1-case002", reward=0.5)
    rows, (line,) = _collect(job, tmp_path=tmp_path)
    assert all(r["reportable"] for r in rows) and line["label"] == "SUBSET"
    assert line["over_budget"] == [{"case": "t1-case001", "trial": "a", "over_budget": 12000}]
    assert line["max_in_flight"] == 3


def test_collect_never_averages_trials_of_different_builds_into_one_line(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _trial(a, "x", "t1-case001", reward=0.9)
    _trial(b, "y", "t1-case002", reward=0.9, pins={"scorer_tree_sha256": "d" * 64})
    _, (line,) = _collect(a, b, tmp_path=tmp_path)
    assert line["label"] == "NOT REPORTABLE"
    _trial(b, "z", "t1-case003", reward=0.9, pins={"scorer_digest": "f" * 64})
    rows, _ = _collect(b, tmp_path=tmp_path)
    assert not next(r for r in rows if r["case"] == "t1-case003")["reportable"]


# -- benchcad.py ----------------------------------------------------------------------------------------------

def _wrapper(tmp: Path, *args: str):
    data = tmp / "data"
    data.mkdir(exist_ok=True)
    (data / "MANIFEST.sha256").write_text("")
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env.update({"BENCHCAD_CACHE": str(tmp / "cache"), "BENCHCAD_DATA": str(data)})
    return subprocess.run([sys.executable, str(OWN / "benchcad.py"), "run", "--jobs-dir", str(tmp / "jobs"),
                           "--dry-run", *args], capture_output=True, text=True, env=env, cwd=ROOT)


def _harbor_cmd(out: str) -> str:
    return next(l for l in out.splitlines() if " run -p " in l)


def test_wrapper_points_each_agent_at_the_meter(tmp_path):
    r = _wrapper(tmp_path, "--agent", "claude-code", "--model", "anthropic/claude-opus-5-5", "--effort", "high",
                 "--cases", "task1/case001")
    assert r.returncode == 0, r.stderr
    assert "run_build.sh" in r.stdout and "--cases task1/case001" in r.stdout
    cmd = _harbor_cmd(r.stdout)
    for want in ("+ ANTHROPIC_BASE_URL=http://meter:8080/anthropic ", "-a claude-code",
                 "-m claude-opus-5-5 ", "--ak reasoning_effort=high", "--ae ANTHROPIC_BASE_URL=http://meter:8080/anthropic",
                 "-i '*t1-case001'", "-k 1", "--max-retries 0", "-y"):
        assert want in cmd, (want, cmd)
    assert "cleanup:" in r.stdout and "note: budget" not in r.stderr
    r = _wrapper(tmp_path, "--agent", "codex", "--model", "openai/gpt-6-astra", "--effort", "xhigh")
    cmd = _harbor_cmd(r.stdout)
    assert "--agent agents.codex_http:CodexHttp" in cmd and "-m openai/gpt-6-astra" in cmd
    assert "--ae OPENAI_BASE_URL=http://meter:8080/openai/v1" in cmd
    r = _wrapper(tmp_path, "--agent", "gemini-cli", "--model", "gemini/g-1")
    assert "--ae GOOGLE_GEMINI_BASE_URL=http://meter:8080/gemini" in _harbor_cmd(r.stdout)
    r = _wrapper(tmp_path, "--agent", "openhands", "--model", "openai/gpt-6-astra")
    cmd = _harbor_cmd(r.stdout)
    assert "-a openhands-sdk" in cmd and "--ak reasoning_effort=high" in cmd and "LLM_BASE_URL=" in cmd


def test_wrapper_refuses_what_would_break_the_budget(tmp_path):
    r = _wrapper(tmp_path, "--agent", "claude-code", "--model", "anthropic/m", "--", "--max-retries", "2")
    assert r.returncode != 0 and "--max-retries is refused" in r.stderr
    r = _wrapper(tmp_path, "--agent", "claude-code", "--model", "openai/m")
    assert r.returncode != 0 and "has not been tested" in r.stderr
    r = _wrapper(tmp_path, "--agent", "mini-swe-agent", "--model", "openai/m", "--budget", "5k")
    assert r.returncode == 0 and "budget 5,000 is not 2.1's 500,000" in r.stderr
    assert "own-agent-tasks-5000" in r.stdout


# -- build.py -------------------------------------------------------------------------------------------------

FIXTURES = {"task1": "t1", "task2": "t2", "task3": "t3", "task4": "t4", "task5": "t5", "task6": "t6"}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Every task kind built from the test fixtures, laid out as a Core download."""
    tmp = tmp_path_factory.mktemp("own_build")
    local = os.environ.get("CADENV_LOCAL")                              # build_case sets it to stage locally
    import envs.common.sandbox as sb
    ready, sb._docker_ready = sb._docker_ready, lambda: False           # and never in docker, whatever the host has
    data, out = tmp / "data", tmp / "out"
    for task, fx in FIXTURES.items():
        shutil.copytree(ROOT / "tests/fixtures" / fx / "case1", data / task / "cases/case001")
    b = _load("own_agent_build", OWN / "build.py")
    sys.path.insert(0, str(ROOT))
    meta = {"harness_commit": "0" * 40, "harness_tag": None, "scorer_child": b.run_py_constants()["SCORER_CHILD"],
            "verifier_memory_mb": {"default": 8192}}
    out.mkdir()
    try:
        rows = {task: b.build_case(data, task, "case001", out, meta) for task in FIXTURES}
    finally:
        os.environ.pop("CADENV_LOCAL") if local is None else os.environ.update(CADENV_LOCAL=local)
        sb._docker_ready = ready
    return b, out, rows


def test_build_writes_a_task_per_case_with_no_reference_on_the_agent_side(built):
    b, out, rows = built
    for task, fx in FIXTURES.items():
        d = out / f"{fx}-case001"
        staged = d / "environment/task"
        assert (staged / "tools.py").exists()
        assert not [p for p in staged.rglob("*") if p.name in ("gt", "case.json") or p.suffix in (".pdf",)]
        assert not list((d / "environment").rglob("gt.step")) and not list((d / "environment").rglob("gt_graph.json"))
        assert (d / "tests/case/case.json").exists() and (d / "solution/answer.py").exists()
        text = (d / "instruction.md").read_text()
        assert "500,000 output tokens" in text and "/work/answer.py" in text and "8 hours" in text
        assert not re.search(r"\d+ rounds|Round \d|```submit", text), text   # no round limit or block protocol
        pins = json.loads((d / "tests/pins.json").read_text())
        assert pins["scorer_digest"] == b.SCORER_DIGEST and pins["token_budget"] == 500_000
        assert pins["score_timeout_s"] == 14400 and pins["scorer_tree_sha256"] == b.tree_sha256(d / "tests/scorer")
        toml = (d / "task.toml").read_text()
        assert "{{" not in toml and 'allowed_hosts = ["meter"]' in toml and 'environment_mode = "separate"' in toml
        assert 'METER_BUDGET: "500000"' in (d / "environment/docker-compose.yaml").read_text()
        assert rows[task]["kind"] == json.loads((d / "tests/case/case.json").read_text())["kind"]


def test_build_scorer_copy_is_this_checkouts(built):
    _, out, _ = built
    scorer = out / "t3-case001/tests/scorer"
    for f in ("envs/common/score_case.py", "envs/common/ecad_graph/spatial_reference.py", "uv.lock",
              "tools/core_score.py"):
        assert (scorer / f).read_bytes() == (ROOT / f).read_bytes(), f


def test_build_refuses_data_that_is_not_core_v21(tmp_path):
    from envs.common.ecad_graph.spatial_reference import scorer_digest
    data = tmp_path / "data"
    data.mkdir()
    for ds, why in [({"name": "benchcad-2.0-core", "version": "1.0", "scorer_digest": "f" * 64}, "scorer_digest"),
                    ({"name": "benchcad-2.0-public", "version": "1.0", "scorer_digest": scorer_digest()}, "not benchcad-2.0-core")]:
        (data / "dataset.json").write_text(json.dumps(ds))
        r = subprocess.run([sys.executable, str(OWN / "build.py"), "--data", str(data), "--out", str(tmp_path / "o")],
                           capture_output=True, text=True)
        assert r.returncode != 0 and why in r.stderr, r.stderr


def test_build_checks_the_manifest_both_ways(tmp_path):
    b = _load("own_agent_build_m", OWN / "build.py")
    case = tmp_path / "task1/cases/case001"
    (case / "input").mkdir(parents=True)
    (case / "case.json").write_text("{}")
    (case / "input/sheet.pdf").write_text("x")
    listed = {p: hashlib.sha256(Path(tmp_path / p).read_bytes()).hexdigest()
              for p in ("task1/cases/case001/case.json", "task1/cases/case001/input/sheet.pdf")}
    (tmp_path / "MANIFEST.sha256").write_text("".join(f"{v}  {k}\n" for k, v in listed.items()))
    b.check_manifest(tmp_path, [case])
    (case / "input/sheet.pdf").unlink()
    with pytest.raises(SystemExit, match="missing"):
        b.check_manifest(tmp_path, [case])
    (case / "input/sheet.pdf").write_text("x")
    (case / "input/extra.png").write_text("y")
    with pytest.raises(SystemExit, match="not in MANIFEST"):
        b.check_manifest(tmp_path, [case])


def test_cleanup_removes_the_core_copies_and_keeps_what_the_agent_made(tmp_path):
    w = _load("own_agent_wrapper", OWN / "benchcad.py")
    staged = tmp_path / "tasks/t1-case001/environment/task"
    (staged / "hires").mkdir(parents=True)
    for f in ("sheet_1.png", "hires/sheet_1.png", "tools.py", "bom.json"):
        (staged / f).write_text("core")
    work = tmp_path / "job/t1-case001__abc/artifacts/work"
    (work / "hires").mkdir(parents=True)
    (work / "out").mkdir()
    for f in ("sheet_1.png", "hires/sheet_1.png", "tools.py", "bom.json", "answer.py", "helpers.py", "out/crop.PNG",
              "out/part.step", "notes.md"):
        (work / f).write_text("x")
    (work.parent.parent / "result.json").write_text(json.dumps({"task_name": "benchcad-2.1-core/t1-case001"}))
    w.cleanup(tmp_path / "tasks", tmp_path / "job", dry=False)
    left = sorted(p.relative_to(work).as_posix() for p in work.rglob("*") if p.is_file())
    assert left == ["answer.py", "helpers.py", "notes.md", "out/part.step"]
    assert not (tmp_path / "tasks").exists()


# -- the verifier's own file handling (template/tests/run) -----------------------------------------------------

RUN = OWN / "template/tests/run"


def test_copy_in_takes_regular_files_only_and_follows_nothing(tmp_path):
    work, out, outside = tmp_path / "work", tmp_path / "run", tmp_path / "outside"
    (work / "sub").mkdir(parents=True)
    outside.mkdir()
    (outside / "reference.step").write_text("the reference")
    (work / "answer.py").write_text("x = 1\n")
    (work / "sub/part.step").write_text("a part")
    (work / "tools.py").write_text("not the harness's")                  # dropped: /tests holds the real one
    (work / "final.step").symlink_to(outside / "reference.step")         # a link out of /work
    (work / "linkdir").symlink_to(outside)                               # a directory link
    (work / "sub/also.step").symlink_to(outside / "reference.step")
    os.mkfifo(work / "pipe")
    r = subprocess.run([sys.executable, str(RUN / "copy_in.py"), str(work), str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = sorted(p.relative_to(out).as_posix() for p in out.rglob("*"))
    assert got == ["answer.py", "sub", "sub/part.step"]
    assert not any(p.is_symlink() for p in out.rglob("*"))
    assert "a link" in r.stderr and "not a regular file" in r.stderr
    assert (outside / "reference.step").read_text() == "the reference"    # untouched


def test_copy_in_caps_what_one_file_can_bring(tmp_path, monkeypatch):
    work, out = tmp_path / "work", tmp_path / "run"
    work.mkdir()
    (work / "big.step").write_bytes(b"x" * 4096)
    (work / "answer.py").write_text("x = 1\n")
    src = (RUN / "copy_in.py").read_text().replace("MAX_FILE = 4 << 30", "MAX_FILE = 1024")
    script = tmp_path / "copy_in_small.py"
    script.write_text(src)
    r = subprocess.run([sys.executable, str(script), str(work), str(out)], capture_output=True, text=True)
    assert r.returncode == 0 and "over the size cap" in r.stderr
    assert sorted(p.name for p in out.iterdir()) == ["answer.py"]


def test_the_seal_catches_a_change_to_the_verifiers_own_files(tmp_path):
    tests = tmp_path / "tests"
    (tests / "run").mkdir(parents=True)
    (tests / "scorer/tools").mkdir(parents=True)
    (tests / "test.sh").write_text("#!/bin/bash\n")
    (tests / "pins.json").write_text("{}")
    (tests / "run/score.py").write_text("print('the real scorer')\n")
    (tests / "scorer/harness_scorer_child.py").write_text("child\n")
    (tests / "scorer/tools/core_score.py").write_text("resolve\n")
    src = (RUN / "seal.py").read_text().replace('"/tests', f'"{tests}')
    seal_py = tmp_path / "seal.py"
    seal_py.write_text(src)
    seal = tmp_path / "seal.json"
    assert subprocess.run([sys.executable, str(seal_py), "record", str(seal)]).returncode == 0
    assert subprocess.run([sys.executable, str(seal_py), "verify", str(seal)]).returncode == 0
    (tests / "run/score.py").write_text("print('whatever the answer wanted')\n")
    r = subprocess.run([sys.executable, str(seal_py), "verify", str(seal)], capture_output=True, text=True)
    assert r.returncode == 1 and "score.py changed during the answer run" in r.stderr


def test_the_tmp_wipe_leaves_nothing_of_the_answers_behind(tmp_path):
    shared, private, outside = tmp_path / "tmp", tmp_path / "private", tmp_path / "outside"
    shared.mkdir()
    outside.mkdir()
    (outside / "keep.txt").write_text("not the answer's to delete")
    (shared / ".cache-sandbox").mkdir()
    (shared / ".cache-sandbox/planted.py").write_text("import os\n")
    (shared / "matplotlibrc").write_text("x")
    (shared / "link").symlink_to(outside / "keep.txt")
    (shared / "dirlink").symlink_to(outside)
    private.mkdir()
    (private / "stale").write_text("from a previous run")
    r = subprocess.run([sys.executable, str(RUN / "wipe_tmp.py"), str(private), str(shared)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert list(shared.iterdir()) == []
    assert sorted(p.name for p in private.iterdir()) == ["cache", "config", "home", "mpl", "tmp"]
    assert (outside / "keep.txt").read_text() == "not the answer's to delete"
    assert oct(private.stat().st_mode)[-3:] == "700"
