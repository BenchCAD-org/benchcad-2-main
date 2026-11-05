"""The case's model-API meter: the agent's only way out, and where its output-token budget is kept.

The agent's container can reach nothing but this service (task.toml: the agent phase allows the host `meter`
only). It forwards by path to one fixed upstream per provider and reads the usage of every response as it
passes, streamed or not:

    /anthropic/...  -> https://api.anthropic.com/...                    usage.output_tokens (thinking included)
    /openai/...     -> https://api.openai.com/...                       usage.output_tokens (reasoning included),
                                                                        or usage.completion_tokens
    /gemini/...     -> https://generativelanguage.googleapis.com/...    usageMetadata.candidatesTokenCount
                                                                        + thoughtsTokenCount

The budget is 2.1's: 500,000 output tokens per case, thinking included, every call counted (a subagent's and a
small side model's too). It is checked before a request is forwarded, never inside one, as harness/run.py
checks it between calls: a request made while budget remains runs to its end, and once the sum reaches the
budget every further request is refused with a 400 in the provider's own error shape, which ends the agent.
Requests made at the same time (subagents) are each admitted against the same balance; how many were in
flight, and how far the total went past the budget, is recorded (max_in_flight, over_budget).

Only the model APIs' generation, token-count and model-list paths are carried (PATHS); anything else (files,
batches, fine-tuning, ...) gets a 403 and is counted in blocked. A reply's tokens are added before its last
byte reaches the agent, and a reply the agent stops reading is read to its end, so its tokens still count.

/meter/usage.json      the running totals (budget, output_tokens, budget_hit, refused, usage_unread, ...)
/meter/requests.jsonl  one line per request: provider, path, status, model, output_tokens, seconds (no bodies)
GET /_meter/status     the totals, for the agent: what 2.1 shows after every round

A 2xx reply from a generation endpoint whose final usage cannot be read is counted in usage_unread and logged;
it is never guessed (an Anthropic stream that ends before its message_delta has only a preliminary count). A
streamed Chat Completions request is sent on with stream_options.include_usage = true, the one way such a
stream reports its usage; nothing else in a request is changed. A WebSocket upgrade is refused (the meter
reads HTTP replies only) and counted in websocket_refused.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

BUDGET = int(os.environ.get("METER_BUDGET", "500000"))
STATE = Path(os.environ.get("METER_DIR", "/meter"))
UPSTREAMS = {
    "anthropic": os.environ.get("METER_UPSTREAM_ANTHROPIC", "https://api.anthropic.com"),
    "openai": os.environ.get("METER_UPSTREAM_OPENAI", "https://api.openai.com"),
    "gemini": os.environ.get("METER_UPSTREAM_GEMINI", "https://generativelanguage.googleapis.com"),
}
GENERATION = re.compile(r"(/messages$|/responses$|/responses/compact$|/chat/completions$|/completions$|"
                        r":generateContent$|:streamGenerateContent$)")
# What the meter carries, per provider: (method, path) after the route prefix. Everything else is refused.
PATHS = {
    "anthropic": [("POST", r"/v1/messages"), ("POST", r"/v1/messages/count_tokens"), ("GET", r"/v1/models(/[^/]+)?")],
    "openai": [("POST", r"/v1/responses"), ("POST", r"/v1/responses/compact"), ("POST", r"/v1/chat/completions"),
               ("POST", r"/v1/completions"), ("GET", r"/v1/models(/[^/]+)?")],
    "gemini": [("POST", r"/v1(beta|alpha)?/models/[^/:]+:(generateContent|streamGenerateContent|countTokens)"),
               ("GET", r"/v1(beta|alpha)?/models(/[^/:]+)?")],
}
HOP = {"host", "connection", "keep-alive", "proxy-connection", "transfer-encoding", "te", "trailer", "upgrade",
       "content-length", "accept-encoding"}
LOCK = threading.Lock()
TOTALS = {"budget": BUDGET, "output_tokens": 0, "requests": 0, "generation_requests": 0, "refused": 0,
          "budget_hit": False, "budget_hit_at": None, "over_budget": 0, "usage_unread": 0, "websocket_refused": 0,
          "blocked": 0, "include_usage_added": 0, "in_flight": 0, "max_in_flight": 0, "upstreams": UPSTREAMS,
          "started_at": time.time()}


def save() -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = STATE / "usage.json.new"
    tmp.write_text(json.dumps(TOTALS, indent=1) + "\n")
    tmp.replace(STATE / "usage.json")


def log(row: dict) -> None:
    with open(STATE / "requests.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


class Usage:
    """Output tokens of one response, from its JSON body or from its server-sent events. One response is one
    model call: Anthropic's stream reports a preliminary output_tokens in message_start and the final,
    cumulative one in message_delta; OpenAI's in response.completed / response.incomplete / response.failed
    (or a chat completion's usage chunk); Gemini's every chunk repeats the totals so far. In each case the last
    final value seen is the call's."""

    def __init__(self, provider: str):
        self.provider, self.buf, self.found, self.model, self.value = provider, b"", False, None, 0

    def _set(self, n, final: bool = True) -> None:
        if isinstance(n, int):
            self.value = n
            self.found = self.found or final

    def _event(self, d) -> None:
        if not isinstance(d, dict):
            return
        p = self.provider
        if p == "anthropic":
            t = d.get("type")
            m = d.get("message") if t == "message_start" else d
            if isinstance(m, dict):
                self.model = m.get("model") or self.model
                self._set((m.get("usage") or {}).get("output_tokens"), final=t != "message_start")
        elif p == "openai":
            r = d.get("response") if d.get("type") in ("response.completed", "response.incomplete",
                                                       "response.failed") else d
            if isinstance(r, dict) and isinstance(r.get("usage"), dict):
                u = r["usage"]
                self._set(u.get("output_tokens", u.get("completion_tokens")))
                self.model = r.get("model") or self.model
        elif p == "gemini":
            u = d.get("usageMetadata")
            if isinstance(u, dict) and ("candidatesTokenCount" in u or "thoughtsTokenCount" in u):
                self._set(int(u.get("candidatesTokenCount") or 0) + int(u.get("thoughtsTokenCount") or 0))
                self.model = d.get("modelVersion") or self.model

    def feed_sse(self, chunk: bytes) -> None:
        self.buf += chunk
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            line = line.strip()
            if line.startswith(b"data:"):
                body = line[5:].strip()
                if body and body != b"[DONE]":
                    try:
                        self._event(json.loads(body))
                    except ValueError:
                        pass

    def feed_json(self, body: bytes) -> None:
        try:
            d = json.loads(body)
        except ValueError:
            return
        for x in d if isinstance(d, list) else [d]:
            self._event(x)

    def total(self) -> int:
        return self.value


def refusal(provider: str) -> bytes:
    msg = (f"BenchCAD: this case's budget of {BUDGET:,} output tokens (thinking included) is spent; "
           f"no further model requests are served. The /work/answer.py that exists now is scored.")
    if provider == "anthropic":
        return json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": msg}}).encode()
    if provider == "openai":
        return json.dumps({"error": {"message": msg, "type": "invalid_request_error", "code": "budget_exhausted"}}).encode()
    return json.dumps({"error": {"code": 400, "message": msg, "status": "FAILED_PRECONDITION"}}).encode()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "benchcad-meter"

    def log_message(self, fmt, *args):            # requests.jsonl is the log
        pass

    def _reply(self, code: int, body: bytes, ctype="application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _proxy(self) -> None:
        t0 = time.time()
        if self.path.startswith("/_meter/status"):
            with LOCK:
                return self._reply(200, json.dumps(TOTALS).encode())
        parts = self.path.split("/", 2)
        provider = parts[1] if len(parts) > 1 else ""
        if provider not in UPSTREAMS:
            return self._reply(404, json.dumps({"error": f"unknown route {self.path[:80]}"}).encode())
        rest = "/" + (parts[2] if len(parts) > 2 else "")
        path_only = urlsplit(rest).path
        gen = bool(GENERATION.search(path_only))
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        if not any(m == self.command and re.fullmatch(rx, path_only) for m, rx in PATHS[provider]) \
                and "websocket" not in (self.headers.get("Upgrade") or "").lower():
            with LOCK:
                TOTALS["blocked"] += 1
                save()
            log({"t": round(t0, 3), "provider": provider, "path": path_only[:200], "method": self.command,
                 "status": 403, "blocked": True})
            return self._reply(403, json.dumps({"error": f"the BenchCAD meter carries the model API's generation "
                                                         f"paths only, not {self.command} {path_only[:80]}"}).encode())
        if "websocket" in (self.headers.get("Upgrade") or "").lower():
            with LOCK:
                TOTALS["websocket_refused"] += 1
                save()
            log({"t": round(t0, 3), "provider": provider, "path": path_only, "status": 400, "websocket": True})
            return self._reply(400, json.dumps({"error": "the BenchCAD meter carries HTTP only; use the HTTP transport"}).encode())
        added = False
        if path_only.endswith("/chat/completions") and body:
            try:
                req = json.loads(body)
            except ValueError:
                req = None
            if isinstance(req, dict) and req.get("stream") and not (req.get("stream_options") or {}).get("include_usage"):
                req["stream_options"] = {**(req.get("stream_options") or {}), "include_usage": True}
                body, added = json.dumps(req).encode(), True
        with LOCK:
            TOTALS["include_usage_added"] += added
            TOTALS["requests"] += 1
            if TOTALS["output_tokens"] >= BUDGET:
                TOTALS["refused"] += 1
                save()
                log({"t": round(t0, 3), "provider": provider, "path": path_only, "status": 400, "refused": True})
                return self._reply(400, refusal(provider))
            if gen:
                TOTALS["generation_requests"] += 1
            TOTALS["in_flight"] += 1
            TOTALS["max_in_flight"] = max(TOTALS["max_in_flight"], TOTALS["in_flight"])
        try:
            self._forward(t0, provider, rest, path_only, gen, body)
        finally:
            with LOCK:
                TOTALS["in_flight"] -= 1
                save()

    def _forward(self, t0, provider, rest, path_only, gen, body) -> None:
        up = urlsplit(UPSTREAMS[provider])
        conn_cls = http.client.HTTPSConnection if up.scheme == "https" else http.client.HTTPConnection
        kw = {"context": ssl.create_default_context()} if up.scheme == "https" else {}
        conn = conn_cls(up.hostname, up.port, timeout=3600, **kw)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        headers["Accept-Encoding"] = "identity"      # the usage has to be readable on the way through
        try:
            conn.request(self.command, (up.path.rstrip("/") + rest) or "/", body=body or None, headers=headers)
            resp = conn.getresponse()
        except OSError as e:
            log({"t": round(t0, 3), "provider": provider, "path": path_only, "status": 502, "error": str(e)[:200]})
            return self._reply(502, json.dumps({"error": f"upstream unreachable: {e}"}).encode())
        usage = Usage(provider)
        sse = "text/event-stream" in (resp.getheader("Content-Type") or "")
        self.send_response(resp.status, resp.reason)
        for k, v in resp.getheaders():
            if k.lower() not in HOP and k.lower() != "content-encoding":
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        whole = b""
        counted = False

        def account() -> None:
            # before the reply's last chunk goes out: once the client has the whole reply, its tokens are in
            # the totals, so a request it sends next is checked against them
            nonlocal counted
            if counted:
                return
            counted = True
            if sse:
                usage.feed_sse(b"\n")
            else:
                usage.feed_json(whole)
            conn.close()
            out = usage.total()
            ok = 200 <= resp.status < 300
            with LOCK:
                if ok and gen and not usage.found:
                    TOTALS["usage_unread"] += 1
                TOTALS["output_tokens"] += out
                if TOTALS["output_tokens"] >= BUDGET and not TOTALS["budget_hit"]:
                    TOTALS["budget_hit"], TOTALS["budget_hit_at"] = True, time.time()
                TOTALS["over_budget"] = max(0, TOTALS["output_tokens"] - BUDGET)
                save()
                log({"t": round(t0, 3), "provider": provider, "path": path_only, "status": resp.status,
                     "model": usage.model, "output_tokens": out, "usage_read": usage.found,
                     "stream": sse, "seconds": round(time.time() - t0, 1), "total": TOTALS["output_tokens"]})
        gone = False                                 # the agent stopped reading: the reply is read to its end anyway
        try:
            while True:
                try:
                    chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                except (OSError, http.client.HTTPException):
                    break                            # the upstream broke off: what was read is what counts
                if not chunk:
                    break
                if sse:
                    usage.feed_sse(chunk)
                elif len(whole) < 64 << 20:
                    whole += chunk
                if not gone:
                    try:
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                        self.wfile.flush()
                    except OSError:
                        gone = True
            account()
            if not gone:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        except OSError:
            pass
        finally:
            account()

    do_POST = _proxy
    do_GET = _proxy
    do_PUT = _proxy
    do_DELETE = _proxy


if __name__ == "__main__":
    save()
    ThreadingHTTPServer((os.environ.get("METER_HOST", "0.0.0.0"), int(os.environ.get("METER_PORT", "8080"))), Handler).serve_forever()
