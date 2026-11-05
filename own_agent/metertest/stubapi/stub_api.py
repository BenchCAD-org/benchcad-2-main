"""A stub model API behind the meter, for the meter tests: no model is called and nothing is spent.

Every generation reply reports STUB_OUTPUT_TOKENS output tokens, in each provider's own usage field. The first
generation reply is a tool call when the request offers a shell tool, so that a real agent CLI asks again (and,
with a budget below one reply, is refused by the meter); later replies are plain text. Every request is logged
to $STUB_DIR/requests.jsonl (method, path, the headers that decide the transport, the request's tool names and
stream options; no bodies).

    /anthropic  POST .../v1/messages (stream or not), .../v1/messages/count_tokens
    /openai     POST .../v1/responses (stream or not), .../v1/chat/completions (stream or not), GET .../models
    /gemini     POST .../models/<m>:generateContent, :streamGenerateContent, :countTokens
(the meter strips its route prefix; the stub serves all three on one port and tells them apart by path)
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

N = int(os.environ.get("STUB_OUTPUT_TOKENS", "1000"))
LOG = Path(os.environ.get("STUB_DIR", "/stub"))
LOG.mkdir(parents=True, exist_ok=True)
LOCK = threading.Lock()
SEEN = {"gen": 0}


def log(row):
    with LOCK, open(LOG / "requests.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def tool_names(req):
    out = []
    for t in req.get("tools") or []:
        if isinstance(t, dict):
            out.append(t.get("name") or (t.get("function") or {}).get("name") or t.get("type"))
            for fd in t.get("functionDeclarations") or t.get("function_declarations") or []:
                out.append(fd.get("name"))
    return [x for x in out if x]


def shell_tool(names, prefer):
    for p in prefer:
        if p in names:
            return p
    for n in names:
        if any(k in n.lower() for k in ("shell", "bash", "exec", "command")):
            return n
    return None


def sse(events):
    return "".join((f"event: {e}\n" if e else "") + f"data: {json.dumps(d)}\n\n" for e, d in events).encode()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        log({"t": time.time(), "method": "GET", "path": self.path, "upgrade": self.headers.get("Upgrade")})
        if self.path.rstrip("/").endswith("/models"):
            return self.send(200, json.dumps({"object": "list", "data": [{"id": "stub-model", "object": "model"}],
                                              "models": [{"name": "models/stub-model"}]}).encode())
        self.send(200, b'{"ok": true}')

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}"
        try:
            req = json.loads(raw)
        except ValueError:
            req = {}
        path = self.path.split("?")[0]
        names = tool_names(req) if isinstance(req, dict) else []
        gen = path.endswith(("/messages", "/responses", "/chat/completions", ":generateContent", ":streamGenerateContent"))
        with LOCK:
            k = SEEN["gen"]
            SEEN["gen"] += gen
        log({"t": time.time(), "method": "POST", "path": self.path, "gen_index": k if gen else None,
             "stream": req.get("stream") if isinstance(req, dict) else None,
             "stream_options": req.get("stream_options") if isinstance(req, dict) else None,
             "model": req.get("model") if isinstance(req, dict) else None, "tools": names[:40],
             "accept": self.headers.get("Accept"), "upgrade": self.headers.get("Upgrade")})
        first = gen and k == 0
        if path.endswith("/count_tokens"):
            return self.send(200, b'{"input_tokens": 10}')
        if path.endswith(":countTokens"):
            return self.send(200, b'{"totalTokens": 10}')
        if path.endswith("/messages"):
            return self.anthropic(req, first, names)
        if path.endswith("/responses"):
            return self.oai_responses(req, first, names)
        if path.endswith("/chat/completions"):
            return self.chat(req, first, names)
        if path.endswith(("generateContent", "streamGenerateContent")):
            return self.gemini(path.endswith("streamGenerateContent"), first, names)
        self.send(404, json.dumps({"error": f"stub: no route {path}"}).encode())

    # -- Anthropic Messages
    def anthropic(self, req, first, names):
        tool = shell_tool(names, ["Bash"]) if first else None
        block = ({"type": "tool_use", "id": "toolu_stub1", "name": tool, "input": {"command": "echo stub", "description": "stub"}}
                 if tool else {"type": "text", "text": "Done."})
        stop = "tool_use" if tool else "end_turn"
        msg = {"id": "msg_stub", "type": "message", "role": "assistant", "model": req.get("model", "stub"),
               "content": [block], "stop_reason": stop, "stop_sequence": None,
               "usage": {"input_tokens": 10, "output_tokens": N}}
        if not req.get("stream"):
            return self.send(200, json.dumps(msg).encode())
        start = dict(msg, content=[], stop_reason=None, usage={"input_tokens": 10, "output_tokens": 1})
        ev = [("message_start", {"type": "message_start", "message": start})]
        if tool:
            ev += [("content_block_start", {"type": "content_block_start", "index": 0, "content_block": dict(block, input={})}),
                   ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                            "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}})]
        else:
            ev += [("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
                   ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Done."}})]
        ev += [("content_block_stop", {"type": "content_block_stop", "index": 0}),
               ("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                                  "usage": {"output_tokens": N}}),
               ("message_stop", {"type": "message_stop"})]
        self.send(200, sse(ev), "text/event-stream")

    # -- OpenAI Responses
    def oai_responses(self, req, first, names):
        tool = shell_tool(names, ["shell", "exec_command", "local_shell"]) if first else None
        if tool == "exec_command":
            item = {"type": "function_call", "id": "fc_stub", "call_id": "call_stub", "name": tool,
                    "arguments": json.dumps({"cmd": "echo stub"}), "status": "completed"}
        elif tool:
            item = {"type": "function_call", "id": "fc_stub", "call_id": "call_stub", "name": tool,
                    "arguments": json.dumps({"command": ["bash", "-lc", "echo stub"]}), "status": "completed"}
        else:
            item = {"type": "message", "id": "msg_stub", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "Done.", "annotations": []}]}
        usage = {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": N,
                 "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 10 + N}
        resp = {"id": "resp_stub", "object": "response", "created_at": int(time.time()), "status": "completed",
                "model": req.get("model", "stub"), "output": [item], "usage": usage}
        if not req.get("stream"):
            return self.send(200, json.dumps(resp).encode())
        base = dict(resp, status="in_progress", output=[], usage=None)
        ev = [("response.created", {"type": "response.created", "response": base, "sequence_number": 0}),
              ("response.output_item.added", {"type": "response.output_item.added", "output_index": 0,
                                              "item": dict(item, status="in_progress"), "sequence_number": 1})]
        if item["type"] == "message":
            ev += [("response.output_text.delta", {"type": "response.output_text.delta", "item_id": "msg_stub",
                                                   "output_index": 0, "content_index": 0, "delta": "Done.", "sequence_number": 2})]
        ev += [("response.output_item.done", {"type": "response.output_item.done", "output_index": 0, "item": item,
                                              "sequence_number": 3}),
               ("response.completed", {"type": "response.completed", "response": resp, "sequence_number": 4})]
        self.send(200, sse(ev), "text/event-stream")

    # -- OpenAI Chat Completions (litellm: mini-swe-agent, OpenHands)
    def chat(self, req, first, names):
        tool = shell_tool(names, ["execute_bash", "bash"]) if first else None
        if tool:
            msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_stub", "type": "function", "function": {"name": tool, "arguments": json.dumps({"command": "echo stub"})}}]}
            fin = "tool_calls"
        else:
            text = "THOUGHT: run it.\n\n```bash\necho stub\n```" if first else "Done."
            msg, fin = {"role": "assistant", "content": text}, "stop"
        usage = {"prompt_tokens": 10, "completion_tokens": N, "total_tokens": 10 + N}
        base = {"id": "chatcmpl-stub", "created": int(time.time()), "model": req.get("model", "stub")}
        if not req.get("stream"):
            return self.send(200, json.dumps(dict(base, object="chat.completion", usage=usage,
                                                  choices=[{"index": 0, "message": msg, "finish_reason": fin}])).encode())
        delta = dict(msg)
        if delta.get("tool_calls"):
            delta["tool_calls"] = [dict(delta["tool_calls"][0], index=0)]
        chunks = [dict(base, object="chat.completion.chunk", choices=[{"index": 0, "delta": delta, "finish_reason": None}]),
                  dict(base, object="chat.completion.chunk", choices=[{"index": 0, "delta": {}, "finish_reason": fin}])]
        if (req.get("stream_options") or {}).get("include_usage"):
            chunks.append(dict(base, object="chat.completion.chunk", choices=[], usage=usage))
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        self.send(200, body.encode(), "text/event-stream")

    # -- Gemini
    def gemini(self, stream, first, names):
        tool = shell_tool(names, ["run_shell_command"]) if first else None
        part = {"functionCall": {"name": tool, "args": {"command": "echo stub"}}} if tool else {"text": "Done."}
        chunk = {"candidates": [{"content": {"role": "model", "parts": [part]}, "finishReason": "STOP", "index": 0}],
                 "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": N - 100, "thoughtsTokenCount": 100,
                                   "totalTokenCount": 10 + N}, "modelVersion": "stub-model"}
        if not stream:
            return self.send(200, json.dumps(chunk).encode())
        if "alt=sse" in self.path:
            return self.send(200, f"data: {json.dumps(chunk)}\n\n".encode(), "text/event-stream")
        self.send(200, json.dumps([chunk]).encode())


ThreadingHTTPServer((os.environ.get("STUB_HOST", "0.0.0.0"), int(os.environ.get("STUB_PORT", "8000"))), H).serve_forever()
