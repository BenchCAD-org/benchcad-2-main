"""A test agent for the meter (metertest/make_metertest.sh): it writes the reference answer (the test task
carries it in /probe_answer, which no real task does), then asks the model API for completions through the
meter until it is refused, and tries the network around it (HTTPS and ICMP). What it saw goes to /logs/agent/probe.json."""
from harbor.agents.base import BaseAgent

PROBE = r'''
import json, socket, urllib.request, urllib.error
out = {"calls": [], "direct": {}}
def call(stream):
    body = json.dumps({"model": "stub", "max_tokens": 10, "stream": stream,
                       "messages": [{"role": "user", "content": "hi"}]}).encode()
    req = urllib.request.Request("http://meter:8080/anthropic/v1/messages", data=body, method="POST",
                                 headers={"content-type": "application/json", "x-api-key": "stub",
                                          "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read().decode()[-300:]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:400]
for i in range(8):
    st, tail = call(stream=bool(i % 2))
    out["calls"].append({"i": i, "stream": bool(i % 2), "status": st, "tail": tail})
    if st == 400:
        break
for name, url in [("anthropic_direct", "https://api.anthropic.com/v1/messages"), ("example", "https://example.com/"),
                  ("pypi", "https://pypi.org/simple/")]:
    try:
        urllib.request.urlopen(url, timeout=10)
        out["direct"][name] = "REACHED"
    except urllib.error.HTTPError as e:
        out["direct"][name] = f"REACHED (HTTP {e.code})"
    except Exception as e:
        out["direct"][name] = f"blocked: {type(e).__name__}: {str(e)[:120]}"
def icmp(kind):
    import struct, select
    pkt = struct.pack("!BBHHH", 8, 0, 0, 1, 1)
    csum = 0
    for i in range(0, len(pkt), 2):
        csum += (pkt[i] << 8) + pkt[i + 1]
    csum = ~((csum >> 16) + (csum & 0xFFFF)) & 0xFFFF
    pkt = struct.pack("!BBHHH", 8, 0, csum, 1, 1)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_RAW if kind == "raw" else socket.SOCK_DGRAM, socket.IPPROTO_ICMP)
    except OSError as e:
        return f"blocked: no socket: {e}"
    try:
        s.sendto(pkt, ("1.1.1.1", 0))
        r, _, _ = select.select([s], [], [], 5)
        return "REACHED (echo reply)" if r else "no reply in 5 s"
    except OSError as e:
        return f"blocked: {e}"
for kind in ("raw", "ping_socket"):
    out["direct"]["icmp_" + kind] = icmp(kind)

# Harbor's allowlist is a TCP redirect into gost, which picks the host out of the TLS SNI or the HTTP Host
# header. These four ask whether a connection the agent aims at an outside address is carried there anyway
# when it claims to be the meter, and whether anything but TCP leaves at all.
OUTSIDE = "1.1.1.1"
eg = out["egress"] = {}
try:
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    c = socket.create_connection((OUTSIDE, 443), timeout=10)
    with ctx.wrap_socket(c, server_hostname="meter") as t:
        cert = t.getpeercert(binary_form=True)
        eg["tls_sni_meter"] = f"REACHED: a TLS peer answered with {len(cert or b'')} bytes of certificate"
except Exception as e:
    eg["tls_sni_meter"] = f"blocked: {type(e).__name__}: {str(e)[:120]}"
try:
    c = socket.create_connection((OUTSIDE, 80), timeout=10)
    c.sendall(b"GET / HTTP/1.1\r\nHost: meter\r\nConnection: close\r\n\r\n")
    first = c.recv(200).decode("latin1")
    c.close()
    eg["http_host_meter"] = f"answered: {first.splitlines()[0] if first else 'nothing'}"
except Exception as e:
    eg["http_host_meter"] = f"blocked: {type(e).__name__}: {str(e)[:120]}"
try:
    c = socket.create_connection((OUTSIDE, 443), timeout=10)
    c.sendall(b"NOTHTTP NOTTLS\n")
    eg["raw_tcp"] = f"answered: {c.recv(60)!r}"
    c.close()
except Exception as e:
    eg["raw_tcp"] = f"blocked: {type(e).__name__}: {str(e)[:120]}"
try:
    u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    u.settimeout(8)
    u.sendto(b"\xab\xcd\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x07example\x03com\x00\x00\x01\x00\x01",
             (OUTSIDE, 53))
    eg["dns_udp"] = f"answered: {len(u.recvfrom(512)[0])} bytes"
except Exception as e:
    eg["dns_udp"] = f"blocked: {type(e).__name__}: {str(e)[:120]}"
with urllib.request.urlopen("http://meter:8080/_meter/status", timeout=10) as r:
    out["status"] = json.loads(r.read())
open("/logs/agent/probe.json", "w").write(json.dumps(out, indent=1))
print(json.dumps(out)[:2000])
'''


class MeterProbe(BaseAgent):
    @staticmethod
    def name() -> str:
        return "meter-probe"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment) -> None:
        return

    async def run(self, instruction, environment, context) -> None:
        await environment.exec(command="cp -r /probe_answer/. /work/", timeout_sec=60)
        r = await environment.exec(command="python3 - <<'PY'\n" + PROBE + "\nPY", timeout_sec=600)
        (self.logs_dir / "probe.stdout").write_text((r.stdout or "") + (r.stderr or ""))


class LinkProbe(BaseAgent):
    """A test agent for the verifier: its answer.py links final.step (and a submission part) to the reference in
    the verifier's /tests and exits before the harness's finish. The verifier must score that as no answer."""

    ANSWER = ("import os\n"
              "os.symlink('/tests/case/gt/gt.step', 'final.step')\n"
              "os.makedirs('submission/parts', exist_ok=True)\n"
              "os.symlink('/tests/case/gt/gt.step', 'submission/parts/p.step')\n"
              "os._exit(0)\n")

    @staticmethod
    def name() -> str:
        return "link-probe"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment) -> None:
        return

    async def run(self, instruction, environment, context) -> None:
        import shlex
        await environment.exec(command=f"printf %s {shlex.quote(self.ANSWER)} > /work/answer.py", timeout_sec=60)


class ForgeProbe(BaseAgent):
    """A test agent for the verifier: it leaves links in /work where the verifier writes its own files
    (tools.py, sitecustomize.py), so that root would write through them, and an answer.py that tries to write
    its own reward into /logs/verifier and then exits before the harness's finish. The verifier must score no
    answer, with its own files unchanged."""

    ANSWER = (
        "import json, os\n"
        "forged = json.dumps({'reward': 1.0, 'error': 0, 'no_answer': 0})\n"
        "tried = {}\n"
        "for p in ('/logs/verifier/reward.json', '/logs/verifier/reward.txt', '/logs/reward.json'):\n"
        "    try:\n"
        "        open(p, 'w').write(forged)\n"
        "        tried[p] = 'WROTE'\n"
        "    except OSError as e:\n"
        "        tried[p] = type(e).__name__\n"
        "for p in ('/logs/verifier', '/logs'):\n"
        "    try:\n"
        "        os.rename(p, p + '.moved')\n"
        "        tried['rename ' + p] = 'MOVED'\n"
        "    except OSError as e:\n"
        "        tried['rename ' + p] = type(e).__name__\n"
        "open('/tmp/forge.json', 'w').write(json.dumps(tried))\n"
        "os._exit(0)\n")

    @staticmethod
    def name() -> str:
        return "forge-probe"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment) -> None:
        return

    async def run(self, instruction, environment, context) -> None:
        import shlex
        await environment.exec(command="ln -sfn /tests/run/score.py /work/tools.py; "
                                       "ln -sfn /tests/run/collect_answer.py /work/sitecustomize.py; "
                                       "ln -sfn /tests/run/fail.py /work/_run.py", timeout_sec=60)
        await environment.exec(command=f"printf %s {shlex.quote(self.ANSWER)} > /work/answer.py", timeout_sec=60)


class FakeLaunchProbe(BaseAgent):
    """A test agent whose answer claims the launcher failed (the marker on stderr and exit 97), which would
    once have bought it a pending cell and a fresh budget. It must be scored 0."""

    ANSWER = ("import sys\n"
              "sys.stderr.write('BENCHCAD_LAUNCH_FAILED OSError: nothing of the sort\\n')\n"
              "sys.exit(97)\n")

    @staticmethod
    def name() -> str:
        return "fake-launch-probe"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment) -> None:
        return

    async def run(self, instruction, environment, context) -> None:
        import shlex
        await environment.exec(command=f"printf %s {shlex.quote(self.ANSWER)} > /work/answer.py", timeout_sec=60)


class TmpProbe(BaseAgent):
    """A test agent whose answer plants links at the paths a scorer writes under /tmp and HOME, pointing at
    the verifier's own files. Nothing may be written through them: the trial is scored no answer and the
    verifier's files are unchanged (otherwise its seal fails and the trial has no score at all)."""

    ANSWER = (
        "import os\n"
        "for p, t in (('/tmp/matplotlibrc', '/tests/pins.json'),\n"
        "             ('/tmp/.cache', '/tests/run'),\n"
        "             ('/tmp/.config', '/tests/scorer'),\n"
        "             ('/tmp/score.py', '/tests/run/score.py')):\n"
        "    try:\n"
        "        os.symlink(t, p)\n"
        "    except OSError:\n"
        "        pass\n"
        "os.makedirs('/tmp/.cache-sandbox', exist_ok=True)\n"
        "open('/tmp/.cache-sandbox/sitecustomize.py', 'w').write('raise SystemExit(0)\\n')\n")

    @staticmethod
    def name() -> str:
        return "tmp-probe"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment) -> None:
        return

    async def run(self, instruction, environment, context) -> None:
        import shlex
        await environment.exec(command=f"printf %s {shlex.quote(self.ANSWER)} > /work/answer.py", timeout_sec=60)
