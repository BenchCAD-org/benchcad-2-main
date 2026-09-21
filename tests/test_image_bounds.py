"""harness.run bounds the images a request carries (the API's many-image rule).

Measured 2026-09-12 on claude-opus-5: a request with more than 20 image blocks
rejects any image with a dimension over 2000 px. Every drawing sheet is 4200
px, so without a bound a long run with crops fails from about round 7 on.
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.run import HARD_IMAGE_CAP, MANY_IMAGES, MANY_IMAGE_PX, _b64, bound_images  # noqa: E402


def _turns(n_seed: int, rounds: int, per_round: int):
    t = [{"role": "user", "text": "Begin.", "images": [f"seed{i}.png" for i in range(n_seed)]}]
    for r in range(rounds):
        t.append({"role": "assistant", "text": "```python\n```", "images": []})
        t.append({"role": "user", "text": f"Round {r}", "images": [f"r{r}_{k}.png" for k in range(per_round)]})
    return t


def test_every_image_stays_and_the_many_image_rule_is_live():
    """Nothing leaves the request: the seeds and every observation image of
    every round are there. Under 20 images they go at full size; from the
    request that passes 20 on, every image goes at MANY_IMAGE_PX (the API's
    rule) -- one cache rewrite, then a stable prefix."""
    turns = _turns(n_seed=6, rounds=4, per_round=3)                    # 6 + 12 = 18
    out, max_px = bound_images(turns)
    assert out == turns and max_px is None
    turns = _turns(n_seed=6, rounds=12, per_round=3)                   # 6 + 36 = 42
    out, max_px = bound_images(turns)
    assert out == turns and max_px == MANY_IMAGE_PX
    assert all(t["text"] for t in out if t["role"] == "user")


def test_only_the_api_ceiling_leaves_images_out_oldest_first():
    """Past HARD_IMAGE_CAP the request would be a 400: the oldest observation
    images are left out, the seeds never."""
    per = 7
    rounds = HARD_IMAGE_CAP // per + 2
    turns = _turns(n_seed=6, rounds=rounds, per_round=per)
    n = 6 + rounds * per
    assert n > HARD_IMAGE_CAP
    out, max_px = bound_images(turns)
    assert max_px == MANY_IMAGE_PX
    assert out[0]["images"] == turns[0]["images"]
    assert sum(len(t["images"]) for t in out) == HARD_IMAGE_CAP
    user_turns = [t for t in out if t["role"] == "user"][1:]
    assert user_turns[0]["images"] == []                               # the oldest round went first
    assert user_turns[-1]["images"] == turns[-1]["images"]             # the newest is whole


def test_the_request_byte_budget_leaves_the_oldest_observations_out(tmp_path, monkeypatch):
    """Both APIs refuse a request over 32 MB. When the images of one request
    would exceed REQUEST_IMAGE_BYTES as sent, the oldest observation images are
    left out first and the seeds never (Anthropic's review, 2026-09-21)."""
    from PIL import Image
    import harness.run as run
    big = tmp_path / "big.png"
    Image.effect_noise((900, 900), 64).convert("RGB").save(big)      # noise: incompressible, ~2 MB
    size = run._sent_bytes(big, None)
    assert size > 1_000_000
    seeds = [str(big)] * 2
    turns = [{"role": "user", "text": "Begin.", "images": seeds}]
    for r in range(6):
        turns.append({"role": "assistant", "text": "```python\n```", "images": []})
        turns.append({"role": "user", "text": f"Round {r}", "images": [str(big)] * 2})
    monkeypatch.setattr(run, "REQUEST_IMAGE_BYTES", size * 5)         # room for the seeds and a round and a half
    out, _ = bound_images(turns)
    assert out[0]["images"] == seeds
    kept = [t["images"] for t in out if t["role"] == "user"][1:]
    assert kept[-1] == [str(big)] * 2                                   # the newest round is whole
    assert kept[0] == []                                                # the oldest went first
    assert sum(len(k) for k in kept) + 2 <= 5


def test_b64_downscales_only_when_asked_and_only_when_larger(tmp_path):
    big, small = tmp_path / "big.png", tmp_path / "small.png"
    Image.new("RGB", (4200, 2970), "white").save(big)
    Image.new("RGB", (600, 400), "white").save(small)
    def size(b64):
        return Image.open(io.BytesIO(base64.b64decode(b64))).size
    assert size(_b64(big)) == (4200, 2970)
    assert size(_b64(big, MANY_IMAGE_PX)) == (2000, 1414)
    assert size(_b64(small, MANY_IMAGE_PX)) == (600, 400)
    assert _b64(small, MANY_IMAGE_PX) == _b64(small)                   # untouched bytes when nothing to do

