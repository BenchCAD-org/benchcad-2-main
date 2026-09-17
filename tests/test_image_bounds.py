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
from harness.run import KEEP_OBS_ROUNDS, MANY_IMAGES, MANY_IMAGE_PX, _b64, bound_images  # noqa: E402


def _turns(n_seed: int, rounds: int, per_round: int):
    t = [{"role": "user", "text": "Begin.", "images": [f"seed{i}.png" for i in range(n_seed)]}]
    for r in range(rounds):
        t.append({"role": "assistant", "text": "```python\n```", "images": []})
        t.append({"role": "user", "text": f"Round {r}", "images": [f"r{r}_{k}.png" for k in range(per_round)]})
    return t


def test_seed_images_always_stay_and_old_observations_leave():
    turns = _turns(n_seed=6, rounds=12, per_round=3)
    out, max_px = bound_images(turns)
    assert out[0]["images"] == turns[0]["images"]                      # the case's inputs, every round
    user_turns = [t for t in out if t["role"] == "user"][1:]
    kept = [t for t in user_turns if t["images"]]
    assert len(kept) == KEEP_OBS_ROUNDS and kept == user_turns[-KEEP_OBS_ROUNDS:]
    assert all(t["text"] for t in user_turns)                          # text is never dropped
    assert sum(len(t["images"]) for t in out) == 6 + KEEP_OBS_ROUNDS * 3 <= MANY_IMAGES
    assert max_px is None


def test_over_the_threshold_every_image_is_downscaled():
    turns = _turns(n_seed=17, rounds=3, per_round=3)                   # 17 + 9 = 26 > 20
    out, max_px = bound_images(turns)
    assert sum(len(t["images"]) for t in out) > MANY_IMAGES
    assert max_px == MANY_IMAGE_PX


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


def test_the_size_decision_is_made_from_the_seeds_not_the_live_count():
    """A T2 case seeds 13 images. Deciding per request, round one went out
    at full size and round five (13 seeds + a few crops > 20) at 2000 px:
    the seed bytes changed, the request prefix changed from turn one, and
    the prompt cache was rewritten instead of read. The episode is sized
    once: 13 + KEEP_OBS_ROUNDS x OBS_PER_ROUND > 20, so 2000 px from round
    one, crops or not; 12 seeds stay at full size."""
    from harness.run import OBS_PER_ROUND
    edge = MANY_IMAGES - KEEP_OBS_ROUNDS * OBS_PER_ROUND        # the last seed count at full size
    out, max_px = bound_images(_turns(n_seed=edge + 1, rounds=0, per_round=0))
    assert sum(len(t["images"]) for t in out) == edge + 1 and max_px == MANY_IMAGE_PX
    out, max_px = bound_images(_turns(n_seed=edge, rounds=0, per_round=0))
    assert max_px is None
    # The live count still rules above the threshold: the API would reject
    # the request otherwise.
    out, max_px = bound_images(_turns(n_seed=edge, rounds=KEEP_OBS_ROUNDS, per_round=OBS_PER_ROUND + 1))
    assert max_px == MANY_IMAGE_PX

