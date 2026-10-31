#!/usr/bin/env python3
"""Broca anlam-planı, yüzey güvenliği ve süreklilik için küçük regresyon pili.

Haricî corpus veya ağırlık dosyası gerekmez. Kalite/fluentlik iddiası yerine
sözleşmenin nedensel denetimleri yapılır: zorunlu/yasaklı terim, tekrar cezası,
görünür hata durumu ve optimizer/RNG ile tek-geçiş devamı.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from nora.language.broca_net import BrocaOrgan
from nora.language.voice import LearnedVoice, NF, SubwordTokenizer


def _organ():
    rows = [
        "Su berrak akar.", "Su temizdir.", "Bilgi zihni değiştirir.",
        "Bilgi paylaşılır.", "Insan su içer.", "Insan düşünür.",
    ] * 4
    tok = SubwordTokenizer()
    tok.train(rows, vocab_size=128)
    return BrocaOrgan(tok, seed=19, d=8, h=8), rows


def plan_contract() -> tuple[bool, str]:
    organ, _ = _organ()
    state = np.full(NF, 0.4, dtype=np.float32)
    texts = {
        1: "Su ve.",             # yarım yüzey
        2: "Su yasak.",          # plan tarafından yasaklandı
        3: "Bilgi var.",          # zorunlu çapa yok
        4: "Su berrak.",          # önceki yüzeyle aynı
        5: "Su akıyor.",
        6: "Su temiz.",
        7: "Su ve bir.",          # yarım yüzey
        8: "Su berrak.",          # aynı adayın ikinci örneği
    }
    organ.generate = lambda *args, **kwargs: [[i, i] for i in texts]
    organ.detok = lambda ids: texts.get(int(ids[0]), "")
    organ.logprob = lambda sequences, state, anchors: [-float(s[0]) for s in sequences]
    result = organ.articulate(
        state, ["su"], speech_plan={"required": ["su"], "forbidden": ["yasak"]},
        recent_surfaces=["Su berrak."], n=6,
    )
    text = str(result.get("text") or "")
    checks = {
        "grounded": result.get("grounded") is True,
        "required": "su" in text.casefold(),
        "forbidden": "yasak" not in text.casefold(),
        "not_exact_repeat": text != "Su berrak.",
        "not_half_surface": not text.casefold().endswith((" ve.", " bir.")),
        "telemetry": result.get("accepted_count", 0) >= 1
        and len(result.get("fluency_values", [])) >= 2,
    }
    # Zorunlu anlam hiçbir adayda yoksa akıcı görünen başka konuya geçilemez.
    organ.generate = lambda *args, **kwargs: [[3, 3]]
    missing = organ.articulate(state, ["su"], speech_plan={"required": ["su"]})
    checks["missing_returns_none"] = missing.get("text") is None and not missing.get("grounded")
    return all(checks.values()), f"{sum(checks.values())}/{len(checks)} {checks} | seçim={text!r}"


def voice_history_roundtrip() -> tuple[bool, str]:
    class FakeOrgan:
        def articulate(self, *args, **kwargs):
            return {"text": "Su berrak.", "sequence": [1, 2], "grounded": True,
                    "fluency_values": [-1.0, -1.2], "coverage": 1.0}

    voice = LearnedVoice(seed=8)
    voice.organ = FakeOrgan()
    state = np.zeros(NF, dtype=np.float32)
    first = voice.speak_organ(state, ["su"])
    voice.rng.random()  # RNG'nin gerçekten ilerlediği bir nokta
    with tempfile.TemporaryDirectory(prefix="nora-voice-contract-") as td:
        path = os.path.join(td, "voice.pkl.gz")
        voice.save_file(path)
        restored = LearnedVoice(seed=99)
        restored.load_file(path)
        same_history = list(restored.recent_organ_surfaces) == list(voice.recent_organ_surfaces)
        same_rng = restored.rng.bit_generator.state == voice.rng.bit_generator.state
        same_last = restored.last_articulation.get("text") == first
    return first == "Su berrak." and same_history and same_rng and same_last, (
        f"metin={first!r} history={same_history} rng={same_rng} telemetry={same_last}"
    )


def optimizer_and_generation_roundtrip() -> tuple[bool, str]:
    organ, rows = _organ()
    state = np.full(NF, 0.4, dtype=np.float32)
    organ.learn_batch(rows[:6], state, loss_mode="full")
    with tempfile.TemporaryDirectory(prefix="nora-broca-contract-") as td:
        path = os.path.join(td, "broca.pt")
        organ.save(path)
        restored = BrocaOrgan.load(path)
        weights = all(torch.allclose(a, b) for a, b in zip(
            organ.net.state_dict().values(), restored.net.state_dict().values()))
        rng = organ.rng.bit_generator.state == restored.rng.bit_generator.state
        torch_rng = torch.equal(organ.torch_rng.get_state(), restored.torch_rng.get_state())
        old_steps = [v["step"] for v in organ.opt.state_dict()["state"].values()]
        new_steps = [v["step"] for v in restored.opt.state_dict()["state"].values()]
        optimizer = len(old_steps) == len(new_steps) and all(
            torch.equal(a, b) for a, b in zip(old_steps, new_steps))
        plan = {"required": ["su"], "act": "ROUNDTRIP"}
        left = organ.articulate(state, ["su"], speech_plan=plan, n=4)
        right = restored.articulate(state, ["su"], speech_plan=plan, n=4)
        continuation = left.get("text") == right.get("text") and left.get("sequence") == right.get("sequence")
    checks = {"weights": weights, "rng": rng, "torch_rng": torch_rng,
              "optimizer": optimizer, "continuation": continuation}
    return all(checks.values()), f"{sum(checks.values())}/{len(checks)} {checks}"


def main():
    tests = [("plan/surface", plan_contract),
             ("voice persistence", voice_history_roundtrip),
             ("optimizer/RNG continuation", optimizer_and_generation_roundtrip)]
    passed = 0
    for name, fn in tests:
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        passed += int(ok)
        print(f"{name}: {'GEÇTİ' if ok else 'KALDI'} | {detail}")
    print(f"SONUÇ {passed}/{len(tests)}")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
