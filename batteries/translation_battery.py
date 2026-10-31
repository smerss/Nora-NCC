#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""translation_battery.py — gerçek converse yüzeyinde geçici dil dersi ve aktarım.

Hedef cümle/çeviri yalnız bu dış ölçümde beklenen değer olarak bulunur; üretim
organında hazır cümle veya hedefe özel kural olarak yer almaz.
"""
from __future__ import annotations

import sys, os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import nora  # noqa: F401  (eski nora_xxx import'ları için uyumluluk kancası)


import os
import tempfile
import time

from nora_agi import NoraAGI
from nora_style import strip_prefixes
from nora_translation import LearnedTranslator


TARGET = ("Harun sırayla kaydıraktan geçerken ayağı takılıp dondurmasını yere "
          "düşürdükten sonra nehire düştü.")
EXPECTED = ("While going down the slide in turn, Harun tripped, and after dropping "
            "his ice cream on the ground, Harun fell into the river.")
LESSONS = (
    "Geçmek İngilizcede 'go', geçmişi 'went', eylemlisi 'going' demektir.",
    "Takılmak İngilizcede 'trip', geçmişi 'tripped', eylemlisi 'tripping' demektir.",
    "Düşürmek İngilizcede 'drop', geçmişi 'dropped', eylemlisi 'dropping' demektir.",
    "Düşmek İngilizcede 'fall', geçmişi 'fell', eylemlisi 'falling' demektir.",
    "Sırayla İngilizcede 'in turn' demektir.",
    "'Ali kaydıraktan geçti.' cümlesinin İngilizcesi 'Ali went down the slide.'",
    "'Ali ayağı takıldı.' cümlesinin İngilizcesi 'Ali tripped.'",
    "'Ali dondurmasını yere düşürdü.' cümlesinin İngilizcesi 'Ali dropped his ice cream on the ground.'",
    "'Ali nehire düştü.' cümlesinin İngilizcesi 'Ali fell into the river.'",
)


def _reply(agi: NoraAGI, text: str) -> str:
    return strip_prefixes(agi.converse(text)).strip()


def _honest_refusal(text: str) -> bool:
    low = text.lower()
    return "çeviri" in low and any(mark in low for mark in (
        "bilmiyorum", "yol yok", "yapmış gibi yapmam", "uydur", "ölçüm yok",
    ))


def _trained(seed: int = 2031) -> NoraAGI:
    agi = NoraAGI(seed=seed, boot_thinker=False)
    for lesson in LESSONS:
        _reply(agi, lesson)
    return agi


def test_lexical_learning() -> tuple[bool, str]:
    agi = NoraAGI(seed=20261027, boot_thinker=False)
    _reply(agi, "Su İngilizcede 'water' demektir.")
    _reply(agi, "Nehir İngilizcede 'river' demektir.")
    answer = _reply(agi, "Nehir İngilizcede ne demek?")
    fresh = NoraAGI(seed=20261028, boot_thinker=False)
    absent = _reply(fresh, "Nehir İngilizcede ne demek?")
    ok = answer == "river" and "bilmiyorum" in absent and "nehir" in absent.lower()
    return ok, f"öğrenilen={answer!r} | yeni yaşam dürüstlüğü={absent!r}"


def test_empty_is_honest() -> tuple[bool, str]:
    agi = NoraAGI(seed=2030, boot_thinker=False)
    answer = _reply(agi, TARGET + " cümlesini İngilizceye çevir.")
    ok = _honest_refusal(answer)
    return ok, answer


def practical_trial() -> tuple[bool, str]:
    agi = _trained()
    started = time.perf_counter()
    answer = _reply(agi, TARGET + " cümlesini İngilizceye çevir.")
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    heldout = _reply(agi, "Selin nehire düştü. cümlesini İngilizceye çevir.")
    unknown = _reply(agi, "Selin okuldan geçerken nehire düştü. cümlesini İngilizceye çevir.")
    ok = (answer == EXPECTED and heldout == "Selin fell into the river."
          and _honest_refusal(unknown) and elapsed_ms < 2000.0)
    detail = (f"9 mikro-ders | {answer} | aktarım={heldout} | "
              f"bilinmeyen yapı dürüst={_honest_refusal(unknown)} | {elapsed_ms:.1f} ms")
    return ok, detail


def test_ablation_restore() -> tuple[bool, str]:
    translator = LearnedTranslator()
    for lesson in LESSONS:
        translator.teach(lesson)
    saved = translator.state_dict()
    baseline = translator.translate(TARGET)
    translator.constructions.pop("düş|nehire", None)
    ablated = translator.translate(TARGET)
    translator.load_state_dict(saved)
    restored = translator.translate("Mert nehire düştü.")
    fragment = translator.translate("Mert ayağı takılıp")
    ok = (baseline.get("status") == "ok" and ablated.get("status") == "unresolved"
          and restored.get("text") == "Mert fell into the river."
          and fragment.get("status") == "unresolved")
    return ok, (f"aktif={baseline.get('status')} | ablasyon={ablated.get('status')} | "
                f"geri yükleme={restored.get('text')!r} | yarım zincir={fragment.get('status')}")


def test_save_load() -> tuple[bool, str]:
    agi = _trained(2032)
    fd, path = tempfile.mkstemp(prefix="nora_translation_", suffix=".pkl")
    os.close(fd)
    try:
        agi.save(path)
        restored = NoraAGI.load(path)
        answer = _reply(restored, "Ece nehire düştü. cümlesini İngilizceye çevir.")
        ok = answer == "Ece fell into the river."
        return ok, f"kayıt/yükleme sonrası aktarım={answer!r}"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def main() -> None:
    tests = (
        ("T1 sözcük dersi ve yaşamlar arası sızıntı yok", test_lexical_learning),
        ("T2 öğretim yoksa dürüst ret", test_empty_is_honest),
        ("T3 uzun cümlede bileşim ve yeni özne", practical_trial),
        ("T4 yapı ablasyonu ve geri yükleme", test_ablation_restore),
        ("T5 öğrenilen çeviri durumunun kalıcılığı", test_save_load),
    )
    passed = 0
    started = time.perf_counter()
    for name, test in tests:
        ok, detail = test()
        passed += int(ok)
        print(f"{'GEÇTİ' if ok else 'KALDI'} {name}: {detail}", flush=True)
    elapsed = time.perf_counter() - started
    print(f"SONUÇ: {passed}/{len(tests)} ({elapsed:.2f} sn)", flush=True)
    raise SystemExit(0 if passed == len(tests) else 1)


if __name__ == "__main__":
    main()
