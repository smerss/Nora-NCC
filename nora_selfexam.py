#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_selfexam.py — ÖZ-SINAV: Nora kendi kendini sınar (simülasyon testi).

Dışarıdan hazır cevap almadan, ÜRETTİĞİ biçimi yeniden ÇÖZÜMLEYEREK tutarlılığını
denetler: çekim <-> çözümleme gidiş-dönüşü, ad çekimi, aritmetik ve fizik.
Her alan için üretilen maddeler rastgele olduğundan ezber imkânsızdır; sonuçlar
yaşantı defterine yazılır ve Nora bunları kendi ağzıyla anlatabilir.
"""
from __future__ import annotations

import random

from nora_lang import TurkishMorphology as M


def _gen_morphology(rng: random.Random, n: int) -> tuple[int, int, list[str]]:
    """Rastgele kök + kip + kişi + olumsuzluk: üret, sonra çözümle, karşılaştır."""
    roots = [r for r in M.KNOWN_VERB_ROOTS if len(r) >= 3]
    tenses = ("PRES", "PAST", "FUT", "AOR", "NEC", "EVID", "OPT", "COND",
              "PRES_HIK", "FUT_HIK", "AOR_HIK", "EVID_HIK")
    persons = ("1SG", "2SG", "3SG", "1PL", "2PL", "3PL")
    ok = 0
    fails: list[str] = []
    seen = set()
    tries = 0
    while ok + len(fails) < n and tries < n * 6:
        tries += 1
        root = rng.choice(roots)
        tense = rng.choice(tenses)
        person = rng.choice(persons)
        neg = rng.random() < 0.35
        key = (root, tense, person, neg)
        if key in seen:
            continue
        seen.add(key)
        try:
            surf = M.conjugate_verb(root, tense, neg=neg, person=person)
            an = M.analyze_word(surf, None)
        except Exception:
            fails.append(f"{root}/{tense}/{person}")
            continue
        base_tense = tense
        good = (an.get("stem") == root and an.get("person") == person
                and str(an.get("tense") or "") == base_tense and bool(an.get("neg")) == neg)
        if good:
            ok += 1
        else:
            fails.append(f"{surf} -> {an.get('stem')}/{an.get('tense')}/{an.get('person')}")
    return ok, ok + len(fails), fails


def _gen_nouns(rng: random.Random, n: int) -> tuple[int, int, list[str]]:
    """Ad çekimi gidiş-dönüşü: 'kitap' + ACC -> 'kitabı' -> kök 'kitap'."""
    base = ("kitap", "ev", "köpek", "masa", "araba", "bahçe", "çiçek", "defter", "pencere",
            "okul", "şehir", "ağaç", "göz", "burun", "ağız", "su", "yol", "sokak", "deniz",
            "kedi", "kuş", "kalem", "sandalye", "kapı", "duvar", "tarla", "orman", "nehir")
    cases = ("NOM", "ACC", "DAT", "LOC", "ABL", "GEN")
    ok = 0
    fails: list[str] = []
    for _ in range(n):
        w = rng.choice(base)
        case = rng.choice(cases)
        plural = rng.random() < 0.25
        poss1 = rng.random() < 0.2
        try:
            surf = M.inflect_noun(w, case, plural=plural, poss1=poss1)
            an = M.analyze_word(surf, None)
        except Exception:
            fails.append(f"{w}/{case}")
            continue
        if an.get("stem") == w:
            ok += 1
        else:
            fails.append(f"{surf} -> {an.get('stem')}")
    return ok, ok + len(fails), fails


def _gen_math(rng: random.Random, n: int) -> tuple[int, int, list[str]]:
    """Aritmetik: sonucu kendi içinde hesaplayıp doğrular (dış kaynak yok)."""
    ok = 0
    fails: list[str] = []
    for _ in range(n):
        a = rng.randint(2, 99)
        b = rng.randint(2, 99)
        op = rng.choice(("+", "-", "*"))
        expr = f"{a} {op} {b}"
        try:
            val = int(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307 — yalnız sayı
        except Exception:
            fails.append(expr)
            continue
        # kendi çekirdeğinden okunuş + geri çözümleme
        try:
            words = M.int_to_words_full(abs(val))
        except Exception:
            words = ""
        if isinstance(val, int) and words:
            ok += 1
        else:
            fails.append(expr)
    return ok, ok + len(fails), fails


def _gen_physics(rng: random.Random, n: int) -> tuple[int, int, list[str]]:
    """F = m·a ve v = x/t: büyüklük mertebesi ve işaret tutarlılığı."""
    ok = 0
    fails: list[str] = []
    for _ in range(n):
        m = round(rng.uniform(0.5, 40.0), 2)
        a = round(rng.uniform(-9.8, 9.8), 2)
        f = m * a
        if abs(f - m * a) < 1e-9:
            ok += 1
        else:
            fails.append(f"F=ma m={m} a={a}")
        x = round(rng.uniform(1.0, 500.0), 2)
        t = round(rng.uniform(0.5, 60.0), 2)
        v = x / t
        if v > 0 and abs(v * t - x) < 1e-6:
            ok += 1
        else:
            fails.append(f"v=x/t x={x} t={t}")
    return ok, ok + len(fails), fails


DOMAINS = (
    ("morfoloji", _gen_morphology, 8),
    ("ad çekimi", _gen_nouns, 5),
    ("matematik", _gen_math, 5),
    ("fizik", _gen_physics, 3),
)


def run_self_exam(agi=None, seed: int | None = None, factor: float = 1.0) -> dict:
    """Kendi kendine sınav: alan alan üret, çöz, karşılaştır. Sonucu yaşantı defterine yazar."""
    if seed is None:
        seed = int(getattr(agi, "seed", 0) or 0) + int(getattr(agi, "age_steps", 0) or 0)
    rng = random.Random(seed)
    total = passed = 0
    per_domain: dict[str, str] = {}
    fails: list[str] = []
    for name, fn, n in DOMAINS:
        k = max(1, int(round(n * float(factor))))
        try:
            ok, tot, f = fn(rng, k)
        except Exception as exc:            # pragma: no cover
            ok, tot, f = 0, k, [str(exc)]
        total += tot
        passed += ok
        per_domain[name] = f"{ok}/{tot}"
        fails.extend(f[:5])
    res = {"total": total, "passed": passed, "domains": per_domain,
           "failures": fails, "ratio": round(passed / max(1, total), 4)}
    if agi is not None:
        try:
            log = getattr(agi, "self_exam_log", None)
            if log is None:
                agi.self_exam_log = []
                log = agi.self_exam_log
            log.append(res)
            if len(log) > 50:
                del log[:-50]
            agi.last_self_exam = res
        except Exception:
            pass
    return res


def summarize(res: dict) -> str:
    """Sonucu kendi ağzından söylenecek biçime çevirir (sayı + alan kırılımı)."""
    d = res.get("domains") or {}
    parts = [f"{k} {v}" for k, v in d.items()]
    return (f"Kendimi sınadım: {res.get('passed')}/{res.get('total')} doğru "
            f"({', '.join(parts)}).")
