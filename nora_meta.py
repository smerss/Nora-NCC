#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_meta.py — ÜST-BİLİŞ: çakışma/tekrar izleme ve zaman bilinci.

Beyindeki anterior singulat (ACC) çakışma izleme ile episodik belleğin birleşimi:
Nora, aynı sözün kısa süre içinde tekrarlandığını FARK EDER, ne zaman söylendiğini
hatırlar, o an ne cevap verdiğini anımsar ve bunu kendi sözüne katar.

Burada hazır cümle yoktur; yalnızca
  * imza çıkarımı (kök + kip + kişi; sözcük sırasından bağımsız),
  * zaman farkı ve tur sayısı,
  * kendi önceki cevabından alıntı
verisinden cümle KURULUR. Metin, durumun kendisinden doğar.
"""
from __future__ import annotations

import time

from nora_lang import TurkishMorphology, tr_lower

_STOP = {
    "bir", "bu", "şu", "o", "ve", "de", "da", "ki", "mi", "mı", "mu", "mü", "ya", "ise",
    "ben", "sen", "biz", "siz", "onlar", "çok", "az", "daha", "en", "ama", "fakat", "ki",
    "için", "ile", "gibi", "kadar", "sonra", "önce", "şimdi", "bugün", "dün", "yarın",
}


class ConflictMonitor:
    """Kısa süreli söz belleği + tekrar/çelişki sezici (ACC benzeri)."""

    def __init__(self, window: int = 24, max_gap_s: float = 1800.0):
        self.window = int(window)
        self.max_gap_s = float(max_gap_s)
        self.history: list[dict] = []       # {t, text, sig, reply, kind}
        self.session_t0 = time.time()
        self.repeats_noticed = 0
        self.last_notice: dict | None = None

    # ------------------------------------------------------------------ imza
    @staticmethod
    def signature(text: str) -> frozenset:
        """Sözcük sırasından bağımsız içerik imzası: kök + (kip, kişi, olumsuzluk)."""
        words = [w for w in tr_lower(text).replace("?", " ").replace("!", " ").replace(",", " ").split() if w]
        sig = set()
        for w in words:
            if w in _STOP:
                continue
            try:
                a = TurkishMorphology.analyze_word(w, None)
            except Exception:
                a = {}
            stem = str(a.get("stem") or w)
            if a.get("is_verb"):
                sig.add((stem, str(a.get("tense") or ""), str(a.get("person") or ""), bool(a.get("neg"))))
            else:
                sig.add((stem, "", "", False))
        return frozenset(sig)

    # ------------------------------------------------------------- gözlemle
    def observe(self, text: str, now: float | None = None) -> dict | None:
        """Bu söz yakın geçmişte (anlamca) söylendi mi? Söylendiyse gerekçeli kayıt döner."""
        now = time.time() if now is None else float(now)
        sig = self.signature(text)
        if not sig:
            return None
        best = None
        for rec in reversed(self.history[-self.window:]):
            gap = now - rec["t"]
            if gap < 0 or gap > self.max_gap_s:
                continue
            inter = len(sig & rec["sig"])
            if not inter:
                continue
            union = len(sig | rec["sig"])
            jac = inter / max(1, union)
            if jac < 0.80:
                continue
            if best is None or jac > best["jac"]:
                best = {"jac": jac, "rec": rec, "gap": gap,
                        "turns_ago": len([r for r in self.history if r["t"] > rec["t"]])}
        if best is None:
            return None
        rec = best["rec"]
        exact = best["jac"] >= 0.999
        return {
            "kind": "REPEAT",
            "exact": bool(exact),
            "similarity": round(float(best["jac"]), 3),
            "prev_text": rec["text"],
            "prev_reply": rec.get("reply") or "",
            "elapsed_s": float(best["gap"]),
            "turns_ago": int(best["turns_ago"]) + 1,
            "count": 1 + sum(1 for r in self.history[-self.window:]
                             if r["sig"] == rec["sig"]),
        }

    # ---------------------------------------------------------------- kaydet
    def note_turn(self, text: str, reply: str = "", now: float | None = None) -> None:
        now = time.time() if now is None else float(now)
        self.history.append({"t": now, "text": text, "sig": self.signature(text),
                             "reply": reply or ""})
        if len(self.history) > 400:
            del self.history[:100]

    # ---------------------------------------------------------------- zaman
    def since_session(self, now: float | None = None) -> float:
        now = time.time() if now is None else float(now)
        return max(0.0, now - self.session_t0)

    @staticmethod
    def time_phrase(seconds: float) -> str:
        """Geçen süreyi insan gibi söyler: 'az önce', '3 dakika önce', 'bir saat kadar önce'."""
        s = float(max(0.0, seconds))
        if s < 45:
            return "az önce"
        if s < 90:
            return "bir dakika kadar önce"
        if s < 3600:
            mnt = int(round(s / 60.0))
            return f"{mnt} dakika önce"
        if s < 7200:
            return "bir saat kadar önce"
        if s < 86400:
            return f"{int(s // 3600)} saat önce"
        return "geçen gün"


# ---------------------------------------------------------------------------
# Söz üretimi: veriden cümle kurma (kalıp cümle değil, parçalardan kurulum)
# ---------------------------------------------------------------------------

def _cap(s: str) -> str:
    return (s[:1].upper() + s[1:]) if s else s


def _quote(reply: str, limit: int = 90) -> str:
    r = (reply or "").strip().rstrip(".")
    if not r:
        return ""
    if len(r) > limit:
        r = r[:limit].rsplit(" ", 1)[0] + "…"
    return r


def compose_noticing(meta: dict, agi=None, rng=None) -> str:
    """Tekrarı fark eden sözü KURAR: geçen zaman + kendi önceki cevabı + kendi sorusu.

    Parçalar sabit değil; geçen süre, önceki söz, önceki cevabın içindeki soru ve
    kaçıncı tekrar olduğu o anki bellekten gelir. Kısa tutulur: insan da böyle yapar.
    """
    import random
    rng = rng or random
    gap = float(meta.get("elapsed_s", 0.0))
    tp = ConflictMonitor.time_phrase(gap)
    same = bool(meta.get("exact"))
    count = int(meta.get("count", 1))
    turns = int(meta.get("turns_ago") or 1)
    prev = _quote(meta.get("prev_reply", ""), limit=60)

    # 1) Farkındalık cümlesi
    if gap < 45:
        leads = ("Bunu az önce de söylemiştin", "Aynı şeyi demin sordun",
                 "Bu sözü biraz önce duymuştum") if same else (
                 "Bunu az önce de söyler gibi oldun", "Demin de benzer bir şey sormuştun")
    else:
        leads = (f"Bunu {tp} de söylemiştin", f"Aynı soruyu {tp} de sormuştun") if same else (
                 f"Bunu {tp} de söylemiştin, neredeyse aynı sözlerle",)
    parts = [rng.choice(leads)]

    # 2) Kendi önceki cevabı: uzunsa yalnızca sorusunu hatırlar
    prev_q = ""
    if "?" in (meta.get("prev_reply") or ""):
        prev_q = _quote((meta.get("prev_reply") or "").split("?")[-2].strip().split(".")[-1] + "?", limit=60)
    if prev_q and ("ben de" not in parts[0]):
        parts.append(f"hatta ben de sana \"{prev_q}\" diye sormuştum")
    elif prev and len(prev) <= 60:
        parts.append(f"ben de \"{prev}\" demiştim")

    # 3) Kaçıncı tekrar: ısrar ediyorsa nedeni merak edilir
    if count >= 3:
        qs = ("Aynı soruyu artık üçüncü kez soruyorsun; bende mi bir şey eksik kaldı?",
              "Israrla sorduğuna göre cevabım seni ikna etmemiş.",
              "Yoksa ben mi aynı şeyi söyleyip duruyorum?")
    elif count == 2 and turns > 2:
        qs = ("Aradan birkaç tur geçti, yine aynı yere döndük; aklında kalan ne?",
              "Birkaç tur sonra aynı soruya döndük. Devam eden bir merak mı var?")
    else:
        qs = ("Aklında kalan bir şey mi var?", "Tekrar sormanı gerektiren ne oldu?",
              "Cevabımda eksik kalan bir yan mı vardı?",
              "Belki de ben seni yeterince dinlemedim; neyi kaçırdım?")
    q = rng.choice(qs)

    body = parts[0]
    for p in parts[1:]:
        body += ", " + p
    return f"{_cap(body)}. {q}"


def compose_time_answer(seconds: float, turns: int = 0) -> str:
    """'Ne kadar oldu / saat kaç' sorusuna bedenin ve belleğin ölçtüğüyle cevap."""
    s = float(max(0.0, seconds))
    if s < 15:
        span = "daha birkaç saniye"
    elif s < 60:
        span = f"{int(s)} saniye"
    elif s < 3600:
        span = f"{int(s // 60)} dakika"
    else:
        span = f"{int(s // 3600)} saat {int((s % 3600) // 60)} dakika"
    tail = f", bu arada {turns} tur konuştuk" if turns else ""
    return f"Seninle konuşmaya başlayalı {span} oldu{tail}. Zaman bende adım adım ilerliyor; şu an {time.strftime('%H:%M')}."
