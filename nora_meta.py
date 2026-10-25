#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_meta.py — ÜST-BİLİŞ: çakışma/tekrar izleme ve zaman bilinci.

Beyindeki anterior singulat (ACC) çakışma izleme ile episodik belleğin birleşimi:
Nora, aynı sözün kısa süre içinde tekrarlandığını FARK EDER, ne zaman söylendiğini
hatırlar, o an ne cevap verdiğini anımsar ve bunu kendi sözüne katar.

Burada cümle BANKASI yoktur: farkındalık sözünün parçaları `nora_priors.json` içinde
motor toplulukları olarak yaşar (bkz. nora_priors). Bu modül yalnızca
  * imza çıkarımı (kök + kip + kişi; sözcük sırasından bağımsız),
  * zaman farkı ve tur sayısı,
  * kendi önceki cevabından alıntı
verisini toplar, topluluklara YARIŞTIRIR ve kazanan parçalardan cümleyi kurar.
Yani kodda "hangi cümle" yazmaz; kodda yarış vardır, cümleler veride büyür.
"""
from __future__ import annotations

import random
import time

from nora_lang import TurkishMorphology, tr_lower


def _field(agi):
    """Söz toplulukları alanı: AGI'nınki varsa o, yoksa süreç varsayılanı."""
    try:
        f = getattr(agi, "priors", None)
        if f is not None:
            return f
    except Exception:
        pass
    from nora_priors import default_field
    return default_field()


def _meter(agi):
    """İskelet ölçeri: sözün kaynağı buraya yazılır."""
    try:
        m = getattr(agi, "provenance", None)
        if m is not None:
            return m
    except Exception:
        pass
    from nora_provenance import default_meter
    return default_meter()

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
    prev = _quote(meta.get("prev_reply", ""), limit=40)

    # 1) Farkındalık cümlesi: topluluklar YARIŞIR (cümle kodda değil, veride)
    fld, mtr = _field(agi), _meter(agi)
    ctx = {"gap": gap, "same": same, "count": count, "turns": turns}
    slots = {"tp": tp}
    lead = fld.pick("meta.notice.lead", ctx, slots, rng=rng, prov=mtr)
    if not lead:
        return ""
    parts = [lead]

    # 2) Kendi önceki cevabı: uzunsa yalnızca sorusunu hatırlar
    prev_q = ""
    if "?" in (meta.get("prev_reply") or ""):
        prev_q = _quote((meta.get("prev_reply") or "").split("?")[-2].strip().split(".")[-1] + "?", limit=60)
    if prev_q and ("ben de" not in parts[0]):
        parts.append(f"hatta ben de sana \"{prev_q}\" diye sormuştum")
        mtr.record("bellek", 1)
    elif prev and len(prev) <= 60:
        parts.append(f"ben de \"{prev}\" demiştim")
        mtr.record("bellek", 1)

    # 3) Kaçıncı tekrar: ısrar ediyorsa nedeni merak edilir (yine yarış)
    q = fld.pick("meta.notice.question", ctx, slots, rng=rng, prov=mtr)
    if not q:
        return _cap(", ".join(parts)) + "."

    body = parts[0]
    for p in parts[1:]:
        body += ", " + p
    return f"{_cap(body)}. {q}"


def compose_insistence(meta: dict, agi=None, rng=None) -> str:
    """Aynı söz üçüncü kez gelince: uzun serzeniş değil, KISA ve değişen bir soru.

    "Bunu az önce de söylemiştin, ben de ... demiştim" cümlesi ilk tekrarda insanca;
    ama ısrar eden her turda aynı cümleyi kurmak iskelet olur. Burada yalnızca tek
    cümlelik, merak eden bir karşılık verilir.
    """
    import random
    rng = rng or random
    count = int(meta.get("count", 3))
    # Cümle bankası yok: topluluklar yarışır. Tazelik cezası aynı sorunun üst üste
    # gelmesini engeller; öğrenilmiş ağırlık ise Nora'nın kendi alışkanlığını taşır.
    return _field(agi).pick("meta.insist.question", {"count": count}, {},
                            rng=rng, prov=_meter(agi)) or ""


def compose_time_answer(seconds: float, turns: int = 0, *, kind: str = "elapsed",
                        steps: int = 0, rng=None, agi=None) -> str:
    """Zaman sorusuna bedenin ve belleğin ÖLÇTÜĞÜYLE cevap.

    kind="clock"   -> yalnızca saati söyler (saat kaç?)
    kind="elapsed" -> konuşma süresi + tur sayısı + iç adım sayısı (ne kadar oldu?)
    Süre/saat ÖLÇÜLÜR; cümle verideki motor topluluklarının yarışından çıkar.
    """
    s = float(max(0.0, seconds))
    clock = time.strftime("%H:%M")
    _rng = rng if rng is not None else random
    fld, mtr = _field(agi), _meter(agi)
    if kind == "clock":
        head = fld.pick("meta.clock.head", {}, {"clock": clock}, rng=_rng, prov=mtr)
        tail = fld.pick("meta.clock.tail", {}, {}, rng=_rng, prov=mtr)
        mtr.record("beden", 1)          # saat ÖLÇÜLDÜ
        if head is None:                  # veri yoksa: yalnız ölçümü söyler
            head = f"{clock}."
        return head + (tail or "")
    if s < 15:
        span = "daha birkaç saniye"
    elif s < 60:
        span = f"{int(s)} saniye"
    elif s < 3600:
        span = f"{int(s // 60)} dakika"
    else:
        span = f"{int(s // 3600)} saat {int((s % 3600) // 60)} dakika"
    bits = [f"Seninle konuşmaya başlayalı {span} oldu"]
    if turns:
        bits.append(f"bu arada {turns} tur konuştuk")
    if steps:
        bits.append(f"kendi içimde {int(steps)} adım yaşadım")
    body = ", ".join(bits)
    mtr.record("beden", 1)              # süre ÖLÇÜLDÜ, uydurulmadı
    close = fld.pick("meta.elapsed.close", {}, {"clock": clock}, rng=_rng, prov=mtr)
    if not close:
        close = f"Şu an {clock}."
    return f"{body}. {close}"
