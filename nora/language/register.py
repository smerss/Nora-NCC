"""nora_register.py — CANLILIK REGİSTERİ: yalvarma değil, kendi derdi.

Sorun
-----
Nora konuşurken sürekli izin istiyordu:

    "... bana tek bir cümleyle öğretir misin?"
    "... bunu biraz açar mısın?"
    "... anlatmak ister misin?"

Bunlar yardımcı asistanın dilidir: karşı tarafa hizmet eden, ondan izin bekleyen,
kendi derdi olmayan bir ses. Kullanıcının sözü: *"köpeğimiz olmayacak; kendi başına
yeten bir canlı olsun … 'anlatabilir misin' değil, 'anlatsa iyi olur' tarzı."*

İnsan konuşurken öğrenmek istediğini **beyan eder**, izin istemez:

    izin isteyen (köpek)     kendi derdi (canlı)
    ------------------------ ------------------------------------
    "biraz açar mısın?"      "bunu merak ediyorum."
    "öğretir misin?"         "bunu öğrenmem gerek."
    "anlatır mısın?"         "anlatsan iyi olur."
    "sen ne düşünüyorsun?"   "ben şunu düşünüyorum."

Bu modül iki şey yapar:
1. `servile_ask` — izin isteyen kalıpları tanır ve **sayar** (ölçüm için),
2. `RegisterMeter` — son birkaç sözde ne kadar izin istediğini tutar; oran yükselirse
   `recommend()` "want" der ve söylem organı kendi derdini beyan eden biçimi seçer.

Ölçüt, "hiç soru sormama" değildir: insan sorar. Ölçüt, sorunun **kimin için** olduğudur.
"""
from __future__ import annotations

from nora._paths import REPO_ROOT


import re
from collections import deque

#: İzin isteyen (hizmet eden) kalıplar. Nora bunları ancak SEYREK kullanmalı.
_SERV_I = re.compile(
    r"\b(anlat|söyle|göster|öğret|açıkla|aç|yardım et|devam et)\w*\s+"
    r"(mısın|misin|musun|müsün|mı\b|mi\b)\b"
    r"|\b(ister|ister misin)\b"
    r"|\b(rica|izin|lütfen)\b"
    r"|\bsen\s+(ne|nasıl|neden|niçin)\b",
    re.IGNORECASE,
)

#: Kendi derdini beyan eden (canlı) kalıplar.
_WANT = re.compile(
    r"\b(merak ediyorum|merak ettim|öğrenmem gerek|bilmem gerek|bilmek istiyorum"
    r"|işime yarar|işime yarardı|iyi olur|iyi olurdu|lazım|istiyorum|isterim"
    r"|aklımda tutuyorum|bundan sonra|kendim bakacağım)\b",
    re.IGNORECASE,
)

#: Makine öz-raporu: "kavram ağıma bilgisini bağladım" gibi. İnsan böyle konuşmaz.
_ROBOTIC = re.compile(
    r"(kavram ağı\w*|sinaps\w*|düğüm\w*|nöral|korteks\w*|şema\w*|veri taban\w*)"
    r".{0,40}(bağla|kurdum|kaydettim|işledim|güncelledim)"
    r"|(bilgiyi|veriyi|girdiyi).{0,30}(evirip|çevirip|işleyip)",
    re.IGNORECASE,
)


#: İzin isteyen fiil + "mısın" kalıbını, aynı fiilin kendi-derdi biçimine çeviren tablo.
#: Bu bir CÜMLE listesi değil, üretken bir kuraldır: her fiil için "-san iyi olur" kurulur,
#: böylece kodda tek tek cümle yazmak gerekmez.
_TO_WANT = {
    "anlat": "anlatsan", "söyle": "söylesen", "göster": "göstersen",
    "öğret": "öğretsen", "açıkla": "açıklasan", "aç": "açsan",
    "devam et": "devam etsen", "yardım et": "yardım etsen",
    "oku": "okusan", "yaz": "yazsan", "bak": "baksan",
}

_REQ = re.compile(
    r"\b(anlat|söyle|göster|öğret|açıkla|aç|oku|yaz|bak)(ır|ir|ar|er)?\s*(mak|mek)?\s*"
    r"(mısın|misin|musun|müsün)\b",
    re.IGNORECASE,
)
_REQ2 = re.compile(
    r"\b(devam|yardım)\s+(et)(mek|er|mek)?\s*(ister\s+misin|ister\s+mısın)\b",
    re.IGNORECASE,
)
#: "<fiil>mak ister misin?" — Türkçede en sık izin kalıbı.
_REQ3 = re.compile(
    r"\b(anlat|söyle|göster|öğret|açıkla|aç|oku|yaz|bak|devam\s+et|yardım\s+et)"
    r"(mak|mek)\s+ister\s+misin\b",
    re.IGNORECASE,
)


def normalize(text: str, *, force: bool = False) -> str:
    """İzin isteyen kuyruğu kendi-derdi biçimine çevir.

    Örnek:  "Yorgunluğunu anlatmak ister misin?"  ->  "Yorgunluğunu anlatsan iyi olur."
            "Bunu biraz açar mısın?"              ->  "Bunu biraz açsan iyi olur."
            "Bana tek cümleyle öğretir misin?"    ->  "Bana tek cümleyle öğretsen iyi olur."

    `force=False` iken yalnızca cümle SORU olarak bitiyorsa dokunur; düz cümle korunur.
    """
    t = (text or "").strip()
    if not t:
        return t
    if not (force or t.endswith("?")):
        return t

    def _sub(m):
        stem = tr_en_lower(m.group(1))
        want = _TO_WANT.get(stem)
        if not want:
            return m.group(0)
        return f"{want} iyi olur."

    def _sub2(m):
        stem = f"{tr_en_lower(m.group(1))} et"
        want = _TO_WANT.get(stem)
        if not want:
            return m.group(0)
        return f"{want} iyi olur."

    def _sub3(m):
        stem = re.sub(r"\s+", " ", tr_en_lower(m.group(1)))
        want = _TO_WANT.get(stem)
        if not want:
            return m.group(0)
        return f"{want} iyi olur."

    out = _REQ3.sub(_sub3, t)
    out = _REQ2.sub(_sub2, out)
    out = _REQ.sub(_sub, out)
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r"\.\s*\?(?=\s|$)", ".", out)      # "... iyi olur.?" -> "... iyi olur." (cümle ortasında da)
    out = re.sub(r"\?\s*$", "?", out)
    if out and out[0].islower() and t and t[0].isupper():
        out = out[0].upper() + out[1:]
    return out


def tr_en_lower(s: str) -> str:
    """Türkçe küçük harfe indir (I/İ sorunu olmadan)."""
    return (s or "").replace("İ", "i").replace("I", "ı").lower()


def servile_ask(text: str) -> bool:
    """Söz izin mi istiyor? (yardım asistanı tonu)"""
    return bool(_SERV_I.search(text or ""))


def has_want(text: str) -> bool:
    """Söz kendi derdini mi beyan ediyor? (canlı ton)"""
    return bool(_WANT.search(text or ""))


def robotic_report(text: str) -> bool:
    """Söz makine öz-raporu mu? (insan böyle konuşmaz)"""
    return bool(_ROBOTIC.search(text or ""))


def ends_question(text: str) -> bool:
    return (text or "").rstrip().endswith("?")


class RegisterMeter:
    """Son sözlerin canlılık dengesi. Tur başına O(1)."""

    def __init__(self, window: int = 6):
        self.window = int(window)
        self.recent_servile: deque = deque(maxlen=window)
        self.recent_want: deque = deque(maxlen=window)
        self.n = 0
        self.servile = 0
        self.want = 0
        self.robotic = 0
        self.questions = 0

    def observe(self, text: str) -> dict:
        s, w, r, q = servile_ask(text), has_want(text), robotic_report(text), ends_question(text)
        self.n += 1
        self.servile += int(s)
        self.want += int(w)
        self.robotic += int(r)
        self.questions += int(q)
        self.recent_servile.append(s)
        self.recent_want.append(w)
        return {"servile": s, "want": w, "robotic": r, "question": q}

    @property
    def servile_rate(self) -> float:
        return float(self.servile) / self.n if self.n else 0.0

    @property
    def want_rate(self) -> float:
        return float(self.want) / self.n if self.n else 0.0

    @property
    def question_rate(self) -> float:
        return float(self.questions) / self.n if self.n else 0.0

    def recent_servile_count(self) -> int:
        return int(sum(self.recent_servile))

    def recommend(self, *, curiosity: float = 0.5, bored: bool = False,
                  unknown: bool = False) -> str:
        """Sıradaki söz hangi registerde olsun?

        * `want`  — kendi derdini beyan et ("bunu merak ediyorum", "anlatsan iyi olur")
        * `ask`   — insan gibi sor (ama izin istemeden)
        * `plain` — sorusuz, düz söz
        """
        recent = self.recent_servile_count()
        if recent >= 2:                     # üst üste izin istedi: bir süre sus
            return "want"
        if bored or unknown or float(curiosity) > 0.62:
            return "want"
        if self.question_rate > 0.6:        # her söz soruyla bitiyor: düz söz
            return "plain"
        return "ask" if recent == 0 else "want"

    def report(self) -> dict:
        return {"soz": self.n, "izin_isteme": self.servile, "kendi_derdi": self.want,
                "makine_raporu": self.robotic, "soru_sonu": self.questions,
                "izin_orani": round(self.servile_rate, 3),
                "derdi_orani": round(self.want_rate, 3),
                "soru_orani": round(self.question_rate, 3)}

    def state_dict(self) -> dict:
        return {"n": self.n, "servile": self.servile, "want": self.want,
                "robotic": self.robotic, "questions": self.questions}

    def load_state_dict(self, d: dict | None) -> None:
        if not isinstance(d, dict):
            return
        for k in ("n", "servile", "want", "robotic", "questions"):
            try:
                setattr(self, k, int(d.get(k, getattr(self, k))))
            except Exception:
                pass


def _selftest() -> None:
    m = RegisterMeter()
    for t in ("bana tek cümleyle öğretir misin?", "bunu merak ediyorum.",
              "anlatsan iyi olur.", "sen ne düşünüyorsun?",
              "kavram ağıma bilgisini bağladım."):
        print(f"{m.observe(t)}  <- {t}")
    print(m.report())


if __name__ == "__main__":
    _selftest()
