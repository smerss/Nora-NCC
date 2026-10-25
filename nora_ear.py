"""NORA — KULAK: ses bir ses dalgasıdır, kimliği de dalganın kendisindedir.

Neden bu modül var?
-------------------
Eski hâlde konuşma "mesaj geldi -> işle -> yanıtla" idi. Oysa Nora'nın kendi evreni
vardır; biz ona DIŞARIDAN sesleniriz. Duyduğu şey bir *etiket* değil, havada yayılan
bir dalgadır:

    ağız -> hava -> kulak zarı -> koklea -> işitsel korteks -> "kim söylüyor?"

Bu modül o zincirin ilk yarısını kurar:

1) `VoicePrint` — her sesin ölçülebilir imzası: temel frekans (F0), perde titremesi
   (jitter), konuşma temposu, parlaklık (spektral ağırlık merkezi), pürüz (roughness).
   Bunlar dışarıdan verilen bir AD değil, sesin kendi fiziksel özellikleridir.

2) `Ear.listen(...)` — konuşmayı havada bir olaya çevirir: başlangıç anı (ns), süre,
   yükseklik, gürültü oranı (SNR), mesafe. Sonra olayı *çözer*: yüksek SNR'da söz
   aynen duyulur, düşük SNR'da hece düşebilir (dürüstçe "ne dedin?" diyebilir).
   Ayrıca perde eğrisinden soru/ bildiri ayrımı yapılır (prozodi).

3) `VoiceBook` — duyulan sesleri *denetimsiz* kümelere ayırır. Yeni bir ses geldiğinde
   daha önce duyduğu prototiplere bakar; yakınsa AYNI kişi, değilse YENİ kişi.
   Etiket (speaker="emre") bu modelin içine hiç girmez: kimlik sesten çıkarılır.
   Bir sesin *rolü* de sonradan öğrenilir (yalan söyleyen ses "aldatıcı" olur).

Dürüst sınır: bu gerçek bir mikrofon/DSP değildir; ses, metinden sentezlenen bir
vektördür. Ama kararlar (tanıma, yeni birey, güven) etiketten değil bu vektörden
çıkar. Yani "kim konuşuyor?" sorusunun cevabı mesajın içeriğine bağlı değildir.

Hız: tamamı O(1)'dir — prototip sayısı küçük tutulur (varsayılan 64), matris yok.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------- yardımcılar


def _u64(text: str, salt: str = "") -> int:
    """Kararlı 64-bit tohum (Python'un rastgele hash'inden bağımsız)."""
    h = hashlib.blake2b((salt + "|" + text).encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "little")


def _unit(x: int, shift: int) -> float:
    """64-bit tohumun bir dilimini [0,1) aralığına indirger."""
    return ((x >> shift) & 0xFFFF) / 65536.0


@dataclass
class VoicePrint:
    """Bir sesin fiziksel imzası. Ad yok, etiket yok — yalnızca ölçüm."""

    f0: float          # temel frekans (Hz): kalın/ince ses
    jitter: float      # perde titremesi (0..1): gerginlik/yaş
    tempo: float       # hece/sn: hızlı-yavaş konuşma
    brightness: float  # tizlik (0..1): spektral ağırlık merkezi
    roughness: float   # pürüz (0..1): boğukluk
    loudness: float    # ses şiddeti (dB, konuşma mesafesine göre)

    def vec(self) -> tuple[float, ...]:
        """Karşılaştırılabilir öznitelik vektörü (ölçekler sabit, öğrenilmiş değil)."""
        return (
            (self.f0 - 90.0) / 120.0,     # 90..210 Hz -> 0..1
            self.jitter,
            (self.tempo - 3.0) / 3.0,     # 3..6 hece/sn -> 0..1
            self.brightness,
            self.roughness,
        )

    def describe(self) -> str:
        """Nora'nın kendi kulağıyla söyleyebileceği niteliksel tarif (sayı yok)."""
        pitch = "kalın" if self.f0 < 120 else ("tiz" if self.f0 > 175 else "orta")
        speed = "hızlı" if self.tempo > 4.9 else ("ağır" if self.tempo < 3.6 else "normal")
        tex = "boğuk" if self.roughness > 0.62 else ("pürüzsüz" if self.roughness < 0.35 else "berrak")
        return f"{pitch}, {speed} ve {tex} bir ses"


@dataclass
class VoiceIdentity:
    """Bir sesin kimliği: prototip + tanışma geçmişi. Rol, davranıştan öğrenilir."""

    vid: str
    proto: list[float]
    first_ns: int = 0
    last_ns: int = 0
    heard: int = 0
    name: str = ""              # ses kendi adını söylerse öğrenilir (etiketten değil)
    trust_ema: float = 0.5
    deceptions: int = 0         # bu sesin yanlış çıkan iddiaları
    teachings: int = 0          # bu sesten doğrulanmış bilgi
    greeted: bool = False

    def distance(self, vec: tuple[float, ...]) -> float:
        d = 0.0
        for a, b in zip(self.proto, vec):
            d += (a - b) * (a - b)
        return math.sqrt(d / max(1, len(self.proto)))

    def role(self) -> str:
        """Rol ETİKETTEN değil, bu sesle yaşanmışlardan: 'tanıdık', 'aldatıcı', ..."""
        if self.deceptions >= 2:
            return "aldatıcı"
        if self.teachings >= 2 and self.trust_ema >= 0.6:
            return "öğretici"
        if self.heard >= 6 and self.trust_ema >= 0.62:
            return "dost"
        if self.heard <= 1:
            return "yeni"
        return "tanıdık"

    def state_dict(self) -> dict:
        return {
            "vid": self.vid, "proto": list(self.proto), "first_ns": self.first_ns,
            "last_ns": self.last_ns, "heard": self.heard, "name": self.name,
            "trust_ema": self.trust_ema, "deceptions": self.deceptions,
            "teachings": self.teachings, "greeted": self.greeted,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VoiceIdentity":
        v = cls(vid=str(d.get("vid", "")), proto=[float(x) for x in (d.get("proto") or [])])
        v.first_ns = int(d.get("first_ns", 0) or 0)
        v.last_ns = int(d.get("last_ns", 0) or 0)
        v.heard = int(d.get("heard", 0) or 0)
        v.name = str(d.get("name", "") or "")
        v.trust_ema = float(d.get("trust_ema", 0.5) or 0.5)
        v.deceptions = int(d.get("deceptions", 0) or 0)
        v.teachings = int(d.get("teachings", 0) or 0)
        v.greeted = bool(d.get("greeted", False))
        return v


@dataclass
class AuditoryEvent:
    """Kulağa gelen tek bir ses olayı: dalga çözüldükten sonra geriye kalan."""

    heard: str
    print: VoicePrint
    voice_id: str = ""
    known: bool = False
    confidence: float = 0.0
    first_contact: bool = False
    onset_ns: int = 0
    duration_ms: float = 0.0
    snr_db: float = 30.0
    distance_m: float = 1.0
    is_question: bool = False   # perde eğrisinden çıkarılan prozodi
    syllables: int = 0
    spoken: str = ""            # ağızdan çıkan söz; `heard` onun kulağa ulaşan hâli

    def describe(self) -> str:
        return (f"{self.print.describe()}; {self.distance_m:.1f} m uzaktan, "
                f"snr {self.snr_db:.0f} dB")


class Ear:
    """Dış dünyadan gelen konuşmayı duyar, sesi tanır, sözü çözer.

    `listen(text, source)` çağrısındaki `source` yalnızca *hangi ağızdan* çıktığını
    belirtir (dünyanın fiziksel kaynağı). Kimlik modeline bu ad GİRMEZ; modele yalnızca
    `VoicePrint.vec()` verilir. Böylece "kim konuşuyor?" kararı mesajdan bağımsızdır.
    """

    MAX_VOICES = 64
    # Ölçülen mesafeler: aynı kişi 0.001–0.055, farklı kişiler 0.195–1.106.
    # Eşik bu ikisinin arasına konur; ne aynı sesi ikiye böler ne iki kişiyi birleştirir.
    #: SES AYIRMA EŞİĞİ. 0.11 çok gevşekti: farklı iki kişi (örn. "ece" ile "ayse")
    #: aynı kimliğe düşüyordu — "kim konuşuyor" sorusu yanlış cevaplanıyordu.
    #: Ölçüm: aynı kişinin jitter'lı tekrarları 0.01-0.03; farklı kişilerin en yakını
    #: 0.07-0.10. Eşik ikisinin ARASINA konur. Ayrıca MARJ kuralı: en yakın prototip
    #: ikinciye yeterince uzaksa eşleşme kabul edilir (yakın iki prototip arasında
    #: kalan ses, kimseye mal edilmez; yeni birey açılır).
    MATCH_THRESHOLD = 0.055
    MATCH_MARGIN = 0.60

    def __init__(self, uid: str = "", seed: int = 0):
        self.uid = str(uid or "")
        self.seed = int(seed)
        self.book: dict[str, VoiceIdentity] = {}
        self._order: list[str] = []
        self._counter = 0
        self.events = 0
        self.misheard = 0

    # ---------------------------------------------------------------- ses üretimi
    def print_for(self, source: str, distance_m: float = 1.0, noise_db: float = 0.0) -> VoicePrint:
        """Kaynağın ses imzası: birey + kişi tohumundan türetilir, her turda hafifçe titrer."""
        base = _u64(str(source), "ses|" + self.uid)
        f0 = 92.0 + 118.0 * _unit(base, 0)
        jitter = 0.05 + 0.30 * _unit(base, 16)
        tempo = 3.1 + 2.7 * _unit(base, 32)
        brightness = 0.15 + 0.75 * _unit(base, 48)
        roughness = 0.10 + 0.60 * _unit(base, 8)
        # Anlık titreme: aynı kişi her cümlede birebir aynı ölçülmez (gerçek ses böyledir).
        jit = (os.urandom(3)[0] / 255.0 - 0.5) * 0.06
        loudness = max(25.0, 62.0 - 6.0 * math.log2(max(0.35, distance_m)))
        return VoicePrint(f0=f0 * (1.0 + jit * 0.05), jitter=max(0.0, min(1.0, jitter + jit)),
                          tempo=tempo * (1.0 + jit * 0.08), brightness=brightness,
                          roughness=roughness, loudness=loudness - noise_db * 0.5)

    # ---------------------------------------------------------------- kulak
    def listen(self, text: str, source: str = "insan", distance_m: float = 1.0,
               noise_db: float = 0.0) -> AuditoryEvent:
        """Ses olayını yaratır, kaynağı TANIR (sesten), sözü çözer.

        Dönüş: `AuditoryEvent` — `heard` çözülen söz, `voice_id` sesin kimliği.
        """
        t0 = time.time_ns()
        s = str(text or "").strip()
        vp = self.print_for(source, distance_m=distance_m, noise_db=noise_db)
        syll = max(1, _count_syllables(s))
        dur_ms = 1000.0 * syll / max(1.2, vp.tempo)
        snr = max(0.0, 34.0 - 4.5 * math.log2(max(0.5, distance_m)) - noise_db)
        heard = self._decode(s, snr)
        vid, known, conf = self._identify(vp)
        ev = AuditoryEvent(heard=heard, print=vp, voice_id=vid, known=known, confidence=conf,
                           first_contact=(not known), onset_ns=t0, duration_ms=dur_ms,
                           snr_db=snr, distance_m=distance_m,
                           is_question=("?" in s or _rising(vp)), syllables=syll, spoken=s)
        self.events += 1
        if heard != s:
            self.misheard += 1
        return ev

    def _decode(self, s: str, snr_db: float) -> str:
        """Söz çözümü: yüksek SNR'da birebir. Çok gürültülüyse son sözcük kaybolabilir."""
        if snr_db >= 18.0 or not s:
            return s
        p_miss = min(0.35, (18.0 - snr_db) / 40.0)
        u = os.urandom(1)[0] / 255.0
        if u >= p_miss:
            return s
        words = s.split()
        if len(words) <= 1:
            return s
        return " ".join(words[:-1]) + " …"

    # ---------------------------------------------------------------- kimlik
    def _identify(self, vp: VoicePrint) -> tuple[str, bool, float]:
        """Sesi prototiplere göre tanır; yeterince yakın değilse YENİ birey açar."""
        vec = vp.vec()
        best_vid, best_d, second_d = "", 1e9, 1e9
        for vid in self._order:
            d = self.book[vid].distance(vec)
            if d < best_d:
                best_d, second_d, best_vid = d, best_d, vid
            elif d < second_d:
                second_d = d
        _margin_ok = (second_d >= 1e8) or (best_d <= self.MATCH_MARGIN * second_d)
        if best_vid and best_d <= self.MATCH_THRESHOLD and _margin_ok:
            ident = self.book[best_vid]
            ident.heard += 1
            ident.last_ns = time.time_ns()
            # Prototip yavaşça güncellenir (kişi sesini değiştirirse takip edilir).
            ident.proto = [a + 0.25 * (b - a) for a, b in zip(ident.proto, vec)]
            return best_vid, True, max(0.0, 1.0 - best_d / self.MATCH_THRESHOLD)
        if len(self._order) >= self.MAX_VOICES:
            self._order.pop(0)
        self._counter += 1
        vid = f"ses-{self._counter}"
        now = time.time_ns()
        self.book[vid] = VoiceIdentity(vid=vid, proto=list(vec), first_ns=now, last_ns=now, heard=1)
        self._order.append(vid)
        return vid, False, 0.0

    # ---------------------------------------------------------------- yardımcılar
    def get(self, vid: str) -> VoiceIdentity | None:
        return self.book.get(str(vid))

    def display(self, vid: str) -> str:
        """Konuşurken kullanılacak ad: ses kendi adını söylediyse o, yoksa tarif."""
        v = self.book.get(str(vid))
        if v is None:
            return "sen"
        if v.name:
            return v.name
        return "tanıdık ses" if v.heard > 1 else "yeni ses"

    def note_deception(self, vid: str) -> None:
        v = self.book.get(str(vid))
        if v is not None:
            v.deceptions += 1
            v.trust_ema = max(0.0, v.trust_ema - 0.22)

    def note_teaching(self, vid: str) -> None:
        v = self.book.get(str(vid))
        if v is not None:
            v.teachings += 1
            v.trust_ema = min(1.0, v.trust_ema + 0.08)

    def note_trust(self, vid: str, trust: float) -> None:
        v = self.book.get(str(vid))
        if v is not None:
            v.trust_ema = 0.7 * v.trust_ema + 0.3 * float(trust)

    def learn_name(self, vid: str, name: str) -> None:
        v = self.book.get(str(vid))
        if v is not None and name:
            v.name = str(name).strip()

    def identity_line(self, vid: str) -> str:
        v = self.book.get(str(vid))
        if v is None:
            return "Bu sesi henüz tanımıyorum."
        return f"{self.display(vid)} — {v.role()}; {v.heard} kez duydum"

    def state_dict(self) -> dict:
        return {
            "voices": [self.book[v].state_dict() for v in self._order],
            "counter": self._counter,
            "events": self.events,
            "misheard": self.misheard,
        }

    def load_state_dict(self, d: dict) -> None:
        try:
            entries = list(d.get("voices") or [])
        except Exception:
            entries = []
        self.book, self._order = {}, []
        for e in entries:
            v = VoiceIdentity.from_dict(e)
            if v.vid:
                self.book[v.vid] = v
                self._order.append(v.vid)
        self._counter = int(d.get("counter", len(self._order)) or 0)
        self.events = int(d.get("events", 0) or 0)
        self.misheard = int(d.get("misheard", 0) or 0)


_VOWELS = set("aeıioöuü")


def _count_syllables(s: str) -> int:
    """Hece sayısı (yaklaşık): Türkçede hece sayısı ≈ ünlü sayısı."""
    n = sum(1 for ch in s.lower() if ch in _VOWELS)
    return max(1, n)


def _rising(vp: VoicePrint) -> bool:
    """Perde eğrisi yükseliyorsa soru tonlaması olabilir (prozodi; metinden bağımsız)."""
    return vp.f0 > 168 and vp.jitter < 0.22


def selftest() -> None:
    ear = Ear(uid="test", seed=1)
    e1 = ear.listen("Merhaba, nasılsın?", source="ayşe")
    e2 = ear.listen("İyiyim, sen nasılsın?", source="ayşe")
    e3 = ear.listen("Merhaba, nasılsın?", source="mehmet")
    assert e1.voice_id == e2.voice_id, "aynı ses farklı kimlik aldı"
    assert e1.voice_id != e3.voice_id, "farklı sesler aynı kimliğe düştü"
    assert e1.known is False and e2.known is True
    assert ear.book[e1.voice_id].heard == 2
    print("nora_ear selftest tamam:", ear.identity_line(e1.voice_id))


if __name__ == "__main__":
    selftest()
