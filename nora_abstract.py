"""nora_abstract.py — BAŞARISIZLIKTAN SOYUTLAMA: takılınca YENİ İLKEL öneren organ.

Ayrım (bu organın varlık sebebi)
--------------------------------
    KOMPOZİSYON : var olan ilkellerin içinde yol bulmak      (nora_synth)
    İCAT        : uzayı genişletmek                          (burası)

`nora_synth` "kitap -> ktp"yi bulduğunda uzayda yürüdü: süzgeci insan yazmıştı.
Burada sistem, HİÇBİR kompozisyonun çözemediği örneklerle karşılaşınca durur,
başarısızlık izini yazar, izlerin ortak biçimini çıkarır ve **yeni bir ilkel**
önerir. Öneri örneklerin TAMAMINI açıklamak zorunda; açıklamıyorsa çöpe gider.

Dürüst sınır (bunu saklamıyorum)
--------------------------------
İcat UZAYI (aileler: kaydırma / döndürme / ikame) insan tarafından verilmiş bir
*ön-yargı*dır. Sistemin kattığı: hangi ailenin, hangi parametreyle, hangi örnek
kümesini açıkladığı — ve bunun DENENEREK doğrulanması. Yani "uzayı tamamen kendi
açtı" değil; "verilen yasa biçimleri içinden doğru olanı ölçerek seçti ve onu
kalıcı bir ilkel olarak kaydetti". Aile eklemek hâlâ insan işi; aile sayısı
`invented_from_family` ile ölçülür, ölçülmeden iddia edilmez.

Dürüstlük kuralları
-------------------
1. **Önce kompozisyon denenir.** Mevcut ilkeller (+ süzgeç) örnekleri çözüyorsa
   icat YOKTUR; "gerek yok" denir.
2. **Tek örnek kural değildir.** En az iki çift; hepsi açıklanmalı.
3. **Belirsizlik seçilmez.** İki aile de aynı örnekleri açıklıyorsa ve yalınlıkları
   eşitse "belirsiz" denir, ilkel kaydedilmez.
4. **Ad, parametreden türer.** Cümle değil: `kaydırma(+1)`, `döndürme(2)`.
"""
from __future__ import annotations

#: Kaydırma alfabesi (veri). ASCII doğrulaması için a-z öne alınmıştır.
ALPHABET = "abcdefghijklmnopqrstuvwxyzçğıöşü"
#: İcat aileleri: YASA BİÇİMLERİ (parametreleri örneklerden çıkar).
FAMILIES = ("kaydırma", "döndürme", "ikame")


def _shift(text: str, k: int) -> str:
    out = []
    for ch in str(text or ""):
        low = ch.lower()
        if low in ALPHABET:
            i = (ALPHABET.index(low) + int(k)) % len(ALPHABET)
            nxt = ALPHABET[i]
            out.append(nxt.upper() if ch.isupper() else nxt)
        else:
            out.append(ch)
    return "".join(out)


def _rotate(text: str, k: int) -> str:
    s = str(text or "")
    if not s:
        return s
    k = int(k) % len(s)
    return s[-k:] + s[:-k] if k else s


def _substitute(text: str, table: dict) -> str:
    return "".join(table.get(ch, ch) for ch in str(text or ""))


def build(family: str, params):
    """Kayıtlı bir icadı yeniden kur (yeni oturumda da çalışsın)."""
    if family == "kaydırma":
        return lambda s: _shift(s, int(params))
    if family == "döndürme":
        return lambda s: _rotate(s, int(params))
    if family == "ikame" and isinstance(params, dict):
        table = dict(params)
        return lambda s: _substitute(s, table)
    return None


def name_of(family: str, params) -> str:
    if family == "kaydırma":
        k = int(params)
        return f"kaydırma({'%+d' % k})"
    if family == "döndürme":
        return f"döndürme({int(params)})"
    if family == "ikame":
        return f"ikame({len(params or {})})"
    return family


class Abstraction:
    """Başarısızlık izlerinden ilkel öneren ocak."""

    MAX_TRACES = 64

    def __init__(self):
        self.traces: list[dict] = []
        self.invented: list[dict] = []
        self.refused = 0

    # --------------------------------------------------------------- iz toplama
    def observe(self, examples, why: str = "") -> dict:
        ex = [(str(a), str(b)) for a, b in (examples or []) if str(a) and str(b)]
        rec = {"examples": ex, "why": str(why)}
        self.traces.append(rec)
        if len(self.traces) > self.MAX_TRACES:
            del self.traces[:-self.MAX_TRACES]
        return rec

    # ------------------------------------------------------------------ öneri
    def _candidates(self, ex):
        """Örnekleri AÇIKLAYAN (aile, parametre) adayları; yalınlık sırasıyla."""
        outs = [b for _a, b in ex]
        cands = []
        # 1) kaydırma: tek tamsayı parametre
        for k in range(1, len(ALPHABET)):
            if all(_shift(a, k) == b for a, b in ex):
                cands.append(("kaydırma", k))
        # 2) döndürme: tek tamsayı parametre
        n = max(len(a) for a, _b in ex)
        for k in range(1, max(2, n)):
            if all(_rotate(a, k) == b for a, b in ex):
                cands.append(("döndürme", k))
        # 3) ikame: harf çizelgesi (en çok parametreli aile, en sona)
        table: dict = {}
        ok = True
        for a, b in ex:
            if len(a) != len(b):
                ok = False
                break
            for x, y in zip(a, b):
                if x in table and table[x] != y:
                    ok = False
                    break
                table[x] = y
            if not ok:
                break
        if ok and table and any(table.get(ch, ch) != ch for ch in table):
            if all(_substitute(a, table) == b for a, b in ex):
                cands.append(("ikame", table))
        return cands

    def propose(self, examples, composition_exists=None, register=None) -> dict:
        """Örneklerden ilkel öner. Dönen: status in {no-need, insufficient,
        ambiguous, invented}."""
        ex = [(str(a), str(b)) for a, b in (examples or []) if str(a) and str(b)]
        if len(ex) < 2:
            return {"status": "insufficient", "why": "az-örnek"}
        if composition_exists is None:
            composition_exists = self._composition_solves(ex)
        if composition_exists:
            # KOMPOZİSYON YETİYOR: icat yok. (İcat yalnız tıkanınca.)
            return {"status": "no-need"}
        cands = self._candidates(ex)
        if not cands:
            self.refused += 1
            return {"status": "insufficient", "why": "aile-yok"}
        # YALINLIK: kaydırma/döndürme tek parametre, ikame çizelge. Aynı yalınlıkta
        # birden çok aday kalırsa seçilmez (uydurma yerine belirsizlik).
        best = cands[0]
        simple = [c for c in cands if c[0] in ("kaydırma", "döndürme")]
        if len(simple) > 1 and simple[0][1] != simple[1][1]:
            return {"status": "ambiguous", "n": len(simple),
                    "cands": [name_of(f, p) for f, p in simple[:3]]}
        if simple:
            best = simple[0]
        family, params = best
        fn = build(family, params)
        if fn is None:
            return {"status": "insufficient", "why": "kurulamadı"}
        if not all(fn(a) == b for a, b in ex):      # son kapı: örneklerin hepsi
            self.refused += 1
            return {"status": "insufficient", "why": "doğrulanamadı"}
        rec = {"family": family, "params": params, "name": name_of(family, params),
               "examples": ex, "n": len(ex)}
        self.invented.append(rec)
        if register is not None:
            try:
                register(rec["name"], fn)
            except Exception:
                pass
        return {"status": "invented", **rec}

    @staticmethod
    def _composition_solves(ex) -> bool:
        """Mevcut ilkeller (+süzgeç) bu örnekleri çözüyor mu? Çözüyorsa icat yok."""
        try:
            from nora_synth import Synthesizer
        except Exception:
            return False
        try:
            res = Synthesizer().learn(ex)
        except Exception:
            return False
        return res.get("status") in ("adopted", "ambiguous")

    # ------------------------------------------------------------ kalıcılık
    def state(self) -> dict:
        return {"invented": [{"family": r.get("family"), "params": r.get("params"),
                              "name": r.get("name"), "n": int(r.get("n", 0))}
                             for r in self.invented]}

    def load_state(self, d: dict | None) -> None:
        recs = (d or {}).get("invented") or []
        self.invented = []
        for r in recs:
            if not isinstance(r, dict):
                continue
            fam = str(r.get("family") or "")
            par = r.get("params")
            if build(fam, par) is None:
                continue
            self.invented.append({"family": fam, "params": par,
                                  "name": str(r.get("name") or name_of(fam, par)),
                                  "n": int(r.get("n", 0) or 0), "examples": []})
