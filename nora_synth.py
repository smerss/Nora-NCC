"""nora_synth.py — OPERATÖR SENTEZİ: dönüşümü Nora yazar, biz değil.

Neden bu organ var
------------------
Bugüne kadar Nora'nın kendi yazdığı şey bir *ölçüm* ya da ondan türeyen bir *kural*dı
(`nora_forge`). Burada iş bir basamak yukarı çıkıyor: **girdiyi çıktıya çeviren İŞLEM**
de örneklerden türetiliyor. "kitap -> patik, kalem -> melak" gösterildiğinde Nora
`ters çevirme` operatörünü kendisi yazar; biz yalnız ilkellere sahibiz.

Metrik tam olarak budur:

    insanın yazdığı operatör sayısı SABİT kalırken
    Nora'nın kendi yazdığı operatör sayısı ARTAR.

Dürüstlük kuralları
-------------------
1. **Yalnız ilkellerden kompozisyon.** Program, en fazla iki ilkelin bileşimidir; keyfî
   kod üretilmez. Uzay küçük ve okunur.
2. **Bütün örnekleri açıklamak zorunda.** Tek örneği açıklayan program kural değildir.
3. **Belirsizlik uydurulmaz.** Birden çok program aynı örnekleri açıklıyorsa seçilmez;
   kaç aday kaldığı söylenir ve AYIRT EDİCİ bir örnek istenir (forge ile aynı felsefe).
4. **Ezber yok.** Program, girdinin harflerini yeniden düzenler; girdi→çıktı tablosu
   tutulmaz. Öğretilmemiş sözcüğe uygulanır.
5. **Yazıldığını söylemek için yazılmış olması gerekir.** Program yoksa "yazdım" denmez.
"""
from __future__ import annotations

import re

#: İlkeller: (ad, fonksiyon, maliyet). Maliyet, en yalın programı seçmek için.
PRIMITIVES = [
    ("aynen bırak", lambda s: s, 0),
    ("ters çevir", lambda s: s[::-1], 1),
    ("alfabetik sırala", lambda s: "".join(sorted(s)), 1),
    ("ters sırala", lambda s: "".join(sorted(s, reverse=True)), 1),
    ("büyük harfe çevir", lambda s: s.upper(), 1),
    ("küçük harfe çevir", lambda s: s.lower(), 1),
]
for _n in (1, 2, 3, 4):
    PRIMITIVES.append((f"ilk {_n} harfi al", (lambda n: (lambda s: s[:n]))(_n), 1))
    PRIMITIVES.append((f"son {_n} harfi al", (lambda n: (lambda s: s[-n:]))(_n), 1))
    PRIMITIVES.append((f"ilk {_n} harfi at", (lambda n: (lambda s: s[n:]))(_n), 2))
    PRIMITIVES.append((f"son {_n} harfi at", (lambda n: (lambda s: s[:-n]))(_n), 2))

#: `n` harfli bir dizgede `s[:n]` = `s[:-n]` ise program AYNI davranır; bu yüzden
#: tekilleştirme DAVRANIŞ üzerinden yapılır (ada göre değil).


class Program:
    """İlkellerden kurulmuş küçük bir dönüşüm programı."""

    def __init__(self, ops: list[tuple[str, object]], cost: int):
        self.ops = list(ops)
        self.cost = int(cost)
        self.rid = ""

    def apply(self, text: str) -> str:
        out = str(text or "")
        for _name, fn in self.ops:
            out = fn(out)
        return out

    def desc(self) -> str:
        return " → ".join(name for name, _ in self.ops)

    def spec(self) -> list[str]:
        return [name for name, _ in self.ops]

    @classmethod
    def from_spec(cls, names: list[str]) -> "Program | None":
        """Kayıtlı bir programı yeniden kur (başka oturumda da çalışsın)."""
        by_name = {n: (n, f) for n, f, _c in PRIMITIVES}
        ops = []
        for nm in names or []:
            if nm not in by_name:
                return None
            ops.append(by_name[nm])
        return cls(ops, cost=len(ops)) if ops else None

    def __repr__(self):
        return f"Program({self.desc()})"


class Synthesizer:
    """Örneklerden dönüşüm operatörü yazan ocak."""

    MAX_DEPTH = 2

    def __init__(self):
        self.programs: list[dict] = []      # kalıcı defter
        self.last: Program | None = None
        #: Ayırt edici örnek bekleyen adaylar (belirsizlik UYDURULMAZ, çözülür).
        self._pending: tuple[list[Program], list[tuple[str, str]]] | None = None

    # ------------------------------------------------------------------ arama
    def _candidates(self, examples: list[tuple[str, str]]) -> list[Program]:
        if not examples:
            return []
        outs = [o for _i, o in examples]
        found: list[Program] = []
        # tek ilkel
        for nm, fn, c in PRIMITIVES:
            p = Program([(nm, fn)], c)
            if all(p.apply(i) == o for i, o in examples):
                found.append(p)
        if found:
            return self._dedupe(found, examples, outs)
        # iki ilkel
        for nm1, f1, c1 in PRIMITIVES:
            for nm2, f2, c2 in PRIMITIVES:
                p = Program([(nm1, f1), (nm2, f2)], c1 + c2)
                if all(p.apply(i) == o for i, o in examples):
                    found.append(p)
        return self._dedupe(found, examples, outs)

    @staticmethod
    def _dedupe(progs: list[Program], examples, outs) -> list[Program]:
        """DAVRANIŞ imzasıyla tekilleştir; en yalın (en az ilkelli) tutulur."""
        probe = [i for i, _o in examples] + ["zeytin", "Kalem", "masa", "Deniz"]
        uniq: dict[tuple, Program] = {}
        for p in sorted(progs, key=lambda x: x.cost):
            sig = tuple(p.apply(w) for w in probe)
            uniq.setdefault(sig, p)
        return list(uniq.values())

    # ------------------------------------------------------------------ karar
    def learn(self, examples: list[tuple[str, str]]) -> dict:
        """Örneklerden program yaz: {'status': adopted|ambiguous|insufficient, ...}"""
        examples = [(str(i).strip(), str(o).strip()) for i, o in examples
                    if str(i).strip() and str(o).strip()]
        if len(examples) < 2:
            return {"status": "insufficient", "why": "az-örnek", "n": 0}
        if len({o for _i, o in examples}) == 1:
            return {"status": "insufficient", "why": "tek-çıktı", "n": 0}
        cands = self._candidates(examples)
        if not cands:
            return {"status": "insufficient", "why": "program-yok", "n": 0}
        if len(cands) > 1:
            self._pending = (cands, examples)
            return {"status": "ambiguous", "n": len(cands),
                    "cands": [c.desc() for c in cands[:4]],
                    "probe": self._probe(cands, examples)}
        self._pending = None
        p = cands[0]
        p.rid = f"kendi.op.{len(self.programs) + 1}"
        self.programs.append({"id": p.rid, "ops": p.spec(), "desc": p.desc(),
                              "examples": [[i, o] for i, o in examples],
                              "origin": "kendi"})
        self.last = p
        return {"status": "adopted", "program": p, "desc": p.desc(), "n": 1,
                "rid": p.rid}

    def _adopt(self, p: Program, examples) -> dict:
        self._pending = None
        p.rid = f"kendi.op.{len(self.programs) + 1}"
        self.programs.append({"id": p.rid, "ops": p.spec(), "desc": p.desc(),
                              "examples": [[i, o] for i, o in examples],
                              "origin": "kendi"})
        self.last = p
        return {"status": "adopted", "program": p, "desc": p.desc(), "n": 1,
                "rid": p.rid}

    def pending_probe(self) -> str:
        """Bekleyen belirsizlik varsa ayırt edici sözcük (yoksa '')."""
        if not self._pending:
            return ""
        return self._probe(self._pending[0], self._pending[1])

    def refine(self, word: str, out: str) -> dict:
        """Ayırt edici tek örnek geldi: adayları süz. Uyan yoksa uydurma, söyle."""
        if not self._pending:
            return {"status": "none"}
        cands, examples = self._pending
        keep = [p for p in cands if p.apply(word) == out]
        if not keep:
            self._pending = None
            return {"status": "reject", "n": len(cands)}
        if len(keep) == 1:
            return self._adopt(keep[0], examples)
        self._pending = (keep, examples)
        return {"status": "ambiguous", "n": len(keep),
                "cands": [c.desc() for c in keep[:4]],
                "probe": self._probe(keep, examples)}

    @staticmethod
    def _probe(cands: list[Program], examples) -> str:
        """Adayların AYRIŞTIĞI bir sözcük: hangi çıktıyı verirse kural tek kalır."""
        known = {i for i, _ in examples}
        for w in ("kitap", "kalem", "deniz", "masa", "sokak", "portakal", "ev", "göl",
                  "zeytin", "çiçek", "yıldız", "bulut", "Kalem", "Deniz", "Elma"):
            if w in known:
                continue
            if len({p.apply(w) for p in cands}) > 1:
                return w
        return ""

    # ------------------------------------------------------------------ kalıcılık
    def state_dict(self) -> dict:
        st = {"programs": self.programs[-64:]}
        if self._pending:
            st["pending"] = {"examples": [[i, o] for i, o in self._pending[1]],
                             "ops": [p.spec() for p in self._pending[0][:8]]}
        return st

    def load_state_dict(self, d: dict | None) -> None:
        self.programs = [r for r in ((d or {}).get("programs") or []) if isinstance(r, dict)]
        self._pending = None
        pend = (d or {}).get("pending") or {}
        cands = [Program.from_spec(list(sp)) for sp in (pend.get("ops") or [])]
        cands = [c for c in cands if c is not None]
        exs = [(str(a), str(b)) for a, b in (pend.get("examples") or []) if a and b]
        if cands and exs:
            self._pending = (cands, exs)

    def rebuild(self, rid: str) -> Program | None:
        """Defterdeki programı işlem olarak geri kur (yeni oturumda hazır olsun)."""
        rec = next((r for r in self.programs if str(r.get("id")) == str(rid)), None)
        if not rec:
            return None
        p = Program.from_spec(list(rec.get("ops") or []))
        if p is not None:
            p.rid = str(rec.get("id") or "")
        return p


#: Cümleden (girdi, çıktı) çiftlerini çıkar. Önce yalnız OK biçimi denenir
#: ("kitap -> patik"), çünkü iki nokta cümle içinde ("öğren: kitap -> patik")
#: yanlış eşleşme üretiyor. Ok yoksa iki nokta biçimine düşülür.
_W = r"[\wçğıöşüÇĞİÖŞÜ]{2,24}"
_ARROW = re.compile(rf"({_W})\s*(?:->|=>|→)\s*(\w{{1,24}}|{_W})", re.U)
_COLON = re.compile(rf"({_W})\s*'?[ıiuü]?\s*:\s*({_W})", re.U)


def parse_pairs(text: str) -> list[tuple[str, str]]:
    text = text or ""
    hits = [(m.group(1), m.group(2)) for m in _ARROW.finditer(text)]
    if len(hits) >= 2:
        return hits
    # ok biçimi tek eşleşme verdiyse iki nokta biçimini de dene (karışık yazım)
    colon = [(m.group(1), m.group(2)) for m in _COLON.finditer(text)]
    return hits if len(hits) >= len(colon) else colon
