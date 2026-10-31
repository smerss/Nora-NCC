"""nora_sets.py — KÜME OPERATÖRLERİ: operatör elle yazılmaz, TANIMDAN SENTEZLENİR.

Fikir
-----
Bir işlemi "elle yazmak" demek, girdiyi çıktıya çeviren adımları insanın seçmesi
demektir. Burada insan yalnızca **tanımı** (aksiyomları) verir:

    f "kesişim"dir  ⟺  (1) f(A,B) ⊆ A ∩ B          (kapsam)
                        (2) A ∩ B ⊆ f(A,B)          (tamlık)
                        (3) f tekrarsızdır           (teklik)
                        (4) f(A,B) = f(B,A)          (değişme)
                        (5) f alfabetiktir            (sıra)

İşlemin KENDİSİNİ Nora arar: ilkellerden (harfler, üyelik süzgeci, tekilleştirme,
sıralama) kurulmuş 96 aday programı tek tek çalıştırır ve aksiyomları sağlayanı
seçer. Aksiyomları sağlayan tek program kaldıysa operatör YAZILMIŞTIR; birden çok
kalırsa uydurulmaz, belirsizlik bildirilir.

Böylece `nora_operators.json` (insan rejistrisi) BÜYÜMEZ: 9 operatör sabit kalır,
küme operatörleri "kendi yazdığı" defterine düşer.
"""
from __future__ import annotations

from nora._paths import REPO_ROOT


import random

from nora_lang import tr_lower

#: İlkeller. Hepsi tek satırlık, birbirinden bağımsız; operatör bunların DİZİLİŞİDİR.
BASES = ("A", "B", "AB", "BA")          # hangi sözcük(ler)ten başlanır
FILTERS = ("yok", "tut", "dışla")        # diğer sözcüğe göre süz
UNIQUES = (False, True)
ORDERS = ("doğal", "alfabetik", "ters alfabetik", "ters")

LETTERS = "abcçdefgğhıijklmnoöprsştuüvyz"


def letters(text: str) -> list[str]:
    """Bir sözcüğün harfleri (küçük harf, yalnız alfabe)."""
    return [c for c in tr_lower(str(text or "")) if c.isalpha()]


def _run(prog: dict, a: str, b: str) -> str:
    """Aday programı çalıştır: operatörün TANIMI bu fonksiyon değil, `prog`dur."""
    la, lb = letters(a), letters(b)
    base = prog["base"]
    seq = {"A": la, "B": lb, "AB": la + lb, "BA": lb + la}[base]
    flt = prog["filter"]
    if flt != "yok":
        other = set(lb) if base == "A" else set(la)
        if flt == "tut":
            seq = [c for c in seq if c in other]
        else:
            seq = [c for c in seq if c not in other]
    if prog["unique"]:
        seq = list(dict.fromkeys(seq))
    order = prog["order"]
    if order == "alfabetik":
        seq = sorted(seq)
    elif order == "ters alfabetik":
        seq = sorted(seq, reverse=True)
    elif order == "ters":
        seq = list(reversed(seq))
    return "".join(seq)


def candidates() -> list[dict]:
    """Arama uzayı: ilkellerin bütün dizilişleri (96 aday)."""
    out = []
    for base in BASES:
        for flt in FILTERS:
            if flt != "yok" and base in ("AB", "BA"):
                continue                      # birleşimde "diğeri" tanımsız
            for uniq in UNIQUES:
                for order in ORDERS:
                    out.append({"base": base, "filter": flt,
                                "unique": uniq, "order": order})
    return out


def _probe_words(n: int, seed: int = 7) -> list[tuple[str, str]]:
    """Aksiyomların sınandığı sözcük çiftleri (sabit tohum: ölçüm tekrarlanabilir)."""
    rng = random.Random(seed)
    pairs = [("kitap", "kalem"), ("deniz", "masa"), ("bahçe", "kalem"),
             ("okul", "yol"), ("çiçek", "keçi"), ("armut", "turşu")]
    while len(pairs) < n:
        w1 = "".join(rng.choice(LETTERS) for _ in range(rng.randint(3, 6)))
        w2 = "".join(rng.choice(LETTERS) for _ in range(rng.randint(3, 6)))
        pairs.append((w1, w2))
    return pairs[:n]


def _laws(name: str, f, a: str, b: str) -> dict[str, bool]:
    """Aksiyomlar: operatörün ne OLMASI gerektiği (insan burada yalnız tanım verir)."""
    sa, sb = set(letters(a)), set(letters(b))
    out = f(a, b)
    so = set(out)
    law: dict[str, bool] = {}
    if name == "kesişim":
        law["kapsam"] = so <= (sa & sb)
        law["tamlık"] = (sa & sb) <= so
        law["değişme"] = f(a, b) == f(b, a)
    elif name == "birleşim":
        law["kapsam"] = so <= (sa | sb)
        law["tamlık"] = (sa | sb) <= so
        law["değişme"] = f(a, b) == f(b, a)
    elif name == "fark":
        law["kapsam"] = so <= (sa - sb)
        law["tamlık"] = (sa - sb) <= so
    else:
        raise ValueError(name)
    law["teklik"] = len(out) == len(so)
    law["sıra"] = out == "".join(sorted(out))
    return law


class SetSynth:
    """Küme operatörlerini aksiyomlardan sentezleyen organ."""

    NAMES = ("kesişim", "birleşim", "fark")

    def __init__(self, n_probe: int = 8):
        self.n_probe = int(n_probe)
        self.derived: dict[str, dict] = {}
        self.ambiguous: dict[str, int] = {}
        self.pairs = _probe_words(self.n_probe)

    # ---------------------------------------------------------------- sentez
    def discover(self, name: str) -> dict:
        """Aksiyomları sağlayan programı ara. Uydurma yok: birden çoksa bildir."""
        if name in self.derived:
            return {"status": "adopted", "entry": self.derived[name]}
        keep = []
        for prog in candidates():
            f = lambda x, y, p=prog: _run(p, x, y)          # noqa: E731
            if all(all(_laws(name, f, a, b).values()) for a, b in self.pairs):
                keep.append(prog)
        keep = self._dedupe(keep)
        if len(keep) == 1:
            p = keep[0]
            entry = {
                "id": f"kendi.kume.{len(self.derived) + 1}",
                "name": name,
                "program": p,
                "ops": [p["base"], p["filter"],
                        "tekilleştir" if p["unique"] else "tekilleştirme",
                        p["order"]],
                "origin": "sentez",
                "laws": sorted(_laws(name, lambda x, y, q=p: _run(q, x, y),
                                     *self.pairs[0]).keys()),
                "probe": len(self.pairs),
            }
            self.derived[name] = entry
            return {"status": "adopted", "entry": entry}
        if not keep:
            return {"status": "none"}
        self.ambiguous[name] = len(keep)
        return {"status": "ambiguous", "n": len(keep),
                "cands": [[k["base"], k["filter"], k["unique"], k["order"]] for k in keep[:4]]}

    def _dedupe(self, progs: list[dict]) -> list[dict]:
        """DAVRANIŞ imzasıyla tekilleştir: aynı fonksiyonu yazan iki program tek sayılır.

        `A üzerinden süz` ile `B üzerinden süz` kesişimde AYNI fonksiyondur; arada
        kalmak sahte bir belirsizlik olurdu. Sıra tercihi: A, B, AB, BA.
        """
        probe = self.pairs + [("abca", "cabd"), ("zzz", "z"), ("", "kalem")]
        seen: dict[tuple, dict] = {}
        pref = {b: i for i, b in enumerate(BASES)}
        for p in sorted(progs, key=lambda x: (pref.get(x["base"], 9), x["unique"], x["order"])):
            sig = tuple(_run(p, a, b) for a, b in probe)
            seen.setdefault(sig, p)
        return list(seen.values())

    # ---------------------------------------------------------------- kullanım
    def apply(self, name: str, a: str, b: str) -> str | None:
        """Operatörü çalıştır (daha önce sentezlenmemişse önce sentezle)."""
        res = self.discover(name)
        if res.get("status") != "adopted":
            return None
        return _run(res["entry"]["program"], a, b)

    def state_dict(self) -> dict:
        return {"derived": self.derived, "ambiguous": self.ambiguous}

    def load_state_dict(self, d: dict | None) -> None:
        """Kayıtlı operatörler, program yeniden kurularak geri getirilir."""
        d = d or {}
        self.derived = {k: v for k, v in (d.get("derived") or {}).items()
                        if isinstance(v, dict) and isinstance(v.get("program"), dict)}
        self.ambiguous = dict(d.get("ambiguous") or {})
