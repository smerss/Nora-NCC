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

# ================================================= PARAMETRELİ AİLE ÜRETİCİLERİ
#: DİKKAT: burada YENİ FORM yok. `PRIMITIVES` içinde zaten yazılmış dört ailenin
#: (`ilk/son n harfi al/at`) PARAMETRESİ açılıyor. Numaralandırılmış üyeler n=1..4;
#: bu üretici aynı mekanizmayı HER n için kurar. Şema bunu kullanarak, insanın
#: SAYMADIĞI bir parametreyi kapsar. Yeni ifade gücü buradan doğar.
PARAM_GENERATORS = {
    "ilk {n} harfi al": lambda n: (lambda t: t[:n]),
    "son {n} harfi al": lambda n: (lambda t: t[-n:]),
    "ilk {n} harfi at": lambda n: (lambda t: t[n:]),
    "son {n} harfi at": lambda n: (lambda t: t[:-n]),
}


def family_of(name):
    """Adı PARAMETRELİ bir aileye çöz: ('ilk {n} harfi al', 3) / ('kaydırma({k})', 1).

    İki kaynak var ve ikisi de AYNI işi yapar — bir ad, bir ailenin üyesidir:
      - dilim aileleri: 'ilk 3 harfi al' → 'ilk {n} harfi al'
      - İCADIN aileleri: 'kaydırma(+2)' → 'kaydırma({k})'  (icat edilen ilkel de
        parametreli bir ailenin üyesidir; onun da şeması olur)
    """
    name = str(name)
    m = re.match(r"^(ilk|son) (\d+) harfi (al|at)$", name)
    if m:
        return f"{m.group(1)} {{n}} harfi {m.group(3)}", int(m.group(2))
    _low = name.replace("İ", "i").replace("I", "ı").lower()
    m2 = re.match(r"^([a-zçğıöşü]+)\(([+-]?\d+)\)$", _low)
    if m2 and m2.group(1) in ("kaydırma", "döndürme"):
        return f"{m2.group(1)}({{k}})", int(m2.group(2))
    return None


def instantiate(family, n):
    """Aile + parametre -> (ad, fonksiyon, maliyet).

    Dilim ailelerinde n >= 1; icat ailelerinde parametre işaretli olabilir (kaydırma).
    """
    fam = str(family)
    if fam.startswith(("kaydırma(", "döndürme(")):
        try:
            from nora_abstract import build as _ab_build, name_of as _ab_name
            base = fam.split("(", 1)[0]
            k = int(n)
            if base == "döndürme" and k < 0:
                return None
            fn = _ab_build(base, k)
            return (_ab_name(base, k) if base != "döndürme" else f"döndürme({k})", fn, 1) \
                if fn is not None else None
        except Exception:
            return None
    gen = PARAM_GENERATORS.get(str(family))
    try:
        n = int(n)
    except Exception:
        return None
    if gen is None or n < 1:
        return None
    return (str(family).replace("{n}", str(n)), gen(n), 1)


# ============================================================ DONMUŞ REÇETE (KİLİT)
#: Tur 40 kuralı: **yeni elle yazılmış form yok.** Bu sayılar ARTMAMALI; artarsa
#: `agi_battery` içindeki A106 donar ve insan katkısı sessizce büyüyemez. Yeni
#: ilkel ancak `nora_reify` (kendi bileşimini adlandırma) ile doğar — ve o ilkel
#: `Synthesizer.extra` içinde yaşar, bu listede DEĞİL.
FROZEN_PRIMITIVES = 22
FROZEN_FILTERS = 5

# ===================================================================== YÜKSEK MERTEBE
#: SÜZGEÇ AİLESİ. Dikkat: burada insan tek bir MEKANİZMA yazıyor (bir yüklemi tut/at),
#: hangi süzgecin kullanılacağını ÖRNEKLER belirliyor. "Sesli harfleri at" diye bir
#: operatörü kimse yazmadı; örneklerden çıktı.
#:
#: Neden ilkel kompozisyonu yetmiyor: k[0], k[-1], sorted, reverse — hiçbiri "harfi
#: kendi özelliğine göre seç" işlemini yapamaz. Bu, ilkel KÜMESİNİN genişlemesi değil,
#: ilkellerin ÜSTÜNE bir kombinator eklenmesi: süzgeç.
_VOWELS = set("aeıioöuüâîû")


def _mk_sieve(pred, keep: bool, label: str):
    """Bir yüklemi tutan/atayan program adımı üret."""
    if keep:
        def _fn(s, _p=pred):
            return "".join(ch for ch in s if _p(ch))
        nm = f"{label} tut"
    else:
        def _fn(s, _p=pred):
            return "".join(ch for ch in s if not _p(ch))
        nm = f"{label} at"
    return (nm, _fn, 1)


def _is_vowel(ch: str) -> bool:
    return ch.lower() in _VOWELS


def _first_only(s: str) -> str:
    """Tekrarları at: her harfin YALNIZ ilk geçişi kalır (sıra korunur)."""
    seen, out = set(), []
    for ch in s:
        k = ch.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(ch)
    return "".join(out)


FILTER_PRIMITIVES = [
    _mk_sieve(_is_vowel, False, "sesli harfleri"),
    _mk_sieve(_is_vowel, True, "sesli harfleri"),
    _mk_sieve(lambda ch: ch.isalpha() and not _is_vowel(ch), False, "sessiz harfleri"),
    _mk_sieve(lambda ch: ch.isalpha() and not _is_vowel(ch), True, "sessiz harfleri"),
    ("tekrarları at", _first_only, 2),
]

#: Ad -> (ad, fonksiyon) kayıt defteri: kaydedilmiş bir program BAŞKA OTURUMDA da
#: yeniden kurulabilsin (süzgeçler dahil).
_PRIM_BY_NAME = {n: (n, f) for n, f, _c in list(PRIMITIVES) + list(FILTER_PRIMITIVES)}


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
    def from_spec(cls, names: list[str], spec_map: dict | None = None) -> "Program | None":
        """Kayıtlı bir programı yeniden kur (başka oturumda da çalışsın).

        `spec_map` verilirse (ör. örnek düzeyinde İCAT EDİLMİŞ ilkeller) ondan kurulur.
        """
        by_name = dict(_PRIM_BY_NAME)
        if spec_map:
            by_name.update({k: (k, f) for k, f in spec_map.items()})
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

    MAX_DEPTH = 3

    def __init__(self):
        self.programs: list[dict] = []      # kalıcı defter
        self.last: Program | None = None
        #: İCAT EDİLMİŞ ilkeller (bu örneğe özel; küresel durumu kirletmesin).
        self.extra: list[tuple] = []
        #: Başarılı sentez kancası: (spec, examples) -> None. `nora_reify` buradan izler.
        self.on_adopt = None
        #: ŞEMALAR (delikli makrolar): {"steps", "hole", "family", "seen"}.
        self.schemas: list[dict] = []
        #: Son çözümde hangi şema/parametre kullanıldı (ölçüm için).
        self.last_schema = None
        #: Beraberlik sayacı: sabit tohumlu "yazı-tura" budur.
        self._tie_n = 0
        #: Ayırt edici örnek bekleyen adaylar (belirsizlik UYDURULMAZ, çözülür).
        self._pending: tuple[list[Program], list[tuple[str, str]]] | None = None

    def add_primitive(self, name: str, fn, cost: int = 1) -> None:
        """Fail izlerinden çıkan ilkeli bu örneğin uzayına ekle."""
        name = str(name or "")
        if not name or any(nm == name for nm, _f, _c in self.extra):
            return
        self.extra.append((name, fn, int(cost)))

    def add_schema(self, steps, hole: int, family: str, seen,
                   inferred: dict | None = None) -> dict | None:
        """Şemayı uzaya ekle: delikli makro (parametre ailesi üzerinden)."""
        steps = [str(x) for x in (steps or [])]
        _ok_fam = (family in PARAM_GENERATORS
                   or str(family).startswith(("kaydırma(", "döndürme("))
                   # Ailesi TABLODA olmayabilir: deliğin anlamı veriden çıkarıldıysa
                   # (ör. "taban dizgenin ilk n harfi") aile adı yalnız bir etikettir.
                   or bool(inferred))
        if not (0 <= int(hole) < len(steps)) or not _ok_fam:
            return None
        if any(sc["family"] == family and sc["steps"] == steps and sc["hole"] == hole
               for sc in self.schemas):
            return None
        sc = {"steps": steps, "hole": int(hole), "family": str(family),
              "seen": sorted({int(x) for x in (seen or [])}),
              "inferred": inferred}
        self.schemas.append(sc)
        return sc

    def _schema_programs(self, examples, ignore_examples: bool = False):
        """Şemayı GÖRÜLMEMİŞ parametreler için somutlaştır ve örnekleri sına.

        `ignore_examples=True`: yalnızca aday ÜRET (bileşim için); doğrulama çağıranda.
        """
        out = []
        for sc in self.schemas:
            # Parametre aralığı VERİDEN: dilimde n, sözcük uzunluğunu aşamaz;
            # kaydırmada k alfabeyi dolaşır. Sabit 1..12 yazmak, 17 gibi bir
            # parametreyi "kapsam dışı" bırakıp şemayı işe yaramaz hâle getiriyordu.
            _shift = str(sc["family"]).startswith(("kaydırma(", "döndürme("))
            if _shift:
                try:
                    from nora_abstract import ALPHABET as _ALFA
                    _maxn = len(_ALFA) - 1
                except Exception:
                    _maxn = 29
            else:
                _maxn = max([len(str(a)) for a, _b in examples] or [12])
            for n in range(1, int(_maxn) + 1):
                if n in sc.get("seen", []):
                    continue
                # ÖNCE VERİDEN ÇIKARILAN KURAL (tablo değil): "taban programın ilk n
                # harfi". Tablo (PARAM_GENERATORS) yalnız yedek.
                inst = None
                _inf = sc.get("inferred")
                if _inf and _inf.get("kind") == "prefix":
                    _base = Program.from_spec(list(_inf.get("base") or []),
                                              spec_map={nm: fn for nm, fn, _c in self.extra})
                    if _base is not None:
                        inst = (f"ilk {n} harfi al",
                                (lambda t, _b=_base, _k=int(n): str(_b.apply(t))[:_k]), 1)
                if inst is None:
                    inst = instantiate(sc["family"], n)
                if inst is None:
                    continue
                names = list(sc["steps"])
                names[sc["hole"]] = inst[0]
                ops = []
                ok = True
                for nm in names:
                    if nm == inst[0]:
                        ops.append((nm, inst[1]))
                        continue
                    hit = next((p for p in list(PRIMITIVES) + list(self.extra) if p[0] == nm), None)
                    hit = hit or next((p for p in FILTER_PRIMITIVES if p[0] == nm), None)
                    if hit is None:
                        ok = False
                        break
                    ops.append((hit[0], hit[1]))
                if not ok:
                    continue
                prog = Program(ops, cost=max(1, len(ops) - 1))
                if ignore_examples or all(prog.apply(i) == o for i, o in examples):
                    out.append((prog, sc, n))
        return out

    def tie_pick(self, cands):
        """BERABERLİK POLİTİKASI: berabere adaylar arasından SEÇ ve %50 raporla.

        Sıra: en kısa betimleme → eşit maliyetliler arasında sabit tohumlu "yazı-tura"
        (sayaç). Dönen olasılık her zaman 0.5: bu bir tahmin değil, **beraberlik**
        beyanıdır. Sessiz kalmak kalibrasyon üretmez; bağlanmak üretir.
        """
        cands = list(cands or [])
        if not cands:
            return None, None
        best_cost = min(p.cost for p in cands)
        tied = sorted([p for p in cands if p.cost == best_cost], key=lambda p: p.desc())
        self._tie_n += 1
        pick = tied[(self._tie_n * 7 + 3) % len(tied)]   # sabit, tekrarlanabilir
        return pick, 0.5

    def drop_schema(self, name: str, family: str) -> bool:
        """Budanan şemayı arama uzayından çıkar."""
        before = len(self.schemas)
        self.schemas = [sc for sc in self.schemas
                        if not (sc.get("family") == family and sc.get("name") == name)]
        if len(self.schemas) != before:
            return True
        self.schemas = [sc for sc in self.schemas if sc.get("family") != family]
        return len(self.schemas) != before

    def drop_primitive(self, name: str) -> bool:
        """Budanan ilkeli uzaydan çıkar (defterde ölü kalır, aramada görünmez)."""
        before = len(self.extra)
        self.extra = [(nm, f, c) for nm, f, c in self.extra if nm != str(name)]
        return len(self.extra) != before

    def spec_map(self) -> dict:
        return {nm: fn for nm, fn, _c in self.extra}

    # ------------------------------------------------------------------ arama
    def _candidates(self, examples: list[tuple[str, str]]) -> list[Program]:
        if not examples:
            return []
        outs = [o for _i, o in examples]
        self._used_schema = None
        self._schema_ops: dict[str, tuple] = {}
        found: list[Program] = []
        # 0) İCAT EDİLMİŞ ilkeller: kendi yazdığım adımlar en önce denenir.
        for nm, fn, c in self.extra:
            p = Program([(nm, fn)], c)
            if all(p.apply(i) == o for i, o in examples):
                found.append(p)
        if found:
            return self._dedupe(found, examples, outs)
        # tek ilkel
        for nm, fn, c in PRIMITIVES:
            p = Program([(nm, fn)], c)
            if all(p.apply(i) == o for i, o in examples):
                found.append(p)
        if found:
            return self._dedupe(found, examples, outs)
        # iki ilkel (kendi yazdığı ilkeller de bu havuzda: REIFY buradan kullanılır)
        pool2 = list(PRIMITIVES) + list(self.extra)
        for nm1, f1, c1 in pool2:
            for nm2, f2, c2 in pool2:
                p = Program([(nm1, f1), (nm2, f2)], c1 + c2)
                if all(p.apply(i) == o for i, o in examples):
                    found.append(p)
        if found:
            return self._dedupe(found, examples, outs)
        # üç ilkel: sınır burada. Daha derini aranmaz (patlama), ve aranmadığı SÖYLENİR.
        pool3 = list(PRIMITIVES) + list(self.extra)
        for nm1, f1, c1 in pool3:
            for nm2, f2, c2 in pool3:
                for nm3, f3, c3 in pool3:
                    p = Program([(nm1, f1), (nm2, f2), (nm3, f3)], c1 + c2 + c3)
                    if all(p.apply(i) == o for i, o in examples):
                        found.append(p)
        if found:
            return self._dedupe(found, examples, outs)
        # SÜZGEÇ AŞAMASI: ilkel kompozisyonu tükendi. Şimdi YÜKSEK MERTEBE kombinator
        # denenir (tut/at). Sıra ÖNEMLİ: buraya yalnız taban arama BOŞ dönünce gelinir,
        # böylece eski davranış (N16, N22) hiç değişmez.
        for nm, fn, c in FILTER_PRIMITIVES:
            p = Program([(nm, fn)], c)
            if all(p.apply(i) == o for i, o in examples):
                found.append(p)
        if found:
            return self._dedupe(found, examples, outs)
        sch = self._schema_programs(examples)
        if sch:
            self.last_schema = (sch[0][1], sch[0][2])
            self._used_schema = (sch[0][1], sch[0][2])
            return self._dedupe([p for p, _sc, _n in sch], examples, outs)
        # ŞEMA + İLKEL: şemanın ürettiği adım tek başına yetmiyorsa, o adım normal
        # 2-adım aramasının HAVUZUNA girer (ör. büyük harfe çevir ∘ sırala∘dilimle(n)).
        # Yeni form değil: mevcut iki yeteneğin (şema, kompozisyon) bileşimi.
        _sch_pool: list[tuple] = []
        for sp, sc, n in self._schema_programs(examples, ignore_examples=True):
            # Şemanın ürettiği program TEK ADIM gibi havuza girer: kendi adımları
            # içinde kalsın, dışarıya "tek yetenek" olarak görünsün.
            _nm = " ∘ ".join(str(x) for x in sp.spec())
            _sch_pool.append((_nm, (lambda t, _p=sp: _p.apply(t)), int(sp.cost)))
            self._schema_ops[_nm] = (sc, n)
        pool = list(PRIMITIVES) + list(FILTER_PRIMITIVES) + list(self.extra) + _sch_pool
        for nm1, f1, c1 in pool:
            for nm2, f2, c2 in pool:
                p = Program([(nm1, f1), (nm2, f2)], c1 + c2)
                if all(p.apply(i) == o for i, o in examples):
                    found.append(p)
        if found:
            # Hangi aday şema kullandıysa onu işaretle (kullanım sayacı için).
            for nm, _f, _c in _sch_pool:
                if any(nm in [x[0] for x in p.ops] for p in found):
                    self.last_schema = self._schema_ops.get(nm)
                    self._used_schema = self._schema_ops.get(nm)
                    break
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
        self._notify(p.spec(), examples)
        return {"status": "adopted", "program": p, "desc": p.desc(), "n": 1,
                "rid": p.rid, "schema": getattr(self, "_used_schema", None)}

    def _notify(self, spec, examples) -> None:
        """Başarılı sentezi dinleyicilere bildir (reify izler)."""
        cb = getattr(self, "on_adopt", None)
        if cb is None:
            return
        try:
            cb(list(spec or []), list(examples or []))
        except Exception:
            pass

    def _adopt(self, p: Program, examples) -> dict:
        self._pending = None
        p.rid = f"kendi.op.{len(self.programs) + 1}"
        self.programs.append({"id": p.rid, "ops": p.spec(), "desc": p.desc(),
                              "examples": [[i, o] for i, o in examples],
                              "origin": "kendi"})
        self.last = p
        self._notify(p.spec(), examples)
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
                  "zeytin", "çiçek", "yıldız", "bulut", "ankara", "kitapçı",
                  "Kalem", "Deniz", "Elma"):
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
        p = Program.from_spec(list(rec.get("ops") or []), spec_map=self.spec_map())
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
