"""NORA — KENDİ ÇIKARIMI: bilgiyi ezberlemek değil, kendi kendine türetmek.

Neden gerekli?
--------------
Bir dil modeli "bilgiye aç" olabilir: sorulduğunda getirir, sorulmadığında durur.
Nora'nın kendi içinde bir şeyler olmalı: elindeki iki bağdan ÜÇÜNCÜ bir bağ çıkarmak.

    Serçe bir kuştur.      (serçe ISA kuş)
    Kuşlar uçar.           (kuş CAN_DO uç)
    ---------------------
    -> serçe uçar           (kendi çıkarımı, kimse söylemedi)

Bu modül üç çıkarım türü yapar:

1. **Sınıf geçişliliği** (ISA zinciri): A ⊂ B, B ⊂ C ⇒ A ⊂ C
2. **Özellik kalıtımı**: A ⊂ B, B şu özelliğe sahip ⇒ A da sahip
3. **Nedensel zincir**: A → B, B → C ⇒ A → C (daha düşük güvenle)

Kurallar (uydurmama):
* Türetilen bağ, ağı `source="çıkarım"` olarak girer; öğretilmiş bağla karıştırılmaz.
* Güven, kanıt bağlarının güvenlerinin ÇARPIMI ve bir indirim katsayısıdır (asla 1.0 değil).
* Zıt kanıt varsa (CANNOT_DO / negatif bağ) çıkarım yapılmaz.
* Aynı bağ ikinci kez türetilmez; adım başına en fazla `budget` çıkarım (O(1) kalır).

Her çıkarım Nora'nın iç monoloğuna bir düşünce olarak düşer ve isterse söze katar:
"Kendi kendime şunu çıkardım: serçe uçar."
"""

from __future__ import annotations

from nora._paths import REPO_ROOT


import re

import time
from dataclasses import dataclass, field

from nora_lang import tr_lower

_INHERIT_RELS = ("HAS_PROP", "CAN_DO", "USED_FOR", "CAUSES", "HAS_PART")


@dataclass
class Inference:
    subj: str
    rel: str
    obj: str
    confidence: float
    basis: list[tuple[str, str, str]] = field(default_factory=list)
    kind: str = "kalıtım"
    step: int = 0
    told: bool = False

    def key(self) -> tuple[str, str, str]:
        return (self.subj, self.rel, self.obj)


class InferenceEngine:
    """Kendi bağlarından yeni bağ türeten küçük motor (adım başına O(kenar penceresi))."""

    MAX_DERIVED = 3000

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self.derived: list[Inference] = []
        self._keys: set[tuple[str, str, str]] = set()
        self._cursor = 0
        self.fresh = False
        self.fresh_until = 0.0
        self.total = 0

    # ---------------------------------------------------------------- türetme
    def step(self, agi, budget: int = 4, window: int = 64) -> list[Inference]:
        """Ağda dolaşır, yeni bağlar türetir. Dönen: bu adımda türetilenler."""
        graph = agi.lang.graph
        edges = getattr(graph, "edges", None)
        if not edges:
            return []
        out: list[Inference] = []
        n = len(edges)
        step_no = int(getattr(agi.mind, "age", 0))
        # Kayan pencere: her adımda farklı bir dilim taranır (tüm ağı taramak O(n) olurdu).
        start = self._cursor % n
        self._cursor = (self._cursor + window) % max(1, n)
        idxs = [(start + i) % n for i in range(min(window, n))]
        for i in idxs:
            if len(out) >= budget:
                break
            e = edges[i]
            if e.rel not in ("ISA",) + _INHERIT_RELS:
                continue
            subj, rel, obj = tr_lower(e.subj), e.rel, tr_lower(e.obj)
            if not subj or not obj or subj == obj:
                continue
            # 1) SINIF GEÇİŞLİLİĞİ ve 2) ÖZELLİK KALITIMI — ikisi de ISA bağını kullanır:
            #    A ISA B, B ISA C    -> A ISA C
            #    A ISA B, B rel X    -> A rel X   (rel: HAS_PROP/CAN_DO/USED_FOR/CAUSES/HAS_PART)
            if rel == "ISA":
                for e2 in graph._out_edges.get(obj, ()):   # noqa: SLF001 (yerel indeks, O(1))
                    if e2.weight <= 0.25:
                        continue
                    o2 = tr_lower(e2.obj)
                    if e2.rel == "ISA":
                        inf = self._make(subj, "ISA", o2,
                                         conf=e.weight * e2.weight * 0.8,
                                         basis=[(e.subj, "ISA", e.obj), (e2.subj, "ISA", e2.obj)],
                                         kind="sınıf geçişliliği", step=step_no)
                    elif e2.rel in _INHERIT_RELS:
                        inf = self._make(subj, e2.rel, o2,
                                         conf=e.weight * e2.weight * 0.75,
                                         basis=[(e.subj, "ISA", e.obj), (e2.subj, e2.rel, e2.obj)],
                                         kind="özellik kalıtımı", step=step_no)
                    else:
                        inf = None
                    if inf:
                        out.append(inf)
            # 3) Nedensel zincir: A CAUSES B, B CAUSES C -> A CAUSES C
            if rel == "CAUSES":
                for e2 in graph._out_edges.get(obj, ()):   # noqa: SLF001
                    if e2.rel != "CAUSES" or e2.weight <= 0.25:
                        continue
                    inf = self._make(subj, "CAUSES", tr_lower(e2.obj),
                                     conf=e.weight * e2.weight * 0.6,
                                     basis=[(e.subj, "CAUSES", e.obj), (e2.subj, "CAUSES", e2.obj)],
                                     kind="nedensel zincir", step=step_no)
                    if inf:
                        out.append(inf)
        if out:
            self.fresh = True
            self.fresh_until = time.time() + 30.0
            self.total += len(out)
        elif time.time() > self.fresh_until:
            self.fresh = False
        return out

    def _make(self, subj: str, rel: str, obj: str, *, conf: float,
              basis: list[tuple[str, str, str]], kind: str, step: int) -> Inference | None:
        if not subj or not obj or subj == obj:
            return None
        key = (subj, rel, obj)
        if key in self._keys:
            return None
        if len(self.derived) >= self.MAX_DERIVED:
            return None
        # Zıt kanıt: aynı özne için CANNOT_DO varsa türetme (bilgi ağına dokunmadan kontrol).
        conf = max(0.05, min(0.9, float(conf)))
        inf = Inference(subj=subj, rel=rel, obj=obj, confidence=conf, basis=basis, kind=kind, step=step)
        self.derived.append(inf)
        self._keys.add(key)
        return inf

    def commit(self, agi, infs: list[Inference]) -> int:
        """Türetilen bağları bilgi ağına işler (kaynağı 'çıkarım' olarak işaretli)."""
        graph = agi.lang.graph
        n = 0
        for inf in infs:
            try:
                ex = graph._edge_map.get((inf.subj, inf.rel, inf.obj))   # noqa: SLF001
                if ex is not None:
                    continue
                if inf.rel == "CAN_DO":
                    for e in graph._out_edges.get(inf.subj, ()):   # noqa: SLF001
                        if e.rel == "CANNOT_DO" and tr_lower(e.obj) == inf.obj:
                            return n
                graph.add_edge(inf.subj, inf.rel, inf.obj, source="çıkarım",
                               speaker="kendi", weight=float(inf.confidence), bind_mb=False)
                n += 1
            except Exception:
                continue
        return n

    # ---------------------------------------------------------------- dile çevirme
    def line_for(self, agi, inf: Inference) -> str:
        """Çıkarımı Türkçe cümleye çevirir (kalıp cevap değil, Broca çekimi)."""
        try:
            from nora_lang import Proposition
            prop = agi.lang.broca.realize_prop(Proposition(subj=inf.subj, rel=inf.rel, obj=inf.obj))
        except Exception:
            prop = f"{inf.subj} {inf.obj}"
        prop = (prop or "").strip().rstrip(".")
        if not prop:
            return ""
        return f"Kendi kendime şunu çıkardım: {prop}."

    @staticmethod
    def _words(t: str) -> set:
        return {w for w in re.findall(r"[a-zçğıöşüâîû]{4,}", (t or "").lower())}

    def last_line(self, agi, topic: str = "") -> str:
        """Söylenmemiş bir çıkarımı cümleye çevirir. Aynı çıkarım İKİ KEZ söylenmez.

        `topic` verilirse önce o konuya değen çıkarım aranır: Nora durup dururken
        konuyla ilgisiz bir türetimini araya sıkıştırmaz (bu, iskelet konuşma kadar
        yapay durur). Konuya değen yoksa sessiz kalır — susmak uydurmaktan iyidir.
        """
        if not self.derived:
            return ""
        cands = [i for i in self.derived[-60:] if not i.told]
        if not cands:
            return ""
        tw = self._words(topic)
        if tw:
            rel = [i for i in cands if tw & self._words(f"{i.subj} {i.obj}")]
            if not rel:
                return ""
            cands = rel
        inf = cands[-1]
        line = self.line_for(agi, inf)
        inf.told = True
        return line

    def explain(self) -> str:
        if not self.derived:
            return "Henüz kendi başıma bir çıkarım yapmadım."
        inf = self.derived[-1]
        basis = "; ".join(f"{s} {r} {o}" for s, r, o in inf.basis)
        return (f"Son çıkarımım ({inf.kind}, güven {inf.confidence:.2f}): "
                f"{inf.subj} {inf.rel} {inf.obj}. Dayanak: {basis}.")

    # ---------------------------------------------------------------- kalıcılık
    def state_dict(self) -> dict:
        return {"derived": [{"s": i.subj, "r": i.rel, "o": i.obj, "c": i.confidence,
                             "b": [list(x) for x in i.basis], "k": i.kind, "st": i.step}
                            for i in self.derived[-600:]],
                "total": self.total}

    def load_state_dict(self, d: dict) -> None:
        self.derived, self._keys = [], set()
        for rec in (d.get("derived") or []):
            try:
                inf = Inference(subj=str(rec["s"]), rel=str(rec["r"]), obj=str(rec["o"]),
                                confidence=float(rec.get("c", 0.5)),
                                basis=[tuple(x) for x in (rec.get("b") or [])],
                                kind=str(rec.get("k", "")), step=int(rec.get("st", 0)))
                self.derived.append(inf)
                self._keys.add(inf.key())
            except Exception:
                continue
        self.total = int(d.get("total", len(self.derived)) or 0)
