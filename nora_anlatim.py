"""NORA — ANLATIM: ansiklopedi değil, konuşan bir zihin.

Sorun neydi?
------------
Aynı soru sorulduğunda Nora **birebir aynı cümleyi** veriyordu:

    Su nedir?  ->  Su susuzluk gidermek için kullanılır; suyun nedeni yaşamın sürmesidir.
    Su nedir?  ->  Su susuzluk gidermek için kullanılır; suyun nedeni yaşamın sürmesidir.
    Su nedir?  ->  Su susuzluk gidermek için kullanılır; suyun nedeni yaşamın sürmesidir.

Bu, insan konuşmasına benzemiyor: insan aynı şeyi ikinci kez sorulduğunda **başka bir
kapıdan** anlatır — kısalır, örnek verir, kendi yaşantısına bağlar, "bunu daha önce
söylemiştim" der, hatta bazen susar. Ansiklopedi maddesi hep aynıdır; insan hep aynı değil.

Bu organ ne yapar?
------------------
Kavram ağından gelen **bağları** (rel, obj) alır ve her turda *farklı bir anlatım*
kurar. Yeni bilgi uydurmaz: kullandığı her içerik grafiktir, beden kaydıdır ya da kendi
çıkarımıdır. Değişen şey **çerçeve**dir:

    çekirdek      tür + işlev (ilk cevap; tanışma anı)
    tek bağ       yalnızca bir bağ, en az kullanılmış olanı
    ad koyma      "Suyun işi susuzluk gidermek." (eylemlikle, kısa)
    duyma         "Suyu kendim deneyimlemedim; bilişsel ağımda ... diye duruyor."
    beden         beden sürüşü yüksekse: "Boğazım kuruyor; su aklımda." + bağ
    bellek        bu kavramı yaşadıysa: "Su bana iyi geldi." + bağ
    çıkarım       kendi türettiği ve henüz söylemediği bir bağ varsa onu ekler
    sıralama      iki bağın sırasını değiştirir, bağlayıcıyı değiştirir

Dürüstlük kuralları (hepsi zorunlu):

* **Beden yoksa el yok.** "Suyu içerim" gibi bir cümle kurmaz; deneyimlemediği şeyi
  "deneyimledim" diye anlatmaz. Somasensory kaydı olmayan kavram için yalnızca
  "duydum/biliyorum" çerçeveleri açıktır.
* **Bilgi uydurulmaz.** Her cümlenin içeriği grafik bağı, bellek kaydı ya da çıkarımdır.
* **Tekrar görünmez.** Aynı kavram için aynı metin iki kez söylenmez; tüm çerçeveler
  tükendiyse en az kullanılan bağa döner, ama cümleyi yeniden kurar.

Bu yüzden Nora'nın tanımı "doğru ama hep aynı" değil: **doğru ve her seferinde yeniden
söylenmiş** olur.
"""

from __future__ import annotations

import re
from collections import Counter, deque

from nora_lang import Proposition, tr_capitalize, tr_lower


def _verbal_to_inf(obj: str) -> str:
    """"susuzluk giderme" -> "susuzluk gidermek", "bilgi aktarma" -> "bilgi aktarmak"."""
    o = (obj or "").strip()
    last = o.split()[-1] if o else ""
    if last.endswith("ma") and len(last) > 4:
        return o[: -len(last)] + last[:-2] + "mak"
    if last.endswith("me") and len(last) > 4:
        return o[: -len(last)] + last[:-2] + "mek"
    return o


def _drop_subject(text: str, subj: str) -> str:
    """Türkçe özne düşürür: 'su susuzluk gidermek için kullanılır' -> 'susuzluk gidermek için kullanılır'."""
    t = (text or "").strip()
    s = (subj or "").strip()
    if s and t.lower().startswith(s.lower() + " "):
        return t[len(s) + 1:]
    return t


class Narrator:
    """Aynı bilgiyi her seferinde başka türlü anlatan organ. Tur başına O(bağ sayısı)."""

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self.turn = 0
        self.seen: dict[str, deque[str]] = {}      # kavram -> söylenen metinler (son 12)
        self.count = Counter()                      # kavram -> kaç kez anlatıldı
        self.used_edges: dict[str, Counter] = {}    # kavram -> bağ kullanım sayısı
        self.last_text: str = ""

    # ------------------------------------------------------------------ iç yardımcılar
    def _remember(self, key: str, text: str) -> str:
        d = self.seen.setdefault(key, deque(maxlen=12))
        d.append(tr_lower(text))
        self.count[key] += 1
        self.last_text = text
        return text

    def _already_said(self, key: str, text: str) -> bool:
        return tr_lower(text) in self.seen.get(key, ())

    def _edge_score(self, key: str, rel: str, obj: str) -> int:
        return self.used_edges.setdefault(key, Counter())[(rel, obj or "")]

    def _use_edge(self, key: str, rel: str, obj: str) -> None:
        self.used_edges.setdefault(key, Counter())[(rel, obj or "")] += 1

    # ------------------------------------------------------------------ ana giriş
    def render(self, agi, subj: str, edges: list[tuple[str, str]], core_text: str,
               *, speaker: str = "", force: bool = False) -> str:
        """Kavram için bu turun anlatımını kurar.

        `core_text` çağıranın (nora_agi) zaten kurduğu çekirdek metindir; ilk tanışmada
        o kullanılır, böylece davranış geriye dönük uyumlu kalır. Sonraki turlarda farklı
        bir çerçeve denenir. `force=True` çekirdeği de atlar (çağıran tekrar istemiyorsa).
        """
        self.turn += 1
        key = tr_lower(subj).strip()
        e = [(str(r), str(o or "")) for r, o in edges if r and str(o or "").strip()]
        if not e:
            return self._remember(key, core_text)

        first_time = self.count[key] == 0
        if first_time and not force and core_text and not self._already_said(key, core_text):
            for rel, obj in e:
                self._use_edge(key, rel, obj)
            return self._remember(key, core_text)

        # Çerçeve sırası her turda kayar: aynı çerçeve üst üste gelmez.
        builders = (self._f_core, self._f_nominal, self._f_single, self._f_heard,
                    self._f_body, self._f_memory, self._f_inference, self._f_order)
        start = self.turn % len(builders)
        order = [builders[(start + i) % len(builders)] for i in range(len(builders))]
        # Çekirdek metin ilk turda kullanıldı; sonraki turlarda en sona kalır.
        for build in order:
            try:
                text = build(agi, subj, key, e)
            except Exception:
                continue
            if not text:
                continue
            text = text.strip()
            if not text.endswith((".", "?", "!")):
                text += "."
            if self._already_said(key, text):
                continue
            return self._remember(key, text)

        # Bütün çerçeveler tükendi: en az kullanılan bağa dön, ama cümleyi yeniden kur.
        rel, obj = min(e, key=lambda ro: self._edge_score(key, ro[0], ro[1]))
        self._use_edge(key, rel, obj)
        tail = tr_lower(_drop_subject(self._one(agi, subj, rel, obj), subj))
        txt = f"Bildiğim bu: {tail}"
        if self._already_said(key, txt):
            # Söyleyecek yeni bir yan kalmadıysa bunu saklamaz: sınırını söyler.
            txt = f"{txt.rstrip('.')}; ötesini henüz bilmiyorum."
        return self._remember(key, txt)

    # ------------------------------------------------------------------ yardımcı cümle
    @staticmethod
    def _one(agi, subj: str, rel: str, obj: str) -> str:
        try:
            return tr_capitalize(agi.lang.broca.realize_prop(
                Proposition(subj=subj, rel=rel, obj=obj))).strip()
        except Exception:
            return f"{tr_capitalize(subj)} — {obj}"

    def _related_edges(self, key: str, e):
        """Önce işlev bağı (USED_FOR), sonra en az kullanılmış bağlar.

        'su susuzluk gidermek için kullanılır' ilk yarı olarak doğal duruyor; nedeni
        ikinci cümleye bırakmak Türkçede daha konuşma diline yakın.
        """
        return sorted(e, key=lambda ro: (0 if ro[0] == "USED_FOR" else 1,
                                         self._edge_score(key, ro[0], ro[1])))

    # ------------------------------------------------------------------ çerçeveler
    def _f_core(self, agi, subj, key, e) -> str | None:
        """Çekirdek: iki bağ, ama sırası ve bağlayıcısı değişir."""
        if len(e) < 2:
            return None
        a, b = self._related_edges(key, e)[:2]
        # Açılış cümlesi öznesiz başlamaz ("Susuzluk gidermek için kullanılır." diye söze
        # girilmez); özne yalnızca İKİNCİ cümlede düşer.
        t1 = self._one(agi, subj, *a)
        t2 = _drop_subject(self._one(agi, subj, *b), subj) if self._edge_score(key, *b) else self._one(agi, subj, *b)
        conn = ("; ", ". Ayrıca ", " — ", ". Bir de ")
        glue = conn[self.turn % len(conn)]
        join_word = glue.strip(" .")
        text = tr_capitalize(f"{t1}{glue}{tr_lower(t2) if join_word in ('Ayrıca', 'Bir de', '—') else t2}")
        for rel, obj in (a, b):
            self._use_edge(key, rel, obj)
        return text if text.endswith((".", "?", "!")) else text + "."

    def _f_nominal(self, agi, subj, key, e) -> str | None:
        """"Suyun işi susuzluk gidermek." — eylemlikle, kısa ve ansiklopedik değil."""
        uses = [(r, o) for r, o in e if r == "USED_FOR"]
        if not uses:
            return None
        rel, obj = min(uses, key=lambda ro: self._edge_score(key, *ro))
        if len(obj.split()) < 2:
            return None
        try:
            gen = agi.lang.morph.inflect_noun(subj, "GEN")
        except Exception:
            gen = subj + "ın"
        self._use_edge(key, rel, obj)
        text = f"{tr_capitalize(gen)} işi {_verbal_to_inf(obj)}."
        # İkinci bir bağ varsa ayrı cümle olarak ekle (aynı kalıba düşmeden).
        others = [(r, o) for r, o in self._related_edges(key, e) if (r, o) != (rel, obj)]
        if others:
            rel2, obj2 = others[0]
            self._use_edge(key, rel2, obj2)
            # İkinci cümlede özne düşer: "Kitabın işi ... . Bilgi ve hayal gücünü aktarır."
            second = _drop_subject(self._one(agi, subj, rel2, obj2), subj)
            text += " " + (tr_capitalize(second) if second else "")
        return text

    def _f_single(self, agi, subj, key, e) -> str | None:
        """Tek bağ: en az kullanılmış olanı, yalın hâlde."""
        rel, obj = self._related_edges(key, e)[0]
        if self._edge_score(key, rel, obj) >= 3:
            return None
        self._use_edge(key, rel, obj)
        return self._one(agi, subj, rel, obj)

    def _f_heard(self, agi, subj, key, e) -> str | None:
        """Bedenle deneyimlenmemiş kavram: 'duydum/biliyorum' dürüstlüğü."""
        try:
            rec = getattr(agi.self_model, "somatic", {}).get(subj)
        except Exception:
            rec = None
        if rec is not None:
            return None
        rel, obj = self._related_edges(key, e)[0]
        self._use_edge(key, rel, obj)
        tail = _drop_subject(self._one(agi, subj, rel, obj), subj)
        frames = (f"{tr_capitalize(subj)} konusunda kendi deneyimim yok; bildiğim şu: {tr_lower(tail)}.",
                  f"{tr_capitalize(subj)} bana anlatılanlardan biliniyor: {tr_lower(tail)}.",
                  f"Bana anlatılanlardan biliyorum: {tr_lower(tail)}.")
        return frames[self.turn % len(frames)]

    def _f_body(self, agi, subj, key, e) -> str | None:
        """Beden sürüşü yüksekse kavram kendi derdine bağlanır (yalnızca bedene değen kavramlar)."""
        if tr_lower(subj) not in ("su", "yiyecek", "besin", "yemek", "yem"):
            return None
        nv = getattr(agi, "nervous", None)
        if nv is None:
            return None
        thirst = float(getattr(nv, "thirst_drive", 0.0) or 0.0)
        hunger = float(getattr(nv, "hunger_drive", 0.0) or 0.0)
        if max(thirst, hunger) <= 0.35:
            return None
        rel, obj = self._related_edges(key, e)[0]
        self._use_edge(key, rel, obj)
        tail = tr_lower(_drop_subject(self._one(agi, subj, rel, obj), subj))
        tail = tr_capitalize(tail) if tail else tail
        if thirst >= hunger:
            return f"Şu an boğazım kuruyor; {tr_lower(subj)} aklımda. {tail}"
        return f"Karnım aç; {tr_lower(subj)} aklımda. {tail}"

    def _f_memory(self, agi, subj, key, e) -> str | None:
        """Yaşanmış kavram: bellek kaydı + bağ."""
        try:
            rec = getattr(agi.self_model, "somatic", {}).get(subj)
        except Exception:
            rec = None
        if rec is None or getattr(rec, "provenance", "") not in ("deneyim", "duyum"):
            return None
        rel, obj = self._related_edges(key, e)[0]
        self._use_edge(key, rel, obj)
        pol = "iyi geldi" if float(getattr(rec, "effective_valence", 0.0) or 0.0) > 0 else "kötü geldi"
        tail = tr_lower(_drop_subject(self._one(agi, subj, rel, obj), subj))
        return f"{tr_capitalize(subj)} bana {pol}; {tail}."

    def _f_inference(self, agi, subj, key, e) -> str | None:
        """Kendi çıkarımı varsa onu ekler (uydurma değil, kendi türetimi)."""
        inf = getattr(agi, "infer", None)
        if inf is None:
            return None
        try:
            line = inf.last_line(agi, topic=subj)
        except Exception:
            line = ""
        if not line:
            return None
        base = self._one(agi, subj, *self._related_edges(key, e)[0])
        self._use_edge(key, *self._related_edges(key, e)[0])
        return f"{base} {line}"

    def _f_order(self, agi, subj, key, e) -> str | None:
        """Aynı bağları ters sırada, tek cümle hâlinde."""
        if len(e) < 2:
            return None
        a, b = self._related_edges(key, e)[:2]
        self._use_edge(key, *a)
        self._use_edge(key, *b)
        t1 = tr_lower(_drop_subject(self._one(agi, subj, *a), subj))
        t2 = tr_lower(_drop_subject(self._one(agi, subj, *b), subj))
        frames = (f"{tr_capitalize(subj)} deyince aklıma iki şey geliyor: {t1}; bir de {t2}.",
                  f"{tr_capitalize(subj)}, iki yönüyle aklımda: {t1}; {t2}.")
        return frames[self.turn % len(frames)]
        return frames[self.turn % len(frames)]

    # ------------------------------------------------------------------ rapor
    def diversity(self) -> dict:
        return {k: v for k, v in self.count.items() if v > 1}

    def state_dict(self) -> dict:
        return {"turn": self.turn,
                "count": dict(self.count),
                "seen": {k: list(v) for k, v in self.seen.items()},
                "edges": {k: {f"{a}|{b}": c for (a, b), c in v.items()}
                          for k, v in self.used_edges.items()}}

    def load_state_dict(self, d: dict) -> None:
        try:
            self.turn = int(d.get("turn", 0) or 0)
            self.count = Counter({str(k): int(v) for k, v in (d.get("count") or {}).items()})
            self.seen = {str(k): deque([str(x) for x in v], maxlen=12)
                         for k, v in (d.get("seen") or {}).items()}
            self.used_edges = {}
            for k, v in (d.get("edges") or {}).items():
                c: Counter = Counter()
                for pair, n in (v or {}).items():
                    rel, _, obj = str(pair).partition("|")
                    c[(rel, obj)] = int(n)
                self.used_edges[str(k)] = c
        except Exception:
            pass


def _selftest() -> None:
    from nora_agi import NoraAGI

    agi = NoraAGI(seed=901, boot_thinker=False)
    outs = [agi.converse("Su nedir?") for _ in range(5)]
    for i, o in enumerate(outs, 1):
        print(f"{i}. {o}")
    uniq = len({tr_lower(o) for o in outs})
    print(f"farklı anlatım: {uniq}/{len(outs)}")


if __name__ == "__main__":
    _selftest()
