"""nora_drive.py — SÜRÜCÜ: talimat olmadan, şaşkınlığa göre kendi öğrenmesini tetikler.

Neden bu organ
--------------
Bugüne kadar tetikleyen insandı: "şu dönüşümü öğren", "şu örneklerden kural çıkar".
Nora cevap veriyordu ama **sürücüsü yoktu**. Bu organ akışı kendi izler, şaşkınlığı
ölçer, kümeler ve en yüksek şaşkınlıkta kendi kendine öğrenme çağırır.

    sürücüsüz sistem : araç  (sen tetiklersin, o cevaplar)
    sürücülü sistem  : ajan  (kendi tetikler)

Kararlılık (fren)
-----------------
Her şaşkınlık öğrenilmeye değmez; gürültü kovalanırsa sistem her cümleye bir kural
uydurur. Kural: **yalnız akışın betimleme uzunluğunu KISALTAN** öğrenme kabul edilir
(MDL). Bir küme en az `MIN_ITEMS` maddede aynı dönüşümü gösteriyorsa, tek adım/tek ad
`MIN_ITEMS` sürümden kısadır; aksi hâlde öğrenme yoktur.

Dürüstlük
---------
1. Şaşkınlık ölçümü veriden: dönüşüm çiftleri, bilinmeyen kavramlar, sayı olguları.
2. Öğrenme iddiası yalnız **doğrulanmış** kayıtla: `synth` programı benimsedi ya da
   `abstract` ilkel önerdi ve örneklerin hepsi tuttu.
3. Hiçbir şey öğrenilmediyse "öğrenmedim" denir; sürücü boş rapor uydurmaz.
"""
from __future__ import annotations

from nora_lang import tr_lower


class Driver:
    """Ham akışı izleyen, şaşkınlığı ölçen ve öğrenmeyi KENDİ tetikleyen sürücü."""

    MIN_ITEMS = 2          # bir kümenin öğrenme sayılması için en az madde
    MIN_SURPRISE = 0.55    # bu eşiğin altındaki şaşkınlık öğrenme tetiklemez
    MAX_ITEMS = 512

    def __init__(self, agi=None):
        self.agi = agi
        self.items: list[dict] = []      # {"text","kind","surprise","key"}
        self.seen = 0
        self.triggered = 0               # KENDİ tetiklediği öğrenme sayısı
        self.learned: list[dict] = []    # {"kind","name","n","keys"}
        self.rejected = 0                # MDL freni: kazancı olmayan öneriler
        self.unresolved = 0              # kanıt yetmedi (belirsiz/bilinmeyen) — sessiz kalır
        #: EŞİK ARTIK SABİT DEĞİL: kendi isabetini ölçüp ayarlar (irade).
        self.thresh = float(self.MIN_SURPRISE)
        #: Kanıt eşiği: kaç maddelik küme öğrenmeye değer. Gürültüde artar.
        self.min_items = int(self.MIN_ITEMS)
        self._specs: dict[tuple, int] = {}
        self._run = {"trig": 0, "unres": 0, "reuse": 0}

    # ------------------------------------------------------------------ algı
    def ingest(self, text: str) -> dict:
        """Bir cümleyi izle: şaşkınlığını ölç ve kaydet."""
        raw = str(text or "").strip()
        self.seen += 1
        rec = {"text": raw, "kind": "noise", "surprise": 0.10, "key": ()}
        if not raw:
            self.items.append(rec)
            return rec
        try:
            from nora_synth import parse_pairs
            pairs = parse_pairs(raw)
        except Exception:
            pairs = []
        if pairs:
            a, b = pairs[0]
            explained = self._explained(a, b)
            rec = {"text": raw, "kind": "pair", "pairs": pairs,
                   # BİÇİM anahtarı: uzunluk farkını DIŞARIDA bırakır. Aynı bileşim
                   # farklı uzunluklarda görünse de tek kümede toplansın; uzunluk
                   # çeşitliliği belirsizliği ÇÖZEN kanıttır, kümeyi bölen değil.
                   "key": (len(a) == len(b), sorted(a) == sorted(b)),
                   "surprise": 0.25 if explained else 1.0}
        else:
            low = tr_lower(raw)
            if " nedir" in low or " ne demek" in low:
                rec = {"text": raw, "kind": "question", "surprise": 0.35, "key": ()}
            elif any(ch.isdigit() for ch in raw):
                rec = {"text": raw, "kind": "number", "surprise": 0.45, "key": ()}
        self.items.append(rec)
        if len(self.items) > self.MAX_ITEMS:
            del self.items[:-self.MAX_ITEMS]
        return rec

    def ingest_stream(self, lines) -> int:
        for ln in (lines or []):
            self.ingest(ln)
        return self.seen

    def _explained(self, a: str, b: str) -> bool:
        """Bu çift şu an HALLEDİLİYOR mu? (kayıtlı programlardan biri uygulanabiliyorsa)"""
        agi = self.agi
        sy = getattr(agi, "synth", None) if agi is not None else None
        if sy is None:
            return False
        for rec in (sy.programs or [])[-8:]:
            try:
                p = sy.rebuild(rec.get("id"))
            except Exception:
                p = None
            if p is not None and p.apply(a) == b:
                return True
        return False

    # ------------------------------------------------------------------ kümeler
    def clusters(self) -> list[dict]:
        """Aynı biçimi paylaşan yüksek-şaşkınlık maddelerini grupla."""
        out: dict[tuple, dict] = {}
        for it in self.items:
            if it.get("kind") != "pair" or it.get("consumed"):
                continue
            if float(it.get("surprise", 0)) < self.thresh:
                continue
            key = it.get("key") or ()
            c = out.setdefault(key, {"key": key, "pairs": [], "surprise": 0.0})
            for pr in it.get("pairs") or []:
                if pr not in c["pairs"]:
                    c["pairs"].append(pr)
            c["surprise"] = max(c["surprise"], float(it["surprise"]))
        res = list(out.values())
        res.sort(key=lambda c: -(c["surprise"] * len(c["pairs"])))
        return res

    # ------------------------------------------------------------------ karar
    def step(self) -> dict | None:
        """En yüksek şaşkınlıklı kümeyi öğren. MDL freni burada."""
        agg = self.agi
        if agg is None:
            return None
        best = None
        for c in self.clusters():
            pairs = [tuple(p) for p in c["pairs"]]
            if len(pairs) < self.min_items:
                continue          # yetersiz kanıt: ne öğrenme ne ret sayılır
            best = (c, pairs)
            break
        if best is None:
            return None
        c, pairs = best
        sy = getattr(agg, "synth", None)
        # 1) mevcut uzay bu kümeyi çözüyor mu?
        try:
            res = sy.learn(pairs) if sy is not None else {"status": "insufficient"}
        except Exception:
            res = {"status": "insufficient"}
        if res.get("status") in ("ambiguous", "insufficient"):
            # Kanıt yetmedi: uydurma yok. Küme TÜKETİLMEZ; akış devam ederse yeniden
            # denenir. "Ret" sayılmaz: ret, denenip başarısız olan öneriye aittir.
            self.unresolved += 1
            self._run["unres"] += 1
            return None
        if res.get("status") == "adopted":
            self._consume(c)
            self.triggered += 1
            spec = tuple(getattr(res.get("program"), "spec", lambda: [])())
            self._run["reuse"] += int(self._specs.get(spec, 0) > 0)
            self._specs[spec] = self._specs.get(spec, 0) + 1
            self._run["trig"] += 1
            rec = {"kind": "kompozisyon", "name": res["program"].desc(),
                   "n": len(pairs), "keys": list(c["key"])}
            self.learned.append(rec)
            if hasattr(agg, "_after_synth"):
                try:
                    agg._after_synth(res["program"])
                except Exception:
                    pass
            return rec
        # 2) uzay yetmiyorsa: soyutlama (icat) — sürücü kendi çağırır
        ab = getattr(agg, "abstract", None)
        if ab is None:
            return None
        try:
            prop = ab.propose(pairs, composition_exists=False,
                              register=(lambda nm, fn: sy.add_primitive(nm, fn)) if sy else None)
        except Exception:
            prop = {"status": "insufficient"}
        if prop.get("status") != "invented":
            self.rejected += 1
            return None
        # MDL FRENİ: icat en az MIN_ITEMS maddeyi tek adla açıklamalı.
        fn = None
        try:
            from nora_abstract import build as _build
            fn = _build(prop["family"], prop["params"])
        except Exception:
            fn = None
        if fn is None or len(pairs) < self.MIN_ITEMS:
            self.rejected += 1
            return None
        if not all(fn(a) == b for a, b in pairs):
            self.rejected += 1
            return None
        self._consume(c)
        self.triggered += 1
        rec = {"kind": "icat", "name": prop["name"], "n": len(pairs),
               "keys": list(c["key"])}
        self.learned.append(rec)
        return rec

    def _consume(self, c: dict) -> None:
        """Öğrenilen kümeyi tüket: aynı şaşkınlık ikinci kez öğrenilmez."""
        key = c.get("key")
        for it in self.items:
            if it.get("kind") == "pair" and (it.get("key") or None) == key:
                it["consumed"] = True

    def run(self, max_steps: int = 6) -> list[dict]:
        """Akışı kendi kendine işle, sonunda eşiği KENDİ isabetine göre ayarla."""
        self._run = {"trig": 0, "unres": 0, "reuse": 0}
        out = []
        for _ in range(int(max_steps)):
            rec = self.step()
            if rec is None:
                break
            out.append(rec)
        self.adapt()
        return out

    def adapt(self) -> float:
        """Kendi isabetini ölç, eşiği oynat. Eşik artık insanın sabiti değil.

        - **Kanıt yetmiyor** (unresolved baskın): eşik DÜŞER — daha çok kanıt topla.
          (Gürültüye tetikleme riski, kanıtsızlıkta susmaktan iyidir; ama sınırlı.)
        - **Öğrendiği tekrar kullanılmıyor** (reuse=0): eşik YÜKSELİR — bu akışta
          öğrenilecek bir şey yok, boşuna tetikleme.
        """
        r = self._run
        if r["unres"] > max(1, 2 * r["trig"]):
            self.thresh = max(0.15, self.thresh * 0.90)
        elif r["trig"] > 0 and r["reuse"] == 0:
            self.thresh = min(0.95, self.thresh * 1.10)
            # Tekrar kullanılmayan öğrenme = kanıt zayıf: daha çok madde iste.
            self.min_items = min(4, self.min_items + 1)
        return self.thresh

    # ------------------------------------------------------------------ rapor
    def summary(self) -> dict:
        kinds = {}
        for r in self.learned:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        # DÜRÜST SAYI: aynı program beş kümede beş kez bulunsa da KEŞİF birdir.
        # Ölçü "kaç kez tetiklendi" değil, "kaç AYRI şey öğrenildi" olmalı.
        names = [r["name"] for r in self.learned]
        return {"seen": self.seen, "triggered": self.triggered,
                "distinct": len(set(names)), "rejected": self.rejected,
                "unresolved": self.unresolved, "thresh": round(self.thresh, 3),
                "min_items": int(self.min_items),
                "kinds": kinds, "names": names[-6:]}

    def state(self) -> dict:
        return {"seen": self.seen, "triggered": self.triggered,
                "rejected": self.rejected, "unresolved": self.unresolved,
                "thresh": round(self.thresh, 3), "min_items": int(self.min_items),
                "learned": self.learned[-24:]}

    def load_state(self, d: dict | None) -> None:
        d = d or {}
        self.seen = int(d.get("seen", 0) or 0)
        self.triggered = int(d.get("triggered", 0) or 0)
        self.rejected = int(d.get("rejected", 0) or 0)
        self.unresolved = int(d.get("unresolved", 0) or 0)
        try:
            self.thresh = float(d.get("thresh", self.MIN_SURPRISE) or self.MIN_SURPRISE)
            self.min_items = int(d.get("min_items", self.MIN_ITEMS) or self.MIN_ITEMS)
        except Exception:
            self.thresh = float(self.MIN_SURPRISE)
        self.learned = [r for r in (d.get("learned") or []) if isinstance(r, dict)]
