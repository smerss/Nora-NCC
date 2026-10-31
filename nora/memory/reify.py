"""nora_reify.py — YENİDEN-ÖRNEKLEME: tekrar eden bileşimi ADLANDIRIP ilkel yap.

Neden bu organ
--------------
`nora_synth` her seferinde sıfırdan program kuruyordu: aynı `ters çevir → ilk 3
harfi al` bileşimi on kez keşfedilse de on kez "iki adım" olarak kalıyordu. İnsan
matematiğinde lemma tam bu işi yapar: tekrar eden bir bileşim AD alır, sonra o ad
İLKEL gibi kullanılır; bir sonraki ispat onun üstüne kurulur.

    seviye 1  kompozisyon      : yol bulmak
    seviye 2  örnekleme        : verili formlardan parametre seçmek
    seviye 3  YENİDEN-ÖRNEKLEME: bileşimi ilkel yapmak        <-- burası
    seviye 4  form icadı       : repertuvarda olmayan yasa biçimi

Kurallar (enflasyonu engelleyen kısım)
--------------------------------------
1. **Eşik**: aynı bileşim en az `K` kez BAŞARILI sentezde görülmeli.
2. **Ad**: addan türetilir, cümle değil (`kendi.bilesim.3`); açıklaması adımlardan.
3. **Kullanım ölçülür**: reify edilen her ilkelin `uses` sayacı vardır. Hiç
   kullanılmayan ilkel **budanır** (`MAX_IDLE` karar sonra). Ölçülen şey ilkel
   SAYISI değil, HAYATTA KALAN (tekrar kullanılan) sayısıdır.
4. **Kalıcılık**: defter `save`/`load` ile taşınır; yeniden kurulamayan kayıt yüklenmez.
"""
from __future__ import annotations

from nora._paths import REPO_ROOT



class Reify:
    """Bileşim izlerini biriktirip tekrar edeni ilkele çeviren ocak."""

    K = 2          # kaç kez görülünce ilkel olur
    MAX_IDLE = 4   # kaç kullanımsız karardan sonra budanır

    def __init__(self):
        self.counts: dict[tuple, dict] = {}   # spec -> {"n", "examples"}
        self.ledger: list[dict] = []          # birinci sınıf ilkeller
        self.promoted = 0
        self.pruned = 0

    # ------------------------------------------------------------------ izleme
    def observe(self, spec, examples=None) -> dict | None:
        """Başarılı bir sentezin adımlarını yaz. Tek adımlılar ilgi alanı dışı."""
        spec = tuple(str(x) for x in (spec or []) if str(x))
        if len(spec) < 2:
            return None
        rec = self.counts.setdefault(spec, {"n": 0, "examples": []})
        rec["n"] = int(rec["n"]) + 1
        for ex in (examples or [])[:4]:
            try:
                pair = [str(ex[0]), str(ex[1])]
            except Exception:
                continue
            if pair not in rec["examples"]:
                rec["examples"].append(pair)
        return rec

    def candidates(self, k: int | None = None) -> list[tuple]:
        """Eşiği geçmiş, henüz ilkel olmamış bileşimler (çoktan aza)."""
        k = self.K if k is None else int(k)
        have = {tuple(r["spec"]) for r in self.ledger}
        out = [(spec, int(rec["n"])) for spec, rec in self.counts.items()
               if int(rec["n"]) >= k and spec not in have]
        out.sort(key=lambda x: -x[1])
        return [s for s, _n in out]

    # ------------------------------------------------------------------ terfi
    def promote(self, spec, build_fn=None, desc: str = "") -> dict | None:
        """Bileşimi birinci sınıf ilkel yap. `build_fn` verilirse fonksiyon kurulur."""
        spec = tuple(str(x) for x in (spec or []))
        if len(spec) < 2 or any(tuple(r["spec"]) == spec for r in self.ledger):
            return None
        rec = self.counts.get(spec) or {}
        self.promoted += 1
        item = {"name": f"kendi.bilesim.{self.promoted}", "spec": list(spec),
                "desc": desc or " → ".join(spec), "origin": "reify",
                "uses": 0, "idle": 0, "alive": True,
                "evidence": int(rec.get("n", 0)),
                "examples": list((rec.get("examples") or [])[:4])}
        self.ledger.append(item)
        if build_fn is not None:
            item["fn"] = build_fn
        return item

    def alive(self) -> list[dict]:
        return [r for r in self.ledger if r.get("alive", True)]

    def names(self) -> list[str]:
        return [str(r["name"]) for r in self.alive()]

    # ------------------------------------------------------------------ kullanım
    def note_use(self, name: str) -> None:
        for r in self.ledger:
            if str(r["name"]) == str(name):
                r["uses"] = int(r.get("uses", 0)) + 1
                r["idle"] = 0
                return

    def note_spec_use(self, spec) -> None:
        """Bir programın adımlarında geçen reify ilkellerini işaretle."""
        for nm in (spec or []):
            self.note_use(str(nm))

    def tick(self) -> list[str]:
        """Bir karar geçti: kullanılmayan ilkellerin `idle` sayacı artar; sınırı
        geçen budanır (hayatta kalma oranı buradan çıkar)."""
        pruned = []
        for r in self.ledger:
            if not r.get("alive", True):
                continue
            r["idle"] = int(r.get("idle", 0)) + 1
            if int(r["idle"]) > self.MAX_IDLE and int(r.get("uses", 0)) == 0:
                r["alive"] = False
                self.pruned += 1
                pruned.append(str(r["name"]))
        return pruned

    def survival(self) -> dict:
        alive = self.alive()
        used = [r for r in alive if int(r.get("uses", 0)) > 0]
        total = len(self.ledger)
        return {"total": total, "alive": len(alive), "used": len(used),
                "survival": (len(used) / total) if total else 1.0,
                "pruned": self.pruned}

    # ------------------------------------------------------------ kalıcılık
    def state(self) -> dict:
        return {"promoted": self.promoted, "pruned": self.pruned,
                "counts": [{"spec": list(s), "n": int(r.get("n", 0))}
                           for s, r in self.counts.items()][-48:],
                "ledger": [{k: v for k, v in r.items() if k != "fn"}
                           for r in self.ledger[-48:]]}

    def load_state(self, d: dict | None) -> None:
        d = d or {}
        self.promoted = int(d.get("promoted", 0) or 0)
        self.pruned = int(d.get("pruned", 0) or 0)
        self.counts = {}
        for rec in (d.get("counts") or []):
            spec = tuple(str(x) for x in (rec.get("spec") or []))
            if len(spec) >= 2:
                self.counts[spec] = {"n": int(rec.get("n", 0)), "examples": []}
        self.ledger = []
        for rec in (d.get("ledger") or []):
            if not isinstance(rec, dict) or not rec.get("spec"):
                continue
            item = {k: v for k, v in rec.items() if k != "fn"}
            item.setdefault("uses", 0)
            item.setdefault("idle", 0)
            item.setdefault("alive", True)
            self.ledger.append(item)
