"""nora_schema.py — ŞEMA: makroda bir delik soyutla, aileyi tarafla.

Merdiven
--------
    3  makro (reify) : sırala∘dilimle(3)          → aynı davranışın kısa adı
    4  ŞEMA          : sırala∘dilimle(n)          → İNSANIN SAYMADIĞI üyeler
    5  taban uzantısı: Nora önerir, insan onaylar
    6  kendini değiştirme yetkisi

Makro sıkıştırır, genişletmez. Şema genişletir: `n=3` ve `n=4` görülmüşse
`sırala∘dilimle(n)` şeması `n=7`yi de kapsar — ve `n=7` **hiçbir ilkelde yok**.
Yeni erişimin ölçülebilir tanımı budur.

Kurallar (şema mı, makro mu)
----------------------------
1. En az **2 farklı görülmüş parametre** (tek üyeli aile şema değil makrodur).
2. Delik **tek** bileşende; ailenin üreticisi `nora_synth.PARAM_GENERATORS`ta var.
3. Betimleme **kısalmalı**: şema 1 adım sayılır; somut makro 2+ adım.
4. Yeni form yazılmaz: şema bir *organ*, form değil. Repertuvar donuk kalır.
"""
from __future__ import annotations

from nora_synth import family_of


class Schema:
    """Delikli makroları bulan ve uzaya kaydeden ocak."""

    MIN_SEEN = 2

    MAX_IDLE = 4      # kullanılmayan şema kaç karar sonra budanır

    def __init__(self):
        self.ledger: list[dict] = []
        self.created = 0
        self.pruned = 0

    # ------------------------------------------------------------------ çıkarım
    def infer(self, specs) -> list[dict]:
        """Adım dizilerinden delikli şablon çıkar (aynı her şey + değişen tek aile)."""
        groups: dict[tuple, dict] = {}
        for spec in specs or []:
            spec = [str(x) for x in spec]
            if len(spec) < 2:
                continue
            for i, nm in enumerate(spec):
                fo = family_of(nm)
                if fo is None:
                    continue
                fam, n = fo
                key = (tuple(spec[:i]), tuple(spec[i + 1:]), fam, i)
                g = groups.setdefault(key, {"steps": spec, "hole": i, "family": fam,
                                            "seen": set()})
                g["seen"].add(int(n))
        out = []
        for g in groups.values():
            if len(g["seen"]) >= self.MIN_SEEN:
                out.append({"steps": list(g["steps"]), "hole": int(g["hole"]),
                            "family": g["family"], "seen": sorted(g["seen"])})
        return out

    def register(self, sc: dict, synth) -> dict | None:
        """Şemayı kalıcı deftere ve sentez uzayına yaz."""
        if not sc or len(sc.get("seen") or []) < self.MIN_SEEN:
            return None
        if any(r["steps"] == sc["steps"] and r["hole"] == sc["hole"]
               and r["family"] == sc["family"] for r in self.ledger):
            return None
        item = {"steps": list(sc["steps"]), "hole": int(sc["hole"]),
                "family": sc["family"], "seen": list(sc["seen"]),
                "name": f"şema.{self.created + 1}", "uses": 0, "idle": 0,
                "alive": True}
        self.created += 1
        self.ledger.append(item)
        if synth is not None:
            synth.add_schema(item["steps"], item["hole"], item["family"], item["seen"])
        return item

    # ------------------------------------------------------------------ kullanım
    def note_use(self, family: str) -> None:
        for r in self.ledger:
            if r["family"] == str(family) and r.get("alive", True):
                r["uses"] = int(r.get("uses", 0)) + 1
                r["idle"] = 0
                return

    def tick(self) -> list[str]:
        """Bir karar geçti: kullanılmayan şema sayaç artırır, sınırı geçen BUDANIR.

        Şema enflasyonu da enflasyondur: iki parametre gördüm diye on şema açmak
        betimlemeyi KISALTMAZ. Ölçü yazılan şema değil, kullanılan şemadır.
        """
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
        alive = [r for r in self.ledger if r.get("alive", True)]
        used = [r for r in alive if int(r.get("uses", 0)) > 0]
        total = len(self.ledger)
        return {"total": total, "alive": len(alive), "used": len(used),
                "pruned": self.pruned,
                "survival": (len(used) / total) if total else 1.0}

    @staticmethod
    def covers(item: dict, n: int) -> bool:
        """Şema bu parametreyi kapsıyor mu? (görülmüş üyeler dahil)"""
        return int(n) >= 1

    def state(self) -> dict:
        return {"created": self.created, "pruned": self.pruned,
                "ledger": self.ledger[-16:]}

    def load_state(self, d, synth=None) -> None:
        d = d or {}
        self.created = int(d.get("created", 0) or 0)
        self.pruned = int(d.get("pruned", 0) or 0)
        self.ledger = [r for r in (d.get("ledger") or []) if isinstance(r, dict)]
        if synth is not None:
            for r in self.ledger:
                if not r.get("alive", True):
                    continue
                synth.add_schema(r.get("steps") or [], int(r.get("hole", 0)),
                                 r.get("family") or "", r.get("seen") or [])
