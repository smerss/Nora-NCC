"""nora_provenance.py — SÖZÜN KAYNAĞI: iskelet mi, bilgi mi, kendi öğrenmesi mi?

Neden?
------
"Nora ansiklopedi gibi konuşuyor" şikâyetinin ölçülebilir bir karşılığı yoktu. Bir cümle
kavram ağından mı (gerçek bilgi), bellekten mi (yaşanmış), kendi çıkarımından mı, yoksa
tasarımcının yazdığı bir kalıptan mı geldi — bunu kimse saymıyordu.

Bu organ her söz PARÇASINI doğduğu yere göre etiketler:

    ag       kavram ağındaki bağ (broca ile çekimlenmiş gerçek bilgi)
    bellek   yaşanmış kayıt / hatırlanan söz
    cikarim  kendi türettiği bağ
    beden    iç-duyum, açlık/susuzluk, ölçülen zaman
    duyum    kulaktan gelen ses
    kalip    doğuştan eğilim (prior) — yani İSKELET
    ogrenim  doğuştan eğilimi geçmiş ÖĞRENİLMİŞ ağırlık (artık iskelet değil)

`iskelet_orani = kalip / (tüm parçalar)`. Faz 1'de bu oran yüksektir; sönümleme
(prior_strength -> 0) ve pekiştirme ilerledikçe düşer. Nora'nın çocukluğunun ölçüsü budur.
"""
from __future__ import annotations

from collections import Counter

SOURCES = ("ag", "bellek", "cikarim", "beden", "duyum", "kalip", "ogrenim")

LABELS = {
    "ag": "kavram ağı",
    "bellek": "bellek",
    "cikarim": "kendi çıkarımı",
    "beden": "beden/ölçüm",
    "duyum": "duyum",
    "kalip": "doğuştan eğilim",
    "ogrenim": "öğrenilmiş söz",
}


class Provenance:
    """Söz parçalarının kaynağını sayan ölçer. O(1) kayıt, sabit bellek."""

    def __init__(self):
        self.c: Counter = Counter()
        self.turns = 0
        self.tagged_turns = 0
        self.last = ""

    # ------------------------------------------------------------------ kayıt
    def record(self, source: str, k: int = 1) -> None:
        s = str(source)
        if s not in SOURCES:
            s = "kalip"
        self.c[s] += int(k)
        self.last = s

    def record_turn(self, *sources: str) -> None:
        """Bir turun tamamı: verilen kaynakların her biri 1 parça sayılır."""
        for s in sources:
            self.record(s, 1)
        self.turns += 1
        self.tagged_turns += 1

    # ------------------------------------------------------------------ ölçüm
    @property
    def total(self) -> int:
        return int(sum(self.c.values()))

    @property
    def skeleton_rate(self) -> float:
        """Doğuştan eğilimden (kalıptan) gelen parçaların oranı."""
        t = self.total
        return float(self.c.get("kalip", 0)) / float(t) if t else 0.0

    @property
    def learned_rate(self) -> float:
        t = self.total
        return float(self.c.get("ogrenim", 0)) / float(t) if t else 0.0

    def report(self) -> dict:
        d = {s: int(self.c.get(s, 0)) for s in SOURCES}
        d.update({
            "toplam": self.total,
            "tur": int(self.turns),
            "kapsanan_tur": int(self.tagged_turns),
            "iskelet_orani": round(self.skeleton_rate, 4),
            "ogrenim_orani": round(self.learned_rate, 4),
        })
        return d

    def describe(self) -> str:
        r = self.report()
        parts = [f"{LABELS[s]}={r[s]}" for s in SOURCES if r[s]]
        return (f"iskelet oranı {r['iskelet_orani']:.2f} | öğrenim {r['ogrenim_orani']:.2f} | "
                + ", ".join(parts))

    # ------------------------------------------------------------------ kalıcılık
    def state_dict(self) -> dict:
        return {"c": dict(self.c), "turns": int(self.turns), "tagged": int(self.tagged_turns)}

    def load_state_dict(self, d: dict | None) -> None:
        if not isinstance(d, dict):
            return
        try:
            self.c = Counter({str(k): int(v) for k, v in (d.get("c") or {}).items()
                              if str(k) in SOURCES})
            self.turns = int(d.get("turns", 0) or 0)
            self.tagged_turns = int(d.get("tagged", 0) or 0)
        except Exception:
            pass


#: Süreç geneli varsayılan ölçer (AGI nesnesi yokken compose_* bunu kullanır).
_DEFAULT: Provenance | None = None


def default_meter() -> Provenance:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Provenance()
    return _DEFAULT


def _selftest() -> None:
    p = Provenance()
    for s in ("ag", "ag", "kalip", "bellek"):
        p.record(s)
    p.record_turn("ag", "beden")
    print(p.describe())
    print(p.report())


if __name__ == "__main__":
    _selftest()
