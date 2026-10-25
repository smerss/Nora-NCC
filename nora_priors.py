"""nora_priors.py — sözün İSKELETİ koddan çıkar: doğuştan motor toplulukları + öğrenilen sinaps.

Neden bu organ var?
-------------------
Nora'nın bazı cümleleri kodun içine gömülü dizgelerdi:

    "Belki de ben seni yeterince dinlemedim; neyi kaçırdım?"
    "Israrın bir şeyi değiştirdi mi?"

Bu, öğrenilmiş bir söz değil YAZILMIŞ bir sözdür. Kullanıcı haklı olarak "sonunda bu
iskeletler kodda görünmemeli" dedi. Çözüm metni silmek değil (o zaman Nora susar):
metni **veriye** taşımak ve seçimi **kuralla** değil **nöral yarışla** yapmak.

Her cümle bir MOTOR HÜCRE TOPLULUĞU (motor assembly) olarak yaşar:

    aktivasyon = w            (öğrenilmiş sinaptik ağırlık, beyinde büyür)
               + ps · bias    (doğuştan eğilim; ps = prior_strength, zamanla söner)
               + gürültü      (sistem entropisi)
               - tazelik      (az önce konuşan topluluk kolay ateşlenmez)

Kazanan-tümü-alır: en yüksek aktivasyonlu topluluk konuşur. Yani kodda "hangi cümle"
yok; kodda yalnızca **yarış** var, cümleler `nora_priors.json` içinde.

Üç katman
---------
1. `MotorAssembly` — tek bir topluluk: metin şablonu + uygunluk koşulları + ağırlıklar.
2. `PriorField`    — bir *alan*: topluluklar, yarış, tazelik, öğrenme, sönümleme (annealing).
3. Öğrenme sinyali — dışarıdan dayatılmaz, Nora'nın kendi sonucundan gelir:
     * aynı söz hemen tekrar geldiyse -> o turda konuşan topluluklar "işe yaramadı",
       ağırlıkları DÜŞER (ceza),
     * konuşma ilerlediyse (tekrar yok) -> ağırlıkları YÜKSELİR (pekiştirme).

Böylece zamanla cümleyi seçen şey doğuştan eğilim değil, Nora'nın kendi yaşantısı olur.
`prior_strength` sıfıra indiğinde doğuştan eğilim tamamen susar: kalan her şey öğrenilmiş
ağırlıktır. İşte "iskeletin koddan silinmesi" bu demektir.
"""
from __future__ import annotations

import json
import os
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

JSON_PATH = Path(__file__).with_name("nora_priors.json")

#: Doğuştan eğilimin tamamen sönmesi için gereken seçim sayısı.
#: 1 tur ≈ 1-3 seçim; 600 ≈ birkaç yüz turlük bir çocukluk.
ANNEAL_PICKS = 600

#: Öğrenme adımları (kendi sonucundan gelen pekiştirme/ceza).
REWARD = 0.06
PUNISH = 0.30
W_MIN, W_MAX = -1.50, 2.50

#: Tazelik: son N seçimde konuşan topluluk bu kadar aktivasyon kaybeder.
FRESH_WINDOW = 3
FRESH_PENALTY = 1.40


def _safe_format(template: str, slots: dict) -> str:
    """Eksik slotta çökmeyen biçimlendirme (None -> '')."""
    class _S(dict):
        def __missing__(self, k):  # noqa: D401 - sözlük sözleşmesi
            return ""
    return str(template).format_map(_S(slots or {}))


@dataclass
class MotorAssembly:
    """Tek bir söz topluluğu: metin + ne zaman uygun olduğu + sinaptik ağırlıklar."""

    id: str
    kind: str
    text: str
    bias: float = 1.0        # doğuştan eğilim (veriden gelir, söner)
    w: float = 0.0           # öğrenilmiş sinaps (beyinde büyür)
    when: dict = field(default_factory=dict)
    need: list = field(default_factory=list)
    uses: int = 0
    wins: int = 0
    losses: int = 0

    # ------------------------------------------------------------------ uygunluk
    def matches(self, ctx: dict, slots: dict) -> bool:
        c = ctx or {}
        for key, want in (self.when or {}).items():
            if key == "gap_lt":
                if not (float(c.get("gap", 0.0)) < float(want)):
                    return False
            elif key == "gap_ge":
                if not (float(c.get("gap", 0.0)) >= float(want)):
                    return False
            elif key == "same":
                if bool(c.get("same", False)) != bool(want):
                    return False
            elif key == "count_min":
                if not (int(c.get("count", 0)) >= int(want)):
                    return False
            elif key == "count_max":
                if not (int(c.get("count", 0)) <= int(want)):
                    return False
            elif key == "turns_gt":
                if not (int(c.get("turns", 0)) > int(want)):
                    return False
            elif key == "turns_le":
                if not (int(c.get("turns", 0)) <= int(want)):
                    return False
            else:
                # Bilinmeyen koşul: temkinli davran, topluluk yarışa girmez.
                return False
        for k in (self.need or []):
            if not str((slots or {}).get(k, "") or "").strip():
                return False
        return True

    def render(self, slots: dict) -> str:
        return _safe_format(self.text, slots)

    def to_dict(self) -> dict:
        return {"w": round(float(self.w), 6), "uses": int(self.uses),
                "wins": int(self.wins), "losses": int(self.losses)}


class PriorField:
    """Motor toplulukları alanı: yarış, tazelik, öğrenme ve sönümleme."""

    def __init__(self, path: str | os.PathLike | None = None, entropy=None, rng=None):
        self.path = str(path or JSON_PATH)
        self.entropy = entropy
        self.rng = rng
        self.kinds: dict[str, list[MotorAssembly]] = {}
        self._by_id: dict[str, MotorAssembly] = {}
        self.picks = 0
        self.turn_ids: list[str] = []
        self.last_turn_ids: list[str] = []
        self.recent: dict[str, deque] = {}
        self.log: deque = deque(maxlen=64)
        self.load()

    # ------------------------------------------------------------------ veri
    def load(self) -> bool:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                blob = json.load(f)
        except Exception:
            self.kinds, self._by_id = {}, {}
            return False
        self.kinds, self._by_id = {}, {}
        for kind, spec in (blob.get("kinds") or {}).items():
            items = []
            for p in (spec.get("priors") or []):
                a = MotorAssembly(
                    id=str(p.get("id") or f"{kind}.{len(items)}"),
                    kind=str(kind),
                    text=str(p.get("text", "")),
                    bias=float(p.get("bias", 1.0)),
                    when=dict(p.get("when") or {}),
                    need=list(p.get("need") or []),
                )
                items.append(a)
                self._by_id[a.id] = a
            if items:
                self.kinds[str(kind)] = items
        return bool(self.kinds)

    # ------------------------------------------------------------------ sönümleme
    @property
    def prior_strength(self) -> float:
        """Doğuştan eğilimin bugünkü gücü: 1.0 (doğum) -> 0.0 (öğrenilmiş zihin)."""
        return max(0.0, 1.0 - float(self.picks) / float(ANNEAL_PICKS))

    # ------------------------------------------------------------------ gürültü
    def _u(self, rng=None) -> float:
        for src in (rng, self.rng):
            if src is not None:
                try:
                    return float(src.random())
                except Exception:
                    pass
        if self.entropy is not None:
            try:
                return float(self.entropy.unit())
            except Exception:
                pass
        return random.random()

    # ------------------------------------------------------------------ yarış
    def eligible(self, kind: str, ctx: dict, slots: dict) -> list[MotorAssembly]:
        return [a for a in self.kinds.get(kind, ()) if a.matches(ctx, slots)]

    def pick(self, kind: str, ctx: dict | None = None, slots: dict | None = None, *,
             prov=None, noise: float = 0.45, rng=None) -> str | None:
        """Topluluklar yarışır; kazanan metni döndürür. Uygun topluluk yoksa None.

        `prov` verilirse sözün kaynağı ölçere yazılır: kazanan ÖĞRENİLMİŞ ağırlığıyla mı
        (ogrenim) yoksa doğuştan eğilimiyle mi (kalip) konuştu?
        """
        ctx = ctx or {}
        slots = slots or {}
        cands = self.eligible(kind, ctx, slots)
        if not cands:
            return None
        ps = self.prior_strength
        recent = self.recent.setdefault(kind, deque(maxlen=FRESH_WINDOW))
        best: MotorAssembly | None = None
        best_act = float("-inf")
        for a in cands:
            act = float(a.w) + ps * float(a.bias)
            if a.id in recent:
                act -= FRESH_PENALTY
            act += noise * self._u(rng)
            if act > best_act:
                best, best_act = a, act
        if best is None:
            return None
        best.uses += 1
        best.wins += 1
        recent.append(best.id)
        self.picks += 1
        self.turn_ids.append(best.id)
        learned = abs(float(best.w)) >= (ps * float(best.bias))
        self.log.append({"kind": kind, "id": best.id, "a": round(best_act, 3),
                         "ps": round(ps, 3), "learned": bool(learned), "w": round(best.w, 3)})
        if prov is not None:
            try:
                prov.record("ogrenim" if learned else "kalip", 1)
            except Exception:
                pass
        return best.render(slots)

    # ------------------------------------------------------------------ öğrenme
    def begin_turn(self) -> None:
        """Yeni tur: geçen turun seçimleri öğrenme sinyaline devredilir, ağırlıklar söner."""
        self.last_turn_ids = list(self.turn_ids)
        self.turn_ids = []
        for a in self._by_id.values():
            a.w *= 0.9995          # unutma eğrisi (yavaş)

    def learn(self, repeated: bool) -> dict:
        """Turun SONUCU: aynı söz tekrar geldiyse o tur konuşanlar işe yaramadı."""
        changed = {"rewarded": [], "punished": []}
        for pid in self.last_turn_ids:
            a = self._by_id.get(pid)
            if a is None:
                continue
            if repeated:
                a.w = max(W_MIN, float(a.w) - PUNISH)
                a.losses += 1
                changed["punished"].append(pid)
            else:
                a.w = min(W_MAX, float(a.w) + REWARD)
                changed["rewarded"].append(pid)
        return changed

    # ------------------------------------------------------------------ rapor
    def report(self) -> dict:
        learned = sum(1 for a in self._by_id.values() if abs(a.w) >= 0.25)
        return {"topluluk": len(self._by_id), "secim": int(self.picks),
                "prior_strength": round(self.prior_strength, 3),
                "ogrenilmis_topluluk": int(learned),
                "son": list(self.log)[-4:]}

    # ------------------------------------------------------------------ kalıcılık
    def state_dict(self) -> dict:
        return {"picks": int(self.picks),
                "learned": {k: a.to_dict() for k, a in self._by_id.items()
                            if a.uses or a.w or a.wins or a.losses},
                "recent": {k: list(v) for k, v in self.recent.items()}}

    def load_state_dict(self, d: dict | None) -> None:
        if not isinstance(d, dict):
            return
        try:
            self.picks = int(d.get("picks", 0) or 0)
            for pid, rec in (d.get("learned") or {}).items():
                a = self._by_id.get(str(pid))
                if a is None:
                    continue
                a.w = float(rec.get("w", a.w))
                a.uses = int(rec.get("uses", a.uses))
                a.wins = int(rec.get("wins", a.wins))
                a.losses = int(rec.get("losses", a.losses))
            for kind, ids in (d.get("recent") or {}).items():
                self.recent[str(kind)] = deque([str(i) for i in ids], maxlen=FRESH_WINDOW)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Varsayılan alan: AGI nesnesi olmadan çağrılan compose_* fonksiyonları için.
# ---------------------------------------------------------------------------
_DEFAULT: PriorField | None = None


def default_field() -> PriorField:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = PriorField()
    return _DEFAULT


def _selftest() -> None:
    f = PriorField()
    print(f"topluluk: {len(f._by_id)} | prior_strength={f.prior_strength:.2f}")
    ctx = {"gap": 10.0, "same": True, "count": 2, "turns": 1}
    outs = [f.pick("meta.notice.question", ctx, {}) for _ in range(4)]
    for i, o in enumerate(outs, 1):
        print(f"{i}. {o}")
    print("farklı:", len(set(outs)), "/", len(outs))
    print(f.report())


if __name__ == "__main__":
    _selftest()
