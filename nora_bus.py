#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_bus.py — çalışma-zamanı bilişsel yayın ve yarışma veri yolu.

Bu modül bir "bilinç" etiketi değil, ölçülebilir bir entegrasyon ilkelidir:

* üretici ve tüketici birbirinin sınıfını bilmez; konu ve yük şeması yeterlidir,
* her sinyalin kaynağı ve ebeveynleri tutulur; bir karar geriye doğru izlenebilir,
* aboneler çalışma zamanında eklenebilir, kapatılabilir ve sınırlanabilir,
* yarışma yalnız gerçek teklifler arasından yapılır; teklif yoksa başarı uydurulmaz,
* geri çağırma hatası yaşam döngüsünü sessizce öldürmez, fakat hata kaydı olarak görünür.

Veri yolu kasıtlı olarak eşzamanlıdır. Böylece aynı tohum ve aynı olay sırasıyla
tekrarlanabilir; iş parçacığı yarışları "entegrasyon" diye ölçülmez.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import fnmatch
import itertools
import time
from typing import Any, Callable


@dataclass(frozen=True)
class Bid:
    """Bir yarışmacının teklifi.

    ``score`` yalnız aynı yarışmadaki teklifleri karşılaştırmak içindir; evrensel
    bir değer veya bilinç derecesi değildir. ``payload`` üreticinin gerçek çıktısıdır.
    """

    score: float
    payload: Any
    reason: str = ""
    confidence: float = 1.0


@dataclass
class Signal:
    id: str
    topic: str
    payload: Any
    source: str
    kind: str = "event"
    confidence: float = 1.0
    parents: tuple[str, ...] = ()
    created_ns: int = field(default_factory=time.time_ns)
    delivered: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def summary(self) -> dict:
        return {
            "id": self.id, "topic": self.topic, "source": self.source,
            "kind": self.kind, "confidence": float(self.confidence),
            "parents": list(self.parents), "created_ns": int(self.created_ns),
            "delivered": list(self.delivered), "errors": list(self.errors),
        }


@dataclass
class _Subscription:
    name: str
    pattern: str
    callback: Callable[[Signal], Any]
    priority: int = 0
    enabled: bool = True
    calls: int = 0
    failures: int = 0


class CognitiveBus:
    """Sınırlı, izlenebilir, dinamik yayın/abonelik ve teklif yarışması."""

    def __init__(self, history: int = 2048, max_subscribers: int = 256):
        self.history_limit = max(16, int(history))
        self.max_subscribers = max(1, int(max_subscribers))
        self._subs: list[_Subscription] = []
        self._signals: dict[str, Signal] = {}
        self._order: list[str] = []
        self._ids = itertools.count(1)
        self._dispatch_depth = 0
        self.max_dispatch_depth = 16
        self.published = 0
        self.dropped = 0
        self.errors: list[dict] = []
        self.last_winner: dict | None = None
        self._module_inboxes: dict[str, list[Signal]] = {}

    # ------------------------------------------------------------------ abonelik
    def subscribe(self, pattern: str, callback: Callable[[Signal], Any], *,
                  name: str | None = None, priority: int = 0) -> str:
        if not callable(callback):
            raise TypeError("subscriber callback must be callable")
        if len(self._subs) >= self.max_subscribers:
            raise RuntimeError("subscriber budget exhausted")
        pat = str(pattern or "*")
        base = str(name or getattr(callback, "__qualname__", None)
                     or getattr(callback, "__name__", None) or "subscriber")
        # Aynı ada sahip iki bağımsız uzman birbirini ezmesin.
        name2 = base
        n = 2
        names = {s.name for s in self._subs}
        while name2 in names:
            name2 = f"{base}#{n}"
            n += 1
        self._subs.append(_Subscription(name2, pat, callback, int(priority)))
        self._subs.sort(key=lambda s: -s.priority)
        return name2

    def unsubscribe(self, name: str) -> bool:
        old = len(self._subs)
        self._subs = [s for s in self._subs if s.name != str(name)]
        return len(self._subs) != old

    def set_enabled(self, name: str, enabled: bool) -> bool:
        found = False
        for sub in self._subs:
            if sub.name == str(name):
                sub.enabled = bool(enabled)
                found = True
        return found

    def subscribers(self) -> list[dict]:
        return [{"name": s.name, "pattern": s.pattern, "priority": s.priority,
                 "enabled": s.enabled, "calls": s.calls, "failures": s.failures}
                for s in self._subs]

    def sync_modules(self, modules: dict[str, Any]) -> None:
        """Mimari organ kümesini genel yayın alıcılarıyla tek seferde eşitle.

        Her organ için ayrı konu köprüsü yazılmaz. Alıcılar özel konu yayınlarını
        dinlemez; ``publish_global`` bütün etkin alıcılara aynı sinyali ulaştırır.
        """
        prefix = "module."
        wanted = {str(name) for name in modules}
        existing = {sub.name[len(prefix):] for sub in self._subs if sub.name.startswith(prefix)}
        for name in existing - wanted:
            self.unsubscribe(prefix + name)
            self._module_inboxes.pop(name, None)
        for name, obj in modules.items():
            key = str(name)
            if key in existing:
                continue

            def receive(signal, key=key, obj=obj):
                inbox = self._module_inboxes.setdefault(key, [])
                inbox.append(signal)
                del inbox[:-8]
                # Ortak alıcı protokolü olan organ sinyali doğrudan işler; diğerleri
                # aynı sinyali sınırlı gelen kutularından okuyabilir.
                try:
                    callback = getattr(obj, "receive_broadcast", None)
                    if callable(callback):
                        callback(signal)
                    else:
                        local = getattr(obj, "broadcast_inbox", None)
                        if local is None:
                            local = []
                            setattr(obj, "broadcast_inbox", local)
                        local.append(signal)
                        del local[:-8]
                except Exception as exc:
                    # Veri yolundaki kayıt alındığını kanıtlar; organ tarafındaki
                    # özel işleme hatası yayını başka organlardan etkilemez.
                    self.errors.append({"signal": signal.id, "subscriber": prefix + key,
                                        "error": f"{type(exc).__name__}: {exc}"})

            self.subscribe("\0global", receive, name=prefix + key)

    def module_inbox(self, name: str) -> list[Signal]:
        return list(self._module_inboxes.get(str(name), ()))

    def has_listeners(self, topic: str) -> bool:
        """Bu konu için etkin abone var mı? Boş yaşam döngüsünde kopya maliyetini önler."""
        return any(s.enabled and fnmatch.fnmatchcase(str(topic), s.pattern)
                   for s in self._subs)

    # ------------------------------------------------------------------ yayın
    def publish(self, topic: str, payload: Any = None, *, source: str = "unknown",
                kind: str = "event", confidence: float = 1.0,
                parents=(), _deliver: bool = True, _everyone: bool = False) -> Signal:
        if self._dispatch_depth > self.max_dispatch_depth:
            self.dropped += 1
            raise RuntimeError("publish feedback depth exceeded")
        sid = f"s{next(self._ids)}"
        sig = Signal(sid, str(topic), payload, str(source), str(kind),
                     float(confidence), tuple(str(p) for p in (parents or ()) if p))
        self._signals[sid] = sig
        self._order.append(sid)
        self.published += 1
        self._trim()
        delivered: list[str] = []
        errors: list[str] = []
        if _deliver:
            # Ebeveyn sinyalinin aboneleri yeni yayın açarsa liste üzerinde değişiklik
            # olabilir; bu yüzden anlık kopya ve derinlik sayacı kullanılır.
            self._dispatch_depth += 1
            try:
                for sub in tuple(self._subs):
                    if not sub.enabled:
                        continue
                    if not _everyone and not fnmatch.fnmatchcase(sig.topic, sub.pattern):
                        continue
                    sub.calls += 1
                    try:
                        result = sub.callback(sig)
                        # Yarışma dışındaki abonenin dönüşü de yeni bir sinyal olabilir;
                        # otomatik yayın yapmayız (geri-besleme ve gizli yan etki olmaz).
                        _ = result
                        delivered.append(sub.name)
                    except Exception as exc:  # hata görünür kalır, yayın yaşamaya devam eder
                        sub.failures += 1
                        msg = f"{sub.name}: {type(exc).__name__}: {exc}"
                        errors.append(msg)
                        self.errors.append({"signal": sid, "subscriber": sub.name,
                                            "error": msg})
                        self.errors = self.errors[-128:]
            finally:
                self._dispatch_depth -= 1
        sig.delivered = tuple(delivered)
        sig.errors = tuple(errors)
        return sig

    broadcast = publish

    def publish_global(self, topic: str, payload: Any = None, *, source: str = "unknown",
                       kind: str = "broadcast", confidence: float = 1.0, parents=()) -> Signal:
        """Aynı bütünleşik sinyali konu filtresinden bağımsız bütün etkin organlara dağıt."""
        return self.publish(topic, payload, source=source, kind=kind, confidence=confidence,
                            parents=parents, _everyone=True)

    # ------------------------------------------------------------------ yarışma
    def auction(self, topic: str, payload: Any = None, *, source: str = "unknown",
                baseline: Bid | None = None, parents=()) -> dict | None:
        """Teklifleri topla, en yüksek puanlı geçerli teklifi yayınla.

        Abone ``Bid`` döndürebilir; kolay entegrasyon için doğrudan payload ya da
        ``None`` da kabul edilir. Doğrudan payload'un puanı sıfırdır. ``None``
        teklif değildir. Eşitlikte önce yüksek öncelikli, sonra kayıt sırasındaki
        yarışmacı kazanır; gizli rastgelelik yoktur.
        """
        # İstek kaydı önce yazılır fakat burada henüz dağıtılmaz. Böylece teklif
        # geri çağrıları tam bir kez çalışır; öğrenen bir uzman çift örnek görmez.
        request = self.publish(topic, payload, source=source, kind="request",
                               parents=parents, _deliver=False)
        offers = self._collect_offers(request)
        if baseline is not None:
            offers.append((float(baseline.score), -1, str(source), baseline))
        if not offers:
            self.last_winner = {"request": request.id, "winner": None, "offers": 0}
            return None
        offers.sort(key=lambda x: (-x[0], x[1], x[2]))
        score, _order, winner, bid = offers[0]
        result = {"request": request.id, "winner": winner, "score": float(score),
                  "payload": bid.payload, "reason": bid.reason,
                  "confidence": float(bid.confidence), "offers": len(offers)}
        selected = self.publish(topic + ".selected", bid.payload, source=winner,
                                kind="selection", confidence=bid.confidence,
                                parents=(request.id,))
        result["signal"] = selected.id
        self.last_winner = result
        return result

    def _collect_offers(self, request: Signal) -> list[tuple[float, int, str, Bid]]:
        """Yarışma geri çağrılarını bir kez çalıştır ve geçerli teklifleri topla."""
        offers: list[tuple[float, int, str, Bid]] = []
        for order, sub in enumerate(tuple(self._subs)):
            if not sub.enabled or not fnmatch.fnmatchcase(request.topic, sub.pattern):
                continue
            sub.calls += 1
            try:
                raw = sub.callback(request)
            except Exception as exc:
                sub.failures += 1
                self.errors.append({"signal": request.id, "subscriber": sub.name,
                                    "error": f"{type(exc).__name__}: {exc}"})
                self.errors = self.errors[-128:]
                continue
            if raw is None:
                continue
            if isinstance(raw, Bid):
                bid = raw
            elif isinstance(raw, dict) and "score" in raw and "payload" in raw:
                bid = Bid(float(raw["score"]), raw["payload"],
                          str(raw.get("reason", "")), float(raw.get("confidence", 1.0)))
            else:
                bid = Bid(0.0, raw)
            try:
                score = float(bid.score)
            except (TypeError, ValueError):
                continue
            if score != score:  # NaN teklif kazanamaz
                continue
            offers.append((score, order, sub.name, bid))
        return offers

    # ------------------------------------------------------------------ iz / durum
    def get(self, signal_id: str) -> Signal | None:
        return self._signals.get(str(signal_id))

    def trace(self, signal_id: str) -> list[dict]:
        """Bir sinyalin bütün bilinen ebeveyn zincirini kökten yaprağa döndür."""
        out: list[dict] = []
        seen: set[str] = set()

        def visit(sid: str):
            sid = str(sid)
            if sid in seen:
                return
            seen.add(sid)
            sig = self._signals.get(sid)
            if sig is None:
                out.append({"id": sid, "missing": True})
                return
            for parent in sig.parents:
                visit(parent)
            out.append(sig.summary())

        visit(signal_id)
        return out

    def recent(self, topic: str = "*", limit: int = 32) -> list[Signal]:
        ids = [sid for sid in reversed(self._order)
               if fnmatch.fnmatchcase(self._signals[sid].topic, str(topic))]
        return [self._signals[sid] for sid in ids[:max(0, int(limit))][::-1]]

    def _trim(self):
        while len(self._order) > self.history_limit:
            old = self._order.pop(0)
            self._signals.pop(old, None)

    def state_dict(self) -> dict:
        # Geri çağırmalar kod oldukları için serileştirilmez; bu açıkça abone
        # metaverisinden ayrılır. Sinyal yükleri de burada saklanmaz: büyük duyusal
        # dizilerin checkpoint'i şişirmesini ve güvensiz pickle'ı önleriz.
        return {"published": int(self.published), "dropped": int(self.dropped),
                "history": [self._signals[s].summary() for s in self._order],
                "subscribers": self.subscribers(), "errors": list(self.errors[-32:]),
                "last_winner": (dict(self.last_winner) if self.last_winner else None)}

    def load_state_dict(self, d: dict | None) -> None:
        # Geçmiş sinyaller yeni oturumda yeniden yayınlanmaz; sadece sayaç ve hata
        # muhasebesi taşınır. Böylece yükleme sırasında eski eylemler tekrar etmez.
        d = d or {}
        self.published = int(d.get("published", 0) or 0)
        self.dropped = int(d.get("dropped", 0) or 0)
        self.errors = [dict(x) for x in (d.get("errors") or []) if isinstance(x, dict)]


__all__ = ["Bid", "Signal", "CognitiveBus"]
