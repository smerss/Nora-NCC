#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_workspace.py — kaynaklı paylaşılan çalışma alanı ve seçici WM köprüsü.

GNWT/bilinç kanıtı değildir. Ölçülen işlev, bir üreticinin yükünün başka bir
tüketicinin aynı olaydaki karar girdisi hâline gelmesidir. Olaylar sınırsız WM
yuvalarına dökülmez; son değer + kaynak + sinyal kimliği tutulur. ``focus``
çağrısı olmadan mevcut Interpreter WM'sine yazılmaz.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nora_state import decode, encode


@dataclass
class WorkspaceValue:
    value: Any
    source: str
    topic: str
    signal: str
    confidence: float


class IntegrationWorkspace:
    def __init__(self, bus=None, working_memory=None, *, capacity: int = 32):
        self.bus = bus
        self.working_memory = working_memory
        self.capacity = max(4, int(capacity))
        self.values: dict[str, WorkspaceValue] = {}
        self.order: list[str] = []
        self.history: list[dict] = []
        self.subscriptions: dict[str, str] = {}
        self.writes = 0
        self.dropped = 0

    def attach(self, topics=("body.observation", "body.transition", "self.state",
                             "prediction", "perception.objects", "source.change",
                             "architecture.execution")) -> None:
        if self.bus is None:
            return
        for topic in topics:
            if topic in self.subscriptions:
                continue
            name = self.bus.subscribe(topic, self._receive,
                                      name=f"workspace.{topic}")
            self.subscriptions[topic] = name

    def detach(self) -> None:
        if self.bus is not None:
            for name in self.subscriptions.values():
                self.bus.unsubscribe(name)
        self.subscriptions.clear()

    def _receive(self, signal) -> None:
        self.write(signal.topic, signal.payload, source=signal.source,
                   signal=signal.id, confidence=signal.confidence)

    def write(self, key: str, value: Any, *, source: str = "", signal: str = "",
              confidence: float = 1.0) -> None:
        key = str(key)
        if key not in self.values:
            self.order.append(key)
            if len(self.order) > self.capacity:
                old = self.order.pop(0)
                self.values.pop(old, None)
                self.dropped += 1
        self.values[key] = WorkspaceValue(value, str(source), key, str(signal), float(confidence))
        self.writes += 1
        self.history.append({"key": key, "source": str(source), "signal": str(signal)})
        self.history = self.history[-128:]

    def read(self, key: str, default=None):
        item = self.values.get(str(key))
        return default if item is None else item.value

    def provenance(self, key: str) -> dict | None:
        item = self.values.get(str(key))
        if item is None:
            return None
        return {"source": item.source, "topic": item.topic, "signal": item.signal,
                "confidence": item.confidence}

    def focus(self, key: str, *, wm_key: str = "focus") -> bool:
        """Seçilen değeri gerçek tipli WM'ye tek bir yuvayla aktar."""
        if self.working_memory is None or key not in self.values:
            return False
        item = self.values[key]
        self.working_memory.write(wm_key, item.value, f"workspace:{item.source}")
        return True

    def snapshot(self) -> dict:
        return {key: self.values[key].value for key in self.order if key in self.values}

    def state_dict(self) -> dict:
        return {"version": 1, "capacity": self.capacity, "order": list(self.order),
                "values": {key: {"value": encode(item.value), "source": item.source,
                                 "topic": item.topic, "signal": item.signal,
                                 "confidence": item.confidence}
                           for key, item in self.values.items()},
                "history": list(self.history), "writes": self.writes, "dropped": self.dropped}

    def load_state_dict(self, data: dict | None) -> None:
        if not data:
            return
        self.capacity = int(data.get("capacity", self.capacity))
        self.order = [str(x) for x in (data.get("order") or [])]
        self.values = {}
        for key, raw in (data.get("values") or {}).items():
            self.values[str(key)] = WorkspaceValue(decode(raw.get("value")),
                                                    str(raw.get("source", "")),
                                                    str(raw.get("topic", key)),
                                                    str(raw.get("signal", "")),
                                                    float(raw.get("confidence", 1.0)))
        self.history = [dict(x) for x in (data.get("history") or [])]
        self.writes = int(data.get("writes", 0))
        self.dropped = int(data.get("dropped", 0))


__all__ = ["WorkspaceValue", "IntegrationWorkspace"]
