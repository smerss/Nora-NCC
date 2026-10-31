#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_crossmodal.py — iç izler arasında kanıta dayalı çapraz-modal bağ.

Modality adları serbesttir. Dış dünya kimliği/anlam etiketi kullanılmaz; organların
ürettiği iç token'lar, zaman ve ölçülen özellikler kullanılır. Olaylar sınırlı bir
günlükte tutulur ve sorgu sırasında zaman-benzerlik sıralı global bire-bir atama
yapılır; çevrimiçi ilk-uygun eşleşmenin çift sayım/distraktör kapması olmaz.
"""
from __future__ import annotations

from nora._paths import REPO_ROOT


import math
from typing import Any

import numpy as np

from nora_state import decode, encode


class CrossModalBinder:
    def __init__(self, *, window: float = 0.75, min_count: int = 3,
                 max_events: int = 512):
        self.window = float(window)
        self.min_count = max(2, int(min_count))
        self.max_events = max(32, int(max_events))
        self.event_log: list[dict] = []
        self.pairs: dict[tuple[tuple[str, str], tuple[str, str]], dict] = {}
        self.events = 0
        self._dirty = True

    @staticmethod
    def _endpoint(modality: str, token: str) -> tuple[str, str]:
        return str(modality), str(token)

    @staticmethod
    def _pair(a, b):
        return (a, b) if a <= b else (b, a)

    @staticmethod
    def _similarity(a: dict, b: dict) -> float:
        common = set(a) & set(b)
        if not common:
            return 0.5  # yalnız zaman kanıtı; tam özellik kanıtı değil
        terms = []
        for key in common:
            scale = max(1.0, abs(float(a[key])), abs(float(b[key])))
            terms.append(((float(a[key]) - float(b[key])) / scale) ** 2)
        return float(math.exp(-float(np.mean(terms))))

    def observe(self, modality: str, token: str, features: dict[str, float] | None = None,
                *, timestamp: float) -> dict:
        endpoint = self._endpoint(modality, token)
        t = float(timestamp)
        if not math.isfinite(t):
            raise ValueError("timestamp must be finite")
        clean = {str(k): float(v) for k, v in (features or {}).items()
                 if isinstance(v, (int, float, np.number)) and np.isfinite(float(v))}
        event = {"endpoint": endpoint, "features": clean, "time": t, "serial": self.events}
        self.events += 1
        self.event_log.append(event)
        self.event_log = self.event_log[-self.max_events:]
        self._dirty = True
        # Bağlar query'de global olarak kurulur; burada erken/yanlış eşleşme yok.
        return {"endpoint": endpoint, "time": t, "pending": True, "bindings": []}

    def _rebuild(self) -> None:
        candidates = []
        events = self.event_log
        for i, left in enumerate(events):
            for right in events[i + 1:]:
                a, b = left["endpoint"], right["endpoint"]
                if a[0] == b[0]:
                    continue
                distance = abs(float(left["time"]) - float(right["time"]))
                if distance > self.window:
                    continue
                sim = self._similarity(left["features"], right["features"])
                candidates.append((-sim, distance, left["serial"], right["serial"], left, right, sim))
        candidates.sort(key=lambda row: row[:4])
        # (olay, karşı modality) başına tek atama; üçüncü modality yine bağlanabilir.
        used: set[tuple[int, str]] = set()
        pairs = {}
        for _neg_sim, _distance, _ls, _rs, left, right, sim in candidates:
            a, b = left["endpoint"], right["endpoint"]
            if (left["serial"], b[0]) in used or (right["serial"], a[0]) in used:
                continue
            used.add((left["serial"], b[0]))
            used.add((right["serial"], a[0]))
            key = self._pair(a, b)
            stat = pairs.setdefault(key, {"n": 0, "similarity": 0.0, "last": 0.0})
            stat["n"] += 1
            stat["similarity"] += (sim - stat["similarity"]) / stat["n"]
            stat["last"] = max(float(left["time"]), float(right["time"]))
        self.pairs = pairs
        self._dirty = False

    def query(self, modality: str, token: str) -> dict:
        if self._dirty:
            self._rebuild()
        endpoint = self._endpoint(modality, token)
        candidates = []
        for (a, b), stat in self.pairs.items():
            if endpoint == a:
                other = b
            elif endpoint == b:
                other = a
            else:
                continue
            if stat["n"] > 0:
                candidates.append((other, stat))
        if not candidates:
            return {"status": "unresolved", "reason": "no co-occurrence", "candidates": 0}
        candidates.sort(key=lambda x: (-x[1]["n"], -x[1]["similarity"], x[0]))
        total = sum(stat["n"] for _other, stat in candidates)
        best, second = candidates[0], candidates[1] if len(candidates) > 1 else (None, {"n": 0})
        other, stat = best
        ratio = stat["n"] / max(1, total)
        margin = (stat["n"] - second[1]["n"]) / max(1, total)
        if stat["n"] < self.min_count or ratio < 0.5 or margin <= 0.0:
            return {"status": "unresolved", "reason": "insufficient count or ambiguous rival",
                    "candidates": len(candidates), "count": stat["n"], "ratio": ratio}
        confidence = float(np.clip((1.0 - math.exp(-stat["n"] / 3.0)) *
                                   ratio * stat["similarity"], 0.0, 0.95))
        if confidence <= 0.0:
            return {"status": "unresolved", "reason": "zero confidence", "candidates": len(candidates)}
        return {"status": "ok", "match": {"modality": other[0], "token": other[1]},
                "confidence": confidence, "count": stat["n"],
                "similarity": stat["similarity"], "margin": margin}

    def state_dict(self) -> dict:
        return {"version": 2, "window": self.window, "min_count": self.min_count,
                "max_events": self.max_events, "events": self.events,
                "log": [encode(event) for event in self.event_log]}

    def load_state_dict(self, data: dict | None) -> None:
        if not data:
            return
        self.window = float(data.get("window", self.window))
        self.min_count = int(data.get("min_count", self.min_count))
        self.max_events = int(data.get("max_events", self.max_events))
        self.events = int(data.get("events", 0))
        self.event_log = []
        for raw in data.get("log") or []:
            event = decode(raw)
            event["endpoint"] = tuple(event["endpoint"])
            self.event_log.append(event)
        self.event_log = self.event_log[-self.max_events:]
        self.pairs = {}
        self._dirty = True


__all__ = ["CrossModalBinder"]
