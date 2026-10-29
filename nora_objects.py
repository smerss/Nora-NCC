#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_objects.py — duyusal listelerden nesne inancı ve occlusion takibi.

Bu organ dünya nesnelerinin gizli kimliğini okumaz. Her çağrıda yalnız sayısal
özellikler/kategorik görünüm ve zaman verilir; iç iz kimliği eşleştirme ile açılır.
Görülmeyen izler konum hızıyla ileri taşınır, görünürlükleri false ve güvenleri
azalır. Bu, nesne permanencesinin sonsuz kesinliği değil, sınırlı ve ölçülebilir
bir inanç güncellemesidir.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np


@dataclass
class Track:
    track_id: str
    features: dict[str, float]
    categories: dict[str, str] = field(default_factory=dict)
    velocity: dict[str, float] = field(default_factory=dict)
    age: int = 0
    last_seen: int = 0
    last_time: float = 0.0
    missed: int = 0
    visible: bool = True
    confidence: float = 0.65
    updates: int = 1


class ObjectBelief:
    def __init__(self, *, max_tracks: int = 64, max_missed: int = 4,
                 gate: float = 9.0, decay: float = 0.82):
        self.max_tracks = max(1, int(max_tracks))
        self.max_missed = max(0, int(max_missed))
        self.gate = float(gate)
        self.decay = float(np.clip(decay, 0.0, 1.0))
        self.tracks: dict[str, Track] = {}
        self.time = 0.0
        self.frame = 0
        self.serial = 0
        self.retired = 0
        self.dropped = 0
        self.last_assignments: dict[int, str] = {}

    @staticmethod
    def _flatten(obj: dict) -> tuple[dict[str, float], dict[str, str]]:
        numeric: dict[str, float] = {}
        categorical: dict[str, str] = {}
        source = dict(obj.get("features") or {}) if isinstance(obj.get("features"), dict) else {}
        source.update({k: v for k, v in obj.items() if k != "features"})
        def visit(key, value):
            key = str(key)
            if key in {"id", "track_id", "visible", "timestamp"}:
                return
            if isinstance(value, (int, float, np.number)) and np.isfinite(float(value)):
                numeric[key] = float(value)
            elif isinstance(value, (list, tuple, np.ndarray)):
                arr = np.asarray(value)
                if arr.ndim == 1 and arr.size <= 32:
                    for i, item in enumerate(arr):
                        visit(f"{key}.{i}", item)
            elif isinstance(value, str):
                categorical[key] = value
        for key, value in source.items():
            visit(key, value)
        return numeric, categorical

    @staticmethod
    def _is_position(key: str) -> bool:
        low = key.lower()
        return (low in {"x", "y", "z", "px", "py", "pz"} or
                low.startswith("position") or low.startswith("pos.") or
                low.endswith(".x") or low.endswith(".y") or low.endswith(".z"))

    def _predicted(self, track: Track, dt: float) -> dict[str, float]:
        out = dict(track.features)
        for key, velocity in track.velocity.items():
            if self._is_position(key):
                out[key] = out.get(key, 0.0) + float(velocity) * dt
        return out

    def _cost(self, track: Track, numeric: dict[str, float],
              categorical: dict[str, str], dt: float) -> float:
        predicted = self._predicted(track, dt)
        common = set(predicted) & set(numeric)
        if not common and not (set(track.categories) & set(categorical)):
            return float("inf")
        terms = []
        for key in common:
            scale = max(1.0, abs(predicted[key]) * 0.05, abs(numeric[key]) * 0.05)
            terms.append(((predicted[key] - numeric[key]) / scale) ** 2)
        mismatch = sum(track.categories.get(k) != v for k, v in categorical.items()
                       if k in track.categories)
        # Görünüm çelişkisi ucuz bir benzerlikle ezilemez.
        return float(np.mean(terms) if terms else 0.0) + 4.0 * mismatch

    def _new(self, numeric: dict[str, float], categorical: dict[str, str]) -> Track | None:
        if len(self.tracks) >= self.max_tracks:
            self.dropped += 1
            return None
        tid = f"t{self.serial}"
        self.serial += 1
        track = Track(tid, dict(numeric), dict(categorical), {}, 0, self.frame,
                      self.time, 0, True, 0.65, 1)
        self.tracks[tid] = track
        return track

    def observe(self, objects: list[dict] | None = None, *, dt: float = 1.0) -> list[dict]:
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        self.time += dt
        self.frame += 1
        observations = [self._flatten(dict(obj)) for obj in (objects or [])]
        candidates = []
        for tid, track in self.tracks.items():
            for index, (numeric, categorical) in enumerate(observations):
                cost = self._cost(track, numeric, categorical, dt)
                if math.isfinite(cost) and cost <= self.gate:
                    candidates.append((cost, tid, index))
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        used_tracks: set[str] = set()
        used_obs: set[int] = set()
        assignments: dict[int, str] = {}
        for _cost, tid, index in candidates:
            if tid in used_tracks or index in used_obs:
                continue
            used_tracks.add(tid)
            used_obs.add(index)
            assignments[index] = tid
            track = self.tracks[tid]
            numeric, categorical = observations[index]
            elapsed = max(dt, self.time - track.last_time)
            for key, value in numeric.items():
                if self._is_position(key) and key in track.features:
                    measured_v = (value - track.features[key]) / elapsed
                    old_v = track.velocity.get(key, measured_v)
                    track.velocity[key] = 0.5 * old_v + 0.5 * measured_v
                track.features[key] = float(value)
            track.categories.update(categorical)
            track.age += 1
            track.last_seen = self.frame
            track.last_time = self.time
            track.missed = 0
            track.visible = True
            track.updates += 1
            track.confidence = min(1.0, track.confidence + 0.12)
        for tid, track in list(self.tracks.items()):
            if tid in used_tracks:
                continue
            track.features = self._predicted(track, dt)
            track.age += 1
            track.missed += 1
            track.visible = False
            track.confidence *= self.decay
            if track.missed > self.max_missed:
                del self.tracks[tid]
                self.retired += 1
        for index, (numeric, categorical) in enumerate(observations):
            if index in used_obs:
                continue
            track = self._new(numeric, categorical)
            if track is not None:
                assignments[index] = track.track_id
        self.last_assignments = assignments
        return self.snapshot()

    def snapshot(self, *, include_occluded: bool = True) -> list[dict]:
        rows = []
        for tid in sorted(self.tracks, key=lambda x: int(x[1:]) if x[1:].isdigit() else x):
            track = self.tracks[tid]
            if not include_occluded and not track.visible:
                continue
            rows.append({"id": tid, "features": dict(track.features),
                         "categories": dict(track.categories),
                         "velocity": dict(track.velocity), "age": track.age,
                         "last_seen": track.last_seen, "missed": track.missed,
                         "visible": track.visible, "occluded": not track.visible,
                         "confidence": float(track.confidence), "updates": track.updates})
        return rows

    def get(self, track_id: str) -> dict | None:
        return next((row for row in self.snapshot() if row["id"] == track_id), None)

    def state_dict(self) -> dict:
        return {"version": 1, "max_tracks": self.max_tracks, "max_missed": self.max_missed,
                "gate": self.gate, "decay": self.decay, "time": self.time,
                "frame": self.frame, "serial": self.serial, "retired": self.retired,
                "dropped": self.dropped,
                "tracks": {tid: vars(track).copy() for tid, track in self.tracks.items()}}

    def load_state_dict(self, data: dict | None) -> None:
        if not data:
            return
        self.max_tracks = int(data.get("max_tracks", self.max_tracks))
        self.max_missed = int(data.get("max_missed", self.max_missed))
        self.gate = float(data.get("gate", self.gate))
        self.decay = float(data.get("decay", self.decay))
        self.time = float(data.get("time", 0.0))
        self.frame = int(data.get("frame", 0))
        self.serial = int(data.get("serial", 0))
        self.retired = int(data.get("retired", 0))
        self.dropped = int(data.get("dropped", 0))
        self.tracks = {}
        for tid, raw in (data.get("tracks") or {}).items():
            self.tracks[tid] = Track(str(raw["track_id"]), dict(raw.get("features") or {}),
                                     dict(raw.get("categories") or {}), dict(raw.get("velocity") or {}),
                                     int(raw.get("age", 0)), int(raw.get("last_seen", 0)),
                                     float(raw.get("last_time", 0.0)), int(raw.get("missed", 0)),
                                     bool(raw.get("visible", False)), float(raw.get("confidence", 0.0)),
                                     int(raw.get("updates", 0)))
        self.last_assignments = {}


__all__ = ["Track", "ObjectBelief"]
