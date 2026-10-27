#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_metacog.py — tahmin hatasından öğrenen işlevsel öz-izleme.

Bilinç/qualia iddiası yoktur. Ölçülen şey, sistemin kendi tahmin güveni ile
sonradan gözlenen doğruluğu arasındaki ilişkidir. Kayıt öncesi kalibrasyon
(prequential) saklanır; böylece aynı örneği öğrenip sonra onu başarı diye
saymak yerine ham ve kalibre Brier skorları karşılaştırılabilir.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


class PredictionMonitor:
    def __init__(self, *, bins: int = 10, prior_strength: float = 8.0,
                 max_pending: int = 256, max_outcomes: int = 1024):
        self.bins = max(2, int(bins))
        self.prior_strength = max(0.1, float(prior_strength))
        self.max_pending = max(8, int(max_pending))
        self.max_outcomes = max(32, int(max_outcomes))
        # kaynak -> güven kutusu -> [doğru, toplam]
        self.stats: dict[str, list[list[float]]] = {}
        self.pending: dict[str, dict] = {}
        self.outcomes: list[dict] = []
        self.serial = 0
        self.missed = 0

    def _bin(self, confidence: float) -> int:
        return min(self.bins - 1, max(0, int(float(confidence) * self.bins)))

    def _source_stats(self, source: str) -> list[list[float]]:
        return self.stats.setdefault(str(source), [[0.0, 0.0] for _ in range(self.bins)])

    def calibrate_one(self, confidence: float, source: str = "unknown") -> float:
        raw = float(np.clip(confidence, 0.0, 1.0))
        good, total = self._source_stats(source)[self._bin(raw)]
        # Veri yoksa ham güven; veri geldikçe ölçülen doğruluğa doğru büzülme.
        return float(np.clip((good + self.prior_strength * raw) /
                             (total + self.prior_strength), 0.0, 1.0))

    def calibrate(self, predictions: list[dict]) -> list[dict]:
        out = []
        for pred in predictions or []:
            row = dict(pred)
            raw = float(np.clip(row.get("confidence", 0.0), 0.0, 1.0))
            row["raw_confidence"] = raw
            row["confidence"] = self.calibrate_one(raw, str(row.get("source", "unknown")))
            out.append(row)
        return out

    def begin(self, predictions: list[dict], before: Any, action: Any = None) -> str | None:
        rows = [dict(p) for p in (predictions or []) if isinstance(p, dict)]
        if not rows:
            return None
        state = np.asarray(before, dtype=np.float64).reshape(-1).copy()
        self.serial += 1
        token = f"p{self.serial}"
        self.pending[token] = {"before": state, "action": action, "predictions": rows}
        if len(self.pending) > self.max_pending:
            oldest = next(iter(self.pending))
            del self.pending[oldest]
            self.missed += 1
        return token

    def resolve(self, token: str | None, after: Any, *, feature_names=None,
                relative_tolerance: float = 0.05) -> list[dict]:
        if not token or token not in self.pending:
            return []
        item = self.pending.pop(token)
        before = item["before"]
        now = np.asarray(after, dtype=np.float64).reshape(-1)
        if now.size != before.size:
            self.missed += 1
            return []
        names = list(feature_names or [f"f{i}" for i in range(now.size)])
        index = {name: i for i, name in enumerate(names)}
        resolved = []
        for pred in item["predictions"]:
            name = str(pred.get("variable", ""))
            if name not in index:
                continue
            i = index[name]
            predicted = float(pred.get("delta", 0.0))
            actual = float(now[i] - before[i])
            error = abs(predicted - actual)
            tolerance = max(1e-6, float(relative_tolerance) *
                            max(1.0, abs(predicted), abs(actual)))
            correct = bool(math.isfinite(error) and error <= tolerance)
            raw = float(np.clip(pred.get("confidence", 0.0), 0.0, 1.0))
            source = str(pred.get("source", "unknown"))
            # Skor, bu örneği modele eklemeden ÖNCE ölçülür.
            calibrated = self.calibrate_one(raw, source)
            stat = self._source_stats(source)[self._bin(raw)]
            stat[0] += float(correct)
            stat[1] += 1.0
            row = {"variable": name, "predicted": predicted, "actual": actual,
                   "error": error, "tolerance": tolerance, "correct": correct,
                   "raw_confidence": raw, "calibrated_confidence": calibrated,
                   "source": source}
            resolved.append(row)
            self.outcomes.append(row)
        self.outcomes = self.outcomes[-self.max_outcomes:]
        return resolved

    def report(self) -> dict:
        rows = self.outcomes
        if not rows:
            return {"status": "insufficient", "n": 0, "coverage": 0.0,
                    "missed": self.missed, "meaning": "functional prediction monitoring"}
        n = len(rows)
        accuracy = sum(r["correct"] for r in rows) / n
        raw_brier = sum((r["raw_confidence"] - float(r["correct"])) ** 2 for r in rows) / n
        cal_brier = sum((r["calibrated_confidence"] - float(r["correct"])) ** 2 for r in rows) / n
        ece = 0.0
        for b in range(self.bins):
            group = [r for r in rows if self._bin(r["raw_confidence"]) == b]
            if group:
                conf = sum(r["raw_confidence"] for r in group) / len(group)
                acc = sum(r["correct"] for r in group) / len(group)
                ece += len(group) / n * abs(conf - acc)
        return {"status": "ok" if n >= 8 else "insufficient", "n": n,
                "accuracy": float(accuracy), "raw_brier": float(raw_brier),
                "calibrated_brier": float(cal_brier), "ece": float(ece),
                "coverage": float(n / max(1, n + self.missed)), "missed": self.missed,
                "meaning": "functional prediction monitoring; not subjective experience"}

    def state_dict(self) -> dict:
        return {"version": 1, "bins": self.bins, "prior_strength": self.prior_strength,
                "max_pending": self.max_pending, "max_outcomes": self.max_outcomes,
                "stats": self.stats, "outcomes": self.outcomes, "serial": self.serial,
                "missed": self.missed}

    def load_state_dict(self, data: dict | None) -> None:
        if not data:
            return
        self.bins = int(data.get("bins", self.bins))
        self.prior_strength = float(data.get("prior_strength", self.prior_strength))
        self.max_pending = int(data.get("max_pending", self.max_pending))
        self.max_outcomes = int(data.get("max_outcomes", self.max_outcomes))
        self.stats = {str(k): [[float(x[0]), float(x[1])] for x in v]
                      for k, v in (data.get("stats") or {}).items()}
        self.outcomes = [dict(x) for x in (data.get("outcomes") or [])]
        self.serial = int(data.get("serial", 0))
        self.missed = int(data.get("missed", 0))
        # Bekleyen tahminin before/action değerleri kasıtlı olarak geri yüklenmez;
        # eski bağlamda kalan ölçümü yeni dünyaya uygulamak sahte kanıt olurdu.
        self.pending = {}


__all__ = ["PredictionMonitor"]
