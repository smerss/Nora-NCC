#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_causal.py — geçişlerden öğrenilen, sınanabilir nedensel model.

Girdi yalnız ``(durum, eylem, sonraki_durum)`` kayıtlarıdır. Dünya nesnelerinin
gizli alanları, hedef/etki sözlüğü veya test cevabı modele verilmez. İki kanıt
yolu vardır:

1. eylem-keşifli karşılaştırma: bir eylemin sonuç ortalaması diğer eylemlerden
   ayrışıyor mu? Müdahale ve gözlem kayıtları ayrı sayılır.
2. öznitelikli geçiş modeli: eylem öznitelikleri verilirse, her değişken için
   recursive sufficient statistics ile doğrusal model öğrenilir. Böylece yeni
   eylem parametresi için tablo ezberi yerine held-out tahmin mümkün olur.

Bu, gözlemsel korelasyonu otomatik olarak nedensellik ilan etmez. ``evidence``
alanı ``intervention`` ya da ``association`` döner; yetersiz örnek/rank veya
başka eylem karşılaştırması yoksa tahmin üretilmez.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Callable

import numpy as np

from nora_state import decode, encode


@dataclass
class Summary:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    interventions: int = 0

    def update(self, value: float, intervened: bool = False) -> None:
        x = float(value)
        if not np.isfinite(x):
            return
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)
        self.interventions += int(bool(intervened))

    @property
    def variance(self) -> float:
        return float(self.m2 / max(1, self.n - 1))

    @property
    def se(self) -> float:
        return float(np.sqrt(max(0.0, self.variance) / max(1, self.n)))

    def dict(self) -> dict:
        return {"n": self.n, "mean": self.mean, "m2": self.m2,
                "interventions": self.interventions}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Summary":
        d = d or {}
        return cls(int(d.get("n", 0)), float(d.get("mean", 0.0)),
                   float(d.get("m2", 0.0)), int(d.get("interventions", 0)))


class LinearTransition:
    """Her hedef değişken için X'X ve X'y; ham örnekleri biriktirmez."""

    def __init__(self, dim: int):
        self.dim = int(dim)
        self.xx = np.zeros((self.dim, self.dim), dtype=np.float64)
        self.xy = np.zeros(self.dim, dtype=np.float64)
        self.n = 0
        self.interventions = 0
        self.residual = Summary()

    def update(self, x: np.ndarray, y: float, intervened: bool) -> None:
        if self.n >= max(4, 2 * self.dim):
            try:
                beta = self.beta()
                self.residual.update(float(y - x @ beta), intervened)
            except (ValueError, np.linalg.LinAlgError):
                pass
        self.xx += np.outer(x, x)
        self.xy += x * float(y)
        self.n += 1
        self.interventions += int(bool(intervened))

    def beta(self, ridge: float = 1e-5) -> np.ndarray:
        reg = float(ridge) * np.eye(self.dim)
        # Kesişim katsayısını da çok hafif düzenle; sıfır-rank'ta çözüm üretme.
        return np.linalg.solve(self.xx + reg, self.xy)

    def ready(self) -> bool:
        return self.n >= max(4, self.dim + 1) and int(np.linalg.matrix_rank(self.xx)) >= self.dim

    def predict(self, x: np.ndarray) -> tuple[float, float, float]:
        if not self.ready():
            raise ValueError("linear transition is not identified")
        beta = self.beta()
        inv = np.linalg.pinv(self.xx + 1e-5 * np.eye(self.dim))
        leverage = max(0.0, float(x @ inv @ x))
        support = 1.0 - math.exp(-self.n / max(2.0, 2.0 * self.dim))
        # Uzak/eksik örneklemde güven düşer; sıfır residual sınırsız güven değildir.
        uncertainty = 1.0 / (1.0 + leverage * self.dim)
        confidence = float(np.clip(0.05 + 0.90 * support * uncertainty, 0.0, 0.95))
        return float(x @ beta), confidence, leverage

    def dict(self) -> dict:
        return {"dim": self.dim, "xx": self.xx.tolist(), "xy": self.xy.tolist(),
                "n": self.n, "interventions": self.interventions,
                "residual": self.residual.dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "LinearTransition":
        out = cls(int(d["dim"]))
        out.xx = np.asarray(d["xx"], dtype=np.float64).reshape(out.dim, out.dim)
        out.xy = np.asarray(d["xy"], dtype=np.float64).reshape(out.dim)
        out.n = int(d.get("n", 0))
        out.interventions = int(d.get("interventions", 0))
        out.residual = Summary.from_dict(d.get("residual"))
        return out


class CausalTransitionModel:
    """Durum/eylem akışından öğrenen sınırlı ve kalıcı geçiş modeli."""

    def __init__(self, feature_names=None, *, min_support: int = 3,
                 max_actions: int = 64, max_features: int = 128):
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.min_support = max(2, int(min_support))
        self.max_actions = max(2, int(max_actions))
        self.max_features = max(8, int(max_features))
        self.global_stats: dict[str, Summary] = {}
        self.by_action: dict[str, dict[str, Summary]] = {}
        self.linear: dict[str, LinearTransition] = {}
        self.actions: dict[str, Any] = {}
        self.action_features: dict[str, list[float]] = {}
        self.transitions = 0
        self.rejected = 0
        self.last_record: dict | None = None

    # ------------------------------------------------------------------ temsil
    def _vector(self, state: Any) -> tuple[np.ndarray, list[str]]:
        if isinstance(state, dict):
            # Yalnız sayısal yapraklar; sıralama deterministik, isimler kullanıcı
            # veya organın verdiği gerçek alanlardır.
            names = sorted(k for k in state if isinstance(state[k], (int, float, np.number))
                           and np.isfinite(float(state[k])))
            if not names:
                raise ValueError("state has no numeric features")
            return np.asarray([float(state[k]) for k in names], dtype=np.float64), [str(k) for k in names]
        arr = np.asarray(state, dtype=np.float64).reshape(-1)
        if arr.size == 0 or arr.size > self.max_features or not np.all(np.isfinite(arr)):
            raise ValueError("invalid state vector")
        if self.feature_names is None:
            self.feature_names = [f"f{i}" for i in range(arr.size)]
        if len(self.feature_names) != arr.size:
            raise ValueError("feature dimension changed")
        return arr, list(self.feature_names)

    @staticmethod
    def _key(value: Any) -> str:
        try:
            return json.dumps(encode(value), ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"), allow_nan=False)
        except Exception:
            return repr(value)

    @staticmethod
    def _features(action: Any, supplied: Any = None) -> np.ndarray | None:
        raw = supplied
        if raw is None and isinstance(action, dict):
            raw = action.get("features", action.get("özellikler"))
            if raw is None:
                # Öznitelikleri açıkça verilmeyen dict'in yalnız sayısal alanları
                # kullanılır; eylem adı/target gibi semantik etiketler özellik değildir.
                numeric = [(k, v) for k, v in action.items()
                           if isinstance(v, (int, float, np.number)) and np.isfinite(float(v))]
                if numeric:
                    raw = [v for _k, v in sorted(numeric, key=lambda kv: str(kv[0]))]
        if raw is None:
            return None
        try:
            vec = np.asarray(raw, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if vec.size == 0 or vec.size > 32 or not np.all(np.isfinite(vec)):
            return None
        return vec

    # ------------------------------------------------------------------ öğrenme
    def observe(self, state: Any, action: Any, next_state: Any, *,
                action_features: Any = None, intervened: bool = False) -> dict:
        before, names = self._vector(state)
        after, names_after = self._vector(next_state)
        if names != names_after:
            raise ValueError("state feature names changed")
        key = self._key(action)
        if key not in self.by_action and len(self.by_action) >= self.max_actions:
            self.rejected += 1
            raise RuntimeError("action capacity exhausted")
        feats = self._features(action, action_features)
        if feats is not None:
            # Bütün hedef istatistikleri güncellenmeden önce boyutu doğrula;
            # aksi hâlde yarım yazılmış bir geçiş modeli kalır.
            for name in names:
                old = self.linear.get(name)
                if old is not None and old.dim != len(feats) + 1:
                    self.rejected += 1
                    raise ValueError("action feature dimension changed")
            self.action_features[key] = feats.tolist()
        if key not in self.by_action:
            self.by_action[key] = {name: Summary() for name in names}
            self.actions[key] = action
        deltas = after - before
        for i, name in enumerate(names):
            self.global_stats.setdefault(name, Summary()).update(float(deltas[i]), intervened)
            self.by_action[key].setdefault(name, Summary()).update(float(deltas[i]), intervened)
            if feats is not None:
                lin = self.linear.get(name)
                if lin is None:
                    lin = self.linear[name] = LinearTransition(len(feats) + 1)
                if lin.dim != len(feats) + 1:
                    self.rejected += 1
                    raise ValueError("action feature dimension changed")
                lin.update(np.r_[1.0, feats], float(deltas[i]), intervened)
        self.transitions += 1
        rec = {"id": self.transitions, "action": key, "n_features": len(names),
               "intervened": bool(intervened), "has_action_features": feats is not None}
        self.last_record = rec
        return rec

    # ------------------------------------------------------------------ tahmin
    def predict(self, state: Any, action: Any, *, action_features: Any = None,
                min_confidence: float = 0.05) -> list[dict]:
        _before, names = self._vector(state)
        key = self._key(action)
        feats = self._features(action, action_features)
        out: list[dict] = []
        for name in names:
            # Öznitelikli model, görülmemiş eylem için de kullanılabilir.
            if feats is not None and len(self.by_action) >= 2 and name in self.linear:
                lin = self.linear[name]
                try:
                    delta, confidence, leverage = lin.predict(np.r_[1.0, feats])
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    pass
                else:
                    if confidence >= min_confidence and abs(delta) > 1e-10:
                        out.append({"variable": name, "delta": float(delta),
                                    "confidence": float(confidence), "support": int(lin.n),
                                    "evidence": "intervention" if lin.interventions == lin.n else "association",
                                    "source": "transition-linear", "leverage": float(leverage)})
                    continue
            stats = self.by_action.get(key, {}).get(name)
            total = self.global_stats.get(name)
            if stats is None or total is None or stats.n < self.min_support:
                continue
            other_n = total.n - stats.n
            if other_n < self.min_support:
                continue
            other_mean = (total.mean * total.n - stats.mean * stats.n) / other_n
            # Tahmin edilen büyüklük eylemin MUTLAK geçişidir. Diğer eylemlerle
            # fark yalnız bunun gerçekten ayrışıp ayrışmadığını (güveni) ölçer.
            delta = float(stats.mean)
            contrast = float(stats.mean - other_mean)
            between = (stats.n * other_n / max(1, total.n)) * (stats.mean - other_mean) ** 2
            other_m2 = max(0.0, total.m2 - stats.m2 - between)
            other_var = other_m2 / max(1, other_n - 1)
            se = math.sqrt(stats.variance / stats.n + other_var / other_n)
            signal = abs(contrast) / max(1e-9, se)
            support = 1.0 - math.exp(-stats.n / 3.0)
            confidence = float(np.clip(0.05 + 0.90 * support * (1.0 - math.exp(-signal / 2.0)),
                                       0.0, 0.95))
            if confidence < min_confidence or abs(delta) <= 1e-10:
                continue
            out.append({"variable": name, "delta": delta, "confidence": confidence,
                        "support": int(stats.n),
                        "evidence": "intervention" if stats.interventions == stats.n else "association",
                        "source": "transition-table"})
        return out

    def require_prediction(self, state: Any, action: Any, **kwargs) -> dict:
        pred = self.predict(state, action, **kwargs)
        return {"status": "ok" if pred else "unresolved", "predictions": pred}

    def rollout(self, state: Any, actions, *, action_features=None) -> dict:
        """Öğrenilmiş modelle salt-okunur çok adımlı önizleme; gözlem yazmaz."""
        current = np.asarray(state, dtype=np.float64).reshape(-1).copy()
        rows = []
        for i, action in enumerate(actions or []):
            feats = action_features[i] if isinstance(action_features, (list, tuple)) and i < len(action_features) else None
            pred = self.predict(current, action, action_features=feats)
            if not pred:
                return {"status": "unresolved", "steps": rows}
            for p in pred:
                # Vektör adları f0... veya kullanıcı adları; yalnız mevcut boyuta yaz.
                name = p["variable"]
                if self.feature_names is not None and name in self.feature_names:
                    j = self.feature_names.index(name)
                elif name.startswith("f") and name[1:].isdigit():
                    j = int(name[1:])
                else:
                    continue
                if 0 <= j < current.size:
                    current[j] += float(p["delta"])
            rows.append({"action": action, "state": current.copy(), "predictions": pred})
        return {"status": "ok" if rows else "unresolved", "state": current, "steps": rows}

    def edges(self) -> list[dict]:
        """Şu an desteklenen eylem→değişken kenarlarını dışa ver."""
        out = []
        for key, stats_by_name in self.by_action.items():
            for name, stats in stats_by_name.items():
                if stats.n < self.min_support:
                    continue
                # Kenarın etkisini tahmin yolundan al; başka eylem yoksa kenar yok.
                if self.feature_names is not None:
                    sample_state = np.zeros(len(self.feature_names), dtype=np.float64)
                else:
                    sample_state = {name: 0.0 for name in self.global_stats}
                try:
                    pred = self.predict(sample_state, self.actions[key])
                except Exception:
                    pred = []
                for p in pred:
                    if p["variable"] == name:
                        out.append({"action": self.actions[key], "variable": name, **p})
        return out

    # ------------------------------------------------------------------ kalıcılık
    def state_dict(self) -> dict:
        return {"version": 1, "feature_names": self.feature_names,
                "min_support": self.min_support, "max_actions": self.max_actions,
                "max_features": self.max_features, "transitions": self.transitions,
                "rejected": self.rejected,
                "global": {k: v.dict() for k, v in self.global_stats.items()},
                "by_action": {k: {n: s.dict() for n, s in d.items()}
                               for k, d in self.by_action.items()},
                "linear": {k: v.dict() for k, v in self.linear.items()},
                "actions": encode(self.actions), "action_features": encode(self.action_features)}

    def load_state_dict(self, d: dict | None) -> None:
        if not d:
            return
        self.feature_names = list(d["feature_names"]) if d.get("feature_names") is not None else None
        self.min_support = int(d.get("min_support", self.min_support))
        self.max_actions = int(d.get("max_actions", self.max_actions))
        self.max_features = int(d.get("max_features", self.max_features))
        self.transitions = int(d.get("transitions", 0))
        self.rejected = int(d.get("rejected", 0))
        self.global_stats = {k: Summary.from_dict(v) for k, v in (d.get("global") or {}).items()}
        self.by_action = {k: {n: Summary.from_dict(s) for n, s in vals.items()}
                          for k, vals in (d.get("by_action") or {}).items()}
        self.linear = {k: LinearTransition.from_dict(v) for k, v in (d.get("linear") or {}).items()}
        self.actions = decode(d.get("actions") or {})
        self.action_features = decode(d.get("action_features") or {})
        self.last_record = None


__all__ = ["Summary", "LinearTransition", "CausalTransitionModel"]
