#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_planning.py — öğrenilen geçiş modeli üzerinde uyarlanır beam planlama.

Planlayıcı eylem/hedef adı bilmez. Geçişleri ``CausalTransitionModel``den alır;
hedef, çağrılabilir bir erişim/ölçüt veya özellik->hedef haritasıdır. Her adımın
güveni birikir; belirsizlik bütçeyi aşarsa derinleşmek yerine durur. Hedefe
varmayan en iyi yol ``partial``dir, ``ok`` değildir.
"""
from __future__ import annotations

from nora._paths import REPO_ROOT


from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np


@dataclass
class PlanNode:
    state: np.ndarray
    actions: tuple[Any, ...]
    states: tuple[np.ndarray, ...]
    score: float
    confidence: float
    depth: int


class AdaptivePlanner:
    def __init__(self, model, *, beam: int = 12, max_horizon: int = 8,
                 min_confidence: float = 0.08, max_expansions: int = 512,
                 monitor=None):
        self.model = model
        self.beam = max(2, int(beam))
        self.max_horizon = max(1, int(max_horizon))
        self.min_confidence = float(min_confidence)
        self.max_expansions = max(16, int(max_expansions))
        # İşlevsel üst-biliş yalnız rapor üretmez: geçmiş tahmin hatasıyla
        # kalibre edilen güven, aynı genel eşik üzerinden plan uzayına girer.
        # Monitor yoksa eski ham-model davranışı aynen korunur.
        self.monitor = monitor
        self.last_trace: dict = {}

    def _index(self, variable: str) -> int | None:
        names = getattr(self.model, "feature_names", None) or []
        if variable in names:
            return names.index(variable)
        if variable.startswith("f") and variable[1:].isdigit():
            index = int(variable[1:])
            return index if index < len(names) else None
        return None

    def _step(self, state: np.ndarray, action: Any) -> tuple[np.ndarray, list[dict], float] | None:
        pred = self.model.predict(state, action)
        if self.monitor is not None:
            # Kalibrasyon yeni satır döndürür; öğrenilen dünya modelini veya ham
            # güveni yerinde değiştirmez. Böylece ham-model kontrolü mümkündür.
            pred = self.monitor.calibrate(pred)
        if not pred:
            return None
        nxt = state.copy()
        used = 0
        for p in pred:
            index = self._index(str(p["variable"]))
            if index is None or not 0 <= index < nxt.size:
                continue
            delta = float(p["delta"])
            if not math.isfinite(delta):
                return None
            nxt[index] += delta
            used += 1
        if not used:
            return None
        confidence = float(min(float(p["confidence"]) for p in pred))
        return nxt, pred, confidence

    def _goal_test(self, state: np.ndarray, goal) -> bool:
        if callable(goal):
            return bool(goal(state))
        if isinstance(goal, dict):
            for name, spec in goal.items():
                index = self._index(str(name))
                if index is None:
                    return False
                if isinstance(spec, dict):
                    target = float(spec.get("target", 0.0))
                    tol = float(spec.get("tol", spec.get("tolerance", 1e-6)))
                else:
                    target, tol = float(spec), 1e-6
                if abs(float(state[index]) - target) > tol:
                    return False
            return True
        raise TypeError("goal must be callable or a feature mapping")

    def _score(self, state: np.ndarray, goal, scoring: Callable | None) -> float:
        if scoring is not None:
            return float(scoring(state))
        if callable(goal) and not isinstance(goal, dict):
            # Erişim fonksiyonu skor vermiyorsa yalnız erişim 0/1 ile yönlendirilir;
            # kullanıcı daha zengin yönlendirme için scoring vermelidir.
            return float(bool(goal(state)))
        if isinstance(goal, dict):
            total = 0.0
            for name, spec in goal.items():
                index = self._index(str(name))
                if index is None:
                    return float("-inf")
                if isinstance(spec, dict):
                    target = float(spec.get("target", 0.0))
                    weight = float(spec.get("weight", 1.0))
                else:
                    target, weight = float(spec), 1.0
                total -= weight * (float(state[index]) - target) ** 2
            return total
        return 0.0

    def plan(self, state, actions, goal, *, scoring: Callable | None = None,
             horizon: int | None = None) -> dict:
        initial = np.asarray(state, dtype=np.float64).reshape(-1).copy()
        actions = tuple(actions or ())
        if not actions:
            return {"status": "unresolved", "reason": "no actions", "actions": []}
        try:
            if self._goal_test(initial, goal):
                return {"status": "ok", "actions": [], "states": [initial],
                        "horizon": 0, "confidence": 1.0, "expanded": 0,
                        "stop_reason": "already-at-goal"}
        except (TypeError, ValueError) as exc:
            return {"status": "invalid-protocol", "reason": str(exc), "actions": []}
        limit = min(self.max_horizon, int(horizon or self.max_horizon))
        root = PlanNode(initial, (), (initial.copy(),), self._score(initial, goal, scoring), 1.0, 0)
        beam = [root]
        best = root
        expanded = 0
        stall = 0
        stop = "horizon"
        for depth in range(1, limit + 1):
            candidates: list[PlanNode] = []
            for node in beam:
                for action in actions:
                    if expanded >= self.max_expansions:
                        stop = "resource-budget"
                        break
                    expanded += 1
                    step = self._step(node.state, action)
                    if step is None:
                        continue
                    nxt, pred, conf = step
                    total_conf = node.confidence * conf
                    if total_conf < self.min_confidence:
                        continue
                    path = node.actions + (action,)
                    states = node.states + (nxt.copy(),)
                    score = self._score(nxt, goal, scoring)
                    child = PlanNode(nxt, path, states, score, total_conf, depth)
                    candidates.append(child)
                    if self._goal_test(nxt, goal):
                        result = {"status": "ok", "actions": list(path),
                                  "states": list(states), "horizon": depth,
                                  "confidence": total_conf, "expanded": expanded,
                                  "stop_reason": "goal-reached", "score": score}
                        self.last_trace = result
                        return result
                    if (score, total_conf, -depth) > (best.score, best.confidence, -best.depth):
                        best = child
                if expanded >= self.max_expansions:
                    break
            if not candidates:
                stop = "no-supported-transition" if expanded < self.max_expansions else stop
                break
            # Güven azaldıkça daralt: düşük kanıtla binlerce dallanma yapmak yerine
            # en iyi birkaç yolu tut. Bu hesap yalnız adayların gerçek güveninden gelir.
            mean_conf = float(np.mean([c.confidence for c in candidates]))
            width = max(2, min(self.beam, int(round(self.beam * (0.35 + 0.65 * mean_conf)))))
            candidates.sort(key=lambda c: (-c.score, -c.confidence, len(c.actions)))
            beam = candidates[:width]
            if best.score <= root.score + 1e-12:
                stall += 1
            else:
                stall = 0
            if stall >= 2:
                stop = "no-goal-progress"
                break
            if expanded >= self.max_expansions:
                stop = "resource-budget"
                break
        status = "partial" if best.actions and best.score > root.score else "unresolved"
        result = {"status": status, "actions": list(best.actions), "states": list(best.states),
                  "horizon": len(best.actions), "confidence": best.confidence,
                  "expanded": expanded, "stop_reason": stop, "score": best.score,
                  "initial_score": root.score}
        self.last_trace = result
        return result

    def affordances(self, state, actions, goal, *, scoring: Callable | None = None) -> list[dict]:
        """Bir adımlık, öğrenilmiş eylem olanaklarını hedef ilerlemesine göre sırala."""
        current = np.asarray(state, dtype=np.float64).reshape(-1).copy()
        base = self._score(current, goal, scoring)
        out = []
        for action in actions or ():
            step = self._step(current, action)
            if step is None:
                continue
            nxt, pred, confidence = step
            out.append({"action": action, "progress": self._score(nxt, goal, scoring) - base,
                        "confidence": confidence, "state": nxt, "predictions": pred})
        out.sort(key=lambda x: (-x["progress"], -x["confidence"]))
        return out


__all__ = ["AdaptivePlanner", "PlanNode"]
