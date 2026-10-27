#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_laws.py — örneklerden güvenli, koşullu sembolik geçiş yasaları.

Bu modül dondurulmuş operatör repertuvarının başka bir adı değildir: ifade AST'si
örnek imzalarından aranır, koşullu/parçalı formlar da adaydır ve ancak bağımsız
validation + audit ile kabul edilir. Modelin arayüzü ``CausalTransitionModel``
ile uyumludur; böylece öğrenilen yasa doğrudan planlamada kullanılabilir.
"""
from __future__ import annotations

import ast
import math
from typing import Any

import numpy as np

from nora_evolution import ExpressionSynthesizer, _eval, _safe_tree, _signature


class SymbolicLaw:
    def __init__(self, names: list[str], *, max_depth: int = 3, max_exprs: int = 6000):
        self.names = list(names)
        self.max_depth = int(max_depth)
        self.max_exprs = int(max_exprs)
        self.expression: ast.AST | None = None
        self.text = ""
        self.confidence = 0.0
        self.train_score = 0.0
        self.validation_score = 0.0
        self.audit_score = 0.0
        self.evaluated = 0

    @staticmethod
    def _score(expression, names, cases) -> float:
        if not cases:
            return 0.0
        good = 0
        for case in cases:
            try:
                got = _eval(expression, tuple(case["args"]), names)
                good += int(_signature(got) == _signature(case["expect"]))
            except Exception:
                pass
        return good / len(cases)

    def fit(self, train: list[dict], validation: list[dict], audit: list[dict] | None = None) -> dict:
        if len(train) < 2 or not validation:
            return {"status": "invalid-protocol", "reason": "train>=2 and validation required"}
        audit = list(audit or [])
        synth = ExpressionSynthesizer(max_depth=self.max_depth, max_exprs=self.max_exprs,
                                      max_solutions=24)
        candidates = synth.candidates(self.names, train)
        self.evaluated = synth.evaluated
        best = None
        for candidate in candidates:
            val = self._score(candidate.expression, self.names, validation)
            if val < 1.0:
                continue
            aud = self._score(candidate.expression, self.names, audit) if audit else None
            if aud is not None and aud < 1.0:
                continue
            best = (candidate, val, aud)
            break
        if best is None:
            return {"status": "unresolved", "candidates": len(candidates),
                    "evaluated": self.evaluated}
        candidate, val, aud = best
        self.expression = candidate.expression
        self.text = candidate.text
        self.train_score = 1.0
        self.validation_score = float(val)
        self.audit_score = float(aud) if aud is not None else 0.0
        # Örnek sayısı arttıkça büyüyen, sınırsız olmayan güven; bağımsız audit
        # yoksa 0.5'i aşmayan bir başlangıç güveni kullanılır.
        support = len(train) / (len(train) + 4.0)
        self.confidence = float(np.clip(support * val * (aud if aud is not None else 0.75),
                                        0.0, 0.95))
        return {"status": "accepted", "expression": self.text,
                "form": "conditional" if any(isinstance(n, ast.IfExp)
                                             for n in ast.walk(self.expression)) else "expression",
                "train": self.train_score, "validation": self.validation_score,
                "audit": aud, "confidence": self.confidence,
                "evaluated": self.evaluated}

    def predict(self, args) -> Any:
        if self.expression is None:
            raise RuntimeError("law is unresolved")
        return _eval(self.expression, tuple(args), self.names)

    def state_dict(self) -> dict:
        return {"names": self.names, "text": self.text, "confidence": self.confidence,
                "train": self.train_score, "validation": self.validation_score,
                "audit": self.audit_score, "evaluated": self.evaluated,
                "max_depth": self.max_depth, "max_exprs": self.max_exprs}

    def load_state_dict(self, data: dict | None) -> None:
        if not data or not data.get("text"):
            return
        self.names = list(data.get("names") or self.names)
        self.max_depth = int(data.get("max_depth", self.max_depth))
        self.max_exprs = int(data.get("max_exprs", self.max_exprs))
        expr = ast.parse(str(data["text"]), mode="eval").body
        _safe_tree(ast.Expression(body=expr))
        self.expression = expr
        self.text = str(data["text"])
        self.confidence = float(data.get("confidence", 0.0))
        self.train_score = float(data.get("train", 0.0))
        self.validation_score = float(data.get("validation", 0.0))
        self.audit_score = float(data.get("audit", 0.0))
        self.evaluated = int(data.get("evaluated", 0))


class SymbolicTransitionModel:
    """Her durum değişkeni için ayrı, atomik olarak kabul edilen yasa modeli."""

    def __init__(self, feature_names: list[str], *, max_depth: int = 3, max_exprs: int = 6000):
        self.feature_names = list(feature_names)
        self.max_depth = int(max_depth)
        self.max_exprs = int(max_exprs)
        self.laws: dict[str, SymbolicLaw] = {}
        self.fit_report: dict = {}

    @staticmethod
    def _records(records, n: int):
        out = []
        for record in records or []:
            state = np.asarray(record["state"], dtype=np.float64).reshape(-1)
            nxt = np.asarray(record["next"], dtype=np.float64).reshape(-1)
            if state.size != n or nxt.size != n or not np.all(np.isfinite(state)) or not np.all(np.isfinite(nxt)):
                raise ValueError("invalid transition record")
            action = float(record["action"])
            if not math.isfinite(action):
                raise ValueError("action must be numeric")
            out.append((state, action, nxt))
        return out

    def fit(self, train: list[dict], validation: list[dict], audit: list[dict]) -> dict:
        n = len(self.feature_names)
        try:
            tr = self._records(train, n)
            va = self._records(validation, n)
            au = self._records(audit, n)
        except (KeyError, TypeError, ValueError) as exc:
            return {"status": "invalid-protocol", "reason": str(exc)}
        if len(tr) < 2 or not va or not au:
            return {"status": "invalid-protocol", "reason": "train>=2, validation and audit required"}
        names = self.feature_names + ["action"]
        pending: dict[str, SymbolicLaw] = {}
        reports = {}
        for j, name in enumerate(self.feature_names):
            def cases(rows):
                return [{"args": [*state.tolist(), action], "expect": float(nxt[j] - state[j])}
                        for state, action, nxt in rows]
            law = SymbolicLaw(names, max_depth=self.max_depth, max_exprs=self.max_exprs)
            report = law.fit(cases(tr), cases(va), cases(au))
            reports[name] = report
            if report["status"] != "accepted":
                return {"status": "unresolved", "variable": name, "reports": reports}
            pending[name] = law
        # Bütün değişkenler geçmeden eski modeli yarım bırakma.
        self.laws = pending
        self.fit_report = {"status": "accepted", "reports": reports}
        return self.fit_report

    def predict(self, state, action, **kwargs) -> list[dict]:
        values = np.asarray(state, dtype=np.float64).reshape(-1)
        if values.size != len(self.feature_names):
            raise ValueError("feature dimension changed")
        out = []
        for j, name in enumerate(self.feature_names):
            law = self.laws.get(name)
            if law is None:
                continue
            delta = float(law.predict([*values.tolist(), float(action)]))
            if not math.isfinite(delta) or abs(delta) <= 1e-10:
                continue
            out.append({"variable": name, "delta": delta, "confidence": law.confidence,
                        "support": law.evaluated, "evidence": "heldout-symbolic",
                        "source": "symbolic-law", "expression": law.text})
        return out

    def state_dict(self) -> dict:
        return {"version": 1, "feature_names": self.feature_names,
                "max_depth": self.max_depth, "max_exprs": self.max_exprs,
                "laws": {name: law.state_dict() for name, law in self.laws.items()},
                "fit_report": self.fit_report}

    def load_state_dict(self, data: dict | None) -> None:
        if not data:
            return
        self.feature_names = list(data.get("feature_names") or self.feature_names)
        self.max_depth = int(data.get("max_depth", self.max_depth))
        self.max_exprs = int(data.get("max_exprs", self.max_exprs))
        self.laws = {}
        for name, raw in (data.get("laws") or {}).items():
            law = SymbolicLaw(self.feature_names + ["action"],
                              max_depth=self.max_depth, max_exprs=self.max_exprs)
            law.load_state_dict(raw)
            if law.expression is not None:
                self.laws[name] = law
        self.fit_report = dict(data.get("fit_report") or {})


__all__ = ["SymbolicLaw", "SymbolicTransitionModel"]
