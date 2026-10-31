#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_architecture.py — canlı, çalıştırılabilir öz-mimari modeli.

``ArchitectureModel`` bir organlar listesi değildir. Kök nesnenin o anki
alanlarını tarar, yetenek bildiren gerçek çağrılabilirleri keşfeder, girdi/çıktı
türleri üzerinden yeni iş zincirleri kurar ve bunları çalıştırır. Sonradan köke
bağlanan bir organ için bu dosyada yeni bir ``if`` veya sınıf adı gerekmez.

Sınırlar da açıktır: yetenek etiketi bir arayüz sözleşmesidir; model, etiketi
yazan kodun doğru olduğunu varsaymaz. Her çağrı ayrı kaydedilir, istisna başarı
sayılmaz ve isteğe bağlı çıktı doğrulayıcıları çalıştırılır. Kaynak AST'si,
çalışan nesnenin yerine geçmez; ikisini karşılaştırmak için kullanılır.
"""
from __future__ import annotations

from nora._paths import REPO_ROOT


import ast
from dataclasses import dataclass, field
import inspect
import itertools
import os
import time
from typing import Any, Callable


def capability(inputs, outputs=None, *, name: str | None = None,
               cost: float = 1.0, validator: Callable | None = None):
    """Bir metodu mimari modelin keşfedebileceği yetenek olarak işaretle.

    Kısa kullanım ``@capability("text", "number")``; çoklu girdi/çıktı için
    ``@capability(("text", "number"), ("result",))``. Türler anlamsal arayüz
    adlarıdır, Python sınıfı olmak zorunda değildir. ``validator`` çıktı gerçekten
    sözleşmeye uyuyor mu diye çalışma zamanında çağrılır.
    """
    if outputs is None:
        inputs, outputs = (), (inputs,)
    elif isinstance(outputs, (str, bytes)):
        outputs = (outputs,)
    if isinstance(inputs, (str, bytes)):
        inputs = (inputs,)
    ins = tuple(str(x) for x in (inputs or ()) if str(x))
    outs = tuple(str(x) for x in (outputs or ()) if str(x))
    if not outs:
        raise ValueError("capability needs an output type")

    def decorate(fn):
        setattr(fn, "__nora_capability__", {
            "inputs": ins, "outputs": outs, "name": name or fn.__name__,
            "cost": float(cost), "validator": validator,
        })
        return fn
    return decorate


@dataclass
class Capability:
    node: str
    method: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    cost: float = 1.0
    name: str = ""
    validator: Callable | None = field(default=None, repr=False, compare=False)

    @property
    def key(self) -> str:
        return f"{self.node}.{self.method}"

    def describe(self) -> dict:
        return {"node": self.node, "method": self.method, "key": self.key,
                "inputs": list(self.inputs), "outputs": list(self.outputs),
                "cost": float(self.cost), "name": self.name or self.method}


@dataclass
class Node:
    name: str
    type_name: str
    module: str
    source: str = ""
    methods: tuple[str, ...] = ()

    def describe(self) -> dict:
        return {"name": self.name, "type": self.type_name, "module": self.module,
                "source": self.source, "methods": list(self.methods)}


class ArchitectureModel:
    """Kök nesnenin canlı organ ve yetenek grafiği."""

    MAX_PLAN_DEPTH = 8
    MAX_PLAN_STATES = 512

    def __init__(self, root: Any, bus=None):
        self.root = root
        self.bus = bus
        self.nodes: dict[str, Node] = {}
        self._objects: dict[str, Any] = {}
        self.capabilities: dict[str, Capability] = {}
        self._manual: dict[str, Any] = {}
        self.executions: list[dict] = []
        self._exec_ids = itertools.count(1)
        self._invariants: dict[str, Callable] = {}
        self._ast_cache: dict[str, tuple[int, dict]] = {}
        self.refresh()

    # ------------------------------------------------------------------ keşif
    @staticmethod
    def _is_object(value: Any) -> bool:
        if value is None or isinstance(value, (str, bytes, int, float, bool,
                                               list, tuple, dict, set, frozenset)):
            return False
        # Rastgelelik üreteci, kilit, modül gibi altyapı nesneleri organ değildir.
        if inspect.ismodule(value) or inspect.isroutine(value):
            return False
        return hasattr(value, "__dict__") or hasattr(type(value), "__dict__")

    @staticmethod
    def _public_methods(obj: Any) -> tuple[str, ...]:
        names: set[str] = set()
        for cls in type(obj).__mro__:
            for name, value in vars(cls).items():
                if name.startswith("_"):
                    continue
                if isinstance(value, (staticmethod, classmethod)):
                    value = value.__func__
                if callable(value):
                    names.add(name)
        for name, value in getattr(obj, "__dict__", {}).items():
            if not name.startswith("_") and callable(value):
                names.add(name)
        return tuple(sorted(names))

    @staticmethod
    def _source_of(obj: Any) -> str:
        try:
            return os.path.abspath(inspect.getsourcefile(type(obj)) or "")
        except (TypeError, OSError):
            return ""

    def register(self, name: str, obj: Any) -> str:
        """Çalışma zamanında bir organı modele ekle; kök alanı da güncellenir."""
        key = str(name)
        if not key or key.startswith("_") or "." in key:
            raise ValueError("invalid organ name")
        self._manual[key] = obj
        setattr(self.root, key, obj)
        self.refresh()
        return key

    def unregister(self, name: str) -> bool:
        key = str(name)
        existed = key in self._manual or key in self._objects
        self._manual.pop(key, None)
        if getattr(self.root, key, None) is not None:
            try:
                delattr(self.root, key)
            except (AttributeError, TypeError):
                pass
        self.refresh()
        return existed

    def refresh(self) -> None:
        candidates: dict[str, Any] = dict(self._manual)
        for name, value in vars(self.root).items():
            if name.startswith("_") or name in {"bus", "architecture"}:
                continue
            if self._is_object(value):
                candidates.setdefault(name, value)
        self.nodes.clear()
        self._objects.clear()
        self.capabilities.clear()
        for name, obj in sorted(candidates.items()):
            methods = self._public_methods(obj)
            self._objects[name] = obj
            self.nodes[name] = Node(name, type(obj).__name__,
                                    type(obj).__module__, self._source_of(obj), methods)
            for method_name in methods:
                try:
                    method = getattr(obj, method_name)
                    meta = getattr(method, "__nora_capability__", None)
                except Exception:
                    continue
                if not meta:
                    continue
                cap = Capability(name, method_name, tuple(meta.get("inputs") or ()),
                                 tuple(meta.get("outputs") or ()),
                                 float(meta.get("cost", 1.0)),
                                 str(meta.get("name") or method_name),
                                 meta.get("validator"))
                self.capabilities[cap.key] = cap
        if self.bus is not None:
            sync = getattr(self.bus, "sync_modules", None)
            if callable(sync):
                sync(self._objects)

    def node(self, name: str) -> Node | None:
        return self.nodes.get(str(name))

    def describe(self) -> dict:
        return {"nodes": [n.describe() for n in self.nodes.values()],
                "capabilities": [c.describe() for c in self.capabilities.values()],
                "execution_count": len(self.executions),
                "failed_executions": sum(r.get("status") == "error" for r in self.executions)}

    # ------------------------------------------------------------------ planlama
    def plan(self, available, goal: str, *, max_depth: int | None = None,
             excluded=()) -> list[Capability] | None:
        """Tip kümesinden hedef türe en düşük maliyetli kısa zinciri ara.

        Arama yalnız kayıtlı arayüzleri kullanır; belirli organ veya görev adı
        içermez. Aynı tip kümesine daha pahalı yoldan gelen dallar budanır.
        """
        start = frozenset(str(x) for x in (available or ()))
        target = str(goal)
        blocked = frozenset(str(x) for x in (excluded or ()))
        if target in start:
            return []
        limit = min(self.MAX_PLAN_DEPTH, int(max_depth or self.MAX_PLAN_DEPTH))
        queue: list[tuple[frozenset[str], tuple[Capability, ...], float]] = [(start, (), 0.0)]
        best: dict[frozenset[str], float] = {start: 0.0}
        expanded = 0
        while queue and expanded < self.MAX_PLAN_STATES:
            queue.sort(key=lambda x: (x[2], len(x[1])))
            types, path, cost = queue.pop(0)
            expanded += 1
            if len(path) >= limit:
                continue
            for cap in self.capabilities.values():
                if cap.node in blocked:
                    continue
                if cap in path or not set(cap.inputs).issubset(types):
                    continue
                new_types = frozenset(set(types) | set(cap.outputs))
                new_cost = cost + max(0.001, float(cap.cost))
                if new_cost >= best.get(new_types, float("inf")):
                    continue
                new_path = path + (cap,)
                if target in new_types:
                    return list(new_path)
                best[new_types] = new_cost
                queue.append((new_types, new_path, new_cost))
        return None

    # ------------------------------------------------------------------ çalıştırma
    def _record(self, cap: Capability | None, status: str, *, inputs=None,
                outputs=None, error: str = "", parents=()) -> dict:
        rec = {"id": f"e{next(self._exec_ids)}",
               "capability": cap.key if cap else "",
               "node": cap.node if cap else "", "method": cap.method if cap else "",
               "status": str(status), "inputs": list(inputs or []),
               "outputs": list(outputs or []), "error": str(error),
               "parents": list(parents or ()), "time_ns": time.time_ns()}
        self.executions.append(rec)
        self.executions = self.executions[-1024:]
        if self.bus is not None:
            try:
                self.bus.publish("architecture.execution", dict(rec),
                                 source="architecture", kind="execution",
                                 parents=tuple(rec["parents"]))
            except Exception:
                pass
        return rec

    def invoke(self, key: str, *args, **kwargs) -> dict:
        cap = self.capabilities.get(str(key))
        if cap is None:
            return {"status": "error", "error": "capability not found", "value": None}
        obj = self._objects.get(cap.node)
        try:
            fn = getattr(obj, cap.method)
            value = fn(*args, **kwargs)
            values = (value,) if len(cap.outputs) == 1 else tuple(value)
            if len(values) != len(cap.outputs):
                raise ValueError("capability output arity mismatch")
            if cap.validator is not None and not bool(cap.validator(value)):
                raise ValueError("capability output rejected")
            rec = self._record(cap, "ok", inputs=cap.inputs, outputs=cap.outputs)
            return {"status": "ok", "value": value, "record": rec}
        except Exception as exc:
            rec = self._record(cap, "error", inputs=cap.inputs,
                               error=f"{type(exc).__name__}: {exc}")
            return {"status": "error", "value": None, "record": rec,
                    "error": str(exc)}

    def predict_removal(self, node_name, values: dict[str, Any], goal: str) -> dict:
        """Bir organı veya organ kombinasyonunu çıkarma sonucunu önceden yanıtla.

        Bu yalnız tip-erişilebilirliği tahminidir; çalıştırma veya doğrulama
        başarısını uydurmaz. Kombinasyon, tekil sonuçların üzerine etiketle
        yazılmaz; aynı grafik araması bütün çıkarılan düğümleri birlikte dışlar.
        """
        names = ([str(node_name)] if isinstance(node_name, (str, bytes))
                 else [str(name) for name in (node_name or ())])
        if not names or len(set(names)) != len(names):
            return {"status": "error", "error": "invalid node combination", "removed_nodes": names}
        unknown = [name for name in names if name not in self.nodes]
        if unknown:
            return {"status": "error", "error": "unknown node", "unknown": unknown,
                    "removed_nodes": names}
        baseline = self.plan(values.keys(), goal)
        remaining = self.plan(values.keys(), goal, excluded=names)
        return {
            "status": "unresolved" if remaining is None else "ok",
            "removed": names[0] if len(names) == 1 else list(names),
            "removed_nodes": names, "goal": str(goal), "basis": "interface_graph",
            "baseline_plan": [cap.key for cap in baseline or []],
            "plan": [cap.key for cap in remaining or []],
        }

    def run(self, values: dict[str, Any], goal: str) -> dict:
        """Tip->değer haritasından hedefe giden zinciri gerçekten çalıştır."""
        path = self.plan(values.keys(), goal)
        if path is None:
            return {"status": "unresolved", "value": None, "plan": [], "records": []}
        current = dict(values)
        records: list[dict] = []
        for cap in path:
            obj = self._objects.get(cap.node)
            args = [current[t] for t in cap.inputs]
            try:
                fn = getattr(obj, cap.method)
                value = fn(*args)
                vals = (value,) if len(cap.outputs) == 1 else tuple(value)
                if len(vals) != len(cap.outputs):
                    raise ValueError("capability output arity mismatch")
                if cap.validator is not None and not bool(cap.validator(value)):
                    raise ValueError("capability output rejected")
            except Exception as exc:
                rec = self._record(cap, "error", inputs=cap.inputs,
                                   error=f"{type(exc).__name__}: {exc}",
                                   parents=[r["id"] for r in records[-1:]])
                records.append(rec)
                return {"status": "error", "value": None,
                        "plan": [c.key for c in path], "records": records,
                        "error": str(exc)}
            for typ, val in zip(cap.outputs, vals):
                current[typ] = val
            rec = self._record(cap, "ok", inputs=cap.inputs, outputs=cap.outputs,
                               parents=[r["id"] for r in records[-1:]])
            records.append(rec)
        return {"status": "ok", "value": current.get(goal), "values": current,
                "plan": [c.key for c in path], "records": records}

    # ------------------------------------------------------------------ öz-inceleme
    def register_invariant(self, name: str, check: Callable[["ArchitectureModel"], Any]) -> None:
        if not callable(check):
            raise TypeError("invariant must be callable")
        self._invariants[str(name)] = check

    def check_invariants(self) -> dict:
        out = {}
        for name, check in self._invariants.items():
            try:
                result = check(self)
                out[name] = {"ok": bool(result), "detail": result if not isinstance(result, bool) else ""}
            except Exception as exc:
                out[name] = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
        return out

    def source_index(self, node_name: str) -> dict:
        """Organın kaynak AST'sinden sınıf/metot/çağrı özetini çıkar."""
        node = self.nodes.get(str(node_name))
        if node is None or not node.source or not os.path.exists(node.source):
            return {"node": str(node_name), "status": "unavailable"}
        path = node.source
        try:
            mtime = os.path.getmtime(path)
            cached = self._ast_cache.get(path)
            if cached and cached[0] == mtime:
                tree_data = cached[1]
            else:
                with open(path, "r", encoding="utf-8") as fh:
                    tree = ast.parse(fh.read(), filename=path)
                classes = []
                for item in ast.walk(tree):
                    if isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        calls = sorted({n.func.id for n in ast.walk(item)
                                        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)})
                        classes.append({"kind": type(item).__name__, "name": item.name,
                                        "line": int(getattr(item, "lineno", 0)),
                                        "calls": calls})
                tree_data = {"file": path, "definitions": classes}
                self._ast_cache[path] = (mtime, tree_data)
            return {"node": node.name, "status": "ok", **tree_data}
        except (OSError, SyntaxError) as exc:
            return {"node": node.name, "status": "error", "error": str(exc)}

    def state_dict(self) -> dict:
        return {"nodes": [n.describe() for n in self.nodes.values()],
                "capabilities": [c.describe() for c in self.capabilities.values()],
                "executions": [dict(r) for r in self.executions[-128:]]}


__all__ = ["capability", "Capability", "Node", "ArchitectureModel"]
