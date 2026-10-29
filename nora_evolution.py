#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_evolution.py — ölçüm-kapılı kaynak kodu öz-değişimi.

Bu modül çalışan beyni keyfî olarak yeniden yazmaz. Bir ``SourceWorkspace``
içindeki hedef fonksiyon için örneklerden AST ifadesi arar, kaynağı atomik olarak
değiştirir, ayrı Python sürecinde çalıştırır ve bağımsız validation/audit
başarısızsa dosyayı geri yükler. Başarısız aday "öğrendim" diye kaydedilmez.

Öncelik/öğrenme ayrımı açıktır: AST operatörleri, güvenli built-in'ler ve arama
bütçesi insan-yazımı **prior**; ifade imzaları, eğitim örneklerini açıklama ve
kaynak değişiminin kabulü çalıştırma ölçümüne dayanan **operatör/sonuç**tur.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import difflib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from typing import Any, Iterable


# --------------------------------------------------------------------------- güvenli ifade
_SAFE_BUILTINS = {
    "len": len, "str": str, "int": int, "float": float, "bool": bool,
    "abs": abs, "round": round, "sum": sum, "min": min, "max": max,
    "sorted": sorted,
}
_SAFE_METHODS = {"upper", "lower", "strip", "title"}
_ALLOWED_NODES = (
    ast.Expression, ast.Name, ast.Load, ast.Constant, ast.Tuple, ast.List,
    ast.Attribute,
    ast.UnaryOp, ast.UAdd, ast.USub, ast.Not, ast.BinOp, ast.Add, ast.Sub,
    ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.Compare, ast.Eq,
    ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.BoolOp, ast.And, ast.Or,
    ast.IfExp, ast.Call, ast.Subscript, ast.Slice,
)


def _safe_tree(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"unsafe AST node: {type(node).__name__}")
        if isinstance(node, ast.Call):
            named = isinstance(node.func, ast.Name) and node.func.id in _SAFE_BUILTINS
            method = isinstance(node.func, ast.Attribute) and node.func.attr in _SAFE_METHODS
            if not (named or method):
                raise ValueError("unsafe call")
            if node.keywords or (method and node.args):
                raise ValueError("call arguments are not in the safe method subset")
        if isinstance(node, ast.Attribute) and node.attr not in _SAFE_METHODS:
            raise ValueError("unsafe attribute")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("private name")
        if isinstance(node, ast.Constant) and isinstance(node.value, (bytes, complex)):
            raise ValueError("unsupported constant")


def _signature(value: Any) -> str:
    # Türü de imzaya kat: True ile 1 aynı program davranışı sayılmaz.
    return f"{type(value).__name__}:{repr(value)}"


def _eval(expr: ast.AST, args: tuple[Any, ...], names: list[str]) -> Any:
    tree = ast.Expression(body=expr)
    ast.fix_missing_locations(tree)
    _safe_tree(tree)
    env = dict(zip(names, args))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        code = compile(tree, "<nora-expression>", "eval")
    return eval(code, {"__builtins__": {}}, {**_SAFE_BUILTINS, **env})


@dataclass
class ExpressionCandidate:
    expression: ast.AST
    cost: int
    train_score: float

    @property
    def text(self) -> str:
        return ast.unparse(self.expression)


class ExpressionSynthesizer:
    """Örnek imzalarını kullanan küçük, tür-güvenli enumeratif sentezci."""

    def __init__(self, *, max_depth: int = 4, max_exprs: int = 5000,
                 max_solutions: int = 12):
        self.max_depth = max(1, int(max_depth))
        self.max_exprs = max(100, int(max_exprs))
        self.max_solutions = max(1, int(max_solutions))
        self.evaluated = 0
        self.pruned = 0

    @staticmethod
    def _constants(cases) -> list[Any]:
        vals = {0, 1, -1, 2}
        # Veriyle gelen küçük sayıları da aday yap; hedef cevap kaynak koda yazılmaz.
        def visit(x):
            if isinstance(x, bool):
                return
            if isinstance(x, int) and abs(x) <= 32:
                vals.add(x)
            elif isinstance(x, float) and abs(x) <= 32 and x.is_integer():
                vals.add(int(x))
            elif isinstance(x, (list, tuple)):
                for y in x:
                    visit(y)
        for case in cases:
            for arg in case.get("args", []):
                visit(arg)
            visit(case.get("expect"))
        return sorted(vals)

    def _terminal(self, value: Any) -> ast.AST:
        return ast.Constant(value=value)

    def candidates(self, names: list[str], cases: list[dict], *,
                   incumbents=()) -> list[ExpressionCandidate]:
        if not names or not cases:
            return []
        # Arama bütçesi çağrı başınadır; önceki yamanın harcadığı bütçe ikinci
        # yapısal adımı daha başlamadan tüketmemelidir.
        self.evaluated = 0
        self.pruned = 0
        expected = tuple(_signature(c.get("expect")) for c in cases)
        # Her semantik imzada en ucuz ifadeyi tut: aynı davranışın yüzlerce
        # sözdizimsel varyantı aramayı şişirmesin.
        pools: dict[str, ast.AST] = {}
        ordered: list[ast.AST] = []
        value_cache: dict[str, tuple[Any, ...]] = {}

        def offer(expr: ast.AST, depth: int) -> tuple[bool, bool]:
            """(havuzda yeni mi, eğitim hedefi mi) döndür."""
            if self.evaluated >= self.max_exprs:
                return False, False
            self.evaluated += 1
            try:
                vals = tuple(_eval(expr, tuple(c["args"]), names) for c in cases)
                value_cache[ast.dump(expr, include_attributes=False)] = vals
                sig = tuple(_signature(v) for v in vals)
            except Exception:
                self.pruned += 1
                return False, False
            if sig == expected:
                return True, True
            # Semantik imza + sonuç türleri aynıysa pahalı/tekrar ifadeyi ele.
            key = repr((sig, depth))
            # Aynı imzanın daha derin kopyasını saklamak zincir kurmayı engeller;
            # imza başına tek ifade yeterlidir (maliyet sırasıyla geliyor).
            key = repr(sig)
            if key in pools:
                return False, False
            pools[key] = expr
            ordered.append(expr)
            return True, False

        terminals = [ast.Name(id=n, ctx=ast.Load()) for n in names]
        terminals += [self._terminal(v) for v in self._constants(cases)]
        # Kabul edilmiş mevcut AST, çok adımlı yamada yeni yapının ilkel girdisi
        # olabilir. Böylece ikinci yama birincinin davranışını genişletebilir.
        terminals += [expr for expr in (incumbents or ()) if isinstance(expr, ast.AST)]
        solutions: list[ExpressionCandidate] = []
        for expr in terminals:
            fresh, hit = offer(expr, 0)
            if hit:
                solutions.append(ExpressionCandidate(expr, 0, 1.0))
        # Derinlik katmanları: her katmanda o ana kadarki ifadelerden yeni güvenli
        # operatörler kur. Sonuçlar eğitim örneklerinde fiilen değerlendirilir.
        for depth in range(1, self.max_depth + 1):
            current = list(ordered)
            if not current or self.evaluated >= self.max_exprs:
                break
            # Tekli dönüşümler.
            for a in current:
                unary = []
                unary += [ast.UnaryOp(op=ast.USub(), operand=a),
                          ast.UnaryOp(op=ast.Not(), operand=a)]
                unary += [ast.Call(func=ast.Name(id=f, ctx=ast.Load()), args=[a], keywords=[])
                          for f in ("len", "str", "int", "float", "bool", "abs", "sorted")]
                # Dilimler özellikle dizi/metin görevlerinde yeni davranış açar;
                # özel bir hedef fonksiyon veya test cevabı içermez.
                unary += [ast.Subscript(value=a, slice=ast.Slice(
                    lower=None, upper=None, step=ast.UnaryOp(op=ast.USub(), operand=ast.Constant(1))),
                    ctx=ast.Load())]
                unary += [ast.Subscript(value=a, slice=ast.Slice(
                    lower=ast.Constant(1), upper=None, step=None), ctx=ast.Load()),
                          ast.Subscript(value=a, slice=ast.Slice(
                    lower=None, upper=ast.Constant(-1), step=None), ctx=ast.Load())]
                values_a = value_cache.get(ast.dump(a, include_attributes=False), ())
                if values_a and all(isinstance(value, str) for value in values_a):
                    unary += [ast.Call(func=ast.Attribute(value=a, attr=method, ctx=ast.Load()),
                                       args=[], keywords=[]) for method in sorted(_SAFE_METHODS)]
                for expr in unary:
                    fresh, hit = offer(expr, depth)
                    if hit:
                        solutions.append(ExpressionCandidate(expr, depth, 1.0))
            # Koşullu ifade: karşılaştırma ve iki dal aynı örneklerden birlikte
            # sınanır. Böylece yalnız doğrusal/tek biçimli yasa değil, veriyle
            # seçilen parçalı bir temsil de sentezlenebilir.
            conditions = [e for e in current if isinstance(e, (ast.Compare, ast.BoolOp))]
            for condition in conditions:
                cvals = value_cache.get(ast.dump(condition, include_attributes=False), ())
                if len(cvals) != len(cases):
                    continue
                try:
                    mask = tuple(bool(v) for v in cvals)
                except Exception:
                    continue
                if all(mask) or not any(mask):
                    continue
                true_branches, false_branches = [], []
                # Tam çözüm için gerekli koşul: dal, koşulun seçtiği bütün
                # örneklerde hedefi açıklamalı. Bu genel alt-küme eşleşmesi,
                # kör cartesian aramayı güvenli biçimde budar.
                for branch in current:
                    vals = value_cache.get(ast.dump(branch, include_attributes=False), ())
                    if len(vals) != len(cases):
                        continue
                    if all(_signature(vals[i]) == expected[i] for i in range(len(cases)) if mask[i]):
                        true_branches.append(branch)
                    if all(_signature(vals[i]) == expected[i] for i in range(len(cases)) if not mask[i]):
                        false_branches.append(branch)
                for a in true_branches:
                    for b in false_branches:
                        expr = ast.IfExp(body=a, test=condition, orelse=b)
                        fresh, hit = offer(expr, depth)
                        if hit:
                            solutions.append(ExpressionCandidate(expr, depth, 1.0))
                        if self.evaluated >= self.max_exprs:
                            break
                    if self.evaluated >= self.max_exprs:
                        break
                if self.evaluated >= self.max_exprs:
                    break
            # İkili dönüşümler. Havuz sınırlı; önce kısa ifadeler denenir.
            for a in current:
                for b in current:
                    ops = [ast.Add(), ast.Sub(), ast.Mult(), ast.FloorDiv(), ast.Mod()]
                    # Üs alma taşması/uzun hesap güvenli aramaya dahil değil.
                    for op in ops:
                        expr = ast.BinOp(left=a, op=op, right=b)
                        fresh, hit = offer(expr, depth)
                        if hit:
                            solutions.append(ExpressionCandidate(expr, depth, 1.0))
                    for op in (ast.Eq(), ast.NotEq(), ast.Lt(), ast.LtE(), ast.Gt(), ast.GtE()):
                        expr = ast.Compare(left=a, ops=[op], comparators=[b])
                        fresh, hit = offer(expr, depth)
                        if hit:
                            solutions.append(ExpressionCandidate(expr, depth, 1.0))
                    if self.evaluated >= self.max_exprs:
                        break
                if self.evaluated >= self.max_exprs:
                    break
        return self._rank(solutions)

    def _rank(self, solutions) -> list[ExpressionCandidate]:
        unique = {}
        for c in solutions:
            unique.setdefault(c.text, c)
        ordered = sorted(unique.values(), key=lambda c: (c.cost, c.text))
        # İlk bulunan aile bütün aday bütçesini yutmasın: her kök AST biçiminden
        # sırayla aday al. Böylece validation'da çakılan koşullu form, aritmetik
        # veya dilim formunun hiç denenmemesine yol açmaz.
        groups: dict[str, list[ExpressionCandidate]] = {}
        for candidate in ordered:
            groups.setdefault(type(candidate.expression).__name__, []).append(candidate)
        out = []
        while len(out) < self.max_solutions and any(groups.values()):
            for kind in sorted(groups):
                if groups[kind] and len(out) < self.max_solutions:
                    out.append(groups[kind].pop(0))
        return out


# --------------------------------------------------------------------------- çalışma alanı
class SourceWorkspace:
    """Tek modül hedefli atomik kaynak ve subprocess değerlendirme alanı."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._original: dict[str, str | None] = {}
        self._created: set[str] = set()

    def path(self, module: str) -> str:
        # Path traversal ile çalışma alanı dışına yazma yok.
        path = os.path.abspath(os.path.join(self.root, module))
        if os.path.commonpath((self.root, path)) != self.root:
            raise ValueError("module outside workspace")
        return path

    def read(self, module: str) -> str:
        with open(self.path(module), "r", encoding="utf-8") as fh:
            return fh.read()

    def snapshot(self, module: str) -> None:
        path = self.path(module)
        if module not in self._original:
            self._original[module] = open(path, encoding="utf-8").read() if os.path.exists(path) else None

    def commit(self, module: str | None = None) -> None:
        """Kabul edilen çalışma ağacını sonraki adımın yeni rollback tabanı yap."""
        names = list(self._original) if module is None else [str(module)]
        for name in names:
            self._original.pop(name, None)
            self._created.discard(name)

    def write(self, module: str, source: str) -> None:
        self.snapshot(module)
        path = self.path(module)
        existed = os.path.exists(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(source)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        if not existed:
            self._created.add(module)

    def restore(self) -> None:
        for module, source in self._original.items():
            path = self.path(module)
            if source is None:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
            else:
                tmp = path + ".rollback"
                with open(tmp, "w", encoding="utf-8") as fh:
                    fh.write(source)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, path)
        self._created.clear()

    @staticmethod
    def _score(cases: list[dict], outputs: list[dict]) -> float:
        if not cases:
            return 0.0
        return float(sum(bool(x.get("ok")) for x in outputs) / len(cases))

    def evaluate(self, module: str, function: str, cases: list[dict], *,
                 timeout: float = 4.0) -> dict:
        """Hedef fonksiyonu ayrı süreçte çalıştır; çıktıyı örneklerle karşılaştır."""
        if not cases:
            return {"score": 0.0, "outputs": [], "status": "empty"}
        path = self.path(module)
        payload = json.dumps(cases, ensure_ascii=False)
        runner = """import json, runpy, sys
module, function, payload = sys.argv[1], sys.argv[2], sys.argv[3]
cases = json.loads(payload)
out = []
try:
    ns = runpy.run_path(module)
    fn = ns[function]
    for case in cases:
        try:
            got = fn(*case.get('args', []))
            out.append({'ok': bool(got == case.get('expect')), 'got': repr(got)})
        except Exception as exc:
            out.append({'ok': False, 'error': type(exc).__name__ + ': ' + str(exc)})
    print(json.dumps({'status': 'ok', 'outputs': out}, ensure_ascii=False))
except Exception as exc:
    print(json.dumps({'status': 'load-error', 'outputs': out,
                      'error': type(exc).__name__ + ': ' + str(exc)}, ensure_ascii=False))
"""
        with tempfile.NamedTemporaryFile("w", suffix=".py", dir=self.root,
                                         prefix=".nora_eval_", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(runner)
            runner_path = fh.name
        try:
            proc = subprocess.run([sys.executable, "-I", runner_path, path, function, payload],
                                  cwd=self.root, capture_output=True, text=True,
                                  timeout=max(0.1, float(timeout)),
                                  env={"PYTHONHASHSEED": "0"})
            lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
            if not lines:
                return {"score": 0.0, "outputs": [], "status": "no-output",
                        "error": proc.stderr[-500:]}
            data = json.loads(lines[-1])
            data["score"] = self._score(cases, data.get("outputs") or [])
            data["returncode"] = proc.returncode
            return data
        except subprocess.TimeoutExpired:
            return {"score": 0.0, "outputs": [], "status": "timeout"}
        except (OSError, json.JSONDecodeError) as exc:
            return {"score": 0.0, "outputs": [], "status": "error", "error": str(exc)}
        finally:
            try:
                os.remove(runner_path)
            except FileNotFoundError:
                pass


# --------------------------------------------------------------------------- öz-değişim
class SelfModifier:
    """Örnek → aday → çalıştır → validation → audit → kalıcılaştır/geri al."""

    def __init__(self, workspace: SourceWorkspace, *, synthesizer: ExpressionSynthesizer | None = None):
        self.workspace = workspace
        self.synthesizer = synthesizer or ExpressionSynthesizer()
        self.ledger: list[dict] = []

    @staticmethod
    def _same_cases(a: list[dict], b: list[dict]) -> bool:
        return json.dumps(a, sort_keys=True, ensure_ascii=False) == json.dumps(b, sort_keys=True, ensure_ascii=False)

    @staticmethod
    def _replace_function(source: str, function: str, args: list[str], expression: str) -> str:
        tree = ast.parse(source)
        found = None
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function:
                found = node
                break
        expr = ast.parse(expression, mode="eval").body
        _safe_tree(ast.Expression(body=expr))
        if found is None:
            found = ast.FunctionDef(name=function,
                                    args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=a) for a in args],
                                                       vararg=None, kwonlyargs=[], kw_defaults=[],
                                                       kwarg=None, defaults=[]),
                                    body=[ast.Return(value=expr)], decorator_list=[])
            tree.body.append(found)
        else:
            if isinstance(found, ast.AsyncFunctionDef):
                raise ValueError("async target is not supported")
            found.body = [ast.Return(value=expr)]
        ast.fix_missing_locations(tree)
        return ast.unparse(tree) + "\n"

    @staticmethod
    def _incumbent_expression(source: str, function: str) -> ast.AST | None:
        if not source:
            return None
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == function:
                if len(node.body) == 1 and isinstance(node.body[0], ast.Return):
                    return node.body[0].value
        return None

    def propose_patch(self, module: str, function: str, args: list[str], *,
                      train: list[dict], validation: list[dict],
                      timeout: float = 4.0) -> dict:
        """Kendi adayını seç; doğrula fakat onay/pil öncesinde kaynağı değiştirme."""
        if not train or not validation or self._same_cases(train, validation):
            return {"status": "invalid-protocol", "reason": "train/validation ayrı ve boş olmamalı"}
        self.workspace.snapshot(module)
        try:
            original = self.workspace.read(module)
        except FileNotFoundError:
            original = ""
        base_train = self.workspace.evaluate(module, function, train, timeout=timeout) if original else {"score": 0.0}
        base_val = self.workspace.evaluate(module, function, validation, timeout=timeout) if original else {"score": 0.0}
        if base_val.get("score", 0.0) >= 1.0 and base_train.get("score", 0.0) >= 1.0:
            return {"status": "no-need", "train": base_train.get("score", 0.0),
                    "validation": base_val.get("score", 0.0)}
        incumbent = self._incumbent_expression(original, function)
        candidates = self.synthesizer.candidates(
            args, train, incumbents=[incumbent] if incumbent is not None else [],
        )
        try:
            for candidate in candidates:
                try:
                    source = self._replace_function(original, function, args, candidate.text)
                    self.workspace.write(module, source)
                    train_result = self.workspace.evaluate(module, function, train, timeout=timeout)
                    if train_result.get("score", 0.0) < 1.0:
                        continue
                    val_result = self.workspace.evaluate(module, function, validation, timeout=timeout)
                    if val_result.get("score", 0.0) < 1.0:
                        continue
                    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
                    patch = "".join(difflib.unified_diff(
                        original.splitlines(keepends=True), source.splitlines(keepends=True),
                        fromfile=f"a/{module}", tofile=f"b/{module}",
                    ))
                    return {
                        "status": "proposed", "module": module, "function": function,
                        "args": list(args), "expression": candidate.text,
                        "incumbent": ast.unparse(incumbent) if incumbent is not None else "",
                        "source": source, "patch": patch, "sha256": digest,
                        "proposal_id": digest[:16], "candidates": len(candidates),
                        "train": train_result.get("score"), "validation": val_result.get("score"),
                        "baseline": {"train": base_train.get("score", 0.0),
                                     "validation": base_val.get("score", 0.0)},
                    }
                except Exception:
                    continue
            return {"status": "unresolved", "candidates": len(candidates), "rolled_back": True}
        finally:
            # Öneri ile kabul birbirinden ayrıdır: bu metot hiçbir adayı bırakmaz.
            self.workspace.restore()

    def apply_patch(self, proposal: dict, *, battery, approved: bool = False) -> dict:
        """Açık onayla dış pili çalıştır; yalnız başarılı yamayı bırak."""
        if not isinstance(proposal, dict) or proposal.get("status") != "proposed":
            return {"status": "invalid-protocol", "reason": "valid proposal required"}
        if approved is not True:
            return {"status": "approval-required", "proposal_id": proposal.get("proposal_id"),
                    "changed": False}
        if not callable(battery):
            return {"status": "invalid-protocol", "reason": "callable battery required"}
        module = str(proposal.get("module") or "")
        source = str(proposal.get("source") or "")
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if not module or not source or digest != proposal.get("sha256"):
            return {"status": "invalid-protocol", "reason": "proposal integrity", "changed": False}
        self.workspace.snapshot(module)
        record = {"module": module, "function": proposal.get("function"),
                  "proposal_id": proposal.get("proposal_id"),
                  "expression": proposal.get("expression")}
        try:
            self.workspace.write(module, source)
            battery_result = battery()
            passed = (battery_result is True or
                      (isinstance(battery_result, dict) and
                       (battery_result.get("passed") is True or
                        battery_result.get("status") in {"ok", "passed"})))
            record["battery"] = battery_result
            if passed:
                self.workspace.commit(module)
                record.update({"status": "accepted", "kept": True, "sha256": digest})
                self.ledger.append(dict(record))
                return record
        except Exception as exc:
            record["battery"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        self.workspace.restore()
        record.update({"status": "rolled_back", "rolled_back": True, "kept": False})
        self.ledger.append(dict(record))
        return record

    def improve(self, module: str, function: str, args: list[str], *,
                train: list[dict], validation: list[dict], audit: list[dict] | None = None,
                timeout: float = 4.0) -> dict:
        if not train or not validation or self._same_cases(train, validation):
            return {"status": "invalid-protocol", "reason": "train/validation ayrı ve boş olmamalı"}
        self.workspace.snapshot(module)
        try:
            original = self.workspace.read(module)
        except FileNotFoundError:
            original = ""
        base_train = self.workspace.evaluate(module, function, train, timeout=timeout) if original else {"score": 0.0}
        base_val = self.workspace.evaluate(module, function, validation, timeout=timeout) if original else {"score": 0.0}
        if base_val.get("score", 0.0) >= 1.0 and base_train.get("score", 0.0) >= 1.0:
            return {"status": "no-need", "train": base_train.get("score", 0.0),
                    "validation": base_val.get("score", 0.0)}
        candidates = self.synthesizer.candidates(args, train)
        record = {"module": module, "function": function, "started_ns": time.time_ns(),
                  "prior": "safe-ast/enumerative", "candidates": len(candidates),
                  "baseline": {"train": base_train.get("score", 0.0), "validation": base_val.get("score", 0.0)}}
        for cand in candidates:
            try:
                source = self._replace_function(original, function, args, cand.text)
                self.workspace.write(module, source)
                # Adayın gerçek kaynak olarak çalışması zorunlu; sentezcinin kendi
                # eval'i tek başına kabul kanıtı değildir.
                train_result = self.workspace.evaluate(module, function, train, timeout=timeout)
                if train_result.get("score", 0.0) < 1.0:
                    self.workspace.restore()
                    continue
                val_result = self.workspace.evaluate(module, function, validation, timeout=timeout)
                if val_result.get("score", 0.0) < 1.0:
                    self.workspace.restore()
                    continue
                accepted = {**record, "status": "accepted", "expression": cand.text,
                            "train": train_result.get("score"), "validation": val_result.get("score"),
                            "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest()}
                if audit is not None:
                    # Audit yalnız son bağımsız kontrol; aday üretiminde kullanılmaz.
                    audit_result = self.workspace.evaluate(module, function, audit, timeout=timeout)
                    accepted["audit"] = audit_result.get("score", 0.0)
                    if audit_result.get("score", 0.0) < 1.0:
                        self.workspace.restore()
                        accepted["status"] = "audit-rollback"
                        self.ledger.append(accepted)
                        return accepted
                self.ledger.append(accepted)
                return accepted
            except Exception as exc:
                self.workspace.restore()
                record["last_error"] = f"{type(exc).__name__}: {exc}"
        self.workspace.restore()
        failed = {**record, "status": "unresolved", "rolled_back": True}
        self.ledger.append(failed)
        return failed

    def state_dict(self) -> dict:
        return {"ledger": list(self.ledger[-64:])}

    def load_state_dict(self, d: dict | None) -> None:
        self.ledger = [dict(x) for x in (d or {}).get("ledger", []) if isinstance(x, dict)]


__all__ = ["ExpressionSynthesizer", "ExpressionCandidate", "SourceWorkspace", "SelfModifier"]
