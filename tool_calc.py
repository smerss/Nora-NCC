#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tool_calc.py — NoRa'nın otonom hesap-aracı (System-2'nin eli).
GÜVENLİK: eval()/exec() YOK. ast beyaz-listesi: Expression, BinOp, UnaryOp,
Constant(int/float), Add, Sub, Mult, Div, UAdd, USub. Hepsi reddedilir: Name,
Call, Attribute, Pow, Mod, FloorDiv, Compare... Uzunluk/düğüm/derinlik tavanlı,
sıfır-bölme ve inf/nan hata döner.
Kullanım: DPY tool_calc.py "12 + 34 * 2"  -> 80
"""
import ast
import sys

MAX_LEN, MAX_NODES, MAX_DEPTH, MAX_ABS = 200, 200, 20, 10 ** 15
OK_BIN = (ast.Add, ast.Sub, ast.Mult, ast.Div)
OK_UN = (ast.UAdd, ast.USub)


def _check(node, depth=0, count=None):
    if count is None:
        count = [0]
    count[0] += 1
    if depth > MAX_DEPTH or count[0] > MAX_NODES:
        raise ValueError("ifade çok derin/geniş")
    if isinstance(node, ast.Expression):
        _check(node.body, depth + 1, count)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, OK_BIN):
        _check(node.left, depth + 1, count)
        _check(node.right, depth + 1, count)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, OK_UN):
        _check(node.operand, depth + 1, count)
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        if abs(node.value) > MAX_ABS:
            raise ValueError("sabit çok büyük")
    else:
        raise ValueError("yasak düğüm: %s" % type(node).__name__)


def _run(node):
    if isinstance(node, ast.Expression):
        return _run(node.body)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        l, r = _run(node.left), _run(node.right)
        if isinstance(node.op, ast.Add):
            v = l + r
        elif isinstance(node.op, ast.Sub):
            v = l - r
        elif isinstance(node.op, ast.Mult):
            v = l * r
        else:
            if r == 0:
                raise ZeroDivisionError("sıfıra bölme")
            v = l / r
    elif isinstance(node, ast.UnaryOp):
        v = +_run(node.operand) if isinstance(node.op, ast.UAdd) else -_run(node.operand)
    else:
        raise ValueError("yasak")
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        raise ValueError("geçersiz sonuç")
    if abs(v) > MAX_ABS:
        raise ValueError("ara sonuç taştı")
    return v


def safe_eval(expr):
    """Başarılıysa (değer, None), değilse (None, hata-mesajı)."""
    if not isinstance(expr, str) or len(expr) > MAX_LEN:
        return None, "girdi hatası"
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None, "sözdizimi"
    try:
        _check(tree)
        return _run(tree), None
    except ZeroDivisionError:
        return None, "sıfıra bölme"
    except ValueError as e:
        return None, str(e)


if __name__ == "__main__":
    for e in sys.argv[1:] or ["12 + 34 * 2"]:
        v, err = safe_eval(e)
        print(("%s = %s" % (e, v)) if err is None else ("%s -> HATA: %s" % (e, err)))
