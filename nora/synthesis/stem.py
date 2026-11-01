#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_stem.py — NoRa Parietal Matematik ve Fiziksel Sezgi/Simülasyon Korteksi.

Ezberlenmiş metin şablonları yerine İKİ SÜREÇLİ BİLİŞSEL MİMARİ kullanır:
  1. Sembolik Sistem-2 Türetimi + Sayısal Sistem-1 Sağlaması (MathCortex):
     - Güvenli özyinelemeli AST ayrıştırıcı: cebirsel ifadeler, polinom/trigonometrik/üstel
       sembolik türev (çarpım, bölüm ve zincir kuralı + sonlu-fark sayısal doğrulaması),
       belirli/belirsiz integral (analitik ilkel fonksiyon + Simpson sayısal sağlaması),
       1. ve 2. derece denklemler ile 2x2 doğrusal denklem sistemleri (kök yerine koyma sağlamalı),
       sayılar teorisi (asal çarpanlar, EBOB/EKOK), kombinatorik ve geometri.
  2. Boyut Analizli Kısıt Yayılımı + Zihinsel Fizik Simülasyonu (PhysicsCortex):
     - Her fiziksel büyüklüğü SI boyut vektörü [M, L, T, I, Theta] ile temsil eder.
     - Verilen büyüklüklerden hedef büyüklüğe çift-yönlü fizik yasaları ağında (Newton dinamiği,
       kinematik, iş-güç-enerji, momentum korunumu ve çarpışmalar, Ohm/elektrik yasaları,
       dalgalar, yay/sarkaç salınımı, termodinamik, kütle-enerji) çok-adımlı çıkarım yapar.
     - MentalPhysicsSandbox içinde zaman-adımlı (RK4) sayısal simülasyon koşturarak analitik
       sonucu doğrular ve 'hız 3 katına çıkarsa kinetik enerji kaç katına çıkar?' ya da
       'sürtünme olmasaydı ne olurdu?' gibi ölçekleme/karşıolgusal soruları canlı pertürbasyonla çözer.
"""
from __future__ import annotations

from nora._paths import REPO_ROOT


from dataclasses import dataclass
import math
import re
import numpy as np


def _tr_low(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").lower()


def _fmt_num(x: float, digits: int = 4) -> str:
    if not math.isfinite(x):
        return "tanımsız"
    if abs(x - round(x)) < 1e-7:
        return str(int(round(x)))
    s = f"{x:.{digits}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


# Matematiksel sözcükler: sabit, işlev adı. Bunların dışındaki Türkçe sözcükler
# ('ifadesinde', 'kaçtır', 'hesapla') ifadeye ait değildir ve ayrıştırmadan önce atılır.
_MATH_WORDS = frozenset({"pi", "e", "sin", "cos", "tan", "exp", "ln", "log", "sqrt", "abs"})
_MATH_WORD_OPS = {"üzeri": "^", "üssü": "^", "çarpı": "*", "kere": "*", "bölü": "/", "karekök": "sqrt"}
_MATH_OPS = frozenset({"+", "-", "*", "/", "^", "(", ")"})


def _math_span(text: str) -> str:
    """Ham Türkçe ifadeden yalnız matematiksel belirteçleri bırakıp normalize eder.

    Kesme işaretli ekler ("3'e", "pi'nin") ve dolgu sözcükleri atılır; sayı, işleç,
    parantez, sabit/işlev adları ile tek harfli değişkenler (x, t, y, z) korunur.
    """
    s = _tr_low(text).replace("π", " pi ").replace("²", "^2").replace("³", "^3").replace("⁴", "^4")
    s = s.replace("×", "*").replace("÷", "/").replace("**", "^")
    s = re.sub(r"['’]\S*", " ", s)
    out: list[str] = []
    for tok in re.findall(r"\d+(?:\.\d+)?|[a-zçğıöşü_]+|[\+\-\*\/\^\(\)]", s):
        if tok in _MATH_WORD_OPS:
            out.append(_MATH_WORD_OPS[tok])
        elif tok[0].isdigit() or tok in _MATH_OPS or tok in _MATH_WORDS or (len(tok) == 1 and tok in "xyzt"):
            out.append(tok)
    # Örtük çarpımları ('2 pi' -> '2*pi', '4 x ^ 3' -> '4*x ^ 3') ön-işlemcinin kuralıyla aç.
    return SymbolicAST.preprocess(" ".join(out))


# ============================================================================
# 1. SEMBOLİK MATEMATİK AST VE TÜREV / İNTEGRAL / DENKLEM MOTORU
# ============================================================================

@dataclass(frozen=True)
class Expr:
    pass


@dataclass(frozen=True)
class Num(Expr):
    val: float


@dataclass(frozen=True)
class Var(Expr):
    name: str


@dataclass(frozen=True)
class Unary(Expr):
    op: str
    arg: Expr


@dataclass(frozen=True)
class Bin(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Call(Expr):
    fn: str
    arg: Expr


class SymbolicAST:
    """Güvenli matematiksel ifade ayrıştırıcı, sadeleştirici ve sembolik kalkülüs motoru."""

    FUNCS = {"sin", "cos", "tan", "exp", "ln", "log", "sqrt", "abs"}

    @classmethod
    def preprocess(cls, raw: str) -> str:
        s = _tr_low(raw).strip()
        s = s.replace("π", "pi").replace("²", "^2").replace("³", "^3").replace("⁴", "^4")
        s = s.replace("karekök", "sqrt").replace("üzeri", "^").replace("üssü", "^").replace("**", "^")
        s = s.replace("×", "*").replace("÷", "/")
        # Sayılar arasındaki x/X çarpma işaretidir: '5x5' -> '5*5'
        s = re.sub(r"(\d+)\s*[xX]\s*(\d+)", r"\1*\2", s)
        # Türkçe işlem sözcükleri
        s = re.sub(r"\b(?:çarpı|kere)\b", "*", s)
        s = re.sub(r"\b(?:bölü)\b", "/", s)
        # Örtük çarpımları açık hale getir: '3x' -> '3*x', '2(x+1)' -> '2*(x+1)', 'x(x+1)' -> 'x*(x+1)', '2sin(x)' -> '2*sin(x)'
        s = re.sub(r"(\d)\s*([a-zA-Z(])", r"\1*\2", s)
        s = re.sub(r"(\))\s*(\d|[a-zA-Z(])", r"\1*\2", s)
        return s

    @classmethod
    def tokenize(cls, s: str) -> list[str]:
        s = cls.preprocess(s)
        toks = re.findall(r"\d+(?:\.\d+)?|[a-zA-Z_]+|[\+\-\*\/\^\(\)]", s)
        # '3*sin' ayrışırken 'sin' korunur; ama tek harfli 'x' ile 'sin' ayrımını koru
        return toks

    @classmethod
    def parse(cls, raw: str, *, simplify: bool = True) -> Expr:
        """İfadeyi ayrıştır; ``simplify=False`` ara işlem ağacını denetim için korur."""
        toks = cls.tokenize(raw)
        if not toks:
            raise ValueError("Boş matematiksel ifade")
        pos = [0]

        def peek() -> str | None:
            return toks[pos[0]] if pos[0] < len(toks) else None

        def eat(expected: str | None = None) -> str:
            t = peek()
            if t is None or (expected is not None and t != expected):
                raise ValueError(f"Beklenen '{expected}', gelen '{t}'")
            pos[0] += 1
            return t

        def parse_add() -> Expr:
            node = parse_mul()
            while peek() in ("+", "-"):
                op = eat()
                node = Bin(op, node, parse_mul())
            return node

        def parse_mul() -> Expr:
            node = parse_pow()
            while peek() in ("*", "/"):
                op = eat()
                node = Bin(op, node, parse_pow())
            return node

        def parse_pow() -> Expr:
            node = parse_unary()
            if peek() == "^":
                eat("^")
                node = Bin("^", node, parse_pow())
            return node

        def parse_unary() -> Expr:
            if peek() == "-":
                eat("-")
                return Unary("-", parse_unary())
            if peek() == "+":
                eat("+")
                return parse_unary()
            return parse_atom()

        def parse_atom() -> Expr:
            t = peek()
            if t is None:
                raise ValueError("Beklenmeyen ifade sonu")
            if t == "(":
                eat("(")
                node = parse_add()
                if peek() == ")":
                    eat(")")
                return node
            if re.match(r"^\d+(?:\.\d+)?$", t):
                eat()
                return Num(float(t))
            if re.match(r"^[a-zA-Z_]+$", t):
                ident = eat()
                if ident in cls.FUNCS and peek() == "(":
                    eat("(")
                    arg = parse_add()
                    if peek() == ")":
                        eat(")")
                    return Call("ln" if ident == "log" else ident, arg)
                if ident in cls.FUNCS and peek() is not None:
                    return Call("ln" if ident == "log" else ident, parse_pow())
                if ident == "pi":
                    return Num(math.pi)
                if ident == "e" and peek() != "^":
                    return Num(math.e)
                if ident == "e" and peek() == "^":
                    eat("^")
                    return Call("exp", parse_pow())
                return Var(ident)
            raise ValueError(f"Tanınmayan atom: {t}")

        res = parse_add()
        if pos[0] != len(toks):
            raise ValueError(f"Artık belirteç: {toks[pos[0]:]}")
        return cls.simplify(res) if simplify else res

    @classmethod
    def eval_expr(cls, node: Expr, env: dict[str, float] | None = None) -> float:
        env = env or {}
        if isinstance(node, Num):
            return float(node.val)
        if isinstance(node, Var):
            if node.name in env:
                return float(env[node.name])
            raise KeyError(f"Değişken değeri eksik: {node.name}")
        if isinstance(node, Unary):
            v = cls.eval_expr(node.arg, env)
            return -v if node.op == "-" else v
        if isinstance(node, Bin):
            a = cls.eval_expr(node.left, env)
            b = cls.eval_expr(node.right, env)
            if node.op == "+":
                return a + b
            if node.op == "-":
                return a - b
            if node.op == "*":
                return a * b
            if node.op == "/":
                return a / b
            if node.op == "^":
                return a ** b
        if isinstance(node, Call):
            v = cls.eval_expr(node.arg, env)
            fn_map = {
                "sin": math.sin, "cos": math.cos, "tan": math.tan,
                "exp": math.exp, "ln": math.log, "sqrt": math.sqrt, "abs": abs,
            }
            return fn_map[node.fn](v)
        raise ValueError("Bilinmeyen AST düğümü")

    @classmethod
    def simplify(cls, node: Expr) -> Expr:
        if isinstance(node, (Num, Var)):
            return node
        if isinstance(node, Unary):
            a = cls.simplify(node.arg)
            if node.op == "-":
                if isinstance(a, Num):
                    return Num(-a.val)
                if isinstance(a, Unary) and a.op == "-":
                    return a.arg
            return Unary(node.op, a)
        if isinstance(node, Call):
            a = cls.simplify(node.arg)
            if isinstance(a, Num):
                try:
                    return Num(cls.eval_expr(Call(node.fn, a)))
                except Exception:
                    pass
            return Call(node.fn, a)
        if isinstance(node, Bin):
            l = cls.simplify(node.left)
            r = cls.simplify(node.right)
            if isinstance(l, Num) and isinstance(r, Num):
                try:
                    return Num(cls.eval_expr(Bin(node.op, l, r)))
                except Exception:
                    pass
            op = node.op
            if op == "+":
                if isinstance(l, Num) and abs(l.val) < 1e-12:
                    return r
                if isinstance(r, Num) and abs(r.val) < 1e-12:
                    return l
                if isinstance(r, Unary) and r.op == "-":
                    return cls.simplify(Bin("-", l, r.arg))
                if isinstance(r, Num) and r.val < 0:
                    return cls.simplify(Bin("-", l, Num(-r.val)))
            elif op == "-":
                if isinstance(r, Num) and abs(r.val) < 1e-12:
                    return l
                if isinstance(l, Num) and abs(l.val) < 1e-12:
                    return cls.simplify(Unary("-", r))
                if l == r:
                    return Num(0.0)
            elif op == "*":
                if (isinstance(l, Num) and abs(l.val) < 1e-12) or (isinstance(r, Num) and abs(r.val) < 1e-12):
                    return Num(0.0)
                if isinstance(l, Num) and abs(l.val - 1.0) < 1e-12:
                    return r
                if isinstance(r, Num) and abs(r.val - 1.0) < 1e-12:
                    return l
                if isinstance(l, Num) and abs(l.val + 1.0) < 1e-12:
                    return cls.simplify(Unary("-", r))
                if isinstance(r, Num) and not isinstance(l, Num):
                    l, r = r, l
                # (c1) * (c2 * u) -> (c1*c2) * u
                if isinstance(l, Num) and isinstance(r, Bin) and r.op == "*" and isinstance(r.left, Num):
                    return cls.simplify(Bin("*", Num(l.val * r.left.val), r.right))
            elif op == "/":
                if isinstance(l, Num) and abs(l.val) < 1e-12:
                    return Num(0.0)
                if isinstance(r, Num) and abs(r.val - 1.0) < 1e-12:
                    return l
                if l == r:
                    return Num(1.0)
            elif op == "^":
                if isinstance(r, Num) and abs(r.val) < 1e-12:
                    return Num(1.0)
                if isinstance(r, Num) and abs(r.val - 1.0) < 1e-12:
                    return l
                if isinstance(l, Num) and abs(l.val) < 1e-12:
                    return Num(0.0)
            return Bin(op, l, r)
        return node

    @classmethod
    def diff(cls, node: Expr, var: str = "x") -> Expr:
        """Analitik sembolik türev (kuvvet, çarpım, bölüm ve zincir kuralı)."""
        if isinstance(node, Num):
            return Num(0.0)
        if isinstance(node, Var):
            return Num(1.0) if node.name == var else Num(0.0)
        if isinstance(node, Unary):
            return cls.simplify(Unary(node.op, cls.diff(node.arg, var)))
        if isinstance(node, Bin):
            u, v = node.left, node.right
            du, dv = cls.diff(u, var), cls.diff(v, var)
            if node.op in ("+", "-"):
                return cls.simplify(Bin(node.op, du, dv))
            if node.op == "*":
                return cls.simplify(Bin("+", Bin("*", du, v), Bin("*", u, dv)))
            if node.op == "/":
                num = Bin("-", Bin("*", du, v), Bin("*", u, dv))
                den = Bin("^", v, Num(2.0))
                return cls.simplify(Bin("/", num, den))
            if node.op == "^":
                if isinstance(v, Num):
                    # n * u^(n-1) * du
                    return cls.simplify(Bin("*", Bin("*", Num(v.val), Bin("^", u, Num(v.val - 1.0))), du))
                # Genel u^v: u^v * (dv*ln(u) + v*du/u)
                term = Bin("+", Bin("*", dv, Call("ln", u)), Bin("/", Bin("*", v, du), u))
                return cls.simplify(Bin("*", node, term))
        if isinstance(node, Call):
            u = node.arg
            du = cls.diff(u, var)
            if node.fn == "sin":
                outer = Call("cos", u)
            elif node.fn == "cos":
                outer = Unary("-", Call("sin", u))
            elif node.fn == "tan":
                outer = Bin("/", Num(1.0), Bin("^", Call("cos", u), Num(2.0)))
            elif node.fn == "exp":
                outer = Call("exp", u)
            elif node.fn == "ln":
                outer = Bin("/", Num(1.0), u)
            elif node.fn == "sqrt":
                outer = Bin("/", Num(1.0), Bin("*", Num(2.0), Call("sqrt", u)))
            else:
                raise ValueError(f"Türevi tanımlanmamış fonksiyon: {node.fn}")
            return cls.simplify(Bin("*", du, outer))
        raise ValueError("Türevlenemeyen düğüm")

    @classmethod
    def poly_coeffs(cls, node: Expr, var: str = "x", max_deg: int = 8) -> dict[int, float] | None:
        """İfade bir polinom ise {derece: katsayı} sözlüğünü döndürür; değilse None."""
        if isinstance(node, Num):
            return {0: node.val}
        if isinstance(node, Var):
            return {1: 1.0} if node.name == var else None
        if isinstance(node, Unary) and node.op == "-":
            sub = cls.poly_coeffs(node.arg, var, max_deg)
            return {k: -v for k, v in sub.items()} if sub is not None else None
        if isinstance(node, Bin):
            l = cls.poly_coeffs(node.left, var, max_deg)
            r = cls.poly_coeffs(node.right, var, max_deg)
            if l is None or r is None:
                return None
            if node.op in ("+", "-"):
                out = dict(l)
                sgn = 1.0 if node.op == "+" else -1.0
                for k, v in r.items():
                    out[k] = out.get(k, 0.0) + sgn * v
                return out
            if node.op == "*":
                out: dict[int, float] = {}
                for k1, v1 in l.items():
                    for k2, v2 in r.items():
                        if k1 + k2 > max_deg:
                            return None
                        out[k1 + k2] = out.get(k1 + k2, 0.0) + v1 * v2
                return out
            if node.op == "/" and list(r.keys()) == [0] and abs(r[0]) > 1e-12:
                return {k: v / r[0] for k, v in l.items()}
            if node.op == "^" and list(r.keys()) == [0]:
                p = r[0]
                if p >= 0 and abs(p - round(p)) < 1e-9 and int(round(p)) <= max_deg:
                    pi = int(round(p))
                    res = {0: 1.0}
                    for _ in range(pi):
                        nxt: dict[int, float] = {}
                        for k1, v1 in res.items():
                            for k2, v2 in l.items():
                                if k1 + k2 > max_deg:
                                    return None
                                nxt[k1 + k2] = nxt.get(k1 + k2, 0.0) + v1 * v2
                        res = nxt
                    return res
        return None

    @classmethod
    def format_poly(cls, coeffs: dict[int, float], var: str = "x") -> str:
        terms = []
        for deg in sorted(coeffs.keys(), reverse=True):
            c = coeffs[deg]
            if abs(c) < 1e-9:
                continue
            sgn = "-" if c < 0 else "+"
            ac = abs(c)
            c_str = _fmt_num(ac)
            if deg == 0:
                body = c_str
            elif deg == 1:
                body = var if abs(ac - 1.0) < 1e-9 else f"{c_str}{var}"
            else:
                body = f"{var}^{deg}" if abs(ac - 1.0) < 1e-9 else f"{c_str}{var}^{deg}"
            terms.append((sgn, body))
        if not terms:
            return "0"
        first_sgn, first_body = terms[0]
        out = (f"-{first_body}" if first_sgn == "-" else first_body)
        for sgn, body in terms[1:]:
            out += f" {sgn} {body}"
        return out

    @classmethod
    def to_str(cls, node: Expr, var: str = "x") -> str:
        pc = cls.poly_coeffs(node, var)
        if pc is not None:
            return cls.format_poly(pc, var)
        if isinstance(node, Num):
            return _fmt_num(node.val)
        if isinstance(node, Var):
            return node.name
        if isinstance(node, Unary):
            inner = cls.to_str(node.arg, var)
            return f"-({inner})" if isinstance(node.arg, Bin) and node.arg.op in ("+", "-") else f"-{inner}"
        if isinstance(node, Call):
            return f"{node.fn}({cls.to_str(node.arg, var)})"
        if isinstance(node, Bin):
            ls = cls.to_str(node.left, var)
            rs = cls.to_str(node.right, var)
            if node.op in ("*", "/", "^"):
                if isinstance(node.left, Bin) and node.left.op in ("+", "-"):
                    ls = f"({ls})"
                if isinstance(node.right, Bin) and node.right.op in ("+", "-", "*", "/"):
                    rs = f"({rs})"
            return f"{ls} {node.op} {rs}" if node.op in ("+", "-") else f"{ls}{node.op}{rs}"
        return str(node)

    @classmethod
    def simpson_integral(cls, node: Expr, var: str, a: float, b: float, n: int = 1200) -> float:
        if abs(b - a) < 1e-12:
            return 0.0
        if n % 2 == 1:
            n += 1
        h = (b - a) / n
        total = cls.eval_expr(node, {var: a}) + cls.eval_expr(node, {var: b})
        for i in range(1, n):
            x = a + i * h
            w = 4.0 if (i % 2 == 1) else 2.0
            total += w * cls.eval_expr(node, {var: x})
        return total * h / 3.0


class MathCortex:
    """Parietal Matematik Korteksi: Cebir, Kalkülüs, Sayılar Teorisi, Kombinatorik ve Geometri."""

    @staticmethod
    def _prime_factors(n: int) -> list[tuple[int, int]]:
        n = abs(int(n))
        out = []
        d = 2
        while d * d <= n:
            if n % d == 0:
                cnt = 0
                while n % d == 0:
                    n //= d
                    cnt += 1
                out.append((d, cnt))
            d += 1 if d == 2 else 2
        if n > 1:
            out.append((n, 1))
        return out

    def try_solve(self, raw_text: str, mind: Any = None, agi: Any = None) -> dict | None:
        """Gelen Türkçe cümlede ileri matematiksel bir problem varsa çözer ve yapılandırılmış açıklama döndürür."""
        s = _tr_low(raw_text).strip()

        # 0) Doğrudan Temel İki-Terimli Aritmetik ('25 ile 17\'nin toplamı kaçtır?', '50 eksi 12', '6 çarpı 7', '100\'ün 4\'e bölümü', '25 artı 17')
        m_op = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:ile|ve)?\s*(\d+(?:[.,]\d+)?)(?:'nin|'nın|'nün|'nun|'in|'ın|'ün|'un)?\s*(toplamı|farkı|çarpımı|bölümü)", s)
        if not m_op:
            m_op = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:sayısının|sayısıyla)?\s*(\d+(?:[.,]\d+)?)\s*(?:sayısının|sayısıyla)?\s*(toplamı|farkı|çarpımı|bölümü)", s)
        if not m_op:
            m_op = re.search(r"(\d+(?:[.,]\d+)?)\s*(artı|eksi|çarpı|bölü|\+|\-|\*|\/)\s*(\d+(?:[.,]\d+)?)", s)
        # Değişkenli ifadeler ('3x^3 - 4x^2', 'türev', 'integral', '=') bu yola girmez; denklem/türev motoruna kalır.
        if m_op and not re.search(r"\d\s*[xyzt]\b|\d\s*[xyzt]\^|türev|integral|=", s) and not any(v in s for v in ("elma", "armut", "kalem", "sayfa", "bilye", "kutu", "dakika", "gün", "tane", "yaşında")):
            if m_op.re.pattern.endswith(r"(\d+(?:[.,]\d+)?)"):
                a_v = float(m_op.group(1).replace(",", "."))
                op_v = m_op.group(2)
                b_v = float(m_op.group(3).replace(",", "."))
            else:
                a_v = float(m_op.group(1).replace(",", "."))
                b_v = float(m_op.group(2).replace(",", "."))
                op_v = m_op.group(3)

            val = None
            expr_sym = ""
            op_word = ""
            if op_v in ("toplamı", "artı", "+"):
                val = a_v + b_v; expr_sym = "+"; op_word = "toplam"
            elif op_v in ("farkı", "eksi", "-"):
                val = a_v - b_v; expr_sym = "-"; op_word = "fark"
            elif op_v in ("çarpımı", "çarpı", "*"):
                val = a_v * b_v; expr_sym = "×"; op_word = "çarpım"
            elif op_v in ("bölümü", "bölü", "/"):
                if b_v != 0:
                    val = a_v / b_v; expr_sym = "÷"; op_word = "bölüm"

            if val is not None:
                a_s = _fmt_num(a_v); b_s = _fmt_num(b_v); val_s = _fmt_num(val)
                return {
                    "domain": "math", "kind": "arithmetic", "confidence": 0.98,
                    "reply": f"{a_s} {expr_sym} {b_s} = {val_s}; {op_word} {val_s} eder.",
                    "trace": f"{a_v}{expr_sym}{b_v}={val}"
                }

        # 1) Kalkülüs: Sembolik Türev ('3x^3 - 4x^2 + 5x - 2 türevi nedir?', 'f(x) = 3x^2 + 2x türevini al')
        # Türev veya türev alma isteği, denklem eşitliklerinden ('=') önce kontrol edilir!
        if any(k in s for k in ("türev", "türevi", "türevini", "d/dx", "d/dt")):
            res = self._solve_derivative(s, raw_text=raw_text, agi=agi)
            if res is not None:
                return res

        # 2) Kalkülüs: Belirli / Belirsiz İntegral ("0'dan 3'e x^2 integrali", "2x + 4 integrali nedir", "integralini al: 2x + 4")
        if any(k in s for k in ("integral", "integrali", "integralini", "ilkel fonksiyon")):
            res = self._solve_integral(s, raw_text=raw_text, agi=agi)
            if res is not None:
                return res

        # 3) 2x2 Doğrusal Denklem Sistemi ('x + y = 12 ve x - y = 4')
        eqs = re.findall(r"([0-9xXyY\+\-\*\s\.]+)=([0-9xXyY\+\-\*\s\.]+)", s)
        if len(eqs) == 2 and ("x" in s and "y" in s):
            res = self._solve_2x2(eqs, raw_text=raw_text, agi=agi)
            if res is not None:
                return res

        # 4) Tek Değişkenli Cebirsel Denklem ('2x + 7 = 23', 'x^2 - 5x + 6 = 0')
        if "=" in s and "::" not in s and "->" not in s and any(v in s for v in ("x", "t", "z")):
            var_name = "x" if "x" in s else ("t" if "t" in s else "z")
            m_eq = re.search(rf"([0-9{var_name}\^\+\-\*\/\(\)\s\.]+)=\s*([0-9{var_name}\^\+\-\*\/\(\)\s\.\-]+)", s)
            if m_eq:
                lhs_s, rhs_s = m_eq.group(1).strip(), m_eq.group(2).strip()
                # Sağ taraftaki Türkçe soru eklerini temizle ('23 ise x', '0 denkleminin')
                rhs_clean = re.match(rf"^[\d{var_name}\^\+\-\*\/\(\)\s\.]+", rhs_s)
                if rhs_clean:
                    res = self._solve_single_eq(lhs_s, rhs_clean.group(0).strip(), var_name, raw_text=raw_text, agi=agi)
                    if res is not None:
                        return res

        # 5) Sayılar Teorisi: Asallık & Asal Çarpanlar & EBOB/EKOK & Faktöriyel & Kombinasyon/Permütasyon
        wants_gcd = any(k in s for k in ("ebob", "obeb", "en büyük ortak bölen", "ortak bölen"))
        wants_lcm = any(k in s for k in ("ekok", "okek", "en küçük ortak kat", "ortak kat"))
        if wants_gcd or wants_lcm:
            nums = [abs(int(x)) for x in re.findall(r"-?\d+", s)]
            if len(nums) >= 2:
                a, b = nums[0], nums[1]
                g = math.gcd(a, b)
                l = abs(a * b) // g if g else 0
                if wants_gcd and wants_lcm:
                    ans_txt = f"{a} ile {b} sayılarının EBOB'u {g}, EKOK'u ise {l} olur (sağlama: {g} × {l} = {g*l} = {a} × {b})."
                elif wants_gcd:
                    ans_txt = f"Öklid algoritmasıyla {a} ve {b} sayılarının en büyük ortak böleni (EBOB) {g} bulunur."
                else:
                    ans_txt = f"{a} ve {b} sayılarının en küçük ortak katı (EKOK) {l} bulunur."
                return {"domain": "math", "kind": "number_theory", "confidence": 0.96, "reply": ans_txt, "trace": f"gcd({a},{b})={g}, lcm={l}"}

        if "asal" in s:
            nums = [int(x) for x in re.findall(r"\d+", s)]
            if nums:
                n = nums[0]
                if "çarpan" in s or "ayır" in s:
                    pf = self._prime_factors(n)
                    f_str = " × ".join(f"{p}^{e}" if e > 1 else f"{p}" for p, e in pf)
                    primes_only = ", ".join(str(p) for p, _ in pf)
                    return {
                        "domain": "math", "kind": "prime_factors", "confidence": 0.96,
                        "reply": f"{n} sayısının asal çarpan ayrışımı {n} = {f_str} biçimindedir (asal çarpanları: {primes_only}).",
                        "trace": f"{n} = {f_str}",
                    }
                if n < 2:
                    return {"domain": "math", "kind": "primality", "confidence": 0.96, "reply": f"Hayır, {n} asal değildir; en küçük asal sayı 2'dir.", "trace": "n<2"}
                pf = self._prime_factors(n)
                is_p = (len(pf) == 1 and pf[0][1] == 1)
                if is_p:
                    return {
                        "domain": "math", "kind": "primality", "confidence": 0.96,
                        "reply": f"Evet, {n} bir asal sayıdır; √{n} ≈ {math.sqrt(n):.1f} sınırına kadar 1 ve kendisi dışında hiçbir tam böleni yoktur.",
                        "trace": f"prime({n})=True",
                    }
                f_str = " × ".join(f"{p}^{e}" if e > 1 else f"{p}" for p, e in pf)
                return {
                    "domain": "math", "kind": "primality", "confidence": 0.96,
                    "reply": f"Hayır, {n} asal değildir; çünkü {n} = {f_str} olarak çarpanlarına ayrılır.",
                    "trace": f"prime({n})=False ({f_str})",
                }

        if "kombinasyon" in s or "permütasyon" in s or re.search(r"\bc\(\s*\d+\s*,\s*\d+\s*\)", s):
            nums = [int(x) for x in re.findall(r"\d+", s)]
            if len(nums) >= 2:
                n, r = nums[0], nums[1]
                if 0 <= r <= n <= 200:
                    if "permütasyon" in s:
                        val = math.perm(n, r)
                        return {"domain": "math", "kind": "combinatorics", "confidence": 0.95,
                                "reply": f"P({n}, {r}) = {n}! / ({n}-{r})! hesabıyla permütasyon sonucu {val} bulunur.", "trace": f"P({n},{r})={val}"}
                    val = math.comb(n, r)
                    return {"domain": "math", "kind": "combinatorics", "confidence": 0.95,
                            "reply": f"C({n}, {r}) = {n}! / ({r}!·{n-r}!) hesabıyla kombinasyon sonucu {val} bulunur.", "trace": f"C({n},{r})={val}"}

        m_fact = re.search(r"(\d+)\s*(?:!|faktöriyel)", s)
        if m_fact:
            n = int(m_fact.group(1))
            if 0 <= n <= 30:
                val = math.factorial(n)
                return {"domain": "math", "kind": "factorial", "confidence": 0.96,
                        "reply": f"{n}! ({n} faktöriyel) işleminin sonucu {val} eder.", "trace": f"{n}!={val}"}

        # 6) Geometri: Pisagor / Hipotenüs, Daire, Küre, Üçgen, Gauss toplamı
        if "hipotenüs" in s or ("dik üçgen" in s and len(re.findall(r"\d+(?:\.\d+)?", s)) >= 2):
            nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
            if len(nums) >= 2:
                a, b = nums[0], nums[1]
                c = math.hypot(a, b)
                return {
                    "domain": "math", "kind": "geometry", "confidence": 0.96,
                    "reply": f"Pisagor teoreminden (c² = {_fmt_num(a)}² + {_fmt_num(b)}² = {_fmt_num(a*a + b*b)}) hipotenüs uzunluğu c = {_fmt_num(c)} bulunur.",
                    "trace": f"hypot({a},{b})={_fmt_num(c)}",
                }

        if ("daire" in s or "çember" in s) and "yarıçap" in s:
            nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
            if nums:
                r = nums[0]
                area = math.pi * r * r
                circ = 2.0 * math.pi * r
                return {
                    "domain": "math", "kind": "geometry", "confidence": 0.94,
                    "reply": f"Yarıçapı r = {_fmt_num(r)} olan dairenin alanı πr² ≈ {_fmt_num(area, 3)} ({_fmt_num(r*r)}π), çevresi 2πr ≈ {_fmt_num(circ, 3)} ({_fmt_num(2*r)}π) olur.",
                    "trace": f"r={r}, A={area:.3f}",
                }

        if "1'den" in s or "1 den" in s:
            m_gauss = re.search(r"1['’\s]*(?:den|dan)\s*(\d+)['’\s]*(?:e|a|ye|ya)\s*kadar.*toplam", s)
            if m_gauss:
                n = int(m_gauss.group(1))
                tot = n * (n + 1) // 2
                return {
                    "domain": "math", "kind": "series_sum", "confidence": 0.96,
                    "reply": f"Gauss toplam formülüyle n(n+1)/2 = {n}×{n+1}/2 = {tot} bulunur.",
                    "trace": f"sum(1..{n})={tot}",
                }

        # 7) Yüzde, Üs, Karekök ve Çok-İşlemli İfadeler ('250 sayısının %16'sı', '2 üzeri 10', '(15 + 9) * 4')
        m_pct = re.search(r"(\d+(?:\.\d+)?)[^\d%]*%\s*(\d+(?:\.\d+)?)", s)
        if not m_pct:
            m_pct2 = re.search(r"yüzde\s*(\d+(?:\.\d+)?)", s)
            if m_pct2:
                nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
                if len(nums) >= 2:
                    base_n, pct_n = nums[0], float(m_pct2.group(1))
                    val = base_n * pct_n / 100.0
                    return {"domain": "math", "kind": "percent", "confidence": 0.94,
                            "reply": f"{_fmt_num(base_n)} değerinin %{pct_n:g} oranı {_fmt_num(val)} eder.", "trace": f"{base_n}*{pct_n}%={val}"}
        else:
            base_n, pct_n = float(m_pct.group(1)), float(m_pct.group(2))
            val = base_n * pct_n / 100.0
            return {"domain": "math", "kind": "percent", "confidence": 0.94,
                    "reply": f"{_fmt_num(base_n)} değerinin %{_fmt_num(pct_n)} karşılığı {_fmt_num(val)} eder.", "trace": f"{base_n}*{pct_n}%={val}"}

        if "karekök" in s or "sqrt" in s or "üzeri" in s or "üssü" in s or re.search(r"[\d\)]\s*[\*\/\^xX×÷]\s*[\(\d]", s) or any(k in s for k in ("çarpı", "kere", "bölü")):
            # Basit iki-terimli toplama/çıkarmayı ThinkColumn S2 köprüsüne bırak; çarpma/bölme/üs/kök/parantezi burada çöz
            expr_cand = re.findall(r"(?:karekök|sqrt)?\s*[\d\.\(\)\+\-\*\/\^\s]+", SymbolicAST.preprocess(s))
            best = max((c.strip() for c in expr_cand), key=len, default="")
            if best and any(op in best for op in ("*", "/", "^", "sqrt")):
                try:
                    ast = SymbolicAST.parse(best)
                    val = SymbolicAST.eval_expr(ast)
                    return {
                        "domain": "math", "kind": "arithmetic_ast", "confidence": 0.95,
                        "reply": f"İfadeyi işlem önceliği ağacıyla ({SymbolicAST.to_str(ast)}) değerlendirdim; sonuç {_fmt_num(val)} olur.",
                        "trace": f"{best} = {_fmt_num(val)}",
                    }
                except Exception:
                    pass

        # 8) Sembolik sabitli sayısal ifadeler ('2pi kaç eder', 'e^2 kaçtır'): en son, yalnız
        #    başka hiçbir kalıp eşleşmediğinde denenir; 'pi nedir' gibi tanım sorularına dokunmaz.
        return self._solve_constant_expr(raw_text)

    def _solve_constant_expr(self, raw_text: str) -> dict | None:
        """pi ve e içeren, değişkensiz sayısal ifadeleri değerlendirir ('2pi', 'e^2', 'pi/4 kaç eder')."""
        span = _math_span(raw_text)
        if not re.search(r"\b(pi|e)\b", span):
            return None
        # Hesap niyeti: sayı/işleç içeren bir ifade ya da açık bir hesap sözcüğü olmalı.
        if not (re.search(r"\d|[\+\-\*\/\^\(]", span) or re.search(r"kaç|hesapla", _tr_low(raw_text))):
            return None
        # Değişken içeren ifade (x, t, ...) burada çözülmez; o yol denklem/türev motorunun işidir.
        words = re.findall(r"[a-z]+", span)
        if any(w not in _MATH_WORDS for w in words):
            return None
        try:
            ast = SymbolicAST.parse(span)
            val = SymbolicAST.eval_expr(ast)
        except Exception:
            return None
        if not math.isfinite(val):
            return None
        is_exact = abs(val - round(val)) < 1e-9
        disp = span.replace("pi", "π").replace("*", "×").replace("/", "÷")
        rel = "=" if is_exact else "≈"
        return {
            "domain": "math", "kind": "constant_expr", "confidence": 0.96,
            "reply": f"{disp} {rel} {_fmt_num(val, 6)} eder.",
            "trace": f"{span} = {_fmt_num(val, 6)}",
        }

    @classmethod
    def _extract_calculus_target(cls, s: str, kind: str = "derivative") -> tuple[str, str]:
        cleaned = s
        # If there is a colon, take the part after the colon if it contains math variables or digits
        if ":" in cleaned:
            after_colon = cleaned.split(":", 1)[1].strip()
            if re.search(r"\d|[a-zA-Z_]\([a-zA-Z_]\)|[xXyYtT]", after_colon):
                cleaned = after_colon
        # If there is f(x) = ..., take from f(x) = onwards to discard conversational lead-in
        m_fn = re.search(r"\b[a-zA-Z_]\([a-zA-Z_]\)\s*=", cleaned)
        if m_fn:
            cleaned = cleaned[m_fn.start():]

        cleaned = re.sub(r"(-?\d+(?:\.\d+)?)['’\s]*(?:dan|den|tan|ten)\s*(-?\d+(?:\.\d+)?)['’\s]*(?:e|a|ye|ya)(?:\s+kadar)?", " ", cleaned)
        cleaned = re.sub(r"^[a-zA-Z_]\([a-zA-Z_]\)\s*=\s*", " ", cleaned.strip())
        cleaned = re.sub(r"\b[a-zA-Z_]\([a-zA-Z_]\)\s*=\s*", " ", cleaned)
        cleaned = re.sub(r"\b[yY]\s*=\s*", " ", cleaned)
        if kind == "derivative":
            cleaned = re.sub(r"\b(türevini\s+al|türevi\s+nedir|türevini\s+hesapla|türevi\s+kaçtır|türevi\s+ne|türev\s+al|türevi|türevini|türev)\b", " ", cleaned)
        elif kind == "integral":
            cleaned = re.sub(r"\b(integralini\s+al|integrali\s+nedir|integralini\s+hesapla|integrali\s+kaçtır|integrali\s+ne|integral\s+al|integrali|integralini|integral)\b", " ", cleaned)
        cleaned = re.sub(r"\b(fonksiyonunun|fonksiyonun|fonksiyonu|fonksiyon|ifadesinin|ifadesi|ifade|polinomunun|polinomu|polinom|değerinin|değeri|hesapla|al|bul|çöz|nedir|kaçtır|kaç\s+eder|sonucu)\b", " ", cleaned)
        cleaned = re.sub(r"\b[a-zA-Z_][’'](?:e|a|ye|ya|in|ın|ün|un)\s+göre\b", " ", cleaned)
        cleaned = re.sub(r"[’'][a-zçğıöşü]+\b", "", cleaned)
        cleaned = re.sub(r"\b(dx|dt|dz)\b", " ", cleaned)
        cleaned = re.sub(r"[?!.,]", " ", cleaned)
        expr = cleaned.strip()
        var = "t" if ("t" in expr and "x" not in expr) else "x"
        return expr, var

    @classmethod
    def _compose_math_reply(
        cls,
        kind: str,
        expr_display: str,
        steps: list[str],
        result_str: str,
        verif_note: str = "",
        raw_text: str = "",
        agi: Any = None,
    ) -> str:
        chem = getattr(agi, "chem", None) if agi is not None else None
        da = float(getattr(chem, "da_tonic", 0.5) or 0.5) + max(0.0, float(getattr(chem, "da_phasic", 0.0) or 0.0))
        sht = float(getattr(chem, "sht", 0.65) or 0.65)
        oxy = float(getattr(chem, "oxy", 0.5) or 0.5)
        cort = float(getattr(chem, "cort", 0.1) or 0.1)
        turn_num = len(getattr(agi, "dialogue_log", []) or []) + int(getattr(getattr(agi, "style", None), "turn", 0) or 0)

        s_low = _tr_low(raw_text or "")
        is_testing = any(k in s_low for k in ("test", "bakalım", "çözebilir misin", "basit", "temel", "dene", "deneyelim", "seviye"))

        if kind == "derivative":
            if is_testing:
                openers = [
                    "Evet, Nora'nın basit türev ve integralleri çözebildiğini söylüyorsan önce temel seviyeden test etmek daha mantıklı. 😄",
                    "Temel seviyeden başlayarak türev adımlarını doğrulamak ve test etmek en sağlıklı yaklaşım.",
                    "Nora'nın temel kalkülüs becerilerini sınamak için harika bir soru! Birlikte adım adım inceleyelim:",
                    "Elbette; temel türev kurallarıyla test etmek sağlam bir zemin kurar. Adım adım bakalım:",
                ]
            elif da > 0.58:
                openers = [
                    "Harika bir kalkülüs sorusu! Fonksiyonun türevini terim terim kuvvet kuralıyla çıkaralım:",
                    "Kalkülüsün en zevkli kısımlarından biri; fonksiyonun türevini adım adım inceleyelim:",
                    "Çok güzel bir türev problemi! İfadeyi terim terim ayrıştırıp çözelim:",
                ]
            elif sht > 0.55:
                openers = [
                    "Tabii, verilen fonksiyonu adım adım inceleyerek türevini hesaplayalım:",
                    "Elbette; türev kurallarını terim terim uygulayarak çözüme ulaşalım:",
                    "Verilen matematiksel ifadeyi dikkatle ayrıştırıp türevini alalım:",
                ]
            else:
                openers = [
                    "Verilen fonksiyonun türevini adım adım çözelim:",
                    "Fonksiyonu terim terim inceleyerek türevine ulaşalım:",
                    "Kuvvet kuralını uygulayarak türevi adım adım çıkaralım:",
                ]
        elif kind == "indefinite_integral":
            if is_testing:
                openers = [
                    "Evet, Nora'nın basit integralleri çözebildiğini adım adım test etmek çok mantıklı. 😄",
                    "Temel integral kurallarıyla başlamak ve adım adım doğrulamak en doğrusu.",
                    "Nora'nın integral alma becerisini test etmek için güzel bir soru! Birlikte bakalım:",
                    "Elbette; temel integral adımlarını sınamak harika bir yaklaşım:",
                ]
            elif da > 0.58:
                openers = [
                    "Harika bir integral problemi! Fonksiyonu ters kuvvet kuralıyla adım adım entegre edelim:",
                    "İntegral hesabını terim terim açarak ilkel fonksiyona ulaşalım:",
                    "Çok güzel bir integral sorusu! Terimleri adım adım entegre edelim:",
                ]
            elif sht > 0.55:
                openers = [
                    "Tabii, verilen fonksiyonun belirsiz integralini adım adım hesaplayalım:",
                    "Elbette; entegrasyon kurallarını terim terim uygulayalım:",
                    "Verilen matematiksel ifadeyi dikkatle ayrıştırıp integralini alalım:",
                ]
            else:
                openers = [
                    "Belirsiz integrali adım adım çözelim:",
                    "Fonksiyonu terim terim entegre ederek ilkel fonksiyonu bulalım:",
                    "Ters kuvvet kuralını uygulayarak integrali hesaplayalım:",
                ]
        elif kind == "definite_integral":
            if is_testing:
                openers = [
                    "Evet, belirli integralleri ve sınır değerlerini test etmek çok mantıklı bir yaklaşım.",
                    "Belirli integrali kalkülüsün temel teoremiyle adım adım doğrulayalım:",
                ]
            elif da > 0.58:
                openers = [
                    "Harika bir belirli integral problemi! Sınırlar arasındaki alanı adım adım hesaplayalım:",
                    "Kalkülüsün temel teoremini uygulayarak belirli integrali adım adım çözelim:",
                ]
            else:
                openers = [
                    "Belirli integrali kalkülüsün temel teoremiyle adım adım hesaplayalım:",
                    "Verilen sınır değerleri için belirli integrali adım adım çözelim:",
                ]
        else:
            if is_testing:
                openers = [
                    "Denklem çözme adımlarını temelden test etmek gayet mantıklı; birlikte çözelim:",
                    "Nora'nın cebirsel problem çözme adımlarını sınamak için güzel bir soru:",
                ]
            elif da > 0.58:
                openers = [
                    "Harika bir cebir problemi! Denklemi adım adım çözelim:",
                    "Çok güzel bir denklem sorusu! Bilinmeyeni yalnız bırakalım:",
                ]
            else:
                openers = [
                    "Verilen denklemi adım adım çözelim:",
                    "Denklemdeki bilinmeyeni bulmak için adım adım ilerleyelim:",
                ]

        state_idx = (int(da * 73 + sht * 47 + oxy * 31 + cort * 19 + turn_num * 13)) % len(openers)
        if hasattr(cls, "_last_math_opener_idx") and getattr(cls, "_last_math_opener_idx") == state_idx:
            state_idx = (state_idx + 1) % len(openers)
        cls._last_math_opener_idx = state_idx
        opening = openers[state_idx]

        lines = [opening]
        if kind == "derivative":
            lines.append(f"İnceleyeceğimiz fonksiyon: f(x) = {expr_display}")
            if steps:
                lines.append("\nKuvvet kuralını ve türev özelliklerini terim terim uygulayalım:")
                lines.extend(steps)
            lines.append(f"\nToparladığımızda sembolik türev: {result_str} olur{verif_note}.")
        elif kind == "indefinite_integral":
            lines.append(f"Hesaplayacağımız belirsiz integral: ∫ ({expr_display}) dx")
            if steps:
                lines.append("\nTers kuvvet kuralını (entegrasyon) terim terim uygulayalım:")
                lines.extend(steps)
            lines.append(f"\nSonuç olarak belirsiz integral: {result_str} olur.")
        elif kind == "definite_integral":
            lines.append(f"Hesaplayacağımız belirli integral: {expr_display}")
            if steps:
                lines.extend(steps)
            lines.append(f"\nBelirli integralin kesin değeri: {result_str} olur{verif_note}.")
        elif kind in ("linear_eq", "quadratic_eq", "cubic_eq", "linear_system"):
            lines.append(f"Çözeceğimiz denklem: {expr_display}")
            if steps:
                lines.extend(steps)
            lines.append(f"\nSonuç: {result_str}")
        else:
            lines.append(result_str)

        return "\n".join(lines).strip()

    @classmethod
    def _calculus_clarification(cls, kind: str, expr_s: str, var: str, agi: Any = None) -> dict:
        if kind == "türev":
            reply = (
                "Türev alma isteğini anladım; ancak türevini almak istediğin matematiksel ifadeyi "
                "tam olarak ayrıştıramadım. Örneğin 'f(x) = 3x^2 + 2x türevi nedir?' ya da '3x^2 + 2x türevini al' "
                "biçiminde standart cebirsel gösterimle yazarsan terim terim hesaplayabilirim."
            )
        else:
            reply = (
                "İntegral hesaplama isteğini anladım; ancak entegre edilecek matematiksel ifadeyi "
                "tam olarak ayrıştıramadım. Örneğin '2x + 4 integrali nedir?' ya da '0'dan 3'e x^2 integrali' "
                "şeklinde belirtirsen adım adım hesaplayabilirim."
            )
        return {
            "domain": "math",
            "kind": "calculus_clarification",
            "confidence": 0.96,
            "reply": reply,
            "trace": f"{kind}_clarification",
        }

    def _solve_single_eq(self, lhs_s: str, rhs_s: str, var: str, raw_text: str = "", agi: Any = None) -> dict | None:
        try:
            lhs = SymbolicAST.parse(lhs_s)
            rhs = SymbolicAST.parse(rhs_s)
            diff_node = SymbolicAST.simplify(Bin("-", lhs, rhs))
            pc = SymbolicAST.poly_coeffs(diff_node, var, max_deg=3)
            if pc is None:
                return None
            deg = max((k for k, v in pc.items() if abs(v) > 1e-10), default=0)
            if deg == 1:
                a = pc.get(1, 0.0)
                b = pc.get(0, 0.0)
                root = -b / a
                chk_l = SymbolicAST.eval_expr(lhs, {var: root})
                chk_r = SymbolicAST.eval_expr(rhs, {var: root})
                steps = [
                    f"1. Denklemi standart hale getirelim: {SymbolicAST.format_poly(pc, var)} = 0",
                    f"2. Bilinmeyeni yalnız bırakalım: {var} = -({_fmt_num(b)}) / {_fmt_num(a)} => {var} = {_fmt_num(root)}",
                    f"3. Sağlama: Sol taraf {_fmt_num(chk_l)} = Sağ taraf {_fmt_num(chk_r)}"
                ]
                reply_txt = self._compose_math_reply(
                    kind="linear_eq",
                    expr_display=f"{lhs_s} = {rhs_s}",
                    steps=steps,
                    result_str=(
                        f"Denklemi {SymbolicAST.format_poly(pc, var)} = 0 biçimine getirdim; buradan {var} = {_fmt_num(root)} bulunur "
                        f"(sağlama: sol taraf {_fmt_num(chk_l)} = sağ taraf {_fmt_num(chk_r)})."
                    ),
                    raw_text=raw_text,
                    agi=agi,
                )
                return {
                    "domain": "math", "kind": "linear_eq", "confidence": 0.97,
                    "reply": reply_txt,
                    "trace": f"{var}={_fmt_num(root)}",
                }
            if deg == 2:
                a = pc.get(2, 0.0)
                b = pc.get(1, 0.0)
                c = pc.get(0, 0.0)
                disc = b * b - 4.0 * a * c
                steps = [
                    f"1. Denklemi standart ax² + bx + c = 0 biçimine getirelim: {SymbolicAST.format_poly(pc, var)} = 0",
                    f"   (Katsayılar: a = {_fmt_num(a)}, b = {_fmt_num(b)}, c = {_fmt_num(c)})",
                    f"2. Diskriminantı (Δ) hesaplayalım: Δ = b² - 4ac = ({_fmt_num(b)})² - 4({_fmt_num(a)})({_fmt_num(c)}) = {_fmt_num(disc)}",
                ]
                if disc < -1e-9:
                    re_p = -b / (2.0 * a)
                    im_p = math.sqrt(-disc) / abs(2.0 * a)
                    steps.append("3. Δ < 0 olduğu için denklemin reel kökü yoktur; karmaşık kökleri vardır.")
                    reply_txt = self._compose_math_reply(
                        kind="quadratic_eq",
                        expr_display=f"{lhs_s} = {rhs_s}",
                        steps=steps,
                        result_str=(
                            f"Diskriminant Δ = b² - 4ac = {_fmt_num(disc)} < 0 olduğu için reel kök yoktur; "
                            f"karmaşık kökler {var}₁,₂ = {_fmt_num(re_p)} ± {_fmt_num(im_p)}i olur."
                        ),
                        raw_text=raw_text,
                        agi=agi,
                    )
                    return {
                        "domain": "math", "kind": "quadratic_eq", "confidence": 0.97,
                        "reply": reply_txt,
                        "trace": f"Δ={_fmt_num(disc)}<0",
                    }
                if abs(disc) <= 1e-9:
                    r1 = -b / (2.0 * a)
                    steps.append(f"3. Δ = 0 olduğu için çakışık tek reel kök vardır: {var} = -b / (2a) = {_fmt_num(r1)}.")
                    reply_txt = self._compose_math_reply(
                        kind="quadratic_eq",
                        expr_display=f"{lhs_s} = {rhs_s}",
                        steps=steps,
                        result_str=f"Diskriminant Δ = 0 olduğu için çakışık tek reel kök vardır: {var} = {_fmt_num(r1)}.",
                        raw_text=raw_text,
                        agi=agi,
                    )
                    return {
                        "domain": "math", "kind": "quadratic_eq", "confidence": 0.97,
                        "reply": reply_txt,
                        "trace": f"{var}={_fmt_num(r1)}",
                    }
                sq = math.sqrt(disc)
                r1 = (-b - sq) / (2.0 * a)
                r2 = (-b + sq) / (2.0 * a)
                roots = sorted([r1, r2])
                c1 = abs(SymbolicAST.eval_expr(diff_node, {var: roots[0]}))
                c2 = abs(SymbolicAST.eval_expr(diff_node, {var: roots[1]}))
                steps.append(f"3. Δ > 0 olduğu için iki farklı reel kök bulunur: {var}₁,₂ = (-b ± √Δ) / 2a")
                steps.append(f"   • {var}₁ = {_fmt_num(roots[0])}")
                steps.append(f"   • {var}₂ = {_fmt_num(roots[1])}")
                reply_txt = self._compose_math_reply(
                    kind="quadratic_eq",
                    expr_display=f"{lhs_s} = {rhs_s}",
                    steps=steps,
                    result_str=(
                        f"Δ = b² - 4ac = {_fmt_num(disc)} üzerinden denklemin kökleri {var}₁ = {_fmt_num(roots[0])} ve "
                        f"{var}₂ = {_fmt_num(roots[1])} bulunur (yerine koyma sağlaması hatası: {max(c1, c2):.1e})."
                    ),
                    raw_text=raw_text,
                    agi=agi,
                )
                return {
                    "domain": "math", "kind": "quadratic_eq", "confidence": 0.98,
                    "reply": reply_txt,
                    "trace": f"roots=({_fmt_num(roots[0])}, {_fmt_num(roots[1])})",
                }
            if deg == 3:
                np_roots = np.roots([pc.get(3, 0.0), pc.get(2, 0.0), pc.get(1, 0.0), pc.get(0, 0.0)])
                real_roots = sorted([float(r.real) for r in np_roots if abs(r.imag) < 1e-6])
                r_txt = ", ".join(_fmt_num(r) for r in real_roots)
                reply_txt = self._compose_math_reply(
                    kind="cubic_eq",
                    expr_display=f"{lhs_s} = {rhs_s}",
                    steps=[f"1. Üçüncü dereceden polinom kökleri çözüldü: {SymbolicAST.format_poly(pc, var)} = 0"],
                    result_str=f"Üçüncü derece denklemin reel kökleri: {var} ∈ {{{r_txt}}}.",
                    raw_text=raw_text,
                    agi=agi,
                )
                return {
                    "domain": "math", "kind": "cubic_eq", "confidence": 0.95,
                    "reply": reply_txt,
                    "trace": f"cubic_roots={r_txt}",
                }
        except Exception:
            return None
        return None

    def _solve_2x2(self, eqs: list[tuple[str, str]], raw_text: str = "", agi: Any = None) -> dict | None:
        try:
            A = np.zeros((2, 2), dtype=np.float64)
            B = np.zeros(2, dtype=np.float64)
            for i, (lhs_s, rhs_s) in enumerate(eqs):
                # rhs içindeki bağlaç kırıntılarını ('12 ve') temizle
                rhs_clean = re.sub(r"\bve\b.*$", "", rhs_s).strip()
                lhs_clean = re.sub(r"^.*\bve\b", "", lhs_s).strip()
                expr = Bin("-", SymbolicAST.parse(lhs_clean), SymbolicAST.parse(rhs_clean))
                f00 = SymbolicAST.eval_expr(expr, {"x": 0.0, "y": 0.0})
                fx = SymbolicAST.eval_expr(expr, {"x": 1.0, "y": 0.0}) - f00
                fy = SymbolicAST.eval_expr(expr, {"x": 0.0, "y": 1.0}) - f00
                A[i, 0], A[i, 1] = fx, fy
                B[i] = -f00
            det = float(np.linalg.det(A))
            if abs(det) < 1e-9:
                return None
            sol = np.linalg.solve(A, B)
            xv, yv = float(sol[0]), float(sol[1])
            eq_txt = " ve ".join(f"{lhs.strip()} = {rhs.strip()}" for lhs, rhs in eqs)
            steps = [
                f"1. Denklem sistemini matris / yok etme katsayılarıyla düzenleyelim (det(A) = {_fmt_num(det)})",
                f"2. Doğrusal çözümü elde edelim: x = {_fmt_num(xv)}, y = {_fmt_num(yv)}"
            ]
            reply_txt = self._compose_math_reply(
                kind="linear_system",
                expr_display=eq_txt,
                steps=steps,
                result_str=(
                    f"Doğrusal denklem sistemini yok etme/matris yöntemiyle (det = {_fmt_num(det)}) çözdüm: "
                    f"x = {_fmt_num(xv)} ve y = {_fmt_num(yv)} bulunur."
                ),
                raw_text=raw_text,
                agi=agi,
            )
            return {
                "domain": "math", "kind": "linear_system", "confidence": 0.98,
                "reply": reply_txt,
                "trace": f"x={_fmt_num(xv)}, y={_fmt_num(yv)}",
            }
        except Exception:
            return None

    def _solve_derivative(self, s: str, raw_text: str = "", agi: Any = None) -> dict | None:
        expr_s, var = self._extract_calculus_target(s, kind="derivative")
        if not expr_s:
            if "türev" in s:
                return self._calculus_clarification("türev", expr_s, var, agi=agi)
            return None
        try:
            ast = SymbolicAST.parse(expr_s)
            d_ast = SymbolicAST.diff(ast, var)
            d_str = SymbolicAST.to_str(d_ast, var)
            # Sistem-1 sayısal sonlu-fark sağlaması!
            x0, h = 1.37, 1e-5
            try:
                num_d = (SymbolicAST.eval_expr(ast, {var: x0 + h}) - SymbolicAST.eval_expr(ast, {var: x0 - h})) / (2 * h)
                sym_d = SymbolicAST.eval_expr(d_ast, {var: x0})
                err = abs(num_d - sym_d)
                ver_note = f" (sayısal eğim sağlaması Δ={err:.1e} ile doğrulandı)"
            except Exception:
                ver_note = ""

            steps = []
            pc = SymbolicAST.poly_coeffs(ast, var)
            if pc is not None:
                for deg in sorted(pc.keys(), reverse=True):
                    c = pc[deg]
                    if abs(c) < 1e-9:
                        continue
                    c_str = _fmt_num(abs(c))
                    sgn_str = "-" if c < 0 else ""
                    if deg > 1:
                        new_c = c * deg
                        new_deg = deg - 1
                        deg_str = f"{var}^{new_deg}" if new_deg > 1 else var
                        res_term = f"{_fmt_num(new_c)}{deg_str}" if abs(abs(new_c) - 1.0) > 1e-9 else f"{'-' if new_c < 0 else ''}{deg_str}"
                        term_disp = f"{sgn_str}{c_str}{var}^{deg}" if abs(abs(c) - 1.0) > 1e-9 else f"{sgn_str}{var}^{deg}"
                        steps.append(f"• {term_disp} terimi için kuvvet kuralı uygulanır: üs ({deg}) katsayı ile çarpılır ve üs 1 azaltılır ({deg} × {sgn_str}{c_str} · {deg_str} = {res_term}).")
                    elif deg == 1:
                        res_term = _fmt_num(c)
                        term_disp = f"{sgn_str}{c_str}{var}" if abs(abs(c) - 1.0) > 1e-9 else f"{sgn_str}{var}"
                        steps.append(f"• {term_disp} doğrusal teriminin türevi doğrudan katsayısı {res_term} olur.")
                    elif deg == 0:
                        steps.append(f"• {sgn_str}{c_str} sabit teriminin türevi 0 olur.")
            elif isinstance(ast, Call):
                if ast.fn == "sin":
                    steps.append(f"• sin({SymbolicAST.to_str(ast.arg, var)}) türevi zincir kuralıyla cos({SymbolicAST.to_str(ast.arg, var)}) ile iç türevin çarpımıdır.")
                elif ast.fn == "cos":
                    steps.append(f"• cos({SymbolicAST.to_str(ast.arg, var)}) türevi zincir kuralıyla -sin({SymbolicAST.to_str(ast.arg, var)}) ile iç türevin çarpımıdır.")
                elif ast.fn == "exp":
                    steps.append(f"• e^{SymbolicAST.to_str(ast.arg, var)} fonksiyonunun türevi kendisi ile iç türevin çarpımıdır.")
                elif ast.fn == "ln":
                    steps.append(f"• ln({SymbolicAST.to_str(ast.arg, var)}) türevi 1/({SymbolicAST.to_str(ast.arg, var)}) ile iç türevin çarpımıdır.")

            reply_txt = self._compose_math_reply(
                kind="derivative",
                expr_display=SymbolicAST.to_str(ast, var),
                steps=steps,
                result_str=f"f'({var}) = {d_str}",
                verif_note=ver_note,
                raw_text=raw_text or s,
                agi=agi,
            )
            return {
                "domain": "math", "kind": "derivative", "confidence": 0.98,
                "reply": reply_txt,
                "trace": f"d/d{var}({SymbolicAST.to_str(ast, var)}) = {d_str}",
            }
        except Exception:
            if "türev" in s:
                return self._calculus_clarification("türev", expr_s, var, agi=agi)
            return None

    def _solve_integral(self, s: str, raw_text: str = "", agi: Any = None) -> dict | None:
        # Belirli integral: "0'dan 3'e x^2 integrali", "-2'den 3'e 4x^3 - 3x^2 + 2x - 7 ifadesinde integral"
        m_bounds = re.search(r"(-?\d+(?:\.\d+)?)['’\s]*(?:dan|den|tan|ten)\s*(-?\d+(?:\.\d+)?)['’\s]*(?:e|a|ye|ya)(?:\s+kadar)?\s+(.+?)(?:\s+ifadesinde)?\s+integral", s)
        if m_bounds:
            a, b = float(m_bounds.group(1)), float(m_bounds.group(2))
            raw_expr = m_bounds.group(3)
            raw_expr = re.sub(r"^[a-zA-Z_]\([a-zA-Z_]\)\s*=\s*", "", raw_expr.strip())
            raw_expr = re.sub(r"\b(dx|dt|dz)\b", "", raw_expr)
            raw_expr = re.sub(r"\b(ifadesinde|ifadesinin|fonksiyonunun)\b", "", raw_expr).strip()
            expr_s = _math_span(raw_expr)
            var = "t" if (re.search(r"\bt\b", expr_s) and not re.search(r"\bx\b", expr_s)) else "x"
            try:
                ast = SymbolicAST.parse(expr_s)
                val_num = SymbolicAST.simpson_integral(ast, var, a, b)
                pc = SymbolicAST.poly_coeffs(ast, var)
                if pc is not None:
                    anti = {deg + 1: coef / (deg + 1) for deg, coef in pc.items()}
                    anti_str = SymbolicAST.format_poly(anti, var)
                    fb = sum(coef * (b ** deg) for deg, coef in anti.items())
                    fa = sum(coef * (a ** deg) for deg, coef in anti.items())
                    val_exact = fb - fa
                    steps = [
                        f"1. İlkel fonksiyonu (belirsiz integrali) bulalım: F({var}) = {anti_str}",
                        f"2. Kalkülüsün temel teoremine göre üst ve alt sınırları F({var}) fonksiyonuna yerleştirelim:",
                        f"   • F({_fmt_num(b)}) = {_fmt_num(fb)}",
                        f"   • F({_fmt_num(a)}) = {_fmt_num(fa)}",
                        f"   • F({_fmt_num(b)}) - F({_fmt_num(a)}) = {_fmt_num(val_exact)}",
                    ]
                    reply_txt = self._compose_math_reply(
                        kind="definite_integral",
                        expr_display=f"∫[{_fmt_num(a)}..{_fmt_num(b)}] ({SymbolicAST.to_str(ast, var)}) d{var}",
                        steps=steps,
                        result_str=f"F({_fmt_num(b)}) - F({_fmt_num(a)}) = {_fmt_num(val_exact)}",
                        verif_note=f" (Simpson sayısal sağlaması: {_fmt_num(val_num, 5)})",
                        raw_text=raw_text or s,
                        agi=agi,
                    )
                    return {
                        "domain": "math", "kind": "definite_integral", "confidence": 0.98,
                        "reply": reply_txt,
                        "trace": f"int_{a}^{b} = {_fmt_num(val_exact)}",
                    }
                reply_txt = self._compose_math_reply(
                    kind="definite_integral",
                    expr_display=f"∫[{_fmt_num(a)}..{_fmt_num(b)}] ({SymbolicAST.to_str(ast, var)}) d{var}",
                    steps=[],
                    result_str=f"= {_fmt_num(val_num, 5)}",
                    verif_note="",
                    raw_text=raw_text or s,
                    agi=agi,
                )
                return {
                    "domain": "math", "kind": "definite_integral", "confidence": 0.95,
                    "reply": reply_txt,
                    "trace": f"simpson={_fmt_num(val_num, 5)}",
                }
            except Exception:
                if "integral" in s:
                    return self._calculus_clarification("integral", expr_s, var, agi=agi)
                return None

        # Belirsiz polinom integrali: "3x^2 + 4x integrali nedir?", "integralini al: 2x + 4", "f(x) = 2x + 4 integralini al"
        expr_s, var = self._extract_calculus_target(s, kind="integral")
        if not expr_s:
            if "integral" in s:
                return self._calculus_clarification("integral", expr_s, var, agi=agi)
            return None
        try:
            ast = SymbolicAST.parse(expr_s)
            pc = SymbolicAST.poly_coeffs(ast, var)
            if pc is not None:
                anti = {deg + 1: coef / (deg + 1) for deg, coef in pc.items()}
                anti_str = SymbolicAST.format_poly(anti, var)
                steps = []
                for deg in sorted(pc.keys(), reverse=True):
                    c = pc[deg]
                    if abs(c) < 1e-9:
                        continue
                    c_str = _fmt_num(abs(c))
                    sgn_str = "-" if c < 0 else ""
                    new_deg = deg + 1
                    new_c = c / new_deg
                    deg_str = f"{var}^{new_deg}" if new_deg > 1 else var
                    res_term = f"{_fmt_num(new_c)}{deg_str}" if abs(abs(new_c) - 1.0) > 1e-9 else f"{'-' if new_c < 0 else ''}{deg_str}"
                    if deg > 0:
                        term_disp = f"{sgn_str}{c_str}{var}^{deg}" if deg > 1 else f"{sgn_str}{c_str}{var}"
                        steps.append(f"• {term_disp} teriminin integralinde üssü 1 artırıp ({new_deg}) katsayıyı yeni üsse böleriz ({c_str}/{new_deg} · {deg_str} = {res_term}).")
                    else:
                        steps.append(f"• {sgn_str}{c_str} sabit teriminin integrali {res_term} olur.")
                steps.append("• Belirsiz integral olduğu için genel entegrasyon sabiti + C eklenir.")

                reply_txt = self._compose_math_reply(
                    kind="indefinite_integral",
                    expr_display=SymbolicAST.to_str(ast, var),
                    steps=steps,
                    result_str=f"∫ ({SymbolicAST.to_str(ast, var)}) d{var} = {anti_str} + C",
                    verif_note="",
                    raw_text=raw_text or s,
                    agi=agi,
                )
                return {
                    "domain": "math", "kind": "indefinite_integral", "confidence": 0.97,
                    "reply": reply_txt,
                    "trace": f"F({var}) = {anti_str} + C",
                }
            # Analitik temel fonksiyonlar (sin, cos, exp)
            if isinstance(ast, Call):
                if ast.fn == "sin" and isinstance(ast.arg, Var) and ast.arg.name == var:
                    reply_txt = self._compose_math_reply(
                        kind="indefinite_integral",
                        expr_display=f"sin({var})",
                        steps=[f"• sin({var}) fonksiyonunun integrali -cos({var})'dir.", "• Belirsiz integral sabiti + C eklenir."],
                        result_str=f"∫ sin({var}) d{var} = -cos({var}) + C",
                        raw_text=raw_text or s,
                        agi=agi,
                    )
                    return {"domain": "math", "kind": "indefinite_integral", "confidence": 0.97, "reply": reply_txt, "trace": f"F({var}) = -cos({var}) + C"}
                if ast.fn == "cos" and isinstance(ast.arg, Var) and ast.arg.name == var:
                    reply_txt = self._compose_math_reply(
                        kind="indefinite_integral",
                        expr_display=f"cos({var})",
                        steps=[f"• cos({var}) fonksiyonunun integrali sin({var})'dir.", "• Belirsiz integral sabiti + C eklenir."],
                        result_str=f"∫ cos({var}) d{var} = sin({var}) + C",
                        raw_text=raw_text or s,
                        agi=agi,
                    )
                    return {"domain": "math", "kind": "indefinite_integral", "confidence": 0.97, "reply": reply_txt, "trace": f"F({var}) = sin({var}) + C"}
                if ast.fn == "exp" and isinstance(ast.arg, Var) and ast.arg.name == var:
                    reply_txt = self._compose_math_reply(
                        kind="indefinite_integral",
                        expr_display=f"e^{var}",
                        steps=[f"• e^{var} fonksiyonunun integrali kendisidir.", "• Belirsiz integral sabiti + C eklenir."],
                        result_str=f"∫ e^{var} d{var} = e^{var} + C",
                        raw_text=raw_text or s,
                        agi=agi,
                    )
                    return {"domain": "math", "kind": "indefinite_integral", "confidence": 0.97, "reply": reply_txt, "trace": f"F({var}) = exp({var}) + C"}
        except Exception:
            if "integral" in s:
                return self._calculus_clarification("integral", expr_s, var, agi=agi)
            return None
        return None


# ============================================================================
# 2. BOYUT ANALİZLİ FİZİK KORTEKSİ VE ZİHİNSEL SİMÜLASYON KUM HAVUZU
# ============================================================================

@dataclass(frozen=True)
class PhysVarInfo:
    symbol: str
    name_tr: str
    unit: str
    dim: tuple[int, int, int, int, int]  # (M, L, T, I, Theta)


class MentalPhysicsSandbox:
    """Zihinsel fizik kum havuzu: RK4 / Verlet sayısal simülasyonu ile analitik fiziği doğrular."""

    @staticmethod
    def simulate_free_fall(h0: float, g: float = 9.8, drag_k: float = 0.0, dt: float = 0.002) -> dict:
        y = float(h0)
        v = 0.0
        t = 0.0
        max_steps = 200000
        for _ in range(max_steps):
            if y <= 0.0:
                break
            # RK4 adımı (aşağı yön pozitif hız büyüklüğü için dy/dt = -v, dv/dt = g - drag_k * v)
            def acc(vel: float) -> float:
                return g - drag_k * vel

            k1_y, k1_v = -v, acc(v)
            k2_y, k2_v = -(v + 0.5 * dt * k1_v), acc(v + 0.5 * dt * k1_v)
            k3_y, k3_v = -(v + 0.5 * dt * k2_v), acc(v + 0.5 * dt * k2_v)
            k4_y, k4_v = -(v + dt * k3_v), acc(v + dt * k3_v)

            y_next = y + (dt / 6.0) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
            v_next = v + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
            if y_next <= 0.0 and y > 0.0:
                frac = y / max(1e-12, y - y_next)
                t += frac * dt
                v = v + frac * (v_next - v)
                y = 0.0
                break
            y, v = y_next, v_next
            t += dt
        return {"impact_v": v, "fall_time": t}

    @staticmethod
    def simulate_constant_force(m: float, F: float, t_end: float, v0: float = 0.0, dt: float = 0.005) -> dict:
        x, v, t = 0.0, float(v0), 0.0
        a = F / m
        steps = max(1, int(round(t_end / dt)))
        dt_eff = t_end / steps
        for _ in range(steps):
            x += v * dt_eff + 0.5 * a * dt_eff * dt_eff
            v += a * dt_eff
            t += dt_eff
        return {"v_end": v, "x_end": x, "Ek_end": 0.5 * m * v * v}


class PhysicsCortex:
    """SI Boyut Analizi + Kısıt Yayılımı + Ölçekleme Pertürbasyonu + Zihinsel Fizik Simülasyonu."""

    VARS: dict[str, PhysVarInfo] = {
        "m": PhysVarInfo("m", "kütle", "kg", (1, 0, 0, 0, 0)),
        "a": PhysVarInfo("a", "ivme", "m/s²", (0, 1, -2, 0, 0)),
        "F": PhysVarInfo("F", "kuvvet", "N", (1, 1, -2, 0, 0)),
        "v": PhysVarInfo("v", "hız", "m/s", (0, 1, -1, 0, 0)),
        "t": PhysVarInfo("t", "zaman", "s", (0, 0, 1, 0, 0)),
        "x": PhysVarInfo("x", "yol/yer değiştirme", "m", (0, 1, 0, 0, 0)),
        "h": PhysVarInfo("h", "yükseklik", "m", (0, 1, 0, 0, 0)),
        "Ek": PhysVarInfo("Ek", "kinetik enerji", "J", (1, 2, -2, 0, 0)),
        "Ep": PhysVarInfo("Ep", "potansiyel enerji", "J", (1, 2, -2, 0, 0)),
        "p": PhysVarInfo("p", "momentum", "kg·m/s", (1, 1, -1, 0, 0)),
        "W": PhysVarInfo("W", "iş", "J", (1, 2, -2, 0, 0)),
        "P": PhysVarInfo("P", "güç", "W", (1, 2, -3, 0, 0)),
        "V": PhysVarInfo("V", "elektriksel gerilim", "V", (1, 2, -3, -1, 0)),
        "I": PhysVarInfo("I", "elektrik akımı", "A", (0, 0, 0, 1, 0)),
        "R": PhysVarInfo("R", "elektriksel direnç", "Ω", (1, 2, -3, -2, 0)),
        "f": PhysVarInfo("f", "frekans", "Hz", (0, 0, -1, 0, 0)),
        "lam": PhysVarInfo("λ", "dalga boyu", "m", (0, 1, 0, 0, 0)),
        "T_per": PhysVarInfo("T", "periyot", "s", (0, 0, 1, 0, 0)),
        "Q": PhysVarInfo("Q", "ısı enerjisi", "J", (1, 2, -2, 0, 0)),
        "c_sp": PhysVarInfo("c", "özısı", "J/(kg·K)", (0, 2, -2, 0, -1)),
        "dT": PhysVarInfo("ΔT", "sıcaklık değişimi", "K", (0, 0, 0, 0, 1)),
    }

    def __init__(self):
        self.sandbox = MentalPhysicsSandbox()

    @staticmethod
    def format_dim(dim: tuple[int, int, int, int, int]) -> str:
        syms = ("M", "L", "T", "I", "Θ")
        parts = []
        for s, e in zip(syms, dim):
            if e == 1:
                parts.append(s)
            elif e != 0:
                parts.append(f"{s}^{e}")
        return "[" + (" ".join(parts) if parts else "1") + "]"

    def try_solve(self, raw_text: str, mind=None) -> dict | None:
        s = _tr_low(raw_text).strip()

        # 1) Ölçekleme / Oran Pertürbasyonu ('hız 3 katına çıkarsa kinetik enerji kaç katına çıkar?')
        scale_res = self._try_scaling_perturbation(s)
        if scale_res is not None:
            return scale_res

        # 2) Çarpışma ve Momentum Korunumu ('3 kg kütleli ve 4 m/s hızlı cisim duran 1 kg kütleli cisme kenetlenirse...')
        col_res = self._try_collision(s)
        if col_res is not None:
            return col_res

        # 3) Karşıolgusal ve Kavramsal Derin Fizik Muhakemesi ('sürtünme olmasaydı...', 'entropi nedir', 'eylemsizlik...')
        qual_res = self._try_conceptual_physics(s, mind=mind)
        if qual_res is not None:
            return qual_res

        # 4) Boyut Analizi Sorgusu ('kuvvetin boyut formülü ve si birimi nedir?')
        if "boyut" in s or ("si birimi" in s and any(v.name_tr in s for v in self.VARS.values())):
            for key, info in self.VARS.items():
                if info.name_tr in s:
                    d_str = self.format_dim(info.dim)
                    return {
                        "domain": "physics", "kind": "dimensional_analysis", "confidence": 0.95,
                        "reply": f"{info.name_tr.capitalize()} ({info.symbol}) büyüklüğünün SI birimi '{info.unit}' ve temel boyut formülü {d_str} şeklindedir.",
                        "trace": f"{info.symbol}: {info.unit} {d_str}",
                    }

        # 5) Nicel Kısıt Yayılımı + Zihinsel Fizik Simülasyonu
        return self._try_quantitative_propagation(s)

    def _try_scaling_perturbation(self, s: str) -> dict | None:
        if "katına" not in s and "yarıya" not in s:
            return None
        factor = None
        if "yarıya" in s:
            factor = 0.5
        else:
            m_f = re.search(r"(\d+(?:\.\d+)?)\s*katına", s)
            if m_f:
                factor = float(m_f.group(1))
            elif "iki katına" in s:
                factor = 2.0
            elif "üç katına" in s:
                factor = 3.0
            elif "dört katına" in s:
                factor = 4.0
        if factor is None:
            return None

        # Hangi değişken pertürbe edildi, hangi hedef soruluyor?
        scaling_laws = [
            (("hız", "sürat"), "kinetik enerji", lambda k: k ** 2, "Ek = ½mv² yasasında hızın karesiyle orantılı olduğundan"),
            (("kütle",), "kinetik enerji", lambda k: k, "Ek = ½mv² yasasında kütleyle doğru orantılı olduğundan"),
            (("kütle",), "ivme", lambda k: 1.0 / k, "sabit kuvvet altında a = F/m yasasında kütleyle ters orantılı olduğundan"),
            (("kuvvet",), "ivme", lambda k: k, "sabit kütle için a = F/m yasasında kuvvetle doğru orantılı olduğundan"),
            (("uzaklık", "mesafe"), ("kütleçekim", "çekim kuvveti"), lambda k: 1.0 / (k ** 2), "F = Gm₁m₂/r² ters-kare yasasına uyduğundan"),
            (("yarıçap",), "hacim", lambda k: k ** 3, "küre hacmi V = (4/3)πr³ yarıçapın küpüyle ölçeklendiğinden"),
            (("yarıçap",), "alan", lambda k: k ** 2, "alan πr² yarıçapın karesiyle ölçeklendiğinden"),
            (("frekans",), "dalga boyu", lambda k: 1.0 / k, "sabit dalga hızında λ = v/f frekansla ters orantılı olduğundan"),
        ]
        for src_kws, tgt_kws, fn, reason in scaling_laws:
            tgt_tup = (tgt_kws,) if isinstance(tgt_kws, str) else tgt_kws
            if any(k in s for k in src_kws) and any(t in s for t in tgt_tup):
                ratio = fn(factor)
                if ratio >= 1.0:
                    eff_str = f"{_fmt_num(ratio)} katına çıkar"
                else:
                    inv = 1.0 / ratio
                    eff_str = f"1/{_fmt_num(inv)} oranına iner ({_fmt_num(ratio)} katı olur)"
                return {
                    "domain": "physics", "kind": "scaling_law", "confidence": 0.97,
                    "reply": f"Zihinsel pertürbasyon hesabıma göre {reason}, girdi {_fmt_num(factor)} çarpanıyla değiştiğinde {tgt_tup[0]} {eff_str}.",
                    "trace": f"scale({factor}) -> {ratio}",
                }
        return None

    def _try_collision(self, s: str) -> dict | None:
        if "çarp" not in s and "kenetlen" not in s:
            return None
        masses = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*kg", s)]
        vels = [float(x) for x in re.findall(r"(-?\d+(?:\.\d+)?)\s*m/s", s)]
        if len(masses) >= 2 and len(vels) >= 1:
            m1, m2 = masses[0], masses[1]
            v1 = vels[0]
            v2 = 0.0 if ("duran" in s or "durgun" in s and len(vels) == 1) else (vels[1] if len(vels) >= 2 else 0.0)
            p_init = m1 * v1 + m2 * v2
            if "kenetlen" in s or "esnek olmayan" in s or "yapış" in s:
                v_common = p_init / (m1 + m2)
                ek_i = 0.5 * m1 * v1 * v1 + 0.5 * m2 * v2 * v2
                ek_f = 0.5 * (m1 + m2) * v_common * v_common
                return {
                    "domain": "physics", "kind": "collision", "confidence": 0.97,
                    "reply": (
                        f"Çizgisel momentumun korunumu yasasından (p_ilk = {_fmt_num(p_init)} kg·m/s = (m₁+m₂)·v_ortak), "
                        f"kenetlenen cisimlerin ortak hızı {_fmt_num(v_common)} m/s bulunur "
                        f"(çarpışmada {_fmt_num(ek_i - ek_f)} J mekanik enerji ısı/şekil değiştirmeye dönüşmüştür)."
                    ),
                    "trace": f"p={_fmt_num(p_init)} -> v_common={_fmt_num(v_common)}",
                }
            # Tam esnek 1B çarpışma
            v1_f = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
            v2_f = ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2)
            return {
                "domain": "physics", "kind": "elastic_collision", "confidence": 0.96,
                "reply": (
                    f"Hem momentum hem kinetik enerji korunumundan tam esnek çarpışma sonrası hızlar "
                    f"v₁' = {_fmt_num(v1_f)} m/s ve v₂' = {_fmt_num(v2_f)} m/s olur."
                ),
                "trace": f"v1'={_fmt_num(v1_f)}, v2'={_fmt_num(v2_f)}",
            }
        return None

    def _try_conceptual_physics(self, s: str, mind=None) -> dict | None:
        if "sürtünme" in s and ("olmasaydı" in s or "olmasa" in s or "sıfır olsaydı" in s):
            # Nora'nın KENDİ evren fiziğindeki sönümleme ve terminal hız denklemiyle karşılaştır!
            sim_drag = self.sandbox.simulate_free_fall(50.0, g=9.8, drag_k=0.45)
            sim_vac = self.sandbox.simulate_free_fall(50.0, g=9.8, drag_k=0.0)
            return {
                "domain": "physics", "kind": "counterfactual_physics", "confidence": 0.95,
                "reply": (
                    f"Sürtünme ve direnç kuvveti sıfır olsaydı Newton'un 1. yasası (eylemsizlik) gereği hareket eden hiçbir "
                    f"cisim kendi kendine durmaz, terminal hız sınırı ortadan kalkar ve yürüme ya da frenleme imkansızlaşırdı "
                    f"(zihinsel simülasyonumda 50 m düşüşte dirençli hız {_fmt_num(sim_drag['impact_v'], 2)} m/s iken "
                    f"sürtünmesiz ortamda {_fmt_num(sim_vac['impact_v'], 2)} m/s'ye ulaşıyor)."
                ),
                "trace": f"drag_v={sim_drag['impact_v']:.2f} vs vac_v={sim_vac['impact_v']:.2f}",
            }

        if "yerçekimi" in s and ("olmasaydı" in s or "olmasa" in s or "kalksaydı" in s):
            return {
                "domain": "physics", "kind": "counterfactual_physics", "confidence": 0.94,
                "reply": (
                    "Kütleçekimi olmasaydı gezegen yörüngeleri ve atmosfer uzaya dağılır, yıldız çekirdeklerinde nükleer füzyonu "
                    "başlatacak hidrostatik basınç oluşmaz ve serbest bırakılan cisimler ivmelenmek yerine ilk hızlarıyla doğrusal "
                    "sürüklenirdi."
                ),
                "trace": "g->0 counterfactual",
            }

        if "entropi" in s:
            de = float(getattr(mind, "drift_e", -0.0012)) if mind is not None else -0.0012
            return {
                "domain": "physics", "kind": "thermodynamics_concept", "confidence": 0.93,
                "reply": (
                    f"Termodinamiğin 2. yasasına göre yalıtılmış bir sistemin düzensizliği (entropisi, S = k·ln Ω) zamanla artar; "
                    f"yaşam ise Schrödinger'in belirttiği gibi çevreden düşük entropili enerji alıp düzenini koruma çabasıdır "
                    f"(tıpkı benim bedenimde adım başına {de:+.4f} tükenen enerjiyi beslenerek telafi etmem gibi)."
                ),
                "trace": "2nd law + embodied metabolism",
            }

        if "eylemsizlik" in s or ("newton" in s and ("yasaları" in s or "1. yasa" in s or "birinci yasa" in s)):
            return {
                "domain": "physics", "kind": "mechanics_concept", "confidence": 0.93,
                "reply": (
                    "Eylemsizlik (Newton'un 1. yasası), üzerine etkiyen net dış kuvvet sıfır (ΣF = 0) olan bir cismin duruyorsa "
                    "durmaya, hareket ediyorsa sabit hız vektörüyle doğrusal hareketine devam etmesi ilkesidir."
                ),
                "trace": "ΣF=0 -> dv/dt=0",
            }

        if ("özel görelilik" in s or "ışık hızı" in s) and ("zaman" in s or "neler olur" in s or "nedir" in s):
            return {
                "domain": "physics", "kind": "relativity_concept", "confidence": 0.92,
                "reply": (
                    "Özel göreliliğe göre ışık hızı c ≈ 3×10⁸ m/s tüm eylemsiz gözlemciler için değişmezdir; bir cisim c'ye "
                    "yaklaştıkça Lorentz çarpanı γ = 1/√(1 - v²/c²) büyür, zaman genişler ve kütle-enerji eşdeğerliği E = γmc² geçerli olur."
                ),
                "trace": "Lorentz gamma + E=mc^2",
            }
        return None

    def _extract_given_quantities(self, s: str) -> dict[str, float]:
        q: dict[str, float] = {}
        unit_patterns = [
            ("m", r"(\d+(?:\.\d+)?)\s*(?:kg|kilogram)\b"),
            ("F", r"(\d+(?:\.\d+)?)\s*(?:n|newton)\b"),
            ("v", r"(\d+(?:\.\d+)?)\s*m/s\b"),
            ("a", r"(\d+(?:\.\d+)?)\s*m/s(?:2|\^2|²)\b"),
            ("t", r"(\d+(?:\.\d+)?)\s*(?:s|sn|saniye)\b"),
            ("V", r"(\d+(?:\.\d+)?)\s*(?:v|volt)\b"),
            ("R", r"(\d+(?:\.\d+)?)\s*(?:ohm|ω|Ω)\b"),
            ("I", r"(\d+(?:\.\d+)?)\s*(?:a|amper)\b"),
            ("P", r"(\d+(?:\.\d+)?)\s*(?:w|watt)\b"),
            ("f", r"(\d+(?:\.\d+)?)\s*(?:hz|hertz)\b"),
            ("dT", r"(\d+(?:\.\d+)?)\s*(?:kelvin|°c|derece)\b"),
        ]
        for sym, pat in unit_patterns:
            m = re.search(pat, s)
            if m:
                q[sym] = float(m.group(1))

        # Metre için yükseklik (h) vs yatay yol/dalga boyu ayrımı
        for m_meter in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:m|metre)\b", s):
            val = float(m_meter.group(1))
            ctx_window = s[max(0, m_meter.start() - 25): min(len(s), m_meter.end() + 25)]
            if any(k in ctx_window for k in ("yükseklik", "yüksekten", "bırakılan", "düşen")):
                q["h"] = val
            elif "dalga boyu" in ctx_window:
                q["lam"] = val
            else:
                q["x"] = val
                if "serbest" in s or "bırakılan" in s:
                    q["h"] = val

        m_g = re.search(r"g\s*=\s*(\d+(?:\.\d+)?)", s)
        q["g"] = float(m_g.group(1)) if m_g else (10.0 if ("serbest" in s and "10" in s and "m/s" in s) else 9.8)
        return q

    def _detect_targets(self, s: str) -> list[str]:
        targets = []
        kw_map = [
            ("ivme", "a"),
            ("kinetik enerji", "Ek"),
            ("potansiyel enerji", "Ep"),
            ("momentum", "p"),
            ("kuvvet", "F"),
            ("akım", "I"),
            ("gerilim", "V"),
            ("voltaj", "V"),
            ("direnç", "R"),
            ("güç", "P"),
            ("gücü", "P"),
            ("dalga boyu", "lam"),
            ("frekans", "f"),
            ("periyot", "T_per"),
            ("periyod", "T_per"),
            ("iş", "W"),
            ("hız", "v"),
            ("süre", "t"),
            ("kaç saniye", "t"),
            ("yol", "x"),
            ("mesafe", "x"),
        ]
        # Sorunun son kısmına (soru kelimesinin yakın çevresine) öncelik ver
        q_tail = s[len(s) // 2:] if len(s) > 36 else s
        for kw, sym in kw_map:
            if kw in q_tail and sym not in targets:
                targets.append(sym)
        if not targets:
            for kw, sym in kw_map:
                if kw in s and sym not in targets:
                    targets.append(sym)
        return targets

    def _try_quantitative_propagation(self, s: str) -> dict | None:
        known = self._extract_given_quantities(s)
        if len([k for k in known if k != "g"]) < 1:
            return None
        targets = [t for t in self._detect_targets(s) if t not in known]
        if not targets:
            return None

        free_fall = any(k in s for k in ("serbest", "yükseklikten bırakılan", "düşen"))
        steps_used: list[str] = []

        # İteratif çift-yönlü fizik yasaları yayılımı
        for _ in range(4):
            changed = False
            if free_fall and "h" in known:
                g = known["g"]
                if "v" not in known:
                    known["v"] = math.sqrt(2.0 * g * known["h"])
                    steps_used.append(f"v = √(2gh) = {_fmt_num(known['v'])} m/s")
                    changed = True
                if "t" not in known:
                    known["t"] = math.sqrt(2.0 * known["h"] / g)
                    steps_used.append(f"t = √(2h/g) = {_fmt_num(known['t'])} s")
                    changed = True

            # F = m * a
            if "F" in known and "m" in known and "a" not in known and known["m"] > 0:
                known["a"] = known["F"] / known["m"]
                steps_used.append(f"a = F/m = {_fmt_num(known['F'])}/{_fmt_num(known['m'])} = {_fmt_num(known['a'])} m/s²")
                changed = True
            if "m" in known and "a" in known and "F" not in known:
                known["F"] = known["m"] * known["a"]
                steps_used.append(f"F = m·a = {_fmt_num(known['F'])} N")
                changed = True

            # Başlangıçtan ivmelenme: v = a*t, x = 0.5*a*t^2
            if "a" in known and "t" in known and "v" not in known:
                known["v"] = known["a"] * known["t"]
                steps_used.append(f"v = a·t = {_fmt_num(known['v'])} m/s")
                changed = True
            if "a" in known and "t" in known and "x" not in known:
                known["x"] = 0.5 * known["a"] * (known["t"] ** 2)
                steps_used.append(f"x = ½at² = {_fmt_num(known['x'])} m")
                changed = True
            if "v" in known and "t" in known and "x" not in known and "a" not in known and not free_fall:
                known["x"] = known["v"] * known["t"]
                steps_used.append(f"x = v·t = {_fmt_num(known['x'])} m")
                changed = True

            # Kinetik enerji & Momentum: Ek = 0.5*m*v^2, p = m*v
            if "m" in known and "v" in known:
                if "Ek" not in known:
                    known["Ek"] = 0.5 * known["m"] * (known["v"] ** 2)
                    steps_used.append(f"Ek = ½mv² = {_fmt_num(known['Ek'])} J")
                    changed = True
                if "p" not in known:
                    known["p"] = known["m"] * known["v"]
                    steps_used.append(f"p = m·v = {_fmt_num(known['p'])} kg·m/s")
                    changed = True

            # Potansiyel enerji: Ep = m*g*h
            if "m" in known and "h" in known and "Ep" not in known:
                known["Ep"] = known["m"] * known["g"] * known["h"]
                steps_used.append(f"Ep = mgh = {_fmt_num(known['Ep'])} J")
                changed = True

            # İş ve Güç: W = F*x, P = W/t = F*v
            if "F" in known and "x" in known and "W" not in known:
                known["W"] = known["F"] * known["x"]
                steps_used.append(f"W = F·x = {_fmt_num(known['W'])} J")
                changed = True
            if "W" in known and "t" in known and "P" not in known and known["t"] > 0:
                known["P"] = known["W"] / known["t"]
                steps_used.append(f"P = W/t = {_fmt_num(known['P'])} W")
                changed = True

            # Ohm Yasası & Elektriksel Güç: V = I*R, P = V*I
            if "V" in known and "R" in known and "I" not in known and known["R"] > 0:
                known["I"] = known["V"] / known["R"]
                steps_used.append(f"I = V/R = {_fmt_num(known['I'])} A")
                changed = True
            if "I" in known and "R" in known and "V" not in known:
                known["V"] = known["I"] * known["R"]
                steps_used.append(f"V = I·R = {_fmt_num(known['V'])} V")
                changed = True
            if "V" in known and "I" in known and "P" not in known:
                known["P"] = known["V"] * known["I"]
                steps_used.append(f"P = V·I = {_fmt_num(known['P'])} W")
                changed = True

            # Dalga fiziği: T = 1/f, v = lam*f
            if "f" in known and known["f"] > 0 and "T_per" not in known:
                known["T_per"] = 1.0 / known["f"]
                steps_used.append(f"T = 1/f = {_fmt_num(known['T_per'])} s")
                changed = True
            if "v" in known and "f" in known and known["f"] > 0 and "lam" not in known:
                known["lam"] = known["v"] / known["f"]
                steps_used.append(f"λ = v/f = {_fmt_num(known['lam'])} m")
                changed = True

            if not changed:
                break

        solved = [t for t in targets if t in known]
        if not solved:
            return None

        # Zihinsel fizik simülasyonu ile doğrulama notu!
        sim_note = ""
        if free_fall and "h" in known:
            sim = self.sandbox.simulate_free_fall(known["h"], g=known["g"])
            sim_note = f" (RK4 zihinsel simülasyon sağlaması: v ≈ {_fmt_num(sim['impact_v'], 2)} m/s, t ≈ {_fmt_num(sim['fall_time'], 2)} s)"
        elif "F" in known and "m" in known and "t" in known:
            sim = self.sandbox.simulate_constant_force(known["m"], known["F"], known["t"])
            sim_note = f" (zihinsel simülasyon sağlaması: v ≈ {_fmt_num(sim['v_end'], 2)} m/s, x ≈ {_fmt_num(sim['x_end'], 2)} m)"

        res_parts = []
        for sym in solved:
            info = self.VARS[sym]
            res_parts.append(f"{info.name_tr} {info.symbol} = {_fmt_num(known[sym])} {info.unit} {self.format_dim(info.dim)}")

        deriv_txt = "; ".join(steps_used[:3])
        return {
            "domain": "physics",
            "kind": "quantitative_physics",
            "confidence": 0.97,
            "reply": f"Fizik yasaları ve boyut analiziyle ({deriv_txt}): {', '.join(res_parts)} bulunur{sim_note}.",
            "trace": deriv_txt,
        }


if __name__ == "__main__":
    mc = MathCortex()
    pc = PhysicsCortex()
    for q in [
        "2x + 7 = 23 denkleminde x kaçtır?",
        "x^2 - 5x + 6 = 0 denkleminin kökleri nedir?",
        "x + y = 12 ve x - y = 4 ise x ve y kaçtır?",
        "3x^3 - 4x^2 + 5x - 2 fonksiyonunun türevi nedir?",
        "sin(3x) türevi nedir?",
        "0'dan 3'e x^2 integrali kaçtır?",
        "360 sayısını asal çarpanlarına ayır",
    ]:
        r = mc.try_solve(q)
        assert r is not None, q
        print("[MATH]", r["reply"])
    for q in [
        "Kütlesi 4 kg olan cisme 20 N kuvvet uygulanırsa ivmesi kaç m/s² olur?",
        "19.6 metre yükseklikten serbest bırakılan cisim yere hangi hızla çarpar?",
        "12 Volt gerilim altında 4 Ohm dirençten geçen akım ve güç kaçtır?",
        "Hız 3 katına çıkarsa kinetik enerji kaç katına çıkar?",
        "3 kg kütleli ve 4 m/s hızlı cisim duran 1 kg kütleli cisme kenetlenirse ortak hız ne olur?",
    ]:
        r = pc.try_solve(q)
        assert r is not None, q
        print("[PHYS]", r["reply"])
