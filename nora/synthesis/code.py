#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_code.py — Nora'nın Çalıştırarak Doğrulayan, Plastik Olarak Öğrenen Kod Sentez Korteksi.

Tasarım ilkesi
--------------
Nora bir dil modeli gibi "kulağa doğru gelen" kod UYDURMAZ. Bir programı ancak ve ancak
  (1) isteği köklerden çözümleyip bir niyet vektörüne indirgediyse,
  (2) bu niyeti, plastik sinaptik ağırlıklarla seçilmiş bir OPERATÖR + gömülü metin üzerinden
      bir AST'ye dönüştürdüyse,
  (3) ve ürettiği programı gerçekten ÇALIŞTIRIP çıktısını istenen metinle karşılaştırdıysa
söyler. Doğrulama geçmezse cevap "bilmiyorum / şu an üretemiyorum" olur. Bu, halüsinasyonu
mimari düzeyde imkânsız kılar: doğrulanmamış hiçbir satır dışarı çıkmaz.

Öğrenme (if/else kural tablosu değil):
  * Her Türkçe ipucu kökü ("yazdır", "ekrana", "bastır"...) ile her operatör (DISPLAY, CONCAT,
    ASSIGN, LOOP) arasındaki bağ bir sinaptik ağırlıktır. Operatör seçimi bu ağırlıkların
    yumuşak maksimumudur (softmax); başarılı çalıştırma ağırlığı güçlendirir, başarısızlık zayıflatır.
  * Çalıştığı kanıtlanmış bir niyet, beceri belleğine (System-1) yazılır: bir dahaki sefere
    hiç arama yapmadan, anında aynı program üretilir.
"""
from __future__ import annotations

from nora._paths import REPO_ROOT


import html
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np

from nora_architecture import capability
from nora_broca import NP, VerbSpec, TurkishSyntaxEngine
from nora_lang import tr_capitalize, tr_lower
from nora_semantic_code import CompositionalCodeLearner, SemanticGoal, ProgramPlan


# ============================================================================
# 1. PLASTİK OPERATÖR AĞI
# ============================================================================

@dataclass
class Operator:
    name: str
    arity: int
    cues: tuple[str, ...]          # bu operatörü çağıran Türkçe kök ipuçları
    py_name: str


class PlasticOperatorCortex:
    """Türkçe kök ipucu -> Python operatörü sinaptik çağrışım matrisi.

    Başlangıç ağırlıkları dilsel önseldir (bir insanın 'yazdır'ı bilmesi gibi); sonrasında
    tamamen çalıştırma geri bildirimiyle değişir. Bu bir kural tablosu değil, öğrenilen bir
    ilişkilendirme matrisidir: yeni bir ipucu eklenirse ağırlığı 0'dan başlar ve denemeyle öğrenir."""

    OPERATORS: tuple[Operator, ...] = (
        Operator("DISPLAY", 1, ("yazdır", "yazdir", "yaz", "ekrana", "bastır", "bas", "print", "çıktı", "göster",
                                "satır", "python", "kod", "program", "betik"), "print"),
        Operator("CONCAT", 2, ("birleştir", "ekle", "yan yana", "concat", "topla"), "concat"),
        Operator("ASSIGN", 2, ("ata", "değişken", "tut", "sakla"), "assign"),
        Operator("LOOP", 2, ("tekrarla", "döngü", "defa", "kez", "loop"), "loop"),
    )

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.W: dict[str, dict[str, float]] = {op.name: {} for op in self.OPERATORS}
        self.op_by_name = {op.name: op for op in self.OPERATORS}
        self.learning_events = 0
        # Dilsel önsel: ipucu -> kendi operatörü (küçük gürültüyle, böylece tek tohum kader değil)
        for op in self.OPERATORS:
            for cue in op.cues:
                self.W[op.name][cue] = 0.85 + 0.05 * float(self.rng.standard_normal())
        for other in self.OPERATORS:
            if other.name == "DISPLAY":
                continue
            for cue in other.cues:
                for opn in self.W:
                    self.W[opn].setdefault(cue, 0.0)

    def presence_features(self, text: str) -> dict[str, float]:
        s = tr_lower(text)
        words = re.findall(r"[\wçğıöşü]+", s)
        feats: dict[str, float] = {}
        for cue in self.W["DISPLAY"].keys():
            if cue in s:
                # Türkçe kök + ek: 'yazdıran', 'yazdıracak', 'yazdırır' hepsi 'yazdır' kökünü içerir
                feats[cue] = 1.0
            else:
                for w in words:
                    if w.startswith(cue) and len(cue) >= 3:
                        feats[cue] = 0.85
                        break
        return feats

    def scores(self, text: str) -> dict[str, float]:
        feats = self.presence_features(text)
        out = {}
        for opn, wmap in self.W.items():
            out[opn] = float(sum(wmap.get(c, 0.0) * v for c, v in feats.items()))
        return out

    def select(self, text: str, threshold: float = 0.30) -> tuple[str | None, dict]:
        sc = self.scores(text)
        if not sc:
            return None, {}
        names = list(sc.keys())
        vals = np.array([sc[n] for n in names], dtype=np.float64)
        if float(np.max(vals)) < threshold:
            return None, {n: round(float(v), 3) for n, v in sc.items()}
        z = np.exp((vals - np.max(vals)) * 3.0)
        probs = z / max(1e-9, float(z.sum()))
        best = names[int(np.argmax(probs))]
        return best, {n: round(float(p), 3) for n, p in zip(names, probs)}

    def reinforce(self, text: str, op_name: str, good: bool, lr: float = 0.12):
        """Çalıştırma sonucuyla sinaptik ağırlığı güncelle (üç-faktörlü öğrenme: ipucu x operatör x ödül)."""
        feats = self.presence_features(text)
        if not feats or op_name not in self.W:
            return
        sign = 1.0 if good else -1.0
        for cue, presence in feats.items():
            cur = self.W[op_name].get(cue, 0.0)
            self.W[op_name][cue] = float(np.clip(cur + sign * lr * presence, -1.0, 2.0))
        self.learning_events += 1


# ============================================================================
# 2. İSTEK ÇÖZÜMLEME (gömülü metin + tek satır kısıtı)
# ============================================================================

@dataclass
class CodeIntent:
    verb: str                     # seçilen operatör
    payload: str = ""
    single_line: bool = True
    language: str = "python"
    raw: str = ""
    scores: dict = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)


class RequestGrounder:
    """Türkçe isteği (operatör, gömülü metin, kısıtlar) üçlüsüne indirger."""

    QUOTE_PATTERNS = (
        r'"([^"]+)"',
        r"“([^”]+)”",
        r"«([^»]+)»",
        r"‘([^’]+)’",
        r"`([^`]+)`",
        # Düz tek tırnak: Türkçe iyelik kesmesiyle ('Python'da) karışmasın diye
        # yalnızca sözcük sınırındaki tırnaklar kabul edilir.
        r"(?<![\wçğıöşü])'([^']{2,})'(?![\wçğıöşü])",
    )

    CUE_ROOTS = ("yazdır", "yazdir", "ekrana", "bastır", "print", "çıktı ver", "göster")
    # Hitap/kanal sözcükleri: gömülü metin DEĞİLDİR, yalnızca nereye yazılacağını söyler
    ADDRESS_ROOTS = ("ekrana", "ekran", "konsola", "konsol", "terminale", "terminal", "çıktı olarak", "alt alta")

    def __init__(self, cortex: PlasticOperatorCortex):
        self.cortex = cortex

    def ground(self, text: str) -> CodeIntent:
        raw = text.strip()
        s = tr_lower(raw)

        payload = ""
        for pat in self.QUOTE_PATTERNS:
            m = re.search(pat, raw)
            if m and m.group(1).strip():
                payload = m.group(1)
                break

        if not payload:
            # Tırnaksız istek: 'Merhaba! Ey Ahali! yazdıran python satırı' / 'ekrana Merhaba! yaz'
            def _cut_payload(span: str) -> str:
                cut = len(span)
                low_span = tr_lower(span)
                for cue in self.CUE_ROOTS + ("yaz", "bas", "yazdır"):
                    j = low_span.find(cue)
                    if j > 0:
                        cut = min(cut, j)
                cand = span[:cut].strip(" ,.:;-\n\t")
                cand = re.sub(r"^(?:bana|lütfen|acaba|hemen|şimdi)\s+", "", cand, flags=re.IGNORECASE)
                if any(k in tr_lower(cand) for k in ("python", "satır", "kod", "program", "betik")):
                    return ""
                return cand

            # (a) Kanal adı ('ekrana', 'konsola') verilmişse gömülü metin ONDAN SONRA gelir
            addr_hits = [(s.rfind(a), a) for a in self.ADDRESS_ROOTS if s.rfind(a) >= 0]
            if addr_hits:
                idx, addr = max(addr_hits)
                cand = _cut_payload(raw[idx + len(addr):])
                if cand:
                    payload = cand
            # (b) Klasik sıra: 'METİN yazdıran python satırı'
            if not payload:
                payload = _cut_payload(raw)

        verb, probs = self.cortex.select(raw)
        single = bool(re.search(r"\b(tek|bir)\s+satır", s)) or ("satır" in s)
        intent = CodeIntent(verb=verb or "", payload=payload, single_line=single, raw=raw, scores=probs)
        intent.trace.append(f"ipuçları={list(self.cortex.presence_features(raw).keys())}")
        intent.trace.append(f"operatör olasılıkları={probs}")
        intent.trace.append(f"gömülü metin={payload!r}")
        return intent


# ============================================================================
# 3. AST + YÜZEY GERÇEKLEŞTİRME + ÇALIŞTIRARAK DOĞRULAMA
# ============================================================================

class PythonEmitter:
    """Soyut sözdizimi ağacını Python metnine çevirir (karar vermez, yalnızca biçimlendirir)."""

    @staticmethod
    def quote(text: str, style: str = '"') -> str:
        if style == '"':
            return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
        return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"

    @classmethod
    def display(cls, text: str, style: str = '"', variant: int = 0) -> str:
        lit = cls.quote(text, style)
        if variant == 0:
            return f"print({lit})"
        if variant == 1:
            return f"import sys; sys.stdout.write({lit} + chr(10))"
        if variant == 2:
            # f-string yolu (kaçış davranışını farklı bir kod yolundan sınar)
            safe = text.replace("{", "{{").replace("}", "}}")
            return "print(f" + cls.quote(safe, style) + ")"
        return f"print({lit})"


class ExecutionVerifier:
    """Programı ayrı bir süreçte çalıştırır; çıktıyı beklenen metinle karşılaştırır."""

    def __init__(self, timeout: float = 6.0):
        self.timeout = float(timeout)
        self.runs = 0
        self.successes = 0
        self.last_stdout = ""

    def run(self, program: str) -> dict:
        self.runs += 1
        t0 = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory() as td:
                p = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True,
                                   timeout=self.timeout, cwd=td)
            out = p.stdout
            self.last_stdout = out
            return {"ok": p.returncode == 0, "stdout": out, "stderr": p.stderr[-300:],
                    "ms": round((time.perf_counter() - t0) * 1000.0, 2), "code": p.returncode}
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "stderr": "zaman_aşımı", "ms": self.timeout * 1000, "code": -1}
        except Exception as e:
            return {"ok": False, "stdout": "", "stderr": type(e).__name__, "ms": 0.0, "code": -2}

    def verify(self, program: str, expected: str) -> dict:
        res = self.run(program)
        out = res["stdout"]
        matched = res["ok"] and out.strip() == expected.strip() and (expected.strip() in out)
        if matched:
            self.successes += 1
        res["expected"] = expected
        res["matched"] = bool(matched)
        return res


# ============================================================================
# 3b. HTML5 VE WEB BİLEŞENİ DOĞRULAYICI VE SENTEZLEYİCİ
# ============================================================================

class HtmlStructuralVerifier(HTMLParser):
    """HTML5 sözdizimi, dengeli etiketler ve zorunlu elemanları denetleyen doğrulayıcı."""
    VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.tags_seen: set[str] = set()
        self.total_tags: int = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        t = tag.lower()
        self.tags_seen.add(t)
        self.total_tags += 1
        if t not in self.VOID_TAGS:
            self.stack.append(t)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]):
        t = tag.lower()
        self.tags_seen.add(t)
        self.total_tags += 1

    def handle_endtag(self, tag: str):
        t = tag.lower()
        if t in self.VOID_TAGS:
            self.errors.append(f"Void etiket '{t}' kapanış etiketi alamaz")
            return
        if not self.stack:
            self.errors.append(f"Eşleşmeyen kapanış etiketi '{t}'")
            return
        expected = self.stack.pop()
        if expected != t:
            self.errors.append(f"Hatalı etiket kapanışı: beklenen '{expected}', bulunan '{t}'")

    def verify(self, markup: str, required_tags: list[str] | None = None) -> dict:
        self.stack.clear()
        self.errors.clear()
        self.tags_seen.clear()
        self.total_tags = 0
        try:
            self.feed(markup)
            self.close()
        except Exception as e:
            return {"ok": False, "reason": f"Sözdizimi hatası: {e}", "errors": [str(e)], "tags_seen": []}
        if self.stack:
            self.errors.append(f"Kapanmamış etiketler kaldı: {', '.join(self.stack)}")
        missing = [t for t in (required_tags or []) if t.lower() not in self.tags_seen]
        if missing:
            self.errors.append(f"Eksik zorunlu etiketler: {', '.join(missing)}")
        return {
            "ok": len(self.errors) == 0,
            "errors": list(self.errors),
            "tags_seen": sorted(list(self.tags_seen)),
            "total_tags": self.total_tags,
        }


@dataclass
class HtmlDesignIntent:
    raw: str
    category: str = "showcase"
    target_text: str = ""
    author_name: str = ""
    role_desc: str = ""
    topic: str = ""
    custom_snippets: list[str] = field(default_factory=list)
    theme_key: str = "dark"
    palette: dict = field(default_factory=dict)
    title_tr: str = "Dinamik Web Uygulaması"
    required_tags: list[str] = field(default_factory=lambda: ["html", "head", "body"])
    has_js: bool = True
    archetype: str = "showcase"
    variant_idx: int = 0


PALETTES = {
    "monochrome": {
        "name": "Siyah & Beyaz (Minimalist Noir)",
        "bg": "#09090b", "card_bg": "#121214", "card_elevated": "#1c1c20",
        "primary": "#ffffff", "hover": "#e4e4e7", "accent": "#a1a1aa",
        "text": "#ffffff", "muted": "#a1a1aa", "border": "rgba(255, 255, 255, 0.16)",
        "accent_badge": "rgba(255, 255, 255, 0.10)", "badge_bg": "rgba(255, 255, 255, 0.10)",
        "badge_text": "#ffffff", "shadow": "0 12px 36px rgba(0, 0, 0, 0.8)",
        "gradient": "linear-gradient(135deg, #ffffff, #888888)",
        "btn_text": "#09090b"
    },
    "dark": {
        "name": "Karanlık Gece (Cyber Dark)",
        "bg": "#0b0f19", "card_bg": "#111827", "card_elevated": "#1f2937",
        "primary": "#38bdf8", "hover": "#0ea5e9", "accent": "#818cf8",
        "text": "#f8fafc", "muted": "#94a3b8", "border": "#1e293b",
        "accent_badge": "rgba(56, 189, 248, 0.12)", "badge_bg": "rgba(56, 189, 248, 0.12)",
        "badge_text": "#38bdf8", "shadow": "0 10px 30px rgba(0, 0, 0, 0.5)",
        "gradient": "linear-gradient(135deg, #38bdf8, #818cf8)",
        "btn_text": "#0b0f19"
    },
    "indigo": {
        "name": "Derin İndigo & Mor",
        "bg": "#0f172a", "card_bg": "#1e293b", "card_elevated": "#334155",
        "primary": "#6366f1", "hover": "#4f46e5", "accent": "#a855f7",
        "text": "#f8fafc", "muted": "#94a3b8", "border": "#334155",
        "accent_badge": "rgba(99, 102, 241, 0.15)", "badge_bg": "rgba(99, 102, 241, 0.15)",
        "badge_text": "#a5b4fc", "shadow": "0 10px 30px rgba(99, 102, 241, 0.2)",
        "gradient": "linear-gradient(135deg, #6366f1, #a855f7)",
        "btn_text": "#ffffff"
    },
    "emerald": {
        "name": "Zümrüt Yeşili (Emerald)",
        "bg": "#064e3b", "card_bg": "#065f46", "card_elevated": "#047857",
        "primary": "#10b981", "hover": "#059669", "accent": "#34d399",
        "text": "#f0fdf4", "muted": "#a7f3d0", "border": "#047857",
        "accent_badge": "rgba(16, 185, 129, 0.15)", "badge_bg": "rgba(16, 185, 129, 0.15)",
        "badge_text": "#6ee7b7", "shadow": "0 10px 30px rgba(0, 0, 0, 0.3)",
        "gradient": "linear-gradient(135deg, #10b981, #059669)",
        "btn_text": "#064e3b"
    },
    "rose": {
        "name": "Modern Gül & Günbatımı",
        "bg": "#1a0b16", "card_bg": "#2d1224", "card_elevated": "#421a36",
        "primary": "#f43f5e", "hover": "#e11d48", "accent": "#fb7185",
        "text": "#fff1f2", "muted": "#fecdd3", "border": "#4c1d3d",
        "accent_badge": "rgba(244, 63, 94, 0.15)", "badge_bg": "rgba(244, 63, 94, 0.15)",
        "badge_text": "#fda4af", "shadow": "0 10px 30px rgba(244, 63, 94, 0.2)",
        "gradient": "linear-gradient(135deg, #f43f5e, #fb7185)",
        "btn_text": "#ffffff"
    },
    "amber": {
        "name": "Sıcak Kehribar (Amber)",
        "bg": "#1c1917", "card_bg": "#292524", "card_elevated": "#44403c",
        "primary": "#f59e0b", "hover": "#d97706", "accent": "#fbbf24",
        "text": "#fffbeb", "muted": "#fde68a", "border": "#44403c",
        "accent_badge": "rgba(245, 158, 11, 0.15)", "badge_bg": "rgba(245, 158, 11, 0.15)",
        "badge_text": "#fcd34d", "shadow": "0 10px 30px rgba(0, 0, 0, 0.4)",
        "gradient": "linear-gradient(135deg, #f59e0b, #fbbf24)",
        "btn_text": "#1c1917"
    },
    "cyan": {
        "name": "Okyanus Turkuazı (Cyan)",
        "bg": "#083344", "card_bg": "#0e4a60", "card_elevated": "#155e75",
        "primary": "#06b6d4", "hover": "#0891b2", "accent": "#67e8f9",
        "text": "#ecfeff", "muted": "#a5f3fc", "border": "#164e63",
        "accent_badge": "rgba(6, 182, 212, 0.15)", "badge_bg": "rgba(6, 182, 212, 0.15)",
        "badge_text": "#67e8f9", "shadow": "0 10px 30px rgba(6, 182, 212, 0.2)",
        "gradient": "linear-gradient(135deg, #06b6d4, #3b82f6)",
        "btn_text": "#083344"
    },
    "violet": {
        "name": "Asil Mor (Violet)",
        "bg": "#1e1035", "card_bg": "#2e1a4d", "card_elevated": "#3e2468",
        "primary": "#8b5cf6", "hover": "#7c3aed", "accent": "#c084fc",
        "text": "#faf5ff", "muted": "#e9d5ff", "border": "#4c2882",
        "accent_badge": "rgba(139, 92, 246, 0.15)", "badge_bg": "rgba(139, 92, 246, 0.15)",
        "badge_text": "#d8b4fe", "shadow": "0 10px 30px rgba(139, 92, 246, 0.25)",
        "gradient": "linear-gradient(135deg, #8b5cf6, #ec4899)",
        "btn_text": "#ffffff"
    },
    "blue": {
        "name": "Safir Mavisi (Sapphire)",
        "bg": "#0f172a", "card_bg": "#1e293b", "card_elevated": "#334155",
        "primary": "#3b82f6", "hover": "#2563eb", "accent": "#60a5fa",
        "text": "#f8fafc", "muted": "#94a3b8", "border": "#334155",
        "accent_badge": "rgba(59, 130, 246, 0.15)", "badge_bg": "rgba(59, 130, 246, 0.15)",
        "badge_text": "#93c5fd", "shadow": "0 10px 30px rgba(59, 130, 246, 0.2)",
        "gradient": "linear-gradient(135deg, #3b82f6, #06b6d4)",
        "btn_text": "#ffffff"
    },
    "clean_light": {
        "name": "Modern Minimalist Açık",
        "bg": "#f8fafc", "card_bg": "#ffffff", "card_elevated": "#f1f5f9",
        "primary": "#2563eb", "hover": "#1d4ed8", "accent": "#38bdf8",
        "text": "#0f172a", "muted": "#64748b", "border": "#e2e8f0",
        "accent_badge": "#dbeafe", "badge_bg": "#dbeafe",
        "badge_text": "#1e40af", "shadow": "0 10px 25px -5px rgba(37, 99, 235, 0.08)",
        "gradient": "linear-gradient(135deg, #2563eb, #38bdf8)",
        "btn_text": "#ffffff"
    }
}

PALETTE_KEYS = ("monochrome", "dark", "indigo", "emerald", "rose", "amber", "cyan", "violet", "blue", "clean_light")


class HtmlCodeSynthesizer:
    """Nora'nın Yaratıcı, Komut Uyumlu ve Dinamik Web Sentez Motoru.
    Kullanıcının verdiği her girdiyi (isim, rol/unvan, sektör, renkler ve özel metinler)
    doğal dil yönergelerinden ayrıştırarak semantik bir DOM ağacına ve uyumlu CSS'e dönüştürür.
    Statik şablonlar yerine kullanıcı talimatlarını harfiyen yerine getiren dinamik yapılar üretir."""

    WEB_CUES = (
        "html", "html5", "web", "website", "web sitesi", "site", "sayfa", "arayüz", "frontend",
        "ui", "şablon", "template", "landing", "portal", "app", "uygulama", "portfolyo", "portfolio"
    )
    APP_CUES = (
        "register", "kayıt", "kaydol", "signup", "sign up", "üye ol", "hesap aç", "hesap oluştur", "auth",
        "login", "giriş", "oturum", "form", "iletişim", "contact",
        "hesap makinesi", "calculator", "dört işlem",
        "tic-tac-toe", "tictactoe", "xox", "oyun", "game",
        "todo", "yapılacaklar", "kanban", "görev", "task",
        "dashboard", "panel", "analitik", "veri tablosu", "istatistik", "admin", "metrik",
        "sayaç", "counter", "kronometre", "zamanlayıcı", "timer",
        "portfolyo", "portfolio", "özgeçmiş", "cv", "kişisel site"
    )
    ACTION_VERBS = (
        "yaz", "oluştur", "hazırla", "ver", "yap", "tasarla", "kodla", "kod", "kodunu", "örnek",
        "sağlayın", "sağla", "geliştir", "kur", "programla", "kullan", "olsun"
    )
    EXCLUSION_CUES = ("nedir", "ne demek", "ne işe yarar", "ne anlama gelir", "nasıl çalışır", "tarihçesi", "kim buldu")

    def __init__(self):
        self.verifier = HtmlStructuralVerifier()

    def is_requested(self, raw: str) -> bool:
        low = tr_lower(raw)
        if any(exc in low for exc in self.EXCLUSION_CUES) and not any(v in low for v in ("yaz", "kod ver", "kod oluştur", "kodu yaz", "tasarla")):
            return False

        has_action = any(v in low for v in self.ACTION_VERBS)
        has_web = any(c in low for c in self.WEB_CUES)
        has_app = any(c in low for c in self.APP_CUES)

        if has_web and (has_action or has_app):
            return True
        if has_app and has_action:
            return True
        if "html" in low or "web sitesi" in low or "website" in low or "portfolyo" in low or "portfolio" in low:
            return True
        return False

    def parse_design_intent(self, raw: str, agi: Any = None) -> HtmlDesignIntent:
        low = tr_lower(raw)

        # 1. Doğal Dil Yönerge Ayrıştırması (Instruction Chunking)
        markers = ["olsun", "yazsın", "yazan", "yazmalı", "yaz", "kullan", "yap", "tasarla", "kodla", "hazırla", "istiyorum"]
        pattern = r"\b(" + "|".join(markers) + r")\b|[,\.;]"

        tokens = []
        last_end = 0
        for match in re.finditer(pattern, raw, re.IGNORECASE):
            chunk = raw[last_end:match.start()].strip()
            marker = match.group(0).lower()
            if chunk:
                tokens.append((chunk, marker))
            last_end = match.end()
        trailing = raw[last_end:].strip()
        if trailing:
            tokens.append((trailing, "son"))

        genre = "showcase"
        author_name = ""
        role_desc = ""
        topic = ""
        custom_snippets = []
        color_req = ""

        # Tırnaklı metinleri topla
        for qm in re.finditer(r'["“«‘`]([^"”»’`]+)["”»’`]', raw):
            val = qm.group(1).strip()
            if val and val not in custom_snippets:
                custom_snippets.append(val)

        for chunk, marker in tokens:
            c_low = chunk.lower()
            clean = re.sub(r"^(?:bana|lütfen|acaba|bir|sayfada|ekrana|ve|renk\s+olarak)\s+", "", chunk, flags=re.IGNORECASE).strip()

            # Renk talimatı
            if any(c in c_low for c in ("siyah", "beyaz", "kırmızı", "yeşil", "mavi", "mor", "sarı", "koyu", "açık", "dark", "light", "renk")):
                color_req = chunk

            # Tür / Kategori
            if any(g in c_low for g in ("portfolyo", "portfolio", "özgeçmiş", "cv", "kişisel site", "kişisel web", "developer")):
                genre = "portfolio"
                topic = "Portfolyo"
            elif any(g in c_low for g in ("register", "kayıt", "signup", "giriş", "login", "auth")):
                genre = "auth"
                topic = "Kayıt & Giriş"
            elif any(g in c_low for g in ("hesap makinesi", "calculator", "dört işlem")):
                genre = "calculator"
                topic = "Hesap Makinesi"
            elif any(g in c_low for g in ("tic-tac-toe", "tictactoe", "xox", "oyun", "game")):
                genre = "tictactoe"
                topic = "Oyun"
            elif any(g in c_low for g in ("todo", "yapılacaklar", "kanban", "görev", "task")):
                genre = "todo"
                topic = "Görevler"
            elif any(g in c_low for g in ("dashboard", "panel", "analitik", "veri tablosu", "admin")):
                genre = "dashboard"
                topic = "Dashboard"
            elif any(g in c_low for g in ("sayaç", "counter", "kronometre")):
                genre = "counter"
                topic = "Sayaç"
            elif marker in ("olsun",) and not topic:
                if not any(w in c_low for w in ("renk", "siyah", "beyaz", "koyu", "açık")):
                    topic = clean

            # İsim veya Rol / Unvan belirleme
            if marker in ("yazsın", "yazan", "yaz", "yazmalı"):
                if any(r in c_low for r in ("developer", "geliştirici", "mühendis", "tasarımcı", "designer", "yaşında", "öğrenci", "freelance", "fullstack", "frontend", "backend", "yazılımcı", "coder")):
                    role_desc = clean
                elif len(clean.split()) <= 4 and not author_name and not any(w in c_low for w in ("renk", "web", "sayfa", "kod")):
                    author_name = clean
                else:
                    custom_snippets.append(clean)

        # Yedek isim kontrolü
        if not author_name:
            for w in raw.split():
                clean_w = re.sub(r"[^\w]", "", w)
                if clean_w and clean_w[0].isupper() and clean_w.lower() not in ("bana", "bir", "web", "sayfası", "sayfa", "yap", "portfolyo", "portfolio", "renk", "olarak", "siyah", "beyaz", "kullan"):
                    author_name = clean_w
                    break

        # 2. Renk Teması Seçimi (Kullanıcı talimatlarına harfiyen uyar)
        low_color = (color_req + " " + raw).lower()
        if any(k in low_color for k in ("siyah ve beyaz", "siyah beyaz", "siyah-beyaz", "black and white", "monochrome", "monokrom", "tek renk")):
            theme_key = "monochrome"
        elif any(k in low_color for k in ("koyu", "dark", "siyah", "karanlık", "gece", "cyber")):
            theme_key = "dark"
        elif any(k in low_color for k in ("açık", "light", "beyaz", "minimal")):
            theme_key = "clean_light"
        elif any(k in low_color for k in ("yeşil", "green", "zümrüt")):
            theme_key = "emerald"
        elif any(k in low_color for k in ("mor", "purple", "eflatun")):
            theme_key = "violet"
        elif any(k in low_color for k in ("mavi", "blue", "deniz")):
            theme_key = "blue"
        elif any(k in low_color for k in ("turkuaz", "cyan", "teal")):
            theme_key = "cyan"
        elif any(k in low_color for k in ("kırmızı", "pembe", "rose")):
            theme_key = "rose"
        elif any(k in low_color for k in ("sarı", "turuncu", "kehribar", "amber")):
            theme_key = "amber"
        else:
            theme_key = "monochrome" if genre == "portfolio" else "dark"

        pal = PALETTES.get(theme_key, PALETTES["monochrome" if genre == "portfolio" else "dark"])

        # Zorunlu etiketler ve başlık
        if genre == "portfolio":
            title = f"{author_name + ' - ' if author_name else ''}Geliştirici Portfolyosu"
            req = ["html", "head", "body", "header", "main", "footer", "button", "script"]
        elif genre == "auth":
            title = f"{author_name + ' ' if author_name else ''}Kullanıcı Kayıt & Giriş Portalı"
            req = ["html", "body", "form", "input", "button", "script"]
        elif genre == "calculator":
            title = "Modern Hesap Makinesi"
            req = ["html", "body", "button", "script"]
        elif genre == "tictactoe":
            title = "Tic-Tac-Toe Web Oyunu"
            req = ["html", "body", "button", "script"]
        elif genre == "todo":
            title = "Dinamik Görev & Kanban Yöneticisi"
            req = ["html", "body", "input", "button", "ul", "script"]
        elif genre == "dashboard":
            title = f"{author_name + ' ' if author_name else ''}İnteraktif Analitik Dashboard"
            req = ["html", "body", "table", "thead", "tbody", "tr", "th", "td", "input", "script"]
        elif genre == "counter":
            title = "Etkileşimli Sayaç"
            req = ["html", "body", "button", "script"]
        else:
            title = author_name if author_name else (topic.title() if topic else "Modern Web Deneyimi")
            req = ["html", "head", "body", "header", "main", "footer", "button", "script"]

        return HtmlDesignIntent(
            raw=raw, category=genre, target_text=author_name or topic,
            author_name=author_name, role_desc=role_desc, topic=topic,
            custom_snippets=custom_snippets, theme_key=theme_key,
            palette=pal, title_tr=title, required_tags=req, has_js=True,
            archetype=genre, variant_idx=0
        )

    def generate_markup(self, intent: HtmlDesignIntent) -> str:
        cat = intent.category
        if cat == "portfolio":
            return self._build_portfolio(intent)
        elif cat == "auth":
            return self._build_auth_portal(intent)
        elif cat == "calculator":
            return self._build_calculator(intent)
        elif cat == "tictactoe":
            return self._build_tictactoe(intent)
        elif cat == "todo":
            return self._build_todo(intent)
        elif cat == "dashboard":
            return self._build_dashboard(intent)
        elif cat == "counter":
            return self._build_counter(intent)
        else:
            return self._build_domain_showcase(intent)

    def _build_portfolio(self, intent: HtmlDesignIntent) -> str:
        p = intent.palette
        author = html.escape(intent.author_name if intent.author_name else "Geliştirici")
        role = html.escape(intent.role_desc if intent.role_desc else "Web Developer")
        initial = author[0].upper() if author else "D"
        btn_text_color = p.get("btn_text", "#09090b")

        return f"""<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{author} - Portfolyo | {role}</title>
  <style>
    :root {{
      --bg: {p['bg']};
      --card-bg: {p['card_bg']};
      --card-elevated: {p['card_elevated']};
      --text: {p['text']};
      --muted: {p['muted']};
      --border: {p['border']};
      --primary: {p['primary']};
      --primary-hover: {p['hover']};
      --accent: {p['accent']};
      --shadow: {p['shadow']};
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 18px 40px;
      border-bottom: 1px solid var(--border);
      background: rgba(9, 9, 11, 0.85);
      backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 100;
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 12px;
      text-decoration: none;
      color: var(--text);
      font-size: 20px;
      font-weight: 800;
      letter-spacing: -0.5px;
    }}
    .brand-icon {{
      width: 36px;
      height: 36px;
      border-radius: 10px;
      background: var(--card-elevated);
      border: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 900;
      color: var(--primary);
    }}
    nav {{ display: flex; gap: 24px; }}
    nav a {{
      color: var(--muted);
      text-decoration: none;
      font-size: 14px;
      font-weight: 500;
      transition: color 0.2s;
    }}
    nav a:hover {{ color: var(--primary); }}
    .status-pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 14px;
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid var(--border);
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      color: var(--text);
    }}
    .status-dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
      box-shadow: 0 0 8px #10b981;
    }}
    main {{ max-width: 1040px; margin: 0 auto; padding: 0 24px; }}
    .hero {{
      text-align: center;
      padding: 90px 20px 60px;
    }}
    .avatar {{
      width: 88px;
      height: 88px;
      border-radius: 50%;
      background: var(--card-bg);
      border: 2px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 36px;
      font-weight: 800;
      margin: 0 auto 24px;
      color: var(--primary);
      box-shadow: 0 0 30px rgba(255, 255, 255, 0.1);
    }}
    .role-badge {{
      display: inline-block;
      padding: 6px 16px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid var(--border);
      font-size: 13px;
      font-weight: 600;
      color: var(--accent);
      margin-bottom: 20px;
      letter-spacing: 0.5px;
      text-transform: uppercase;
    }}
    .hero h1 {{
      font-size: 52px;
      font-weight: 800;
      letter-spacing: -1.5px;
      margin-bottom: 16px;
      line-height: 1.15;
    }}
    .hero p {{
      font-size: 20px;
      color: var(--muted);
      max-width: 680px;
      margin: 0 auto 36px;
      font-weight: 400;
    }}
    .hero-actions {{ display: flex; gap: 14px; justify-content: center; }}
    .btn {{
      padding: 13px 28px;
      border-radius: 10px;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      text-decoration: none;
      transition: all 0.2s ease;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .btn-primary {{
      background: var(--primary);
      color: {btn_text_color};
      border: none;
    }}
    .btn-primary:hover {{
      background: var(--primary-hover);
      transform: translateY(-2px);
    }}
    .btn-outline {{
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text);
    }}
    .btn-outline:hover {{
      background: rgba(255, 255, 255, 0.05);
      border-color: var(--text);
    }}
    .section-title {{
      font-size: 28px;
      font-weight: 700;
      margin: 80px 0 24px;
      letter-spacing: -0.5px;
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .skills-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 20px;
    }}
    .skill-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 24px;
      transition: transform 0.2s, border-color 0.2s;
    }}
    .skill-card:hover {{
      transform: translateY(-4px);
      border-color: rgba(255, 255, 255, 0.35);
    }}
    .skill-card h3 {{ font-size: 18px; margin-bottom: 8px; }}
    .skill-card p {{ font-size: 14px; color: var(--muted); }}
    .filter-tabs {{
      display: flex;
      gap: 10px;
      margin-bottom: 24px;
    }}
    .filter-btn {{
      padding: 8px 18px;
      background: transparent;
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .filter-btn.active, .filter-btn:hover {{
      background: var(--text);
      color: var(--bg);
      border-color: var(--text);
    }}
    .projects-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(310px, 1fr));
      gap: 24px;
    }}
    .project-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 16px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      transition: transform 0.2s, border-color 0.2s;
    }}
    .project-card:hover {{
      transform: translateY(-4px);
      border-color: rgba(255, 255, 255, 0.35);
    }}
    .project-body {{
      padding: 24px;
      flex: 1;
      display: flex;
      flex-direction: column;
    }}
    .project-tag {{
      font-size: 11px;
      text-transform: uppercase;
      font-weight: 700;
      color: var(--accent);
      margin-bottom: 8px;
    }}
    .project-body h3 {{ font-size: 20px; margin-bottom: 8px; }}
    .project-body p {{ font-size: 14px; color: var(--muted); margin-bottom: 20px; flex: 1; }}
    .contact-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 40px;
      margin: 40px 0 80px;
    }}
    .form-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
    .form-group label {{
      display: block;
      font-size: 13px;
      font-weight: 600;
      color: var(--text);
      margin-bottom: 8px;
    }}
    .form-group input, .form-group textarea {{
      width: 100%;
      padding: 12px 16px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--border);
      border-radius: 10px;
      color: var(--text);
      font-size: 14px;
      outline: none;
      font-family: inherit;
    }}
    .form-group textarea {{ height: 110px; resize: vertical; margin-bottom: 20px; }}
    .form-group input:focus, .form-group textarea:focus {{
      border-color: var(--primary);
    }}
    .alert-box {{
      display: none;
      padding: 14px;
      border-radius: 10px;
      margin-bottom: 16px;
      font-size: 14px;
      text-align: center;
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid var(--primary);
      color: #fff;
    }}
    footer {{
      border-top: 1px solid var(--border);
      padding: 40px 24px;
      text-align: center;
      color: var(--muted);
      font-size: 14px;
    }}
  </style>
</head>
<body>
  <header>
    <a href="#" class="brand">
      <div class="brand-icon">{initial}</div>
      <span>{author}</span>
    </a>
    <nav>
      <a href="#hakkimda">Hakkımda</a>
      <a href="#yetenekler">Yetenekler</a>
      <a href="#projeler">Projeler</a>
      <a href="#iletisim">İletişim</a>
    </nav>
    <div class="status-pill">
      <div class="status-dot"></div>
      <span>Projelere Açık</span>
    </div>
  </header>

  <main>
    <section class="hero" id="hakkimda">
      <div class="avatar">{initial}</div>
      <div class="role-badge">{role}</div>
      <h1>Merhaba, Ben {author}</h1>
      <p>{role} olarak modern web standartları, dinamik JavaScript uygulamaları ve temiz arayüz mimarileri geliştiriyorum. Kullanıcı odaklı, performanslı ve estetik dijital deneyimler inşa ediyorum.</p>
      <div class="hero-actions">
        <a href="#projeler" class="btn btn-primary">Projelerimi İncele</a>
        <a href="#iletisim" class="btn btn-outline">İletişime Geç</a>
      </div>
    </section>

    <section id="yetenekler">
      <h2 class="section-title">⚡ Teknik Yetenekler</h2>
      <div class="skills-grid">
        <div class="skill-card">
          <h3>Modern Web & Frontend</h3>
          <p>HTML5 semantik yapı standartları, modern CSS3 Flexbox/Grid mimarisi ve Vanilla JavaScript (ES6+).</p>
        </div>
        <div class="skill-card">
          <h3>Dinamik Uygulama Mimarisi</h3>
          <p>İstemci taraflı durum yönetimi, DOM manipülasyonu, asenkron veri iletişimi ve interaktif bileşenler.</p>
        </div>
        <div class="skill-card">
          <h3>Kullanıcı Deneyimi & UI/UX</h3>
          <p>Duyarlı (responsive) tasarım, A11y erişilebilirlik ilkeleri ve yüksek kontrastlı modern tipografi.</p>
        </div>
      </div>
    </section>

    <section id="projeler">
      <h2 class="section-title">💼 Öne Çıkan Projeler</h2>
      <div class="filter-tabs">
        <button type="button" class="filter-btn active" onclick="filterProjects('all')">Tümü</button>
        <button type="button" class="filter-btn" onclick="filterProjects('app')">Web Uygulamaları</button>
        <button type="button" class="filter-btn" onclick="filterProjects('ui')">Arayüzler</button>
      </div>
      <div class="projects-grid">
        <div class="project-card" data-cat="app">
          <div class="project-body">
            <span class="project-tag">Web Uygulaması</span>
            <h3>Dinamik Görev Yöneticisi</h3>
            <p>Gerçek zamanlı görev ekleme, tamamlanma durumları ve dinamik DOM yönetimi sunan modern web uygulaması.</p>
            <button type="button" class="btn btn-outline" onclick="alert('Dinamik Görev Yöneticisi demosu açılıyor...')">Demoyu İncele</button>
          </div>
        </div>
        <div class="project-card" data-cat="ui">
          <div class="project-body">
            <span class="project-tag">Arayüz & Dashboard</span>
            <h3>Operasyonel Analitik Paneli</h3>
            <p>Anlık arama filtreleme, KPI metrik kartları ve koyu tema veri tablosu içeren yönetim arayüzü.</p>
            <button type="button" class="btn btn-outline" onclick="alert('Analitik Paneli demosu açılıyor...')">Demoyu İncele</button>
          </div>
        </div>
        <div class="project-card" data-cat="app">
          <div class="project-body">
            <span class="project-tag">Etkileşimli Deneyim</span>
            <h3>Gelişmiş Hesap Makinesi</h3>
            <p>JavaScript durum makinesi ile dört işlem, hafıza yönetimi ve klavye destekli hesaplama motoru.</p>
            <button type="button" class="btn btn-outline" onclick="alert('Hesap Makinesi demosu açılıyor...')">Demoyu İncele</button>
          </div>
        </div>
      </div>
    </section>

    <section id="iletisim">
      <h2 class="section-title">📬 İletişime Geçin</h2>
      <div class="contact-card">
        <div id="contact-alert" class="alert-box"></div>
        <form onsubmit="handleContact(event)">
          <div class="form-row">
            <div class="form-group">
              <label for="c-name">Adınız Soyadınız</label>
              <input type="text" id="c-name" required placeholder="Adınız">
            </div>
            <div class="form-group">
              <label for="c-email">E-posta Adresiniz</label>
              <input type="email" id="c-email" required placeholder="ornek@alanadi.com">
            </div>
          </div>
          <div class="form-group">
            <label for="c-msg">Mesajınız</label>
            <textarea id="c-msg" required placeholder="Projeniz veya iş birliği hakkında mesajınızı yazın..."></textarea>
          </div>
          <button type="submit" class="btn btn-primary">Mesajı {author}'ya Gönder</button>
        </form>
      </div>
    </section>
  </main>

  <footer>
    <p>&copy; 2026 {author}. Tüm hakları saklıdır. &bull; {role}</p>
  </footer>

  <script>
    function filterProjects(cat) {{
      const cards = document.querySelectorAll('.project-card');
      const btns = document.querySelectorAll('.filter-btn');
      btns.forEach(b => b.classList.remove('active'));
      event.target.classList.add('active');

      cards.forEach(card => {{
        if (cat === 'all' || card.dataset.cat === cat) {{
          card.style.display = 'flex';
        }} else {{
          card.style.display = 'none';
        }}
      }});
    }}

    function handleContact(e) {{
      e.preventDefault();
      const name = document.getElementById('c-name').value;
      const alertBox = document.getElementById('contact-alert');
      alertBox.textContent = 'Teşekkürler ' + name + '! Mesajınız ' + '{author}' + ' kişisine başarıyla iletildi.';
      alertBox.style.display = 'block';
      e.target.reset();
    }}
  </script>
</body>
</html>"""

    def _build_auth_portal(self, intent: HtmlDesignIntent) -> str:
        p = intent.palette
        is_register_first = not any(k in intent.raw.lower() for k in ("login", "giriş", "oturum"))
        reg_active = "active" if is_register_first else ""
        log_active = "" if is_register_first else "active"
        reg_style = "display: block;" if is_register_first else "display: none;"
        log_style = "display: none;" if is_register_first else "display: block;"

        return f"""<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Kayıt Ol & Giriş Yap - {html.escape(intent.title_tr)}</title>
  <style>
    :root {{
      --bg: {p['bg']};
      --card-bg: {p['card_bg']};
      --text: {p['text']};
      --muted: {p['muted']};
      --border: {p['border']};
      --primary: {p['primary']};
      --primary-hover: {p['hover']};
      --accent: {p['accent']};
      --shadow: {p['shadow']};
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 24px;
    }}
    .auth-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 36px 32px;
      width: 100%;
      max-width: 440px;
      box-shadow: var(--shadow);
    }}
    .auth-header {{ text-align: center; margin-bottom: 24px; }}
    .auth-header h2 {{ font-size: 24px; font-weight: 700; margin-bottom: 8px; }}
    .auth-header p {{ color: var(--muted); font-size: 14px; }}
    .auth-tabs {{
      display: flex;
      background: rgba(0,0,0,0.25);
      border-radius: 10px;
      padding: 4px;
      margin-bottom: 24px;
    }}
    .tab-btn {{
      flex: 1;
      padding: 10px;
      border: none;
      background: transparent;
      color: var(--muted);
      font-size: 14px;
      font-weight: 600;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .tab-btn.active {{
      background: var(--primary);
      color: #fff;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }}
    .form-group {{ margin-bottom: 16px; position: relative; }}
    .form-group label {{
      display: block;
      font-size: 13px;
      font-weight: 600;
      color: var(--text);
      margin-bottom: 6px;
    }}
    .form-group input {{
      width: 100%;
      padding: 11px 14px;
      background: rgba(0,0,0,0.2);
      border: 1.5px solid var(--border);
      border-radius: 10px;
      color: var(--text);
      font-size: 14px;
      outline: none;
      transition: border-color 0.2s;
    }}
    .form-group input:focus {{ border-color: var(--primary); }}
    .pw-strength {{
      height: 4px;
      background: var(--border);
      border-radius: 2px;
      margin-top: 6px;
      overflow: hidden;
    }}
    .pw-bar {{
      height: 100%;
      width: 0%;
      transition: width 0.3s, background-color 0.3s;
    }}
    .pw-desc {{ font-size: 11px; color: var(--muted); margin-top: 4px; }}
    .btn-submit {{
      width: 100%;
      padding: 13px;
      background: var(--primary);
      color: #fff;
      border: none;
      border-radius: 10px;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      margin-top: 10px;
      transition: background 0.2s, transform 0.1s;
    }}
    .btn-submit:hover {{ background: var(--primary-hover); transform: translateY(-1px); }}
    .auth-alert {{
      display: none;
      padding: 12px;
      border-radius: 8px;
      font-size: 13px;
      margin-bottom: 16px;
      text-align: center;
    }}
    .alert-success {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }}
    .alert-error {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }}
  </style>
</head>
<body>
  <div class="auth-card">
    <div class="auth-header">
      <h2 id="portal-title">Hesap Oluştur</h2>
      <p id="portal-desc">Aramıza katılmak için formu doldurun</p>
    </div>

    <div class="auth-tabs">
      <button type="button" class="tab-btn {reg_active}" onclick="switchAuth('register')">Kayıt Ol (Register)</button>
      <button type="button" class="tab-btn {log_active}" onclick="switchAuth('login')">Giriş Yap (Login)</button>
    </div>

    <div id="auth-alert" class="auth-alert"></div>

    <form id="form-register" style="{reg_style}" onsubmit="handleRegister(event)">
      <div class="form-group">
        <label for="reg-name">Ad Soyad</label>
        <input type="text" id="reg-name" required placeholder="Adınız Soyadınız">
      </div>
      <div class="form-group">
        <label for="reg-email">E-posta Adresi</label>
        <input type="email" id="reg-email" required placeholder="ornek@alanadi.com">
      </div>
      <div class="form-group">
        <label for="reg-password">Şifre</label>
        <input type="password" id="reg-password" required placeholder="En az 8 karakter" oninput="checkStrength(this.value)">
        <div class="pw-strength"><div id="pw-bar" class="pw-bar"></div></div>
        <div id="pw-desc" class="pw-desc">Şifre güvenliği</div>
      </div>
      <div class="form-group">
        <label for="reg-confirm">Şifre Tekrar</label>
        <input type="password" id="reg-confirm" required placeholder="Şifrenizi tekrar girin">
      </div>
      <button type="submit" class="btn-submit">Kayıt Ol</button>
    </form>

    <form id="form-login" style="{log_style}" onsubmit="handleLogin(event)">
      <div class="form-group">
        <label for="log-email">E-posta Adresi</label>
        <input type="email" id="log-email" required placeholder="ornek@alanadi.com">
      </div>
      <div class="form-group">
        <label for="log-password">Şifre</label>
        <input type="password" id="log-password" required placeholder="••••••••">
      </div>
      <button type="submit" class="btn-submit">Giriş Yap</button>
    </form>
  </div>

  <script>
    function switchAuth(mode) {{
      const regForm = document.getElementById('form-register');
      const logForm = document.getElementById('form-login');
      const tabs = document.querySelectorAll('.tab-btn');
      const title = document.getElementById('portal-title');
      const desc = document.getElementById('portal-desc');
      const alertBox = document.getElementById('auth-alert');
      alertBox.style.display = 'none';

      if (mode === 'register') {{
        regForm.style.display = 'block';
        logForm.style.display = 'none';
        tabs[0].classList.add('active');
        tabs[1].classList.remove('active');
        title.textContent = 'Hesap Oluştur';
        desc.textContent = 'Aramıza katılmak için formu doldurun';
      }} else {{
        regForm.style.display = 'none';
        logForm.style.display = 'block';
        tabs[0].classList.remove('active');
        tabs[1].classList.add('active');
        title.textContent = 'Hoş Geldiniz';
        desc.textContent = 'Lütfen hesabınıza giriş yapın';
      }}
    }}

    function checkStrength(val) {{
      const bar = document.getElementById('pw-bar');
      const desc = document.getElementById('pw-desc');
      let score = 0;
      if (val.length >= 6) score += 25;
      if (val.length >= 8) score += 25;
      if (/[A-Z]/.test(val) && /[0-9]/.test(val)) score += 25;
      if (/[^A-Za-z0-9]/.test(val)) score += 25;

      bar.style.width = score + '%';
      if (score <= 25) {{
        bar.style.backgroundColor = '#ef4444';
        desc.textContent = 'Zayıf şifre';
      }} else if (score <= 75) {{
        bar.style.backgroundColor = '#f59e0b';
        desc.textContent = 'Orta seviye şifre';
      }} else {{
        bar.style.backgroundColor = '#10b981';
        desc.textContent = 'Güçlü şifre';
      }}
    }}

    function handleRegister(e) {{
      e.preventDefault();
      const p1 = document.getElementById('reg-password').value;
      const p2 = document.getElementById('reg-confirm').value;
      const alertBox = document.getElementById('auth-alert');

      if (p1 !== p2) {{
        alertBox.className = 'auth-alert alert-error';
        alertBox.textContent = 'Hata: Şifreler birbiriyle eşleşmiyor!';
        alertBox.style.display = 'block';
        return;
      }}
      alertBox.className = 'auth-alert alert-success';
      alertBox.textContent = 'Tebrikler! Kayıt başarılı, hesabınız oluşturuldu.';
      alertBox.style.display = 'block';
    }}

    function handleLogin(e) {{
      e.preventDefault();
      const alertBox = document.getElementById('auth-alert');
      alertBox.className = 'auth-alert alert-success';
      alertBox.textContent = 'Giriş başarılı! Yönlendiriliyorsunuz...';
      alertBox.style.display = 'block';
    }}
  </script>
</body>
</html>"""

    def _build_calculator(self, intent: HtmlDesignIntent) -> str:
        p = intent.palette
        return f"""<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Modern Hesap Makinesi</title>
  <style>
    :root {{
      --bg: {p['bg']};
      --card-bg: {p['card_bg']};
      --text: {p['text']};
      --border: {p['border']};
      --primary: {p['primary']};
    }}
    body {{
      font-family: system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
      margin: 0;
    }}
    .calc-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 24px;
      width: 320px;
      box-shadow: 0 12px 36px rgba(0,0,0,0.4);
    }}
    .calc-screen {{
      background: rgba(0,0,0,0.3);
      border-radius: 12px;
      padding: 16px;
      text-align: right;
      margin-bottom: 20px;
      min-height: 50px;
    }}
    .calc-history {{ font-size: 13px; color: #94a3b8; min-height: 18px; }}
    .calc-value {{ font-size: 32px; font-weight: 700; color: #fff; word-break: break-all; }}
    .calc-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
    }}
    .calc-btn {{
      padding: 16px;
      font-size: 18px;
      font-weight: 600;
      border: none;
      border-radius: 10px;
      background: rgba(255,255,255,0.08);
      color: var(--text);
      cursor: pointer;
      transition: all 0.15s;
    }}
    .calc-btn:hover {{ background: rgba(255,255,255,0.15); }}
    .calc-btn.op {{ background: var(--primary); color: #fff; }}
    .calc-btn.clear {{ background: #ef4444; color: #fff; }}
  </style>
</head>
<body>
  <div class="calc-card">
    <div class="calc-screen">
      <div id="calc-history" class="calc-history"></div>
      <div id="calc-val" class="calc-value">0</div>
    </div>
    <div class="calc-grid">
      <button type="button" class="calc-btn clear" onclick="clearCalc()">C</button>
      <button type="button" class="calc-btn" onclick="appendOp('/')">÷</button>
      <button type="button" class="calc-btn" onclick="appendOp('*')">×</button>
      <button type="button" class="calc-btn op" onclick="deleteLast()">⌫</button>

      <button type="button" class="calc-btn" onclick="appendNum('7')">7</button>
      <button type="button" class="calc-btn" onclick="appendNum('8')">8</button>
      <button type="button" class="calc-btn" onclick="appendNum('9')">9</button>
      <button type="button" class="calc-btn op" onclick="appendOp('-')">-</button>

      <button type="button" class="calc-btn" onclick="appendNum('4')">4</button>
      <button type="button" class="calc-btn" onclick="appendNum('5')">5</button>
      <button type="button" class="calc-btn" onclick="appendNum('6')">6</button>
      <button type="button" class="calc-btn op" onclick="appendOp('+')">+</button>

      <button type="button" class="calc-btn" onclick="appendNum('1')">1</button>
      <button type="button" class="calc-btn" onclick="appendNum('2')">2</button>
      <button type="button" class="calc-btn" onclick="appendNum('3')">3</button>
      <button type="button" class="calc-btn op" style="grid-row: span 2;" onclick="calculate()">=</button>

      <button type="button" class="calc-btn" style="grid-column: span 2;" onclick="appendNum('0')">0</button>
      <button type="button" class="calc-btn" onclick="appendNum('.')">.</button>
    </div>
  </div>

  <script>
    let currentInput = '0';
    let prevInput = '';
    let op = null;

    function updateDisplay() {{
      document.getElementById('calc-val').textContent = currentInput;
      document.getElementById('calc-history').textContent = prevInput + (op ? ' ' + op : '');
    }}

    function appendNum(n) {{
      if (currentInput === '0' && n !== '.') currentInput = n;
      else if (n === '.' && currentInput.includes('.')) return;
      else currentInput += n;
      updateDisplay();
    }}

    function appendOp(operator) {{
      if (op && currentInput !== '') calculate();
      prevInput = currentInput;
      op = operator;
      currentInput = '0';
      updateDisplay();
    }}

    function calculate() {{
      if (!op || prevInput === '') return;
      const a = parseFloat(prevInput);
      const b = parseFloat(currentInput);
      let res = 0;
      if (op === '+') res = a + b;
      else if (op === '-') res = a - b;
      else if (op === '*') res = a * b;
      else if (op === '/') res = b !== 0 ? a / b : 'Hata';
      currentInput = String(res);
      op = null;
      prevInput = '';
      updateDisplay();
    }}

    function clearCalc() {{
      currentInput = '0';
      prevInput = '';
      op = null;
      updateDisplay();
    }}

    function deleteLast() {{
      if (currentInput.length > 1) currentInput = currentInput.slice(0, -1);
      else currentInput = '0';
      updateDisplay();
    }}
  </script>
</body>
</html>"""

    def _build_tictactoe(self, intent: HtmlDesignIntent) -> str:
        p = intent.palette
        return f"""<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Tic-Tac-Toe Web Oyunu</title>
  <style>
    :root {{
      --bg: {p['bg']};
      --card-bg: {p['card_bg']};
      --text: {p['text']};
      --primary: {p['primary']};
    }}
    body {{
      font-family: system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
      margin: 0;
    }}
    .game-card {{
      background: var(--card-bg);
      padding: 32px;
      border-radius: 20px;
      text-align: center;
      box-shadow: 0 12px 30px rgba(0,0,0,0.5);
    }}
    .game-grid {{
      display: grid;
      grid-template-columns: repeat(3, 80px);
      grid-template-rows: repeat(3, 80px);
      gap: 10px;
      margin: 24px auto;
    }}
    .cell {{
      background: rgba(255,255,255,0.08);
      border-radius: 12px;
      font-size: 36px;
      font-weight: 800;
      display: flex;
      justify-content: center;
      align-items: center;
      cursor: pointer;
      color: var(--primary);
      user-select: none;
    }}
    .btn-reset {{
      padding: 10px 20px;
      background: var(--primary);
      color: #fff;
      border: none;
      border-radius: 8px;
      font-weight: 700;
      cursor: pointer;
    }}
  </style>
</head>
<body>
  <div class="game-card">
    <h2>Tic-Tac-Toe</h2>
    <div id="status">Sıra: X</div>
    <div class="game-grid" id="grid">
      <div class="cell" onclick="makeMove(0)"></div>
      <div class="cell" onclick="makeMove(1)"></div>
      <div class="cell" onclick="makeMove(2)"></div>
      <div class="cell" onclick="makeMove(3)"></div>
      <div class="cell" onclick="makeMove(4)"></div>
      <div class="cell" onclick="makeMove(5)"></div>
      <div class="cell" onclick="makeMove(6)"></div>
      <div class="cell" onclick="makeMove(7)"></div>
      <div class="cell" onclick="makeMove(8)"></div>
    </div>
    <button type="button" class="btn-reset" onclick="resetGame()">Yeniden Başlat</button>
  </div>
  <script>
    let board = ['', '', '', '', '', '', '', '', ''];
    let turn = 'X';
    let active = true;

    function makeMove(i) {{
      if (board[i] !== '' || !active) return;
      board[i] = turn;
      document.getElementsByClassName('cell')[i].textContent = turn;
      if (checkWin()) {{
        document.getElementById('status').textContent = 'Kazanan: ' + turn + '! 🎉';
        active = false;
        return;
      }}
      if (!board.includes('')) {{
        document.getElementById('status').textContent = 'Berabere!';
        active = false;
        return;
      }}
      turn = turn === 'X' ? 'O' : 'X';
      document.getElementById('status').textContent = 'Sıra: ' + turn;
    }}

    function checkWin() {{
      const wins = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];
      return wins.some(([a,b,c]) => board[a] && board[a] === board[b] && board[a] === board[c]);
    }}

    function resetGame() {{
      board = ['', '', '', '', '', '', '', '', ''];
      active = true;
      turn = 'X';
      document.getElementById('status').textContent = 'Sıra: X';
      Array.from(document.getElementsByClassName('cell')).forEach(c => c.textContent = '');
    }}
  </script>
</body>
</html>"""

    def _build_todo(self, intent: HtmlDesignIntent) -> str:
        p = intent.palette
        return f"""<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Görev & Kanban Yöneticisi</title>
  <style>
    :root {{
      --bg: {p['bg']};
      --card-bg: {p['card_bg']};
      --text: {p['text']};
      --primary: {p['primary']};
      --border: {p['border']};
      --muted: {p['muted']};
    }}
    body {{
      font-family: system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      justify-content: center;
      padding: 40px 20px;
      margin: 0;
    }}
    .todo-app {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 16px;
      width: 100%;
      max-width: 480px;
      padding: 28px;
    }}
    .add-row {{ display: flex; gap: 8px; margin: 20px 0; }}
    .add-row input {{
      flex: 1;
      padding: 12px;
      background: rgba(0,0,0,0.2);
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--text);
      outline: none;
    }}
    .add-row button {{
      padding: 12px 20px;
      background: var(--primary);
      border: none;
      border-radius: 8px;
      color: #fff;
      font-weight: 700;
      cursor: pointer;
    }}
    ul {{ list-style: none; padding: 0; margin: 0; }}
    li {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px;
      background: rgba(255,255,255,0.04);
      margin-bottom: 8px;
      border-radius: 8px;
    }}
    .completed {{ text-decoration: line-through; opacity: 0.6; }}
    .del-btn {{ background: transparent; border: none; color: #ef4444; cursor: pointer; }}
  </style>
</head>
<body>
  <div class="todo-app">
    <h2>Görev Yöneticisi</h2>
    <div class="add-row">
      <input type="text" id="task-inp" placeholder="Yeni görev ekle...">
      <button type="button" onclick="addTask()">Ekle</button>
    </div>
    <ul id="task-list"></ul>
  </div>
  <script>
    let tasks = ['HTML5 Şablonunu incele', 'Canlı test gerçekleştir'];
    function render() {{
      const ul = document.getElementById('task-list');
      ul.innerHTML = '';
      tasks.forEach((t, i) => {{
        const li = document.createElement('li');
        li.innerHTML = '<span onclick="toggleTask(' + i + ')" style="cursor:pointer;">' + t + '</span><button type="button" class="del-btn" onclick="delTask(' + i + ')">Sil</button>';
        ul.appendChild(li);
      }});
    }}
    function addTask() {{
      const inp = document.getElementById('task-inp');
      if (!inp.value.trim()) return;
      tasks.push(inp.value.trim());
      inp.value = '';
      render();
    }}
    function delTask(i) {{ tasks.splice(i, 1); render(); }}
    function toggleTask(i) {{
      const el = document.querySelectorAll('#task-list li span')[i];
      el.classList.toggle('completed');
    }}
    render();
  </script>
</body>
</html>"""

    def _build_dashboard(self, intent: HtmlDesignIntent) -> str:
        p = intent.palette
        brand = html.escape(intent.author_name or intent.topic or "Operasyonel")
        return f"""<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{brand} Analitik Dashboard</title>
  <style>
    :root {{
      --bg: {p['bg']};
      --card-bg: {p['card_bg']};
      --text: {p['text']};
      --primary: {p['primary']};
      --border: {p['border']};
      --muted: {p['muted']};
    }}
    body {{
      font-family: system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 30px;
    }}
    .dashboard {{ max-width: 1000px; margin: 0 auto; }}
    .stats-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 30px;
    }}
    .stat-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
    }}
    .stat-val {{ font-size: 28px; font-weight: 800; color: var(--primary); margin: 8px 0; }}
    .stat-label {{ color: var(--muted); font-size: 13px; }}
    .table-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
    }}
    table {{ width: 100%; border-collapse: collapse; text-align: left; }}
    th, td {{ padding: 14px 18px; font-size: 14px; border-bottom: 1px solid var(--border); }}
    th {{ background: rgba(0,0,0,0.2); }}
    .search-inp {{
      width: 100%;
      box-sizing: border-box;
      padding: 12px;
      background: rgba(0,0,0,0.2);
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--text);
      margin-bottom: 16px;
    }}
  </style>
</head>
<body>
  <div class="dashboard">
    <h2>{brand} Analitik Dashboard</h2>
    <div class="stats-row">
      <div class="stat-card"><div class="stat-label">Toplam Gelir</div><div class="stat-val">₺142.500</div><div>+%14.2 Bu Ay</div></div>
      <div class="stat-card"><div class="stat-label">Aktif Kullanıcı</div><div class="stat-val">3.420</div><div>+%8.5 Artış</div></div>
      <div class="stat-card"><div class="stat-label">Sistem Uptime</div><div class="stat-val">%99.98</div><div>Kesintisiz</div></div>
    </div>
    <input type="text" id="search" class="search-inp" placeholder="Tabloda anında filtrele..." onkeyup="filterTable()">
    <div class="table-card">
      <table id="data-table">
        <thead><tr><th>#ID</th><th>Kullanıcı / Proje</th><th>Rol</th><th>Durum</th></tr></thead>
        <tbody>
          <tr><td>#101</td><td>{brand} Core</td><td>Motor</td><td>Aktif</td></tr>
          <tr><td>#102</td><td>Web Synthesizer</td><td>Korteks</td><td>Aktif</td></tr>
          <tr><td>#103</td><td>Episodic Graph</td><td>Bellek</td><td>Hazır</td></tr>
        </tbody>
      </table>
    </div>
  </div>
  <script>
    function filterTable() {{
      const q = document.getElementById('search').value.toLowerCase();
      const rows = document.querySelectorAll('#data-table tbody tr');
      rows.forEach(r => {{
        r.style.display = r.textContent.toLowerCase().includes(q) ? '' : 'none';
      }});
    }}
  </script>
</body>
</html>"""

    def _build_counter(self, intent: HtmlDesignIntent) -> str:
        p = intent.palette
        return f"""<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Etkileşimli Sayaç</title>
  <style>
    body {{
      margin: 0;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
      background: {p['bg']};
      color: {p['text']};
      font-family: system-ui, sans-serif;
    }}
    .counter-card {{
      background: {p['card_bg']};
      border: 1px solid {p['border']};
      border-radius: 16px;
      padding: 36px;
      text-align: center;
      box-shadow: {p['shadow']};
      width: 320px;
    }}
    h2 {{ color: {p['text']}; margin: 0 0 16px; font-size: 20px; }}
    .counter-display {{
      font-size: 54px;
      font-weight: 800;
      color: {p['primary']};
      margin: 20px 0;
    }}
    .controls {{ display: flex; gap: 12px; justify-content: center; }}
    .btn {{
      padding: 10px 18px;
      font-size: 16px;
      font-weight: 700;
      border-radius: 8px;
      border: none;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .btn-inc {{ background: {p['primary']}; color: #fff; }}
    .btn-inc:hover {{ background: {p['hover']}; }}
    .btn-dec {{ background: {p['accent_badge']}; color: {p['badge_text']}; }}
    .btn-reset {{ background: transparent; border: 1px solid {p['border']}; color: {p['muted']}; }}
  </style>
</head>
<body>
  <div class="counter-card">
    <h2>Etkileşimli Sayaç</h2>
    <div id="counter-val" class="counter-display">0</div>
    <div class="controls">
      <button type="button" id="btn-dec" class="btn btn-dec" onclick="step(-1)">-</button>
      <button type="button" id="btn-reset" class="btn btn-reset" onclick="resetCounter()">Sıfırla</button>
      <button type="button" id="btn-inc" class="btn btn-inc" onclick="step(1)">+</button>
    </div>
  </div>
  <script>
    let count = 0;
    const disp = document.getElementById("counter-val");
    function step(n) {{
      count += n;
      disp.textContent = count;
    }}
    function resetCounter() {{
      count = 0;
      disp.textContent = count;
    }}
  </script>
</body>
</html>"""

    def _build_domain_showcase(self, intent: HtmlDesignIntent) -> str:
        p = intent.palette
        brand = html.escape(intent.author_name or intent.topic or "Modern Web Deneyimi")
        role = html.escape(intent.role_desc if intent.role_desc else "Yaratıcı Web Deneyimi")
        return f"""<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{brand} | {role}</title>
  <style>
    :root {{
      --bg: {p['bg']};
      --card-bg: {p['card_bg']};
      --card-elevated: {p['card_elevated']};
      --text: {p['text']};
      --muted: {p['muted']};
      --primary: {p['primary']};
      --primary-hover: {p['hover']};
      --accent: {p['accent']};
      --border: {p['border']};
      --gradient: {p['gradient']};
      --shadow: {p['shadow']};
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 20px 40px;
      border-bottom: 1px solid var(--border);
      background: rgba(0,0,0,0.2);
      backdrop-filter: blur(10px);
      position: sticky;
      top: 0;
      z-index: 100;
    }}
    .brand {{ font-size: 22px; font-weight: 800; background: var(--gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-decoration: none; }}
    nav a {{ color: var(--muted); text-decoration: none; margin-left: 24px; font-weight: 500; transition: color 0.2s; }}
    nav a:hover {{ color: var(--primary); }}
    .btn-header {{
      background: var(--primary);
      color: #fff;
      border: none;
      padding: 9px 20px;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
    }}
    .hero {{
      text-align: center;
      padding: 80px 24px;
      max-width: 900px;
      margin: 0 auto;
    }}
    .pill {{
      display: inline-block;
      padding: 6px 16px;
      background: {p['badge_bg']};
      color: {p['badge_text']};
      border-radius: 999px;
      font-size: 13px;
      font-weight: 600;
      margin-bottom: 24px;
    }}
    .hero h1 {{
      font-size: 48px;
      font-weight: 800;
      line-height: 1.2;
      margin-bottom: 20px;
      background: var(--gradient);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .hero p {{ font-size: 19px; color: var(--muted); margin-bottom: 32px; max-width: 650px; margin-left: auto; margin-right: auto; }}
    .hero-actions {{ display: flex; gap: 14px; justify-content: center; }}
    .btn-primary {{
      background: var(--primary);
      color: #fff;
      padding: 14px 32px;
      border-radius: 10px;
      border: none;
      font-size: 16px;
      font-weight: 700;
      cursor: pointer;
      box-shadow: 0 4px 16px rgba(0,0,0,0.3);
      transition: transform 0.15s;
    }}
    .btn-primary:hover {{ transform: translateY(-2px); }}
    .btn-outline {{
      background: transparent;
      border: 1.5px solid var(--border);
      color: var(--text);
      padding: 14px 28px;
      border-radius: 10px;
      font-size: 16px;
      font-weight: 600;
      cursor: pointer;
    }}
    .features-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 24px;
      max-width: 1000px;
      margin: 40px auto 80px auto;
      padding: 0 24px;
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 32px 24px;
      transition: transform 0.2s, box-shadow 0.2s;
    }}
    .card:hover {{ transform: translateY(-4px); box-shadow: var(--shadow); }}
    .card h3 {{ font-size: 20px; margin-bottom: 12px; color: var(--text); }}
    .card p {{ color: var(--muted); font-size: 14px; }}
    footer {{
      text-align: center;
      padding: 36px;
      border-top: 1px solid var(--border);
      color: var(--muted);
      font-size: 14px;
    }}
  </style>
</head>
<body>
  <header>
    <a href="#" class="brand">{brand}</a>
    <nav>
      <a href="#ozellikler">Özellikler</a>
      <a href="#hakkinda">Hakkında</a>
      <a href="#iletisim">İletişim</a>
    </nav>
    <button type="button" class="btn-header" onclick="alert('Hoş Geldiniz!')">Keşfet</button>
  </header>
  <main>
    <section class="hero">
      <div class="pill">✨ {role}</div>
      <h1>{brand} İle Geleceği Şekillendirin</h1>
      <p>Yüksek performanslı, semantik ve tamamen duyarlı arayüz tasarımı ile kullanıcılarınıza benzersiz bir deneyim sunun.</p>
      <div class="hero-actions">
        <button type="button" class="btn-primary" onclick="alert('Başlatıldı!')">Hemen Başla</button>
        <button type="button" class="btn-outline" onclick="alert('Detaylar gösteriliyor...')">Detaylı Bilgi</button>
      </div>
    </section>
    <section id="ozellikler" class="features-grid">
      <div class="card">
        <h3>⚡ Yüksek Hız</h3>
        <p>W3C semantik standartları ve optimize edilmiş CSS mimarisi ile anında yüklenir.</p>
      </div>
      <div class="card">
        <h3>🛡️ Güvenli & Erişilebilir</h3>
        <p>A11y erişilebilirlik yönergelerine ve mobil uyumlu viewport yapısına tam uygundur.</p>
      </div>
      <div class="card">
        <h3>🎨 Dinamik Tasarım</h3>
        <p>Kullanıcı etkileşimlerine duyarlı, esnek ve modern görsel bileşenler.</p>
      </div>
    </section>
  </main>
  <footer>
    <p>&copy; 2026 {brand}. Tüm hakları saklıdır.</p>
  </footer>
  <script>
    function toggleTheme() {{
      const b = document.body;
      b.dataset.theme = b.dataset.theme === 'alt' ? 'default' : 'alt';
      if (b.dataset.theme === 'alt') {{
        b.style.filter = 'invert(0.9) hue-rotate(180deg)';
      }} else {{
        b.style.filter = 'none';
      }}
    }}
    function exploreAction() {{
      alert('Keşfet butonu tıklandı! {brand} deneyimi aktif.');
    }}
  </script>
</body>
</html>"""

    def synthesize(self, request: str, agi: Any = None) -> dict:
        t0 = time.perf_counter()
        intent = self.parse_design_intent(request, agi=agi)
        markup = self.generate_markup(intent)

        # Doğrulama (Structural HTML verification)
        ver = self.verifier.verify(markup, intent.required_tags)
        if not ver["ok"]:
            return {
                "ok": False,
                "language": "html",
                "path": "html_error",
                "errors": ver["errors"],
                "reason": "html_dogrulama_hatasi",
                "reply_hint": f"HTML kod yapısında doğrulama hatası tespit edildi: {', '.join(ver['errors'])}",
                "ms": round((time.perf_counter() - t0) * 1000, 2),
            }

        return {
            "ok": True,
            "language": "html",
            "path": "html",
            "category": intent.category,
            "archetype": intent.archetype,
            "title_tr": intent.title_tr,
            "author_name": intent.author_name,
            "role_desc": intent.role_desc,
            "program": markup,
            "tags_seen": ver["tags_seen"],
            "total_tags": ver["total_tags"],
            "target_text": intent.target_text,
            "topic": intent.topic,
            "palette_name": intent.palette.get("name", "Modern Web"),
            "theme_key": intent.theme_key,
            "verified": True,
            "ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def speak(self, res: dict, agi: Any = None) -> str:
        if not res.get("ok"):
            return str(res.get("reply_hint") or "İstenen HTML bileşeni doğrulanamadı.")

        category = res.get("category", "showcase")
        title_tr = res.get("title_tr", "Dinamik Web Deneyimi")
        author_name = res.get("author_name", "")
        role_desc = res.get("role_desc", "")
        markup = res.get("program", "")
        target_text = res.get("target_text", "")
        topic = res.get("topic", "")
        palette_name = res.get("palette_name", "Modern Web")

        chem = getattr(agi, "chem", None) if agi else None
        da = float(getattr(chem, "da", 0.5)) if chem else 0.5
        sht = float(getattr(chem, "sht", 0.5)) if chem else 0.5
        oxy = float(getattr(chem, "oxy", 0.5)) if chem else 0.5
        cort = float(getattr(chem, "cort", 0.2)) if chem else 0.2
        seed_turn = int(getattr(getattr(agi, "mind", None), "age", 0)) if agi else 0
        rot = (seed_turn + len(markup) + int(da * 100)) % 3

        if author_name and role_desc:
            lead_sentence = f"İstediğin gibi {author_name} için hazırladığım, {role_desc} odaklı ve {palette_name} temalı dinamik portfolyo sayfası hazır:"
        elif author_name:
            lead_sentence = f"Tam senin istediğin gibi {author_name} adına özel, {palette_name} renk düzeniyle çalışan dinamik {title_tr.lower()} oluşturdum:"
        elif da >= 0.65:
            leads = [
                f"Harika bir fikir! {palette_name} temalı, tamamen dinamik ve etkileşimli bir {title_tr.lower()} tasarladım:",
                f"Web geliştirmenin en keyifli parçalarından biri! Senin isteğine göre çalışan dinamik bir {title_tr.lower()} hazırladım:",
                f"Hemen modern 2026 web standartlarıyla kurguladım; işte doğrudan kullanabileceğin etkileyici bir {title_tr.lower()}:",
            ]
            lead_sentence = leads[rot]
        else:
            leads = [
                f"İstediğin amaca uygun, {palette_name} stili ve etkileşimli JavaScript mantığıyla hazırlanmış {title_tr.lower()} hazır:",
                f"Web arayüzlerinde doğrudan çalıştırabileceğin, yapısal sözdizimi doğrulanmış {title_tr.lower()}:",
                f"Modern web geliştirme standartlarına uygun, temiz ve dinamik bir {title_tr.lower()} hazırladım:",
            ]
            lead_sentence = leads[rot]

        points = []
        if author_name:
            points.append(f"**Kişiselleştirilmiş İsim & Marka**: İlettiğin **'{author_name}'** adı tepe logo, başlık ve altbilgiye doğrudan yerleştirildi.")
        if role_desc:
            points.append(f"**Unvan & Tanıtım**: Belirttiğin **'{role_desc}'** unvanı ve uzmanlık odağı sayfanın ana vitrininde vurgulandı.")

        points.append(f"**Görsel Tema & CSS**: Tasarım **{palette_name}** renk paleti, modern CSS Grid/Flexbox ve akıcı kutu gölgeleriyle donatıldı.")

        if category == "portfolio":
            points.extend([
                "**Bölüm Mimarisi**: Hakkımda, Teknik Yetenekler, Öne Çıkan Projeler ve İletişim blokları responsive hiyerarşiyle yapılandırıldı.",
                "**İnteraktif Proje Filtreleme**: Projeler arasında kategoriye göre (Tümü / Web Uygulaması / Arayüz) anlık filtreleme yapan JavaScript mekanizması eklendi.",
                "**Canlı İletişim Formu**: Form gönderildiğinde istemci tarafında doğrulama yapıp anında geri bildirim veren JavaScript kodu entegre edildi."
            ])
        elif category == "auth":
            points.extend([
                "**İnteraktif Tab Geçişi**: JavaScript ile Kayıt Ol ve Giriş Yap formları arasında sayfa yenilenmeden anında sekme geçişi sağlanır.",
                "**Canlı Şifre Gücü Ölçer (Password Strength)**: Kullanıcı şifre yazarken uzunluk, büyük/küçük harf, rakam ve sembolleri dinamik analiz ederek renkli ilerleme çubuğuyla anında geri bildirim sunar.",
                "**Şifre Eşleşme Doğrulaması**: Kayıt esnasında girilen iki şifrenin eşitliği istemci tarafında anlık kontrol edilir."
            ])
        elif category == "calculator":
            points.extend([
                "**Çalışan Hesaplama Mantığı**: JavaScript durum makinesi ile toplama, çıkarma, çarpma ve bölme işlemleri gerçek zamanlı hesaplanır.",
                "**İşlem Geçmişi Ekranı**: Önceki işlemler ve mevcut girdi çift katmanlı dijital ekranda canlı gösterilir."
            ])
        elif category == "tictactoe":
            points.extend([
                "**Etkileşimli Oyun Tahtası**: 3x3 ızgarada oyuncu sırası (X ve O) JavaScript ile dinamik olarak takip edilir.",
                "**Otomatik Kazanma Kontrolü**: Tüm yatay, dikey ve çapraz kombinasyonlar her hamlede denetlenir ve kazanan duyurulur."
            ])
        elif category == "todo":
            points.extend([
                "**Dinamik DOM Yönetimi**: Kullanıcı yeni görevler ekleyebilir, tamamlanan görevlerin üzerini tıklayarak çizebilir ve silebilir.",
                "**Hafif ve Hızlı**: Harici hiçbir kütüphaneye ihtiyaç duymadan saf Vanilla JavaScript ile çalışır."
            ])
        elif category == "dashboard":
            points.extend([
                "**Anlık Arama ve Filtreleme**: Arama kutusuna yazılan metne göre tablo satırları klavye vuruşunda gerçek zamanlı filtrelenir.",
                "**Özet İstatistik Kartları**: Gelir, kullanıcı ve çalışma süresi KPI metrikleri modern kart bloklarında sergilenir."
            ])
        elif category == "counter":
            points.extend([
                "**Dinamik Sayaç Yönetimi**: Artırma, azaltma ve sıfırlama butonlarıyla sayısal durum anında güncellenir."
            ])
        else:
            points.extend([
                "**W3C ve HTML5 Standartları**: Belge `<!DOCTYPE html>`, meta `viewport` ve semantik etiketlerle eksiksiz kurgulandı.",
                "**Duyarlı Tasarım (Responsive)**: Mobil cihazlardan geniş masaüstü ekranlara kadar tam uyumlu akıcı yerleşim."
            ])

        pedagogy = "\n".join(f"- {p}" for p in points)
        tails = [
            "Bu kodu bir `.html` dosyası olarak kaydedip doğrudan tarayıcında açarak canlı etkileşimleri anında deneyimleyebilirsin!",
            "İstersen tasarıma yeni veri alanları ekleyebilir ya da renk paletini kurumsal renklerine göre güncelleyebilirim.",
            "Canlı JavaScript mantığı tamamen tarayıcı üzerinde çalıştığı için herhangi bir sunucu kurulumu gerektirmeden anında çalışır!"
        ]
        tail_sentence = tails[(seed_turn + 1) % len(tails)]

        return f"{lead_sentence}\n\n```html\n{markup}\n```\n\n### 💡 Nasıl Çalışır ve Tasarım Kararları:\n{pedagogy}\n\n{tail_sentence}"


# 4. KOD SENTEZ KORTEKSİ (cephe)
# ============================================================================

class CodeSynthesisCortex:
    """Nora'nın kod üretme organı: niyet -> aday programlar -> ÇALIŞTIRARAK DOĞRULAMA -> beyan.
    Doğrulanmamış program asla dışarı verilmez; bilinmeyen istekte dürüstçe 'yapamam' der."""

    MAX_CANDIDATES = 6
    # Her ipucu kökünün kanıt gücü: "çıktı" tek başına kod istemi sayılmaz (Türkçede 'ortaya çıktı'
    # anlamına da gelir); "yazdır/print/kod" ise tek başına yeter. Bu bir kural tablosu değil,
    # duyusal özellik ağırlığıdır — toplam kanıt eşiği geçilirse korteks ateşlenir.
    CUE_STRENGTH = {
        "yazdır": 1.0, "yazdir": 1.0, "print": 1.0, "kod": 0.9, "python": 1.0, "program": 0.8,
        "satır": 0.7, "bastır": 0.9, "ekrana": 0.6, "göster": 0.5, "çıktı ver": 0.6,
    }
    REQUEST_THRESHOLD = 1.0

    def requested_evidence(self, raw: str) -> float:
        feats = self.cortex.presence_features(raw)
        return float(sum(self.CUE_STRENGTH.get(c, 0.5) * v for c, v in feats.items()))

    def is_requested(self, raw: str) -> bool:
        if getattr(self, "html_synthesizer", None) is not None and self.html_synthesizer.is_requested(raw):
            return True
        if self.requested_evidence(raw) < self.REQUEST_THRESHOLD:
            return False
        op, probs = self.cortex.select(raw)
        return op == "DISPLAY" and float(probs.get("DISPLAY", 0.0)) >= 0.45

    def __init__(self, seed: int = 0, pretrain: bool = True):
        self.cortex = PlasticOperatorCortex(seed=seed)
        self.grounder = RequestGrounder(self.cortex)
        self.verifier = ExecutionVerifier()
        self.html_synthesizer = HtmlCodeSynthesizer()
        self.syn = TurkishSyntaxEngine
        self.skills: dict[str, dict] = {}      # niyet imzası -> doğrulanmış program (System-1)
        self.feedback_log: list[dict] = []
        self.min_score = 0.30
        self.semantic = CompositionalCodeLearner()
        if pretrain:
            primer = REPO_ROOT / "data" / "python_semantics_primer.md"
            try:
                self.semantic.load_primer(primer)
            except (OSError, UnicodeError):
                pass

    # --------------------------------------------------------------- yardımcılar
    @staticmethod
    def _signature(intent: CodeIntent) -> str:
        return f"{intent.verb}|{'tek' if intent.single_line else 'cok'}|{intent.payload}"

    def _candidates(self, intent: CodeIntent) -> list[str]:
        if intent.verb != "DISPLAY" or not intent.payload:
            return []
        styles = ['"', "'"]
        variants = [0, 1] if intent.single_line else [0, 1, 2]
        out: list[str] = []
        for style in styles:
            for v in variants:
                prog = PythonEmitter.display(intent.payload, style=style, variant=v)
                if intent.single_line and "\n" in prog:
                    continue
                if prog not in out:
                    out.append(prog)
        return out[: self.MAX_CANDIDATES]

    def _clarify(self, missing: str) -> str:
        syn = self.syn
        if missing == "payload":
            c1 = syn.realize_verb(VerbSpec(
                verb_root="söyle", obj=syn.nominalize_dik("yazdır", poss=2, case="ACC"),
                neg=True, tense="PAST", person="2SG",
            ))
            c2 = syn.realize_verb(VerbSpec(verb_root="söyle", obj=NP("metin", case="ACC"), tense="IMP", person="2SG"))
            c3 = syn.realize_verb(VerbSpec(
                verb_root="ver", obj=NP("satır", case="ACC"),
                advs=("ancak çalıştırıp doğruladıktan sonra",), tense="AOR", person="1SG",
            ))
            return f"{tr_capitalize(c1)}. {tr_capitalize(c2)}; {c3}."
        c1 = syn.realize_verb(VerbSpec(verb_root="çözümle", obj=NP("istek", poss=2, case="ACC"), tense="PAST", person="1SG"))
        c2 = syn.realize_verb(VerbSpec(verb_root="anla", advs=("hangi işlemi istediğini",),
                                       neg=True, tense="PAST", person="1SG"))
        return f"{tr_capitalize(c1)}, ama {c2}. Ne yapmamı istediğini tek cümleyle söyle; uydurmak yerine soruyorum."

    # --------------------------------------------------------------- anlamsal bileşim
    def _handle_semantic(self, goal: SemanticGoal, request: str, started: float) -> dict:
        if isinstance(goal, SemanticGoal) and goal.layout != "vertical":
            solved = {"ok": False, "reason": "layout_gap", "missing": [goal.layout],
                      "gap_stage": "layout", "trace": list(goal.trace) + ["gap_stage=layout"]}
        else:
            solved = self.semantic.solve(goal)
        if isinstance(solved, dict):
            missing = [str(value) for value in solved.get("missing") or ()]
            c_plan = self.syn.realize_verb(VerbSpec(
                verb_root="kur", obj=NP("plan", case="ACC"), neg=True,
                tense="PAST", person="1SG",
            ))
            return {"ok": False, "path": "semantic_gap", "program": "", "verified": False,
                    "confidence": 0.0, "reason": solved.get("reason", "semantic_gap"),
                    "missing": missing, "reply_hint": tr_capitalize(c_plan) + ": " + ", ".join(missing),
                    "ms": round((time.perf_counter() - started) * 1000, 2),
                    "trace": solved.get("trace") or list(goal.trace), "intent": goal}
        plan: ProgramPlan = solved
        ver = self.verifier.run(plan.program)
        matched = bool(ver.get("ok") and ver.get("stdout") == plan.expected)
        trace = list(plan.trace) + [f"exact_output={matched}"]
        if not matched:
            return {"ok": False, "path": "unverified", "program": "", "verified": False,
                    "confidence": 0.0, "reason": "doğrulanamadı", "reply_hint": self._clarify("intent"),
                    "ms": round((time.perf_counter() - started) * 1000, 2),
                    "trace": trace + [f"stderr={ver.get('stderr', '')}"], "intent": goal}
        self.verifier.successes += 1
        self.semantic.adopt(goal, plan)
        self.cortex.reinforce(request, "DISPLAY", good=True)
        if "iterate" in plan.operators:
            self.cortex.reinforce(request, "LOOP", good=True)
        self.feedback_log.append({"sig": "semantic:" + "|".join(goal.key),
                                  "program": plan.program, "ok": True})
        return {"ok": True, "path": "composed", "program": plan.program,
                "stdout": ver["stdout"], "expected": plan.expected, "verified": True,
                "confidence": 1.0, "operators": list(plan.operators), "run_ms": ver["ms"],
                "ms": round((time.perf_counter() - started) * 1000, 2),
                "trace": trace, "intent": goal}

    @staticmethod
    def is_document(request: str) -> bool:
        low = tr_lower(request or "")
        has_url = bool(re.search(r"https?://", low))
        return (bool(re.match(r"^\s*(?:doküman|belge|document)\s*:", low))
                or ("```" in low and any(cue in low for cue in ("öğren", "oku", "belge")))
                or (has_url and any(cue in low for cue in ("öğren", "oku", "belge", "doküman"))))

    def handle_document(self, request: str) -> dict:
        url = re.search(r"https?://[^\s<>]+", request)
        if url:
            result = self.semantic.learn_url(url.group(0).rstrip(".,;"))
        else:
            body = re.sub(r"^\s*(?:doküman|belge|document)\s*:\s*", "", request,
                          count=1, flags=re.IGNORECASE)
            result = self.semantic.learn_document(body, source="conversation")
        result.update({"path": "document", "program": "", "verified": False})
        if not result.get("ok"):
            c_read = self.syn.realize_verb(VerbSpec(
                verb_root="oku", obj=NP("belge", case="ACC"), neg=True,
                tense="PAST", person="1SG",
            ))
            result["reply_hint"] = tr_capitalize(c_read) + ": " + str(result.get("error", "0"))
        return result

    @capability("request", "code_result",
                validator=lambda value: isinstance(value, dict) and bool(value.get("ok")))
    def handle(self, request: str, agi: Any = None) -> dict:
        t0 = time.perf_counter()
        if getattr(self, "html_synthesizer", None) is not None and self.html_synthesizer.is_requested(request):
            return self.html_synthesizer.synthesize(request, agi=agi)
        goal = self.semantic.request(request)
        if goal is not None:
            return self._handle_semantic(goal, request, t0)
        intent = self.grounder.ground(request)
        trace = list(intent.trace)

        # 1) System-1: aynı niyet daha önce çalıştırılarak doğrulandıysa anında dön
        sig = self._signature(intent)
        cached = self.skills.get(sig)
        if cached is not None:
            ver = self.verifier.verify(cached["program"], intent.payload)
            if ver["matched"]:
                cached["hits"] = cached.get("hits", 0) + 1
                return {
                    "ok": True, "path": "system1", "program": cached["program"],
                    "stdout": ver["stdout"], "confidence": 1.0, "verified": True,
                    "ms": round((time.perf_counter() - t0) * 1000, 2), "trace": trace + ["System-1 beceri belleğinden"],
                    "intent": intent,
                }

        # 2) Niyet belirsizse: uydurma, açıkça söyle
        if intent.verb != "DISPLAY":
            self.cortex.reinforce(request, intent.verb or "DISPLAY", good=False, lr=0.02)
            return {
                "ok": False, "path": "abstain", "program": "", "verified": False, "confidence": 0.0,
                "reason": "belirsiz_niyet", "reply_hint": self._clarify("intent"),
                "ms": round((time.perf_counter() - t0) * 1000, 2), "trace": trace, "intent": intent,
            }
        if not intent.payload.strip():
            return {
                "ok": False, "path": "abstain", "program": "", "verified": False, "confidence": 0.0,
                "reason": "metin_yok", "reply_hint": self._clarify("payload"),
                "ms": round((time.perf_counter() - t0) * 1000, 2), "trace": trace, "intent": intent,
            }

        # 3) Adayları üret ve ÇALIŞTIRARAK doğrula
        candidates = self._candidates(intent)
        for prog in candidates:
            ver = self.verifier.verify(prog, intent.payload)
            trace.append(f"aday={prog} -> çıktı={ver['stdout']!r} eşleşti={ver['matched']}")
            if ver["matched"]:
                self.cortex.reinforce(request, "DISPLAY", good=True)
                self.skills[sig] = {"program": prog, "payload": intent.payload, "hits": 1,
                                    "verified_at": time.time()}
                self.feedback_log.append({"sig": sig, "program": prog, "ok": True})
                return {
                    "ok": True, "path": "verified", "program": prog, "stdout": ver["stdout"],
                    "confidence": 1.0, "verified": True, "run_ms": ver["ms"],
                    "ms": round((time.perf_counter() - t0) * 1000, 2), "trace": trace, "intent": intent,
                }

        # 4) Hiçbir aday doğrulanmadıysa: başarısızlığı gizleme, öğren ve dürüst ol
        self.cortex.reinforce(request, "DISPLAY", good=False)
        self.feedback_log.append({"sig": sig, "program": candidates[0] if candidates else "", "ok": False})
        return {
            "ok": False, "path": "unverified", "program": "", "verified": False, "confidence": 0.0,
            "reason": "doğrulanamadı", "reply_hint": self._clarify("intent"),
            "ms": round((time.perf_counter() - t0) * 1000, 2), "trace": trace, "intent": intent,
        }

    # --------------------------------------------------------------- beyan
    def speak(self, res: dict, agi: Any = None) -> str:
        """Doğrulanmış programı, kendi kök çekimiyle kurulmuş Türkçe bir beyana dönüştürür."""
        if res.get("language") == "html" or res.get("path") == "html":
            return self.html_synthesizer.speak(res, agi=agi)
        syn = self.syn
        if not res.get("ok"):
            return str(res.get("reply_hint") or "")
        if res.get("path") == "document":
            learned = list(res.get("new_operators") or ()) + list(res.get("new_facts") or ())
            if not learned:
                learned = list(res.get("operators") or ()) + list((res.get("facts") or {}).keys())
            c_read = syn.realize_verb(VerbSpec(verb_root="oku", obj=NP("belge", case="ACC"),
                                               tense="PAST", person="1SG"))
            return tr_capitalize(c_read) + ": " + (", ".join(learned) or "0") + "."
        if res.get("path") == "composed":
            prog = str(res.get("program") or "")
            out = str(res.get("stdout") or "").rstrip("\n")
            c_write = syn.realize_verb(VerbSpec(verb_root="yaz", obj=NP("program", case="ACC"),
                                                tense="PAST", person="1SG"))
            c_run = syn.realize_verb(VerbSpec(verb_root="çalıştır", obj="onu",
                                              tense="PAST", person="1SG"))
            head = tr_capitalize(c_write) + "."
            return (head + "\n```python\n" + prog + "\n```\n"
                    + tr_capitalize(c_run) + "; çıktı:\n```text\n" + out + "\n```")
        prog = res.get("program", "")
        out = str(res.get("stdout", "")).strip()
        c_run = syn.realize_verb(VerbSpec(verb_root="çalıştır", obj="onu", advs=("hemen",),
                                          tense="PAST", person="1SG"))
        c_chk = syn.realize_verb(VerbSpec(verb_root="karşılaştır", obj=NP("çıktı", poss=3, case="ACC"),
                                          advs=("istediğin metinle",), tense="PAST", person="1SG"))
        if res.get("path") == "system1":
            c1 = syn.realize_verb(VerbSpec(verb_root="hatırla", obj="bunu", advs=("düşünmeden",),
                                           tense="PRES", person="1SG"))
            head = f"{tr_capitalize(c1)}; bu satırı daha önce çalıştırıp doğrulamıştım."
        else:
            c1 = syn.realize_verb(VerbSpec(verb_root="yaz", obj=NP("satır", case="ACC"),
                                           advs=("istediğin gibi",), tense="PAST", person="1SG"))
            head = f"{tr_capitalize(c1)}."
        verify = f"{tr_capitalize(c_run)} ve {c_chk}; çıktısı birebir şu: {out}"
        stop = "" if out.endswith(("!", "?", ".")) else "."
        return f"{head} `{prog}` — {verify}{stop} Doğrulanmamış hiçbir kodu sana vermem."

    def snapshot(self) -> dict:
        return {
            "skills": len(self.skills), "learning_events": self.cortex.learning_events,
            "verified_runs": self.verifier.runs, "successes": self.verifier.successes,
            "semantic_operators": len(self.semantic.operators),
            "composite_operators": len(self.semantic.composites),
            "weights": {op: {k: round(v, 2) for k, v in list(w.items())[:5]} for op, w in self.cortex.W.items()},
        }

    def state_dict(self) -> dict:
        return {"W": {k: dict(v) for k, v in self.cortex.W.items()},
                "skills": {k: {"program": v["program"], "payload": v["payload"], "hits": v.get("hits", 1)}
                           for k, v in list(self.skills.items())[-400:]},
                "events": self.cortex.learning_events,
                "semantic": self.semantic.state_dict()}

    def load_state_dict(self, d: dict):
        for opn, wmap in (d.get("W") or {}).items():
            if opn in self.cortex.W:
                self.cortex.W[opn].update({k: float(v) for k, v in wmap.items()})
        self.skills = {k: dict(v, payload=v.get("payload", "")) for k, v in (d.get("skills") or {}).items()}
        self.cortex.learning_events = int(d.get("events", 0))
        if d.get("semantic") is not None:
            self.semantic.load_state_dict(d.get("semantic") or {})
