#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import html
from dataclasses import dataclass, field
from typing import Any
from nora.synthesis.code import HtmlStructuralVerifier

PALETTES = {
    "monochrome": {
        "name": "Siyah & Beyaz (Minimalist Noir)",
        "bg": "#09090b",
        "card_bg": "#121214",
        "card_elevated": "#1c1c20",
        "primary": "#ffffff",
        "hover": "#e4e4e7",
        "accent": "#a1a1aa",
        "text": "#ffffff",
        "muted": "#a1a1aa",
        "border": "rgba(255, 255, 255, 0.16)",
        "accent_badge": "rgba(255, 255, 255, 0.10)",
        "badge_bg": "rgba(255, 255, 255, 0.10)",
        "badge_text": "#ffffff",
        "shadow": "0 12px 36px rgba(0, 0, 0, 0.8)",
        "gradient": "linear-gradient(135deg, #ffffff, #888888)"
    },
    "dark": {
        "name": "Karanlık Gece (Cyber Dark)",
        "bg": "#0b0f19", "card_bg": "#111827", "card_elevated": "#1f2937",
        "primary": "#38bdf8", "hover": "#0ea5e9", "accent": "#818cf8",
        "text": "#f8fafc", "muted": "#94a3b8", "border": "#1e293b",
        "accent_badge": "rgba(56, 189, 248, 0.12)", "badge_bg": "rgba(56, 189, 248, 0.12)",
        "badge_text": "#38bdf8",
        "shadow": "0 10px 30px rgba(0, 0, 0, 0.5)",
        "gradient": "linear-gradient(135deg, #38bdf8, #818cf8)"
    },
    "clean_light": {
        "name": "Modern Minimalist Açık",
        "bg": "#f8fafc", "card_bg": "#ffffff", "card_elevated": "#f1f5f9",
        "primary": "#2563eb", "hover": "#1d4ed8", "accent": "#38bdf8",
        "text": "#0f172a", "muted": "#64748b", "border": "#e2e8f0",
        "accent_badge": "#dbeafe", "badge_bg": "#dbeafe",
        "badge_text": "#1e40af",
        "shadow": "0 10px 25px -5px rgba(37, 99, 235, 0.08)",
        "gradient": "linear-gradient(135deg, #2563eb, #38bdf8)"
    }
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

def parse_user_intent(raw: str) -> HtmlDesignIntent:
    low = raw.lower()

    # 1. Metin / İbare Ayıklama ("X yazsın", "Y yazan", tırnaklar)
    custom_snippets = []
    # Tırnak içi
    for qm in re.finditer(r'["“«‘`]([^"”»’`]+)["”»’`]', raw):
        val = qm.group(1).strip()
        if val and val not in custom_snippets:
            custom_snippets.append(val)

    # "X yazsın" / "X yazan"
    for ym in re.finditer(r'([a-zA-ZçğıöşüÇĞİÖŞÜ0-9\s!?,.-]{2,50}?)\s+(?:yazsın|yazan|yazmalı|yazacak)', raw, re.IGNORECASE):
        cand = ym.group(1).strip()
        cand = re.sub(r"^(?:bana|lütfen|acaba|ekrana|bir|sayfada|burada)\s+", "", cand, flags=re.IGNORECASE).strip()
        if cand and not any(w in cand.lower().split() for w in ("html", "kod", "sayfa", "şablon", "site", "website")):
            if cand not in custom_snippets:
                custom_snippets.append(cand)

    # "adı X olsun" / "ismi X olsun"
    name_m = re.search(r'(?:adı|ismi|isim)\s+([a-zA-ZçğıöşüÇĞİÖŞÜ0-9_-]+)', raw, re.IGNORECASE)
    author_name = name_m.group(1).strip() if name_m else ""

    role_desc = ""
    # Snippet'ları analiz et: İsim mi, rol/unvan mı?
    for s in custom_snippets:
        s_low = s.lower()
        if any(r in s_low for r in ("developer", "geliştirici", "mühendis", "tasarımcı", "designer", "yaşında", "öğrenci", "freelance", "fullstack", "frontend")):
            if not role_desc:
                role_desc = s
        elif len(s.split()) <= 3 and not author_name and not any(w in s_low for w in ("merhaba", "selam", "portfolyo", "portfolio")):
            author_name = s

    # 2. Alan / Konu (Topic)
    topic = ""
    tm = re.search(r'([a-zA-ZçğıöşüÇĞİÖŞÜ0-9\s]{2,30}?)\s+(?:için|hakkında|olsun)', raw, re.IGNORECASE)
    if tm:
        cand = re.sub(r"^(?:bana|lütfen|acaba|bir)\s+", "", tm.group(1), flags=re.IGNORECASE).strip()
        if cand and not any(w in cand.lower().split() for w in ("web", "html", "kod", "kendim", "örnek", "basit")):
            topic = cand

    # 3. Kategori Tespiti
    if any(k in low for k in ("portfolyo", "portfolio", "özgeçmiş", "cv", "kişisel site", "kişisel web", "developer sitesi", "profil sitesi")):
        cat = "portfolio"
        title = f"{author_name.title() + ' - ' if author_name else ''}Geliştirici Portfolyosu"
        req = ["html", "head", "body", "header", "main", "footer", "button", "script"]
    elif any(k in low for k in ("register", "kayıt", "kaydol", "signup", "sign up", "üye ol", "hesap aç", "hesap oluştur", "auth", "login", "giriş", "oturum")):
        cat = "auth"
        title = "Kullanıcı Kayıt & Giriş Portalı"
        req = ["html", "body", "form", "input", "button", "script"]
    elif any(k in low for k in ("hesap makinesi", "calculator", "dört işlem")):
        cat = "calculator"
        title = "Modern Hesap Makinesi"
        req = ["html", "body", "button", "script"]
    elif any(k in low for k in ("tic-tac-toe", "tictactoe", "xox", "oyun", "game")):
        cat = "tictactoe"
        title = "Tic-Tac-Toe Web Oyunu"
        req = ["html", "body", "button", "script"]
    elif any(k in low for k in ("todo", "yapılacaklar", "kanban", "görev", "task", "liste")):
        cat = "todo"
        title = "Dinamik Görev & Kanban Yöneticisi"
        req = ["html", "body", "input", "button", "ul", "script"]
    elif any(k in low for k in ("dashboard", "panel", "analitik", "veri tablosu", "istatistik", "admin", "metrik")):
        cat = "dashboard"
        title = "İnteraktif Analitik Dashboard"
        req = ["html", "body", "table", "thead", "tbody", "tr", "th", "td", "input", "script"]
    elif any(k in low for k in ("sayaç", "counter", "kronometre", "zamanlayıcı", "sayı art", "sayı azalsın")):
        cat = "counter"
        title = "Etkileşimli Sayaç & Durum Aracı"
        req = ["html", "body", "button", "script"]
    else:
        cat = "showcase"
        title = topic.title() if topic else (author_name if author_name else "Yaratıcı Web Deneyimi")
        req = ["html", "head", "body", "header", "main", "footer", "button", "script"]

    # 4. Tema Seçimi
    if any(k in low for k in ("siyah ve beyaz", "siyah-beyaz", "siyah beyaz", "black and white", "monochrome", "monokrom", "tek renk")):
        explicit_theme = "monochrome"
    elif any(k in low for k in ("koyu", "dark", "siyah", "karanlık", "gece", "cyber")):
        explicit_theme = "dark"
    elif any(k in low for k in ("açık", "light", "beyaz", "minimal")):
        explicit_theme = "clean_light"
    else:
        explicit_theme = "monochrome" if cat == "portfolio" else "dark"

    pal = PALETTES.get(explicit_theme, PALETTES["dark"])

    return HtmlDesignIntent(
        raw=raw, category=cat, target_text=custom_snippets[0] if custom_snippets else "",
        author_name=author_name, role_desc=role_desc, topic=topic,
        custom_snippets=custom_snippets, theme_key=explicit_theme,
        palette=pal, title_tr=title, required_tags=req, has_js=True,
        archetype=cat, variant_idx=0
    )

prompt = "Bana bir web sayfası yap Portfolyo olsun Offihito yazsın 16 yaşında web developer yazsın renk olarak siyah ve beyaz kullan."
intent = parse_user_intent(prompt)
print("Parsed intent:")
print(f"  Category: {intent.category}")
print(f"  Author Name: {intent.author_name}")
print(f"  Role Desc: {intent.role_desc}")
print(f"  Theme Key: {intent.theme_key}")
print(f"  Title: {intent.title_tr}")
print(f"  Custom Snippets: {intent.custom_snippets}")
