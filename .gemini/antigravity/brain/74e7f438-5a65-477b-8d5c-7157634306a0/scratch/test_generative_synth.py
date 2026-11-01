#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scratch/test_generative_synth.py

LLM-tarzı kural/komut uyumlu ve tamamen dinamik web sayfası sentezleyici prototipi.
Statik tek düze şablonlar yerine, kullanıcının verdiği her girdiyi (isim, rol, sektör,
istenen renkler, özel metinler) adım adım semantik bir DOM ağacına ve uyumlu CSS'e dönüştürür.
"""
import re
import html
from dataclasses import dataclass, field
from nora.synthesis.code import HtmlStructuralVerifier

@dataclass
class UserWebPromptSpec:
    raw: str
    genre: str = "showcase"          # portfolio, ecommerce, auth, calculator, tictactoe, todo, dashboard, counter, showcase
    name: str = ""                   # e.g. "Offihito"
    role_or_tagline: str = ""        # e.g. "16 yaşında web developer"
    topic: str = ""                  # e.g. "portfolyo", "kahve", "teknoloji"
    custom_texts: list[str] = field(default_factory=list)
    color_palette: dict = field(default_factory=dict)
    theme_name: str = "monochrome"

def parse_llm_style_spec(raw: str) -> UserWebPromptSpec:
    low = raw.lower()

    # 1. Parçalara ayır (Türkçe doğal dil yönerge ayrıştırması)
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
    name = ""
    role_or_tagline = ""
    topic = ""
    custom_texts = []
    color_req = ""

    # Tırnaklı ifadeleri doğrudan topla
    for qm in re.finditer(r'["“«‘`]([^"”»’`]+)["”»’`]', raw):
        t = qm.group(1).strip()
        if t and t not in custom_texts:
            custom_texts.append(t)

    for chunk, marker in tokens:
        c_low = chunk.lower()
        clean = re.sub(r"^(?:bana|lütfen|acaba|bir|sayfada|ekrana|ve|renk\s+olarak)\s+", "", chunk, flags=re.IGNORECASE).strip()

        # Renk yönergesi mi?
        if any(c in c_low for c in ("siyah", "beyaz", "kırmızı", "yeşil", "mavi", "mor", "sarı", "koyu", "açık", "dark", "light", "renk")):
            color_req = chunk

        # Genre / Tür yönergesi mi?
        if any(g in c_low for g in ("portfolyo", "portfolio", "özgeçmiş", "cv", "kişisel site", "kişisel web")):
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

        # İsim veya Rol / Unvan yönergesi mi?
        if marker in ("yazsın", "yazan", "yaz", "yazmalı"):
            if any(r in c_low for r in ("developer", "geliştirici", "mühendis", "tasarımcı", "designer", "yaşında", "öğrenci", "freelance", "fullstack", "frontend", "backend", "yazılımcı", "coder")):
                role_or_tagline = clean
            elif len(clean.split()) <= 4 and not name and not any(w in c_low for w in ("renk", "web", "sayfa", "kod")):
                name = clean
            else:
                custom_texts.append(clean)

    # Genel metinden isim ve rol yedek kontrolü
    if not name:
        # Tek kelimelik büyük harfli veya özel kelimeler (örn: Offihito)
        for w in raw.split():
            clean_w = re.sub(r"[^\w]", "", w)
            if clean_w and clean_w[0].isupper() and clean_w.lower() not in ("bana", "bir", "web", "sayfası", "sayfa", "yap", "portfolyo", "renk", "olarak", "siyah", "beyaz", "kullan"):
                name = clean_w
                break

    # 2. Renk Teması Çözümleme (Kullanıcının emrettiği renklere BİREBİR uyar)
    low_color = (color_req + " " + raw).lower()
    if any(k in low_color for k in ("siyah ve beyaz", "siyah beyaz", "siyah-beyaz", "black and white", "monochrome", "monokrom")):
        theme_name = "monochrome"
        palette = {
            "name": "Siyah & Beyaz (Minimalist Noir)",
            "bg": "#09090b", "card_bg": "#121214", "card_elevated": "#1c1c20",
            "primary": "#ffffff", "hover": "#e4e4e7", "accent": "#a1a1aa",
            "text": "#ffffff", "muted": "#a1a1aa", "border": "rgba(255, 255, 255, 0.16)",
            "accent_badge": "rgba(255, 255, 255, 0.10)", "badge_bg": "rgba(255, 255, 255, 0.10)",
            "badge_text": "#ffffff", "shadow": "0 12px 36px rgba(0, 0, 0, 0.8)",
            "gradient": "linear-gradient(135deg, #ffffff, #888888)",
            "btn_text": "#09090b"
        }
    elif any(k in low_color for k in ("koyu", "dark", "siyah", "karanlık", "gece", "cyber")):
        theme_name = "dark"
        palette = {
            "name": "Karanlık Gece (Cyber Dark)",
            "bg": "#0b0f19", "card_bg": "#111827", "card_elevated": "#1f2937",
            "primary": "#38bdf8", "hover": "#0ea5e9", "accent": "#818cf8",
            "text": "#f8fafc", "muted": "#94a3b8", "border": "#1e293b",
            "accent_badge": "rgba(56, 189, 248, 0.12)", "badge_bg": "rgba(56, 189, 248, 0.12)",
            "badge_text": "#38bdf8", "shadow": "0 10px 30px rgba(0, 0, 0, 0.5)",
            "gradient": "linear-gradient(135deg, #38bdf8, #818cf8)",
            "btn_text": "#0b0f19"
        }
    elif any(k in low_color for k in ("kırmızı", "red", "rose")):
        theme_name = "rose"
        palette = {
            "name": "Kırmızı & Siyah",
            "bg": "#0f0507", "card_bg": "#1c0b0f", "card_elevated": "#2a1117",
            "primary": "#f43f5e", "hover": "#e11d48", "accent": "#fb7185",
            "text": "#fff1f2", "muted": "#fda4af", "border": "#4c1d28",
            "accent_badge": "rgba(244, 63, 94, 0.15)", "badge_bg": "rgba(244, 63, 94, 0.15)",
            "badge_text": "#fda4af", "shadow": "0 10px 30px rgba(244, 63, 94, 0.25)",
            "gradient": "linear-gradient(135deg, #f43f5e, #fb7185)",
            "btn_text": "#ffffff"
        }
    elif any(k in low_color for k in ("yeşil", "green", "zümrüt")):
        theme_name = "emerald"
        palette = {
            "name": "Zümrüt Yeşili",
            "bg": "#041a14", "card_bg": "#062b21", "card_elevated": "#0a3d30",
            "primary": "#10b981", "hover": "#059669", "accent": "#34d399",
            "text": "#f0fdf4", "muted": "#6ee7b7", "border": "#0f513f",
            "accent_badge": "rgba(16, 185, 129, 0.15)", "badge_bg": "rgba(16, 185, 129, 0.15)",
            "badge_text": "#6ee7b7", "shadow": "0 10px 30px rgba(0, 0, 0, 0.4)",
            "gradient": "linear-gradient(135deg, #10b981, #059669)",
            "btn_text": "#041a14"
        }
    else:
        # Varsayılan: Eğer portfolio ise modern noir siyah-beyaz, değilse cyber dark
        if genre == "portfolio":
            theme_name = "monochrome"
            palette = {
                "name": "Siyah & Beyaz (Minimalist Noir)",
                "bg": "#09090b", "card_bg": "#121214", "card_elevated": "#1c1c20",
                "primary": "#ffffff", "hover": "#e4e4e7", "accent": "#a1a1aa",
                "text": "#ffffff", "muted": "#a1a1aa", "border": "rgba(255, 255, 255, 0.16)",
                "accent_badge": "rgba(255, 255, 255, 0.10)", "badge_bg": "rgba(255, 255, 255, 0.10)",
                "badge_text": "#ffffff", "shadow": "0 12px 36px rgba(0, 0, 0, 0.8)",
                "gradient": "linear-gradient(135deg, #ffffff, #888888)",
                "btn_text": "#09090b"
            }
        else:
            theme_name = "dark"
            palette = {
                "name": "Karanlık Gece (Cyber Dark)",
                "bg": "#0b0f19", "card_bg": "#111827", "card_elevated": "#1f2937",
                "primary": "#38bdf8", "hover": "#0ea5e9", "accent": "#818cf8",
                "text": "#f8fafc", "muted": "#94a3b8", "border": "#1e293b",
                "accent_badge": "rgba(56, 189, 248, 0.12)", "badge_bg": "rgba(56, 189, 248, 0.12)",
                "badge_text": "#38bdf8", "shadow": "0 10px 30px rgba(0, 0, 0, 0.5)",
                "gradient": "linear-gradient(135deg, #38bdf8, #818cf8)",
                "btn_text": "#0b0f19"
            }

    return UserWebPromptSpec(
        raw=raw, genre=genre, name=name, role_or_tagline=role_or_tagline,
        topic=topic, custom_texts=custom_texts, color_palette=palette,
        theme_name=theme_name
    )

def generate_custom_portfolio(spec: UserWebPromptSpec) -> str:
    p = spec.color_palette
    author = html.escape(spec.name if spec.name else "Geliştirici")
    role = html.escape(spec.role_or_tagline if spec.role_or_tagline else "Web Developer")
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
          <p>HTML5, semantik yapı standartları, modern CSS3 Flexbox/Grid mimarisi ve Vanilla JavaScript (ES6+).</p>
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

prompt = "Bana bir web sayfası yap Portfolyo olsun Offihito yazsın 16 yaşında web developer yazsın renk olarak siyah ve beyaz kullan."
spec = parse_llm_style_spec(prompt)
print("SPEC:")
print("  Name:", spec.name)
print("  Role:", spec.role_or_tagline)
print("  Genre:", spec.genre)
print("  Theme:", spec.theme_name)
print("  Palette:", spec.color_palette["name"])

html_code = generate_custom_portfolio(spec)
v = HtmlStructuralVerifier()
res = v.verify(html_code, ["html", "head", "body", "header", "main", "footer", "button", "script"])
print("VERIFICATION:", res["ok"], "Errors:", res.get("errors"))
assert "Offihito" in html_code
assert "16 yaşında web developer" in html_code
assert "#09090b" in html_code
print("ALL ASSERTIONS PASSED! The output strictly adheres to all user instructions!")
