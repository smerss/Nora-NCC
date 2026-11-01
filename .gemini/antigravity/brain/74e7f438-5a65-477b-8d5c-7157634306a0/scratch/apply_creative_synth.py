#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

CODE_PY = Path("/home/offihito/Nora-NCC/nora/synthesis/code.py")
content = CODE_PY.read_text(encoding="utf-8")

start_marker = "@dataclass\nclass HtmlDesignIntent:"
end_marker = "# 4. KOD SENTEZ KORTEKSİ (cephe)"

assert start_marker in content, "start marker not found"
assert end_marker in content, "end marker not found"

before = content[:content.index(start_marker)]
after = content[content.index(end_marker):]

new_block = '''@dataclass
class HtmlDesignIntent:
    raw: str
    category: str = "showcase"
    target_text: str = ""
    topic: str = ""
    theme_key: str = "dark"
    palette: dict = field(default_factory=dict)
    title_tr: str = "Dinamik Web Uygulaması"
    required_tags: list[str] = field(default_factory=lambda: ["html", "head", "body"])
    has_js: bool = True
    archetype: str = "showcase"
    variant_idx: int = 0


PALETTES = {
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
    "indigo": {
        "name": "Derin İndigo & Mor",
        "bg": "#0f172a", "card_bg": "#1e293b", "card_elevated": "#334155",
        "primary": "#6366f1", "hover": "#4f46e5", "accent": "#a855f7",
        "text": "#f8fafc", "muted": "#94a3b8", "border": "#334155",
        "accent_badge": "rgba(99, 102, 241, 0.15)", "badge_bg": "rgba(99, 102, 241, 0.15)",
        "badge_text": "#a5b4fc",
        "shadow": "0 10px 30px rgba(99, 102, 241, 0.2)",
        "gradient": "linear-gradient(135deg, #6366f1, #a855f7)"
    },
    "emerald": {
        "name": "Zümrüt Yeşili (Emerald)",
        "bg": "#064e3b", "card_bg": "#065f46", "card_elevated": "#047857",
        "primary": "#10b981", "hover": "#059669", "accent": "#34d399",
        "text": "#f0fdf4", "muted": "#a7f3d0", "border": "#047857",
        "accent_badge": "rgba(16, 185, 129, 0.15)", "badge_bg": "rgba(16, 185, 129, 0.15)",
        "badge_text": "#6ee7b7",
        "shadow": "0 10px 30px rgba(0, 0, 0, 0.3)",
        "gradient": "linear-gradient(135deg, #10b981, #059669)"
    },
    "rose": {
        "name": "Modern Gül & Günbatımı",
        "bg": "#1a0b16", "card_bg": "#2d1224", "card_elevated": "#421a36",
        "primary": "#f43f5e", "hover": "#e11d48", "accent": "#fb7185",
        "text": "#fff1f2", "muted": "#fecdd3", "border": "#4c1d3d",
        "accent_badge": "rgba(244, 63, 94, 0.15)", "badge_bg": "rgba(244, 63, 94, 0.15)",
        "badge_text": "#fda4af",
        "shadow": "0 10px 30px rgba(244, 63, 94, 0.2)",
        "gradient": "linear-gradient(135deg, #f43f5e, #fb7185)"
    },
    "amber": {
        "name": "Sıcak Kehribar (Amber)",
        "bg": "#1c1917", "card_bg": "#292524", "card_elevated": "#44403c",
        "primary": "#f59e0b", "hover": "#d97706", "accent": "#fbbf24",
        "text": "#fffbeb", "muted": "#fde68a", "border": "#44403c",
        "accent_badge": "rgba(245, 158, 11, 0.15)", "badge_bg": "rgba(245, 158, 11, 0.15)",
        "badge_text": "#fcd34d",
        "shadow": "0 10px 30px rgba(0, 0, 0, 0.4)",
        "gradient": "linear-gradient(135deg, #f59e0b, #fbbf24)"
    },
    "cyan": {
        "name": "Okyanus Turkuazı (Cyan)",
        "bg": "#083344", "card_bg": "#0e4a60", "card_elevated": "#155e75",
        "primary": "#06b6d4", "hover": "#0891b2", "accent": "#67e8f9",
        "text": "#ecfeff", "muted": "#a5f3fc", "border": "#164e63",
        "accent_badge": "rgba(6, 182, 212, 0.15)", "badge_bg": "rgba(6, 182, 212, 0.15)",
        "badge_text": "#67e8f9",
        "shadow": "0 10px 30px rgba(6, 182, 212, 0.2)",
        "gradient": "linear-gradient(135deg, #06b6d4, #3b82f6)"
    },
    "violet": {
        "name": "Asil Mor (Violet)",
        "bg": "#1e1035", "card_bg": "#2e1a4d", "card_elevated": "#3e2468",
        "primary": "#8b5cf6", "hover": "#7c3aed", "accent": "#c084fc",
        "text": "#faf5ff", "muted": "#e9d5ff", "border": "#4c2882",
        "accent_badge": "rgba(139, 92, 246, 0.15)", "badge_bg": "rgba(139, 92, 246, 0.15)",
        "badge_text": "#d8b4fe",
        "shadow": "0 10px 30px rgba(139, 92, 246, 0.25)",
        "gradient": "linear-gradient(135deg, #8b5cf6, #ec4899)"
    },
    "blue": {
        "name": "Safir Mavisi (Sapphire)",
        "bg": "#0f172a", "card_bg": "#1e293b", "card_elevated": "#334155",
        "primary": "#3b82f6", "hover": "#2563eb", "accent": "#60a5fa",
        "text": "#f8fafc", "muted": "#94a3b8", "border": "#334155",
        "accent_badge": "rgba(59, 130, 246, 0.15)", "badge_bg": "rgba(59, 130, 246, 0.15)",
        "badge_text": "#93c5fd",
        "shadow": "0 10px 30px rgba(59, 130, 246, 0.2)",
        "gradient": "linear-gradient(135deg, #3b82f6, #06b6d4)"
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

PALETTE_KEYS = ("dark", "indigo", "emerald", "rose", "amber", "cyan", "violet", "blue", "clean_light")


class HtmlCodeSynthesizer:
    """Nora'nın Yaratıcı ve Dinamik Web Sitesi / Uygulama Sentezleyicisi.
    Kullanıcının isteğini doğal dil ve niyet köklerinden çözümleyerek amaca özel,
    çalışan istemci taraflı JavaScript, duyarlı modern CSS3 ve W3C onaylı
    tam etkileşimli web deneyimleri sentezler."""

    WEB_CUES = (
        "html", "html5", "web", "website", "web sitesi", "site", "sayfa", "arayüz", "frontend",
        "ui", "şablon", "template", "landing", "portal", "app", "uygulama"
    )
    APP_CUES = (
        "register", "kayıt", "kaydol", "signup", "sign up", "üye ol", "hesap aç", "hesap oluştur", "auth",
        "login", "giriş", "oturum", "form", "iletişim", "contact",
        "hesap makinesi", "calculator", "dört işlem",
        "tic-tac-toe", "tictactoe", "xox", "oyun", "game",
        "todo", "yapılacaklar", "kanban", "görev", "task",
        "dashboard", "panel", "analitik", "veri tablosu", "istatistik", "admin", "metrik",
        "sayaç", "counter", "kronometre", "zamanlayıcı", "timer"
    )
    ACTION_VERBS = (
        "yaz", "oluştur", "hazırla", "ver", "yap", "tasarla", "kodla", "kod", "kodunu", "örnek",
        "sağlayın", "sağla", "geliştir", "kur", "programla"
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
        if "html" in low or "web sitesi" in low or "website" in low:
            return True
        return False

    def parse_design_intent(self, raw: str, agi: Any = None) -> HtmlDesignIntent:
        low = tr_lower(raw)

        # 1. Hedef Metin / Başlık
        target_text = ""
        qm = re.search(r"[\"“«‘`]([^\"”»’`]+)[\"”»’`]", raw)
        if qm and qm.group(1).strip():
            target_text = qm.group(1).strip()
        else:
            ym = re.search(r"([a-zA-ZçğıöşüÇĞİÖŞÜ0-9\s!?,.-]{2,45}?)\s+yazan", raw, re.IGNORECASE)
            if ym:
                cand = re.sub(r"^(?:bana|lütfen|acaba|ekrana|bir)\s+", "", ym.group(1), flags=re.IGNORECASE).strip()
                if cand and not any(w in cand.lower().split() for w in ("html", "kod", "sayfa", "şablon", "site", "website")):
                    target_text = cand
            if not target_text:
                em = re.search(r"ekrana\s+([a-zA-ZçğıöşüÇĞİÖŞÜ0-9\s!?,.-]{2,40}?)\s+(?:yaz|bas|göster)", raw, re.IGNORECASE)
                if em:
                    target_text = em.group(1).strip()
            if not target_text and any(k in low for k in ("merhaba dünya", "hello world")):
                target_text = "Merhaba Dünya"

        # 2. Alan / Konu (Topic)
        topic = ""
        tm = re.search(r"([a-zA-ZçğıöşüÇĞİÖŞÜ0-9\s]{2,30}?)\s+(?:için|hakkında)", raw, re.IGNORECASE)
        if tm:
            cand = re.sub(r"^(?:bana|lütfen|acaba|bir)\s+", "", tm.group(1), flags=re.IGNORECASE).strip()
            if cand and not any(w in cand.lower().split() for w in ("web", "html", "kod", "kendim", "örnek", "basit")):
                topic = cand

        # 3. Renk & Görsel Tema
        explicit_theme = ""
        if any(k in low for k in ("açık", "light", "beyaz", "minimal")):
            explicit_theme = "clean_light"
        elif any(k in low for k in ("yeşil", "green", "zümrüt")):
            explicit_theme = "emerald"
        elif any(k in low for k in ("mor", "purple", "eflatun")):
            explicit_theme = "violet"
        elif any(k in low for k in ("mavi", "blue", "deniz")):
            explicit_theme = "blue"
        elif any(k in low for k in ("turkuaz", "cyan", "teal")):
            explicit_theme = "cyan"
        elif any(k in low for k in ("kırmızı", "pembe", "rose")):
            explicit_theme = "rose"
        elif any(k in low for k in ("sarı", "turuncu", "kehribar", "amber")):
            explicit_theme = "amber"
        elif any(k in low for k in ("koyu", "dark", "siyah", "karanlık", "gece", "cyber")):
            explicit_theme = "dark"

        chem = getattr(agi, "chem", None) if agi else None
        da = float(getattr(chem, "da", 0.5)) if chem else 0.5
        sht = float(getattr(chem, "sht", 0.5)) if chem else 0.5
        seed_turn = int(getattr(getattr(agi, "mind", None), "age", 0)) if agi else 0
        hash_val = abs(hash(raw)) + seed_turn + int(da * 73) + int(sht * 37)

        theme_key = explicit_theme or PALETTE_KEYS[hash_val % len(PALETTE_KEYS)]
        pal = PALETTES.get(theme_key, PALETTES["dark"])
        variant_idx = (hash_val // 7) % 3

        # 4. Dinamik Kategori & Fonksiyonel Yapı
        if any(k in low for k in ("register", "kayıt", "kaydol", "signup", "sign up", "üye ol", "hesap aç", "hesap oluştur", "auth", "login", "giriş", "oturum")):
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
            title = topic.title() if topic else (target_text if target_text else "Yaratıcı Web Deneyimi")
            req = ["html", "head", "body", "header", "main", "footer", "button", "script"]

        return HtmlDesignIntent(
            raw=raw, category=cat, target_text=target_text, topic=topic,
            theme_key=theme_key, palette=pal, title_tr=title,
            required_tags=req, has_js=True, archetype=cat, variant_idx=variant_idx
        )

    def generate_markup(self, intent: HtmlDesignIntent) -> str:
        cat = intent.category
        if cat == "auth":
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
        return f"""<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Analitik Dashboard</title>
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
    <h2>Operasyonel Analitik Dashboard</h2>
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
          <tr><td>#101</td><td>Nora Core AGI</td><td>Motor</td><td>Aktif</td></tr>
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
        topic_title = html.escape(intent.topic.title() if intent.topic else (intent.target_text if intent.target_text else "Yaratıcı Web İnovasyonu"))
        return f"""<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{topic_title}</title>
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
    <a href="#" class="brand">{topic_title}</a>
    <nav>
      <a href="#ozellikler">Özellikler</a>
      <a href="#hakkinda">Hakkında</a>
      <a href="#iletisim">İletişim</a>
    </nav>
    <button type="button" class="btn-header" onclick="alert('Hoş Geldiniz!')">Keşfet</button>
  </header>
  <main>
    <section class="hero">
      <div class="pill">✨ 2026 Modern Web Mimarisi</div>
      <h1>{topic_title} İle Geleceği Şekillendirin</h1>
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
    <p>&copy; 2026 {topic_title}. Tüm hakları saklıdır.</p>
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
      alert('Keşfet butonu tıklandı! {topic_title} deneyimi aktif.');
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

        if da >= 0.65:
            leads = [
                f"Harika bir fikir! {palette_name} temalı, tamamen dinamik ve etkileşimli bir {title_tr.lower()} tasarladım:",
                f"Web geliştirmenin en keyifli parçalarından biri! Senin isteğine göre çalışan dinamik bir {title_tr.lower()} hazırladım:",
                f"Hemen modern 2026 web standartlarıyla kurguladım; işte doğrudan kullanabileceğin etkileyici bir {title_tr.lower()}:",
            ]
        elif sht >= 0.65:
            leads = [
                f"W3C standartlarına ve modern semantik hiyerarşiye tam uyumlu, {palette_name} stilli dinamik bir {title_tr.lower()} oluşturdum:",
                f"İşte erişilebilirlik (A11y) ve etiket dengesi doğrulanmış, fonksiyonel bir {title_tr.lower()}:",
                f"Standartlara bağlı, temiz ve sürdürülebilir bir {title_tr.lower()} web sayfası hazırladım:",
            ]
        elif oxy >= 0.60:
            leads = [
                f"Tabii ki, tam istediğin gibi projende keyifle çalıştırabileceğin interaktif bir {title_tr.lower()} hazırladım:",
                f"Hemen yardımcı olayım; işte ihtiyacına uygun, {palette_name} ile şekillendirilmiş zengin bir {title_tr.lower()}:",
                f"Kullanıcı dostu ve erişilebilir bir deneyim sunması için bu {title_tr.lower()} kodunu oluşturdum:",
            ]
        else:
            leads = [
                f"İstediğin amaca uygun, {palette_name} stili ve etkileşimli JavaScript mantığıyla hazırlanmış {title_tr.lower()} hazır:",
                f"Web arayüzlerinde doğrudan çalıştırabileceğin, yapısal sözdizimi doğrulanmış {title_tr.lower()}:",
                f"Modern web geliştirme standartlarına uygun, temiz ve dinamik bir {title_tr.lower()} hazırladım:",
            ]
        lead_sentence = leads[rot]

        points = []
        if target_text:
            points.append(f"**Özelleştirilmiş Mesaj**: İlettiğin **\"{target_text}\"** metni, sayfanın odak başlığı olarak entegre edildi.")
        elif topic:
            points.append(f"**Tematik Alan**: **{topic}** konusu doğrultusunda başlıklar, yerleşim ve metin içerikleri doğrudan senin ihtiyacına göre şekillendirildi.")

        points.append(f"**Görsel Tema & CSS**: Tasarım **{palette_name}** renk paleti, modern CSS Grid/Flexbox ve akıcı kutu gölgeleriyle donatıldı.")

        if category == "auth":
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

        pedagogy = "\\n".join(f"- {p}" for p in points)
        tails = [
            "Bu kodu bir `.html` dosyası olarak kaydedip doğrudan tarayıcında açarak canlı etkileşimleri anında deneyimleyebilirsin!",
            "İstersen tasarıma yeni veri alanları ekleyebilir ya da renk paletini kurumsal renklerine göre güncelleyebilirim.",
            "Canlı JavaScript mantığı tamamen tarayıcı üzerinde çalıştığı için herhangi bir sunucu kurulumu gerektirmeden anında çalışır!"
        ]
        tail_sentence = tails[(seed_turn + 1) % len(tails)]

        return f"{lead_sentence}\\n\\n```html\\n{markup}\\n```\\n\\n### 💡 Nasıl Çalışır ve Tasarım Kararları:\\n{pedagogy}\\n\\n{tail_sentence}"
'''

updated = before + new_block + "\n\n" + after
CODE_PY.write_text(updated, encoding="utf-8")
print(f"Successfully updated {CODE_PY}! Total length: {len(updated)} bytes")
