#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import html

def build_portfolio_test(author_name="Offihito", role_desc="16 yaşında Web Developer", p=None):
    if not p:
        p = {
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
        }
    
    author = html.escape(author_name if author_name else "Geliştirici")
    role = html.escape(role_desc if role_desc else "Web Developer")
    initial = author[0].upper() if author else "D"

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
      color: var(--bg);
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
      <p>Modern web standartları, dinamik JavaScript uygulamaları ve temiz arayüz mimarileri geliştiriyorum. Kullanıcı odaklı, performanslı ve estetik dijital deneyimler inşa ediyorum.</p>
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

from nora.synthesis.code import HtmlStructuralVerifier

v = HtmlStructuralVerifier()
html_out = build_portfolio_test("Offihito", "16 yaşında Web Developer")
res = v.verify(html_out, ["html", "head", "body", "header", "main", "footer", "button", "script"])
print("Portfolio Verification:", res["ok"], "Total tags:", res["total_tags"], "Errors:", res.get("errors"))
