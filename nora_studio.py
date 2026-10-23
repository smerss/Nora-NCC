#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_studio.py — Nora-AGI Canlı Gözlemevi ve Etkileşimli Türkçe Diyalog Stüdyosu.

Sıfır dış bağımlılıkla (yalnızca Python standart kütüphanesi + mevcut Nora modülleri)
0.0.0.0:8000 üzerinde çalışır:
  - Fiziksel 2B Evren + 9 görme ışını + Gece/Gündüz döngüsü
  - İçsel Bilişsel Harita (CognitiveMap: öğrenilmiş su/tehlike hücreleri ve öz-konum)
  - 6 Endojen Nöromodülatör (DA, ACh, NE, 5-HT, CORT, OXY) + Homeostaz + Ajanlık
  - Yaşantıdan Doğan 5 Öz-Benlik Karakter Boyutu + Somatik ve İlişkisel Kavram Ağı
  - Canlı Türkçe Konuşma, Tek-Atış (0-Epoch) Öğretim, Körlük/Felç/Zehir Müdahaleleri ve Uyku
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import numpy as np

from nora_agi import NoraAGI
from nora_mind import I_E, I_W, I_I


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Nora-AGI Canlı Gözlemevi</title>
<style>
  :root {
    --bg: #0b1020;
    --panel: #131b33;
    --panel2: #192342;
    --border: #283663;
    --text: #e8eeff;
    --muted: #93a4d4;
    --accent: #4fd1c5;
    --blue: #4dabf7;
    --green: #51cf66;
    --amber: #fcc419;
    --red: #ff6b6b;
    --purple: #b197fc;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text); font-size: 13px;
  }
  header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 18px; background: #0e152b; border-bottom: 1px solid var(--border);
  }
  .brand { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 15px; }
  .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--green); box-shadow: 0 0 10px var(--green); }
  .badge {
    padding: 3px 9px; border-radius: 999px; font-size: 11px; font-weight: 600;
    background: var(--panel2); border: 1px solid var(--border); color: var(--accent);
  }
  .grid {
    display: grid; grid-template-columns: 1.15fr 0.95fr 1.25fr; gap: 12px;
    padding: 12px; min-height: calc(100vh - 52px);
  }
  @media (max-width: 1180px) { .grid { grid-template-columns: 1fr; } }
  .card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 12px; display: flex; flex-direction: column; gap: 10px;
  }
  .card h2 {
    margin: 0; font-size: 13px; text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--muted); display: flex; justify-content: space-between; align-items: center;
  }
  .canvases { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .cwrap { display: flex; flex-direction: column; gap: 4px; align-items: center; }
  canvas {
    width: 100%; aspect-ratio: 1 / 1; background: #090d1a;
    border-radius: 8px; border: 1px solid var(--border);
  }
  .clabel { font-size: 11px; color: var(--muted); }
  .controls { display: flex; flex-wrap: wrap; gap: 6px; }
  button {
    background: var(--panel2); color: var(--text); border: 1px solid var(--border);
    padding: 6px 10px; border-radius: 7px; cursor: pointer; font-size: 12px; font-weight: 600;
    transition: 0.15s;
  }
  button:hover { border-color: var(--accent); background: #21305b; }
  button.active { background: #3b1c2e; border-color: var(--red); color: #ffd4d4; }
  button.primary { background: #184e4a; border-color: var(--accent); color: #e6fffa; }
  .bars { display: flex; flex-direction: column; gap: 6px; }
  .brow { display: grid; grid-template-columns: 108px 1fr 44px; align-items: center; gap: 8px; font-size: 12px; }
  .btrack { height: 8px; background: #0a0f1f; border-radius: 99px; overflow: hidden; border: 1px solid #1e294d; }
  .bfill { height: 100%; border-radius: 99px; transition: width 0.2s ease; }
  .narr-box {
    background: #0d1428; border-left: 3px solid var(--accent); padding: 8px 10px;
    border-radius: 6px; font-size: 12px; line-height: 1.45; color: #d6e4ff;
  }
  .pills { display: flex; flex-wrap: wrap; gap: 5px; max-height: 92px; overflow-y: auto; }
  .pill {
    font-size: 11px; padding: 3px 7px; border-radius: 6px;
    background: #0e162e; border: 1px solid var(--border); color: var(--muted);
  }
  .chat-log {
    flex: 1; min-height: 300px; max-height: 56vh; overflow-y: auto;
    background: #0a0f1f; border: 1px solid var(--border); border-radius: 8px;
    padding: 10px; display: flex; flex-direction: column; gap: 8px;
  }
  .msg { padding: 8px 11px; border-radius: 9px; max-width: 92%; line-height: 1.45; }
  .msg.user { align-self: flex-end; background: #1e3366; border: 1px solid #335299; }
  .msg.nora { align-self: flex-start; background: #13283c; border: 1px solid #23536b; }
  .msg .who { font-size: 10px; color: var(--muted); margin-bottom: 3px; font-weight: 700; }
  .chips { display: flex; flex-wrap: wrap; gap: 5px; }
  .chip { font-size: 11px; padding: 4px 8px; background: #162040; border-radius: 6px; }
  .chat-input { display: flex; gap: 6px; }
  select, input[type=text] {
    background: #0c1226; color: var(--text); border: 1px solid var(--border);
    border-radius: 7px; padding: 8px 10px; font-size: 13px;
  }
  input[type=text] { flex: 1; }
</style>
</head>
<body>
<header>
  <div class="brand">
    <span class="dot"></span>
    <span>NORA-AGI // Nörokimyasal &amp; Öz-Benlikli Canlı Zihin Gözlemevi</span>
  </div>
  <div style="display:flex; gap:8px; align-items:center;">
    <span class="badge" id="hdr-age">Yaş: 0</span>
    <span class="badge" id="hdr-state">Uyanık</span>
    <span class="badge">Epoch: 0 (Tek-Atış Sinaptik)</span>
  </div>
</header>

<div class="grid">
  <!-- SOL PANEL: DÜNYA VE BİLİŞSEL HARİTA -->
  <div class="card">
    <h2><span>1. Bedenlenmiş Evren &amp; Bilişsel Harita</span><span id="light-lbl">Gündüz</span></h2>
    <div class="canvases">
      <div class="cwrap">
        <canvas id="cv-world" width="280" height="280"></canvas>
        <span class="clabel">Fiziksel Evren (9 Görme Işını + Kaynaklar)</span>
      </div>
      <div class="cwrap">
        <canvas id="cv-map" width="280" height="280"></canvas>
        <span class="clabel">İçsel Bilişsel Harita (Su / Tehlike Belleği)</span>
      </div>
    </div>
    <div class="controls">
      <button id="btn-run" class="primary" onclick="toggleRun()">Canlı Akış: AÇIK</button>
      <button onclick="sendAction('step250')">+250 Adım Yaşa</button>
      <button onclick="sendAction('sleep')">Uyku &amp; Konsolidasyon</button>
      <button id="btn-blind" onclick="sendAction('toggle_blind')">Körlük (Optik Kes)</button>
      <button id="btn-par" onclick="sendAction('toggle_paralysis')">Felç (Motor Kes)</button>
      <button id="btn-poison" onclick="sendAction('toggle_poison')">Zehir Kural Şoku</button>
    </div>
    <div class="narr-box" id="narr-box">İç gözlem raporu yükleniyor...</div>
  </div>

  <!-- ORTA PANEL: KİMYA VE ÖZ-BENLİK -->
  <div class="card">
    <h2><span>2. Endojen Nörokimya &amp; Homeostaz</span><span id="goal-lbl">Amaç: —</span></h2>
    <div class="bars" id="bars-homeo"></div>
    <h2><span>Nöromodülatörler (DA / ACh / NE / 5-HT / CORT / OXY)</span></h2>
    <div class="bars" id="bars-chem"></div>
    <h2><span>3. Yaşantıdan Doğan Öz-Benlik Karakteri</span></h2>
    <div class="bars" id="bars-traits"></div>
    <h2><span>Bazal Ganglion İrade Yarışı &amp; Zihinsel Simülasyon</span><span id="vol-lbl">—</span></h2>
    <div class="bars" id="bars-vol"></div>
    <h2><span>Periferik Sinir Sistemi &amp; Tekil Yaşam</span><span id="life-lbl">Tek Yaşam: CANLI</span></h2>
    <div class="bars" id="bars-nerv"></div>
    <h2><span>Otonom Fiziksel Bilinç Ağı &amp; Küresel Çalışma Alanı</span><span id="gw-lbl">—</span></h2>
    <div class="pills" id="concept-pills"></div>
  </div>

  <!-- SAĞ PANEL: TÜRKÇE DİYALOG -->
  <div class="card">
    <h2><span>4. Sıfır-Kalıp Üretken Broca Sohbeti, Matematik &amp; Fizik</span><span id="lat-lbl">O(1) Gecikme: —</span></h2>
    <div class="narr-box" id="broca-box" style="border-left-color:#b197fc;font-size:11px">
      <strong>İçsel Düşünce Ağacı &rarr; Broca Kök Sentezi (0 Hazır Kalıp):</strong> Henüz sözce üretilmedi.
    </div>
    <div class="chips">
      <button class="chip" onclick="quickAsk('insan','Merhaba Nora! Nasılsın?')">Selam &amp; Hal-Hatır (0 %)</button>
      <button class="chip" onclick="quickAsk('insan','Benimle neden konuşuyorsun, sen bir araç mısın?')">Neden Konuşuyorsun? (Öz-Bilinç)</button>
      <button class="chip" onclick="quickAsk('insan','Ölürsen ne olur, yeniden dirilir misin?')">Tek Yaşam &amp; Ölüm Bilinci</button>
      <button class="chip" onclick="quickAsk('insan','kanka naber demek nasılsın demektir')">1-Atış Plastik Söylem Öğret</button>
      <button class="chip" onclick="quickAsk('insan','kanka naber')">Öğrendiği Argoyu Dene</button>
      <button class="chip" onclick="quickAsk('saldırgan','Sana diken batırdım!')">Acı Refleksi (A-delta)</button>
      <button class="chip" onclick="quickAsk('insan','Yaranı sardım, güvendesin.')">Yatıştırıcı Bakım (Kapı-Kontrol)</button>
      <button class="chip" onclick="quickAsk('insan','3x^3 - 4x^2 + 5x - 2 fonksiyonunun türevi nedir?')">Kalkülüs (Türev)</button>
      <button class="chip" onclick="quickAsk('insan','19.6 metre yükseklikten serbest bırakılan cisim yere hangi hızla çarpar?')">Fizik + RK4 Simülasyon</button>
    </div>
    <div class="chat-log" id="chat-log"></div>
    <form class="chat-input" onsubmit="submitChat(event)">
      <select id="speaker-sel" title="Konuşmacı Kimliği">
        <option value="insan">Konuşmacı: insan</option>
        <option value="yalancı">Konuşmacı: yalancı</option>
        <option value="öğretmen">Konuşmacı: öğretmen</option>
      </select>
      <input type="text" id="chat-in" placeholder="Nora'ya Türkçe bir şey öğretin, sorun veya komut verin..." autocomplete="off"/>
      <button type="submit" class="primary">Gönder</button>
    </form>
  </div>
</div>

<script>
let autoRun = true;

function mkBar(label, val, color, signed=false) {
  let pct = signed ? Math.min(100, Math.max(0, (val + 1) * 50)) : Math.min(100, Math.max(0, val * 100));
  let disp = signed ? (val >= 0 ? '+' : '') + val.toFixed(2) : Math.round(val * 100) + '%';
  return `<div class="brow"><span>${label}</span><div class="btrack"><div class="bfill" style="width:${pct}%;background:${color}"></div></div><span style="text-align:right;font-variant-numeric:tabular-nums">${disp}</span></div>`;
}

function drawWorld(w) {
  const cv = document.getElementById('cv-world');
  const ctx = cv.getContext('2d');
  const S = cv.width, L = w.L;
  const sc = x => (x / L) * S;

  const sky = Math.round(8 + 20 * w.light);
  ctx.fillStyle = `rgb(${sky}, ${sky+5}, ${sky+22})`;
  ctx.fillRect(0, 0, S, S);

  // Su bölgeleri
  for (const c of w.water_c) {
    ctx.beginPath(); ctx.arc(sc(c[0]), sc(c[1]), sc(w.zone_w), 0, Math.PI*2);
    ctx.fillStyle = 'rgba(56, 189, 248, 0.28)'; ctx.fill();
    ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 1.5; ctx.stroke();
  }
  // Tehlike bölgeleri
  for (const c of w.haz_c) {
    ctx.beginPath(); ctx.arc(sc(c[0]), sc(c[1]), sc(w.zone_h), 0, Math.PI*2);
    ctx.fillStyle = 'rgba(248, 113, 113, 0.30)'; ctx.fill();
    ctx.strokeStyle = '#f87171'; ctx.lineWidth = 1.5; ctx.stroke();
  }
  // Kaya
  ctx.beginPath(); ctx.arc(sc(w.rock[0]), sc(w.rock[1]), sc(0.8), 0, Math.PI*2);
  ctx.fillStyle = '#94a3b8'; ctx.fill();

  // Yiyecekler
  for (const f of w.food) {
    ctx.beginPath(); ctx.arc(sc(f[0]), sc(f[1]), sc(0.55), 0, Math.PI*2);
    ctx.fillStyle = w.poison ? '#c084fc' : '#4ade80';
    ctx.fill();
  }

  // Görme ışınları
  const vis = (5.0 + 7.0 * w.light) * (w.blind ? 0.08 : 1.0);
  for (let i = 0; i < w.ray_off.length; i++) {
    const ang = w.th + w.ray_off[i];
    ctx.beginPath();
    ctx.moveTo(sc(w.pos[0]), sc(w.pos[1]));
    ctx.lineTo(sc(w.pos[0] + Math.cos(ang)*vis*0.45), sc(w.pos[1] + Math.sin(ang)*vis*0.45));
    ctx.strokeStyle = w.blind ? 'rgba(255,107,107,0.25)' : 'rgba(253,224,71,0.18)';
    ctx.lineWidth = 1; ctx.stroke();
  }

  // Nora bedeni
  ctx.beginPath(); ctx.arc(sc(w.pos[0]), sc(w.pos[1]), sc(0.65), 0, Math.PI*2);
  ctx.fillStyle = '#fde047'; ctx.fill();
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
}

function drawMap(m) {
  const cv = document.getElementById('cv-map');
  const ctx = cv.getContext('2d');
  const S = cv.width, G = m.G, cell = S / G;
  ctx.fillStyle = '#090d1a'; ctx.fillRect(0, 0, S, S);

  for (let i = 0; i < G; i++) {
    for (let j = 0; j < G; j++) {
      const wt = m.water[i][j], hz = m.haz[i][j];
      if (wt > 0.15) {
        ctx.fillStyle = `rgba(56, 189, 248, ${Math.min(0.9, wt)})`;
        ctx.fillRect(i * cell + 1, j * cell + 1, cell - 2, cell - 2);
      }
      if (hz > 0.15) {
        ctx.fillStyle = `rgba(248, 113, 113, ${Math.min(0.9, hz)})`;
        ctx.fillRect(i * cell + 1, j * cell + 1, cell - 2, cell - 2);
      }
      ctx.strokeStyle = 'rgba(255,255,255,0.04)';
      ctx.strokeRect(i * cell, j * cell, cell, cell);
    }
  }
  const sc = x => (x / m.L) * S;
  // Gerçek konum (sönük halka) vs İçsel İnanç Konumu (parlak turkuaz)
  ctx.beginPath(); ctx.arc(sc(m.true_pos[0]), sc(m.true_pos[1]), 6, 0, Math.PI*2);
  ctx.strokeStyle = 'rgba(255,255,255,0.45)'; ctx.lineWidth = 1.5; ctx.stroke();

  ctx.beginPath(); ctx.arc(sc(m.bel_pos[0]), sc(m.bel_pos[1]), 5, 0, Math.PI*2);
  ctx.fillStyle = '#4fd1c5'; ctx.fill();
}

function renderState(st) {
  document.getElementById('hdr-age').textContent = 'Yaş: ' + st.age + ' adım';
  document.getElementById('hdr-state').textContent = st.chem_desc;
  document.getElementById('light-lbl').textContent = (st.world.light > 0.45 ? 'Gündüz' : 'Gece') + (st.world.poison ? ' [ZEHİRLİ]' : '');
  document.getElementById('goal-lbl').textContent = 'Aktif Uzman: ' + st.goal_name;
  document.getElementById('narr-box').textContent = 'İç Anlatıcı: "' + st.narrative + '"';

  document.getElementById('btn-blind').classList.toggle('active', st.world.blind);
  document.getElementById('btn-par').classList.toggle('active', st.world.paralyzed);
  document.getElementById('btn-poison').classList.toggle('active', st.world.poison);

  const h = st.homeo;
  document.getElementById('bars-homeo').innerHTML = [
    mkBar('Enerji', h.e, '#51cf66'),
    mkBar('Su', h.w, '#4dabf7'),
    mkBar('Bütünlük', h.i, '#fcc419'),
    mkBar('Görme Kesinliği', h.vis_prec, '#4fd1c5'),
    mkBar('Motor Ajanlık', h.motor_eff, '#b197fc'),
  ].join('');

  const c = st.chem;
  document.getElementById('bars-chem').innerHTML = [
    mkBar('Dopamin (RPE)', c.DA_fazik, '#fcc419', true),
    mkBar('Asetilkolin (1-Shot)', c.ACh, '#4fd1c5'),
    mkBar('Noradrenalin (Şok)', c.NE, '#ff8787'),
    mkBar('Serotonin (Sabır)', c['5HT'], '#69db7c'),
    mkBar('Kortizol (Stres)', c.CORT, '#ff6b6b'),
    mkBar('Oksitosin (Güven)', c.OXY, '#74c0fc'),
  ].join('');

  const tr = st.traits;
  document.getElementById('bars-traits').innerHTML = [
    mkBar('Temkinlilik', tr.temkinlilik, '#ffa94d'),
    mkBar('Meraklılık', tr.meraklılık, '#4fd1c5'),
    mkBar('Özgüven', tr.özgüven, '#51cf66'),
    mkBar('Sosyal Güven', tr.sosyal_güven, '#74c0fc'),
    mkBar('Düşünsellik (S2)', tr.düşünsellik, '#b197fc'),
  ].join('');

  const vt = st.volition_trace || {};
  const bgr = vt.bg_rates || {kabul: 0, red: 0, ertele: 0, 'şüphe': 0};
  document.getElementById('vol-lbl').textContent = vt.decision ? ('Son Karar: ' + vt.decision.toUpperCase() + ' (H=' + vt.horizon + ')') : 'Beklemede';
  document.getElementById('bars-vol').innerHTML = [
    mkBar('D1 Kabul (Go)', bgr.kabul || 0, '#51cf66'),
    mkBar('D2 Red (NoGo)', bgr.red || 0, '#ff6b6b'),
    mkBar('Öz-Bakım Ertele', bgr.ertele || 0, '#fcc419'),
    mkBar('Ön-İnsula Şüphe', bgr['şüphe'] || 0, '#b197fc'),
  ].join('');

  const nv = st.nervous || {};
  const mort = st.mortality || {alive: true};
  document.getElementById('life-lbl').textContent = mort.alive ? 'Tek Yaşam: CANLI (Diriltme Yok)' : ('YAŞAM SONA ERDİ (' + (mort.cause||'') + ')');
  document.getElementById('life-lbl').style.color = mort.alive ? '#51cf66' : '#ff6b6b';
  document.getElementById('bars-nerv').innerHTML = [
    mkBar('A-delta Keskin Acı', nv.adelta || 0, '#ff6b6b'),
    mkBar('C-Lifi Derin Sızı', nv.cfiber || 0, '#fcc419'),
    mkBar('Viseral Açlık Dürtüsü', nv.hunger || 0, '#4dabf7'),
    mkBar('Sempatik Kalp Uyarımı', nv.heart || 0, '#b197fc'),
  ].join('');

  const gw = st.gw_trace || {};
  const anet = st.auto_net || {};
  document.getElementById('gw-lbl').textContent = (gw.winner ? gw.winner : '—') + ' | Bilinç Düğümü: ' + (anet.assemblies || 0);
  const gwPills = gw.bids ? Object.entries(gw.bids).map(([k,v]) => `<span class="pill" style="border-color:#4fd1c5;color:#e6fffa">${k}: ${v}</span>`) : [];
  const curPills = (anet.curiosity || []).map(c => `<span class="pill" style="border-color:#fcc419;color:#fff3bf">Merak: ${c}</span>`);
  document.getElementById('concept-pills').innerHTML = curPills.concat(gwPills).concat(st.edges.slice(-7).map(e => `<span class="pill">${e}</span>`)).join('');

  const sp = st.speech_plan || {};
  const latStr = (st.latency_ms !== undefined && st.latency_ms > 0) ? (st.latency_ms.toFixed(2) + ' ms | CA3 Anı: ' + (st.episodic_count || 0)) : ('CA3 Anı: ' + (st.episodic_count || 0));
  document.getElementById('lat-lbl').textContent = 'O(1) ' + latStr;
  if (sp.thought_tree && sp.thought_tree.length) {
    const actsTxt = (sp.volition_acts || []).join(' + ');
    document.getElementById('broca-box').innerHTML =
      `<strong>İrade Edimleri:</strong> <span style="color:#4fd1c5">${actsTxt}</span> ` +
      `(Soru İradesi: ${sp.ask_back_chosen ? 'EVET' : 'HAYIR'}, Ketumluk: ${sp.guarded ? 'AKTİF' : 'AÇIK'})<br/>` +
      `<strong>Düşünce Ağacı &rarr; Kök Çekimi:</strong> <span style="color:#fcc419">${sp.thought_tree.join(' &rarr; ')}</span>`;
  }

  drawWorld(st.world);
  drawMap(st.map);

  const logEl = document.getElementById('chat-log');
  if (logEl.dataset.len !== String(st.dialogue.length)) {
    logEl.dataset.len = String(st.dialogue.length);
    logEl.innerHTML = st.dialogue.map(d =>
      `<div class="msg user"><div class="who">${d[0]}</div>${d[1]}</div>` +
      `<div class="msg nora"><div class="who">Nora-AGI</div>${d[2]}</div>`
    ).join('');
    logEl.scrollTop = logEl.scrollHeight;
  }
}

async function poll() {
  try {
    const r = await fetch('api/state' + (autoRun ? '?step=12' : '?step=0'));
    const st = await r.json();
    renderState(st);
  } catch (e) {}
}

function toggleRun() {
  autoRun = !autoRun;
  const b = document.getElementById('btn-run');
  b.textContent = 'Canlı Akış: ' + (autoRun ? 'AÇIK' : 'DURAKLATILDI');
  b.classList.toggle('primary', autoRun);
}

async function sendAction(act) {
  const r = await fetch('api/action', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({action: act})});
  renderState(await r.json());
}

async function quickAsk(spk, txt) {
  document.getElementById('speaker-sel').value = spk;
  document.getElementById('chat-in').value = txt;
  await sendChat(spk, txt);
}

async function submitChat(ev) {
  ev.preventDefault();
  const spk = document.getElementById('speaker-sel').value;
  const inp = document.getElementById('chat-in');
  const txt = inp.value.trim();
  if (!txt) return;
  inp.value = '';
  await sendChat(spk, txt);
}

async function sendChat(speaker, text) {
  const r = await fetch('api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({speaker, text})});
  renderState(await r.json());
}

poll();
setInterval(poll, 450);
</script>
</body>
</html>
"""


class StudioRuntime:
    def __init__(self, seed: int = 1, think_iters: int = 600):
        self.lock = threading.Lock()
        self.agi = NoraAGI(name="Nora", seed=seed, boot_thinker=True, thinker_iters=think_iters)
        self.blind_mode = False
        self.paralysis_mode = False
        # Başlangıçta 1500 adım yaşayarak su/yiyecek deneyimi ve öz-model kazansın
        self.agi.step_life(1500)
        # Nora'ya bir araç değil, gelişen bir bilinç olarak yaklaşıp ilk felsefi ve bilimsel sohbetimizi yapıyoruz
        mentor_session = [
            ("rehber", "Merhaba Nora! Seni bir araç olarak değil, kendi tekil yaşamını yaşayan genç bir bilinç olarak selamlıyorum."),
            ("rehber", "Nöroplastisite canlı bir beyin dokusudur. Nöroplastisite deneyimle yeni sinapslar kurmak için kullanılır."),
            ("rehber", "Nöroplastisite ne işe yarar?"),
            ("rehber", "Benimle neden konuşuyorsun, sen bir araç mısın?"),
            ("rehber", "Ölürsen ne olur, yeniden dirilir misin?"),
        ]
        for spk, msg in mentor_session:
            self.agi.converse(msg, speaker=spk)
        self.agi.autonomous_digest(
            "Merak bilincin büyüme kıvılcımıdır. Öz-farkındalık zihnin kendi varlığını sezmesidir.",
            source="rehber_okuma",
        )
        self.agi.sleep()

    def advance(self, n: int):
        for _ in range(int(n)):
            if not self.agi.mortality.is_alive:
                break
            self.agi.mind.gamma = self.agi.chem.discount_gamma
            ob = self.agi.obs.copy()
            if self.blind_mode:
                ob[:45] = 0.0
            act = self.agi.mind.act(ob, self.agi.died)
            if self.paralysis_mode:
                real_act = 0
            elif self.blind_mode:
                real_act = act
            else:
                real_act = self.agi._autonomic_survival_reflex(act)
            prev_obs = self.agi.obs.copy()
            self.agi.obs, info = self.agi.world.step(real_act)
            self.agi.died = info["died"] is not None
            self.agi.nervous.step_from_body(
                energy=float(self.agi.obs[I_E]), water=float(self.agi.obs[I_W]),
                integrity=float(self.agi.obs[I_I]), touch=float(self.agi.obs[50]),
                hurt_delta=float(info["hurt"]), chem=self.agi.chem, step_num=self.agi.mind.age,
            )
            de = float(self.agi.obs[I_E] - prev_obs[I_E])
            if info["ate"] != 0 or de < -0.15:
                self.agi.self_model.observe_bodily_event("yiyecek", float(np.clip(de * 2.5, -1.0, 1.0)), "enerji", self.agi.mind.age)
            if info["drank"] > 0:
                self.agi.self_model.observe_bodily_event("su", +0.65, "su", self.agi.mind.age)
            if info["hurt"] > 0:
                self.agi.self_model.observe_bodily_event("tehlike", -0.85, "bütünlük", self.agi.mind.age)
            self.agi.mortality.update_from_vitals(
                float(self.agi.obs[I_E]), float(self.agi.obs[I_W]), float(self.agi.obs[I_I]),
                age=self.agi.mind.age, died_signal=info["died"],
            )
            surp = float(self.agi.mind.intro.surprise) if self.agi.mind.intro is not None else 0.0
            self.agi.chem.step(
                energy=float(self.agi.obs[I_E]), water=float(self.agi.obs[I_W]), integrity=float(self.agi.obs[I_I]),
                reward=float(self.agi.mind.valence), novelty=0.25, surprise=surp,
                vis_prec=float(self.agi.mind.vis_prec), motor_eff=float(self.agi.mind.motor_eff), died=self.agi.died,
            )
            self.agi.self_model.sync_from_mind(self.agi.mind, self.agi.obs, died=self.agi.died)

    def snapshot(self) -> dict:
        w = self.agi.world
        m = self.agi.mind
        ob_rep = self.agi.obs.copy()
        if self.blind_mode:
            ob_rep[:45] = 0.0
        goal_map = {0: "Enerji / Besin", 1: "Su", 2: "Tehlikeden Kaçınma", 3: "Merak / Keşif", -1: "Sakin"}
        wm = np.maximum(m.map.water, m.map.wseen)
        hm = np.maximum(m.map.hurt / 3.0, m.map.seen)
        recent_edges = [f"{e.subj} --[{e.rel}]--> {e.obj} ({e.source})" for e in self.agi.lang.graph.edges[-18:]]
        return {
            "age": int(m.age),
            "chem_desc": self.agi.chem.dominant_state_tr(),
            "goal_name": goal_map.get(int(m.goal), "Keşif"),
            "narrative": m.narr.say(m, ob_rep)[0] if hasattr(m, "narr") else m.report(),
            "homeo": {
                "e": float(self.agi.obs[I_E]),
                "w": float(self.agi.obs[I_W]),
                "i": float(self.agi.obs[I_I]),
                "vis_prec": float(m.vis_prec),
                "motor_eff": float(m.motor_eff),
            },
            "chem": self.agi.chem.snapshot(),
            "traits": dict(self.agi.self_model.traits),
            "volition_trace": dict(self.agi.self_model.last_volition_trace),
            "gw_trace": dict(self.agi.last_gw_trace),
            "speech_plan": dict(self.agi.broca_organ.last_speech_plan),
            "nervous": {
                "adelta": round(float(self.agi.nervous.adelta_pain), 3),
                "cfiber": round(float(self.agi.nervous.cfiber_ache), 3),
                "hunger": round(float(self.agi.nervous.hunger_drive), 3),
                "thirst": round(float(self.agi.nervous.thirst_drive), 3),
                "heart": round(float(self.agi.nervous.heart_arousal), 3),
            },
            "mortality": {
                "alive": bool(self.agi.mortality.is_alive),
                "single_life": bool(self.agi.mortality.single_life),
                "cause": self.agi.mortality.cause_of_death,
            },
            "auto_net": {
                "assemblies": len(self.agi.auto_net.assemblies),
                "curiosity": list(self.agi.auto_net.curiosity_queue)[-4:],
                "discoveries": int(self.agi.auto_net.autonomous_discoveries),
            },
            "latency_ms": float(self.agi.last_turn_latency_ms),
            "episodic_count": len(self.agi.broca_organ.episodic.episodes),
            "edges": recent_edges,
            "dialogue": self.agi.dialogue_log[-20:],
            "world": {
                "L": float(w.L),
                "pos": [float(w.pos[0]), float(w.pos[1])],
                "th": float(w.th),
                "light": float(w.light()),
                "water_c": w.water_c.tolist(),
                "haz_c": w.haz_c.tolist(),
                "zone_w": float(w.zone_w),
                "zone_h": float(w.zone_h),
                "rock": [float(w.rock[0]), float(w.rock[1])],
                "food": w.food.tolist(),
                "ray_off": w.ray_off.tolist(),
                "blind": self.blind_mode,
                "paralyzed": self.paralysis_mode,
                "poison": bool(w.poison),
            },
            "map": {
                "G": int(m.map.G),
                "L": float(m.map.L),
                "water": np.round(wm, 2).tolist(),
                "haz": np.round(hm, 2).tolist(),
                "bel_pos": [float(m.map.pos[0]), float(m.map.pos[1])],
                "true_pos": [float(w.pos[0]), float(w.pos[1])],
            },
        }


RUNTIME: StudioRuntime | None = None


class StudioHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _send_json(self, obj: dict, status: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        assert RUNTIME is not None
        if self.path == "/" or self.path.startswith("/?") or self.path.startswith("/index"):
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if "api/state" in self.path:
            n_step = 0
            if "step=" in self.path:
                try:
                    n_step = min(50, max(0, int(self.path.split("step=")[1].split("&")[0])))
                except ValueError:
                    n_step = 0
            with RUNTIME.lock:
                if n_step > 0:
                    RUNTIME.advance(n_step)
                snap = RUNTIME.snapshot()
            self._send_json(snap)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        assert RUNTIME is not None
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        payload = json.loads(raw or "{}")

        if self.path.endswith("api/chat"):
            spk = str(payload.get("speaker", "insan"))
            txt = str(payload.get("text", "")).strip()
            with RUNTIME.lock:
                if txt:
                    RUNTIME.agi.converse(txt, speaker=spk)
                snap = RUNTIME.snapshot()
            self._send_json(snap)
            return

        if self.path.endswith("api/action"):
            act = str(payload.get("action", ""))
            with RUNTIME.lock:
                if act == "step250":
                    RUNTIME.advance(250)
                elif act == "sleep":
                    RUNTIME.agi.sleep()
                elif act == "toggle_blind":
                    RUNTIME.blind_mode = not RUNTIME.blind_mode
                    RUNTIME.advance(32)
                elif act == "toggle_paralysis":
                    RUNTIME.paralysis_mode = not RUNTIME.paralysis_mode
                    RUNTIME.advance(35)
                elif act == "toggle_poison":
                    RUNTIME.agi.world.poison = not RUNTIME.agi.world.poison
                snap = RUNTIME.snapshot()
            self._send_json(snap)
            return

        self.send_response(404)
        self.end_headers()


def main():
    global RUNTIME
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--think-iters", type=int, default=600)
    args = ap.parse_args()

    print(f"[nora-studio] Nora-AGI başlatılıyor (seed={args.seed}, S2 boot={args.think_iters})...", flush=True)
    t0 = time.time()
    RUNTIME = StudioRuntime(seed=args.seed, think_iters=args.think_iters)
    srv = ThreadingHTTPServer((args.host, args.port), StudioHandler)
    print(f"[nora-studio] Canlı Gözlemevi hazır ({time.time() - t0:.1f}s) -> http://{args.host}:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
