#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_sayfa.py — SAYFA KORTEKSİ: etkileşimli web sayfasını örnekten öğren, birleştir, tıklayarak doğrula.

Bu organda hazır sayfa, düğme, betik ya da "sayaç" kalıbı YOKTUR. Nora bir sayfayı ancak
kendisine öğretilmiş örneklerin parçalarından kurabilir ve kurduğu sayfayı gerçekten
çalıştırıp düğmelerine basarak doğrulamadan dışarı vermez.

Öğretim (tek sunum, epoch yok)
------------------------------
Bir örnek = Türkçe bir açıklama + bir HTML sayfası. Nora örneği okurken:
  1. sayfayı node altında küçük bir DOM kabuğunda GERÇEKTEN çalıştırır, her düğmeye basar
     ve görünen sayının nasıl değiştiğini GÖZLER (anlam = gözlenen etki, yazılı tanım değil);
  2. gözlenen etkiyi açıklamadaki sözcüklerle hizalar: hangi fiil hangi etkiyi anlatıyor,
     büyüklük metindeki sayıya mı bağlı yoksa sabit mi;
  3. düğmenin kodunu, durum değişkeni / ekran kimliği / büyüklük delikleri açık bir PARÇA
     olarak saklar; sayfanın geri kalanı (baş, stil, ekran) iskelet değil, o örneğin kendisidir.

Genelleme
---------
Sözcük → etki bağı çapraz-durumsaldır: bir sözcük farklı etkilerle birlikte görüldükçe
tutarlılığı düşer ('sayıyı' her örnekte geçer, hiçbir etkiyi anlatmaz). Yeni bir istekte her
yan cümle öğrenilmiş bir etkiye bağlanır, büyüklük metinden alınır, farklı örneklerden gelen
parçalar tek bir durum değişkeni ve ekran üzerinde yeniden adlandırılarak birleştirilir.

İnsan yazımı önseller (prior)
-----------------------------
HTML sözdizimi (ayrıştırma/yazma), JS deyim sınırları, tarayıcı yerine geçen DOM kabuğu ve
sayısal etki hipotez uzayı (ekle / çarp / ata). Bunlar dilin ve ölçümün biçimidir; hangi
sözcüğün ne yaptığı, hangi kodun hangi etkiyi doğurduğu ve sayfanın nasıl göründüğü öğrenilir.
"""
from __future__ import annotations

from nora._paths import REPO_ROOT


import collections
import copy
import html
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import threading
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

from nora_broca import NP, VerbSpec, TurkishSyntaxEngine
from nora_lang import TurkishMorphology, tr_capitalize, tr_lower


# ============================================================================
# 1. HTML AĞACI (prior: sözdizimi)
# ============================================================================

VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
             "param", "source", "track", "wbr"}
RAW_TAGS = {"style"}


@dataclass
class Page:
    """Ayrıştırılmış sayfa. Betikler ağaçtan ayrı tutulur; kabukta tek bir bağlamda koşar."""
    nodes: list[dict] = field(default_factory=list)
    roots: list[int] = field(default_factory=list)
    scripts: list[str] = field(default_factory=list)
    doctype: str = ""

    def text(self, i: int) -> str:
        return "".join(c for c in self.nodes[i]["content"] if isinstance(c, str)).strip()

    def children(self, i: int) -> list[int]:
        return [c for c in self.nodes[i]["content"] if isinstance(c, int)]

    def find_id(self, ident: str) -> int | None:
        for i, n in enumerate(self.nodes):
            if n.get("live", True) and n["attrs"].get("id") == ident:
                return i
        return None

    def find_tag(self, tag: str) -> int | None:
        for i, n in enumerate(self.nodes):
            if n.get("live", True) and n["tag"] == tag:
                return i
        return None

    def script(self) -> str:
        return "\n;\n".join(self.scripts)

    def harness_payload(self) -> dict:
        return {
            "nodes": [{"tag": n["tag"], "attrs": n["attrs"], "text": self.text(i),
                       "children": self.children(i)} for i, n in enumerate(self.nodes)],
            "roots": list(self.roots),
            "script": self.script(),
        }


class _TreeBuilder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.page = Page()
        self.stack: list[int] = []
        self._script: list[str] | None = None

    def handle_decl(self, decl):
        self.page.doctype = decl

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self._script = []
            return
        nodes = self.page.nodes
        idx = len(nodes)
        parent = self.stack[-1] if self.stack else None
        nodes.append({"tag": tag, "attrs": {k: ("" if v is None else v) for k, v in attrs},
                      "content": [], "parent": parent})
        if parent is None:
            self.page.roots.append(idx)
        else:
            nodes[parent]["content"].append(idx)
        if tag not in VOID_TAGS:
            self.stack.append(idx)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag == "script":
            self.handle_endtag(tag)
        elif tag not in VOID_TAGS and self.stack:
            self.stack.pop()

    def handle_endtag(self, tag):
        if tag == "script":
            if self._script is not None:
                self.page.scripts.append("".join(self._script))
            self._script = None
            return
        for pos in range(len(self.stack) - 1, -1, -1):
            if self.page.nodes[self.stack[pos]]["tag"] == tag:
                del self.stack[pos:]
                break

    def handle_data(self, data):
        if self._script is not None:
            self._script.append(data)
        elif self.stack:
            self.page.nodes[self.stack[-1]]["content"].append(data)


def parse_page(markup: str) -> Page:
    builder = _TreeBuilder()
    builder.feed(markup)
    builder.close()
    return builder.page


def _attr(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def serialize(page: Page) -> str:
    def emit(i: int) -> str:
        n = page.nodes[i]
        tag = n["tag"]
        attrs = "".join(f' {k}="{_attr(v)}"' if v != "" else f" {k}" for k, v in n["attrs"].items())
        if tag in VOID_TAGS:
            return f"<{tag}{attrs}>"
        inner = []
        for c in n["content"]:
            if isinstance(c, int):
                if page.nodes[c].get("live", True):
                    inner.append(emit(c))
            elif isinstance(c, tuple):
                inner.append(c[1])
            else:
                inner.append(c if tag in RAW_TAGS else html.escape(c, quote=False))
        return f"<{tag}{attrs}>{''.join(inner)}</{tag}>"

    head = f"<!{page.doctype}>\n" if page.doctype else ""
    return head + "\n".join(emit(i) for i in page.roots if page.nodes[i].get("live", True)) + "\n"


# ============================================================================
# 2. JS DEYİMLERİ (prior: deyim sınırı, bildirim, dinleyici bağlama)
# ============================================================================

_IDENT = r"[A-Za-z_$][\w$]*"
_CONTINUES = tuple("=+-*/,(.&|?:![{<>")


def js_statements(src: str) -> list[str]:
    """Üst düzey JS deyimleri: ';' ya da derinlik-0 satır sonu (otomatik noktalı virgül yaklaşımı)."""
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    quote = ""
    i = 0

    def flush():
        stmt = "".join(buf).strip().rstrip(";").strip()
        if stmt:
            out.append(stmt)
        buf.clear()

    while i < len(src):
        ch = src[i]
        if quote:
            buf.append(ch)
            if ch == "\\" and i + 1 < len(src):
                buf.append(src[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = ""
            i += 1
            continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = len(src) if j < 0 else j
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            i = len(src) if j < 0 else j + 2
            continue
        if ch in "'\"`":
            quote = ch
        elif ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        if depth == 0 and ch == ";":
            flush()
            i += 1
            continue
        if depth == 0 and ch == "\n":
            done = "".join(buf).strip()
            rest = src[i + 1:].lstrip()
            if done and not done.endswith(_CONTINUES) and not rest.startswith(
                    (".", "+", "-", "*", "/", ",", "?", ":", ")", "&", "|", "=")):
                flush()
                i += 1
                continue
        buf.append(ch)
        i += 1
    flush()
    return out


def declared_names(stmt: str) -> set[str]:
    names = set()
    m = re.match(rf"\s*(?:let|const|var)\s+({_IDENT})", stmt)
    if m:
        names.add(m.group(1))
    m = re.match(rf"\s*(?:async\s+)?function\s+({_IDENT})", stmt)
    if m:
        names.add(m.group(1))
    return names


def _tidy_statement(stmt: str) -> str:
    """Örnekten koparılan deyimin iç girintisini düzeltir; blok değilse ';' ile kapatır."""
    lines = stmt.splitlines()
    if len(lines) > 1:
        body = textwrap.dedent("\n".join(lines[1:-1]))
        lines = [lines[0]] + ["  " + line if line.strip() else line for line in body.splitlines()] \
            + [lines[-1].strip()]
    text = "\n".join(lines)
    return text if text.endswith("}") else text + ";"


def attaches_listener(stmt: str) -> bool:
    return "addEventListener(" in stmt or bool(re.search(r"\.on[a-z]+\s*=(?!=)", stmt))


def _strip_strings(code: str) -> str:
    return re.sub(r"(['\"`])(?:\\.|(?!\1).)*\1", "''", code)


def identifiers(code: str) -> set[str]:
    return set(re.findall(rf"(?<![\w$.]){_IDENT}", _strip_strings(code)))


def assigned_names(code: str) -> set[str]:
    bare = _strip_strings(code)
    return (set(re.findall(rf"(?<![\w$.])({_IDENT})\s*(?:[-+*/%]?=(?!=)|\+\+|--)", bare))
            | set(re.findall(rf"(?:\+\+|--)({_IDENT})", bare)))


def rename_identifier(code: str, old: str, new: str) -> str:
    if old == new:
        return code
    return re.sub(rf"(?<![\w$.]){re.escape(old)}(?![\w$])", new, code)


def rename_element_id(code: str, old: str, new: str) -> str:
    if not old or old == new:
        return code
    code = re.sub(rf"(['\"`])#?{re.escape(old)}\1",
                  lambda m: m.group(0).replace(old, new), code)
    return code


def function_body(src: str) -> str:
    s = src.strip()
    brace = s.find("{")
    head = s if brace < 0 else s[:brace]
    if "=>" in head:
        after = s[s.find("=>") + 2:].strip()
        return after[1:-1].strip() if after.startswith("{") and after.endswith("}") else after
    if brace >= 0 and s.endswith("}"):
        return s[brace + 1:-1].strip()
    return s


# ============================================================================
# 3. DOM KABUĞU (prior: tarayıcının yerine geçen ölçüm aleti; JS gerçekten koşar)
# ============================================================================

_SHIM = r"""
const vm = require('vm');
let data = null;
class El {
  constructor(tag, attrs, doc) {
    this.tagName = String(tag).toUpperCase(); this.attrs = Object.assign({}, attrs || {});
    this.children = []; this.parentNode = null; this._text = ''; this.listeners = {};
    this.value = this.attrs.value || ''; this.style = {}; this.onclick = null; this.dataset = {};
    this.ownerDocument = doc; this.disabled = false; this.checked = false;
    const cls = new Set(String(this.attrs['class'] || '').split(/\s+/).filter(Boolean));
    this.classList = { add: (...c) => c.forEach(x => cls.add(x)), remove: (...c) => c.forEach(x => cls.delete(x)),
      toggle: c => cls.has(c) ? (cls.delete(c), false) : (cls.add(c), true), contains: c => cls.has(c) };
  }
  get id() { return this.attrs.id || ''; } set id(v) { this.attrs.id = String(v); }
  get className() { return this.attrs['class'] || ''; } set className(v) { this.attrs['class'] = String(v); }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
  set textContent(v) { this.children = []; this._text = String(v); }
  get innerText() { return this.textContent; } set innerText(v) { this.textContent = v; }
  get innerHTML() { return this.textContent; }
  set innerHTML(v) { this.children = []; this._text = String(v).replace(/<[^>]*>/g, ''); }
  get firstChild() { return this.children[0] || null; }
  get lastChild() { return this.children[this.children.length - 1] || null; }
  appendChild(c) { if (c.parentNode) c.parentNode.removeChild(c); c.parentNode = this; this.children.push(c); return c; }
  append(...cs) { cs.forEach(c => typeof c === 'string' ? (this._text += c) : this.appendChild(c)); }
  prepend(c) { c.parentNode = this; this.children.unshift(c); }
  removeChild(c) { this.children = this.children.filter(x => x !== c); c.parentNode = null; return c; }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  setAttribute(k, v) { this.attrs[k] = String(v); } getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  removeAttribute(k) { delete this.attrs[k]; }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  removeEventListener(t, f) { this.listeners[t] = (this.listeners[t] || []).filter(x => x !== f); }
  walk(fn) { fn(this); this.children.forEach(c => c.walk(fn)); }
  matches(sel) {
    sel = String(sel).trim();
    if (sel.startsWith('#')) return this.id === sel.slice(1);
    if (sel.startsWith('.')) return this.classList.contains(sel.slice(1));
    return this.tagName === sel.toUpperCase();
  }
  querySelector(sel) { let r = null; this.children.forEach(c => c.walk(e => { if (!r && e.matches(sel)) r = e; })); return r; }
  querySelectorAll(sel) { const r = []; this.children.forEach(c => c.walk(e => { if (e.matches(sel)) r.push(e); })); return r; }
  getElementsByTagName(t) { return this.querySelectorAll(t); }
  click() { fire(this, this.ownerDocument._page); }
  focus() {} blur() {}
}
function build() {
  const doc = {};
  const root = new El('#document', {}, doc);
  const els = data.nodes.map(n => new El(n.tag, n.attrs, doc));
  data.nodes.forEach((n, i) => { els[i]._text = n.text;
    n.children.forEach(j => { els[j].parentNode = els[i]; els[i].children.push(els[j]); }); });
  data.roots.forEach(i => { els[i].parentNode = root; root.children.push(els[i]); });
  doc.documentElement = root;
  doc.body = root.querySelector('body') || root; doc.head = root.querySelector('head') || root;
  doc.getElementById = id => { let r = null; root.walk(e => { if (!r && e.attrs.id === id) r = e; }); return r; };
  doc.querySelector = s => root.querySelector(s); doc.querySelectorAll = s => root.querySelectorAll(s);
  doc.getElementsByTagName = t => root.querySelectorAll(t);
  doc.getElementsByClassName = c => root.querySelectorAll('.' + c);
  doc.createElement = t => new El(t, {}, doc);
  doc.createTextNode = t => { const e = new El('#text', {}, doc); e._text = String(t); return e; };
  doc.listeners = {}; doc.addEventListener = (t, f) => (doc.listeners[t] = doc.listeners[t] || []).push(f);
  const page = { els, errors: [], alerts: [] };
  const sandbox = { document: doc, console: { log() {}, error() {}, warn() {}, info() {} },
    alert: m => page.alerts.push(String(m)), setTimeout: () => 0, clearTimeout() {}, setInterval: () => 0,
    clearInterval() {}, parseInt, parseFloat, Number, String, Math, JSON, isNaN, isFinite, Array, Object, Date,
    listeners: {} };
  sandbox.window = sandbox;
  sandbox.addEventListener = (t, f) => (sandbox.listeners[t] = sandbox.listeners[t] || []).push(f);
  page.ctx = vm.createContext(sandbox); doc._page = page;
  try { vm.runInContext(data.script, page.ctx, { timeout: 2500 }); } catch (e) { page.errors.push('script: ' + e.message); }
  const ready = (doc.listeners['DOMContentLoaded'] || []).concat(sandbox.listeners['DOMContentLoaded'] || [],
                                                                 sandbox.listeners['load'] || []);
  for (const f of ready) call(page, f, doc, { type: 'load' }, 'load');
  if (typeof sandbox.onload === 'function') call(page, sandbox.onload, sandbox, { type: 'load' }, 'onload');
  return page;
}
// Sayfa kodu yalnız zaman aşımlı runInContext içinde çağrılır: sonsuz döngü ölçümü kilitlemez.
function call(page, fn, self, ev, tag) {
  page.ctx.__fn = fn; page.ctx.__self = self; page.ctx.__ev = ev;
  try { vm.runInContext('__fn.call(__self, __ev)', page.ctx, { timeout: 2500 }); }
  catch (e) { page.errors.push(tag + ': ' + e.message); }
}
function fire(el, page) {
  const ev = { type: 'click', target: el, currentTarget: el, preventDefault() {}, stopPropagation() {} };
  if (el.attrs.onclick) {
    let fn = null;
    try { fn = vm.runInContext('(function(event){\n' + el.attrs.onclick + '\n})', page.ctx, { timeout: 2500 }); }
    catch (e) { page.errors.push('onclick: ' + e.message); }
    if (fn) call(page, fn, el, ev, 'onclick');
  }
  if (typeof el.onclick === 'function') call(page, el.onclick, el, ev, 'onclick');
  for (const f of (el.listeners.click || [])) call(page, f, el, ev, 'click');
}
const snap = p => p.els.map(e => [e._text, e.textContent, String(e.value)]);
function measure(request) {
  data = request;
  const out = { handlers: [], runs: [] };
  { const p = build();
    out.handlers = p.els.map(e => ({ inline: e.attrs.onclick || null,
      prop: typeof e.onclick === 'function' ? e.onclick.toString() : null,
      listeners: (e.listeners.click || []).map(f => f.toString()) }));
    out.load_errors = p.errors; }
  for (const seq of data.actions) {
    const p = build(); const snaps = [snap(p)];
    for (const i of seq) { fire(p.els[i], p); snaps.push(snap(p)); }
    out.runs.push({ snaps, errors: p.errors, alerts: p.alerts });
  }
  return out;
}
require('readline').createInterface({ input: process.stdin }).on('line', line => {
  let out;
  try { out = measure(JSON.parse(line)); } catch (e) { out = { error: String((e && e.message) || e) }; }
  process.stdout.write(JSON.stringify(out) + '\n');
});
"""


class DomHarness:
    """Sayfayı node altında koşturur, verilen düğme dizilerine basar ve her adımda görüneni kaydeder."""

    # Süreç başına tek kalıcı node işçisi: her ölçümde node başlatmak saniyeler sürüyor.
    _shim_path = ""
    _worker: subprocess.Popen | None = None
    _lock = threading.Lock()

    def __init__(self):
        self.node = shutil.which("node")

    @property
    def available(self) -> bool:
        return bool(self.node)

    @classmethod
    def _shim(cls) -> str:
        if not cls._shim_path or not os.path.isfile(cls._shim_path):
            fd, path = tempfile.mkstemp(prefix="nora-dom-", suffix=".js")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(_SHIM)
            DomHarness._shim_path = path
        return cls._shim_path

    def run(self, page: Page, actions: list[list[int]]) -> dict | None:
        """Bir kez yeniden dener: alt süreç ara sıra geçici bir hatayla düşebilir."""
        out = self._run_once(page, actions)
        if out is not None and out.get("error"):
            out = self._run_once(page, actions)
        return out

    def _process(self) -> subprocess.Popen:
        worker = DomHarness._worker
        if worker is None or worker.poll() is not None:
            worker = subprocess.Popen([self.node, self._shim()], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            DomHarness._worker = worker
        return worker

    @staticmethod
    def _drop(worker: subprocess.Popen) -> None:
        DomHarness._worker = None
        try:
            worker.kill()
        except OSError:
            pass

    def _run_once(self, page: Page, actions: list[list[int]], timeout: float = 30.0) -> dict | None:
        if not self.node:
            return None
        payload = json.dumps({**page.harness_payload(), "actions": actions}, ensure_ascii=False)
        with DomHarness._lock:
            worker = self._process()
            box: list[bytes] = []
            try:
                worker.stdin.write(payload.encode("utf-8") + b"\n")
                worker.stdin.flush()
            except OSError as exc:
                self._drop(worker)
                return {"error": str(exc)}
            reader = threading.Thread(target=lambda: box.append(worker.stdout.readline()), daemon=True)
            reader.start()
            reader.join(timeout)
            if not box or not box[0]:
                self._drop(worker)
                return {"error": "kabuk_yanit_yok"}
        try:
            return json.loads(box[0].decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return {"error": str(exc)}


# ============================================================================
# 4. ETKİ GÖZLEMİ (prior: ekle / çarp / ata hipotez uzayı)
# ============================================================================

_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


def _first_number(text: str) -> float | None:
    m = _NUM.search(text or "")
    return float(m.group(0).replace(",", ".")) if m else None


def fmt_num(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def infer_effect(n0: float, n1: float, n2: float) -> tuple[str, float] | None:
    """Üç gözlemden (önce, bir basış, iki basış) etkiyi çıkarır; hiçbiri tutmazsa None."""
    if math.isclose(n1, n2) and not math.isclose(n0, n1):
        return "set", n1
    step = n1 - n0
    if not math.isclose(step, 0.0) and math.isclose(n2 - n1, step):
        return "add", step
    if not math.isclose(n0, 0.0) and not math.isclose(n1, 0.0):
        ratio = n1 / n0
        if not math.isclose(ratio, 1.0) and math.isclose(n2, n1 * ratio):
            return ("mul", ratio) if ratio > 1 else ("div", 1 / ratio)
    return None


def effect_holds(op: str, k: float, n0: float, n1: float, n2: float) -> bool:
    if op == "set":
        return math.isclose(n1, k) and math.isclose(n2, k)
    if op == "add":
        return math.isclose(n1 - n0, k) and math.isclose(n2 - n1, k)
    if op == "mul":
        return math.isclose(n1, n0 * k) and math.isclose(n2, n1 * k)
    if op == "div":
        return math.isclose(n1, n0 / k) and math.isclose(n2, n1 / k)
    return False


def apply_effect(op: str, k: float, a: float) -> float:
    return {"add": a + k, "mul": a * k, "div": a / k, "set": k}[op]


# Gözlenen etkinin sembolü: Nora'nın kendi matematik korteksine soruyu bu biçimde sorar.
EFFECT_SYMBOL = {"add+": "+", "add-": "-", "mul": "*", "div": "/"}

# Fiilden ad yapan -(I)m eki (çarp-ım, böl-üm, topla-m): biçimbilim önseli.
_DEVERBAL_M = ("ım", "im", "um", "üm", "m")


def deverbal_root(noun: str) -> str:
    noun = tr_lower(noun)
    for tail in _DEVERBAL_M:
        if noun.endswith(tail) and len(noun) - len(tail) >= 3:
            return noun[:-len(tail)]
    return noun


# ============================================================================
# 5. SÖZCÜK (Türkçe biçimbilim önseli üstünde benzerlik)
# ============================================================================

_WORD = re.compile(r"[\wçğıöşüÇĞİÖŞÜ]+(?:['’][\wçğıöşü]+)?")
# Fiil çekim/sıfat-fiil sonları: tanınmayan bir eylemi sezmek için, anlamını vermek için değil.
_VERBAL_TAILS = ("an", "en", "yan", "yen", "sın", "sin", "sun", "sün", "ır", "ir", "ur", "ür",
                 "ar", "er", "acak", "ecek", "mış", "miş")


def _stem(word: str) -> str:
    try:
        return str(TurkishMorphology.analyze_word(word).get("stem") or word)
    except Exception:
        return word


def _is_verb(word: str) -> bool:
    try:
        return bool(TurkishMorphology.analyze_word(word).get("is_verb"))
    except Exception:
        return False


def same_word(a: str, b: str) -> bool:
    a, b = tr_lower(a), tr_lower(b)
    if a == b:
        return True
    sa, sb = _stem(a), _stem(b)
    if sa == sb and len(sa) >= 3:
        return True
    p = len(os.path.commonprefix([a, b]))
    return p >= 5 or (p >= 4 and p >= min(len(sa), len(sb)))


def word_number(word: str) -> float | None:
    m = re.match(r"\d+(?:[.,]\d+)?", word)
    if m:
        return float(m.group(0).replace(",", "."))
    if tr_lower(word) == "bir":
        # Ekisiz 'bir' çoğunlukla belirsizlik sözcüğüdür ('bir sayfa'), sayı değil.
        return None
    try:
        value = TurkishMorphology.words_to_int([_stem(tr_lower(word))])
    except Exception:
        value = None
    return float(value) if isinstance(value, int) else None


def tokens(text: str) -> list[str]:
    return [tr_lower(m.group(0)) for m in _WORD.finditer(text or "")]


def clauses(text: str) -> list[str]:
    # 'ile' ayırıcı değildir: '3 ile katlayan' içinde büyüklüğü fiile bağlayan edattır.
    parts = re.split(r"[,;.:\n]|\s+ve\s+", text or "")
    return [p.strip() for p in parts if p and p.strip()]


def quoted(text: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r"['\"“‘«]([^'\"”’»]+)['\"”’»]", text or "")]


# ============================================================================
# 6. ÖĞRENİLEN YAPILAR
# ============================================================================

@dataclass
class Fragment:
    """Bir örneğin bir düğmesi: kodu, etkisi ve yeniden bağlanabilir delikleri."""
    example: int
    node: int
    label: str
    code: str
    via_listener: bool
    meaning: str
    k: float
    slot: bool
    state: str
    display_id: str
    words: list[str]
    origin: str = "öğretim"   # öğretim | deney (Nora'nın kendi denemesi)


def meaning_key(op: str, k: float, slot: bool) -> str:
    if op == "add":
        return "add+" if k > 0 else "add-"
    if op == "set" and not slot:
        return "set=" + fmt_num(k)
    return op


class PageCortex:
    """Örnekten öğrenen, birleştiren ve tıklayarak doğrulayan sayfa organı."""

    CONSISTENCY = 0.75

    def __init__(self):
        self.harness = DomHarness()
        self.syn = TurkishSyntaxEngine
        self.examples: list[dict] = []
        self.fragments: list[Fragment] = []
        self.verb_words: dict[str, collections.Counter] = {}
        self.control_nouns: collections.Counter = collections.Counter()
        self.seen_words: set[str] = set()
        self.last: dict = {}
        # Öz-çalışma: kimse öğretmeden kendi deneyiyle bulduğu davranışlar ve onlara verdiği adlar.
        self.self_taught: list[dict] = []
        self.studied = False
        self.oracle = None   # Nora'nın matematik korteksi (adlandırmada danışılan kendi organı)

    # ---------------------------------------------------------------- öz-çalışma
    def explore(self, probe: float = 3.0) -> list[Fragment]:
        """Öğretilmiş düğme kodlarında işleci değiştir, çalıştır ve doğan etkiyi GÖZLE.

        İşleç kümesi (+ - * /) JS sözdiziminin önselidir; hangi değişikliğin hangi etkiyi
        doğurduğu ve bunun hangi sayfada çalıştığı ölçümle bulunur. Yeni etki adsız kaydedilir."""
        known = {f.meaning for f in self.fragments}
        found: list[Fragment] = []
        for frag in [f for f in self.fragments if f.origin == "öğretim" and f.state]:
            state = re.escape(frag.state)
            m = re.search(rf"(?<![\w$.]){state}\s*=\s*{state}\s*([-+*/])\s*(\d+(?:\.\d+)?)", frag.code)
            if not m:
                continue
            for op in "+-*/":
                if op == m.group(1):
                    continue
                code = (frag.code[:m.start(1)] + op + frag.code[m.end(1):m.start(2)]
                        + fmt_num(probe) + frag.code[m.end(2):])
                part = {"code": code, "label": frag.label, "deps": [], "source": frag,
                        "k": probe, "meaning": ""}
                page = parse_page(self._assemble(frag.example, [part]))
                display = page.find_id(frag.display_id)
                controls = [i for i, n in enumerate(page.nodes) if n["attrs"].get("onclick")]
                if display is None or len(controls) != 1:
                    continue
                obs = self.harness.run(page, [[controls[0], controls[0]]])
                if not obs or obs.get("error") or obs.get("load_errors") or obs["runs"][0]["errors"]:
                    continue
                n0, n1, n2 = (_first_number(s[display][0]) for s in obs["runs"][0]["snaps"])
                guess = None if None in (n0, n1, n2) else infer_effect(n0, n1, n2)
                if guess is None:
                    continue
                eff, k = guess
                magnitude = abs(k) if eff == "add" else k
                if not math.isclose(magnitude, probe):
                    continue
                key = meaning_key(eff, k, True)
                if key in known:
                    continue
                known.add(key)
                found.append(Fragment(
                    example=frag.example, node=frag.node, label="", code=code, via_listener=False,
                    meaning=key, k=probe, slot=True, state=frag.state, display_id=frag.display_id,
                    words=[], origin="deney"))
        self.fragments.extend(found)
        return found

    def _oracle(self):
        if self.oracle is None:
            from nora_stem import MathCortex
            self.oracle = MathCortex()
        return self.oracle

    def name_behavior(self, frag: Fragment, a: float = 6.0) -> dict | None:
        """Adsız davranışı kendi matematik korteksine sorarak adlandırır.

        Soru yalnız sembolle sorulur ('6 * 3'). Korteksin cevabındaki sonuç, deneyde gözlenen
        sonuçla aynı değilse ad verilmez; aynıysa sonucu niteleyen ad ('çarpım') fiil köküne
        indirilir ('çarp') ve davranışa bağlanır."""
        symbol = EFFECT_SYMBOL.get(frag.meaning)
        if symbol is None:
            return None
        question = f"{fmt_num(a)} {symbol} {fmt_num(frag.k)}"
        reply = str((self._oracle().try_solve(question) or {}).get("reply") or "")
        op = frag.meaning.rstrip("+-")
        k = -frag.k if frag.meaning == "add-" else frag.k
        expected = apply_effect(op, k, a)
        words = tokens(reply)
        at = [i for i, w in enumerate(words) if word_number(w) is not None
              and math.isclose(word_number(w), expected)]
        if not at or at[-1] == 0 or word_number(words[at[-1] - 1]) is not None:
            return None
        root = deverbal_root(words[at[-1] - 1])
        self.verb_words.setdefault(root, collections.Counter())[frag.meaning] += 1
        frag.words, frag.label = [root], root
        record = {"word": root, "meaning": frag.meaning, "question": question, "reply": reply,
                  "source": "deney+matematik"}
        self.self_taught.append(record)
        return record

    def self_study(self) -> list[dict]:
        """Öğretmensiz çalışma: deneyle yeni davranış bul, kendi bilgisiyle adlandır."""
        self.studied = True
        named = []
        for frag in self.explore():
            rec = self.name_behavior(frag)
            if rec:
                named.append(rec)
        return named

    def authorship(self) -> dict:
        return {"öğretim": sum(f.origin == "öğretim" for f in self.fragments),
                "deney": sum(f.origin == "deney" and bool(f.words) for f in self.fragments)}

    # ---------------------------------------------------------------- öğretim
    @staticmethod
    def has_markup(text: str) -> bool:
        return bool(re.search(r"```html\s*\n", text or "", re.IGNORECASE))

    @staticmethod
    def split_document(text: str) -> list[tuple[str, str]]:
        """'açıklama ```html ... ```' çiftleri; önek ('Öğren:') ve boş satırlar ayıklanır."""
        pairs = []
        last = 0
        for m in re.finditer(r"```html\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE):
            desc = text[last:m.start()]
            desc = re.sub(r"^\s*(?:öğren|belge|doküman|document)\s*:\s*", "", desc.strip(),
                          flags=re.IGNORECASE)
            pairs.append((desc.strip(), m.group(1)))
            last = m.end()
        return pairs

    def teach_document(self, text: str) -> dict:
        learned, failed = [], []
        for desc, markup in self.split_document(text):
            res = self.teach(desc, markup)
            (learned if res.get("ok") else failed).append(res)
        names = [f"{w}:{f}" for r in learned for w, f in r.get("words", [])]
        return {"ok": bool(learned), "learned": learned, "failed": failed,
                "new_operators": names, "language": "html"}

    def teach(self, description: str, markup: str, *, record: bool = True) -> dict:
        if not self.harness.available:
            return {"ok": False, "reason": "kabuk_yok"}
        page = parse_page(markup)
        first = self.harness.run(page, [])
        if not first or first.get("error"):
            return {"ok": False, "reason": "çalışmadı", "detail": (first or {}).get("error")}
        controls = [i for i, h in enumerate(first["handlers"])
                    if h["inline"] or h["prop"] or h["listeners"]]
        if not controls:
            return {"ok": False, "reason": "düğme_yok"}
        obs = self.harness.run(page, [[c, c] for c in controls])
        if not obs or obs.get("error"):
            return {"ok": False, "reason": "çalışmadı", "detail": (obs or {}).get("error")}

        ex_index = len(self.examples)
        stmts = js_statements(page.script())
        decls = {name: s for s in stmts for name in declared_names(s)}
        desc_clauses = clauses(description)
        desc_tokens = tokens(description)
        self.seen_words.update(desc_tokens)
        new_fragments, words = [], []
        for c, run in zip(controls, obs["runs"]):
            if run["errors"]:
                continue
            s0, s1, s2 = run["snaps"]
            effect = None
            for d in range(len(page.nodes)):
                if s0[d][0] == s1[d][0]:
                    continue
                n0, n1, n2 = (_first_number(s[d][0]) for s in (s0, s1, s2))
                if None in (n0, n1, n2):
                    continue
                guess = infer_effect(n0, n1, n2)
                if guess:
                    effect = (d, *guess)
                    break
            if effect is None:
                continue
            display, op, k = effect
            handler = first["handlers"][c]
            via_listener = not handler["inline"]
            code = handler["inline"] or function_body(handler["prop"] or handler["listeners"][0])
            label = page.text(c)
            clause = self._clause_for(label, desc_clauses, len(controls) == 1, description)
            numbers = [n for n in (word_number(t) for t in tokens(clause)) if n is not None]
            magnitude = abs(k) if op == "add" else k
            slot = any(math.isclose(n, magnitude) for n in numbers)
            key = meaning_key(op, k, slot)
            label_words = tokens(label)
            verb_like = [t for t in tokens(clause)
                         if word_number(t) is None
                         and (_is_verb(t) or any(same_word(t, lw) for lw in label_words))]
            for t in verb_like or label_words:
                self.verb_words.setdefault(t, collections.Counter())[key] += 1
                words.append((t, key))
            for t in tokens(clause):
                if t not in verb_like and word_number(t) is None and len(t) >= 3:
                    self.control_nouns[t] += 1
            # Durum: düğmenin kodunda atanan, sayfa betiğinde bildirilmiş değişken.
            assigned = sorted(assigned_names(code))
            state = next((n for n in assigned if n in decls), assigned[0] if assigned else "")
            new_fragments.append(Fragment(
                example=ex_index, node=c, label=label, code=code, via_listener=via_listener,
                meaning=key, k=magnitude, slot=slot, state=state,
                display_id=page.nodes[display]["attrs"].get("id", ""), words=verb_like or label_words,
            ))
        if not new_fragments:
            return {"ok": False, "reason": "etki_gözlenmedi"}
        self.examples.append({"description": description, "markup": markup,
                              "controls": controls, "decls": decls, "stmts": stmts})
        self.fragments.extend(new_fragments)
        return {"ok": True, "example": ex_index, "words": words,
                "fragments": [(f.label, f.meaning, fmt_num(f.k), f.slot) for f in new_fragments]}

    @staticmethod
    def _clause_for(label: str, desc_clauses: list[str], only: bool, description: str) -> str:
        if only:
            return description
        label_words = tokens(label)
        for cl in desc_clauses:
            if any(q.strip() and tr_lower(q.strip()) == tr_lower(label.strip()) for q in quoted(cl)):
                return cl
        for cl in desc_clauses:
            if any(same_word(t, lw) for t in tokens(cl) for lw in label_words):
                return cl
        return ""

    # ---------------------------------------------------------------- istek
    def _meaning_of(self, word: str) -> tuple[str, float] | None:
        best = None
        for learned, counter in self.verb_words.items():
            if not same_word(word, learned):
                continue
            key, count = counter.most_common(1)[0]
            consistency = count / sum(counter.values())
            if consistency < self.CONSISTENCY:
                continue
            score = consistency * math.log1p(count)
            if best is None or score > best[1]:
                best = (key, score)
        return best

    def _is_control_noun(self, word: str) -> bool:
        return any(same_word(word, n) for n, c in self.control_nouns.items() if c >= 1)

    def plan(self, request: str) -> dict:
        items, unknown = [], []
        for cl in clauses(request):
            toks = tokens(cl)
            hits = [(t, m) for t in toks for m in [self._meaning_of(t)] if m]
            numbers = [n for n in (word_number(t) for t in toks) if n is not None]
            if hits:
                word, (key, _score) = max(hits, key=lambda h: h[1][1])
                items.append({"meaning": key, "word": word, "number": numbers[0] if numbers else None})
                continue
            verbal = [t for t in toks if t not in self.seen_words and len(_stem(t)) >= 4
                      and t.endswith(_VERBAL_TAILS) and word_number(t) is None]
            if verbal or numbers:
                unknown.append(" ".join(verbal) or cl)
        return {"items": items, "unknown": unknown}

    def requested_evidence(self, raw: str) -> float:
        if not self.fragments:
            return 0.0
        toks = tokens(raw)
        verb_hit = any(self._meaning_of(t) for t in toks)
        noun_hit = any(self._is_control_noun(t) for t in toks if not self._meaning_of(t))
        explicit = "html" in toks
        if verb_hit and (noun_hit or explicit):
            return 1.0
        if explicit or (noun_hit and self.plan(raw)["unknown"]):
            return 0.6
        return 0.0

    def is_requested(self, raw: str) -> bool:
        return self.requested_evidence(raw) >= 0.6

    # ---------------------------------------------------------------- birleştirme
    def _magnitude(self, item: dict, frag: Fragment) -> float:
        if frag.slot and item["number"] is not None:
            return float(item["number"])
        return frag.k

    def _adapt(self, frag: Fragment, k: float, base: Fragment, used: dict[str, str]) -> dict | None:
        ex = self.examples[frag.example]
        code, label = frag.code, frag.label
        if not math.isclose(k, frag.k):
            old = fmt_num(frag.k)
            pattern = rf"(?<![\w.]){re.escape(old)}(?![\w.])"
            if not re.search(pattern, code):
                return None
            code = re.sub(pattern, fmt_num(k), code)
            label = re.sub(pattern, fmt_num(k), label)
        deps: list[str] = []
        if frag.example != base.example:
            code = rename_identifier(code, frag.state, base.state)
            code = rename_element_id(code, frag.display_id, base.display_id)
            pending = [n for n in identifiers(code) if n in ex["decls"] and n != base.state]
            seen = set()
            while pending:
                name = pending.pop()
                if name in seen:
                    continue
                seen.add(name)
                stmt = rename_identifier(ex["decls"][name], frag.state, base.state)
                stmt = rename_element_id(stmt, frag.display_id, base.display_id)
                if name in used and used[name] != stmt:
                    fresh = f"{name}{frag.example}"
                    code = rename_identifier(code, name, fresh)
                    stmt = rename_identifier(stmt, name, fresh)
                    deps = [rename_identifier(d, name, fresh) for d in deps]
                    name = fresh
                if used.get(name) != stmt:
                    used[name] = stmt
                    deps.append(stmt)
                pending.extend(n for n in identifiers(stmt)
                               if n in ex["decls"] and n not in seen and n != base.state)
        return {"code": code, "label": label, "deps": deps, "source": frag}

    def compose(self, plan_items: list[dict]) -> dict:
        by_key: dict[str, list[Fragment]] = collections.defaultdict(list)
        for f in self.fragments:
            by_key[f.meaning].append(f)
        if any(not by_key.get(it["meaning"]) for it in plan_items):
            return {"ok": False, "reason": "parça_yok"}
        coverage = collections.Counter(f.example for it in plan_items for f in by_key[it["meaning"]])
        bases = sorted({f.example for f in self.fragments if f.display_id},
                       key=lambda e: (-coverage[e], e))
        attempts = []
        for base_ex in bases:
            base = next(f for f in self.fragments if f.example == base_ex and f.display_id)
            ex = self.examples[base_ex]
            used = {n: s for n, s in ex["decls"].items()}
            parts = []
            for it in plan_items:
                # Önce istekteki sözcüğe benzeyen parça (adı da doğru olsun), sonra tabanın kendisi.
                cands = sorted(by_key[it["meaning"]], key=lambda f: (
                    not any(same_word(it["word"], w) for w in tokens(f.label)),
                    not any(same_word(it["word"], w) for w in f.words),
                    f.example != base_ex, f.example))
                part = None
                for frag in cands:
                    part = self._adapt(frag, self._magnitude(it, frag), base, dict(used))
                    if part is not None:
                        used.update({n: s for s in part["deps"] for n in declared_names(s)})
                        break
                if part is None:
                    break
                part["k"] = self._magnitude(it, part["source"])
                part["meaning"] = it["meaning"]
                parts.append(part)
            if len(parts) != len(plan_items):
                continue
            markup = self._assemble(base_ex, parts)
            check = self.verify(markup, base.display_id, parts)
            attempts.append({"base": base_ex, "ok": check["ok"], "detail": check.get("detail")})
            if check["ok"]:
                return {"ok": True, "markup": markup, "parts": parts, "checks": check["checks"],
                        "base": base_ex, "attempts": attempts}
        return {"ok": False, "reason": "doğrulanamadı", "attempts": attempts}

    def _assemble(self, base_ex: int, parts: list[dict]) -> str:
        ex = self.examples[base_ex]
        page = parse_page(ex["markup"])
        anchor = None
        for c in sorted(ex["controls"]):
            parent = page.nodes[c]["parent"]
            if parent is not None:
                content = page.nodes[parent]["content"]
                at = content.index(c)
                if at > 0 and isinstance(content[at - 1], str) and not content[at - 1].strip():
                    del content[at - 1]
                    at -= 1
                if anchor is None:
                    anchor = (parent, at)
            page.nodes[c]["live"] = False
        if anchor is None:
            body = page.find_tag("body")
            anchor = (body, len(page.nodes[body]["content"])) if body is not None else (page.roots[-1], 0)
        parent, pos = anchor
        indent = next((c for c in reversed(page.nodes[parent]["content"][:pos])
                       if isinstance(c, str) and "\n" in c), "\n")
        new_items: list[Any] = []
        for part in parts:
            src_page = parse_page(self.examples[part["source"].example]["markup"])
            src = src_page.nodes[part["source"].node]
            attrs = {k: v for k, v in src["attrs"].items() if k != "id" and not k.startswith("on")}
            attrs["onclick"] = " ".join(line.strip() for line in part["code"].splitlines() if line.strip())
            page.nodes.append({"tag": src["tag"], "attrs": attrs, "content": [part["label"]],
                               "parent": parent})
            new_items += [indent, len(page.nodes) - 1]
        content = page.nodes[parent]["content"]
        page.nodes[parent]["content"] = content[:pos] + new_items + content[pos:]
        stmts = [s for s in ex["stmts"] if not attaches_listener(s)]
        stmts += [d for part in parts for d in part["deps"]]
        script = "\n".join(_tidy_statement(s) for s in stmts)
        body = page.find_tag("body")
        holder = body if body is not None else page.roots[-1]
        content = page.nodes[holder]["content"]
        while content and isinstance(content[-1], str) and not content[-1].strip():
            content.pop()
        content.append(("raw", f"{indent}<script>\n{script}\n</script>\n"))
        return serialize(page)

    def verify(self, markup: str, display_id: str, parts: list[dict]) -> dict:
        page = parse_page(markup)
        display = page.find_id(display_id)
        controls = [i for i, n in enumerate(page.nodes) if n["attrs"].get("onclick")]
        if display is None or len(controls) != len(parts):
            return {"ok": False, "detail": "yapı"}
        obs = self.harness.run(page, [[c, c] for c in controls])
        if not obs or obs.get("error") or obs.get("load_errors"):
            return {"ok": False, "detail": (obs or {}).get("error") or (obs or {}).get("load_errors")}
        checks = []
        for part, run in zip(parts, obs["runs"]):
            n0, n1, n2 = (_first_number(s[display][0]) for s in run["snaps"])
            op = part["meaning"].rstrip("+-").split("=")[0]
            k = part["k"] * (-1 if part["meaning"] == "add-" else 1)
            if op == "set" and "=" in part["meaning"]:
                k = float(part["meaning"].split("=")[1])
            ok = (not run["errors"] and None not in (n0, n1, n2) and effect_holds(op, k, n0, n1, n2))
            checks.append({"label": part["label"], "op": op, "k": k, "ok": ok,
                           "seen": [n0, n1, n2], "errors": run["errors"]})
        return {"ok": all(c["ok"] for c in checks), "checks": checks,
                "detail": [c for c in checks if not c["ok"]]}

    # ---------------------------------------------------------------- giriş/beyan
    def handle(self, request: str) -> dict:
        plan = self.plan(request)
        learned_now: list[dict] = []
        if (plan["unknown"] or not plan["items"]) and not self.studied and self.fragments:
            # Bilmediği bir etkiyle karşılaşınca önce kendi başına çalışır, sonra yeniden dener.
            learned_now = self.self_study()
            if learned_now:
                plan = self.plan(request)
        if plan["unknown"] or not plan["items"]:
            gap = {"ok": False, "language": "html", "path": "page_gap", "reason": "bilinmeyen_etki",
                   "unknown": plan["unknown"], "known": self.known_words(), "program": "",
                   "verified": False}
            gap["reply_hint"] = self.speak(gap)
            return gap
        res = self.compose(plan["items"])
        self.last = {"request": request, "plan": plan, **{k: v for k, v in res.items() if k != "markup"}}
        if not res["ok"]:
            fail = {"ok": False, "language": "html", "path": "page_unverified", "reason": res["reason"],
                    "program": "", "verified": False, "attempts": res.get("attempts"),
                    "unknown": [it["word"] for it in plan["items"]], "known": self.known_words()}
            fail["reply_hint"] = self.speak(fail)
            return fail
        used = [p["source"] for p in res["parts"]]
        return {"ok": True, "language": "html", "path": "page", "program": res["markup"],
                "verified": True, "confidence": 1.0, "checks": res["checks"],
                "sources": sorted({f.example for f in used}),
                "self_taught": [r for r in self.self_taught
                                if any(f.origin == "deney" and r["word"] in f.words for f in used)],
                "learned_now": learned_now}

    def known_words(self) -> list[str]:
        """Her öğrenilmiş etki için onu en sık anlatan sözcük."""
        best: dict[str, tuple[int, str]] = {}
        for word, counter in self.verb_words.items():
            key, count = counter.most_common(1)[0]
            if count / sum(counter.values()) < self.CONSISTENCY:
                continue
            if key not in best or (count, -len(word)) > (best[key][0], -len(best[key][1])):
                best[key] = (count, word)
        return sorted(w for _, w in best.values())

    def speak(self, res: dict) -> str:
        syn = self.syn
        if res.get("path") == "document":
            c_read = syn.realize_verb(VerbSpec(verb_root="oku", obj=NP("örnek", case="ACC"),
                                               tense="PAST", person="1SG"))
            return tr_capitalize(c_read) + ": " + (", ".join(res.get("new_operators") or ()) or "0") + "."
        if not res.get("ok"):
            c_learn = syn.realize_verb(VerbSpec(verb_root="öğren", obj=NP("etki", case="ACC"), neg=True,
                                                tense="PAST", person="1SG"))
            c_know = syn.realize_verb(VerbSpec(verb_root="bil", tense="PRES", person="1SG"))
            unknown = ", ".join(res.get("unknown") or ()) or "—"
            known = ", ".join(res.get("known") or ()) or "—"
            return f"{tr_capitalize(c_learn)}: {unknown}. {tr_capitalize(c_know)}: {known}."
        c_build = syn.realize_verb(VerbSpec(verb_root="kur", obj=NP("sayfa", case="ACC"),
                                            tense="PAST", person="1SG"))
        c_try = syn.realize_verb(VerbSpec(verb_root="dene", obj=NP("düğme", plural=True, case="ACC"),
                                          tense="PAST", person="1SG"))
        def effect(c: dict) -> str:
            if c["op"] == "add":
                return ("+" if c["k"] >= 0 else "−") + fmt_num(abs(c["k"]))
            return {"mul": "×", "div": "÷", "set": "="}.get(c["op"], c["op"]) + fmt_num(c["k"])
        checks = "; ".join(
            f"«{c['label']}» {effect(c)} {'✓' if c['ok'] else '✗'}"
            for c in res.get("checks") or ())
        head = ""
        if res.get("self_taught"):
            # Öğretilmediği hâlde kullandığı şeyi kaynağıyla söyler: deney + kendi matematiği.
            c_self = syn.realize_verb(VerbSpec(verb_root="öğren", obj="bunu", advs=("kendi kendime",),
                                               tense="PAST", person="1SG"))
            head = tr_capitalize(c_self) + ": " + "; ".join(
                f"{r['word']} ({r['question']} → {r['reply'].split(';')[0]})" for r in res["self_taught"]) + ".\n"
        return (head + tr_capitalize(c_build) + ".\n```html\n" + str(res.get("program") or "").rstrip()
                + "\n```\n" + tr_capitalize(c_try) + ": " + checks + ".")

    # ---------------------------------------------------------------- kalıcılık
    def state_dict(self) -> dict:
        return {"examples": [{"description": e["description"], "markup": e["markup"]}
                             for e in self.examples],
                "studied": self.studied}

    def load_state_dict(self, data: dict | None) -> None:
        oracle = self.oracle
        self.__init__()
        self.oracle = oracle
        for ex in (data or {}).get("examples") or ():
            self.teach(ex.get("description", ""), ex.get("markup", ""))
        # Öz-çalışma deterministiktir: aynı örneklerden aynı davranışlar yeniden bulunur.
        if (data or {}).get("studied"):
            self.self_study()
