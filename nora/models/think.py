#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_think.py — NoRa Düşünce Motoru: düşün → doğrula → uyu → sezgi.

Döngü (her "gün"):
  1. MERAK      kendi sorularını üretir; öğrenme-ilerlemesi en yüksek zorluğu seçer.
  2. SİSTEM-1   doğrudan cevap + güven (cevap token'larının geometrik-ort. olasılığı).
  3. ÜSTBİLİŞ   öğrenilmiş eşik τ: güven ≥ τ ise sezgiyle cevapla, değilse düşün.
                τ, "hangi güvende haklı çıktım" geçmişinden kalibre edilir; ara sıra
                güvenli cevaplar da denetlenir (öz-denetim) ki kalibrasyon yalan söylemesin.
  4. SİSTEM-2   '[' açılır, sütun sütun adım yazılır:  47 + 38 = [7+8+0=15>5|4+3+1=8>85] 85
                Her adımda K aday örneklenir; HAKEM cevabı BİLMEDEN yalnızca yerel
                tutarlılığı denetler (doğru sütun mu, elde/ödünç doğru mu, hesap tutuyor mu,
                kısmi sonuç doğru mu). Reddedilirse daha sıcak yeniden dener.
  5. UYKU       doğrulanmış düşünceler hem İZ hem DOĞRUDAN CEVAP olarak yeni adaptöre
                damıtılır (düşünce → sezgi). Eski adaptörler donuk; eski günler tekrar edilir.
Öğretmen YALNIZCA 1-2 basamak örnek gösterir (gün 0). 3-4 basamak hiç gösterilmez.

Kullanım (DPY = D venv python):
  DPY nora_think.py --days 5                       # tam döngü, models/think_brain.pt
  DPY nora_think.py --days 3 --resume              # kaldığı yerden
  DPY nora_think.py --watch "347 + 589"            # düşünceyi canlı izle
"""

from nora._paths import REPO_ROOT
import argparse, json, math, os, random, re, sys, time

PROJECT_ROOT = os.path.join("D:" + os.sep, "NoRa_AGI") if os.name == "nt" else os.path.join(os.path.expanduser("~"), "NoRa_AGI")
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as FF
from micro_gpt import Block, MicroGPT, batchify
from grow_gpt import rebuild_brain, MODEL_DIR, CKPT
from tool_calc import safe_eval

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OPS = ("+", "-")
LEVELS = [(op, n) for op in OPS for n in (1, 2, 3, 4)]  # (işlem, basamak)
TEACH_MAX_DIGITS = 2
STEP_RE = re.compile(r"^(\d+)([+\-])(\d)([+\-])(\d)=(\d+)>(\d+)$")


# ---------------------------------------------------------------- yöntem (prosedür)
def col(v, k):
    s = str(v)
    return int(s[-1 - k]) if k < len(s) else 0


def n_cols(a, b):
    return max(len(str(a)), len(str(b)))


def verify_step(a, op, b, k, carry, partial, text):
    """HAKEM: cevabı bilmez; yalnızca adımın soruyla ve önceki adımla tutarlılığını denetler.
    Döner: (geçerli, neden, (yeni_elde, yeni_kısmi))."""
    m = STEP_RE.match(text)
    if not m:
        return False, "biçim", None
    X, o1, Y, o2, C, S, P = m.groups()
    X, Y, C = int(X), int(Y), int(C)
    if o1 != op or o2 != op:
        return False, "işlem", None
    xa, yb = col(a, k), col(b, k)
    if Y != yb:
        return False, "2. sütun", None
    if C != carry:
        return False, "elde/ödünç", None
    last = k == n_cols(a, b) - 1
    if op == "+":
        if X != xa:
            return False, "1. sütun", None
        v, _ = safe_eval("%d+%d+%d" % (X, Y, C))
        if S != str(v):
            return False, "hesap", None
        newp = (str(v) if last else str(v % 10)) + partial
        nc = 0 if last else v // 10
    else:
        need = xa - yb - C < 0
        if X != xa + (10 if need else 0):
            return False, "1. sütun/ödünç", None
        v, _ = safe_eval("%d-%d-%d" % (X, Y, C))
        if S != str(v):
            return False, "hesap", None
        newp = str(v) + partial
        nc = 1 if need else 0
    if P != newp:
        return False, "kısmi sonuç", None
    return True, "ok", (nc, newp)


def step_prefix(a, op, b, k, carry):
    """DİKKAT İŞARETÇİSİ: adımın 'nereye bakılır' kısmı (sütun rakamları + elde/ödünç).
    Model 'ne olduğu'nu (sonuç > kısmi) kendisi üretir; hakem tamamını yine denetler."""
    x, y = col(a, k), col(b, k)
    if op == "+":
        return "%d+%d+%d=" % (x, y, carry)
    return "%d-%d-%d=" % (x + (10 if x - y - carry < 0 else 0), y, carry)


def teacher_steps(a, op, b):
    """Öğretmen izi: hakemin kabul ettiği tek kanonik adım dizisi."""
    steps, c, p = [], 0, ""
    for k in range(n_cols(a, b)):
        x, y = col(a, k), col(b, k)
        last = k == n_cols(a, b) - 1
        if op == "+":
            s = x + y + c
            p = (str(s) if last else str(s % 10)) + p
            st = "%d+%d+%d=%d>%s" % (x, y, c, s, p)
            c = 0 if last else s // 10
        else:
            need = x - y - c < 0
            xx = x + (10 if need else 0)
            d = xx - y - c
            p = str(d) + p
            st = "%d-%d-%d=%d>%s" % (xx, y, c, d, p)
            c = 1 if need else 0
        steps.append(st)
    return steps


def answer_of(a, op, b):
    return a + b if op == "+" else a - b


def prompt_of(a, op, b):
    return "%d %s %d =" % (a, op, b)


def think_line(a, op, b, steps, ans):
    return "%s [%s] %d" % (prompt_of(a, op, b), "|".join(steps), ans)


def direct_line(a, op, b, ans):
    return "%s %d" % (prompt_of(a, op, b), ans)


def make_problem(rng, op, n):
    a = rng.randint(0, 9) if n == 1 else rng.randint(10 ** (n - 1), 10 ** n - 1)
    b = rng.randint(0, 10 ** rng.randint(1, n) - 1)
    if op == "-" and b > a:
        a, b = b, a
    return a, op, b


# ---------------------------------------------------------------- beyin
_DEFAULT_THINK_CHARS = list("0123456789+-=[]|>\n ")


def extract_think_coords(idx, stoi):
    """Çok satırlı ve nedensel sözdizimsel koordinat çıkarıcı (hiçbir aritmetik kuralı bilmez):
    Mutlak dizi indeksi yerine basamak/adım göreli koordinatlar üretir; böylece yalnızca 1-2 basamakta
    eğitilen sütun, daha önce hiç görmediği 3, 4, 5+ basamaklı işlemlere sıfır-atış genelleme yapar."""
    digits = {stoi[c] for c in "0123456789" if c in stoi}
    id_plus, id_minus = stoi.get("+", -1), stoi.get("-", -1)
    id_eq, id_lb, id_rb = stoi.get("=", -1), stoi.get("[", -1), stoi.get("]", -1)
    id_pipe, id_gt, id_sp, id_nl = stoi.get("|", -1), stoi.get(">", -1), stoi.get(" ", -1), stoi.get("\n", -1)

    B, T = idx.shape
    dev = idx.device
    arr = idx.tolist()
    tok_canon = idx.clone()
    phase = torch.zeros((B, T), dtype=torch.long, device=dev)
    q_col = torch.zeros((B, T), dtype=torch.long, device=dev)
    k_col = torch.zeros((B, T), dtype=torch.long, device=dev)
    line_id = torch.zeros((B, T), dtype=torch.long, device=dev)
    q_step = torch.zeros((B, T), dtype=torch.long, device=dev)
    k_step = torch.full((B, T), -10, dtype=torch.long, device=dev)
    step_u = torch.zeros((B, T), dtype=torch.long, device=dev)
    flags = torch.zeros((B, T), dtype=torch.long, device=dev)

    for b in range(B):
        row = arr[b]
        s, lid = 0, 0
        while s < T:
            try:
                end = row.index(id_nl, s) + 1
            except ValueError:
                end = T
            eq0 = end
            for p in range(s, end):
                if row[p] == id_eq:
                    eq0 = p
                    break
            seen_op = False
            has_a, has_b = set(), set()
            i = s
            while i < eq0:
                line_id[b, i] = lid
                if row[i] in digits:
                    j = i
                    while j < eq0 and row[j] in digits:
                        j += 1
                    for p in range(i, j):
                        line_id[b, p] = lid
                        cv = (j - 1) - p
                        phase[b, p] = 2 if seen_op else 0
                        k_col[b, p] = cv
                        q_col[b, p] = cv
                        k_step[b, p] = -10
                        (has_b if seen_op else has_a).add(cv)
                    i = j
                else:
                    if row[i] in (id_plus, id_minus) and i > s and not seen_op:
                        seen_op = True
                        phase[b, i] = 1
                    else:
                        phase[b, i] = 3
                    k_step[b, i] = -10
                    i += 1

            max_col = max(has_a | has_b) if (has_a or has_b) else 0
            st, k, su, ns_toks, head_rem, c_rem, cur_p, last_p = 3, 0, 0, 0, 0, 0, 0, 0
            for t in range(eq0, end):
                line_id[b, t] = lid
                tok = row[t]
                more = 1 if k < max_col else 0
                in_a, in_b = int(k in has_a), int(k in has_b)
                if st == 3:
                    phase[b, t] = 3
                    k_step[b, t] = 0
                    q_step[b, t] = -10
                    step_u[b, t] = min(su, 15)
                    su += 1
                    if tok == id_lb:
                        st, k, su = 4, 0, 1
                        phase[b, t] = 4
                        q_col[b, t] = 0
                        k_col[b, t] = 0
                        k_step[b, t] = 1
                        q_step[b, t] = 0
                        step_u[b, t] = 0
                        flags[b, t] = int(0 in has_a) + 2 * int(0 in has_b)
                elif st == 4:
                    k_step[b, t] = k + 1
                    q_step[b, t] = k
                    q_col[b, t] = k
                    k_col[b, t] = k
                    step_u[b, t] = min(su, 15)
                    su += 1
                    if tok == id_eq:
                        phase[b, t] = 5
                        ns_toks = 0
                        flags[b, t] = 0
                        st = 5
                    else:
                        phase[b, t] = 4
                        flags[b, t] = in_a + 2 * in_b
                elif st == 5:
                    k_step[b, t] = k + 1
                    q_step[b, t] = k
                    q_col[b, t] = k
                    k_col[b, t] = k
                    step_u[b, t] = min(su, 15)
                    su += 1
                    if tok == id_gt:
                        head_rem = ns_toks if more == 0 else 1
                        c_rem = head_rem + k - 1
                        cur_p = 0
                        phase[b, t] = 6
                        k_col[b, t] = k
                        q_col[b, t] = max(0, c_rem)
                        flags[b, t] = int(head_rem >= 2)
                        st = 6
                    else:
                        ns_toks += 1
                        phase[b, t] = 5
                        flags[b, t] = min(ns_toks, 3)
                elif st == 6:
                    if tok == id_pipe:
                        if id_lb >= 0:
                            tok_canon[b, t] = id_lb
                        last_p = cur_p
                        k += 1
                        su = 1
                        st = 4
                        phase[b, t] = 4
                        k_col[b, t] = k
                        q_col[b, t] = k
                        k_step[b, t] = k + 1
                        q_step[b, t] = k
                        step_u[b, t] = 0
                        flags[b, t] = int(k in has_a) + 2 * int(k in has_b)
                    elif tok == id_rb:
                        last_p = cur_p
                        phase[b, t] = 7
                        k_step[b, t] = k + 1
                        q_step[b, t] = k + 1
                        k_col[b, t] = 0
                        q_col[b, t] = max(0, last_p - 1)
                        step_u[b, t] = 0
                        flags[b, t] = 0
                        st = 7
                    else:
                        k_step[b, t] = k + 1
                        q_step[b, t] = k
                        mc = max(0, c_rem)
                        cur_p += 1
                        if head_rem > 0:
                            head_rem -= 1
                        c_rem -= 1
                        phase[b, t] = 6
                        k_col[b, t] = mc
                        q_col[b, t] = max(0, c_rem)
                        step_u[b, t] = 15
                        flags[b, t] = 0 if head_rem >= 1 else (2 if c_rem >= 0 else (4 if more == 1 else 3))
                else:
                    k_step[b, t] = k + 2
                    q_step[b, t] = k + 1
                    step_u[b, t] = 15
                    if tok == id_sp:
                        c_rem = last_p - 1
                        phase[b, t] = 8
                        k_col[b, t] = 0
                        q_col[b, t] = max(0, c_rem)
                        flags[b, t] = int(c_rem < 0)
                    elif tok != id_nl:
                        mc = max(0, c_rem)
                        c_rem -= 1
                        phase[b, t] = 8
                        k_col[b, t] = mc
                        q_col[b, t] = max(0, c_rem)
                        flags[b, t] = int(c_rem < 0)
                    else:
                        phase[b, t] = 9
                        flags[b, t] = 0
            s = end
            lid += 1
    return tok_canon, phase, q_col, k_col, line_id, q_step, k_step, step_u, flags


class MetricRelAttn(nn.Module):
    """4 uzmanlaşmış metrik göreli dikkat kafası (monotonik üstel sönüm, sıfır periyodik taşma):
    Kafa 0: soru basamak okuyucu (k_step == -10, k_col == q_col)
    Kafa 1: önceki adım elde/ödünç okuyucu (k_step == q_step)
    Kafa 2: basamak-hizalı kısmi/nihai sonuç kopyalayıcı (k_step == q_step ve k_col == q_col)
    Kafa 3: adım-içi yerel tek-basamak aritmetik (aynı adım k_step_i == k_step_j)"""

    def __init__(self, d, n_head=4):
        super().__init__()
        assert n_head == 4 and d % 4 == 0
        self.hd = d // 4
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.raw_col = nn.Parameter(torch.tensor([3.5, 0.0, 3.5, 0.5]))
        self.raw_step = nn.Parameter(torch.tensor([3.0, 3.5, 3.5, 3.5]))

    def forward(self, x, q_col, k_col, q_step, k_step, step_u, mask):
        B, T, D = x.shape
        hd = self.hd
        q, k, v = [t.view(B, T, 4, hd).transpose(1, 2) for t in self.qkv(x).chunk(3, dim=-1)]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        d_col = (q_col.unsqueeze(2) - k_col.unsqueeze(1)).abs().float()
        d_qstep = (q_step.unsqueeze(2) - k_step.unsqueeze(1)).abs().float()
        d_same = (k_step.unsqueeze(2) - k_step.unsqueeze(1)).abs().float()
        k_np = (k_step != -10).float().unsqueeze(1).expand(B, T, T)
        ac, bs = FF.softplus(self.raw_col), FF.softplus(self.raw_step)
        rel = torch.stack([
            -ac[0] * d_col - bs[0] * k_np,
            -bs[1] * d_qstep,
            -ac[2] * d_col - bs[2] * d_qstep,
            -bs[3] * d_same - 0.1 * ac[3] * (step_u.unsqueeze(2) - step_u.unsqueeze(1)).abs().float(),
        ], dim=1)
        out = (FF.softmax(att + rel + mask, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class ThinkColumn(nn.Module):
    """KORTEKS SÜTUNU (Progressive Nets + Kompozisyonel Göreli Dikkat): adaptör kapasitesi yetmediğinde
    büyütülen, SIFIRDAN eğitilen küçük transformer. Mutlak pozisyon gömmesi yerine göreli basamak/adım
    metrik dikkati kullanır; böylece 1-2 basamaktan öğrenilen sütun kuralı 3+ basamağa sıfır-atış taşınır."""

    def __init__(self, V, block, d=128, n_layer=3, n_head=4, d_lat=256, lateral=True, stoi=None):
        super().__init__()
        self.block = block
        self.stoi = stoi if stoi is not None else {c: i for i, c in enumerate(_DEFAULT_THINK_CHARS[:V])}
        self.tok = nn.Embedding(V, d)
        self.phase_emb = nn.Embedding(12, d)
        self.su_emb = nn.Embedding(16, d)
        self.flag_emb = nn.Embedding(8, d)
        self.lateral = lateral
        if lateral:
            self.lat = nn.Linear(d_lat, d)
            nn.init.zeros_(self.lat.weight)
            nn.init.zeros_(self.lat.bias)
        self.lns1 = nn.ModuleList([nn.LayerNorm(d, bias=False) for _ in range(n_layer)])
        self.attns = nn.ModuleList([MetricRelAttn(d, n_head) for _ in range(n_layer)])
        self.lns2 = nn.ModuleList([nn.LayerNorm(d, bias=False) for _ in range(n_layer)])
        self.mlps = nn.ModuleList([nn.Sequential(
            nn.Linear(d, 4 * d, bias=False), nn.GELU(), nn.Linear(4 * d, d, bias=False)
        ) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d, bias=False)
        self.head = nn.Linear(d, V, bias=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)) and m is not getattr(self, "lat", None):
                nn.init.normal_(m.weight, 0, 0.02)

    def precompute_coords(self, idx):
        return extract_think_coords(idx, self.stoi)

    def forward(self, idx, lat=None, coords=None):
        B, T = idx.shape
        tok_c, phase, q_col, k_col, line_id, q_step, k_step, step_u, flags = (
            coords if coords is not None else extract_think_coords(idx, self.stoi)
        )
        x = self.tok(tok_c) + self.phase_emb(phase) + self.su_emb(step_u) + self.flag_emb(flags)
        if self.lateral and lat is not None:
            x = x + self.lat(lat)
        causal = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), 1).unsqueeze(0)
        same_line = line_id.unsqueeze(2) == line_id.unsqueeze(1)
        mask = torch.where(same_line, causal, torch.full_like(causal, float("-inf"))).unsqueeze(1)
        for ln1, attn, ln2, mlp in zip(self.lns1, self.attns, self.lns2, self.mlps):
            x = x + attn(ln1(x), q_col, k_col, q_step, k_step, step_u, mask)
            x = x + mlp(ln2(x))
        return self.head(self.ln(x))


class Mind:
    def __init__(self, src, bb, si, cfg, cortex="column", col_cfg=None):
        self.src, self.bb, self.si, self.cfg = src, bb, si, cfg
        self.brain = rebuild_brain(src, bb)
        self.allc = src["chars"] + src.get("ext_chars", [])
        self.stoi = {c: i for i, c in enumerate(self.allc)}
        self.block = cfg["block"]
        self.V0 = len(src["chars"])
        self.col = None
        if src.get("think_column") is not None:
            cortex, col_cfg = "column", src["think_col_cfg"]
        self.cortex = cortex
        if cortex == "column":
            self.col_cfg = col_cfg or {}
            self.col = ThinkColumn(len(self.allc), self.block, stoi=self.stoi, **self.col_cfg)
            if src.get("think_column") is not None:
                self.col.load_state_dict(src["think_column"])
            for p in self.col.parameters():
                p.requires_grad = False

    def enc(self, s):
        return [self.stoi.get(c, 0) for c in s]

    @torch.no_grad()
    def lateral(self, idx):
        """Donuk omurga gizli durumu; omurganın bilmediği semboller boşluğa eşlenir (T > block ise sağdan hizalanır)."""
        sp = self.stoi[" "]
        B, T = idx.shape
        sub = idx[:, -self.block:]
        clean = torch.where(sub >= self.V0, torch.full_like(sub, sp), sub)
        feat = self.brain.frozen_features(clean)
        if T <= self.block:
            return feat
        pad = torch.zeros((B, T - self.block, feat.shape[-1]), dtype=feat.dtype, device=feat.device)
        return torch.cat([pad, feat], dim=1)

    def fwd(self, idx):
        if self.col is not None:
            return self.col(idx, self.lateral(idx) if self.col.lateral else None)
        return self.brain.forward_seg(idx, self.si)

    def params(self):
        return [p for p in (self.col or self.brain).parameters() if p.requires_grad]

    def n_params(self):
        if self.col is not None:
            return sum(p.numel() for p in self.col.parameters())
        return sum(p.numel() for p in self.brain.adapts[self.si].parameters())

    @torch.no_grad()
    def sample(self, ctx, K, temp, stops, max_new, greedy_first=True, ban=""):
        """K paralel devam; her satır kendi durdurucusunda kesilir. ban: seçimde yasak
        semboller (logp yine yasaksız dağılımdan: modelin gerçek inancı).
        Döner: [(metin, durdurucu|None, logp_toplam, token_logp_listesi)]."""
        self.brain.eval()
        if self.col is not None:
            self.col.eval()
        max_ctx = max(self.block, 256) if self.col is not None else self.block
        idx = torch.tensor([self.enc(ctx)], dtype=torch.long).repeat(K, 1)
        outs = [[] for _ in range(K)]
        lps = [[] for _ in range(K)]
        stop_ch = [None] * K
        done = [False] * K
        stop_ids = {self.stoi[c] for c in stops if c in self.stoi}
        ban_ids = [self.stoi[c] for c in ban if c in self.stoi]
        for _ in range(max_new):
            lo = self.fwd(idx[:, -max_ctx:])[:, -1, :]
            logp = FF.log_softmax(lo, -1)
            ls = lo.clone()
            if ban_ids:
                ls[:, ban_ids] = float("-inf")
            pr = FF.softmax(ls / max(temp, 1e-3), -1)
            nxt = torch.multinomial(pr, 1).squeeze(1)
            if greedy_first:
                nxt[0] = ls[0].argmax()
            for r in range(K):
                if done[r]:
                    continue
                t = int(nxt[r])
                if t in stop_ids:
                    done[r], stop_ch[r] = True, self.allc[t]
                else:
                    outs[r].append(self.allc[t])
                    lps[r].append(float(logp[r, t]))
            if all(done):
                break
            idx = torch.cat([idx, nxt.unsqueeze(1)], 1)
        return [("".join(outs[r]), stop_ch[r], sum(lps[r]), lps[r]) for r in range(K)]

    def intuition(self, a, op, b):
        """SİSTEM-1: doğrudan cevap + güven. '[' (düşünmeye kaçış) seçimde yasak; ama güven
        yasaksız olasılıktan ölçülür → modelin 'düşünmek istiyorum' eğilimi güveni düşürür."""
        txt, st, _, lps = self.sample(prompt_of(a, op, b), 1, 1.0, "\n", max(8, n_cols(a, b) + 4), ban="[")[0]
        m = re.match(r"\s*(-?\d+)", txt)
        if not m:
            return None, 0.0
        nd = len(m.group(0))
        conf = math.exp(sum(lps[:nd]) / max(1, nd))
        return int(m.group(1)), conf

    def deliberate(self, a, op, b, K=8, retries=3, watch=False, scaffold=False):
        """SİSTEM-2: adım adım düşün; her adım hakemden geçmeli. scaffold: dikkat işaretçisi
        adım önekini verir (nereye bakılır). Her adımda ayrıca İSKELESİZ greedy algı sondası
        alınır (model öneki kendi okuyabiliyor mu?) → iskele sökme ölçütü. Döner bölüm sözlüğü."""
        ctx = prompt_of(a, op, b) + " ["
        carry, partial, steps, samples, rejects = 0, "", [], 0, {}
        perc = [0, 0]
        n = n_cols(a, b)
        for k in range(n):
            ok_step = None
            pre = step_prefix(a, op, b, k, carry)
            probe = self.sample(ctx, 1, 1.0, "=|]\n", 8)[0][0] + "="
            perc[0] += probe == pre
            perc[1] += 1
            if watch:
                print(f"    adım {k + 1}/{n} algı sondası (iskelesiz): «{probe}» "
                      f"{'✓ sütunları kendi okudu' if probe == pre else '✗ beklenen ' + pre}")
            for att in range(retries):
                temp = 0.6 + 0.4 * att
                if scaffold:
                    cands = [(pre + t, st, lp, l) for t, st, lp, l in self.sample(ctx + pre, K, temp, "|]\n", max(12, n + 6))]
                else:
                    cands = self.sample(ctx, K, temp, "|]\n", max(18, n + 12))
                samples += K
                scored = []
                for txt, st, lp, _ in cands:
                    ok, why, nxt = verify_step(a, op, b, k, carry, partial, txt)
                    scored.append((ok, lp, txt, why, nxt))
                    if not ok:
                        rejects[why] = rejects.get(why, 0) + 1
                if watch:
                    seen = set()
                    print(f"    adım {k + 1}/{n} deneme {att + 1} (sıcaklık {temp:.1f}"
                          f"{', iskele: ' + pre if scaffold else ', iskelesiz'}):")
                    for ok, lp, txt, why, _ in sorted(scored, key=lambda t: -t[1]):
                        if txt in seen:
                            continue
                        seen.add(txt)
                        print(f"      {'✓' if ok else '✗'} {txt or '∅':<22} logp={lp:6.2f}" + ("" if ok else f"  [{why}]"))
                good = [s for s in scored if s[0]]
                if good:
                    ok_step = max(good, key=lambda s: s[1])
                    break
            if ok_step is None:
                return {"ok": False, "steps": steps, "samples": samples, "rejects": rejects, "fail_at": k,
                        "perc": perc}
            _, _, txt, _, (carry, partial) = ok_step
            steps.append(txt)
            ctx += txt + ("]" if k == n - 1 else "|")
        ans = int(partial)  # sonuç akıl yürütmeden çıkar
        tail, _, _, _ = self.sample(ctx, 1, 1.0, "\n", max(8, n + 4))[0]
        m = re.match(r"\s*(-?\d+)", tail)
        copy_ok = bool(m) and int(m.group(1)) == ans
        return {"ok": True, "ans": ans, "steps": steps, "samples": samples,
                "rejects": rejects, "copy_ok": copy_ok, "perc": perc}

    # ---- büyüme + uyku
    def grow(self, max_units):
        if self.col is not None:
            # Sütun EN YENİ modüldür: tamamı eğitilir; omurga + eski segmentler donuk.
            for p in self.col.parameters():
                p.requires_grad = True
            return False, "sütun"
        units = self.brain.adapts[self.si]
        for p in self.brain.parameters():
            p.requires_grad = False
        if len(units) < max_units:
            units.append(self.brain._make_adapter())
            grew = True
        else:
            grew = False
        for p in list(units[-1].parameters()) + list(self.brain.heads[self.si].parameters()):
            p.requires_grad = True
        if self.brain.n_extra:
            for p in list(self.brain.emb_ext.parameters()) + list(self.brain.head_ext.parameters()):
                p.requires_grad = True
        return grew, len(units)

    def train_lines(self, lines, iters, batch, lr, seed, log_every=50, win=64):
        opt = torch.optim.AdamW(self.params(), lr=lr)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters, pct_start=0.1)
        rng = torch.Generator().manual_seed(seed)
        (self.col or self.brain).train()
        t0, loss = time.time(), None
        if self.col is not None:
            enc = [self.enc(l + "\n")[:max(win, 160)] for l in lines]
            L = max(len(e) for e in enc)
            X = torch.zeros(len(enc), L, dtype=torch.long)
            M = torch.zeros(len(enc), L, dtype=torch.bool)
            Wt = torch.ones(len(enc), L)
            for i, (e, l) in enumerate(zip(enc, lines)):
                X[i, :len(e)] = torch.tensor(e, dtype=torch.long)
                eq = l.index("=") if "=" in l else 0
                M[i, eq:len(e) - 1] = True
                if "[" not in l:
                    Wt[i] = 1.2
            all_coords = self.col.precompute_coords(X[:, :-1])
            for it in range(1, iters + 1):
                ix = torch.randint(0, len(enc), (batch,), generator=rng)
                inp, tgt, m, wt = X[ix, :-1], X[ix, 1:], M[ix, :-1], Wt[ix, :-1]
                sub_c = tuple(t[ix] for t in all_coords)
                lat = self.lateral(inp) if self.col.lateral else None
                logits = self.col(inp, lat=lat, coords=sub_c)
                ce = FF.cross_entropy(logits[m], tgt[m], reduction="none")
                loss = (ce * wt[m]).sum() / wt[m].sum().clamp(min=1.0)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                if it % log_every == 0 or it == iters:
                    print(f"      iter {it}/{iters} kayıp={loss.item():.3f} {time.time() - t0:.0f}sn", flush=True)
        else:
            text = "\n".join(lines) + "\n"
            ids = torch.tensor(self.enc(text), dtype=torch.long)
            for it in range(1, iters + 1):
                x, y = batchify(ids, win, batch, rng)
                loss = FF.cross_entropy(self.fwd(x).view(-1, len(self.allc)), y.view(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                if it % log_every == 0 or it == iters:
                    print(f"      iter {it}/{iters} kayıp={loss.item():.3f} {time.time() - t0:.0f}sn", flush=True)
        (self.col or self.brain).eval()
        for p in list(self.brain.parameters()) + (list(self.col.parameters()) if self.col else []):
            p.requires_grad = False
        return loss.item()

    def save(self, path, state):
        b = self.brain
        out = dict(self.src)
        out.update({"adapts": [[a.state_dict() for a in seg] for seg in b.adapts],
                    "heads": [h.state_dict() for h in b.heads],
                    "emb_ext": b.emb_ext.state_dict() if b.n_extra else None,
                    "head_ext": b.head_ext.state_dict() if b.n_extra else None,
                    "apos": b.apos, "nested": True, "think_state": state,
                    "think_column": self.col.state_dict() if self.col else None,
                    "think_col_cfg": self.col_cfg if self.col else None})
        torch.save(out, path)


# ---------------------------------------------------------------- üstbiliş + merak
def calibrate(pairs, target, min_n=8):
    """En düşük τ: güven ≥ τ olan cevapların doğruluğu ≥ hedef. Yoksa inf (hep düşün)."""
    if len(pairs) < min_n:
        return float("inf")
    ps = sorted(pairs, key=lambda t: -t[0])
    best, hit = float("inf"), 0
    for i, (c, ok) in enumerate(ps, 1):
        hit += ok
        if i >= min_n and hit / i >= target:
            best = c
    return best


class Curiosity:
    """Öğrenme-ilerlemesi (Oudeyer): hızlı/yavaş başarı ortalaması farkı. Seviye n, n-1'de
    düşünce başarısı ≥ %50 olunca açılır. Sezgisi oturmuş (S1 ≥ %95) seviye sıkıcıdır."""

    def __init__(self, st=None, mode="auto", fade_at=0.8):
        self.st = st or {f"{op}{n}": {"fast": 0.0, "slow": 0.0, "s1": 0.0, "tries": 0} for op, n in LEVELS}
        for s in self.st.values():
            s.setdefault("perc", 0.0)  # iskelesiz algı (sütun okuma) başarısı
        self.mode, self.fade_at = mode, fade_at

    def scaffold(self, op, n):
        """İskele sökme: seviyenin algısı fade_at'i geçince dikkat işaretçisi kalkar."""
        if self.mode != "auto":
            return self.mode == "on"
        return self.st[f"{op}{n}"]["perc"] < self.fade_at

    def unlocked(self, op, n):
        return n == 1 or self.st[f"{op}{n - 1}"]["fast"] >= 0.5

    def pick(self, rng):
        keys, ws = [], []
        for op, n in LEVELS:
            if not self.unlocked(op, n):
                continue
            s = self.st[f"{op}{n}"]
            w = abs(s["fast"] - s["slow"]) + 0.05 + (0.5 if s["tries"] < 5 else 0.0)
            if s["s1"] > 0.95:
                w *= 0.2
            keys.append((op, n))
            ws.append(w)
        return rng.choices(keys, weights=ws)[0]

    def update(self, op, n, s2_ok, s1_ok, perc=None):
        s = self.st[f"{op}{n}"]
        s["tries"] += 1
        if perc and perc[1]:
            s["perc"] += 0.2 * (perc[0] / perc[1] - s["perc"])
        s["fast"] += 0.3 * (s2_ok - s["fast"])
        s["slow"] += 0.05 * (s2_ok - s["slow"])
        if s1_ok is not None:
            s["s1"] += 0.2 * (s1_ok - s["s1"])


# ---------------------------------------------------------------- sınav
def exam(mind, tau, per_level, K, retries, cur):
    rng = random.Random(4242)
    rows, tot = {}, {"n": 0, "s1": 0, "s2": 0, "gate": 0, "think": 0, "samp": 0, "copy": 0}
    for op, n in LEVELS:
        sc = cur.scaffold(op, n)
        r = {"n": 0, "s1": 0, "s2": 0, "p_ok": 0, "p_n": 0, "scaffold": sc}
        for _ in range(per_level):
            a, o, b = make_problem(rng, op, n)
            truth = answer_of(a, o, b)
            g, conf = mind.intuition(a, o, b)
            d = mind.deliberate(a, o, b, K, retries, scaffold=sc)
            s1 = g == truth
            s2 = d["ok"] and d["ans"] == truth
            think = conf < tau
            r["n"] += 1
            r["s1"] += s1
            r["s2"] += s2
            r["p_ok"] += d["perc"][0]
            r["p_n"] += d["perc"][1]
            tot["n"] += 1
            tot["s1"] += s1
            tot["s2"] += s2
            tot["think"] += think
            tot["gate"] += s2 if think else s1
            tot["samp"] += d["samples"] if think else 0
            tot["copy"] += bool(d.get("copy_ok"))
        rows[f"{op}{n}"] = r
    return rows, tot


def print_exam(day, rows, tot, tau):
    print(f"  SINAV gün {day}  (τ={tau:.3f})")
    print("    seviye  " + "  ".join(f"{k:>5}" for k in rows) + "   | basamak 3-4 öğretilmedi")
    print("    S1 %    " + "  ".join(f"{100 * r['s1'] / r['n']:5.0f}" for r in rows.values()))
    print("    S2 %    " + "  ".join(f"{100 * r['s2'] / r['n']:5.0f}" for r in rows.values()))
    print("    algı %  " + "  ".join(f"{100 * r['p_ok'] / max(1, r['p_n']):5.0f}" for r in rows.values())
          + "   | iskelesiz sütun okuma")
    print("    iskele  " + "  ".join(f"{'açık' if r['scaffold'] else 'yok':>5}" for r in rows.values()))
    n = tot["n"]
    print(f"    TOPLAM: S1 %{100 * tot['s1'] / n:.1f} | S2 %{100 * tot['s2'] / n:.1f} | "
          f"kapılı %{100 * tot['gate'] / n:.1f} | düşünme oranı %{100 * tot['think'] / n:.0f} | "
          f"soru-başı örnek {tot['samp'] / n:.1f} | sonuç-kopyası %{100 * tot['copy'] / n:.0f}", flush=True)


# ---------------------------------------------------------------- ana döngü
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(MODEL_DIR, "ab_block16.pt"), help="başlangıç beyni (AutoBrain kaydı)")
    ap.add_argument("--seg", default="M", help="düşünce segmenti etiketi")
    ap.add_argument("--out", default=os.path.join(MODEL_DIR, "think_brain.pt"))
    ap.add_argument("--resume", action="store_true", help="--out'taki beyin ve durumdan devam")
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--wake", type=int, default=40, help="gün-başı merak sorusu")
    ap.add_argument("--boot-iters", type=int, default=400, help="gün 0: öğretmen gösterimi eğitimi")
    ap.add_argument("--boot-n", type=int, default=3000, help="gün 0: öğretmen örnek sayısı (yalnızca 1-2 basamak)")
    ap.add_argument("--sleep-iters", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--cortex", choices=["column", "adapter"], default="column",
                    help="column: sıfırdan eğitilen yeni korteks sütunu (+yanal bağ); adapter: omurga üstü adaptör (eski)")
    ap.add_argument("--col-d", type=int, default=128)
    ap.add_argument("--col-layers", type=int, default=3)
    ap.add_argument("--no-lateral", action="store_true", help="sütun omurgayı okumasın")
    ap.add_argument("--k", type=int, default=8, help="adım-başı aday")
    ap.add_argument("--retries", type=int, default=3, help="adım-başı yeniden deneme (sıcaklık artar)")
    ap.add_argument("--target", type=float, default=0.95, help="üstbiliş: sezgiye güvenme doğruluk hedefi")
    ap.add_argument("--audit", type=float, default=0.2, help="öz-denetim: güvenli cevabı yine de düşünme olasılığı")
    ap.add_argument("--max-grow", type=int, default=5, help="segment adaptör tavanı (uyku başı büyüme)")
    ap.add_argument("--exam-per-level", type=int, default=10)
    ap.add_argument("--scaffold", choices=["auto", "on", "off"], default="auto",
                    help="dikkat işaretçisi: auto = seviye algısı %%80 olunca sökülür")
    ap.add_argument("--watch", default=None, help='tek soruyu canlı düşün: "347 + 589"')
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    rng = random.Random(args.seed)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    bb = MicroGPT(len(ck["chars"]), cfg["n_layer"], cfg["n_head"], cfg["n_embd"], cfg["block"])
    bb.load_state_dict(ck["state"])
    del ck

    src_path = args.out if (args.resume or args.watch) and os.path.exists(args.out) else args.base
    src = torch.load(src_path, map_location="cpu", weights_only=False)
    si = list(src["tags"]).index(args.seg) if args.seg in src["tags"] else 0
    mind = Mind(src, bb, si, cfg, args.cortex,
                {"d": args.col_d, "n_layer": args.col_layers, "lateral": not args.no_lateral})
    state = src.get("think_state") or {"day": 0, "tau": float("inf"), "calib": [], "memory": [],
                                         "curiosity": None, "history": []}
    cur = Curiosity(state["curiosity"], args.scaffold)
    print(f"beyin: {os.path.basename(src_path)} seg={src['tags'][si]} apos={mind.brain.apos} "
          f"korteks={mind.cortex} ({mind.n_params() / 1e3:.0f}k param) gün={state['day']} τ={state['tau']}")

    if args.watch:
        m = re.match(r"\s*(\d+)\s*([+\-])\s*(\d+)", args.watch)
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        if op == "-" and b > a:
            a, b = b, a
        g, conf = mind.intuition(a, op, b)
        tau = state["tau"]
        print(f"\nSORU: {prompt_of(a, op, b)}")
        print(f"  SİSTEM-1 (sezgi): {g}  güven={conf:.3f}  eşik τ={tau:.3f}")
        print(f"  ÜSTBİLİŞ: {'sezgiye güveniyorum' if conf >= tau else 'emin değilim → DÜŞÜNÜYORUM'}")
        d = mind.deliberate(a, op, b, args.k, args.retries, watch=True, scaffold=cur.scaffold(op, min(4, n_cols(a, b))))
        truth = answer_of(a, op, b)
        if d["ok"]:
            print(f"  SİSTEM-2 izi: {think_line(a, op, b, d['steps'], d['ans'])}")
            print(f"  sonuç {d['ans']} ({d['samples']} örnek)  |  gerçek {truth}  "
                  f"{'✓' if d['ans'] == truth else '✗'}  |  sezgi {'✓' if g == truth else '✗'}")
        else:
            print(f"  SİSTEM-2: adım {d['fail_at'] + 1}'de takıldı ({d['samples']} örnek), red nedenleri {d['rejects']}")
        return

    # ---- GÜN 0: öğretmen yalnızca 1-2 basamağı gösterir
    if state["day"] == 0:
        print("\n=== GÜN 0: öğretmen gösterimi (yalnızca 1-2 basamak) ===")
        teach = []
        trng = random.Random(7)
        for i in range(args.boot_n):
            op = OPS[i % 2]
            a, op, b = make_problem(trng, op, trng.randint(1, TEACH_MAX_DIGITS))
            teach.append(think_line(a, op, b, teacher_steps(a, op, b), answer_of(a, op, b)))
            if i % 4 == 0:
                teach.append(direct_line(a, op, b, answer_of(a, op, b)))
        state["teach_replay"] = teach[:400]
        grew, nu = mind.grow(args.max_grow)
        print(f"  büyüme: {nu}, {len(teach)} satır, örn: {teach[0]}")
        mind.train_lines(teach, args.boot_iters, args.batch, args.lr, args.seed)
        rows, tot = exam(mind, state["tau"], args.exam_per_level, args.k, args.retries, cur)
        print_exam(0, rows, tot, state["tau"])
        state["history"].append({"day": 0, "tau": None, "rows": rows, "tot": tot})
        state["day"] = 1
        state["curiosity"] = cur.st
        mind.save(args.out, state)

    # ---- GÜNLER: uyanıklık (merak + kapı + düşünme) → uyku (damıtma + büyüme) → sınav
    for _ in range(args.days):
        day = state["day"]
        print(f"\n=== GÜN {day}: uyanıklık ({args.wake} soru) ===")
        t0 = time.time()
        today, n_think, n_ok, n_s1, samp = [], 0, 0, 0, 0
        for q in range(args.wake):
            op, n = cur.pick(rng)
            a, op, b = make_problem(rng, op, n)
            g, conf = mind.intuition(a, op, b)
            think = conf < state["tau"] or rng.random() < args.audit
            if not think:
                n_s1 += 1  # sezgi cevabı: doğrulanmadı → öğrenme verisi değil
                continue
            n_think += 1
            d = mind.deliberate(a, op, b, args.k, args.retries, scaffold=cur.scaffold(op, n))
            samp += d["samples"]
            if d["ok"]:
                n_ok += 1
                s1_ok = g == d["ans"]
                state["calib"].append((conf, int(s1_ok)))
                today.append(think_line(a, op, b, d["steps"], d["ans"]))
                today.append(direct_line(a, op, b, d["ans"]))  # düşünce → sezgi hedefi
                state["memory"].append([a, op, b])
            else:
                s1_ok = None
            cur.update(op, n, int(d["ok"]), s1_ok, d["perc"])
        state["calib"] = state["calib"][-400:]
        lv = {k: f"{v['fast']:.2f}" for k, v in cur.st.items() if cur.unlocked(k[0], int(k[1:]))}
        print(f"  düşündü {n_think}, doğruladı {n_ok}, sezgiyle geçti {n_s1}, örnek {samp}, "
              f"{time.time() - t0:.0f}sn | açık seviyeler (S2 başarı): {lv}")

        print(f"=== GÜN {day}: uyku (damıtma) ===")
        # Eski günlerin doğrulanmış düşünceleri tekrar edilir (hakem yalnızca kanonik izi
        # kabul ettiğinden kayıtlı iz = teacher_steps; cevap da izden türemiştir).
        replay = []
        if state["memory"]:
            for a, op, b in rng.sample(state["memory"], min(len(state["memory"]), max(40, len(today)))):
                ans = answer_of(a, op, b)
                replay += [think_line(a, op, b, teacher_steps(a, op, b), ans), direct_line(a, op, b, ans)]
        lines = today * 2 + replay + rng.sample(state["teach_replay"], min(200, len(state["teach_replay"])))
        rng.shuffle(lines)
        grew, nu = mind.grow(args.max_grow)
        print(f"  {'yeni adaptör' if grew else 'tavan: son adaptör'} ({nu}), {len(lines)} satır "
              f"(bugün {len(today)}, tekrar {len(replay)})")
        mind.train_lines(lines, args.sleep_iters, args.batch, args.lr, args.seed + day)

        state["tau"] = calibrate(state["calib"], args.target)
        rows, tot = exam(mind, state["tau"], args.exam_per_level, args.k, args.retries, cur)
        print_exam(day, rows, tot, state["tau"])
        state["history"].append({"day": day, "tau": state["tau"], "rows": rows, "tot": tot,
                                 "wake": {"think": n_think, "ok": n_ok, "s1": n_s1, "samples": samp}})
        state["day"] = day + 1
        state["curiosity"] = cur.st
        mind.save(args.out, state)
        json.dump(state["history"], open(os.path.join(PROJECT_ROOT, "runs", "think_history.json"), "w"),
                  ensure_ascii=False, indent=1, default=str)

    print("\nÖZET (gün: S1 / S2 / kapılı / düşünme oranı)")
    for h in state["history"]:
        t = h["tot"]
        print(f"  gün {h['day']}: S1 %{100 * t['s1'] / t['n']:.0f} / S2 %{100 * t['s2'] / t['n']:.0f} / "
              f"kapılı %{100 * t['gate'] / t['n']:.0f} / düşünme %{100 * t['think'] / t['n']:.0f}")


if __name__ == "__main__":
    main()
