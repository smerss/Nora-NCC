#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""grow_gpt.py — GrowingNet'in transformer karşılığı: omurga DONUK, bilgi EKLEMEYLE gelir.
(a) adaptör: darboğaz 256→64→256 (sıfır-init rezidüel = başlangıçta identity) + yeni head (~50k, %1.5)
(b) taban çizgisi: full-finetune (unutmayı göstermek için)
Metrik: F = L_A(sonra) - L_A(önce); F<0.05 sıfır unutma. Kazanma: F_ad<0.05 ve F_full-F_ad>0.10 ve B farkı<0.05.
Tümü D diskinde. Kullanim (DPY = D venv python):
  DPY grow_gpt.py --stage adapt --iters 200
  DPY grow_gpt.py --stage full --iters 200
"""
import argparse, os, re, sys, time
sys.path.insert(0, os.path.join("D:" + os.sep, "NoRa_AGI") if os.name == "nt" else os.path.join(os.path.expanduser("~"), "NoRa_AGI"))
from micro_gpt import MicroGPT, batchify

import torch
import torch.nn as nn
import torch.nn.functional as FF

PROJECT_ROOT = os.path.join("D:" + os.sep, "NoRa_AGI") if os.name == "nt" else os.path.join(os.path.expanduser("~"), "NoRa_AGI")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
SA = os.path.join(PROJECT_ROOT, "data", "text", "tiny_slice.txt")
SB = os.path.join(PROJECT_ROOT, "data", "text", "tiny_sliceB.txt")
CKPT = os.path.join(MODEL_DIR, "micro_tiny.pt")


class GrowGPT(nn.Module):
    """Donuk omurga + İSTİFLENEBİLİR adaptörler + B kafası. Kural: yalnızca EN YENİ
    adaptör eğitilir, eskiler donar (GrowingNet blok kuralının transformer hali)."""

    def __init__(self, backbone: MicroGPT, bn=64):
        super().__init__()
        self.bb = backbone
        self.bn = bn
        for p in self.bb.parameters():
            p.requires_grad = False
        e = self.bb.head.in_features
        self.adapts = nn.ModuleList()
        self.headB = nn.Linear(e, self.bb.head.out_features, bias=False)
        self.headB.weight.data.copy_(self.bb.head.weight.data)  # transfer: eskiden başla
        self.add_adapter()

    def _make_one(self):
        e = self.bb.head.in_features
        a = nn.Sequential(nn.Linear(e, self.bn), nn.GELU(), nn.Linear(self.bn, e))
        nn.init.zeros_(a[-1].weight)
        nn.init.zeros_(a[-1].bias)
        return a

    def add_adapter(self):
        """YENİ adaptör: eskileri dondur, sıfır-init ile başla (ilk adım identity)."""
        for a in self.adapts:
            for p in a.parameters():
                p.requires_grad = False
        self.adapts.append(self._make_one())
        return len(self.adapts)

    def features(self, idx):
        B, T = idx.shape
        mask = torch.triu(torch.full((T, T), float("-inf")), 1)
        x = self.bb.tok(idx) + self.bb.pos(torch.arange(T))
        for l in self.bb.layers:
            x = l(x, mask)
        return self.bb.ln(x)

    def forward_B(self, idx):
        idx = idx.long()
        h = self.features(idx)
        for a in self.adapts:
            h = h + a(h)
        return self.headB(h)

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@torch.no_grad()
def evalloss(fn, ids, chars, block, batch, n=20):
    tot = 0.0
    rng = torch.Generator().manual_seed(0)
    for _ in range(n):
        ix = torch.randint(0, len(ids) - block - 1, (batch,), generator=rng)
        x = torch.stack([ids[i:i + block] for i in ix]).long()
        y = torch.stack([ids[i + 1:i + block + 1] for i in ix]).long()
        lo = fn(x)
        tot += FF.cross_entropy(lo.view(-1, lo.size(-1)), y.view(-1)).item()
    return tot / n


class AutoBrain(nn.Module):
    """KAPALI DÖNGÜ beyni: donuk omurga + segment-başı (adaptör, kafa) çiftleri.
    Kural (GrowingNet): yalnızca EN YENİ çift eğitilir; eskiler donar, unutma sıfırdır.
    Yönlendirici: her parti için en düşük kayıp veren kafa (1-adım zihinsel yoklama).
    apos: 'head' = adaptör yalnızca son gizli durumda (eski; dikkate dokunamaz),
          'block' = her bloğun çıkışında bir adaptör (Houlsby-tarzı; sonraki katmanların
          dikkati de değişir). Bir 'adaptör birimi' block kipinde katman-başı listedir."""

    def __init__(self, backbone: MicroGPT, n_extra=0, bn=64, apos="head"):
        super().__init__()
        self.bb = backbone
        self.bn = bn
        self.apos = apos
        for p in self.bb.parameters():
            p.requires_grad = False
        self.adapts = nn.ModuleList()
        self.heads = nn.ModuleList()
        # YENİ SEMBOLLER = YENİ NÖRONLAR: genişletilmiş gömme/kafa satırları (paylaşılan
        # sembol kapasitesi; her zaman eğitilebilir — felsefi sapma değil, alfabe büyümesidir).
        self.n_extra = n_extra
        if n_extra:
            e = self.bb.head.in_features
            self.emb_ext = nn.Embedding(n_extra, e)
            nn.init.normal_(self.emb_ext.weight, 0, 0.02)
            self.head_ext = nn.Linear(e, n_extra, bias=False)
            nn.init.normal_(self.head_ext.weight, 0, 0.02)

    @property
    def V_total(self):
        return self.bb.head.out_features + self.n_extra

    def _make_one(self):
        e = self.bb.head.in_features
        a = nn.Sequential(nn.Linear(e, self.bn), nn.GELU(), nn.Linear(self.bn, e))
        nn.init.zeros_(a[-1].weight)
        nn.init.zeros_(a[-1].bias)
        return a

    def _make_adapter(self):
        """Bir adaptör birimi: head kipinde tek darboğaz, block kipinde katman-başı liste."""
        if self.apos == "block":
            return nn.ModuleList([self._make_one() for _ in self.bb.layers])
        return self._make_one()

    def _make_head(self):
        e, V = self.bb.head.in_features, self.bb.head.out_features
        h = nn.Linear(e, V, bias=False)
        h.weight.data.copy_(self.bb.head.weight.data)
        return h

    def _make_pair(self):
        return self._make_adapter(), self._make_head()

    def open_segment(self):
        for a in self.adapts:
            for p in a.parameters():
                p.requires_grad = False
        for h in self.heads:
            for p in h.parameters():
                p.requires_grad = False
        a, h = self._make_pair()
        self.adapts.append(nn.ModuleList([a]))
        self.heads.append(h)
        return len(self.adapts)

    def grow_current(self):
        """Segment-içi BÜYÜME (doğrusu): AYNI kafanın altına yeni adaptör istifle.
        Eskiler donar, kafa korunur — öğrenme sıfırlanmaz (yeni-kafa açmak hataydı)."""
        for p in self.adapts[-1].parameters():
            p.requires_grad = False
        self.adapts[-1].append(self._make_adapter())
        return len(self.adapts[-1])

    @torch.no_grad()
    def frozen_features(self, idx):
        """Donuk omurga gizli durumu (şaşkınlık-vektörü için; adaptörsüz)."""
        idx = idx.long()
        B, T = idx.shape
        mask = torch.triu(torch.full((T, T), float("-inf")), 1)
        V0 = self.bb.head.out_features
        x = self.bb.tok(torch.clamp(idx, 0, V0 - 1)) + self.bb.pos(torch.arange(T))
        for l in self.bb.layers:
            x = l(x, mask)
        return self.bb.ln(x)

    def forward_seg(self, idx, s):
        idx = idx.long()
        B, T = idx.shape
        mask = torch.triu(torch.full((T, T), float("-inf")), 1)
        V0 = self.bb.head.out_features
        if self.n_extra:
            nov = (idx >= V0)
            x = self.bb.tok(torch.clamp(idx, 0, V0 - 1)) + self.bb.pos(torch.arange(T))
            x = x + self.emb_ext(torch.clamp(idx - V0, 0, self.n_extra - 1)) * nov.unsqueeze(-1)
        else:
            x = self.bb.tok(idx) + self.bb.pos(torch.arange(T))
        blk = self.apos == "block"
        for li, l in enumerate(self.bb.layers):
            x = l(x, mask)
            if blk:
                for u in self.adapts[s]:
                    x = x + u[li](x)
        h = self.bb.ln(x)
        if not blk:
            for a in self.adapts[s]:
                h = h + a(h)
        lo = self.heads[s](h)
        if self.n_extra:
            lo = torch.cat([lo, self.head_ext(h)], dim=-1)
        return lo

    @torch.no_grad()
    def route(self, x, y):
        """En düşük kayıplı segment. Döner: (seg_idx, loss)."""
        best, bl = -1, float("inf")
        for s in range(len(self.heads)):
            l = FF.cross_entropy(self.forward_seg(x, s).view(-1, self.V_total), y.view(-1)).item()
            if l < bl:
                best, bl = s, l
        return best, bl

def resolve_seg(sel, tags):
    try:
        i = int(sel)
        return i if 0 <= i < len(tags) else 0
    except (ValueError, TypeError):
        return tags.index(sel) if sel in tags else 0


ARITH_RE = re.compile(r"(\d+)\s*([+\-*])\s*(\d+)")


def parse_arith(text):
    m = ARITH_RE.search(text)
    return (int(m.group(1)), m.group(2), int(m.group(3))) if m else None


def arith_result(a, op, b):
    return a + b if op == "+" else (a - b if op == "-" else a * b)


REV_TRACE = False  # --rev-trace: izdeki sonuç ters yazılır (make_think*.py --reverse ile uyumlu)


def trace_num(s):
    """İz içindeki sonuç dizgesini sayıya çevir (ters kipte önce çevrilir)."""
    return int(s[::-1] if REV_TRACE else s)


def expected_trace(a, op, b):
    r = arith_result(a, op, b)
    return " [%d %s %d = %s] %d" % (a, op, b, str(r)[::-1] if REV_TRACE else r, r)


def rebuild_brain(src, bb):
    """Kayıtlı beyni AYNI mimariyle dirilt (iç içe adaptörler + ext dahil); hepsi donuk döner."""
    n_ext = len(src.get("ext_chars", []))
    brain = AutoBrain(bb, n_ext, src.get("bn", 64), src.get("apos", "head"))
    raw = src["adapts"] if src.get("nested") else [[s] for s in src["adapts"]]
    for seg_sds, h_sd in zip(raw, src["heads"]):
        brain.open_segment()
        brain.adapts[-1] = nn.ModuleList()
        for a_sd in seg_sds:
            na = brain._make_adapter()
            na.load_state_dict(a_sd)
            brain.adapts[-1].append(na)
        brain.heads[-1].load_state_dict(h_sd)
    if n_ext:
        brain.emb_ext.load_state_dict(src["emb_ext"])
        brain.head_ext.load_state_dict(src["head_ext"])
    for p in brain.parameters():
        p.requires_grad = False
    return brain


def load_ids(path, stoi, nchars=None):
    """Akışlı kodlayıcı: dev Python-int listesi YOK (MemoryError koruması).
    Alfabe latin-1 ise bayt-tablosu hızlı yolu, yoksa parçalı doldurma.
    nchars verilirse dosyanın yalnızca başı okunur (çapa/plan için yeterli)."""
    import numpy as _np
    t = open(path, encoding="utf-8").read(nchars) if nchars else open(path, encoding="utf-8").read()
    n = len(t)
    unseen = sorted(set(t) - set(stoi)) if n < 5000000 else []
    if max(stoi.values()) < 256:
        try:
            raw = t.encode("latin-1")
            del t
            lut = _np.full(256, stoi.get(" ", 0), dtype=_np.int64)
            for b in range(256):
                lut[b] = stoi.get(chr(b), stoi.get(" ", 0))
            ids = torch.from_numpy(lut[_np.frombuffer(raw, dtype=_np.uint8)])
            del raw
            return ids, unseen  # uint8: V<256 iken 4x küçük commit; tüketimde .long()
        except UnicodeEncodeError:
            t = open(path, encoding="utf-8").read(nchars) if nchars else open(path, encoding="utf-8").read()
    n = len(t)
    if max(stoi.values()) >= 256:  # dev alfabe: hızlı yol kapalı, klasik long
        ids = torch.tensor([stoi.get(x, stoi.get(" ", 0)) for x in t], dtype=torch.long)
        del t
        return ids, unseen
    ids = torch.empty(n, dtype=torch.uint8)
    st, sp = stoi.get(" ", 0), stoi
    for i in range(0, n, 200000):  # küçük parça: commit darlığında liste şişmesin
        c = t[i:i + 200000]
        ids[i:i + len(c)] = torch.tensor([sp.get(x, st) for x in c], dtype=torch.uint8)
        del c
    del t
    return ids, unseen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["adapt", "full", "detect", "auto", "sleep", "refine", "think", "rft", "evalarith"], required=True,
                    help="detect: etiketsiz karışık akışta dağılım sınırlarını şaşkınlıkla bul (eğitim yok); "
                         "auto: KAPALI DÖNGÜ — detect öttükçe adaptör+kafa aç, eğit, yönlendir (insan yok); "
                         "sleep: UYKU — benzer segmentleri kaynaştır (işlevsel eşdeğerlik), yönlendiriciyi koru; "
                         "refine: İNCE AYAR — kayıtlı beynin bir segmentine kapasite istifle, kaldığı yerden eğit; "
                         "think: SYSTEM-2 — üret-ve-test muhakeme döngüsü (yavaş düşünme); "
                         "rft: araç-doğrulanmış izlerle ince-eğitim (reddetme-örneklemesi); "
                         "evalarith: sabit 200 soruda greedy iz/cevap doğruluğu (ölçüm, eğitim yok)")
    ap.add_argument("--max-adapt", type=int, default=3, help="en fazla adaptör sayısı (büyüme tavanı)")
    ap.add_argument("--plateau-eps", type=float, default=0.02, help="bu iyileşmenin altı = plato")
    ap.add_argument("--plateau-w", type=int, default=3, help="plato penceresi (B-val sayısı)")
    ap.add_argument("--bdata", default=SB, help="B dilimi (varsayılan TinyStories-B; dagilim kaymasi icin tinyshakespeare verilebilir)")
    ap.add_argument("--pool", choices=["nar", "codemath", "three"], default="nar",
                    help="nar: anlatı akışı [A/B/S]; codemath: [A/kod/matematik]; three: [kod/mat/Türkçe]")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32, help="eğitim partisi (commit darlığında 16 verin)")
    ap.add_argument("--adapt-bn", type=int, default=64, help="adaptör darboğazı (felsefe: omurga donuk, kapasite yalnızca yeni-nöronlarda)")
    ap.add_argument("--adapt-pos", choices=["head", "block"], default="block",
                    help="auto: adaptör konumu — block: her blok çıkışında (dikkate etki eder), head: yalnızca sonda (eski)")
    ap.add_argument("--out", default=None, help="auto: beyin kayıt yolu (varsayılan models/grow_auto.pt)")
    ap.add_argument("--seg-iters", type=int, default=80, help="kapalı-döngüde segment-başı eğitim iterasyonu")
    ap.add_argument("--max-seg-adapt", type=int, default=2,
                    help="segment-başı tavan (plato-büyümesiyle aynı-etiketli devam segmentleri; uyku birleştirir)")
    ap.add_argument("--surprise-kf", type=float, default=2.5,
                    help="vektör-şaşkınlık eşiği: min(uzaklık/ADAY-yayılımı) > kf (M 6, T ~3, aynı ~1; böl-uykuda-birleştir)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seg", default="C", help="refine/think hedefi: etiket (C/M/T/S) veya indeks")
    ap.add_argument("--prompt", default="12 + 34 =", help="think: başlangıç bağlamı")
    ap.add_argument("--topk", type=int, default=6, help="think: tur-başı aday sayısı")
    ap.add_argument("--cand-len", type=int, default=8, help="think: aday uzunluğu (token)")
    ap.add_argument("--steps", type=int, default=2, help="think: düşünme turu")
    ap.add_argument("--tool", action="store_true",
                    help="think: Şaşkınlık Kapısı + hesap-aracı (aritmetikte model yerine araç doğrular)")
    ap.add_argument("--brain", default=os.path.join(MODEL_DIR, "grow_auto.pt"), help="refine/think: kaynak beyin")
    ap.add_argument("--data", default=None, help="refine: veri dosyası (varsayılan etikete göre seçilir)")
    ap.add_argument("--rft-rounds", type=int, default=3, help="rft: örnekle-filtrele-eğit turu")
    ap.add_argument("--rft-k", type=int, default=6, help="rft: prompt-başı aday")
    ap.add_argument("--rft-n", type=int, default=24, help="rft: tur-başı prompt sayısı")
    ap.add_argument("--rft-iters", type=int, default=30, help="rft: tur-başı eğitim iterasyonu")
    ap.add_argument("--rft-max", type=int, default=99, help="rft: operand tavanı (soğuk-başlatma için 9 verin)")
    ap.add_argument("--rev-trace", action="store_true",
                    help="think/rft: izdeki sonuç TERS (make_think*.py --reverse korpusuyla eğitilmiş beyin)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    global REV_TRACE
    REV_TRACE = args.rev_trace
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    chars, cfg = ck["chars"], ck["cfg"]
    stoi = {c: i for i, c in enumerate(chars)}
    bb = MicroGPT(len(chars), cfg["n_layer"], cfg["n_head"], cfg["n_embd"], cfg["block"])
    bb.load_state_dict(ck["state"])
    V = len(chars)
    import gc as _gc  # ck tutuluyor (full/adapt dalları kullanır); ihtiyaçta dellenebilir
    _gc.collect()

    # Commit darlığı (limit 18GB, boş ~0.7GB): her kip yalnızca İHTİYACINI yükler.
    # A kapağı: adapt/full tam (val için), detect 100k, auto-nar 300k, auto-codemath 100k, sleep 300k.
    if args.stage in ("adapt", "full"):
        A_cap = None
    elif args.stage == "auto" and args.pool == "nar":
        A_cap = 300000
    elif args.stage == "sleep":
        A_cap = 300000
    else:
        A_cap = 100000
    A, unA = load_ids(SA, stoi, A_cap)
    # B tembel yüklenir (yalnızca gerçekten kullanan kipte): 42M'luk çift yükleme yok.
    need_b = args.stage in ("adapt", "full") or (args.stage == "detect" and args.pool == "nar")
    B, unB = (load_ids(args.bdata, stoi, 100000 if args.stage == "detect" else None) if need_b else (torch.empty(0, dtype=torch.long), []))
    print(f"A: {len(A)/1e6:.1f}M char" + (f", B: {len(B)/1e6:.1f}M char" if need_b else ", B: atlandı")
          + f", görülmemiş: A={len(unA)} B={len(unB)}")
    CODE_TXT = os.path.join(PROJECT_ROOT, "data", "text", "code_py.txt")
    MATH_TXT = os.path.join(PROJECT_ROOT, "data", "text", "math_synth.txt")
    TURK_TXT = os.path.join(PROJECT_ROOT, "data", "text", "turkish.txt")
    ext_chars, stoi2 = [], dict(stoi)
    if args.pool in ("codemath", "three"):
        # ALFABE BÜYÜMESİ: kod/matematik/Türkçedeki yeni semboller için yeni nöron satırları.
        for p in (CODE_TXT, MATH_TXT, TURK_TXT):
            if os.path.exists(p):
                for c in sorted(set(open(p, encoding="utf-8").read()) - set(stoi2)):
                    stoi2[c] = len(stoi2)
                    ext_chars.append(c)
        print(f"alfabe büyümesi: +{len(ext_chars)} sembol ({len(stoi2)} toplam)")

    if args.stage == "detect":
        # v4-dil: [A,A,B,B,S,S,A,B] akışı; gerçek kayma 4 ve 6'da (S giriş/çıkış).
        # 2'deki A→B aynı-dağılım: ÖTMEMELİ (özgüllük testi). Eğitim yok, yalnızca şaşkınlık.
        S, _ = load_ids(os.path.join(PROJECT_ROOT, "data", "text", "tinyshakespeare.txt"), stoi, 100000)
        if args.pool == "codemath":
            C0, unC = load_ids(CODE_TXT, stoi, 100000)
            M0, unM = load_ids(MATH_TXT, stoi, 100000)
            print(f"  (alt-sınır notu: eşlenen sembol C={len(unC)} M={len(unM)})")
        CH = 20000
        # Yenilik PARÇA-başınadır (geçiş-başına değil): S hiç öğrenilmedi -> 4 VE 5
        # ikisi de yenidir; 7 B≈A olduğundan sessizlik DOĞRUDUR (özgüllük).
        plan = [("A", A, False), ("A", A, False), ("B", B, False), ("B", B, False),
                ("S", S, True), ("S", S, True), ("A", A, True), ("B", B, False)]
        exp = [4, 5, 6]
        if args.pool == "codemath":
            plan = [("A", A, False), ("A", A, False), ("C", C0, True), ("C", C0, True),
                    ("M", M0, True), ("M", M0, True), ("A", A, True)]
            exp = [2, 3, 4, 5, 6]
        offs = {"A": 0, "B": 0, "S": 0, "C": 0, "M": 0}
        NB = 8
        mus = sig = p90r = sig90 = prev = None
        hits, fires = 0, []
        for i, (tag, src, should) in enumerate(plan):
            o = offs[tag] % (len(src) - CH - 1)
            offs[tag] += CH
            seg = src[o:o + CH]
            bl = []
            rg = torch.Generator().manual_seed(i)
            for _ in range(NB):
                s0 = int(torch.randint(0, CH - cfg["block"] - 1, (1,), generator=rg))
                x = seg[s0:s0 + cfg["block"]].unsqueeze(0).repeat(32, 1).long()
                y = seg[s0 + 1:s0 + cfg["block"] + 1].unsqueeze(0).repeat(32, 1).long()
                with torch.no_grad():
                    l = FF.cross_entropy(bb(x).view(-1, V), y.view(-1)).item()
                bl.append(l)
            import numpy as _np
            mu_c, p90_c = float(_np.mean(bl)), float(_np.quantile(bl, 0.9))
            if i == 0:  # referans çapa: pekleşmiş bilgi (A)
                mus, sig = mu_c, max(float(_np.std(bl)), 0.05 * mu_c)
                p90r, sig90 = p90_c, max(float(_np.std(bl)), 0.05 * p90_c)
                prev, tag2 = mu_c, "ÇAPA"
            else:
                lvl = (mu_c > mus + 2.5 * sig) or (p90_c > p90r + 2.5 * sig90)
                chg = abs(mu_c - prev) > 1.5 * sig
                tag2 = "SINIR" if (lvl or chg) else "tanidik"
                if tag2 == "SINIR":
                    fires.append(i)
                    if should:
                        hits += 1
                prev = mu_c
            print(f"  parça {i} [{tag}]: ort={mu_c:.3f} p90={p90_c:.3f} -> {tag2} (beklenen: {'SINIR' if should else 'sessiz'})")
        fp = [f for f in fires if f not in exp]
        fn = [e for e in exp if e not in fires]
        print(f"\nDETECT: beklenen {exp}, bulunan {fires} | isabet {hits}/{len(exp)}, "
              f"yanlış-alarm {fp}, kaçırma {fn}")
        return

    if args.stage == "auto":
        # KAPALI DÖNGÜ: [S,S,S,S,A,A] akışı (100k'lık parçalar), etiket YOK.
        # detect öttükçe yeni (adaptör,kafa) açılır; ötmeyince aynı segment birikir.
        # Sonunda: segment-val kayıpları + etiketsiz yönlendirici isabeti. İnsan yok.
        print("\n=== KAPALI DÖNGÜ (detect->büyüt->eğit, insansız) ===")
        V0, SP = len(chars), stoi[" "]
        NB = 8
        import numpy as _np

        def base_view(t):
            # Şaşkınlık donuk omurgayla ölçülür (95 sembol); yeni semboller boşluğa
            # eşlenir (alt-sınır; eğitim tam-alfabeyle yapılır).
            return torch.where(t >= V0, torch.full_like(t, SP), t)

        if args.pool == "nar":
            S, _ = load_ids(os.path.join(PROJECT_ROOT, "data", "text", "tinyshakespeare.txt"), stoi)
            CH, SEG_ITERS = 100000, args.seg_iters
            stream = [("S", S[i:i + CH]) for i in range(0, 400000, CH)]
            stream += [("A", A[i:i + CH]) for i in range(0, 200000, CH)]
        elif args.pool == "codemath":
            C0e, _ = load_ids(CODE_TXT, stoi2)
            M0e, _ = load_ids(MATH_TXT, stoi2)
            CH, SEG_ITERS = 100000, args.seg_iters
            stream = [("C", C0e[i:i + CH]) for i in range(0, 400000, CH)]
            stream += [("M", M0e[i:i + CH]) for i in range(0, 200000, CH)]
            print(f"  akış: {len(stream)} parça [C×4, M×2] (şaşkınlık alt-sınır, eğitim geniş alfabe)")
        else:
            C0e, _ = load_ids(CODE_TXT, stoi2)
            M0e, _ = load_ids(MATH_TXT, stoi2)
            T0e, _ = load_ids(TURK_TXT, stoi2)
            CH, SEG_ITERS = 100000, args.seg_iters
            stream = [("C", C0e[i:i + CH]) for i in range(0, 300000, CH)]
            stream += [("M", M0e[i:i + CH]) for i in range(0, 200000, CH)]
            stream += [("T", T0e[i:i + CH]) for i in range(0, min(len(T0e), 200000), CH)]
            print(f"  akış: {len(stream)} parça [C×3, M×2, T×?] (Türkçe sondaj, geniş alfabe)")
        rg0 = torch.Generator().manual_seed(99)
        bl0 = []
        for _ in range(NB):
            s0 = int(torch.randint(0, len(A) - cfg["block"] - 1, (1,), generator=rg0))
            x = A[s0:s0 + cfg["block"]].unsqueeze(0).repeat(32, 1).long()
            y = A[s0 + 1:s0 + cfg["block"] + 1].unsqueeze(0).repeat(32, 1).long()
            with torch.no_grad():
                bl0.append(FF.cross_entropy(bb(x).view(-1, V), y.view(-1)).item())
        mu, sig = float(_np.mean(bl0)), max(float(_np.std(bl0)), 0.05 * float(_np.mean(bl0)))
        anchors = [(mu, sig)]  # pekleşen her bilgi çapaya EKLENİR; yenilik=hiçbirine benzememe
        print(f"  referans çapa (A): mu={mu:.3f} sig={sig:.3f}")

        def feat_mean(ten, seed, nw=4):
            # Şaşkınlık-VEKTÖRÜ: donuk omurgada gizli-ortalama + yayılım.
            rgf = torch.Generator().manual_seed(seed)
            vs = []
            Bw, Tw = 8, cfg["block"]
            tb = base_view(ten)
            for _ in range(nw):
                s0 = int(torch.randint(0, len(tb) - Tw - 1, (1,), generator=rgf))
                x = tb[s0:s0 + Tw].unsqueeze(0).repeat(Bw, 1).long()
                with torch.no_grad():
                    mask = torch.triu(torch.full((Tw, Tw), float("-inf")), 1)
                    h = bb.tok(x) + bb.pos(torch.arange(Tw))
                    for l in bb.layers:
                        h = l(h, mask)
                    vs.append(bb.ln(h).mean(dim=(0, 1)))
            Mm = torch.stack(vs)
            return Mm.mean(0), float(((Mm - Mm.mean(0)) ** 2).sum(1).mean())

        pA, sA = feat_mean(A, 999)
        seg_protos = [(pA, sA)]  # vektör çapalar: [A-prototip]; eğitimle büyür
        print(f"  vektör çapa (A): yayılım={sA:.2f}")
        if args.pool == "codemath":
            # A bu akışta bir daha KULLANILMAZ: commit iadesi (168MB tensor + artıklar).
            import gc as _gc
            del A
            _gc.collect()
        brain = AutoBrain(bb, len(ext_chars), args.adapt_bn, args.adapt_pos)
        acc, acc_tags, s_prev, seg_tags = [], [], None, []
        t0 = time.time()

        def train_seg(data, si):
            # Segment-içi PLATO-BÜYÜMESİ: 50'lik raundlar, held-out val izlenir; plato +
            # tavan varsa AYNI etiketle devam segmenti açılır (felsefe: kapasite sayıda
            # aranır; fazlalık varsa uyku birleştirir). Bütçe: SEG_ITERS + büyüme başına +50.
            T = torch.cat(data)
            tr, vl = T[:int(.9 * len(T))], T[int(.9 * len(T)):]
            rng = torch.Generator().manual_seed(si)
            vals, budget = [], SEG_ITERS
            it = 0
            while it < budget:
                opt = torch.optim.AdamW([p for p in brain.parameters() if p.requires_grad], lr=args.lr)
                for _ in range(min(25, budget - it)):
                    x, y = batchify(tr, cfg["block"], args.batch, rng)
                    loss = FF.cross_entropy(brain.forward_seg(x, si).view(-1, brain.V_total), y.view(-1))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    it += 1
                    if it % 10 == 0:
                        print(f"    seg{si} iter {it} loss={loss.item():.3f}", flush=True)
                vals.append(evalloss(lambda x, s=si: brain.forward_seg(x, s), vl, chars, cfg["block"], 8, n=6))
                print(f"  segment {si} ({seg_tags[si]}) raund-sonu val={vals[-1]:.3f} adaptör={len(brain.adapts[si])}", flush=True)
                if (len(vals) >= 3 and (vals[-3] - vals[-1]) < args.plateau_eps
                        and len(brain.adapts[si]) < args.max_seg_adapt and it >= budget - 25):
                    k = brain.grow_current()
                    budget += 50
                    print(f"  PLATO segment-içi -> seg{si} altına adaptör {k} istiflendi (tavan {args.max_seg_adapt})")
            print(f"  segment {si} ({seg_tags[si]}) eğitildi: {it} iter, son train={loss.item():.3f}")
            # ÇAPA GENİŞLETME: bu bilgi pekleşti; şaşkınlık imzası çapaya eklenir.
            Tt = torch.cat(data)
            rgc = torch.Generator().manual_seed(si + 1000)
            blc = []
            for _ in range(NB):
                s0 = int(torch.randint(0, len(Tt) - cfg["block"] - 1, (1,), generator=rgc))
                x = base_view(Tt[s0:s0 + cfg["block"]]).unsqueeze(0).repeat(32, 1).long()
                y = base_view(Tt[s0 + 1:s0 + cfg["block"] + 1]).unsqueeze(0).repeat(32, 1).long()
                with torch.no_grad():
                    blc.append(FF.cross_entropy(bb(x).view(-1, V), y.view(-1)).item())
            m2 = float(_np.mean(blc))
            anchors.append((m2, max(float(_np.std(blc)), 0.05 * m2)))
            print(f"  çapa genişledi: {len(anchors)} bilgi (son mu={m2:.3f})")
            pf, sf = feat_mean(base_view(torch.cat(data)), si + 5000)
            seg_protos.append((pf, sf))
            print(f"  vektör çapa: {len(seg_protos)} prototip (son yayılım={sf:.2f})")

        def feat_ratio(ten, seed):
            # ASİMETRİK test: adayın KENDİ yayılımına göre uzaklık. Sıkı-yabancı küme
            # (matematik: yayılım 1.6, uzaklık 10.5) geniş-bilinen bulutun (kod: 15.4)
            # içinde kaybolmaz; tersi (aynı-dağılım ~1) sessiz kalır.
            pf, sf = feat_mean(ten, seed)
            return min(float(((pf - p) ** 2).sum()) / max(sf, 0.5) for p, sp in seg_protos)

        def acc_novel():
            # Birikimde PEKLEŞMEMİŞ bilgi var mı? (kayıp-seviyesi + vektör)
            Tac = torch.cat(acc)
            rga = torch.Generator().manual_seed(777)
            bla = []
            for _ in range(NB):
                s0 = int(torch.randint(0, len(Tac) - cfg["block"] - 1, (1,), generator=rga))
                x = base_view(Tac[s0:s0 + cfg["block"]]).unsqueeze(0).repeat(32, 1).long()
                y = base_view(Tac[s0 + 1:s0 + cfg["block"] + 1]).unsqueeze(0).repeat(32, 1).long()
                with torch.no_grad():
                    bla.append(FF.cross_entropy(bb(x).view(-1, V), y.view(-1)).item())
            ma = float(_np.mean(bla))
            lvl = all(ma > m + 2.5 * s for m, s in anchors)
            return (lvl or feat_ratio(Tac, 778) > args.surprise_kf), ma

        for ci, (tag, ten) in enumerate(stream):
            rg = torch.Generator().manual_seed(ci)
            bl = []
            for _ in range(NB):
                s0 = int(torch.randint(0, len(ten) - cfg["block"] - 1, (1,), generator=rg))
                x = base_view(ten[s0:s0 + cfg["block"]]).unsqueeze(0).repeat(32, 1).long()
                y = base_view(ten[s0 + 1:s0 + cfg["block"] + 1]).unsqueeze(0).repeat(32, 1).long()
                with torch.no_grad():
                    bl.append(FF.cross_entropy(bb(x).view(-1, V), y.view(-1)).item())
            mu_c = float(_np.mean(bl))
            fr = feat_ratio(ten, 2000 + ci)
            sig_min = min(s for _, s in anchors)
            lvl = all(mu_c > m + 2.5 * s for m, s in anchors)  # hiçbir pekleşmiş bilgiye benzemiyor
            chg = s_prev is not None and abs(mu_c - s_prev) > 1.5 * sig_min
            feat_fire = fr > args.surprise_kf
            fire = lvl or chg or feat_fire
            print(f"  parça {ci} [{tag}]: saskinlik={mu_c:.3f} vektör={fr:.2f} -> {'SINIR' if fire else 'tanidik'}")
            if fire:
                if acc:
                    novel, ma = acc_novel()
                    if novel:
                        n = brain.open_segment()
                        # Etiket BİRİKİMDEN gelir (o anki parçadan değil): çoğunluk oyu.
                        seg_tags.append(max(set(acc_tags), key=acc_tags.count))
                        train_seg(acc, n - 1)
                    else:
                        print(f"  birikim pekleşmiş (şaşkınlık={ma:.3f}): ek eğitim YOK, atıldı")
                else:
                    print("  (ilk kanıt birikiyor, segment yok)")
                acc, acc_tags = [], []
            acc.append(ten)
            acc_tags.append(tag)
            s_prev = mu_c
        if acc:
            # Kalan birikim gerçekten YENİ mi? Tanıdıksa büyüme YOK (bilgi yok, atılır).
            Tr = torch.cat(acc)
            rgr = torch.Generator().manual_seed(1234)
            blr = []
            for _ in range(NB):
                s0 = int(torch.randint(0, len(Tr) - cfg["block"] - 1, (1,), generator=rgr))
                x = base_view(Tr[s0:s0 + cfg["block"]]).unsqueeze(0).repeat(32, 1).long()
                y = base_view(Tr[s0 + 1:s0 + cfg["block"] + 1]).unsqueeze(0).repeat(32, 1).long()
                with torch.no_grad():
                    blr.append(FF.cross_entropy(bb(x).view(-1, V), y.view(-1)).item())
            mr = float(_np.mean(blr))
            fr_r = feat_ratio(Tr, 1235)
            if all(mr > m + 2.5 * s for m, s in anchors) or fr_r > args.surprise_kf:
                n = brain.open_segment()
                seg_tags.append(max(set(acc_tags), key=acc_tags.count))
                train_seg(acc, n - 1)
            else:
                print(f"  kalan birikim tanıdık (saskinlik={mr:.3f}): yeni bilgi yok, büyüme YOK")
        # DEĞERLENDİRME: havuza göre. nar: yenilik->segment / tanıdık->omurga.
        # codemath: omurga ikisini de bilmez -> segmentler-arası kimlik yönlendirmesi.
        brain.eval()
        if args.pool == "nar":
            Sv = S[int(.9 * len(S)):]
            A2 = A[:200000]
            Av2 = A2[int(.9 * len(A2)):]
            pools = {"S": Sv, "A": Av2}
        elif args.pool == "codemath":
            Cv = C0e[int(.9 * len(C0e)):]
            Mv = M0e[int(.9 * len(M0e)):]
            pools = {"C": Cv, "M": Mv}
        else:
            Cv = C0e[int(.9 * len(C0e)):]
            Mv = M0e[int(.9 * len(M0e)):]
            pools = {"C": Cv, "M": Mv, "T": T0e[int(.9 * len(T0e)):]}
        for si, tg in enumerate(seg_tags):
            vv = pools[tg]
            print(f"  segment {si} [{tg}] kendi-val kaybı: {evalloss(lambda x, s=si: brain.forward_seg(x, s), vv, chars, cfg['block'], 32):.3f}")
        ok = tot = 0
        rg = torch.Generator().manual_seed(7)
        for _ in range(20):
            if args.pool == "nar":
                truth = "S" if int(torch.randint(0, 2, (1,), generator=rg)) == 0 else "A"
                vv = pools[truth]
                ix = torch.randint(0, len(vv) - cfg["block"] - 1, (32,), generator=rg)
                x = torch.stack([vv[i:i + cfg["block"]] for i in ix]).long()
                y = torch.stack([vv[i + 1:i + cfg["block"] + 1] for i in ix]).long()
                with torch.no_grad():
                    bl = FF.cross_entropy(bb(x).view(-1, V), y.view(-1)).item()
                pick, pl = brain.route(x, y)
                if truth == "S":
                    ok += (seg_tags[pick] == "S" and pl < bl)
                else:
                    ok += (bl <= pl)
            else:
                truth = int(torch.randint(0, len(seg_tags), (1,), generator=rg))
                vv = pools[seg_tags[truth]]
                ix = torch.randint(0, len(vv) - cfg["block"] - 1, (32,), generator=rg)
                x = torch.stack([vv[i:i + cfg["block"]] for i in ix]).long()
                y = torch.stack([vv[i + 1:i + cfg["block"] + 1] for i in ix]).long()
                pick, _ = brain.route(x, y)
                ok += (pick == truth)
            tot += 1
        print(f"\nAUTO: {len(seg_tags)} segment {seg_tags} | yönlendirici: {ok}/{tot} | süre={time.time()-t0:.0f}sn")
        torch.save({"n_seg": len(seg_tags), "tags": seg_tags, "pool": args.pool,
                    "ext_chars": ext_chars, "bn": args.adapt_bn, "nested": True, "apos": brain.apos,
                    "protos": torch.stack([p for p, _ in seg_protos]),
                    "spreads": [s for _, s in seg_protos],
                    "adapts": [[a.state_dict() for a in seg] for seg in brain.adapts],
                    "heads": [h.state_dict() for h in brain.heads],
                    "emb_ext": brain.emb_ext.state_dict() if brain.n_extra else None,
                    "head_ext": brain.head_ext.state_dict() if brain.n_extra else None,
                    "chars": chars, "cfg": cfg}, args.out or os.path.join(MODEL_DIR, "grow_auto.pt"))
        return

    if args.stage == "refine":
        # İNCE AYAR: yapıyı bozmadan hedef segmente kapasite istifle, kaldığı yerden eğit.
        # Yalnızca en-yeni adaptör + kafa (+ext) eğitilir; gerisi donuk (F=0 korunur).
        print("\n=== İNCE AYAR (yapı korunur, kapasite istiflenir) ===")
        src = torch.load(args.brain, map_location="cpu", weights_only=False)
        assert src["chars"] == chars, "vocab uyuşmuyor"
        brain = rebuild_brain(src, bb)
        si = resolve_seg(args.seg, list(src["tags"]))
        tag = src["tags"][si]
        files = {"C": CODE_TXT, "M": MATH_TXT, "T": TURK_TXT,
                 "S": os.path.join(PROJECT_ROOT, "data", "text", "tinyshakespeare.txt"),
                 "A": SA, "B": args.bdata}
        stx = stoi2 if src.get("ext_chars") else stoi
        Dfull, _ = load_ids(args.data or files.get(tag, CODE_TXT), stx)
        Dtr, Dv = Dfull[:int(.9 * len(Dfull))], Dfull[int(.9 * len(Dfull)):]
        before = evalloss(lambda x: brain.forward_seg(x, si), Dv, chars, cfg["block"], 32)
        print(f"  segment {si} [{tag}] önce-val: {before:.3f} (hedef <2.0)" if tag == "C" else f"  segment {si} [{tag}] önce-val: {before:.3f}")
        for p in brain.adapts[si].parameters():
            p.requires_grad = False
        brain.adapts[si].append(brain._make_adapter())
        for p in list(brain.adapts[si][-1].parameters()) + list(brain.heads[si].parameters()):
            p.requires_grad = True
        if brain.n_extra:
            for p in list(brain.emb_ext.parameters()) + list(brain.head_ext.parameters()):
                p.requires_grad = True
        print(f"  adaptör istiflendi: seg{si} artık {len(brain.adapts[si])} adaptör, "
              f"eğitilebilir={sum(p.numel() for p in brain.parameters() if p.requires_grad)/1e3:.1f}k")
        opt = torch.optim.AdamW([p for p in brain.parameters() if p.requires_grad], lr=args.lr)
        rng = torch.Generator().manual_seed(args.seed)
        brain.train()
        t0 = time.time()
        for it in range(1, args.iters + 1):
            x, y = batchify(Dtr, cfg["block"], args.batch, rng)
            loss = FF.cross_entropy(brain.forward_seg(x, si).view(-1, brain.V_total), y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it % 25 == 0 or it == 1:
                print(f"  iter {it}/{args.iters} train={loss.item():.3f} {time.time()-t0:.0f}sn", flush=True)
        after = evalloss(lambda x: brain.forward_seg(x, si), Dv, chars, cfg["block"], 32)
        print(f"  SONUÇ seg{si} [{tag}]: {before:.3f} -> {after:.3f} ({'HEDEF TUTTU (<2.0)' if tag == 'C' and after < 2.0 else 'devam gerekli' if tag == 'C' else 'tamam'})")
        out = {"n_seg": len(src["tags"]), "tags": list(src["tags"]), "pool": src.get("pool", "nar"),
               "ext_chars": src.get("ext_chars", []), "bn": src.get("bn", 64), "nested": True, "apos": brain.apos,
               "protos": src.get("protos"), "spreads": src.get("spreads"),
               "adapts": [[a.state_dict() for a in seg] for seg in brain.adapts],
               "heads": [h.state_dict() for h in brain.heads],
               "emb_ext": brain.emb_ext.state_dict() if brain.n_extra else None,
               "head_ext": brain.head_ext.state_dict() if brain.n_extra else None,
               "chars": chars, "cfg": cfg}
        torch.save(out, args.out or os.path.join(MODEL_DIR, "grow_auto_r1.pt"))
        print("  kaydedildi: grow_auto_r1.pt (eski beyin duruyor)")
        return

    if args.stage == "rft":
        # RFT-lite (araç-doğrulanmış reddetme-örneklemesi): K iz örnekle, araçla
        # DOĞRUYU filtrele, doğrularla ince-eğit. Yalnızca hedef segment (F=0).
        # Ölçü: tur-başı isabet oranı (yükselmeli) + think-val (düşmeli).
        import random as _rnd
        print("\n=== RFT (araç-filtreli iz eğitimi) ===")
        src = torch.load(args.brain, map_location="cpu", weights_only=False)
        brain = rebuild_brain(src, bb)
        si = resolve_seg(args.seg, list(src["tags"]))
        stx = stoi2 if src.get("ext_chars") else stoi
        allc = chars + src.get("ext_chars", [])
        for p in brain.adapts[si].parameters():
            p.requires_grad = False
        brain.adapts[si].append(brain._make_adapter())
        for p in list(brain.adapts[si][-1].parameters()) + list(brain.heads[si].parameters()):
            p.requires_grad = True
        if brain.n_extra:
            for p in list(brain.emb_ext.parameters()) + list(brain.head_ext.parameters()):
                p.requires_grad = True
        print(f"  seg{si} [{src['tags'][si]}]: adaptör istiflendi ({len(brain.adapts[si])}), tur={args.rft_rounds}")
        _rnd.seed(args.seed)
        t0 = time.time()
        for rd in range(args.rft_rounds):
            good, tries = [], 0
            brain.eval()
            for _ in range(args.rft_n):
                a, b, op = _rnd.randint(0, args.rft_max), _rnd.randint(0, args.rft_max), _rnd.choice(["+", "-", "*"])
                if op == "-" and b > a:
                    a, b = b, a
                pr = "%d %s %d =" % (a, op, b)
                exp = expected_trace(a, op, b)
                for _ in range(args.rft_k):
                    idx = torch.tensor([[stx.get(c, 0) for c in pr]], dtype=torch.long)
                    with torch.no_grad():
                        for _ in range(20):
                            lo = brain.forward_seg(idx[:, -cfg["block"]:], si)[0, -1] / 0.9
                            nxt = torch.multinomial(FF.softmax(lo, -1), 1).item()
                            idx = torch.cat([idx, torch.tensor([[nxt]])], 1)
                    txt = "".join(allc[t] for t in idx[0].tolist())
                    tries += 1
                    if exp in txt:  # araç-hükmü: öğretmen-iz harfi-harfi içeride
                        good.append(txt + "\n")
            print(f"  tur{rd}: isabet {len(good)}/{tries} (%{100 * len(good) / max(1, tries):.1f})", flush=True)
            if good:
                pool = torch.tensor([stx.get(c, 0) for c in "".join(good)], dtype=torch.long)
                opt = torch.optim.AdamW([p for p in brain.parameters() if p.requires_grad], lr=args.lr)
                rng = torch.Generator().manual_seed(rd)
                brain.train()
                for it in range(args.rft_iters):
                    x, y = batchify(pool, cfg["block"], args.batch, rng)
                    loss = FF.cross_entropy(brain.forward_seg(x, si).view(-1, brain.V_total), y.view(-1))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        out = {"n_seg": len(src["tags"]), "tags": list(src["tags"]), "pool": src.get("pool", "nar"),
               "ext_chars": src.get("ext_chars", []), "bn": src.get("bn", 64), "nested": True, "apos": brain.apos,
               "protos": src.get("protos"), "spreads": src.get("spreads"),
               "adapts": [[a.state_dict() for a in seg] for seg in brain.adapts],
               "heads": [h.state_dict() for h in brain.heads],
               "emb_ext": brain.emb_ext.state_dict() if brain.n_extra else None,
               "head_ext": brain.head_ext.state_dict() if brain.n_extra else None,
               "chars": chars, "cfg": cfg}
        torch.save(out, args.out or os.path.join(MODEL_DIR, "grow_auto_rft.pt"))
        print(f"  kaydedildi: grow_auto_rft.pt  süre={time.time()-t0:.0f}sn")
        return

    if args.stage == "evalarith":
        # ÖLÇÜM: sabit soru kümesi (seed 1234), greedy; iz-sadakati ve son-cevap AYRI sayılır.
        # Basamak sınıfı: 1 = iki operand < 10, 2 = en az biri 10-99.
        import random as _rnd
        import re
        src = torch.load(args.brain, map_location="cpu", weights_only=False)
        brain = rebuild_brain(src, bb)
        brain.eval()
        si = resolve_seg(args.seg, list(src["tags"]))
        allc = chars + src.get("ext_chars", [])
        stx = {c: i for i, c in enumerate(allc)}  # alfabe beyinden: --pool gerekmez
        rq = _rnd.Random(1234)
        qs = []
        while len(qs) < 200:
            mx = 9 if len(qs) < 100 else args.rft_max
            a, b, op = rq.randint(0, mx), rq.randint(0, mx), rq.choice(["+", "-", "*"])
            if op == "-" and b > a:
                a, b = b, a
            qs.append((a, op, b))
        st = {1: [0, 0, 0], 2: [0, 0, 0]}  # [n, iz-doğru, cevap-doğru]
        ops_ok = {"+": [0, 0], "-": [0, 0], "*": [0, 0]}
        t0 = time.time()
        for qi, (a, op, b) in enumerate(qs):
            pr = "%d %s %d =" % (a, op, b)
            idx = torch.tensor([[stx.get(c, 0) for c in pr]], dtype=torch.long)
            with torch.no_grad():
                for _ in range(len(expected_trace(a, op, b)) + 2):
                    nxt = int(brain.forward_seg(idx[:, -cfg["block"]:], si)[0, -1].argmax())
                    idx = torch.cat([idx, torch.tensor([[nxt]])], 1)
            gen = "".join(allc[t] for t in idx[0].tolist())[len(pr):]
            r = arith_result(a, op, b)
            m = re.match(r"\s*\[\s*\d+\s*[+\-*]\s*\d+\s*=\s*(\d+)\s*\]\s*(-?\d+)", gen)
            tr_ok = bool(m) and trace_num(m.group(1)) == r
            an_ok = bool(m) and int(m.group(2)) == r
            k = 1 if max(a, b) < 10 else 2
            st[k][0] += 1
            st[k][1] += tr_ok
            st[k][2] += an_ok
            ops_ok[op][0] += 1
            ops_ok[op][1] += an_ok
            if qi % 40 == 0:
                print(f"  örnek: {pr}{gen.rstrip()}  iz={'✓' if tr_ok else '✗'} cevap={'✓' if an_ok else '✗'}", flush=True)
        for k in (1, 2):
            n, t_, c_ = st[k]
            if n:
                print(f"  {k}-basamak (n={n}): iz-doğru %{100*t_/n:.1f}  cevap-doğru %{100*c_/n:.1f}")
        print("  işlem-başı cevap: " + "  ".join(f"{o} %{100*v[1]/max(1,v[0]):.0f}" for o, v in ops_ok.items()))
        n = sum(v[0] for v in st.values())
        print(f"EVALARITH [{os.path.basename(args.brain)} seg={src['tags'][si]} rev={REV_TRACE}]: "
              f"iz %{100*sum(v[1] for v in st.values())/n:.1f}  cevap %{100*sum(v[2] for v in st.values())/n:.1f}  "
              f"süre={time.time()-t0:.0f}sn")
        return

    if args.stage == "think":
        # SYSTEM-2 (ilk adım): ÜRET-ve-TEST. K aday devam örnekle, her birini kendi
        # kafasının NLL'si + harici hesap-doğrulayıcıyla puanla, en iyiyi bağlama ekle.
        # Greedy (System-1) tek-yolludur, geri dönüşü yoktur; bu çok-yollu aramadır.
        import re
        print("\n=== SYSTEM-2 (üret-ve-test muhakemesi) ===")
        src = torch.load(args.brain, map_location="cpu", weights_only=False)
        brain = rebuild_brain(src, bb)
        brain.eval()
        si = resolve_seg(args.seg, list(src["tags"]))
        stx = stoi2 if src.get("ext_chars") else stoi
        allc = chars + src.get("ext_chars", [])

        def calc_bonus(text):
            # Son CEVABI denetle: köşeli iz (kapanmamışı da) çıkarılır; iz trace_bonus'un işi.
            text = re.sub(r"\[[^\]]*(\]|$)", "", text)
            m = re.search(r"(\d+)\s*([+\-*])\s*(\d+)\s*=\s*(-?\d+)", text)
            if not m:
                return 0.0
            a, op, b, r = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
            ok = (a + b if op == "+" else (a - b if op == "-" else a * b)) == r
            return -1.0 if ok else +1.0  # doğru hesap güçlü ödül, yanlış güçlü ceza

        def trace_bonus(text):
            # İZ SADAKATİ (C+A füzyonu): köşeli izdeki HER hesabı araçla doğrula;
            # sadık iz ödüllendirilir — model doğru DÜŞÜNMEYE yönlendirilir.
            from tool_calc import safe_eval
            bonus = 0.0
            for br in re.findall(r"\[([^\[\]]+)\]", text):
                parts = br.split("=")
                if len(parts) != 2:
                    continue
                v, err = safe_eval(parts[0])
                if err is not None:
                    continue
                try:
                    r = trace_num(parts[1].strip().split()[0])
                except (ValueError, IndexError):
                    continue
                bonus += -0.5 if (int(v) if float(v).is_integer() else v) == r else +0.5
            return bonus

        def gen_from(ctx_ids, n_tok, temp, greedy=False):
            idx = ctx_ids.clone()
            with torch.no_grad():
                for _ in range(n_tok):
                    lo = brain.forward_seg(idx[:, -cfg["block"]:], si)[0, -1] / temp
                    nxt = int(lo.argmax()) if greedy else torch.multinomial(FF.softmax(lo, -1), 1).item()
                    idx = torch.cat([idx, torch.tensor([[nxt]])], 1)
            return idx[0, len(ctx_ids[0]):].tolist()

        def score_of(ctx_ids, cont):
            full = torch.cat([ctx_ids[0], torch.tensor(cont)]).unsqueeze(0)
            with torch.no_grad():
                lo = brain.forward_seg(full[:, -cfg["block"]:], si)
            lp = FF.log_softmax(lo[0, -len(cont) - 1:-1], -1)
            nll = -lp[range(len(cont)), torch.tensor(cont)].mean().item()
            txt = "".join(allc[t] for t in full[0].tolist())
            # YOĞUN ÖDÜL: araçla kurulan öğretmen-izin en-uzun-ortak-öneki.
            dense = 0.0
            pa = parse_arith(args.prompt)
            if pa is not None:
                exp = expected_trace(*pa)
                cand_txt = "".join(allc[t] for t in cont)
                k = 0
                for x, y in zip(cand_txt, exp):
                    if x == y:
                        k += 1
                    else:
                        break
                dense = -1.5 * k / max(1, len(exp))
            return nll + calc_bonus(txt) + trace_bonus(txt) + dense

        ctx = torch.tensor([[stx.get(c, 0) for c in args.prompt]], dtype=torch.long)
        g0 = gen_from(ctx, args.cand_len * args.steps, 1.0, greedy=True)
        print("  System-1 (greedy): %s%s" % (args.prompt, "".join(allc[t] for t in g0)))
        # ToT-lite: her tur K dal simüle edilir, min-şaşkınlık seçilir; tur kötüleşirse
        # GERİ DÖNÜLÜR (son kazanan atılır, sıcaklık artırılıp yeniden dallanılır).
        hist, prev_min, backtracks, temp = [ctx.clone()], None, 0, 0.8
        st = 0
        while st < args.steps:
            cands = [gen_from(ctx, args.cand_len, temp) for _ in range(args.topk)]
            scored = sorted(((score_of(ctx, c), c) for c in cands), key=lambda t: t[0])
            for rank, (sc, c) in enumerate(scored):
                print(f"    tur{st} aday{rank}: skor={sc:+.3f} «{''.join(allc[t] for t in c)}»")
            rmin = scored[0][0]
            if prev_min is not None and rmin > prev_min + 0.3 and backtracks < 2:
                ctx, temp, backtracks = hist[-1], round(temp + 0.2, 2), backtracks + 1
                print(f"  tur{st} kötüleşti ({prev_min:+.3f}->{rmin:+.3f}): GERİ DÖNÜLDÜ, sıcaklık={temp}")
                continue
            hist.append(ctx.clone())
            ctx = torch.cat([ctx, torch.tensor(scored[0][1]).unsqueeze(0)], 1)
            prev_min = rmin
            print(f"  tur{st} kazanan eklendi.")
            st += 1
        print("  System-2 sonuç: %s" % "".join(allc[t] for t in ctx[0].tolist()))
        if args.tool:
            # ŞAŞKINLIK KAPISI + ARAÇ: aritmetik soru omurga sınırını aşıyorsa
            # (modelin cevabı hesapla uyuşmuyorsa) araç hükmeder, iz bırakılır.
            from tool_calc import safe_eval
            m = re.search(r"([\d\s+\-*/().]+?)\s*=\s*$", args.prompt)
            v, err = safe_eval(m.group(1)) if m else (None, "ifade yok")
            final = "".join(allc[t] for t in ctx[0].tolist())
            if err is None:
                tv = int(v) if float(v).is_integer() else v
                agree = str(tv) in final.split("=")[-1][:12]
                print(f"  ARAÇ KAPISI: ifade='{m.group(1).strip()}' araç={tv} "
                      f"-> {'MODEL DOĞRULANDI' if agree else 'ARAÇ DÜZELTTİ: ' + args.prompt + str(tv)}")
            else:
                print(f"  ARAÇ KAPISI: devrede değil ({err})")
        return

    if args.stage == "sleep":
        # UYKU: işlevsel eşdeğer segmentleri kaynaştır. Birincil kriter kayıp farkı
        # (|L_i(j)-L_i(i)|<0.025, aynı sabit partilerde); kosinüs yalnızca ucuz ön-eleme.
        print("\n=== UYKU (birleştirme, yıkım yok — karar öncesi rapor) ===")
        src = torch.load(os.path.join(MODEL_DIR, "grow_auto.pt"), map_location="cpu", weights_only=False)
        assert src["chars"] == chars, "vocab uyuşmuyor"
        n_ext = len(src.get("ext_chars", []))
        brain = AutoBrain(bb, n_ext, src.get("bn", 64), src.get("apos", "head"))
        raw_ad = src["adapts"]
        if not src.get("nested"):  # eski düz kayıtlar: her segment tek adaptör
            raw_ad = [[sd] for sd in raw_ad]
        for seg_sds, h_sd in zip(raw_ad, src["heads"]):
            brain.open_segment()
            brain.adapts[-1] = nn.ModuleList()
            for a_sd in seg_sds:
                na = brain._make_adapter()
                na.load_state_dict(a_sd)
                for p in na.parameters():
                    p.requires_grad = False
                brain.adapts[-1].append(na)
            brain.heads[-1].load_state_dict(h_sd)
        if n_ext:
            brain.emb_ext.load_state_dict(src["emb_ext"])
            brain.head_ext.load_state_dict(src["head_ext"])
        tags = list(src["tags"])
        if src.get("pool", "nar") == "nar":
            S, _ = load_ids(os.path.join(PROJECT_ROOT, "data", "text", "tinyshakespeare.txt"), stoi)
            Sv = S[int(.9 * len(S)):]
            A2 = A[:200000]
            Av2 = A2[int(.9 * len(A2)):]
            pools = {"S": Sv, "A": Av2}
        else:
            C0e, _ = load_ids(CODE_TXT, stoi2)
            M0e, _ = load_ids(MATH_TXT, stoi2)
            pools = {"C": C0e[int(.9 * len(C0e)):], "M": M0e[int(.9 * len(M0e)):]}
            if src.get("pool") == "three" and os.path.exists(TURK_TXT):
                T0e, _ = load_ids(TURK_TXT, stoi2)
                pools["T"] = T0e[int(.9 * len(T0e)):]
        EPS = 0.025
        # sabit parti setleri (karşılaştırma gürültüsüz olsun)
        fixed = {}
        rg = torch.Generator().manual_seed(123)
        for tg, vv in pools.items():
            fb = []
            for _ in range(10):
                ix = torch.randint(0, len(vv) - cfg["block"] - 1, (32,), generator=rg)
                fb.append((torch.stack([vv[i:i + cfg["block"]] for i in ix]).long(),
                           torch.stack([vv[i + 1:i + cfg["block"] + 1] for i in ix]).long()))
            fixed[tg] = fb

        def path_loss(si, fb):
            with torch.no_grad():
                return sum(FF.cross_entropy(brain.forward_seg(x, si).view(-1, brain.V_total), y.view(-1)).item()
                           for x, y in fb) / len(fb)

        def cos_sim(si, sj):
            import torch.nn.functional as _F
            vi = torch.cat([p.detach().view(-1) for a in brain.adapts[si] for p in a.parameters()])
            vj = torch.cat([p.detach().view(-1) for a in brain.adapts[sj] for p in a.parameters()])
            return float(_F.cosine_similarity(vi, vj, dim=0))

        drop = set()
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                if tags[i] != tags[j]:
                    continue
                tg = tags[i]
                li_i, li_j = path_loss(i, fixed[tg]), path_loss(j, fixed[tg])
                own_j = li_j  # j'nin kendi yolu
                cos = cos_sim(i, j)
                # karşılıklı kapsama: i, j'nin işini VE j, i'nin işini eps içinde yapmalı
                lj_i = path_loss(i, fixed[tg])
                print(f"  çift ({i},{j}) [{tg}]: cos={cos:.3f} L_i={li_i:.3f} L_j(i-yolunda)={lj_i:.3f} fark={abs(lj_i-own_j):.3f}")
                if cos > 0.95 and abs(lj_i - own_j) < EPS and abs(li_i - own_j) < EPS:
                    drop.add(j)
                    print(f"    -> {j} birleştirildi (işlevsel eşdeğer), {i} yaşar")
        keep = [i for i in range(len(tags)) if i not in drop]
        print(f"  sonuç: {len(tags)} -> {len(keep)} segment (yaşayan: {keep})")
        torch.save({"n_seg": len(keep), "tags": [tags[i] for i in keep], "pool": src.get("pool", "nar"),
                    "ext_chars": src.get("ext_chars", []), "bn": src.get("bn", 64), "nested": True, "apos": brain.apos,
                    "adapts": [[a.state_dict() for a in brain.adapts[i]] for i in keep],
                    "heads": [brain.heads[i].state_dict() for i in keep],
                    "emb_ext": brain.emb_ext.state_dict() if brain.n_extra else None,
                    "head_ext": brain.head_ext.state_dict() if brain.n_extra else None,
                    "chars": chars, "cfg": cfg}, os.path.join(MODEL_DIR, "grow_merged.pt"))
        # yönlendirici koruma testi (birleşmiş beyin + omurga)
        print("  yönlendirici koruma testi:")
        ok = tot = 0
        rg = torch.Generator().manual_seed(7)
        is_nar = src.get("pool", "nar") == "nar"
        for _ in range(20):
            if is_nar:
                truth = "S" if int(torch.randint(0, 2, (1,), generator=rg)) == 0 else "A"
            else:
                truth = tags[keep[int(torch.randint(0, len(keep), (1,), generator=rg))]]
            vv = pools[truth]
            ix = torch.randint(0, len(vv) - cfg["block"] - 1, (32,), generator=rg)
            x = torch.stack([vv[i:i + cfg["block"]] for i in ix]).long()
            y = torch.stack([vv[i + 1:i + cfg["block"] + 1] for i in ix]).long()
            best, bvl = -1, float("inf")
            for si, k in enumerate(keep):
                with torch.no_grad():
                    l = FF.cross_entropy(brain.forward_seg(x, k).view(-1, brain.V_total), y.view(-1)).item()
                if l < bvl:
                    best, bvl = si, l
            if is_nar:
                with torch.no_grad():
                    bl = FF.cross_entropy(bb(x).view(-1, V), y.view(-1)).item()
                if truth == "S":
                    ok += (tags[keep[best]] == "S" and bvl < bl)
                else:
                    ok += (bl <= bvl)
            else:
                ok += (tags[keep[best]] == truth)
            tot += 1
        print(f"  MERGED yönlendirici: {ok}/{tot} (hedef: birleşme öncesi skor korunur)")
        return
    Av, Bv = A[int(.9 * len(A)):], B[int(.9 * len(B)):]
    Atr, Btr = A[:int(.9 * len(A))], B[:int(.9 * len(B))]

    baseA = evalloss(bb, Av, chars, cfg["block"], 32)
    print(f"baz L_A* (dilim A val, ckpt): {baseA:.3f}")

    t0 = time.time()
    if args.stage == "adapt":
        g = GrowGPT(bb, args.adapt_bn)
        print(f"eğitilebilir: {g.n_trainable()/1e3:.1f}k / {sum(p.numel() for p in bb.parameters())/1e6:.2f}M (tavan: {args.max_adapt} adaptör)")
        rng = torch.Generator().manual_seed(args.seed)
        g.train()
        bvals, done = [], 0
        while done < args.iters:
            opt = torch.optim.AdamW([p for p in g.parameters() if p.requires_grad], lr=args.lr)
            for _ in range(min(50, args.iters - done)):
                x, y = batchify(Btr, cfg["block"], 32, rng)
                loss = FF.cross_entropy(g.forward_B(x).view(-1, V), y.view(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
                done += 1
            bv = evalloss(g.forward_B, Bv, chars, cfg["block"], 32)
            bvals.append(bv)
            print(f"  iter {done}/{args.iters} B-train={loss.item():.3f} B-val={bv:.3f} adaptör={len(g.adapts)} {time.time()-t0:.0f}sn")
            # BÜYÜME KARARI: son W B-val iyileşmesi eps altındaysa ve tavan dolmadıysa büyü.
            if (len(bvals) >= args.plateau_w and len(g.adapts) < args.max_adapt
                    and (bvals[-args.plateau_w] - bvals[-1]) < args.plateau_eps):
                n = g.add_adapter()
                print(f"  PLATO (iyileşme={bvals[-args.plateau_w]-bvals[-1]:.3f}<{args.plateau_eps}) -> adaptör {n} açıldı, eskiler dondu")
        torch.save({"adapts": [a.state_dict() for a in g.adapts], "headB": g.headB.state_dict(),
                    "chars": chars, "cfg": cfg}, os.path.join(MODEL_DIR, "grow_adapt.pt"))
        LA = evalloss(bb, Av, chars, cfg["block"], 32)  # omurga donuk -> değişemez
        LB = evalloss(g.forward_B, Bv, chars, cfg["block"], 32)
    else:
        if args.pool == "codemath":
            # FULL-ceiling ardışık: CODE sonra MATH (her faz args.iters). Geniş alfabe
            # (yeni satırlar da eğitilir — tavan ölçümü, unutma serbest).
            C0, _ = load_ids(CODE_TXT, stoi2)
            M0, _ = load_ids(MATH_TXT, stoi2)
            V2 = len(stoi2)
            m = MicroGPT(V2, cfg["n_layer"], cfg["n_head"], cfg["n_embd"], cfg["block"])
            sd = ck["state"]
            sd["tok.weight"] = torch.cat([sd["tok.weight"],
                                          torch.randn(V2 - V, cfg["n_embd"]) * 0.02])
            sd["head.weight"] = torch.cat([sd["head.weight"],
                                           torch.randn(V2 - V, cfg["n_embd"]) * 0.02])
            m.load_state_dict(sd)
            phases = [("C", C0), ("M", M0)]
            Cv, Mv = C0[int(.9 * len(C0)):], M0[int(.9 * len(M0)):]
            opt = torch.optim.AdamW(m.parameters(), lr=args.lr)
            rng = torch.Generator().manual_seed(args.seed)
            m.train()
            for pname, P in phases:
                Ptr = P[:int(.9 * len(P))]
                for it in range(1, args.iters + 1):
                    x, y = batchify(Ptr, cfg["block"], args.batch, rng)
                    loss = FF.cross_entropy(m(x).view(-1, V2), y.view(-1))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    if it % 50 == 0 or it == 1:
                        print(f"  faz {pname} iter {it}/{args.iters} train={loss.item():.3f} {time.time()-t0:.0f}sn", flush=True)
                lA = evalloss(m, Av, chars, cfg["block"], 32)
                lC = evalloss(m, Cv, chars, cfg["block"], 32)
                print(f"  faz {pname} sonu: L_A={lA:.3f} (F_A={lA-baseA:+.3f}) L_C={lC:.3f}")
            LA = evalloss(m, Av, chars, cfg["block"], 32)
            LB = evalloss(m, Mv, chars, cfg["block"], 32)
            print(f"  final L_C={evalloss(m, Cv, chars, cfg['block'], 32):.3f} (kod unutması: MATH sonrası)")
        else:
            m = MicroGPT(len(chars), cfg["n_layer"], cfg["n_head"], cfg["n_embd"], cfg["block"])
            m.load_state_dict(ck["state"])
            opt = torch.optim.AdamW(m.parameters(), lr=args.lr)
            rng = torch.Generator().manual_seed(args.seed)
            m.train()
            for it in range(1, args.iters + 1):
                x, y = batchify(Btr, cfg["block"], 32, rng)
                loss = FF.cross_entropy(m(x).view(-1, V), y.view(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
                if it % 50 == 0 or it == 1:
                    print(f"  iter {it}/{args.iters} B-train={loss.item():.3f} {time.time()-t0:.0f}sn")
            LA = evalloss(m, Av, chars, cfg["block"], 32)
            LB = evalloss(m, Bv, chars, cfg["block"], 32)

    F = LA - baseA
    print(f"\nSONUÇ [{args.stage}]: L_A*={baseA:.3f} L_A={LA:.3f} L_B={LB:.3f} "
          f"unutma F={F:+.3f} ({'SIFIR' if F < 0.05 else 'HAFİF' if F < 0.20 else 'CİDDİ'}) "
          f"süre={time.time()-t0:.0f}sn")


if __name__ == "__main__":
    main()
