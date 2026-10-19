#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_neuron.py — NoRa Nöron Temeli (sinek beyni örnek alındı).

Kaynak: Shiu et al. 2024 (Nature) "A Drosophila computational brain model reveals sensorimotor
processing" — github.com/philshiu/Drosophila_brain_model. Orada tüm yetişkin sinek beyni
(FlyWire konektomu, 138.639 nöron, ~15M bağlantı) Brian2 ile LIF olarak çalıştırılıyor.
Biz aynı denklemleri saf NumPy'de (CPU, Brian2/derleyici yok) olay-güdümlü yazdık ve
referans çıktıyla doğruluyoruz.

Sinekten alınan ilkeler (FlyWire verisinde ölçüldü, bkz. NEURON_TEMELI.md):
  - LIF nöron: dv/dt=(v0-v+g)/t_mbr, dg/dt=-g/tau, eşik -45mV, reset -52mV, refrakter 2.2ms
  - Sinaps = sinaps-sayısı x işaret x w_syn (0.275 mV), gecikme 1.8 ms. Öğrenilmiş ağırlık yok:
    davranışı yalnızca BAĞLANTI YAPISI üretiyor.
  - Dale yasası: her nöron tek işaretli (verideki 138.639 nöronun 0'ı karışık), %30 baskılayıcı.
  - Seyrek (yoğunluk %0.08) ve ağır kuyruklu dereceler (medyan 79, max ~9800 çıkış).
  - Mantar cismi (mushroom body): seyrek genişleme (PN→KC ~7 pençe), APL geri-baskılama ile
    ~%5 aktif KC, dopamin (DAN) kapılı KC→MBON plastisitesi = sineğin öğrenme ve bellek
    merkezi. Yenilik (familiarity) MBON'u tekrarla bastırılır → NoRa'nın şaşkınlık kapısı.

Bileşenler:
  Connectome    seyrek CSR bağlantı (FlyWire'dan veya sinek-benzeri sentetik), Dale, susturma
  LIFBrain      olay-güdümlü LIF simülatörü, paralel deneme (batch), Poisson uyarım
  MushroomBody  öğrenen modül: seyrek kod + 3-faktörlü (pre x post x dopamin) öğrenme + yenilik

Kullanım:
  DPY nora_neuron.py --stage test                          # öz-sınama (saniyeler)
  DPY nora_neuron.py --stage convert --flywire <klasör>    # parquet → kompakt npz (pandas+pyarrow bir kez)
  DPY nora_neuron.py --stage fly --trials 30               # sinek beyni: şeker → MN9 (hortum uzatma)
  DPY nora_neuron.py --stage fly --silence <flyid,...>     # susturma deneyi
  DPY nora_neuron.py --stage synth                         # sentetik sinek-benzeri beyin (veri gerekmez)
  DPY nora_neuron.py --stage mb                            # mantar cismi: koku→ödül/ceza, yenilik
  DPY nora_neuron.py --stage mbcl --dataset fashion        # mantar cismiyle sürekli öğrenme (5 görev)
"""
import argparse, os, sys, time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = os.path.join("D:" + os.sep, "NoRa_AGI") if os.name == "nt" else os.path.join(os.path.expanduser("~"), "NoRa_AGI")
FLY_DIR = os.path.join(PROJECT_ROOT, "data", "flywire")
FLY_NPZ = os.path.join(FLY_DIR, "flywire_783.npz")

# FlyWire kimlikleri (Shiu et al., sürüm 630 ve 783'te aynı)
SUGAR_R = [  # sağ labellar şeker-algılayan tat reseptör nöronları (GRN)
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
    720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
    720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
    720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
MN9 = 720575940660219265  # hortum (proboscis) uzatma motor nöronu

# Shiu et al. default_params (birimler: mV, ms)
LIF = dict(
    v0=-52.0, v_rst=-52.0, v_th=-45.0, t_mbr=20.0, tau=5.0, t_rfc=2.2, t_dly=1.8,
    w_syn=0.275, f_poi=250.0, dt=0.1,
)


# ================================================================ bağlantı (konektom)
class Connectome:
    """Seyrek yönlü çizge, çıkış-CSR. weight = işaretli sinaps sayısı (Dale: satır başına tek işaret)."""

    def __init__(self, n, pre, post, w, ids=None, regions=None):
        pre = np.asarray(pre, np.int64)
        order = np.argsort(pre, kind="stable")
        self.n = int(n)
        self.indices = np.asarray(post, np.int32)[order]
        self.weight = np.asarray(w, np.float32)[order]
        self.indptr = np.zeros(self.n + 1, np.int64)
        np.cumsum(np.bincount(pre, minlength=self.n), out=self.indptr[1:])
        self.ids = None if ids is None else np.asarray(ids, np.int64)
        self.regions = dict(regions or {})
        self._id2i = None

    # ---- kimlik / bölge
    def idx(self, flyids):
        if self.ids is None:
            return np.asarray(flyids, np.int64)
        if self._id2i is None:
            self._id2i = {int(f): i for i, f in enumerate(self.ids)}
        return np.array([self._id2i[int(f)] for f in flyids], np.int64)

    def sign(self):
        """Nöron başına işaret (+1 uyarıcı, -1 baskılayıcı, 0 çıkışsız)."""
        s = np.zeros(self.n, np.int8)
        has = np.diff(self.indptr) > 0
        s[has] = np.sign(self.weight[self.indptr[:-1][has]])
        return s

    def out_edges(self, src):
        """src nöronlarının tüm çıkış kenarları: (kaynak-sırası, hedef, ağırlık). Döngüsüz vektör."""
        st = self.indptr[src]
        ln = self.indptr[src + 1] - st
        tot = int(ln.sum())
        if tot == 0:
            return np.zeros(0, np.int64), self.indices[:0], self.weight[:0]
        which = np.repeat(np.arange(len(src)), ln)
        pos = np.arange(tot) - np.repeat(np.cumsum(ln) - ln, ln) + np.repeat(st, ln)
        return which, self.indices[pos], self.weight[pos]

    def stats(self):
        deg = np.diff(self.indptr)
        s = self.sign()
        mixed = 0
        if len(self.weight):
            sg = np.sign(self.weight)
            row = np.repeat(np.arange(self.n), deg)
            mixed = int(np.unique(row[sg != s[row]]).size)
        return dict(n=self.n, edges=len(self.indices), density=len(self.indices) / max(1, self.n) ** 2,
                    deg_med=float(np.median(deg)), deg_max=int(deg.max()) if self.n else 0,
                    inh_frac=float((s < 0).sum() / max(1, (s != 0).sum())), dale_violations=mixed)

    # ---- kayıt
    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        wmax = np.abs(self.weight).max() if len(self.weight) else 0
        w = self.weight.astype(np.int16) if wmax < 32767 and np.all(self.weight == np.round(self.weight)) else self.weight
        np.savez(path, n=self.n, indptr=self.indptr, indices=self.indices, weight=w,
                 ids=self.ids if self.ids is not None else np.zeros(0, np.int64))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        c = cls.__new__(cls)
        c.n, c.indptr, c.indices = int(z["n"]), z["indptr"], z["indices"]
        c.weight = z["weight"].astype(np.float32)
        c.ids = z["ids"] if len(z["ids"]) else None
        c.regions, c._id2i = {}, None
        return c

    # ---- kaynaklar
    @classmethod
    def from_flywire(cls, comp_csv, con_parquet):
        import pandas as pd  # yalnızca dönüştürmede gerekir
        comp = pd.read_csv(comp_csv, index_col=0)
        con = pd.read_parquet(con_parquet, columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
        return cls(len(comp), con["Presynaptic_Index"].values, con["Postsynaptic_Index"].values,
                   con["Excitatory x Connectivity"].values, ids=comp.index.values)

    @classmethod
    def synthetic(cls, n=20000, n_regions=8, deg_med=40, inh_frac=0.3, p_local=0.8, seed=0):
        """Sinek-benzeri sentetik beyin: bölgeli, ağır kuyruklu derece, Dale, log-normal sinaps sayısı.
        İstatistikler FlyWire'dan: medyan çıkış 79 (burada küçük n için ölçekli), %30 baskılayıcı,
        ort. |sinaps sayısı| ~3.6, bölge-içi bağlanma tercihi."""
        rng = np.random.default_rng(seed)
        region = np.sort(rng.integers(0, n_regions, n))
        bounds = np.searchsorted(region, np.arange(n_regions + 1))
        deg = np.minimum(rng.lognormal(np.log(deg_med), 0.8, n).astype(np.int64) + 1, n - 1)
        sign = np.where(rng.random(n) < inh_frac, -1.0, 1.0)
        pre = np.repeat(np.arange(n), deg)
        local = rng.random(pre.size) < p_local
        r = region[pre]
        lo, hi = bounds[r], bounds[r + 1]
        post = np.where(local, lo + (rng.random(pre.size) * (hi - lo)).astype(np.int64), rng.integers(0, n, pre.size))
        keep = post != pre
        pre, post = pre[keep], post[keep]
        cnt = np.maximum(1, np.round(rng.lognormal(0.6, 0.9, pre.size)))
        regions = {"R%d" % k: np.arange(bounds[k], bounds[k + 1]) for k in range(n_regions)}
        return cls(n, pre, post, cnt * sign[pre], regions=regions)


# ================================================================ LIF simülatörü
class LIFBrain:
    """Shiu et al. LIF modelinin NumPy karşılığı (Brian2 'linear' = tam üstel çözüm).
    B deneme paralel koşar (durum B x N). Sinaptik iletim olay-güdümlü: yalnızca ateşleyen
    nöronların çıkış kenarları dokunulur; gecikme halka-kuyrukla. Bellek ~ 4*B*N*4 bayt."""

    def __init__(self, con, params=None, seed=0):
        self.c = con
        self.p = dict(LIF, **(params or {}))
        self.rng = np.random.default_rng(seed)
        self.silenced = np.zeros(con.n, bool)

    def silence(self, idx):
        """Shiu: susturulan nöronun çıkış sinapsları 0 (ateşleyebilir ama etkisi yok)."""
        self.silenced[np.asarray(idx, np.int64)] = True

    def run(self, t_ms=1000.0, stim=None, trials=30, record=None, verbose=True):
        """stim: {nöron_indeksi: Hz} Poisson uyarım (optogenetik aktivasyon).
        Döner: oranlar (N,) Hz [denemeler ortalaması], isteğe bağlı record indeksleri için
        ani-ateşleme listesi [(deneme, nöron, t_ms)]."""
        p, n, B = self.p, self.c.n, int(trials)
        dt = p["dt"]
        steps = int(round(t_ms / dt))
        D = max(1, int(round(p["t_dly"] / dt)))
        ref_steps = int(round(p["t_rfc"] / dt))
        ev, eg = np.exp(-dt / p["t_mbr"]), np.exp(-dt / p["tau"])
        cg = p["tau"] / (p["tau"] - p["t_mbr"]) * (eg - ev)  # g'nin v'ye tam-integral katkısı
        v = np.full((B, n), p["v0"], np.float32)
        g = np.zeros((B, n), np.float32)
        rc = np.zeros((B, n), np.int16)  # refrakter geri sayım
        v_flat, g_flat = v.reshape(-1), g.reshape(-1)
        stim = stim or {}
        s_idx = np.array(list(stim.keys()), np.int64)
        s_prob = np.array([stim[k] * dt * 1e-3 for k in s_idx], np.float32)
        no_ref = np.zeros(n, bool)
        no_ref[s_idx] = True  # Shiu: Poisson hedeflerinde refrakter yok
        kick = p["w_syn"] * p["f_poi"]
        wscale = np.float32(p["w_syn"])
        queue = [None] * D  # halka: D adım önce ateşleyenler (düz indeks b*n+i)
        counts = np.zeros(n, np.int64)
        rec_set = None if record is None else np.zeros(n, bool)
        if rec_set is not None:
            rec_set[np.asarray(record, np.int64)] = True
        rec = []
        t0 = time.time()
        for step in range(steps):
            # 1) durum güncelleme (refrakter olmayanlar)
            free = rc == 0
            vn = p["v0"] + (v - p["v0"]) * ev + g * cg
            np.copyto(v, vn, where=free)
            np.copyto(g, g * eg, where=free)
            np.subtract(rc, 1, out=rc, where=~free)
            # 2) eşik
            spk = np.flatnonzero((v_flat > p["v_th"]) & (rc.reshape(-1) == 0))
            # 3) sinaps: D adım önceki ateşlemeler şimdi ulaşır; Poisson girdisi v'ye
            arr = queue[step % D]
            if arr is not None and arr.size:
                b, i = np.divmod(arr, n)
                which, post, w = self.c.out_edges(i)
                np.add.at(g_flat, b[which] * n + post, w * wscale)
            if s_idx.size:
                hit = self.rng.random((B, s_idx.size)) < s_prob
                hb, hk = np.nonzero(hit)
                v_flat[hb * n + s_idx[hk]] += kick
            # 4) sıfırlama
            if spk.size:
                v_flat[spk] = p["v_rst"]
                g_flat[spk] = 0.0
                bi = spk % n
                rc.reshape(-1)[spk] = np.where(no_ref[bi], 0, ref_steps)
                counts += np.bincount(bi, minlength=n)
                if rec_set is not None:
                    m = rec_set[bi]
                    rec.extend(zip((spk[m] // n).tolist(), bi[m].tolist(), [step * dt] * int(m.sum())))
                out = spk[~self.silenced[bi]]
                queue[step % D] = out
            else:
                queue[step % D] = None
            if verbose and steps >= 10 and step % (steps // 10) == 0 and step:
                print("    t=%6.0f ms  ateşleme/ms=%.1f  %.0fs" % (step * dt, counts.sum() / (step * dt) / B, time.time() - t0), flush=True)
        rates = counts / (B * t_ms * 1e-3)
        return (rates, rec) if record is not None else rates


# ================================================================ mantar cismi (öğrenme/bellek)
class MushroomBody:
    """Sinek mantar cismi (Kenyon hücreleri + MBON/DAN), oran-tabanlı, saf NumPy.

    PN  : girdi + yanal-baskılama ile kazanç normalizasyonu (ort. çıkar, norm böl)
    KC  : rastgele seyrek genişleme, her KC ~claws PN'den beslenir (sinekte ~7)
    APL : tek baskılayıcı nöron geri-beslemesi → yalnızca en güçlü %sparsity KC ateşler (k-WTA)
    MBON: bölmeler (compartment). KC→MBON sinapsları yalnızca o bölmenin DAN'ı ateşlerse ve
          KC aktifse değişir (3 faktör: pre x bölme x dopamin) → başka bilgiyi bozmaz.
    NOV : yenilik MBON'u (sinekte α'3): ağırlıklar 1'den başlar, gördükçe bastırılır.
          novelty(x) = aktif KC'lerin kalan ağırlık ortalaması (1=hiç görülmemiş, 0=çok tanıdık)."""

    def __init__(self, n_in, n_out, n_kc=2000, claws=7, sparsity=0.05, lr=1.0, nov_rate=0.3, seed=0):
        rng = np.random.default_rng(seed)
        self.n_in, self.n_out, self.n_kc = n_in, n_out, n_kc
        self.k = max(1, int(round(sparsity * n_kc)))
        cl = min(claws, n_in)
        self.claw = np.stack([rng.choice(n_in, cl, replace=False) for _ in range(n_kc)])  # (KC, claws)
        self.w_out = np.zeros((n_out, n_kc), np.float32)  # KC→MBON
        self.w_nov = np.ones(n_kc, np.float32)
        self.lr, self.nov_rate = lr, nov_rate

    def pn(self, X):
        X = np.atleast_2d(np.asarray(X, np.float32))
        X = X - X.mean(1, keepdims=True)
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)

    def kc(self, X):
        """Girdi → ikili seyrek KC kodu (B, n_kc)."""
        P = self.pn(X)
        drive = P[:, self.claw].sum(2)  # her KC kendi pençelerini toplar
        top = np.argpartition(-drive, self.k - 1, axis=1)[:, :self.k]  # APL: k-WTA
        K = np.zeros_like(drive, dtype=np.float32)
        np.put_along_axis(K, top, 1.0, axis=1)
        return K

    def mbon(self, X):
        return self.kc(X) @ self.w_out.T

    def reinforce(self, X, comp, dopamine=1.0):
        """DAN 'comp' bölmesine dopamin salar: aktif KC→comp sinapsları güçlenir (dopamin<0: zayıflar).
        Yalnızca o bölme ve o KC'ler değişir — eski bilgi yerinde kalır."""
        K = self.kc(X)
        comp = np.broadcast_to(np.asarray(comp), (len(K),))
        da = np.broadcast_to(np.asarray(dopamine, np.float32), (len(K),))
        np.add.at(self.w_out, comp, self.lr * da[:, None] * K)
        return K

    def novelty(self, X, learn=False):
        K = self.kc(X)
        nov = (K @ self.w_nov) / self.k
        if learn:
            seen = K.max(0) > 0
            self.w_nov[seen] *= (1.0 - self.nov_rate)
        return nov

    def predict(self, X):
        return self.mbon(X).argmax(1)


class LobedMushroomBody:
    """Loblu mantar cismi: sinekte KC alt-tipleri (α/β, α'/β', γ) FARKLI duyusal girdi tiplerini alır.
    Burada her duyusal grup (ör. 'yiyecek ışınları', 'duvar ışınları', 'propriosepsiyon') kendi KC
    popülasyonuna sahip; her KC grubundan rastgele `claws` girdi alır ve KENDİ rastgele eşiğine sahiptir
    (heterojen eşik -> yakınlık, ateşleyen KC SAYISIYLA kodlanır, yani analog bilgi seyrek ikili kodda
    korunur; girdi sessizse kod da sessizdir). APL geri-baskılaması toplamı `cap`'te keser.
    Kod, KC indeks dizisi olarak döner: Q ve yenilik sinapsları yalnızca bu indekslere dokunur."""

    def __init__(self, groups, n_per=500, claws=3, cap_frac=0.12, nov_rate=0.02, seed=0, thr_range=(0.04, 0.9)):
        rng = np.random.default_rng(seed)
        self.groups = [np.asarray(g) for g in groups]
        self.n_per, self.cap_frac = n_per, cap_frac
        self.claw = [g[np.stack([rng.choice(len(g), min(claws, len(g)), replace=False) for _ in range(n_per)])]
                     for g in self.groups]
        self.thr = [rng.uniform(thr_range[0], thr_range[1], n_per).astype(np.float32) for _ in self.groups]
        self.n_kc = n_per * len(self.groups)
        self.w_nov = np.ones(self.n_kc, np.float32)
        self.nov_rate = nov_rate

    def kc(self, x):
        out = []
        for gi, (cl, th) in enumerate(zip(self.claw, self.thr)):
            drive = x[cl].mean(1)
            fire = np.flatnonzero(drive > th)
            cap = int(self.cap_frac * self.n_per)
            if len(fire) > cap:
                fire = fire[np.argpartition(-drive[fire], cap - 1)[:cap]]
            out.append(fire + gi * self.n_per)
        return np.concatenate(out) if out else np.zeros(0, np.int64)

    def novelty(self, idx, learn=True):
        if len(idx) == 0:
            return 0.0
        nov = float(self.w_nov[idx].mean())
        if learn:
            self.w_nov[idx] *= (1.0 - self.nov_rate)
            self.w_nov += 0.002 * (1.0 - self.w_nov)
        return nov


# ================================================================ aşamalar
def stage_test(args):
    print("[test] sentetik konektom + LIF tutarlılığı")
    c = Connectome.synthetic(n=3000, deg_med=20, seed=1)
    st = c.stats()
    print("  ", st)
    assert st["dale_violations"] == 0, "Dale yasası bozuk"
    # tek nöron analitik: sabit g olmadan v dinlenmede kalır; büyük tekme → ateşler
    lone = Connectome(2, [0], [1], [40.0])  # 0 → 1, 40 sinaps * 0.275 = 11 mV tepe-g
    br = LIFBrain(lone, seed=0)
    r = br.run(t_ms=200, stim={0: 100.0}, trials=4, verbose=False)
    print("  2-nöron zinciri: kaynak %.0f Hz, hedef %.0f Hz" % (r[0], r[1]))
    assert 60 < r[0] < 140 and r[1] > 0, "zincir iletimi beklenen değil"
    br.silence([0])
    r2 = br.run(t_ms=200, stim={0: 100.0}, trials=4, verbose=False)
    assert r2[1] == 0, "susturma çalışmıyor"
    print("  susturma: hedef %.0f Hz (beklenen 0)" % r2[1])
    # out_edges vektör toplama doğruluğu
    src = np.array([5, 17, 5])
    which, post, w = c.out_edges(src)
    ref = np.concatenate([c.indices[c.indptr[s]:c.indptr[s + 1]] for s in src])
    assert np.array_equal(post, ref)
    # mantar cismi: yenilik düşer, öğrenme ayrık
    mb = MushroomBody(50, 2, n_kc=500, seed=0)
    x = np.random.default_rng(0).random((1, 50))
    n0 = mb.novelty(x, learn=True)[0]
    n1 = mb.novelty(x)[0]
    assert n1 < n0
    print("  yenilik: ilk %.2f → tekrar %.2f" % (n0, n1))
    print("[test] TAMAM")


def stage_convert(args):
    src = args.flywire
    comp = os.path.join(src, "Completeness_783.csv")
    con = os.path.join(src, "Connectivity_783.parquet")
    if args.v630:
        comp = os.path.join(src, "2023_03_23_completeness_630_final.csv")
        con = os.path.join(src, "2023_03_23_connectivity_630_final.parquet")
    t0 = time.time()
    c = Connectome.from_flywire(comp, con)
    out = args.npz or (FLY_NPZ if not args.v630 else FLY_NPZ.replace("783", "630"))
    c.save(out)
    print("dönüştürüldü: %s  (%.0fs, %.0f MB)" % (out, time.time() - t0, os.path.getsize(out) / 1e6))
    print("  ", c.stats())


def stage_fly(args):
    path = args.npz or FLY_NPZ
    if not os.path.exists(path):
        sys.exit("Konektom yok: %s\nÖnce: git clone https://github.com/philshiu/Drosophila_brain_model "
                 "ve  nora_neuron.py --stage convert --flywire <o klasör>" % path)
    t0 = time.time()
    c = Connectome.load(path)
    print("sinek beyni yüklendi: %d nöron, %d bağlantı (%.1fs)" % (c.n, len(c.indices), time.time() - t0))
    br = LIFBrain(c, seed=args.seed)
    if args.silence:
        br.silence(c.idx([int(s) for s in args.silence.split(",")]))
    sugar = c.idx(SUGAR_R)
    mn9 = c.idx([MN9])[0]
    print("uyarım: %d sağ şeker GRN @ %.0f Hz, %d deneme x %.0f ms" % (len(sugar), args.hz, args.trials, args.ms))
    t0 = time.time()
    rates = br.run(t_ms=args.ms, stim={int(i): args.hz for i in sugar}, trials=args.trials)
    print("süre: %.0fs" % (time.time() - t0))
    act = np.flatnonzero(rates > 0)
    print("aktif nöron: %d   MN9 (hortum uzatma): %.1f Hz" % (act.size, rates[mn9]))
    top = act[np.argsort(-rates[act])][:15]
    for i in top:
        print("   %d  %6.1f Hz%s" % (c.ids[i], rates[i], "  <- şeker GRN" if i in set(sugar) else ""))
    if args.ref:
        compare_ref(args.ref, c, rates)
    if args.out:
        np.savez(args.out, ids=c.ids[act], rates=rates[act])
        print("kaydedildi:", args.out)


def compare_ref(ref, c, rates):
    """Brian2 referans parquet'i (Drosophila_brain_model/results/...) ile nöron-nöron karşılaştırma."""
    import pandas as pd
    d = pd.read_parquet(ref)
    nt = d["trial"].nunique()
    T = max(1.0, np.ceil(d["t"].max()))
    rr = d.groupby("flywire_id").size() / nt / T
    ref_r = np.zeros(c.n)
    ref_r[c.idx(rr.index.values)] = rr.values
    both = (ref_r > 0) | (rates > 0)
    corr = np.corrcoef(ref_r[both], rates[both])[0, 1]
    mn9 = c.idx([MN9])[0]
    print("BRIAN2 KARŞILAŞTIRMA: aktif ref=%d biz=%d ortak=%d | oran korelasyonu r=%.3f | MN9 ref=%.1f biz=%.1f Hz"
          % ((ref_r > 0).sum(), (rates > 0).sum(), ((ref_r > 0) & (rates > 0)).sum(), corr, ref_r[mn9], rates[mn9]))


def stage_synth(args):
    c = Connectome.synthetic(n=args.n, seed=args.seed)
    print("sentetik sinek-benzeri beyin:", c.stats())
    br = LIFBrain(c, seed=args.seed)
    src = c.regions["R0"][:20]
    t0 = time.time()
    rates = br.run(t_ms=args.ms, stim={int(i): args.hz for i in src}, trials=args.trials)
    for name, ix in c.regions.items():
        print("   %s  ort %.2f Hz  aktif %d/%d" % (name, rates[ix].mean(), (rates[ix] > 0).sum(), ix.size))
    print("süre %.1fs" % (time.time() - t0))


def make_odors(rng, n_odor, n_pn, active=0.2):
    return (rng.random((n_odor, n_pn)) < active).astype(np.float32) * rng.random((n_odor, n_pn)).astype(np.float32)


def stage_mb(args):
    """Klasik sinek deneyi: kokular ödül (şeker) veya ceza (şok) ile eşlenir; sonra yaklaş/kaç."""
    rng = np.random.default_rng(args.seed)
    n_pn, n_odor = 50, 40  # sinekte ~50 glomerül
    odors = make_odors(rng, n_odor, n_pn)
    mb = MushroomBody(n_pn, 2, n_kc=2000, seed=args.seed)  # MBON 0=kaç (approach'u bastırır), 1=yaklaş
    val = rng.choice([-1, 1], n_odor)
    print("[mb] %d koku, yarısı ödül yarısı ceza; her koku 1 kez eşlenir (tek-atış öğrenme)" % n_odor)
    nov_first = mb.novelty(odors)
    for i in range(n_odor):
        x = odors[i:i + 1]
        noisy = x + 0.05 * rng.standard_normal(x.shape).astype(np.float32)
        mb.novelty(noisy, learn=True)
        mb.reinforce(noisy, 1 if val[i] > 0 else 0, 1.0)
    test = odors + 0.1 * rng.standard_normal(odors.shape).astype(np.float32)
    choice = np.where(mb.mbon(test)[:, 1] > mb.mbon(test)[:, 0], 1, -1)
    acc = (choice == val).mean()
    novel = make_odors(rng, n_odor, n_pn)
    print("  gürültülü kokularda doğru yaklaş/kaç: %.2f (şans 0.50)" % acc)
    print("  yenilik  ilk görüş %.2f | öğrenilmiş %.2f | hiç görülmemiş %.2f"
          % (nov_first.mean(), mb.novelty(test).mean(), mb.novelty(novel).mean()))
    kc = mb.kc(odors)
    ov = (kc @ kc.T)[~np.eye(n_odor, dtype=bool)].mean() / mb.k
    print("  KC kod seyrekliği %.1f%%, kokular arası ortalama KC örtüşmesi %.2f (girişim azlığı)" % (100 * kc.mean(), ov))


def stage_mbcl(args):
    """Sınıf-artımlı sürekli öğrenme: görevler sırayla gelir, eski görevler tekrar edilmez,
    test anında görev kimliği VERİLMEZ (en zor senaryo). Tek geçiş, geri yayılım yok."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from growing_net import load_dataset, parse_tasks
    data_dir = args.data_dir or os.path.join(PROJECT_ROOT, "data", args.dataset)
    Xtr, ytr, Xte, yte = load_dataset(args.dataset, data_dir)
    tasks = parse_tasks(args.tasks)
    mb = MushroomBody(Xtr.shape[1], 10, n_kc=args.kc, claws=args.claws, sparsity=args.sparsity, seed=args.seed)
    R = np.full((len(tasks), len(tasks)), np.nan)
    t0 = time.time()
    for ti, cls in enumerate(tasks):
        m = np.isin(ytr, cls)
        for s in range(0, m.sum(), 2000):
            xb, yb = Xtr[m][s:s + 2000], ytr[m][s:s + 2000]
            mb.reinforce(xb, yb, 1.0)
        for tj in range(ti + 1):
            mt = np.isin(yte, tasks[tj])
            seen = sorted(set(sum(tasks[:ti + 1], [])))
            out = mb.mbon(Xte[mt])
            norm = out[:, seen] / (mb.w_out[seen].sum(1) + 1e-6)  # bölme başına toplam ağırlıkla normalize
            R[ti, tj] = (np.array(seen)[norm.argmax(1)] == yte[mt]).mean()
        print("  görev %d %s  ort.acc(görülenler)=%.3f  %.1fs" % (ti, cls, np.nanmean(R[ti, :ti + 1]), time.time() - t0))
    bwt = np.mean([R[-1, j] - R[j, j] for j in range(len(tasks) - 1)]) if len(tasks) > 1 else 0.0
    print("SONUÇ %s: son ort.acc (görev-bilinmez, 10-yollu) %.3f | BWT %.3f | %.1fs"
          % (args.dataset, np.nanmean(R[-1]), bwt, time.time() - t0))


def main():
    ap = argparse.ArgumentParser(description="NoRa nöron temeli (sinek beyni)")
    ap.add_argument("--stage", choices=["test", "convert", "fly", "synth", "mb", "mbcl"], default="test")
    ap.add_argument("--flywire", default=".", help="Drosophila_brain_model klasörü (convert)")
    ap.add_argument("--v630", action="store_true", help="makaledeki 630 sürümünü kullan")
    ap.add_argument("--npz", default=None)
    ap.add_argument("--ref", default=None, help="Brian2 referans parquet (fly karşılaştırma)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--silence", default=None, help="susturulacak FlyWire kimlikleri, virgüllü")
    ap.add_argument("--hz", type=float, default=200.0)
    ap.add_argument("--ms", type=float, default=1000.0)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--dataset", choices=["mnist", "fashion"], default="fashion")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--tasks", default="0,1;2,3;4,5;6,7;8,9")
    ap.add_argument("--kc", type=int, default=8000)
    ap.add_argument("--claws", type=int, default=40)
    ap.add_argument("--sparsity", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"test": stage_test, "convert": stage_convert, "fly": stage_fly, "synth": stage_synth,
     "mb": stage_mb, "mbcl": stage_mbcl}[args.stage](args)


if __name__ == "__main__":
    main()
