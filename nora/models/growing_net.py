from nora._paths import REPO_ROOT
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
growing_net.py — Büyüyerek öğrenen (structural plasticity) minimal sürekli öğrenme prototipi v3
================================================================================================
v3 düzeltmeleri (kod incelemesi 8 madde):
  1. _train v1'e döndü (tek softmax); log_softmax YALNIZCA NLL raporunda (nll_loss).
     exp(log_softmax) == softmax olduğundan gradyanda kullanımı regresyondu.
  2. _build_prototypes ölü satırı silindi.
  3. reuse_hist çift korumalı: learn_task büyüdüyse None döner, main yalnızca g==0 iken
     ekler. Plato kuralı artık büyümeyi atlatmaz; her zaman önce mevcutlarla denenir.
  4. ensemble temiz z-skor + inf düşme (DENEYSEL etiketiyle); energy/proto ayrı raporlanmalı.
  5. --fast = hızlı-NCM TABAN ÇİZGİSİ (rastgele/sabit özellik + nearest-mean), eğitim
     doğruluğu kalite kapısı olarak loglanır. NCM→lineer çevrimi yorumda türetildi.
  6. PROJECT_ROOT taşınabilir: Windows'ta D:\\NoRa_AGI, başka OS'te ~/NoRa_AGI.
  7. sleep gerçek süre ölçer; |W| metriği "aday" olarak etiketli (gerçek ölü nöron aktivasyondadır).
  8. growth_map npz'ye kaydedilir/yüklenir; yoksa rapor yalnızca bu oturumu gösterir.
v4 (AGI yönü: etiket yok, maliyet sabit):
  - surprise()/recall(): dondurulmuş özelliklerde şaşkınlık + epizodik geri-çağırma.
  - --stream: görev sınırları verilmez; referans-çapalı çift kural (seviye+değişim)
    otomatik böler. Parça maliyeti sabit -> veri büyüse de adım maliyeti büyümez.
v2 iyileştirmeleri (öneriler uygulandı):
  1. log_softmax + NLL: gradyan daha stabil, 2-3 satır daha kısa (softmax'taki max çıkarma korundu).
  2. MIRRORS["fashion"]: HTTPS github raw İLK sıraya alındı (http s3 yedek).
  3. save/load: pickle yerine np.savez (.npz). Eski .pkl dosyaları load'da otomatik tanınır (geriye uyumlu).
  4. Büyüme haritası: raporun başında tablo (görev, blok, nöron, param, val-acc, eşik, karar).
  5. Kademeli eşik + eğilim kuralı: T(k)=max(0.85, base-0.03*(k-1)); son W görevde plato varsa büyü.
  6. Yönlendirici v2 (yol haritası m.1): --router proto|energy|ensemble|conf (varsayılan ensemble).
     - proto: her sınıfın özellik ortalamasına (prototip) en yakın kafa
     - energy: E=-logsumexp(logit) en düşük kafa
  7. Tüm varsayılan yollar D diskinde (C'de yer yok): D:\\NoRa_AGI\\data, D:\\NoRa_AGI\\beyin.npz

FİKİR (tek cümle):
    Önce mevcut nöronlarla öğrenmeyi dene; yetmezse YENİ nöronlar ekle.
    Eski nöronlar dondurulur, yeni nöronlar eskilere yan bağlantılarla bağlanır.

KULLANIM (D diskinde):
    python D:\\NoRa_AGI\\growing_net.py                                   # MNIST, 5 görev
    python D:\\NoRa_AGI\\growing_net.py --dataset fashion                 # Fashion-MNIST
    python D:\\NoRa_AGI\\growing_net.py --tasks "0,1;2,3;4,5" --save D:\\NoRa_AGI\\beyin.npz
    python D:\\NoRa_AGI\\growing_net.py --resume D:\\NoRa_AGI\\beyin.npz --tasks "6,7;8,9" --save D:\\NoRa_AGI\\beyin.npz
"""

import argparse
import gzip
import os
import pickle
import struct
import sys
import time
import urllib.request

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------------------------------------------------------------------
# 0) VARSAYILAN YOLLAR (bu makinede C dolu -> D; başka OS'te ev dizini)
# ---------------------------------------------------------------------------
if os.name == "nt":
    PROJECT_ROOT = os.path.join("D:" + os.sep, "NoRa_AGI")
else:  # Linux/Mac: "D:" adlı klasöre sessizce yazma, ev dizinine git
    PROJECT_ROOT = os.path.join(os.path.expanduser("~"), "NoRa_AGI")
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
DEFAULT_SAVE = os.path.join(PROJECT_ROOT, "beyin.npz")

# ---------------------------------------------------------------------------
# 1) VERİ
# ---------------------------------------------------------------------------
MIRRORS = {
    "mnist": [
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
    ],
    "fashion": [
        # ÖNERİ: HTTPS github raw ilk sıraya alındı (daha güvenli/hızlı).
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
    ],
}
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}


def read_idx(path):
    """IDX .gz -> numpy. data[3]=ndim okuma IDX standardına uygun."""
    with gzip.open(path, "rb") as f:
        data = f.read()
    assert data[0] == 0 and data[1] == 0, f"IDX magic bozuk: {path}"
    ndim = data[3]
    assert 1 <= ndim <= 4, f"IDX ndim garip: {ndim}"
    dims = struct.unpack(">" + "I" * ndim, data[4:4 + 4 * ndim])
    arr = np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)
    return arr


def download(name, filename, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    last_err = None
    for base in MIRRORS[name]:
        try:
            print(f"  indiriliyor: {base}{filename}")
            urllib.request.urlretrieve(base + filename, dest)
            return
        except Exception as e:
            print(f"  ! ayna başarısız ({base}): {e}")
            last_err = e
    raise RuntimeError(
        f"{filename} indirilemedi ({last_err}).\n"
        f"Dosyaları elle indirip '{os.path.dirname(dest)}' klasörüne koyun ve --data-dir ile gösterin."
    )


def load_dataset(name, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    raw = {}
    for key, fn in FILES.items():
        path = os.path.join(data_dir, fn)
        if not os.path.exists(path):
            download(name, fn, path)
        raw[key] = read_idx(path)
    Xtr = raw["train_x"].reshape(len(raw["train_x"]), -1).astype(np.float32) / 255.0
    Xte = raw["test_x"].reshape(len(raw["test_x"]), -1).astype(np.float32) / 255.0
    return Xtr, raw["train_y"].astype(np.int64), Xte, raw["test_y"].astype(np.int64)


def parse_tasks(s):
    return [[int(c) for c in part.split(",")] for part in s.split(";") if part.strip()]


def task_subset(X, y, classes):
    m = np.isin(y, classes)
    local = np.searchsorted(np.array(sorted(classes)), y[m])
    return X[m], y[m], local


# ---------------------------------------------------------------------------
# 2) YARDIMCI MATEMATİK
# ---------------------------------------------------------------------------
def relu(z):
    return np.maximum(z, 0.0)


def log_softmax(z):
    """YALNIZCA NLL loss raporlama için (bkz. nll_loss). Gradyan için KULLANMA:
    exp(log_softmax(z)) == softmax(z) olduğundan gradyanda faydası yok, sadece yavaşlatır."""
    z = z - z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1, keepdims=True) + 1e-12)
    return z - lse


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def nll_loss(logits, y):
    """Gerçek NLL: -mean(log p_y). log_softmax'in stabilize ettiği tek yer burası."""
    logp = log_softmax(logits)
    return float(-logp[np.arange(len(y)), y].mean())


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


# ---------------------------------------------------------------------------
# 2b) KADEMELİ EŞİK + EĞİLİM KURALI
# ---------------------------------------------------------------------------
def tiered_threshold(base, task_idx_0based, floor=0.85, step=0.03):
    """Görev grubuna göre kademeli eşik: G1=base, G2=base-0.03, ... taban 0.85."""
    return max(floor, base - step * task_idx_0based)


def ema_trend(acc_history, window=3, beta=0.6):
    """Son W görevdeki mevcut-nöron başarısının EMA eğimi. (eğim, plato_mu) döner."""
    if len(acc_history) < 1:
        return 0.0, False
    w = acc_history[-window:]
    ema = w[0]
    for a in w[1:]:
        ema = beta * a + (1 - beta) * ema
    slope = (ema - w[0]) / max(1, len(w) - 1)
    plato = (max(w) - min(w)) < 0.005 and slope < 0.003
    return slope, plato


# ---------------------------------------------------------------------------
# 3) BÜYÜYEN AĞ
# ---------------------------------------------------------------------------
class GrowingNet:
    def __init__(self, in_dim, block_size, seed=0):
        self.in_dim = in_dim
        self.block_size = block_size
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.blocks = []
        self.heads = []  # {"V","c","n_feat","classes","protos": {cls: mean_vec}}
        self.growth_map = []  # rapor tablosu satırları

    @property
    def n_features(self):
        return len(self.blocks) * self.block_size

    def n_params(self):
        n = sum(b["W"].size + b["U"].size + b["b"].size for b in self.blocks)
        return n + sum(h["V"].size + h["c"].size for h in self.heads)

    def add_block(self):
        k, D, F_old = self.block_size, self.in_dim, self.n_features
        self.blocks.append({
            "W": self.rng.normal(0, np.sqrt(2.0 / D), (D, k)).astype(np.float32),
            "U": (self.rng.normal(0, np.sqrt(2.0 / max(F_old, 1)), (F_old, k)) * 0.5).astype(np.float32),
            "b": np.zeros(k, dtype=np.float32),
        })

    def new_head(self, classes):
        F = self.n_features
        return {"V": self.rng.normal(0, np.sqrt(1.0 / max(F, 1)), (F, len(classes))).astype(np.float32),
                "c": np.zeros(len(classes), dtype=np.float32),
                "n_feat": F, "classes": sorted(classes), "protos": {}}

    def features(self, X, n_blocks=None):
        n_blocks = len(self.blocks) if n_blocks is None else n_blocks
        H = np.zeros((len(X), 0), dtype=np.float32)
        for blk in self.blocks[:n_blocks]:
            z = X @ blk["W"] + blk["b"]
            if H.shape[1]:
                z += H @ blk["U"]
            H = np.concatenate([H, relu(z)], axis=1)
        return H

    def _train(self, head, X, y_local, train_new_block, epochs, lr, batch):
        n_frozen = len(self.blocks) - (1 if train_new_block else 0)
        H_frozen = self.features(X, n_frozen)
        params = [head["V"], head["c"]]
        if train_new_block:
            blk = self.blocks[-1]
            params += [blk["W"], blk["b"], blk["U"]]
        opt = Adam(params, lr=lr)
        N = len(X)
        for _ in range(epochs):
            perm = self.rng.permutation(N)
            for s in range(0, N, batch):
                idx = perm[s:s + batch]
                x, yb, Hf = X[idx], y_local[idx], H_frozen[idx]
                if train_new_block:
                    z = x @ blk["W"] + blk["b"]
                    if Hf.shape[1]:
                        z += Hf @ blk["U"]
                    H = np.concatenate([Hf, relu(z)], axis=1)
                else:
                    H = Hf
                # v1'e dönüş: tek softmax, gradyan dL/dlogits = (softmax-onehot)/B.
                # (log_softmax+exp eşdeğerdir, sadece yavaşlatır; NLL yalnızca loss raporunda.)
                logits = H @ head["V"] + head["c"]
                p = softmax(logits)
                p[np.arange(len(yb)), yb] -= 1.0
                d = p / len(yb)
                grads = [H.T @ d, d.sum(0)]
                if train_new_block:
                    k = self.block_size
                    dz = (d @ head["V"][-k:].T) * (z > 0)
                    grads += [x.T @ dz, dz.sum(0), Hf.T @ dz]
                opt.step(grads)

    def _head_acc(self, head, X, y_local):
        H = self.features(X)[:, :head["n_feat"]]
        return float(((H @ head["V"] + head["c"]).argmax(1) == y_local).mean())

    def _build_prototypes(self, head_idx, X, y_local):
        """Her sınıfın özellik-ortalamasını sakla (yönlendirici v2 için)."""
        head = self.heads[head_idx]
        H = self.features(X)[:, :head["n_feat"]]
        protos = {}
        for li, cls in enumerate(head["classes"]):
            m = (y_local == li)
            if m.sum() > 0:
                protos[int(cls)] = H[m].mean(axis=0).astype(np.float32)
        head["protos"] = protos

    def fast_add_task(self, classes, X, y_local, log=print):
        """HIZLI TABAN ÇİZGİSİ (büyümeli öğrenme DEĞİL: sabit/rastgele özellik + NCM).
        Geri yayılım yok (<1sn). İlk görevde omurga rastgeledir (eğitimsiz ReLU) —
        bu "rastgele özellik + nearest-mean" taban çizgisidir, dürüstçe böyle sunulur.
        NCM→lineer çevrimi: -||h-mu||² = -||h||² + 2h·mu - ||mu||²; -||h||² tüm
        sınıflarda ortak olduğundan argmax'ta düşer → V[:,c]=2*mu_c, c_c=-||mu_c||²
        (argmax aynı, ama softmax olasılıkları NCM'e göre kalibre değildir).
        Kalite kapısı: eğitim kümesi doğruluğu hesaplanıp growth_map'e yazılır."""
        import time as _t
        t0 = _t.time()
        if not self.blocks:  # ilk görev: rastgele, EĞİTİMSİZ omurga (taban çizgisi!)
            self.add_block()
            grown = 1
        else:
            grown = 0
        head = self.new_head(classes)
        H = self.features(X)[:, :head["n_feat"]]
        mus = []
        for li in range(len(head["classes"])):
            m = (y_local == li)
            mu = H[m].mean(axis=0).astype(np.float32) if m.sum() else np.zeros(H.shape[1], np.float32)
            mus.append(mu)
        M = np.stack(mus, axis=1)  # (F, C)
        head["V"][:] = 2.0 * M
        head["c"][:] = -(M ** 2).sum(axis=0)
        self.heads.append(head)
        self._build_prototypes(len(self.heads) - 1, X, y_local)
        train_acc = float((((H @ head["V"] + head["c"]).argmax(1)) == y_local).mean())
        dt = _t.time() - t0
        self.growth_map.append(dict(task=list(classes), added=grown, total_n=self.n_features,
                                    total_p=self.n_params(), dparam=0,
                                    val_acc=train_acc, thr=float('nan'),
                                    karar=f"hızlı-NCM({dt:.2f}sn)"))
        log(f"    hızlı-NCM taban çizgisi: {dt:.2f} sn, blok+{grown}, eğitim-doğruluğu={train_acc:.3f} (eğitim yok)")
        return grown

    def sleep_consolidate(self, log=print, act_thresh=1e-3):
        """UYKU FAZI (yıkıcı değil v3): yalnızca rapor + sayaç, budama YOK (U şekilleri bozulmasın).
        "Ölü aday" metriği |W| ortalamasıdır — gerçek ölü nöron AKTİVASYONA bakılarak
        bulunur (girdi dağılımına bağlı); buradaki sayı üstünkörü bir aday sayımıdır."""
        import time as _t
        t0 = _t.time()
        n_dead = 0
        for b in self.blocks:
            n_dead += int((np.abs(b["W"]).mean(axis=0) < act_thresh).sum())
        dt = _t.time() - t0
        log(f"    uyku: {n_dead} ölü ADAY nöron (|W| vekili, aktivasyon değil), "
            f"{len(self.heads)} prototip kaydı mevcut ({dt * 1000:.1f} ms)")
        return n_dead

    # --- v4: dünya-modelciği (şaşkınlık) + epizodik hafıza ---
    def _all_protos(self):
        """(vektör, global_etiket, head_idx) listesi — epizodik hafıza deposu."""
        out = []
        for ji, h in enumerate(self.heads):
            for c, v in h.get("protos", {}).items():
                out.append((v, int(c), ji))
        return out

    def surprise(self, X):
        """Şaşkınlık: her örneğin en yakın prototipe uzaklığı.
        Kafalar farklı boyutta olabilir (büyüme sonrası) -> her kafa kendi
        alt-uzayında ölçer, en küçük uzaklık alınır."""
        H = self.features(X)
        best = np.full(len(X), np.inf)
        for h in self.heads:
            P = h.get("protos")
            if not P:
                continue
            M = np.stack([P[c] for c in h["classes"] if c in P])
            d = ((H[:, :h["n_feat"], None] - M.T[None, :, :]) ** 2).sum(1).min(1)
            best = np.minimum(best, d)
        return best

    def recall(self, h_vec, k=1):
        """Epizodik geri-çağırma: verilen özellik vektörüne en yakın k anı.
        Döner: [(global_etiket, uzaklık, head_idx)]. Planlamanın 1-adım yoklaması burada çalışır."""
        P = self._all_protos()
        if not P:
            return []
        best = []
        for v, c, ji in P:
            F = len(v)
            best.append((((h_vec[:F] - v) ** 2).sum(), c, ji))
        best.sort(key=lambda t: t[0])
        return [(c, float(d), ji) for d, c, ji in best[:k]]

    def learn_task(self, classes, X, y_local, Xval, yval_local, args, log,
                   task_idx_0based=0, reuse_acc_history=None):
        reuse_acc_history = reuse_acc_history or []
        thr = tiered_threshold(args.grow_threshold, task_idx_0based) if args.adaptive_threshold else args.grow_threshold
        grown = 0
        params_before = self.n_params()
        val_acc_reuse = None

        # Eğilim kuralı SADECE bilgi amaçlı: plato varsa loglanır ama yeniden-kullanma
        # denemesi HER ZAMAN yapılır ("önce mevcutlarla dene" felsefesine sadakat).
        _, plato = ema_trend(reuse_acc_history, window=args.trend_window)
        if plato and len(reuse_acc_history) >= 2:
            log(f"    (not: son {args.trend_window} görevde plato var, yine de önce mevcutlarla deneniyor)")

        if self.blocks:
            head = self.new_head(classes)
            self._train(head, X, y_local, False, args.epochs, args.lr, args.batch)
            Hv = self.features(Xval)[:, :head["n_feat"]]  # bir kez hesapla: acc + NLL ikisi de kullanır
            acc = float(((Hv @ head["V"] + head["c"]).argmax(1) == yval_local).mean())
            val_acc_reuse = acc
            log(f"    mevcut {self.n_features} nöronla doğrulama: {acc:.3f} (eşik {thr:.2f}, NLL={nll_loss(Hv @ head['V'] + head['c'], yval_local):.3f})")
            if acc >= thr:
                log("    -> yeterli, BÜYÜMEYE GEREK YOK (eski bilgi yeniden kullanıldı)")
                self.heads.append(head)
                self._build_prototypes(len(self.heads) - 1, X, y_local)
                self.growth_map.append(dict(task=list(classes), added=0, total_n=self.n_features,
                                            total_p=self.n_params(), dparam=self.n_params() - params_before,
                                            val_acc=acc, thr=thr, karar="yeniden-kullan"))
                return grown, val_acc_reuse
            log("    -> yetersiz, BÜYÜYOR (yeni nöronlar eklenecek)")

        while grown < args.max_grow:
            self.add_block()
            grown += 1
            head = self.new_head(classes)
            self._train(head, X, y_local, True, args.epochs, args.lr, args.batch)
            acc = self._head_acc(head, Xval, yval_local)
            log(f"    +{self.block_size} yeni nöron (toplam {self.n_features}) -> doğrulama: {acc:.3f}")
            if acc >= thr:
                break
        self.heads.append(head)
        self._build_prototypes(len(self.heads) - 1, X, y_local)
        self.growth_map.append(dict(task=list(classes), added=grown, total_n=self.n_features,
                                    total_p=self.n_params(), dparam=self.n_params() - params_before,
                                    val_acc=acc, thr=thr,
                                    karar=f"büyü(+{grown})" if grown else "yeniden-kullan"))
        # Büyüme sonrası acc BURAYA karışmaz: eğilim kuralı yalnızca yeniden-kullanma
        # denemelerini görmeli. Büyüdüysek None dön.
        return grown, (val_acc_reuse if grown == 0 else None)

    # --- tahmin ---
    def predict_task(self, head_idx, X):
        h = self.heads[head_idx]
        H = self.features(X)[:, :h["n_feat"]]
        return np.array(h["classes"])[(H @ h["V"] + h["c"]).argmax(1)]

    def _head_logits(self, X):
        H = self.features(X)
        out = []
        for h in self.heads:
            out.append(H[:, :h["n_feat"]] @ h["V"] + h["c"])
        return H, out

    def predict_any(self, X, method="ensemble"):
        """Görev bilinmiyor. method: conf|proto|energy|ensemble"""
        H, logits_list = self._head_logits(X)
        n = len(X)
        if method == "conf":
            best_conf = np.full(n, -np.inf)
            pred = np.zeros(n, dtype=np.int64)
            for h, logits in zip(self.heads, logits_list):
                p = softmax(logits)
                conf = p.max(1)
                win = conf > best_conf
                best_conf[win] = conf[win]
                pred[win] = np.array(h["classes"])[p.argmax(1)][win]
            return pred
        # enerji + prototip skorları
        # düzgün enerji: E = -logsumexp(logit)
        energies = []
        for logits in logits_list:
            m = logits.max(1, keepdims=True)
            energies.append(-(m[:, 0] + np.log(np.exp(logits - m).sum(1) + 1e-20)))
        energies = np.stack(energies)  # (T, N) düşük = iyi
        # prototip mesafesi: her kafa için min sınıf mesafesi
        dists = []
        for h in self.heads:
            if h.get("protos"):
                P = np.stack([h["protos"][c] for c in h["classes"]])  # (C, F)
                Hh = H[:, :h["n_feat"]]
                # (N, C) öklid
                d = ((Hh[:, None, :] - P[None, :, :]) ** 2).sum(-1).min(1)
            else:
                d = np.full(n, np.inf)
            dists.append(d)
        dists = np.stack(dists)  # (T, N) düşük = iyi
        if method == "energy":
            pick = energies.argmin(0)
        elif method == "proto":
            pick = dists.argmin(0)
        else:  # ensemble (DENEYSEL): enerji + prototip z-skorlarının eşit ağırlıklı toplamı.
            # Dürüstlük notu: iki farklı fiziksel anlamı eşit ağırlıkla karıştırmak keyfi
            # bir varsayımdır; varyansı büyük olan ezer. Güvenilir karşılaştırma için
            # energy ve proto'yu ayrı raporlayın (--router ile tek tek koşun).
            def z(v):
                mu = np.nanmean(v, axis=0, keepdims=True)
                sd = np.nanstd(v, axis=0, keepdims=True) + 1e-6
                return (v - mu) / sd
            d_clean = np.where(np.isinf(dists), np.nan, dists)
            if np.isnan(d_clean).all():  # hiç prototip yoksa enerjiye düş
                pick = energies.argmin(0)
            else:
                col_max = np.nanmax(d_clean, axis=0, keepdims=True)
                d_clean = np.where(np.isnan(d_clean), col_max, d_clean)  # inf -> sütun max'ı
                score = z(energies) + z(d_clean)
                pick = score.argmin(0)
        pred = np.zeros(n, dtype=np.int64)
        for t, (h, logits) in enumerate(zip(self.heads, logits_list)):
            m = (pick == t)
            if m.sum():
                pred[m] = np.array(h["classes"])[logits[m].argmax(1)]
        return pred

    # --- kaydet / yükle (.npz; eski .pkl'e geriye uyumlu) ---
    def save(self, path):
        if path.endswith(".pkl"):
            path = path[:-4] + ".npz"
            print(f"  (not: .pkl yerine .npz yazılıyor: {path})")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        arr = dict(in_dim=np.array(self.in_dim), block_size=np.array(self.block_size),
                   n_blocks=np.array(len(self.blocks)), n_heads=np.array(len(self.heads)))
        for i, b in enumerate(self.blocks):
            arr[f"b{i}_W"], arr[f"b{i}_U"], arr[f"b{i}_b"] = b["W"], b["U"], b["b"]
        for j, h in enumerate(self.heads):
            arr[f"h{j}_V"], arr[f"h{j}_c"] = h["V"], h["c"]
            arr[f"h{j}_nfeat"] = np.array(h["n_feat"])
            arr[f"h{j}_classes"] = np.array(h["classes"], dtype=np.int64)
            if h.get("protos"):
                for c, v in h["protos"].items():
                    arr[f"h{j}_proto_{c}"] = v
        # growth_map kalıcıdır; yoksa rapor "yalnızca bu oturumun kararlarını" gösterir.
        if self.growth_map:
            arr["gm_task"] = np.array([",".join(map(str, g["task"])) for g in self.growth_map])
            arr["gm_added"] = np.array([g["added"] for g in self.growth_map], dtype=np.int64)
            arr["gm_total_n"] = np.array([g["total_n"] for g in self.growth_map], dtype=np.int64)
            arr["gm_total_p"] = np.array([g["total_p"] for g in self.growth_map], dtype=np.int64)
            arr["gm_val_acc"] = np.array([g["val_acc"] for g in self.growth_map], dtype=np.float64)
            arr["gm_thr"] = np.array([g["thr"] for g in self.growth_map], dtype=np.float64)
            arr["gm_karar"] = np.array([g["karar"] for g in self.growth_map])
        np.savez_compressed(path, **arr)
        return path

    @classmethod
    def load(cls, path, seed=0):
        if path.endswith(".pkl") and not os.path.exists(path):
            alt = path[:-4] + ".npz"
            if os.path.exists(alt):
                path = alt
        if path.endswith(".pkl"):
            with open(path, "rb") as f:  # eski dosyalar için
                d = pickle.load(f)
            net = cls(d["in_dim"], d["block_size"], seed)
            net.blocks, net.heads = d["blocks"], d["heads"]
            for h in net.heads:
                h.setdefault("protos", {})
            return net
        z = np.load(path, allow_pickle=True)
        net = cls(int(z["in_dim"]), int(z["block_size"]), seed)
        for i in range(int(z["n_blocks"])):
            net.blocks.append({"W": z[f"b{i}_W"], "U": z[f"b{i}_U"], "b": z[f"b{i}_b"]})
        for j in range(int(z["n_heads"])):
            h = {"V": z[f"h{j}_V"], "c": z[f"h{j}_c"],
                 "n_feat": int(z[f"h{j}_nfeat"]), "classes": list(map(int, z[f"h{j}_classes"])),
                 "protos": {}}
            for c in h["classes"]:
                k = f"h{j}_proto_{c}"
                if k in z:
                    h["protos"][c] = z[k]
            net.heads.append(h)
        if "gm_task" in z:  # eski kayıtlarda yoksa growth_map boş kalır (yalnızca bu oturum raporlanır)
            for i in range(len(z["gm_task"])):
                net.growth_map.append(dict(
                    task=[int(c) for c in str(z["gm_task"][i]).split(",") if c],
                    added=int(z["gm_added"][i]), total_n=int(z["gm_total_n"][i]),
                    total_p=int(z["gm_total_p"][i]), val_acc=float(z["gm_val_acc"][i]),
                    thr=float(z["gm_thr"][i]), karar=str(z["gm_karar"][i])))
        return net


# ---------------------------------------------------------------------------
# 4) KLASİK AĞ
# ---------------------------------------------------------------------------
class PlainMLP:
    def __init__(self, in_dim, hidden, n_out, seed=0):
        r = np.random.default_rng(seed)
        self.rng = r
        self.W1 = r.normal(0, np.sqrt(2.0 / in_dim), (in_dim, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, np.float32)
        self.W2 = r.normal(0, np.sqrt(1.0 / hidden), (hidden, n_out)).astype(np.float32)
        self.b2 = np.zeros(n_out, np.float32)

    def n_params(self):
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def logits(self, X):
        return relu(X @ self.W1 + self.b1) @ self.W2 + self.b2

    def learn_task(self, X, y_global, args):
        opt = Adam([self.W1, self.b1, self.W2, self.b2], lr=args.lr)
        N = len(X)
        for _ in range(args.epochs):
            perm = self.rng.permutation(N)
            for s in range(0, N, args.batch):
                idx = perm[s:s + args.batch]
                x, yb = X[idx], y_global[idx]
                z1 = x @ self.W1 + self.b1
                h = relu(z1)
                p = softmax(h @ self.W2 + self.b2)  # v1: NLL yalnızca raporda, bkz. nll_loss
                p[np.arange(len(yb)), yb] -= 1.0
                d = p / len(yb)
                dz1 = (d @ self.W2.T) * (z1 > 0)
                opt.step([x.T @ dz1, dz1.sum(0), h.T @ d, d.sum(0)])

    def predict_task(self, classes, X):
        cl = sorted(classes)
        return np.array(cl)[self.logits(X)[:, cl].argmax(1)]

    def predict_any(self, seen, X):
        s = sorted(seen)
        return np.array(s)[self.logits(X)[:, s].argmax(1)]


# ---------------------------------------------------------------------------
# 5) DENEY
# ---------------------------------------------------------------------------
def print_matrix(title, R, names):
    print(f"\n  {title}  (satır: o göreve kadar öğrendikten sonra, sütun: test edilen görev)")
    print("  " + " " * 12 + "".join(f"{n:>9}" for n in names))
    for i, row in enumerate(R):
        cells = "".join(f"{v:9.3f}" if v is not None else f"{'':>9}" for v in row)
        print(f"  {'sonra ' + names[i]:<12}{cells}")


def print_growth_map(net, names):
    # NOT: growth_map npz'de saklanır; eski kayıtlarda yoksa yalnızca bu oturum görünür.
    # names[-len(growth_map):] hizalaması bu yüzden oturum-içidir, resume'da kayma olmaz.
    def f(x, fmt):
        import math
        return ("-" * 5) if (isinstance(x, float) and math.isnan(x)) else format(x, fmt)
    print("\n  BÜYÜME HARİTASI (rapor başı özeti):")
    print("  {:<8} {:>5} {:>8} {:>10} {:>9} {:>7}  {}".format(
        "görev", "eklenen", "toplamN", "toplamP", "val-acc", "eşik", "karar"))
    for n, g in zip(names[-len(net.growth_map):] if net.growth_map else [], net.growth_map):
        print("  {:<8} {:>5} {:>8} {:>10,} {:>9} {:>7}  {}".format(
            n, g["added"], g["total_n"], g["total_p"], f(g["val_acc"], ".3f"), f(g["thr"], ".2f"), g["karar"]))
    if net.growth_map:
        tot_add = sum(g["added"] for g in net.growth_map)
        print(f"  Toplam eklenen blok: {tot_add}, son nöron: {net.n_features}, son param: {net.n_params():,}")


def summarize(R):
    T = len(R)
    final = [R[T - 1][j] for j in range(T)]
    bwt = np.mean([R[T - 1][j] - R[j][j] for j in range(T - 1)]) if T > 1 else 0.0
    return float(np.mean(final)), float(bwt)


def main():
    ap = argparse.ArgumentParser(description="Büyüyerek öğrenen sürekli öğrenme prototipi v2 (D diski)")
    ap.add_argument("--dataset", choices=["mnist", "fashion"], default="mnist")
    ap.add_argument("--data-dir", default=None, help="IDX .gz klasörü (varsayılan D:\\NoRa_AGI\\data\\<dataset>)")
    ap.add_argument("--tasks", default="0,1;2,3;4,5;6,7;8,9")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--grow-threshold", type=float, default=0.97, help="baz eşik (kademeli azalır)")
    ap.add_argument("--no-adaptive", dest="adaptive_threshold", action="store_false",
                    help="kademeli eşiği kapat, sabit eşik kullan")
    ap.set_defaults(adaptive_threshold=True)
    ap.add_argument("--trend-window", type=int, default=3, help="eğilim kuralı penceresi")
    ap.add_argument("--max-grow", type=int, default=2)
    ap.add_argument("--router", choices=["conf", "proto", "energy", "ensemble"], default="proto")
    ap.add_argument("--fast", action="store_true", help="hızlı-NCM TABAN ÇİZGİSİ (büyüme yok): sabit özellik + prototip kafa (<1sn/görev)")
    ap.add_argument("--sleep", action="store_true", help="her görevden sonra uyku konsolidasyonu çalıştır")
    ap.add_argument("--stream", action="store_true",
                    help="v4 AKIŞ kipi: görev sınırları VERİLMEZ; şaşkınlık kapısı otomatik böler (--tasks yoksayılır)")
    ap.add_argument("--chunk", type=int, default=12000, help="akış parçası (varsayılan 2 sınıf ≈ 12000)")
    ap.add_argument("--surprise-k", type=float, default=2.5, help="şaşkınlık eşiği: mu + k*sigma")
    ap.add_argument("--surprise-k2", type=float, default=1.5,
                    help="değişim eşiği: |s - s_onceki| > k2*sigma (seviye yetmezse sıçrama yakalar)")
    ap.add_argument("--save", default=None, help=f"beyin dosyası (varsayılan kullanılmaz; ör. {DEFAULT_SAVE})")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    data_dir = args.data_dir or os.path.join(DEFAULT_DATA_ROOT, args.dataset)
    Xtr, ytr, Xte, yte = load_dataset(args.dataset, data_dir)
    new_tasks = parse_tasks(args.tasks)
    rng = np.random.default_rng(args.seed)
    print(f"Veri: {args.dataset}  eğitim={len(Xtr)}  test={len(Xte)}  (D diski: {data_dir})")

    if args.resume:
        net = GrowingNet.load(args.resume, args.seed)
        print(f"Kaldığı yerden devam: {len(net.heads)} görev ve {net.n_features} nöron yüklendi.")
    else:
        net = GrowingNet(Xtr.shape[1], args.block_size, args.seed)
    old_tasks = [h["classes"] for h in net.heads]
    all_tasks = old_tasks + new_tasks
    names = ["G" + "".join(map(str, c)) for c in all_tasks]

    def eval_grow(upto):
        row = []
        for j in range(len(all_tasks)):
            if j < upto:
                Xj, yj, _ = task_subset(Xte, yte, all_tasks[j])
                row.append(float((net.predict_task(j, Xj) == yj).mean()))
            else:
                row.append(None)
        return row

    R_task, R_class, growth = [], [], []
    reuse_hist = []
    if old_tasks:
        start = eval_grow(len(old_tasks))
        print("\nYüklenen beynin eski görevlerdeki başarısı:",
              ", ".join(f"{names[j]}={start[j]:.3f}" for j in range(len(old_tasks))))

    if args.stream:
        # v4 AKIŞ: sınırlar verilmez. Sınıflar sıralı akar (0..9), parça parça şaşkınlık
        # ölçülür; sıçrama (mu+k*sigma) = otomatik sınır. Parça maliyeti SABİT:
        # veri 10x artsa da her adımda yalnızca 1 parça işlenir (sıfırdan eğitim yok).
        print("\n=== AKIŞ KİPİ (sınırlar şaşkınlıkla bulunuyor) ===")
        all_tasks = [h["classes"] for h in net.heads]
        names = ["G" + "".join(map(str, c)) for c in all_tasks]
        order = sorted(set(ytr.tolist()))
        SX = np.concatenate([Xtr[ytr == c][rng.permutation((ytr == c).sum())] for c in order])
        SY = np.concatenate([ytr[ytr == c][rng.permutation((ytr == c).sum())] for c in order])
        chunks = [(SX[i:i + args.chunk], SY[i:i + args.chunk]) for i in range(0, len(SX), args.chunk)]
        mu = sig = s_prev = None
        acc_X, acc_y, seg_times = [], [], []

        def close_segment(force=False):
            X = np.concatenate(acc_X)
            y = np.concatenate(acc_y)
            classes = sorted(set(y.tolist()))
            if len(classes) < 2 and not force:
                print(f"  (tek-sınıf birikim {classes}: ayırt edici kafa kurulamaz, sınır ertelendi)")
                return False
            yl = np.searchsorted(np.array(classes), y)
            perm = rng.permutation(len(X))
            nval = len(X) // 10
            vi, tr = perm[:nval], perm[nval:]
            print(f"  SEGMENT kapanıyor (sınıflar {classes}, n={len(X)}):")
            t_seg = time.time()
            if args.fast:
                g = net.fast_add_task(classes, X[tr], yl[tr], print)
            else:
                g, ra = net.learn_task(classes, X[tr], yl[tr], X[vi], yl[vi], args, print,
                                       task_idx_0based=len(net.heads), reuse_acc_history=reuse_hist)
                if g == 0 and ra is not None:
                    reuse_hist.append(ra)
            growth.append(g)
            seg_times.append(time.time() - t_seg)
            if args.sleep:
                net.sleep_consolidate(print)
            all_tasks.append(classes)
            names.append("G" + "".join(map(str, classes)))
            R_task.append(eval_grow(len(all_tasks)))
            seen = sorted(set(c for cl in all_tasks for c in cl))
            Xs, ys, _ = task_subset(Xte, yte, seen)
            R_class.append(float((net.predict_any(Xs, method=args.router) == ys).mean()))
            # REFERANS ÇAPA: yenilik, "son akan veri" ile değil, PEKİŞMİŞ BİLGİ ile kıyaslanır.
            # Bu segmentin eğitim parçasındaki şaşkınlık = tanıdık dağılımın imzası.
            nonlocal mu, sig
            ref_s = net.surprise(X[tr][:2000])
            mu = float(ref_s.mean())
            sig = max(float(ref_s.std()), 0.05 * mu)
            print(f"    referans çapa: mu={mu:.1f} sig={sig:.1f} (pekleşmiş bilgi)")
            return True

        for ci, (cx, cy) in enumerate(chunks):
            if not net.heads:
                # ÖNYÜKLEME: şaşkınlık için önce temsil gerekir; ilk parça doğrudan öğrenilir.
                acc_X, acc_y = [cx], [cy]
                print(f"  parça {ci}: önyükleme (temsil yok) -> ilk SEGMENT")
                close_segment(force=True)
                acc_X, acc_y = [], []
                # NOT: çapayı SİLME; close_segment referansı kurdu, akış ondan devam eder.
                continue
            s = float(net.surprise(cx).mean())
            # Çift kural: SEVİYE (bilinmeyene uzaklık) + DEĞİŞİM (ani sıçrama/düşüş).
            # Seviye tek başına yetmez: bilgi büyüdükçe referans genişler, yakın-yeni
            # sınıflar içine gizlenir. Değişim kuralı dağılım kaymasını yakalar.
            lvl = mu is not None and np.isfinite(s) and s > mu + args.surprise_k * sig
            chg = (mu is not None and s_prev is not None and np.isfinite(s)
                   and abs(s - s_prev) > args.surprise_k2 * sig)
            if lvl or chg:
                tag = "SINIR"
            else:
                tag = "tanidik"
            print(f"  parça {ci}: saskinlik={s:.1f} (mu={mu or 0:.1f}) -> {tag}")
            if tag == "SINIR" and acc_X:
                if close_segment():
                    acc_X, acc_y = [], []
            acc_X.append(cx)
            acc_y.append(cy)
            # NOT: mu/sig yalnızca close_segment'te referans çapayla güncellenir;
            # akan verinin EMA'sı yeniyi eskiye benzetir (çapa kayması), o yüzden yok.
            if np.isfinite(s):
                s_prev = s
        if acc_X:
            close_segment(force=True)
        print(f"  parça-maliyetleri: {[f'{t:.1f}sn' for t in seg_times]} (sabit; veriyle büyümez)")
    else:
        print("\n=== BÜYÜYEN AĞ ===")
        for t, classes in enumerate(new_tasks):
            ti = len(old_tasks) + t
            X, _, yl = task_subset(Xtr, ytr, classes)
            perm = rng.permutation(len(X))
            nval = len(X) // 10
            vi, tr = perm[:nval], perm[nval:]
            print(f"  Görev {names[ti]} (sınıflar {classes}):")
            if args.fast:
                t_fast0 = time.time()
                g = net.fast_add_task(classes, X[tr], yl[tr], print)
                print(f"    süre: {time.time() - t_fast0:.2f} sn (hedef <5sn, eğitim yok)")
                growth.append(g)
            else:
                g, reuse_acc = net.learn_task(classes, X[tr], yl[tr], X[vi], yl[vi], args, print,
                                              task_idx_0based=ti, reuse_acc_history=reuse_hist)
                # Çift koruma: yalnızca büyüMEyen görevlerin yeniden-kullanma skoru girer.
                if g == 0 and reuse_acc is not None:
                    reuse_hist.append(reuse_acc)
                growth.append(g)
            if args.sleep:
                net.sleep_consolidate(print)
            R_task.append(eval_grow(ti + 1))
            seen = sorted(set(c for cl in all_tasks[:ti + 1] for c in cl))
            Xs, ys, _ = task_subset(Xte, yte, seen)
            R_class.append(float((net.predict_any(Xs, method=args.router) == ys).mean()))

    if not args.no_baseline and not old_tasks and not args.stream:
        print("\n=== KLASİK AĞ (karşılaştırma) ===")
        # VARSAYIM: etiketler 0..K-1 bitişik (MNIST/Fashion için doğru); predict_task [:, cl] ile kolon seçer.
        n_cls = int(max(ytr)) + 1
        base = PlainMLP(Xtr.shape[1], 256, n_cls, args.seed)
        B_task, B_class = [], []
        for t, classes in enumerate(new_tasks):
            X, yg, _ = task_subset(Xtr, ytr, classes)
            base.learn_task(X, yg, args)
            row = []
            for j in range(len(all_tasks)):
                if j <= t:
                    Xj, yj, _ = task_subset(Xte, yte, all_tasks[j])
                    row.append(float((base.predict_task(all_tasks[j], Xj) == yj).mean()))
                else:
                    row.append(None)
            B_task.append(row)
            seen = sorted(set(c for cl in all_tasks[:t + 1] for c in cl))
            Xs, ys, _ = task_subset(Xte, yte, seen)
            B_class.append(float((base.predict_any(seen, Xs) == ys).mean()))
            print(f"  Görev {names[t]} öğrenildi.")

    print("\n" + "=" * 70 + "\nSONUÇLAR\n" + "=" * 70)
    print_growth_map(net, names)
    offset = len(old_tasks)
    # Hız notu: eval_grow her görev-altkümesinde features'ı yeniden hesaplar (altkümeler
    # farklı X olduğu için paylaşılamaz). Ama R_task ile R_full eski-kodsuz durumda
    # birebir aynıdır -> tekrar hesaplamayı atla.
    R_full = R_task if not old_tasks else [eval_grow(offset + i + 1) for i in range(len(new_tasks))]
    if old_tasks:
        print_matrix("Büyüyen ağ - görev biliniyor", R_full, names[offset:])
        print("  (ilk sütunlar devam modunda yüklenen eski görevlerdir)")
        now = R_full[-1]
        diffs = [now[j] - start[j] for j in range(len(old_tasks))]
        print(f"\n  Eski görevlerdeki değişim (unutma): {['%+.4f' % d for d in diffs]}")
    else:
        print_matrix("Büyüyen ağ - görev biliniyor", R_task, names)
        acc, bwt = summarize(R_task)
        print(f"\n  Büyüyen ağ : ort. başarı={acc:.3f}  unutma(BWT)={bwt:+.4f}  "
              f"parametre={net.n_params():,}  nöron={net.n_features}  büyümeler={growth}  router={args.router}")
        if not args.no_baseline:
            print_matrix("Klasik ağ - görev biliniyor", B_task, names)
            bacc, bbwt = summarize(B_task)
            print(f"\n  Klasik ağ  : ort. başarı={bacc:.3f}  unutma(BWT)={bbwt:+.4f}  "
                  f"parametre={base.n_params():,}")
        print(f"\n  Görev BİLİNMEDEN [router={args.router}] (tüm görülen sınıflar):")
        print("  " + "  ".join(f"{names[i]}:{R_class[i]:.3f}" for i in range(len(R_class))) + "   <- büyüyen ağ")
        if not args.no_baseline:
            print("  " + "  ".join(f"{names[i]}:{B_class[i]:.3f}" for i in range(len(B_class))) + "   <- klasik ağ")

    if args.save:
        out = net.save(args.save)
        print(f"\nBeyin kaydedildi: {out}  (devam için: --resume {out})")
    print(f"Süre: {time.time() - t0:.1f} sn")


if __name__ == "__main__":
    main()
