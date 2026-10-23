#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_mirror.py — AYNA / ÖZ-DIŞ AYRIMI (benlik planı, aşama 5). Duyu-motor ŞARTLILIĞI (contingency) ile 'ben mi, başkası mı, kendi kaydımın tekrarı mı'.

Ajan gördüğü bir nesnenin (açısal hızının) kendi motor komutlarına bağlı olup olmadığına bakar. Üç durum, hepsi GERÇEK evren izlerinden üretilir:
  ben        ayna: nesne, ajanın kendi gerçek dönüş hızını 1 adım gecikmeyle (+gürültü) yansıtır
  başka      başka bir ajanın (bağımsız rastgele politika, ayrı evren) gerçek dönüş hızı
  tekrar     ajanın KENDİ eski dönüş hızı kaydı, L≫1 adım gecikmeyle oynatılıyor (aynı istatistik, ama şartlılık gecikmeli)
Algılayıcı (parametresiz): motor kopyası c_t (sola +1, sağa −1) ile nesne hızı y_t arasında lag 0..100 çapraz ilişki; tepe gücü sıfır-dağılımından (dairesel kaydırma) anlamlı mı,
tepe gecikmesi ≤3 mü? → ben | anlamlı ama gecikme büyük → tekrar (gecikme raporlanır) | anlamsız → başka.
Karşı sınama: (a) 'papağan': yalnızca nesne izinden (efferans kopyası YOK) sınıflandırıcı; (b) efferans kopyası karıştırılmış ablasyon; (c) hiç görülmemiş gecikmeler/gürültü düzeyleri.

Kullanım:  python nora_mirror.py --trials 300
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nora_world import Universe
from nora_mind import Mind, wrap

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def record(seed, steps, policy):
    """Gerçek evren izi: (komut c_t, gerçek dönüş hızı dθ_t)."""
    w = Universe(seed=seed)
    mind = Mind(seed=seed, mode="random" if policy == "random" else "full")
    obs, died = w.observe(), False
    c, dth = np.zeros(steps), np.zeros(steps)
    th0 = w.th
    for t in range(steps):
        a = mind.act(obs, died)
        obs, info = w.step(a)
        died = info["died"] is not None
        c[t] = 1.0 if a == 2 else (-1.0 if a == 3 else 0.0)
        dth[t] = wrap(w.th - th0)
        th0 = w.th
    return c, dth


def make_trial(kind, c_self, d_self, d_other, t0, W, lag_replay, noise, rng):
    """Bir pencere: (efferans kopyası c, nesne hızı y)."""
    c = c_self[t0:t0 + W]
    if kind == "ben":
        y = d_self[t0 - 1:t0 + W - 1]                            # 1 adım gecikmeli yansıma (komuttan sonra)
    elif kind == "başka":
        j = int(rng.integers(0, len(d_other) - W))
        y = d_other[j:j + W]
    else:                                                        # tekrar: kendi eski kaydım L adım gecikmeyle
        y = d_self[t0 - lag_replay:t0 - lag_replay + W]
    y = y + noise * np.std(d_self) * rng.standard_normal(W)
    return c, y


def xcorr(c, y, lmax=100):
    """r(l) = corr(c[t-l], y[t]) , l=0..lmax."""
    r = np.zeros(lmax + 1)
    for l in range(lmax + 1):
        a, b = c[:len(c) - l], y[l:]
        if a.std() > 1e-9 and b.std() > 1e-9:
            r[l] = np.corrcoef(a, b)[0, 1]
    return r


def detect(c, y, null_thr):
    r = xcorr(c, y)
    a = np.abs(r)
    l = int(np.flatnonzero(a >= 0.9 * a.max())[0])               # en erken yakın-tepe (komut öbeklerinin özilişkisi tepeyi yaymasın)
    if a[l] < null_thr:
        return "başka", l
    return ("ben" if l <= 3 else "tekrar"), l


def null_threshold(c, y, rng, n=60):
    peaks = []
    for _ in range(n):
        s = int(rng.integers(110, len(y) - 110))                    # kaydırma, aranan gecikme aralığının (≤100) dışında kalmalı
        peaks.append(np.abs(xcorr(c, np.roll(y, s))).max())
    return float(np.quantile(peaks, 0.99))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=300)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    steps = 6000
    c_self, d_self = record(args.seed, steps, "full")
    _, d_other = record(args.seed + 100, steps, "random")
    kinds = ["ben", "başka", "tekrar"]
    W = args.window
    print("AYNA / ÖZ-DIŞ AYRIMI | pencere %d adım | sınıf başına %d deneme | gerçek evren izleri" % (W, args.trials))
    for lag in (40, 70):
        for noise in (0.0, 0.5, 1.0, 2.0):
            conf = {k: {j: 0 for j in kinds} for k in kinds}
            conf_abl = {k: {j: 0 for j in kinds} for k in kinds}
            Xp, Yp = [], []
            for k in kinds:
                for _ in range(args.trials):
                    t0 = int(rng.integers(lag + 5, steps - W - 5))
                    c, y = make_trial(k, c_self, d_self, d_other, t0, W, lag, noise, rng)
                    thr = null_threshold(c, y, rng, n=30)
                    conf[k][detect(c, y, thr)[0]] += 1
                    ca = rng.permutation(c)                       # ablasyon: efferans kopyası karıştırıldı
                    conf_abl[k][detect(ca, y, thr)[0]] += 1
                    Xp.append(np.r_[np.abs(np.fft.rfft(y - y.mean()))[:20], y.std(), np.abs(y).mean()]); Yp.append(kinds.index(k))
            acc = np.mean([conf[k][k] for k in kinds]) / args.trials
            acc_abl = np.mean([conf_abl[k][k] for k in kinds]) / args.trials
            # papağan: yalnızca nesne izinden 5-NN (eğitim/test bölünmüş)
            Xp, Yp = np.array(Xp), np.array(Yp)
            idx = rng.permutation(len(Xp)); cut = int(0.6 * len(idx))
            mu, sd = Xp[idx[:cut]].mean(0), Xp[idx[:cut]].std(0) + 1e-9
            Z = (Xp - mu) / sd
            correct = 0
            for i in idx[cut:]:
                d = ((Z[idx[:cut]] - Z[i]) ** 2).sum(1)
                nn = idx[:cut][np.argsort(d)[:5]]
                correct += np.bincount(Yp[nn], minlength=3).argmax() == Yp[i]
            acc_p = correct / (len(idx) - cut)
            print("  gecikme(tekrar)=%2d gürültü=%.1f | efferanslı doğruluk %.2f (ben %.2f, başka %.2f, tekrar %.2f) | efferans karıştırılmış %.2f | papağan (yalnız nesne izi) %.2f | şans 0.33" %
                  (lag, noise, acc, conf["ben"]["ben"] / args.trials, conf["başka"]["başka"] / args.trials, conf["tekrar"]["tekrar"] / args.trials, acc_abl, acc_p), flush=True)


if __name__ == "__main__":
    main()
