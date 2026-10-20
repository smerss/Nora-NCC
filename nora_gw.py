#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_gw.py — ÇALIŞMA ALANI GEREKLİ Mİ? Küresel çalışma alanı kuramının kontrollü sınama tezgâhı.

Kuram (Baars/Dehaene): beyin, her biri yalnızca KENDİ bilgisine erişen, paralel ve modüler uzmanlardan oluşur. Hiçbir uzman tek başına
yeterli bilgiye sahip değildir; bir yarışmanın kazananı içeriğini TÜM uzmanlara yayınlar, böylece her uzman kendi özel bilgisini yayınlanan
içerikle BİRLEŞTİREREK değerlendirme yapabilir. Çalışma alanı tam da bu birleştirmeyi mümkün kılar.

Tasarım (tek sinir bütçesi, tek öğrenme kuralı; yalnızca bilgi akışı değişir):
  • 4 uzman: yiyecek (yalnız yiyecek ışınları + enerji açığı), su (yalnız su ışınları + susuzluk), tehlike (yalnız tehlike ışınları + bütünlük + dokunma),
    duvar/hareket (duvar ışınları + pusula + hız). Her uzmanın KENDİ seyrek Kenyon-hücre kodu ve amaç-başına Q kafaları vardır.
  • Amaç k için toplam değer  Q_k(a) = Σ_j Q_j[k](a | kod_j(özel_j, yayın_k)). Yayın_k = amaç k'nın kendi algısı (yiyecek ışınları vb.) + amaç kimliği.
  • Öğrenme: TD (dopamin), her amaç k için; ödül = o amacın ödülü + herkes için yara cezası.
Koşullar (yalnızca bilgi akışı farklı):
  gw        modüler + yayın            (özel bilgi + yayınlanan içerik birleşir)
  yalıtık   modüler, yayın YOK         (her uzman yalnız kendi bilgisini görür; kazanan-hepsini-alır amaç seçimi)
  karışım   modüler, yayın YOK, kazanan YOK (acil durum ağırlıklı toplam)
  monolit   tek uzman, tüm algıyı tek kodda görür (üst sınır: birleştirme zaten mümkün)
Görev: yiyeceğin tehlike bölgesinin KENDİ kenarına konduğu evren (food_haz). Doğru yaklaşma tarafı = f(yiyecek yönü, tehlike yönü): bir KONJONKSİYON.
Toplamsal ayrık uzmanlar bunu temsil edemez; yayınla alıcı uzman (tehlike) 'yiyecek şurada' bilgisini kendi tehlike bilgisiyle birleştirir.

Kullanım:  python nora_gw.py --seeds 4 --steps 60000
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nora_world import Universe
from nora_neuron import LobedMushroomBody

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PRIOR = np.array([0.0, 0.8, 0.0, 0.0, -1.0], np.float32)
NA = Universe.N_ACT
I_E, I_W, I_I = Universe.I_INTERO, Universe.I_INTERO + 1, Universe.I_INTERO + 2


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def privates(obs):
    """Uzmanların ÖZEL girdileri (başka uzmanın bilgisini içermez)."""
    r = obs[:45].reshape(Universe.R, 5)
    d = 1.0 - obs[I_E:I_I + 1]
    return [np.r_[r[:, 1], d[0]],                       # 0 yiyecek uzmanı
            np.r_[r[:, 2], d[1]],                       # 1 su uzmanı
            np.r_[r[:, 3], d[2], obs[Universe.I_TOUCH]],  # 2 tehlike uzmanı
            np.r_[r[:, 0], obs[Universe.I_COMPASS:Universe.I_COMPASS + 2], obs[Universe.I_SPEED:Universe.I_SPEED + 2]]]  # 3 duvar/hareket


class Agent:
    def __init__(self, cond="gw", seed=0, n_per=700, claws=4, gamma=0.95, lr=0.25):
        self.cond, self.rng, self.gamma, self.lr = cond, np.random.default_rng(seed), gamma, lr
        self.G = 3                                       # öğrenen amaçlar: yiyecek, su, tehlike
        self.bc_dim = 3 + Universe.R                     # amaç kimliği (one-hot) + amacın kendi algısı (9 ışın)
        if cond == "monolit":
            self.n_in = [9 * 4 + 3 + 1 + 2 + 2]
        else:
            base = [Universe.R + 1, Universe.R + 1, Universe.R + 2, Universe.R + 4]
            self.n_in = [b + (self.bc_dim if cond == "gw" else 0) for b in base]
        self.J = len(self.n_in)
        self.mb = [LobedMushroomBody([np.arange(n)], n_per=n_per, claws=claws, cap_frac=0.15, seed=seed * 31 + j, thr_range=(0.05, 0.9))
                   for j, n in enumerate(self.n_in)]
        self.W = [np.zeros((self.G, NA, m.n_kc), np.float32) for m in self.mb]
        self.prev = None
        self.goal, self.age = -1, 0
        self.acc = np.zeros(4)

    # ---- kodlar ve değer
    def codes(self, obs):
        """cod[k][j] = j. uzmanın amaç k için seyrek kodu (yayına bağlıysa k'ya bağlı)."""
        P = privates(obs)
        if self.cond == "monolit":
            r = obs[:45].reshape(Universe.R, 5)
            x = np.r_[r[:, 1], r[:, 2], r[:, 3], r[:, 0], 1.0 - obs[I_E:I_I + 1], obs[Universe.I_TOUCH],
                      obs[Universe.I_COMPASS:Universe.I_COMPASS + 2], obs[Universe.I_SPEED:Universe.I_SPEED + 2]]
            c = [self.mb[0].kc(x.astype(np.float32))]
            return [c for _ in range(self.G)]
        if self.cond == "gw":
            out = []
            for k in range(self.G):
                bc = np.r_[np.eye(3)[k], P[k][:Universe.R]]          # yayın: amaç kimliği + amacın kendi algısı
                out.append([self.mb[j].kc(np.r_[P[j], bc].astype(np.float32)) for j in range(self.J)])
            return out
        c = [self.mb[j].kc(P[j].astype(np.float32)) for j in range(self.J)]
        return [c for _ in range(self.G)]

    def qvals(self, cod):
        """(G, NA): amaç başına toplam değer."""
        Q = np.zeros((self.G, NA), np.float32)
        for k in range(self.G):
            for j in range(self.J):
                Q[k] += self.W[j][k][:, cod[k][j]].sum(1)
        return Q

    @staticmethod
    def urgency(obs):
        r = obs[:45].reshape(Universe.R, 5)
        d = 1.0 - obs[I_E:I_I + 1]
        return np.array([1.5 * d[0] ** 2, 1.5 * d[1] ** 2, max(1.5 * d[2] ** 2, 1.2 * float(r[:, 3].max()) ** 2)])

    def act(self, obs, died=False):
        self.age += 1
        cod = self.codes(obs)
        Q = self.qvals(cod)
        # ---- öğrenme
        if self.prev is not None:
            pcod, pa, pobs = self.prev
            de, dw, di = obs[I_E] - pobs[I_E], obs[I_W] - pobs[I_W], obs[I_I] - pobs[I_I]
            hurt = 5.0 * min(0.0, di)
            r = np.array([3.0 * max(-0.5, de), 3.0 * max(0.0, dw), 0.0]) + hurt
            if died:
                r = r - 5.0
            nxt = Q.max(1) if not died else np.zeros(self.G, np.float32)
            for k in range(self.G):
                q_prev = sum(self.W[j][k][pa, pcod[k][j]].sum() for j in range(self.J))
                delta = float(np.clip(r[k] + self.gamma * nxt[k] - q_prev, -3.0, 3.0))
                n_act = sum(len(pcod[k][j]) for j in range(self.J)) + 1
                for j in range(self.J):
                    self.W[j][k][pa, pcod[k][j]] += self.lr * delta / n_act
        # ---- uzmanlar arası yarışma (kazanan-hepsini-alır; karışımda yok)
        urg = self.urgency(obs)
        if self.cond == "karışım":
            w = urg + 0.1
            logits = (w[:, None] * 8.0 * Q).sum(0) / w.sum() + PRIOR
            self.goal = -1
        else:
            sal = np.r_[urg, 0.15] + 0.02 * self.rng.standard_normal(4)
            if self.goal >= 0:
                sal[self.goal] += 0.15                      # kısa histerezis: titremeyi önler
            win = int(sal.argmax())
            self.goal = win if win < 3 and sal[win] > 0.25 else -1
            logits = 8.0 * Q[self.goal] + PRIOR if self.goal >= 0 else PRIOR.copy()
        act = int(self.rng.choice(NA, p=softmax(logits)))
        self.prev = (cod, act, obs.copy())
        return act


def run(job):
    cond, seed, steps, fh, size, nfood = job
    kw = dict(food_haz=fh)
    if nfood:
        kw.update(size=float(size), n_food=nfood)
    w = Universe(seed=seed, **kw)
    ag = Agent(cond, seed=seed)
    obs, died = w.observe(), False
    nwin = 6
    W = int(steps // nwin)
    rows, d, hurt, ate, conf = [], 0, 0, 0.0, 0.0
    for t in range(1, steps + 1):
        obs, info = w.step(ag.act(obs, died))
        died = info["died"] is not None
        d += died
        hurt += info["hurt"] > 0
        ate += info["ate"]
        conf += min(obs[I_E], obs[I_W], obs[I_I])
        if t % W == 0:
            rows.append((1000.0 * d / W, 1000.0 * hurt / W, 1000.0 * ate / W, conf / W))
            d, hurt, ate, conf = 0, 0, 0.0, 0.0
    return cond, seed, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--conds", default="monolit,gw,yalıtık,karışım")
    ap.add_argument("--food-haz", type=float, default=0.85)
    ap.add_argument("--size", type=float, default=24.0)
    ap.add_argument("--nfood", type=int, default=0)
    args = ap.parse_args()
    from multiprocessing import Pool
    conds = args.conds.split(",")
    jobs = [(c, s, args.steps, args.food_haz, args.size, args.nfood) for c in conds for s in range(args.seed0, args.seed0 + args.seeds)]
    t0 = time.time()
    with Pool(min(len(jobs), os.cpu_count() or 1)) as pool:
        out = pool.map(run, jobs)
    print("ÇALIŞMA ALANI SINAMASI | yiyecek tehlike kenarında olasılığı %.2f | %d adım | %d tohum | son 3 pencere ortalaması (%.0fs)" %
          (args.food_haz, args.steps, args.seeds, time.time() - t0))
    for c in conds:
        R = [r[2] for r in out if r[0] == c]
        last = np.array([np.mean(np.array(x[-3:]), 0) for x in R])         # (tohum, 4)
        print("  %-9s ölüm/1k %.2f ± %.2f | yaralanma/1k %.1f | yeme/1k %.2f | konfor %.3f" %
              (c, last[:, 0].mean(), last[:, 0].std(), last[:, 1].mean(), last[:, 2].mean(), last[:, 3].mean()), flush=True)


if __name__ == "__main__":
    main()
