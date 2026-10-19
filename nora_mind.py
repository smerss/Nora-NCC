#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_mind.py — NoRa'nın yaşayan zihni: nöronsal temel + global çalışma alanı + değer + öz-model.

Bilinç kuramlarının ORTAK olarak aradığı dört bileşen tek döngüde:

  1) KAPALI DÖNGÜ   algı -> karar -> eylem -> evren -> algı (nora_world.py). Eylem kopyası (efference)
                    iç duruma geri beslenir; ajan kendi eyleminin sonucunu öğrenir.
  2) DEĞER/VALANS   Homeostatik hata H = sum(d_k^2) (d_k = 1 - ihtiyaç_k). Duygu-tonu v_t = -(H_t - H_{t-1}):
                    ihtiyaç giderilince iyi, yara/açlık artınca kötü. Ruh hali = v'nin yavaş ortalaması;
                    kötü ruh hali merakı/keşfi artırır (aloströz düzenleme). Ödül dışarıdan verilmez,
                    VÜCUTTAN doğar.
  3) GLOBAL ÇALIŞMA ALANI (Baars/Dehaene)  Uzmanlar (açlık, susuzluk, tehlike, merak) AYNI duyusal kodu
                    okur, her biri kendi eylem önerisini + çıkar miktarını (salience) üretir. Yanal
                    baskılamalı sızıntılı toplayıcılarda yarışırlar; kazanan eşik+marj aşarsa "ATEŞLEME":
                    amacı tüm sisteme yayınlanır (iç duruma, öz-modele, öğrenmeye, rapora). Sığa=1 amaç.
  4) SÜREKLİLİK/ÖZ-MODEL  Hiç sıfırlanmayan sızıntılı tekrarlayan durum h (ölüm ve doğumlar boyunca sürer).
                    Öz-model: (h, eylem, beden) -> bir sonraki beden/hareket durumunu öngörür; öngörü
                    hatasından (a) şaşkınlık, (b) ÖZ-DIŞ AYRIMI/ajanlık (rüzgâr gibi dış kuvvet hatası
                    öz-eylem hatasından büyük mü), (c) üstbilişsel güven (kendi hatasını önceden tahmin).

Nöronsal temel: MushroomBody (nora_neuron.py) — seyrek Kenyon kodu; uzman Q-kafaları KC->MBON
sinapsları; öğrenme sinekteki gibi dopamin(TD hatası)-kapılı, yalnızca aktif KC'ler değişir.

Dürüstlük: buradaki "ateşleme, değer, ajanlık, güven" İŞLEVSEL göstergelerdir; öznel deneyim olduğunu
kanıtlamazlar. Amaç bu göstergeleri ölçülebilir tutup mimariyi bunlar üzerinden iyileştirmek.

Kullanım:
  DPY nora_mind.py --stage run --steps 60000 --mode full [--save beyin_mind.npz] [--resume beyin_mind.npz]
  DPY nora_mind.py --stage compare --steps 60000          # random | flat (çalışma alanı yok) | full
"""
import argparse, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nora_neuron import LobedMushroomBody
from nora_world import Universe, ACTIONS

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SPEC = ("aç", "susuz", "tehlike", "merak")
I_E, I_W, I_I = Universe.I_INTERO, Universe.I_INTERO + 1, Universe.I_INTERO + 2
RAY_FOOD, RAY_WATER, RAY_HAZ = 1, 2, 3
BODY_IDX = [45, 46, 47, 48, 49, 50]                 # öz-modelin öngördüğü: e,w,i,hız_f,hız_y,dokunma
PRIOR = np.array([0.0, 0.8, 0.0, 0.0, -1.0], np.float32)   # doğuştan: ileri yönelim, geri kaçınma


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class Mind:
    def __init__(self, mode="full", seed=0, n_kc=3000, n_res=96, gamma=0.95, lr=0.25, self_model=True, shape=2.0, ctx=True):
        self.mode, self.rng, self.gamma, self.lr = mode, np.random.default_rng(seed), gamma, lr
        self.use_sm, self.shape, self.ctx = self_model, shape, ctx
        self.n_ctx = 48
        n_act = Universe.N_ACT
        # --- nöronsal temel: KC kodu yalnızca duyusal girdiden (iç duyu HARİÇ) -> Q ihtiyaçtan bağımsız
        groups = [np.arange(c, 45, 5) for c in range(5)] + [np.array([48, 49, 50])]   # kanal başına ışınlar | proprio
        if ctx:                                                # bağlam lobu: sürekli iç durum h davranışa girer
            groups.append(np.arange(Universe.OBS_DIM, Universe.OBS_DIM + self.n_ctx))
        self.mb = LobedMushroomBody(groups, n_per=n_kc // len(groups), seed=seed)
        n_kc = self.mb.n_kc
        self.Wq = np.zeros((len(SPEC), n_act, n_kc), np.float32)       # uzman Q kafaları (KC->MBON)
        # --- süreklilik: tekrarlayan durum (HİÇ sıfırlanmaz)
        self.n_res, self.alpha = n_res, 0.3
        self.n_B = len(SPEC) + 1 + 1 + n_act + 1                         # yayın vektörü boyutu
        rr = self.rng
        Wr = rr.standard_normal((n_res, n_res)) * (rr.random((n_res, n_res)) < 0.1)
        Wr *= 0.9 / max(1e-6, np.abs(np.linalg.eigvals(Wr)).max())
        self.Wr = Wr.astype(np.float32)
        self.Win = (rr.standard_normal((n_res, Universe.OBS_DIM + self.n_B)) * 0.4 *
                    (rr.random((n_res, Universe.OBS_DIM + self.n_B)) < 0.3)).astype(np.float32)
        self.h = np.zeros(n_res, np.float32)
        # --- öz-model (NLMS) ve güven başlığı
        self.nz = n_res + n_act + len(BODY_IDX) + 2 * n_act + 1 + 1
        self.Wsm = np.zeros((len(BODY_IDX), self.nz), np.float32)
        self.Wc = np.zeros(self.nz, np.float32)
        self.scale = np.full(len(BODY_IDX), 0.05, np.float32)           # sinyal değişkenliği (normalizasyon)
        # --- çalışma alanı ve duygu
        self.acc = np.zeros(len(SPEC), np.float32)
        self.B = np.zeros(self.n_B, np.float32)
        self.mood, self.valence, self.goal, self.ignited = 0.0, 0.0, -1, False
        self.prev = None        # (kc_idx, action, obs, H)
        self.pred, self.z, self.conf = None, None, 0.0
        self.err_n = 0.0
        self.err_vel = 0.0
        self.age = 0
        self.last_obs = None

    # ------------------------------------------------------------ yardımcılar
    def _H(self, obs):
        return float(((1 - obs[I_E:I_I + 1]) ** 2).sum())

    @staticmethod
    def _phi(obs):
        """Duyusal potansiyel Φ_k (koku gradyanı gibi): kanalın en yakın ışın yakınlığı. Potansiyel-tabanlı
        şekillendirme r' = γΦ(s') - Φ(s) en iyi politikayı DEĞİŞTİRMEZ (Ng 1999), yalnızca öğrenmeyi yoğunlaştırır."""
        rays = obs[:45].reshape(Universe.R, 5)
        return np.array([rays[:, RAY_FOOD].max(), rays[:, RAY_WATER].max(), -rays[:, RAY_HAZ].max(), 0.0], np.float32)

    def _q(self, idx):
        return self.Wq[:, :, idx].sum(2)                 # (uzman, eylem)

    def _urgency(self, obs):
        rays = obs[:45].reshape(Universe.R, 5)
        d = 1 - obs[I_E:I_I + 1]
        haz = float(rays[:, RAY_HAZ].max())
        return np.array([1.5 * d[0] ** 2, 1.5 * d[1] ** 2, max(1.5 * d[2] ** 2, 1.2 * haz ** 2), 0.0], np.float32)

    # ------------------------------------------------------------ ana döngü
    def act(self, obs, died=False):
        self.age += 1
        rr = self.rng
        # ---------- algı: seyrek kod + yenilik
        idx = self.mb.kc(np.concatenate([np.abs(obs), np.clip(3.0 * np.abs(self.h[:self.n_ctx]), 0, 1)]))
        nov = self.mb.novelty(idx)
        H = self._H(obs)
        # ---------- valans (homeostatik hata türevi) ve ruh hali
        if self.prev is not None:
            raw = -(H - self.prev[3]) * 5.0
            hurt = max(0.0, self.prev[2][I_I] - obs[I_I])
            self.valence = float(np.clip(raw - 30.0 * hurt, -2.0, 2.0)) if not died else -2.0
        self.mood += 0.01 * (self.valence - self.mood)
        # ---------- öğrenme (dopamin = TD hatası, yalnızca aktif KC'ler)
        if self.prev is not None:
            pidx, pa, pobs, _ = self.prev
            r = np.array([3.0 * max(0.0, obs[I_E] - pobs[I_E]),
                          3.0 * max(0.0, obs[I_W] - pobs[I_W]),
                          5.0 * (obs[I_I] - pobs[I_I]),
                          1.0 * nov], np.float32)
            r += self.shape * (self.gamma * self._phi(obs) - self._phi(pobs)) * np.array([1, 1, 1, 0], np.float32)
            qn = self._q(idx).max(1) if not died else np.zeros(len(SPEC), np.float32)
            delta = np.clip(r + self.gamma * qn - self._q(pidx)[:, pa], -3.0, 3.0)
            self.Wq[:, pa, pidx] += (self.lr * delta / max(1, len(pidx)))[:, None]
        # ---------- öz-model: önceki öngörüyü gerçekle karşılaştır
        y = obs[BODY_IDX]
        if self.pred is not None and self.use_sm:
            err = (y - self.last_obs[BODY_IDX]) - self.pred          # öz-model DEĞİŞİMİ öngörür
            en = np.abs(err) / (self.scale + 1e-3)
            self.err_n = float(en[3:5].mean())          # hareket boyutları: öngörülebilir olan öz-etki
            self.err_vel = float(en[3:5].mean())
            zz = float(self.z @ self.z) + 1e-3
            self.Wsm += 0.5 * np.outer(err, self.z) / zz
            self.Wc += 0.2 * (self.err_n - float(self.Wc @ self.z)) * self.z / zz
            self.scale += 0.01 * (np.abs(y - self.last_obs[BODY_IDX]) - self.scale)
        # ---------- süreklilik: tekrarlayan durum (algı + önceki yayın)
        u = np.concatenate([obs, self.B])
        self.h = ((1 - self.alpha) * self.h + self.alpha * np.tanh(self.Win @ u + self.Wr @ self.h)).astype(np.float32)
        # ---------- uzmanlar: öneriler + çıkar (salience)
        Q = self._q(idx)                                              # (4, 5)
        urg = self._urgency(obs)
        opp = np.tanh(np.maximum(Q.max(1), 0.0))
        sal = urg * (0.6 + 0.8 * opp)
        sal[3] = 0.15 + 0.4 * nov
        sal = sal + 0.02 * rr.standard_normal(len(SPEC))
        temp = 1.0 + 0.8 * float(np.clip(-self.mood, 0, 1))
        if self.mode == "random":
            logits, self.goal, self.ignited = PRIOR.copy(), -1, False
        elif self.mode == "flat":                                      # çalışma alanı YOK: ağırlıklı karışım
            w = urg + 0.1
            logits = 8.0 * (w[:, None] * Q).sum(0) / w.sum() + PRIOR
            self.goal, self.ignited = -1, False
        else:
            # global çalışma alanı: yanal baskılamalı sızıntılı toplayıcılar
            a = self.acc
            for _ in range(3):
                a = np.clip(0.75 * a + 0.5 * sal - 0.35 * (a.sum() - a), 0.0, 1.5)
            self.acc = a
            order = np.argsort(-a)
            win, run = int(order[0]), int(order[1])
            self.ignited = bool(a[win] > 0.25 and a[win] - a[run] > 0.08)
            self.goal = win if self.ignited else -1
            logits = (8.0 * Q[win] + PRIOR) if self.ignited else PRIOR.copy()
        p = softmax(logits / temp)
        act = int(rr.choice(Universe.N_ACT, p=p))
        # ---------- yayın (tüm modüllere): amaç, valans, ruh hali, eylem kopyası, değer
        B = np.zeros(self.n_B, np.float32)
        if self.goal >= 0:
            B[self.goal] = 1.0
        B[4], B[5] = np.clip(self.valence, -2, 2) / 2.0, np.clip(self.mood, -1, 1)
        B[6 + act] = 1.0
        B[-1] = float(np.tanh(Q[self.goal].max())) if self.goal >= 0 else 0.0
        self.B = B
        # ---------- öz-model öngörüsü (eyleme koşullu)
        a1 = np.zeros(Universe.N_ACT, np.float32)
        a1[act] = 1.0
        self.z = np.concatenate([0.25 * self.h, 3.0 * a1, 3.0 * obs[BODY_IDX], 3.0 * np.outer(a1, obs[48:50]).ravel(),
                                 [3.0 * obs[51]], [3.0]]).astype(np.float32)
        self.pred = self.Wsm @ self.z                                  # öngörülen Δ(beden)
        self.conf = float(self.Wc @ self.z)
        self.prev = (idx, act, obs.copy(), H)
        self.last_obs = obs.copy()
        return act

    # ------------------------------------------------------------ iç gözlem raporu
    def report(self):
        mood = "iyi" if self.mood > 0.05 else ("kötü" if self.mood < -0.05 else "nötr")
        goal = SPEC[self.goal] if self.goal >= 0 else "bilinçdışı gezinme"
        return ("amaç=%s | duygu-tonu=%+.2f ruh-hali=%s(%+.2f) | öz-model hatası=%.2f güven-beklentisi=%.2f"
                % (goal, self.valence, mood, self.mood, self.err_n, self.conf))

    # ------------------------------------------------------------ süreklilik: kayıt
    def save(self, path):
        np.savez(path, Wq=self.Wq, Wsm=self.Wsm, Wc=self.Wc, scale=self.scale, h=self.h, w_nov=self.mb.w_nov,
                 claw=np.stack(self.mb.claw), thr=np.stack(self.mb.thr), Wr=self.Wr, Win=self.Win, mood=self.mood, age=self.age)

    def load(self, path):
        z = np.load(path)
        for k in ("Wq", "Wsm", "Wc", "scale", "h", "Wr", "Win"):
            setattr(self, k, z[k])
        self.mb.w_nov, self.mb.claw, self.mb.thr = z["w_nov"], list(z["claw"]), list(z["thr"])
        self.mood, self.age = float(z["mood"]), int(z["age"])


# ================================================================ yaşam döngüsü + göstergeler
def auc(pos, neg):
    """ROC-AUC (Mann-Whitney): pozitif örneklerin negatiflerden büyük olma olasılığı."""
    if len(pos) < 3 or len(neg) < 3:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def live(mind, world, steps, window=5000, verbose=True, tag=""):
    obs = world.observe()
    died = False
    rows, W = [], dict(steps=0, deaths=0, comfort=0.0, share=np.zeros(5), ign=0, sw=0, ate=0.0, drank=0.0,
                       hurt=0.0, errn=[], conf=[], ev_ext=[], ev_non=[], val=0.0, mood=0.0, life=[])
    last_goal, life_start = -2, 0
    t0 = time.time()
    for t in range(1, steps + 1):
        a = mind.act(obs, died)
        obs, info = world.step(a)
        died = info["died"] is not None
        W["steps"] += 1
        W["deaths"] += died
        W["comfort"] += float(obs[I_E:I_I + 1].min())
        W["share"][mind.goal if mind.goal >= 0 else 4] += 1
        W["ign"] += mind.ignited
        W["sw"] += mind.ignited and mind.goal != last_goal and last_goal >= 0
        if mind.ignited:
            last_goal = mind.goal
        W["ate"] += info["ate"]; W["drank"] += info["drank"]; W["hurt"] += info["hurt"]
        W["val"] += mind.valence; W["mood"] += mind.mood
        if mind.use_sm and mind.pred is not None and mind.age > 500:
            W["errn"].append(mind.err_n); W["conf"].append(mind.conf)
            (W["ev_ext"] if info["ext"] else W["ev_non"]).append(mind.err_vel)
        if died:
            W["life"].append(t - life_start); life_start = t
        if t % window == 0:
            n = W["steps"]
            ec, cf = np.array(W["errn"]), np.array(W["conf"])
            row = dict(t=t, deaths=W["deaths"], per1k=1000.0 * W["deaths"] / n, comfort=W["comfort"] / n,
                       life=float(np.mean(W["life"])) if W["life"] else float(n), share=W["share"] / n,
                       ign=W["ign"] / n, sw=100.0 * W["sw"] / n, ate=W["ate"], drank=W["drank"], hurt=W["hurt"],
                       skill=1.0 - float(ec.mean()) if len(ec) else float("nan"),
                       calib=float(np.corrcoef(ec, cf)[0, 1]) if len(ec) > 10 and cf.std() > 1e-9 else float("nan"),
                       agency=auc(np.array(W["ev_ext"]), np.array(W["ev_non"])),
                       val=W["val"] / n, mood=W["mood"] / n)
            rows.append(row)
            if verbose:
                print("%s t=%6d ölüm/1k=%5.1f ömür=%6.0f konfor=%.2f | pay aç/su/teh/merak/karanlık=%s ateşleme=%.2f "
                      "geçiş/100=%.1f | yem=%d su=%.1f yara=%.1f | öz-model beceri=%+.2f kalibrasyon r=%+.2f ajanlık AUC=%.2f | duygu %+.3f ruh %+.3f | %.0fs"
                      % (tag, t, row["per1k"], row["life"], row["comfort"],
                         "/".join("%.2f" % x for x in row["share"]), row["ign"], row["sw"], row["ate"] / 0.4, row["drank"], row["hurt"], row["skill"], row["calib"],
                         row["agency"], row["val"], row["mood"], time.time() - t0), flush=True)
                print("      iç gözlem:", mind.report())
            W = dict(steps=0, deaths=0, comfort=0.0, share=np.zeros(5), ign=0, sw=0, ate=0.0, drank=0.0, hurt=0.0,
                     errn=[], conf=[], ev_ext=[], ev_non=[], val=0.0, mood=0.0, life=[])
    return rows


def stage_run(args):
    world = Universe(seed=args.seed)
    mind = Mind(mode=args.mode, seed=args.seed, self_model=not args.no_sm, ctx=not args.no_ctx)
    if args.resume and os.path.exists(args.resume):
        mind.load(args.resume)
        print("kaldığı yerden devam: yaş", mind.age)
    live(mind, world, args.steps, window=args.window, tag=args.mode)
    if args.save:
        mind.save(args.save)
        print("kaydedildi:", args.save)


def _one(job):
    mode, seed, steps, window, kw = job
    base, _, flag = mode.partition("-")
    kw = dict(kw, ctx=(flag != "noctx"), self_model=(flag != "nosm"))
    world, mind = Universe(seed=seed), Mind(mode=base, seed=seed, **kw)
    return mode, seed, live(mind, world, steps, window=window, verbose=False)


def stage_compare(args):
    """Aynı evren/tohumlarda modlar yan yana. Son %40 pencerelerin ortalaması (öğrenmenin oturduğu bölge)."""
    from multiprocessing import Pool
    modes = args.modes.split(",")
    seeds = list(range(args.seeds))
    jobs = [(m, s, args.steps, args.window, {}) for m in modes for s in seeds]
    with Pool(min(len(jobs), os.cpu_count() or 1)) as pool:
        out = pool.map(_one, jobs)
    print("evren tohumu=%s adım=%d | ölüm/1000 adım, konfor=min(enerji,su,bütünlük) ortalaması" % (seeds, args.steps))
    for mode in modes:
        runs = [r for (m, s, r) in out if m == mode]
        k = max(1, int(0.6 * len(runs[0])))
        first = np.mean([r[0]["per1k"] for r in runs])
        last = np.array([np.mean([w["per1k"] for w in r[k:]]) for r in runs])
        conf = np.array([np.mean([w["comfort"] for w in r[k:]]) for r in runs])
        print("%-7s ilk pencere %5.2f | son: %5.2f ± %4.2f (tohum başı %s) | konfor %.3f" %
              (mode, first, last.mean(), last.std(), np.round(last, 2).tolist(), conf.mean()), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["run", "compare"], default="run")
    ap.add_argument("--mode", choices=["full", "flat", "random"], default="full")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--window", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-sm", action="store_true")
    ap.add_argument("--no-ctx", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--modes", default="random,flat,full")
    ap.add_argument("--save", default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    {"run": stage_run, "compare": stage_compare}[args.stage](args)


if __name__ == "__main__":
    main()
