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
_RAY_OFF = np.linspace(-Universe.FOV / 2, Universe.FOV / 2, Universe.R)
# öz-model hata normalizasyonunun TABANI: eylemlerin tipik etkisi (hareketsizlik ölçeği çökertmesin)
REF = np.array([1e-3, 1e-3, 1e-3, 0.03, 0.03, 0.1, 0.1], np.float32)
PLAN_G = 3.0                                          # planın eylem üstündeki logit ağırlığı
PRIOR = np.array([0.0, 0.8, 0.0, 0.0, -1.0], np.float32)   # doğuştan: ileri yönelim, geri kaçınma


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class CognitiveMap:
    """Bilişsel harita (yol integrasyonu + duvar çapası + deneyimden öğrenilen kaynak/tehlike haritası).

    Ajana konum VERİLMEZ. Konum, pusula (yön) ile propriosepsiyon hızının integralinden kestirilir
    (böceklerde merkezi kompleks), kayma duvar ışınlarının geometrik tutarlılığıyla düzeltilir (yer çapası).
    Hücre başına öğrenilen: su bulunan yerler, yaralanılan yerler, en son ziyaret zamanı (taze/bayat).
    Planlayıcı: hedeften maliyet-alanı (Bellman), tehlike hücreleri pahalı -> güvenli rota."""

    def __init__(self, L=24.0, G=12, dt=0.25):
        self.L, self.G, self.dt, self.cell = L, G, dt, L / G
        self.pos, self.unc = np.array([L / 2, L / 2]), 1.0
        self.water = np.zeros((G, G), np.float32)
        self.hurt = np.zeros((G, G), np.float32)
        self.seen = np.zeros((G, G), np.float32)                # görerek haritalanan tehlike (acı çekmeden)
        self.last = np.zeros((G, G), np.float32)
        self.th = 0.0
        self.target = {}                                      # uzman -> (hücre, atandığı zaman, alan)

    def cell_of(self, p):
        return tuple(np.clip((np.asarray(p) / self.cell).astype(int), 0, self.G - 1))

    def update(self, obs, prev, died, now):
        if died:
            self.pos, self.unc = np.array([self.L / 2, self.L / 2]), 1.0
        self.th = float(np.arctan2(obs[Universe.I_COMPASS + 1], obs[Universe.I_COMPASS]))
        f = np.array([np.cos(self.th), np.sin(self.th)])
        l = np.array([-f[1], f[0]])
        self.pos = self.pos + (obs[48] * f + obs[49] * l) * 2.0 * self.dt              # yol integrasyonu
        # duvar çapası: görünen duvar noktası gerçekte bir duvar üzerinde olmalı
        vis = 5.0 + 7.0 * obs[51]
        rays = obs[:45].reshape(Universe.R, 5)[:, 0]
        ang = self.th + _RAY_OFF
        corr, n = np.zeros(2), 0
        for k in np.flatnonzero(rays > 0.02):
            hit = self.pos + vis * (1 - rays[k]) * np.array([np.cos(ang[k]), np.sin(ang[k])])
            dx0, dx1, dy0, dy1 = hit[0], hit[0] - self.L, hit[1], hit[1] - self.L
            j = int(np.argmin(np.abs([dx0, dx1, dy0, dy1])))
            corr[j // 2] -= [dx0, dx1, dy0, dy1][j]
            n += 1
        if n:
            self.pos = self.pos + (0.12 + 0.5 * self.unc) * corr / n
        if self.unc < 0.3:                                    # konum güvenilirse görülen tehlikeyi haritala
            hz = obs[:45].reshape(Universe.R, 5)[:, RAY_HAZ]
            for k in np.flatnonzero(hz > 0.02):
                dvec = np.array([np.cos(ang[k]), np.sin(ang[k])])
                for extra in (0.3, 1.2, 2.2):
                    p = self.pos + dvec * (vis * (1 - hz[k]) + extra)
                    if 0 <= p[0] < self.L and 0 <= p[1] < self.L:
                        self.seen[self.cell_of(p)] = 1.0
        self.unc *= 0.98
        self.pos = np.clip(self.pos, 0.3, self.L - 0.3)
        c = self.cell_of(self.pos)
        self.last[c] = now
        if prev is not None:
            if obs[I_W] - prev[I_W] > 0.02:
                self.water[c] += 1.0
            if prev[I_I] - obs[I_I] > 0.01:
                self.hurt[c] += 1.0

    def _field(self, goal_mask):
        cost = 1.0 + 8.0 * np.minimum(self.hurt, 3.0) + 20.0 * self.seen
        d = np.full((self.G, self.G), 1e9, np.float32)
        d[goal_mask] = 0.0
        for _ in range(2 * self.G):
            p = np.pad(d, 1, constant_values=1e9)
            m = np.minimum(np.minimum(p[:-2, 1:-1], p[2:, 1:-1]), np.minimum(p[1:-1, :-2], p[1:-1, 2:]))
            nd = np.minimum(d, m + cost)
            if np.array_equal(nd, d):
                break
            d = nd
        return d

    def choose_target(self, kind, now):
        """kind 'su': öğrenilmiş su hücresi (yoksa bayat). 'gez': en bayat ve yakın, tehlikesiz hücre."""
        if kind == "su" and self.water.max() > 0:
            return np.argwhere(self.water >= 0.5 * self.water.max())
        age = now - self.last
        ci, cj = self.cell_of(self.pos)
        ii, jj = np.meshgrid(np.arange(self.G), np.arange(self.G), indexing="ij")
        score = age / 100.0 - 0.15 * np.hypot(ii - ci, jj - cj) - 5.0 * ((self.hurt > 0) | (self.seen > 0))
        return np.array([np.unravel_index(int(np.argmax(score)), score.shape)])

    def plan(self, key, kind, now):
        """Hedefe giden rotanın bir sonraki yönü için yön hatası (rad) ya da None. Eylemi Mind seçer (öğrenilmiş modelle)."""
        cur = self.cell_of(self.pos)
        tg = self.target.get(key)
        if tg is None or now - tg[1] > 120 or tuple(tg[0]) == cur and kind == "gez" or kind != tg[3]:
            cells = self.choose_target(kind, now)
            mask = np.zeros((self.G, self.G), bool)
            mask[tuple(cells.T)] = True
            tg = (cells[0], now, self._field(mask), kind)
            self.target[key] = tg
        d = tg[2]
        if cur in {tuple(c) for c in np.argwhere(d == 0)}:
            wp = (np.array(cur) + 0.5) * self.cell                                # hedef hücrede: merkeze yaklaş
        else:
            best, bv = None, d[cur]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    n = (cur[0] + di, cur[1] + dj)
                    if (di or dj) and 0 <= n[0] < self.G and 0 <= n[1] < self.G and d[n] < bv:
                        best, bv = n, d[n]
            if best is None:
                return None
            wp = (np.array(best) + 0.5) * self.cell
        return wrap(np.arctan2(wp[1] - self.pos[1], wp[0] - self.pos[0]) - self.th)      # hedef yön hatası (rad)

    def state(self):
        return dict(water=self.water, hurt=self.hurt, last=self.last)


class Mind:
    def __init__(self, mode="full", seed=0, n_kc=3000, n_res=96, gamma=0.95, lr=0.25, self_model=True, shape=2.0, ctx=True, use_map=True, grow=True, model=True):
        self.mode, self.rng, self.gamma, self.lr = mode, np.random.default_rng(seed), gamma, lr
        self.use_sm, self.shape, self.ctx, self.use_map = self_model, shape, ctx, use_map
        self.grow, self.use_model = grow, model
        self.map = CognitiveMap()
        self.n_ctx = 48
        n_act = Universe.N_ACT
        # --- nöronsal temel: KC kodu yalnızca duyusal girdiden (iç duyu HARİÇ) -> Q ihtiyaçtan bağımsız
        groups = [np.arange(c, 45, 5) for c in range(5)] + [np.array([48, 49, 50])]   # kanal başına ışınlar | proprio
        if ctx:                                                # bağlam lobu: sürekli iç durum h davranışa girer
            groups.append(np.arange(Universe.OBS_DIM, Universe.OBS_DIM + self.n_ctx))
        self.mb = LobedMushroomBody(groups, n_per=n_kc // len(groups), seed=seed)
        n_kc = self.mb.n_kc
        self.n_kc = n_kc
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
        self.nz = n_res + n_act + len(BODY_IDX) + 2 * n_act + 1 + 1 + 6
        self.n_body = len(BODY_IDX) + 1                                 # beden(6) + yön değişimi
        self.scale = np.full(self.n_body, 0.05, np.float32)             # sinyal değişkenliği (normalizasyon)
        # --- BAĞLAMLAR: (refleks Q-kafaları + öz-model). Sürprizle yeni bağlam açılır, eskiler DONAR (sıfır unutma)
        self.ctxs = [self._new_ctx()]
        self.cur, self.cerr, self.oerr, self.cool = 0, [1.0], [0.0], 0
        self.prob = None                                               # deneme (probation) durumu: {'old': idx, 't': adım}
        self.freeze = False                                            # sürprizde aktif bağlamın plastisitesi kapanır
        self.n_created = 1
        # --- çalışma alanı ve duygu
        self.acc = np.zeros(len(SPEC), np.float32)
        self.B = np.zeros(self.n_B, np.float32)
        self.mood, self.valence, self.goal, self.ignited = 0.0, 0.0, -1, False
        self.prev = None        # (kc_idx, action, obs, H)
        self.pred, self.z, self.conf = None, None, 0.0
        self.err_n = 0.0
        self.err_vel = 0.0
        self.err_head = 0.0
        self.age = 0
        self.last_obs = None
        self.last_y = None

    # ------------------------------------------------------------ bağlamlar
    def _new_ctx(self, src=None):
        c = dict(Wq=np.zeros((len(SPEC), Universe.N_ACT, self.n_kc), np.float32),
                 Wsm=np.zeros((self.n_body, self.nz), np.float32), Wc=np.zeros(self.nz, np.float32),
                 good=False, n=0, ok=0, V=np.full(3, np.nan), Vn=np.zeros(3))                                      # good: bu bağlam dünyayı bir kez açıkladı mı (olgun mu)
        if src is not None:                                           # sıcak başlangıç: bilgi KOPYALANIR, eskisi donuk kalır
            for k in ("Wq", "Wsm", "Wc"):
                c[k][:] = src[k]
            pass                                                      # anlamsal bellek (V) KOPYALANMAZ: aday sonuçları sıfırdan öğrenir
        return c

    Wq = property(lambda s: s.ctxs[s.cur]["Wq"])
    Wsm = property(lambda s: s.ctxs[s.cur]["Wsm"], lambda s, v: s.ctxs[s.cur].__setitem__("Wsm", v))
    Wc = property(lambda s: s.ctxs[s.cur]["Wc"], lambda s, v: s.ctxs[s.cur].__setitem__("Wc", v))

    def _infer_context(self):
        """Bağlam çıkarımı: olgun bağlam bir kez dünyayı açıklar (good); sonra sürpriz görürse
        (1) önce başka bir olgun bağlam daha iyi mi diye bakılır (eski dünyaya DÖNÜŞ),
        (2) değilse aktif bağlam DONAR ve klonu deneme (probation) bağlamı olarak eğitilir;
        deneme 300 adım sonra eski bağlam daha iyi açıklıyorsa (yanlış alarm) geri alınır."""
        score = np.array(self.cerr) + 2.0 * np.array(self.oerr)
        c = self.ctxs[self.cur]
        cs = float(score[self.cur])
        self.cool = max(0, self.cool - 1)
        c["ok"] = c["ok"] + 1 if cs < 0.7 else 0
        if not c["good"] and c["ok"] > 200 and c["n"] > 4000:           # olgun: dünyayı uzun süre tutarlı açıkladı
            c["good"] = True
        if self.prob is not None:
            self.prob["t"] += 1
            self.freeze = False
            if self.prob["t"] >= 400:
                cand = self.ctxs[self.cur]

                def same_world(i):                                      # anlamsal imza (V işaretleri) aday ile örtüşüyor mu?
                    o = self.ctxs[i]
                    ks = [k for k in range(3) if np.isfinite(o["V"][k]) and np.isfinite(cand["V"][k]) and o["Vn"][k] >= 3 and cand["Vn"][k] >= 2]
                    return len(ks) >= 1 and all(o["V"][k] * cand["V"][k] > 0 for k in ks) and self.cerr[i] < 1.0
                others = [i for i in range(len(self.ctxs)) if i != self.cur and self.ctxs[i]["good"] and same_world(i)]
                o = min(others, key=lambda i: score[i]) if others else None
                if o is not None:                                       # eski bir dünya bu olguyu zaten açıklıyor -> aday ATILIR
                    self.ctxs.pop(self.cur)
                    self.cerr.pop(self.cur)
                    self.oerr.pop(self.cur)
                    self.cur = o
                    self.n_created -= 1
                self.prob = None
                self.cool = 1500
            return
        self.freeze = False
        if not c["good"] or self.cool > 0:
            return
        best = int(np.argmin(score))
        if best != self.cur and self.ctxs[best]["good"] and score[best] < 0.6 * cs:   # eski bir dünyaya dönüldü
            self.cur, self.cool = best, 100
        elif cs > 1.2 and len(self.ctxs) < 8:                           # yeni bir dünya: eskiyi dondur, klonu eğit
            old = self.cur
            self.ctxs.append(self._new_ctx(self.ctxs[old]))
            self.cerr.append(float(self.cerr[old]))
            self.oerr.append(0.0)
            self.cur, self.prob = len(self.ctxs) - 1, {"old": old, "t": 0}
            self.n_created += 1

    @staticmethod
    def _bodyvec(obs):
        return np.r_[obs[BODY_IDX], np.arctan2(obs[Universe.I_COMPASS + 1], obs[Universe.I_COMPASS])].astype(np.float32)

    @staticmethod
    def _delta(y, y0):
        d = y - y0
        d[-1] = wrap(d[-1])
        return d

    def _zvec(self, obs, act):
        a1 = np.zeros(Universe.N_ACT, np.float32)
        a1[act] = 1.0
        return np.concatenate([0.25 * self.h, 3.0 * a1, 3.0 * obs[BODY_IDX], 3.0 * np.outer(a1, obs[48:50]).ravel(),
                               [3.0 * obs[51]], [3.0], 3.0 * np.abs(self._phi(obs)[:3]), 3.0 * a1[1] * np.abs(self._phi(obs)[:3])]
                              ).astype(np.float32)

    def _steer(self, err, obs):
        """Hedef yön hatasından eylem: ÖĞRENİLMİŞ modelin 'hangi eylem yönümü nasıl değiştirir' öngörüsüyle.
        Kumandalar ters çevrilirse ezber değil model değişir; plan kendiliğinden uyum sağlar."""
        if abs(err) <= 0.3:
            return 1
        if self.use_model and self.use_sm and self.age > 300:
            pr = np.array([float(self.Wsm[-1] @ self._zvec(obs, a)) for a in (2, 3)])
            s = np.sign(err) * pr
            if s.max() > 0.05:
                return 2 + int(np.argmax(s))
        return 2 if err > 0 else 3

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
        if self.use_map:
            self.map.update(obs, self.prev[2] if self.prev is not None else None, died, self.age)
        # ---------- valans (homeostatik hata türevi) ve ruh hali
        if self.prev is not None:
            raw = -(H - self.prev[3]) * 5.0
            hurt = max(0.0, self.prev[2][I_I] - obs[I_I])
            self.valence = float(np.clip(raw - 30.0 * hurt, -2.0, 2.0)) if not died else -2.0
        self.mood += 0.01 * (self.valence - self.mood)
        # ---------- öğrenme (dopamin = TD hatası, yalnızca aktif KC'ler)
        if self.prev is not None:
            pidx, pa, pobs, _ = self.prev
            r = np.array([3.0 * max(-0.5, obs[I_E] - pobs[I_E]),
                          3.0 * max(0.0, obs[I_W] - pobs[I_W]),
                          5.0 * (obs[I_I] - pobs[I_I]),
                          1.0 * nov], np.float32)
            r += self.shape * (self.gamma * self._phi(obs) - self._phi(pobs)) * np.array([1, 1, 1, 0], np.float32)
            qn = self._q(idx).max(1) if not died else np.zeros(len(SPEC), np.float32)
            delta = np.clip(r + self.gamma * qn - self._q(pidx)[:, pa], -3.0, 3.0)
            if not self.freeze:
                self.Wq[:, pa, pidx] += (self.lr * delta / max(1, len(pidx)))[:, None]
                self.ctxs[self.cur]["n"] += 1
        # ---------- öz-model: önceki öngörüyü gerçekle karşılaştır; bağlam çıkarımı
        yv = self._bodyvec(obs)
        if self.pred is not None and self.use_sm and not died:        # ölüm/yeniden doğuş geçişi öğrenme verisi değildir
            d = self._delta(yv, self.last_y)
            sc = np.maximum(self.scale, REF)
            evs = []                                                   # sonuç olayları: (değişken, değişim)
            if abs(d[0]) > 0.05:
                evs.append((0, float(d[0])))                           # yem (yedim -> enerji +/-)
            if d[1] > 0.02:
                evs.append((1, float(d[1])))                           # içtim
            if d[2] < -0.02:
                evs.append((2, float(d[2])))                           # yaralandım
            for ci, c in enumerate(self.ctxs):
                pd = c["Wsm"] @ self.z
                en_c = np.abs(d - pd) / sc
                self.cerr[ci] = 0.97 * self.cerr[ci] + 0.03 * float(en_c[[3, 4, 6]].mean())
                for k, dv in evs:                                       # anlamsal bellek: 'şey X bedene ne yapar' işareti doğru muydu?
                    v = c["V"][k]
                    if np.isfinite(v) and c["Vn"][k] >= 3:
                        self.oerr[ci] = 0.6 * self.oerr[ci] + 0.4 * (1.0 if v * dv < 0 else 0.0)
            cc = self.ctxs[self.cur]
            if not self.freeze:
                for k, dv in evs:                                       # çelişen olay olgun bağlamın belleğini KİRLETMEZ
                    v = cc["V"][k]
                    if not np.isfinite(v):
                        cc["V"][k], cc["Vn"][k] = dv, 1.0
                    elif cc["Vn"][k] < 3 or v * dv >= 0 or not cc["good"]:
                        cc["V"][k] += 0.2 * (dv - v)
                        cc["Vn"][k] += 1.0
            err = d - self.pred
            en = np.abs(err) / sc
            self.err_n = float(en[3:5].mean())          # hareket boyutları: öngörülebilir olan öz-etki
            self.err_vel = float(en[3:5].mean())
            self.err_head = float(en[6])
            zz = float(self.z @ self.z) + 1e-3
            if not self.freeze:                                        # sürprizde aktif bağlam KİRLENMESİN
                self.Wsm += 0.5 * np.outer(err, self.z) / zz
                self.Wc += 0.2 * (self.err_n - float(self.Wc @ self.z)) * self.z / zz
            self.scale += 0.01 * (np.abs(d) - self.scale)
            if self.grow:
                self._infer_context()
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
        phi = self._phi(obs)

        def proposal(k):
            """Uzman k'nın eylem logitleri: refleks Q (öğrenilmiş) + gerekirse bellek/plan (kaynak görünmüyorsa)."""
            lg = 8.0 * Q[k] + PRIOR
            if self.use_map and k != 2 and not (k < 2 and phi[k] > 0.05):
                pe = self.map.plan(k, "su" if k == 1 else "gez", self.age)
                if pe is not None:
                    lg[self._steer(pe, obs)] += PLAN_G
            return lg
        if self.mode == "random":
            logits, self.goal, self.ignited = PRIOR.copy(), -1, False
        elif self.mode == "flat":                                      # çalışma alanı YOK: ağırlıklı karışım
            w = urg + 0.1
            logits = sum(w[k] * proposal(k) for k in range(len(SPEC))) / w.sum()
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
            logits = proposal(win) if self.ignited else PRIOR.copy()
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
        self.z = self._zvec(obs, act)
        self.pred = self.Wsm @ self.z                                  # öngörülen Δ(beden)
        self.conf = float(self.Wc @ self.z)
        self.prev = (idx, act, obs.copy(), H)
        self.last_obs = obs.copy()
        self.last_y = yv
        return act

    # ------------------------------------------------------------ iç gözlem raporu
    def report(self):
        mood = "iyi" if self.mood > 0.05 else ("kötü" if self.mood < -0.05 else "nötr")
        goal = SPEC[self.goal] if self.goal >= 0 else "bilinçdışı gezinme"
        return ("amaç=%s | duygu-tonu=%+.2f ruh-hali=%s(%+.2f) | öz-model hatası=%.2f güven-beklentisi=%.2f"
                % (goal, self.valence, mood, self.mood, self.err_n, self.conf))

    # ------------------------------------------------------------ süreklilik: kayıt
    def save(self, path):
        np.savez(path, Wq=np.stack([c['Wq'] for c in self.ctxs]), Wsm=np.stack([c['Wsm'] for c in self.ctxs]),
                 Wc=np.stack([c['Wc'] for c in self.ctxs]), cerr=np.array(self.cerr), oerr=np.array(self.oerr), good=np.array([c['good'] for c in self.ctxs]), V=np.stack([c['V'] for c in self.ctxs]), Vn=np.stack([c['Vn'] for c in self.ctxs]), cur=self.cur, scale=self.scale, h=self.h, w_nov=self.mb.w_nov,
                 claw=np.stack(self.mb.claw), thr=np.stack(self.mb.thr), Wr=self.Wr, Win=self.Win, mood=self.mood, age=self.age,
                 m_water=self.map.water, m_hurt=self.map.hurt, m_last=self.map.last, m_seen=self.map.seen)

    def load(self, path):
        z = np.load(path)
        for k in ("scale", "h", "Wr", "Win"):
            setattr(self, k, z[k])
        self.ctxs = [dict(Wq=z["Wq"][i], Wsm=z["Wsm"][i], Wc=z["Wc"][i], good=bool(z["good"][i]), n=0, ok=0, V=z["V"][i], Vn=z["Vn"][i]) for i in range(len(z["Wq"]))]
        self.cerr, self.oerr, self.cur = list(z["cerr"]), list(z["oerr"]), int(z["cur"])
        self.mb.w_nov, self.mb.claw, self.mb.thr = z["w_nov"], list(z["claw"]), list(z["thr"])
        self.mood, self.age = float(z["mood"]), int(z["age"])
        self.map.water, self.map.hurt, self.map.last, self.map.seen = z["m_water"], z["m_hurt"], z["m_last"], z["m_seen"]


# ================================================================ yaşam döngüsü + göstergeler
def auc(pos, neg):
    """ROC-AUC (Mann-Whitney): pozitif örneklerin negatiflerden büyük olma olasılığı."""
    if len(pos) < 3 or len(neg) < 3:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def live(mind, world, steps, window=5000, verbose=True, tag="", events=None):
    obs = world.observe()
    died = False
    rows, W = [], dict(steps=0, deaths=0, comfort=0.0, share=np.zeros(5), ign=0, sw=0, ate=0.0, drank=0.0,
                       hurt=0.0, errn=[], conf=[], ev_ext=[], ev_non=[], val=0.0, mood=0.0, life=[], cause={}, loc=[], head=[])
    last_goal, life_start = -2, 0
    t0 = time.time()
    for t in range(1, steps + 1):
        if events and t in events:
            events[t](world)
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
            W["errn"].append(mind.err_n); W["conf"].append(mind.conf); W["head"].append(mind.err_head)
            (W["ev_ext"] if info["ext"] else W["ev_non"]).append(mind.err_vel)
        if died:
            W["life"].append(t - life_start); life_start = t
            W["cause"][info["died"]] = W["cause"].get(info["died"], 0) + 1
        if mind.use_map and mind.age > 50:
            W["loc"].append(float(np.hypot(*(mind.map.pos - world.pos))))
        if t % window == 0:
            n = W["steps"]
            ec, cf = np.array(W["errn"]), np.array(W["conf"])
            row = dict(t=t, deaths=W["deaths"], per1k=1000.0 * W["deaths"] / n, comfort=W["comfort"] / n,
                       life=float(np.mean(W["life"])) if W["life"] else float(n), share=W["share"] / n,
                       ign=W["ign"] / n, sw=100.0 * W["sw"] / n, ate=W["ate"], drank=W["drank"], hurt=W["hurt"],
                       skill=1.0 - float(ec.mean()) if len(ec) else float("nan"),
                       calib=float(np.corrcoef(ec, cf)[0, 1]) if len(ec) > 10 and cf.std() > 1e-9 else float("nan"),
                       agency=auc(np.array(W["ev_ext"]), np.array(W["ev_non"])),
                       val=W["val"] / n, mood=W["mood"] / n, cause=dict(W["cause"]), nctx=len(mind.ctxs), cur=mind.cur, head=1.0 - float(np.mean(W["head"])) if W["head"] else float("nan"),
                       loc=float(np.mean(W["loc"])) if W["loc"] else float("nan"))
            rows.append(row)
            if verbose:
                print("%s t=%6d ölüm/1k=%5.1f ömür=%6.0f konfor=%.2f | pay aç/su/teh/merak/karanlık=%s ateşleme=%.2f "
                      "geçiş/100=%.1f | yem=%d su=%.1f yara=%.1f | öz-model beceri=%+.2f kalibrasyon r=%+.2f ajanlık AUC=%.2f | konum hatası=%.2f | ölüm nedeni=%s | duygu %+.3f ruh %+.3f | %.0fs"
                      % (tag, t, row["per1k"], row["life"], row["comfort"],
                         "/".join("%.2f" % x for x in row["share"]), row["ign"], row["sw"], row["ate"] / 0.4, row["drank"], row["hurt"], row["skill"], row["calib"],
                         row["agency"], row["loc"], row["cause"], row["val"], row["mood"], time.time() - t0), flush=True)
                print("      iç gözlem:", mind.report())
            W = dict(steps=0, deaths=0, comfort=0.0, share=np.zeros(5), ign=0, sw=0, ate=0.0, drank=0.0, hurt=0.0,
                     errn=[], conf=[], ev_ext=[], ev_non=[], val=0.0, mood=0.0, life=[], cause={}, loc=[], head=[])
    return rows


def stage_run(args):
    world = Universe(seed=args.seed)
    mind = Mind(mode=args.mode, seed=args.seed, self_model=not args.no_sm, ctx=not args.no_ctx, use_map=not args.no_map)
    if args.resume and os.path.exists(args.resume):
        mind.load(args.resume)
        print("kaldığı yerden devam: yaş", mind.age)
    live(mind, world, args.steps, window=args.window, tag=args.mode)
    if args.save:
        mind.save(args.save)
        print("kaydedildi:", args.save)


SHIFT = False
SHIFT_KIND = "invert"


def _SHIFT_EVENTS(steps):
    q = steps // 4                                    # evreler: normal | değişmiş | normal | değişmiş
    def flip(v):
        return lambda w: setattr(w, SHIFT_KIND, v)
    return {q + 1: flip(True), 2 * q + 1: flip(False), 3 * q + 1: flip(True)}


def _one(job):
    mode, seed, steps, window, kw = job
    base, *flags = mode.split("-")
    kw = dict(kw, ctx=("noctx" not in flags), self_model=("nosm" not in flags), use_map=("nomap" not in flags),
              grow=("nogrow" not in flags), model=("nomodel" not in flags))
    world, mind = Universe(seed=seed), Mind(mode=base, seed=seed, **kw)
    return mode, seed, live(mind, world, steps, window=window, verbose=False, events=_SHIFT_EVENTS(steps) if SHIFT else None)


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
        cause = {}
        for r in runs:
            for w in r[k:]:
                for c, v in w["cause"].items():
                    cause[c] = cause.get(c, 0) + v
        loc = np.nanmean([w["loc"] for r in runs for w in r[k:]]) if any(not np.isnan(w["loc"]) for r in runs for w in r) else float("nan")
        print("%-11s ilk %5.2f | son: %5.2f ± %4.2f (tohum başı %s) | konfor %.3f | konum hatası %.2f | nedenler %s" %
              (mode, first, last.mean(), last.std(), np.round(last, 2).tolist(), conf.mean(), loc, cause), flush=True)
    return out


def stage_shift(args):
    """Kural değişimi: kumanda ters (çeyrek başına: normal | ters | normal | ters). Sıfır unutma ve uyum ölçümü.
    Eski bağlamlar donar; dönüşte A'nın beceri ve modeli hemen geri gelmeli (yeniden öğrenme olmadan)."""
    global SHIFT, SHIFT_KIND
    from multiprocessing import Pool
    SHIFT, SHIFT_KIND = True, args.kind
    modes = args.modes.split(",")
    seeds = list(range(args.seeds))
    jobs = [(m, s, args.steps, args.window, {}) for m in modes for s in seeds]
    with Pool(min(len(jobs), os.cpu_count() or 1)) as pool:
        out = pool.map(_one, jobs)
    nw = args.steps // args.window
    q = nw // 4
    names = ["A normal", "B değişmiş", "A' normal", "B' değişmiş"]
    print("kural değişimi (%s)" % args.kind, " | pencere=%d adım, evre=%d pencere | ölüm/1000 adım: [evrenin ilk 3 penceresi -> son 3 penceresi]" % (args.window, q))
    for mode in modes:
        runs = [r for (mm, s, r) in out if mm == mode]
        cells = []
        for p in range(4):
            f = np.mean([np.mean([w["per1k"] for w in r[p * q:p * q + 3]]) for r in runs])
            l = np.mean([np.mean([w["per1k"] for w in r[(p + 1) * q - 3:(p + 1) * q]]) for r in runs])
            cf = np.mean([np.mean([w["comfort"] for w in r[p * q:p * q + 3]]) for r in runs])
            cl = np.mean([np.mean([w["comfort"] for w in r[(p + 1) * q - 3:(p + 1) * q]]) for r in runs])
            cells.append("%s ölüm %.2f->%.2f konfor %.2f->%.2f" % (names[p], f, l, cf, cl))
        head = [np.nanmean([w["head"] for r in runs for w in r[(p + 1) * q - 2:(p + 1) * q]]) for p in range(4)]
        print("%-20s %s | bağlam sayısı %.1f | yön-öngörü becerisi (A,B,A',B') %s" %
              (mode, "\n    ".join(cells), np.mean([r[-1]["nctx"] for r in runs]), np.round(head, 2).tolist()), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["run", "compare", "shift"], default="run")
    ap.add_argument("--mode", choices=["full", "flat", "random"], default="full")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--window", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-sm", action="store_true")
    ap.add_argument("--no-ctx", action="store_true")
    ap.add_argument("--kind", choices=["invert", "poison"], default="invert")
    ap.add_argument("--no-map", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--modes", default="random,flat,full")
    ap.add_argument("--save", default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    {"run": stage_run, "compare": stage_compare, "shift": stage_shift}[args.stage](args)


if __name__ == "__main__":
    main()
