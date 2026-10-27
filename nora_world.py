#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_world.py — NoRa'nın evreni: küçük, deterministik, fizik yasalı, kapalı döngü bir dünya.

Amaç eğitim ortamı değil YAŞAM ortamı: NoRa burada doğar, ihtiyaç duyar, hareket eder, ölür, yeniden doğar;
beyni ölümler arasında KESİNTİSİZ sürer. İleride her AGI/bilinç testi bu evrende yapılacak.

Fizik (basitleştirilmiş ama gerçek yasalar):
  - Newton: kuvvet -> ivme -> hız -> konum; sürtünme (v *= exp(-k dt)); eylemsizlik (itiş bitince kayar)
  - Esnek olmayan çarpışma (restitüsyon), kaya çarpışmada MOMENTUM korunur (itilebilir cisim, kütle 3)
  - Metabolizma: hareket enerji yakar (itiş^2), gece daha pahalı; su ve yiyecek tüketilir, geri gelir
  - Gün/gece döngüsü: görüş menzili değişir (karanlıkta kör)
  - Rüzgâr: ajanın EYLEMİ DIŞINDA gelen kısa dış kuvvetler -> "bunu ben mi yaptım?" (ajanlık) testi
Duyular: 9 ışınlık egosantrik görme x 5 kanal (duvar, yiyecek, su, tehlike, kaya), iç duyu (enerji,
  su, bütünlük), propriosepsiyon (ileri/yan hız), dokunma, ışık, pusula (yön; konum VERİLMEZ).
İhtiyaçlar: enerji (yiyecek), su (havuz), bütünlük (tehlike bölgeleri yaralar). Biri 0 olursa ölüm.

Kullanım:  DPY nora_world.py --stage test      # fizik ve görme öz-sınaması
"""
import argparse, sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CH = ("duvar", "yiyecek", "su", "tehlike", "kaya")
ACTIONS = ("dur", "ileri", "sola", "sağa", "geri")


class Universe:
    N_ACT = 5
    R = 9                                   # ışın sayısı
    FOV = np.deg2rad(150.0)
    OBS_DIM = R * 5 + 3 + 2 + 1 + 1 + 2     # 45 görme | 3 iç duyu | 2 hız | 1 dokunma | 1 ışık | 2 pusula
    I_RAYS, I_INTERO, I_SPEED, I_TOUCH, I_LIGHT, I_COMPASS = 0, 45, 48, 50, 51, 52

    def __init__(self, seed=0, size=24.0, n_food=14, dt=0.25, day_len=600, wind=True, n_term=0, metabolism=True, food_haz=0.0, metab_scale=1.0, mortal=False):
        self.food_haz = food_haz                              # yiyeceğin tehlike bölgesinin kenarına konma olasılığı (konjonktif görev: yiyeceğe yaklaşma tarafı tehlikeye bağlı)
        self.mortal = bool(mortal)                            # True -> ölüp ölüp dirilmez; tek bir geri dönüşsüz yaşamı vardır!
        self.dead_forever = False
        self.rng = np.random.default_rng(seed)
        self.L, self.dt, self.day_len, self.wind_on = float(size), dt, day_len, wind
        self.A, self.fric, self.turn = 4.0, 2.0, 0.40       # itiş ivmesi, sürtünme, dönüş açısı
        self.rock_fric = 4.0
        self.c_base, self.c_move, self.food_gain = 0.0006, 0.0006, 0.5      # metabolizma kalibrasyonu
        self.beam = 0.35                                     # ışın yarı-genişliği (ışınlar arası boşluğu kapatır)
        self.ra, self.rr = 0.5, 0.8                          # ajan ve kaya yarıçapı
        self.m_agent, self.m_rock = 1.0, 3.0
        self.rest = 0.25                                     # restitüsyon
        self.food_r, self.zone_w, self.zone_h = 0.6, 2.4, 2.2
        self.ray_off = np.linspace(-self.FOV / 2, self.FOV / 2, self.R)
        lay = np.random.default_rng(seed + 7919)             # düzen seed'e bağlı, SABİT evren
        self.water_c, self.haz_c = self._layout(lay)
        self.rock = lay.uniform(4, self.L - 4, 2)
        self.rock_v = np.zeros(2)
        self.metab = (1.0 if metabolism else 0.0) * metab_scale                # 0: ihtiyaçlar tükenmez (çürütme testi için 'cansız' evren)
        self.is_term = np.zeros(n_food, bool)                # 'bilmeceli kutu': yiyecek gibi görünür, ancak soru çözülünce enerji verir
        self.is_term[:n_term] = True
        self.food = np.zeros((n_food, 2))
        for i in range(n_food):
            self.food[i] = self._spawn_food()
        self.t = 0
        self.wind_left, self.wind_f = 0, np.zeros(2)
        self.lives, self.touch = 0, 0.0
        self.poison = False                                  # kural değişimi 2: yiyecek zehirli (değer tersine döner)
        self.swap = False                                    # algısal yeniden eşleme ('ters gözlük'): yiyecek/su/tehlike kanalları döner
        self.invert = False                                  # kural değişimi: sağ/sol kumanda ters (duyu-motor yeniden eşleme)
        self._respawn()

    # ---- kurulum
    def _layout(self, lay):
        pts = []
        while len(pts) < 4:
            p = lay.uniform(4, self.L - 4, 2)
            if all(np.hypot(*(p - q)) > 8.0 for q in pts):
                pts.append(p)
        return np.array(pts[:2]), np.array(pts[2:])

    def relayout(self, layout_seed):
        """Düzen değişimi: su/tehlike bölgeleri yeni yerlere taşınır (harita bilgisi eskir). Aynı seed her zaman aynı düzen."""
        self.water_c, self.haz_c = self._layout(np.random.default_rng(layout_seed + 7919))
        for i in range(len(self.food)):
            if np.any(np.hypot(*(self.haz_c - self.food[i]).T) <= self.zone_h + 1.5):
                self.food[i] = self._spawn_food()

    def _spawn_food(self):
        if self.food_haz > 0 and self.rng.random() < self.food_haz:
            for _ in range(20):
                c = self.haz_c[self.rng.integers(len(self.haz_c))]
                a = self.rng.uniform(0, 2 * np.pi)
                p = c + (self.zone_h + 0.55) * np.array([np.cos(a), np.sin(a)])
                if np.all((p > 1.0) & (p < self.L - 1.0)):
                    return p
        while True:
            p = self.rng.uniform(1.0, self.L - 1.0, 2)
            if np.all(np.hypot(*(self.haz_c - p).T) > self.zone_h + 1.5):
                return p

    def _respawn(self):
        while True:
            self.pos = self.rng.uniform(2.0, self.L - 2.0, 2)
            if np.all(np.hypot(*(self.haz_c - self.pos).T) > self.zone_h + 2.0):
                break
        self.vel, self.th = np.zeros(2), self.rng.uniform(0, 2 * np.pi)
        self.e = self.w = self.i = 0.8
        self.lives += 1

    # ---- gün/gece
    def light(self):
        return 0.5 + 0.5 * np.cos(2 * np.pi * self.t / self.day_len)

    # ---- görme
    def _rays(self):
        vis = 5.0 + 7.0 * self.light()
        ang = self.th + self.ray_off
        d = np.stack([np.cos(ang), np.sin(ang)], 1)
        p = self.pos
        out = np.zeros((self.R, 5), np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(d[:, 0] > 1e-9, (self.L - p[0]) / d[:, 0], np.where(d[:, 0] < -1e-9, -p[0] / d[:, 0], np.inf))
            ty = np.where(d[:, 1] > 1e-9, (self.L - p[1]) / d[:, 1], np.where(d[:, 1] < -1e-9, -p[1] / d[:, 1], np.inf))
        out[:, 0] = np.clip(1 - np.minimum(tx, ty) / vis, 0, 1)
        sets = [(self.food, self.food_r), (self.water_c, self.zone_w), (self.haz_c, self.zone_h),
                (self.rock[None, :], self.rr)]
        for ch, (C, rad) in enumerate(sets, start=1):
            oc = C - p
            t = d @ oc.T
            c2 = (oc ** 2).sum(1)
            disc = (rad + self.beam) ** 2 - (c2[None, :] - t * t)   # kalın ışın (koni izleme)
            hit = (disc > 0) & (t > 0)
            dist = np.where(hit, t - np.sqrt(np.maximum(disc, 0.0)), np.inf)
            dist = np.where((c2 < rad * rad)[None, :], 0.0, np.maximum(dist, 0.0))
            out[:, ch] = np.clip(1 - dist.min(1) / vis, 0, 1)
        if self.swap:
            out[:, 1:4] = out[:, [2, 3, 1]]                   # yiyecek görünümü=su, su=tehlike, tehlike=yiyecek
        return out, vis

    def observe(self):
        rays, _ = self._rays()
        f = np.array([np.cos(self.th), np.sin(self.th)])
        l = np.array([-f[1], f[0]])
        o = np.zeros(self.OBS_DIM, np.float32)
        o[:45] = rays.reshape(-1)
        o[45:48] = (self.e, self.w, self.i)
        o[48:50] = (self.vel @ f / 2.0, self.vel @ l / 2.0)
        o[50] = self.touch
        o[51] = self.light()
        o[52:54] = (np.cos(self.th), np.sin(self.th))      # pusula (yön hücreleri): mutlak yön, konum DEĞİL
        return o

    def grant(self, gain):
        """Bilmece ödülü: çözülen terminalin enerjisi."""
        self.e = min(1.0, self.e + gain)

    def predict_transition(self, a):
        """Salt-okunur bir fizik önizlemesi döndür; gerçek evreni ilerletme.

        ``deepcopy`` üzerindeki adım, ajanın henüz yaşamadığı bir geleceği
        deneyim belleğine veya gerçek metabolizmaya yazmaz. Dönen ``info``
        fiziksel olayları içerebilir; üst katman bunlardan yalnız ajanın
        meşru biçimde bilebileceği mekanik/geometrik büyüklükleri seçmelidir.
        """
        import copy

        sim = copy.deepcopy(self)
        next_obs, info = sim.step(int(a))
        return {
            "obs": next_obs,
            "info": info,
            "position": sim.pos.copy(),
            "velocity": sim.vel.copy(),
            "time": int(sim.t),
        }

    # ---- fizik
    def step(self, a):
        dt, info = self.dt, dict(ate=0.0, drank=0.0, hurt=0.0, died=None, ext=False, push=False, puzzle=0.0)
        if self.dead_forever:
            self.vel[:] = 0.0
            return self.observe(), info
        if self.invert and a in (2, 3):
            a = 5 - a
        f = np.array([np.cos(self.th), np.sin(self.th)])
        thrust = 1.0 if a == 1 else (-0.5 if a == 4 else 0.0)
        if a == 2:
            self.th += self.turn
        elif a == 3:
            self.th -= self.turn
        acc = f * thrust * self.A
        if self.wind_on:
            if self.wind_left == 0 and self.rng.random() < 0.004:
                ang = self.rng.uniform(0, 2 * np.pi)
                self.wind_f, self.wind_left = 6.0 * np.array([np.cos(ang), np.sin(ang)]), 3
            if self.wind_left > 0:
                acc = acc + self.wind_f
                self.wind_left -= 1
                info["ext"] = True
        self.vel = (self.vel + acc * dt) * np.exp(-self.fric * dt)
        self.pos = self.pos + self.vel * dt
        self.rock = self.rock + self.rock_v * dt
        self.rock_v = self.rock_v * np.exp(-self.rock_fric * dt)
        self.touch = 0.0
        for pos, vel, r in ((self.pos, self.vel, self.ra), (self.rock, self.rock_v, self.rr)):
            for k in range(2):
                if pos[k] < r:
                    pos[k], vel[k] = r, abs(vel[k]) * self.rest
                    self.touch = 1.0 if vel is self.vel else self.touch
                elif pos[k] > self.L - r:
                    pos[k], vel[k] = self.L - r, -abs(vel[k]) * self.rest
                    self.touch = 1.0 if vel is self.vel else self.touch
        dvec = self.rock - self.pos
        dist = np.hypot(*dvec)
        if dist < self.ra + self.rr:                     # ajan-kaya çarpışması (momentum korunur)
            n = dvec / max(dist, 1e-6)
            self.rock = self.rock + n * (self.ra + self.rr - dist) * (self.m_agent / (self.m_agent + self.m_rock))
            self.pos = self.pos - n * (self.ra + self.rr - dist) * (self.m_rock / (self.m_agent + self.m_rock))
            vn = (self.rock_v - self.vel) @ n
            if vn < 0:
                j = -(1 + self.rest) * vn / (1 / self.m_agent + 1 / self.m_rock)
                self.vel = self.vel - j / self.m_agent * n
                self.rock_v = self.rock_v + j / self.m_rock * n
            self.touch, info["push"] = 1.0, True
        # metabolizma
        night = 1.25 - 0.25 * self.light()
        self.e -= (self.c_base + self.c_move * thrust * thrust + 0.0002 * (a in (2, 3))) * night * self.metab
        self.w -= 0.0010 * self.metab
        d_food = np.hypot(*(self.food - self.pos).T)
        for k in np.flatnonzero(d_food < self.ra + self.food_r):
            if self.is_term[k]:                              # terminal: ödül çözüm sonrası grant() ile verilir
                info["puzzle"] = min(self.food_gain, 1.0 - self.e)
                self.food[k] = self._spawn_food()
                continue
            gain = -0.35 if self.poison else min(self.food_gain, 1.0 - self.e)
            self.e, info["ate"] = self.e + gain, info["ate"] + max(gain, 0.0)
            self.food[k] = self._spawn_food()
        if np.any(np.hypot(*(self.water_c - self.pos).T) < self.zone_w):
            gain = min(0.06, 1.0 - self.w)
            self.w, info["drank"] = self.w + gain, gain
            if self.poison:                                  # zehirli evrende hayatta kalmanın yolu: su bölgesi az da olsa enerji verir
                self.e = min(1.0, self.e + 0.03)
        if np.any(np.hypot(*(self.haz_c - self.pos).T) < self.zone_h):
            self.i -= 0.05
            info["hurt"] = 0.05
        elif self.e > 0.4 and self.w > 0.4:
            self.i = min(1.0, self.i + 0.0015)
        self.t += 1
        for name, v in (("açlık", self.e), ("susuzluk", self.w), ("yara", self.i)):
            if v <= 0:
                info["died"] = name
        if info["died"]:
            if self.mortal:
                self.e, self.w, self.i = max(0.0, self.e), max(0.0, self.w), max(0.0, self.i)
                self.vel[:] = 0.0
                self.dead_forever = True
            else:
                self._respawn()
        return self.observe(), info


# ---------------------------------------------------------------- öz-sınama
def stage_test():
    u = Universe(seed=1, wind=False)
    # 1) eylemsizlik + sürtünme: itiş bırakılınca hız üstel sönmeli; terminal hız ayrık-zaman sabit noktası
    u.pos, u.vel, u.th = np.array([3.0, 12.0]), np.zeros(2), 0.0
    for _ in range(25):
        u.step(1)
    q = np.exp(-u.fric * u.dt)
    v_star = u.A * u.dt * q / (1 - q)
    v0 = np.hypot(*u.vel)
    assert abs(v0 - v_star) < 0.02 and u.pos[0] < u.L - 1, (v0, v_star, u.pos)
    for _ in range(8):
        u.step(0)
    v1 = np.hypot(*u.vel)
    assert abs(v1 / v0 - q ** 8) < 0.03, (v1, v0)
    print("  terminal hız %.3f (ayrık kuram %.3f); itiş bitince v %.3f -> %.3f (oran %.3f, beklenen %.3f)"
          % (v0, v_star, v0, v1, v1 / v0, q ** 8))
    # 2) momentum: ajan durağan kayaya çarpar (sürtünme kapalı)
    u.fric = u.rock_fric = 0.0
    u.pos, u.vel, u.th = np.array([6.0, 12.0]), np.array([2.0, 0.0]), 0.0
    u.rock, u.rock_v = np.array([7.2, 12.0]), np.zeros(2)
    p0 = u.m_agent * u.vel + u.m_rock * u.rock_v
    u.step(0)
    p1 = u.m_agent * u.vel + u.m_rock * u.rock_v
    print("  çarpışma momentumu: önce %s sonra %s; kaya hızı %.3f" % (np.round(p0, 3), np.round(p1, 3), u.rock_v[0]))
    assert np.allclose(p0, p1, atol=1e-6) and u.rock_v[0] > 0
    # 3) görme: tam önde yiyecek, 5 birim
    u = Universe(seed=2, wind=False)
    u.pos, u.th = np.array([8.0, 8.0]), 0.0
    u.food[:] = 100.0
    u.food[0] = [13.0, 8.0]
    u.haz_c[:] = [[40.0, 40.0], [40.0, 41.0]]
    u.water_c[:] = [[40.0, 20.0], [41.0, 20.0]]
    u.t = 0                                          # öğle: görüş 10
    rays, vis = u._rays()
    mid = Universe.R // 2
    exp = 1 - (5.0 - u.food_r - u.beam) / vis   # kalın ışın: etkin yarıçap r+beam
    assert abs(rays[mid, 1] - exp) < 1e-4 and rays[0, 1] == 0, (rays[mid, 1], exp)
    print("  görme: orta ışın yiyecek yakınlığı %.3f (beklenen %.3f), uç ışın 0" % (rays[mid, 1], exp))
    # 4) gece daha kısa görüş
    u.t = u.day_len // 2
    _, vis_n = u._rays()
    assert vis_n < vis
    print("  görüş menzili gündüz %.1f gece %.1f" % (vis, vis_n))
    # 5) rastgele politikada bir süre hayatta kal / metabolizma
    u = Universe(seed=3)
    rng, deaths, ate = np.random.default_rng(0), 0, 0.0
    for _ in range(5000):
        _, info = u.step(rng.integers(5))
        deaths += info["died"] is not None
        ate += info["ate"]
    print("  5000 rastgele adım: %d ölüm, %.1f yiyecek" % (deaths, ate))
    print("[world] TAMAM")


if __name__ == "__main__":
    stage_test()
