#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_chem.py — NoRa Endojen Nörokimya ve Üç-Faktörlü Sinaptik Plastisite Temeli.

Beyin, milyonlarca veri üzerinde çoklu-epoch geri-yayılım (brute-force SGD) çalıştıran
statik bir matris çarpıcı değildir; kimyasal modülatörlerin anlık olarak sinaptik kapıları
açıp kapattığı canlı bir elektro-kimyasal süreçtir:

  - Dopamin (DA)      : Ödül/değer öngörü hatası (RPE) ve güdüsel belirginlik. Aktif uygunluk
                        izlerini (eligibility traces) kalıcı sinaps ağırlığına damgalar.
  - Asetilkolin (ACh) : Beklenen belirsizlik ve yenilik kapısı (Hasselmo 2006). Yüksek ACh,
                        kortikal/hipokampal devreyi TEK-ATIŞ (1-shot, 0-epoch) hızlı kodlama
                        kipine sokar; eski önbilgilerin yeni öğrenmeyi ezmesini önler.
  - Noradrenalin (NE) : Beklenmedik şok, kural çelişkisi ve içsel sürpriz (Locus Coeruleus).
                        Bilişsel sıfırlama, Sistem-2 derin düşünme ve yapısal nörogenez tetikler.
  - Serotonin (5-HT)  : Sabır, zaman ufku (iskonto faktörü gamma) ve dürtü ketleme.
  - Kortizol (CORT)   : Kronik bedensel tehdit / allostatik yük; varoluşsal öz-korumayı öne alır.
  - Oksitosin (OXY)   : Sosyal/epistemik güven; dürüst öğreticilerle yükselir, yalan/zararlı
                        yönlendirmede hızla düşerek epistemik bağışıklık sağlar.

Bileşenler:
  NeuroChemistry       6 endojen nöromodülatörün homeostatik ve bilişsel dinamikleri
  ChemMushroomBody     Sinek mantar cismi + Tamamlayıcı Öğrenme Sistemleri (CLS):
                       - Seyrek Kenyon hücreleri (k-WTA, ~%3.5 aktif -> sıfır girişim)
                       - Hızlı hipokampal sinapslar (W_fast: ACh x DA kapılı 1-shot öğrenme)
                       - Yavaş neokortikal sinapslar (W_slow: uyku tekrarında sinaptik
                         etiketleme-yakalama ve metaplofastik sertlik S ile yıkıcı unutma koruması)
                       - Yapısal nörogenez (kapasite dolduğunda yeni KC grupları büyütme)
"""
from __future__ import annotations

import numpy as np


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


class NeuroChemistry:
    """Canlı bedensel durum, sürpriz ve sosyal etkileşimden beslenen endojen kimya sistemi."""

    MOD_NAMES = ("DA", "ACh", "NE", "5HT", "CORT", "OXY")

    def __init__(self):
        self.da_tonic = 0.50       # bazal dopamin (merak / hareket canlılığı)
        self.da_phasic = 0.00      # anlık RPE patlaması [-1, +1]
        self.ach = 0.55            # dikkat ve tek-atış kodlama açıklığı [0, 1]
        self.ne = 0.15             # beklenmedik şok / uyanıklık [0, 1]
        self.sht = 0.65            # serotonin: sabır ve dürtü dengesi [0, 1]
        self.cort = 0.10           # kortizol: allostatik stres yükü [0, 1]
        self.oxy = 0.60            # genel sosyal güven tabanı [0, 1]
        self.speaker_trust: dict[str, float] = {}   # konuşmacı bazlı epistemik güven
        self.val_ema = 0.0         # beklenen değer izi (RPE için)
        self.step_count = 0
        self.history: list[np.ndarray] = []

    def trust_of(self, speaker: str = "insan") -> float:
        return float(self.speaker_trust.get(speaker, 0.60))

    def update_social(self, speaker: str, verified_helpful: float):
        """verified_helpful in [-1, +1]: +1 doğrulanan faydalı bilgi, -1 yalan/zararlı yönlendirme."""
        cur = self.trust_of(speaker)
        # İhanet/yalan güveni hızlı düşürür (asimetrik biyolojik ihtiyat), doğruluk kademeli artırır
        rate = 0.30 if verified_helpful > 0 else 0.50
        target = 1.0 if verified_helpful > 0 else 0.0
        mag = min(1.0, abs(float(verified_helpful)))
        new_t = _clip01(cur + rate * mag * (target - cur))
        self.speaker_trust[speaker] = new_t
        mean_spk = float(np.mean(list(self.speaker_trust.values())))
        self.oxy = _clip01(0.30 * 0.60 + 0.70 * mean_spk)
        if verified_helpful < -0.3:
            self.ne = _clip01(self.ne + 0.35 * mag)
            self.cort = _clip01(self.cort + 0.15 * mag)
            self.da_phasic = float(np.clip(self.da_phasic - 0.5 * mag, -1.0, 1.0))
        elif verified_helpful > 0.3:
            self.da_phasic = float(np.clip(self.da_phasic + 0.35 * mag, -1.0, 1.0))

    def step(self, energy: float = 1.0, water: float = 1.0, integrity: float = 1.0,
             reward: float = 0.0, novelty: float = 0.2, surprise: float = 0.0,
             vis_prec: float = 1.0, motor_eff: float = 1.0, died: bool = False) -> dict[str, float]:
        """Her yaşam veya diyalog adımında nörokimyasal dengeyi günceller."""
        self.step_count += 1
        wellbeing = float(min(energy, water, integrity))
        deficit = 1.0 - wellbeing

        # 1) Dopamin: fazik RPE + tonik canlılık
        rpe = float(reward) - self.val_ema
        self.val_ema = 0.92 * self.val_ema + 0.08 * float(reward)
        if died:
            rpe = -1.0
        self.da_phasic = float(np.clip(0.35 * self.da_phasic + 0.85 * np.tanh(1.8 * rpe) + 0.18 * (novelty - 0.3), -1.0, 1.0))
        self.da_tonic = _clip01(0.94 * self.da_tonic + 0.06 * (0.45 + 0.35 * wellbeing + 0.20 * novelty - 0.30 * self.cort))

        # 2) Asetilkolin: yenilik ve öğrenme açıklığı (tek-atış sinaptik kapı)
        ach_drive = 0.25 + 0.65 * _clip01(novelty) + 0.20 * _clip01(surprise / 3.0)
        self.ach = _clip01(0.60 * self.ach + 0.40 * ach_drive)

        # 3) Noradrenalin: beklenmedik şok, duyusal/motor kayıp veya ölüm
        sensory_motor_loss = max(0.0, 1.0 - min(vis_prec, motor_eff))
        ne_drive = _clip01(0.45 * (surprise / 2.5) + 0.45 * sensory_motor_loss + (0.8 if died else 0.0))
        self.ne = _clip01(0.72 * self.ne + 0.28 * ne_drive)

        # 4) Kortizol: kronik eksiklik, yara ve kontrol kaybı birikimi (yavaş yükselir, yavaş iner)
        stress_in = _clip01(0.55 * (deficit ** 1.5) + 0.35 * sensory_motor_loss + (0.65 if died else 0.0))
        self.cort = _clip01(0.96 * self.cort + 0.04 * stress_in + (0.18 if died else 0.0))

        # 5) Serotonin: güvenlik, bütünlük ve düşük stres altında yükselir; sabrı belirler
        sht_target = _clip01(0.75 * wellbeing + 0.25 * (1.0 - self.cort) - 0.20 * self.ne)
        self.sht = _clip01(0.93 * self.sht + 0.07 * sht_target)

        snap = self.snapshot()
        if len(self.history) >= 512:
            self.history.pop(0)
        self.history.append(self.vector())
        return snap

    def pulse_learning(self, novelty: float = 0.85, valence: float = 0.4):
        """Yeni bir cümle, kavram veya kural öğretildiğinde ACh + DA öğrenme darbesi üretir."""
        self.ach = _clip01(max(self.ach, 0.45 + 0.55 * _clip01(novelty)))
        self.da_phasic = float(np.clip(0.4 * self.da_phasic + 0.6 * valence, -1.0, 1.0))

    @property
    def fast_lr_gate(self) -> float:
        """Tek-atış (1-shot) hipokampal yazma katsayısı: ACh yüksekken tek örnekte tam bağ kurar."""
        return float(np.clip(self.ach * (0.75 + 0.65 * abs(self.da_phasic) + 0.25 * self.da_tonic), 0.15, 1.5))

    @property
    def discount_gamma(self) -> float:
        """Serotonin tarafından ayarlanan gelecek ufku (sabır)."""
        return float(np.clip(0.82 + 0.16 * self.sht - 0.06 * self.cort, 0.78, 0.985))

    @property
    def system2_drive(self) -> float:
        """Sistem-2 (derin düşünme / hayal etme) eğilimi: NE + ACh yüksek, dürtüsel panik düşükken artar."""
        return _clip01(0.45 * self.ne + 0.40 * self.ach + 0.15 * self.sht)

    def vector(self) -> np.ndarray:
        return np.array([self.da_tonic, self.da_phasic, self.ach, self.ne, self.sht, self.cort, self.oxy], dtype=np.float32)

    def snapshot(self) -> dict[str, float]:
        return {
            "DA_tonik": round(self.da_tonic, 3),
            "DA_fazik": round(self.da_phasic, 3),
            "ACh": round(self.ach, 3),
            "NE": round(self.ne, 3),
            "5HT": round(self.sht, 3),
            "CORT": round(self.cort, 3),
            "OXY": round(self.oxy, 3),
        }

    def dominant_state_tr(self) -> str:
        """Nörokimyasal durumu Türkçe içsel duygu/kip olarak özetler."""
        if self.cort > 0.55:
            return "stresli ve öz-koruma odaklı"
        if self.ne > 0.50:
            return "teyakkuzda ve dikkatli"
        if self.da_phasic > 0.35 and self.ach > 0.60:
            return "meraklı ve hızlı öğrenmeye açık"
        if self.sht > 0.65 and self.cort < 0.25:
            return "dengeli, sakin ve sabırlı"
        if self.da_phasic < -0.35:
            return "hayal kırıklığı / hata düzeltme kipinde"
        return "uyanık ve dengeli"

    def state_dict(self) -> dict:
        return {
            "vals": [self.da_tonic, self.da_phasic, self.ach, self.ne, self.sht, self.cort, self.oxy, self.val_ema, self.step_count],
            "speaker_trust": dict(self.speaker_trust),
        }

    def load_state_dict(self, d: dict):
        v = d["vals"]
        self.da_tonic, self.da_phasic, self.ach, self.ne, self.sht, self.cort, self.oxy, self.val_ema = [float(x) for x in v[:8]]
        self.step_count = int(v[8])
        self.speaker_trust = {str(k): float(val) for k, val in d.get("speaker_trust", {}).items()}


class ChemMushroomBody:
    """Nörokimyasal Tamamlayıcı Öğrenme Sistemi (Sinek Mantar Cismi + Sinaptik Konsolidasyon + Nörogenez).

    Neden epoch gerekmez ve neden yıkıcı unutma olmaz?
      1. Seyrek Ortogonal Kod (APL k-WTA, ~%3 aktif KC): Her kavram yalnızca ~60-90 nöronu ateşler;
         farklı kavramların aktif KC kesişimi ~%0.1'dir -> yeni bilgi eski sinapslara dokunmaz.
      2. Hızlı Sinaps (W_fast, Hipokampus / α'β' lobu): ACh kapısıyla TEK ÖRNEKTE (1-shot) tam güçte
         yazılır; uygunluk izi (elig) ve sinaptik etiket (tag) bırakır.
      3. Yavaş Sinaps (W_slow, Neokorteks / αβ lobu): Uyku konsolidasyonunda (sleep_consolidate)
         etiketli sinapslar W_slow'a aktarılır ve sinaptik sertlik (stiffness S) kazanır;
         S yüksek olan konsolide sinapslar yeni girdilerle silinmez!
      4. Yapısal Nörogenez (grow_neurogenesis): Kapasite yoğunlaştığında mevcut sinapsları bozmadan
         yeni Kenyon hücre sütunları ekler (GrowingNet ilkesi).
    """

    def __init__(self, n_in: int, n_out: int = 16, n_kc: int = 2048, claws: int = 7,
                 sparsity: float = 0.035, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.n_kc = int(n_kc)
        self.claws = min(int(claws), self.n_in)
        self.sparsity = float(sparsity)
        self.k = max(4, int(round(self.sparsity * self.n_kc)))

        self.claw_idx = self._sample_claws(self.n_kc)              # (n_kc, claws)
        self.claw_sign = self.rng.choice(np.array([-1.0, 1.0], np.float32), size=(self.n_kc, self.claws))
        self.W_fast = np.zeros((self.n_out, self.n_kc), np.float32)
        self.W_slow = np.zeros((self.n_out, self.n_kc), np.float32)
        self.stiff = np.zeros((self.n_out, self.n_kc), np.float32)  # konsolide sinaps koruması [0, 1]
        self.tags = np.zeros((self.n_out, self.n_kc), np.float32)   # sinaptik etiketleme (uyku için)
        self.elig = np.zeros(self.n_kc, np.float32)                 # uygunluk izi
        self.w_nov = np.ones(self.n_kc, np.float32)                 # yenilik (α'3 MBON)
        self.kc_usage = np.zeros(self.n_kc, np.float32)             # kapasite izleme
        self.episodic_buffer: list[tuple[np.ndarray, int, float]] = []
        self.growth_events = 0

    def _sample_claws(self, count: int) -> np.ndarray:
        return np.stack([self.rng.choice(self.n_in, self.claws, replace=False) for _ in range(count)]).astype(np.int32)

    def ensure_outputs(self, min_out: int):
        """Yeni sınıf/kavram eklendiğinde çıkış bölmelerini mevcut sinapsları bozmadan genişletir."""
        if min_out <= self.n_out:
            return
        extra = int(min_out - self.n_out)
        z = lambda: np.zeros((extra, self.n_kc), np.float32)
        self.W_fast = np.vstack([self.W_fast, z()])
        self.W_slow = np.vstack([self.W_slow, z()])
        self.stiff = np.vstack([self.stiff, z()])
        self.tags = np.vstack([self.tags, z()])
        self.n_out = int(min_out)

    def grow_neurogenesis(self, add_kc: int = 512):
        """Yapısal nörogenez: mevcut tüm sinapsları koruyarak yeni Kenyon hücreleri ekler."""
        add_kc = int(add_kc)
        new_claws = self._sample_claws(add_kc)
        new_signs = self.rng.choice(np.array([-1.0, 1.0], np.float32), size=(add_kc, self.claws))
        self.claw_idx = np.vstack([self.claw_idx, new_claws])
        self.claw_sign = np.vstack([self.claw_sign, new_signs])
        pad = lambda: np.zeros((self.n_out, add_kc), np.float32)
        self.W_fast = np.hstack([self.W_fast, pad()])
        self.W_slow = np.hstack([self.W_slow, pad()])
        self.stiff = np.hstack([self.stiff, pad()])
        self.tags = np.hstack([self.tags, pad()])
        self.elig = np.concatenate([self.elig, np.zeros(add_kc, np.float32)])
        self.w_nov = np.concatenate([self.w_nov, np.ones(add_kc, np.float32)])
        self.kc_usage = np.concatenate([self.kc_usage, np.zeros(add_kc, np.float32)])
        self.n_kc += add_kc
        self.k = max(4, int(round(self.sparsity * self.n_kc)))
        self.growth_events += 1

    def encode_kc(self, x: np.ndarray) -> np.ndarray:
        """Girdi vektörü -> seyrek aktif KC indeksleri (boyut: k)."""
        v = np.asarray(x, np.float32).ravel()
        if len(v) < self.n_in:
            v = np.pad(v, (0, self.n_in - len(v)))
        elif len(v) > self.n_in:
            v = v[:self.n_in]
        norm = float(np.linalg.norm(v))
        if norm < 1e-8:
            return np.zeros(0, dtype=np.int64)
        vn = (v - v.mean()) / (norm + 1e-6)
        drive = (vn[self.claw_idx] * self.claw_sign).sum(axis=1)
        top = np.argpartition(-drive, self.k - 1)[:self.k]
        top = top[np.argsort(-drive[top])]
        return top.astype(np.int64)

    def novelty(self, x_or_idx: np.ndarray, update: bool = False) -> float:
        idx = x_or_idx if (x_or_idx.ndim == 1 and np.issubdtype(x_or_idx.dtype, np.integer)) else self.encode_kc(x_or_idx)
        if len(idx) == 0:
            return 0.0
        nov = float(self.w_nov[idx].mean())
        if update:
            self.w_nov[idx] *= 0.65
            self.w_nov = np.minimum(1.0, self.w_nov + 0.0005)
        return nov

    def scores(self, x_or_idx: np.ndarray) -> np.ndarray:
        idx = x_or_idx if (x_or_idx.ndim == 1 and np.issubdtype(x_or_idx.dtype, np.integer)) else self.encode_kc(x_or_idx)
        if len(idx) == 0:
            return np.zeros(self.n_out, np.float32)
        W_eff = self.W_slow + self.W_fast
        return W_eff[:, idx].sum(axis=1) / max(1.0, float(len(idx)))

    def predict(self, x: np.ndarray) -> tuple[int, float, float]:
        """(en_iyi_bölme, güven_marjı, yenilik) döndürür."""
        idx = self.encode_kc(x)
        sc = self.scores(idx)
        if len(sc) == 0 or float(np.max(np.abs(sc))) < 1e-6:
            return -1, 0.0, self.novelty(idx, update=False)
        best = int(np.argmax(sc))
        if len(sc) > 1:
            part = np.partition(sc, -2)
            margin = float(part[-1] - part[-2])
        else:
            margin = float(sc[0])
        return best, margin, self.novelty(idx, update=False)

    def learn_oneshot(self, x: np.ndarray, target: int, chem: NeuroChemistry | None = None,
                      strength: float = 1.0, contrastive: bool = True) -> float:
        """Üç-faktörlü (KC x hedef_bölme x [ACh, DA]) TEK-ATIŞ (0-epoch) sinaptik bağlama!"""
        self.ensure_outputs(int(target) + 1)
        idx = self.encode_kc(x)
        if len(idx) == 0:
            return 0.0
        nov = self.novelty(idx, update=True)
        if chem is not None:
            chem.pulse_learning(novelty=max(0.4, nov), valence=0.35 * np.sign(strength))
            gate = chem.fast_lr_gate * float(strength)
        else:
            gate = (0.75 + 0.5 * nov) * float(strength)

        # Kapasite doygunluğu kontrolü -> gerekirse nörogenez
        self.kc_usage[idx] = np.minimum(1.0, self.kc_usage[idx] + 0.25)
        if float((self.kc_usage > 0.75).mean()) > 0.65 and self.n_kc < 8192:
            self.grow_neurogenesis(add_kc=512)
            idx = self.encode_kc(x)

        # Yalnızca aktif KC sinapslarında yerel hata-güdümlü 3-faktörlü güncelleme (stiffness korumalı!)
        sc = self.scores(idx)
        plasticity = 1.0 - 0.85 * self.stiff[target, idx]
        cur_target = float(sc[target])
        want = 1.0 if strength >= 0 else -0.6
        delta_pos = max(0.15, want - cur_target) if strength >= 0 else (want - cur_target)
        step_w = gate * delta_pos * plasticity
        self.W_fast[target, idx] = np.clip(self.W_fast[target, idx] + step_w, -2.5, 2.5)
        self.tags[target, idx] = np.maximum(self.tags[target, idx], np.abs(step_w))

        # Yan-baskılama: yanlış ateşleyen rakip bölme varsa yalnızca konsolide OLMAYAN hızlı sinapsı zayıflat
        if contrastive and self.n_out > 1 and strength > 0:
            sc_other = sc.copy()
            sc_other[target] = -1e9
            rival = int(np.argmax(sc_other))
            if sc_other[rival] > 0.20:
                riv_plast = 1.0 - 0.92 * self.stiff[rival, idx]
                self.W_fast[rival, idx] -= 0.45 * gate * max(0.0, sc_other[rival] - 0.1) * riv_plast

        self.elig *= 0.75
        self.elig[idx] = 1.0
        if len(self.episodic_buffer) >= 512:
            self.episodic_buffer.pop(0)
        self.episodic_buffer.append((idx.copy(), int(target), float(strength)))
        return nov

    def sleep_consolidate(self, replay_steps: int = 32) -> dict[str, float]:
        """Keskin-dalga dalgacığı (SWR) uykusu:
        1) Epizodik tamponu yeniden oynatarak kenar ayrımlarını keskinleştirir.
        2) Sinaptik etiketli (tags) hızlı hipokampal ağırlıkları yavaş neokortikal W_slow'a aktarır.
        3) Konsolide sinapsların sertliğini (stiff) artırarak gelecekteki yıkıcı unutmayı kilitler."""
        replayed = 0
        if self.episodic_buffer:
            n_rep = min(int(replay_steps), len(self.episodic_buffer))
            order = self.rng.choice(len(self.episodic_buffer), size=n_rep, replace=False)
            for bi in order:
                idx, tgt, strg = self.episodic_buffer[int(bi)]
                if tgt >= self.n_out or len(idx) == 0:
                    continue
                # Nörogenez sonrası eski indeksler her zaman geçerlidir (append-only KC büyümesi!)
                sc = self.scores(idx)
                err = max(0.0, 1.05 - float(sc[tgt])) if strg > 0 else 0.0
                if err > 0.02:
                    self.W_fast[tgt, idx] += 0.35 * err
                    self.tags[tgt, idx] = np.maximum(self.tags[tgt, idx], 0.35 * err)
                replayed += 1

        # Sinaptik yakalama (Synaptic Tagging & Capture): W_fast -> W_slow
        tagged = self.tags > 0.05
        n_cons = int(tagged.sum())
        if n_cons > 0:
            transfer = 0.65 * self.W_fast[tagged]
            self.W_slow[tagged] = np.clip(self.W_slow[tagged] + transfer, -3.5, 3.5)
            self.W_fast[tagged] *= 0.35
            self.stiff[tagged] = np.clip(self.stiff[tagged] + 0.35 * np.tanh(np.abs(self.W_slow[tagged])), 0.0, 0.98)
            self.tags[tagged] *= 0.2
        # Etiketlenmemiş geçici gürültü sönümlenir (sinaptik homeostaz, Tononi & Cirelli SHY)
        self.W_fast[~tagged] *= 0.95
        return {"replayed": float(replayed), "consolidated_synapses": float(n_cons), "n_kc": float(self.n_kc)}


if __name__ == "__main__":
    chem = NeuroChemistry()
    mb = ChemMushroomBody(n_in=32, n_out=5, n_kc=1500, seed=42)
    rng = np.random.default_rng(7)
    protos = [rng.standard_normal(32).astype(np.float32) for _ in range(5)]
    # 5 görevi SIRA İLE ve YALNIZCA 1 KEZ (0 epoch!) gösterelim:
    for cls, p in enumerate(protos):
        mb.learn_oneshot(p, cls, chem=chem)
    mb.sleep_consolidate()
    hits = sum(mb.predict(p + 0.18 * rng.standard_normal(32))[0] == cls for cls, p in enumerate(protos))
    print(f"[chem-test] 1-atış 5 ardışık kavram (0 epoch, gürültülü test): {hits}/5 doğru | kimya: {chem.snapshot()}")
    assert hits == 5
