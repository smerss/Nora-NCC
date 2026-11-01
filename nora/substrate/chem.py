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
  ChemMushroomBody     Biyolojik esinli mantar cismi + Tamamlayıcı Öğrenme Sistemleri (CLS):
                       - Seyrek Kenyon hücreleri (k-WTA; düşük girişimli kod)
                       - Hızlı hipokampal sinapslar (W_fast: ACh x DA kapılı 1-shot öğrenme)
                       - Yavaş neokortikal sinapslar (W_slow: replay'siz uyku bakımında
                         etiketleme-yakalama ve metaplofastik sertlik S ile korunum)
                       - Yapısal nörogenez (kapasite dolduğunda yeni KC grupları büyütme)
  HDCMushroomBody      Rol–dolgu HDC/VSA bağlama, %6 k-WTA ve MB karar okuması
"""
from __future__ import annotations

from nora._paths import REPO_ROOT


import hashlib
import os
import sqlite3
import tempfile
from collections import OrderedDict
from pathlib import Path

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
        # Konuşmacı bazlı epistemik güven. Anahtar SES KİMLİĞİdir; dış dünyanın etiketi
        # (speaker="emre" gibi) yalnızca takma addır ve `alias_resolver` ile çözülür.
        self.speaker_trust: dict[str, float] = {}
        self.alias_resolver = None   # callable(etiket) -> ses kimliği (NoraAGI bağlar)
        self.val_ema = 0.0         # beklenen değer izi (RPE için)
        self.step_count = 0
        self.history: list[np.ndarray] = []

    def _canon(self, speaker: str) -> str:
        """Etiketi ses kimliğine çevirir. Çözücü yoksa anahtar olduğu gibi kalır."""
        r = self.alias_resolver
        if r is not None:
            try:
                c = r(str(speaker))
                if c:
                    return str(c)
            except Exception:
                pass
        return str(speaker)

    def trust_of(self, speaker: str = "insan") -> float:
        return float(self.speaker_trust.get(self._canon(speaker), 0.60))

    def update_social(self, speaker: str, verified_helpful: float):
        """verified_helpful in [-1, +1]: +1 doğrulanan faydalı bilgi, -1 yalan/zararlı yönlendirme."""
        speaker = self._canon(speaker)
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

    def intero_word_tr(self) -> str:
        """Bedensel duyumun TEK sözcüklük adı (insula okuması): kimyadan türetilir, cümle değildir."""
        if self.cort > 0.55:
            return "gerginlik"
        if self.ne > 0.50:
            return "tedirginlik"
        if self.da_phasic > 0.35 and self.ach > 0.60:
            return "merak"
        if self.sht > 0.65 and self.cort < 0.25:
            return "huzur"
        if self.da_phasic < -0.35:
            return "burukluk"
        return "sakinlik"

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
    """Nörokimyasal Tamamlayıcı Öğrenme Sistemi (Mantar Cismi + Sinaptik Bakım + Nörogenez).

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
        self.last_learning: dict = {}

    def _sample_claws(self, count: int) -> np.ndarray:
        return np.stack([self.rng.choice(self.n_in, self.claws, replace=False) for _ in range(count)]).astype(np.int32)

    def ensure_outputs(self, min_out: int):
        """Yeni sınıf/kavram eklendiğinde çıkış bölmelerini mevcut sinapsları bozmadan genişletir."""
        if min_out <= self.n_out:
            return
        # Amortize reallocation: grow by at least 256 or 50% to prevent frequent vstack overhead
        target_out = max(min_out, self.n_out + 256, int(self.n_out * 1.5))
        extra = int(target_out - self.n_out)
        z = lambda: np.zeros((extra, self.n_kc), np.float32)
        self.W_fast = np.vstack([self.W_fast, z()])
        self.W_slow = np.vstack([self.W_slow, z()])
        self.stiff = np.vstack([self.stiff, z()])
        self.tags = np.vstack([self.tags, z()])
        self.n_out = int(target_out)

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
        # Yalnız aktif KC sütunlarını birleştir; bütün W_eff matrisini her çağrıda
        # kopyalamak hem gereksiz bellek hem de tek-atış çıkarımında darboğazdır.
        return (self.W_slow[:, idx] + self.W_fast[:, idx]).sum(axis=1) / max(1.0, float(len(idx)))

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

        # Yalnızca aktif KC sinapslarında yerel hata-güdümlü 3-faktörlü güncelleme.
        # Contrastive istenmediğinde tüm matrisi toplamak yerine yalnız hedef satırı O(K) hesapla.
        if contrastive and self.n_out > 1 and strength > 0:
            sc = self.scores(idx)
            cur_target = float(sc[target])
        else:
            sc = None
            cur_target = float((self.W_slow[target, idx] + self.W_fast[target, idx]).sum()) / max(1.0, float(len(idx)))

        want = 1.0 if strength >= 0 else -0.6
        # Zıt işaretli ve sertleşmiş bir bağla çakışma, eski sinapsı ezmek için
        # gerekçe değildir: önce yeni KC kapasitesi açılır ve örnek yeni kodla yazılır.
        conflict = bool(want * cur_target < -0.04
                        and float(np.mean(self.stiff[target, idx])) > 0.45)
        grew_for_conflict = False
        if conflict and self.n_kc < 8192:
            self.grow_neurogenesis(add_kc=512)
            idx = self.encode_kc(x)
            if sc is not None:
                sc = self.scores(idx)
                cur_target = float(sc[target])
            else:
                cur_target = float((self.W_slow[target, idx] + self.W_fast[target, idx]).sum()) / max(1.0, float(len(idx)))
            grew_for_conflict = True

        plasticity = 1.0 - 0.85 * self.stiff[target, idx]
        delta_pos = max(0.15, want - cur_target) if strength >= 0 else (want - cur_target)
        step_w = gate * delta_pos * plasticity
        self.W_fast[target, idx] = np.clip(self.W_fast[target, idx] + step_w, -2.5, 2.5)
        self.tags[target, idx] = np.maximum(self.tags[target, idx], np.abs(step_w))
        # Uyanık öğrenilen bilgi hemen korunur; stiffness için uyku beklenmez.
        self.stiff[target, idx] = np.maximum(
            self.stiff[target, idx],
            np.clip(0.08 + 0.24 * np.tanh(np.abs(self.W_fast[target, idx])), 0.0, 0.32),
        )

        # Yarışma skoru çıkışta okunur; rakip sınıfın eski sinapsını silmek
        # "contrastive learning" değildir. Bu nedenle burada negatif ağırlık
        # güncellemesi yoktur. Çakışma yalnız izlenir ve gerekirse nörogenez olur.
        rival = -1
        if sc is not None and contrastive and self.n_out > 1 and strength > 0:
            sc_other = sc.copy()
            sc_other[target] = -1e9
            rival = int(np.argmax(sc_other))
        self.last_learning = {
            "target": int(target), "active_kc": int(len(idx)),
            "conflict": conflict, "grew_for_conflict": grew_for_conflict,
            "rival_observed_without_weight_erasure": rival,
            "replay": 0, "gradient": 0,
        }

        self.elig *= 0.75
        self.elig[idx] = 1.0
        if len(self.episodic_buffer) >= 512:
            self.episodic_buffer.pop(0)
        self.episodic_buffer.append((idx.copy(), int(target), float(strength)))
        return nov

    def sleep_consolidate(self, replay_steps: int = 0) -> dict[str, float]:
        """Uyku bakımı: örnek replay'i olmadan etiket yakalama ve homeostaz.

        İlk öğrenme uyanıkken ``learn_oneshot`` ile tamamlanır. Uyku artık
        epizodik tampondan gradyan veya ek sunum üretmez; yalnız zaten yazılmış
        sinapsları W_slow'a aktarır, stiffness'i artırır ve etiketsiz geçici
        gürültüyü budar. Pozitif ``replay_steps`` istemek açıkça hata verir;
        böylece eski gizli multi-epoch yolu geri dönemez.
        """
        if int(replay_steps) != 0:
            raise ValueError("uyku replay'i kaldırıldı; replay_steps=0 olmalı")

        # Sinaptik yakalama (Synaptic Tagging & Capture): W_fast -> W_slow.
        # Bu işlem yeni bir örnek sunumu değil, mevcut sinapsın bakımıdır.
        tagged = self.tags > 0.05
        n_cons = int(tagged.sum())
        if n_cons > 0:
            transfer = 0.65 * self.W_fast[tagged]
            self.W_slow[tagged] = np.clip(self.W_slow[tagged] + transfer, -3.5, 3.5)
            self.W_fast[tagged] *= 0.35
            self.stiff[tagged] = np.clip(
                self.stiff[tagged] + 0.35 * np.tanh(np.abs(self.W_slow[tagged])), 0.0, 0.98,
            )
            self.tags[tagged] *= 0.2
        # Etiketlenmemiş geçici gürültü sönümlenir (sinaptik homeostaz).
        self.W_fast[~tagged] *= 0.95
        return {
            "replayed": 0.0, "consolidated_synapses": float(n_cons),
            "n_kc": float(self.n_kc), "mode": "maintenance_only",
        }


class HDCMushroomBody:
    """HDC/VSA rol–dolgu bağlamasıyla çalışan içerik adresli mantar cismi.

    Bir önerme yalnız bir kez dışarıdan sunulur. Olayın özne erişim anahtarı ve
    tam ``özne–ilişki–nesne`` anahtarı, aynı çıkış hücresine iki ayrı seyrek
    sinaptik bağla yazılır; ikinci yazım aynı örneği yeniden göstermek değildir.
    Her önermenin ayrı çıkış bölmesi olduğundan yeni bilgi eski çıkış ağırlıklarını
    ezmez. Okuma yalnız ``W_fast + W_slow`` aktivasyonlarından yapılır; sembolik
    graf bu sınıfın içinde yoktur ve sonradan bağımsız doğrulayıcı olarak kalır.
    """

    def __init__(self, dim: int = 256, n_kc: int = 2048, claws: int = 9,
                 sparsity: float = 0.06, seed: int = 0, working_set_size: int = 64):
        self.dim = int(dim)
        self.seed = int(seed)
        self.initial_n_kc = int(n_kc)
        self.sparsity = float(sparsity)
        self.mb = ChemMushroomBody(
            n_in=self.dim, n_out=1, n_kc=n_kc, claws=claws,
            sparsity=sparsity, seed=seed + 503,
        )
        self.activation_threshold = 0.32
        self.bound_slots: set[int] = set()
        # Her erişim anahtarının yazıldığı andaki aktif KC sayısı. Nörogenez
        # k-WTA koduna yeni hücreler ekler; eski olayı yeni toplam k'ya bölmek
        # aktivasyonu yapay olarak seyreltir. Olay-kendi normu bunu önler.
        self.slot_code_norm: dict[tuple[int, str], float] = {}
        self.presentations = 0
        self.synaptic_writes = 0
        self._symbol_cache: dict[tuple[str, str], np.ndarray] = {}
        self.last_recall: dict = {}
        # Uyanık çalışma kümesi sınırlıdır; ``bound_slots`` bütün olayların
        # defteridir. Pasif ağırlıklar RAM sözlüğünde değil SQLite'tadır.
        # HDC özeti diskteki erişim anahtarını bulur, yalnız eşleşen satır
        # RAM'e taşınır; ardından aktivasyon ve graf doğrulaması yapılır.
        self.working_set_size = max(1, int(working_set_size))
        self.active_slots: set[int] = set()
        self._slot_lru: OrderedDict[int, None] = OrderedDict()
        self.passive_slots: set[int] = set()
        self.evictions = 0
        self.passive_reactivations = 0
        self._db = None
        self._disk_tmp = None
        self.disk_path = ""
        self._init_disk_store()

    def _init_disk_store(self) -> None:
        root = Path(os.environ.get("NORA_HDC_CACHE", ".cache/nora_hdc"))
        root.mkdir(parents=True, exist_ok=True)
        self._disk_tmp = tempfile.TemporaryDirectory(prefix="mb-", dir=root)
        self.disk_path = str(Path(self._disk_tmp.name) / "passive.sqlite3")
        self._db = sqlite3.connect(self.disk_path)
        self._db.execute("PRAGMA journal_mode=DELETE")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute(
            "CREATE TABLE cues (slot INTEGER NOT NULL, kind TEXT NOT NULL, "
            "digest BLOB NOT NULL, PRIMARY KEY(slot, kind))")
        self._db.execute(
            "CREATE TABLE passive (slot INTEGER PRIMARY KEY, fast BLOB NOT NULL, "
            "slow BLOB NOT NULL, nkc INTEGER NOT NULL)")
        self._db.execute("CREATE INDEX cues_lookup ON cues(kind, digest)")
        self._db.commit()

    def _close_disk(self) -> None:
        db, tmp = getattr(self, "_db", None), getattr(self, "_disk_tmp", None)
        self._db = None
        self._disk_tmp = None
        if db is not None:
            try:
                db.close()
            except Exception:
                pass
        if tmp is not None:
            try:
                tmp.cleanup()
            except Exception:
                pass

    def __del__(self):
        try:
            self._close_disk()
        except Exception:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        # İki nesne aynı SQLite dosyasını paylaşmasın: pasif satırlar ve HDC
        # erişim anahtarları snapshot'a alınır, load sırasında yeni depoya yazılır.
        state["_pickle_passive"] = list(self._db.execute(
            "SELECT slot, fast, slow, nkc FROM passive ORDER BY slot")) if self._db else []
        state["_pickle_cues"] = list(self._db.execute(
            "SELECT slot, kind, digest FROM cues ORDER BY slot, kind")) if self._db else []
        state.pop("_db", None)
        state.pop("_disk_tmp", None)
        state["disk_path"] = ""
        return state

    def __setstate__(self, state):
        passive = state.pop("_pickle_passive", [])
        cues = state.pop("_pickle_cues", [])
        self.__dict__.update(state)
        # Çok eski, yalnız aktif W satırları taşıyan kayıtlar için geriye uyum.
        if not hasattr(self, "active_slots"):
            self.active_slots = set(self.bound_slots)
            self._slot_lru = OrderedDict((slot, None) for slot in sorted(self.active_slots))
        if not hasattr(self, "passive_slots"):
            self.passive_slots = set()
        self._init_disk_store()
        self._db.executemany("INSERT OR REPLACE INTO cues(slot, kind, digest) VALUES(?, ?, ?)", cues)
        self._db.executemany(
            "INSERT OR REPLACE INTO passive(slot, fast, slow, nkc) VALUES(?, ?, ?, ?)", passive)
        self._db.commit()

    def reset(self) -> None:
        self._close_disk()
        self.__init__(dim=self.dim, n_kc=self.initial_n_kc, claws=self.mb.claws,
                      sparsity=self.sparsity, seed=self.seed,
                      working_set_size=self.working_set_size)

    @staticmethod
    def _cue_digest(cue: np.ndarray) -> bytes:
        data = np.ascontiguousarray(cue, dtype=np.float32).tobytes()
        return hashlib.blake2b(data, digest_size=16).digest()

    def passive_storage_info(self) -> dict:
        rows = int(self._db.execute("SELECT COUNT(*) FROM passive").fetchone()[0])
        return {"backend": "sqlite", "path": self.disk_path,
                "disk_rows": rows, "ram_passive_weight_rows": 0,
                "file_exists": Path(self.disk_path).is_file()}

    def _symbol(self, namespace: str, text: str) -> np.ndarray:
        key = (str(namespace), str(text))
        cached = self._symbol_cache.get(key)
        if cached is not None:
            return cached
        digest = hashlib.blake2b(
            f"{self.seed}|{key[0]}|{key[1]}".encode("utf-8"), digest_size=16,
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "little"))
        vec = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=self.dim)
        self._symbol_cache[key] = vec
        return vec

    def encode(self, **roles: str) -> np.ndarray:
        """Rol ve dolguyu elementwise-binding ile bağla, katkıları bundle et."""
        terms = []
        for role, filler in sorted(roles.items()):
            if not str(filler).strip():
                continue
            f = self._symbol("filler", str(filler))
            r = self._symbol("role", str(role))
            # Bağlı rol–dolgu ana temsil; zayıf dolgu yolu kısmi ipucuyla
            # içerik adresli erişimi korur. Normalizasyon k-WTA öncesindedir.
            terms.append(r * f + 0.20 * f)
        if not terms:
            return np.zeros(self.dim, dtype=np.float32)
        # Tam önermede çapraz bağlar, yalnız kavram listesi ezberlenmesini
        # engeller: aynı sözcükler başka rol/ilişkide başka KC kodu üretir.
        if {"subject", "relation", "object"} <= set(roles):
            s = self._symbol("filler", str(roles["subject"]))
            r = self._symbol("filler", str(roles["relation"]))
            o = self._symbol("filler", str(roles["object"]))
            terms.extend([s * r, r * o, s * o, s * r * o])
        vec = np.sum(terms, axis=0, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / max(1e-6, norm)

    def set_working_set_size(self, size: int) -> None:
        """Aktif sinaps kümesini daralt/genişlet; pasif olayları diskte koru."""
        self.working_set_size = max(1, int(size))
        while len(self.active_slots) > self.working_set_size:
            oldest = next(iter(self._slot_lru))
            self._deactivate_slot(oldest)
        self._db.commit()

    @staticmethod
    def _restore_row(saved: np.ndarray, target: np.ndarray) -> None:
        # Nörogenez pasif olayın KC alanını da büyütebilir. Eski sinapslar
        # aynen korunur; yeni hücreler o olay için henüz bağlanmamıştır.
        if len(saved) < len(target):
            padded = np.zeros_like(target)
            padded[:len(saved)] = saved
            saved = padded
        target[:] = saved[:len(target)]

    def _deactivate_slot(self, slot: int) -> None:
        slot = int(slot)
        if slot not in self.active_slots:
            return
        if slot < len(self.mb.W_fast):
            fast = self.mb.W_fast[slot].copy()
            slow = self.mb.W_slow[slot].copy()
            self._db.execute(
                "INSERT OR REPLACE INTO passive(slot, fast, slow, nkc) VALUES(?, ?, ?, ?)",
                (slot, fast.tobytes(), slow.tobytes(), int(self.mb.n_kc)))
            self.mb.W_fast[slot].fill(0.0)
            self.mb.W_slow[slot].fill(0.0)
            self.passive_slots.add(slot)
        self.active_slots.discard(slot)
        self._slot_lru.pop(slot, None)
        self.evictions += 1

    def _activate_slot(self, slot: int) -> None:
        slot = int(slot)
        was_passive = slot in self.passive_slots
        if was_passive:
            row = self._db.execute(
                "SELECT fast, slow FROM passive WHERE slot=?", (slot,)).fetchone()
            if row is None:
                raise RuntimeError(f"pasif HDC satırı diskte bulunamadı: {slot}")
            fast = np.frombuffer(row[0], dtype=np.float32)
            slow = np.frombuffer(row[1], dtype=np.float32)
            if slot < len(self.mb.W_fast):
                self._restore_row(fast, self.mb.W_fast[slot])
                self._restore_row(slow, self.mb.W_slow[slot])
            self._db.execute("DELETE FROM passive WHERE slot=?", (slot,))
            self.passive_slots.discard(slot)
        self.active_slots.add(slot)
        self._slot_lru.pop(slot, None)
        self._slot_lru[slot] = None
        if was_passive:
            self.passive_reactivations += 1
        while len(self.active_slots) > self.working_set_size:
            oldest = next((item for item in self._slot_lru if item != slot), None)
            if oldest is None:
                break
            self._deactivate_slot(oldest)

    def _matching_cue_slots(self, cue: np.ndarray, cue_kind: str) -> list[int]:
        """HDC içerik adresiyle ölçülecek sinapsları daralt; cevap üretme.

        Aktif ve pasif olaylar aynı digest defterini kullanır. Bu yalnız
        erişim adresidir: her adayın ağırlığı aşağıda yeniden ölçülür, eşik
        ve bağımsız graf doğrulaması değişmez.
        """
        digest = self._cue_digest(cue)
        rows = self._db.execute(
            "SELECT slot FROM cues WHERE kind=? AND digest=? ORDER BY slot",
            (cue_kind, digest)).fetchall()
        return [int(row[0]) for row in rows]

    def _matching_passive_slots(self, cue: np.ndarray, cue_kind: str) -> list[int]:
        """Diskteki HDC içerik anahtarıyla yalnız eşleşen pasif satırları bul."""
        return [slot for slot in self._matching_cue_slots(cue, cue_kind)
                if slot in self.passive_slots]

    def weight_energy(self) -> dict[str, float]:
        """Pahalı bütün-matris enerjisini yalnız açık tanı isteğinde ölç."""
        return {"fast": float(np.linalg.norm(self.mb.W_fast)),
                "slow": float(np.linalg.norm(self.mb.W_slow))}

    def bind_relation(self, subject: str, relation: str, obj: str, slot: int) -> None:
        slot = int(slot)
        self.mb.ensure_outputs(slot + 1)
        cues = (
            ("subject", self.encode(subject=subject)),
            ("full_relation", self.encode(subject=subject, relation=relation, object=obj)),
            # Aynı tek sunumun ters erişim anahtarı: “kim?” yerine “bu
            # ilişki+nesneyi kimler taşır?” sorusu da MB'den aday bulur.
            # Yeni örnek/epoch değil, aynı olayın üçüncü içerik adresidir.
            ("relation_object", self.encode(relation=relation, object=obj)),
        )
        for cue_kind, cue in cues:
            self.mb.learn_oneshot(cue, slot, chem=None, strength=1.0, contrastive=False)
            self.slot_code_norm[(slot, cue_kind)] = float(
                self.mb.last_learning.get("active_kc", self.mb.k))
            self._db.execute(
                "INSERT OR REPLACE INTO cues(slot, kind, digest) VALUES(?, ?, ?)",
                (slot, cue_kind, self._cue_digest(cue)))
            self.synaptic_writes += 1
        self.bound_slots.add(slot)
        self._activate_slot(slot)
        self.presentations += 1
        if getattr(self, "_auto_commit", True):
            self._db.commit()

    def _recall(self, cue: np.ndarray, cue_kind: str) -> list[tuple[int, float]]:
        idx = self.mb.encode_kc(cue)
        # ChemMushroomBody.scores genel sınıflandırıcı için geçerli k-WTA
        # ortalaması kullanır. Burada her olay kendi yazım kodunun normuyla
        # okunur; aksi hâlde nörogenez eski olayların aktivasyonunu seyreltir.
        # Pasif eşleşmeler küme kapasitesinden büyükse bile bilgi kaybolmaz:
        # her satır geri yüklenip ÖLÇÜLÜR, sonra başka bir olay yerini alabilir.
        scores: dict[int, float] = {}
        addressed_slots = self._matching_cue_slots(cue, cue_kind)
        passive_matches = [slot for slot in addressed_slots if slot in self.passive_slots]

        def measure(slot: int) -> None:
            if slot >= len(self.mb.W_fast):
                return
            norm = max(1.0, float(self.slot_code_norm.get((slot, cue_kind), len(idx))))
            value = float((self.mb.W_slow[slot, idx] + self.mb.W_fast[slot, idx]).sum()) / norm
            scores[slot] = value

        # Adres yalnız aday kümesini daraltır. Her olayın sinapsı yine ölçülür;
        # LRU baskısı bir adayı pasife taşırsa kendi sırasında geri yüklenir.
        # Böylece çalışma kümesi sınırlıyken de hiçbir adreslenmiş olay atlanmaz.
        for slot in addressed_slots:
            if slot not in self.active_slots:
                self._activate_slot(slot)
            measure(slot)
        scored_slots = sorted(scores.items())
        candidates = [(slot, score) for slot, score in scored_slots
                      if score >= self.activation_threshold]
        candidates.sort(key=lambda row: (-row[1], row[0]))
        top = max((score for _slot, score in scored_slots), default=0.0)
        self.last_recall = {
            "engine": "hdc_mb", "cue": cue_kind, "awake": True, "replay": 0,
            "active_kc": int(len(idx)), "n_kc": int(self.mb.n_kc),
            "actual_sparsity": float(len(idx) / max(1, self.mb.n_kc)),
            "target_sparsity": float(self.sparsity),
            "candidates": [{"slot": slot, "activation": score}
                           for slot, score in candidates[:24]],
            "top_activation": top, "threshold": float(self.activation_threshold),
            "addressed_slots": int(len(addressed_slots)),
            "presentations": int(self.presentations),
            "synaptic_writes": int(self.synaptic_writes),
            "growth_events": int(self.mb.growth_events),
            "working_set_size": int(self.working_set_size),
            "active_slots": int(len(self.active_slots)),
            "passive_slots": int(len(self.passive_slots)),
            "passive_matches": int(len(passive_matches)),
            "scanned_slots": int(len(scored_slots)),
            "evictions": int(self.evictions),
            "passive_reactivations": int(self.passive_reactivations),
            "passive_backend": "sqlite",
            "ram_passive_weight_rows": 0,
        }
        self._db.commit()
        return candidates

    def recall_subject(self, subject: str) -> list[tuple[int, float]]:
        return self._recall(self.encode(subject=subject), "subject")

    def recall_relation(self, subject: str, relation: str, obj: str) -> list[tuple[int, float]]:
        return self._recall(
            self.encode(subject=subject, relation=relation, object=obj), "full_relation",
        )

    def recall_relation_object(self, relation: str, obj: str) -> list[tuple[int, float]]:
        """İlişki+nesne HDC ipucuyla ters adayları getir; graf yönü sonra doğrular."""
        return self._recall(self.encode(relation=relation, object=obj), "relation_object")

    def sleep_maintenance(self) -> dict[str, float]:
        """Uyku yalnız bakım yapar; örnek tamponunu yeniden oynatmaz."""
        result = self.mb.sleep_consolidate(replay_steps=0)
        result["replayed"] = 0.0
        return result


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
