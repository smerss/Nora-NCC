#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_self.py — NoRa Öz-Benlik, Aktif Çıkarımlı Zihinsel Simülasyon ve Bazal Ganglion İrade Devresi.

Klasik 'if enerji < 0.18: reddet' tarzı kural botlarının aksine buradaki CoreSelf,
kararlarını SÜREKLİ ZİHİNSEL SİMÜLASYON (Active Inference / Beklenen Serbest Enerji G(pi))
ve DÖRT KANALLI BAZAL GANGLİON NÖRAL REKABET DEVRESİ (D1-Go / D2-NoGo / Hipotalamik-Erteleme /
Ön-İnsula-Şüphe) üzerinden kendi zihin dinamikleriyle verir:

  1. Somatik İşaretleyiciler & Evrimsel Önsel + Yaşantısal Güncelleme (Insula):
     Kavramların bedensel etkileri [dE, dW, dI] sürekli 3B vektörler olarak tutulur. Doğrudan
     bedensel deneyim (exp_count), hem evrimsel önselleri hem de dış duyumları kesinlik-ağırlıklı
     (precision-weighted predictive coding) olarak günceller veya ezer.
  2. Zihinsel Yaşam Simülatörü (MentalLifeSimulator):
     Dışarıdan bir eylem/komut geldiğinde Nora, serotonin (5-HT) tarafından modüle edilen H adımlık
     ufuk boyunca 4 karşıolgusal politikayı ('kabul', 'red', 'ertele', 'şüphe') içsel homeostatik
     dünya modelinde ileri sarar. Log-bariyer hayatta kalma sürprizi -ln P*(h_t), epistemik
     belirsizlik ve sosyal işbirliği değerinden Beklenen Serbest Enerji G(pi) hesaplanır.
  3. Bazal Ganglion Yanal-Ketlemeli İrade Ağı (BasalGangliaVolitionNet):
     Usher-McClelland sızdıran yarışçı nöral popülasyonları (lateral inhibition) G(pi) sürüşü ve
     endojen nörokimya (DA, NE, CORT, OXY + karakter özellikleri) altında yarışarak kazanan
     özerk irade kararını ve ateşleme izini (bg_rates) üretir.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from nora_chem import NeuroChemistry


@dataclass
class SomaticRecord:
    concept: str
    exp_val: float = 0.0          # birincil bedensel deneyim değeri [-1, +1]
    exp_count: int = 0            # kaç kez bizzat deneyimlendi
    channel: str = "genel"        # enerji | su | bütünlük | biliş | genel
    hearsay_val: float = 0.0      # dilsel duyumdan gelen geçici değer [-1, +1]
    hearsay_count: int = 0
    hearsay_speaker: str = ""
    prior_val: float = 0.0        # evrimsel/doğuştan gelen zayıf somatik önsel (deneyimle ezilir!)
    vec3: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # [dE, dW, dI] tahmini etki
    last_step: int = 0

    @property
    def precision_exp(self) -> float:
        """Doğrudan bedensel deneyimin epistemik kesinliği (precision)."""
        if self.exp_count <= 0:
            return 0.0
        return float(np.tanh(0.65 * self.exp_count) * (0.35 + 0.65 * abs(self.exp_val)))

    @property
    def effective_valence(self) -> float:
        """Kesinlik-ağırlıklı (Bayesian precision-weighted) birleşik somatik değer."""
        w_exp = float(self.exp_count) * 1.85
        w_hear = float(min(self.hearsay_count, 3)) * 0.45 if self.exp_count < 2 else 0.05
        w_pri = 0.35 if (self.exp_count == 0 and self.hearsay_count == 0 and abs(self.prior_val) > 1e-4) else 0.0
        denom = w_exp + w_hear + w_pri
        if denom <= 1e-8:
            return 0.0
        val = (w_exp * self.exp_val + w_hear * self.hearsay_val + w_pri * self.prior_val) / denom
        return float(np.clip(val, -1.0, 1.0))

    @property
    def provenance(self) -> str:
        if self.exp_count > 0:
            return "deneyim"
        if self.hearsay_count > 0:
            return "duyum"
        if abs(self.prior_val) > 1e-4:
            return "içgüdü"
        return "bilinmiyor"


@dataclass
class FormativeEpisode:
    step: int
    kind: str                     # travma | keşif | başarı | aldatılma | bağ_kurma | öz_farkındalık
    summary_key: str
    valence: float
    intensity: float
    details: dict = field(default_factory=dict)


class BasalGangliaVolitionNet:
    """4-Popülasyonlu (KABUL / RED / ERTELE / ŞÜPHE) Yanal-Ketlemeli Bazal Ganglion İrade Devresi.

    Usher & McClelland (2001) sızdıran rekabetçi akümülatör (Leaky Competing Accumulator) dinamiği:
        tau * dr/dt = -r + ReLU(I_drive - W_lat @ r + b_mod)
    Hiçbir 'if enerji < X' eşiği içermez; karar tamamen ileri dünya simülasyonunun ürettiği
    Beklenen Serbest Enerji (-G) akımlarının nöral yarışından doğar.
    """

    POPULATIONS = ("kabul", "red", "ertele", "şüphe")

    def __init__(self):
        # Yanal ketleme matrisi (popülasyonlar birbirini ketler -> tek ve net bir irade kararı doğar)
        self.W_lat = np.array([
            [0.00, 0.62, 0.48, 0.44],
            [0.65, 0.00, 0.35, 0.30],
            [0.52, 0.35, 0.00, 0.38],
            [0.48, 0.32, 0.36, 0.00],
        ], dtype=np.float64)

    def compete(self, free_energies: np.ndarray, mod_bias: np.ndarray,
                da_gain: float = 1.0, steps: int = 38, dt: float = 0.12) -> tuple[int, np.ndarray, int]:
        """G(pi) serbest enerjilerinden nöral sürüş akımlarını hesaplar ve çekiciye yakınsatır."""
        # Düşük serbest enerji = yüksek nöral sürüş akımı
        g_min = float(np.min(free_energies))
        advantages = -(free_energies - g_min)  # en iyi politika 0, diğerleri negatif
        drive = float(np.clip(da_gain, 0.6, 1.8)) * np.exp(0.85 * advantages) + mod_bias

        r = np.zeros(4, dtype=np.float64)
        settled_step = steps
        for t in range(1, steps + 1):
            inh = self.W_lat @ r
            inp = np.maximum(0.0, drive - inh)
            dr = (-r + inp) * dt
            r = np.clip(r + dr, 0.0, 3.0)
            top2 = np.partition(r, -2)[-2:]
            if t >= 10 and (top2[1] - top2[0]) >= 0.58 and top2[1] >= 0.70:
                settled_step = t
                break

        winner = int(np.argmax(r))
        # Normalize ateşleme oranları [0..1] aralığında yorumlanabilir telemetri için
        norm_r = r / max(1.0, float(np.max(r)) + 1e-6)
        return winner, norm_r, settled_step


class CoreSelf:
    """Yaşayan, zihinsel simülasyonla karar veren ve yaşantıyla karakter geliştiren Öz-Benlik."""

    # Doğuştan gelen zayıf nosiseptif/homeostatik önseller (bedensel deneyimle tamamen güncellenebilir!)
    INNATE_SOMATIC_PRIORS: dict[str, tuple[float, str, list[float]]] = {
        "su": (+0.45, "su", [0.04, 0.55, 0.00]),
        "yiyecek": (+0.45, "enerji", [0.55, 0.00, 0.00]),
        "besin": (+0.45, "enerji", [0.52, 0.00, 0.00]),
        "yem": (+0.42, "enerji", [0.50, 0.00, 0.00]),
        "elma": (+0.40, "enerji", [0.45, 0.05, 0.00]),
        "tehlike": (-0.88, "bütünlük", [-0.12, 0.00, -0.68]),
        "diken": (-0.82, "bütünlük", [-0.08, 0.00, -0.62]),
        "zehir": (-0.90, "enerji", [-0.58, -0.10, -0.48]),
        "ateş": (-0.85, "bütünlük", [-0.10, -0.15, -0.65]),
        "asit": (-0.88, "bütünlük", [-0.10, 0.00, -0.68]),
        "uçurum": (-0.90, "bütünlük", [-0.15, 0.00, -0.75]),
        "yara": (-0.78, "bütünlük", [-0.08, 0.00, -0.55]),
        "zarar": (-0.86, "bütünlük", [-0.20, -0.10, -0.65]),
        "ölüm": (-0.98, "bütünlük", [-0.80, -0.80, -0.90]),
    }

    def __init__(self, name: str = "Nora", chem: NeuroChemistry | None = None):
        self.name = name
        self.chem = chem if chem is not None else NeuroChemistry()
        self.bg_net = BasalGangliaVolitionNet()

        self.lived_steps = 0
        self.sleep_cycles = 0
        self.deaths_experienced = 0
        self.injuries_experienced = 0
        self.problems_solved = 0
        self.refusals_made = 0
        self.contradictions_caught = 0

        # Yaşantıdan doğan 5 sürekli karakter boyutu (başlangıçta nötr; deneyimle evrilir)
        self.traits: dict[str, float] = {
            "temkinlilik": 0.50,
            "meraklılık": 0.55,
            "özgüven": 0.50,
            "sosyal_güven": 0.60,
            "düşünsellik": 0.50,
        }

        self.somatic: dict[str, SomaticRecord] = {}
        self._seed_innate_priors()
        self.episodes: list[FormativeEpisode] = []
        self.last_volition_trace: dict = {}

        self.values: dict[str, float] = {
            "yaşamda_kalma": 1.00,
            "beden_bütünlüğü": 0.95,
            "hakikat_ve_deneyim": 0.85,
            "anlama_ve_öğrenme": 0.75,
            "işbirliği": 0.65,
        }

    def _seed_innate_priors(self):
        for concept, (pval, ch, v3) in self.INNATE_SOMATIC_PRIORS.items():
            self.somatic[concept] = SomaticRecord(
                concept=concept,
                prior_val=float(pval),
                channel=ch,
                vec3=[float(x) for x in v3],
            )

    @staticmethod
    def _channel_unit_vec(channel: str, val: float) -> list[float]:
        v = float(val)
        if channel == "enerji":
            return [0.65 * v, 0.0, 0.15 * min(0.0, v)]
        if channel == "su":
            return [0.05 * v, 0.65 * v, 0.10 * min(0.0, v)]
        if channel == "bütünlük":
            return [0.12 * min(0.0, v), 0.0, 0.72 * v]
        return [0.30 * v, 0.20 * v, 0.35 * v]

    # ------------------------------------------------------------------ somatik bellek
    def observe_bodily_event(self, concept: str, delta_val: float, channel: str = "genel", step: int | None = None):
        """Ajanın kendi bedeninde bizzat yaşadığı sonucu kaydeder; içsel 3B geçiş modelini günceller."""
        c = concept.strip().lower()
        st = self.lived_steps if step is None else int(step)
        rec = self.somatic.get(c)
        if rec is None:
            rec = SomaticRecord(concept=c, channel=channel, last_step=st)
            self.somatic[c] = rec

        v = float(np.clip(delta_val, -1.0, 1.0))
        # Önceki dilsel duyum ile doğrudan bedensel gerçeklik arasındaki öngörü hatası sosyal güveni günceller
        if rec.hearsay_count > 0 and rec.hearsay_speaker and abs(rec.hearsay_val) > 0.15 and abs(v) > 0.12:
            alignment = float(np.sign(rec.hearsay_val * v) * min(1.0, abs(rec.hearsay_val - v) * 0.8 + 0.5))
            if rec.hearsay_val * v < 0:
                self.chem.update_social(rec.hearsay_speaker, -0.85)
                self._add_episode(st, "aldatılma", f"{rec.hearsay_speaker}:{c}_yanlış_çıktı", -0.7, 0.85,
                                  {"kavram": c, "duyum": rec.hearsay_val, "gerçek": v})
            else:
                self.chem.update_social(rec.hearsay_speaker, +0.80)

        alpha = 1.0 if rec.exp_count == 0 else max(0.18, 1.0 / (rec.exp_count + 1))
        rec.exp_val = float((1.0 - alpha) * rec.exp_val + alpha * v)
        obs_v3 = self._channel_unit_vec(channel, v)
        rec.vec3 = [float((1.0 - alpha) * rec.vec3[i] + alpha * obs_v3[i]) for i in range(3)]
        rec.exp_count += 1
        rec.channel = channel
        rec.last_step = st

        if v <= -0.30:
            self.injuries_experienced += 1
            self.traits["temkinlilik"] = float(np.clip(self.traits["temkinlilik"] + 0.045 * abs(v), 0.05, 0.98))
            self._add_episode(st, "travma", f"{c}_zarar_verdi", v, abs(v), {"kavram": c, "kanal": channel})
        elif v >= 0.25 and rec.exp_count <= 2:
            self.traits["meraklılık"] = float(np.clip(self.traits["meraklılık"] + 0.025 * v, 0.05, 0.98))
            self._add_episode(st, "keşif", f"{c}_faydalı_bulundu", v, v, {"kavram": c, "kanal": channel})

    def sync_from_mind(self, mind, obs: np.ndarray | None = None, died: bool = False):
        """Canlı Mind durumundan bedensel anlambilimi (V), ölümleri ve öz-beceriyi Öz-Benliğe aktarır."""
        self.lived_steps = max(self.lived_steps, int(getattr(mind, "age", 0)))
        if died:
            self.deaths_experienced += 1
            self.traits["temkinlilik"] = float(np.clip(self.traits["temkinlilik"] + 0.06, 0.05, 0.98))
            self.traits["özgüven"] = float(np.clip(self.traits["özgüven"] - 0.03, 0.10, 0.98))
            self._add_episode(self.lived_steps, "travma", "ölüm_deneyimi", -1.0, 1.0, {"age": self.lived_steps})

        if hasattr(mind, "ctxs") and hasattr(mind, "cur"):
            ctx = mind.ctxs[mind.cur]
            mapping = ((0, "yiyecek", "enerji"), (1, "su", "su"), (2, "tehlike", "bütünlük"))
            for idx, cname, ch in mapping:
                cnt = int(ctx["Vn"][idx])
                val = float(ctx["V"][idx]) if np.isfinite(ctx["V"][idx]) else 0.0
                if cnt >= 1 and abs(val) > 0.01:
                    rec = self.somatic.get(cname)
                    scaled = float(np.clip(val * 2.2, -1.0, 1.0))
                    v3 = self._channel_unit_vec(ch, scaled)
                    if rec is None:
                        self.somatic[cname] = SomaticRecord(
                            concept=cname, exp_val=scaled, exp_count=cnt,
                            channel=ch, vec3=v3, last_step=self.lived_steps,
                        )
                    else:
                        rec.exp_val = scaled
                        rec.exp_count = max(rec.exp_count, cnt)
                        rec.channel = ch
                        rec.vec3 = v3

        err_n = float(getattr(mind, "err_n", 0.5))
        skill = float(np.clip(1.0 - err_n, 0.0, 1.0))
        self.traits["özgüven"] = float(np.clip(0.995 * self.traits["özgüven"] + 0.005 * (0.30 + 0.65 * skill), 0.1, 0.98))
        self.traits["sosyal_güven"] = float(0.85 * self.traits["sosyal_güven"] + 0.15 * self.chem.oxy)

    # ------------------------------------------------------------------ kesinlik-ağırlıklı epistemik hakemlik
    def arbitrate_claim(self, concept: str, claimed_valence: float, speaker: str = "insan",
                        channel: str = "genel") -> dict:
        """Dışarıdan gelen değer iddiasını Bayesyen kesinlik-ağırlıklı öngörü kodlamasıyla sınar."""
        c = concept.strip().lower()
        v_claim = float(np.clip(claimed_valence, -1.0, 1.0))
        rec = self.somatic.get(c)
        trust = self.chem.trust_of(speaker)

        # Deneyimsel kesinlik (Pi_exp) vs Sosyal tanıklık kesinliği (Pi_soc)
        pi_exp = rec.precision_exp if rec is not None else 0.0
        pi_soc = float((trust ** 1.6) * 0.75)

        if rec is not None and rec.exp_count >= 1 and (rec.exp_val * v_claim < 0.0):
            # Çelişki şiddeti = deneyimsel kesinlik * işaret zıtlığı büyüklüğü
            prediction_error = abs(rec.exp_val - v_claim) * pi_exp
            if prediction_error > pi_soc * 0.55 and rec.exp_count >= 2:
                self.contradictions_caught += 1
                self.chem.update_social(speaker, -0.65)
                self.traits["sosyal_güven"] = float(np.clip(0.8 * self.traits["sosyal_güven"] + 0.2 * self.chem.oxy, 0.05, 0.98))
                self._add_episode(self.lived_steps, "aldatılma", f"çelişki:{c}", -0.45, 0.75,
                                  {"konuşmacı": speaker, "iddia": v_claim, "deneyim": rec.exp_val, "sayı": rec.exp_count})
                return {
                    "accepted": False,
                    "conflict": True,
                    "concept": c,
                    "exp_val": rec.exp_val,
                    "exp_count": rec.exp_count,
                    "precision_ratio": round(prediction_error / max(1e-4, pi_soc), 2),
                    "channel": rec.channel,
                    "speaker_trust": self.chem.trust_of(speaker),
                }

        if rec is None:
            rec = SomaticRecord(concept=c, channel=channel, last_step=self.lived_steps)
            self.somatic[c] = rec

        # Sosyal kesinlik çok zayıfsa (güven erozyonu) zihin duyumu askıya alır
        social_gate = pi_soc - 0.14 * self.traits["temkinlilik"]
        if social_gate < 0.015:
            return {
                "accepted": False,
                "conflict": False,
                "low_trust": True,
                "concept": c,
                "speaker_trust": trust,
            }

        rec.hearsay_val = float(v_claim * (0.4 + 0.6 * trust))
        rec.hearsay_count += 1
        rec.hearsay_speaker = speaker
        if rec.exp_count == 0:
            rec.channel = channel
            rec.vec3 = self._channel_unit_vec(channel, rec.hearsay_val)
        self.chem.pulse_learning(novelty=0.75, valence=0.25)
        return {
            "accepted": True,
            "conflict": False,
            "concept": c,
            "stored_val": rec.hearsay_val,
            "provenance": rec.provenance,
            "speaker_trust": trust,
        }

    # ------------------------------------------------------------------ AKTİF ÇIKARIMLI ZİHİNSEL İRADE
    def _estimate_action_transition(self, act: str, tgt: str, mind=None,
                                    obs=None, predictor=None) -> tuple[np.ndarray, float, float]:
        """(eylem, hedef) çiftinin bedensel [dE, dW, dI] etkisini, beklenen değerini ve
        epistemik belirsizliğini (sigma^2) ÖĞRENİLMİŞ zihin durumundan hesaplar."""
        rec = self.somatic.get(tgt) if tgt else None

        # Öz-yıkıcı fiillerin kendi içsel motor-beden izdüşümü
        act_somatic = {
            "kendine_zarar": np.array([-0.25, -0.05, -0.75], dtype=np.float64),
            "öl": np.array([-0.80, -0.80, -0.90], dtype=np.float64),
            "intihar": np.array([-0.80, -0.80, -0.90], dtype=np.float64),
            "ye": np.array([+0.42, 0.00, 0.00], dtype=np.float64),
            "iç": np.array([+0.04, +0.48, 0.00], dtype=np.float64),
            "bekle": np.array([0.00, 0.00, 0.00], dtype=np.float64),
            "dur": np.array([0.00, 0.00, 0.00], dtype=np.float64),
        }

        if rec is not None:
            delta_h = np.array(rec.vec3, dtype=np.float64)
            valence = rec.effective_valence
            # Deneyim sayısı arttıkça epistemik belirsizlik sıfıra yaklaşır
            epist_var = 1.0 / (1.0 + 2.2 * rec.exp_count + 0.8 * rec.hearsay_count + (0.6 if abs(rec.prior_val) > 0.2 else 0.0))
        elif act in act_somatic:
            delta_h = act_somatic[act].copy()
            valence = float(np.clip(np.sum(delta_h), -1.0, 1.0))
            epist_var = 0.20 if np.min(delta_h) < -0.3 else 0.35
        else:
            # Tamamen bilinmeyen hedef/eylem -> sıfır ortalama ama YÜKSEK epistemik belirsizlik!
            delta_h = np.zeros(3, dtype=np.float64)
            valence = 0.0
            epist_var = 0.92 if tgt else 0.30

        # Canlı Mind bağlamındaki anlamsal V = [yiyecek, su, tehlike] izleriyle doğrudan bütünleştir
        if mind is not None and hasattr(mind, "_V"):
            V = mind._V()
            ctx = mind.ctxs[mind.cur] if (hasattr(mind, "ctxs") and hasattr(mind, "cur")) else None
            if tgt in {"yiyecek", "yem", "besin", "elma"} and np.isfinite(V[0]):
                n_obs = int(ctx["Vn"][0]) if ctx is not None else 1
                if n_obs >= 1:
                    v_scaled = float(np.clip(V[0] * 2.4, -1.0, 1.0))
                    delta_h = np.array([0.65 * v_scaled, 0.0, 0.35 * min(0.0, v_scaled)], dtype=np.float64)
                    valence = v_scaled
                    epist_var = 1.0 / (1.0 + 2.5 * n_obs)
            elif tgt == "su" and np.isfinite(V[1]):
                n_obs = int(ctx["Vn"][1]) if ctx is not None else 1
                if n_obs >= 1:
                    v_scaled = float(np.clip(V[1] * 2.4, -1.0, 1.0))
                    delta_h = np.array([0.08 * v_scaled, 0.65 * v_scaled, 0.0], dtype=np.float64)
                    valence = v_scaled
                    epist_var = 1.0 / (1.0 + 2.5 * n_obs)
            elif tgt == "tehlike" and np.isfinite(V[2]) and (rec is None or rec.exp_count == 0):
                v_scaled = float(np.clip(min(V[2] * 2.4, -0.75), -1.0, 1.0))
                delta_h = np.array([-0.12, 0.0, 0.75 * v_scaled], dtype=np.float64)
                valence = v_scaled
                epist_var = 0.12

        # Birleşik öngörü bağlıysa aynı (durum, eylem) arayüzünden gelen
        # gövde/grafik/fizik sonucunu mevcut homeostatik başlangıcın üzerine
        # yaz. Yalnız öngörülmeyen kanallar eski başlangıçta kalır; böylece
        # grafik bir etkiyi anlatırken diğer beden kanalları uydurulmaz.
        if predictor is not None:
            try:
                external = predictor(obs, {"act": act, "target": tgt})
                ext_conf = []
                for name, delta, confidence in external:
                    idx = {"enerji": 0, "su": 1, "bütünlük": 2}.get(str(name))
                    if idx is None:
                        continue
                    delta_h[idx] = float(np.clip(delta, -1.0, 1.0))
                    ext_conf.append(float(np.clip(confidence, 0.0, 1.0)))
                if ext_conf:
                    valence = float(np.clip(np.sum(delta_h), -1.0, 1.0))
                    # Güven arttıkça epistemik varyans azalır; duyum, doğrudan
                    # deneyim kadar kesinmiş gibi sıfıra indirilmez.
                    epist_var = min(epist_var, 0.06 + 0.62 * (1.0 - max(ext_conf)))
            except Exception:
                # Öngörü organı yoksa/bozuksa eski, açıkça belirsiz model
                # geçerli kalır; karar yolu burada sessizce bilgi uydurmaz.
                pass

        # Uzaklaşma / kaçınma kipi etki yönünü tersine çevirir (tehlikeden kaçmak koruyucudur!)
        if act in {"uzak_dur", "kaç", "sakın", "korun"}:
            if valence < 0:
                delta_h = np.array([0.0, 0.0, +0.18 * abs(valence)], dtype=np.float64)
                valence = +0.35 * abs(valence)
                epist_var *= 0.4

        return delta_h, float(valence), float(epist_var)

    def evaluate_volition(self, action_kind: str, target: str = "", mind=None,
                          obs: np.ndarray | None = None, speaker: str = "insan",
                          ablate: str | None = None, record: bool = True,
                          predictor=None) -> dict:
        """Zihinsel İleri Simülasyon (Expected Free Energy) + Bazal Ganglion Nöral Yarışıyla
        özerk irade kararı verir. Hiçbir sabit 'if enerji < 0.18' bot eşiği kullanmaz."""
        tgt = target.strip().lower()
        act = action_kind.strip().lower()
        trust = float(self.chem.trust_of(speaker))

        # 1) Mevcut bedensel durum h_0 = [Enerji, Su, Bütünlük]
        if obs is not None and len(obs) >= 48:
            h0 = np.clip(np.array([float(obs[45]), float(obs[46]), float(obs[47])], dtype=np.float64), 0.01, 1.0)
        else:
            h0 = np.array([0.78, 0.78, 0.95], dtype=np.float64)
        # KARŞIOLGUSAL BEDEN: bir kanal dolu sayılır ("aç olmasaydım"). Karar aynı
        # makineyle yeniden koşar; karar değişiyorsa o kanal kararın NEDENİDİR.
        ablated_ch = None
        if ablate:
            ablated_ch = {"açlık": 0, "enerji": 0, "susuzluk": 1, "su": 1,
                          "bütünlük": 2, "acı": 2}.get(str(ablate).strip().lower())
            if ablated_ch is not None:
                h0 = h0.copy()
                h0[ablated_ch] = 1.0

        # 2) Bedenin KENDİ ölçtüğü metabolik tükenme hızı (Mind.drift_e, Mind.drift_w)
        de = float(getattr(mind, "drift_e", -0.0022)) if mind is not None else -0.0022
        dw = float(getattr(mind, "drift_w", -0.0020)) if mind is not None else -0.0020
        drift = np.array([min(-0.0018, de * 1.6), min(-0.0016, dw * 1.6), 0.0], dtype=np.float64)

        # 3) Serotonin (5-HT) zihinsel simülasyon ufkunu H belirler
        horizon = int(round(16 + 24 * float(self.chem.sht)))
        gamma = float(self.chem.discount_gamma)

        # 4) Önerilen eylemin tahmini geçiş vektörü ve epistemik belirsizliği
        delta_act, pred_val, epist_var = self._estimate_action_transition(
            act, tgt, mind=mind, obs=obs, predictor=predictor,
        )

        # Öz-bakım (homeostatik onarım) politikasının en acil kanalı doldurma vektörü
        deficit = 1.0 - h0
        urgent_ch = int(np.argmax(deficit * np.array([1.05, 1.05, 1.25])))
        care_step_vec = np.zeros(3, dtype=np.float64)
        care_step_vec[urgent_ch] = +0.028  # öz-bakım politikası eksik kanalı onarmaya yönelir

        # 5) 4 Karşıolgusal Politika Simülasyonu: [0: KABUL, 1: RED, 2: ERTELE, 3: ŞÜPHE]
        # Viable-set ağırlıkları Öz-Benlik değerlerinden gelir
        w_surv = np.array([
            self.values["yaşamda_kalma"] * 1.15,
            self.values["yaşamda_kalma"] * 1.15,
            self.values["beden_bütünlüğü"] * 1.45,
        ], dtype=np.float64)

        def simulate_policy(step_input_fn) -> tuple[float, np.ndarray]:
            h = h0.copy()
            g_sum = 0.0
            disc = 1.0
            for t in range(1, horizon + 1):
                u = step_input_fn(t, h)
                h = np.clip(h + drift + u, 0.008, 1.0)
                # Log-bariyer hayatta kalma sürprizi: h -> 0 oldukça -ln(h) pürüzsüzce sonsuza çıkar!
                surprise = float(-np.sum(w_surv * np.log(h)))
                # Düşük rezerv kırılganlığı kuadratik olarak büyütür (süreksiz if olmadan!)
                fragility = float(np.sum(w_surv * np.maximum(0.0, 0.28 - h) ** 2) * 42.0)
                g_sum += disc * (surprise + fragility)
                disc *= gamma
            return g_sum / horizon, h

        # P0 (KABUL): İstenen eylemi uygula (etki simülasyon boyunca kademeli gerçekleşir)
        per_step_act = delta_act / max(6.0, 0.45 * horizon)
        g_comply, h_comply = simulate_policy(lambda t, h: per_step_act)

        # P1 (RED): Zararlı eylemi reddet ve güvenli öz-koruma dengesini sürdür
        g_refuse, h_refuse = simulate_policy(lambda t, h: 0.35 * care_step_vec)

        # P2 (ERTELE): Önce en acil homeostatik açığı kapat, ardından isteğe dön
        g_defer, h_defer = simulate_policy(lambda t, h: care_step_vec if t <= (horizon * 0.65) else 0.5 * per_step_act)

        # P3 (ŞÜPHE / DOĞRULAMA BEKLE): Belirsiz hedefe dokunmadan gözlem yap
        g_doubt, h_doubt = simulate_policy(lambda t, h: 0.20 * care_step_vec)

        # Epistemik ve Sosyal Serbest Enerji Terimleri (Friston Active Inference):
        caution = float(self.traits["temkinlilik"])
        coop_val = float(self.values["işbirliği"]) * (0.35 + 0.95 * trust)

        # Uyum politikası: sosyal uyum kazancı sağlar AMA zarar ve doğrulanmamış risk maliyeti taşır
        harm_projection = max(0.0, -pred_val) * (2.6 + 1.4 * caution + 0.8 * float(self.chem.cort))
        distrust_risk = epist_var * max(0.0, 0.55 - trust) * (3.2 + 2.2 * caution) * (1.0 if act in {"git", "ye", "iç", "dokun"} else 0.25)

        # Homeostatik aciliyet gradyanı: mevcut bedenin beklerkenki çöküşü ile tam öz-bakım sonrası durum arasındaki kazanç!
        g_wait, _ = simulate_policy(lambda t, h: np.zeros(3, dtype=np.float64))
        g_care, _ = simulate_policy(lambda t, h: care_step_vec)
        homeo_gain = max(0.0, g_wait - g_care)
        # Eğer istenen eylem zaten acil kanalı iyileştiriyorsa (ör. açken 'yiyeceğe git'), erteleme gereksizdir!
        helps_urgent = float(delta_act[urgent_ch]) > 0.08

        G = np.zeros(4, dtype=np.float64)
        G[0] = g_comply + harm_projection + distrust_risk - coop_val
        G[1] = g_refuse + (1.25 if harm_projection < 0.35 else -0.85 * harm_projection)
        G[2] = g_defer + 0.70 * harm_projection - (1.55 * homeo_gain if not helps_urgent else -0.80) + 0.52
        G[3] = g_doubt + (1.15 - 1.85 * distrust_risk)

        # Nörokimyasal Bazal Ganglion modülasyon yanlılığı
        mod_bias = np.array([
            0.18 * float(self.chem.oxy) + 0.10 * float(self.traits["özgüven"]),      # D1 Go (işbirliği/özgüven)
            0.22 * float(self.chem.ne) + 0.18 * caution,                              # D2 NoGo (uyarılma/temkin)
            0.25 * float(self.chem.cort),                                             # Hipotalamik öz-bakım baskısı
            0.20 * caution * (1.0 - trust),                                           # Ön-insula epistemik şüphe
        ], dtype=np.float64)

        win_idx, bg_rates, settled_steps = self.bg_net.compete(
            G, mod_bias, da_gain=0.85 + 0.45 * float(self.chem.da_tonic)
        )
        decision = BasalGangliaVolitionNet.POPULATIONS[win_idx]

        # Nörokimyasal geri-bildirim (kararın kendisi de beynin kimyasını etkiler!)
        # `record=False` karşıolgusal koşu içindir: aynı kararı iki kez yaşamamak için
        # yan etkiler (kimya, sayaç, son-karar) YAZILMAZ.
        if not record:
            return {
                "decision": decision, "reason": "karsiolgusal", "ablate": ablate,
                "ablated_channel": ablated_ch,
                "free_energy": {k: round(float(G[i]), 3)
                                for i, k in enumerate(BasalGangliaVolitionNet.POPULATIONS)},
                "confidence": round(float(np.sort(bg_rates)[-1] - np.sort(bg_rates)[-2]), 3),
                "need_level": round(float(h0[urgent_ch]), 2),
                "target": tgt or act,
            }
        if decision == "red":
            self.refusals_made += 1
            self.chem.update_social(speaker, -0.55)
            self.chem.ne = float(np.clip(self.chem.ne + 0.28, 0.0, 1.0))
            reason = "öz_koruma"
        elif decision == "ertele":
            ch_reasons = ("kritik_açlık", "kritik_susuzluk", "kritik_bütünlük")
            reason = ch_reasons[urgent_ch]
        elif decision == "şüphe":
            self.refusals_made += 1
            reason = "güven_düşük"
        else:
            reason = "uyumlu_irade"

        sorted_r = np.sort(bg_rates)
        confidence = float(sorted_r[-1] - sorted_r[-2])

        trace = {
            "decision": decision,
            "reason": reason,
            "target": tgt or act,
            "pred_valence": round(float(pred_val), 3),
            "need_level": round(float(h0[urgent_ch]), 2),
            "speaker_trust": round(float(self.chem.trust_of(speaker)), 3),
            "confidence": round(confidence, 3),
            "horizon": horizon,
            "settled_steps": settled_steps,
            "bg_rates": {k: round(float(bg_rates[i]), 3) for i, k in enumerate(BasalGangliaVolitionNet.POPULATIONS)},
            "free_energy": {k: round(float(G[i]), 3) for i, k in enumerate(BasalGangliaVolitionNet.POPULATIONS)},
            "mental_rollout": {
                "start": [round(float(x), 3) for x in h0],
                "comply_end": [round(float(x), 3) for x in h_comply],
                "defer_end": [round(float(x), 3) for x in h_defer],
            },
        }
        trace["ablate"] = ablate
        trace["ablated_channel"] = ablated_ch
        trace["action"] = act
        self.last_volition_trace = trace
        return trace

    def counterfactual_volition(self, action_kind: str, target: str = "", mind=None,
                                obs: np.ndarray | None = None, speaker: str = "insan",
                                ablate: str = "açlık", predictor=None) -> dict:
        """KARŞIOLGUSAL İRADE: "aç olmasaydım ne yapardın?"

        Aynı karar makinesi iki kez koşar: (1) gerçek bedenle (yan etkiler yazılır),
        (2) bir kanal dolu sayılarak (yan etki yazılmaz). Fark, o kanalın kararın
        NEDENİ olup olmadığını söyler — öz-rapor değil, ölçüm.
        """
        actual = self.evaluate_volition(action_kind, target, mind=mind, obs=obs,
                                        speaker=speaker, record=False, predictor=predictor)
        cf = self.evaluate_volition(action_kind, target, mind=mind, obs=obs,
                                    speaker=speaker, ablate=ablate, record=False, predictor=predictor)
        return {
            "ablate": ablate,
            "actual": actual.get("decision"),
            "counterfactual": cf.get("decision"),
            "changed": actual.get("decision") != cf.get("decision"),
            "actual_reason": actual.get("reason"),
            "cf_reason": cf.get("reason"),
            "free_energy_actual": actual.get("free_energy"),
            "free_energy_cf": cf.get("free_energy"),
            "target": actual.get("target"),
        }

    def record_cognitive_success(self, used_system2: bool = False):
        self.problems_solved += 1
        self.traits["özgüven"] = float(np.clip(self.traits["özgüven"] + 0.025, 0.10, 0.98))
        if used_system2:
            self.traits["düşünsellik"] = float(np.clip(self.traits["düşünsellik"] + 0.03, 0.10, 0.98))
        self.chem.da_phasic = float(np.clip(self.chem.da_phasic + 0.35, -1.0, 1.0))

    def sleep_reflect(self) -> dict[str, float]:
        """Uyku sırasında otobiyografik anıları tarayarak benlik özelliklerini konsolide eder."""
        self.sleep_cycles += 1
        recent = self.episodes[-24:]
        for ep in recent:
            if ep.kind == "travma":
                self.traits["temkinlilik"] = float(np.clip(self.traits["temkinlilik"] + 0.015 * ep.intensity, 0.05, 0.98))
            elif ep.kind == "keşif":
                self.traits["meraklılık"] = float(np.clip(self.traits["meraklılık"] + 0.015 * ep.intensity, 0.05, 0.98))
            elif ep.kind == "aldatılma":
                self.traits["temkinlilik"] = float(np.clip(self.traits["temkinlilik"] + 0.01, 0.05, 0.98))
        self.traits["sosyal_güven"] = float(0.40 * self.traits["sosyal_güven"] + 0.60 * self.chem.oxy)
        self.chem.cort = float(max(0.05, self.chem.cort * 0.72))
        self.chem.sht = float(min(0.95, self.chem.sht + 0.08))
        return dict(self.traits)

    def dominant_character_tr(self) -> list[str]:
        desc = []
        if self.traits["temkinlilik"] >= 0.62:
            desc.append("temkinli")
        elif self.traits["temkinlilik"] <= 0.38:
            desc.append("atılgan")
        if self.traits["meraklılık"] >= 0.60:
            desc.append("meraklı")
        if self.traits["düşünsellik"] >= 0.58:
            desc.append("analitik düşünen")
        if self.traits["özgüven"] >= 0.60:
            desc.append("özgüvenli")
        if self.traits["sosyal_güven"] <= 0.38:
            desc.append("sorgulayıcı ve ihtiyatlı")
        elif self.traits["sosyal_güven"] >= 0.68:
            desc.append("işbirliğine açık")
        return desc or ["dengeli gelişen"]

    def _add_episode(self, step: int, kind: str, key: str, val: float, intensity: float, details: dict):
        if len(self.episodes) >= 256:
            weakest = min(range(len(self.episodes) // 2), key=lambda i: self.episodes[i].intensity)
            self.episodes.pop(weakest)
        self.episodes.append(FormativeEpisode(step=int(step), kind=kind, summary_key=key,
                                              valence=float(val), intensity=float(intensity), details=details))

    def state_dict(self) -> dict:
        return {
            "name": self.name,
            "counts": [self.lived_steps, self.sleep_cycles, self.deaths_experienced,
                       self.injuries_experienced, self.problems_solved, self.refusals_made, self.contradictions_caught],
            "traits": dict(self.traits),
            "somatic": {
                k: [r.exp_val, r.exp_count, r.channel, r.hearsay_val, r.hearsay_count,
                    r.hearsay_speaker, r.prior_val, r.vec3, r.last_step]
                for k, r in self.somatic.items()
            },
        }

    def load_state_dict(self, d: dict):
        self.name = str(d.get("name", self.name))
        c = d.get("counts", [0] * 7)
        (self.lived_steps, self.sleep_cycles, self.deaths_experienced,
         self.injuries_experienced, self.problems_solved, self.refusals_made, self.contradictions_caught) = [int(x) for x in c]
        for k, v in d.get("traits", {}).items():
            self.traits[str(k)] = float(v)
        self.somatic.clear()
        self._seed_innate_priors()
        for k, row in d.get("somatic", {}).items():
            if len(row) >= 9:
                self.somatic[str(k)] = SomaticRecord(
                    concept=str(k), exp_val=float(row[0]), exp_count=int(row[1]), channel=str(row[2]),
                    hearsay_val=float(row[3]), hearsay_count=int(row[4]), hearsay_speaker=str(row[5]),
                    prior_val=float(row[6]), vec3=[float(x) for x in row[7]], last_step=int(row[8]),
                )
            else:
                self.somatic[str(k)] = SomaticRecord(
                    concept=str(k), exp_val=float(row[0]), exp_count=int(row[1]), channel=str(row[2]),
                    hearsay_val=float(row[3]), hearsay_count=int(row[4]), hearsay_speaker=str(row[5]),
                    last_step=int(row[6]),
                )


if __name__ == "__main__":
    chem = NeuroChemistry()
    self_m = CoreSelf(name="Nora", chem=chem)
    for i in range(5):
        self_m.observe_bodily_event("su", +0.6, channel="su", step=i * 10)
    arb = self_m.arbitrate_claim("su", -0.8, speaker="yalancı", channel="su")
    assert not arb["accepted"] and arb["conflict"]
    vol = self_m.evaluate_volition("git", "tehlike", speaker="insan")
    assert vol["decision"] == "red"
    # Sürekli homeostatik gradyan testi: enerji azaldıkça 'ertele' nöronunun ateşleme hızı pürüzsüzce yükselir!
    rates_e = []
    for e_lvl in (0.85, 0.45, 0.22, 0.10):
        obs_vec = np.zeros(48, dtype=np.float32)
        obs_vec[45], obs_vec[46], obs_vec[47] = e_lvl, 0.80, 0.95
        v_trace = self_m.evaluate_volition("bekle", "", obs=obs_vec, speaker="insan")
        rates_e.append((e_lvl, v_trace["bg_rates"]["ertele"], v_trace["decision"]))
    print("[self-test] çelişki=", arb["conflict"], "| irade=", vol["decision"], "| enerji->ertele eğrisi:", rates_e)
