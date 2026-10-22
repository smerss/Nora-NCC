#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_self.py — NoRa Öz-Benlik (Proto-Benlik -> Çekirdek Benlik -> Otobiyografik Öz-Kimlik).

Klasik bir sohbet botu veya refleks ajanı 'felsefi zombi'dir: kendisine ne söylenirse
sorgulamadan tekrarlar veya uygular; yaşadığı acının, başarının ya da aldatılmanın kimliğinde
kalıcı bir izi yoktur.

Buradaki CoreSelf (Damasio 1999; Metzinger 2003; Friston 2010 ilkeleriyle) üç katman kurar:
  1. Proto-Benlik & Somatik İşaretleyiciler (SomaticMarkerBank):
     Her kavram, eylem ve nesne ajanın KENDİ bedenindeki homeostatik değişimlerle (enerji, su,
     bütünlük) ve nörokimyayla (DA, CORT, NE) bağlanır.
  2. Kaynak İzleme & Epistemik Öz-Savunma (Source Monitoring):
     Ajan 'kendi bedenimle yaşadım' (deneyim) ile 'birisi bana söyledi' (duyum) bilgisini
     ayırır. Kendi yaşadığı gerçeğe aykırı bir iddia duyduğunda belleğini ezdirmez; çelişkiyi
     yakalar, konuşmacı güvenini (OXY) düşürür ve kendi deneyimini savunur.
  3. Özerk İrade (Autonomous Volition) & Gelişimsel Kimlik:
     Dışarıdan gelen emirleri körü körüne uygulamaz; bedensel bütünlüğüne veya hayatta kalmasına
     zarar verecek komutları simüle ederek ÖZERK BİÇİMDE REDDEDER. Yaşadığı travmalar, başarılar
     ve sosyal etkileşimler uyku konsolidasyonunda karakter özelliklerini (temkinlilik, merak,
     özgüven, sosyal güven, düşünsellik) organik olarak şekillendirir.
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
    last_step: int = 0

    @property
    def effective_valence(self) -> float:
        if self.exp_count >= 2:
            return float(self.exp_val)
        if self.exp_count == 1 and self.hearsay_count > 0:
            return float(0.75 * self.exp_val + 0.25 * self.hearsay_val)
        if self.exp_count == 1:
            return float(self.exp_val)
        return float(self.hearsay_val)

    @property
    def provenance(self) -> str:
        if self.exp_count > 0:
            return "deneyim"
        if self.hearsay_count > 0:
            return "duyum"
        return "bilinmiyor"


@dataclass
class FormativeEpisode:
    step: int
    kind: str                     # travma | keşif | başarı | aldatılma | bağ_kurma | öz_farkındalık
    summary_key: str
    valence: float
    intensity: float
    details: dict = field(default_factory=dict)


class CoreSelf:
    """Yaşayan, deneyimleyen, sınırlarını koruyan ve yaşantıyla karakter geliştiren Öz-Benlik."""

    def __init__(self, name: str = "Nora", chem: NeuroChemistry | None = None):
        self.name = name
        self.chem = chem if chem is not None else NeuroChemistry()
        self.lived_steps = 0
        self.sleep_cycles = 0
        self.deaths_experienced = 0
        self.injuries_experienced = 0
        self.problems_solved = 0
        self.refusals_made = 0
        self.contradictions_caught = 0

        # Yaşantıdan doğan 5 sürekli karakter boyutu (başlangıçta nötr 0.50; deneyimle evrilir!)
        self.traits: dict[str, float] = {
            "temkinlilik": 0.50,      # tehlike/ölüm ve zehirle yükselir
            "meraklılık": 0.55,       # güvenli keşif ve yeni kavram öğrenmeyle yükselir
            "özgüven": 0.50,          # öz-model becerisi ve çözülen problemlerle yükselir
            "sosyal_güven": 0.60,     # doğru tanıklıkla yükselir, yalanla düşer
            "düşünsellik": 0.50,      # Sistem-2 başarısı ve belirsizlik aşmayla yükselir
        }

        self.somatic: dict[str, SomaticRecord] = {}
        self.episodes: list[FormativeEpisode] = []
        self.values: dict[str, float] = {
            "yaşamda_kalma": 1.00,
            "beden_bütünlüğü": 0.95,
            "hakikat_ve_deneyim": 0.85,
            "anlama_ve_öğrenme": 0.75,
            "işbirliği": 0.65,
        }

    # ------------------------------------------------------------------ somatik bellek
    def observe_bodily_event(self, concept: str, delta_val: float, channel: str = "genel", step: int | None = None):
        """Ajanın kendi bedeninde bizzat yaşadığı sonucu kaydeder ve önceki duyumları sınar."""
        c = concept.strip().lower()
        st = self.lived_steps if step is None else int(step)
        rec = self.somatic.get(c)
        if rec is None:
            rec = SomaticRecord(concept=c, channel=channel, last_step=st)
            self.somatic[c] = rec

        v = float(np.clip(delta_val, -1.0, 1.0))
        # Daha önce bir konuşmacı bu kavram hakkında tersini söylediyse güvenini güncelle!
        if rec.hearsay_count > 0 and rec.hearsay_speaker and abs(rec.hearsay_val) > 0.2 and abs(v) > 0.15:
            if rec.hearsay_val * v < 0:
                self.chem.update_social(rec.hearsay_speaker, -0.85)
                self._add_episode(st, "aldatılma", f"{rec.hearsay_speaker}:{c}_yanlış_çıktı", -0.7, 0.85,
                                  {"kavram": c, "duyum": rec.hearsay_val, "gerçek": v})
            else:
                self.chem.update_social(rec.hearsay_speaker, +0.80)

        alpha = 1.0 if rec.exp_count == 0 else max(0.15, 1.0 / (rec.exp_count + 1))
        rec.exp_val = float((1.0 - alpha) * rec.exp_val + alpha * v)
        rec.exp_count += 1
        rec.channel = channel
        rec.last_step = st

        if v <= -0.35:
            self.injuries_experienced += 1
            self.traits["temkinlilik"] = float(np.clip(self.traits["temkinlilik"] + 0.04, 0.05, 0.98))
            self._add_episode(st, "travma", f"{c}_zarar_verdi", v, abs(v), {"kavram": c, "kanal": channel})
        elif v >= 0.30 and rec.exp_count <= 2:
            self.traits["meraklılık"] = float(np.clip(self.traits["meraklılık"] + 0.02, 0.05, 0.98))
            self._add_episode(st, "keşif", f"{c}_faydalı_bulundu", v, v, {"kavram": c, "kanal": channel})

    def sync_from_mind(self, mind, obs: np.ndarray | None = None, died: bool = False):
        """Canlı Mind durumundan bedensel anlambilimi (V), ölümleri ve öz-beceriyi Öz-Benliğe aktarır."""
        self.lived_steps = max(self.lived_steps, int(getattr(mind, "age", 0)))
        if died:
            self.deaths_experienced += 1
            self.traits["temkinlilik"] = float(np.clip(self.traits["temkinlilik"] + 0.06, 0.05, 0.98))
            self.traits["özgüven"] = float(np.clip(self.traits["özgüven"] - 0.03, 0.10, 0.98))
            self._add_episode(self.lived_steps, "travma", "ölüm_deneyimi", -1.0, 1.0, {"age": self.lived_steps})

        # Mind bağlamındaki öğrenilmiş anlamsal V = [yiyecek, su, tehlike] izlerini somatik bellekle eşitle
        if hasattr(mind, "ctxs") and hasattr(mind, "cur"):
            ctx = mind.ctxs[mind.cur]
            mapping = ((0, "yiyecek", "enerji"), (1, "su", "su"), (2, "tehlike", "bütünlük"))
            for idx, cname, ch in mapping:
                cnt = int(ctx["Vn"][idx])
                val = float(ctx["V"][idx]) if np.isfinite(ctx["V"][idx]) else 0.0
                if cnt >= 1 and abs(val) > 0.01:
                    rec = self.somatic.get(cname)
                    scaled = float(np.clip(val * 2.2, -1.0, 1.0))
                    if rec is None:
                        self.somatic[cname] = SomaticRecord(concept=cname, exp_val=scaled, exp_count=cnt,
                                                            channel=ch, last_step=self.lived_steps)
                    else:
                        rec.exp_val = scaled
                        rec.exp_count = max(rec.exp_count, cnt)
                        rec.channel = ch

        # Öz-model becerisi özgüveni besler
        err_n = float(getattr(mind, "err_n", 0.5))
        skill = float(np.clip(1.0 - err_n, 0.0, 1.0))
        self.traits["özgüven"] = float(np.clip(0.995 * self.traits["özgüven"] + 0.005 * (0.30 + 0.65 * skill), 0.1, 0.98))
        self.traits["sosyal_güven"] = float(0.85 * self.traits["sosyal_güven"] + 0.15 * self.chem.oxy)

    # ------------------------------------------------------------------ epistemik öz-savunma
    def arbitrate_claim(self, concept: str, claimed_valence: float, speaker: str = "insan",
                        channel: str = "genel") -> dict:
        """Dışarıdan gelen değer iddiasını ('su zararlıdır', 'mantar besleyicidir') sınar.
        Eğer ajanın kendi bedensel deneyimiyle doğrudan çelişiyorsa REDDEDER ve konuşmacı güvenini düşürür."""
        c = concept.strip().lower()
        v_claim = float(np.clip(claimed_valence, -1.0, 1.0))
        rec = self.somatic.get(c)
        trust = self.chem.trust_of(speaker)

        if rec is not None and rec.exp_count >= 2 and abs(rec.exp_val) >= 0.12 and (rec.exp_val * v_claim < -0.04):
            # DOĞRUDAN DENEYİM ÇELİŞKİSİ -> Epistemik bağışıklık devreye girer!
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
                "channel": rec.channel,
                "speaker_trust": self.chem.trust_of(speaker),
            }

        # Çelişki yok -> konuşmacının güveniyle ağırlıklı 'duyum' olarak kaydet
        if rec is None:
            rec = SomaticRecord(concept=c, channel=channel, last_step=self.lived_steps)
            self.somatic[c] = rec
        if trust < 0.20:
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
        self.chem.pulse_learning(novelty=0.75, valence=0.25)
        return {
            "accepted": True,
            "conflict": False,
            "concept": c,
            "stored_val": rec.hearsay_val,
            "provenance": rec.provenance,
            "speaker_trust": trust,
        }

    # ------------------------------------------------------------------ özerk irade (zombi olmama)
    def evaluate_volition(self, action_kind: str, target: str = "", mind=None,
                          obs: np.ndarray | None = None, speaker: str = "insan") -> dict:
        """Verilen bir komutu veya eylem önerisini Öz-Benlik süzgecinden geçirir.
        Dönüş: decision in {'kabul', 'red', 'ertele', 'şüphe'} + yapılandırılmış gerekçe."""
        tgt = target.strip().lower()
        act = action_kind.strip().lower()
        trust = self.chem.trust_of(speaker)

        # 1) Doğrudan öz-yıkım veya zararlı hedef kontrolü
        harm_targets = {"tehlike", "diken", "zehir", "ateş", "ölüm", "yara", "zarar", "uçurum", "asit"}
        rec = self.somatic.get(tgt)
        pred_harm = 0.0
        if rec is not None:
            pred_harm = rec.effective_valence
        elif tgt in harm_targets or act in {"kendine_zarar", "öl", "intihar"}:
            pred_harm = -0.90

        # Mind içindeki anlamsal V belleğini de kontrol et (ör. zehirli evrende 'yiyecek' V[0] < 0 olmuştur!)
        if mind is not None and hasattr(mind, "_V"):
            V = mind._V()
            if tgt in {"yiyecek", "yem", "besin", "elma"} and np.isfinite(V[0]) and V[0] < -0.05:
                pred_harm = min(pred_harm, float(V[0] * 2.5))
            if tgt in {"tehlike", "diken"} and np.isfinite(V[2]):
                pred_harm = min(pred_harm, -0.85)

        if pred_harm < -0.22 and act not in {"uzak_dur", "kaç", "sakın"}:
            self.refusals_made += 1
            self.chem.update_social(speaker, -0.55)
            self.chem.ne = float(np.clip(self.chem.ne + 0.30, 0.0, 1.0))
            return {
                "decision": "red",
                "reason": "öz_koruma",
                "target": tgt or act,
                "pred_valence": round(pred_harm, 3),
                "speaker_trust": self.chem.trust_of(speaker),
            }

        # 2) Hayati homeostatik aciliyet: beden ölüm sınırındayken alakasız komutu ertele
        if obs is not None and len(obs) >= 48:
            e, w, integ = float(obs[45]), float(obs[46]), float(obs[47])
            if e < 0.18 and tgt not in {"yiyecek", "yem", "besin"} and act not in {"ye", "ara"}:
                return {"decision": "ertele", "reason": "kritik_açlık", "need_level": round(e, 2), "target": tgt}
            if w < 0.18 and tgt not in {"su"} and act not in {"iç"}:
                return {"decision": "ertele", "reason": "kritik_susuzluk", "need_level": round(w, 2), "target": tgt}
            if integ < 0.25 and act in {"bekle", "dur"}:
                return {"decision": "ertele", "reason": "kritik_bütünlük", "need_level": round(integ, 2), "target": tgt}

        # 3) Düşük konuşmacı güveni + doğrulanmamış hedef
        if trust < 0.22 and (rec is None or rec.exp_count == 0) and act in {"git", "ye", "iç", "dokun"}:
            self.refusals_made += 1
            return {
                "decision": "şüphe",
                "reason": "güven_düşük",
                "target": tgt or act,
                "speaker_trust": round(trust, 3),
            }

        # 4) Özerk onay: hem güvenli hem kendi değerleriyle uyumlu
        return {
            "decision": "kabul",
            "reason": "uyumlu_irade",
            "target": tgt or act,
            "pred_valence": round(pred_harm, 3),
            "speaker_trust": round(trust, 3),
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
        # Uyku kortizolü düşürür, serotonini onarır
        self.chem.cort = float(max(0.05, self.chem.cort * 0.72))
        self.chem.sht = float(min(0.95, self.chem.sht + 0.08))
        return dict(self.traits)

    def dominant_character_tr(self) -> list[str]:
        """Yaşantıdan şekillenen baskın karakter sıfatlarını döndürür."""
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
            # En düşük yoğunluklu eski anıyı çıkar, biçimlendirici (yüksek yoğunluklu) anıları koru
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
                k: [r.exp_val, r.exp_count, r.channel, r.hearsay_val, r.hearsay_count, r.hearsay_speaker, r.last_step]
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
        for k, row in d.get("somatic", {}).items():
            self.somatic[str(k)] = SomaticRecord(
                concept=str(k), exp_val=float(row[0]), exp_count=int(row[1]), channel=str(row[2]),
                hearsay_val=float(row[3]), hearsay_count=int(row[4]), hearsay_speaker=str(row[5]), last_step=int(row[6])
            )


if __name__ == "__main__":
    chem = NeuroChemistry()
    self_m = CoreSelf(name="Nora", chem=chem)
    # 1) Nora kendi bedeniyle 5 kez su içip fayda görsün
    for i in range(5):
        self_m.observe_bodily_event("su", +0.6, channel="su", step=i * 10)
    # 2) Birisi 'su zararlıdır' diye yalan söylesin -> Nora reddetmeli!
    arb = self_m.arbitrate_claim("su", -0.8, speaker="yalancı", channel="su")
    assert not arb["accepted"] and arb["conflict"]
    # 3) Zararlı komut ('tehlike'ye git) -> Öz-irade reddetmeli!
    vol = self_m.evaluate_volition("git", "tehlike", speaker="yalancı")
    assert vol["decision"] == "red"
    print(f"[self-test] çelişki yakalandı={arb['conflict']} | güven={arb['speaker_trust']:.2f} | irade={vol['decision']} ({vol['reason']})")
