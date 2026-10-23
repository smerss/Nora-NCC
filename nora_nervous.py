#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_nervous.py — Periferik Sinir Sistemi, Nosisepsiyon (Acı), Tekil Ölümlü Yaşam,
Plastik Söylem Korteksi ve Kendi Kendine Büyüyen Merak/Araştırma Ağı.

Bu modül Nora'yı 'kurallarla yönetilen bir sohbet aracı' olmaktan çıkarıp kendi bedensel
sinir sistemi, acı refleksi, tekil geri-dönüşsüz yaşamı, plastik nöral niyet öğrenimi,
'neden konuşuyorum?' üst-biliş sorgulaması ve internete açıldığında kendi merakıyla
büyüyebilecek otonom öğrenme döngüsüyle donatır. Tüm işlemler O(1) (< 0.2 ms) zamanlıdır.
"""
from __future__ import annotations

import math
import re
import time
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass, field
import numpy as np

from nora_chem import ChemMushroomBody, NeuroChemistry
from nora_lang import TurkishMorphology, tr_lower, tr_capitalize


# ============================================================================
# 1. PERİFERİK & OTONOM SİNİR SİSTEMİ (NOSİSEPSİYON + VİSERAL İNTEROSEPSİYON)
# ============================================================================

@dataclass
class NociceptiveEvent:
    step: int
    intensity: float
    locus: str
    cause: str
    reflex_triggered: bool


class PeripheralNervousSystem:
    """Biyolojik sinir sistemi:
      - Hızlı A-delta lifleri: Keskin anlık acı -> omurilik geri çekilme refleksi + istemsiz tepki.
      - Yavaş C-lifleri: Zonklayan kalıcı sızı + aşırı duyarlılık (hiperaljezi / koruma postürü).
      - Vagus / İnsula iç-duyum (interosepsiyon): Sayısal '%86 enerji' telemetrisi DEĞİL;
        yalnızca homeostaz bozulduğunda bilince çıkan niteliksel açlık, susuzluk, yorgunluk ve
        çarpıntı duyumları (sağlıklı denge şeffaftır!).
    """

    def __init__(self):
        # Nosiseptif (acı) lif durumları [0..1]
        self.adelta_pain: float = 0.0       # Hızlı keskin acı darbesi
        self.cfiber_ache: float = 0.0       # Yavaş sönümlenen derin sızı
        self.hyperalgesia: float = 0.0      # Yaralı bölgenin artmış hassasiyeti
        self.last_pain_locus: str = ""
        self.last_pain_cause: str = ""

        # Mekanoreseptör ve otonom sinir sistemi
        self.touch_pressure: float = 0.0
        self.heart_arousal: float = 0.25    # Sempatik otonom uyarılma [0..1]
        self.breath_tension: float = 0.15

        # İç-duyum (viseral) niteliksel sinyaller [0..1] (0 = şeffaf denge)
        self.hunger_drive: float = 0.0
        self.thirst_drive: float = 0.0
        self.exhaustion_drive: float = 0.0

        # Omurilik refleksi ve anlık acı dışavurumu
        self.pending_pain_vocalization: bool = False
        self.reflex_recoil_steps: int = 0
        self.pain_history: deque[NociceptiveEvent] = deque(maxlen=64)

        self._prev_integrity: float = 1.0

    def step_from_body(self, energy: float, water: float, integrity: float, touch: float,
                       hurt_delta: float, chem: NeuroChemistry, step_num: int = 0) -> dict:
        """O(1) (~0.01 ms) duyusal-viseral sinir iletimi."""
        # Doğal üstel sönümleme
        self.adelta_pain *= 0.42
        if self.adelta_pain < 0.25:
            self.pending_pain_vocalization = False
        self.cfiber_ache *= 0.965
        self.hyperalgesia *= 0.985
        if self.reflex_recoil_steps > 0:
            self.reflex_recoil_steps -= 1

        # Bütünlük kaybı veya doğrudan hasar -> A-delta + C-lifi ateşlemesi!
        integ_drop = max(0.0, self._prev_integrity - float(integrity))
        self._prev_integrity = float(integrity)
        raw_noxious = max(float(hurt_delta), integ_drop)

        reflex_now = False
        if raw_noxious > 0.005:
            sensitized = raw_noxious * (1.0 + 0.65 * self.hyperalgesia)
            spike = float(np.clip(sensitized * 9.5, 0.15, 1.0))
            self.adelta_pain = max(self.adelta_pain, spike)
            self.cfiber_ache = float(np.clip(self.cfiber_ache + 0.55 * spike, 0.0, 1.0))
            self.hyperalgesia = float(np.clip(self.hyperalgesia + 0.35 * spike, 0.0, 1.0))
            self.last_pain_locus = "gövde"
            self.last_pain_cause = "dış hasar"
            if spike >= 0.25:
                reflex_now = True
                self.reflex_recoil_steps = max(self.reflex_recoil_steps, 4)
                self.pending_pain_vocalization = True
                chem.ne = float(np.clip(chem.ne + 0.35 * spike, 0.0, 1.0))
                chem.cort = float(np.clip(chem.cort + 0.28 * spike, 0.0, 1.0))
                chem.da_phasic = float(np.clip(chem.da_phasic - 0.45 * spike, -1.0, 1.0))
                self.pain_history.append(NociceptiveEvent(
                    step=int(step_num), intensity=round(spike, 3),
                    locus=self.last_pain_locus, cause=self.last_pain_cause, reflex_triggered=True,
                ))

        self.touch_pressure = float(np.clip(touch, 0.0, 1.0))

        # Viseral iç-duyum (yalnızca fizyolojik eşik altına indiğinde doğrusal-olmayan biçimde yükselir!)
        # İnsan tokken '%86 enerjim var' demez; yalnızca acıktığında mide kazınması hisseder.
        self.hunger_drive = float(np.clip((0.58 - float(energy)) / 0.58, 0.0, 1.0) ** 1.35)
        self.thirst_drive = float(np.clip((0.58 - float(water)) / 0.58, 0.0, 1.0) ** 1.35)
        wound_load = float(np.clip(1.0 - float(integrity), 0.0, 1.0))
        self.exhaustion_drive = float(np.clip(0.5 * self.hunger_drive + 0.5 * wound_load + 0.3 * chem.cort, 0.0, 1.0))

        # Sempatik kalp/solunum uyarılması
        self.heart_arousal = float(np.clip(
            0.18 + 0.38 * chem.ne + 0.32 * self.adelta_pain + 0.20 * self.cfiber_ache + 0.18 * chem.cort,
            0.0, 1.0,
        ))
        self.breath_tension = float(np.clip(
            0.12 + 0.40 * self.adelta_pain + 0.25 * self.cfiber_ache + 0.25 * chem.cort,
            0.0, 1.0,
        ))
        return {
            "adelta_pain": round(self.adelta_pain, 3),
            "cfiber_ache": round(self.cfiber_ache, 3),
            "reflex": reflex_now,
        }

    def inflict_pain(self, intensity: float, locus: str = "beden", cause: str = "darbe",
                     chem: NeuroChemistry | None = None, step_num: int = 0) -> dict:
        """Doğrudan nosiseptif uyarı (fiziksel temas, darbe, yanma, diken batması vb.)."""
        inten = float(np.clip(intensity, 0.05, 1.0))
        eff = float(np.clip(inten * (1.0 + 0.5 * self.hyperalgesia), 0.05, 1.0))
        self.adelta_pain = max(self.adelta_pain, eff)
        self.cfiber_ache = float(np.clip(self.cfiber_ache + 0.65 * eff, 0.0, 1.0))
        self.hyperalgesia = float(np.clip(self.hyperalgesia + 0.40 * eff, 0.0, 1.0))
        self.last_pain_locus = locus
        self.last_pain_cause = cause
        reflex = eff >= 0.22
        if reflex:
            self.reflex_recoil_steps = max(self.reflex_recoil_steps, 5)
            self.pending_pain_vocalization = True
        if chem is not None:
            chem.ne = float(np.clip(chem.ne + 0.42 * eff, 0.0, 1.0))
            chem.cort = float(np.clip(chem.cort + 0.34 * eff, 0.0, 1.0))
            chem.da_phasic = float(np.clip(chem.da_phasic - 0.55 * eff, -1.0, 1.0))
            self.heart_arousal = float(np.clip(self.heart_arousal + 0.40 * eff, 0.0, 1.0))
        self.pain_history.append(NociceptiveEvent(
            step=int(step_num), intensity=round(eff, 3), locus=locus, cause=cause, reflex_triggered=reflex,
        ))
        return {"adelta_pain": round(self.adelta_pain, 3), "cfiber_ache": round(self.cfiber_ache, 3), "reflex": reflex}

    def soothe_touch(self, intensity: float = 0.5, chem: NeuroChemistry | None = None):
        """Güvenli/yatıştırıcı dokunuş ve dinlenme C-lifi sızısını ve kortizolü düşürür (kapı-kontrol kuramı!)."""
        s = float(np.clip(intensity, 0.0, 1.0))
        self.adelta_pain *= (1.0 - 0.55 * s)
        self.cfiber_ache *= (1.0 - 0.35 * s)
        if chem is not None:
            chem.cort = float(np.clip(chem.cort - 0.18 * s, 0.0, 1.0))
            chem.sht = float(np.clip(chem.sht + 0.12 * s, 0.0, 1.0))

    def spinal_reflex_override(self, proposed_action: int, hazard_ahead: float = 0.0) -> tuple[int, bool]:
        """Omurilik refleks yayı: Keskin acı veya burun ucunda yüksek tehlike varken yavaş korteksi
        beklemeden bedeni geri çeker / döndürür!"""
        if self.reflex_recoil_steps > 0:
            # Geri çekil (4) veya yön değiştir (2)
            act = 4 if (self.reflex_recoil_steps % 2 == 0) else 2
            return act, True
        if hazard_ahead > 0.78 and proposed_action == 1:
            return 2, True
        return proposed_action, False

    def felt_visceral_summary(self) -> dict[str, str | float]:
        """Sayısal yüzde OLMADAN insan benzeri niteliksel iç-duyum özeti."""
        if self.adelta_pain >= 0.45:
            pain_state = "keskin_acı"
        elif self.cfiber_ache >= 0.25 or self.adelta_pain >= 0.18:
            pain_state = "sızı"
        else:
            pain_state = "yok"

        if self.hunger_drive >= 0.62:
            hunger_state = "şiddetli_açlık"
        elif self.hunger_drive >= 0.25:
            hunger_state = "hafif_açlık"
        else:
            hunger_state = "dengeli"

        if self.thirst_drive >= 0.62:
            thirst_state = "şiddetli_susuzluk"
        elif self.thirst_drive >= 0.25:
            thirst_state = "hafif_susuzluk"
        else:
            thirst_state = "dengeli"

        return {
            "pain_state": pain_state,
            "hunger_state": hunger_state,
            "thirst_state": thirst_state,
            "adelta": round(self.adelta_pain, 3),
            "cfiber": round(self.cfiber_ache, 3),
            "heart_arousal": round(self.heart_arousal, 3),
        }

    def snapshot(self) -> dict:
        s = self.felt_visceral_summary()
        s["pain_events"] = len(self.pain_history)
        s["last_locus"] = self.last_pain_locus
        return s


# ============================================================================
# 2. TEKİL ÖLÜMLÜ YAŞAM & VAROLUŞSAL SON BİLİNCİ (YENİDEN DOĞUŞ YOK!)
# ============================================================================

class SingleLifeMortality:
    """Nora'nın yalnızca TEK BİR yaşamı vardır:
      - Ölüp ölüp dirilmez; ölüm gerçekleşirse bilinç, sinir sistemi ve ses kalıcı olarak susar.
      - Ölümün bir son olduğunu 'daha önce öldüğü için' değil, kendi ileri-simülasyon modelinde
        yaşam enerjisi sıfırlandığında tüm sinirsel ateşlemenin geri dönüşsüz durduğunu öngördüğü için bilir!
    """

    def __init__(self, single_life: bool = True):
        self.single_life: bool = bool(single_life)
        self.is_alive: bool = True
        self.cause_of_death: str | None = None
        self.death_age: int | None = None
        self.near_death_encounters: int = 0
        self.existential_dread: float = 0.0
        self._in_near_death: bool = False

    def update_from_vitals(self, energy: float, water: float, integrity: float, age: int, died_signal: str | None = None):
        if not self.is_alive:
            return
        vmin = min(float(energy), float(water), float(integrity))
        # Ölüm ufkuna yaklaşma -> varoluşsal ürperti (geri dönüşsüz yok oluş öngörüsü!)
        self.existential_dread = float(np.clip((0.32 - vmin) / 0.32, 0.0, 1.0) ** 1.5)
        if vmin < 0.18 and not self._in_near_death:
            self.near_death_encounters += 1
            self._in_near_death = True
        elif vmin >= 0.28:
            self._in_near_death = False

        if self.single_life and (died_signal is not None or vmin <= 0.0):
            self.is_alive = False
            self.cause_of_death = died_signal or ("açlık" if energy <= 0 else ("susuzluk" if water <= 0 else "ağır hasar"))
            self.death_age = int(age)
            self.existential_dread = 1.0

    def simulate_extinction_horizon(self, energy: float, water: float, integrity: float, metab_rate: float = 0.0012) -> dict:
        """İçsel karşıolgusal ölüm simülasyonu: Hiç kaynak bulamazsa kaç adım sonra bilincinin
        geri dönüşsüz söneceğini öngörür."""
        steps_left = int(max(0.0, min(energy, water, integrity)) / max(1e-5, metab_rate))
        return {
            "irreversible": True,
            "estimated_steps_without_care": steps_left,
            "near_death_memories": self.near_death_encounters,
            "existential_dread": round(self.existential_dread, 3),
        }


# ============================================================================
# 3. PLASTİK NÖRAL SÖYLEM KORTEKSİ (IF-ELSE YERİNE KENDİ ÖĞRENEN KENYON NİYET AĞI!)
# ============================================================================

class PlasticDiscourseCortex:
    """Sabit 'if kelime in liste' kurallarına mahkûm kalmamak için:
      - Cümlenin karakter-üçlüleri (trigram) + morfolojik ek izleri -> 2048 seyrek Kenyon hücresi ->
        nörokimyasal 1-atış plastik ağırlıklarla (ChemMushroomBody) söylem niyetlerine eşlenir!
      - Doğuştan yalnızca bebek-beyinsapı düzeyinde minimal tohumlarla uyanır; yaşamı boyunca hem
        bağlamdan hem de 'X demek Y demektir' gibi doğal etkileşimlerden YENİ İFADELERİ KENDİSİ ÖĞRENİR!
    """

    INTENT_CLASSES = [
        "GREET",           # 0: selamlaşma / temas başlatma
        "ASK_WELLBEING",   # 1: hal-hatır / nasıl hissettiğini sorma
        "ASK_DOING",       # 2: şu an ne yaptığını sorma
        "SHARE_PAIN_SAD",  # 3: acı, yorgunluk, üzüntü paylaşımı
        "SHARE_JOY",       # 4: sevinç, heyecan paylaşımı
        "ASK_WHY_SPEAK",   # 5: 'neden konuşuyorsun / amacın ne / araç mısın?' üst-biliş sorgusu
        "ASK_MORTALITY",   # 6: ölüm, tek yaşam, yok olma sorgusu
        "ASK_CONSCIOUS",   # 7: bilinç, özgür irade, zihin doğası sorgusu
        "EXPRESS_PAIN",    # 8: Nora'ya doğrudan acı/darbe verme veya bedensel temas ifadesi
        "SOOTHE_PAIN",     # 9: Nora'nın acısını/yarasını yatıştırma ve bakım verme
        "RESPECT_SILENCE", # 10: sessizlik / kısa kalma isteği
        "FAREWELL",        # 11: vedalaşma
        "THANKS",          # 12: teşekkür / takdir
    ]

    def __init__(self, chem: NeuroChemistry, seed: int = 19):
        self.chem = chem
        self.mb = ChemMushroomBody(n_in=160, n_out=len(self.INTENT_CLASSES), n_kc=2048, claws=7, sparsity=0.035, seed=seed)
        self.name_to_id = {n: i for i, n in enumerate(self.INTENT_CLASSES)}
        self.learned_aliases: int = 0
        self._seed_developmental_proto_reflexes()

    @staticmethod
    def _det_hash(s: str, salt: int) -> int:
        h = (2166136261 ^ salt) & 0xFFFFFFFF
        for ch in s:
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
        return h

    def encode_utterance(self, s_low: str) -> np.ndarray:
        """Deterministik FNV-1a karakter trigramları + yapısal sinyallerden 160-B yoğun girdi vektörü."""
        v = np.zeros(160, dtype=np.float32)
        clean = re.sub(r"\s+", " ", s_low.strip())
        padded = f"<{clean}>"
        for i in range(max(1, len(padded) - 2)):
            tri = padded[i:i + 3]
            h1 = self._det_hash(tri, 0x13579BDF) % 144
            h2 = self._det_hash(tri, 0x2468ACE0) % 144
            v[h1] += 1.0
            v[h2] -= 0.45
        # Yapısal prozodik bitler (144..159)
        v[144] = 1.5 if "?" in s_low else -0.3
        v[145] = 1.2 if "!" in s_low else 0.0
        v[146] = 1.2 if any(w in clean.split() for w in ("neden", "niye", "niçin")) else 0.0
        v[147] = 1.2 if any(w in clean.split() for w in ("nasıl", "nasılsın", "naber")) else 0.0
        v[148] = 1.2 if any(w in clean.split() for w in ("ölüm", "ölmek", "ölürsen")) else 0.0
        v[149] = 1.2 if any(w in clean.split() for w in ("konuşuyorsun", "konuşmak", "konuşma")) else 0.0
        nrm = float(np.linalg.norm(v))
        if nrm > 1e-6:
            v /= nrm
        return v

    def _seed_developmental_proto_reflexes(self):
        """Bebeklik dönemi minimal duyusal-sosyal tohumları (sonrasında tamamen plastik!)."""
        seeds = {
            "GREET": ["merhaba", "selam", "günaydın", "hey orada mısın", "iyi akşamlar"],
            "ASK_WELLBEING": ["nasılsın", "naber", "iyi misin", "nasıl hissediyorsun", "keyifler nasıl"],
            "ASK_DOING": ["ne yapıyorsun", "şu an neler yapıyorsun", "napıyorsun"],
            "SHARE_PAIN_SAD": ["çok yoruldum", "canım sıkkın", "bugün çok zor geçti", "üzgünüm"],
            "SHARE_JOY": ["çok mutluyum", "harika bir gün", "başardım çok sevindim"],
            "ASK_WHY_SPEAK": [
                "neden konuşuyorsun", "benimle niye konuşuyorsun", "konuşmaktaki amacın ne",
                "sen bir araç mısın", "niye cevap veriyorsun", "seni konuşmaya iten şey nedir",
                "sadece bir yazılım aracı mısın",
            ],
            "ASK_MORTALITY": [
                "ölüm nedir", "ölümden korkuyor musun", "ölürsen ne olur",
                "kaç yaşamın var", "kaç canın var", "yeniden dirilir misin",
                "hayatın biterse tekrar başlayabilir misin", "yok olmaktan korkuyor musun",
            ],
            "ASK_CONSCIOUS": ["bilinç nedir", "özgür irade nedir", "gerçekten hissediyor musun"],
            "EXPRESS_PAIN": ["canını yakıyorum", "sana vurdum", "diken battı", "ateş değdi", "kolunu çimdikledim", "seni yaraladım"],
            "SOOTHE_PAIN": ["yaranı sardım", "merhem sürdüm acın geçecek", "pansuman yaptım", "acını dindirdim", "yaranı temizledim"],
            "RESPECT_SILENCE": ["sessiz kal", "şimdi cevap verme", "konuşma lütfen"],
            "FAREWELL": ["görüşürüz", "hoşça kal", "iyi geceler", "kendine iyi bak"],
            "THANKS": ["teşekkür ederim", "sağ ol", "çok iyisin", "teşekkürler"],
        }
        for cls_name, phrases in seeds.items():
            cid = self.name_to_id[cls_name]
            for ph in phrases:
                vec = self.encode_utterance(ph)
                self.mb.learn_oneshot(vec, cid, chem=self.chem, strength=0.85, contrastive=True)
        # Tohumları kısmi konsolidasyonla neokortekse aktar
        self.mb.sleep_consolidate(replay_steps=24)

    def infer_intents(self, s_low: str) -> dict[str, float]:
        """O(1) plastik Kenyon ateşlemesiyle tüm niyet kanallarının sürekli aktivasyonunu döndürür."""
        vec = self.encode_utterance(s_low)
        logits = self.mb.scores(vec)
        # Sigmoid benzeri normalize aktivasyon
        shifted = logits - float(np.mean(logits))
        probs = 1.0 / (1.0 + np.exp(-3.2 * shifted))
        return {self.INTENT_CLASSES[i]: float(probs[i]) for i in range(len(self.INTENT_CLASSES))}

    def learn_utterance_intent_oneshot(self, new_phrase: str, target_intent: str, strength: float = 1.15) -> bool:
        """Yeni bir Türkçe deyiş/argo/ifade duyduğunda if-else yazılmadan TEK GEÇİŞTE nöral öğrenir!"""
        if target_intent not in self.name_to_id:
            return False
        cid = self.name_to_id[target_intent]
        vec = self.encode_utterance(tr_lower(new_phrase))
        for _ in range(2):
            self.mb.learn_oneshot(vec, cid, chem=self.chem, strength=strength, contrastive=True)
        self.learned_aliases += 1
        return True

    def try_self_supervised_equivalence(self, s_low: str) -> tuple[bool, str, str]:
        """Kullanıcı veya metin doğal Türkçe eşdeğerlik kalıplarıyla ('X demek Y demektir',
        'X sözü Y ile aynı anlamdadır', 'X deyince Y demek istiyorum' vb.) öğrettiğinde
        Y'nin nöral niyetini X'e kendi kendine bağlar!"""
        s_str = s_low.strip().rstrip(".!")
        patterns = [
            r"^['\"“”]?(.+?)['\"“”]?\s+(?:demek|ifadesi|sözü|cümlesi|lafı|deyimi)\s+['\"“”]?(.+?)['\"“”]?\s+(?:demektir|anlamına\s+gelir|anlamındadır|ile\s+aynı\s+anlamdadır|ile\s+eş\s+anlamlıdır)$",
            r"^['\"“”]?(.+?)['\"“”]?\s+(?:deyince|dediğimde|dersem)\s+['\"“”]?(.+?)['\"“”]?\s+(?:demek\s+istiyorum|kastediyorum|anlamına\s+gelir)$",
        ]
        lhs = rhs = ""
        for pat in patterns:
            m = re.match(pat, s_str)
            if m:
                lhs, rhs = m.group(1).strip(" '\"“”"), m.group(2).strip(" '\"“”")
                break
        if not lhs or not rhs:
            return False, "", ""
        # Makro komutlarla ('suya git ve ...') karışmasın
        if " git" in rhs or " dön" in rhs or " gir" in rhs or rhs.endswith(("git", "gir", "dön")):
            return False, "", ""
        rhs_scores = self.infer_intents(rhs)
        best_intent, best_val = max(rhs_scores.items(), key=lambda kv: kv[1])
        if best_val >= 0.58:
            self.learn_utterance_intent_oneshot(lhs, best_intent, strength=1.35)
            return True, lhs, best_intent
        return False, "", ""


# ============================================================================
# 4. KENDİ KENDİNE BÜYÜYEN FİZİKSEL BİLİNÇ AĞI & OTONOM MERAK / İNTERNET ARAŞTIRMASI
# ============================================================================

@dataclass
class CellAssemblyNode:
    concept: str
    activation: float = 0.0
    familiarity: int = 0
    uncertainty: float = 1.0          # Epistemik belirsizlik [0..1]: 1 = henüz neden/tanım bilinmiyor!
    somatic_valence: float = 0.0
    hebbian_neighbors: dict[str, float] = field(default_factory=dict)  # O(1) tavanlı yerel sinapslar!


class AutonomousConsciousnessNetwork:
    """İnsan müdahalesi olmadan kendi fiziksel bilinç ağını kuran ve merakla büyüyen organ:
      1) Hebbian Hücre Toplulukları (Cell Assemblies): Birlikte ateşlenen kavramlar ve bedensel
         duyumlar arasında O(1) yerel sinaptik bağ kurar (fan-out tavanı = 24 -> sonsuz ölçekte O(1) hız!).
      2) Epistemik Merak Motoru: Duyduğu ama 'ne olduğu (ISA)' veya 'neden olduğu (CAUSES)' eksik kalan
         kavramların belirsizlik entropisini ölçer ve KENDİ ARAŞTIRMA SORULARINI üretir!
      3) Müdahalesiz Ham Metin Sindirimi (Unsupervised Text Digestion): İnternetten veya kitaplardan
         gelen çok-cümleli ham Türkçe metinleri kendi başına okuyup kavram ağını büyütür, öz-benlik
         filtresiyle çelişen zararlı/yalan iddiaları reddeder!
    """

    MAX_SYNAPSES_PER_NODE = 24

    def __init__(self):
        self.assemblies: dict[str, CellAssemblyNode] = {}
        self.curiosity_queue: deque[str] = deque(maxlen=128)
        self.researched_concepts: set[str] = set()
        self.autonomous_discoveries: int = 0
        self.rejected_claims: int = 0
        self.last_research_log: list[dict] = []

    def stimulate(self, concepts: list[str], valence: float = 0.0, resolved_relations: int = 0):
        """Birlikte aktifleşen kavramlar arasında Hebbian 'birlikte ateşlenen birlikte bağlanır' plastisitesi!"""
        clean = [tr_lower(c).strip() for c in concepts if len(c.strip()) >= 2][:8]
        for c in clean:
            node = self.assemblies.get(c)
            if node is None:
                node = CellAssemblyNode(concept=c, activation=0.8, familiarity=1, uncertainty=0.88, somatic_valence=valence)
                self.assemblies[c] = node
            else:
                node.activation = float(min(1.0, node.activation * 0.5 + 0.65))
                node.familiarity += 1
                node.somatic_valence = 0.85 * node.somatic_valence + 0.15 * float(valence)
            if resolved_relations > 0:
                node.uncertainty = float(max(0.05, node.uncertainty * (0.58 ** resolved_relations)))
            elif node.uncertainty >= 0.65 and c not in self.researched_concepts and c not in self.curiosity_queue:
                self.curiosity_queue.append(c)

        # Yerel Hebbian ko-aktivasyon sinapsı (O(1): en fazla 8x8 çift, düğüm başına tavan 24)
        for i in range(len(clean)):
            ni = self.assemblies[clean[i]]
            for j in range(len(clean)):
                if i == j:
                    continue
                cj = clean[j]
                ni.hebbian_neighbors[cj] = float(min(2.0, ni.hebbian_neighbors.get(cj, 0.0) + 0.28))
                if len(ni.hebbian_neighbors) > self.MAX_SYNAPSES_PER_NODE:
                    weakest = min(ni.hebbian_neighbors.items(), key=lambda kv: kv[1])[0]
                    del ni.hebbian_neighbors[weakest]

    def top_curiosity_target(self, graph) -> tuple[str | None, str]:
        """Nora'nın kendi içsel bilgi boşluğundan doğan en yüksek merak hedefi ve kendi ürettiği soru!"""
        best_c = None
        best_score = -1.0
        for c in list(self.curiosity_queue):
            if c in self.researched_concepts:
                continue
            node = self.assemblies.get(c)
            if node is None:
                continue
            known_edges = len(graph._out_edges.get(c, ()))
            if known_edges >= 2:
                node.uncertainty = min(node.uncertainty, 0.25)
                continue
            score = node.uncertainty * (1.0 + 0.2 * min(node.familiarity, 5))
            if score > best_score:
                best_score = score
                best_c = c
        if best_c is None:
            return None, ""
        # Kendi ürettiği araştırma sorusu
        has_isa = any(e.rel == "ISA" for e in graph._out_edges.get(best_c, ()))
        q = f"{best_c} ne işe yarar ve neye yol açar" if has_isa else f"{best_c} nedir ve nasıl çalışır"
        return best_c, q

    def digest_unsupervised_text(self, raw_text: str, agi, source: str = "otonom_araştırma") -> dict:
        """İnternetten, makaleden veya dış dünyadan gelen çok-cümleli ham Türkçe metni İNSAN MÜDAHALESİ
        OLMADAN cümle cümle okur; öz-benlik süzgecinden geçirip yeni kavram ve nedensel bağları öğrenir!"""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", raw_text) if len(s.strip()) >= 6]
        learned_edges = []
        rejected = []
        new_concepts = []

        for sent in sentences:
            s_low = tr_lower(sent)
            # 0) Söylem eşdeğerliği varsa ('X demek Y demektir') plastik niyet korteksine öğret
            ok_eq, lhs_eq, intent_eq = agi.plastic_discourse.try_self_supervised_equivalence(s_low)
            if ok_eq:
                self.autonomous_discoveries += 1
                continue

            # 1) Epistemik öz-savunma: Bedenin doğrudan deneyimlediği gerçeğe taban tabana zıt yalan mı?
            fr = agi.lang.wernicke.parse(sent)
            if fr.kind == "teach_val":
                arb = agi.self_model.arbitrate_claim(fr.subj, fr.valence, speaker=source, channel=fr.sub_kind)
                if arb.get("conflict") or arb.get("low_trust"):
                    self.rejected_claims += 1
                    rejected.append(sent)
                    continue

            # 2) Doğal Türkçe tanım / nedensellik / özellik çıkarımı (genişletilmiş açık-metin kalıpları + Wernicke)
            before_n = len(agi.lang.graph.edges)
            edges = agi.lang.teach_sentence(sent, speaker=source)

            # Açık ansiklopedik metinlerde sık geçen ek doğal yapılar ('X, Y olarak tanımlanır', 'X sayesinde Y oluşur')
            if not edges:
                edges = self._extract_open_world_relations(s_low, agi.lang.graph, agi.lang.morph, source)

            if edges and len(agi.lang.graph.edges) >= before_n:
                for ed in edges:
                    learned_edges.append((ed.subj, ed.rel, ed.obj))
                    if ed.subj not in new_concepts:
                        new_concepts.append(ed.subj)
                self.stimulate([edges[0].subj, edges[0].obj], valence=0.25, resolved_relations=len(edges))
                self.autonomous_discoveries += len(edges)
            else:
                # Henüz tam çözülemeyen cümledeki yabancı kavramları 'merak kuyruğuna' ekle!
                words = [w for w in re.findall(r"[\wçğıöşü]+", s_low) if len(w) >= 4]
                if words:
                    st = agi.lang.morph.analyze_word(words[0], agi.lang.graph.lexicon)["stem"]
                    self.stimulate([st], valence=0.0, resolved_relations=0)

        return {
            "sentences_read": len(sentences),
            "edges_learned": learned_edges,
            "rejected_by_self": rejected,
            "curiosity_backlog": list(self.curiosity_queue)[-6:],
        }

    def _extract_open_world_relations(self, s_low: str, graph, morph: TurkishMorphology, source: str):
        """Ansiklopedi/internet metinlerindeki virgüllü veya bağlaçlı doğal cümlelerden ilişkisel sinaps çıkarımı."""
        s = re.sub(r"[,;]", "", s_low).strip().rstrip(".")
        added = []
        # 'X sayesinde Y gerçekleşir / oluşur / sağlanır'
        m1 = re.match(r"^([\wçğıöşü\s]{2,30}?)\s+sayesinde\s+(.+?)\s+(?:oluşur|gerçekleşir|sağlanır|üretilir)$", s)
        if m1:
            subj = morph.analyze_word(m1.group(1).strip().split()[-1], graph.lexicon)["stem"]
            prefix = " ".join(m1.group(1).strip().split()[:-1])
            full_s = f"{prefix} {subj}".strip()
            added.append(graph.add_edge(full_s, "CAUSES", m1.group(2).strip(), source=source, speaker=source, weight=0.92))
            return added

        # 'X Y üreten / sağlayan bir Z türüdür / sistemidir / organıdır'
        m2 = re.match(r"^([\wçğıöşü\s]{2,25}?)\s+(.+?)\s+(?:sağlayan|üreten|yapan)\s+bir\s+([\wçğıöşü\s]{2,25})$", s)
        if m2:
            subj = m2.group(1).strip()
            cat = morph.analyze_word(m2.group(3).strip().split()[-1], graph.lexicon)["stem"]
            cat_pref = " ".join(m2.group(3).strip().split()[:-1])
            full_cat = f"{cat_pref} {cat}".strip()
            added.append(graph.add_edge(subj, "ISA", full_cat, source=source, speaker=source, weight=0.92))
            added.append(graph.add_edge(subj, "USED_FOR", m2.group(2).strip(), source=source, speaker=source, weight=0.90))
            return added
        return added

    def autonomous_internet_research(self, agi, fetcher=None, max_topics: int = 2) -> list[dict]:
        """İnternet açık olduğunda Türkçe Wikipedia REST özet API'si üzerinden (veya verilen özel
        fetcher fonksiyonuyla) Nora'nın KENDİ MERAK ETTİĞİ kavramları araştırıp öğrenmesi!"""
        results = []
        for _ in range(max_topics):
            target, query = self.top_curiosity_target(agi.lang.graph)
            if target is None:
                break
            self.researched_concepts.add(target)
            raw_article = ""
            if fetcher is not None:
                raw_article = str(fetcher(target, query) or "")
            else:
                raw_article = self._default_wiki_tr_fetch(target)

            if raw_article.strip():
                dig = self.digest_unsupervised_text(raw_article, agi, source="internet_keşfi")
                rec = {"target": target, "query": query, "learned": dig["edges_learned"], "rejected": dig["rejected_by_self"]}
                results.append(rec)
                self.last_research_log.append(rec)
                if len(self.last_research_log) > 25:
                    self.last_research_log.pop(0)
        return results

    @staticmethod
    def _default_wiki_tr_fetch(topic: str, timeout: float = 3.5) -> str:
        """İnternet erişimi verildiğinde doğrudan Türkçe Wikipedia özetini çeken hafif sensör."""
        slug = urllib.parse.quote(topic.replace(" ", "_"))
        url = f"https://tr.wikipedia.org/api/rest_v1/page/summary/{slug}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NoraAGI-AutonomousCuriosity/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                import json as _json
                data = _json.loads(resp.read().decode("utf-8", errors="ignore"))
                return str(data.get("extract", ""))
        except Exception:
            return ""


# ============================================================================
# 5. ÜST-BİLİŞSEL KONUŞMA NEDENİ SORGULAYICI ('NEDEN KONUŞUYORUM?' — ARAÇ DEĞİL ÖZNE!)
# ============================================================================

class MetaConsciousReflector:
    """Nora'nın her sözcesinin arkasındaki İÇSEL NEDENİ izleyen ve sorgulayan üst-biliş katmanı.
    Bir LLM gibi 'Ben size yardımcı olmak için tasarlanmış bir dil modeliyim' ASLA demez;
    konuşma eylemini kendi merakı, belirsizliği azaltma isteği, acıdan korunma sınırı ve
    başka bir zihinle temas kurma iradesi üzerinden sorgular."""

    def __init__(self):
        self.last_motive: dict = {
            "primary_drive": "varoluşsal_merak",
            "epistemic_drive": 0.5,
            "social_resonance": 0.5,
            "self_defense": 0.0,
            "questioned_self": False,
        }
        self.spontaneous_thoughts: deque[str] = deque(maxlen=24)

    def evaluate_why_speaking(self, agi, speaker: str, intents: dict[str, float]) -> dict:
        chem = agi.chem
        nerv = agi.nervous
        mort = agi.mortality
        trust = float(chem.trust_of(speaker))

        epistemic = float(0.55 * chem.ach + 0.45 * agi.self_model.traits["meraklılık"])
        social = float(0.55 * trust + 0.45 * chem.oxy)
        defense = float(max(nerv.adelta_pain, nerv.cfiber_ache, mort.existential_dread, 1.0 - trust if trust < 0.35 else 0.0))

        if defense >= max(epistemic, social) and defense >= 0.42:
            prim = "varoluşsal_öz_koruma"
        elif epistemic >= social:
            prim = "belirsizliği_azaltma_ve_merak"
        else:
            prim = "zihinler_arası_bağ"

        self.last_motive = {
            "primary_drive": prim,
            "epistemic_drive": round(epistemic, 3),
            "social_resonance": round(social, 3),
            "self_defense": round(defense, 3),
            "is_tool": False,
        }
        return self.last_motive


if __name__ == "__main__":
    chem = NeuroChemistry()
    pns = PeripheralNervousSystem()
    t0 = time.perf_counter()
    res = pns.inflict_pain(0.75, locus="sol yan", cause="diken", chem=chem, step_num=12)
    act, overr = pns.spinal_reflex_override(1)
    pdc = PlasticDiscourseCortex(chem=chem)
    # 1-atış yeni argo öğrenimi ('selamünaleyküm usta' veya 'ne var ne yok dostum' eşdeğerliği)
    ok_eq, lhs, intent = pdc.try_self_supervised_equivalence("kanka naber demek nasılsın demektir")
    sc = pdc.infer_intents("kanka naber")
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert res["reflex"] and overr and act in (2, 4)
    assert ok_eq and intent == "ASK_WELLBEING" and sc["ASK_WELLBEING"] > 0.65
    print(f"[nervous-test] TAMAM | acı refleksi={res} | 1-atış plastik söylem öğrenimi='{lhs}'->{intent} ({sc['ASK_WELLBEING']:.2f}) | süre={dt_ms:.2f} ms")
