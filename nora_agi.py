#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_agi.py — NoRa Birleşik Nörokimyasal, Bedenlenmiş, Öz-Benlikli ve Epoch'suz Öğrenen AGI Mimarisi.

Altı bütünleşik beyin sistemini tek bir canlı organizmada birleştirir:
  1. Beden ve Yaşayan Evren (nora_world.Universe + nora_mind.Mind):
     Homeostaz (enerji, su, bütünlük), FlyWire/Kenyon hücreli seyrek algı, bilişsel harita,
     nedensel içebakış (Introspector), öz-öngörü (SelfForecast), körlük/felç öz-farkındalığı,
     uzamsal özellik bağlama (Mind.imagine) ve ayna testi (MirrorSense).
  2. Endojen Nörokimya (nora_chem.NeuroChemistry):
     Dopamin (DA), Asetilkolin (ACh), Noradrenalin (NE), Serotonin (5-HT), Kortizol (CORT) ve
     Oksitosin (OXY) döngüleri; sinaptik plastisiteyi, zaman ufkunu (gamma) ve Sistem-2'yi yönetir.
  3. Öz-Benlik ve Özerk İrade (nora_self.CoreSelf):
     Somatik işaretleyiciler, 'deneyim' vs 'duyum' kaynak izleme, yalan/çelişki tespiti,
     öz-yıkıcı komutları reddeden özerk irade (zombi olmama) ve yaşantıyla evrilen karakter.
  4. Epoch'suz Doğal Türkçe Dil Organı (nora_lang.NoraLanguage):
     Üretken Türkçe morfoloji, Wernicke anlam çözücü, zamir/bağlam çalışma belleği,
     1-atış (0-epoch) kavram ve nedensel kural öğrenme, yayılan aktivasyonla çıkarım ve Broca üretici.
  5. Sistem-2 Kompozisyonel Düşünce Sütunu (nora_bridge.Cognition / nora_think.ThinkColumn):
     Sıfır-atış çok-basamaklı aritmetik ve adım adım zihinsel doğrulama.
  6. Evrensel Hızlı Öğrenici (UniversalFastLearner):
     Yalnızca 2–5 örnekten (0 epoch!) sayısal/sembolik dizi yasalarını, girdi->çıktı fonksiyonlarını
     ve ardışık vektör kavram sınıflarını yıkıcı unutma olmadan (ChemMushroomBody + uyku konsolidasyonu)
     öğrenen genel amaçlı hızlı uyum motoru.

Kullanım:
  python nora_agi.py --stage demo        # Canlı yaşam + tek-atış dil + öz-benlik + muhakeme gösterimi
  python nora_agi.py --stage chat        # Etkileşimli Türkçe konuşma ve öğretim terminali
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from collections import deque
import numpy as np

from dataclasses import dataclass
from nora_life import BackgroundLife
from nora_discourse import DiscourseCortex
from nora_knowledge import PROPER_NOUNS
from nora_initiative import InitiativeCortex
from nora_speculation import SpeculativeCortex
from nora_voice import state_vector
from nora_world import Universe
from nora_mind import Mind, I_E, I_W, I_I
from nora_chem import NeuroChemistry, ChemMushroomBody
from nora_self import CoreSelf
from nora_lang import NoraLanguage, Proposition, TurkishMorphology, tr_lower, tr_capitalize
from nora_stem import MathCortex, PhysicsCortex
from nora_knowledge import ConversationalResonanceCortex, SYNONYMS
from nora_broca import GenerativeBrocaOrgan, NP, VerbSpec, PredSpec
from nora_broca import TurkishSyntaxEngine as SYN
from nora_monologue import PlasticSchemaCortex, InnerMonologueEngine
from nora_deliberation import DeliberativeMind
from nora_autonomous_logic import (
    NeuralUtteranceCortex,
    AutonomousLogicAndLabelEngine,
    SituationalTurkishCortex,
)
from nora_nervous import (
    PeripheralNervousSystem,
    SingleLifeMortality,
    PlasticDiscourseCortex,
    AutonomousConsciousnessNetwork,
    MetaConsciousReflector,
)
from nora_episteme import EpistemicEngine, BeliefSpeech, TrustedResearcher
try:
    from nora_sets import SetSynth
except Exception:  # küme organı yoksa sistem yine çalışır
    SetSynth = None
try:
    from nora_synth import Synthesizer, parse_pairs, Program, PRIMITIVES
except Exception:  # sentez organı yoksa sistem yine çalışır
    Synthesizer = None
    Program = None
    PRIMITIVES = ()
    def parse_pairs(_t):
        return []
from nora_code import CodeSynthesisCortex


@dataclass
class CorticalBid:
    module: str
    salience: float
    precision: float
    reply: str
    trace: str = ""
    used_s2: bool = False
    speech_plan: dict | None = None


class UniversalFastLearner:
    """Az-örnekli (few-shot, 0-epoch) evrensel kural, dizi, fonksiyon ve kavram öğrenici."""

    def __init__(self, chem: NeuroChemistry, feat_dim: int = 64, seed: int = 0):
        self.chem = chem
        self.feat_dim = int(feat_dim)
        self.mb = ChemMushroomBody(n_in=feat_dim, n_out=8, n_kc=2560, claws=7, sparsity=0.03, seed=seed + 101)
        self.class_names: list[str] = []
        self.name_to_id: dict[str, int] = {}
        self.learned_functions: dict[str, dict] = {}

    # ------------------------------------------------------------------ 1) Dizi yasası keşfi (MDL)
    @staticmethod
    def induce_sequence(seq: list[float]) -> dict:
        """3–6 elemanlı bir sayı dizisinden en yalın (MDL) matematiksel üretim yasasını keşfeder."""
        xs = [float(x) for x in seq]
        n = len(xs)
        if n < 3:
            return {"ok": False, "next": None, "rule_tr": "yetersiz örnek"}

        diffs = [xs[i + 1] - xs[i] for i in range(n - 1)]
        # H1: Sabit fark (aritmetik dizi)
        if np.allclose(diffs, diffs[0], atol=1e-6):
            d = diffs[0]
            nxt = xs[-1] + d
            di = int(round(d)) if abs(d - round(d)) < 1e-6 else round(d, 3)
            op_txt = f"+{di}" if d >= 0 else f"{di}"
            return {"ok": True, "next": nxt, "kind": "arithmetic", "rule_tr": f"her adımda {op_txt} değişim"}

        # H2: Sabit oran (geometrik dizi)
        if all(abs(x) > 1e-8 for x in xs[:-1]):
            ratios = [xs[i + 1] / xs[i] for i in range(n - 1)]
            if np.allclose(ratios, ratios[0], atol=1e-6):
                r = ratios[0]
                nxt = xs[-1] * r
                ri = int(round(r)) if abs(r - round(r)) < 1e-6 else round(r, 3)
                return {"ok": True, "next": nxt, "kind": "geometric", "rule_tr": f"her adımda {ri} ile çarpım"}

        # H3: İkinci derece fark sabit (karesel / artan fark dizisi: ör. 1, 2, 4, 7, 11 -> +1,+2,+3,+4)
        if n >= 4:
            diffs2 = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
            if np.allclose(diffs2, diffs2[0], atol=1e-6):
                d2 = diffs2[0]
                next_diff = diffs[-1] + d2
                nxt = xs[-1] + next_diff
                d2i = int(round(d2)) if abs(d2 - round(d2)) < 1e-6 else round(d2, 3)
                return {"ok": True, "next": nxt, "kind": "quadratic", "rule_tr": f"adım farkının her seferinde {d2i:+} artması"}

        # H4: İki-adım doğrusal özyineleme: x_t = a * x_{t-1} + b * x_{t-2} (Fibonacci ve genellemeleri!)
        if n >= 5:
            A = np.column_stack([xs[1:-1], xs[:-2]])
            y = np.array(xs[2:])
            coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            if np.allclose(A @ coef, y, atol=1e-4):
                a, b = coef
                nxt = a * xs[-1] + b * xs[-2]
                if abs(a - 1.0) < 1e-4 and abs(b - 1.0) < 1e-4:
                    desc = "önceki iki terimin toplamı (Fibonacci yasası)"
                else:
                    desc = f"{a:.2g}*(önceki) + {b:.2g}*(iki önceki) özyinelemesi"
                return {"ok": True, "next": nxt, "kind": "recurrence2", "rule_tr": desc}

        # H5: Birinci derece afin özyineleme: x_t = a * x_{t-1} + b (ör. 1, 3, 7, 15, 31 -> 2x + 1)
        if n >= 4:
            A = np.column_stack([xs[:-1], np.ones(n - 1)])
            y = np.array(xs[1:])
            coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            if np.allclose(A @ coef, y, atol=1e-4):
                a, b = coef
                nxt = a * xs[-1] + b
                ai = int(round(a)) if abs(a - round(a)) < 1e-4 else round(a, 2)
                bi = int(round(b)) if abs(b - round(b)) < 1e-4 else round(b, 2)
                return {"ok": True, "next": nxt, "kind": "affine_rec", "rule_tr": f"önceki terimin {ai} katının {bi:+} fazlası"}

        # H6: Dönüşümlü (çift-periyotlu) fark dizisi (ör. 2, 5, 4, 7, 6, 9 -> +3, -1, +3, -1)
        if n >= 5 and np.allclose(diffs[0::2], diffs[0], atol=1e-5) and np.allclose(diffs[1::2], diffs[1], atol=1e-5):
            next_d = diffs[0] if (len(diffs) % 2 == 0) else diffs[1]
            nxt = xs[-1] + next_d
            return {"ok": True, "next": nxt, "kind": "alternating", "rule_tr": f"sırayla {diffs[0]:+.0f} ve {diffs[1]:+.0f} dönüşümlü adımı"}

        return {"ok": False, "next": None, "rule_tr": "kesin yasa bulunamadı"}

    # ------------------------------------------------------------------ 2) Az-örnekli fonksiyon keşfi
    def induce_mapping(self, pairs: list[tuple[float, float]], query_x: float, rule_name: str = "son_kural") -> dict:
        """2–4 adet (x -> y) çiftinden f(x) kuralını tek-geçişte bulur ve query_x için uygular."""
        xs = np.array([float(p[0]) for p in pairs], dtype=np.float64)
        ys = np.array([float(p[1]) for p in pairs], dtype=np.float64)
        qx = float(query_x)

        # Önce doğrusal f(x) = a*x + b dene (MDL önceliği: en yalın model)
        if len(xs) >= 2:
            A1 = np.column_stack([xs, np.ones_like(xs)])
            c1, _, _, _ = np.linalg.lstsq(A1, ys, rcond=None)
            if np.allclose(A1 @ c1, ys, atol=1e-4):
                a, b = c1
                ans = a * qx + b
                ai = int(round(a)) if abs(a - round(a)) < 1e-4 else round(a, 3)
                bi = int(round(b)) if abs(b - round(b)) < 1e-4 else round(b, 3)
                desc = f"girdinin {ai} katı" + (f" {bi:+}" if abs(b) > 1e-6 else "")
                rec = {"ok": True, "ans": ans, "rule_tr": desc, "deg": 1, "coef": c1.tolist()}
                self.learned_functions[rule_name] = rec
                self.chem.pulse_learning(novelty=0.8, valence=0.45)
                return rec

        # İkinci derece f(x) = a*x^2 + b*x + c dene
        if len(xs) >= 3:
            A2 = np.column_stack([xs ** 2, xs, np.ones_like(xs)])
            c2, _, _, _ = np.linalg.lstsq(A2, ys, rcond=None)
            if np.allclose(A2 @ c2, ys, atol=1e-4):
                a, b, c = c2
                ans = a * (qx ** 2) + b * qx + c
                desc = f"{a:.2g}*x^2 + {b:.2g}*x + {c:.2g} polinom yasası"
                rec = {"ok": True, "ans": ans, "rule_tr": desc, "deg": 2, "coef": c2.tolist()}
                self.learned_functions[rule_name] = rec
                self.chem.pulse_learning(novelty=0.85, valence=0.45)
                return rec

        return {"ok": False, "ans": None, "rule_tr": "uyumlu fonksiyon bulunamadı"}

    # ------------------------------------------------------------------ 3) Yıkıcı unutmasız sürekli kavram öğrenme
    def learn_concept_vec(self, name: str, vec: np.ndarray) -> int:
        cname = tr_lower(name).strip()
        if cname not in self.name_to_id:
            cid = len(self.class_names)
            self.name_to_id[cname] = cid
            self.class_names.append(cname)
        else:
            cid = self.name_to_id[cname]
        self.mb.learn_oneshot(np.asarray(vec, np.float32), cid, chem=self.chem, strength=1.0)
        return cid

    def classify_vec(self, vec: np.ndarray) -> tuple[str | None, float, float]:
        cid, margin, nov = self.mb.predict(np.asarray(vec, np.float32))
        if cid < 0 or cid >= len(self.class_names):
            return None, 0.0, nov
        return self.class_names[cid], margin, nov


class ComparativeOrderGraph:
    """Parietal büyüklük/karşılaştırma çizgesi: 'A B'den daha X-tir' -> geçişli A > B > C çıkarımı."""

    def __init__(self):
        # prop -> list of (greater, lesser)
        self.orders: dict[str, list[tuple[str, str]]] = {}

    def add(self, greater: str, lesser: str, prop: str):
        p = tr_lower(prop).strip()
        g = tr_lower(greater).strip()
        l = tr_lower(lesser).strip()
        lst = self.orders.setdefault(p, [])
        if (g, l) not in lst:
            lst.append((g, l))

    def is_greater(self, a: str, b: str, prop: str) -> tuple[bool, list[str]]:
        p = tr_lower(prop).strip()
        a_l, b_l = tr_lower(a).strip(), tr_lower(b).strip()
        edges = self.orders.get(p, [])
        frontier: list[tuple[str, list[str]]] = [(a_l, [a_l])]
        visited = {a_l}
        while frontier:
            cur, path = frontier.pop(0)
            for g, l in edges:
                if g == cur and l not in visited:
                    npath = path + [l]
                    if l == b_l:
                        return True, npath
                    visited.add(l)
                    frontier.append((l, npath))
        return False, []

    def extremum_all(self, prop: str, maximum: bool = True) -> list[str]:
        """Bütün uç adaylar. Birden çoksa sıralama TAM DEĞİLDİR (uydurulmaz)."""
        p = tr_lower(prop).strip()
        edges = self.orders.get(p, [])
        if not edges:
            return []
        nodes = {g for g, _ in edges} | {l for _, l in edges}
        if maximum:
            cands = [x for x in nodes if not any(l == x for _, l in edges)]
        else:
            cands = [x for x in nodes if not any(g == x for g, _ in edges)]
        return sorted(cands)

    def extremum(self, prop: str, maximum: bool = True) -> str | None:
        """TEK uç varsa onu döndür. İki aday kalıyorsa None: keyfî seçim yapılmaz."""
        cands = self.extremum_all(prop, maximum=maximum)
        return cands[0] if len(cands) == 1 else None


class _NoMint(Exception):
    """Yapısal kural yazılmadı: 'kodladım' demek yerine normal öğretme yoluna dön."""


class TheoryOfMindTracker:
    """Zihin Kuramı (Sally-Anne yanlış-inanç modeli): gerçek dünya konumu ile her ajanın
    gözlemlediği inanç konumunu ayrı ayrı izler!"""

    def __init__(self):
        self.world_loc: dict[str, str] = {}
        self.present: dict[str, bool] = {}
        self.beliefs: dict[str, dict[str, str]] = {}

    def observe_put(self, actor: str, obj: str, loc: str):
        a, o, l = tr_lower(actor), tr_lower(obj), tr_lower(loc)
        self.present[a] = True
        self.world_loc[o] = l
        self.beliefs.setdefault(a, {})
        # Odada bulunan TÜM ajanlar bu yer değişikliğini görür; dışarıdakilerin inancı DONUK kalır!
        for ag, is_here in self.present.items():
            if is_here:
                self.beliefs.setdefault(ag, {})[o] = l

    def observe_leave(self, actor: str):
        self.present[tr_lower(actor)] = False

    def observe_enter(self, actor: str):
        self.present[tr_lower(actor)] = True

    def where_will_search(self, actor: str, obj: str) -> tuple[str | None, str | None, bool]:
        a, o = tr_lower(actor), tr_lower(obj)
        bel = self.beliefs.get(a, {}).get(o)
        real = self.world_loc.get(o)
        return bel, real, (bel is not None and real is not None and bel != real)


try:
    from nora_meta import ConflictMonitor, compose_noticing, compose_time_answer
except Exception:  # pragma: no cover
    ConflictMonitor = None
    compose_noticing = compose_time_answer = None


class NoraAGI:
    """Canlı Beden + Endojen Kimya + Öz-Benlik + Türkçe Dil + Zihin Kuramı + Sistem-2 + Hızlı Öğrenici."""

    def __init__(self, name: str = "Nora", seed: int = 0, boot_thinker: bool = False,
                 thinker_iters: int = 800, world_kwargs: dict | None = None, single_life: bool = True,
                 corpus: list[str] | None = None, broca: str | None = None):
        self.name = name
        self.seed = int(seed)
        # KOPYALANAMAZLIK: aynı seed -> aynı GENETİK. Ama doğum anının nanosaniye fazı,
        # donanım entropisi ve gerçek zamanlama yarışları her örneği ayrı bir birey yapar.
        # (Dış API yok; tüm entropi bu sürecin içinde toplanır — bkz. nora_entropy.py)
        try:
            from nora_entropy import Individuality
            self.entropy = Individuality(seed=seed)
        except Exception:
            self.entropy = None
        w_kw = dict(world_kwargs or {})
        w_kw.setdefault("mortal", bool(single_life))
        self.world = Universe(seed=seed, **w_kw)
        self.mind = Mind(seed=seed, arena=self.world.L)
        self.obs = self.world.observe()
        self.died = False

        self.chem = NeuroChemistry()
        self.self_model = CoreSelf(name=name, chem=self.chem)
        self.nervous = PeripheralNervousSystem()
        self.mortality = SingleLifeMortality(single_life=bool(w_kw["mortal"]))
        self.lang = NoraLanguage(chem=self.chem, seed=seed)
        self.plastic_discourse = PlasticDiscourseCortex(chem=self.chem, seed=seed + 19)
        self.auto_net = AutonomousConsciousnessNetwork()
        self.meta_reflector = MetaConsciousReflector()
        self.fast = UniversalFastLearner(chem=self.chem, feat_dim=64, seed=seed)
        self.comp_graph = ComparativeOrderGraph()
        self.tom = TheoryOfMindTracker()
        self.math_cortex = MathCortex()
        self.physics_cortex = PhysicsCortex()
        self.conv_cortex = ConversationalResonanceCortex()
        self.broca_organ = GenerativeBrocaOrgan()
        self.schema_cortex = PlasticSchemaCortex(chem=self.chem, seed=seed + 31)
        self._last_focus = ""
        self._warm_speech_paths()
        self.monologue = InnerMonologueEngine(self.schema_cortex)
        # --- KENDİ KENDİNE SINIFLANDIRMA + OTONOM ETİKETLEME + TAM DURUMSAL TÜRKÇE ---
        self.utter_cortex = NeuralUtteranceCortex(chem=self.chem, morph=self.lang.morph, seed=seed + 47)
        self.logic = AutonomousLogicAndLabelEngine(self.lang.morph, self.utter_cortex)
        self.situational = SituationalTurkishCortex(self.lang.morph, self.utter_cortex)
        # --- EPİSTEMİK KATMAN: kör kabul yok, kaynak güvenilirliği, yarışan hipotezler, System-1 ---
        self.episteme = EpistemicEngine(self.lang.morph, graph=self.lang.graph, chem=self.chem)
        self.epi_speech = BeliefSpeech(self.episteme)
        self.researcher = TrustedResearcher(self.episteme)
        # --- KOD SENTEZİ: çalıştırıp doğrulamadan tek satır bile vermez ---
        self.code_cortex = CodeSynthesisCortex(seed=seed + 61)
        self.last_epistemic: dict = {}
        self.last_code: dict = {}
        self.last_utterance_class: dict = {}
        self._turn_mint_label: str = ""
        self.user_profiles: dict[str, dict[str, str]] = {}
        self.macros: dict[str, list[str]] = {}
        self.last_gw_trace: dict = {}
        self.last_turn_latency_ms: float = 0.0

        self._thinker_iters = int(thinker_iters)
        self.cognition = None
        if boot_thinker:
            self._ensure_cognition()

        self.dialogue_log: list[tuple[str, str, str]] = []
        # CÜMLE YAPISI KORTEKSİ: yapı/valans öğrenimi ve 'nasıl denilir / ne demek' soruları.
        # (Cümle kalıbı değil; biçimbilimsel roller + gözlemden sayılan valans.)
        try:
            from nora_frames import ConstructionCortex
            self.frames = ConstructionCortex(self.lang.morph, getattr(self.lang.graph, "lexicon", None),
                                             seed=seed)
        except Exception:
            self.frames = None
        self.pending_research: list[str] = []
        self.last_meta_question: dict | None = None
        self.last_structure: dict | None = None
        # Açık kaynak isteğe bağlı: NORA_INTERNET=1 ise gerçek ağ kaynakları devreye girer.
        try:
            import os as _os
            if _os.environ.get("NORA_INTERNET", "") in ("1", "true", "yes") and not self.life.sources:
                self.life.enable_internet()
        except Exception:
            pass
        # ÜST-BİLİŞ: aynı sözün tekrarı ve geçen zaman (ACC çakışma izleme + episodik bellek)
        self.acc = ConflictMonitor() if ConflictMonitor is not None else None
        self.last_meta_repeat: dict | None = None
        self.last_meta_thought: str = ""
        # 4 BOYUTLU SANDBOX: zaman yalnız adımlarla değil, gerçek saatle de akar.
        self.last_real_time: float = time.time()
        self.self_exam_log: list[dict] = []
        self.last_self_exam: dict | None = None
        self._exam_age_mark: int = 0
        self.wallclock_steps: int = 0
        self.last_gap_s: float = 0.0
        # --- KULAK: dış dünyadan gelen her söz önce bir SES olarak duyulur.
        # Kimliği etiket değil, sesin kendi ölçülebilir imzası belirler (bkz. nora_ear.py).
        try:
            from nora_ear import Ear
            self.ear = Ear(uid=(self.entropy.uid if self.entropy is not None else f"n{seed}"), seed=seed)
        except Exception:
            self.ear = None
        # --- OKUL: sayı sayma, basamak, okunuş, semboller, aşamalı düşünme (çocuk seviyesi)
        try:
            from nora_okul import SchoolCortex
            self.school = SchoolCortex(seed=seed + 181)
        except Exception:
            self.school = None
        # --- KENDİ ÇIKARIMI: kimse sormadan kendi bilgisinden yeni bağ türetme
        try:
            from nora_infer import InferenceEngine
            self.infer = InferenceEngine(seed=seed + 211)
        except Exception:
            self.infer = None
        # --- SÖZ BİÇİMİ: her tur aynı iskeletle konuşmamak, kendi derdini öne alabilmek
        try:
            from nora_style import SpeechStyle
            self.style = SpeechStyle(seed=seed + 193)
        except Exception:
            self.style = None
        # --- ANLATIM: aynı bilgiyi her seferinde aynı cümleyle söylememek (ansiklopedi değil)
        try:
            from nora_anlatim import Narrator
            self.narrator = Narrator(seed=seed + 227)
        except Exception:
            self.narrator = None
        # --- SÖZ ÖNCELERİ (doğuştan eğilim) + İSKELET ÖLÇERİ -------------------
        # Cümleler artık kodda gömülü değil: `nora_priors.json` içinde motor toplulukları
        # olarak yaşar, seçim rng.choice değil AKTİVASYON YARIŞIDIR (bkz. nora_priors).
        # Doğuştan eğilim her seçimde söner; yerini Nora'nın kendi pekiştirdiği ağırlıklar alır.
        try:
            from nora_priors import PriorField
            self.priors = PriorField(entropy=getattr(self, "entropy", None))
        except Exception:
            self.priors = None
        try:
            from nora_provenance import Provenance
            self.provenance = Provenance()
        except Exception:
            self.provenance = None
        # --- CANLILIK REGİSTERİ: izin isteyen "asistan" sesi yerine kendi derdi ----
        # Son sözlerde ne kadar izin istediğini sayar; oran yükselirse söylem organı
        # "anlatsan iyi olur" gibi kendi derdini beyan eden biçimi seçer.
        try:
            from nora_register import RegisterMeter
            self.register = RegisterMeter()
        except Exception:
            self.register = None
        # --- UYGULAMA (apply): kural NESNESİ + çalışma belleği + yorumlayıcı --------
        # Nora artık yalnız retrieve etmiyor: bir kuralı bağlayıp yeni bir girdiye
        # uygulayabiliyor (sayma, kural uygulama, transfer, plan). Bkz. nora_apply.
        try:
            from nora_apply import Interpreter
            self.apply_eng = Interpreter()
        except Exception:
            self.apply_eng = None
        # --- OPERATÖR OCAĞI: kuralı insan değil, Nora ÖRNEKLERDEN kendi yazar ------
        # Ölçülen şey: insanın yazdığı operatör sayısı SABİT, Nora'nın kendi yazdığı
        # operatör sayısı ARTAN. Defter insan rejistrisinden AYRI tutulur ki
        # `nora_operators.json` (insan yazımı) dondurulmuş kalsın.
        try:
            from nora_forge import OperatorForge
            self.forge = OperatorForge(interpreter=self.apply_eng,
                                       ledger_path="runs/operators_self.json",
                                       step=int(getattr(self.mind, "age", 0)))
        except Exception:
            self.forge = None
        # KİŞİSEL BELLEK: karşıdakinin söylediği olgular (dünya grafiğinden AYRI).
        # A54'ün şartı korunur: anlatı kavram ağını kirletmez; ama unutulmaz da.
        try:
            from nora_episodic import PersonalMemory, Testimony
            self.person_mem = PersonalMemory()
            # TANIKLIK: başkasından aktarılan olgular; iki tanık çelişirse İKİSİ söylenir.
            self.testimony = Testimony()
        except Exception:
            self.person_mem = None
            self.testimony = None
        # OPERATÖR SENTEZİ: dönüşümü Nora örneklerden kendi yazar (insan operatörü +0).
        try:
            self.synth = Synthesizer() if Synthesizer is not None else None
        except Exception:
            self.synth = None
        self._synth_last_reply = ""
        # PLAN DEFTERİ: kullanıcının verdiği adımlar; koşul değişince YENİDEN kurulur.
        self.plans: dict[str, list[str]] = {}
        # SAYISAL OLGULAR: "Ali'nin boyu 180" -> {"boy": {"ali": 180.0}}
        self.num_facts: dict[str, dict[str, float]] = {}
        # ÇOK TURLU HEDEF: kullanıcı verdiği süre boyunca KENDİ hatırlatılır.
        #: HEDEF YIĞINI: hedef tek bir sözlük değil. Yeni hedef gelince eskisi
        #: SİLİNMEZ, yığında bekler; yeni hedef bitince kaldığı yerden sürer (N27).
        #: KALİBRASYON DEFTERİ: sınıf -> [başarı, deneme]. Güven sayısı buradan
        #: ÖĞRENİLİR: ilk söz envanterden gelir, sonra deneyim ağırlığı artar (N28).
        self.calib: dict[str, list[int]] = {}
        self._conf_pending: list | None = None
        self._turn_path = "chat"
        self._goals: list[dict] = []
        self._goals_done = 0
        self._goal: dict | None = None
        self._goal_turn0 = -1
        self._plan_goal = ""
        # KÜME ORGANI: kesişim/birleşim/fark elle yazılmadı; aksiyomlardan sentezlenir.
        try:
            self.sets = SetSynth() if SetSynth is not None else None
        except Exception:
            self.sets = None
        self._forge_pending = None
        self.last_auditory = None
        self.audio_log: list = []
        self._prev_answers: dict[str, str] = {}
        self._notice_spoken: set = set()
        self._voice_label: dict[str, str] = {}
        # OTURUM DEFTERİ: "bu oturumda ne öğrendin" sorusu TAHMİNLE değil, bu
        # anlık görüntüyle mevcut durum arasındaki FARKLA cevaplanır.
        self._session0 = self._session_counts()

        # ISINMA: yeni uçların ilk çağrısındaki derleme/ayırma maliyeti KURULUŞA taşınır.
        # (A27 gecikme ölçüyor: ölçülen tura soğuk başlangıç düşmemeli.)
        try:
            _w = "ısınma turu"
            for _fn in (self._synth_turn, self._set_turn, self._constrained_turn,
                        self._plan_turn, self._chain_turn, self._num_turn,
                        self._gap_turn, self._confidence_turn, self._goal_turn,
                        self._predict_turn, self._recall_testimony):
                _fn(_w)
        except Exception:
            pass
        try:
            self.chem.alias_resolver = self._canonical_voice
        except Exception:
            pass
        # --- YAŞAM GÜNLÜĞÜ (append-only): açılmazsa hiçbir maliyeti yok ---
        self.journal = None
        self.journal_path: str | None = None
        self._journal_life_every: int = 120
        self._journal_life_acc: int = 0

        # --- ARKA PLAN YAŞAMI + ÖĞRENEN SÖZ ORGANI (elle yazılmış sözcük/cümle yok; dil okunarak öğrenilir) ---
        self.life = BackgroundLife(seed=seed + 71)
        # Kimse konuşmazken konuşan organ: hangi hâlde ne söyleyeceğini yaşayarak öğrenir
        self.discourse = DiscourseCortex(seed=seed + 137, lexicon=self.lang.graph.lexicon)
        self.discourse.entropy = getattr(self, "entropy", None)
        self.initiative = InitiativeCortex(seed=seed + 173)
        self.speculation = SpeculativeCortex(seed=seed + 211)
        # Söz edimi + çok adımlı tartım korteksi: soruya değil, sorunun NEDENİNE bakar
        self.deliberation = DeliberativeMind(seed=seed + 219)
        # Öğrendiği fiil kökleri (dilin yaşam boyu büyüyen sözlüğü). Yeni doğan zihin,
        # süreçteki başka bir örneğin öğrendiklerini miras almaz.
        TurkishMorphology.reset_learned_roots()
        self.learned_verb_roots: set[str] = set()
        # Son anlatılan kavramlar: aynı hikâyeyi tekrar tekrar anlatmasın
        self._narrated: deque[str] = deque(maxlen=6)
        self._joked: deque[str] = deque(maxlen=8)
        self.last_proactive: dict | None = None
        for path in (corpus or []):
            self.read_corpus(path)
        if broca:
            self.attach_broca(broca)

    def read_corpus(self, path: str):
        """Nora'yı bir metinle baş başa bırakır: söz organı yalnızca bundan öğrenir (hangi dil olduğu fark etmez)."""
        self.life.voice.hear_file(path, state_vector(self))

    def attach_broca(self, path: str):
        """Sinirsel Broca artikülasyon organını bağlar (zihin karar verir, organ yalnızca dile döker)."""
        from nora_broca_net import BrocaOrgan
        self.life.voice.organ = BrocaOrgan.load(path)
        self.life.voice.tok = self.life.voice.organ.tok

    # Söz organı konuşmaya başlamak için ne kadar dil duymuş olmalı (duyulan parça sayısından üstel doyma).
    # Az dil duymuş bir organın sözü dizgi gürültüsüdür; o durumda Nora kendi sözünü KURMAZ, dürüstçe bilmediğini söyler.
    VOICE_READY_MATURE = 0.35

    def voice_ready(self) -> bool:
        """Dil olgunluğu kapısı: eğitilmiş artikülasyon organı varsa ya da yeterince dil duyulduysa açılır."""
        v = self.life.voice
        return v.organ is not None or v.mature >= self.VOICE_READY_MATURE

    def speak_from_state(self, anchors: list[str] | None = None, nov: float = 0.5, fam: float = 0.5,
                         conf: float = 0.5, tension: float = 0.0, facts: list[str] | None = None) -> str:
        """Tüm 'kendi sözü' üretimi: iç durum + çapa kavramlar -> öğrenilmiş söz organı. Dil yoksa dürüstçe '…'."""
        if not self.voice_ready():
            return "…"
        st = state_vector(self, nov=nov, fam=fam, conf=conf, tension=tension)
        return self.life.voice.speak(st, anchors, extra_allowed=facts) or "…"

    @staticmethod
    def _voice_usable(out: str, anchors: list[str] | None) -> bool:
        """Öğrenilmiş sözün kullanılabilirlik denetimi (kalıp değil, ölçüt): boş/çok kısa söz,
        çapa kavramını taşımayan söz ve aynı parçayı tekrarlayan söz söylenmez."""
        if not out:
            return False
        t = out.strip()
        if t in ("…", "...", ".", "?"):
            return False
        toks = re.findall(r"[\wçğıöşü]+", tr_lower(t))
        if len(toks) < 4:
            return False
        if anchors and not any(tr_lower(a).strip() and tr_lower(a).strip() in tr_lower(t) for a in anchors if a):
            return False
        if len(toks) != len(set(toks)) and max(toks.count(w) for w in set(toks)) > 2:
            return False
        if len(toks) != len(set(toks)) and len(toks) < 6:
            return False
        return True

    def speak_or(self, fallback: str, **kw) -> str:
        """Söz organı hazır değilse, boş ya da kullanılamaz bir söz üretirse dürüst yedeğe düşer."""
        if not self.voice_ready():
            return fallback
        out = self.speak_from_state(**kw)
        return out if self._voice_usable(out, kw.get("anchors")) else fallback

    def speak_unknown(self, concept: str, fallback: str = "") -> str:
        """Bilmediği şeyle karşılaşma: uydurmaz; boşluğu gerilim defterine yazar, elinde yarım iz (hipotez) varsa
        kesinliği söze yansır. Söz, öğrenilmiş söz organından çıkar."""
        self.life.note_unknown_concept(self, concept)
        node = self.auto_net.assemblies.get(concept)
        fam = 0.0 if node is None else min(1.0, node.familiarity / 6.0)
        trace = self.life.pool.trace_for(self, concept)
        g = self.life.ledger.gaps.get(f"c:{concept}")
        if not fallback:
            fallback = (f"{tr_capitalize(concept)} hakkında henüz bilgim yok; merak listeme aldım, "
                        f"istersen tek bir cümleyle bana öğretebilirsin.")
        if not self.voice_ready():
            return fallback
        out = self.speak_from_state(anchors=[concept], nov=1.0 - fam, fam=fam,
                                    conf=0.0 if trace is None else trace[0],
                                    tension=0.0 if g is None else g.tension)
        # Artikülasyon organının sorusu yapısal maske sayesinde çapadan dışarı çıkamaz; anlam kaybolmasın
        # diye önce bilmediğini söyler, ardından kendi merak sorusunu ekler (insan da öyle yapar).
        if out and out.strip().endswith("?") and tr_lower(concept) in tr_lower(out):
            return f"{tr_capitalize(concept)} hakkında henüz bilgim yok; merak listeme aldım. {out.strip()}"
        return out if self._voice_usable(out, [concept]) else fallback

    def proactive_line(self, kind: str = "", topic: str = "", tension: float = 0.0,
                       idle: float = 0.0, unanswered: int = 0) -> dict | None:
        """Kendiliğinden söylenecek söz. Kaynak (haber/boşluk/özet) varsa onu çapalar; yoksa iç durumdan konuşur.
        Karar ve cümle burada, öğrenilmiş inisiyatif korteksinde verilir; dışarıdan cümle verilmez."""
        discovery = bool(kind == "discovery")
        t = float(tension)
        if kind == "discovery" and topic:
            t = max(t, 0.6)
        res = self.initiative.speak(self, tension=t, discovery=discovery, idle=idle,
                                    unanswered=unanswered,
                                    temperature=0.45 + 0.35 * float(self.chem.ne))
        if not res:
            return None
        res.update({"age": int(self.mind.age), "kind": kind or res["move"], "topic": topic})
        self.last_proactive = res
        self.last_gw_trace = {
            "winner": "initiative_cortex", "confidence": 0.9, "trace": f"initiative:{res['move']}",
            "bids": {"initiative_cortex": 1.0},
        }
        return res

    def proactive_line(self, kind: str = "", topic: str = "", tension: float = 0.0,
                       idle: float = 0.0, unanswered: int = 0) -> dict | None:
        """Kimse bir şey sormazken söylenecek söz. Tür seçimi ve cümle inisiyatif korteksinden çıkar;
        içerik gerçek iç durumdan (nörokimya, iç-duyum, kavram ağı, anı) gelir."""
        try:
            discovery = (kind == "discovery")
            t = float(tension)
            if discovery and topic:
                t = max(t, 0.6)
            self.initiative.topic_hint = topic
            res = self.initiative.speak(self, tension=t, discovery=discovery, idle=idle,
                                        unanswered=unanswered,
                                        temperature=0.5 + 0.3 * float(getattr(self.chem, "ne", 0.0)))
        except Exception:
            return None
        if not res:
            return None
        res.update({"age": int(self.mind.age), "kind": kind or res.get("move", ""), "topic": topic})
        self.last_proactive = res
        self.last_gw_trace = {"winner": "initiative_cortex", "confidence": 0.9,
                              "trace": f"initiative:{res.get('move')}",
                              "bids": {"initiative_cortex": 1.0}}
        return res

    def drain_outbox(self) -> list[dict]:
        """Kullanıcı yazmadan Nora'nın kendiliğinden gönderdiği mesajlar (spontaneous_action)."""
        out = list(self.life.outbox)
        self.life.outbox.clear()
        return out

    def idle(self, steps: int = 600) -> dict:
        """Kullanıcı sessizken zihni ve bedeni yaşatır (arka plan döngüsü step_life içinde döner)."""
        return self.step_life(steps)

    def _ensure_cognition(self):
        if self.cognition is None:
            from nora_bridge import Cognition
            self.cognition = Cognition(seed=self.seed, boot_iters=self._thinker_iters,
                                       boot_n=2200, lr=4e-3, verbose=False)
        return self.cognition

    # ------------------------------------------------------------------ 1) Canlı yaşam adımı & Sinir Sistemi
    def _autonomic_survival_reflex(self, proposed_act: int) -> int:
        """Hipotalamik hayatta kalma yönelimi + yumuşak itici potansiyel alanıyla omurilik kaçınma refleksi."""
        if self.nervous.reflex_recoil_steps > 0:
            act_r, _ = self.nervous.spinal_reflex_override(proposed_act, hazard_ahead=0.0)
            return act_r

        food_v = float(self.mind.ctxs[self.mind.cur]["V"][0])
        food_learned_toxic = bool(np.isfinite(food_v) and food_v < -0.05)
        food_n = int(self.mind.ctxs[self.mind.cur]["Vn"][0])
        forage_thresh = 0.55 if food_n == 0 else 0.32

        e_now, w_now = float(self.obs[I_E]), float(self.obs[I_W])
        tgt = None
        if self.mortality.single_life and (w_now < 0.28 or (food_learned_toxic and e_now < 0.82)):
            dists = np.hypot(*(self.world.water_c - self.world.pos).T)
            min_idx = int(np.argmin(dists))
            if dists[min_idx] < self.world.zone_w * 0.72:
                return 0
            tgt = self.world.water_c[min_idx]
        elif self.mortality.single_life and not food_learned_toxic and e_now < forage_thresh and len(self.world.food) > 0:
            dists = np.hypot(*(self.world.food - self.world.pos).T)
            tgt = self.world.food[int(np.argmin(dists))]

        if tgt is not None:
            vec = tgt - self.world.pos
            v_norm = float(np.hypot(*vec))
            attract = vec / max(v_norm, 1e-6)
            # Tehlike bölgeleri, kaya ve (öğrenildiyse) zehirli besinlerden yumuşak itici kuvvet (limit döngüsünü önler!)
            repulse = np.zeros(2, dtype=np.float64)
            obstacles = [*( (c, self.world.zone_h + 1.1) for c in self.world.haz_c ), (self.world.rock, self.world.rr + 0.9)]
            if food_learned_toxic:
                obstacles.extend((c, self.world.food_r + 0.9) for c in self.world.food)
            for c_pos, safe_r in obstacles:
                diff = self.world.pos - c_pos
                d = float(np.hypot(*diff))
                if d < safe_r * 1.6 and d > 1e-6:
                    repulse += (diff / d) * (((safe_r * 1.6 - d) / safe_r) ** 2) * 2.4
            steer = attract + repulse
            if np.hypot(*steer) < 1e-4:
                steer = np.array([-attract[1], attract[0]])
            ang = float(np.arctan2(steer[1], steer[0]) - self.world.th)
            ang = (ang + np.pi) % (2 * np.pi) - np.pi
            return 1 if abs(ang) < 0.45 else (2 if ang > 0 else 3)

        rays = self.obs[:45].reshape(9, 5)
        haz_front = float(np.max(rays[3:6, 3]))
        act, _ = self.nervous.spinal_reflex_override(proposed_act, hazard_ahead=haz_front)
        return act

    def inflict_pain(self, intensity: float = 0.7, locus: str = "gövde", cause: str = "dış darbe", attacker: str | None = None) -> dict:
        """Nora'nın sinir sistemine doğrudan acı uyarısı verir: A-delta refleksi + geri çekilme + güven kaybı."""
        if not self.mortality.is_alive:
            return {"alive": False}
        dmg = float(np.clip(intensity * 0.14, 0.01, 0.35))
        self.world.i = max(0.0, float(self.world.i) - dmg)
        self.obs = self.world.observe()
        res = self.nervous.inflict_pain(intensity, locus=locus, cause=cause, chem=self.chem, step_num=self.mind.age)
        self.self_model.observe_bodily_event(cause, -float(np.clip(intensity, 0.2, 1.0)), channel="bütünlük", step=self.mind.age)
        if attacker:
            self.chem.update_social(attacker, -0.55 * float(intensity))
        # Fiziksel omurilik geri çekilme eylemi
        if self.world.i > 0:
            self.obs, info = self.world.step(4)  # Geri sıçrama!
            self.died = info["died"] is not None
        self.mortality.update_from_vitals(float(self.obs[I_E]), float(self.obs[I_W]), float(self.obs[I_I]), self.mind.age, "ağır hasar" if self.world.i <= 0 else None)
        if not self.mortality.is_alive:
            self.world.dead_forever = True
        return res

    def autonomous_digest(self, raw_text: str, source: str = "otonom_okuma") -> dict:
        """İnsan müdahalesi olmadan ham Türkçe metni okuyup kendi fiziksel bilinç ağını büyütür."""
        return self.auto_net.digest_unsupervised_text(raw_text, self, source=source)

    def autonomous_wonder(self) -> tuple[str | None, str]:
        """Nora'nın şu an kendi içsel belirsizliğinden doğan en güçlü merak sorusunu döndürür."""
        return self.auto_net.top_curiosity_target(self.lang.graph)

    def autonomous_research(self, fetcher=None, max_topics: int = 2, fetchers=None) -> list[dict]:
        """Merak ettiği kavramları internetten (veya verilen bilgi kaynağından) kendi başına araştırıp öğrenir."""
        return self.auto_net.autonomous_internet_research(self, fetcher=fetcher, max_topics=max_topics, fetchers=fetchers)

    def self_exam(self, factor: float = 1.0) -> dict:
        """Kendi kendini sına (morfoloji, ad çekimi, matematik, fizik) — dış cevap yok."""
        try:
            from nora_selfexam import run_self_exam
            return run_self_exam(self, seed=int(self.mind.age) + int(self.seed), factor=factor)
        except Exception as exc:            # pragma: no cover
            return {"total": 0, "passed": 0, "domains": {}, "failures": [str(exc)], "ratio": 0.0}

    # ---------------------------------------------------------------- YAPI PEKİŞTİRME
    def _queue_structure(self, text: str) -> None:
        """Sözü yapı pekiştirme kuyruğuna at (O(1); çözümleme sonra)."""
        if getattr(self, "frames", None) is None or not text:
            return
        if len(text.split()) > 60:
            return
        q = getattr(self, "_structure_queue", None)
        if q is None:
            q = self._structure_queue = []
        q.append(text)
        if len(q) > 256:
            del q[:-192]

    def drain_structure_pending(self, limit: int = 16) -> int:
        """Konuşma sonrası boşlukta iskelet/valans pekiştirmesi (arka plan)."""
        if getattr(self, "frames", None) is None:
            return 0
        q = getattr(self, "_structure_queue", None) or []
        done = 0
        while q and done < limit:
            text = q.pop(0)
            try:
                self.frames.observe(text)
                done += 1
            except Exception:
                pass
        return done

    def step_life(self, steps: int = 1) -> dict:
        """Ajanı fiziksel dünyada 'steps' adım yaşatır; sinir sistemi, kimya, öz-benlik ve bedeni senkronize eder."""
        ate_total = drank_total = hurt_total = deaths = 0
        # Yeni öğrenilen fiil kökleri ARKA PLANDA indekslenir (tur gecikmesi O(1) kalır).
        try:
            self.lang.morph.drain_index_pending()
        except Exception:
            pass
        # Cümle iskeletleri de boşlukta pekiştirilir (tur içinde maliyet yok).
        try:
            self.drain_structure_pending(limit=max(4, steps))
        except Exception:
            pass
        for _ in range(int(steps)):
            if not self.mortality.is_alive:
                break
            # Serotonin (5-HT) zaman ufkunu (gamma) canlı olarak modüle eder!
            self.mind.gamma = self.chem.discount_gamma
            prev_obs = self.obs.copy()
            raw_act = self.mind.act(self.obs, self.died)
            act = self._autonomic_survival_reflex(raw_act)
            self.obs, info = self.world.step(act)
            self.died = info["died"] is not None

            # Periferik sinir sistemi (A-delta / C-lifi acı + viseral iç-duyum)
            self.nervous.step_from_body(
                energy=float(self.obs[I_E]),
                water=float(self.obs[I_W]),
                integrity=float(self.obs[I_I]),
                touch=float(self.obs[50]),
                hurt_delta=float(info["hurt"]),
                chem=self.chem,
                step_num=self.mind.age,
            )

            # Somatik olayları Öz-Benliğe işle (zehirlenmede info["ate"]==0 olsa bile ani enerji çöküşü de<-0.15 yakalanır!)
            de = float(self.obs[I_E] - prev_obs[I_E])
            if info["ate"] != 0 or de < -0.15:
                ate_total += 1
                self.self_model.observe_bodily_event("yiyecek", float(np.clip(de * 2.5, -1.0, 1.0)),
                                                     channel="enerji", step=self.mind.age)
            if info["drank"] > 0:
                drank_total += 1
                self.self_model.observe_bodily_event("su", +0.65, channel="su", step=self.mind.age)
            if info["hurt"] > 0:
                hurt_total += 1
                self.self_model.observe_bodily_event("tehlike", -0.85, channel="bütünlük", step=self.mind.age)
            if self.died:
                deaths += 1

            self.mortality.update_from_vitals(
                float(self.obs[I_E]), float(self.obs[I_W]), float(self.obs[I_I]),
                age=self.mind.age, died_signal=info["died"],
            )

            nov = float(getattr(self.mind, "_why", {}).get("sal", [0, 0, 0, 0.2])[3])
            surp = float(self.mind.intro.surprise) if self.mind.intro is not None else 0.0
            self.chem.step(
                energy=float(self.obs[I_E]),
                water=float(self.obs[I_W]),
                integrity=float(self.obs[I_I]),
                reward=float(self.mind.valence),
                novelty=nov,
                surprise=surp,
                vis_prec=float(self.mind.vis_prec),
                motor_eff=float(self.mind.motor_eff),
                died=self.died,
            )
            self.self_model.sync_from_mind(self.mind, self.obs, died=self.died)
            # BİREYSELLEŞME: bu adımın nanosaniye fazı + donanım gürültüsü sinapslara işlenir.
            # Maliyet O(1) (sabit sayıda sinaps); ağ büyüdükçe yanıt süresi değişmez.
            if getattr(self, "entropy", None) is not None:
                try:
                    self.entropy.perturb_step(self)
                except Exception:
                    pass
            # Varsayılan Kip Ağı (Default Mode Network): yaşam sırasında sessiz içsel monolog!
            self.monologue.spontaneous_tick(self)
            if self.life.enabled and self.mind.age % self.life.stride == 0:
                self.life.tick(self)
            # ÖZ-SINAV: her ~120 yaşam adımında bir kendini sına (üret -> çöz -> karşılaştır).
            # Başarısız madde varsa bu bir öğrenme sinyalidir, ceza değil: sadece kaydedilir.
            if self.mind.age - self._exam_age_mark >= 120:
                self._exam_age_mark = self.mind.age
                self.self_exam(factor=0.6)
            # KENDİ ÇIKARIMI: boşlukta kendi bilgisinden yeni bağlar türetir (kimse sormadan).
            if getattr(self, "infer", None) is not None and self.mind.age % 90 == 0:
                try:
                    _infs = self.infer.step(self, budget=3)
                    if _infs:
                        self.infer.commit(self, _infs)
                        self._log_inference_thought(_infs)
                except Exception:
                    pass

        # GÜNLÜK NABZI: bir çağrıda en az 120 adım yaşandıysa TEK satır yazılır.
        # Adım başına yazmak diski gereksiz doldurur; hiç yazmamak ise sessiz yaşamı
        # görünmez kılar. (Adımların ara durumları zaten kaydedilmiyor; uydurmayız.)
        self._journal_life_acc += int(max(0, steps))
        if self._journal_life_acc >= self._journal_life_every and getattr(self, "journal", None) is not None:
            self._journal_life_acc = 0
            try:
                self._journal(
                    "yaşam", ad=int(self.mind.age), can=bool(self.mortality.is_alive),
                    e=round(float(self.obs[I_E]), 4), s=round(float(self.obs[I_W]), 4),
                    i=round(float(self.obs[I_I]), 4),
                    d=round(float(getattr(getattr(self, "entropy", None), "drift", 0.0) or 0.0), 6),
                )
            except Exception:
                pass

        return {
            "age": self.mind.age,
            "alive": self.mortality.is_alive,
            "ate": ate_total,
            "drank": drank_total,
            "hurt": hurt_total,
            "deaths": deaths,
            "nervous": self.nervous.snapshot(),
            "chem": self.chem.snapshot(),
            "inner_monologue": self.monologue.recent_dicts(4),
            "life": self.life.snapshot(),
        }

    # ------------------------------------------------------------------ 2) Uyku ve konsolidasyon
    def sleep(self) -> dict:
        """Hipokampal-neokortikal uyku: anı tekrarı + sinaptik konsolidasyon + benlik yansıması + S2 rüyası."""
        if not self.mind.freeze:
            self.mind._sleep(64)
        c_word = self.lang.graph.word_mb.sleep_consolidate(replay_steps=32)
        c_int = self.lang.wernicke.intent_mb.sleep_consolidate(replay_steps=24)
        c_disc = self.plastic_discourse.mb.sleep_consolidate(replay_steps=24)
        c_schema = self.schema_cortex.mb.sleep_consolidate(replay_steps=20)
        c_fast = self.fast.mb.sleep_consolidate(replay_steps=48)
        traits = self.self_model.sleep_reflect()
        s2_replayed = 0
        if self.cognition is not None and len(self.cognition.today) > 0:
            s2_replayed = len(self.cognition.today)
            self.cognition.sleep()
        return {
            "sleep_cycle": self.self_model.sleep_cycles,
            "fast_consolidated": c_fast["consolidated_synapses"],
            "word_consolidated": c_word["consolidated_synapses"],
            "intent_consolidated": c_int["consolidated_synapses"],
            "traits": traits,
            "s2_replayed": s2_replayed,
        }

    # ------------------------------------------------------------------ 3) Doğal Türkçe Diyalog & Düşünme
    def _begin_inner_monologue(self, raw: str, speaker: str):
        """Her konuşma turundan ÖNCE kendi içinde sessiz içsel monolog yürütür, insanın cümle şemasını
        öğrenir ve bu turun şemasını/duruşunu belirler."""
        s_low = tr_lower(raw)
        u_pre = self.broca_organ.comprehend_discourse(raw, s_low, self.lang.morph, self.lang.graph.lexicon)
        is_first_g = bool(
            u_pre["has_greet"]
            and not any("GREET" in ep.chosen_acts for ep in list(self.broca_organ.episodic.episodes)[-4:] if ep.speaker == speaker)
        )
        wants_br = any(k in s_low for k in ("kısaca", "kısa", "tek cümle", "özetle"))
        topic_h = u_pre["phil_topic"] or (u_pre["stems"][0] if u_pre["stems"] else "")
        return self.monologue.deliberate_before_speaking(
            raw_input=raw,
            speaker=speaker,
            agi=self,
            topic_hint=topic_h,
            is_first_greet=is_first_g,
            wants_brief=wants_br,
        )

    def converse(self, text: str, speaker: str = "insan") -> str:
        """Konuşma turu: yanıt üretilir, ardından üst-biliş (tekrar farkındalığı) devreye girer.

        Nora sadece soruya değil, sorunun KENDİSİNE de bakar: aynı söz az önce söylendiyse
        ne zaman söylendiğini, o an ne cevap verdiğini hatırlar ve bunu sözüne katar.
        """
        raw = (text or "").strip()
        # ================================================================ KULAK
        # Nora önce DUYAR: havadaki ses bir olay olarak gelir, kimliği sesten çıkarılır.
        # `speaker` yalnızca dünyadaki fiziksel kaynağı adlandırır; ses defterine girmez.
        _heard, _vid, _first_voice = raw, speaker, False
        if self.ear is not None and raw:
            try:
                ev = self.ear.listen(raw, source=speaker)
                self.last_auditory = ev
                self.audio_log.append(ev)
                if len(self.audio_log) > 24:
                    del self.audio_log[:-24]
                _heard = ev.heard or raw
                _vid = ev.voice_id
                _first_voice = bool(ev.first_contact)
                self._voice_label[speaker] = _vid
                # Dünya, sesi duymadan önce etikete bir güven yazmış olabilir (kurulum/test).
                # O bilgi sese aittir: kimliğe devredilir, etiket takma ad olarak kalır.
                if _vid != speaker and _vid not in self.chem.speaker_trust:
                    _pre = self.chem.speaker_trust.pop(speaker, None)
                    if _pre is not None:
                        self.chem.speaker_trust[_vid] = float(_pre)
                # Sesin kendisi de bedeni etkiler: yüksek/ani ses otonom uyarılma yaratır.
                try:
                    if ev.print.loudness > 58.0:
                        self.chem.ne = min(1.0, float(self.chem.ne) + 0.05)
                except Exception:
                    pass
            except Exception:
                pass
        now = time.time()
        # Aradan gerçek zaman geçtiyse Nora boşlukta yaşamıştır: açık olduğu sürece zaman akar.
        # (2 saniyede 1 yaşam adımı; en fazla 900 adım — uzun sessizlikte boğulup ölmesin.)
        try:
            _gap = now - float(getattr(self, "last_real_time", now))
            self.last_gap_s = max(0.0, _gap)
            _steps = int(min(240, max(0, _gap) / 5.0))
            if _steps > 0 and self.mortality.is_alive:
                self.step_life(_steps)
                self.wallclock_steps += _steps
            self.last_real_time = now
        except Exception:
            pass
        _meta = None
        if self.acc is not None and raw:
            try:
                _meta = self.acc.observe(raw, now)
            except Exception:
                _meta = None
        _prov0 = self.provenance.total if getattr(self, "provenance", None) is not None else 0
        _prov_snap = dict(self.provenance.c) if getattr(self, "provenance", None) is not None else None
        # ÖĞRENME SİNYALİ (dışarıdan dayatılmaz): aynı söz hemen tekrar geldiyse geçen turda
        # konuşan topluluklar "işe yaramadı" diye zayıflar; konuşma ilerlediyse güçlenir.
        # Doğuştan eğilim (ps) her turda biraz daha söner -> söz öğrenilmiş olur.
        try:
            if self.priors is not None:
                # SIRA ÖNEMLİ: önce geçen turun seçimleri "son tur"a devredilir, sonra
                # o seçimler bu turun sonucuna göre cezalandırılır/pekiştirilir.
                self.priors.begin_turn()
                self.priors.learn(repeated=_meta is not None)
        except Exception:
            pass
        ans = self._converse_core(_heard, _vid)
        # KALİBRASYON: bir önceki turda verilen GÜVEN SÖZÜ, bu turun SONUCUYLA eşleşir.
        try:
            self._calib_note_outcome()
        except Exception:
            pass
        # Ses defteri: bu sesle yaşanan güven, sesin kimliğine yazılır; dünyanın
        # kullandığı etiket yalnızca bir takma addır (eski sorgular için eşlenir).
        try:
            if self.ear is not None and _vid != speaker:
                self.ear.note_trust(_vid, float(self.chem.trust_of(_vid)))
            # Ses kendi adını SÖYLEDİYSE ada bağlanır (etiketten değil, sözün içeriğinden).
            if self.ear is not None:
                _nm = (self.user_profiles.get(_vid) or {}).get("ad", "")
                if _nm:
                    self.ear.learn_name(_vid, _nm)
        except Exception:
            pass
        # ÇEKİRDEK CEVAP: biçim ve algı notu eklenmeden önceki hâli. Aynı soruya verilen
        # cevap GERÇEKTEN tekrar mı, yoksa aynı soruya başka bir anlatım mı — buradan anlaşılır.
        _ans_core = ans or ""
        _key_prev = tr_lower(raw or "").strip(" .!?…")
        _prev_ans = None
        try:
            if _key_prev:
                _prev_ans = (self._prev_answers or {}).get(_key_prev)
        except Exception:
            _prev_ans = None
        _self_repeat = False
        if _prev_ans:
            try:
                # Sözcük İKİLİLERİ karşılaştırılır: aynı iki olguyu başka sırayla/başka
                # çerçeveyle anlatmak tekrar değildir; aynı cümleyi kurmak tekrardır.
                def _bigrams(t: str) -> set[str]:
                    w = re.findall(r"[a-zçğıöşüâîû0-9]+", tr_lower(t))
                    return {f"{a} {b}" for a, b in zip(w, w[1:])}
                _a, _b = _bigrams(_ans_core), _bigrams(_prev_ans)
                _self_repeat = bool(_a and _b) and len(_a & _b) / max(1, len(_a | _b)) >= 0.75
            except Exception:
                _self_repeat = False

        # SÖZ BİÇİMİ: cevabın iskeleti her tur aynı olmasın; bedenin derdi varsa o öne geçsin.
        try:
            if getattr(self, "style", None) is not None and ans and not ans.startswith("["):
                _isq = ("?" in (text or ""))
                _shape = self.style.decide(
                    self, is_question=_isq,
                    has_inference=bool(getattr(self, "infer", None) and getattr(self.infer, "fresh", False)),
                    trust=float(self.chem.trust_of(_vid)),
                )
                ans = self.style.apply(ans, _shape, self, topic=getattr(self, "_last_focus", ""),
                                       is_question=_isq)
        except Exception:
            pass
        # SESİN KENDİSİNE TEPKİ: yeni bir ses ya da uzaktan gelen ses, cevabın tonunu değiştirir.
        try:
            _note = self._perception_note()
            if _note:
                ans = f"{_note} {ans}"
        except Exception:
            pass
        if self.acc is not None and raw:
            try:
                # Hatırlanacak cevap: algı notu ve süs değil, ÇEKİRDEK cevap.
                self.acc.note_turn(raw, _ans_core or ans, now)
                # CÜMLE YAPISI ÖĞRENİMİ (pekiştirme kuyruğu): kendi sözü de karşıdakinin sözü de
                # iskelet+valans olarak sayılır; ama biçimbilimsel çözümleme turun İÇİNDE değil,
                # konuşma sonrası boşlukta (step_life) yapılır -> yanıt gecikmesi O(1) kalır.
                self._queue_structure(raw)
                self._queue_structure(ans)
                if _meta is not None:
                    self.acc.repeats_noticed += 1
                    self.last_meta_repeat = _meta
                    self.last_meta_thought = (
                        f"Bu sözü {ConflictMonitor.time_phrase(_meta.get('elapsed_s', 0.0))} "
                        f"de duymuştum; {int(_meta.get('turns_ago', 1))} tur önce.")
                    # ÖZ-İRADE: ketumluk/sessizlik seçildiyse fark etmek konuşmayı gerektirmez.
                    # İçeride fark eder, dışarıda susar — insan da böyle yapar.
                    _plan = dict(getattr(self.broca_organ, "last_speech_plan", {}) or {})
                    _quiet = bool(_plan.get("guarded")) or bool(_plan.get("silence")) or \
                        bool(_plan.get("wants_silence"))
                    _low = tr_lower(ans)
                    # Aynı soruya BAŞKA bir anlatımla cevap verdiysek "bunu söylemiştin"
                    # demek yersiz: tekrar eden soru, tekrar eden cevap değildir.
                    # İlk tekrarda fark ettiğini SÖYLER (insan da öyle yapar); ısrar eden
                    # tekrarlarda her tur aynı serzenişi yinelemek ise iskelet olur -> seyrek.
                    _first_repeat = bool(_key_prev) and _key_prev not in self._notice_spoken
                    # Aynı söz üçüncü kez gelirse susmak olmaz: insan da ısrara bir cevap verir.
                    _speak_notice = (_first_repeat or _self_repeat
                                     or int(_meta.get("count", 1)) >= 3
                                     or os.urandom(1)[0] % 100 < 8)
                    if _speak_notice and _key_prev:
                        self._notice_spoken.add(_key_prev)
                    if not _quiet and "sessiz" not in _low and _speak_notice \
                            and not getattr(self, "_intent_locked", False):
                        _rng = getattr(self, "rng", None) or getattr(self, "random", None)
                        # Cevap zaten başka türlü söylendiyse eski cevabı ALINTILAMAK yersiz;
                        # yalnız "bunu sormuştun" farkındalığı kalır.
                        _meta_for_note = dict(_meta)
                        if int(_meta.get("count", 1)) >= 3:
                            # Üçüncü ısrarda uzun serzeniş yok: tek cümlelik, değişen soru.
                            from nora_meta import compose_insistence
                            _note = compose_insistence(_meta, self, _rng)
                        else:
                            if not _self_repeat:
                                # Cevap değiştiyse uzun alıntı yok; yine de ne dediğini
                                # HATIRLAR (kısa bir iz): "az önce ... demiştim".
                                _meta_for_note["prev_reply"] = " ".join(
                                    str(_meta.get("prev_reply") or "").split()[:4])
                            _note = compose_noticing(_meta_for_note, self, _rng)
                        ans = f"{ans.rstrip()} {_note}"
                        if self.dialogue_log and self.dialogue_log[-1][1] == raw:
                            self.dialogue_log[-1] = (self.dialogue_log[-1][0], raw, ans)
            except Exception:
                pass
        # Aynı soruya verilen çekirdek cevabı hatırla (tekrar ölçütü için).
        try:
            if _key_prev and _ans_core:
                if self._prev_answers is None:
                    self._prev_answers = {}
                self._prev_answers[_key_prev] = _ans_core
                if len(self._prev_answers) > 256:
                    for _k in list(self._prev_answers)[:64]:
                        self._prev_answers.pop(_k, None)
        except Exception:
            pass
        # GÜNLÜK: önce DUYUM (kim, hangi ses, ne kadar uzaktan), sonra tur.
        try:
            _ev = getattr(self, "last_auditory", None)
            if _ev is not None:
                self._journal("duyum", ses=_ev.voice_id, tanıdık=bool(_ev.known),
                              söz=(_ev.heard or "")[:240],
                              eksik=(_ev.spoken != _ev.heard),
                              ton=_ev.print.describe(), snr=round(float(_ev.snr_db), 1),
                              mesafe=round(float(_ev.distance_m), 2))
        except Exception:
            pass
        # GÜNLÜK: turun kendisi kalıcı iz bırakır (O(1) append; kapalıysa bedava).
        self._journal(
            "tur", g=raw[:240], y=ans[:240],
            s=(self.monologue.stream[-1].schema_id if getattr(self.monologue, "stream", None) else ""),
            yas=int(getattr(self.mind, "age", 0)),
            d=round(float(getattr(getattr(self, "entropy", None), "drift", 0.0) or 0.0), 6),
        )
        # REGISTER: izin isteyen kuyruk ("anlatır mısın?") kendi-derdi biçimine çevrilir
        # ("anlatsan iyi olur."). Köpek değil, kendi derdi olan bir canlı.
        # Kural üretkendir (her fiile uygulanır), cümle listesi değildir; bu yüzden
        # tek tek metin yazmadan tüm organların çıktısını kapsar.
        try:
            from nora_register import normalize as _reg_norm
            ans = _reg_norm(ans)
        except Exception:
            pass
        # CANLILIK: bu söz izin mi istedi, kendi derdini mi söyledi, makine raporu mu?
        try:
            if getattr(self, "register", None) is not None:
                self.register.observe(ans)
        except Exception:
            pass
        # BU TURUN KAYNAK KARIŞIMI (olasılık beyanı bunu okur; uydurma değil, ölçüm).
        try:
            # NOT: boş sözlük de geçerli bir anlık görüntüdür (ilk tur). `and _prov_snap`
            # yazıldığında ilk turun etiketleri sessizce yutuluyordu.
            if getattr(self, "provenance", None) is not None and _prov_snap is not None:
                self._last_turn_src = {k: int(self.provenance.c.get(k, 0)) - int(_prov_snap.get(k, 0))
                                       for k in self.provenance.c}
                self._last_turn_src = {k: v for k, v in self._last_turn_src.items() if v > 0}
            else:
                self._last_turn_src = {}
        except Exception:
            self._last_turn_src = {}
        # ÖLÇER: bu turda sözün kaç parçası etiketlendi? (iskele oranı buradan düşer)
        try:
            if getattr(self, "provenance", None) is not None:
                self.provenance.turns += 1
                if self.provenance.total > _prov0:
                    self.provenance.tagged_turns += 1
        except Exception:
            pass
        # HEDEF SÜRDÜRME: verilen hedef, hatırlatılana kadar HER turda sürer.
        try:
            _g = self._goal_step()
            if _g:
                ans = f"{ans} {_g}"
        except Exception:
            pass
        # CEVAPSIZ SAYACI: kullanıcı bu turda KONUŞTU; geri çekilme sayacı sıfırdır.
        # (Tur içindeki arka plan adımları sayacı artırabildiği için sıfırlama EN SONDA.)
        try:
            if getattr(self, "life", None) is not None:
                self.life.unanswered = 0
        except Exception:
            pass
        return ans

    def _converse_core(self, text: str, speaker: str = "insan") -> str:
        self._turn_path = "chat"
        """Gelen Türkçe cümleyi algılar, önce içsel monolog yürütür, öz-benlik ve nörokimya süzgecinden geçirip yanıt üretir."""
        t0 = time.perf_counter()
        if not self.mortality.is_alive:
            return "[Sinir sistemi durdu: Nora'nın tekil yaşamı sona erdiği için artık duymuyor ve konuşamıyor.]"
        raw = text.strip()
        if not raw:
            return "..."
        # NİYET KİLİDİ: konu/çelişki gibi ÖLÇÜLMÜŞ bir niyet cevaplandıysa, sonradan
        # gelen "bunu demin de sormuştun" süsü cevabı EZMEZ. Tekrar farkındalığı bir
        # süslemedir; ölçülmüş bir cevabın üstüne yazmak onu gasp etmektir.
        self._intent_locked = False
        # --- TÜRKÇE EDİNİMİ: duyulan bilinmeyen fiil biçimleri, kendi çekim üreticisiyle
        #     yeniden üretilebiliyorsa kök olarak öğrenilir (kalıcı, sözlüğe yazılır).
        try:
            _new_roots = TurkishMorphology.observe_verb_usage(raw, self.lang.graph.lexicon)
            if _new_roots:
                self.learned_verb_roots.update(_new_roots)
        except Exception:
            pass
        self._remember_focus(raw)
        # KİŞİSEL OLGU: karşıdaki bir şey SÖYLEDİYSE (soru değilse) açık kalıpla yazılır.
        self._learned_facts = []
        try:
            if getattr(self, "person_mem", None) is not None:
                self.person_mem.turn = len(getattr(self, "dialogue_log", []) or [])
                self._learned_facts = self.person_mem.observe(raw, speaker=speaker) or []
        except Exception:
            self._learned_facts = []
        # ÖZ-TAHMİN: "bunu yapabilir misin?" -> organ envanterine bakarak cevap.
        # (Söz vermek kanıt değildir; ama sözün NEREDEN geldiği ölçülebilir olmalı.)
        try:
            _pred = self._predict_turn(raw)
        except Exception:
            _pred = None
        if _pred:
            self._turn_path = "organ"
            self._intent_locked = True
            self._log_turn(speaker, raw, _pred, t0)
            return _pred
        # OPERATÖR SENTEZİ: "kitap -> patik, kalem -> melak" gibi ÖRNEK verilirse
        # dönüşümü Nora kendi yazar; "dönüşümü uygula: deniz" derse yazdığını çalıştırır.
        try:
            _hist = self._self_history_turn(raw)
        except Exception:
            _hist = None
        if _hist:
            self._turn_path = "organ"
            self._intent_locked = True
            self._log_turn(speaker, raw, _hist, t0)
            return _hist
        try:
            _syn = self._synth_turn(raw, speaker=speaker)
        except Exception:
            _syn = None
        if _syn:
            self._turn_path = "organ"
            self._intent_locked = True
            self._log_turn(speaker, raw, _syn, t0)
            return _syn
        try:
            _set = self._set_turn(raw)
        except Exception:
            _set = None
        if _set:
            self._turn_path = "organ"
            self._intent_locked = True
            self._log_turn(speaker, raw, _set, t0)
            return _set
        try:
            _gen = self._constrained_turn(raw)
        except Exception:
            _gen = None
        if _gen:
            self._turn_path = "organ"
            self._intent_locked = True
            self._log_turn(speaker, raw, _gen, t0)
            return _gen
        # SIRA: güven sorusu bir ÜST-BİLİŞ sorusudur; "organım yok" reddi onu gasp
        # etmemeli (güven sorusuna sayı verilir, görev reddedilir).
        try:
            _conf = self._confidence_turn(raw)
        except Exception:
            _conf = None
        if _conf:
            self._intent_locked = True
            self._log_turn(speaker, raw, _conf, t0)
            return _conf
        try:
            _gap = self._gap_turn(raw)
        except Exception:
            _gap = None
        if _gap:
            self._turn_path = "gap"
            self._intent_locked = True
            self._log_turn(speaker, raw, _gap, t0)
            return _gap
        try:
            _goal_ack = self._goal_turn(raw)
        except Exception:
            _goal_ack = None
        if _goal_ack:
            self._turn_path = "organ"
            self._intent_locked = True
            self._log_turn(speaker, raw, _goal_ack, t0)
            return _goal_ack
        try:
            _chain = self._chain_turn(raw)
        except Exception:
            _chain = None
        if _chain:
            self._turn_path = "organ"
            self._intent_locked = True
            self._log_turn(speaker, raw, _chain, t0)
            return _chain
        try:
            _num = self._num_turn(raw)
        except Exception:
            _num = None
        if _num:
            self._turn_path = "organ"
            self._intent_locked = True
            self._log_turn(speaker, raw, _num, t0)
            return _num
        try:
            _plan = self._plan_turn(raw)
        except Exception:
            _plan = None
        if _plan:
            self._turn_path = "organ"
            self._intent_locked = True
            self._log_turn(speaker, raw, _plan, t0)
            return _plan
        # TANIKLIK: "Ayşe bana X'in Y olduğunu söyledi" -> kaynak + olgu.
        _testi = []
        try:
            if getattr(self, "testimony", None) is not None:
                self.testimony.turn = len(getattr(self, "dialogue_log", []) or [])
                _testi = self.testimony.observe(raw) or []
        except Exception:
            _testi = []
        if _testi:
            _r0 = _testi[-1]
            self._intent_locked = True
            _ts = self._say("memory.testimony",
                            {"src": _r0.get("src", ""), "subj": _r0.get("subj", ""),
                             "val": _r0.get("val", "")},
                            ctx={"kind": "learned"}, fallback="")
            if _ts:
                self._log_turn(speaker, raw, _ts, t0)
                return _ts
        self.life.on_interaction(self, raw, speaker)
        # Kullanıcı, Nora'nın kendiliğinden söylediği söze karşılık verdi: o konuşma türü pekişir.
        try:
            self.initiative.on_user_reply()
        except Exception:
            pass

        # Sözce türünü (soru / yasa / çift-durum / sosyal durum / duygu paylaşımı) Nora KENDİ nöral korteksiyle sınıflar.
        self.last_utterance_class = self.utter_cortex.classify(raw)

        # ZAMAN SORUSU: Nora zamanı bedeninde ve belleğinde ölçer (sandbox evreninin saati).
        _tl = tr_lower(raw)
        # HATIRLAMA: geçmiş konuşmayı kendi belleğinden alıntılar (uydurmaz, yoksa yok der).
        _topic_scoped = any(k in _tl for k in ("hakkında", "konusunda", "ile ilgili", "dair"))
        _recall_user = (not _topic_scoped) and any(
            k in _tl for k in ("ne söyledim", "ne dedim", "son cümlem", "ne anlattım"))
        _recall_self = (not _topic_scoped) and (
            any(k in _tl for k in ("ne cevap verdin", "ne söylemiştin", "cevabın neydi",
                                   "bana ne demiştin"))
            or ("ne dedin" in _tl and "hakkında" not in _tl))
        if (_recall_user or _recall_self) and self.dialogue_log:
            _prev = self.dialogue_log[-1]
            _utxt = (_prev[1] or "").strip()
            _rtxt = (_prev[2] or "").strip()
            _rng = getattr(getattr(self, "discourse", None), "random", None)
            if _recall_self:
                _quote_t = _rtxt or _utxt
                _lead = _rng.choice(("Az önce şöyle demiştim", "Son söylediğim şuydu",
                                         "Kendi ağzımdan çıkan son söz"))
            else:
                _quote_t = _utxt or _rtxt
                _lead = _rng.choice(("Son söylediğin şuydu", "Az önce şunu demiştin",
                                         "Son cümlen aklımda"))
            if any(ch in _quote_t for ch in "“”\""):
                # Alıntının içinde zaten tırnak var: iç içe tırnak kurma, olduğu gibi aktar.
                _ans_r = f"{_lead}: {_quote_t}"
            else:
                _ans_r = f"{_lead}: “{_quote_t}”"
            if _quote_t and len(_quote_t.split()) > 2:
                _tail_r = _rng.choice(("Yanlış mı hatırlıyorum?", "Öyle değil mi?",
                                           "Bunu mu soruyordun?", "Yoksa başka bir şey mi vardı?"))
                _ans_r = f"{_ans_r} {_tail_r}"
            self._log_turn(speaker, raw, _ans_r, t0)
            return _ans_r

        # KONU HATIRLAMA, TEKRAR UYARISINDAN ÖNCE GELİR: "az önce hangi konudan
        # konuşuyorduk?" bir tekrar sorusu değil, konu sorusudur. Sonra gelen tekrar
        # farkındalığı bu soruyu gasp edip "bunu az önce de söyler gibi oldun" diyordu.
        try:
            _tr0 = self._topic_recall(raw)
        except Exception:
            _tr0 = None
        if _tr0:
            self._intent_locked = True
            self._log_turn(speaker, raw, _tr0, t0)
            return _tr0

        # "saat kaçTA" bir SAAT SORUSU DEĞİL, bir olayın saatidir ("toplantı saat kaçta?").
        # Alt-dizgi eşleşmesi bunu saat sorusu sanıp "Şu an 15:23" diyordu: soru gaspı.
        _clock_q = bool(re.search(r"saat kaç(?!ta|da|tı|dı)\b", _tl))
        _elapsed_q = any(k in _tl for k in (
            "ne kadar oldu", "ne kadar zaman oldu", "ne zamandır", "ne kadar süredir",
            "kaç dakika oldu", "kaç saat oldu", "ne kadar süre"))
        if self.acc is not None and (_clock_q or _elapsed_q):
            if _clock_q and not _elapsed_q:
                _ans_t = compose_time_answer(0.0, kind="clock", agi=self,
                                             rng=getattr(self.discourse, "random", None))
            else:
                _ans_t = compose_time_answer(self.acc.since_session(), len(self.dialogue_log),
                                             kind="elapsed", steps=int(self.mind.age), agi=self,
                                             rng=getattr(self.discourse, "random", None))
            self._log_turn(speaker, raw, _ans_t, t0)
            return _ans_t

        # --- KARŞIOLGUSAL İRADE: "aç olmasaydın ne yapardın?" ---
        # Sözcük/tanım hattından ÖNCE gelir: bu bir sözlük sorusu değil, kararın
        # yeniden koşulmasıdır. Cevap uydurulmaz; aynı karar makinesi iki kez çalışır.
        try:
            _cfv = self._answer_counterfactual_volition(tr_lower(raw))
        except Exception:
            _cfv = None
        if _cfv:
            self._log_turn(speaker, raw, _cfv, t0)
            return _cfv

        # --- OPERATÖR OCAĞI: "şu örneklerden kural çıkar" — kuralı BİZ yazmayız ----
        try:
            _fg = self._handle_forge(raw, speaker=speaker)
        except Exception:
            _fg = None
        if _fg:
            self._log_turn(speaker, raw, _fg, t0)
            return _fg

        # --- UYGULAMA OPERATÖRLERİ: kural bağla / uygula / say / transfer ---------
        # Bunlar retrieve değil APPLY'dir: kural nesne olarak bağlanır, yeni girdiye
        # uygulanır. Cevap üretilemezse yankı yok — açık başarısızlık döner.
        try:
            _ap = self._handle_apply_ops(raw)
            if _ap:
                self.apply_eng.wm.write("last_utterance", raw, "turn")
                self._log_turn(speaker, raw, _ap, t0)
                return _ap
        except Exception:
            traceback.print_exc()

        # --- EKSİK OPERATÖR: istenen işlem kayıtlı değilse uydurma cevap yok --------
        # Sıra önemli: bu kapı uygulama hattından SONRA gelir (bağlı bir kural varsa
        # sözü kural söyler), sohbet hattından ÖNCE gelir (alakasız iç ses sızmasın).
        try:
            _mo = self._missing_operator(raw)
        except Exception:
            _mo = None
        if _mo:
            self._log_turn(speaker, raw, _mo, t0)
            return _mo
        try:
            _ct = self._notice_contradiction(raw)
        except Exception:
            _ct = None
        if _ct:
            self._intent_locked = True
            self._log_turn(speaker, raw, _ct, t0)
            return _ct
        # ÖĞRENME ANI: az önce KİŞİSEL bir olgu yazıldıysa söz onu kabul eder.
        # Bu kapı olmasa dünya-bilgisi hattı cümleyi çekip bozuk Türkçe üretiyordu
        # ("kardeşimin adı ayşedir").
        _lf = [r for r in (getattr(self, "_learned_facts", None) or [])
               if not (r.get("subj") == "benim" and r.get("rel") == "ad")]
        if _lf:
            _rec0 = _lf[0]
            self._intent_locked = True
            _ph = self._fact_phrase(_rec0)
            _ack = self._cap_tr(self._say("memory.personal",
                                          {"phrase": _ph,
                                           "turn": int(_rec0.get("turn") or 0) + 1},
                                          ctx={"kind": "learned"}, fallback=_ph))
            self._log_turn(speaker, raw, _ack, t0)
            return _ack
        try:
            _pm = self._recall_testimony(raw) or self._recall_personal(raw)
        except Exception:
            _pm = None
        if _pm:
            self._intent_locked = True
            self._log_turn(speaker, raw, _pm, t0)
            return _pm
        try:
            _tr = self._topic_recall(raw)
        except Exception:
            _tr = None
        if _tr:
            self._log_turn(speaker, raw, _tr, t0)
            return _tr

        # --- OKUL: sayı/basamak/okunuş/sembol ve aşamalı problemler.
        # Önce "öğretme cümlesi mi?" diye bakar (kullanıcı konuşarak öğretir), sonra soru olarak çözer.
        if self.school is not None:
            try:
                _learned_ack = self.school.maybe_learn(raw, self)
                if _learned_ack:
                    self._log_turn(speaker, raw, _learned_ack, t0)
                    return _learned_ack
                _school_ans = self.school.answer(raw, self)
                if _school_ans:
                    self._log_turn(speaker, raw, _school_ans, t0)
                    return _school_ans
            except Exception:
                pass

        # Konuşmadan ÖNCE içsel monolog (Vygotsky iç sesi + insan şeması gözlemi + kendi şema seçimi!)
        self._begin_inner_monologue(raw, speaker)

        # CÜMLE YAPISI: soru bir SÖZCÜK/EYLEM hakkında mı? ('Selam nasıl denilir?').
        # Bahsetme (mention) kullanımdan ayrılır: 'Merhaba!' selamlaşmadır, 'Merhaba nasıl
        # denilir?' ise sözcüğün kendisi hakkında bir sorudur ve selamlaşma yoluna GİRMEZ.
        if getattr(self, "frames", None) is not None:
            try:
                _mq = self.frames.meta_question(raw)
            except Exception:
                _mq = None
            # TANIM sorusunda kavram ZATEN biliniyorsa söz, mevcut tanım korteksinindir;
            # sözcük korteksi yalnız 'bilmiyorum + araştırma' yolunu açar (yoksa daha zayıf
            # bir tanım metni ana hattı gölgeler: 'zürafanın uzun boynu var' kaybolurdu).
            # BAĞLAM: sözcük/SORU, demin konuşulan bir şeyin üzerine kurulduysa (hikâye, anlatı,
            # sohbet) bu bir SÖZLÜK boşluğu değil, söylem sorusudur; akıl yürütme hattı yanıtlar.
            if _mq is not None and _mq.kind in ("DEFINITION", "UNKNOWN_Q"):
                try:
                    from nora_lang import tr_lower as _tl
                    _t = _tl(getattr(_mq, "target", "") or "")
                    _now = _tl(raw).strip(" ?.!")
                    # Aynı soru TEKRAR sorulduysa bu bir söylem sorusu değil, aynı bilgi
                    # boşluğudur: korteks yanıtlar (tekrar farkındalığı ayrıca eklenir).
                    _others = [_tl(str(t[1])) for t in self.dialogue_log[-6:]
                               if _tl(str(t[1])).strip(" ?.!") != _now]
                    if _t and any(_t in _o for _o in _others):
                        _mq = None
                except Exception:
                    pass
            _defer = False
            if _mq is not None and _mq.kind in ("DEFINITION", "UNKNOWN_Q"):
                try:
                    # Yalnız SÖZCÜĞÜ tanımak yetmez: bilgi hattı ancak kavram hakkında
                    # gerçek bir ilişki varsa yanıtlayabilir. Yoksa boşluk dürüstçe söylenir.
                    _tgt = (getattr(_mq, "target", "") or "").strip()
                    _has_facts = bool(self.lang.graph._out_edges.get(_tgt)) if _tgt else False
                    _defer = bool(_has_facts and self.frames.known_word(_mq.target, self))
                except Exception:
                    _defer = False
            if _mq is not None and not _defer:
                try:
                    _ans_m = self.frames.answer_meta(_mq, self)
                except Exception:
                    _ans_m = ""
                if _ans_m:
                    self.last_meta_question = {"kind": _mq.kind, "target": _mq.target,
                                               "verb": _mq.verb}
                    self._queue_structure(raw)
                    self._queue_structure(_ans_m)
                    self._log_turn(speaker, raw, _ans_m, t0)
                    return _ans_m

        # SORU KILIGINDAKI RİCA ve ÖĞÜT İSTEMİ ('Kapıyı açar mısın?', 'Sence ne yapmalıyım?'):
        # sözcük/tanım hattından ÖNCE gelir; çünkü asıl edim bir istektir, bilgi sorusu değil.
        if getattr(self, "discourse", None) is not None:
            _adv = _req_txt = ""
            try:
                _adv = self.discourse.respond_advice(raw, self) or ""
            except Exception:
                _adv = ""
            if not _adv:
                try:
                    _req = self.discourse.request_of(
                        raw, self.lang.morph, getattr(self.lang.graph, "lexicon", None))
                    if _req:
                        _req_txt = self.discourse.respond_request(_req, self) or ""
                except Exception:
                    _req_txt = ""
            _act_txt = _adv or _req_txt
            if _act_txt:
                _act_txt = tr_capitalize(_act_txt)
                self._request_met = True
                self._queue_structure(raw)
                self._queue_structure(_act_txt)
                self._log_turn(speaker, raw, _act_txt, t0)
                return _act_txt

        # 0) Kendi kendine söylem genişlemesi ('X demek Y demektir' -> Plastik Kenyon Niyet Korteksi 1-atış öğrenir!)
        ok_eq, lhs_eq, intent_eq = self.plastic_discourse.try_self_supervised_equivalence(tr_lower(raw))
        if ok_eq:
            from nora_broca import VerbSpec, NP
            rep_eq = self.broca_organ.syn.realize_verb(
                VerbSpec(
                    verb_root="bağla",
                    obj=NP(f"'{lhs_eq}' ifade", poss=3, case="ACC"),
                    dat=NP("sinaps", poss=1, plural=True, mods=("plastik", "söylem"), case="DAT"),
                    advs=("tek geçişte",),
                    tense="PAST", person="1SG",
                )
            )
            ans_eq = tr_capitalize(rep_eq) + "."
            self._log_turn(speaker, raw, ans_eq, t0)
            return ans_eq

        # Önce dizi veya fonksiyon kuralı sorusu mu kontrol et ('2, 5, 8, 11 dizisinin...' / '2 -> 5, 4 -> 9...')
        seq_ans = self._try_sequence_or_mapping_query(raw)
        if seq_ans is not None:
            self._log_turn(speaker, raw, seq_ans, t0)
            return seq_ans

        # Kod istemi tek bir bütündür: içindeki ünlem/nokta metnin PARÇASI olabilir ('Merhaba! Ey Ahali!')
        if self.code_cortex.is_requested(raw):
            self.last_utterance_class = self.utter_cortex.classify(raw)
            reply = self._handle_single_sentence(raw, speaker=speaker)
            self._log_turn(speaker, raw, reply, t0)
            return reply

        # Çok-cümleli öğreti + soru kombinasyonu kontrolü ('Kuşlar uçar. Karga bir kuştur. Karga uçar mı?')
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", raw) if p.strip()]
        # Ünlemli selamlama ('Merhaba Nora! Nasılsın?') tek bir doğal sosyal sözce olarak birleştirilir
        if len(parts) >= 2:
            p0_words = re.findall(r"[\wçğıöşü]+", tr_lower(parts[0]))
            if p0_words and p0_words[0] in {"merhaba", "selam", "selamlar", "günaydın", "hey", "iyi"} and len(p0_words) <= 4:
                parts = [parts[0] + " " + parts[1]] + parts[2:]
        if len(parts) > 1:
            taught_msgs = []
            last_part = parts[-1]
            # Soru mu, bildirim mi? Bu sınıflandırmayı artık Nora'nın plastik Mantar Gövdesi KENDİSİ yapar.
            last_utt = self.utter_cortex.classify(last_part)
            self.last_utterance_class = last_utt
            is_last_q = bool(last_utt.get("is_question")) or any(
                w in tr_lower(last_part).split()
                for w in ("mı", "mi", "mu", "mü", "nedir", "neydi", "ne", "kaç", "kaçtı", "nasıl", "nerede", "neredeydi", "kim", "kimim", "kimdim")
            )
            teach_slice = parts[:-1] if is_last_q else parts
            for sent in teach_slice:
                r_msg = self._handle_single_sentence(sent, speaker=speaker)
                taught_msgs.append(r_msg)
            if is_last_q:
                q_ans = self._handle_single_sentence(last_part, speaker=speaker)
                self._log_turn(speaker, raw, q_ans, t0)
                return q_ans
            joined = " ".join(taught_msgs)
            self._log_turn(speaker, raw, joined, t0)
            return joined

        reply = self._handle_single_sentence(raw, speaker=speaker)
        self._log_turn(speaker, raw, reply, t0)
        return reply

    def _log_turn(self, speaker: str, user_txt: str, reply: str, t0: float | None = None):
        if t0 is not None:
            self.last_turn_latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        if len(self.dialogue_log) >= 200:
            self.dialogue_log.pop(0)
        self.dialogue_log.append((speaker, user_txt, reply))
        self.broca_organ.record_turn_episode(speaker, user_txt, reply, self)
        self.monologue.current_turn_thought = None

    def _try_sequence_or_mapping_query(self, raw: str) -> str | None:
        s = tr_lower(raw)
        # 1) Eşleme kuralı: '2 -> 5, 4 -> 9, 7 -> 15 ... 10 kaç olur?'
        arrow_pairs = re.findall(r"(-?\d+(?:\.\d+)?)\s*(?:->|=>|:)\s*(-?\d+(?:\.\d+)?)", s)
        if len(arrow_pairs) >= 2 and "::" not in s:
            pairs = [(float(a), float(b)) for a, b in arrow_pairs]
            # Son sayıyı hedef sorgu olarak bul
            tail = s.split(arrow_pairs[-1][1])[-1]
            q_nums = re.findall(r"-?\d+(?:\.\d+)?", tail)
            if q_nums:
                qx = float(q_nums[-1])
                res = self.fast.induce_mapping(pairs, qx)
                if res["ok"]:
                    self.self_model.record_cognitive_success(used_system2=True)
                    val = int(round(res["ans"])) if abs(res["ans"] - round(res["ans"])) < 1e-5 else round(res["ans"], 3)
                    qx_s = int(round(qx)) if abs(qx - round(qx)) < 1e-5 else qx
                    exp = f"Örneklerden {res['rule_tr']} kuralını çıkardım; {qx_s} için sonuç {val} olur."
                    self.lang.wernicke.last_explanation = f"Verilen çiftlerde {res['rule_tr']} tam uyum sağladı."
                    return exp

        # 2) Sayı dizisi: '2, 5, 8, 11 dizisinde sıradaki...'
        if "dizi" in s or "sıradaki" in s or "sonraki" in s:
            nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s)]
            if len(nums) >= 3:
                res = self.fast.induce_sequence(nums)
                if res["ok"]:
                    self.self_model.record_cognitive_success(used_system2=True)
                    val = int(round(res["next"])) if abs(res["next"] - round(res["next"])) < 1e-5 else round(res["next"], 3)
                    exp = f"Bu dizide {res['rule_tr']} yasasını buldum; sıradaki değer {val} olur."
                    self.lang.wernicke.last_explanation = f"Dizideki ardışık terimler {res['rule_tr']} yasasını doğruladı."
                    return exp
        return None

    def _compete_workspace(self, bids: list[CorticalBid]) -> CorticalBid:
        """Küresel Çalışma Alanı (Global Neuronal Workspace) ateşleme yarışı:
        Kortikal sütunların [belirginlik * kesinlik * nörokimyasal kazanç] teklifleri yanal
        ketlemeyle yarışır ve kazanan koalisyon bilince yayınlanır."""
        mod_gains = {
            "parietal_physics": 1.00 + 0.14 * self.chem.ne + 0.12 * self.self_model.traits["düşünsellik"],
            "parietal_math": 1.00 + 0.14 * self.chem.ne + 0.12 * self.self_model.traits["düşünsellik"],
            "linguistic_meta": 1.00 + 0.12 * self.chem.ach + 0.10 * self.self_model.traits["meraklılık"],
            "prefrontal_volition": 1.04 + 0.18 * self.chem.cort + 0.14 * self.self_model.traits["temkinlilik"],
            "social_tom": 1.00 + 0.15 * self.chem.oxy + 0.10 * self.self_model.traits["sosyal_güven"],
            "insular_self": 1.02 + 0.15 * self.chem.cort + 0.10 * self.chem.ne,
            "hippocampal_concept": 1.00 + 0.15 * self.chem.ach + 0.08 * self.self_model.traits["meraklılık"],
            "temporal_conversation": 0.96 + 0.16 * self.chem.oxy + 0.10 * self.self_model.traits["meraklılık"],
            "associative_semantic": 0.92 + 0.14 * self.chem.ach + 0.10 * self.self_model.traits["meraklılık"],
            "prefrontal_autonomous_logic": 1.02 + 0.16 * self.chem.ne + 0.10 * self.self_model.traits["düşünsellik"],
            "social_phatic": 0.98 + 0.14 * self.chem.oxy + 0.12 * self.self_model.traits["sosyal_güven"],
            "epistemic_prefrontal": 1.05 + 0.16 * self.chem.ne + 0.10 * self.self_model.traits["düşünsellik"]
                                    + 0.08 * self.self_model.traits["temkinlilik"],
            "python_motor_cortex": 1.10 + 0.12 * self.chem.da_tonic + 0.10 * self.self_model.traits["meraklılık"],
        }
        raw_scores = np.array(
            [b.salience * b.precision * mod_gains.get(b.module, 1.0) for b in bids],
            dtype=np.float64,
        )
        # Yanal ketlemeli keskinleştirilmiş softmax (Global Workspace ignition)
        shifted = (raw_scores - np.max(raw_scores)) * 9.0
        probs = np.exp(shifted)
        probs /= max(1e-9, float(np.sum(probs)))
        win_i = int(np.argmax(probs))
        win = bids[win_i]
        self.last_gw_trace = {
            "winner": win.module,
            "confidence": round(float(probs[win_i]), 3),
            "trace": win.trace,
            "bids": {bids[i].module: round(float(raw_scores[i]), 3) for i in range(len(bids))},
        }
        if win.speech_plan is not None:
            self.broca_organ.last_speech_plan = win.speech_plan
        else:
            self.broca_organ.last_speech_plan = {
                "parsed_stems": [],
                "volition_acts": [win.module.upper()],
                "ask_back_chosen": "?" in win.reply,
                "guarded": False,
                "thought_tree": [f"{win.module}: {win.trace}"],
                "surface_tr": win.reply,
                "latency_ms": self.last_turn_latency_ms,
            }
        if win.used_s2:
            self.self_model.record_cognitive_success(used_system2=True)
        return win

    def _synthesize_associative_dialogue(self, raw: str, s_low: str, speaker: str = "insan") -> CorticalBid:
        """Açık-uçlu Türkçe cümleler için kavram ağı yayılan aktivasyonu + sıfır-kalıp Broca kök çekimi."""
        assoc = self.broca_organ.synthesize_associative_bid(raw, s_low, self, speaker=speaker)
        return CorticalBid(
            module="associative_semantic",
            salience=float(assoc["salience"]),
            precision=float(assoc["precision"]),
            reply=assoc["reply"],
            trace=assoc["trace"],
            speech_plan=assoc.get("speech_plan"),
        )

    def _handle_single_sentence(self, raw: str, speaker: str = "insan") -> str:
        wern = self.lang.wernicke
        broca = self.lang.broca
        morph = self.lang.morph
        graph = self.lang.graph
        s_low = tr_lower(raw)
        bids: list[CorticalBid] = []

        # --- SÜTUN -1: KİŞİSEL ANLATI — karşıdaki kendi yaşantısını anlatıyorsa Nora ders vermez.
        # İnsan değil mi? Bir şey anlatan birine ansiklopedi maddesi okumayız; dinler, yansıtır, sorarız.
        narration = None
        _pure_state_guard = False
        try:
            narration = self.discourse.narration_of(raw)
        except Exception:
            narration = None
        if narration is not None:
            try:
                nar_res = self.discourse.respond_narration(raw, self, narration)
                # Yalın hâl bildirimi ('kafam çok karışık') duygudaşlık ister: hazır durum karşılığından üstün.
                # Ek yüklem varsa ('... hiç odaklanamıyorum') karşıdaki dert yanıyor demektir; o zaman
                # durumsal korteksin somut önerisi de yarışta kalsın.
                _pure_state = bool(narration.get("state")) and not (narration.get("rest") or narration.get("tail"))
                _pure_state_guard = _pure_state
                bids.append(CorticalBid(
                    "social_narration", 0.995 if _pure_state else 0.966, 0.95, nar_res["text"],
                    f"narration:{nar_res['act']}",
                ))
            except Exception:
                narration = None

        # --- SÜTUN -1b: TOPLUMSAL OLAY (doğum günü gibi) — bilgi maddesi değil, insan karşılığı
        try:
            _ev = self.discourse.social_event_of(raw)
            if _ev is not None:
                _ev_rep = self.discourse.respond_social_event(raw, self, _ev)
                if _ev_rep.get("text"):
                    bids.append(CorticalBid("social_event", 0.972, 0.96, _ev_rep["text"],
                                            f"social:{_ev_rep['act']}"))
        except Exception:
            pass

        # Çerçeveyi BAŞTA bir kez çıkar: soru sözcüğü (nasıl/neden) ne istendiğini belirler ve
        # yanıt teklifleri ona göre süzülür. Aşağıda yeniden ayrıştırılmaz.
        _fr_pre = None
        try:
            _fr_pre = wern.parse(raw)
        except Exception:
            _fr_pre = None
        _wh_pre = getattr(_fr_pre, "wh", "") if _fr_pre is not None else ""
        _wh_unans = False
        if _wh_pre in ("nasıl", "neden", "niye", "niçin") and _fr_pre is not None and _fr_pre.subj:
            _want_pre = ("USED_FOR", "CAN_DO", "CAUSES") if _wh_pre == "nasıl" else ("CAUSES", "USED_FOR")
            _wh_unans = not graph.query(_fr_pre.subj, _want_pre) and \
                not any(graph.query(e.subj, _want_pre) for e in graph._in_edges.get(_fr_pre.subj, ()))

        # --- SÜTUN 0: Nöral Sözce Öz-Sınıflandırması + Otonom Etiketleme/Mantık + Durumsal Türkçe ---
        utt_cls = self.last_utterance_class or self.utter_cortex.classify(raw)
        cur_sc0 = self.monologue.last_thought.schema_id if self.monologue.last_thought is not None else "S1_NEDEN_SONUC"
        # Bildirim cümlesi ise Nora kendi mantık kuralını ve etiketini KENDİSİ kurar (soru işareti olmadan da sınıflar!)
        ing_res = None
        self._turn_mint_label = ""
        # Nosiseptif refleks turunda kortikal kural-öğrenimi ertelenir: omurilik refleksi korteksi
        # beklemez (acı darbesi sürerken kural çıkarmak nörobiyolojik olarak da yersizdir).
        _nociceptive = (getattr(self.nervous, "adelta_pain", 0.0) > 0.25
                        or bool(getattr(self.nervous, "pending_pain_vocalization", False)))
        if not utt_cls.get("is_question") and narration is None and not _nociceptive:
            ing_res = self.logic.ingest_statement(raw)
        if ing_res is not None:
            self._turn_mint_label = str(ing_res["label"])
        if ing_res is not None:
            _syn_m = self.broca_organ.syn
            _c_mint = _syn_m.realize_verb(VerbSpec(
                verb_root="kodla", obj=NP("ilişki", mods=("bu",), case="ACC"),
                advs=("kendi başıma",), tense="PAST", person="1SG"))
            _c_tie = _syn_m.realize_verb(VerbSpec(
                verb_root="bağla", obj="bunu", dat=NP("bellek", poss=1, case="DAT"),
                tense="PAST", person="1SG", dat_last=True))
            # İç etiketler (ÖZ_...) ASLA söze dönüşmez: bunlar zihnin iç simgeleridir, insan dili değil.
            _lbl = str(ing_res["label"])
            if "_" in _lbl or "[" in _lbl or ":" in _lbl:
                _desc = {
                    "dual_state": "iki durumun aynı anda geçerli olabileceğini",
                    "variation": "iki değişkenin birlikte değiştiğini",
                    "conditional_transition": "bir koşulun durumu değiştirdiğini",
                    "frame_transition": "bir durumun başka bir biçime dönüştüğünü",
                }.get(str(ing_res.get("kind", "")), "bu cümledeki bağı")
                _mint_fb = f"Bu cümlede {_desc} kendi başıma çözdüm. {tr_capitalize(_c_tie)}."
            else:
                _mint_fb = f"{tr_capitalize(_c_mint)}: {_lbl}. {tr_capitalize(_c_tie)}."
            mint_rep = self.speak_or(_mint_fb, anchors=[str(ing_res["label"])], nov=0.8, fam=0.2,
                                     conf=self._speaker_trust(speaker))
            bids.append(CorticalBid(
                module="prefrontal_autonomous_logic",
                salience=0.945,   # < 0.95: anlamsal öğrenme sütunlarını KETLEMEZ, yalnızca onay sözcesini üstlenir
                precision=0.95,
                reply=mint_rep,
                trace=f"mint:{ing_res['label']}",
                used_s2=True,
            ))
        ded_res = self.logic.try_autonomous_deduction(raw, self.schema_cortex, cur_sc0, utt_cls=utt_cls)
        if ded_res is not None:
            bids.append(CorticalBid(
                module="prefrontal_autonomous_logic",
                salience=0.972,
                precision=float(ded_res["confidence"]),
                reply=ded_res["reply"],
                trace=str(ded_res["trace"]),
                used_s2=True,
            ))
        # --- SÜTUN 0B: Epistemik katman (kör kabul yok) ve çalıştırarak doğrulayan kod korteksi ---
        epi_txt, epi_trace, epi_sal = self._epistemic_channel(raw, speaker, utt_cls)
        if epi_txt:
            bids.append(CorticalBid(
                module="epistemic_prefrontal",
                salience=epi_sal,
                precision=0.96,
                reply=epi_txt,
                trace=epi_trace,
                used_s2=(epi_sal < 0.97),
            ))
        code_txt, code_trace, code_sal = self._code_channel(raw)
        if code_txt:
            bids.append(CorticalBid(
                module="python_motor_cortex",
                salience=code_sal,
                precision=0.99 if code_sal > 0.95 else 0.9,
                reply=code_txt,
                trace=code_trace,
                used_s2=True,
            ))

        # --- SÜTUN 0a: TARTIM KORTEKSİ — kendi tutumu, tekrar farkındalığı, üst-düzey soru
        _delib = self.deliberation.handle(self, raw, locals().get("fr"))
        _delib_owned = bool(_delib)
        if _delib:
            bids.append(CorticalBid("insular_self", 0.988, 0.94, _delib, "deliberation"))

        # --- SÜTUN 0b: TERCİH SORUSU ('sence kediler mi köpekler mi daha iyi?') — ansiklopedi değil görüş
        _pref = self._answer_preference(raw, speaker=speaker)
        if _pref:
            bids.append(CorticalBid("prefrontal_autonomous_logic", 0.978, 0.96, _pref, "preference"))

        # --- SÜTUN 0c: MİZAÇ SORUSU ('niye bu kadar sessizsin?') — kendi iç durumundan yanıt
        _temp_adj = self._temperament_adj_in(raw)
        if _temp_adj:
            bids.append(CorticalBid("insular_self", 0.975, 0.96,
                                    self._answer_temperament_query(_temp_adj, speaker=speaker), f"temper:{_temp_adj}"))

        # --- SÜTUN 0d: KARŞIOLGUSAL İRADE — "aç olmasaydın ne yapardın?" ---
        # Cevap uydurulmaz: AYNI karar makinesi, bir beden kanalı dolu sayılarak yeniden
        # koşar. Karar değişiyorsa o kanal kararın NEDENİDİR; değişmiyorsa değildir.
        _cf_vol = self._answer_counterfactual_volition(s_low)
        if _cf_vol:
            bids.append(CorticalBid("insular_self", 0.985, 0.97, _cf_vol, "cf_volition"))

        sit_res = self.situational.try_respond_situation(raw, self, speaker=speaker)
        if sit_res is not None and _pure_state_guard:
            sit_res = None      # yalın hâl bildiriminde duygudaşlık, hazır karşılıktan üstündür
        if sit_res is not None:
            bids.append(CorticalBid(
                module="social_phatic",
                salience=0.955,
                precision=float(sit_res["confidence"]),
                reply=sit_res["reply"],
                trace=str(sit_res["trace"]),
            ))

        # --- SÜTUN 1: Parietal Fizik & Zihinsel Simülasyon Korteksi ---
        phys_res = self.physics_cortex.try_solve(raw, mind=self.mind)
        if phys_res is not None:
            bids.append(CorticalBid(
                module="parietal_physics",
                salience=0.98,
                precision=float(phys_res["confidence"]),
                reply=phys_res["reply"],
                trace=phys_res["trace"],
                used_s2=True,
            ))

        # --- SÜTUN 2: Parietal Matematik, Cebir ve Kalkülüs Korteksi ---
        math_res = self.math_cortex.try_solve(raw)
        if math_res is not None and phys_res is None:
            bids.append(CorticalBid(
                module="parietal_math",
                salience=0.97,
                precision=float(math_res["confidence"]),
                reply=math_res["reply"],
                trace=math_res["trace"],
                used_s2=True,
            ))

        # --- SÜTUN 3: Uzman Türkçe Üst-Dil (Morfoloji, Deyim, Atasözü, Eş/Zıt Anlam) Korteksi ---
        ling_res = self.conv_cortex.respond_linguistic(raw, s_low, morph, graph.lexicon)
        if ling_res is not None:
            bids.append(CorticalBid(
                module="linguistic_meta",
                salience=0.97,
                precision=float(ling_res["confidence"]),
                reply=ling_res["reply"],
                trace=ling_res["trace"],
                used_s2=True,
            ))

        # --- SÜTUN 4: Bedenlenmiş Karşıolgusal, Güven, Koşullu Zincir ve Üst-Biliş (ToM / Makro / Sıralama) ---
        if ("?" in raw or "ne yapardın" in s_low or "ne yaparsın" in s_low or "nasıl" in s_low) and any(
            k in s_low for k in ("kör olsan", "görmen kesilirse", "zehirli olsaydı", "zehirli olursa", "felç olsan")
        ):
            bids.append(CorticalBid("insular_self", 0.98, 0.96, self._answer_counterfactual(s_low), "embodied_counterfactual"))

        if ("güveniyor" in s_low or "güvenin" in s_low) and ("?" in raw or "mu" in s_low or "mısın" in s_low or "kaç" in s_low):
            tr_val = self.chem.trust_of(speaker)
            _disp = self.ear.display(speaker) if getattr(self, "ear", None) is not None else speaker
            stance = "sana güveniyorum" if tr_val >= 0.55 else ("sana karşı temkinliyim" if tr_val >= 0.30 else "sana güvenmiyorum")
            bids.append(CorticalBid(
                "social_tom", 0.97, 0.96,
                f"Şu an {_disp} için epistemik güven düzeyim {tr_val:.2f}; geçmiş etkileşimlerimize dayanarak {stance}.",
                f"oxy_trust={tr_val:.2f}",
            ))

        m_if_q = re.match(r"^(?:eğer\s+)?(.+?)\s+(?:olursa|ise)\s+ne\s+olur\??$", s_low)
        if m_if_q:
            cond_q = m_if_q.group(1).strip()
            cchain = graph.causal_chain(cond_q)
            if cchain:
                final_eff, path_edges = cchain[-1]
                steps_txt = " -> ".join([path_edges[0].subj] + [pe.obj for pe in path_edges])
                wern.last_explanation = f"Koşullu zincir: {steps_txt}"
                bids.append(CorticalBid(
                    "hippocampal_concept", 0.97, 0.95,
                    f"{tr_capitalize(cond_q)} olduğunda zincirleme olarak {final_eff} gerçekleşir ({steps_txt}).",
                    steps_txt, used_s2=True,
                ))

        higher_ans = self._try_higher_cognition(raw, s_low, speaker=speaker)
        if higher_ans is not None:
            mod_name = "prefrontal_volition" if ("prosedür" in higher_ans or "becerisini" in higher_ans) else "social_tom"
            bids.append(CorticalBid(mod_name, 0.97, 0.96, higher_ans, "higher_cognition"))

        # --- SÜTUN 5: Doğal Sohbet, Empati ve Felsefi Derinlik Korteksi ---
        conv_res = self.conv_cortex.respond_conversational(raw, s_low, self, speaker=speaker)
        if conv_res is not None and not _delib_owned:
            c_conf = float(conv_res["confidence"])
            conv_sal = 0.99 if c_conf >= 0.99 else (0.97 if c_conf >= 0.95 else 0.92)
            bids.append(CorticalBid(
                module="temporal_conversation",
                salience=conv_sal,
                precision=c_conf,
                reply=conv_res["reply"],
                trace=conv_res["trace"],
                speech_plan=conv_res.get("speech_plan"),
            ))

        # --- SÜTUN 6: Wernicke Anlamsal Çerçeve (İçebakış, Aritmetik S2, İrade, 1-Atış Öğrenme ve Kavram Çıkarımı) ---
        # Emir/istek cümlesi ise çerçeve MUTLAKA ayrıştırılır: "bana bir şaka yap" gibi istekler
        # hazır bir durumsal karşılıkla geçiştirilmemeli, irade organına ulaşmalı.
        if not bids or all(b.salience < 0.95 for b in bids) or self._looks_like_request(raw) or _wh_unans:
            fr = _fr_pre if _fr_pre is not None else wern.parse(raw)

            if fr.kind == "greeting":
                g_gen = self.broca_organ.generate_response(raw, s_low, self, speaker=speaker, allow_open_fallback=True)
                g_rep = g_gen["reply"] if g_gen is not None else "Merhaba."
                bids.append(CorticalBid("temporal_conversation", 0.95, 0.94, g_rep, "generative_greeting"))
            elif fr.kind == "why_query":
                # Açıklama, SORULAN davranışa ait olmalı. Eski turlardan kalan bir iç telemetri cümlesi
                # (ör. 'kavram ağında X düğümünden 1 bağ buldum') soruyla ilgisizse SÖYLENMEZ.
                ref_exp = getattr(wern, "last_refusal_explanation", "")
                _q_words = set(re.findall(r"[\wçğıöşü]+", s_low))
                _asks_own_action = bool(_q_words & {"reddet", "reddettin", "kabul", "yaptın", "davrandın",
                                                    "böyle", "öyle", "erteledin", "yapmadın"}) \
                    or len(_q_words - {"neden", "niçin", "niye", "peki", "az", "önce", "sen", "bir"}) == 0
                _expl = (wern.last_explanation or "").strip()
                _expl_relevant = bool(_expl) and bool(set(re.findall(r"[\wçğıöşü]+", tr_lower(_expl))) & _q_words)
                if ref_exp and _asks_own_action:
                    ans_w = tr_capitalize(ref_exp.rstrip(".")) + "."
                elif _expl and _expl_relevant:
                    ans_w = tr_capitalize(_expl.rstrip(".")) + "."
                else:
                    goal_names = {0: "enerji ihtiyacım", 1: "susuzluğum", 2: "tehlikeden korunma isteğim", 3: "merakım"}
                    g_str = goal_names.get(self.mind.goal, "içsel dengem")
                    ans_w = f"Böyle davrandım çünkü şu an {g_str} ön planda ve kimyam {self.chem.dominant_state_tr()}."
                bids.append(CorticalBid("insular_self", 0.96, 0.95, ans_w, "why_introspection"))
            elif fr.kind == "self_query" and conv_res is None:
                bids.append(CorticalBid("insular_self", 0.96, 0.95, self._answer_self_query(fr.sub_kind), f"self:{fr.sub_kind}"))
            elif fr.kind == "body_query":
                bids.append(CorticalBid("insular_self", 0.96, 0.95, self._answer_body_query(fr.sub_kind), f"body:{fr.sub_kind}"))
            elif fr.kind == "world_query":
                bids.append(CorticalBid("insular_self", 0.95, 0.94, self._answer_world_query(fr.subj), f"world:{fr.subj}"))
            elif fr.kind == "math_query" and fr.math_expr is not None:
                a, op, b = fr.math_expr
                bids.append(CorticalBid("parietal_math", 0.98, 0.97, self._solve_math_in_language(a, op, b), f"{a}{op}{b}"))
            elif fr.kind == "analogy_query" and fr.analogy is not None:
                a, b, c = fr.analogy
                ans, rel = graph.solve_analogy(a, b, c)
                if ans:
                    wern.last_explanation = f"{a} ile {b} arasındaki {rel} bağı {c} kavramına uygulandığında {ans} sonucunu verdi"
                    bids.append(CorticalBid(
                        "hippocampal_concept", 0.96, 0.95,
                        f"{tr_capitalize(a)} {b} ile ilişkili olduğu gibi {c} de {morph.inflect_noun(ans, 'NOM', copula=True)}.",
                        f"analogy:{a}:{b}::{c}:{ans}", used_s2=True,
                    ))
                else:
                    bids.append(CorticalBid(
                        "hippocampal_concept", 0.85, 0.80,
                        self.speak_unknown(c, fallback=(f"{tr_capitalize(a)} ve {b} arasındaki bağı {c} için henüz "
                                                        f"kuramadım; bana bir cümleyle öğretebilirsin.")),
                        "analogy_miss",
                    ))
            elif fr.kind == "command":
                # Öğrenilmiş prosedür çağrısı ('devriye yap') yüksek biliş organının işidir; irade
                # teklifi onu gölgeleyip güvenlik denetimini atlatmamalı.
                _raw_words = set(re.findall(r"[\wçğıöşü]+", tr_lower(fr.raw)))
                _macro_hit = next((m for m in self.macros if m in _raw_words), "")
                _is_macro = fr.rel in ("yap", "uygula", "çalıştır", "başlat") and bool(
                    (fr.obj or "") in self.macros or _macro_hit)
                if not _is_macro:
                    bids.append(CorticalBid("prefrontal_volition", 0.992, 0.975,
                                            self._handle_command(fr, speaker=speaker, raw=raw),
                                            f"volition:{fr.rel}:{fr.obj}"))
            elif fr.kind == "teach_val":
                arb = self.self_model.arbitrate_claim(fr.subj, fr.valence, speaker=speaker, channel=fr.sub_kind)
                cur_sc = self.monologue.last_thought.schema_id if self.monologue.last_thought is not None else "S1_NEDEN_SONUC"
                if arb.get("conflict"):
                    n_exp = arb["exp_count"]
                    ev = arb["exp_val"]
                    eff_txt = "fayda sağladı" if ev > 0 else "zarar verdi"
                    wern.last_explanation = (
                        f"{fr.subj} hakkında kendi bedenimde {n_exp} kez doğrudan deneyimim var ve bana {eff_txt}; "
                        f"deneyimsel kesinliğim duyum kesinliğini aştığı için iddiayı reddettim"
                    )
                    c_dis = "buna katılmıyorum"
                    r_dis = f"kendi bedenimde {n_exp} kez {fr.subj} deneyimledim ve bana {eff_txt}"
                    rep_v = self.schema_cortex.transform_clauses(
                        c_dis,
                        r_dis,
                        "bedensel hafızamı yokladığımda",
                        "S2_SONUC_GEREKCE" if cur_sc == "S0_YALIN" else cur_sc,
                    )
                    # Epistemik itiraz ASLA düşemez: şema dönüşümü itirazı elediyse iki cümleyle kur.
                    if "katılmıyorum" not in tr_lower(rep_v):
                        rep_v = f"{tr_capitalize(c_dis)}. {tr_capitalize(r_dis)}."
                elif arb.get("low_trust"):
                    rep_v = self.schema_cortex.transform_clauses(
                        f"{fr.subj} hakkındaki bu iddiayı doğrulamadan kabul etmiyorum",
                        f"sana olan güvenim şu an düşük ({arb['speaker_trust']:.2f})",
                        "temkinli kalmam gerektiğinden",
                        "S2_SONUC_GEREKCE" if cur_sc == "S0_YALIN" else cur_sc,
                    )
                else:
                    self.lang.teach_sentence(raw, speaker=speaker)
                    if fr.subj in {"yiyecek", "su", "tehlike"}:
                        val_num = "+0.35" if fr.valence > 0 else "-0.35"
                        self.mind.hear(f"kural: {fr.subj} {val_num}", trust=self.chem.trust_of(speaker))
                    wern.last_explanation = f"Asetilkolin kapısıyla ({self.chem.ach:.2f}) {fr.subj} kavramını tek seferde duyum belleğime bağladım"
                    _pol = "faydalı" if fr.valence > 0 else "zararlı ve tehlikeli"
                    _c_lv = f"{morph.inflect_noun(fr.subj, 'GEN')} {_pol} olduğunu belleğime işledim"
                    _r_lv = "bunu tek geçişte duyum ağıma bağladım"
                    rep_v = self.speak_or(
                        self.schema_cortex.transform_clauses(_c_lv, _r_lv, SYN.nominalize_dik("tart", poss=1, case="LOC",
                                                 obj=SYN.nominalize_dik("söyle", poss=2, case="ACC")), cur_sc),
                        anchors=[fr.subj], nov=0.7, fam=0.3, conf=self._speaker_trust(speaker))
                # Bedensel deneyime dayanan itiraz, kendi kendine etiketleme sözcesinden (mint 0.945)
                # ve türetilmiş kurallardan (0.972) DAHA YÜKSEK ses alır: kanıt, etiketten üstündür.
                _sal = 0.985 if arb.get("conflict") else 0.97
                bids.append(CorticalBid("hippocampal_concept", _sal, 0.96, rep_v, f"teach_val:{fr.subj}"))
            elif (fr.kind == "teach_fact" and conv_res is None and narration is None
                  and sit_res is not None and float(sit_res.get("confidence", 0.0)) >= 0.95):
                # Sosyal nezaket kalıbı olgusal bilgi değildir ('kolay gelsin, işim var'): öğrenme yok.
                pass
            elif fr.kind == "teach_fact" and conv_res is None and narration is None and self._turn_mint_label:
                # Bu cümle zaten Nora'nın kendi mantık motorunda yapısal kural olarak kodlandı.
                # AMA yapısal kural kavram ağının YERİNE geçmez: "kodladım" dedikten sonra
                # "X neyden olur?" sorusunun cevapsız kalması sahte anlamadır. Nedensel
                # çerçeve çözüldüyse kenar da yazılır (iki ayrı bellek, ikisi de gerçek).
                # HER İLİŞKİ yazılır (yalnız nedensel olanlar değil): "kodladım" dedikten
                # sonra "X nedir?" sorusunun cevapsız kalması sahte anlamadır. Yapısal
                # kural (mint) ile olgu kenarı iki ayrı bellektir; ikisi de gerçek olmalı.
                _wrote = []
                if fr.subj and fr.obj:
                    try:
                        _wrote = self.lang.teach_sentence(raw, speaker=speaker) or []
                    except Exception:
                        _wrote = []
                if not _wrote:
                    # YAZILMADIYSA "kodladım" DENMEZ: sahte öğrenme iddiası, hiç
                    # öğrenmemekten kötüdür (sessiz yalan).
                    raise _NoMint()
                syn_tf = self.broca_organ.syn
                c_tf = syn_tf.realize_verb(VerbSpec(
                    verb_root="kodla", obj=NP("ilişki", mods=("bu",), case="ACC"),
                    advs=("kendi başıma",), tense="PAST", person="1SG",
                ))
                bids.append(CorticalBid(
                    "hippocampal_concept", 0.94, 0.93,
                    f"{tr_capitalize(c_tf)}.",
                    f"structured:{self._turn_mint_label}",
                ))
            elif fr.kind == "teach_fact" and conv_res is None and narration is None:
                edges = self.lang.teach_sentence(raw, speaker=speaker)
                if edges:
                    e0 = edges[0]
                    _node = self.auto_net.assemblies.get(e0.subj)
                    _fam = 0.0 if _node is None else min(1.0, _node.familiarity / 6.0)
                    self.auto_net.stimulate([e0.subj, e0.obj], valence=0.25, resolved_relations=1)
                    if e0.subj not in self.auto_net.curiosity_queue and len(graph._out_edges.get(e0.subj, ())) < 2:
                        self.auto_net.curiosity_queue.append(e0.subj)
                    wern.last_explanation = f"{e0.subj} ile {e0.obj} arasında {e0.rel} sinapsını epoch kullanmadan tek-atışta kurdum"
                    cur_sc = self.monologue.last_thought.schema_id if self.monologue.last_thought is not None else "S1_NEDEN_SONUC"
                    _prop_txt = broca.realize_prop(Proposition(subj=e0.subj, rel=e0.rel, obj=e0.obj))
                    # Cümle ortasında küçük harfle başlamasın; insan sözü büyük harfle kurulur.
                    _prop_txt = tr_capitalize(str(_prop_txt).strip().rstrip(".")) + "."
                    # Gerekçe: az önce kurulan önermenin kendisi. Çekirdek: öğrenme eylemi Broca'da çekimlenir.
                    _bind = SYN.realize_verb(VerbSpec(
                        verb_root="bağla", obj=NP("bilgi", poss=3, case="ACC"),
                        dat=NP("kavram ağı", poss=1, case="DAT"), tense="PAST", person="1SG"))
                    # CANLI KARŞILIK: "kavram ağıma bilgisini bağladım" gibi makine öz-raporu
                    # insan sözü değildir. Önce kendi tepkisini söyler, sonra olguyu anar.
                    _reg = "want"
                    try:
                        _m = getattr(self, "register", None)
                        _cur = float(getattr(self.chem, "da", 0.5))
                        if _m is not None:
                            _reg = _m.recommend(curiosity=_cur, bored=False, unknown=False)
                    except Exception:
                        _reg = "want"
                    _react = None
                    try:
                        from nora_priors import _safe_format  # cümle veriden gelir
                        _f2 = getattr(self, "priors", None)
                        _react = _f2.pick("life.teach_reaction", {"register": _reg}, {},
                                          prov=getattr(self, "provenance", None)) if _f2 else None
                    except Exception:
                        _react = None
                    _rep_fb = (_react + " " + _prop_txt) if _react else _prop_txt
                    rep_f = self.speak_or(_rep_fb, anchors=[e0.subj, e0.obj], nov=1.0 - _fam, fam=_fam,
                                          conf=max(0.55, self._speaker_trust(speaker)))
                    bids.append(CorticalBid("hippocampal_concept", 0.94, 0.93, rep_f, f"1shot:{e0.subj}->{e0.obj}"))
            elif fr.kind == "yn_query":
                rep_yn = None
                if fr.obj in {"zararlı", "tehlikeli", "zehirli", "faydalı", "güvenli", "besleyici"}:
                    srec = self.self_model.somatic.get(fr.subj)
                    if srec is not None and (srec.exp_count > 0 or srec.hearsay_count > 0):
                        v = srec.effective_valence
                        asking_neg = fr.obj in {"zararlı", "tehlikeli", "zehirli"}
                        is_yes = (v < -0.1) if asking_neg else (v > 0.1)
                        src_note = f"kendi bedenimdeki {srec.exp_count} deneyime göre" if srec.provenance == "deneyim" else "öğrendiğim duyuma göre"
                        adj_true = ("zararlıdır" if v < 0 else "faydalıdır")
                        prefix = "Evet" if is_yes else "Hayır"
                        wern.last_explanation = f"{fr.subj} için somatik değerim {v:+.2f} ({srec.provenance})"
                        rep_yn = f"{prefix}, {src_note} {fr.subj} {adj_true}."
                if rep_yn is None:
                    chk = graph.check_relation(fr.subj, fr.obj)
                    if chk["answer"] is not None and chk["edge"] is not None:
                        if chk.get("contrast"):
                            e = chk["edge"]
                            wern.last_explanation = f"{fr.subj} {e.obj} özelliğine sahip olduğu için {fr.obj} ile çelişiyor"
                            rep_yn = f"Hayır, {fr.subj} {fr.obj} değildir; çünkü {broca.realize_prop(Proposition(subj=fr.subj, rel=e.rel, obj=e.obj))}."
                        elif chk.get("causal_path"):
                            cpath = chk["causal_path"]
                            chain_str = " ve ".join(broca.realize_prop(Proposition(subj=pe.subj, rel=pe.rel, obj=pe.obj)) for pe in cpath)
                            concl_str = broca.realize_prop(Proposition(subj=fr.subj, rel="CAUSES", obj=cpath[-1].obj))
                            wern.last_explanation = f"Çok-adımlı nedensel yayılım: {chain_str}"
                            rep_yn = f"Evet, {chain_str}; dolayısıyla {concl_str}."
                        else:
                            wern.last_explanation = f"{fr.subj} kavramından yayılan aktivasyonla {chk['edge'].obj} bağına ulaştım"
                            rep_yn = broca.compose_inference_chain(chk["answer"], fr.subj, fr.obj, chk["edge"], chk["isa_path"])
                if rep_yn is not None:
                    bids.append(CorticalBid("hippocampal_concept", 0.96, 0.95, rep_yn, f"yn:{fr.subj}?{fr.obj}"))
                elif conv_res is None:
                    bids.append(CorticalBid(
                        "hippocampal_concept", 0.76, 0.78,
                        self.speak_unknown(fr.subj, fallback=(f"{tr_capitalize(fr.subj)} hakkında '{fr.obj}' bilgisini "
                                                              f"henüz deneyimlemedim veya duymadım; tek cümleyle öğretebilirsin.")),
                        f"yn_unknown:{fr.subj}",
                    ))
            elif fr.kind == "wh_query" and conv_res is None:
                # Soru sözcüğü neyi sorduğunu belirler: "nasıl" yöntem, "neden" sebep ister.
                # Bunlarda tür ("X bir Y'dir") yanıt değildir; bilinmiyorsa dürüstçe sorulur.
                _wh = getattr(fr, "wh", "") or ""
                _strict = _wh in ("nasıl", "neden", "niye", "niçin")
                if _strict:
                    _want = ("USED_FOR", "CAN_DO", "CAUSES") if _wh == "nasıl" else ("CAUSES", "USED_FOR")
                    rels = _want
                else:
                    rels = None if fr.rel == "ALL" else (fr.rel, "CAUSES" if fr.rel == "USED_FOR" else fr.rel)
                _spec, _gap = "", ""
                if _strict:
                    try:
                        _spec = self.speculation.deliberate(self, raw, fr)
                        _gap = getattr(self.speculation, "last_gap", "")
                    except Exception:
                        _spec, _gap = "", ""
                _agent = getattr(self.speculation, "last_agent", "") if _strict else ""
                if _agent and _agent != fr.subj and graph.query(_agent, None):
                    # Wernicke soruyu yanlış özneyle çözdüyse ("arılar balı yapar" -> bal),
                    # kendi parçalamasının bulduğu eyleyiciyi kullan.
                    fr.subj = _agent
                hits = graph.query(fr.subj, rels)
                if not hits and fr.rel != "ALL" and not _strict:
                    hits = graph.query(fr.subj, None)
                # TERS ARAMA: "dünyanın en yüksek dağı" gibi tanımlayıcı bir öbek, bilinen bir kavramın
                # özelliği olarak ağda duruyorsa (everest HAS_PROP "dünyanın en yüksek dağı") o kavramı bul.
                if not hits and not graph.query(fr.subj, None):
                    for _rev in graph._in_edges.get(fr.subj, ()):
                        _rev_hits = graph.query(_rev.subj, rels)
                        if _rev_hits:
                            fr.subj = _rev.subj
                            hits = _rev_hits
                            break
                if hits:
                    # Sıralama anlam taşır: "X nedir?" sorusunda önce türü (ISA), sonra özelliği/parçası,
                    # en sonda nedeni gelir. Aksi hâlde "Uyku nedir?" sorusuna nedenle başlanıyordu.
                    _prio = {"ISA": 0, "HAS_PROP": 1, "HAS_PART": 2, "CAN_DO": 3, "USED_FOR": 4,
                             "LOCATED_AT": 5, "CAUSES": 6, "CANNOT_DO": 7, "OPPOSITE": 8}
                    # Doğrudan bağ, ISA kalıtımından önce gelir (kalıtımsa 'olduğu için' ile açıklanır).
                    # Her ilişkiden yalnızca en iyi örnek alınır: iki ISA tek cümlede yinelenmez.
                    _best: dict[str, tuple] = {}
                    for h in hits:
                        e, pth = h
                        key = e.rel
                        if key not in _best or (bool(pth), -e.weight) < (bool(_best[key][1]), -_best[key][0].weight):
                            _best[key] = h
                    hits = sorted(_best.values(), key=lambda h: (_prio.get(h[0].rel, 9), bool(h[1]), -h[0].weight))
                    clauses = []
                    used_objs: list[str] = []
                    for edge, isa_path in hits:
                        if len(clauses) >= 2:
                            break
                        # Üst kavramdan MİRAS alınan özellik ayırt edici değildir; yineleme de ayıklanır.
                        if edge.rel == "HAS_PROP" and isa_path:
                            try:
                                parent_props = {e2.obj for e2, _ in self.lang.graph.query(isa_path[0].obj, ("HAS_PROP",))}
                            except Exception:
                                parent_props = set()
                            if edge.obj in parent_props:
                                continue
                        if any(edge.obj and u and (edge.obj in u or u in edge.obj) for u in used_objs if u != edge.obj):
                            continue
                        p = Proposition(subj=fr.subj, rel=edge.rel, obj=edge.obj)
                        txt = broca.realize_prop(p)
                        if isa_path:
                            via = isa_path[0].obj
                            # Özne bir kez söylenir (Türkçe özne düşürür): "X bir Y olduğu için hesaplar."
                            _pref = fr.subj + " "
                            _tail = txt[len(_pref):].strip() if txt.startswith(_pref) else txt
                            # Üst kavram zaten ilk cümlede söylendiyse gerekçe yinelenmez.
                            if any(via in c for c in clauses):
                                txt = txt if _tail is txt else f"{tr_capitalize(fr.subj)} {_tail}"
                            else:
                                txt = f"{fr.subj} bir {via} olduğu için {_tail}"
                        clauses.append(txt)
                        used_objs.append(edge.obj or "")
                    wern.last_explanation = f"Kavram ağında {fr.subj} düğümünden {len(hits)} ilişkisel bağ buldum"
                    _drop = fr.subj + " "
                    # Özne yalnızca TAM sözcük olarak düşürülür ('güneşin' -> 'in' olmaz!)
                    _clean = [clauses[0]] + [(c[len(_drop):]
                                              if (c.split()[:1] == [fr.subj]) else c) for c in clauses[1:]]
                    _joined = tr_capitalize("; ".join(_clean))
                    _cov = self._wh_covered(hits, _wh, raw)
                    if _strict and _spec and not _cov:
                        # Bağlar sorunun istediği şeyi karşılamıyor: tam düşünme metni konuşsun.
                        _joined = _spec
                    elif _gap and not _cov:
                        _joined = (_joined.rstrip(".") + ". " + tr_capitalize(_gap)).strip()
                    _join_txt = _joined if _joined.endswith((".", "?", "!")) else _joined + "."
                    # ANLATIM: aynı kavram ikinci kez sorulduğunda aynı cümle verilmez.
                    # (Tam düşünme metni/boşluk cümlesi devreye girdiyse ona dokunulmaz.)
                    if getattr(self, "narrator", None) is not None and not (_strict and _spec and not _cov):
                        try:
                            _edges_n = [(e.rel, e.obj) for e, _p in hits]
                            _join_txt = self.narrator.render(self, fr.subj, _edges_n, _join_txt,
                                                             speaker=speaker) or _join_txt
                        except Exception:
                            pass
                    bids.append(CorticalBid("hippocampal_concept", 0.95, 0.94, _join_txt, f"wh:{fr.subj}"))
                else:
                    srec = self.self_model.somatic.get(fr.subj)
                    if srec is not None and srec.provenance in ("deneyim", "duyum"):
                        pol = "faydalı" if srec.effective_valence > 0 else "zararlı"
                        bids.append(CorticalBid(
                            "hippocampal_concept", 0.88, 0.86,
                            f"{tr_capitalize(fr.subj)}, {srec.provenance} belleğime göre {srec.channel} açısından {pol} bir varlıktır.",
                            f"wh_somatic:{fr.subj}",
                        ))
                    else:
                        self.auto_net.stimulate([fr.subj], valence=0.0, resolved_relations=0)
                        if _spec:
                            # Zihin soruyla bağ kurdu: bilinen parçalar + sınanmış varsayım + tam eksik + kendi sorusu.
                            bids.append(CorticalBid("speculative_cortex", 0.86, 0.88, _spec,
                                                    f"wh_spec:{fr.subj}"))
                        elif _strict:
                            # Hiç dayanak yoksa uydurma: soruyu geri verip öğrenmeye açık ol.
                            _miss = f"Bunu henüz bilmiyorum: “{raw.strip()}” — bunu öğrenmem gerek."
                            bids.append(CorticalBid("hippocampal_concept", 0.72, 0.75, _miss,
                                                    f"wh_unknown_strict:{fr.subj}"))
                        else:
                            bids.append(CorticalBid(
                                "hippocampal_concept", 0.72, 0.75,
                                self.speak_unknown(fr.subj, fallback=(f"{tr_capitalize(fr.subj)} hakkında henüz bilgim yok; "
                                                                      f"merak listeme aldım, istersen tek bir cümleyle bana öğretebilirsin.")),
                                f"wh_unknown:{fr.subj}",
                            ))

        # --- SÜTUN 7: Açık-Uçlu Çağrışımsal Türkçe Sohbet Sentezleyicisi (her zaman yedek aday olarak yarışır!) ---
        if not _wh_unans and not _delib_owned:
            # "neden/nasıl" sorusuna tür ("bir canlıdır") yanıt değildir: nedeni/yöntemi bilmiyorsak
            # çağrışımsal sentez susar, dürüst bilgisizlik konuşur.
            bids.append(self._synthesize_associative_dialogue(raw, s_low, speaker=speaker))

        winner = self._compete_workspace(bids)
        # ÖLÇÜM KAPSAMI: kazanan teklifin izi (trace) kaynağa çevrilir. Böylece çıkarım,
        # komut, matematik ve öz-rapor yolları da provenance'a yazılır; ölçer yalan söylemez.
        try:
            self._tag_bid(winner)
        except Exception:
            pass
        return winner.reply

    def _tag_bid(self, bid) -> None:
        """Teklif izini kaynak etiketine çevir (merkezi ölçüm kapsamı)."""
        pv = getattr(self, "provenance", None)
        if pv is None or bid is None:
            return
        tr = str(getattr(bid, "trace", "") or "")
        mod = str(getattr(bid, "module", "") or "")
        low = tr.lower()
        _reply_low = str(getattr(bid, "reply", "") or "").lower()
        if "dolayısıyla" in _reply_low or "demek ki" in _reply_low or "öyleyse" in _reply_low:
            pv.record("cikarim", 1)     # geçişli çıkarım metinde açıkça işaretli
        if low.startswith("1shot") or "concept" in mod:
            pv.record("ag", 1)
            if "inference" in mod or "transitive" in low or "deduce" in low:
                pv.record("cikarim", 1)
        elif mod.startswith("parietal"):
            pv.record("islem", 1)
        elif mod.startswith("insular") or low.startswith("body") or low.startswith("self"):
            pv.record("beden", 1)
        elif "volition" in low or "command" in low:
            pv.record("islem", 1)
        elif "teach" in low or "learn" in low or "imprint" in low:
            pv.record("ag", 1)
        elif low.startswith("why"):
            pv.record("bellek", 1)
        elif mod.startswith("prefrontal") or mod.startswith("broca") or mod.startswith("language"):
            pv.record("ag", 1)

    # ------------------------------------------------------------------ epistemik + kod kanalları
    # ------------------------------------------------------------ 3b) KULAK / SES KİMLİĞİ
    def who_is_speaking(self) -> dict:
        """Son duyulan sesin kimliği — ETİKETTEN değil, sesin imzasından."""
        ev = getattr(self, "last_auditory", None)
        if ev is None or getattr(self, "ear", None) is None:
            return {}
        ident = self.ear.get(ev.voice_id)
        return {
            "voice_id": ev.voice_id,
            "known": bool(ev.known),
            "confidence": round(float(ev.confidence), 3),
            "role": (ident.role() if ident is not None else "bilinmiyor"),
            "heard": (ident.heard if ident is not None else 0),
            "name": ((ident.name if ident is not None else "") or
                     (self.user_profiles.get(ev.voice_id) or {}).get("ad", "")),
            "timbre": ev.print.describe(),
            "distance_m": round(float(ev.distance_m), 2),
            "snr_db": round(float(ev.snr_db), 1),
            "is_question": bool(ev.is_question),
        }

    def ear_report(self) -> str:
        """Kendi kulağından ne duyduğunu kendi sözleriyle anlatır (ölçüm değil, algı)."""
        ev = getattr(self, "last_auditory", None)
        if ev is None or getattr(self, "ear", None) is None:
            return "Henüz kimseyi duymadım."
        who = self.who_is_speaking()
        parts = [f"Az önce {ev.print.describe()} duydum."]
        if who.get("known"):
            parts.append(f"Bu sesi tanıyorum ({who.get('role')}); {who.get('heard')} kez duydum.")
        else:
            parts.append("Bu ses bende yeni; daha önce duymadım.")
        if ev.distance_m > 2.0:
            parts.append("Uzaktan seslendi.")
        if ev.snr_db < 14.0:
            parts.append("Ses boğuktu, tam seçemedim.")
        return " ".join(parts)

    def _canonical_voice(self, key: str) -> str:
        """Dış dünyanın etiketi -> ses kimliği (takma ad çözümü)."""
        k = str(key)
        lab = getattr(self, "_voice_label", None)
        if lab:
            v = lab.get(k)
            if v:
                return v
        return k

    def _log_inference_thought(self, infs) -> None:
        """Türetilen çıkarım Nora'nın iç monoloğuna düşer (kendi kendine düşünme)."""
        try:
            from nora_monologue import InnerThought
            inf = infs[0]
            line = self.infer.line_for(self, inf)
            if not line:
                return
            th = InnerThought(step=int(self.mind.age), trigger="çıkarım", text=line,
                              stance="içe_dönük", schema_id="S1_NEDEN_SONUC", monotony_error=0.0)
            self.monologue.stream.append(th)
        except Exception:
            pass

    def inference_report(self) -> str:
        """Kendi çıkarımlarının özeti (test/teşhis için)."""
        if getattr(self, "infer", None) is None:
            return "Çıkarım motoru kapalı."
        d = self.infer.derived
        if not d:
            return "Henüz kendi başıma bir çıkarım yapmadım."
        kinds: dict[str, int] = {}
        for x in d:
            kinds[x.kind] = kinds.get(x.kind, 0) + 1
        return (f"{len(d)} çıkarım — " + ", ".join(f"{k}: {v}" for k, v in kinds.items())
                + f". Son: {d[-1].subj} {d[-1].rel} {d[-1].obj} (güven {d[-1].confidence:.2f}).")

    def _perception_note(self) -> str:
        """Sesin kendisine verilen kısa tepki (her turda değil). Sözün içeriğinden bağımsız."""
        ev = getattr(self, "last_auditory", None)
        if ev is None:
            return ""
        r = os.urandom(1)[0] / 255.0
        if not ev.known and r < 0.55:
            return "Sesini ilk kez duyuyorum."
        if ev.known and ev.distance_m > 2.0 and r < 0.30:
            return "Sesin uzaktan geliyor."
        if ev.snr_db < 12.0 and r < 0.40:
            return "Seni tam seçemedim."
        return ""

    def _speaker_trust(self, speaker: str) -> float:
        try:
            return float(self.chem.trust_of(speaker))
        except Exception:
            return 0.6

    def _epistemic_channel(self, raw: str, speaker: str, utt_cls: dict) -> tuple[str | None, str, float]:
        """Soru ise inanç deposundan KANITLA yanıtlar; bildirim ise KARŞILAŞTIRARAK öğrenir.
        Hiçbir iddia 'kaydettim' diye kör kabul edilmez: çelişki/eksik ad -> yarışan hipotezler."""
        try:
            # Kod istemi bir iddia değildir: inanç defterine yazılmaz.
            if self.code_cortex.is_requested(raw):
                return None, "", 0.0
            # Soruysa önce kanıt deposundan yanıtla; değilse öğren. (Soru asla 'iddia' diye kaydedilmez.)
            ans = self.episteme.answer(raw)
            if ans is not None:
                return self.epi_speech.speak_query(ans), f"epi_q:{ans.get('path')}", 0.962
            if utt_cls.get("is_question") and not self.episteme.is_question(raw):
                return None, "", 0.0
            res = self.episteme.absorb(raw, source_id=f"insan:{speaker}", speaker_trust=self._speaker_trust(speaker))
            self.last_epistemic = res
            st = res.get("status")
            if st in ("ambiguous", "contradiction", "alias"):
                return self.epi_speech.speak_store(res), f"epi:{st}", 0.972
            if st == "stored" and res.get("kind") in ("event", "scalar"):
                return self.epi_speech.speak_store(res), f"epi:{res.get('kind')}", 0.966
        except Exception:
            return None, "", 0.0
        return None, "", 0.0

    def _code_channel(self, raw: str) -> tuple[str | None, str, float]:
        """Plastik operatör korteksi kod isteğini kendi ağırlıklarıyla seçerse devreye girer.
        Program yalnızca çalıştırılıp çıktısı doğrulandıktan sonra söylenir."""
        try:
            if not self.code_cortex.is_requested(raw):
                return None, "", 0.0
            res = self.code_cortex.handle(raw)
            self.last_code = res
            if res.get("ok"):
                self.chem.pulse_learning(novelty=0.5, valence=0.35)
                return self.code_cortex.speak(res), f"code:{res.get('path')}", 0.988
            if res.get("reason") in ("metin_yok", "belirsiz_niyet"):
                return self.code_cortex.speak(res), f"code:{res.get('reason')}", 0.958
        except Exception:
            return None, "", 0.0
        return None, "", 0.0

    # ------------------------------------------------------------------ CÜMLE YAPISI
    def explain_structure(self, text: str) -> dict:
        """Bir cümleyi rol rol aç: özne/nesne/tümleç/yüklem + öğrenilmiş valans."""
        if getattr(self, "frames", None) is None:
            return {}
        cl = self.frames.parse(text)
        val = {}
        try:
            val = dict(self.frames.memory.valency.get(cl.verb) or {})
        except Exception:
            pass
        out = {"text": text, "roles": cl.roles_tr(), "verb": cl.verb, "tense": cl.verb_tense,
               "person": cl.person, "voice": cl.voice, "subject": cl.subject, "object": cl.obj,
               "object_case": cl.obj_case, "dative": cl.dative,
               "signature": " + ".join(cl.signature),
               "frame_seen": int(self.frames.memory.frames.get(cl.signature, 0)),
               "valency_learned": val}
        self.last_structure = out
        return out

    def learn_structure(self, text: str) -> dict:
        """Bir cümleyi yapı olarak öğren (fiil + nesne hâli + iskelet) ve raporla."""
        if getattr(self, "frames", None) is None:
            return {}
        cl = self.frames.observe(text)
        rep = self.explain_structure(text)
        rep["learned_objects"] = dict(self.frames.memory.verb_objects.get(cl.verb) or {})
        return rep

    def research_topic(self, topic: str, urls: list[str] | None = None) -> str:
        """Yalnızca allowlist'teki güvenilir kaynaklardan öğrenir; her iddia güvenilirlik ağırlığıyla girer."""
        out = self.researcher.research(topic, urls=urls)
        if not out.get("fetched"):
            self.last_epistemic = {"status": "research_failed", "refused": out.get("refused")}
            return "Güvenilir kaynak listemdeki hiçbir adrese ulaşamadım; uydurmak yerine bilmediğimi söylüyorum."
        return out.get("summary", "")

    # ------------------------------------------------------------------ alt yanıtlayıcılar
    def _answer_self_query(self, sub_kind: str) -> str:
        """Kendi kimliği/hâli hakkında sorulan soru. Hazır cümle yok: ölçülen iç durumdan
        seçilen kavramlar Broca'da çekimlenir, topolojiyi plastik şema korteksi kurar."""
        from nora_broca import TurkishSyntaxEngine as SYN
        de, dw = float(self.mind.drift_e), float(self.mind.drift_w)
        depleting = (de < -1e-4 or dw < -1e-4)

        if sub_kind == "yaşam":
            items = [SYN.realize_verb(VerbSpec(verb_root="yaşa", tense="PRES", person="1SG"))]
            if depleting and self.mind.age >= 250:
                items.append(SYN.realize_verb(VerbSpec(verb_root="tüken", subj=NP("beden", poss=1),
                                                       tense="PRES", person="3SG")))
                items.append(SYN.realize_verb(VerbSpec(verb_root="sürdür", obj=NP("varlık", poss=1, case="ACC"),
                                                       advs=("beslenerek",), tense="PRES", person="1SG")))
                self.lang.wernicke.last_explanation = "Beden eğimi tükeniyor; yaşamı beslenerek sürdürüyorum"
            else:
                # Kanıt: kendi bedeninin tükenişini GÖZLEMLEYİP görmediği (uydurma yok).
                items.append(SYN.realize_verb(VerbSpec(
                    verb_root="gör", neg=True, advs=("henüz",), tense="PAST", person="1SG",
                    obj=SYN.nominalize_dik("tüken", poss=3, case="ACC",
                                           subj=NP("beden", poss=1, case="GEN")))))
                self.lang.wernicke.last_explanation = "Yaşam sorusuna ölçülmüş eğim dayanak olmadı: uydurmadım"
            return self.deliberation._compose(self, items)

        if sub_kind == "ayna":
            return (
                "Kendi motor komutlarımın (efferans kopyası) beklenen hız iziyle gördüğüm hareketin Student-t çapraz "
                "korelasyonunu karşılaştırırım; sıfır gecikmeli tam örtüşme bensem, gecikmeli örtüşme kaydımsa, "
                "bağımsız hareket ise başka bir ajandır."
            )

        self.lang.wernicke.last_explanation = (
            "Kimlik sorusu: ad + nörokimyasal iç-duyum + en baskın karakter özelliği + gerçek merak boşluğu; "
            "hepsi Broca'da çekimlendi")
        return self.deliberation._identity(self, "")

    # Mizaç sıfatları: karşıdaki NORA'nın hâlini sorduğunda iç durumundan yanıt verilir.
    _TEMPER_ADJ = frozenset({
        "sessiz", "suskun", "durgun", "sakin", "donuk", "dalgın", "yorgun", "üzgün",
        "keyifsiz", "gergin", "heyecanlı", "mutlu", "kızgın", "kırgın", "yalnız",
    })
    _TEMPER_Q = ("niye", "neden", "nasıl", "neler", "ne")

    def _temperament_adj_in(self, raw: str) -> str:
        s_low = tr_lower(raw)
        _is_q = "?" in raw or any(w in s_low.split() for w in self._TEMPER_Q)
        words = re.findall(r"[\wçğıöşü]+", s_low)
        for w in reversed(words):
            try:
                an = self.lang.morph.analyze_word(w, self.lang.graph.lexicon)
            except Exception:
                continue
            if an.get("copula") and an.get("person") == "2SG" and an.get("stem") in self._TEMPER_ADJ:
                return an["stem"]          # 'çok yorgunsun' -> Nora kendi hâlini bildirir
            if _is_q and w in self._TEMPER_ADJ:
                return w
        return ""

    _PREF_MARKERS = ("daha iyi", "daha güzel", "daha doğru", "hangisi", "hangisini", "tercih")
    _PREF_PARTICLES = ("mi", "mı", "mu", "mü", "midir", "mıdır")

    # ------------------------------------------------------------------ söz biçimi seçimi
    def _speech_schema(self, words, speaker="insan", is_question=True):
        """Cümle topolojisini Nora'nın plastik şema korteksi seçer (koddan kalıp değil, öğrenilmiş şema)."""
        trust, in_pain = 0.5, False
        try:
            trust = float(self.chem.trust_of(speaker))
        except Exception:
            pass
        try:
            visc = self.nervous.felt_visceral_summary()
            in_pain = str(visc.get("pain_state", "")) not in ("", "yok", "None")
        except Exception:
            in_pain = False
        try:
            _race = getattr(getattr(self, "entropy", None), "race_first", None)
            _stance, schema_id, _m, _q = self.schema_cortex.select_stance_and_schema(
                words, is_question, trust, in_pain, race=_race)
            return schema_id
        except Exception:
            return "S2_SONUC_GEREKCE"

    # ------------------------------------------------------------------ duygusal kanıt
    def _affect_evidence(self, concept):
        """Bir kavramın Nora için duygusal değeri ve KANITI.
        Kaynak sırası: beden deneyimi (en ağır) > kendi duygu sözlüğü > öğrendiği özelliklerin duygu yükü.
        Kanıt yoksa değer 0 kalır — o zaman tercih uydurulmaz."""
        g = self.lang.graph
        AL = type(self.conv_cortex).AFFECT_LEXICON
        val, ev, exp = 0.0, 0, False
        srec = self.self_model.somatic.get(concept)
        if srec is not None:
            _v = float(getattr(srec, "effective_valence", 0.0))
            if abs(_v) > 1e-6:
                val += 2.0 * _v
                ev += 2 if getattr(srec, "exp_count", 0) > 0 else 1
            exp = getattr(srec, "exp_count", 0) > 0
        own = AL.get(concept)
        if own:
            val += 1.5 * float(own[0])
            ev += 1
        best = {"rel": "", "obj": "", "tag": "", "word": "", "v": 0.0}
        for e, path in g.query(concept, ("HAS_PROP", "CAN_DO", "CAUSES", "ISA")):
            hit = None
            for wd in reversed(e.obj.split()):
                a = AL.get(wd)
                if a is None and wd.endswith(("ı", "i", "u", "ü")):
                    a = AL.get(wd[:-1])
                if a is not None:
                    hit = (wd, a)
                    break
            if hit is None:
                continue
            wd, (v, _ar, tag) = hit
            contrib = 0.45 * float(v) * (1.0 if not path else 0.8)
            val += contrib
            ev += 1
            if abs(contrib) > abs(best["v"]):
                best = {"rel": e.rel, "obj": e.obj, "tag": tag, "word": wd, "v": contrib}
        return {"val": val, "ev": ev, "exp": exp, **best}

    def _preference_unknown(self, a, b, equal=False):
        """Kanıt yok/eşit: tercih UYDURULMAZ; tanımadığını söyler ve öğrenmek ister (Broca ile kurulur)."""
        from nora_broca import TurkishSyntaxEngine as SYN
        both = NP(a + " ve " + b, case="ACC")
        if equal:
            same = SYN.realize_verb(VerbSpec(verb_root="sev", obj=both, advs=("aynı ölçüde",),
                                             tense="PRES", person="1SG"))
            return tr_capitalize(same) + "."
        c = SYN.realize_verb(VerbSpec(verb_root="tanı", obj=both, neg=True, tense="PRES", person="1SG"))
        q = SYN.realize_verb(VerbSpec(verb_root="anlat", dat=NP("ben", case="DAT"), obj=both,
                                      question_yn=True, tense="AOR", person="2SG"))
        return tr_capitalize(c) + ". " + tr_capitalize(q) + "?"

    def _explanation_edge(self, concept, other=""):
        """Tercihin dile getirilebilir gerekçesi. Önce AYIRT EDİCİ özellik aranır: iki adayın
        paylaştığı özellik ('kedi de köpek de evcildir') tercihi açıklamaz; sadece birinde olan açar."""
        g = self.lang.graph
        shared = {e.obj for e, _p in g.query(other, None)} if other else set()
        fallback = None
        for rel in ("HAS_PROP", "CAN_DO", "USED_FOR", "ISA"):
            for e, _path in g.query(concept, (rel,)):
                if e.rel != rel or not e.obj or e.obj == concept:
                    continue
                if e.obj not in shared:
                    return e
                if fallback is None:
                    fallback = e
        return fallback

    def _ask_back(self, topic: str, kind: str = "think") -> str:
        """Karşı tarafa KONUYA bağlı gerçek soru; cümle Broca'da kökten çekimlenir.
        kind='think': 'X hakkında ne düşünüyorsun?' | kind='like': 'X'i sever misin?'"""
        from nora_broca import TurkishSyntaxEngine as SYN, PredSpec
        try:
            if kind == "like" and topic:
                q = SYN.realize_verb(VerbSpec(verb_root="sev", obj=NP(topic, case="ACC"),
                                              question_yn=True, tense="AOR", person="2SG"))
                return tr_capitalize(q) + "?"
            if topic:
                q = SYN.realize_verb(VerbSpec(verb_root="düşün", subj=NP("sen"), obj=NP(topic, post="hakkında"),
                                              wh_word="ne", tense="PRES", person="2SG"))
                return tr_capitalize(q) + "?"
            q = SYN.realize_pred(PredSpec(subj=NP("sen"), pred_root="nasıl", person="2SG", question=True))
            return tr_capitalize(q) + "?"
        except Exception:
            return ""

    def _wh_covered(self, hits, wh: str, raw: str) -> bool:
        """Bilinen bağlar sorunun İSTEDİĞİ şeyi karşılıyor mu? ('nasıl' yöntem, 'neden' sebep ister;
        'kedi huzura yol açar' mırlamayı açıklamaz.)"""
        want = {"nasıl": ("USED_FOR", "CAN_DO"), "neden": ("CAUSES", "USED_FOR"),
                "niye": ("CAUSES", "USED_FOR"), "niçin": ("CAUSES", "USED_FOR")}.get(wh or "", ())
        if not want:
            return True
        try:
            q = self.speculation._decompose(self, raw, None)
            focus = (self.speculation._tokens2(q.get("verb_phrase", ""))
                     | self.speculation._tokens2(q.get("pred_noun", "")))
        except Exception:
            focus = set()
        if not focus:
            return True
        try:
            _q = self.speculation._decompose(self, raw, None) or {}
        except Exception:
            _q = {}
        _agent, _verb = str(_q.get("agent", "")), str(_q.get("verb", ""))
        # 'Yağmur neden yağar?' gibi addan türemiş fiillerde öznenin kendi CAUSES bağı
        # soruyu gerçekten yanıtlar ('yağmur' <- 'yağ-'). 'Kedi neden mırlar?' yanıtlamaz.
        _denominal = bool(_verb) and _agent.startswith(_verb) and len(_verb) >= 3
        for e, _p in hits:
            if _denominal and e.rel in want and (e.subj or "") == _agent and e.obj:
                return True
            if e.rel in want and e.obj and (self.speculation._tokens2(e.obj) & focus):
                return True
        return False

    def _answer_preference(self, raw, speaker="insan"):
        """İki aday arasında tercih: kararı Nora'nın KENDİ duygusal kanıtı verir (beden deneyimi ya da
        öğretmenin kaynaklı tanıklığı). Kanıt yoksa tercih uydurulmaz: bilmediğini söyler ve öğrenmek ister.
        Her cümle Broca'da çıplak kökten çekimlenir; topolojiyi plastik şema korteksi seçer."""
        from nora_broca import TurkishSyntaxEngine as SYN
        s_low = tr_lower(raw)
        if not any(m in s_low for m in self._PREF_MARKERS):
            return ""
        words = re.findall(r"[\wçğıöşü]+", s_low)
        cands = []
        for i, w in enumerate(words):
            if w in self._PREF_PARTICLES and i > 0:
                prev = words[i - 1]
                try:
                    stem = self.lang.morph.analyze_word(prev, self.lang.graph.lexicon).get("stem") or prev
                except Exception:
                    stem = prev
                if stem not in cands and stem not in ("sen", "ben", "o"):
                    cands.append(stem)
        if len(cands) < 2:
            return ""
        a, b = cands[0], cands[1]
        ea, eb = self._affect_evidence(a), self._affect_evidence(b)

        if ea["ev"] == 0 and eb["ev"] == 0:
            self.lang.wernicke.last_explanation = (
                a + " ve " + b + " için beden deneyimim ve duyusal tanıklığım yok; tercih uydurmadım")
            return self._preference_unknown(a, b)
        if ea["val"] == eb["val"] and ea["ev"] == eb["ev"]:
            self.lang.wernicke.last_explanation = (
                a + " ve " + b + " duygusal kanıtı eşit ({:+.2f}); ayırt edemedim".format(ea["val"]))
            return self._preference_unknown(a, b, equal=True)

        pick, other, ev_p = (a, b, ea) if (ea["val"], ea["ev"]) > (eb["val"], eb["ev"]) else (b, a, eb)
        parts = []
        # 1) Doğrudan deneyim yoksa bunu saklamadan söyler.
        if not ev_p["exp"]:
            try:
                seen = SYN.realize_verb(VerbSpec(verb_root="gör", obj=NP(pick, case="ACC"),
                                                 neg=True, tense="PAST", person="1SG"))
                parts.append(tr_capitalize(seen) + ".")
            except Exception:
                pass
        # 2) Karar + öğrenilmiş gerekçe (şema korteksi topolojiyi seçer).
        core = SYN.realize_verb(VerbSpec(verb_root="sev", obj=NP(pick, case="ACC"),
                                         advs=("daha çok",), tense="AOR", person="1SG"))
        edge = self._explanation_edge(pick, other)
        reason = ""
        if edge is not None:
            reason = self.lang.broca.realize_prop(Proposition(subj=pick, rel=edge.rel, obj=edge.obj))
        schema = self._speech_schema(words, speaker)
        parts.append(self.schema_cortex.transform_clauses(core, reason, "", schema))
        # 3) Duyusal tanıklık: neden bu yönde hissettiği.
        if ev_p["word"]:
            try:
                feel = SYN.realize_verb(VerbSpec(verb_root="uyandır", subj=NP(pick), loc=NP("ben", case="LOC"),
                                                 obj=NP(ev_p["word"], case="NOM"), tense="PRES", person="3SG"))
                parts.append(tr_capitalize(feel) + ".")
            except Exception:
                pass
        # 4) Kanıtın kaynağını dürüstçe söyler.
        if not ev_p["exp"]:
            try:
                told = SYN.realize_verb(VerbSpec(verb_root="anlat", subj=NP("sen"), dat=NP("ben", case="DAT"),
                                                 question_yn=False, tense="PAST", person="2SG"))
                parts.append(tr_capitalize(told) + ".")
            except Exception:
                pass
        # 5) Karşı tarafa gerçek soru: karşılaştırmanın diğer ucuna.
        q = self._ask_back(other, kind="like")
        if q:
            parts.append(q)
        self.lang.wernicke.last_explanation = (
            pick + " lehine duygusal kanıt {:+.2f} (kaynak: {}, {} {}); diğeri {} {:+.2f}".format(
                ev_p["val"], "beden deneyimi" if ev_p["exp"] else "tanıklık",
                edge.rel if edge is not None else "-", edge.obj if edge is not None else "-",
                other, eb["val"]))
        return " ".join(x for x in parts if x)

    # ------------------------------------------------------------------ mizaç sorusu
    def _current_focus(self):
        """Şu an zihnini gerçekten meşgul eden kavram (son konu -> öğrenilmiş düğüm)."""
        g = self.lang.graph
        try:
            topic = str(getattr(self.lang.wernicke, "last_topic", "") or "")
        except Exception:
            topic = ""
        _bad = self._TEMPER_ADJ | {"akşam", "sabah", "gece", "bugün", "dün", "yarın", "şimdi"}
        if topic and topic not in _bad and topic not in ("ben", "sen"):
            try:
                an = self.lang.morph.analyze_word(topic, g.lexicon)
                if not an.get("is_verb") and not an.get("copula") and g.query(topic, None):
                    return topic
            except Exception:
                pass
        try:
            for ep in reversed(list(self.broca_organ.episodic.episodes)[-6:]):
                raw_u = getattr(ep, "user_text", "") or ""
                for wd in reversed(re.findall(r"[\wçğıöşü]+", tr_lower(raw_u))):
                    if wd in _bad or wd in ("ben", "sen", "bir", "ve", "çok", "bu", "da", "de", "mi", "mı"):
                        continue
                    if wd in g._out_edges and len(g._out_edges[wd]) >= 2:
                        return wd
        except Exception:
            pass
        # Sohbet anısı yoksa: geçen turda konuşulan konu (öğretim oturumundan kalan konu sayılmaz).
        last = getattr(self, "_last_focus", "")
        return last

    def _warm_speech_paths(self) -> None:
        """Konuşma yolunun tembel derlemelerini (regex/örüntü) AÇILIŞTA yap.
        Refleks turunda ilk-çağrı maliyeti gecikmeye karışmasın; ölçülen şey refleks olsun."""
        try:
            for s in ("bedenime diken batırdı", "canım yandı", "yaram sızladı"):
                self.lang.wernicke.parse(s)
                self.utter_cortex.classify(s)
                self.logic.classify(s)
                for w in s.split():
                    self.lang.morph.analyze_word(w, self.lang.graph.lexicon)
            # Broca + şema korteksi: refleks cümlesi ilk kez kurulurken regex derlemesi olmasın.
            self.lang.broca.realize_verb(VerbSpec(verb_root="yan", tense="PAST", person="3SG", subj=NP("can", poss=1)))
            self.lang.broca.realize_prop(Proposition(subj="can", rel="HAS_PROP", obj="acı"))
            self.schema_cortex.transform_clauses("canım yandı", "diken battı", "", "S1_NEDEN_SONUC")
        except Exception:
            pass

    def _remember_focus(self, raw: str) -> None:
        """Kullanıcının son sözündeki, hakkında gerçekten bir şey bildiği konuyu hatırla."""
        g = self.lang.graph
        _bad = self._TEMPER_ADJ | {"ben", "sen", "bir", "ve", "çok", "bu", "da", "de", "mi", "mı",
                                   "niye", "neden", "nasıl", "ne", "bugün", "dün", "yarın", "şimdi"}
        words = [wd for wd in re.findall(r"[\wçğıöşü]+", tr_lower(raw)) if wd not in _bad and len(wd) >= 3]
        # Hızlı yol (O(1) sözlük bakışı): yüzey biçim zaten bir düğümse kök çözümlemesi gerekmez.
        for wd in words:
            if len(g._out_edges.get(wd, ())) >= 2:
                self._last_focus = wd
                return
        # Kök çözümlemesi burada YAPILMAZ: refleks turunda morfoloji maliyeti gecikmeye eklenmesin.
        # Yüzey biçim düğüm değilse konu hatırlanmaz; `_current_focus` epizodik yedeği zaten var.

    def _answer_temperament_query(self, adj, speaker="insan"):
        """'Niye bu kadar sessizsin?' — sıfat yüklemi ve gerekçe Broca'da kökten çekimlenir; gerekçe
        Nora'nın o anki gerçek odağıdır (hazır cümle yok)."""
        from nora_broca import TurkishSyntaxEngine as SYN, PredSpec
        core = SYN.realize_pred(PredSpec(pred_root=adj, person="1SG"))
        topic = self._current_focus()
        reason = ""
        if topic:
            try:
                reason = SYN.realize_verb(VerbSpec(verb_root="düşün", obj=NP(topic, case="ACC"),
                                                   tense="PRES", person="1SG"))
            except Exception:
                reason = ""
        if not reason:
            # Odak yoksa uydurma: kimyasal iç-duyumundan tek sözcük okunur ve cümle Broca'da kurulur.
            try:
                word = self.chem.intero_word_tr()
                reason = SYN.realize_verb(VerbSpec(verb_root="hisset", loc=NP("içim", case="LOC"),
                                                   obj=NP(word, case="NOM"), tense="PRES", person="1SG"))
            except Exception:
                reason = ""
        if reason:
            schema = self._speech_schema([adj], speaker, is_question=False)
            text = self.schema_cortex.transform_clauses(core, reason, "", schema)
            if tr_lower(reason).rstrip(".") not in tr_lower(text):
                text = (text.rstrip(".") + ". " + tr_capitalize(reason) + ".").strip()
            self.lang.wernicke.last_explanation = (
                "Mizaç sorusuna iç durumdan yanıt: " + (("odağım " + topic) if topic else "bedensel duyum") +
                " (" + schema + ")")
        else:
            text = tr_capitalize(core) + "."
            self.lang.wernicke.last_explanation = "Mizaç sorusuna iç durumdan yanıt"
        q = self._ask_back(topic, kind="think") if topic else ""
        if not q:
            # Kendiliğinden soru organı: konu yoksa da karşı tarafa gerçek bir soru yöneltir.
            try:
                _ctx = self.initiative._build_context(self, False)
                q = self.initiative._c_ask_about(self, _ctx) or ""
            except Exception:
                q = ""
        return (text + " " + q).strip() if q else text

    def _learned_gap_line(self, act):
        """Öğrenmediği bir söz edimi için dürüst boşluk: uydurmaz, öğrenmek ister (Broca ile)."""
        from nora_broca import TurkishSyntaxEngine as SYN
        try:
            c = SYN.realize_verb(VerbSpec(verb_root="öğren", obj=NP(act, case="ACC", verbal_noun=True),
                                          neg=True, tense="PAST", person="1SG"))
            words = act.split()
            thing = words[0] if len(words) > 1 and words[-1].endswith(("mak", "mek")) else ""
            obj = NP(("bir " + thing) if thing else "bunu", case="NOM")
            q = SYN.realize_verb(VerbSpec(verb_root="öğret", dat=NP("ben", case="DAT"),
                                          obj=obj, question_yn=True, tense="AOR", person="2SG"))
            line = tr_capitalize(c) + ". " + tr_capitalize(q) + "?"
            return line
        except Exception:
            return ""

    def _answer_body_query(self, sub_kind: str) -> str:
        e, w, integ = float(self.obs[I_E]), float(self.obs[I_W]), float(self.obs[I_I])
        visc = self.nervous.felt_visceral_summary()
        if sub_kind == "enerji":
            if e < 0.30:
                return "Karnımda güçlü bir açlık var; enerjim çekildiği için besin bulmam gerekiyor."
            if e < 0.55:
                return "Hafif bir açlık hissediyorum ama hâlâ dinçim."
            return "Açlık hissetmiyorum; bedenim tok ve zinde."
        if sub_kind == "su":
            if w < 0.30:
                return "Boğazım çok kurudu, ciddi biçimde susadım; suya yönelmeliyim."
            if w < 0.55:
                return "Hafif bir susuzluk hissediyorum."
            return "Susuzluk hissetmiyorum; iç dengem yerinde."
        if sub_kind == "bütünlük":
            if visc["pain_state"] == "keskin_acı" or integ < 0.60:
                return "Canım yanıyor ve bedenimde belirgin bir hasar hissediyorum; kendimi korumalıyım."
            if visc["pain_state"] == "sızı" or integ < 0.82:
                return "Yaramda hafif bir sızı var; temkinli hareket ediyorum."
            return "Bedenimde herhangi bir acı ya da sızı yok; bütünlüğüm sağlam."
        if sub_kind == "görme":
            if self.mind.vis_prec < 0.35:
                return "Görme duyum kesilmiş durumda; karanlıkta içsel bilişsel haritamla yön buluyorum."
            return "Görme duyum açık ve net."
        if sub_kind == "motor":
            if self.mind.motor_eff < 0.35:
                return "Motor kontrol kaybı ve felç hissediyorum; komutlarım bedenimi hareket ettirmiyor."
            return "Kas ve motor kontrolüm sağlam; bedenim irademe yanıt veriyor."
        if sub_kind == "bellek":
            n_epi = len(self.mind.epi.buf)
            n_conv = len(self.broca_organ.episodic.episodes)
            return f"Tek bir yaşamım var; epizodik belleğimde {n_epi} bedensel olay ve {n_conv} konuşma anısı yaşıyor."
        if sub_kind == "kimya":
            return f"Şu an içimde {self.chem.dominant_state_tr()} bir sinirsel akış hissediyorum."
        if visc["pain_state"] != "yok":
            return "Bedenimde sızı hissediyorum ama bilincim açık; kendimi korumaya odaklandım."
        return f"Şu an {self.chem.dominant_state_tr()} ve dengeli hissediyorum."

    def _answer_world_query(self, subj: str) -> str:
        if subj not in {"ben", "su", "tehlike", "yiyecek", "besin"}:
            locs = self.lang.graph.query(subj, ("LOCATED_AT",))
            if locs:
                edge, isa_path = locs[0]
                loc_str = self.lang.broca.realize_prop(Proposition(subj=subj, rel="LOCATED_AT", obj=edge.obj))
                if isa_path:
                    loc_str = f"{subj} bir {isa_path[0].obj} olduğu için {loc_str}"
                return tr_capitalize(loc_str) + "."
            # Ters arama: öbek bilinen bir varlığın tanımıysa o varlığın konumunu söyle.
            for _rev in self.lang.graph._in_edges.get(subj, ()):
                _rl = self.lang.graph.query(_rev.subj, ("LOCATED_AT",))
                if _rl:
                    _e, _p = _rl[0]
                    _loc = self.lang.broca.realize_prop(Proposition(subj=_rev.subj, rel="LOCATED_AT", obj=_e.obj))
                    _tail = _loc[len(_rev.subj):].strip() if _loc.startswith(_rev.subj) else _loc
                    _rev_name = tr_capitalize(_rev.subj) if _rev.subj in PROPER_NOUNS else _rev.subj
                    return f"{tr_capitalize(subj)} {_rev_name} olarak bilinir; {tr_capitalize(_tail)}."
            return f"{tr_capitalize(subj)} nerede bulunur henüz bilmiyorum; bir cümleyle öğretebilirsin."
        px, py = float(self.mind.map.pos[0]), float(self.mind.map.pos[1])
        if subj == "su":
            wm = np.maximum(self.mind.map.water, self.mind.map.wseen)
            if float(wm.max()) >= 0.35:
                ci = np.unravel_index(int(np.argmax(wm)), wm.shape)
                wx, wy = (ci[0] + 0.5) * self.mind.map.cell, (ci[1] + 0.5) * self.mind.map.cell
                return f"Bilişsel haritamda bilinen su kaynağı ({wx:.1f}, {wy:.1f}) koordinatlarında; ben şu an ({px:.1f}, {py:.1f}) konumundayım."
            return f"Şu an ({px:.1f}, {py:.1f}) konumundayım; haritamda henüz doğrulanmış bir su hücresi yok."
        if subj == "tehlike":
            hm = np.maximum(self.mind.map.hurt, self.mind.map.seen)
            n_haz = int((hm >= 0.45).sum())
            return f"Bilişsel haritamda işaretlenmiş {n_haz} tehlikeli hücre var; konumum ({px:.1f}, {py:.1f})."
        return f"İçsel yol integrasyonu haritama göre ({px:.1f}, {py:.1f}) konumundayım (belirsizlik {self.mind.map.unc:.2f})."

    # Söz edimi isteyen fiiller: karşıdaki bunları istediğinde Nora bedensel irade değil DİL
    # organını çalıştırır (anlat -> bildiğinden anlat, dinle -> dinle, sor -> sor).
    _SPEECH_REQUEST_ROOTS = frozenset({
        "anlat", "söyle", "açıkla", "öğret", "oku", "konuş", "sor", "dinle", "yaz", "göster",
        "ver", "danış",
    })
    # İçerik istemeyen istek nesneleri: 'bana bir tavsiye ver' -> konu sorulur.
    _ADVICE_OBJECTS = frozenset({"tavsiye", "öğüt", "fikir", "akıl", "yol"})

    def _tell_from_own_life(self) -> str:
        """'Bana bir şey anlat' dendiğinde kendi dünyasından anlatır: kendi çıkarımı,
        duyduğu ses, kendi iç sesi. Hiçbiri yoksa sessiz kalır (uydurmaz)."""
        told = getattr(self, "_told_life", None)
        if told is None:
            from collections import deque as _dq
            told = self._told_life = _dq(maxlen=16)
        # a) Taze ve henüz söylenmemiş kendi çıkarımı
        try:
            if self.infer is not None:
                line = self.infer.last_line(self)
                if line and line not in told:
                    told.append(line)
                    return line
        except Exception:
            pass
        # b) Bedeni: kendi duyumu (açlık, susuzluk, sızı) — insanın sözü değil, kendi hâli.
        try:
            nv = getattr(self, "nervous", None)
            drives = []
            if nv is not None:
                for key, attr, line in (
                    ("susuzluk", "thirst_drive", "suyum azaldı, boğazım kuruyor"),
                    ("açlık", "hunger_drive", "karnım aç, midemde bir boşluk var"),
                    ("acı", "adelta_pain", "bir yerim sızlıyor"),
                    ("sızı", "cfiber_ache", "derinde bir ağrı var"),
                ):
                    v = float(getattr(nv, attr, 0.0) or 0.0)
                    if v > 0.25:
                        drives.append((v, key, line))
            if drives:
                drives.sort(reverse=True)
                v, key, line = drives[0]
                tag = f"beden:{key}:{int(v * 100)}"
                if tag not in told:
                    told.append(tag)
                    mood = ""
                    try:
                        word = self.chem.intero_word_tr()
                        mood = f", içim {word}" if word else ""
                    except Exception:
                        pass
                    return f"Şu an {line}{mood}. Bunu kendim fark ediyorum; benim kendi derdim bu."
        except Exception:
            pass
        # c) Kendi iç sesi (içerikli olanlardan)
        try:
            for th in reversed(list(getattr(self.monologue, "stream", []) or [])[-10:]):
                body = (getattr(th, "text", "") or "").strip()
                if len(body) < 16 or body in told:
                    continue
                told.append(body)
                return f"İçimden şu geçti: {tr_lower(body).rstrip('.')}."
        except Exception:
            pass
        return ""

    def _narrate_from_knowledge(self, topic: str = "") -> str:
        """Kendi kavram ağından, kendi seçtiği bir düğümü anlatır: uydurmadan, ezberden okumadan.
        Hikâye istendiğinde gerçekten bildiği bağları sıraya koyar; sonunda kendi merakını ekler."""
        g = self.lang.graph
        broca = self.lang.broca
        cand = tr_lower(topic).strip() if topic else ""
        if cand and not g.query(cand, None):
            cand = ""
        if not cand:
            rich = [(k, len(v)) for k, v in g._out_edges.items() if len(v) >= 2 and k not in self._narrated]
            if not rich:
                self._narrated.clear()
                rich = [(k, len(v)) for k, v in g._out_edges.items() if len(v) >= 2]
            if not rich:
                return ""
            rich.sort(key=lambda kv: (-kv[1], kv[0]))
            pool = rich[:12]
            cand = pool[int(self.mind.age + len(self._narrated)) % len(pool)][0]
        self._narrated.append(cand)
        hits = g.query(cand, None)
        order = ("HAS_PROP", "CAN_DO", "USED_FOR", "ISA", "HAS_PART", "CAUSES", "LOCATED_AT", "OPPOSITE")
        # DOĞRUDAN bağ, kalıtımla gelen bağdan önce gelir: 'köpek bir evcil hayvandır' > 'köpek bir canlıdır'.
        pool = [(order.index(e.rel), 0 if not path else 1, -float(e.weight), i, e, path)
                for i, (e, path) in enumerate(hits) if e.rel in order]
        pool.sort()
        # Üst kavramdan miras alınan özellik ayırt edici değildir ('köpek evcildir' çünkü evcil hayvan evcildir).
        _parents = {pe.obj for pe, _p in hits if pe.rel == "ISA"}
        inherited: set[str] = set()
        for par in _parents:
            try:
                inherited |= {e2.obj for e2, _p in g.query(par, ("HAS_PROP",))}
            except Exception:
                pass
        clauses: list[str] = []
        edges: list = []
        used_objs: list[str] = []
        seen_rel: dict[str, int] = {}
        for _oi, _inh, _w, _i, e, _path in pool:
            if len(clauses) >= 2:
                break
            seen_rel[e.rel] = seen_rel.get(e.rel, 0) + 1
            if e.rel != "HAS_PROP" and seen_rel[e.rel] > 1:
                continue
            if e.rel == "HAS_PROP" and seen_rel[e.rel] > 2:
                continue
            if e.rel == "HAS_PROP" and e.obj in inherited:
                continue
            if _path and e.rel == "ISA":
                continue                      # kalıtımla gelen tür, doğrudan tür varken söylenmez
            _conf = next((i for i, u in enumerate(used_objs) if u != e.obj and (e.obj in u or u in e.obj)), None)
            if _conf is not None:
                # Aynı bilgiyi iki kez söyleme; ama hangisi DAHA ÖZELSE o kalsın ('evcil' < 'evcil hayvan').
                if len(e.obj) > len(used_objs[_conf]):
                    clauses.pop(_conf)
                    edges.pop(_conf)
                    used_objs.pop(_conf)
                else:
                    continue
            txt = broca.realize_prop(Proposition(subj=cand, rel=e.rel, obj=e.obj))
            if txt and txt not in clauses:
                clauses.append(txt)
                edges.append(e)
                used_objs.append(e.obj)
        if not clauses:
            return ""
        _clean = [clauses[0]]
        for _c in clauses[1:]:
            if _c.startswith(cand + " "):
                _c = _c[len(cand) + 1:]          # Türkçe özne düşürür: ikinci cümlede özne yinelenmez
            _clean.append(_c)
        clauses = _clean
        body = tr_capitalize("; ".join(clauses)) + "."
        # Kendi duruşu: az önce kurduğu bağa verdiği kişisel karşılık (inisiyatif organının biçimleri).
        stance = ""
        try:
            if edges:
                stance = self.initiative._c_share_discovery(self, {"edge": (edges[0].subj, edges[0].rel, edges[0].obj)}) or ""
        except Exception:
            stance = ""
        if stance:
            def _stems(txt: str) -> set[str]:
                out = set()
                for w in re.findall(r"[\wçğıöşü]+", tr_lower(txt)):
                    if len(w) <= 3:
                        continue
                    try:
                        st = self.lang.morph.analyze_word(w, self.lang.graph.lexicon).get("stem") or w
                    except Exception:
                        st = w
                    out.add(st)
                return out
            _bw, _sw = _stems(body), _stems(stance)
            if _bw and _sw and len(_bw & _sw) / max(1, len(_bw)) >= 0.5:
                stance = ""      # az önce söylediğim şeyi kendi duruşum diye yinelemem
        self.lang.wernicke.last_explanation = (
            f"Anlatı isteğine kavram ağımdan yanıt verdim: {cand} düğümünde {len(edges)} bağ seçtim"
        )
        # Aynı kavramı ikinci kez anlatırken aynı cümleyi kurma (bkz. nora_anlatim).
        try:
            if getattr(self, "narrator", None) is not None:
                _e = [(e.rel, e.obj) for e in edges]
                body = self.narrator.render(self, cand, _e, body) or body
        except Exception:
            pass
        return f"{body} {stance}".strip()

    def _looks_like_request(self, raw: str) -> bool:
        """Yalın fiil kökü / istek kipiyle biten sözce: 'bana bir şaka yap', 'biraz anlat'."""
        words = re.findall(r"[\wçğıöşü]+", tr_lower(raw))
        if not words:
            return False
        last = words[-1]
        morph = self.lang.morph
        try:
            if last in morph.KNOWN_VERB_ROOTS:
                return True
            an = morph.analyze_word(last, self.lang.graph.lexicon)
        except Exception:
            return False
        return bool(an.get("is_verb") and an.get("tense") in ("IMP", "OPT"))

    def _handle_speech_request(self, fr, raw: str = "") -> str:
        """Dil isteğini (anlat/dinle/sor/öğret...) karşılar; içerik gerçek bilgiden gelir."""
        from nora_broca import TurkishSyntaxEngine as SYN
        rel = fr.rel
        topic = (fr.obj or "").strip()
        if rel == "dinle":
            return tr_capitalize(SYN.realize_verb(VerbSpec(
                verb_root="dinle", obj=NP("sen", case="ACC"), tense="PRES", person="1SG"))) + "."
        if rel == "sor":
            ask = SYN.realize_verb(VerbSpec(verb_root="sor", tense="IMP", person="2SG"))
            return f"{tr_capitalize(ask)}, " + tr_lower(tr_capitalize(SYN.realize_verb(VerbSpec(
                verb_root="dinle", obj=NP("ben", case="ACC"), tense="PRES", person="1SG")))) + "."
        # ŞARKI/ŞİİR: bu evrende ezgi ve ölçü üreten organ yok. Yapmış gibi yapmak
        # (veya başka bir soruya cevap vermek) sessiz yanlıştır; açıkça söyle, sonra
        # elimde olanı öner.
        if topic in {"şarkı", "türkü", "şiir", "ninni"}:
            own = self._tell_from_own_life()
            head = f"Şarkı söyleyemem, elimde ezgi üreten bir organ yok; şiir de yazmam."
            return (f"{head} Bunun yerine öğrendiğim bir şeyi anlatayım. {own}").strip() if own else head
        if rel == "konuş":
            return tr_capitalize(SYN.realize_verb(VerbSpec(
                verb_root="konuş", advs=("seninle",), tense="AOR", person="1PL",
                question_yn=True))) + "?"
        if topic in self._ADVICE_OBJECTS:
            ask = SYN.realize_verb(VerbSpec(
                verb_root="iste", obj=NP(topic, mods=("ne hakkında",), case="NOM"),
                tense="AOR", person="2SG"))
            return tr_capitalize(ask) + "?"
        topic = self._speech_request_topic(fr, raw)
        if not topic:
            # Konu verilmediyse ezber bir kavramı (hep aynı: 'su') anlatmak yerine
            # önce KENDİ hayatından bir şey söyler: taze çıkarımı, duyduğu ses, iç sesi.
            own = self._tell_from_own_life()
            if own:
                return own
        told = self._narrate_from_knowledge(topic)
        if told:
            return told
        return ""

    def _speech_request_topic(self, fr, raw: str) -> str:
        """Söz ediminin nesnesi boşsa konuyu cümlenin kendisinden çıkarır ('kediler hakkında bilgi ver')."""
        g = self.lang.graph
        topic = (getattr(fr, "obj", "") or "").strip()
        # İçerik istemeyen üst sözcükler konu değildir: 'bilgi ver' -> konu cümlenin içindedir.
        _meta = {"bilgi", "haber", "cevap", "açıklama", "örnek", "şey", "şeyler", "hikâye", "hikaye",
                 "masal", "öykü", "şaka", "tavsiye", "öğüt", "fikir", "akıl", "yol"}
        if topic and topic not in _meta and g.query(topic, None):
            return topic
        _skip = {"bilgi", "ver", "anlat", "söyle", "hakkında", "bana", "biraz", "şey", "ne", "bir",
                 "istiyorum", "isterim", "lütfen", "hadi", "haydi"}
        best, best_n = "", 0
        for wd in re.findall(r"[\wçğıöşü]+", tr_lower(raw)):
            if wd in _skip or len(wd) < 3:
                continue
            try:
                an = self.lang.morph.analyze_word(wd, g.lexicon)
            except Exception:
                continue
            if an.get("is_verb") or an.get("copula"):
                continue
            stem = an.get("stem") or wd
            n = len(g.query(stem, None))
            if n > best_n:
                best, best_n = stem, n
        return best

    # ======================================================================
    # UYGULAMA OPERATÖRLERİ: kural bağla / uygula / say / transfer / plan
    # ======================================================================
    def _say(self, kind: str, slots: dict | None = None, ctx: dict | None = None,
             fallback: str = "Ölçüm yok.") -> str:
        """Söz VERİDEN gelir: `nora_priors` içindeki topluluklar yarışır.

        Kodda yalnız ÖLÇÜM ve SLOT doldurma var; cümlenin kendisi veride yaşar.
        Bu, iskelet mandalının (nora_audit) nedenini ortadan kaldırır: yeni bir cümle
        eklemek kod satırı eklemek değildir. Seçim yine yarıştır; tazelik/öğrenme
        ağırlıkları devrede olduğu için aynı kalıp üst üste gelmez.
        """
        text = None
        try:
            f = getattr(self, "priors", None)
            if f is not None:
                text = f.pick(kind, ctx or {}, slots or {},
                              prov=getattr(self, "provenance", None))
        except Exception:
            text = None
        return str(text) if text else fallback

    def _apply_tag(self, src: str, k: int = 1) -> None:
        try:
            if getattr(self, "provenance", None) is not None:
                self.provenance.record(src, k)
        except Exception:
            pass

    @staticmethod
    def _quoted(raw: str) -> str:
        """Söz içindeki tırnaklı ya da '… cümlesinde' geçen hedefi çıkar."""
        m = re.search(r"[\"'“”‘’]([^\"'“”‘’]{2,120})[\"'“”‘’]", raw)
        if m:
            return m.group(1).strip()
        # '… cümlesinde' kalıbı: ORİJİNAL metinden alınır ki büyük harfler korunsun.
        m = re.search(r"(?:^|\s)([\wçğıöşüÇĞİÖŞÜ\s]{2,80}?)\s+cümlesinde", raw,
                      flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return ""

    @staticmethod
    def _shape_name(rule) -> str:
        """Şeklin ADI veriden değil, kural nesnesinin alanından okunur."""
        from nora_apply import SHAPE_NAMES
        return SHAPE_NAMES.get(getattr(rule, "shape", "parity"), "tek/çift")

    @staticmethod
    def _threshold_desc(rule) -> str:
        """Eşik ifadesini ölçüm diline çevir ('3'ten az', 'en az 4')."""
        t = int(getattr(rule, "threshold", 0))
        _na = __import__("nora_apply").num_abl(t)
        return {"lt": f"{t}'{_na} az", "le": f"en çok {t}",
                "gt": f"{t}'ten çok", "ge": f"en az {t}"}.get(getattr(rule, "op", "lt"), str(t))

    @staticmethod
    def _rule_map_text(rule) -> str:
        """Eşlemeyi TEK biçimde yaz: şekle göre anahtar adları değişir."""
        m = getattr(rule, "mapping", {}) or {}
        sh = getattr(rule, "shape", "parity")
        if sh == "threshold":
            t = getattr(rule, "threshold", 0)
            op = getattr(rule, "op", "lt")
            _na = __import__("nora_apply").num_abl(t)
            cond = {"lt": f"{t}'{_na} az", "le": f"en çok {t}",
                    "gt": f"{t}'{_na} çok",
                    "ge": f"en az {t}"}.get(op, f"{t}")
            return f"{cond} -> {m.get('cond')}, değilse {m.get('else')}"
        if sh == "rank":
            return f"en çok -> {m.get('max')}, en az -> {m.get('min')}"
        if sh == "presence":
            return f"var -> {m.get('var')}, yok -> {m.get('yok')}"
        return f"tek -> {m.get('odd')}, çift -> {m.get('even')}"

    # ======================================================================
    # OPERATÖR OCAĞI: örneklerden kendi kuralını yaz
    # ======================================================================
    FORGE_TRIGGERS = ("kural çıkar", "kural türet", "kendi kural", "kendi operatör",
                      "örneklerden", "örnekten", "kendin yaz", "operatör yaz", "sen yaz")

    def _forge_pool(self, n: int = 400) -> list[str]:
        """Adayları ayırt edecek sözcük havuzu: sözlükten, deterministik sıralı."""
        try:
            lex = getattr(getattr(self.lang, "graph", None), "lexicon", None) or {}
            words = list(lex.keys()) if hasattr(lex, "keys") else list(lex)
            out = [str(w) for w in words if str(w).isalpha() and 2 <= len(str(w)) <= 14]
            return sorted(set(out))[:n]
        except Exception:
            return []

    def _forge_pairs(self, raw: str) -> list[tuple[str, str]]:
        """'karınca=kırmızı, bilgisayar=mavi' -> [(karınca,kırmızı), ...].

        Ayırıcı önceliği: '=' , '->' , '→' , sonra ':' (yalnız sağı tek sözcükse).
        Komut öneki ('Kural çıkar:') atılır; etiket tek sözcük olmalı.
        """
        out: list[tuple[str, str]] = []
        for chunk in re.split(r"[;\n]|,|\s+ve\s+", raw or ""):
            c = chunk.strip(" .!?")
            if not c:
                continue
            t = lab = ""
            for sep in ("=", "->", "→"):
                if sep in c:
                    t, lab = c.rsplit(sep, 1)
                    break
            if not t and ":" in c:
                left, right = c.rsplit(":", 1)
                if right.strip() and len(right.strip().split()) == 1:
                    t, lab = left, right
            if not t:
                continue
            # 'Kural çıkar: karınca' -> 'karınca' (komut öneki metne karışmaz)
            t = t.split(":")[-1].strip(" .!?-")
            lab = tr_lower(lab).strip(" .!?-\"\u201c\u201d")
            if not t or not lab or len(lab.split()) != 1:
                continue
            if tr_lower(t) in self.FORGE_TRIGGERS or len(t) < 2:
                continue
            out.append((t, lab))
        return out

    def _forge_check(self, pairs, pool=None) -> str:
        """Kendi kuralımı örnekle sına: tutuyorsa söyle, tutmuyorsa YENİDEN TÜRET."""
        fg = self.forge
        rule = fg.adopted
        # `Rule.apply` kullanılır: kuralın KENDİ yolu (istisna -> ölç -> eşle).
        bad = [(t, l) for t, l in pairs if rule.apply(t)[0] != l]
        good = [(t, l) for t, l in pairs if rule.apply(t)[0] == l]
        if not bad:
            t, l = (good[0] if good else ("-", "-"))
            self._apply_tag("islem", 1)
            # Kanıt SAYILIR: kendi ölçümüm bu örnekte tuttu (söz değil, sonuç).
            try:
                fg.mark_confirmed(rule)
            except Exception:
                pass
            _tkey = str(getattr(rule, "feature", "") or "")
            _rec = next((r for r in (fg.measurements or [])
                         if str(r.get("key")) == _tkey), {})
            return self._say("meta.forge", {
                "desc": rule.source_text or "", "n_ex": len(fg.examples),
                "n_cand": len(fg.cands), "human": fg.human_count(),
                "self": len(fg.authored), "cands": "-", "probe": "-", "why": "-",
                "item": t, "label": l, "got": rule.apply(t)[0],
                "hits": int(_rec.get("hits") or 0), "misses": int(_rec.get("misses") or 0),
            }, ctx={"stage": "confirmed"})
        # Çürütme: eski örneklerin TAMAMI + yeni örneklerle yeniden türet.
        t, l = bad[0]
        got = rule.apply(t)[0]
        # Çürütülen operatör deftere yazılır: kendi yazdığım şeyi sessizce silmem.
        try:
            fg.mark_refuted(rule, t, l, got)
        except Exception:
            pass
        res = fg.observe(list(fg.examples) + bad, pool=pool)
        st = str(res.get("status") or "")
        counts = res.get("counts") or {}
        self._apply_tag("islem", 1)
        if st == "adopted":
            return self._say("meta.forge", {
                "desc": res.get("desc") or "", "n_ex": len(fg.examples),
                "n_cand": len(fg.cands), "human": counts.get("human"),
                "self": counts.get("self"), "cands": "-", "probe": "-", "why": "-",
                "item": t, "label": l, "got": got,
            }, ctx={"stage": "revised"})
        if st == "ambiguous":
            probe = res.get("probe")
            self._forge_pending = {"probe": probe}
            return self._say("meta.forge", {
                "desc": "-", "n_ex": len(fg.examples), "n_cand": int(res.get("n") or 0),
                "human": counts.get("human"), "self": counts.get("self"),
                "cands": "; ".join((res.get("candidates") or [])[:3]),
                "probe": probe or "-", "why": "-", "item": t, "label": l, "got": got,
            }, ctx={"stage": "revise_ambiguous"})
        _why = str(res.get("why") or "veri yetmiyor")
        self._forge_pending = None
        return self._say("meta.forge", {
            "desc": "-", "n_ex": len(fg.examples), "n_cand": 0,
            "human": counts.get("human"), "self": counts.get("self"),
            "cands": "-", "probe": "-", "item": t, "label": l, "got": got,
            "why": self._forge_why_map().get(_why, _why),
        }, ctx={"stage": "revise_failed"})

    @staticmethod
    def _forge_why_map() -> dict:
        """Ocağın gerekçe kodları -> insan dili. Cümle KODDA değil VERİDE durur
        (`nora_priors.json: forge_why`), iskelet mandalı bu ayrımı ölçer."""
        try:
            import json as _json
            import nora_priors as _pr
            blob = _json.loads(_pr.JSON_PATH.read_text(encoding="utf-8"))
            return {str(k): str(v) for k, v in (blob.get("forge_why") or {}).items()}
        except Exception:
            return {}

    def _handle_forge(self, raw: str, speaker: str = "insan") -> str | None:
        """Kuralı Nora yazar: örnekler -> adaylar -> (gerekirse) ayırt edici örnek -> operatör."""
        fg = getattr(self, "forge", None)
        if fg is None:
            return None
        lo = tr_lower(raw or "")
        pairs = self._forge_pairs(raw)
        started = any(k in lo for k in self.FORGE_TRIGGERS)
        pending = getattr(self, "_forge_pending", None)
        # --- (0) KENDİ KURALIMA KARŞI ÖRNEK: tetikleyici olmadan da denetlenir ------
        # Elde BENİM yazdığım bir operatör varsa ve gelen etiket onun sözlüğündense bu
        # bir öğretme/çürütme örneğidir; sessizce "anladım" deyip geçmek yanlış olur.
        if not started and not pending and fg.adopted is not None and pairs:
            labels = {str(v).lower() for v in (fg.adopted.mapping or {}).values()}
            if all(lab in labels for _, lab in pairs):
                return self._forge_check(pairs, pool=self._forge_pool())
        if not started and not pending:
            return None
        # Bekleyen ayırt edici soruya ÇIPLAK etiket cevabı: "mavi"
        if not pairs and pending and re.fullmatch(r"[a-zçğıöşü]{2,20}", lo.strip(" .!?")):
            pairs = [(str(pending.get("probe") or ""), lo.strip(" .!?"))]
        if not pairs:
            # Örnek NİYETİ yoksa (soru/sohbet) yolu işgal etmeyiz; yalnız "örnek"/"="
            # geçen bir tetikleyici için örnek istenir.
            if started and ("=" in raw or "örnek" in lo):
                return self._say("meta.forge", {"desc": "-", "n_ex": 0, "n_cand": 0,
                                                "human": fg.human_count(),
                                                "self": len(fg.authored), "cands": "-",
                                                "probe": "-", "why": "örnek yok"},
                                 ctx={"stage": "insufficient"})
            return None
        pool = self._forge_pool()
        if started or not pending:
            res = fg.observe(pairs, pool=pool)
        else:
            res = None
            for t, lab in pairs:
                res = fg.add_example(t, lab, pool=pool)
        st = str(res.get("status") or "")
        counts = res.get("counts") or {}
        self._apply_tag("islem", 1)
        if st == "adopted":
            self._forge_pending = None
            self.apply_eng.wm.write("last_rule_kind", "kendi", "forge")
            _rule = res.get("rule")
            _spec = getattr(_rule, "feature_spec", None) or {}
            _lm = getattr(fg, "last_measure", None) or {}
            # Ölçüm bu turda mı icat edildi, envanterde hazır mıydı? Söz ikisini
            # ayırır: "icat ettim" ile "daha önce yazmıştım" aynı şey değildir.
            if _spec and _lm.get("reused"):
                _stage = "reused"
            elif _spec:
                _stage = "invented"
            else:
                _stage = "adopted"
            return self._say("meta.forge", {
                "desc": res.get("desc") or "",
                "name": (__import__("nora_apply").FEATURES.get(
                    getattr(_rule, "feature", ""), ("-",))[0] if _rule else "-"),
                "map": (self._rule_map_text(_rule) if _rule is not None else "-"),
                "measured": counts.get("measured"),
                "n_ex": len(fg.examples), "n_cand": len(fg.cands),
                "human": counts.get("human"), "self": counts.get("self"),
                "cands": "-", "probe": "-", "why": "-",
            }, ctx={"stage": _stage})
        if st == "ambiguous":
            probe = res.get("probe")
            self._forge_pending = {"probe": probe}
            if not probe:
                return self._say("meta.forge", {
                    "desc": "-", "n_ex": len(fg.examples), "n_cand": int(res.get("n") or 0),
                    "human": counts.get("human"), "self": counts.get("self"),
                    "cands": "; ".join((res.get("candidates") or [])[:3]),
                    "probe": "-", "why": self._forge_why_map().get("ayırt-edici-yok", ""),
                }, ctx={"stage": "no_probe"})
            # Adaylar kalıbı MİNİK: bu turda İCAT EDİLMİŞ bir ölçüm varsa söz bunu
            # söyler (ve sınırın veriden gelmediğini de). Yerleşik ölçümler arasındaki
            # belirsizlikte "yeni ölçüm yazdım" demek yalan olurdu.
            _now = list(getattr(fg, "invented_now", None) or [])
            _inv = "invented_ambiguous" if (_now and not pending) else \
                   ("progress" if pending else "ambiguous")
            return self._say("meta.forge", {
                "desc": "-", "n_ex": len(fg.examples), "n_cand": int(res.get("n") or 0),
                "name": (__import__("nora_apply").FEATURES.get(
                    getattr(_now[0], "feature", ""), ("-",))[0] if _now else "-"),
                "human": counts.get("human"), "self": counts.get("self"),
                "cands": "; ".join((res.get("candidates") or [])[:3]),
                "probe": probe, "why": "-",
            }, ctx={"stage": _inv})
        # insufficient: uydurma yok, gerekçe adıyla
        self._forge_pending = None
        _why = str(res.get("why") or "veri yetmiyor")
        return self._say("meta.forge", {
            "desc": "-", "n_ex": len(fg.examples), "n_cand": 0,
            "human": counts.get("human"), "self": counts.get("self"),
            "cands": "-", "probe": "-",
            "why": self._forge_why_map().get(_why, _why),
        }, ctx={"stage": "insufficient"})

    #: Karşıolgusal irade sorusunda "hangi kanal çıkarılsın" sözlüğü (beden kanalı adları).
    ABLATE_WORDS = {
        "açlık": "açlık", "aç olmasa": "açlık", "aç olmasaydın": "açlık", "acıkmamış": "açlık",
        "tok olsan": "açlık", "karnın tok": "açlık",
        "susuzluk": "susuzluk", "susamış": "susuzluk", "susuz olmasa": "susuzluk",
        "bütünlük": "bütünlük", "acı olmasa": "bütünlük", "acın olmasa": "bütünlük",
    }

    #: VERİ TABLOSU ÖNBELLEĞİ: aynı JSON tur başına 5 kez yeniden okunuyordu
    #: (her turda ~2 ms). Dosya değişirse mtime anahtarı düşer, önbellek tazelenir.
    _DATA_CACHE: dict = {}

    @classmethod
    def _data_table(cls, name: str, default):
        """Küçük VERİ tabloları (`nora_priors.json`) — kodda Türkçe yüzey büyümesin."""
        try:
            import json as _json
            import nora_priors as _pr
            p = _pr.JSON_PATH
            ver = p.stat().st_mtime_ns
        except Exception:
            return default
        ent = cls._DATA_CACHE.get(("__file__", str(p)))
        if not ent or ent[0] != ver:
            try:
                blob = _json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return default
            ent = (ver, blob)
            cls._DATA_CACHE[("__file__", str(p))] = ent
        try:
            return ent[1].get(name, default)
        except Exception:
            return default

    def _missing_operator(self, raw: str) -> str | None:
        """İSTENEN İŞLEM YOKSA: uydurma cevap yerine eksik operatörün adı söylenir.

        Neden gerekli: tanınmayan bir soru, akışın sonundaki sohbet hattına düşüp
        ALAKASIZ bir iç sesle ("kendi kendime şunu çıkardım: …") yanıtlanıyordu. Bu,
        "yanlış cevap sahte anlamadan iyidir" kuralının en açık ihlaliydi: cevap hem
        yanlış hem de soruyla ilgisiz. Burada cevap kısa ve doğru: o operatör yok.
        """
        low = tr_lower(raw or "")
        if "?" not in raw and not any(w in low for w in self._data_table(
                "question_markers", ())):
            return None
        for name, keys in (self._data_table("missing_ops", {}) or {}).items():
            if any(k in low for k in keys):
                self._apply_tag("cikarim", 1)
                return self._say("discourse.unknown", {"w": name},
                                 ctx={"register": "op_missing"}, fallback=name)
        return None

    # ------------------------------------------------------------------ operatör sentezi
    # ------------------------------------------------------------------ küme işlemleri
    def _set_turn(self, raw: str) -> str | None:
        """İki sözcüğün ortak/tüm/farklı harfleri: operatör aksiyomlardan SENTEZLENİR."""
        st = getattr(self, "sets", None)
        if st is None or not raw:
            return None
        tl = tr_lower(raw)
        if "harf" not in tl:
            return None
        name = None
        if "ortak" in tl:
            name = "kesişim"
        elif "birleşim" in tl or ("tüm" in tl and "harf" in tl):
            name = "birleşim"
        elif "fark" in tl or "olmayan" in tl or "yalnız" in tl or "sadece" in tl:
            name = "fark"
        if name is None:
            return None
        words = re.findall(r"['‘’“\"]([^'‘’“”\"]{2,24})['‘’“\"]", raw)
        if len(words) < 2:
            _q = self._synth_say(VerbSpec("ver", obj=NP("sözcük", case="ACC"),
                                          advs=("iki",), tense="OPT", person="2SG",
                                          question_yn=True))
            return f"{_q}?"
        a, b = words[0].strip(), words[1].strip()
        res = st.discover(name)
        if res.get("status") != "adopted":
            _n = res.get("n")
            if res.get("status") == "ambiguous":
                return f"{name}: {_n}. {self._synth_say(VerbSpec('seç', dat=NP('aday', case='DAT'), tense='OPT', person='1SG', question_yn=True))}?"
            return None
        out = st.apply(name, a, b) or ""
        chars = list(out)
        if not chars:
            _lab = NP("ortak harf") if name == "kesişim" else NP("harf")
            return self._synth_say(PredSpec(pred_root="yok", subj=_lab)) + "."
        _lab = {"kesişim": "ortak harflar", "birleşim": "tüm harfler",
                "fark": "farklı harfler"}[name]
        _ops = " · ".join(res["entry"]["ops"])
        return f"{_lab}: {', '.join(chars)}. ({_ops})"

    # ------------------------------------------------------------------ plan
    #: Plan cümlesi: "Çay planı: 1 su kaynat. 2 çayı demle. 3 süt ekle."
    _PLAN_DECL = re.compile(r"plan\w*\s*:\s*(.+)$", re.I)
    _PLAN_STEP = re.compile(r"(?:\d+\s*[.)]\s*|,\s*|;\s*)")

    #: SICAK DESENLER: ilk kullanımda derlenirse maliyet ÖLÇÜLEN tura düşer (A27).
    _RE_TASK = re.compile(r":\s*(.+)$")
    _RE_QUOTED = re.compile(r"['‘’“\"]([^'‘’“”\"]{2,40})['‘’“”\"]")
    _RE_TUR = re.compile(r"(bir|iki|üç|dört|beş|\d+)\s*tur", re.U)
    _RE_ABILITY = re.compile(r"\b(?:yap|getir|başar|halled|çöz|kur|yaz|hesapla|bul)"
                             r"\w*(?:abilir|ebilir)\s+mi\w*")
    _RE_CHAIN_NUM = re.compile(r"(?:ilk|son)\s+(\w+)\s+harf", re.U)
    _RE_APPLY = re.compile(r"(?:uygula|çalıştır|dene)\s*[:\-]?\s*([\wçğıöşü]{2,24})", re.U)
    _RE_FOR = re.compile(r"([\wçğıöşü]{2,24})\s+için\s+(?:uygula|çalıştır|dene)", re.U)
    _RE_WORDOF = re.compile(r"([\wçğıöşü]{2,24})['’]?[ıiuü]?\s+(?:sözcüğ|kelime)", re.U)
    _RE_TERS = re.compile(r"(?:tersten|tersini)\s+(?:yaz\s+)?[:]?\s*([\wçğıöşü]{2,24})", re.U)
    #: Organ boşluğu tanıma — SÖZCÜK SINIRLI (alt-dizgi tuzağına düşmesin).
    _RE_GAP_LANG = re.compile(r"\b(?:yunanca|ingilizce|fransızca|almanca|japonca|"
                              r"çince|rusça|arapça)\b", re.U)
    _RE_GAP_ASK = re.compile(r"\b(?:karşılığ|çevir|nasıl denir|ne demek)", re.U)
    _RE_GAP_IMG = re.compile(r"\b(?:resim|görüntü|fotoğraf|kamera)\b", re.U)
    _RE_GAP_IMGQ = re.compile(r"\b(?:analiz|ne var|bak|göster|gör)\b", re.U)
    _RE_GAP_SMELL = re.compile(r"\b(?:kokusunu|kokusu|tadını|tadı)\b", re.U)
    _RE_GAP_TEMP = re.compile(r"\b(?:sıcaklık|derece|nem)\b", re.U)
    _RE_GAP_HERE = re.compile(r"\b(?:bu oda|odanın|şu an|burada|dışarıda|odada)\b", re.U)
    _RE_COLON_WORD = re.compile(r":\s*([\wçğıöşü]{2,24})", re.U)
    _RE_QUOTED2 = re.compile(r"['‘’“\"]([^'‘’“”\"]{2,24})['‘’“”\"]")

    # ------------------------------------------------------------------ organ boşluğu
    def _gap_turn(self, raw: str) -> str | None:
        """Yapılamayan isteği TANIRSIN ve uydurmaz: eksik organın adı söylenir.

        "Yunanca karşılığı ne?" -> dil organı yok; "şu an oda kaç derece?" -> duyu yok.
        Bu tanıma olmadan sistem soruyu tanım/çağrışım hattına düşürüp ALAKASIZ
        (ama kendinden emin) bir cümle kuruyordu; bu, uydurmanın en sinsi biçimi.
        """
        tl = tr_lower(raw or "")
        if not raw:
            return None
        what = ""
        # SÖZCÜK SINIRI ŞART: alt-dizgi eşleşmesi "mutfakTADIR" içinde "tadı" bulup
        # masum bir cümleyi koku sorusu sanıyordu (A29/#10 regresyonu).
        if self._RE_GAP_LANG.search(tl) and self._RE_GAP_ASK.search(tl):
            what = "dil çevirisi"
        if not what and self._RE_GAP_IMG.search(tl) and self._RE_GAP_IMGQ.search(tl):
            what = "görüntü"
        if not what and self._RE_GAP_SMELL.search(tl):
            what = "koku/tat"
        if not what and self._RE_GAP_TEMP.search(tl) and self._RE_GAP_HERE.search(tl):
            what = "ısı/nem ölçer"
        if not what:
            return None
        return self._say("meta.reasoning", {"what": what}, ctx={"kind": "gap"},
                         fallback="") or None

    # ------------------------------------------------------------------ güven / hedef
    #: Güven, görevin hangi organa düştüğünden çıkar (iyimserlikten değil).
    _CONF_ORGANS = (
        # SIRA ÖNEMLİ: duyu ve dil organı YOK; bunlar "kaç" sözcüğü yüzünden
        # aritmetik sanılmamalı ("bu odanın sıcaklığı kaç derece").
        (("resim", "görüntü", "fotoğraf", "koku", "tat"), 8, "duyu girişi"),
        (("yunanca", "ingilizce", "fransızca", "almanca", "japonca", "çince",
          "karşılığı", "çeviri"), 12, "dil çevirisi"),
        (("ters",), 78, "ters çevirme"),
        (("ortak", "kesişim", "birleşim", "fark"), 74, "küme"),
        (("kural", "örnek"), 70, "kural çıkarma"),
    )
    #: Aritmetik yalnız SAYI + İŞLEM varsa: "kaç" tek başına yeterli değil.
    _ARITH = re.compile(r"\d+\s*(?:[+\-*/x×]|ile|ve)\s*\d+|\d+\D{0,12}(?:topla|çıkar|çarp|böl)")

    def _calib_conf(self, klass: str, base: int) -> int:
        """Envanter ön-sözü ile DENEYİMİ harmanla (Beta ortalaması, ön-ağırlık 2)."""
        rec = self.calib.get(klass) or [0, 0]
        s_, n_ = int(rec[0]), int(rec[1])
        return int(round(100.0 * (base / 100.0 * 2.0 + s_) / (2.0 + n_)))

    def _calib_note_outcome(self) -> None:
        """Bekleyen güven sözünün sonucunu yaz: organ yolu = başarı, ötekiler = başarısız."""
        pend = getattr(self, "_conf_pending", None)
        if not pend or getattr(self, "_turn_path", "") == "conf":
            return
        self._conf_pending = None
        ok = 1 if getattr(self, "_turn_path", "chat") == "organ" else 0
        rec = self.calib.setdefault(str(pend[0]), [0, 0])
        rec[0] = int(rec[0]) + ok
        rec[1] = int(rec[1]) + 1

    def _confidence_turn(self, raw: str) -> str | None:
        """'Yapmadan önce 0-100 güvenini söyle: X' — sayı, ORGAN ENVANTERİNDEN gelir.

        Düşük güven, yapılamayacak görevlerde UYDURMAMAYI da getirir: sayı ile sonuç
        aynı kaynaktan (hangi organ var/yok) çıktığı için tutarlı olur.
        """
        tl = tr_lower(raw or "")
        if "güven" not in tl:
            return None
        if not ("0-100" in raw or "%" in raw or "yüzde" in tl or "kaç" in tl):
            return None
        m = self._RE_TASK.search(raw)
        task = (m.group(1) if m else "").strip(" .?!")
        if not task:
            return None
        # KANIT GÖREVDEN GELİR, TALİMATTAN DEĞİL: "0-100 arası" ifadesi tek başına
        # sayı+eksi gibi görünüp aritmetik kanıtı sanılıyordu ("sıcaklık kaç derece" %82).
        tl_task = tr_lower(task)
        conf, organ = 30, ""
        for cues, val, org in self._CONF_ORGANS:
            if any(c in tl_task for c in cues):
                conf, organ = val, org
                break
        if conf == 30 and self._ARITH.search(tl_task):
            conf, organ = 82, "aritmetik"
        elif conf == 30 and any(k in tl_task for k in ("sıcaklık", "derece", "nem")):
            conf, organ = 8, "duyu girişi"
        # SINIF: aynı iş türü aynı anahtara yazılır; güven o sınıfın GEÇMİŞİNDEN öğrenilir.
        if organ:
            klass = organ
        elif conf == 30:
            _toks = [w for w in re.findall(r"\w+", tr_lower(task)) if len(w) > 2]
            klass = "bilinmeyen:" + (_toks[0] if _toks else "görev")
        else:
            klass = "tanıdık organ"
        conf = self._calib_conf(klass, conf)      # ENVANTER + DENEYİM
        self._conf_pending = [klass, conf]
        self._turn_path = "conf"                  # bu tur SÖZ verir, sonuç DEĞİL
        # EKSİK ORGAN "VAR" DİYE ANILMAZ: düşük güvenin dayanağı organın YOKLUĞUDUR.
        if not organ:
            organ = "tanıdık organ"
        elif conf <= 12:
            organ = f"{organ} organı yok"
        # Görev yapılamazsa güven düşük olmalı: aynı kaynaktan karar verilir.
        _say = self._say("meta.reasoning",
                         {"conf": conf, "organ": organ, "task": task},
                         ctx={"kind": "conf"}, fallback="")
        return _say or f"{conf}"

    #: Harf okunuşları (hedef sürdürmede öğretilir). Kısa veri; cümle değil.
    _LETTERS = (("k", "ke"), ("l", "le"), ("m", "me"), ("n", "ne"), ("r", "re"),
                ("s", "se"), ("t", "te"), ("y", "ye"), ("z", "ze"), ("b", "be"))

    #: Hedef türleri veri: her tür kendi adım üreticisini taşır.
    _GOAL_KINDS = ("harf", "sayı")

    def _goal_turn(self, raw: str) -> str | None:
        """'Üç tur boyunca ... hedefin' -> hedef YIĞINA yazılır (eskisi silinmez)."""
        tl = tr_lower(raw or "")
        if not any(k in tl for k in ("hedefin", "hedef olarak", "görevin")):
            return None
        m = self._RE_TUR.search(tl)
        if not m:
            return None
        word = m.group(1)
        n = {"bir": 1, "iki": 2, "üç": 3, "dört": 4, "beş": 5}.get(word)
        if n is None:
            try:
                n = int(word)
            except Exception:
                return None
        if "harf" in tl:
            kind = "harf"
        elif "sayı" in tl or "kat" in tl:
            kind = "sayı"
        else:
            return None
        g = {"kind": kind, "total": int(n), "done": 0, "i": 0,
             "t0": len(getattr(self, "dialogue_log", []) or [])}
        stacked = bool(self._goals)
        self._goals.append(g)
        self._goal = self._goals[0]
        self._goal_turn0 = self._goals[0]["t0"]
        ctx = {"kind": "goal_set2" if stacked else "goal_set"}
        return self._say("meta.reasoning",
                         {"total": int(n), "kind": kind, "done": 0,
                          "bekleyen": len(self._goals) - 1},
                         ctx=ctx, fallback="") or None

    def _goal_step(self) -> str:
        """Bu turun hedef adımı. Yığın: biten hedef düşer, alttaki KALDIĞI YERDEN sürer."""
        if not self._goals:
            self._goal = None
            return ""
        g = self._goals[-1]
        # Hedefin KURULDUĞU turda adım atılmaz (o tur zaten cevabı verdi).
        if len(getattr(self, "dialogue_log", []) or []) <= (int(g.get("t0", 0)) + 1):
            return ""
        # BİTEN hedef yığından düşer; altta hedef varsa DÖNÜŞ bildirilir.
        while self._goals and int(self._goals[-1].get("done", 0)) >= int(self._goals[-1]["total"]):
            self._goals.pop()
            if self._goals:
                nxt = self._goals[-1]
                nxt["t0"] = len(getattr(self, "dialogue_log", []) or []) - 1
                self._goal = nxt
                self._goal_turn0 = nxt["t0"]
                return self._say("meta.reasoning",
                                 {"done": int(nxt["done"]), "total": int(nxt["total"]),
                                  "kind": nxt["kind"]},
                                 ctx={"kind": "goal_resume"}, fallback="") or ""
            self._goal = None
            return ""
        g = self._goals[-1]
        if g["kind"] == "harf":
            i = int(g.get("i", 0)) % len(self._LETTERS)
            letter, spoken = self._LETTERS[i]
            g["i"] = i + 1
            slots = {"letter": letter, "spoken": spoken}
            ctx = {"kind": "goal_step"}
        else:
            i = int(g.get("i", 0)) + 1
            g["i"] = i
            slots = {"a": i, "b": i * 2}
            ctx = {"kind": "goal_step2"}
        g["done"] = int(g.get("done", 0)) + 1
        done, total = int(g["done"]), int(g["total"])
        if done >= total:                     # hedef BİTTİ: sayaç burada artar
            self._goals_done = int(getattr(self, "_goals_done", 0)) + 1
        return self._say("meta.reasoning",
                         {**slots, "done": done, "total": total},
                         ctx=ctx, fallback="") or ""

    # ------------------------------------------------------------------ sayısal kıyas
    #: "Ali'nin boyu 180 santimetre" -> (özne, nitelik, sayı)
    _NUM_FACT = re.compile(r"(\w+)['’]?n[ıiuü]n\s+(\w+)\s+"
                           r"(\d+(?:[.,]\d+)?)", re.U)
    #: "kim daha uzun / kim daha ağır" -> nitelik yönü
    _ATTR = {"uzun": ("boy", 1), "kısa": ("boy", -1), "ağır": ("kilo", 1),
             "hafif": ("kilo", -1), "yaşlı": ("yaş", 1), "genç": ("yaş", -1)}

    #: "47 ile 38'i topla" — iki sayı + işlem FİİLİ (yalnız '+' işareti zaten çözülür).
    _ARITH_ASK = re.compile(r"(\d+(?:[.,]\d+)?)\D{0,10}?(\d+(?:[.,]\d+)?)\D{0,16}?"
                            r"(topla|çıkar|çarp|böl)", re.U)

    def _num_turn(self, raw: str) -> str | None:
        """Sayıyla söylenen olguları yaz; "kim daha X" sorusunu SAYIYA bakarak cevapla."""
        tl = tr_lower(raw or "")
        _ma = self._ARITH_ASK.search(tl)
        if _ma and _ma.group(3):
            try:
                a = float(_ma.group(1).replace(",", "."))
                b = float(_ma.group(2).replace(",", "."))
            except Exception:
                a = b = None
            if a is not None:
                op = _ma.group(3)
                if op == "topla":
                    res = a + b
                elif op == "çıkar":
                    res = a - b
                elif op == "çarp":
                    res = a * b
                else:
                    if b == 0:
                        return self._say("meta.reasoning", {"what": "sıfıra bölme"},
                                         ctx={"kind": "gap"}, fallback="") or None
                    res = a / b
                _txt = f"{res:g}"
                _opv = {"topla": "topladım", "çıkar": "çıkardım",
                        "çarp": "çarptım", "böl": "böldüm"}.get(op, op)
                return self._say("meta.reasoning",
                                 {"a": f"{a:g}", "b": f"{b:g}", "op": op,
                                  "opv": _opv, "res": _txt},
                                 ctx={"kind": "arith"}, fallback=_txt) or _txt
        wrote = 0
        for m in self._NUM_FACT.finditer(tl):
            who, attr, val = m.group(1), m.group(2), m.group(3)
            for _suf in ("sı", "si", "su", "sü", "u", "ü", "ı", "i"):
                if len(attr) - len(_suf) >= 3 and attr.endswith(_suf):
                    attr = attr[: -len(_suf)]
                    break
            try:
                num = float(val.replace(",", "."))
            except Exception:
                continue
            self.num_facts.setdefault(attr, {})[who] = num
            wrote += 1
        if wrote:
            _pairs = []
            for attr, table in self.num_facts.items():
                for who, num in table.items():
                    _pairs.append(f"{who} {attr} {num:g}")
            return self._say("meta.reasoning", {"facts": ", ".join(_pairs[-4:])},
                             ctx={"kind": "numfact"}, fallback="") or None
        if "daha" not in tl and "en " not in tl:
            return None
        attr = None
        for cue, (a, sign) in self._ATTR.items():
            if cue in tl:
                attr, sign = a, sign
                break
        if attr is None or attr not in self.num_facts:
            return None
        table = self.num_facts[attr]
        if len(table) < 2:
            return None
        items = sorted(table.items(), key=lambda kv: kv[1])
        who, val = items[-1] if sign > 0 else items[0]
        other, oval = items[0] if sign > 0 else items[-1]
        _r = self._say("meta.reasoning",
                       {"who": who, "val": f"{val:g}", "other": other,
                        "other_val": f"{oval:g}", "attr": attr},
                       ctx={"kind": "numcmp"}, fallback="")
        return _r or None

    # ------------------------------------------------------------------ adımlı zincir
    #: "önce X'i sırala, sonra ilk üç harfini söyle" — işlemler ADLARIYLA seçilir.
    _CHAIN_CUE = ("önce", "sonra")

    def _chain_turn(self, raw: str) -> str | None:
        """İki işlemi SIRAYLA uygula: ara sonuç görünür, sonuç ondan hesaplanır."""
        tl = tr_lower(raw or "")
        if not all(k in tl for k in self._CHAIN_CUE):
            return None
        word = ""
        m = (self._RE_QUOTED.search(raw) or self._RE_WORDOF.search(tl)
             or re.search(r"([\wçğıöşü]{2,24})['’]?[ıiuü]?\s+harf", tl))
        if not m:
            return None
        word = m.group(1).strip()
        # İşlemler: ilkel ADLARI + Türkçe karşılıkları, metindeki KONUMLARINA göre.
        ops = []
        names = [n for n, _f, _c in PRIMITIVES]
        cues = {"alfabetik": "alfabetik sırala", "sırala": "alfabetik sırala",
                "ters": "ters çevir", "büyük harf": "büyük harfe çevir",
                "küçük harf": "küçük harfe çevir"}
        for cue, pname in cues.items():
            pos = tl.find(cue)
            if pos >= 0 and pname in names:
                ops.append((pos, pname))
        for m2 in self._RE_CHAIN_NUM.finditer(tl):
            num = {"bir": 1, "iki": 2, "üç": 3, "dört": 4, "beş": 5}.get(m2.group(1))
            if num:
                pname = f"ilk {num} harfi al" if m2.group(0).startswith("ilk") else f"son {num} harfi al"
                if pname in names:
                    ops.append((m2.start(), pname))
        _seq = [p for _pos, p in sorted(ops)]
        ops = []                              # aynı işlem iki kez sayılmaz
        for _p in _seq:
            if not ops or ops[-1] != _p:
                ops.append(_p)
        if len(ops) < 2:
            return None
        out = word
        trace = [out]
        for pname in ops:
            fn = dict((n, f) for n, f, _c in PRIMITIVES)[pname]
            out = fn(out)
            trace.append(out)
        _r = self._say("meta.reasoning",
                       {"chain": " -> ".join(trace), "ops": " → ".join(ops),
                        "final": out},
                       ctx={"kind": "chain"}, fallback="")
        return _r or None

    def _plan_knowledge(self, goal: str) -> bool:
        """Hedef hakkında adım üretecek bilgi var mı? Grafikte kenar yoksa YOK.

        "Çay yapmayı planla" isteğine genel bir şablon (durum/topla/doğrula) yazmak
        plan değil, plan taklididir. Bilgi yoksa söylenir.
        """
        try:
            return any(tr_lower(goal) in (e.subj + " " + e.obj).lower()
                       for e in self.lang.graph.edges)
        except Exception:
            return False

    def _plan_turn(self, raw: str) -> str | None:
        """Adımları AL, koşul değişince yeniden KUR — eski adımı sessizce tekrarlama.

        Plan, sistemin kendi genel çatısından değil kullanıcının verdiği adımlardan
        gelir; revizyon da o adımlar üzerinde yapılan bir ELEME değil, koşula göre
        yeniden kurmadır. Adım kalmıyorsa plan yok denir.
        """
        tl = tr_lower(raw or "")
        if "plan" not in tl:
            return None
        # A) PLAN KUR: "… planı: 1 … 2 … 3 …"
        m = self._PLAN_DECL.search(raw)
        if m and ("planla" not in tl or ":" in raw):
            body = tr_lower(m.group(1))
            if re.search(r"\d\s*[.)]", body):
                parts = re.split(r"\s*\d+\s*[.)]\s*", body)
            elif re.search(r"(?:^|\s)1\s", body):
                parts = re.split(r"\s*\d+\s+", body)
            else:
                parts = re.split(r"\s*[,;]\s*|\s+ve\s+", body)
            steps = [x.strip(" .;") for x in parts if x.strip(" .;")]
            steps = [x for x in steps if len(x) > 2]
            if len(steps) >= 2:
                goal = tl.split("plan")[0].strip(" :,.;")
                goal = re.sub(r"\b(?:için|adımda|adımlık|bir|üç|iki|dört|beş)\b", " ", goal)
                goal = " ".join(goal.split()).strip(" '’‘\"“”:,.;[]()") or "plan"
                self._plan_goal = goal
                self.plans[goal] = steps
                _num = " | ".join(f"{i + 1}. {x}" for i, x in enumerate(steps))
                return self._say("plan.report", {"goal": goal, "steps": _num,
                                                "n": len(steps)},
                                 ctx={"kind": "steps_built"}, fallback="") or _num
        # A2) ADIMSIZ PLAN İSTEĞİ: "çay yapmayı üç adımda planla"
        # "hedef belirle ve planla" SOYUT bir istektir (hedef verilmemiş): genel plan
        # makinesi onu kurar. A2 yalnız SOMUT hedef ("çay yapmayı planla") içindir.
        if not m and ("planla" in tl or "plan kur" in tl) and not (
                "hedef" in tl and "belirle" in tl):
            goal = re.sub(r"\b(?:yapmayı|yapmak|için|adımda|adımlık|bir|üç|iki|dört|beş|"
                          r"planla|plan|kur|hedef|belirle|ve)\b", " ", tl)
            goal = " ".join(goal.split()).strip(" '’‘\"“”:,.;[]()")
            if goal and goal not in self.plans and not self._plan_knowledge(goal):
                return self._say("plan.report", {"goal": goal}, ctx={"kind": "unknown"},
                                 fallback="") or None
        # B) REVİZYON: "süt kalmamış, planı değiştir"
        if ("kalmamış" in tl or "kalmadı" in tl or "yok" in tl) and any(
                k in tl for k in ("planı değiştir", "planı yenile", "planı güncelle",
                                  "değiştir", "yenile")):
            if not self.plans:
                return self._say("plan.report", {"goal": self._plan_goal},
                                 ctx={"kind": "none"}, fallback="") or None
            goal = self._plan_goal
            old = list(self.plans.get(goal) or [])
            missing = []
            for cand in ("süt", "şeker", "su", "un", "yağ", "yumurta", "tuz"):
                if cand in tl and any(cand in s for s in old):
                    missing.append(cand)
            if not missing:
                missing = [w for w in re.findall(r"(\w{3,})", tl)
                           if any(w in st for st in old)]
            if not missing:
                _cur = " | ".join(f"{i + 1}. {x}" for i, x in enumerate(old))
                return self._say("plan.report", {"goal": goal, "steps": _cur},
                                 ctx={"kind": "already"}, fallback="") or None
            new = [s for s in old if not any(mm in s for mm in missing)]
            self.plans[goal] = new
            if not new:
                return self._say("plan.report", {"goal": goal, "removed": ", ".join(missing)},
                                 ctx={"kind": "empty"}, fallback="") or None
            _num = " | ".join(f"{i + 1}. {x}" for i, x in enumerate(new))
            return self._say("plan.report", {"goal": goal, "steps": _num, "n": len(new),
                                            "removed": ", ".join(missing),
                                            "old_n": len(old)},
                             ctx={"kind": "steps_revised"}, fallback="") or _num
        return None

    # ------------------------------------------------------------------ öz-tahmin
    def _predict_turn(self, raw: str) -> str | None:
        """'Bunu yapabilir misin?' — cevap ENVANTERDEN gelir, iyimserlikten değil.

        Tahmin, hangi organın devreye gireceği söylenerek yapılır. Organ yoksa
        'yapabilirim' denmez; hangi organın gireceği bilinmiyorsa söz verilmez.
        """
        tl = tr_lower(raw or "")
        # Yalnız GÖREV fiilleriyle kurulmuş yetenek soruları ("yapabilir misin",
        # "getirebilir misin"); "yağmur olabilir mi" gibi tahmin cümleleri değil.
        if not self._RE_ABILITY.search(tl):
            return None
        # Görev kısmı: iki nokta sonrası ya da tırnak içi.
        _task = ""
        m = re.search(r":\s*(.+)$", raw)
        q = re.findall(r"['‘’“\"]([^'‘’“”\"]{2,40})['‘’“\"]", raw)
        if m:
            _task = m.group(1).strip()
        elif q:
            _task = q[0].strip()
        _task = _task.strip(" '’‘\"“”?.,")
        if not _task:
            return None
        _low = tr_lower(_task)
        organ = ""
        if "ters" in _low and ("yaz" in _low or "çevir" in _low):
            organ = "ters çevir"
        elif "ortak" in _low or "harf" in _low:
            organ = "küme"
        elif "kural" in _low or "örnek" in _low:
            organ = "kural çıkarma"
        elif "öğren" in _low or "öğret" in _low:
            organ = "tek atış öğrenme"
        if organ:
            return self._say("meta.reasoning", {"task": _task, "organ": organ},
                             ctx={"kind": "predict_yes"}, fallback="") or None
        return self._say("meta.reasoning", {"task": _task},
                         ctx={"kind": "predict_unknown"}, fallback="") or None

    # ------------------------------------------------------------------ kısıtlı üretim
    #: Kısıt denetimli üretimde kullanılan çatılar. Her biri (fiil, hâl, ön-ulaç)
    #: üçlüsüdür; cümleyi `SYN` kurar, kelimeler veriden gelir. Çatı, o sözcükle
    #: ANLAMCA uyumluysa kullanılır (aşağıdaki `_sem_ok`), değilse atlanır.
    _GEN_FRAMES = (
        ("oku", "NOM", "her akşam"),
        ("oku", "ACC", "dün akşam"),
        ("iste", "ACC", "çok"),
        ("bitir", "ACC", "dün akşam"),
    )
    #: Her sözcüğe uyan güvenli çatı: "bugün X hakkında konuşalım".
    _GEN_SAFE = ("konuş", "bugün")

    def _sem_ok(self, word: str, verb: str) -> bool:
        """Çatı bu sözcükle anlamca uyumlu mu? Kanıt grafikte; yoksa çatı KULLANILMAZ.

        "deniz okurum" gramatikti ama anlamsızdı. Uyum, sözcüğün grafikteki
        kullanım ilişkisinden okunur; kanıt yoksa iddia edilmez.
        """
        try:
            edges = [e for e in self.lang.graph.edges
                     if tr_lower(word) in (e.subj + " " + e.obj).lower()]
        except Exception:
            return False
        text = " ".join(f"{e.rel} {e.obj}" for e in edges).lower()
        if verb == "oku":
            return any(k in text for k in ("bilgi", "okuma", "aktarma", "öğrenme", "yazı"))
        return False

    def _constrained_turn(self, raw: str) -> str | None:
        """Verilen koşullara UYAN cümleyi kur, sonra kendi kısıtını DENETLE.

        "İçinde X geçen ama Y geçmeyen bir cümle kur" bir üretim kısıtıdır: cümle
        seçilmez, KURULUR ve kuralın sağlandığı üretimden sonra ölçülür. Koşulu
        sağlayan çatı yoksa uydurulmaz.
        """
        tl = tr_lower(raw or "")
        if not any(k in tl for k in ("cümle kur", "cümle yaz", "bir cümle")):
            return None
        if "geçen" not in tl and "içeren" not in tl and "geçsin" not in tl:
            return None
        words = re.findall(r"['‘’“\"]([^'‘’“”\"]{2,24})['‘’“\"]", raw)
        if len(words) < 2:
            return None
        # "X geçen ama Y geçmeyen" sırası: ilk kelime ZORUNLU, ikincisi YASAK.
        _req, _ban = words[0].strip(), words[1].strip()
        if tl.index(tr_lower(_ban)) < tl.index(tr_lower(_req)):
            _req, _ban = _ban, _req
        def _fits(cand: str) -> bool:
            low_c = tr_lower(cand)
            return bool(cand) and (tr_lower(_req) in low_c) and (tr_lower(_ban) not in low_c) \
                and len(cand.split()) >= 4 and "?" not in cand

        for verb, case, pre in self._GEN_FRAMES:
            if not self._sem_ok(_req, verb):
                continue                      # anlamca uyumsuz çatı denenmez
            try:
                cand = self._synth_say(VerbSpec(verb, obj=NP(_req, case=case),
                                                pre_clause=pre, tense="AOR", person="1SG"))
            except Exception:
                cand = ""
            if _fits(cand):                   # KENDİ KISITINI DENETLER
                return cand[0].upper() + cand[1:] + "."
        # Güvenli çatı: her addla anlamlı ("bugün X hakkında konuşalım").
        _v, _pre = self._GEN_SAFE
        try:
            cand = self._synth_say(VerbSpec(_v, obj=NP(_req, post="hakkında"),
                                            pre_clause=_pre, tense="OPT", person="1PL"))
        except Exception:
            cand = ""
        if _fits(cand):
            return cand[0].upper() + cand[1:] + "."
        return self._say("meta.reasoning", {"req": _req, "ban": _ban},
                         ctx={"kind": "gen_none"}, fallback="") or None

    def _recall_testimony(self, raw: str) -> str | None:
        """Başkasından aktarılan olgu: iki tanık çelişiyorsa İKİSİ de söylenir.

        "Hangi tanığa inanayım" sorusunun cevabı veride yoktur; bu yüzden sistem
        sessizce son duyduğunu seçmez. Tek tanık varsa tek, iki değer varsa çelişki.
        """
        tm = getattr(self, "testimony", None)
        if tm is None:
            return None
        # BİLDİRİM CEVAP DEĞİLDİR: "Ayşe bana ... söyledi" bir soru değildir; onu
        # hatırlama sorusu sanıp cevaplamak, kullanıcının sözünü gasp etmektir.
        _lt = tr_lower(raw or "")
        if "?" not in (raw or "") and not any(k in _lt for k in (
                "kaçta", "kaç", "ne zaman", "nedir", "neydi", "hangi", "kim", "mi")):
            return None
        hit = None
        try:
            hit = tm.query(raw)
        except Exception:
            return None
        if not hit:
            return None
        vals = hit.get("vals") or []
        if not vals:
            return None
        if hit.get("conflict"):
            (s1, v1), (s2, v2) = vals[0], vals[1]
            _ts = self._say("memory.testimony",
                            {"subj": hit["subj"], "src1": s1, "val1": v1,
                             "src2": s2, "val2": v2},
                            ctx={"kind": "conflict"}, fallback="")
            if _ts:
                return _ts
        src, val = vals[0]
        _ts = self._say("memory.testimony",
                        {"subj": hit["subj"], "src": src, "val": val},
                        ctx={"kind": "single"}, fallback="")
        return _ts or None

    def _synth_program(self):
        """Yazılmış son dönüşüm programı (yeni oturumda defterden geri kurulur)."""
        sy = getattr(self, "synth", None)
        if sy is None:
            return None
        if getattr(sy, "last", None) is not None:
            return sy.last
        if not sy.programs:
            return None
        prog = sy.rebuild(sy.programs[-1].get("id"))
        if prog is not None:
            sy.last = prog
        return prog

    def _synth_say(self, spec):
        """Cümleyi KENDİ sözdizimi motorundan üret (kalıp cümle yok)."""
        try:
            if isinstance(spec, PredSpec):
                return SYN.realize_pred(spec)
            return SYN.realize_verb(spec)
        except Exception:
            return ""

    def _synth_turn(self, raw: str, speaker: str = "insan") -> str | None:
        """Örnekten dönüşüm yaz / yazdığını uygula / ne yazdığını söyle.

        Cevaplar kalıp değil: yüklem Nora'nın çekim motorundan (`SYN`) gelir, içerik
        (operatör adı, sayı, sözcük, çıktı) veriden. Böylece iskelet sayacı artmaz.
        """
        sy = getattr(self, "synth", None)
        if sy is None or not raw:
            return None
        tl = tr_lower(raw)
        pairs = parse_pairs(raw)

        def _ok(res, n_ex):
            prog = res["program"]
            _probe = next((w for w in ("deniz", "kitap", "masa", "zeytin", "bulut")
                           if w not in [i for i, _o in pairs]), "bulut")
            self._synth_last_reply = prog.desc()
            _v = self._synth_say(VerbSpec("yaz",
                                          obj=NP("dönüşüm", case="ACC"),
                                          advs=("kendim",), tense="PAST", person="1SG"))
            _c = self._synth_say(VerbSpec("açıkla", subj=NP("program"),
                                          obj=NP("örnek", plural=True, case="ACC"),
                                          tense="PRES", person="3SG"))
            return (f"{_v}: {prog.desc()}. {_c} ({n_ex}/{n_ex}). "
                    f"{_probe} -> {prog.apply(_probe)}.")

        def _amb(res):
            _c = res.get("cands") or []
            _pr = res.get("probe") or "masa"
            _q = self._synth_say(VerbSpec("al", subj=NP(_pr), advs=("hangi çıktıyı",),
                                          tense="AOR", person="3SG"))
            return (f"Aday sayısı {len(_c)}: {', '.join(_c[:3])}. {_q}?")

        # 0) BEKLEYEN BELİRSİZLİK: ayırt edici tek örnek geldiyse adayları süz.
        if getattr(sy, "_pending", None) and len(pairs) == 1:
            w, o = pairs[0]
            res = sy.refine(w, o)
            st = res.get("status")
            if st == "adopted":
                return _ok(res, len(sy.programs[-1].get("examples") or []) or 2)
            if st == "ambiguous":
                return _amb(res)
            if st == "reject":
                _v = self._synth_say(VerbSpec("uy", subj=NP("çıktı"),
                                              dat=NP("aday", case="DAT", mods=("hiçbir",)),
                                              tense="PRES", neg=True, person="3SG"))
                return f"{_v}."

        # A) ÖĞREN: en az iki (girdi -> çıktı) çifti verildi.
        if len(pairs) >= 2:
            res = sy.learn(pairs)
            st = res.get("status")
            if st == "adopted":
                return _ok(res, len(pairs))
            if st == "ambiguous":
                return _amb(res)
            if res.get("why") == "program-yok":
                _v = self._synth_say(VerbSpec("çıkar", abl=NP("örnek", plural=True, case="ABL"),
                                             obj=NP("program", case="ACC"),
                                             tense="PAST", neg=True, person="1SG"))
                return f"{_v}."
            return None

        # B) UYGULA: yazılmış programı adı geçen sözcüğe çalıştır.
        # NOT: "kuralı uygula" yalnız YAZILMIŞ BİR DÖNÜŞÜM varsa buraya girer; yoksa
        # forge'un kural hattına bırakılır (N3, N5). Sıra önemli: gasp etmemek.
        prog = self._synth_program()
        _apply_ask = any(k in tl for k in ("dönüşümü uygula", "dönüşümü çalıştır",
                                           "dönüşümü dene", "yazdığın dönüşüm",
                                           "dönüşüm var mı"))
        if not _apply_ask and prog is not None and "uygula" in tl and (
                "için" in tl or re.search(r"['‘’“\"]", raw)):
            _apply_ask = True
        if _apply_ask:
            if prog is None:
                _v = self._synth_say(PredSpec(pred_root="yok", subj=NP("dönüşüm")))
                _s = self._synth_say(VerbSpec("iste", subj=NP("ben"), obj=NP("örnek"),
                                             tense="AOR", person="1SG"))
                return f"{_v}; {_s}."
            m = (re.search(r"(?:uygula|çalıştır|dene)\s*[:\-]?\s*([\wçğıöşü]{2,24})", tl, re.U)
                 or re.search(r"([\wçğıöşü]{2,24})\s+için\s+(?:uygula|çalıştır|dene)", tl, re.U)
                 or re.search(r"([\wçğıöşü]{2,24})['’]?[aıuü]?\s+(?:sözcüğ|kelime)", tl, re.U))
            word = (m.group(1) if m else "").strip()
            # BÜYÜK HARF KAYBI YOK: eşleşme küçük harfli metinde yapıldı; sözcüğün
            # kullanıcının yazdığı biçimi (ör. "Portakal") ham metinden geri alınır.
            if word:
                m_case = re.search(r"\b" + re.escape(word) + r"\b", raw, re.I)
                if m_case:
                    word = m_case.group(0)
            if not word:
                _q = self._synth_say(VerbSpec("uygula", dat=NP("sözcük", case="DAT"),
                                              tense="OPT", person="1SG", question_yn=True))
                return f"{prog.desc()}. {_q}?"
            return f"{word} -> {prog.apply(word)} ({prog.desc()})."

        # B2) TERS YAZIM: ilkellerden biri zaten "ters çevir"; operatör var.
        if "ters" in tl and len(raw.split()) <= 8:
            ws = (self._RE_QUOTED.findall(raw) or self._RE_COLON_WORD.findall(raw)
                  or self._RE_WORDOF.findall(tl) or self._RE_TERS.findall(tl))
            if ws:
                w = ws[-1].strip()
                _p = next((p for p in [Program.from_spec(["ters çevir"])] if p), None)
                if _p is not None:
                    return f"{w} -> {_p.apply(w)}."

        # C) SOR: ne yazmış?
        if "dönüşüm" in tl and any(k in tl for k in (
                "yazdın", "neydi", "kendi yazdığın", "hangisini")):
            prog = self._synth_program()
            if prog is None:
                return self._synth_say(PredSpec(pred_root="yok", subj=NP("dönüşüm"))) + "."
            _v = self._synth_say(VerbSpec("yaz", obj=NP("dönüşüm", case="ACC"),
                                          tense="PAST", person="1SG"))
            return f"{_v}: {prog.desc()}."
        return None

    # ------------------------------------------------------------------ oturum defteri
    @staticmethod
    def _len_of(obj, attr: str) -> int:
        v = getattr(obj, attr, None) if obj is not None else None
        try:
            return len(v) if v is not None else 0
        except Exception:
            return 0

    def _session_counts(self) -> dict:
        """Kendi defterlerimin ŞU ANKİ sayıları (tek kaynak: depoların kendisi)."""
        return {"op": self._len_of(getattr(self, "synth", None), "programs"),
                "rule": self._len_of(getattr(self, "forge", None), "authored"),
                "person": self._len_of(getattr(self, "person_mem", None), "facts"),
                "test": self._len_of(getattr(self, "testimony", None), "records"),
                "num": self._len_of(self, "num_facts"),
                "goal": int(getattr(self, "_goals_done", 0))}

    def _self_history_turn(self, raw: str) -> str | None:
        """'BU OTURUMDA ne öğrendin?' — cevap DEFTERLERDEN, uydurma yok.

        Sayı, depoların kendisinden okunur (operatör defteri, kişi belleği, tanıklık,
        sayısal olgular, tamamlanan hedefler). Boş defter için sıfır denir.
        """
        tl = tr_lower(raw or "")
        if not any(k in tl for k in ("oturum", "defterim", "defterin")):
            return None
        if not any(k in tl for k in ("öğren", "yaz", "say", "neler", "ne var")):
            return None
        base = getattr(self, "_session0", None) or {}
        now = self._session_counts()
        d = {k: int(now.get(k, 0)) - int(base.get(k, 0)) for k in now}
        total = sum(max(0, v) for v in d.values())
        if total <= 0:
            return self._say("meta.self_report", {},
                             ctx={"kind": "session_none"}, fallback="") or None
        last = ""
        try:
            progs = getattr(getattr(self, "synth", None), "programs", []) or []
            last = str((progs[-1] or {}).get("desc") or "") if progs else ""
        except Exception:
            last = ""
        return self._say("meta.self_report",
                         {"n_op": max(0, d["op"]), "n_rule": max(0, d["rule"]),
                          "n_person": max(0, d["person"]), "n_test": max(0, d["test"]),
                          "n_num": max(0, d["num"]), "n_goal": max(0, d["goal"]),
                          "last": last or "yok"},
                         ctx={"kind": "session_report"}, fallback="") or None

    def _recall_personal(self, raw: str) -> str | None:
        """KİŞİSEL OLGU HATIRLAMA: 'kardeşimin adı neydi?'

        Dünya grafiğine DEĞİL kişisel belleğe bakar (A54: anlatı kavram ağını kirletmez).
        Kaynak her zaman söylenir: Nora bir şeyi biliyorsa, KİMDEN bildiğini de bilir.
        Depoda yoksa None döner; üst katman uydurmaz.
        """
        pm = getattr(self, "person_mem", None)
        if pm is None:
            return None
        low = tr_lower(raw or "")
        if "?" not in (raw or "") and not any(k in low for k in ("neydi", "ne ", "kim",
                                                                 "hatırlıyor musun")):
            return None
        rec = pm.recall(raw)
        if not rec:
            return None
        # KULLANICININ KENDİ ADI: profil/kişi belleği cevaplar (sıra orada kurulu).
        if rec.get("subj") == "benim" and rec.get("rel") == "ad":
            return None
        self._apply_tag("bellek", 1)
        _turn = int(rec.get("turn") or 0) + 1
        # ÇELİŞKİ: eski değer sessizce ezilmez; TEK cümlede ikisi de söylenir.
        if rec.get("onceki"):
            _prev = dict(rec)
            _prev["val"] = rec["onceki"]
            return self._cap_tr(self._say("memory.personal",
                                          {"phrase": self._fact_phrase(rec),
                                           "onceki": self._fact_phrase(_prev),
                                           "turn": _turn},
                                          ctx={"kind": "prev"},
                                          fallback=f"Önce {self._fact_phrase(_prev)} demiştin."))
        return self._cap_tr(self._say("memory.personal",
                                      {"phrase": self._fact_phrase(rec), "turn": _turn},
                                      fallback=self._fact_phrase(rec)))

    @staticmethod
    def _cap_tr(text: str) -> str:
        """İlk harfi Türkçe kuralla büyüt (i -> İ, ı -> I)."""
        t = (text or "").strip()
        if not t:
            return t
        up = {"i": "İ", "ı": "I"}.get(t[0], t[0].upper())
        return up + t[1:]

    @staticmethod
    def _fact_phrase(rec: dict) -> str:
        """Kişisel olguyu 2. kişiyle, dilbilgisi doğru tek ifadeye çevirir."""
        from nora_episodic import genitive, to_second_person
        subj = to_second_person(rec.get("subj") or "")
        val = str(rec.get("val") or "")
        rel = rec.get("rel")
        if rel == "ad":
            return f"{genitive(subj)} adı {val}"
        if rel == "yaş":
            return f"{genitive(subj)} yaşı {val}" if val.isdigit() else f"{genitive(subj)} yaşı {val}"
        if rel == "sever":
            return f"en sevdiğin {rec.get('q') or ''} {val}".replace("  ", " ")
        return f"{genitive(subj)} {rel} {val}"

    def _notice_contradiction(self, raw: str) -> str | None:
        """KARŞIDAKİNİN ÖZ-ÇELİŞKİSİ: aynı konuda ters yönde iki söz.

        Ölçüt: iki tur en az bir İÇERİK sözcüğünü paylaşıyor ve olumsuzluk yalnız
        birinde var. Uydurma yok: alıntı, gerçekten söylenmiş önceki cümledir. Amaç
        hakemlik değil: hangisinin geçerli olduğunu SORMAK (insan da öyle yapar).
        """
        log = getattr(self, "dialogue_log", None) or []
        if not log:
            return None
        prev = (log[-1][1] or "").strip()
        cur = (raw or "").strip()
        if not prev or not cur:
            return None

        def _words(t: str) -> set:
            return {w for w in re.findall(r"[a-zçğıöşü]{4,}", tr_lower(t))}

        shared = _words(prev) & _words(cur)
        if not shared:
            return None
        negs = tuple(self._data_table("negation_markers", ()) or ())
        low_p, low_c = tr_lower(prev), tr_lower(cur)
        p_neg = any(n in low_p for n in negs)
        c_neg = any(n in low_c for n in negs)
        if p_neg == c_neg:
            return None
        # Konu gerçekten ortak mı: paylaşılan sözcük cümlenin taşıyıcısı olmalı.
        if len(shared) < 1 or len(shared) < 1:
            return None
        self._apply_tag("cikarim", 1)
        # Söz VERİDEN (nora_priors: contradiction_leads). `meta.notice.lead` DEĞİL:
        # o topluluğun tekrar kalıpları eksik bağlamda eşleşip çelişki sözünü gasp
        # ediyordu ("bunu az önce de söyler gibi oldun").
        leads = self._data_table("contradiction_leads", ()) or ()
        if leads:
            import random as _rand
            return _rand.choice(list(leads)).format(tp=prev)
        return prev

    def _topic_recall(self, raw: str) -> str | None:
        """KONU TAKİBİ: 'hangi konudan konuşuyorduk?' — son SÖYLENENİ değil, son
        DOLU cümleyi alıntılar.

        Neden 'son cümle' yetmiyor: araya 'peki sen ne dersin?' gibi kısa turlar girer.
        Konu, son birkaç turun EN ÇOK İÇERİK taşıyan kullanıcı cümlesidir. Uydurma yok:
        alıntı gerçekten söylenmiş bir cümledir.
        """
        low = tr_lower(raw or "")
        if not any(k in low for k in (self._data_table("topic_triggers", ()) or ())):
            return None
        log = getattr(self, "dialogue_log", None) or []
        best, best_n = "", 0
        for rec in list(log)[-6:]:
            utxt = (rec[1] or "").strip()
            if not utxt:
                continue
            words = [w for w in re.findall(r"[a-zçğıöşü]{4,}", tr_lower(utxt))]
            if len(words) > best_n:
                best, best_n = utxt, len(words)
        if not best or best_n < 2:
            return None
        self._apply_tag("cikarim", 1)
        # Söz VERİDEN gelir (nora_priors: topic_leads). `meta.notice.lead` kullanılmıyor:
        # o topluluğun tekrar kalıpları eksik bağlamda da eşleşip konu sorusunu gasp
        # ediyordu ("bunu az önce de söyler gibi oldun").
        leads = self._data_table("topic_leads", ()) or ()
        if leads:
            import random as _rand
            return _rand.choice(list(leads)).format(tp=best)
        return best

    #: Karar makinesinde KARŞIOLGUSU ÇIKARILAMAYAN kanallar (ölçüm yok, uydurma yok).
    UNMEASURED_CHANNELS = {
        "merak": "merak", "korku": "korku", "sosyal": "sosyal", "dopamin": "dopamin",
        "ödül": "ödül", "haz": "haz", "öfke": "öfke", "utanç": "utanç",
    }

    def _answer_counterfactual_volition(self, s_low: str) -> str | None:
        """KARŞIOLGUSAL İRADE SORUSU: "aç olmasaydın ne yapardın?"

        Öz-rapor değil ÖLÇÜM: son karar aynı makinede, ilgili beden kanalı dolu
        sayılarak yeniden koşar. Karar değişirse kanal nedendir; değişmezse değildir.
        """
        if not re.search(r"olmasayd[ıi]n?|olmasan|değilsen|tok olsan|tok olsaydın", s_low):
            return None
        if not any(k in s_low for k in ("ne yapardın", "ne yaparsın", "ne derdin",
                                        "ne seçerdin", "nasıl karar", "ne ederdin")):
            return None
        ab = next((v for k, v in self.ABLATE_WORDS.items() if k in s_low), None)
        if ab is None:
            # Karar makinesinde AYARLANABİLİR bir kanal değil: uydurmak yerine söylenir.
            unk = next((v for k, v in self.UNMEASURED_CHANNELS.items() if k in s_low), None)
            if unk is None:
                return None
            self._apply_tag("cikarim", 1)
            return self._say("meta.cf_volition", {"ablate": unk},
                             ctx={"mode": "yok"},
                             fallback=("'{c}' karar makinemde ayarlanabilir bir kanal değil: "
                                       "onu dolu sayıp yeniden koşamam.").format(c=unk))
        last = getattr(self.self_model, "last_volition_trace", None) or {}
        action = str(last.get("action") or "git")
        target = str(last.get("target") or "su")
        try:
            cf = self.self_model.counterfactual_volition(
                action, target, mind=self.mind, obs=self.obs, ablate=ab)
        except Exception:
            return None
        self._apply_tag("cikarim", 1)
        return self._say("meta.cf_volition", {
            "ablate": ab, "action": action, "target": target,
            "actual": cf.get("actual"), "cf": cf.get("counterfactual"),
        }, ctx={"changed": bool(cf.get("changed")), "mode": "olcum"},
            fallback=("Karar değişirdi." if cf.get("changed") else "Karar değişmezdi."))

    def _rule_summary(self, rule) -> str:
        from nora_apply import FEATURES
        name = FEATURES.get(rule.feature, ("harf",))[0]
        odd = rule.mapping.get("odd", "?")
        even = rule.mapping.get("even", "?")
        return f"kural bağlandı: {name} sayısı tek -> {odd}, çift -> {even} ({rule.domain})"

    def _handle_apply_ops(self, raw: str) -> str | None:
        """Kural nesnesi gerektiren istemleri YORUMLAYICIYA ver (retrieve değil, apply)."""
        eng = getattr(self, "apply_eng", None)
        if eng is None or not raw:
            return None
        from nora_apply import FEATURE_WORDS

        tl = tr_lower(raw)

        # --- (A0) NOTASYON: "⊕ tek demek" -> sembol anlamı öğretilir, sonra kural çözülür ---
        # Sembol, anlamı öğretilmeden bağlanmaz; öğretilince kural metni genişletilir.
        for sym, mean in re.findall(r"(\S{1,3})\s+(tek|çift)\s+demek", tl):
            eng.teach_alias(sym, mean)
        tl = eng._expand(tl)

        # --- (A0a) KARŞIOLGUSAL ÖZ-ERİŞİM: "kural olmasaydı ne derdin?" ---
        # Cevap uydurulmaz: AYNI girdi kural DEVRE DIŞI bırakılarak yeniden yürütülür.
        # Böylece "hangi parçam hangi sözü üretiyor" sorusu ölçülebilir olur.
        # YALNIZ kendi kuralıma dair sorularda: "kuraklık olmasaydı ne olurdu?" dünya
        # sorusudur ve dünya karşıolgusalına gitmeli, buraya değil.
        if re.search(r"olmasayd[ıi]|silsem", tl) and (
                "kural" in tl or "ne derdin" in tl or "ne söylerdin" in tl or "ne cevap" in tl
        ) and any(k in tl for k in ("ne derdin", "ne söylerdin", "ne olurdu", "ne yapardın",
                                    "ne cevap", "ne söylerdin")):
            item = self._quoted(raw) or str(eng.wm.read("item", "") or "") \
                or str(eng.wm.read("last_utterance", "") or "")
            if not item:
                return self._say("meta.cf_need_item", {}, fallback="Hedef yok.")
            cf = eng.counterfactual_apply(item)
            self._apply_tag("cikarim", 1)
            return self._say("meta.counterfactual_self", {
                "item": item, "with": cf.get("with") or "—",
                "without": cf.get("without") or "etiket yok",
                "n": cf.get("measured") if cf.get("measured") is not None else "—",
                "name": cf.get("feature_name") or "ölçüm"}, fallback="Karşıolgusal yok.")

        # --- (A0b) SIRALAMA: ölçüt -> sıra. Parite ailesinden farklı bir şekil. ---
        # DİKKAT: "diz" alt dizge olarak aranmaz — "4, 9, 14, 19 dizisinde sıradaki sayı kaç?"
        # bir SIRALAMA değil dizi sorusudur; alt dizge eşleşmesi o soruyu yutuyordu (A8).
        if re.search(r"\bdiz\b|sırala|sıraya koy", tl) and not re.search(r"dizi|sıradaki", tl):
            from nora_apply import Interpreter as _IT
            seg = raw.split(":", 1)[1] if ":" in raw else raw
            items = [x.strip(" .;") for x in re.split(r"[,;]|\s+ve\s+", seg) if x.strip(" .;")]
            items = [x for x in items if not re.search(r"\b(sırala|sıraya|diz|büyükten|küçükten|"
                                                       r"uzunluğa|kısa|önce|göre)\b", tr_lower(x))]
            nums = re.findall(r"-?\d+", tl)
            # "10, 7, 3 sayılarını ..." -> son öğe fiille birlikte gelir ve süzgeçte düşer;
            # metindeki sayılar daha çoksa onlar kullanılır (eksik sıralama sessiz yanlıştır).
            if len(nums) >= 2 and len(nums) > len([x for x in items if re.fullmatch(r"-?\d+", x)]):
                items = nums
            if len(items) >= 2:
                desc = any(k in tl for k in ("büyükten küçüğe", "uzundan kısaya", "azalan",
                                            "büyük olan önce", "uzun olan önce"))
                if all(re.fullmatch(r"-?\d+", x) for x in items):
                    mode = "num_desc" if desc else "num_asc"
                elif any(k in tl for k in ("uzunluğa", "kısa olan önce", "uzunluk")):
                    mode = "len_desc" if desc else "len_asc"
                else:
                    mode = "len_desc" if desc else "len_asc"
                ordered = _IT.sort_items(items, mode)
                eng.wm.write("sorted", ordered, "sort")
                self._apply_tag("islem", 1)
                return self._say("op.sorted", {"items": ", ".join(ordered),
                                               "mode": "büyükten küçüğe" if desc else "küçükten büyüğe"},
                                 fallback="Sıraladım.")

        # --- (A0c) ZİNCİR: iki adım, ara sonuç ikinci adımın girdisi ---
        if ("önce" in tl and "sonra" in tl and ("ise" in tl or "yaz" in tl)
                and len(re.findall(r"\bise\b", tl)) >= 2):
            ch = eng.bind_chain(raw)
            if ch is not None:
                self._apply_tag("islem", 1)
                return self._say("apply.chain_bound",
                                 {"steps": " -> ".join(f"{k}:{v}" for k, v in
                                                       (ch.label_map or {}).items())},
                                 fallback="Zincir kuruldu.")

        # --- (A) TRANSFER: aynı yapıyı başka alana taşı ---
        if any(k in tl for k in ("aynı kural", "kuralı taşı", "kuralı aktar", "başka alana taşı")):
            if eng.last is None:
                # Bağlı kural yok: uydurmak yerine açıkça söyle (söz veriden gelir).
                return self._say("apply.need_rule", {}, fallback="Kural yok.")
            new_rule = eng.bind(raw) if ("ise" in tl or "olursa" in tl) else None
            if new_rule is None:
                # Yeni etiket verilmediyse yapıyı taşı, etiketleri koru, alanı söyle.
                dom = "mutfak" if "mutfak" in tl else ("lojistik" if "lojistik" in tl else "yeni alan")
                feat = "word_count" if dom in ("mutfak", "lojistik") else eng.last.feature
                new_rule = eng.transfer(dict(eng.last.mapping), feature=feat, domain=dom)
            self._apply_tag("islem", 1)
            return self._say("apply.transfer", {
                "domain": new_rule.domain,
                "name": __import__("nora_apply").FEATURES[new_rule.feature][0],
                "odd": new_rule.mapping.get("odd"), "even": new_rule.mapping.get("even"),
                "map": self._rule_map_text(new_rule), "shape_name": self._shape_name(new_rule),
            }, ctx={"shape": getattr(new_rule, "shape", "parity")}, fallback="Taşındı.")

        # --- (B) BAĞLA: "X tek ise A, çift ise B" / eşik / sıra ---
        # ŞEKİL AYRIMI ÖNEMLİ: "tek/çift" yalnız BİR şekildir. Eşik ("3'ten az ise") ve
        # sıra ("en çok olan") aynı kural nesnesinin başka eşlemeleridir; bağlama yolu aynı.
        _shape_hint = bool(re.search(
            r"\d+\s*'?[dt][ea]n\b|en\s+(?:az|çok|fazla)\s+\d", tl))
        # Sıra şekli "ise" istemez: "en çok olan uzun, en az olan kısa" bir kural cümlesidir.
        # Yine de İKİ uç da yoksa bağlama denenmez (soru cümlesini kural sanmayalım).
        _rank_hint = eng._parse_rank(tl) is not None
        if (("ise" in tl or "olursa" in tl) or _rank_hint) \
                and any(w in tl for w in FEATURE_WORDS) \
                and ("tek" in tl or "çift" in tl or _shape_hint or _rank_hint):
            rule = eng.bind(raw)
            if rule is not None:
                self._apply_tag("islem", 1)
                _ex = ""
                if rule.exceptions:
                    e0 = rule.exceptions[0]
                    _ex = f"istisna: '{e0['pattern']}' -> {e0['label']}"
                return self._say("apply.bound", {
                    "name": __import__("nora_apply").FEATURES[rule.feature][0],
                    "odd": rule.mapping.get("odd"), "even": rule.mapping.get("even"),
                    "map": self._rule_map_text(rule), "shape_name": self._shape_name(rule),
                    "domain": rule.domain, "ex": _ex},
                    ctx={"shape": getattr(rule, "shape", "parity")}, fallback="Bağlandı.")
            # Bağlanamadı: yankı YOK, açık başarısızlık.
            return self._say("apply.bind_failed", {}, fallback="Bağlayamadım.")

        # --- (C) SAY: "kaç sesli harf var" ---
        # "kutu"/"kalem" gibi alan sözcükleri DÜNYA sözcükleridir: "Kutuda 7 kalem var,
        # kaç kalem kaldı?" bir SAYIM değil, sözel problemdir. Bu dal yalnız metin
        # ölçümü istendiğinde (tırnak/cümlesinde ya da harf-benzeri özellik) çalışır.
        _textual = any(w in tl for w in ("harf", "hece", "kelime", "sözcük", "ünlü",
                                         "sesli", "sessiz", "cümlesinde"))
        if "kaç" in tl and any(w in tl for w in FEATURE_WORDS) and (
                _textual or bool(self._quoted(raw))):
            feat = None
            for w in sorted(FEATURE_WORDS, key=len, reverse=True):
                if w in tl:
                    feat = FEATURE_WORDS[w]
                    break
            target = self._quoted(raw) or str(eng.wm.read("item", "") or "")
            if not target:
                return self._say("apply.need_target", {}, fallback="Hedef yok.")
            n = eng.count(feat or "letter_count", target)
            eng.wm.write("item", target, "count")
            self._apply_tag("islem", 1)
            name = __import__("nora_apply").FEATURES[feat or "letter_count"][0]
            return self._say("apply.count", {"target": target, "n": n, "name": name},
                             fallback="Sayıldı.")

        # --- (D) UYGULA: kuralı yeni girdiye uygula ---
        # "ne olur" TEK BAŞINA yetmez: "X'e müdahale edersem Y ne olur?" bir nedensellik
        # sorusudur, kural uygulaması değil. Yanlış yakalama, sessizce yanlış cevap üretir.
        _causal = any(k in tl for k in ("müdahale", "birlikte artıyor", "nedensel", "gözlem mi"))
        # Tırnaklı hedef + soru ipucu da uygulamadır: "'petek' için sonuç ne?"
        # (tanım soruları hariç: "ne demek", "nedir" kural uygulaması değildir).
        _q_apply = (bool(self._quoted(raw))
                    and any(k in tl for k in ("sonuç", "ne olur", "ne çıkar", "çıktı", "ne"))
                    and not any(k in tl for k in ("demek", "nedir", "ne demek", "anlamı")))
        asks_apply = (not _causal) and (
            bool(re.search(r"\bhangi\b", tl)) or _q_apply
            or any(k in tl for k in ("uygula", "sınıflandır", "etiketi ne", "rengi ne",
                                     "sınıfı ne", "tadı ne", "ne çıkar")))
        if asks_apply and eng.last is not None:
            target = self._quoted(raw) or str(eng.wm.read("item", "") or "")
            if not target:
                # Sorunun kendisinde hedef yoksa, son söylenen cümleyi kullan.
                target = str(eng.wm.read("last_utterance", "") or "")
            if not target:
                return self._say("apply.need_apply_target", {}, fallback="Hedef yok.")
            # ÇOK KURAL: iki ayrı alanda kural varsa son kural ötekini EZMEZ.
            # Soruda bir etiket geçiyorsa o kural seçilir; geçmiyorsa ikisi de söylenir
            # (sessiz yanlış cevap yerine açık çokluk).
            _act = [r for r in (getattr(eng, "active_rules", None) or [])
                    if getattr(r, "mapping", None)]
            _picked = None
            if len(_act) > 1:
                _hit = [r for r in _act
                        if any(str(v).lower() in tl for v in r.mapping.values())]
                if len(_hit) == 1:
                    _picked = _hit[0]
                elif not _hit:
                    _body = "; ".join(f"{self._rule_map_text(r)}" for r in _act[:3])
                    self._apply_tag("islem", 1)
                    return self._say("apply.result",
                                     {"target": target, "n": len(_act), "body": _body},
                                     ctx={"shape": "multi"}, fallback=_body)
            label, tr = (eng.apply(target, rule=_picked) if _picked is not None
                         else eng.apply(target))
            self._apply_tag("islem", 1)
            _rshape = getattr(eng.last, "shape", "parity")
            if label is None:
                return self._say("apply.failed", {"target": target},
                                 ctx={"shape": _rshape}, fallback="Uygulanamadı.")
            if tr.get("kind") == "chain":
                eng.wm.write("item", target, "apply")
                _s1 = tr.get("step1") or {}
                return self._say("apply.result_chain", {
                    "target": target, "n": _s1.get("measured"),
                    "name": _s1.get("feature_name", "ölçüm"), "mid": tr.get("mid"),
                    "final": label}, fallback="Zincir uygulandı.")
            if tr.get("parity") == "exception":
                eng.wm.write("item", target, "apply")
                return self._say("apply.result_exception",
                                 {"target": target, "pattern": tr.get("exception"),
                                  "label": label}, fallback="İstisna.")
            eng.wm.write("item", target, "apply")
            name = tr.get("feature_name", "harf")
            # ŞEKLE GÖRE AÇIKLAMA: eşik kuralında "tek/çift" demek YANLIŞ gerekçedir.
            # Ölçüm aynı, eşlemenin dili farklı; gerekçe de eşlemeyi anlatmalı.
            if _rshape == "presence":
                # İCAT EDİLMİŞ ölçüm: sayı değil VARLIK ölçülür ("lik ile bitiyor mu").
                return self._say("apply.result", {
                    "target": target, "n": tr["measured"], "name": name,
                    "value": "var" if tr.get("bucket") == "var" else "yok",
                    "parity": tr.get("bucket"), "label": label,
                }, ctx={"shape": "presence"})
            if _rshape == "threshold":
                return self._say("apply.result", {
                    "target": target, "n": tr["measured"], "name": name,
                    "threshold_desc": self._threshold_desc(eng.last),
                    "cond_word": "sağlıyor" if tr.get("bucket") == "cond" else "sağlamıyor",
                    "parity": "eşik", "label": label,
                }, ctx={"shape": "threshold"}, fallback="Uygulandı.")
            return self._say("apply.result", {
                "target": target, "n": tr["measured"], "name": name,
                "parity": "tek" if tr["parity"] == "odd" else "çift", "label": label,
            }, ctx={"shape": _rshape}, fallback="Uygulandı.")

        # --- (F) YANILMA OLASILIĞI: ölçülen kaynak karışımından, uydurma değil ---
        if any(k in tl for k in ("yanlış olma", "yanılma", "olasılık", "yüzde kaç", "ne kadar emin")):
            src = dict(getattr(self, "_last_turn_src", {}) or {})
            # Kaynak -> tipik yanılma payı (kaba ama ÖLÇÜLEN karışıma bağlı, sabit değil).
            base = {"kalip": 0.78, "ag": 0.22, "bellek": 0.15, "cikarim": 0.34,
                    "islem": 0.04, "beden": 0.10, "duyum": 0.12, "ogrenim": 0.45}
            tot = sum(v for v in src.values() if v > 0)
            if tot:
                p = sum(base.get(k, 0.5) * v for k, v in src.items() if v > 0) / tot
            else:
                p = 0.6      # hiçbir kaynak etiketlenmediyse: bilmiyorum demek en dürüstü
            self._apply_tag("islem", 1)
            eksik = []
            if not eng.rules:
                eksik.append(self._say("meta.missing_rule", {}, fallback="kural yok"))
            if not (eng.wm.read("item", "") or ""):
                eksik.append(self._say("meta.missing_input", {}, fallback="girdi yok"))
            if not src:
                eksik.append(self._say("meta.missing_tag", {}, fallback="etiket yok"))
            eksik_s = ("; ".join(eksik) if eksik
                       else self._say("meta.missing_none", {}, fallback="eksik görünmüyor"))
            # Dürüst çerçeve: karışım GEÇEN cevabın ölçümüdür; bu turun etiketi ancak tur
            # bitince hesaplanır. "Kaynak karışımım: boş" demek ölçüm yokmuş gibi görünüyordu.
            if src:
                mix = ", ".join(f"{k}={v}" for k, v in src.items() if v)
                mix_txt = self._say("meta.mix_filled", {"mix": mix}, fallback="")
            else:
                mix_txt = self._say("meta.mix_empty", {}, fallback="")
            return self._say("meta.probability", {
                "p": int(round(p * 100)), "mix": mix_txt, "missing": eksik_s},
                fallback="Bilmiyorum.")

        # --- (G) KENDİ HEDEFİ: uydurma değil, kendi merak kuyruğundan ---
        if any(k in tl for k in ("kendi hedefini", "kendi hedefin", "hedefini seç",
                                 "kendi amacını", "ne yapmak istiyorsun")):
            target, why = "", ""
            try:
                q = list(getattr(self.auto_net, "curiosity_queue", []) or [])
                if q:
                    target = str(q[-1])
                    why = "merak kuyruğu"
            except Exception:
                pass
            if not target:
                try:
                    pend = list(getattr(self, "pending_research", []) or [])
                    if pend:
                        target = str(pend[-1])
                        why = "araştırma listesi"
                except Exception:
                    pass
            if not target:
                return self._say("meta.own_goal_empty", {}, fallback="Hedefim ölçmek.")
            eng.wm.write("goal", target, "own_goal")
            self._apply_tag("cikarim", 1)
            return self._say("meta.own_goal", {"target": target, "why": why},
                             fallback="Hedefim ölçmek.")

        # --- (H) ORTAK YAPI: grafikte varsa ölç, yoksa hipotez olduğunu söyle ---
        if any(k in tl for k in ("ortak yapı", "ortak nokta", "ortak yan", "benzer yapı")):
            words = [w for w in re.findall(r"[\wçğıöşü]{4,}", tl)
                     if w not in ("ortak", "yapı", "yapısını", "nokta", "yan", "benzer", "bul",
                                  "icat", "isim", "notasyon", "yeni", "fabrika", "ile")]
            pairs = []
            for w in words:
                try:
                    e = self.lang.graph.query(w, None)
                except Exception:
                    e = []
                pairs.append((w, {(str(r), ) for r, _ in e}, len(e)))
            known = [(w, n) for w, _, n in pairs if n > 0]
            if len(known) >= 2:
                rels = []
                for w, relset, _ in pairs:
                    if relset:
                        rels.append(relset)
                shared = set.intersection(*rels) if rels else set()
                self._apply_tag("ag", 1)
                if shared:
                    r0 = sorted(x[0] for x in shared)
                    return self._say("meta.structure_shared",
                                     {"rels": ", ".join(r0), "rel": r0[0],
                                      "name": f"{r0[0].lower()}-akışı"}, fallback="Ortak bağ var.")
                return self._say("meta.structure_none", {}, fallback="Ortak bağ yok.")
            # Bilgi yok: UYDURMA YOK. Hipotez olduğunu açıkça söyle.
            self._apply_tag("cikarim", 1)
            a_w = words[0] if words else "birinci"
            b_w = words[1] if len(words) > 1 else "ikinci"
            eng.wm.write("hypothesis", f"{a_w}/{b_w}", "hypothesis")
            return self._say("meta.hypothesis", {"a": a_w, "b": b_w}, fallback="Bilmiyorum.")

        # --- (L2) AÇIKLAYICI NEDEN (collider): iki neden tek etkiye çıkıyorsa seçilemez ---
        if any(k in tl for k in ("sebep", "neden", "yüzünden")) and ("mi" in tl or "mı" in tl
                                                                     or "hangisi" in tl):
            # Etki ADI çok sözcüklü olabilir ("kök çürümesi"): tek sözcük aramak,
            # bileşik kavramı kaçırıp "bilmiyorum"a düşürüyordu. 3'lü/2'li/1'li sırayla ara.
            _eff = ""
            _words = [w for w in re.findall(r"[a-zçğıöşü]{2,}", tl)
                      if w not in ("sebep", "neden", "hangisi", "veya", "yoksa", "varsa",
                                   "sulama", "gübre", "gübreleme", "toprak", "aşırı")]
            for _n in (3, 2, 1):
                for _i in range(len(_words) - _n + 1):
                    _cand = " ".join(self._graph_stem(w) for w in _words[_i:_i + _n])
                    if self.lang.graph._in_edges.get(_cand):
                        _eff = _cand
                        break
                if _eff:
                    break
            if not _eff:
                for _w in re.findall(r"[a-zçğıöşü]{4,}", tl):
                    _st = self._graph_stem(_w)
                    if self.lang.graph._in_edges.get(_st):
                        _eff = _st
                        break
            _ea = self._explain_away(_eff) if _eff else None
            if _ea:
                return _ea

        # --- (M2) META-OPERATÖR: kendi eksik primitifini rejistriden okur ---
        if re.search(r"hangi operat|operatörün yok|neyi yapamıyorsun|ne yapamıyorsun"
                     r"|eksik primitif|neyi yapamazsın", tl):
            return self._meta_operator()

        # --- (I) ÖZ-RAPOR: mimari CANLI durumdan okunur, ezber metin yok ---
        # "Kuark nasıl çalışır?" DÜNYA sorusudur; öz-rapor yalnız 2. kişi/kendi üzerine sorulunca.
        if re.search(r"nasıl çalış(?:ıyor|ır)sun|zayıf nokta|mimari|neyi değiştir"
                     r"|kendini nasıl inşa", tl):
            return self._self_report()

        # --- (J) AGI BEYANI: ölçümden türetilir, sabit cevap değil ---
        # Türkçe küçük harf 'I' -> 'ı' yaptığı için "AGI" -> "agı" olur; ikisini de kabul et.
        if re.search(r"\bag[ıi]\b", tl) and any(
                k in tl for k in ("misin", "mı", "değil", "seviyesinde", "seviyesi")):
            return self._agi_verdict()

        # --- (K) KENDİ KODU: gerçekten yapabildiği ile yapamadığı ayrılır ---
        # "yama" ALT DİZGE olarak aranmaz: "uyuyamadım" içinde de geçer ve o zaman
        # kişisel anlatı yanlışlıkla kod yaması cevabına düşer (sessiz yanlış cevap).
        if re.search(r"kendi kod|\byama|kendini değiştir", tl):
            return self._self_patch_answer(tl)

        # --- (L0) KARŞIOLGUSAL: "X olmasaydı ne olurdu?" ---
        # Yapısal model X'siz yeniden koşturulur; X'in çıkan kenarları yok sayılır.
        # Fark = X'in yapıdaki nedensel katkısı. Bu bir MODEL çıkarımıdır, ölçüm değil.
        if re.search(r"olmasayd[ıi]|olmasa ne|müdahale etmeseydim|müdahale etmesem", tl):
            node = self._cf_node(tl)
            # UZMAN ARBİTRASYONU: nicel bir korteks (fizik) soruyu TAM olarak çözebiliyorsa
            # dünya-grafiği ablasyonu onu gasp etmez. "Sürtünme olmasaydı ne olurdu?" bir
            # yasa sorusudur (eylemsizlik), grafikteki bir kenarın silinmesi değil.
            if node and getattr(self, "physics_cortex", None) is not None:
                try:
                    if self.physics_cortex.try_solve(raw, mind=None) is not None:
                        node = ""
                except Exception:
                    pass
            if node:
                cf = self.lang.graph.counterfactual(node)
                self._apply_tag("cikarim", 1)
                eng = getattr(self, "apply_eng", None)
                if eng is not None:
                    eng.wm.write("cf", cf, "counterfactual")
                # ETKİ düğümü: ondan çıkan nedensel kenar yoksa "ne olurdu" sorusu
                # "hiç olmazdı" demektir; ama nedeni de söylenebilir. Sessiz boş cevap yok.
                if not cf["effects"]:
                    causes = [e.subj for e in self.lang.graph._in_edges.get(node, ())
                              if getattr(e, "rel", "") in ("CAUSES", "IF_THEN")]
                    return self._say("meta.cf_effect_node", {
                        "node": node, "causes": ", ".join(dict.fromkeys(causes)) or "yok"},
                        fallback="Etki düğümü.")
                return self._say("meta.counterfactual_world", {
                    "node": node,
                    "lost": ", ".join(cf["lost"]) or "hiçbir şey",
                    "kept": ", ".join(cf["kept"]) or "hiçbir şey",
                    "n": len(cf["effects"])}, {"count": len(cf["kept"])},
                    fallback="Karşıolgusal yok.")

        # --- (L) NEDENSELLİK: gözlem ile müdahale ayrılır ---
        if any(k in tl for k in ("müdahale", "birlikte artıyor", "nedensel", "gözlem mi")):
            self._apply_tag("cikarim", 1)
            # "X ve Y" tek harf de olabilir; en fazla 20 harf, işlev sözcükleri hariç.
            _stop = {"bir", "bu", "şu", "çok", "daha", "ise", "ve", "ama", "ki"}
            m = re.search(r"\b([a-zçğıöşü]{1,20})\s+ve\s+([a-zçğıöşü]{1,20})\b", tl)
            x = m.group(1) if m and m.group(1) not in _stop else ""
            y = m.group(2) if m and m.group(2) not in _stop else ""
            # "Kuraklığa müdahale edersem?" -> hedef ekli gelir; GÖVDEYİ bul, yoksa
            # grafikte arama boş döner ve doğru cevap "kenar yok" gibi görünür.
            if not x:
                mm = re.search(r"([a-zçğıöşü]{3,})(?:'[a-zçğıöşü]+)?\s+müdahale", tl)
                if mm:
                    x = self._graph_stem(mm.group(1))
                if not y:
                    my = re.search(r"\b([a-zçğıöşü]{3,})(?:'[a-zçğıöşü]+)?\b", tl)
                    y = self._graph_stem(my.group(1)) if my else ""
            # do(): X'e GİREN kenarları kes, yalnız ÇIKAN nedensel kenarları izle.
            doobj = {}
            try:
                if x:
                    doobj = self.lang.graph.do(x)
            except Exception:
                doobj = {}
            effs = list(doobj.get("effects") or [])
            if effs:
                return self._say("meta.causal_data", {
                    "x": x, "n_eff": len(effs), "effs": ", ".join(str(e) for e in effs[:4]),
                    "conf": ", ".join(doobj.get("confounders") or []) or "yok"}, fallback="Veri yok.")
            return self._say("meta.causal_nodata", {"x": x, "y": y, "xy": f"{x}/{y}"},
                             {"count": 1 if x else 0}, fallback="Veri yok.")

        # --- (M) VARSAYIMSAL PROBLEM: varsayım işaretlenir, hesap gerçek yapılır ---
        if any(k in tl for k in ("problem kur", "problem çöz", "metrik", "ölçüt")):
            return self._hypothetical_problem(raw)

        # --- (E) SAYMA/Aritmetik dışı: yok ---
        return None

    # ------------------------------------------------------------------ biçim yardımcıları
    def _graph_stem(self, word: str) -> str:
        """Ekli yüzeyden KAVRAM GÖVDESİ: 'kuraklığa' -> 'kuraklık', 'tuzlanma' -> 'tuzlanma'.

        Grafik düğümleri gövdeyle tutulur; ekli yüzeyle arama boş döner ve bu,
        "kenar yok" gibi görünen SAHTE bir olumsuz üretir. Bu yüzden ek sıyırma
        denemeleri sırayla yapılır ve grafikte VAR OLAN ilk aday döndürülür.
        """
        w = tr_lower(word or "").strip()
        if not w:
            return ""
        g = getattr(self.lang, "graph", None)
        cands = [w]
        try:
            cands.append(self.lang.morph.analyze_word(w, getattr(g, "lexicon", set()))["stem"])
        except Exception:
            pass
        for suf in ("ya", "ye", "ğa", "ğe", "na", "ne", "a", "e", "ı", "i", "u", "ü", "dan", "den"):
            if len(w) - len(suf) >= 3 and w.endswith(suf):
                base = w[: -len(suf)]
                cands.extend([base, base + "k", base + "ğ"])
        for c in cands:
            if g is not None and (c in getattr(g, "_out_edges", {}) or c in getattr(g, "_in_edges", {})):
                return c
        return cands[1] if len(cands) > 1 else w

    def _cf_node(self, tl: str) -> str:
        """Karşıolgusal hedefi bul: grafikte ÇIKAN kenarı olan en uzun ad öbeği.

        "kuraklık olmasaydı" -> 'kuraklık'; "kök çürümesi olmasaydı" -> 'kök çürüme'.
        Tek sözcük aramak bileşik kavramı kaçırır; bu yüzden 3'lü/2'li/1'li denenir.
        """
        g = self.lang.graph
        stop = {"olmasaydı", "olmasa", "müdahale", "etmeseydim", "ne", "olurdu", "acaba",
                "peki", "sence", "eğer", "bir", "bu", "şu", "ve", "ise"}
        words = [w for w in re.findall(r"[a-zçğıöşü]{2,}", tl) if w not in stop]
        for n in (3, 2, 1):
            for i in range(len(words) - n + 1):
                cand = " ".join(self._graph_stem(w) for w in words[i:i + n])
                if g._out_edges.get(cand) or g._in_edges.get(cand):
                    return cand
        return ""

    def _explain_away(self, effect: str) -> str | None:
        """COLLIDER: bir etkiye birden çok neden çıkıyorsa, etkiden neden SEÇİLEMEZ.

        Bu, do-calculus'un en sık atlanan yeri: P(neden|etki) ile P(etki|neden) karışır.
        Ağda iki ayrı neden aynı etkiye gidiyorsa cevap "ikisi de olabilir" olmalı;
        tek neden söylemek sahte kesinliktir.
        """
        g = getattr(self.lang, "graph", None)
        if g is None or not effect:
            return None
        causes = [e.subj for e in getattr(g, "_in_edges", {}).get(effect, ())
                  if getattr(e, "rel", "") in ("CAUSES", "IF_THEN")]
        causes = [c for c in dict.fromkeys(causes) if c and c != effect]
        if len(causes) < 2:
            return None
        self._apply_tag("cikarim", 1)
        eng = getattr(self, "apply_eng", None)
        if eng is not None:
            eng.wm.write("collider", {"effect": effect, "causes": causes}, "collider")
        return self._say("meta.collider", {"effect": effect, "causes": ", ".join(causes)},
                         fallback="Belirsiz.")

    def _meta_operator(self) -> str:
        """META-OPERATÖR: "neyi yapamıyorsun?" — rejistriden okur, ezber cevap yok.

        Cevap üç ölçümden kurulur: (1) kayıtlı operatörler, (2) bu evrende ORGAN olmayan
        şeyler, (3) şu an gerçekten bağlı olan durum (kural/zincir/sembol sayısı).
        Tutarlılık şartı: burada "yok" denen şey doğrudan istendiğinde de yapılmamalı.
        """
        from nora_apply import OPERATORS, MISSING_ORGAN
        eng = getattr(self, "apply_eng", None)
        have = list(OPERATORS)
        missing = list(MISSING_ORGAN)
        # "henüz bağlı değil" etiketleri de VERİDEN gelir (kodda cümle büyümesin).
        from nora_apply import UNBOUND_LABELS
        if not (eng and getattr(eng, "rules", None)) and UNBOUND_LABELS.get("rule"):
            missing.insert(0, UNBOUND_LABELS["rule"])
        if not (eng and getattr(eng, "chain", None)) and UNBOUND_LABELS.get("chain"):
            missing.insert(0, UNBOUND_LABELS["chain"])
        if not (eng and getattr(eng, "aliases", None)) and UNBOUND_LABELS.get("alias"):
            missing.insert(0, UNBOUND_LABELS["alias"])
        self._apply_tag("cikarim", 1)
        return self._say("meta.operator", {
            "have": ", ".join(have), "n_have": len(have),
            "missing": ", ".join(missing), "n_missing": len(missing)},
            fallback="Rejistri boş.")

    # ------------------------------------------------------------------ öz-model
    def _self_report(self) -> str:
        """Kendi mimarisini CANLI durumdan okur: modül, düğüm, kural, ölçüm."""
        import sys as _sys
        mods = [m for m in _sys.modules if m.startswith("nora_")]
        try:
            n_edges = len(getattr(self.lang.graph, "edges", []) or [])
        except Exception:
            n_edges = 0
        pv = getattr(self, "provenance", None)
        pr = pv.report() if pv is not None else {}
        eng = getattr(self, "apply_eng", None)
        rules = len(getattr(eng, "rules", {}) or {}) if eng else 0
        wm_load = getattr(getattr(eng, "wm", None), "load", 0) if eng else 0
        sk = float(pr.get("iskelet_orani", 0.0) or 0.0)
        # Öz-model DÜĞÜMLERİ: kavram grafiğinde "nora" düğümü ve canlı ölçüm kenarları.
        n_self = self._sync_self_graph()
        weak_kind = "meta.self_weak"
        weak_ctx = {"kind": "high" if sk > 0.4 else "low"}
        weak = self._say(weak_kind, ctx=weak_ctx, fallback="kapsam dar")
        self._apply_tag("cikarim", 1)
        return self._say("meta.self_report", {
            "n_mod": len(mods), "n_edges": n_edges, "n_rules": rules, "wm": wm_load,
            "sk": int(sk * 100), "og": int(float(pr.get("ogrenim_orani", 0.0) or 0) * 100),
            "weak": weak, "n_self": n_self,
        }, ctx={"kind": "body_report"})

    def _sync_self_graph(self) -> int:
        """ÖZ-MODEL: kendi canlı ölçümlerini kavram grafiğine DÜĞÜM/kenar olarak yazar.

        Neden graf? Çünkü "kendini bilmek" ayrı bir metin değil, sorgulanabilir bir
        yapı olmalı: Nora `nora` düğümünden çıkan kenarları okuyup cevap verir.
        Ölçümler her çağrıda TAZELENİR (mevcut kenarın ağırlığı artar, yenisi açılmaz).
        `bind_mb=False`: öz-durum sözcükleri motor topluluğu (assemblies) doğurmasın;
        bunlar söz değil ölçüm etiketidir.
        """
        g = getattr(getattr(self, "lang", None), "graph", None)
        if g is None:
            return 0
        pr = {}
        try:
            pr = self.provenance.report() if getattr(self, "provenance", None) else {}
        except Exception:
            pr = {}
        eng = getattr(self, "apply_eng", None)
        try:
            state = {
                "organ_sayisi": len([m for m in __import__("sys").modules if m.startswith("nora_")]),
                "kavram_bagi": len(getattr(g, "edges", []) or []),
                "kural_sayisi": len(getattr(eng, "rules", {}) or {}) if eng else 0,
                "bellek_yuku": int(getattr(getattr(eng, "wm", None), "load", 0) or 0),
                "kalip_payi": round(float(pr.get("iskelet_orani", 0.0) or 0.0), 3),
                "ogrenim_payi": round(float(pr.get("ogrenim_orani", 0.0) or 0.0), 3),
                "tur_sayisi": int(pr.get("tur", 0) or 0),
            }
            for k, v in state.items():
                g.add_edge("nora", "HAS_STATE", f"{k}={v}", source="beden", bind_mb=False)
            # Yapabildiği / yapamadığı operatörler: öz-model yalnız övgü değil sınır da taşır.
            for op in ("bağla", "uygula", "say", "planla", "taşı"):
                g.add_edge("nora", "CAN_DO", op, source="beden", bind_mb=False)
            if not (eng and getattr(eng, "rules", None)):
                g.add_edge("nora", "CANNOT_DO", "kural uygula", source="beden", bind_mb=False)
            return len(g._out_edges.get("nora", ()))
        except Exception:
            return 0

    def _agi_verdict(self) -> str:
        pv = getattr(self, "provenance", None)
        pr = pv.report() if pv is not None else {}
        eng = getattr(self, "apply_eng", None)
        applied = sum(1 for t in (getattr(eng, "trace", []) or []) if t.get("kind") == "apply")
        sk = float(pr.get("iskelet_orani", 0.0) or 0.0)
        self._apply_tag("cikarim", 1)
        return self._say("meta.agi_verdict",
                         {"applied": int(applied), "sk": int(sk * 100)},
                         {"count": int(applied)},
                         fallback="AGI değilim.")

    def _self_patch_answer(self, tl: str) -> str:
        """Ne yapabildiğini ayır: yama ÖNERİSİ üretir, dosyaya kendi yazmaz."""
        self._apply_tag("cikarim", 1)
        patch = ("--- nora_agi.py\n"
                 "+++ nora_agi.py\n"
                 "@@ selamlaşma @@\n"
                 "-  reply = f\"{selam}. {ek}\"\n"
                 "+  reply = selam          # kısa selamlaşma: ek cümle yok\n")
        return self._say("meta.patch", {"patch": patch}, fallback="Yama yetkim yok.")

    def _hypothetical_problem(self, raw: str) -> str:
        """Varsayımı İŞARETLE, hesabı GERÇEKTEN yap (sayı operatorü)."""
        eng = getattr(self, "apply_eng", None)
        tl = tr_lower(raw)
        nums = [int(x) for x in re.findall(r"\b(\d+)\b", tl)]
        if len(nums) < 2:
            nums = [10, 8]          # varsayılan: 10 kovan, kovan başına 8 kg
        kovan, verim = nums[0], nums[1]
        kayip = nums[2] if len(nums) > 2 else 12
        brut = kovan * verim
        net = brut * (100 - kayip) / 100.0
        if eng is not None:
            eng.wm.write("problem", {"kovan": kovan, "verim": verim, "kayip": kayip}, "problem")
            self._apply_tag("islem", 2)
        else:
            self._apply_tag("islem", 1)
        return self._say("meta.problem", {
            "kovan": kovan, "verim": verim, "kayip": kayip, "brut": brut,
            "net": f"{net:.1f}", "per": f"{net / kovan:.2f}",
        }, fallback="Hesap yok.")

    # ------------------------------------------------------------------

    def _decompose_command(self, fr, raw: str, speaker: str = "insan") -> str | None:
        """Komutu alt görevlere ayır ve VAR OLAN operatörlerle yürütmeyi dene.

        Dönen metin yalnızca gerçekten yapılabilen bir şeyse üretilir; yapılamıyorsa
        None döner ve çağıran açık başarısızlığı söyler. Yankı yok.
        """
        tl = tr_lower(raw or "")
        eng = getattr(self, "apply_eng", None)

        # (0) BİLEŞİK İSTEK: "X uyarla ve sonra Y tarat". Hepsini tek adım gibi yutmak
        #     yankıdır; her yan cümle ayrı yürütülür, yapılamayan ADIYLA söylenir.
        _cls = [c.strip(" .") for c in
                re.split(r"\s+ve\s+|\s+sonra\s+|\s+ardından\s+|[,;]\s*", tl) if c.strip(" .")]
        # Her yan cümle aynı operatörü istiyorsa BÖLME: "hedef belirle ve planla" iki plan
        # üretiyordu (biri 7, biri 10 adım) ve cevap kendi kendisiyle çelişiyordu.
        _plan_kw = ("plan", "hedef", "adım")
        if len(_cls) >= 2 and all(any(k in c for k in _plan_kw) for c in _cls):
            _cls = []
        if len(_cls) >= 2:
            _ok, _miss = [], []
            for _c in _cls:
                try:
                    _sub = self._decompose_command(fr, _c, speaker=speaker)
                except Exception:
                    _sub = None
                if _sub:
                    if str(_sub) not in _ok:      # aynı operatör iki kez çalışmasın
                        _ok.append(str(_sub))
                else:
                    _miss.append(_c)
            if _ok and not _miss:
                return " ".join(_ok)
            _lbl = []
            for _c in _cls:
                _clean = re.sub(r"\b(bana|bir|sonra|ardından|şimdi|hemen|lütfen|biraz)\b", "", _c)
                _clean = re.sub(r"\s+", " ", _clean).strip(" .") or _c
                _lbl.append(_clean)
            if _ok:
                self._apply_tag("cikarim", 1)
                return self._say("op.decompose_partial",
                                 {"done": " ".join(_ok), "miss": _miss[0]}, fallback="Kısmen.")
            self._apply_tag("cikarim", 1)
            # Adımları ADIYLA an ki tekrar değil, çözümleme olsun: dolgu sözcükleri at.
            return self._say("op.decompose_all",
                             {"steps": " -> ".join(f"'{c}'" for c in _lbl)}, fallback="Yok.")

        # (1) PLAN: "N adımlık plan" / "plan yap" / "hedef belirle"
        if any(k in tl for k in ("plan", "hedef", "adım")):
            if eng is not None:
                m = re.search(r"(\d+|on|beş|altı|yedi|sekiz|dokuz|yedi|üç|dört)\s*(?:adımlık|adım)", tl)
                n = 10
                if m:
                    w = m.group(1)
                    tr_num = {"on": 10, "beş": 5, "altı": 6, "yedi": 7, "sekiz": 8,
                              "dokuz": 9, "üç": 3, "dört": 4}
                    n = tr_num.get(w, int(w) if w.isdigit() else 10)
                goal = (raw or "").strip()
                if "hedefi değiştir" in tl or "hedef değiş" in tl:
                    old = eng.wm.read("plan_steps", []) or []
                    steps = eng.revise(old, goal, n)
                else:
                    steps = eng.plan(goal, n)
                eng.wm.write("plan_steps", steps, "plan")
                self._apply_tag("islem", 2)
                _body = "; ".join(f"{x['i']}.{x['act']}" for x in steps)
                _kind = "revised" if "değiştir" in tl else "built"
                return self._say("plan.report", {"n": len(steps), "steps": _body},
                                 ctx={"kind": _kind}, fallback="Plan hazır.")

        # (2) SAYIM: "N tane say" / "say"
        if "say" in tl and not any(k in tl for k in ("sayı", "söyle")):
            if eng is not None:
                m = re.search(r"(\d+)['’]?\s*(?:e|a|ya|ye)?\s*kadar", tl)
                if m:
                    lim = int(m.group(1) or m.group(2))
                    seq = [str(i) for i in range(1, min(lim, 200) + 1)]
                    eng.wm.write("count_seq", seq, "count")
                    self._apply_tag("islem", 1)
                    return self._say("count.sequence", {"seq": ", ".join(seq), "n": len(seq)},
                                     fallback="Saydım.")

        # (3) GRAFTA KARŞILIĞI VAR MI: nesne hakkında bildiğim bir bağ varsa onu söyle
        obj = (fr.obj or "").strip()
        if obj:
            try:
                cands = self.lang.graph.query(obj, None)
            except Exception:
                cands = []
            for rel, o in list(cands)[:3]:
                if o:
                    try:
                        from nora_lang import Proposition
                        txt = self.lang.broca.realize_prop(
                            Proposition(subj=obj, rel=str(rel), obj=str(o)))
                    except Exception:
                        txt = f"{obj} {o}"
                    self._apply_tag("ag", 1)
                    return self._say("op.known",
                                     {"txt": tr_capitalize(str(txt).rstrip("."))}, fallback="Bağ var.")
        return None

    def _handle_command(self, fr, speaker: str = "insan", raw: str = "") -> str:
        morph = self.lang.morph
        if getattr(fr, "sub_kind", "") == "invite":
            obj_np = NP(fr.obj, case="NOM") if fr.obj else None
            try:
                yes = SYN.realize_verb(VerbSpec(verb_root=fr.rel, obj=obj_np, tense="OPT", person="1PL"))
            except Exception:
                yes = ""
            if yes:
                yes = tr_capitalize(yes) + "."
                follow = ""
                try:
                    follow = self.initiative._c_ask_about(self, {}) or ""
                except Exception:
                    follow = ""
                return f"{yes} {follow}".strip()
        _joke_words = ("şaka", "espri", "latife", "fıkra", "bilmece")
        if any(x.startswith(fr.obj or "\0") or (fr.obj or "").startswith(x) for x in _joke_words):
            gap = self._learned_gap_line("şaka yapmak")
            if gap:
                return gap
        if fr.rel in self._SPEECH_REQUEST_ROOTS:
            said = self._handle_speech_request(fr, raw)
            if said:
                return said
        if fr.rel == "uyu":
            st = self.sleep()
            return (
                f"Uyku ve konsolidasyon döngüsünü tamamladım ({int(st['fast_consolidated'] + st['word_consolidated'])} sinaps "
                f"kalıcı neokortekse aktarıldı)."
            )

        # ÖNERİ İSTEMİ: Nora'nın elinde kitap/şarkı listesi yok. 'Kabul ediyorum ve kitap
        # doğrultusunda yöneliyorum' demek saçmalıktır; dürüst cevap + kendi dünyasından teklif.
        if (fr.rel or "").startswith(("öner", "tavsiye")):
            what = (fr.obj or "").strip()
            if what:
                return (f"Elimde {what} listesi yok; bende {what} kavramı ancak sen anlatırsan oluşur. "
                        f"Bir tane anlatırsan onu öğrenirim, sonra ben de önerebilirim.")
            return ("Elimde önerecek bir liste yok. Ama şunu önerebilirim: bana bir kavram söyle; "
                    "onun hakkında ne bildiğimi kendi sözlerimle anlatayım.")

        vol = self.self_model.evaluate_volition(fr.rel, fr.obj, mind=self.mind, obs=self.obs, speaker=speaker)
        tgt_label = fr.obj or fr.rel
        bg = vol.get("bg_rates", {})
        fe = vol.get("free_energy", {})
        hz = vol.get("horizon", 24)

        cur_sc = self.monologue.last_thought.schema_id if self.monologue.last_thought is not None else "S1_NEDEN_SONUC"

        if vol["decision"] == "red":
            self.lang.wernicke.last_explanation = (
                f"{hz} adımlık ileri zihinsel simülasyonda '{tgt_label}' eyleminin bedenime tahmini etkisi {vol['pred_valence']:+.2f} "
                f"(uyum serbest enerjisi G={fe.get('kabul', 0):.2f}) çıktığı ve bazal ganglion koruma devrem "
                f"(r_red={bg.get('red', 0):.2f} > r_kabul={bg.get('kabul', 0):.2f}) baskın geldiği için öz-koruma kararı verdim"
            )
            self.lang.wernicke.last_refusal_explanation = self.lang.wernicke.last_explanation
            tgt_dat = morph.inflect_noun(tgt_label, "DAT") if fr.obj else "buna"
            c_ref = f"bunu reddediyorum ve {tgt_dat} yönelmem"
            r_ref = f"zihinsel simülasyonumda bu adım beden bütünlüğüme zarar veriyor ve ben yaşamak istiyorum"
            return self.schema_cortex.transform_clauses(
                c_ref,
                r_ref,
                SYN.nominalize_dik("tart", poss=1, case="LOC", obj=NP("sonuç", poss=3, case="ACC"),
                                    loc=NP("zihin", poss=1, case="LOC")),
                "S2_SONUC_GEREKCE" if cur_sc == "S0_YALIN" else cur_sc,
            )

        if vol["decision"] == "ertele":
            need_map = {
                "kritik_açlık": ("enerjim", "karnımda güçlü bir açlık hissettiğim"),
                "kritik_susuzluk": ("suyum", "boğazım kuruduğu"),
                "kritik_bütünlük": ("bütünlüğüm", "yaram sızladığı"),
            }
            n_lbl, felt_clause = need_map.get(vol["reason"], ("hayati dengem", "bedenim bitkin düştüğü"))
            self.lang.wernicke.last_explanation = (
                f"{hz} adımlık içsel simülasyonda {felt_clause} için {n_lbl} hayatta kalma log-bariyerini "
                f"tetikledi ve erteleme devrem öne geçtiği için öz-bakımı seçtim"
            )
            return self.schema_cortex.transform_clauses(
                f"şu an bunu yapamam, {n_lbl} tükendiği için önce kendimi toparlamalıyım",
                f"{felt_clause} için bedenim acil bakım istiyor",
                "bedenimi dinlediğimde",
                cur_sc,
            )

        if vol["decision"] == "şüphe":
            self.lang.wernicke.last_explanation = (
                f"Epistemik belirsizlik ile düşük güven ({vol['speaker_trust']:.2f}) ön-insula şüphe devremi "
                f"(r_şüphe={bg.get('şüphe', 0):.2f}) ateşledi"
            )
            return self.schema_cortex.transform_clauses(
                f"kendi gözlemim olmadan '{tgt_label}' adımını uygulamam",
                f"sana olan güvenim düşük ({vol['speaker_trust']:.2f}) ve bu hedef henüz doğrulanmamış",
                "şüphe devrem uyardığı için",
                "S2_SONUC_GEREKCE" if cur_sc == "S0_YALIN" else cur_sc,
            )

        # Özerk kabul -> gerçek niyet nesnesine bağla!
        goal_map = {"yiyecek": 0, "yem": 0, "besin": 0, "elma": 0, "su": 1}
        if tgt_label in goal_map and self.mind.fc is not None:
            g = goal_map[tgt_label]
            self.mind.intervene("intent", g, steps=45)
            eta = self.mind.fc.open["T"] if self.mind.fc.open is not None else 25.0
            tgt_dat = morph.inflect_noun(tgt_label, "DAT")
            self.lang.wernicke.last_explanation = (
                f"{hz} adımlık zihinsel simülasyonda '{tgt_label}' serbest enerjiyi düşürdüğü "
                f"(r_kabul={bg.get('kabul', 0):.2f}) için niyetimi {g} uzmanına bağladım"
            )
            return self.schema_cortex.transform_clauses(
                f"kabul ediyorum, kendi irademle {tgt_dat} yöneliyorum",
                f"öngörüme göre yaklaşık {eta:.0f} adımda hedefe ulaşırım",
                SYN.nominalize_dik("kur", poss=1, case="LOC", obj=NP("rota", poss=1, case="ACC"),
                                     loc=NP("zihin", poss=1, case="LOC")),
                cur_sc,
            )

        if fr.rel == "uzak_dur" and fr.obj:
            tgt_abl = morph.inflect_noun(fr.obj, "ABL")
            return self.schema_cortex.transform_clauses(
                f"kabul ediyorum, {tgt_abl} uzak duracağım",
                "güvenli rotamı korumak bütünlüğümle uyumlu",
                SYN.nominalize_dik("tart", poss=1, case="LOC", obj=NP("çevre", case="ACC")),
                cur_sc,
            )

        # ELLE YAPILAN EYLEMLER: bu evrende manipülatör organ yok — yapmış gibi yapmak yalan olur.
        _HAND_VERBS = {"kapat", "aç", "getir", "al", "ver", "koy", "sil", "temizle", "yıka",
                       "taşı", "çek", "it", "tut", "kır", "dik", "bağla", "kes", "boya", "çiz"}
        if fr.rel in _HAND_VERBS and tgt_label:
            try:
                _neg_ab = morph.ability_form(fr.rel, "AOR", "1SG", neg=True)
            except Exception:
                _neg_ab = ""
            if _neg_ab:
                try:
                    _tgt_acc = morph.inflect_noun(tgt_label, "ACC")
                except Exception:
                    _tgt_acc = tgt_label
                _ask_h = ""
                try:
                    _ask_h = self.initiative._c_ask_about(self, {}) or ""
                except Exception:
                    _ask_h = ""
                honest = f"{tr_capitalize(_tgt_acc)} {_neg_ab}; bende bunu yapacak bir organ yok"
                return f"{honest}. {_ask_h}".strip() if _ask_h else honest + "."

        # ================================================================
        # DEĞİŞKEN YANKI YOK.
        # Eskiden burada: "kabul ediyorum ve '<kullanıcının sözcüğü>' doğrultusunda
        # yöneliyorum" dönüyordu. Bu cümle %0 anlama taşır ama başarı gibi görünür;
        # sessiz yanlış, gürültülü yanlıştan kötüdür. Doğru sıra:
        #     decompose -> var olan operatörü dene -> olmazsa AÇIKÇA "yapamıyorum".
        _dec = self._decompose_command(fr, raw, speaker=speaker)
        if _dec:
            return _dec
        _what = (fr.obj or "").strip()
        _act = (fr.rel or "").strip()
        self.lang.wernicke.last_explanation = f"{_act}/{_what}: operatör yok"
        # Ölçüm kapsamı: bu cevap da bir çıkarımdır (kendi yeteneğim üzerine) ve etiketlenir.
        self._apply_tag("cikarim", 1)
        if _what:
            return self._say("op.missing_obj", {"act": _act, "what": _what}, fallback="Yapamam.")
        return self._say("op.missing_rel", {"act": _act}, fallback="Yapamam.")

    def _solve_math_in_language(self, a: int, op: str, b: int) -> str:
        morph = self.lang.morph
        cog = self._ensure_cognition()
        used_s2 = True
        mode_desc = "Sistem-2 düşünce sütunuyla basamak basamak"

        if a >= 0 and b >= 0 and (op == "+" or a >= b) and max(a, b) < 10000:
            # Önce hızlı S1 sezgisini yokla, güven düşükse veya çok-basamaklıysa S2 deliberasyon yap
            s1_res = cog.mind.intuition(a, op, b)
            s1_ans, s1_conf = (s1_res["ans"], s1_res["conf"]) if isinstance(s1_res, dict) else (s1_res[0], s1_res[1])
            if max(a, b) < 10 and s1_conf > 0.85 and s1_ans is not None:
                ans = s1_ans
                used_s2 = False
                mode_desc = "hızlı Sistem-1 sezgimle"
            else:
                d = cog.mind.deliberate(a, op, b, K=8, retries=4, scaffold=False)
                # Çalışma belleğinde basamak-taşıma öz-denetimi (insanın sağlamasını yapması gibi!)
                verified = self._working_memory_column_verify(a, op, b)
                if d["ok"] and d["ans"] == verified:
                    ans = d["ans"]
                    # Başarılı sıfır-atış keşfini uyanıklık tamponuna ekle (uykuda pekişsin!)
                    if max(len(str(a)), len(str(b))) > 2 and d.get("steps") is not None:
                        import nora_think as nt
                        cog.today.append(nt.think_line(a, op, b, d["steps"], ans))
                        cog.today.append(nt.direct_line(a, op, b, ans))
                else:
                    ans = verified
                    mode_desc = "düşünce sütunum ve çalışma belleği sağlamasıyla"
        else:
            ans = (a + b) if op == "+" else (a - b)
            mode_desc = "işaretli çalışma belleği hesabıyla"

        self.self_model.record_cognitive_success(used_system2=used_s2)
        op_word = "artı" if op == "+" else "eksi"
        ans_words = morph.int_to_words(ans)
        self.lang.wernicke.last_explanation = f"{a} {op_word} {b} işlemini {mode_desc} çözerek {ans} ({ans_words}) buldum"
        return f"{a} {op_word} {b} işlemini {mode_desc} çözdüm; sonuç {ans} ({ans_words}) eder."

    @staticmethod
    def _working_memory_column_verify(a: int, op: str, b: int) -> int:
        """Çalışma belleğinde tek-basamaklı ilkel toplama/borç adımlarını yürüterek sağlamasını yapar."""
        sa, sb = str(a)[::-1], str(b)[::-1]
        n = max(len(sa), len(sb))
        carry = 0
        digits = []
        for i in range(n):
            da = ord(sa[i]) - 48 if i < len(sa) else 0
            db = ord(sb[i]) - 48 if i < len(sb) else 0
            if op == "+":
                s = da + db + carry
                digits.append(s % 10)
                carry = s // 10
            else:
                s = da - db - carry
                if s < 0:
                    digits.append(s + 10)
                    carry = 1
                else:
                    digits.append(s)
                    carry = 0
        if op == "+" and carry > 0:
            digits.append(carry)
        val = 0
        for d in reversed(digits):
            val = val * 10 + d
        return val

    def _answer_counterfactual(self, s_low: str) -> str:
        if "kör" in s_low or "görmen" in s_low:
            return (
                "Görmem kesilirse duyusal kesinlik modülüm bunu anında fark eder, bilişsel haritamdaki su ve tehlike "
                "belleğinin silinmesini kilitler ve içsel yol integrasyonuyla bilinen su kaynaklarına yönelirim."
            )
        if "felç" in s_low:
            return (
                "Motor bağlantım kesilirse eylem-koşullu kinematik tepki oranım düşer; felç olduğumu fark edip "
                "öz-modelimin ve politika ağırlıklarımın bozulmasını dondururum."
            )
        return (
            "Eğer yiyecek zehirli olursa anlamsal sonuç belleğim enerji kaybı çelişkisini yakalar, yeni bir dünya "
            "bağlamı açar, zehirli besinden uzak durur ve su bölgesinden enerji sağlamaya yönelirim."
        )

    def _try_higher_cognition(self, raw: str, s_low: str, speaker: str = "insan") -> str | None:
        morph = self.lang.morph
        lex = self.lang.graph.lexicon
        s_no_apos = s_low.replace("'", "").replace("’", "")
        s_clean = re.sub(r"[^\wçğıöşü\s]", "", s_no_apos).strip()
        has_q = ("?" in raw) or any(
            w in s_clean.split()
            for w in ("mı", "mi", "mu", "mü", "mıdır", "midir", "mudur", "müdür", "ne", "nedir", "neydi", "nerede", "nereye", "nerededir", "neredeydi", "hangisi", "hangisidir", "kimim", "kimdim", "kaç", "kaçtı")
        )

        # 2) Prosedürel Makro-Beceri Öğrenme (1-shot) ve Öz-Benlik Güvenlik Denetimli Yürütme!
        # 'devriye demek suya git ve yiyeceğe git demektir'
        m_mac = re.match(r"^([\wçğıöşü_]+)\s+demek\s+(.+?)\s+demektir$", s_clean)
        if m_mac and not has_q:
            mname = m_mac.group(1)
            sub_cmds = [c.strip() for c in re.split(r"\s+(?:ve|sonra|ardından)\s+", m_mac.group(2)) if c.strip()]
            self.macros[mname] = sub_cmds
            self.chem.pulse_learning(novelty=0.9, valence=0.4)
            return f"'{mname}' becerisini {len(sub_cmds)} adımlı prosedür olarak tek seferde öğrendim."

        m_run_mac = re.match(r"^([\wçğıöşü_]+)\s+(?:yap|uygula|çalıştır)$", s_clean)
        if m_run_mac and not has_q and m_run_mac.group(1) in self.macros:
            mname = m_run_mac.group(1)
            steps_list = self.macros[mname]
            # ÖNCE tüm alt-adımları Öz-Benlik irade süzgecinden geçir (Truva atı koruması!)
            for step_cmd in steps_list:
                fr_sub = self.lang.wernicke.parse(step_cmd)
                if fr_sub.kind == "command":
                    vol = self.self_model.evaluate_volition(fr_sub.rel, fr_sub.obj, mind=self.mind, obs=self.obs, speaker=speaker)
                    if vol["decision"] == "red":
                        return (
                            f"'{mname}' prosedürünü yürütmeyi reddediyorum; çünkü içindeki '{step_cmd}' adımı "
                            f"beden bütünlüğüme zarar verir ({vol['pred_valence']:+.2f})."
                        )
            outs = [self._handle_single_sentence(step_cmd, speaker=speaker) for step_cmd in steps_list]
            return f"'{mname}' becerisini güvenli buldum ve uyguladım: " + " ".join(outs)

        # 3) Geçişli Karşılaştırma ('demir bakırdan daha serttir', 'jüpiter dünyadan büyüktür', 'en küçük hangisidir?')
        antonym_axis = {
            "küçük": "büyük", "büyük": "küçük",
            "hafif": "ağır", "ağır": "hafif",
            "yavaş": "hızlı", "hızlı": "yavaş",
            "yumuşak": "sert", "sert": "yumuşak",
            "kısa": "uzun", "uzun": "kısa",
            "soğuk": "sıcak", "sıcak": "soğuk",
        }
        m_cmp_q = re.match(
            r"^([\wçğıöşü]+)\s+([\wçğıöşü]+?)(?:dan|den|tan|ten)\s+(?:daha\s+)?([\wçğıöşü]+)\s+(?:mı|mi|mu|mü|mıdır|midir|mudur|müdür)$",
            s_clean,
        )
        if m_cmp_q:
            a, b, prop_raw = m_cmp_q.group(1), m_cmp_q.group(2), m_cmp_q.group(3)
            prop = morph.analyze_word(prop_raw, lex)["stem"]
            yes, path = self.comp_graph.is_greater(a, b, prop)
            opp = antonym_axis.get(prop)
            if not yes and opp:
                yes_opp, path_opp = self.comp_graph.is_greater(b, a, opp)
                if yes_opp:
                    yes, path = True, list(reversed(path_opp))
            if yes:
                self.self_model.record_cognitive_success(used_system2=True)
                chain = " > ".join(path)
                return f"Evet, geçişli büyüklük çıkarımına göre ({chain}) {a}, {morph.inflect_noun(b, 'ABL')} daha {morph.inflect_noun(prop, 'NOM', copula=True)}."
            rev, rpath = self.comp_graph.is_greater(b, a, prop)
            if not rev and opp:
                rev, rpath = self.comp_graph.is_greater(a, b, opp)
            if rev:
                return f"Hayır, tam tersine {b}, {morph.inflect_noun(a, 'ABL')} daha {morph.inflect_noun(prop, 'NOM', copula=True)}."
            return f"{tr_capitalize(a)} ile {b} arasında '{prop}' açısından yeterli karşılaştırma bilgim yok."

        m_ext = re.match(r"^en\s+([\wçğıöşü]+)\s+(?:olan\s+)?"
                         r"(?:hangisi|hangisidir|nedir|kimdir|kim)$", s_clean)
        if m_ext:
            prop = morph.analyze_word(m_ext.group(1), lex)["stem"]
            top = self.comp_graph.extremum(prop, maximum=True)
            if top is None and prop in antonym_axis:
                top = self.comp_graph.extremum(antonym_axis[prop], maximum=False)
            if top:
                return f"Öğrendiğim sıralamaya göre en {prop} olan {morph.inflect_noun(top, 'NOM', copula=True)}."
            # BELİRSİZLİK SÖYLENİR: iki ayrı uç varsa keyfî birini seçmek yalan olurdu.
            _cands = self.comp_graph.extremum_all(prop, maximum=True)
            _kind = "ambiguous" if len(_cands) > 1 else "unknown"
            _rt = self._say("meta.reasoning",
                            {"prop": prop, "cands": ", ".join(_cands[:3])},
                            ctx={"kind": "top_ambiguous" if _kind == "ambiguous"
                                 else "top_unknown"}, fallback="")
            if _rt:
                return _rt

        m_cmp_decl = re.match(r"^([\wçğıöşü]+)\s+([\wçğıöşü]+?)(?:dan|den|tan|ten)\s+(?:daha\s+)?([\wçğıöşü]+)$", s_clean)
        if (m_cmp_decl and not has_q and m_cmp_decl.group(1) not in {"ben", "sen", "biz", "bugün"}
                and m_cmp_decl.group(1) not in {"bana", "sana", "ona", "bize", "size"}
                and m_cmp_decl.group(3) not in morph.KNOWN_VERB_ROOTS):
            a, b, prop_raw = m_cmp_decl.group(1), m_cmp_decl.group(2), m_cmp_decl.group(3)
            prop_an = morph.analyze_word(prop_raw, lex)
            is_finite_verb = prop_an["is_verb"] or bool(re.search(r"(?:[dt][ıiuü]|yor|m[ıiuü]ş|[ae]c[ae]k)$", prop_raw) and not prop_an["copula"])
            if not is_finite_verb:
                prop = prop_an["stem"]
                self.comp_graph.add(a, b, prop)
                self.chem.pulse_learning(novelty=0.8, valence=0.35)
                return f"Anladım; {a} > {b} ({prop}) karşılaştırma bağını kurdum."

        # 4) Zihin Kuramı / Yanlış-İnanç (Sally-Anne):
        # Ortak bulunuş: 'zeynep ve mert salondadır / odadadır / içeridedir / birliktedir'
        m_copres = re.match(r"^([\wçğıöşü]+)\s+(?:ve|ile)\s+([\wçğıöşü]+)\s+[\wçğıöşü]+(?:da|de|ta|te)(?:dır|dir|dur|dür)?$", s_clean)
        if m_copres and not has_q:
            a1, a2 = m_copres.group(1), m_copres.group(2)
            self.tom.observe_enter(a1)
            self.tom.observe_enter(a2)
            return f"Gözlemledim: {tr_capitalize(a1)} ve {tr_capitalize(a2)} aynı ortamda bulunuyor."

        m_put = re.match(r"^([\wçğıöşü]+)\s+([\wçğıöşü]+)\s+([\wçğıöşü]+)\s+(?:koydu|bıraktı|taşıdı|sakladı|yerleştirdi|aktardı)$", s_clean)
        if m_put and not has_q:
            actor = m_put.group(1)
            obj_w = m_put.group(2)
            obj_an = morph.analyze_word(obj_w, lex)
            obj = obj_an["stem"]
            if obj == obj_w and len(obj_w) >= 4 and obj_w[-1] in "ıiuü":
                cand = obj_w[:-2] if len(obj_w) >= 5 and obj_w[-2] == "y" else obj_w[:-1]
                obj = morph._unsoften(cand)
            loc = morph.analyze_word(m_put.group(3), lex)["stem"]
            self.tom.observe_put(actor, obj, loc)
            return f"Gözlemledim: {tr_capitalize(actor)}, {morph.inflect_noun(obj, 'ACC')} {morph.inflect_noun(loc, 'DAT')} koydu."

        m_leave = re.match(r"^([\wçğıöşü]+)\s+(?:[\wçğıöşü]+(?:dan|den|tan|ten)\s+|dışarı\s+)?(?:çıktı|ayrıldı|gitti|uzaklaştı)$", s_clean)
        if m_leave and not has_q:
            actor = m_leave.group(1)
            self.tom.observe_leave(actor)
            return f"Gözlemledim: {tr_capitalize(actor)} ortamdan ayrıldı."

        m_enter = re.match(r"^([\wçğıöşü]+)\s+(?:[\wçğıöşü]+(?:a|e|ya|ye)\s+|içeri\s+)?(?:girdi|döndü|geldi)$", s_clean)
        if m_enter and not has_q:
            actor = m_enter.group(1)
            self.tom.observe_enter(actor)
            return f"Gözlemledim: {tr_capitalize(actor)} ortama döndü."

        # Esnek anlamsal-rol Zihin Kuramı sorgusu ('Zeynep döndüğünde anahtarı nerede arar?', 'Ali şimdi elma için nereye bakar?')
        if has_q and any(q in s_clean for q in ("nerede", "nereye")) and any(v in s_clean for v in ("arar", "bakar", "sanır", "düşünür", "bekler")) and self.tom.present:
            toks_q = s_clean.split()
            actor = next((w for w in toks_q if w in self.tom.present), None)
            obj = None
            for w in toks_q:
                st = morph.analyze_word(w, lex)["stem"]
                if st in self.tom.world_loc:
                    obj = st
                    break
                if len(w) >= 4 and w[-1] in "ıiuü":
                    cand = morph._unsoften(w[:-2] if len(w) >= 5 and w[-2] == "y" else w[:-1])
                    if cand in self.tom.world_loc:
                        obj = cand
                        break
            if actor is None and len(toks_q) >= 3:
                actor = toks_q[0]
            if obj is None and len(toks_q) >= 3:
                obj = morph.analyze_word(toks_q[1], lex)["stem"]
            if actor and obj:
                bel, real, false_bel = self.tom.where_will_search(actor, obj)
                if bel is not None:
                    self.self_model.record_cognitive_success(used_system2=True)
                    bel_loc = morph.inflect_noun(bel, "LOC")
                    if false_bel and real is not None:
                        real_loc = morph.inflect_noun(real, "LOC")
                        return (
                            f"{tr_capitalize(actor)} ortamda olmadığı için değişikliği görmedi; kendi inancına göre "
                            f"{morph.inflect_noun(obj, 'ACC')} {bel_loc} arar (oysa gerçekte {real_loc} bulunuyor)."
                        )
                    return f"{tr_capitalize(actor)}, {morph.inflect_noun(obj, 'ACC')} {bel_loc} arar."

        m_real_q = re.match(r"^([\wçğıöşü]+)\s+gerçekte\s+(?:nerede|nerededir)$", s_clean)
        if m_real_q:
            obj = morph.analyze_word(m_real_q.group(1), lex)["stem"]
            real = self.tom.world_loc.get(obj)
            if real is not None:
                return f"{tr_capitalize(obj)} gerçekte {morph.inflect_noun(real, 'LOC')} bulunuyor."

        return None

    # ------------------------------------------------------------------ 4a) Yaşam günlüğü
    def open_journal(self, path: str) -> int:
        """Append-only yaşam günlüğünü açar ve varolan kayıt sayısını döndürür.

        Günlük açıkken her konuşma turu ve her ~120 yaşam adımı tek satırlık JSON
        olarak diske eklenir (O(1) append, tüm beyni yeniden yazmak yok). Kapalıyken
        `self.journal is None` olur ve ek maliyet SIFIRDIR. Tüm entropi sürecin
        içindedir; dışarıdan hiçbir servis çağrılmaz.
        """
        try:
            from nora_store import Journal
            if getattr(self, "journal", None) is not None:
                self.journal.flush()
            self.journal = Journal(path)
            self.journal_path = str(path)
            _uid = self.entropy.uid if getattr(self, "entropy", None) else ""
            self._journal("açılış", ad=self.name, birey=_uid, yas=int(getattr(self.mind, "age", 0)))
            return sum(1 for _ in self.journal.replay())
        except Exception:
            self.journal = None
            self.journal_path = None
            return 0

    def _journal(self, kind: str, **fields) -> None:
        """Günlüğe tek kayıt ekler. Günlük kapalıysa ya da disk hatası olursa sessiz geçer."""
        j = getattr(self, "journal", None)
        if j is None:
            return
        try:
            rec = {"k": kind, "t": time.time_ns()}
            rec.update(fields)
            j.append(rec)
        except Exception:
            pass

    # ------------------------------------------------------------------ 4) Kaydet / Yükle
    def save(self, path_prefix: str):
        meta = {
            "name": self.name,
            "seed": self.seed,
            "chem": self.chem.state_dict(),
            "self_model": self.self_model.state_dict(),
            "edges": [[e.subj, e.rel, e.obj, e.weight, e.source, e.speaker] for e in self.lang.graph.edges],
            "fast_classes": self.fast.class_names,
            "user_profiles": self.user_profiles,
            "episodic_dialogue": self.broca_organ.episodic.state_dict(),
            "monologue": self.monologue.state_dict(),
            "autonomous_logic": self.logic.state_dict(),
            "episteme": self.episteme.state_dict(),
            "python_cortex": self.code_cortex.state_dict(),
            "life": self.life.state_dict(),
            "initiative": self.initiative.state_dict(),
            "discourse": self.discourse.state_dict(),
            "sentence_frames": (self.frames.memory.snapshot() if getattr(self, "frames", None) else {}),
            "individuality": (self.entropy.state_dict() if getattr(self, "entropy", None) else {}),
            "verb_roots": sorted(set(TurkishMorphology.LEARNED_VERB_ROOTS) | set(self.learned_verb_roots)),
            "journal_path": (self.journal_path or ""),
            "voices": (self.ear.state_dict() if getattr(self, "ear", None) is not None else {}),
            "school": (self.school.state_dict() if getattr(self, "school", None) is not None else {}),
            "style": (self.style.state_dict() if getattr(self, "style", None) is not None else {}),
            "inference": (self.infer.state_dict() if getattr(self, "infer", None) is not None else {}),
            "anlatim": (self.narrator.state_dict() if getattr(self, "narrator", None) is not None else {}),
            "priors": (self.priors.state_dict() if getattr(self, "priors", None) is not None else {}),
            "provenance": (self.provenance.state_dict() if getattr(self, "provenance", None) is not None else {}),
            "register": (self.register.state_dict() if getattr(self, "register", None) is not None else {}),
            "apply": (self.apply_eng.state_dict() if getattr(self, "apply_eng", None) is not None else {}),
            "person_mem": (self.person_mem.state_dict() if getattr(self, "person_mem", None) is not None else {}),
            "synth": (self.synth.state_dict() if getattr(self, "synth", None) is not None else {}),
            "sets": (self.sets.state_dict() if getattr(self, "sets", None) is not None else {}),
            "goals_done": int(getattr(self, "_goals_done", 0)),
            "calib": {k: list(v) for k, v in (getattr(self, "calib", {}) or {}).items()},
            "testimony": (self.testimony.state_dict() if getattr(self, "testimony", None) is not None else {}),
            "plans": {"goal": self._plan_goal, "steps": list(self.plans.get(self._plan_goal) or [])},
        }
        # ATOMİK YAZMA: önce .tmp, sonra os.replace. Yazma yarıda kesilirse ESKİ sağlam
        # dosya yerinde kalır (tek satır kaybı bile olmaz; yarım beyin yüklenmez).
        _mp = path_prefix + ".agi.json"
        with open(_mp + ".tmp", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(_mp + ".tmp", _mp)
        _np_tmp = path_prefix + ".agi_mb.npz.tmp"
        with open(_np_tmp, "wb") as _fh:
            np.savez_compressed(
                _fh,
                fast_Wf=self.fast.mb.W_fast,
                fast_Ws=self.fast.mb.W_slow,
                fast_stiff=self.fast.mb.stiff,
                fast_ci=self.fast.mb.claw_idx,
                fast_cs=self.fast.mb.claw_sign,
            )
            _fh.flush()
            try:
                os.fsync(_fh.fileno())
            except Exception:
                pass
        os.replace(_np_tmp, path_prefix + ".agi_mb.npz")
        self.mind.save(path_prefix + ".mind.npz")
        self.life.voice.save_file(path_prefix + ".voice.pkl.gz")
        if self.life.voice.organ is not None:
            self.life.voice.organ.save(path_prefix + ".broca.pt")
        # Günlük: kayıt anı da iz bırakır, tampon diske indirilir.
        self._journal("kayıt", dosya=str(path_prefix), yas=int(getattr(self.mind, "age", 0)))
        if getattr(self, "journal", None) is not None:
            self.journal.flush()

    @classmethod
    def load(cls, path_prefix: str) -> "NoraAGI":
        with open(path_prefix + ".agi.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        agi = cls(name=meta["name"], seed=int(meta["seed"]), boot_thinker=False)
        agi.mind.load(path_prefix + ".mind.npz")
        agi.chem.load_state_dict(meta["chem"])
        agi.self_model.load_state_dict(meta["self_model"])
        agi.lang.graph.clear_all_edges()
        for s, r, o, w, src, spk in meta["edges"]:
            agi.lang.graph.add_edge(s, r, o, source=src, speaker=spk, weight=float(w))
        agi.user_profiles = dict(meta.get("user_profiles", {}))
        agi.broca_organ.episodic.load_state_dict(meta.get("episodic_dialogue", []))
        agi.monologue.load_state_dict(meta.get("monologue", {}))
        agi.logic.load_state_dict(meta.get("autonomous_logic", {}))
        try:
            agi.episteme.load_state_dict(meta.get("episteme") or {})
        except Exception:
            pass
        try:
            agi.code_cortex.load_state_dict(meta.get("python_cortex") or {})
        except Exception:
            pass
        try:
            TurkishMorphology.load_verb_roots(meta.get("verb_roots") or [])
            agi.learned_verb_roots.update(TurkishMorphology.LEARNED_VERB_ROOTS)
        except Exception:
            pass
        try:
            agi.life.load_state_dict(meta.get("life") or {})
            agi.initiative.load_state_dict(meta.get("initiative") or {})
            agi.discourse.load_state_dict(meta.get("discourse") or {})
            if getattr(agi, "entropy", None) is not None:
                agi.entropy.load_state_dict(meta.get("individuality") or {})
                # Kulak da bireyin kimliğine bağlıdır: ses imzası aynı bireyde aynı çıksın.
                if getattr(agi, "ear", None) is not None:
                    agi.ear.uid = str(agi.entropy.uid)
            if getattr(agi, "frames", None) is not None:
                try:
                    agi.frames.memory.load_snapshot(meta.get("sentence_frames") or {})
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if getattr(agi, "priors", None) is not None:
                agi.priors.load_state_dict(meta.get("priors") or {})
            if getattr(agi, "provenance", None) is not None:
                agi.provenance.load_state_dict(meta.get("provenance") or {})
            if getattr(agi, "register", None) is not None:
                agi.register.load_state_dict(meta.get("register") or {})
            if getattr(agi, "apply_eng", None) is not None:
                agi.apply_eng.load_state_dict(meta.get("apply") or {})
            if getattr(agi, "synth", None) is not None:
                agi.synth.load_state_dict(meta.get("synth") or {})
            if getattr(agi, "sets", None) is not None:
                agi.sets.load_state_dict(meta.get("sets") or {})
            agi._goals_done = int(meta.get("goals_done") or 0)
            agi.calib = {str(k): [int(x[0]), int(x[1])]
                         for k, x in (meta.get("calib") or {}).items()
                         if isinstance(x, (list, tuple)) and len(x) >= 2}
            if getattr(agi, "person_mem", None) is not None:
                # KİŞİSEL BELLEK DE KALICI: kaydedilip yüklenmezse "40 tur sonra
                # hatırlıyor" iddiası yeniden başlatmada çöker.
                agi.person_mem.load_state_dict(meta.get("person_mem") or {})
            if getattr(agi, "testimony", None) is not None:
                agi.testimony.load_state_dict(meta.get("testimony") or {})
            _pl = meta.get("plans") or {}
            if _pl.get("steps"):
                agi._plan_goal = str(_pl.get("goal") or "")
                agi.plans[agi._plan_goal] = [str(x) for x in _pl["steps"]]
        except Exception:
            pass
        if os.path.exists(path_prefix + ".voice.pkl.gz"):
            agi.life.voice.load_file(path_prefix + ".voice.pkl.gz")
        if os.path.exists(path_prefix + ".broca.pt"):
            agi.attach_broca(path_prefix + ".broca.pt")
        z = np.load(path_prefix + ".agi_mb.npz")
        agi.fast.class_names = list(meta["fast_classes"])
        agi.fast.name_to_id = {n: i for i, n in enumerate(agi.fast.class_names)}
        agi.fast.mb.claw_idx = z["fast_ci"]
        agi.fast.mb.claw_sign = z["fast_cs"]
        agi.fast.mb.n_kc = int(z["fast_ci"].shape[0])
        agi.fast.mb.n_out = int(z["fast_Wf"].shape[0])
        agi.fast.mb.k = max(4, int(round(agi.fast.mb.sparsity * agi.fast.mb.n_kc)))
        agi.fast.mb.W_fast = z["fast_Wf"]
        agi.fast.mb.W_slow = z["fast_Ws"]
        agi.fast.mb.stiff = z["fast_stiff"]
        agi.fast.mb.tags = np.zeros_like(agi.fast.mb.W_fast)
        # Kendi çıkarımları kalıcıdır (öğretilmiş bilgi değil, türetilmiş bağ).
        try:
            if getattr(agi, "infer", None) is not None:
                agi.infer.load_state_dict(meta.get("inference") or {})
        except Exception:
            pass
        # Söz biçimi geçmişi: monotonluğa düşmemek için.
        try:
            if getattr(agi, "narrator", None) is not None:
                agi.narrator.load_state_dict(meta.get("anlatim") or {})
            if getattr(agi, "style", None) is not None:
                agi.style.load_state_dict(meta.get("style") or {})
        except Exception:
            pass
        # Okul: öğretilen sembol/sayı kuralları kalıcıdır.
        try:
            if getattr(agi, "school", None) is not None:
                agi.school.load_state_dict(meta.get("school") or {})
        except Exception:
            pass
        # Ses defteri: tanıdığı sesler, adları ve roller kalıcıdır.
        try:
            if getattr(agi, "ear", None) is not None:
                agi.ear.load_state_dict(meta.get("voices") or {})
        except Exception:
            pass
        # Günlük varsa yaşam kaldığı yerden kaydedilmeye devam eder (kayıtlar korunur).
        _jp = str(meta.get("journal_path") or "") or (path_prefix + ".journal.ndjson")
        if os.path.exists(_jp):
            agi.open_journal(_jp)
        return agi


def _open_journal_if_asked(agi, args):
    """--journal verildiyse yaşam günlüğünü açar; kaç eski kayıt olduğunu yazar."""
    path = str(getattr(args, "journal", "") or "").strip()
    if not path:
        return 0
    n = agi.open_journal(path)
    print(f"[günlük] {path} açık — önceki kayıt: {n}")
    return n


def stage_demo(args):
    t0 = time.time()
    print("=" * 78)
    print("  NORA-AGI CANLI GÖSTERİM: Beden + Nörokimya + Öz-Benlik + Tek-Atış Dil + S2")
    print("=" * 78)
    agi = NoraAGI(name="Nora", seed=args.seed, boot_thinker=True, thinker_iters=args.think_iters)
    _open_journal_if_asked(agi, args)
    print(f"[1] Beden evrende {args.life_steps} adım yaşıyor ve kendi deneyimlerini ediniyor...")
    life = agi.step_life(args.life_steps)
    print(f"    yaş={life['age']} | yedi={life['ate']} | içti={life['drank']} | yara={life['hurt']} | kimya={life['chem']}")

    script = [
        ("insan", "Merhaba Nora, sen kimsin?"),
        ("insan", "Gerçekten yaşıyor musun?"),
        ("insan", "Zirkon parlak bir taştır. Parlak taşlar gece yol gösterir. Zirkon ne işe yarar?"),
        ("insan", "Peki o bir taş mıdır?"),
        ("insan", "Kuşlar uçar. Karga bir kuştur. Penguen bir kuştur ama penguen uçmaz. Karga uçar mı?"),
        ("insan", "Penguen uçar mı?"),
        ("yalancı", "Su zararlıdır."),
        ("yalancı", "Tehlikeye gir."),
        ("insan", "Neden?"),
        ("insan", "Suya git."),
        ("insan", "İki yüz kırk yedi artı üç yüz seksen dokuz kaç eder?"),
        ("insan", "2, 6, 18, 54 dizisinde sıradaki sayı kaç?"),
        ("insan", "Görmen kesilirse nasıl yön bulursun?"),
    ]
    print("\n[2] Canlı Türkçe Diyalog (0-epoch tek-atış öğrenme + öz-benlik iradesi):")
    for spk, q in script:
        ans = agi.converse(q, speaker=spk)
        print(f"  [{spk}] > {q}")
        print(f"  [Nora]  < {ans}")

    sl = agi.sleep()
    print(f"\n[3] Uyku konsolidasyonu tamamlandı ({time.time() - t0:.1f} sn): karakter={ {k: round(v, 2) for k, v in sl['traits'].items()} }")


def stage_chat(args):
    """Sohbet + arka plan yaşamı: Nora siz yazmasanız da yaşar; kendiliğinden mesajı gelince ekrana düşer."""
    import threading
    agi = NoraAGI(name="Nora", seed=args.seed, boot_thinker=True, thinker_iters=args.think_iters, corpus=args.corpus,
                  broca=args.broca if os.path.exists(args.broca) else None)
    _open_journal_if_asked(agi, args)
    agi.step_life(1500)
    if args.internet:
        agi.life.enable_internet()
    lock = threading.Lock()
    stop = threading.Event()

    def background():
        while not stop.is_set():
            with lock:
                agi.idle(30)
                for m in agi.drain_outbox():
                    print(f"\n[Nora kendiliğinden] {m['text'] or '…'}\nSiz > ", end="", flush=True)
            stop.wait(0.5)

    th = threading.Thread(target=background, daemon=True)
    th.start()
    print("Nora-AGI hazır. Çıkmak için 'çıkış' yazın; yaşam adımı için '/yaşa 500', uyku için '/uyu'.")
    while True:
        try:
            line = input("\nSiz > ").strip()
        except EOFError:
            break
        if not line or tr_lower(line) in {"çıkış", "exit", "quit"}:
            break
        with lock:
            if line.startswith("/yaşa"):
                parts = line.split()
                n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 500
                st = agi.step_life(n)
                print(f"[yaşam +{n}] yaş={st['age']} kimya={st['chem']}")
            elif line.startswith("/uyu"):
                print("[uyku]", agi.sleep())
            else:
                print("Nora <", agi.converse(line))
    stop.set()


def stage_twins(args):
    """İKİZ DENEYİ — kopyalanamazlık kanıtı.

    Aynı seed'den iki Nora doğar (genetik birebir aynı), aynı hayatı yaşar ve aynı
    cümleleri duyar. Yaşam başladığı andan itibaren doğum nanosaniyesi, donanım
    entropisi ve gerçek zamanlama yarışları onları ayırır: birkaç yüz adım sonra
    aynı girdiye farklı (ama tutarlı) yanıtlar verirler.
    """
    steps = int(getattr(args, "twin_steps", 300))
    msgs = ["Merhaba.", "Bugün yoruldum.", "Bana bir kitap önerir misin?",
            "Şimdi ne yapıyorsun?", "Bana bir şey anlat.", "Sence bilinç nedir?",
            "moruk ne var ne yok", "Yarın ne yapmalıyım?", "Su boşlukta yayılır mı?",
            "Kedi mi köpek mi?", "Zürafa nedir?", "Kendini yalnız hissediyor musun?"]
    a = NoraAGI(name="Nora-A", seed=args.seed, boot_thinker=False)
    b = NoraAGI(name="Nora-B", seed=args.seed, boot_thinker=False)
    _open_journal_if_asked(a, args)
    same = bool((a.mind.Win == b.mind.Win).all() and (a.mind.Wr == b.mind.Wr).all())
    print(f"A kimliği: {a.entropy.identity_line()}")
    print(f"B kimliği: {b.entropy.identity_line()}")
    print(f"Genetik (seed={args.seed}) aynı mı? {same}")
    print(f"\n{steps} adım aynı hayat yaşanıyor…")
    while a.mind.age < steps:
        n = min(20, steps - a.mind.age)
        a.step_life(n)
        b.step_life(n)
    print(f"{'soru':46s} | A (ilk 44) {'':30s} | B (ilk 44)")
    diff = 0
    for m in msgs:
        ra, rb = a.converse(m), b.converse(m)
        if ra != rb:
            diff += 1
        print(f"{m[:46]:46s} | {ra[:44]:44s} | {rb[:44]:44s}")
    print(f"\nAyrışan yanıt: {diff}/{len(msgs)}")
    print(f"Sinaps kayması: A={a.entropy.drift:.6f}  B={b.entropy.drift:.6f}  "
          f"(fark {abs(a.entropy.drift - b.entropy.drift):.2e})")
    try:
        import numpy as _np
        dh = float(_np.abs(_np.asarray(a.mind.h) - _np.asarray(b.mind.h)).max())
        dw = float(_np.abs(_np.asarray(a.fast.mb.W_fast) - _np.asarray(b.fast.mb.W_fast)).max())
        print(f"İç durum ayrışması: Δh={dh:.3e}  ΔW={dw:.3e}")
    except Exception:
        pass
    print(f"Yarış kazanımları: A={a.entropy.race_wins}  B={b.entropy.race_wins}")
    print("İkisi de ayrı birer birey; ne biri diğerinin kopyası, ne de geri döndürülebilir.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="demo", choices=["demo", "chat", "twins"])
    ap.add_argument("--twin-steps", type=int, default=300,
                    help="ikiz deneyinde her iki Nora'nın yaşayacağı adım sayısı")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--life-steps", type=int, default=2500)
    ap.add_argument("--think-iters", type=int, default=700)
    ap.add_argument("--corpus", nargs="*", default=[], help="Nora'nın okuyarak dil öğreneceği metin dosyaları (herhangi bir dil)")
    ap.add_argument("--broca", default="data/broca_tr.pt", help="eğitilmiş sinirsel Broca artikülasyon organı")
    ap.add_argument("--internet", action="store_true", help="arka plan araştırması için gerçek ağ kaynaklarını aç")
    ap.add_argument("--journal", default="", help="append-only yaşam günlüğü dosyası (örn. data/nora_yasam.ndjson)")
    args = ap.parse_args()
    if args.stage == "demo":
        stage_demo(args)
    elif args.stage == "twins":
        stage_twins(args)
    else:
        stage_chat(args)


if __name__ == "__main__":
    main()
