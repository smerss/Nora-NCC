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
from nora_broca import GenerativeBrocaOrgan, NP, VerbSpec
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

    def extremum(self, prop: str, maximum: bool = True) -> str | None:
        p = tr_lower(prop).strip()
        edges = self.orders.get(p, [])
        if not edges:
            return None
        nodes = {g for g, _ in edges} | {l for _, l in edges}
        if maximum:
            cands = [x for x in nodes if not any(l == x for _, l in edges)]
        else:
            cands = [x for x in nodes if not any(g == x for g, _ in edges)]
        return cands[0] if len(cands) == 1 else (cands[0] if cands else None)


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

        # --- ARKA PLAN YAŞAMI + ÖĞRENEN SÖZ ORGANI (elle yazılmış sözcük/cümle yok; dil okunarak öğrenilir) ---
        self.life = BackgroundLife(seed=seed + 71)
        # Kimse konuşmazken konuşan organ: hangi hâlde ne söyleyeceğini yaşayarak öğrenir
        self.discourse = DiscourseCortex(seed=seed + 137, lexicon=self.lang.graph.lexicon)
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
            # Varsayılan Kip Ağı (Default Mode Network): yaşam sırasında sessiz içsel monolog!
            self.monologue.spontaneous_tick(self)
            if self.life.enabled and self.mind.age % self.life.stride == 0:
                self.life.tick(self)
            # ÖZ-SINAV: her ~120 yaşam adımında bir kendini sına (üret -> çöz -> karşılaştır).
            # Başarısız madde varsa bu bir öğrenme sinyalidir, ceza değil: sadece kaydedilir.
            if self.mind.age - self._exam_age_mark >= 120:
                self._exam_age_mark = self.mind.age
                self.self_exam(factor=0.6)

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
        ans = self._converse_core(text, speaker)
        if self.acc is not None and raw:
            try:
                self.acc.note_turn(raw, ans, now)
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
                    if not _quiet and "sessiz" not in _low:
                        _rng = getattr(self, "rng", None) or getattr(self, "random", None)
                        _note = compose_noticing(_meta, self, _rng)
                        ans = f"{ans.rstrip()} {_note}"
                        if self.dialogue_log and self.dialogue_log[-1][1] == raw:
                            self.dialogue_log[-1] = (self.dialogue_log[-1][0], raw, ans)
            except Exception:
                pass
        return ans

    def _converse_core(self, text: str, speaker: str = "insan") -> str:
        """Gelen Türkçe cümleyi algılar, önce içsel monolog yürütür, öz-benlik ve nörokimya süzgecinden geçirip yanıt üretir."""
        t0 = time.perf_counter()
        if not self.mortality.is_alive:
            return "[Sinir sistemi durdu: Nora'nın tekil yaşamı sona erdiği için artık duymuyor ve konuşamıyor.]"
        raw = text.strip()
        if not raw:
            return "..."
        # --- TÜRKÇE EDİNİMİ: duyulan bilinmeyen fiil biçimleri, kendi çekim üreticisiyle
        #     yeniden üretilebiliyorsa kök olarak öğrenilir (kalıcı, sözlüğe yazılır).
        try:
            _new_roots = TurkishMorphology.observe_verb_usage(raw, self.lang.graph.lexicon)
            if _new_roots:
                self.learned_verb_roots.update(_new_roots)
        except Exception:
            pass
        self._remember_focus(raw)
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

        _clock_q = any(k in _tl for k in ("saat kaç", "saat kaçtır", "saat kaç oldu"))
        _elapsed_q = any(k in _tl for k in (
            "ne kadar oldu", "ne kadar zaman oldu", "ne zamandır", "ne kadar süredir",
            "kaç dakika oldu", "kaç saat oldu", "ne kadar süre"))
        if self.acc is not None and (_clock_q or _elapsed_q):
            if _clock_q and not _elapsed_q:
                _ans_t = compose_time_answer(0.0, kind="clock", rng=getattr(self.discourse, "random", None))
            else:
                _ans_t = compose_time_answer(self.acc.since_session(), len(self.dialogue_log),
                                             kind="elapsed", steps=int(self.mind.age),
                                             rng=getattr(self.discourse, "random", None))
            self._log_turn(speaker, raw, _ans_t, t0)
            return _ans_t

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
            stance = "sana güveniyorum" if tr_val >= 0.55 else ("sana karşı temkinliyim" if tr_val >= 0.30 else "sana güvenmiyorum")
            bids.append(CorticalBid(
                "social_tom", 0.97, 0.96,
                f"Şu an {speaker} için epistemik güven düzeyim {tr_val:.2f}; geçmiş etkileşimlerimize dayanarak {stance}.",
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
                # Bu cümle zaten Nora'nın kendi mantık motorunda yapısal kural olarak kodlandı;
                # kaba sözcük-dizisi kenarı üretmek kavram ağını kirletir, onun yerine öz-etiketini bildirir.
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
                    # Gerekçe: az önce kurulan önermenin kendisi. Çekirdek: öğrenme eylemi Broca'da çekimlenir.
                    _bind = SYN.realize_verb(VerbSpec(
                        verb_root="bağla", obj=NP("bilgi", poss=3, case="ACC"),
                        dat=NP("kavram ağı", poss=1, case="DAT"), tense="PAST", person="1SG"))
                    _rep_fb = self.schema_cortex.transform_clauses(
                        _bind, _prop_txt,
                        SYN.converb("evir", kind="ip", obj=NP("bilgi", case="ACC", mods=("yeni",)))
                        + " " + SYN.converb("çevir", kind="ip", loc=NP("iç", poss=1, case="LOC")), cur_sc)
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
                            _miss = f"Bunu henüz bilmiyorum: “{raw.strip()}” — bana bir cümleyle öğretir misin?"
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
        return winner.reply

    # ------------------------------------------------------------------ epistemik + kod kanalları
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
            _stance, schema_id, _m, _q = self.schema_cortex.select_stance_and_schema(
                words, is_question, trust, in_pain)
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

        return self.schema_cortex.transform_clauses(
            f"kabul ediyorum ve '{tgt_label}' doğrultusunda yöneliyorum",
            "isteğin zihinsel simülasyonumda içsel dengemle uyumlu çıktı",
            SYN.nominalize_dik("kontrol et", poss=1, case="LOC", obj=NP("simülasyon", poss=1, case="ACC")),
            cur_sc,
        )

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

        m_ext = re.match(r"^en\s+([\wçğıöşü]+)\s+(?:olan\s+)?(?:hangisi|hangisidir|nedir|kimdir)$", s_clean)
        if m_ext:
            prop = morph.analyze_word(m_ext.group(1), lex)["stem"]
            top = self.comp_graph.extremum(prop, maximum=True)
            if top is None and prop in antonym_axis:
                top = self.comp_graph.extremum(antonym_axis[prop], maximum=False)
            if top:
                return f"Öğrendiğim sıralamaya göre en {prop} olan {morph.inflect_noun(top, 'NOM', copula=True)}."

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
            "verb_roots": sorted(set(TurkishMorphology.LEARNED_VERB_ROOTS) | set(self.learned_verb_roots)),
        }
        with open(path_prefix + ".agi.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        np.savez_compressed(
            path_prefix + ".agi_mb.npz",
            fast_Wf=self.fast.mb.W_fast,
            fast_Ws=self.fast.mb.W_slow,
            fast_stiff=self.fast.mb.stiff,
            fast_ci=self.fast.mb.claw_idx,
            fast_cs=self.fast.mb.claw_sign,
        )
        self.mind.save(path_prefix + ".mind.npz")
        self.life.voice.save_file(path_prefix + ".voice.pkl.gz")
        if self.life.voice.organ is not None:
            self.life.voice.organ.save(path_prefix + ".broca.pt")

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
            if getattr(agi, "frames", None) is not None:
                try:
                    agi.frames.memory.load_snapshot(meta.get("sentence_frames") or {})
                except Exception:
                    pass
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
        return agi


def stage_demo(args):
    t0 = time.time()
    print("=" * 78)
    print("  NORA-AGI CANLI GÖSTERİM: Beden + Nörokimya + Öz-Benlik + Tek-Atış Dil + S2")
    print("=" * 78)
    agi = NoraAGI(name="Nora", seed=args.seed, boot_thinker=True, thinker_iters=args.think_iters)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="demo", choices=["demo", "chat"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--life-steps", type=int, default=2500)
    ap.add_argument("--think-iters", type=int, default=700)
    ap.add_argument("--corpus", nargs="*", default=[], help="Nora'nın okuyarak dil öğreneceği metin dosyaları (herhangi bir dil)")
    ap.add_argument("--broca", default="data/broca_tr.pt", help="eğitilmiş sinirsel Broca artikülasyon organı")
    ap.add_argument("--internet", action="store_true", help="arka plan araştırması için gerçek ağ kaynaklarını aç")
    args = ap.parse_args()
    if args.stage == "demo":
        stage_demo(args)
    else:
        stage_chat(args)


if __name__ == "__main__":
    main()
