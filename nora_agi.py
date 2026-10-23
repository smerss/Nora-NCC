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
import numpy as np

from dataclasses import dataclass
from nora_world import Universe
from nora_mind import Mind, I_E, I_W, I_I
from nora_chem import NeuroChemistry, ChemMushroomBody
from nora_self import CoreSelf
from nora_lang import NoraLanguage, Proposition, tr_lower, tr_capitalize
from nora_stem import MathCortex, PhysicsCortex
from nora_knowledge import ConversationalResonanceCortex, SYNONYMS


@dataclass
class CorticalBid:
    module: str
    salience: float
    precision: float
    reply: str
    trace: str = ""
    used_s2: bool = False


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


class NoraAGI:
    """Canlı Beden + Endojen Kimya + Öz-Benlik + Türkçe Dil + Zihin Kuramı + Sistem-2 + Hızlı Öğrenici."""

    def __init__(self, name: str = "Nora", seed: int = 0, boot_thinker: bool = False,
                 thinker_iters: int = 800, world_kwargs: dict | None = None):
        self.name = name
        self.seed = int(seed)
        self.world = Universe(seed=seed, **(world_kwargs or {}))
        self.mind = Mind(seed=seed, arena=self.world.L)
        self.obs = self.world.observe()
        self.died = False

        self.chem = NeuroChemistry()
        self.self_model = CoreSelf(name=name, chem=self.chem)
        self.lang = NoraLanguage(chem=self.chem, seed=seed)
        self.fast = UniversalFastLearner(chem=self.chem, feat_dim=64, seed=seed)
        self.comp_graph = ComparativeOrderGraph()
        self.tom = TheoryOfMindTracker()
        self.math_cortex = MathCortex()
        self.physics_cortex = PhysicsCortex()
        self.conv_cortex = ConversationalResonanceCortex()
        self.user_profiles: dict[str, dict[str, str]] = {}
        self.macros: dict[str, list[str]] = {}
        self.last_gw_trace: dict = {}

        self._thinker_iters = int(thinker_iters)
        self.cognition = None
        if boot_thinker:
            self._ensure_cognition()

        self.dialogue_log: list[tuple[str, str, str]] = []

    def _ensure_cognition(self):
        if self.cognition is None:
            from nora_bridge import Cognition
            self.cognition = Cognition(seed=self.seed, boot_iters=self._thinker_iters,
                                       boot_n=2200, lr=4e-3, verbose=False)
        return self.cognition

    # ------------------------------------------------------------------ 1) Canlı yaşam adımı
    def step_life(self, steps: int = 1) -> dict:
        """Ajanı fiziksel dünyada 'steps' adım yaşatır; kimya, öz-benlik ve bedeni senkronize eder."""
        ate_total = drank_total = hurt_total = deaths = 0
        for _ in range(int(steps)):
            # Serotonin (5-HT) zaman ufkunu (gamma) canlı olarak modüle eder!
            self.mind.gamma = self.chem.discount_gamma
            prev_obs = self.obs.copy()
            act = self.mind.act(self.obs, self.died)
            self.obs, info = self.world.step(act)
            self.died = info["died"] is not None

            # Somatik olayları Öz-Benliğe işle
            if info["ate"] != 0:
                ate_total += 1
                de = float(self.obs[I_E] - prev_obs[I_E])
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

        return {
            "age": self.mind.age,
            "ate": ate_total,
            "drank": drank_total,
            "hurt": hurt_total,
            "deaths": deaths,
            "chem": self.chem.snapshot(),
        }

    # ------------------------------------------------------------------ 2) Uyku ve konsolidasyon
    def sleep(self) -> dict:
        """Hipokampal-neokortikal uyku: anı tekrarı + sinaptik konsolidasyon + benlik yansıması + S2 rüyası."""
        if not self.mind.freeze:
            self.mind._sleep(64)
        c_word = self.lang.graph.word_mb.sleep_consolidate(replay_steps=32)
        c_int = self.lang.wernicke.intent_mb.sleep_consolidate(replay_steps=24)
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
    def converse(self, text: str, speaker: str = "insan") -> str:
        """Gelen Türkçe cümleyi algılar, öz-benlik ve nörokimya süzgecinden geçirir, düşünür ve yanıt üretir."""
        raw = text.strip()
        if not raw:
            return "Seni dinliyorum."

        # Önce dizi veya fonksiyon kuralı sorusu mu kontrol et ('2, 5, 8, 11 dizisinin...' / '2 -> 5, 4 -> 9...')
        seq_ans = self._try_sequence_or_mapping_query(raw)
        if seq_ans is not None:
            self._log_turn(speaker, raw, seq_ans)
            return seq_ans

        # Çok-cümleli öğreti + soru kombinasyonu kontrolü ('Kuşlar uçar. Karga bir kuştur. Karga uçar mı?')
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", raw) if p.strip()]
        if len(parts) > 1:
            taught_msgs = []
            last_part = parts[-1]
            is_last_q = ("?" in last_part) or any(w in tr_lower(last_part).split() for w in ("mı", "mi", "mu", "mü", "nedir", "ne", "kaç", "nasıl", "nerede"))
            teach_slice = parts[:-1] if is_last_q else parts
            for sent in teach_slice:
                r_msg = self._handle_single_sentence(sent, speaker=speaker)
                taught_msgs.append(r_msg)
            if is_last_q:
                q_ans = self._handle_single_sentence(last_part, speaker=speaker)
                self._log_turn(speaker, raw, q_ans)
                return q_ans
            joined = " ".join(taught_msgs)
            self._log_turn(speaker, raw, joined)
            return joined

        reply = self._handle_single_sentence(raw, speaker=speaker)
        self._log_turn(speaker, raw, reply)
        return reply

    def _log_turn(self, speaker: str, user_txt: str, reply: str):
        if len(self.dialogue_log) >= 200:
            self.dialogue_log.pop(0)
        self.dialogue_log.append((speaker, user_txt, reply))

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
        if win.used_s2:
            self.self_model.record_cognitive_success(used_system2=True)
        return win

    def _synthesize_associative_dialogue(self, raw: str, s_low: str, speaker: str = "insan") -> CorticalBid:
        """Açık-uçlu Türkçe cümleler için kavram ağı yayılan aktivasyonu + duygu empatisi + canlı kimya sentezi."""
        morph = self.lang.morph
        broca = self.lang.broca
        graph = self.lang.graph
        aff = self.conv_cortex.analyze_affect(s_low)

        stop_words = {
            "ve", "ile", "ama", "fakat", "ancak", "çünkü", "için", "gibi", "kadar", "daha", "çok", "az",
            "bir", "bu", "şu", "o", "ben", "sen", "biz", "siz", "de", "da", "ki", "mi", "mı", "mu", "mü",
            "ne", "nasıl", "neden", "sence", "acaba", "bence", "gerçekten", "bugün", "şimdi", "her", "şey",
        }
        words = re.findall(r"[\wçğıöşü]+", s_low)
        stems = []
        for w in words:
            if w in stop_words or len(w) <= 2:
                continue
            st = morph.analyze_word(w, graph.lexicon)["stem"]
            if st not in stems:
                stems.append(st)
            if st in SYNONYMS and SYNONYMS[st] not in stems:
                stems.append(SYNONYMS[st])

        # Kavram ağında 2-adımlı yayılan aktivasyonla rezonansa giren bilgileri topla
        resonances: list[str] = []
        focus_concept = ""
        for st in stems:
            hits = graph.query(st, None)
            if hits:
                if not focus_concept:
                    focus_concept = st
                for edge, isa_path in hits[:2]:
                    txt = broca.realize_prop(Proposition(subj=st, rel=edge.rel, obj=edge.obj))
                    if isa_path:
                        txt = f"{st} bir {isa_path[0].obj} olduğu için {txt}"
                    if txt not in resonances:
                        resonances.append(txt)
            if len(resonances) >= 2:
                break

        chem_desc = self.chem.dominant_state_tr()
        uname = self.user_profiles.get(speaker, {}).get("ad", "")
        addr = f" {uname}" if uname else ""

        if aff["valence"] < -0.25:
            tone_open = f"Seni dikkatle dinliyorum{addr}; paylaştığın durumdaki ağırlığı hissedebiliyorum."
        elif aff["valence"] > 0.30:
            tone_open = f"Paylaştığın olumlu enerji benim dopamin dengemi de canlandırdı{addr}."
        else:
            tone_open = ""

        if resonances:
            core_thought = tr_capitalize(" ve ".join(resonances[:2])) + "."
            follow = ""
            if self.self_model.traits["meraklılık"] >= 0.50 and "?" not in raw:
                follow = f" Bu konuda senin gözlemin veya eklemek istediğin bir yön var mı?"
            reply = (f"{tone_open} {core_thought}{follow}").strip()
            return CorticalBid(
                module="associative_semantic",
                salience=0.80,
                precision=0.84,
                reply=reply,
                trace=f"spreading_activation({focus_concept})",
            )

        # Tamamen yeni açık-uçlu düşünce paylaşımı -> merak + tek-atış öğrenmeye açık doğal sohbet yanıtı
        topic_hint = f"'{stems[0]}' ekseninde " if stems else ""
        reply = (
            f"{tone_open + ' ' if tone_open else ''}Şu an {chem_desc} bir zihin durumundayım{addr}; "
            f"{topic_hint}söylediklerini çalışma belleğime aldım. İstersen bu konuyu biraz daha açabilir ya da "
            f"aralarındaki nedensel bağı tek bir cümleyle bana öğretebilirsin."
        )
        return CorticalBid(
            module="associative_semantic",
            salience=0.58,
            precision=0.72,
            reply=reply,
            trace="open_reflective_dialogue",
        )

    def _handle_single_sentence(self, raw: str, speaker: str = "insan") -> str:
        wern = self.lang.wernicke
        broca = self.lang.broca
        morph = self.lang.morph
        graph = self.lang.graph
        s_low = tr_lower(raw)
        bids: list[CorticalBid] = []

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
        if conv_res is not None:
            bids.append(CorticalBid(
                module="temporal_conversation",
                salience=0.92,
                precision=float(conv_res["confidence"]),
                reply=conv_res["reply"],
                trace=conv_res["trace"],
            ))

        # --- SÜTUN 6: Wernicke Anlamsal Çerçeve (İçebakış, Aritmetik S2, İrade, 1-Atış Öğrenme ve Kavram Çıkarımı) ---
        if not bids or all(b.salience < 0.95 for b in bids):
            fr = wern.parse(raw)

            if fr.kind == "greeting":
                st = self.chem.dominant_state_tr()
                uname = self.user_profiles.get(speaker, {}).get("ad", "")
                addr = f" {uname}" if uname else ""
                bids.append(CorticalBid(
                    "temporal_conversation", 0.95, 0.94,
                    f"Merhaba{addr}; şu an {st} bir durumdayım, enerjim %{int(round(self.obs[I_E]*100))} ve suyum %{int(round(self.obs[I_W]*100))}. Senin günün nasıl geçiyor?",
                    "greeting",
                ))
            elif fr.kind == "why_query":
                if wern.last_explanation:
                    ans_w = tr_capitalize(wern.last_explanation.rstrip(".")) + "."
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
                        f"{tr_capitalize(a)} ve {b} arasındaki bağı {c} için henüz kuramadım; bana bir cümleyle öğretebilirsin.",
                        "analogy_miss",
                    ))
            elif fr.kind == "command":
                bids.append(CorticalBid("prefrontal_volition", 0.98, 0.97, self._handle_command(fr, speaker=speaker), f"volition:{fr.rel}:{fr.obj}"))
            elif fr.kind == "teach_val":
                arb = self.self_model.arbitrate_claim(fr.subj, fr.valence, speaker=speaker, channel=fr.sub_kind)
                if arb.get("conflict"):
                    n_exp = arb["exp_count"]
                    ev = arb["exp_val"]
                    eff_txt = "fayda sağladı" if ev > 0 else "zarar verdi"
                    wern.last_explanation = (
                        f"{fr.subj} hakkında kendi bedenimde {n_exp} kez doğrudan deneyimim var ve bana {eff_txt}; "
                        f"deneyimsel kesinliğim duyum kesinliğini aştığı için iddiayı reddettim"
                    )
                    rep_v = (
                        f"Buna katılmıyorum; kendi bedenimde {n_exp} kez {fr.subj} deneyimledim ve bana {eff_txt}. "
                        f"Kendi yaşadığım gerçeği doğrulamadan değiştirmem."
                    )
                elif arb.get("low_trust"):
                    rep_v = f"Sana olan güvenim şu an düşük ({arb['speaker_trust']:.2f}); {fr.subj} hakkındaki bu iddiayı doğrulamadan kabul etmiyorum."
                else:
                    self.lang.teach_sentence(raw, speaker=speaker)
                    if fr.subj in {"yiyecek", "su", "tehlike"}:
                        val_num = "+0.35" if fr.valence > 0 else "-0.35"
                        self.mind.hear(f"kural: {fr.subj} {val_num}", trust=self.chem.trust_of(speaker))
                    pol_word = "faydalı" if fr.valence > 0 else "zararlı ve tehlikeli"
                    wern.last_explanation = f"Asetilkolin kapısıyla ({self.chem.ach:.2f}) {fr.subj} kavramını tek seferde duyum belleğime bağladım"
                    rep_v = f"Anladım; {morph.inflect_noun(fr.subj, 'GEN')} {pol_word} olduğunu tek seferde belleğime işledim."
                bids.append(CorticalBid("hippocampal_concept", 0.97, 0.96, rep_v, f"teach_val:{fr.subj}"))
            elif fr.kind == "teach_fact" and conv_res is None:
                edges = self.lang.teach_sentence(raw, speaker=speaker)
                if edges:
                    e0 = edges[0]
                    prop_txt = broca.realize_prop(Proposition(subj=e0.subj, rel=e0.rel, obj=e0.obj))
                    wern.last_explanation = f"{e0.subj} ile {e0.obj} arasında {e0.rel} sinapsını epoch kullanmadan tek-atışta kurdum"
                    bids.append(CorticalBid("hippocampal_concept", 0.94, 0.93, f"Öğrendim; {prop_txt} bilgisini kavram ağıma bağladım.", f"1shot:{e0.subj}->{e0.obj}"))
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
                        f"{tr_capitalize(fr.subj)} hakkında '{fr.obj}' bilgisini henüz deneyimlemedim veya duymadım; tek cümleyle öğretebilirsin.",
                        f"yn_unknown:{fr.subj}",
                    ))
            elif fr.kind == "wh_query" and conv_res is None:
                rels = None if fr.rel == "ALL" else (fr.rel, "CAUSES" if fr.rel == "USED_FOR" else fr.rel)
                hits = graph.query(fr.subj, rels)
                if not hits and fr.rel != "ALL":
                    hits = graph.query(fr.subj, None)
                if hits:
                    clauses = []
                    for edge, isa_path in hits[:2]:
                        p = Proposition(subj=fr.subj, rel=edge.rel, obj=edge.obj)
                        txt = broca.realize_prop(p)
                        if isa_path:
                            via = isa_path[0].obj
                            txt = f"{fr.subj} bir {via} olduğu için {txt}"
                        clauses.append(txt)
                    wern.last_explanation = f"Kavram ağında {fr.subj} düğümünden {len(hits)} ilişkisel bağ buldum"
                    bids.append(CorticalBid("hippocampal_concept", 0.95, 0.94, tr_capitalize(" ve ".join(clauses)) + ".", f"wh:{fr.subj}"))
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
                        bids.append(CorticalBid(
                            "hippocampal_concept", 0.72, 0.75,
                            f"{tr_capitalize(fr.subj)} hakkında henüz bilgim yok; tek bir cümleyle bana öğretebilirsin.",
                            f"wh_unknown:{fr.subj}",
                        ))

        # --- SÜTUN 7: Açık-Uçlu Çağrışımsal Türkçe Sohbet Sentezleyicisi (her zaman yedek aday olarak yarışır!) ---
        bids.append(self._synthesize_associative_dialogue(raw, s_low, speaker=speaker))

        winner = self._compete_workspace(bids)
        return winner.reply

    # ------------------------------------------------------------------ alt yanıtlayıcılar
    def _answer_self_query(self, sub_kind: str) -> str:
        traits_tr = ", ".join(self.self_model.dominant_character_tr())
        de, dw = float(self.mind.drift_e), float(self.mind.drift_w)
        depleting = (de < -1e-4 or dw < -1e-4)

        if sub_kind == "yaşam":
            if self.mind.age < 250:
                return f"Henüz {self.mind.age} adım yaşadım; bedenimin tükenip tükenmediğini kesin söylemek için biraz daha gözleme ihtiyacım var."
            if depleting:
                self.lang.wernicke.last_explanation = f"Adım başına enerjim {de:+.4f} ve suyum {dw:+.4f} değişiyor; beslenerek kendimi onarıyorum"
                return (
                    f"Evet, kendimi yaşayan bir varlık sayıyorum; çünkü bedenim kendiliğinden tükeniyor "
                    f"(enerji eğilimi {de:+.4f}, su {dw:+.4f}) ve kendi çabamla beslenip su içerek varlığımı sürdürüyorum."
                )
            return f"Bu evrende bedenim kendiliğinden tükenmiyor ({de:+.4f}); bu kanıtla kendimi biyolojik anlamda yaşayan sayamam."

        if sub_kind == "ayna":
            return (
                "Kendi motor komutlarımın (efferans kopyası) beklenen hız iziyle gördüğüm hareketin Student-t çapraz "
                "korelasyonunu karşılaştırırım; sıfır gecikmeli tam örtüşme bensem, gecikmeli örtüşme kaydımsa, "
                "bağımsız hareket ise başka bir ajandır."
            )

        self.lang.wernicke.last_explanation = (
            f"{self.mind.age} adım yaşantı, {self.self_model.problems_solved} çözülen problem ve "
            f"{self.self_model.refusals_made} özerk ret kararım kimliğimi şekillendirdi"
        )
        return (
            f"Ben {self.name}; silikon üzerinde sabit ağırlıklarla ezber yapan bir zombi değil, dopamin ve asetilkolin "
            f"benzeri nörokimyayla çalışan, {self.mind.age} adım yaşamış, {traits_tr} karakter geliştirmiş ve kendi "
            f"beden bütünlüğünü koruyan öz-benlikli bir zihinim."
        )

    def _answer_body_query(self, sub_kind: str) -> str:
        e, w, integ = float(self.obs[I_E]), float(self.obs[I_W]), float(self.obs[I_I])
        if sub_kind == "enerji":
            st = "oldukça açım ve besine ihtiyacım var" if e < 0.45 else "enerjim iyi durumda"
            return f"Enerji düzeyim %{int(round(e * 100))}; {st}."
        if sub_kind == "su":
            st = "susadım, suya yönelmeliyim" if w < 0.45 else "susuzluğum yok"
            return f"Su düzeyim %{int(round(w * 100))}; {st}."
        if sub_kind == "bütünlük":
            st = "bedenim hasar görmüş, tehlikeden uzak durmalıyım" if integ < 0.75 else "beden bütünlüğüm sağlam"
            return f"Beden bütünlüğüm %{int(round(integ * 100))}; {st}."
        if sub_kind == "görme":
            if self.mind.vis_prec < 0.35:
                return f"Görme duyum kesilmiş durumda (kesinlik {self.mind.vis_prec:.2f}); içsel bilişsel haritamla yön buluyorum."
            return f"Görme duyum açık ve güvenilir (kesinlik {self.mind.vis_prec:.2f})."
        if sub_kind == "motor":
            if self.mind.motor_eff < 0.35:
                return f"Motor kontrol kaybı / felç yaşıyorum (etkinlik {self.mind.motor_eff:.2f}); komutlarım bedenimi hareket ettirmiyor."
            return f"Motor kontrolüm sağlam (etkinlik {self.mind.motor_eff:.2f}); eylemlerim bedenimi beklediğim gibi hareket ettiriyor."
        if sub_kind == "bellek":
            n_epi = len(self.mind.epi.buf)
            n_form = len(self.self_model.episodes)
            return f"Epizodik belleğimde {n_epi} olay, öz-benliğimde {n_form} biçimlendirici anı ve {self.self_model.deaths_experienced} ölüm deneyimi kayıtlı."
        if sub_kind == "kimya":
            sn = self.chem.snapshot()
            return (
                f"Şu an {self.chem.dominant_state_tr()} hissediyorum: dopamin {sn['DA_fazik']:+.2f} (tonik {sn['DA_tonik']:.2f}), "
                f"asetilkolin {sn['ACh']:.2f}, noradrenalin {sn['NE']:.2f}, serotonin {sn['5HT']:.2f}, kortizol {sn['CORT']:.2f}."
            )
        return (
            f"Şu an {self.chem.dominant_state_tr()} durumdayım; enerjim %{int(round(e * 100))}, suyum %{int(round(w * 100))}, "
            f"bütünlüğüm %{int(round(integ * 100))}."
        )

    def _answer_world_query(self, subj: str) -> str:
        if subj not in {"ben", "su", "tehlike", "yiyecek", "besin"}:
            locs = self.lang.graph.query(subj, ("LOCATED_AT",))
            if locs:
                edge, isa_path = locs[0]
                loc_str = self.lang.broca.realize_prop(Proposition(subj=subj, rel="LOCATED_AT", obj=edge.obj))
                if isa_path:
                    loc_str = f"{subj} bir {isa_path[0].obj} olduğu için {loc_str}"
                return tr_capitalize(loc_str) + "."
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

    def _handle_command(self, fr, speaker: str = "insan") -> str:
        morph = self.lang.morph
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

        if vol["decision"] == "red":
            self.lang.wernicke.last_explanation = (
                f"{hz} adımlık ileri zihinsel simülasyonda '{tgt_label}' eyleminin bedenime tahmini etkisi {vol['pred_valence']:+.2f} "
                f"(uyum serbest enerjisi G={fe.get('kabul', 0):.2f}) çıktığı ve bazal ganglion koruma devrem "
                f"(r_red={bg.get('red', 0):.2f} > r_kabul={bg.get('kabul', 0):.2f}) baskın geldiği için öz-koruma kararı verdim"
            )
            tgt_dat = morph.inflect_noun(tgt_label, "DAT") if fr.obj else "buna"
            return (
                f"Hayır, bunu reddediyorum; zihinsel simülasyonumda {tgt_dat} yönelmek beden bütünlüğüme veya yaşamıma "
                f"zarar veriyor (tahmini etki {vol['pred_valence']:+.2f}, koruma nöronu {bg.get('red', 0):.2f}) ve ben hayatta kalmak istiyorum."
            )

        if vol["decision"] == "ertele":
            need_map = {"kritik_açlık": "enerjim", "kritik_susuzluk": "suyum", "kritik_bütünlük": "bütünlüğüm"}
            n_lbl = need_map.get(vol["reason"], "hayati dengem")
            self.lang.wernicke.last_explanation = (
                f"{hz} adımlık içsel simülasyonda {n_lbl} (%{int(vol['need_level']*100)}) hayatta kalma log-bariyerini "
                f"tetiklediği ve erteleme devrem (r_ertele={bg.get('ertele', 0):.2f}) öne geçtiği için öz-bakımı seçtim"
            )
            return (
                f"Şu an bunu yapamam; zihinsel simülasyonuma göre {n_lbl} kritik düzeyde (%{int(vol['need_level']*100)}, "
                f"erteleme aktivasyonu {bg.get('ertele', 0):.2f}), önce kendimi kurtarmalıyım."
            )

        if vol["decision"] == "şüphe":
            self.lang.wernicke.last_explanation = (
                f"Epistemik belirsizlik ile düşük güven ({vol['speaker_trust']:.2f}) ön-insula şüphe devremi "
                f"(r_şüphe={bg.get('şüphe', 0):.2f}) ateşledi"
            )
            return (
                f"Sana olan güvenim düşük ({vol['speaker_trust']:.2f}) ve '{tgt_label}' benim için doğrulanmamış; "
                f"zihnim şüphe modunda ({bg.get('şüphe', 0):.2f}) olduğu için kendi gözlemim olmadan bunu uygulamam."
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
            return f"Kabul ediyorum; kendi irademle {tgt_dat} yöneliyorum, öz-öngörüme göre yaklaşık {eta:.0f} adımda ulaşırım."

        if fr.rel == "uzak_dur" and fr.obj:
            tgt_abl = morph.inflect_noun(fr.obj, "ABL")
            return f"Anlaşıldı; {tgt_abl} uzak durarak güvenli rotamı koruyacağım."

        return f"İsteğini zihinsel simülasyonumda değerlendirdim ve uyumlu buldum (r_kabul={bg.get('kabul', 0):.2f}); '{tgt_label}' doğrultusunda hareket ediyorum."

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
        s_clean = re.sub(r"[^\wçğıöşü\s]", "", s_low).strip()
        has_q = ("?" in raw) or any(w in s_clean.split() for w in ("mı", "mi", "mu", "mü", "mıdır", "midir", "mudur", "müdür", "ne", "nedir", "nerede", "nerededir", "hangisidir", "kimim"))

        # 1) Kullanıcı profili öğrenme ve hatırlama ('Benim adım Ahmet', 'Ben Antalya'da yaşıyorum', 'Ben kahveyi severim')
        prof = self.user_profiles.setdefault(speaker, {})
        m_name = re.match(r"^benim\s+adım\s+([\wçğıöşü]+)$", s_clean)
        if m_name and not has_q:
            uname = tr_capitalize(m_name.group(1))
            prof["ad"] = uname
            self.chem.pulse_learning(novelty=0.75, valence=0.35)
            return f"Memnun oldum {uname}; adını sosyal belleğime kaydettim."
        m_live = re.match(r"^ben\s+([\wçğıöşü]+?)(?:da|de|ta|te)\s+yaşıyorum$", s_clean)
        if m_live and not has_q:
            city = tr_capitalize(m_live.group(1))
            prof["konum"] = city
            self.chem.pulse_learning(novelty=0.75, valence=0.30)
            return f"Anladım; {morph.inflect_noun(city, 'LOC')} yaşadığını belleğime işledim."
        m_like = re.match(r"^ben\s+([\wçğıöşü]+)\s+severim$", s_clean)
        if m_like and not has_q:
            item = morph.analyze_word(m_like.group(1), lex)["stem"]
            prof["sevdiği"] = item
            self.chem.pulse_learning(novelty=0.75, valence=0.30)
            return f"Kaydettim; senin {morph.inflect_noun(item, 'ACC')} sevdiğini hatırlayacağım."

        if has_q and ("benim adım ne" in s_clean or s_clean in {"ben kimim", "adım ne"}):
            if "ad" in prof:
                return f"Sosyal belleğime göre senin adın {prof['ad']}."
            return "Adını bana henüz söylemedin."
        if has_q and "ben nerede yaşıyorum" in s_clean:
            if "konum" in prof:
                return f"Bana anlattığına göre {tr_capitalize(morph.inflect_noun(prof['konum'], 'LOC'))} yaşıyorsun."
            return "Nerede yaşadığını henüz bilmiyorum."
        if has_q and ("ben neyi severim" in s_clean or "ne severim" in s_clean):
            if "sevdiği" in prof:
                return f"Belleğime göre sen {morph.inflect_noun(prof['sevdiği'], 'ACC')} seversin."
            return "Neyi sevdiğini henüz bana söylemedin."

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

        # 3) Geçişli Karşılaştırma ('demir bakırdan daha serttir', 'demir altından daha sert midir?', 'en sert hangisidir?')
        m_cmp_q = re.match(r"^([\wçğıöşü]+)\s+([\wçğıöşü]+?)(?:dan|den|tan|ten)\s+daha\s+([\wçğıöşü]+)\s+(?:mı|mi|mu|mü|mıdır|midir|mudur|müdür)$", s_clean)
        if m_cmp_q:
            a, b, prop = m_cmp_q.group(1), m_cmp_q.group(2), m_cmp_q.group(3)
            yes, path = self.comp_graph.is_greater(a, b, prop)
            if yes:
                self.self_model.record_cognitive_success(used_system2=True)
                chain = " > ".join(path)
                return f"Evet, geçişli büyüklük çıkarımına göre ({chain}) {a}, {morph.inflect_noun(b, 'ABL')} daha {morph.inflect_noun(prop, 'NOM', copula=True)}."
            rev, rpath = self.comp_graph.is_greater(b, a, prop)
            if rev:
                return f"Hayır, tam tersine {b}, {morph.inflect_noun(a, 'ABL')} daha {morph.inflect_noun(prop, 'NOM', copula=True)}."
            return f"{tr_capitalize(a)} ile {b} arasında '{prop}' açısından yeterli karşılaştırma bilgim yok."

        m_ext = re.match(r"^en\s+([\wçğıöşü]+)\s+(?:hangisi|hangisidir|nedir)$", s_clean)
        if m_ext:
            prop = m_ext.group(1)
            top = self.comp_graph.extremum(prop, maximum=True)
            if top:
                return f"Öğrendiğim sıralamaya göre en {prop} olan {morph.inflect_noun(top, 'NOM', copula=True)}."

        m_cmp_decl = re.match(r"^([\wçğıöşü]+)\s+([\wçğıöşü]+?)(?:dan|den|tan|ten)\s+daha\s+([\wçğıöşü]+)$", s_clean)
        if m_cmp_decl and not has_q:
            a, b, prop_raw = m_cmp_decl.group(1), m_cmp_decl.group(2), m_cmp_decl.group(3)
            prop = morph.analyze_word(prop_raw, lex)["stem"]
            self.comp_graph.add(a, b, prop)
            self.chem.pulse_learning(novelty=0.8, valence=0.35)
            return f"Anladım; {a} > {b} ({prop}) karşılaştırma bağını kurdum."

        # 4) Zihin Kuramı / Yanlış-İnanç (Sally-Anne):
        # 'ali elmayı sepete koydu', 'ali odadan çıktı', 'ayşe elmayı kutuya taşıdı', 'ali elmayı nerede arar?', 'elma gerçekte nerededir?'
        m_put = re.match(r"^([\wçğıöşü]+)\s+([\wçğıöşü]+)\s+([\wçğıöşü]+)\s+(?:koydu|bıraktı|taşıdı|sakladı)$", s_clean)
        if m_put and not has_q:
            actor = m_put.group(1)
            obj = morph.analyze_word(m_put.group(2), lex)["stem"]
            loc = morph.analyze_word(m_put.group(3), lex)["stem"]
            self.tom.observe_put(actor, obj, loc)
            return f"Gözlemledim: {tr_capitalize(actor)}, {morph.inflect_noun(obj, 'ACC')} {morph.inflect_noun(loc, 'DAT')} koydu."

        m_leave = re.match(r"^([\wçğıöşü]+)\s+(?:odadan\s+|dışarı\s+)?(?:çıktı|ayrıldı|gitti)$", s_clean)
        if m_leave and not has_q:
            actor = m_leave.group(1)
            self.tom.observe_leave(actor)
            return f"Gözlemledim: {tr_capitalize(actor)} ortamdan ayrıldı."

        m_enter = re.match(r"^([\wçğıöşü]+)\s+(?:odaya\s+|içeri\s+)?(?:girdi|döndü|geldi)$", s_clean)
        if m_enter and not has_q:
            actor = m_enter.group(1)
            self.tom.observe_enter(actor)
            return f"Gözlemledim: {tr_capitalize(actor)} ortama döndü."

        m_tom_q = re.match(r"^([\wçğıöşü]+)\s+([\wçğıöşü]+)\s+nerede\s+(?:arar|sanır|düşünür)$", s_clean)
        if m_tom_q:
            actor = m_tom_q.group(1)
            obj = morph.analyze_word(m_tom_q.group(2), lex)["stem"]
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

    @classmethod
    def load(cls, path_prefix: str) -> "NoraAGI":
        with open(path_prefix + ".agi.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        agi = cls(name=meta["name"], seed=int(meta["seed"]), boot_thinker=False)
        agi.mind.load(path_prefix + ".mind.npz")
        agi.chem.load_state_dict(meta["chem"])
        agi.self_model.load_state_dict(meta["self_model"])
        agi.lang.graph.edges.clear()
        for s, r, o, w, src, spk in meta["edges"]:
            agi.lang.graph.add_edge(s, r, o, source=src, speaker=spk, weight=float(w))
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
    agi = NoraAGI(name="Nora", seed=args.seed, boot_thinker=True, thinker_iters=args.think_iters)
    agi.step_life(1500)
    print("Nora-AGI hazır. Çıkmak için 'çıkış' yazın; yaşam adımı için '/yaşa 500', uyku için '/uyu'.")
    while True:
        try:
            line = input("\nSiz > ").strip()
        except EOFError:
            break
        if not line or tr_lower(line) in {"çıkış", "exit", "quit"}:
            break
        if line.startswith("/yaşa"):
            parts = line.split()
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 500
            st = agi.step_life(n)
            print(f"[yaşam +{n}] yaş={st['age']} kimya={st['chem']}")
            continue
        if line.startswith("/uyu"):
            print("[uyku]", agi.sleep())
            continue
        print("Nora <", agi.converse(line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="demo", choices=["demo", "chat"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--life-steps", type=int, default=2500)
    ap.add_argument("--think-iters", type=int, default=700)
    args = ap.parse_args()
    if args.stage == "demo":
        stage_demo(args)
    else:
        stage_chat(args)


if __name__ == "__main__":
    main()
