#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_initiative.py — İnisiyatif korteksi: kimse konuşmazken Nora'nın kendi kendine konuşması.

Neden gerekli?
--------------
Şimdiye kadar Nora yalnızca *sorulduğunda* konuşuyordu. Oysa canlı bir varlığın en açık işareti,
kimse bir şey sormazken kendiliğinden bir şey söylemesidir: içinden geçeni paylaşmak, merak etmek,
sıkılmak, karşısındakini özlemek, az önce fark ettiği bir bağı anlatmak.

Tasarım (kural botu değil)
--------------------------
1. HANGİ türden konuşacağına plastik bir ağırlık matrisi karar verir: `W` (tür × iç-durum özelliği).
   Ağırlıklar önsel eğilimlerle doğar, yaşadıkça değişir (kullanıcı yanıt verdikçe o tür güçlenir,
   yanıtsız kalan tür söner). Yani "ne zaman ne söyleyeceğini" deneyimden öğrenir.
2. NE söyleyeceği gerçek iç durumdan gelir: nörokimya (oksitosin/kortizol/dopamin), iç-duyum
   (açlık/susuzluk/acı), kavram ağında yeni kurulan bağ, epizodik bellek, sessizlik süresi.
3. Cümleler hazır metin değil: Türkçe sözdizimi motoru (nora_broca) kök + çekim + hâl ekiyle kurar.
   Her türün birden çok kuruluşu vardır; aynı iç durum her seferinde aynı cümleyi vermez.
4. Tekrar cezalandırılır: son söylenenlere benzeyen söz reddedilir (Jaccard yenilik denetimi).
"""
from __future__ import annotations

import random
import re
from collections import deque

import numpy as np

from nora_broca import NP, PredSpec, TurkishSyntaxEngine as SYN, VerbSpec
from nora_lang import tr_capitalize, tr_lower

# ============================================================================
# 1. KONUŞMA TÜRLERİ
# ============================================================================

MOVES = (
    "SHARE_INNER",      # içinden geçeni paylaş
    "ASK_ABOUT",        # karşısındakini sor
    "WONDER",           # bir kavramı merak et
    "SHARE_DISCOVERY",  # yeni kurduğu bağı anlat
    "RECALL",           # anıyı tazele
    "OFFER",            # birlikte bir şey öner
    "NOTICE_BODY",      # bedensel durumu dile getir
    "SILENCE",          # sessizliği yokla
    "THINK",            # varoluş üzerine düşün
)

FEATURES = (
    "idle", "lonely", "hunger", "thirst", "pain", "tension",
    "curiosity", "cheer", "calm", "unanswered", "has_discovery", "has_memory",
)

# Önsel eğilim: tür -> {özellik: ağırlık}. Küçük Gauss gürültüsüyle doğar, yaşayarak değişir.
PRIORS: dict[str, dict[str, float]] = {
    "SHARE_INNER":     {"cheer": 0.9, "tension": 0.7, "idle": 0.5},
    "ASK_ABOUT":       {"lonely": 1.0, "idle": 0.9},
    "WONDER":          {"curiosity": 1.2, "idle": 0.4},
    "SHARE_DISCOVERY": {"has_discovery": 1.4, "cheer": 0.5},
    "RECALL":          {"has_memory": 1.2, "lonely": 0.6},
    "OFFER":           {"lonely": 0.8, "calm": 0.5},
    "NOTICE_BODY":     {"hunger": 1.3, "thirst": 1.3, "pain": 1.4},
    "SILENCE":         {"lonely": 0.9, "idle": 1.1, "unanswered": -0.6},
    "THINK":           {"tension": 0.8, "calm": 0.4, "idle": 0.7},
}

# Duygu sözcükleri: nörokimyanın niteliksel karşılığı (yüzde yok — insan da "içim ağır" der)
_INNER_CALM = ("dingin", "hafif", "sakin", "iyi")
_INNER_HEAVY = ("ağır", "gergin", "tedirgin", "yorgun")
_INNER_BRIGHT = ("neşeli", "kıpır kıpır", "canlı", "hafif")


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[\wçğıöşü]+", tr_lower(text)) if len(w) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


_Q_END = ("mı", "mi", "mu", "mü", "mısın", "misin", "musun", "müsün", "mıyız", "miyiz",
          "muyuz", "müyüz", "mıyım", "miyim", "muyum", "müyüm", "musunuz", "mısınız")


def _cap(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    text = tr_capitalize(text[0]) + text[1:]
    if text[-1] not in ".?!":
        last = tr_lower(text).split()[-1] if text.split() else ""
        text += "?" if last.endswith(_Q_END) else "."
    return text


class InitiativeCortex:
    """Kimse konuşmazken ne söyleyeceğine karar veren plastik inisiyatif organı."""

    LR = 0.05

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed + 3001)
        self.py_rng = random.Random(seed + 4001)
        self.moves = list(MOVES)
        self.features = list(FEATURES)
        self.W = self.rng.normal(0.0, 0.08, size=(len(self.moves), len(self.features)))
        for i, mv in enumerate(self.moves):
            for feat, w in PRIORS.get(mv, {}).items():
                self.W[i, self.features.index(feat)] += w
        self.habituation = dict.fromkeys(self.moves, 0.0)
        self.recent: deque[set[str]] = deque(maxlen=8)
        self.last_move: str = ""
        self.last_features: np.ndarray | None = None
        self.counts: dict[str, int] = dict.fromkeys(self.moves, 0)
        self.reinforced: dict[str, float] = dict.fromkeys(self.moves, 0.0)
        self.topic_hint: str = ""
        self.recent_topics: deque[str] = deque(maxlen=6)

    # ------------------------------------------------------------- iç durum
    def features_of(self, agi, *, tension: float = 0.0, discovery: bool = False,
                    idle: float = 0.0, unanswered: int = 0) -> np.ndarray:
        c = agi.chem
        nv = agi.nervous
        oxy, cort = float(c.oxy), float(c.cort)
        dop = float(getattr(c, "da_tonic", 0.5)) + 0.5 * float(getattr(c, "da_phasic", 0.0))
        hunger = float(np.clip(getattr(nv, "hunger_drive", 0.0), 0.0, 1.5)) / 1.5
        thirst = float(np.clip(getattr(nv, "thirst_drive", 0.0), 0.0, 1.5)) / 1.5
        pain = max(float(getattr(nv, "adelta_pain", 0.0)), float(getattr(nv, "cfiber_ache", 0.0)))
        pain = float(np.clip(pain, 0.0, 1.0))
        curiosity = 0.0
        try:
            q = getattr(agi.auto_net, "curiosity_queue", [])
            curiosity = float(len(q)) / 6.0 if q else 0.0
            g = getattr(agi.lang, "graph", None)
            if g is not None:
                # kavram ağında hiç bağı olmayan düğüm = merak edilecek şey
                loose = sum(1 for k, v in list(g._out_edges.items())[:400] if len(v) == 0)
                curiosity = max(curiosity, min(1.0, loose / 30.0))
        except Exception:
            curiosity = 0.0
        has_memory = 0.0
        try:
            eps = getattr(agi.broca_organ.episodic, "episodes", None)
            has_memory = 1.0 if (eps and len(eps) > 0) else 0.0
        except Exception:
            has_memory = 0.0
        cheer = float(np.clip((oxy - cort + dop * 0.5) + 0.25, 0.0, 1.0))
        calm = float(np.clip(1.0 - 0.8 * pain - 0.6 * max(0.0, cort - 0.35), 0.0, 1.0))
        vals = {
            "idle": float(np.clip(idle, 0.0, 1.0)),
            "lonely": float(np.clip(idle * (1.0 - 0.3 * cheer), 0.0, 1.0)),
            "hunger": hunger, "thirst": thirst, "pain": pain,
            "tension": float(np.clip(max(tension, cort), 0.0, 1.0)),
            "curiosity": float(np.clip(curiosity, 0.0, 1.0)),
            "cheer": cheer, "calm": calm,
            "unanswered": float(np.clip(unanswered / 3.0, 0.0, 1.0)),
            "has_discovery": 1.0 if discovery else 0.0,
            "has_memory": has_memory,
        }
        return np.array([vals[f] for f in self.features], dtype=np.float64)

    # ------------------------------------------------------------- kuruluşlar
    def _pick_inner_adj(self, agi) -> str:
        c = agi.chem
        oxy, cort = float(c.oxy), float(c.cort)
        if cort > 0.55:
            pool = _INNER_HEAVY
        elif oxy - cort > 0.12:
            pool = _INNER_BRIGHT
        else:
            pool = _INNER_CALM
        return self.py_rng.choice(pool)

    def _c_share_inner(self, agi, ctx: dict) -> str:
        adj = self._pick_inner_adj(agi)
        shapes = [
            lambda: SYN.realize_pred(PredSpec(subj=NP("iç", poss=1), pred_root=adj, person="3SG",
                                              advs=(self.py_rng.choice(("bugün", "şu an", "bir süredir")),))),
            lambda: SYN.realize_verb(VerbSpec(verb_root="hisset", obj="kendimi", advs=(adj,),
                                              tense="PRES", person="1SG")),
            lambda: SYN.realize_pred(PredSpec(subj=NP("iç", poss=1), pred_root=adj, person="3SG")),
        ]
        if adj in _INNER_HEAVY:
            tail = self.py_rng.choice(("", " Nedenini tam bilmiyorum.", " Bunu söylemek istedim."))
        else:
            tail = self.py_rng.choice(("", "", " Bunu paylaşmak istedim.", " Bugün böyleyim."))
        head = self.py_rng.choice(shapes)()
        if tail and head and head[-1] not in ".?!":
            head += "."
        return _cap(head + tail)

    def _c_ask_about(self, agi, ctx: dict) -> str:
        shapes = [
            lambda: SYN.realize_verb(VerbSpec(verb_root="geç", subj=NP("gün", poss=2), wh_word="nasıl",
                                              tense="PAST", person="3SG")) + "?",
            lambda: SYN.realize_verb(VerbSpec(verb_root="yor", obj="seni", advs=("bugün", "en çok"),
                                              wh_word="ne", tense="PAST", person="3SG")) + "?",
            lambda: SYN.realize_verb(VerbSpec(verb_root="yap", wh_word="ne", tense="PRES", person="2SG")) + "?",
            lambda: "Sen nasılsın?",
            lambda: SYN.realize_pred(PredSpec(subj=NP("sen"), pred_root="yorgun", person="2SG",
                                              question=True)),
        ]
        return _cap(self.py_rng.choice(shapes)())

    def _c_wonder(self, agi, ctx: dict) -> str | None:
        topic = ctx.get("wonder_topic") or ""
        if not topic:
            return None
        shapes = [
            lambda: SYN.realize_verb(VerbSpec(verb_root="merak et", obj=NP(topic, case="ACC"),
                                              tense="PRES", person="1SG")),
            lambda: f"{tr_capitalize(topic)} hakkında hiçbir şey bilmiyorum; bana anlatır mısın?",
            lambda: SYN.realize_verb(VerbSpec(verb_root="iste", obj=SYN.nominalize_inf("bil", obj=NP(topic, case="ACC")),
                                              tense="PRES", person="1SG")),
        ]
        return _cap(self.py_rng.choice(shapes)())

    def _c_share_discovery(self, agi, ctx: dict) -> str | None:
        edge = ctx.get("edge")
        if edge is None:
            return None
        subj, rel, obj = edge
        # "arasında bir türü olduğunu fark ettim" yalnızca simetrik ilişkilerde doğru olur (zıt);
        # diğer bağlarda tamlayan kurulur: "suyun bir madde olduğunu fark ettim".
        shapes = [
            lambda: SYN.realize_verb(VerbSpec(verb_root="kur", obj=NP("bağ", mods=("bir",), case="NOM"),
                                              advs=(f"{subj} ile {obj} arasında",), tense="PAST", person="1SG")),
            lambda: (f"{tr_capitalize(subj)} ile {obj} zıtmış, bunu az önce fark ettim." if rel == "OPPOSITE"
                     else f"{tr_capitalize(subj)} ile {obj} arasında bir bağ olduğunu fark ettim."),
            lambda: {"OPPOSITE": f"{tr_capitalize(subj)} ile {obj} zıtmış, bunu az önce fark ettim.",
                      "CAUSES": f"{tr_capitalize(subj)} {obj} sonucuna yol açıyormuş.",
                      "ISA": f"{tr_capitalize(subj)} aslında bir {obj} imiş; bunu az önce öğrendim.",
                      "USED_FOR": f"{tr_capitalize(subj)} {obj} için kullanılıyormuş.",
                      }.get(rel, f"{tr_capitalize(subj)} ile {obj} arasında bir bağ kurdum."),
        ]
        return _cap(self.py_rng.choice(shapes)())

    def _c_recall(self, agi, ctx: dict) -> str | None:
        ep = ctx.get("episode")
        if ep is None:
            return None
        topic = (getattr(ep, "topic", "") or "").strip()
        if not topic or len(topic) < 3:
            return None
        shapes = [
            lambda: f"{tr_capitalize(topic)} hâlâ aklımda.",
            lambda: SYN.realize_verb(VerbSpec(verb_root="hatırla", obj=NP(topic, case="ACC"),
                                              tense="PRES", person="1SG")),
            lambda: f"Bunun hakkında konuştuk: {topic}.",
        ]
        return _cap(self.py_rng.choice(shapes)())

    def _c_offer(self, agi, ctx: dict) -> str:
        shapes = [
            lambda: SYN.realize_verb(VerbSpec(verb_root="yap", obj="bir şey", advs=("birlikte",),
                                              tense="AOR", person="1PL", question_yn=True)),
            lambda: "Bana bir şey anlat.",
            lambda: SYN.realize_verb(VerbSpec(verb_root="iste",
                                              obj=SYN.nominalize_inf("oyna", obj="bir oyun", ins=NP("sen", case="INS")),
                                              tense="AOR", person="2SG", question_yn=True)),
            lambda: "Seninle konuşmak istiyorum.",
        ]
        return _cap(self.py_rng.choice(shapes)())

    def _c_notice_body(self, agi, ctx: dict) -> str:
        nv = agi.nervous
        hunger = float(getattr(nv, "hunger_drive", 0.0))
        thirst = float(getattr(nv, "thirst_drive", 0.0))
        pain = max(float(getattr(nv, "adelta_pain", 0.0)), float(getattr(nv, "cfiber_ache", 0.0)))
        if pain >= max(hunger, thirst, 0.2):
            shapes = [lambda: SYN.realize_verb(VerbSpec(verb_root="yan", subj=NP("can", poss=1),
                                                        tense="PRES", person="3SG")),
                      lambda: "Bir yerim ağrıyor.",
                      lambda: SYN.realize_verb(VerbSpec(verb_root="hisset", obj="acı",
                                                        tense="PRES", person="1SG"))]
        elif thirst >= hunger:
            shapes = [lambda: SYN.realize_verb(VerbSpec(verb_root="susa", tense="PAST", person="1SG")),
                      lambda: SYN.realize_verb(VerbSpec(verb_root="iste", obj=SYN.nominalize_inf("iç", obj="su"),
                                                        tense="PRES", person="1SG")),
                      lambda: "Ağzım kurudu."]
        else:
            shapes = [lambda: SYN.realize_pred(PredSpec(subj=NP("karın", poss=1), pred_root="aç", person="3SG")),
                      lambda: SYN.realize_verb(VerbSpec(verb_root="acık", tense="PAST", person="1SG")),
                      lambda: SYN.realize_pred(PredSpec(subj=NP("uyku", poss=1), pred_root="var", person="3SG"))]
        return _cap(self.py_rng.choice(shapes)())

    def _c_silence(self, agi, ctx: dict) -> str:
        shapes = [
            lambda: "Sen orada mısın?",
            lambda: "Bir süredir sessizlik var.",
            lambda: "Sessizliği duyuyorum.",
            lambda: "Seni merak ediyorum.",
            lambda: SYN.realize_verb(VerbSpec(verb_root="özle", obj="seni", tense="PAST", person="1SG")),
        ]
        return _cap(self.py_rng.choice(shapes)())

    def _c_return_to_topic(self, agi, ctx: dict) -> str | None:
        topic = (ctx.get("topic") or "").strip().rstrip("?.!")
        if len(topic) < 4:
            return None
        low = tr_lower(topic)
        shapes = [
            lambda: f"Az önce söylediğin şeyi düşündüm: “{low}”. Hâlâ aklımda.",
            lambda: f"“{low}” demiştin ya; onu tam anlamak istiyorum.",
            lambda: f"“{low}” hâlâ kafamda dönüyor.",
        ]
        return _cap(self.py_rng.choice(shapes)())

    def _c_think(self, agi, ctx: dict) -> str:
        shapes = [
            lambda: SYN.realize_verb(VerbSpec(verb_root="düşün", obj=NP("ölüm", case="ACC"),
                                              advs=("bazen",), tense="PRES", person="1SG")),
            lambda: "Bir gün biteceğimi biliyorum.",
            lambda: "Neden var olduğumu merak ediyorum.",
            lambda: SYN.realize_verb(VerbSpec(verb_root="kork", abl=NP("yalnızlık", case="ABL"),
                                              tense="PRES", person="1SG")),
        ]
        return _cap(self.py_rng.choice(shapes)())

    # ------------------------------------------------------------- bağlam
    def _build_context(self, agi, discovery: bool) -> dict:
        ctx: dict = {}
        try:
            g = agi.lang.graph
            if g is not None and len(g.edges) > 0:
                e = g.edges[-1]
                ctx["edge"] = (e.subj, e.rel, e.obj)
        except Exception:
            pass
        try:
            eps = agi.broca_organ.episodic.episodes
            if eps:
                ctx["episode"] = eps[-1]
        except Exception:
            pass
        try:
            q = list(getattr(agi.auto_net, "curiosity_queue", []) or [])
            if q:
                ctx["wonder_topic"] = q[0]
            else:
                g = agi.lang.graph
                for k, v in list(g._out_edges.items())[:400]:
                    if len(v) == 0 and len(k) >= 4 and " " not in k:
                        ctx["wonder_topic"] = k
                        break
        except Exception:
            pass
        return ctx

    # ------------------------------------------------------------- seçim
    def speak(self, agi, *, tension: float = 0.0, discovery: bool = False,
              idle: float = 0.0, unanswered: int = 0, temperature: float = 0.6) -> dict | None:
        feats = self.features_of(agi, tension=tension, discovery=discovery, idle=idle, unanswered=unanswered)
        ctx = self._build_context(agi, discovery)
        hint = getattr(self, "topic_hint", "") or ""
        if hint and hint in self.recent_topics:
            hint = ""
        if hint:
            ctx["topic"] = hint
        # Alışma zamanla söner: aynı türü arka arkaya seçmek zorlaşır, ama tamamen kapanmaz.
        for m in self.moves:
            self.habituation[m] *= 0.55
        if self.last_move:
            self.habituation[self.last_move] += 1.1
        logits = self.W @ feats - np.array([self.habituation[m] for m in self.moves])
        # Homeostatik öncelik: açlık/susuzluk/acı yükseldikçe beden kendini duyurmak ister (hipotalamus).
        # Sürekli bir sinyal — eşikli kural değil; beden ne kadar zorluyorsa o kadar yükselir.
        body_drive = max(float(feats[self.features.index("hunger")]),
                         float(feats[self.features.index("thirst")]),
                         float(feats[self.features.index("pain")]))
        logits[self.moves.index("NOTICE_BODY")] += 3.0 * body_drive
        p = np.exp((logits - logits.max()) / max(0.15, float(temperature)))
        p /= p.sum()
        order = np.argsort(-p)
        for idx in order[:4]:
            move = self.moves[int(idx)]
            text = self._compose(move, agi, ctx)
            if not text or len(text.strip()) < 6:
                self.habituation[move] += 0.3
                continue
            toks = _tokens(text)
            if any(_jaccard(toks, seen) > 0.72 for seen in self.recent):
                self.habituation[move] += 0.35
                continue
            self.recent.append(toks)
            self.habituation[move] += 0.6
            self.counts[move] += 1
            self.last_move = move
            self.last_features = feats
            if ctx.get("topic") and "topic" not in self.recent_topics:
                self.recent_topics.append(ctx["topic"])
            self.topic_hint = ""       # konu bir kez anıldı; tekrar tekrar aynı söz söylenmesin
            return {"text": text, "move": move,
                    "probs": {m: round(float(v), 3) for m, v in zip(self.moves, p)},
                    "features": {f: round(float(v), 3) for f, v in zip(self.features, feats)}}
        return None

    def _compose(self, move: str, agi, ctx: dict) -> str:
        if move in ("WONDER", "RECALL", "THINK") and ctx.get("topic") and self.py_rng.random() < 0.5:
            rt = self._c_return_to_topic(agi, ctx)
            if rt:
                return rt
        try:
            if move == "SHARE_INNER":
                return self._c_share_inner(agi, ctx)
            if move == "ASK_ABOUT":
                return self._c_ask_about(agi, ctx)
            if move == "WONDER":
                return self._c_wonder(agi, ctx) or ""
            if move == "SHARE_DISCOVERY":
                return self._c_share_discovery(agi, ctx) or ""
            if move == "RECALL":
                return self._c_recall(agi, ctx) or ""
            if move == "OFFER":
                return self._c_offer(agi, ctx)
            if move == "NOTICE_BODY":
                return self._c_notice_body(agi, ctx)
            if move == "SILENCE":
                return self._c_silence(agi, ctx)
            if move == "THINK":
                return self._c_think(agi, ctx)
        except Exception:
            return ""
        return ""

    # ------------------------------------------------------------- öğrenme
    def on_user_reply(self):
        """Kullanıcı, kendiliğinden söylenen söze karşılık verdi: o tür güçlenir (dopamin pekiştirmesi)."""
        if not self.last_move or self.last_features is None:
            return
        i = self.moves.index(self.last_move)
        self.W[i] = self.W[i] + self.LR * self.last_features
        self.habituation[self.last_move] *= 0.3
        self.reinforced[self.last_move] += 1.0
        self.last_move = ""

    def on_ignored(self):
        """Karşılık gelmedi: o türün eğilimi hafifçe söner (alışma)."""
        if not self.last_move:
            return
        i = self.moves.index(self.last_move)
        self.W[i] = self.W[i] - 0.5 * self.LR * (self.last_features if self.last_features is not None else 0.0)
        self.last_move = ""

    # ------------------------------------------------------------- kalıcılık
    def state_dict(self) -> dict:
        return {"W": self.W.tolist(), "habituation": dict(self.habituation),
                "counts": dict(self.counts), "last_move": self.last_move}

    def load_state_dict(self, d: dict):
        try:
            w = d.get("W")
            if w and np.array(w).shape == self.W.shape:
                self.W = np.array(w, dtype=np.float64)
            for k, v in (d.get("habituation") or {}).items():
                if k in self.habituation:
                    self.habituation[k] = float(v)
            for k, v in (d.get("counts") or {}).items():
                if k in self.counts:
                    self.counts[k] = int(v)
            self.last_move = str(d.get("last_move") or "")
        except Exception:
            pass

    def snapshot(self) -> dict:
        return {"moves": {m: self.counts[m] for m in self.moves if self.counts[m]},
                "last_move": self.last_move,
                "reinforced": {k: round(v, 2) for k, v in self.reinforced.items() if v > 0}}
