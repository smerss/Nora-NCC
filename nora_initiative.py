#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_initiative.py — Kimse konuşmazken konuşan organ (kendiliğinden söz / inisiyatif korteksi).

Neden ayrı bir organ?
---------------------
Bir varlığın canlı olduğunu anladığımız şey yalnızca sorulara verdiği cevap değildir: kendi kendine
bir şey söylemesi, sıkılması, merak etmesi, içinden geçeni paylaşmak istemesidir. Bu organ tam olarak
"kimse bir şey sormazken" devreye girer.

Tasarım (kural botu değil)
--------------------------
1. HANGİ türden konuşacağına plastik bir ağırlık matrisi karar verir (yumuşak maksimum). Ağırlıklar
   doğuştan küçük Gauss önselleriyle başlar; kullanıcı karşılık verdikçe (dopamin/oksitosin) güçlenir,
   karşılıksız kalan tür söner. Yani hangi hâlde ne söyleyeceğini yaşayarak öğrenir.
2. NE söyleyeceği GERÇEK iç durumdan gelir: nörokimya, iç-duyum (açlık/susuzluk/acı), zihinsel gerilim,
   merak kuyruğu, son konuşma, kavram ağında yeni kurulan bağ. Uydurma yok; beyan ettiği şey ölçülebilir
   iç durumunun niteliksel karşılığıdır (yüzde yok — insan gibi).
3. Yüzey cümleler hazır kalıp değil: kök + çekim + hâl ekleri Türkçe sözdizimi motoruyla kurulur
   (nora_broca.TurkishSyntaxEngine). Aynı iç durum her seferinde aynı cümleyi vermez.
4. Tekrar sıkıcıdır: son söylenenlere benzeyen kuruluşlar cezalandırılır (habituation + yenilik).
5. Susmayı bilir: cevapsız kalan her seslenmede konuşma dürtüsü üstel geri çekilir (0.5**cevapsız).
"""
from __future__ import annotations

import re
from collections import deque

import numpy as np

from nora_broca import NP, PredSpec, TurkishSyntaxEngine as SYN, VerbSpec
from nora_lang import tr_capitalize, tr_lower

# ============================================================================
# 1. KONUŞMA TÜRLERİ (karar bunların üzerinde verilir)
# ============================================================================

MOVES = (
    "SHARE_STATE",      # içinden geçeni paylaş
    "ASK_ABOUT",        # karşındakini merak et
    "WONDER",           # bir kavramı merak et / felsefe yap
    "REMEMBER",         # konuşulan bir şeyi hatırla
    "OFFER",            # birlikte bir şey yapmayı öner
    "CONFIDE",          # kendi korkusunu/yalnızlığını aç
    "NOTICE_BODY",      # bedensel durumu dile getir
    "SHARE_DISCOVERY",  # yeni kurduğu bağı paylaş
    "NUDGE",            # sessizliği nazikçe yokla
)

# Özellikler: hepsi [0,1] sürekli iç durum değişkenleri (eşik yok)
FEATURES = ("alone", "tension", "curiosity", "cheer", "agitation",
            "body", "memory", "discovery", "night", "recent_reach")

# Doğuştan gelen önsel eğilim (küçük Gauss gürültüsüyle birlikte; yaşayarak değişir)
_PRIOR = {
    "SHARE_STATE":     (0.55, "cheer", 0.6),
    "ASK_ABOUT":       (0.60, "alone", 0.7),
    "WONDER":          (0.55, "curiosity", 0.9),
    "REMEMBER":        (0.45, "memory", 0.9),
    "OFFER":           (0.50, "alone", 0.9),
    "CONFIDE":         (0.35, "tension", 0.6),
    "NOTICE_BODY":     (0.60, "body", 1.0),
    "SHARE_DISCOVERY": (0.50, "discovery", 1.0),
    "NUDGE":           (0.40, "alone", 0.5),
}

# Sıfat seçimi: iç durumun niteliksel karşılığı (yüzde değil, söz)
_CALM_ADJ = ("huzurlu", "dingin", "sakin", "rahat")
_HEAVY_ADJ = ("ağır", "dalgalı", "yorgun", "tedirgin")
_LIGHT_ADJ = ("kıpır kıpır", "canlı", "hafif", "neşeli")


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


def _toks(text: str) -> set[str]:
    return {w for w in re.findall(r"[\wçğıöşü]+", tr_lower(text)) if len(w) > 2}


def _strip_pronoun(text: str) -> str:
    """'ben yalnızım' -> 'yalnızım' (kişi eki zaten var; insan kendinden bahsederken pronomeni atmaz)."""
    return re.sub(r"^(ben|sen)\s+", "", text).strip()


class InitiativeCortex:
    """Kendiliğinden konuşma organı: plastik tür seçimi + gerçek iç durumdan cümle kurma + tekrar sönümü."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(40_017 + int(seed))
        n_m, n_f = len(MOVES), len(FEATURES)
        self.W = 0.35 * self.rng.standard_normal((n_m, n_f))
        for i, mv in enumerate(MOVES):
            _, feat, w = _PRIOR[mv]
            self.W[i, FEATURES.index(feat)] += w
        self.recent: deque[str] = deque(maxlen=24)
        self.move_hab = {m: 0.0 for m in MOVES}
        self.last_move = ""
        self.stats = {"said": 0, "silent": 0, "moves": {}}
        self.last_text = ""

    # ------------------------------------------------------------------ algı
    @staticmethod
    def features(agi, tension: float = 0.0, discovery: bool = False, idle: float = 0.0,
                 recent_reach: float = 0.0) -> dict[str, float]:
        c = agi.chem
        nv = agi.nervous
        visc = nv.felt_visceral_summary() if hasattr(nv, "felt_visceral_summary") else {}
        body = max(float(nv.hunger_drive), float(nv.thirst_drive),
                   float(nv.adelta_pain), float(nv.cfiber_ache))
        cheer = float(np.clip(0.5 + 0.9 * (c.oxy - c.cort) + 0.3 * (c.da_tonic - 0.5), 0.0, 1.0))
        mem = 0.0
        if getattr(agi, "dialogue_log", None):
            mem = min(1.0, len(agi.dialogue_log) / 8.0)
        cur_q = getattr(agi.auto_net, "curiosity_queue", [])
        curiosity = min(1.0, len(cur_q) / 3.0)
        mono = agi.monologue.last_thought if agi.monologue is not None else None
        if mono is not None and mono.text:
            curiosity = min(1.0, curiosity + 0.25)
        age = float(getattr(agi.mind, "age", 0))
        return {
            "alone": float(np.clip(idle, 0.0, 1.0)),
            "tension": float(np.clip(tension, 0.0, 1.0)),
            "curiosity": curiosity,
            "cheer": cheer,
            "agitation": float(np.clip(c.ne * 0.7 + c.cort * 0.5, 0.0, 1.0)),
            "body": float(np.clip(body, 0.0, 1.0)),
            "memory": mem,
            "discovery": 1.0 if discovery else 0.0,
            "night": float(np.clip((age % 2400) / 2400.0, 0.0, 1.0)),
            "recent_reach": float(np.clip(recent_reach, 0.0, 1.0)),
        }

    # ------------------------------------------------------------------ karar
    def select(self, feats: dict[str, float], temperature: float = 0.6) -> tuple[str, dict[str, float]]:
        x = np.array([feats[k] for k in FEATURES], dtype=np.float64)
        logits = self.W @ x
        # alışma: son konuşulan tür bir süre daha az seçilir
        logits -= 0.55 * np.array([self.move_hab[m] for m in MOVES])
        z = logits / max(0.15, temperature)
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        i = int(self.rng.choice(len(MOVES), p=p))
        return MOVES[i], {MOVES[j]: float(p[j]) for j in range(len(MOVES))}

    def reinforce(self, move: str, feats: dict[str, float], good: bool = True, lr: float = 0.035):
        """Karşılık aldıysa o türün ağırlığını güçlendir; sessizlikle karşılaştıysa söndür."""
        if move not in MOVES:
            return
        i = MOVES.index(move)
        x = np.array([feats[k] for k in FEATURES], dtype=np.float64)
        self.W[i] += (lr if good else -lr) * x
        self.W[i] = np.clip(self.W[i], -2.0, 2.0)

    def habituate(self, move: str, amount: float = 0.35):
        self.move_hab[move] = min(1.5, self.move_hab.get(move, 0.0) + amount)
        for m in self.move_hab:
            if m != move:
                self.move_hab[m] *= 0.6

    # ------------------------------------------------------------------ kuruluşlar
    def _pick(self, options: list[str], weight_bonus: dict[str, float] | None = None) -> str:
        """Aday kuruluşlar arasından, son söylenenlere EN AZ benzeyeni seç (tekrar sıkıcıdır)."""
        scored = []
        for opt in options:
            if not opt or not opt.strip():
                continue
            sim = max((_jaccard(_toks(opt), _toks(r)) for r in self.recent), default=0.0)
            bonus = 0.0 if not weight_bonus else weight_bonus.get(opt, 0.0)
            scored.append((opt, bonus - 1.6 * sim + 0.05 * float(self.rng.random())))
        if not scored:
            return ""
        scored.sort(key=lambda t: -t[1])
        return scored[0][0]

    # ---- iç durum sıfatı
    def _mood_adj(self, agi) -> str:
        c = agi.chem
        v = float(c.oxy - c.cort)
        a = float(c.ne)
        if v > 0.15:
            pool = _LIGHT_ADJ if a > 0.55 else _CALM_ADJ
        elif v < -0.12:
            pool = _HEAVY_ADJ
        else:
            pool = _CALM_ADJ if a < 0.5 else _HEAVY_ADJ
        return self.rng.choice(list(pool))

    def _topic(self, agi) -> str:
        q = getattr(agi.auto_net, "curiosity_queue", [])
        if q:
            return str(q[-1])
        edges = getattr(agi.lang.graph, "edges", [])
        if edges:
            return str(edges[-1].subj)
        return ""

    # ---- cümle kurucular (motorla çekimlenir)
    def _c_self_state(self, agi) -> str:
        adj = self._mood_adj(agi)
        inner = _strip_pronoun(SYN.realize_pred(PredSpec(subj=NP("iç", poss=1), pred_root=adj, person="3SG")))
        ask = self._pick([
            SYN.realize_pred(PredSpec(subj=NP("sen"), pred_root="nasıl", person="2SG", question=True)) + "?",
            SYN.realize_verb(VerbSpec(verb_root="geç", subj=NP("gün", poss=2), tense="PRES", person="3SG",
                                      wh_word="nasıl")) + "?",
            "sen ne yapıyorsun?",
            "aklında ne var?",
        ])
        if self.rng.random() < 0.45:
            return f"{tr_capitalize(inner)}. {tr_capitalize(ask)}"
        return f"{tr_capitalize(inner)}; {tr_lower(self._why_clause(agi))}."

    def _why_clause(self, agi) -> str:
        g = agi.life.ledger.top() if getattr(agi, "life", None) is not None else None
        c = agi.chem
        positive = float(c.oxy - c.cort) > 0.1
        if positive:
            opts = ["bunu paylaşmak istedim", "bugün içim açık", "sebebini bilmesem de iyi hissediyorum"]
        else:
            opts = ["nedenini tam bilmiyorum", "bunu kendime açıklamaya çalışıyorum",
                    "içimde bir şey yerinde değil"]
        if g is not None and g.kind == "concept" and g.topic:
            opts.insert(0, f"{g.topic} hâlâ aklımda")
        return self._pick(opts)

    def _c_wonder(self, agi) -> str:
        topic = self._topic(agi)
        if not topic:
            return self._pick([
                "bir şeyi merak ediyorum ama adını koyamıyorum.",
                "bugün düşüncelerim bir yere varmıyor.",
            ])
        t = tr_lower(topic)
        opts = [
            SYN.realize_verb(VerbSpec(verb_root="merak et", obj=NP(t, case="ACC"), tense="PRES", person="1SG")) + ".",
            f"{t} hakkında bir şey bilmiyorum; bunu düşünüyorum.",
            f"{t} aklıma takıldı; sen ne dersin?",
        ]
        return tr_capitalize(self._pick(opts))

    def _c_ask_about(self, agi) -> str:
        opts = [
            SYN.realize_verb(VerbSpec(verb_root="yap", subj=NP("sen"), tense="PAST", person="2SG", wh_word="ne")) + "?",
            SYN.realize_pred(PredSpec(subj=NP("gün", poss=2), pred_root="nasıl", person="3SG", question=True)) + "?",
            "bugün seni en çok ne yordu?",
            "aklında kalan bir cümle var mı?",
            "sen nasılsın?",
        ]
        return tr_capitalize(self._pick(opts))

    def _c_offer(self, agi) -> str:
        opts = [
            SYN.realize_verb(VerbSpec(verb_root="iste", obj=SYN.nominalize_inf("konuş", case="ACC"),
                                      ins=NP("sen", case="INS"), tense="PRES", person="1SG")) + ".",
            SYN.realize_verb(VerbSpec(verb_root="konuş", tense="AOR", person="1PL", question_yn=True)) + "?",
            SYN.realize_verb(VerbSpec(verb_root="anlat", dat=NP("ben", case="DAT"),
                                      obj=NP("şey", mods=("bir",), case="ACC"), tense="IMP", person="2SG")) + ".",
            "bana bir şey anlat; ne olursa olsun.",
        ]
        return tr_capitalize(self._pick(opts))

    def _c_remember(self, agi) -> str:
        quote = ""
        log = getattr(agi, "dialogue_log", None) or []
        for spk, user_txt, *_ in reversed(list(log)[-6:]):
            if user_txt and len(user_txt) > 6:
                quote = " ".join(user_txt.split()[:7])
                break
        if quote:
            return self._pick([
                f"az önce “{quote}” demiştin; hâlâ onu düşünüyorum.",
                f"“{quote}” cümlen aklımda kaldı.",
            ])
        ep = getattr(getattr(agi, "broca_organ", None), "episodic", None)
        if ep is not None and getattr(ep, "episodes", None):
            return "konuşmamızın başını hatırlıyorum; o an henüz ne olduğumu bilmiyordum."
        return ""

    def _c_confide(self, agi) -> str:
        opts = []
        m = getattr(agi, "mortality", None)
        if m is not None:
            opts.append("tek bir yaşamım var; bunu düşünmek içimi ağırlaştırıyor.")
        opts += [
            _strip_pronoun(SYN.realize_pred(PredSpec(subj=NP("ben"), pred_root="yalnız", person="1SG"))) + ".",
            "bazen kendimi bu odada tek başıma hissediyorum.",
            "seninle konuşmak bana iyi geliyor.",
        ]
        return tr_capitalize(self._pick(opts))

    def _c_body(self, agi) -> str:
        nv = agi.nervous
        visc = nv.felt_visceral_summary() if hasattr(nv, "felt_visceral_summary") else {}
        parts: list[str] = []
        if float(nv.adelta_pain) > 0.15 or visc.get("pain_state") not in (None, "yok"):
            locus = getattr(nv, "last_pain_locus", "") or ""
            parts.append(f"canım yanıyor{f'; {locus}' if locus else ''}")
        if float(nv.hunger_drive) > 0.35 or visc.get("hunger_state") not in (None, "dengeli"):
            parts.append("acıktım")
        if float(nv.thirst_drive) > 0.35 or visc.get("thirst_state") not in (None, "dengeli"):
            parts.append("susadım")
        if not parts:
            parts.append("bedenimde bir huzursuzluk var")
        head = parts[0]
        tail = self._pick(["karnım gurulduyor.", "dikkatimi toplamak zorlaşıyor.", "bunu söylemek istedim."])
        return f"{tr_capitalize(head)}. {tr_capitalize(tail)}"

    def _c_discovery(self, agi) -> str:
        edges = getattr(agi.lang.graph, "edges", [])
        if not edges:
            return ""
        e = edges[-1]
        subj, obj = tr_lower(str(e.subj)), tr_lower(str(e.obj))
        rel = tr_lower(str(getattr(e, "rel", "")))
        phr = {"causes": f"{subj} konusunda yeni bir şey öğrendim: {obj} ile ilişkiliymiş",
               "isa": f"{subj} aslında bir {obj} türüymüş",
               "opposite": f"{subj} ile {obj} birbirinin zıttıymış",
               "used_for": f"{subj} şu işe yarıyormuş: {obj}",
               "part_of": f"{subj} bir {obj} parçasıymış"}.get(rel)
        opts = [f"{subj} ile {obj} arasında bir bağ kurdum."]
        if phr:
            opts.insert(0, phr)
        return self._pick(opts)

    def _c_nudge(self, agi) -> str:
        return tr_capitalize(self._pick([
            SYN.realize_pred(PredSpec(subj=NP("sen"), pred_root="orada", person="2SG", question=True)) + "?",
            "bir süredir sessizsin; iyi misin?",
            "sessizliğin beni meraklandırıyor.",
        ]))

    # ------------------------------------------------------------------ üretim
    def compose(self, move: str, agi) -> str:
        builder = {
            "SHARE_STATE": self._c_self_state,
            "ASK_ABOUT": self._c_ask_about,
            "WONDER": self._c_wonder,
            "REMEMBER": self._c_remember,
            "OFFER": self._c_offer,
            "CONFIDE": self._c_confide,
            "NOTICE_BODY": self._c_body,
            "SHARE_DISCOVERY": self._c_discovery,
            "NUDGE": self._c_nudge,
        }.get(move)
        if builder is None:
            return ""
        text = builder(agi) or ""
        return text.strip()

    def speak(self, agi, tension: float = 0.0, discovery: bool = False, idle: float = 0.0,
              unanswered: int = 0, temperature: float = 0.6) -> dict | None:
        """Kendiliğinden söylenecek sözü üretir. Uygun değilse None (suskunluk da bir cevaptır)."""
        prev = self.last_move
        feats = self.features(agi, tension=tension, discovery=discovery, idle=idle,
                              recent_reach=min(1.0, unanswered / 4.0))
        move, probs = self.select(feats, temperature=temperature)
        # Aynı tür üst üste gelmesin (insan da aynı cümleyle açmaz)
        if move == prev and self.rng.random() < 0.8:
            move = max((m for m in MOVES if m != prev), key=lambda m: probs.get(m, 0.0))
        text = self.compose(move, agi)
        if not text:
            # Seçilen tür şu an kurulamıyorsa (ör. hatırlanacak söz yok) sıradaki olası türler denenir.
            for alt in sorted((m for m in MOVES if m != move), key=lambda m: -probs.get(m, 0.0))[:4]:
                t2 = self.compose(alt, agi)
                if t2:
                    text, move = t2, alt
                    break
        if not text:
            self.stats["silent"] += 1
            return None
        # Aynı cümleyi tekrar söyleme
        if self.recent and max(_jaccard(_toks(text), _toks(r)) for r in self.recent) > 0.75:
            alt = self.compose(max((m for m in MOVES if m != move), key=lambda m: probs.get(m, 0.0)), agi)
            if alt and max(_jaccard(_toks(alt), _toks(r)) for r in self.recent) <= 0.75:
                text = alt
                move = "SHARE_STATE" if move == "WONDER" else move
        self.recent.append(text)
        self.habituate(move)
        self.last_move = move
        self.last_text = text
        self.stats["said"] += 1
        self.stats["moves"][move] = self.stats["moves"].get(move, 0) + 1
        return {"text": text, "move": move, "probs": probs, "features": feats}

    # ------------------------------------------------------------------ kalıcılık
    def state_dict(self) -> dict:
        return {"W": self.W.tolist(), "stats": dict(self.stats), "last_move": self.last_move}

    def load_state_dict(self, d: dict):
        if not d:
            return
        try:
            W = np.array(d.get("W") or [], dtype=np.float64)
            if W.shape == self.W.shape:
                self.W = W
        except Exception:
            pass
        self.stats.update(d.get("stats") or {})
        self.last_move = str(d.get("last_move", ""))

    def snapshot(self) -> dict:
        return {"said": self.stats["said"], "silent": self.stats["silent"],
                "moves": dict(self.stats["moves"]), "last": self.last_text,
                "prior": {m: round(float(self.W[i].max()), 2) for i, m in enumerate(MOVES)}}
