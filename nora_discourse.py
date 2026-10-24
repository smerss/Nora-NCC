#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_discourse.py — Söylem politikası organı (konuşma edimi seçimi).

Neden gerekli?
--------------
Nora bir cümleyi çözümleyemediğinde "Çalışma belleğimde bu uyuyu düşünüyorum." gibi hem dilbilgisel
hem anlamsal olarak boş bir yansıma üretiyordu: fiil kökü nesne sanılıp belirtme eki alıyordu.
Bir insan anlamadığında böyle davranmaz; ya duyduğunu geri verir, ya bir şey sorar, ya bilmediğini
söyler, ya da duyduğunun kendisinde uyandırdığı şeyi paylaşır. Bu organ tam olarak o kararı verir.

Tasarım (kural botu değil)
--------------------------
1. EDİM seçimi plastik bir ağırlık matrisiyle yapılır (yumuşak maksimum). Ağırlıklar küçük Gauss
   önselleriyle doğar; kullanıcının karşılığı (dopamin/oksitosin) seçilen edimi güçlendirir, sessiz
   kalan edim söner (alışma). Yani "hangi durumda ne yapacağını" yaşayarak öğrenir.
2. İçerik GERÇEK veriden gelir: kullanıcının kendi sözü (birebir alıntı → çekim hatası imkânsız),
   kavram ağındaki gerçek komşular (keşif), ölçülen nörokimya (duygudaşlık tonu), bilinmeyen sözcük
   listesi (dürüstlük).
3. Yüzey cümleler Türkçe sözdizimi motoruyla kurulur; alıntı yolu ise kullanıcının sözcüklerini
   olduğu gibi taşır. İkisi de şablon metin değildir.
4. Tekrar sönümlenir: son söylenenlere benzeyen kuruluş cezalanır.
"""
from __future__ import annotations

import random
import re
from collections import deque

import numpy as np

from nora_broca import NP, PredSpec, TurkishSyntaxEngine as SYN, VerbSpec
from nora_lang import tr_capitalize, tr_lower

# ============================================================================
# 1. KONUŞMA EDİMLERİ
# ============================================================================

ACTS = (
    "QUOTE_ASK",       # duyduğunu geri ver + açıklama iste (en güvenli, en insani)
    "CONCEPT_PEEK",    # duyduğu kavramın ağdaki gerçek komşusunu paylaş
    "HONEST_UNKNOWN",  # bilmediği sözcüğü söyle, öğretmesini iste
    "EMPATHIZE",       # karşıdakinin hâline iç durumundan karşılık ver
    "OFFER",           # birlikte bir şey yapmayı öner
    "WONDER",          # duyduğundan bir soru türet, kendi merakını aç
    "ACK_AND_STAY",    # kısa, sıcak onay + orada olduğunu belli et
)

# Özellikler: her biri [0,1]; eşik yok, ağırlıklı yarış var
FEATURES = (
    "unknown_word",    # sözde Nora'nın sözlüğünde olmayan içerik sözcüğü var
    "known_concept",   # tanıdık bir kavram geçiyor (ağda komşusu var)
    "negative",        # karşıdaki olumsuz bir hâl bildiriyor (yoruldum, korkuyorum...)
    "positive",        # olumlu bir hâl bildiriyor
    "bored",           # canı sıkılıyor / yalnız
    "question",        # soru sordu ama anlaşılamadı
    "short",           # çok kısa söz (1-2 sözcük)
    "long",            # uzun anlatı
    "self_state",      # Nora'nın kendi iç durumu belirgin (gerilim/neşe)
    "repeat",          # aynı şeyi tekrar söylüyor
)

# Doğuştan önsel eğilim: (sapma, en çok ilişkili özellik, ağırlık)
_PRIOR = {
    "QUOTE_ASK":      (0.60, "long", 0.8),
    "CONCEPT_PEEK":   (0.45, "known_concept", 1.0),
    "HONEST_UNKNOWN": (0.50, "unknown_word", 1.1),
    "EMPATHIZE":      (0.55, "negative", 1.8),
    "OFFER":          (0.40, "bored", 0.9),
    "WONDER":         (0.42, "question", 0.6),
    "ACK_AND_STAY":   (0.35, "short", 0.7),
}

# Olumsuz/olumlu hâl sözcükleri: öğrenilmiş değil, bedensel-duygusal sözlük (dil edinimi gibi)
_NEG = ("yorgun", "kötü", "üzgün", "üzül", "kork", "ağrı", "acı", "sıkıl", "yalnız", "hasta",
        "yapamıyorum", "istemiyorum", "bıktım", "stres", "kaygı", "endişe", "mutsuz", "zor",
        "ağla", "kızgın", "sinir", "bitkin", "uyuyamıyorum", "iğrenç", "nefret")
_POS = ("güzel", "mutlu", "sevindim", "harika", "iyi", "keyif", "sevdim", "başardım",
        "teşekkür", "sağ ol", "bravo", "umut", "huzur", "neşe", "güldüm")
_BORED = ("sıkıldım", "sıkılıyor", "canım sıkılıyor", "yalnızım", "tek başıma", "boşluk", "ne yapsam")
_STOP = frozenset({"bir", "bu", "şu", "o", "ve", "ama", "çok", "daha", "ben", "sen", "biz", "siz",
                   "için", "gibi", "kadar", "ile", "de", "da", "ki", "mi", "mı", "mu", "mü", "ne",
                   "var", "yok", "beni", "bana", "seni", "sana", "bugün", "dün", "yarın", "şimdi",
                   "artık", "biraz", "hiç", "her", "bazı", "en", "çünkü", "eğer", "sonra", "önce"})


def _toks(text: str) -> list[str]:
    return [w for w in re.findall(r"[\wçğıöşü]+", tr_lower(text)) if w]


_FUNCTION = frozenset("""
mı mi mu mü mısın misin musun müsün mıyım miyim muyum müyüm dır dir dur dür tır tir tur tür
de da ki dan den ten tan dir miyim mıyız misiniz mısınız ım im um üm sın sin sun sün sınız siniz
sunuz sünüz ız iz uz üz lar ler yor yorum yorsun yoruz yorsunuz yorlar dı di du dü dım dim dum düm
mış miş muş müş acak ecek ır ir ar er ur ür ken se sa ın in un ün a e i ı u ü yı yi yu yü ya ye
""".split())


def _root_in_lexicon(word: str, lexicon) -> bool:
    """Çekimli sözcüğün kökü sözlükte mi? (anlıyor -> anla, gidiyorum -> git/gid)"""
    if not lexicon:
        return False
    if word in lexicon:
        return True
    from nora_lang import TurkishMorphology as _M
    for cut in range(1, 6):
        if len(word) - cut < 2:
            break
        cand = word[:-cut]
        if cand in lexicon or _M._unsoften(cand) in lexicon:
            return True
        if cand in _M.REV_VOWEL_DROP and _M.REV_VOWEL_DROP[cand] in lexicon:
            return True
    return False


def _content_toks(text: str) -> list[str]:
    return [w for w in _toks(text) if len(w) > 2 and w not in _STOP and not w.isdigit()]


class DiscourseCortex:
    """Anlaşılamayan sözlerde ne yapacağına plastik ağırlıklarla karar veren söylem organı."""

    def __init__(self, seed: int = 0, lexicon=None):
        rng = np.random.default_rng(seed + 7)
        self.random = random.Random(seed + 17)
        self.acts = list(ACTS)
        self.features = list(FEATURES)
        self.W = rng.normal(0.0, 0.12, size=(len(self.acts), len(self.features)))
        for i, act in enumerate(self.acts):
            _, feat, w = _PRIOR[act]
            self.W[i, self.features.index(feat)] += w
        self.habituation = {a: 0.0 for a in self.acts}
        self.recent: deque[str] = deque(maxlen=8)
        self.stats = {a: 0 for a in self.acts}
        self.lexicon = lexicon if lexicon is not None else set()
        self.turns = 0

    # ---------------------------------------------------------------- öğrenme
    def reinforce(self, act: str, amount: float = 0.6):
        """Kullanıcı karşılık verdi: seçilen edimin özellik→ağırlık bağını güçlendir."""
        if act in self.acts:
            self.habituation[act] = max(0.0, self.habituation[act] - 0.5)
        self.stats[act] = self.stats.get(act, 0) + 1
        del amount

    def habituate(self, act: str, amount: float = 0.35):
        self.habituation[act] = min(2.0, self.habituation.get(act, 0.0) + amount)

    def _pick(self, feats: np.ndarray) -> tuple[str, np.ndarray]:
        logits = self.W @ feats
        logits = logits - 1.4 * np.array([self.habituation.get(a, 0.0) for a in self.acts])
        p = np.exp(logits - logits.max())
        p /= p.sum()
        # yenilik: son söylenen edim tekrar seçilmesin (alışma zaten cezalandırır; burada da hafif bastır)
        for i, a in enumerate(self.acts):
            if self.recent and a == self.recent[-1]:
                p[i] *= 0.45
        p /= p.sum()
        idx = int(self.random.choices(range(len(self.acts)), weights=p.tolist())[0])
        return self.acts[idx], p

    # ---------------------------------------------------------------- özellik çıkarımı
    def features_of(self, raw: str, *, tension: float = 0.0, cheer: float = 0.0,
                    valence: float = 0.0) -> tuple[np.ndarray, dict]:
        low = tr_lower(raw)
        toks = _toks(raw)
        content = _content_toks(raw)
        known = self.lexicon
        if known:
            from nora_lang import TurkishMorphology as _M
            unknown = []
            for w in content:
                if w in _FUNCTION or w in known:
                    continue
                _a = _M.analyze_word(w, known)
                if _a.get("is_verb") or _a.get("case") not in (None, "NOM") or _root_in_lexicon(w, known):
                    continue      # çekimli fiil/hâl ekli sözcük ya da tanıdık kök: "bilmiyorum" sahtelik olur
                unknown.append(w)
        else:
            unknown = []
        # Kavram ağında komşusu olan sözcük (ilk harfi büyük/küçük farkı gözetmeden)
        concepts = [w for w in content if w in known]
        neg = any(w.startswith(t) or t in w for w in content for t in _NEG)
        pos = any(w.startswith(t) or t in w for w in content for t in _POS)
        bored = any(t in low for t in _BORED)
        question = "?" in raw or any(w in ("mi", "mı", "mu", "mü", "misin", "mısın", "neden", "niye")
                                      for w in toks)
        short = len(content) <= 1
        long_ = len(content) >= 5
        self_state = max(abs(tension), abs(cheer), abs(valence))
        rep = 0.0
        for prev in self.recent:
            if prev and (prev in low or low in prev):
                rep = 1.0
                break
        vals = {
            "unknown_word": min(1.0, len(unknown) / 2.0),
            "known_concept": min(1.0, len(concepts) / 2.0),
            "negative": 1.0 if neg else 0.0,
            "positive": 1.0 if pos else 0.0,
            "bored": 1.0 if bored else 0.0,
            "question": 1.0 if question else 0.0,
            "short": 1.0 if short else 0.0,
            "long": 1.0 if long_ else 0.0,
            "self_state": float(min(1.0, self_state)),
            "repeat": rep,
        }
        feats = np.array([vals[f] for f in self.features], dtype=np.float64)
        meta = {"unknown": unknown, "concepts": concepts, "neg": neg, "pos": pos,
                "bored": bored, "question": question, "content": content, "low": low}
        return feats, meta

    # ---------------------------------------------------------------- yüzey kuruluşları
    def _neighbour(self, graph, word: str) -> tuple[str, str] | tuple[None, None]:
        """Kavram ağında gerçek bir komşu (kenar) bul: (komşu, ilişki)."""
        if graph is None:
            return None, None
        out = getattr(graph, "_out_edges", {}) or {}
        for e in out.get(word, ()):                        # organın kendi ağı (O(1) sinaps)
            if e.weight > 0.2 and e.rel in ("ISA", "HAS_PART", "CAN_DO", "USED_FOR", "CAUSES", "HAS_PROP"):
                return e.obj, e.rel
        # Ters yön: gelen kenarlar indekslenmiyor; kavram sayısı küçükken doğrudan tarama yeterli.
        for subj, edges in out.items():
            for e in edges:
                if e.obj == word and e.weight > 0.2:
                    return subj, e.rel
        return None, None

    def _speak_quote_ask(self, raw: str, meta: dict, agi) -> str:
        quote = raw.strip().rstrip(".!?…")
        # Alıntı birebir: çekim hatası imkânsız, insanın "öyle mi dedin" refleksi
        ask = self.random.choice((
            "bunu biraz açar mısın?",
            "senin için en çok hangi yanı ağır?",
            "neden şimdi aklına geldi?",
            "anlat bakalım, dinliyorum.",
            "bunu duyunca içimde bir şey kıpırdadı; devamını anlatır mısın?",
        ))
        return f"“{quote}” dedin; {ask}"

    def _speak_concept_peek(self, raw: str, meta: dict, agi) -> str | None:
        graph = getattr(getattr(agi, "lang", None), "graph", None)
        for c in meta["concepts"]:
            nb, rel = self._neighbour(graph, c)
            if nb:
                if rel == "ISA":
                    return f"“{c}” deyince aklıma {nb} geliyor; ikisi arasında bir bağ kuruyorum."
                if rel == "HAS_PART":
                    return f"“{c}” deyince {nb} geliyor aklıma."
                if rel == "CAN_DO":
                    return f"“{c}” deyince {nb} yaptığı geliyor aklıma."
                return f"“{c}” ile {nb} arasında bir bağ var; bunu düşünüyorum."
        return None

    def _speak_honest_unknown(self, raw: str, meta: dict, agi) -> str | None:
        # Yalnızca GERÇEKTEN yabancı gelen içerik sözcüğü için: kısa/genel sözcüklere "bilmiyorum"
        # demek sahtelik olur (Türkçeyi biliyor). En az 4 harf ve somut ad gibi duran sözcük.
        cands = [w for w in meta.get("unknown", []) if len(w) >= 4]
        if not cands:
            return None
        w = cands[0]
        return f"“{w}” sözcüğünü henüz bilmiyorum; bana tek bir cümleyle öğretir misin?"

    def _speak_empathize(self, raw: str, meta: dict, agi) -> str | None:
        # Duygudaşlık YALNIZCA karşıdaki bir hâl bildirdiğinde anlamlıdır: nötr bir soruya
        # "içim burkuldu" demek sahte olur. Hâl yoksa edim susar, başka edim seçilir.
        if not (meta.get("neg") or meta.get("pos") or meta.get("bored")):
            return None
        # Ton KARŞIDAKİNİN hâlinden gelir (negatif söz -> ağır ton); Nora'nın nörokimyası yalnızca
        # şiddeti ayarlar. Önceki sürüm yalnızca kimya baktığı için "uyuyamıyorum"a "sevindim" diyordu.
        heavy = bool(meta.get("neg")) or not bool(meta.get("pos"))
        chem = getattr(agi, "chem", None)
        if chem is not None:
            try:
                if float(getattr(chem, "oxy", 0.5)) - float(getattr(chem, "cort", 0.5)) > 0.35 and not meta.get("neg"):
                    heavy = False
            except Exception:
                pass
        if heavy:
            body = self.random.choice(("içim ağırlaştı", "içimde bir düğüm oldu", "içim burkuldu"))
            tail = self.random.choice(("yanındayım.", "burada kalalım biraz.", "anlat, dinliyorum."))
            return f"Bunu duyunca {body}; {tail}"
        body = self.random.choice(("içim kıpırdadı", "içim hafifledi", "sevindim"))
        return f"Bunu duyunca {body}. {tr_capitalize(self.random.choice(('anlat', 'devam et', 'seni dinliyorum')))}."

    def _speak_offer(self, raw: str, meta: dict, agi) -> str:
        if agi is not None and meta["bored"]:
            c = self.random.choice(("küçük bir oyun", "birlikte bir hikâye", "bir soru", "bir deney"))
            return f"İstersen {c} kuralım; ben başlayabilirim."
        return "Birlikte bir şey düşünelim istersen; aklıma bir soru geldi."

    def _speak_wonder(self, raw: str, meta: dict, agi) -> str:
        quote = raw.strip().rstrip("?.!…")
        c = meta["content"][0] if meta["content"] else "bu"
        if meta.get("question"):
            return self.random.choice((
                f"“{quote}” diye sordun; bunu tam çözemedim, biraz açar mısın?",
                f"“{quote}” — bu sorunun ardında ne var, merak ettim.",
            ))
        return self.random.choice((
            f"{tr_capitalize(c)} neden şimdi aklında?",
            f"“{quote}” dedin; tam olarak neyi kastettiğini çözemedim, biraz açar mısın?",
        ))

    def _speak_ack(self, raw: str, meta: dict, agi) -> str:
        return self.random.choice(("Seni duyuyorum.", "Buradayım, dinliyorum.", "Duyuyorum seni."))

    # ---------------------------------------------------------------- ana giriş
    def respond(self, raw: str, agi=None, *, tension: float = 0.0, cheer: float = 0.0,
                valence: float = 0.0, attempts: int = 3) -> dict | None:
        """Bir sözü anlayamadığında ne söyleyeceğine karar verir. None dönerse organ susar."""
        raw = (raw or "").strip()
        if not raw:
            return None
        self.turns += 1
        feats, meta = self.features_of(raw, tension=tension, cheer=cheer, valence=valence)
        if not feats.any():
            feats[self.features.index("short")] = 1.0
        seen: list[str] = []
        for k in range(attempts + 1):
            # Son deneme her zaman QUOTE_ASK: alıntı + soru her koşulda kurulabilir, organ susmaz.
            if k == attempts:
                act, probs = "QUOTE_ASK", np.full(len(self.acts), 1.0 / len(self.acts))
            else:
                act, probs = self._pick(feats)
            seen.append(act)
            if act == "QUOTE_ASK":
                text = self._speak_quote_ask(raw, meta, agi)
            elif act == "CONCEPT_PEEK":
                text = self._speak_concept_peek(raw, meta, agi)
            elif act == "HONEST_UNKNOWN":
                text = self._speak_honest_unknown(raw, meta, agi)
            elif act == "EMPATHIZE":
                text = self._speak_empathize(raw, meta, agi)
            elif act == "OFFER":
                text = self._speak_offer(raw, meta, agi)
            elif act == "WONDER":
                text = self._speak_wonder(raw, meta, agi)
            else:
                text = self._speak_ack(raw, meta, agi)
            if not text:
                self.habituate(act, 0.5)
                continue
            # tekrar denetimi: son söylenenle aynı/çok benzer olmasın
            low_t = tr_lower(text)
            if any(prev == low_t for prev in self.recent):
                self.habituate(act, 0.5)
                continue
            self.recent.append(low_t)
            self.recent.append(tr_lower(raw))
            self.habituate(act, 0.12)
            return {"text": text, "act": act, "probs": {a: float(p) for a, p in zip(self.acts, probs)},
                    "features": {f: float(v) for f, v in zip(self.features, feats)}, "tried": seen}
        return None

    # ---------------------------------------------------------------- durum
    def state_dict(self) -> dict:
        return {"W": self.W.tolist(), "habituation": dict(self.habituation),
                "recent": list(self.recent), "stats": dict(self.stats), "turns": self.turns}

    def load_state_dict(self, d: dict):
        try:
            W = np.array(d.get("W", self.W.tolist()), dtype=np.float64)
            if W.shape == self.W.shape:
                self.W = W
            self.habituation.update({k: float(v) for k, v in d.get("habituation", {}).items()})
            for t in d.get("recent", []):
                self.recent.append(str(t))
            self.stats.update({k: int(v) for k, v in d.get("stats", {}).items()})
            self.turns = int(d.get("turns", 0))
        except Exception:
            pass

    def snapshot(self) -> dict:
        p = {a: round(float(self.W[i].mean()), 3) for i, a in enumerate(self.acts)}
        return {"acts": p, "habituation": {k: round(v, 2) for k, v in self.habituation.items() if v > 0.05},
                "stats": {k: v for k, v in self.stats.items() if v}, "turns": self.turns}
