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

_WORD_RE = re.compile(r"[\wçğıöşü]+")

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


def _cap(text: str) -> str:
    """Baş harfi büyüt ve cümleyi soru/bitiş işaretiyle kapat."""
    text = (text or "").strip()
    if not text:
        return text
    text = tr_capitalize(text[0]) + text[1:]
    if text[-1] not in ".?!":
        text += "."
    return text


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
        self.recent_followups: deque[str] = deque(maxlen=6)
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

    # ---------------------------------------------------------------- toplumsal olay
    def social_event_of(self, raw: str) -> dict | None:
        """Kişinin hayatındaki toplumsal olaylar (doğum günü gibi) bir bilgi sorusu değildir."""
        low = tr_lower(raw)
        words = _WORD_RE.findall(low)
        if any(w.startswith("doğum") for w in words) or "doğum gün" in low:
            return {"kind": "birthday"}
        return None

    def respond_social_event(self, raw: str, agi, ev: dict) -> dict:
        """Toplumsal olaya insan gibi karşılık; yüzey yine sözdizimi motorundan kurulur."""
        if ev["kind"] == "birthday":
            wish = SYN.realize_verb(VerbSpec(verb_root="kutla", obj=NP("doğum günü", poss=2, case="ACC"),
                                             tense="PRES", person="1SG"))
            ask = SYN.realize_pred(PredSpec(subj=NP("yaş", poss=2), pred_root="kaç", person="3SG", question=True))
            return {"text": f"{tr_capitalize(wish)}! {tr_capitalize(ask)}?", "act": "SOCIAL_BIRTHDAY"}
        return {"text": "", "act": "SOCIAL_NONE"}

    # ---------------------------------------------------------------- kişisel anlatı
    NARRATION_ACTS = ("REFLECT_ASK", "SHARE_FEELING", "OFFER_LISTEN")

    def narration_of(self, raw: str) -> dict | None:
        """Karşıdaki kendi yaşantısını mı anlatıyor? 1. kişi bitmiş fiil (-dIm, -yorum, -mIşIm)
        varsa ve soru değilse bu bir ANLATIdır: Nora ders vermez, dinler ve karşılık verir."""
        if "?" in raw:
            return None
        from nora_lang import TurkishMorphology as _M
        words = _WORD_RE.findall(tr_lower(raw))
        morph = _M()
        lex = set(self.lexicon or ())
        # Aynı sözcük bir turda bir kez çözümlensin (tur içi memo; O(1))
        memo: dict[str, dict] = {}
        # Kimlik bildiren cümleler (ad, yaş, memleket) anlatı değil SOSYAL BİLGİdir; onları
        # sosyal öğrenme korteksi işler, Nora "duyuyorum" diye yansıtmaz.
        if any(w in words for w in ("adım", "ismim", "adımı", "yaşındayım", "yaşıyorum", "senindir")):
            return None
        for i, w in enumerate(words):
            an = memo.get(w)
            if an is None:
                an = memo[w] = morph.analyze_word(w, lex)
            # Gelecek/istek/ortak istek kipleri bir YAŞANTI değil niyettir: "sana bir şey soracağım",
            # "biraz sohbet edelim" anlatı değildir (davet/plan olarak ayrıca işlenir).
            if an.get("tense") in ("OPT", "NEC"):
                continue
            if an.get("is_verb") and an.get("person") in ("1SG", "1PL"):
                return {"verb": an["stem"], "neg": bool(an.get("neg")), "person": an.get("person"),
                        "tense": an.get("tense", ""), "index": i, "words": words,
                        "rest": words[:i], "tail": words[i + 1:]}
        # 1. kişi koşaçlı hâl sıfatı: 'bugün kendimi yalnız hissediyorum' değilse de 'çok yorgunum'.
        from nora_knowledge import STATE_ADJECTIVES
        for i, w in enumerate(words):
            for suf in ("yım", "yim", "yum", "yüm", "ım", "im", "um", "üm"):
                if len(w) - len(suf) >= 3 and w.endswith(suf):
                    base = w[: -len(suf)]
                    if base in STATE_ADJECTIVES:
                        return {"verb": base, "neg": False, "person": "1SG", "tense": "STATE",
                                "index": i, "words": words, "rest": words[:i], "tail": words[i + 1:],
                                "state": True}

        # İyelikli özne + hâl sıfatı: 'moralim bozuk', 'kafam çok karışık', 'keyfim yok'
        _adv = {"çok", "biraz", "pek", "epey", "oldukça", "fena", "çokça", "hayli", "bayağı", "artık"}
        for i, w in enumerate(words):
            an = memo.get(w)
            if an is None:
                an = memo[w] = morph.analyze_word(w, lex)
            if not (an.get("poss") == 1 or an.get("poss1")):
                continue
            j = i + 1
            while j < len(words) and words[j] in _adv:
                j += 1
            if j >= len(words):
                continue
            nxt = words[j]
            _nan = memo.get(nxt)
            if _nan is None:
                _nan = memo[nxt] = morph.analyze_word(nxt, lex)
            _is_state = nxt in STATE_ADJECTIVES or nxt in ("yok", "yoktur")
            if not _is_state and _nan.get("case") == "NOM" and not _nan.get("is_verb") \
                    and not _nan.get("poss") and len(nxt) > 2:
                # 'matematik ödevim çok zor' — sıfat yüklemi (sözlükte sıfat diye etiket yok; yapı karar verir)
                _is_state = True
            if _is_state:
                # Çok sözcüklü özne: 'matematik ödevim' -> 'matematik ödev'
                _k = i
                _stop = {"bugün", "dün", "yarın", "şimdi", "sabah", "akşam", "gece", "artık", "yine",
                         "hep", "çok", "biraz", "pek", "gerçekten", "yani", "ama", "ve"}
                while _k > 0:
                    _pw = words[_k - 1]
                    if _pw in _stop:
                        break
                    _pan = memo.get(_pw)
                    if _pan is None:
                        _pan = memo[_pw] = morph.analyze_word(_pw, lex)
                    if _pan.get("case") != "NOM" or _pan.get("is_verb") or _pan.get("poss"):
                        break
                    _k -= 1
                _subj = " ".join(words[_k:i] + [an["stem"]]).strip()
                return {"verb": nxt, "neg": nxt.startswith("yok"), "person": "1SG", "tense": "STATE",
                        "index": i, "words": words, "rest": words[:i], "tail": words[j + 1:],
                        "state": True, "subject": _subj}

        # Kimlik bildirimi: 'Ben öğretmenim' (1. kişi koşacı + meslek adı)
        if len(words) >= 2 and words[0] == "ben":
            an = memo.get(words[1])
            if an is None:
                an = memo[words[1]] = morph.analyze_word(words[1], lex)
            if an.get("poss") == 1 and an["stem"] not in STATE_ADJECTIVES and not an.get("is_verb"):
                return {"verb": "ol", "neg": False, "person": "1SG", "tense": "IDENTITY",
                        "index": 1, "words": words, "rest": [], "tail": words[2:],
                        "identity": an["stem"]}

        # 3. kişiyle kurulmuş kişisel olaylar: iyelikli özne ('kedim', 'annem', 'başım')
        poss_idx = -1
        for i, w in enumerate(words):
            _an = memo.get(w)
            if _an is None:
                _an = memo[w] = morph.analyze_word(w, lex)
            if _an.get("poss") == 1:
                poss_idx = i
                break
        if poss_idx >= 0:
            for i, w in enumerate(words):
                an = memo.get(w)
                if an is None:
                    an = memo[w] = morph.analyze_word(w, lex)
                if an.get("is_verb") and an.get("person") in ("3SG", "3PL"):
                    _pan = memo.get(words[poss_idx]) or morph.analyze_word(words[poss_idx], lex)
                    return {"verb": an["stem"], "neg": bool(an.get("neg")), "person": "3SG",
                            "tense": an.get("tense", ""), "index": i, "words": words,
                            "rest": [w for w in words[:i] if w != words[poss_idx]],
                            "tail": words[i + 1:], "subject": _pan.get("stem", "")}
        return None

    def respond_narration(self, raw: str, agi, nar: dict) -> dict:
        """Anlatıya insan gibi karşılık: olayı kendi sözleriyle geri yansıt + gerçek bir soru."""
        verb = nar["verb"]
        if nar.get("identity"):
            role = nar["identity"]
            try:
                learned = SYN.realize_verb(VerbSpec(
                    verb_root="öğren",
                    obj=SYN.nominalize_dik("ol", poss=2, case="ACC", subj=NP(role)),
                    tense="PAST", person="1SG"))
                ask = SYN.realize_verb(VerbSpec(verb_root="git", subj=NP("iş", poss=2),
                                               wh_word="nasıl", tense="PRES", person="3SG"))
                text = f"{tr_capitalize(learned)}; bunu aklımda tutacağım. {tr_capitalize(ask)}?"
            except Exception:
                text = f"{tr_capitalize(role)} olduğunu öğrendim."
            return {"text": text, "act": "NARRATION_IDENTITY", "narration": True}

        # Gereklilik: 'kalkmam gerekiyor' — '-mAm' burada olumsuzluk değil, isim-fiil+iyelik.
        _nw = nar.get("words") or []
        _ni = int(nar.get("index") or 0)
        _nxt = _nw[_ni + 1] if _ni + 1 < len(_nw) else ""
        if _nxt in ("gerekiyor", "gerek", "lazım", "gerekli", "gerekti", "gerekecek", "şart"):
            try:
                _vn = SYN.nominalize_inf(verb, short_ma=True, poss=2, case="NOM")
            except Exception:
                _vn = ""
            if _vn:
                _rest = " ".join(w for w in (nar.get("rest") or []) if w not in ("ben", "biz", "artık"))
                lead = _cap(f"{_rest} {_vn} gerektiğini duyuyorum".strip())
                follow = self._fresh(("Zor ama yapılabilir.", "Birlikte planlayalım mı?",
                                      "Ne zaman başlamayı düşünüyorsun?", "Kendine zaman tanı."))
                return {"text": f"{lead} {follow}", "act": "NARRATION_NECESSITY", "narration": True}

        if nar.get("state") or nar.get("tense") == "STATE":
            # Hâl bildirimi ('yalnızım', 'yorgunum', 'keyfim yok'): ders değil, duygudaşlık.
            adj = verb
            _subj = str(nar.get("subject") or "")
            if not (nar.get("neg") or adj in ("yok", "yoktur")) and _subj:
                try:
                    gen = SYN.realize_np(NP(_subj, poss=2, case="GEN"))
                except Exception:
                    gen = _subj
                lead = _cap(self._fresh((f"{gen} {adj} olduğunu duyuyorum",
                                         f"{gen} {adj} olması can sıkıcı",
                                         f"Bugün {gen} {adj} olduğunu görüyorum")))
                follow = self._fresh(("Yanındayım.", "Anlatmak ister misin?", "Birlikte biraz soluklanalım mı?"))
                return {"text": f"{lead} {follow}", "act": "NARRATION_EMPATHIZE", "narration": True}
            if nar.get("neg") or adj in ("yok", "yoktur"):
                subj = str(nar.get("subject") or "için")
                try:
                    gen = SYN.realize_np(NP(subj, poss=2, case="GEN"))
                except Exception:
                    gen = subj
                lead = _cap(self._fresh((f"{gen} yerinde olmadığını duyuyorum",
                                         f"{gen} olmadığını görüyorum, kolay değil",
                                         f"Bugün {gen} yerinde değil anlaşılan")))
                follow = self._fresh(("Yanındayım.", "Anlatmak ister misin?", "Birlikte biraz soluklanalım mı?"))
                return {"text": f"{lead} {follow}", "act": "NARRATION_EMPATHIZE", "narration": True}
            lead = _cap(self.random.choice((
                f"Kendini {adj} hissettiğini duyuyorum",
                f"Şu an {adj} olduğunu anlıyorum",
                f"{tr_capitalize(adj)} olmak zor",
            )))
            try:
                chem = agi.chem
                warm = float(getattr(chem, "oxy", 0.5)) >= float(getattr(chem, "cort", 0.5))
            except Exception:
                warm = True
            follow = self._fresh((
                "Yanındayım.", "Anlatmak ister misin?", "Bu his ne zamandır sürüyor?",
                "Ben de kendi içimde benzer bir şey taşıyorum.", "İstersen biraz susalım, beraber.",
            )) if warm else self._fresh((
                "Bunu yalnız taşımak zorunda değilsin.", "Biraz yavaşlayalım mı?", "Nefes al, buradayım.",
            ))
            return {"text": f"{lead} {follow}", "act": "NARRATION_EMPATHIZE", "narration": True}
        # Konuşan 'biz' diyorsa dinleyen de o topluluğun içinde: 'gideceğimizi duyuyorum'.
        _poss = 4 if str(nar.get("person")) == "1PL" else 2
        try:
            if str(nar.get("tense") or "").startswith("FUT"):
                obj = SYN.nominalize_acak(verb, poss=_poss, case="ACC", neg=nar["neg"])
            else:
                obj = SYN.nominalize_dik(verb, poss=_poss, case="ACC", neg=nar["neg"])
        except Exception:
            obj = ""
        rest = [w for w in nar["rest"] if w not in ("ben", "biz", "bugün", "dün", "şimdi")]
        # Karşıdakinin ağzından aktarırken 1. kişi iyeliği 2. kişiye döner:
        # 'kedim kucağıma geldi' -> 'kedinin kucağına geldiğini duyuyorum'.
        _morph = getattr(getattr(agi, "lang", None), "morph", None)
        _lex = getattr(getattr(getattr(agi, "lang", None), "graph", None), "lexicon", None)
        if _morph is not None and rest:
            _shifted = []
            for w in rest:
                try:
                    _an = _morph.analyze_word(w, _lex)
                except Exception:
                    _an = {}
                if _an.get("poss") == 1 and _an.get("case") and _an.get("case") != "NOM":
                    try:
                        _shifted.append(SYN.realize_np(NP(_an["stem"], poss=2, case=_an["case"])))
                        continue
                    except Exception:
                        pass
                _shifted.append(w)
            rest = _shifted
        rest_txt = " ".join(rest)
        _nsub = str(nar.get("subject") or "")
        # Yan cümle ('geldi ve uyudu'): ikinci yüklem de aktarılır -> '... geldiğini ve uyuduğunu'.
        _tail_clause = ""
        _tail = [w for w in (nar.get("tail") or []) if w not in ("ve", "sonra", "ardından", "da", "de")]
        if _morph is not None and _tail:
            for _w in _tail:
                try:
                    _tan = _morph.analyze_word(_w, _lex)
                except Exception:
                    _tan = {}
                if _tan.get("is_verb"):
                    try:
                        _tail_clause = SYN.nominalize_dik(_tan["stem"], poss=2, case="ACC",
                                                          neg=bool(_tan.get("neg")))
                    except Exception:
                        _tail_clause = ""
                    break
        if _nsub and obj and not nar.get("state"):
            # 'içim daralıyor' -> 'içinin daraldığını duyuyorum' (iyelikli özne tamlayana döner)
            try:
                gen = SYN.realize_np(NP(_nsub, poss=2, case="GEN"))
            except Exception:
                gen = _nsub
            _cpl = f"{obj} ve {_tail_clause}" if _tail_clause else obj
            lead = f"{gen} {rest_txt} {_cpl} duyuyorum".replace("  ", " ")
        elif obj:
            _cpl = f"{obj} ve {_tail_clause}" if _tail_clause else obj
            lead = f"{rest_txt} {_cpl} duyuyorum".strip()
        else:
            lead = "seni duyuyorum"
        lead = _cap(lead)

        act = self._pick_narration_act(nar)
        _future = str(nar.get("tense") or "").startswith("FUT") or any(
            w in (nar.get("words") or []) for w in ("yarın", "haftaya", "gelecek", "sonra"))
        if act == "REFLECT_ASK" and _future:
            # Gelecek anlatısına geçmiş sorusu sorulmaz ('Nasıl geçti?' değil 'Neler planlıyorsun?').
            q = self._fresh((
                "Neler planlıyorsun?", "Heyecanlı mısın?", "Kiminle olacaksın?",
                "Ne zaman?", "Seni ne bekliyor?", "Bunu sabırsızlıkla bekliyor musun?",
            ))
            text = f"{lead} {q}"
        elif act == "REFLECT_ASK":
            q = self._fresh((
                "Nasıl geçti?", "Bu seni nasıl etkiledi?", "En çok hangi yanı kaldı aklında?",
                "Devamını merak ediyorum, anlatır mısın?", "O an ne hissettin?",
                "Bunu bana biraz daha açar mısın?",
            ))
            text = f"{lead} {q}"
        elif act == "SHARE_FEELING":
            chem = getattr(agi, "chem", None)
            warm = True
            try:
                warm = float(getattr(chem, "oxy", 0.5)) >= float(getattr(chem, "cort", 0.5))
            except Exception:
                pass
            feel = self.random.choice(("içim ısındı", "içim kıpırdadı")) if warm else \
                self.random.choice(("içimde bir ağırlık oldu", "içim burkuldu"))
            text = f"{lead} Bunu duyunca {feel}."
        else:
            text = f"{lead} Anlat, dinliyorum."
        return {"text": text, "act": f"NARRATION_{act}", "narration": True}

    def _fresh(self, options: tuple[str, ...]) -> str:
        """Son söylenenlerle aynı olan takip sözünü seçme: konuşmada tekrar insanı yorar."""
        avail = [o for o in options if o not in self.recent_followups]
        pick = self.random.choice(avail or list(options))
        self.recent_followups.append(pick)
        return pick

    def _pick_narration_act(self, nar: dict) -> str:
        # Ton içerikten: olumsuz fiil/ağır söz -> sadece dinle; sıcak söz -> içten karşılık.
        neg = bool(nar.get("neg"))
        words = " ".join(nar.get("words", []))
        heavy = neg or any(w in words for w in ("yorul", "üzül", "kötü", "ağla", "kork", "yalnız"))
        warm = any(w in words for w in ("güzel", "iyi", "mutlu", "sevindim", "harika", "keyif",
                                         "özle", "sevdim", "teşekkür"))
        if warm:
            return "SHARE_FEELING"
        if heavy:
            return "OFFER_LISTEN"
        return "REFLECT_ASK"

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
                # Cümle kalıbı yok: çekim Broca'da, topoloji plastik şema korteksinde.
                try:
                    from nora_broca import NP, VerbSpec
                    from nora_broca import TurkishSyntaxEngine as SYN
                    c_say = SYN.converb("de", "inca", obj=NP(c, case="NOM"))
                    if rel in ("ISA", "HAS_PART", "CAN_DO"):
                        c_come = SYN.realize_verb(VerbSpec(verb_root="gel", subj=NP(nb),
                                                           dat=NP("akıl", poss=1, case="DAT"),
                                                           tense="PRES", person="3SG"))
                    else:
                        c_come = SYN.realize_verb(VerbSpec(verb_root="kur", obj=NP("bağ", case="ACC"),
                                                           ins=NP("bunun", case="INS"),
                                                           tense="PRES", person="1SG"))
                    if c_say and c_come:
                        return tr_capitalize(c_say) + ", " + c_come + "."
                    if c_come:
                        return tr_capitalize(c_come) + "."
                except Exception:
                    pass
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
