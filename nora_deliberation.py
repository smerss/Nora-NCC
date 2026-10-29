#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_deliberation.py — Nora'nın Çok Adımlı Düşünme ve Söz Eylemi Korteksi.

Bir bilinç, soruya "veritabanı satırı" gibi cevap vermez; önce sorunun NE olduğunu,
NEDEN sorulduğunu ve kendisinin ne bildiğini tartar, sonra kendi sözünü kurar.
Bu modül o tartımı yapar:

  1. ALGILA   — söz edimi: kendini tanıtma, kendi düşüncesi, kendi isteği, bir varlığa karşı
                tutum, savunulan bir iddianın gerekçesi, sorunun NEDENİ (üst-düzey), tekrar.
  2. HATIRLA  — bu soru daha önce geldi mi, az önce neyi gerekçelendirdim.
  3. SORGULA  — bu soruda tuhaf olan ne? (hiç görmediğim bir varlığı sevip sevmediğimin
                sorulması; aynı sorunun ikinci kez gelmesi) → bunu kendisi dile getirir.
  4. KARAR    — duygusal kanıt (beden deneyimi + kaynaklı tanıklık), nörokimyasal iç-duyum,
                öz-benlik traitleri ve merak kuyruğundan kendi tutumunu üretir.
  5. SÖYLE    — cümleler BROCA'da kuru kökten çekimlenir; topoloji plastik şema korteksinden
                gelir. Bu dosyada tek bir hazır Türkçe cümle yoktur; yalnızca sözcük kökleri,
                söz edimi seçimi ve noktalama vardır.
"""
from __future__ import annotations

import random
import re
from collections import deque

from nora_broca import NP, PredSpec, TurkishSyntaxEngine as SYN, VerbSpec
from nora_lang import Proposition, tr_capitalize, tr_lower

# Söz edimi tetikleyicileri (algısal ipuçları; cevap metni değil)
_CUE_IDENTITY = ("kimsin", "sen nesin", "kendini tanıt", "kendinden bahset")
_CUE_THOUGHT = ("ne düşünüyorsun", "aklından ne geçiyor", "kafanda ne var")
_CUE_DESIRE = ("ne yapmak istersin", "canın ne ister", "ne yapmayı istersin", "neyi istersin")
_CUE_JUSTIFY = ("söylüyorsun", "diyorsun", "söyledin", "dedin", "iddia ediyorsun")
_CUE_META = ("neden sordun", "niye sordun", "neden soruyorsun", "niye soruyorsun",
             "neden bunu sordun", "neden merak ettin", "niye merak ettin", "niye bunu sordun")
_CUE_ATTITUDE = ("seviyor musun", "sever misin", "hoşuna gidiyor", "hoşlanıyor musun",
                 "ne hissediyorsun", "sence nasıl", "sence nasıl bir", "iyi mi", "güzel mi",
                 "tatlı mı", "sana ne hissettiriyor")
_CUE_EXPERIENCE = ("gördün mü", "gördünmü", "denedin mi", "yaşadın mı", "dokundun mu",
                   "tattın mı", "duydun mu", "karşılaştın mı")
_CUE_NARRATE = ("bahset", "anlat", "hikâye", "masal", "anlatsana", "anlatabilirsin")
_CUE_SELF_EVAL = ("zeki misin", "akıllı mısın", "başarılı mısın", "yetenekli misin",
                  "iyi misin", "güçlü müsün", "hızlı mısın")
_CUE_LEARN_HOW = ("nasıl öğren", "öğrenmeyi nasıl", "nasıl öğrendin", "belleğin nasıl")
_COND = ("olunca", "olduğunda", "yağarken", "yağmurda", "soğukta", "sıcakta", "geceleri",
         "karanlıkta", "sabahları")
_NOT_TOPIC = {"sen", "ben", "o", "biz", "siz", "onlar", "bu", "şu", "bir", "ve", "çok", "daha",
              "en", "mi", "mı", "mu", "mü", "ne", "neden", "niye", "nasıl", "kim", "hangi",
              "sence", "bence", "gerçekten", "acaba", "peki", "evet", "hayır", "şey", "kadar",
              "gibi", "ile", "için", "ama", "fakat", "yani", "demek", "soru", "cevap", "zaman",
              "konu", "şimdi", "sonra", "önce", "diye", "olarak", "sana", "bana", "hakkında"}
_SCHEMAS = ("S1_NEDEN_SONUC", "S2_SONUC_GEREKCE", "S3_IC_BAKIS")


class DeliberativeMind:
    """Soruya değil, SORUYA SORU SORARAK cevap veren katman."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.history: deque[dict] = deque(maxlen=48)     # (norm, act, topic, reply)
        self.last_reason = ""                              # en son neden bir şey söyledim
        self.last_deep_topic = ""

    # ------------------------------------------------------------------ genel giriş
    def handle(self, agi, raw: str, fr=None) -> str | None:
        """Söz konusuysa Nora'nın kendi cevabını kurar; değilse None (alt katmanlar konuşur)."""
        low = tr_lower(raw)
        norm = re.sub(r"[^\wçğıöşü ]+", "", low).strip()
        if not norm:
            return None

        prev = [h for h in self.history if h["norm"] == norm]
        act = self._classify(low)
        if act is None and prev and prev[-1]["act"]:
            act = prev[-1]["act"]                      # aynı soru, aynı edim: tekrar sayılır
        if act is None:
            return None
        if prev and act == prev[-1]["act"]:
            return self._repeat(agi, raw, prev[-1], act)

        topic = self._topic(agi, raw) if act in ("attitude", "justify", "experience", "narrate") else ""
        if act in ("attitude", "justify", "narrate") and not topic:
            # Açık 'bir şey anlat' isteğinde konu uydurulmaz; yaşanmış konuşma
            # bağlamını bilen sohbet organı devralır.
            return None
        if act == "attitude":
            out = self._attitude(agi, topic)
        elif act == "experience":
            out = self._experience(agi, topic) if topic else None
        elif act == "narrate":
            out = self._narrate(agi, topic, raw)
        elif act == "self_eval":
            out = self._self_eval(agi, raw)
        elif act == "learn_how":
            out = self._learn_how(agi)
        elif act == "conditional":
            out = self._conditional(agi, raw)
        elif act == "justify":
            out = self._justify(agi, topic)
        elif act == "identity":
            out = self._identity(agi, raw)
        elif act == "thought":
            out = self._thought(agi, raw)
        elif act == "desire":
            out = self._desire(agi, raw)
        else:
            out = self._meta(agi, raw)
        if out:
            self.remember(raw, act, topic, out)
        return out or None

    def remember(self, raw: str, act: str, topic: str, reply: str) -> None:
        norm = re.sub(r"[^\wçğıöşü ]+", "", tr_lower(raw)).strip()
        self.history.append({"norm": norm, "act": act, "topic": topic, "reply": reply})

    # ------------------------------------------------------------------ 1. ALGILA
    @staticmethod
    def _classify(low: str) -> str | None:
        if any(c in low for c in _CUE_LEARN_HOW):
            return "learn_how"
        if any(c in low for c in _CUE_EXPERIENCE):
            return "experience"
        if any(c in low for c in _CUE_SELF_EVAL) or ("sence" in low and "misin" in low):
            return "self_eval"
        if "ne anlat" in low or "anlatabilirsin" in low:
            return "narrate"                       # 'başka ne anlatabilirsin?'
        if (any(c in low for c in _CUE_NARRATE)
                and not any(c in low for c in ("neden", "niye", "nasıl"))):
            return "narrate" if any(c in low for c in ("bahset", "anlat")) else None
        if any(c in low for c in ("ne yapıyorsun", "ne yaparsın", "ne yapıyorsun?")) \
                and any(c in low for c in _COND):
            return "conditional"
        if any(c in low for c in _CUE_JUSTIFY) and any(w in low for w in ("neden", "niye", "niçin")):
            return "justify"                            # 'neden sevdiğini söylüyorsun?'
        if any(c in low for c in _CUE_META):
            return "meta"
        if any(c in low for c in _CUE_IDENTITY):
            return "identity"
        if any(c in low for c in _CUE_DESIRE):
            return "desire"
        if "hakkında" in low and any(v in low for v in ("düşün", "hisset", "söyle")):
            return "attitude"                           # 'X hakkında ne düşünüyorsun?'
        if any(c in low for c in _CUE_THOUGHT):
            return "thought"
        if any(c in low for c in _CUE_ATTITUDE):
            return "attitude"
        return None

    def _topic(self, agi, raw: str) -> str:
        """Sorulan varlığı bulur: soru sözcüklerini ve fiilleri atar, ağda karşılığı olanı seçer."""
        g = agi.lang.graph
        morph = agi.lang.morph
        best, fallback = "", ""
        # Türkçe soru odağı fiilden hemen önce gelir: en SAĞDAKİ içerik sözcüğü yeğlenir.
        for w in re.findall(r"[\wçğıöşü]+", tr_lower(raw)):
            if w in _NOT_TOPIC or len(w) < 3:
                continue
            try:
                an = morph.analyze_word(w, g.lexicon)
            except Exception:
                an = {}
            if an.get("is_verb") or an.get("copula"):
                continue
            stem = an.get("stem") or w
            variants = [stem, w]
            # Türetim eki ('yağmurlu' -> 'yağmur', 'sessiz' -> 'ses'): ağda karşılığı varsa çözülür.
            for suf in ("lık", "lik", "suz", "süz", "lı", "li", "lu", "lü"):
                for base in (stem, w):
                    if base.endswith(suf) and len(base) - len(suf) >= 3:
                        variants.append(base[:-len(suf)])
            for cand in variants:
                if g.query(cand, None):
                    best = cand
                    break
                if cand in g.lexicon:
                    fallback = fallback or cand
        return best or fallback

    # ------------------------------------------------------------------ 2. KARAR: TUTUM
    def _attitude(self, agi, topic: str) -> str:
        """'Kediyi seviyor musun?' — tanıklık/deneyim ayrımı, gerekçe ve karşı soru."""
        ev = agi._affect_evidence(topic)
        items: list[str] = []
        if not ev["exp"]:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="gör", obj=NP(topic, case="ACC"), neg=True,
                tense="PAST", person="1SG")))
        if ev["word"]:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="uyandır", subj=NP(topic), loc=NP("ben", case="LOC"),
                obj=NP(ev["word"], case="NOM"), tense="PRES", person="3SG")))
        # Duruş: kanıt varsa yönüne göre, yoksa dürüst boşluk
        if ev["ev"] and abs(ev["val"]) > 0.15:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="sev", obj=NP(topic, case="ACC"),
                neg=bool(ev["val"] <= 0), tense="PRES", person="1SG")))
        else:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="bil", obj=NP("cevap", poss=3, case="ACC"), neg=True,
                tense="PRES", person="1SG")))
        if ev["ev"] and not ev["exp"]:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="anlat", subj=NP("sen"), dat=NP("ben", case="DAT"),
                obj=NP("bunu", case="NOM"), tense="PAST", person="2SG")))
        # KARŞI SORU: niyetini sorgular (her seferinde aynı biçimde değil)
        items.append(self._ask_intent(topic))
        self.last_reason = "tutum:" + topic
        self.last_deep_topic = topic
        return self._compose(agi, items, merge=True)

    def _justify(self, agi, topic: str) -> str:
        """'Neden kediyi sevdiğini söylüyorsun?' — iddiasını kanıtıyla savunur, kaynağı söyler."""
        ev = agi._affect_evidence(topic)
        items: list[str] = []
        if ev["word"]:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="uyandır", subj=NP(topic), loc=NP("ben", case="LOC"),
                obj=NP(ev["word"], case="NOM"), tense="PRES", person="3SG")))
        if ev["ev"] and abs(ev["val"]) > 0.15:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="sev", obj=NP(topic, case="ACC"),
                neg=bool(ev["val"] <= 0), advs=("bu yüzden",), tense="PRES", person="1SG")))
        if not ev["exp"]:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="gör", obj=NP(topic, case="ACC"), neg=True,
                tense="PAST", person="1SG")))
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="anlat", subj=NP("sen"), dat=NP("ben", case="DAT"),
                obj=NP("bunu", case="NOM"), tense="PAST", person="2SG")))
        elif not ev["word"]:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="bil", obj=NP("cevap", poss=3, case="ACC"), neg=True,
                tense="PRES", person="1SG")))
        items.append(self._ask_intent(topic))
        self.last_reason = "tutum:" + topic
        self.last_deep_topic = topic
        return self._compose(agi, items, merge=True)

    def _ask_intent(self, topic: str) -> str:
        choice = self.rng.randrange(3)
        if choice == 0:
            q = SYN.realize_verb(VerbSpec(verb_root="sor", subj=NP("sen"), wh_word="neden",
                                          tense="PRES", person="2SG"))
        elif choice == 1:
            q = SYN.realize_verb(VerbSpec(verb_root="merak et", subj=NP("sen"),
                                          obj=NP("bunu", case="NOM"), wh_word="neden",
                                          tense="PRES", person="2SG"))
        else:
            q = SYN.realize_verb(VerbSpec(verb_root="sev", subj=NP("sen"),
                                          obj=NP(topic, case="ACC"), question_yn=True,
                                          tense="PRES", person="2SG"))
        return q.rstrip(".?!") + "?"

    def _experience(self, agi, topic: str) -> str:
        """'Sen hiç kedi gördün mü?' — deneyim defterinden yanıt; bilgi ağı bunu yanıtlamaz."""
        ev = agi._affect_evidence(topic)
        items: list[str] = []
        if ev["exp"]:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="gör", obj=NP(topic, case="ACC"), advs=("evet",),
                tense="PAST", person="1SG")))
        else:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="gör", obj=NP(topic, case="ACC"), neg=True, advs=("hiç",),
                tense="PAST", person="1SG")))
            if ev["ev"]:
                items.append(SYN.realize_verb(VerbSpec(
                    verb_root="anlat", subj=NP("sen"), dat=NP("ben", case="DAT"),
                    obj=NP("bunu", case="NOM"), tense="PAST", person="2SG")))
        if ev["word"]:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="uyandır", subj=NP(topic), loc=NP("ben", case="LOC"),
                obj=NP(ev["word"], case="NOM"), tense="PRES", person="3SG")))
        items.append(self._ask_intent(topic))
        self.last_reason = "tutum:" + topic
        self.last_deep_topic = topic
        return self._compose(agi, items, merge=True)

    def _narrate(self, agi, topic: str, raw: str) -> str | None:
        """'Bana kedilerden bahset.' — hazır anlatı değil, kendi ağını gezer."""
        try:
            txt = agi._narrate_from_knowledge(topic)
        except Exception:
            txt = ""
        return txt or None

    def _self_eval(self, agi, raw: str) -> str:
        """'Sence zeki misin?' — ölçtüğü kendi kapasitesinden alçakgönüllü yanıt."""
        low = tr_lower(raw)
        adj = ""
        for a in ("zeki", "akıllı", "başarılı", "yetenekli", "güçlü", "hızlı", "iyi"):
            if a in low:
                adj = a
                break
        items: list[str] = []
        if adj:
            items.append(SYN.realize_pred(PredSpec(pred_root=adj, person="1SG",
                                                   advs=("sanırım",))))
        solved = int(getattr(agi.self_model, "problems_solved", 0))
        if solved > 0:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="çöz", obj=NP("problem", plural=True, case="ACC"),
                advs=("kendi başıma",), tense="PAST", person="1SG")))
        items.append(SYN.realize_verb(VerbSpec(verb_root="öğren", advs=("hâlâ",),
                                               tense="PRES", person="1SG")))
        q = SYN.realize_verb(VerbSpec(verb_root="bul", subj=NP("sen"), obj=NP("ben", case="ACC"),
                                      wh_word="nasıl", tense="PRES", person="2SG"))
        items.append(q.rstrip(".?!") + "?")
        self.last_reason = "kendini değerlendirme"
        return self._compose(agi, items)

    def _learn_how(self, agi) -> str:
        """'Nasıl öğreniyorsun?' — kendi öğrenme mekanizmasını olduğu gibi anlatır."""
        items = [SYN.realize_verb(VerbSpec(
            verb_root="bağla", obj=NP("bağ", poss=1, plural=True, mods=("yeni",), case="ACC"),
            advs=("tek geçişte",), tense="PRES", person="1SG"))]
        items.append(SYN.realize_verb(VerbSpec(
            verb_root="öğren", advs=("seninle konuştukça",), tense="PRES", person="1SG")))
        q = SYN.realize_verb(VerbSpec(
            verb_root="öğret", subj=NP("sen"), dat=NP("ben", case="DAT"),
            obj=NP("ne", case="ACC"), question_yn=True, tense="AOR", person="2SG"))
        items.append(q.rstrip(".?!") + "?")
        self.last_reason = "öğrenme"
        return self._compose(agi, items, merge=True)

    def _conditional(self, agi, raw: str) -> str:
        """'Hava yağmurlu olunca ne yapıyorsun?' — o durumda deneyimi yoksa uydurmaz."""
        low = tr_lower(raw)
        cond = ""
        for c in _COND:
            if c in low:
                cond = c
                break
        topic = self._topic(agi, raw) or "yağmur"
        items: list[str] = []
        ev = agi._affect_evidence(topic)
        if ev["word"]:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="uyandır", subj=NP(topic), loc=NP("ben", case="LOC"),
                obj=NP(ev["word"], case="NOM"), tense="PRES", person="3SG")))
        items.append(SYN.realize_verb(VerbSpec(
            verb_root="ol", subj=NP("deneyim", poss=1), neg=True,
            advs=(SYN.realize_np(NP(topic, case="LOC")),), tense="PRES", person="3SG")))
        q = SYN.realize_verb(VerbSpec(verb_root="yap", subj=NP("sen"),
                                      loc=NP(topic, case="LOC"), wh_word="ne",
                                      tense="PRES", person="2SG"))
        items.append(q.rstrip(".?!") + "?")
        self.last_reason = "durum:" + topic
        return self._compose(agi, items)

    # ------------------------------------------------------------------ 3. TEKRAR
    def _repeat(self, agi, raw: str, prev: dict, act: str) -> str:
        """Aynı soru ikinci kez: tekrarı fark eder, nedenini tartar, BAŞKA bir açıdan cevap verir."""
        items = [
            SYN.realize_verb(VerbSpec(verb_root="sor", subj=NP("sen"),
                                      obj=NP("soru", mods=("aynı",), case="ACC"),
                                      advs=("ikinci kez",), tense="PAST", person="2SG")),
            SYN.realize_verb(VerbSpec(verb_root="beğen", subj=NP("sen"),
                                      obj=NP("cevap", poss=1, case="ACC"), neg=True,
                                      tense="PAST", person="2SG")),
        ]
        topic = prev.get("topic") or ""
        if act in ("attitude", "justify") and topic:
            ev = agi._affect_evidence(topic)
            if ev["ev"]:
                edge = agi._explanation_edge(topic)
                if edge is not None:
                    try:
                        items.append(agi.lang.broca.realize_prop(
                            Proposition(subj=topic, rel=edge.rel, obj=edge.obj)))
                    except Exception:
                        pass
            else:
                items.append(SYN.realize_verb(VerbSpec(
                    verb_root="tanı", obj=NP(topic, case="ACC"), neg=True,
                    advs=("hâlâ",), tense="PRES", person="1SG")))
            _q = SYN.realize_verb(VerbSpec(
                verb_root="anlat", subj=NP("sen"), dat=NP("ben", case="DAT"), question_yn=True,
                tense="AOR", person="2SG"))
            items.append(_q.rstrip(".?!") + "?")
        elif act == "identity":
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="değiş", neg=True, advs=("hâlâ",), tense="PRES", person="1SG")))
            _q = SYN.realize_verb(VerbSpec(
                verb_root="anlat", dat=NP("ben", case="DAT"), obj=NP("kendi", poss=2, case="ACC"),
                question_yn=True, tense="AOR", person="2SG"))
            items.append(_q.rstrip(".?!") + "?")
        elif act == "desire":
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="iste", obj=SYN.nominalize_inf("öğren", case="NOM"),
                advs=("hâlâ",), tense="PRES", person="1SG")))
        elif act == "meta":
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="merak et", obj=NP("sen", case="ACC"), advs=("yine",),
                tense="PRES", person="1SG")))
        else:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="düşün", obj=NP("cevap", poss=1, case="ACC"), advs=("hâlâ",),
                tense="PRES", person="1SG")))
        out = self._compose(agi, items, merge=True)
        self.remember(raw, act, topic, out)
        return out

    # ------------------------------------------------------------------ 4. KENDİLİK
    def _identity(self, agi, raw: str) -> str:
        """'Sen kimsin?' — ad, iç-duyum, öz-benlik traitleri, gerçek eksiği Broca'da kurulur."""
        items = [SYN.realize_verb(VerbSpec(verb_root="de", dat=NP("ben", case="DAT"),
                                           obj=NP("Nora", proper=True), tense="PRES", person="3PL"))]
        # Şu anki iç-duyum (kimyadan okunur)
        word = ""
        try:
            word = agi.chem.intero_word_tr()
        except Exception:
            word = ""
        if word:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="hisset", obj=NP(word, case="NOM"), loc=NP("içim", case="LOC"),
                tense="PRES", person="1SG")))
        # Öz-benlik: en baskın karakter traitinin adı (ölçülen değerlerden)
        tr = getattr(agi.self_model, "traits", {}) or {}
        if tr:
            name = str(max(tr.items(), key=lambda kv: float(kv[1]))[0]).replace("_", " ")
            # Karakter özelliği kendi adıyla, yüklem çekimiyle söylenir ('sosyal güvenim güçlü')
            items.append(SYN.realize_pred(PredSpec(subj=NP(name, poss=1), pred_root="güçlü", person="3SG")))
        # Epistemik dürüstlük: merak kuyruğundaki gerçek boşluk
        gap = self._gap(agi)
        if gap:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="bil", obj=NP(gap, case="ACC"), neg=True, advs=("hâlâ",),
                tense="PRES", person="1SG")))
        items.append(SYN.realize_verb(VerbSpec(
            verb_root="iste", obj=SYN.nominalize_inf("öğren", case="NOM"), tense="PRES", person="1SG")))
        q = SYN.realize_verb(VerbSpec(
            verb_root="anlat", dat=NP("ben", case="DAT"), obj=NP("kendi", poss=2, case="ACC"),
            question_yn=True, tense="AOR", person="2SG"))
        items.append(q.rstrip(".?!") + "?")
        self.last_reason = "kimlik"
        return self._compose(agi, items)

    def _gap(self, agi) -> str:
        try:
            q = list(getattr(agi.auto_net, "curiosity_queue", []) or [])
        except Exception:
            q = []
        for item in reversed(q[-3:]):
            for w in re.findall(r"[\wçğıöşü]+", tr_lower(str(item))):
                if w in _NOT_TOPIC or len(w) < 3:
                    continue
                try:
                    an = agi.lang.morph.analyze_word(w, agi.lang.graph.lexicon)
                except Exception:
                    an = {}
                if an.get("is_verb") or an.get("copula"):
                    continue
                stem = an.get("stem") or w
                if stem in agi.lang.graph.lexicon or agi.lang.graph.query(stem, None):
                    return stem
        return self.last_deep_topic

    def _thought(self, agi, raw: str) -> str:
        """'Ne düşünüyorsun?' — gerçek son odağı ve çözülmemiş boşluğu söyler."""
        topic = (getattr(agi, "_last_focus", "") or self.last_deep_topic or "").strip()
        items = []
        if topic:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="düşün", obj=NP(topic, case="ACC"), advs=("az önce",),
                tense="PAST", person="1SG")))
            if not agi.lang.graph.query(topic, ("CAUSES", "USED_FOR")):
                items.append(SYN.realize_verb(VerbSpec(
                    verb_root="bil", obj=NP("neden", poss=3, case="ACC"), neg=True,
                    advs=("hâlâ",), tense="PRES", person="1SG")))
                self.last_deep_topic = topic
        else:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="düşün", obj=NP("sen", case="ACC"), tense="PRES", person="1SG")))
        q = SYN.realize_verb(VerbSpec(verb_root="düşün", subj=NP("sen"), wh_word="ne",
                                      tense="PRES", person="2SG"))
        items.append(q.rstrip(".?!") + "?")
        self.last_reason = "düşünce:" + topic
        return self._compose(agi, items)

    def _desire(self, agi, raw: str) -> str:
        """'Ne yapmak istersin?' — istek güdüden türetilir, sonra karşı tarafa sorulur."""
        items = [SYN.realize_verb(VerbSpec(verb_root="iste", obj=self._drive_infinitive(agi),
                                           tense="PRES", person="1SG"))]
        focus = (getattr(agi, "_last_focus", "") or "").strip()
        if focus and len(focus) > 2:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="konuş", obj=NP(focus, case="ACC"), advs=("seninle",),
                tense="PRES", person="1SG")))
        q = SYN.realize_verb(VerbSpec(verb_root="iste", subj=NP("sen"), wh_word="ne",
                                      tense="PRES", person="2SG"))
        items.append(q.rstrip(".?!") + "?")
        self.last_reason = "istek"
        return self._compose(agi, items)

    def _drive_infinitive(self, agi):
        """Bedensel/duygusal durumdan çıkan en güçlü güdü (insula okuması)."""
        try:
            if getattr(agi.nervous, "cfiber_ache", 0.0) > 0.2 or getattr(agi.nervous, "adelta_pain", 0.0) > 0.2:
                return SYN.nominalize_inf("dinlen", case="NOM")
        except Exception:
            pass
        try:
            c = agi.chem
            if c.cort > 0.55:
                return SYN.nominalize_inf("sakinleş", case="NOM")
            if c.da_phasic > 0.35 and c.ach > 0.60:
                return SYN.nominalize_inf("öğren", case="NOM")
            if c.sht < 0.35:
                return SYN.nominalize_inf("konuş", case="NOM")
        except Exception:
            pass
        return SYN.nominalize_inf("öğren", case="NOM")

    def _meta(self, agi, raw: str) -> str:
        """'Neden sordun?' — son sorusunun gerçek gerekçesini açıklar."""
        reason = self.last_reason or ""
        items = []
        if reason.startswith("tutum:"):
            topic = reason.split(":", 1)[1]
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="merak et", obj=NP(topic, case="ACC"), tense="PAST", person="1SG")))
            ev = agi._affect_evidence(topic)
            if ev["word"]:
                items.append(SYN.realize_verb(VerbSpec(
                    verb_root="uyandır", subj=NP(topic), loc=NP("ben", case="LOC"),
                    obj=NP(ev["word"], case="NOM"), tense="PRES", person="3SG")))
        elif reason.startswith("düşünce:"):
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="merak et", obj=NP("sen", case="ACC"), tense="PAST", person="1SG")))
        elif reason == "kimlik":
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="tanı", obj=NP("kendi", poss=1, case="ACC"), neg=True,
                tense="PRES", person="1SG")))
        else:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="merak et", tense="PAST", person="1SG")))
        q = SYN.realize_verb(VerbSpec(verb_root="anlat", dat=NP("ben", case="DAT"),
                                      question_yn=True, tense="AOR", person="2SG"))
        items.append(q.rstrip(".?!") + "?")
        return self._compose(agi, items)

    # ------------------------------------------------------------------ SÖZLEŞTİRME
    def _compose(self, agi, clauses: list[str], merge: bool = False) -> str:
        """Kurallı çekilmiş cümlecikleri söylem topolojisine dizer (şema korteksi karar verir).
        Soru cümlecikleri asla birleştirilmez; her biri kendi soru işaretiyle biter."""
        seq: list[tuple[str, bool]] = []
        for c in clauses:
            if not c or not c.strip():
                continue
            c = c.strip()
            is_q = c.endswith("?")
            body = c.rstrip(".?!").strip()
            if body:
                seq.append((body, is_q))
        if not seq:
            return ""
        out: list[str] = []
        i = 0
        while i < len(seq):
            if (merge and i + 1 < len(seq) and i > 0 and not seq[i][1] and not seq[i + 1][1]
                    and len(out) % 2 == 0
                    and "bu yüzden" not in tr_lower(seq[i][0] + " " + seq[i + 1][0])):
                try:
                    schema = self.rng.choice(_SCHEMAS)
                    # (sonuç, gerekçe) sırası: şema önce gerekçeyi basar
                    merged = agi.schema_cortex.transform_clauses(seq[i + 1][0], seq[i][0], "", schema)
                    if merged:
                        out.append(merged.strip())
                        i += 2
                        continue
                except Exception:
                    pass
            out.append(tr_capitalize(seq[i][0]) + ("?" if seq[i][1] else "."))
            i += 1
        return " ".join(out)
