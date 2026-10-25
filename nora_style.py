"""NORA — SÖZ BİÇİMİ: aynı kalıba düşmemek, kendi derdini öne almak.

Sorun neydi?
------------
Nora her soruya aynı iskeletle cevap veriyordu:

    "Su yararlıdır; suyu içerim. Sen su içer misin?"

Üç parça: iddia + gerekçe + karşı soru. Hep aynı sıra, hep aynı yer. Bu, insan
konuşmasına benzemez; insan bazen tek kelime eder, bazen susar, bazen aklına geleni
söyler, bazen karşısındakine sormaz — çünkü derdi illa ona hizmet etmek değildir.

Bu organ kararı iç durumdan verir:

* **Nörokimya**: dopamine (merak) yüksekse soru sormaya meyilli; serotonin düşükse
  kısa ve kuru; kortizol yüksekse savunmacı.
* **Beden**: enerji/su/bütünlük azaldıysa KENDİ DERDİ öne geçer. Tok bir Nora
  konuşkan; aç bir Nora kısa konuşur ve kendi ihtiyacını söyler.
* **Sıra**: üst üste iki turda soru sorduysa üçüncüde sormaz. Aynı biçim üç kez
  tekrarlanmaz (monotonluk hatası).
* **İç ses**: az önce kendi çıkarımını yaptıysa onu söze katar — konuşma yalnızca
  gelen mesaja verilen cevap olmaktan çıkar.

Biçimler
--------
    cevap          yalnızca cevap (kısa)
    cevap+düşünce  cevap + kendi iç sesinden bir cümle
    cevap+soru     cevap + konuya özgü soru (her turda değil)
    kendine        önce kendi kendine mırıldanır, sonra cevap
    parça          tek cümlelik kısa cevap (uzun cevabın özü)
    ihtiyaç        kendi bedensel ihtiyacını söyler, sonra (belki) cevap
    sessizlik      yalnızca "…" (çok seyrek; derin iç odakta)

Dürüst sınır: bu bir "kişilik simülatörü" değildir; seçimler ölçülen iç durumdan
ve geçmiş söz sırasından çıkar. Aynı durum aynı biçimi verir — rastgele süs değil.
"""

from __future__ import annotations

import os
import re
from collections import Counter, deque

from nora_lang import tr_lower, tr_capitalize

# Karşı tarafa sorulan "nezaket soruları": bunlar bilgi boşluğundan doğmaz, alışkanlıktır.
_SOCIAL_ASK = re.compile(
    r"(hakkında\s+ne\s+düşünüyorsun"
    r"|(anlatır|söyler|gösterir|öğretir|anlatır)\s+mısın"
    r"|\bsever\s+misin\b|\bseviyor\s+musun\b"
    r"|\bgünün\s+nasıl|\bkeyfin\b|\bmoralin\b"
    r"|\bsen\s+ne\s+düşünüyorsun|\bsen\s+olsan\b"
    r"|\bbugün\s+ne\s+oldu\b)"
)

SHAPES = ("cevap", "cevap+düşünce", "cevap+soru", "kendine", "parça", "ihtiyaç", "sessizlik")

# Kulağın ilk tepkisi ve iç mırıldanma cevabın GÖVDESİNDEN önce gelebilir. Gövdeyi
# ölçmek isteyen yerler (selftest, kıstas) bu yardımcıyı kullanır.
_PERCEPTION_NOTES = ("Sesini ilk kez duyuyorum.", "Sesin uzaktan geliyor.", "Seni tam seçemedim.")


def strip_prefixes(text: str) -> str:
    """Algı notunu ve baştaki iç mırıldanmayı ayırır; geriye cevabın kendisi kalır."""
    t = (text or "").strip()
    changed = True
    while changed:
        changed = False
        for note in _PERCEPTION_NOTES:
            if t.startswith(note):
                t = t[len(note):].strip()
                changed = True
        m = re.match(r"^İçimden '[^']{1,80}' diye geçirdim\.\s*", t)
        if m:
            t = t[m.end():].strip()
            changed = True
    return t


class SpeechStyle:
    """Biçim kararı: iç durum + söz geçmişi. Tur başına O(1)."""

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self.history: deque[str] = deque(maxlen=10)
        self.turn = 0
        self.asked_recently = 0     # üst üste soruyla biten tur sayısı
        self.shapes_used: Counter[str] = Counter()
        self.silences = 0
        self._last_need = -999
        self._said: deque[str] = deque(maxlen=24)

    # ---------------------------------------------------------------- karar
    def decide(self, agi, *, is_question: bool, has_inference: bool = False,
               trust: float = 0.6, direct: bool = True) -> str:
        """Bu turun biçimini seçer. Dönen değer `SHAPES` içinden biridir.

        `direct=False` yalnızca Nora KENDİ KENDİNE konuşurken kullanılır (o zaman
        susabilir). Bir insan ona seslendiyse her zaman bir cevap vardır.
        """
        self.turn += 1
        chem = agi.chem
        da = float(getattr(chem, "da_tonic", 0.5)) + max(0.0, float(getattr(chem, "da_phasic", 0.0)))
        sht = float(getattr(chem, "sht", 0.65))
        cort = float(getattr(chem, "cort", 0.1))
        deficit = self._deficit(agi)
        idle_need = (self.turn - self._last_need) > 6

        # 0) KETUMLUK: söz planı "guarded" ise Nora süs yapmaz — ne mırıldanma, ne karşı
        #    soru, ne iç ses. Ketum insan kısa konuşur.
        try:
            _plan = dict(getattr(getattr(agi, "broca_organ", None), "last_speech_plan", {}) or {})
        except Exception:
            _plan = {}
        if _plan.get("guarded") and not _plan.get("ask_back_chosen"):
            return "parça" if (not is_question and os.urandom(1)[0] % 2) else "cevap"

        # 1) SES: aynı biçim üst üste İKİ kez gelmez. Üç tur aynı iskelet, insanın
        #    "acaba iskelet mi?" demesinin ta kendisiydi.
        hist = list(self.history)
        banned = {hist[-1]} if len(hist) >= 2 and hist[-1] == hist[-2] else set()
        allow_q = self.asked_recently < 2

        options: list[tuple[str, int]] = []

        def add(shape: str, weight: int) -> None:
            if weight > 0:
                options.append((shape, weight))

        # 2) Bedenin derdi varsa kendi ihtiyacı öne geçer (karşı tarafa hizmet değil).
        if direct and is_question and deficit > 0.55 and idle_need and trust < 0.72:
            add("ihtiyaç", 45)
        # 3) Kendi çıkarımını söze katma (dopamin = merak; çıkarım taze).
        if has_inference and da > 0.50:
            add("cevap+düşünce", 40)
        # 4) Soru sorma: yalnızca merak yüksek ve üst üste soru sorulmadıysa.
        if allow_q and is_question and da > 0.52 and sht > 0.35:
            add("cevap+soru", 34)
        # 5) Kendi kendine mırıldanma: iç ses canlıysa.
        if sht > 0.45 and cort < 0.55:
            add("kendine", 22)
        # 6) Kısa kesme: yalnızca anlatı/anlatım turlarında.
        if not is_question:
            add("parça", 16)
        # 7) Susma: yalnızca kimse sormadığında ve iç odak derinse (çok seyrek).
        if (not direct) and deficit < 0.35 and self.silences < 2:
            add("sessizlik", 4)
        # 8) Yalın cevap: varsayılan; her cümleye soru eklemek yok.
        add("cevap", 30)

        pool = [(sh, w) for sh, w in options if sh not in banned]
        if not pool:
            # Yasaklı biçimden başka seçenek yoksa yalın cevap değil, en yakın farklı biçim.
            pool = [(sh, w) for sh, w in options if sh != "cevap"] or options
        total = sum(w for _, w in pool) or 1
        # Bireysellik: seçim donanım entropisinden de beslenir (aynı seed aynı kararı vermez).
        roll = (int.from_bytes(os.urandom(4), "big") / 2 ** 32) * total
        acc = 0.0
        for sh, w in pool:
            acc += w
            if roll <= acc:
                return sh
        return pool[-1][0]

    def _deficit(self, agi) -> float:
        """Bedenin ne kadar 'derdi var' (0..1). Duyumdan okunur, sayı ekranda görünmez."""
        try:
            nv = agi.nervous
            # Sürekli sürüşler (0..1): açlık, susuzluk, A-delta acı, C-lifi sızısı.
            drives = [float(getattr(nv, "hunger_drive", 0.0) or 0.0),
                      float(getattr(nv, "thirst_drive", 0.0) or 0.0),
                      float(getattr(nv, "adelta_pain", 0.0) or 0.0),
                      float(getattr(nv, "cfiber_ache", 0.0) or 0.0)]
            return max(0.0, min(1.0, max(drives)))
        except Exception:
            return 0.0

    # ---------------------------------------------------------------- uygulama
    def apply(self, ans: str, shape: str, agi, *, topic: str = "", is_question: bool = False) -> str:
        """Seçilen biçimi metne uygular. Anlamı bozmaz: ilk cümle (cevap) korunur."""
        text = (ans or "").strip()
        if not text:
            return text
        self.history.append(shape)
        self.shapes_used[shape] += 1

        if shape == "sessizlik":
            return "…"

        # 1) Nezaket sorusunu at (yalnızca *son* cümle ve yalnızca sosyal kalıpsa)
        if shape in ("cevap", "cevap+düşünce", "parça", "ihtiyaç") and self.asked_recently >= 1:
            text = self._strip_social_ask(text)

        if shape == "parça":
            # Kısa cevabı daha da kısaltmak anlamı siler ("Çok geçmiş olsun." gibi tek cümle
            # zaten öz). Yalnızca UZUN ve çok cümleli cevaplar kısaltılır.
            parts = re.split(r"(?<=[.!?])\s+", text.strip())
            first = parts[0].strip()
            if len(parts) >= 3 and len(first) >= 45:
                text = first

        if shape == "kendine":
            mutter = self._own_mutter(agi)
            if mutter:
                # Mırıldanma her zaman ÖNCE gelmez: bazen cevabı söyler, sonra içinden geçirir.
                # Sabit bir sıra (hep mırıldan→cevap) da iskelettir.
                if os.urandom(1)[0] % 2:
                    text = f"{mutter} {tr_capitalize(text[0]) + text[1:] if text else text}"
                else:
                    text = f"{text.rstrip()} {mutter}"

        if shape == "cevap+düşünce":
            thought = self._own_thought(agi, text=text, topic=topic)
            if thought:
                text = f"{text.rstrip()} {thought}"

        if shape == "ihtiyaç":
            need = self._own_need(agi)
            if need:
                self._last_need = self.turn
                text = f"{need} {text}"

        if shape == "cevap+soru":
            q = self._topic_question(agi, topic)
            if q:
                text = f"{text.rstrip()} {q}"

        if text.rstrip().endswith("?"):
            self.asked_recently += 1
        else:
            self.asked_recently = 0
        return text.strip()

    # ---------------------------------------------------------------- parçalar
    @staticmethod
    def _first_sentence(text: str) -> str:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return parts[0] if parts else text

    @staticmethod
    def _strip_social_ask(text: str) -> str:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        if len(parts) >= 2 and parts[-1].rstrip().endswith("?"):
            if _SOCIAL_ASK.search(tr_lower(parts[-1])):
                return " ".join(parts[:-1]).strip()
        return text

    def _own_thought(self, agi, *, text: str = "", topic: str = "") -> str:
        """Kendi iç sesinden bir cümle — ama yalnızca KONUYA değiyorsa.

        İlgisiz bir türetimi araya sıkıştırmak, kalıp cevap kadar yapay durur; üstelik
        aynı cümleyi tekrar tekrar söylemek iskeleti ele verir. Çıkarım bir kez söylenir.
        """
        try:
            inf = getattr(agi, "infer", None)
            if inf is not None:
                context = f"{text} {topic}".strip()
                line = inf.last_line(agi, topic=context)
                if line:
                    # Çıkarım motoru cümleyi kendi başlangıcıyla verir; üstüne bir de
                    # "şunu fark ettim" eklenirse "şunu fark ettim: kendi kendime şunu
                    # çıkardım:" gibi çift başlangıç olur.
                    _lead = self._lead_in()
                    for _pre in ("Kendi kendime şunu çıkardım:", "Kendi kendime şunu çıkardım",
                                 "İçimden şu geçti:"):
                        if line.startswith(_pre):
                            line = line[len(_pre):].strip()
                            break
                    return f"{_lead} {line}"
            tw = {w for w in re.split(r"[^a-zçğıöşüâîû]+", tr_lower(f"{text} {topic}")) if len(w) >= 4}
            if not tw:
                return ""
            for th in reversed(list(getattr(agi.monologue, "stream", []) or [])[-12:]):
                body = tr_lower(getattr(th, "text", "") or "")
                if not body or len(body) < 12:
                    continue
                if body in self._said:
                    continue
                if not (tw & {w for w in re.split(r"[^a-zçğıöşüâîû]+", body) if len(w) >= 4}):
                    continue
                self._said.append(body)
                lead = self._lead_in()
                return f"{lead} {body.rstrip('.')}."
        except Exception:
            pass
        return ""

    def _lead_in(self) -> str:
        """İç sesi söze sokan kalıp her tur aynı olmasın."""
        choices = ("Kendi kendime şunu çıkardım:", "İçimden şu geçti:",
                   "Az önce kendi kendime bağladım:", "Şunu fark ettim kendi kendime:")
        return choices[int.from_bytes(os.urandom(1), "big") % len(choices)]

    def _own_mutter(self, agi) -> str:
        """Kendi kendine mırıldanma: beden/kimya durumundan, karşı tarafa değil.

        Aynı iç durum aynı sözcüğü verse bile Nora her tur aynı cümleyi mırıldanmaz:
        son üç turda kullandığı mırıltıyı yeniden kurmaz.
        """
        try:
            chem = agi.chem
            word = chem.intero_word_tr()
        except Exception:
            word = ""
        options = []
        if word:
            options.append(f"İçimden '{word}' diye geçirdim.")
        # Beden/kimya dışındaki iç hâller: son duyum, son düşünce, beden sürüşü.
        try:
            ev = getattr(agi, "last_auditory", None)
            if ev is not None and getattr(ev, "distance_m", 0.0) > 1.5:
                options.append("İçimden 'ses uzaktan gelince içim boş kaldı' diye geçirdim.")
            drives = max(float(getattr(agi.nervous, a, 0.0) or 0.0)
                         for a in ("hunger_drive", "thirst_drive", "adelta_pain", "cfiber_ache"))
            if drives > 0.3:
                options.append("İçimden 'bedenim kendi derdini fısıldıyor' diye geçirdim.")
            if getattr(agi, "infer", None) is not None and getattr(agi.infer, "fresh", False):
                options.append("İçimden 'aklımın bir köşesi yeni bağla meşgul' diye geçirdim.")
        except Exception:
            pass
        if not options:
            return ""
        recent = set(list(self._said)[-3:])
        for opt in options:
            if opt not in recent:
                self._said.append(opt)
                return opt
        return options[0]

    def _own_need(self, agi) -> str:
        """Kendi bedensel derdi — karşı tarafın sorusundan önce gelir."""
        try:
            visc = agi.nervous.felt_visceral_summary()
            for key, line in (("thirst_state", "Suyum azaldı; önce onu düşünmem gerek."),
                              ("hunger_state", "Karnım aç; aklım orada."),
                              ("pain_state", "Canım yanıyor; dikkatim orada.")):
                v = str(visc.get(key, "") or "")
                if v and v not in ("yok", "tok", "None"):
                    return line
        except Exception:
            pass
        return ""

    def _topic_question(self, agi, topic: str) -> str:
        """Konuya özgü soru: bilgi boşluğu varsa onu sorar, yoksa kısa bir merak sorusu."""
        t = (topic or "").strip()
        if not t:
            try:
                q = list(getattr(agi.auto_net, "curiosity_queue", []) or [])
                t = str(q[-1]) if q else ""
            except Exception:
                t = ""
        if not t:
            return ""
        try:
            from nora_broca import TurkishSyntaxEngine as SYN, NP, VerbSpec
            return tr_capitalize(SYN.realize_verb(VerbSpec(
                verb_root="bil", obj=NP(t, case="ACC"), question_wh="ne",
                tense="PRES", person="2SG"))) + "?"
        except Exception:
            return ""

    # ---------------------------------------------------------------- rapor
    def diversity(self) -> dict:
        total = sum(self.shapes_used.values()) or 1
        return {k: round(v / total, 3) for k, v in self.shapes_used.items()}

    def state_dict(self) -> dict:
        return {"turn": self.turn, "asked_recently": self.asked_recently,
                "shapes": dict(self.shapes_used), "silences": self.silences,
                "last_need": self._last_need}

    def load_state_dict(self, d: dict) -> None:
        try:
            self.turn = int(d.get("turn", 0) or 0)
            self.asked_recently = int(d.get("asked_recently", 0) or 0)
            self.shapes_used = Counter({str(k): int(v) for k, v in (d.get("shapes") or {}).items()})
            self.silences = int(d.get("silences", 0) or 0)
            self._last_need = int(d.get("last_need", -999) or -999)
        except Exception:
            pass
