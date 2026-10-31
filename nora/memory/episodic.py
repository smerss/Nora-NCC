"""nora_episodic.py — KİŞİSEL BELLEK: karşıdakinin SÖYLEDİĞİ olgular.

Neden ayrı bir bellek?
----------------------
İki bilgi türü aynı yere yazılırsa biri ötekini kirletir:

    DÜNYA BİLGİSİ  ("kuşlar uçar")        -> kavram ağı (`nora_lang.graph`)
    KİŞİ BİLGİSİ   ("kardeşimin adı Elif") -> BURASI

A54 tam olarak bunu ölçüyor: kişisel anlatı kavram ağına "olgu" diye yazılmamalı.
Ama söyleneni de unutmamalı. Bu yüzden ayrı bir depo: kaynak etiketli, turlu, alıntılı.

Dürüstlük kuralları
-------------------
1. **Yalnız AÇIK kalıplar.** "X'in adı Y", "benim adım Y", "X Y yaşında" gibi. Tahmin
   edilen, çıkarılan, "belki" denen bir olgu yazılmaz (uydurma olur).
2. **Kaynak saklanır.** Her kayıt "kim söyledi, kaçıncı turda, hangi cümlede" bilgisini
   taşır. Sorulduğunda Nora bunu söyleyebilir: "bana söylemiştin".
3. **Çelişki sessizce ezilmez.** Aynı özneye ikinci kez farklı değer gelirse eski kayıt
   silinmez, `onceki` alanına düşer; cevap "önce şöyle demiştin" diyebilir.
4. **Bilinmeyen soruya olgu uydurulmaz.** Depoda yoksa `None` döner; üst katman
   "bilmiyorum" der.
"""
from __future__ import annotations

from nora._paths import REPO_ROOT


import re

from nora_lang import tr_lower

#: Açık olgu kalıpları. Hepsi "özne + ilişki + değer" üretir. Sıra önemli:
#: en özgül kalıp önce denenir.
_PATTERNS = (
    # "kardeşimin adı Elif" / "benim adım Ali" / "adım Ali"
    (re.compile(r"^(?:benim\s+)?ad[ıi]m\s+([A-Za-zÇĞİÖŞÜçğıöşü]{2,20})\b", re.I),
     "benim", "ad"),
    (re.compile(r"^([\wçğıöşü]{2,20}?)(?:'?[ıiuü]n|'?[ıiuü]nın|'?[ıiuü]nın)?\s+ad[ıi]\s+"
                r"([A-ZÇĞİÖŞÜ]{1}[\wçğıöşü]{1,19})\b", re.U),
     None, "ad"),
    # "kedimin adı Pamuk" (küçük harfli değer de kabul: 'elif')
    (re.compile(r"^([\wçğıöşü]{2,20}?)(?:'?[ıiuü]n)?\s+ad[ıi]\s+([\wçğıöşü]{2,20})\b", re.U),
     None, "ad"),
    # "kardeşim 12 yaşında" / "ben 30 yaşındayım"
    (re.compile(r"^([\wçğıöşü]{2,20})\s+(\d{1,3})\s+yaş(?:ında|ındayım|ında)?\b", re.I),
     None, "yaş"),
    # "en sevdiğim renk mavi" / "sevdiğim yemek mantı"
    (re.compile(r"^(?:benim\s+)?en\s+sevdi[ğg]im\s+([\wçğıöşü]{2,20})\s+([\wçğıöşü]{2,20})\b",
                re.I), "benim", "sever"),
)


_BACK_V = "aıou"
_FRONT_V = "eiöü"


def to_second_person(subj: str) -> str:
    """'kardeşim' -> 'kardeşin' (1. kişi iyelik -> 2. kişi).

    Karşıdaki kendi yakınını 1. kişiyle anlatır ("kardeşim"); Nora aynı kişiden
    bahsederken 2. kişiyi kullanır ("kardeşinin adı"). Bu bir çeviri değil, dilbilgisi:
    aynı gönderge, konuşanın değişmesiyle eki değişir.
    """
    w = tr_lower(subj or "")
    if w in ("benim", "ben"):
        return "senin"
    # 1. kişi iyelik eki sonda 'm'dir (kardeşim, annem, kedim, babam); 2. kişide 'n'
    # olur (kardeşin, annen, kedin, baban). Tek istisna 'benim' (yukarıda).
    if len(w) >= 3 and w.endswith("m"):
        return w[:-1] + "n"
    return w


_BACK = "aıou"
_FRONT = "eiöü"


def genitive(word: str) -> str:
    """Tamlayan eki: 'kardeşin' -> 'kardeşinin', 'kedin' -> 'kedinin', 'senin' -> 'senin'."""
    w = tr_lower(word or "")
    if not w:
        return w
    if w in ("senin", "sen"):
        return "senin"
    last_v = next((ch for ch in reversed(w) if ch in _BACK + _FRONT), "a")
    if w[-1] in _BACK + _FRONT:
        return w + ("nın" if last_v in "aı" else "nun" if last_v in "ou"
                    else "nin" if last_v in "ei" else "nün")
    if last_v in "aı":
        return w + "ın"
    if last_v in "ou":
        return w + "un"
    if last_v in "ei":
        return w + "in"
    return w + "ün"


def _clean_word(w: str) -> str:
    return tr_lower((w or "").strip(" .,!?;:'\"“”"))


#: Soru izleri: bunlardan biri geçiyorsa cümle olgu bildirimi değil SORUDUR.
_QUESTIONISH = ("neydi", "nedir", "ne ", " ne", "kim", "kaç", "hangi", "nasıl", "nerede",
                "mi?", "mı?", "mu?", "mü?", "mi ", "mı ", "mu ", "mü ")

#: Değer olarak yazılamayacak sözcükler (soru sözcükleri).
_STOP_VALUES = {"neydi", "nedir", "ne", "kim", "kaç", "hangi", "nasıl", "nerede", "ne?",
                "bilmiyorum", "hatırlamıyorum"}


class PersonalMemory:
    """Karşıdakinin söylediği olgular: (özne, ilişki) -> değer + kaynak."""

    MAX_RECORDS = 512

    def __init__(self):
        self.facts: dict[tuple[str, str], dict] = {}
        self.turn = 0

    # ---------------------------------------------------------------- yazma
    def observe(self, text: str, speaker: str = "insan") -> list[dict]:
        """Cümleden AÇIK olguları çıkar ve yaz. Bulunan kayıtları döndürür."""
        raw = (text or "").strip()
        if not raw or raw.endswith("?"):
            return []
        _low_all = tr_lower(raw)
        if any(k in _low_all for k in _QUESTIONISH):
            return []
        out: list[dict] = []
        for clause in re.split(r"[,;]|\s+ve\s+|\s+ama\s+", raw):
            c = clause.strip(" .!?")
            if not c or len(c) > 120:
                continue
            low = tr_lower(c)
            for pat, fixed_subj, rel in _PATTERNS:
                m = pat.match(c) or pat.match(low)
                if not m:
                    continue
                q = ""
                if fixed_subj is not None and pat.groups == 1:
                    subj, val = fixed_subj, m.group(1)
                elif pat.groups == 2:
                    subj, val = m.group(1), m.group(2)
                else:
                    continue
                if rel == "sever":
                    # "en sevdiğim RENK MAVİ": renk = niteleyici, mavi = değer
                    q, val = subj, val
                    subj = "benim"
                subj, val = _clean_word(subj), _clean_word(val)
                if not subj or not val or subj == val:
                    continue
                if val in _STOP_VALUES or val.endswith(("ne", "mi", "mı")):
                    continue
                if len(val) < 2 and not val.isdigit():
                    continue
                rec = self.add(subj, rel, val, source=speaker, quote=c)
                if q:
                    rec["q"] = _clean_word(q)
                out.append(rec)
                break
        return out

    def add(self, subj: str, rel: str, val: str, source: str = "insan",
            quote: str = "") -> dict:
        key = (str(subj).lower(), str(rel).lower())
        prev = self.facts.get(key)
        rec = {"subj": key[0], "rel": key[1], "val": str(val).lower(),
               "quotes": [(str(quote), int(self.turn))][-4:],
               "source": source, "turn": int(self.turn), "hits": 0}
        if prev is not None:
            if prev["val"] == rec["val"]:
                # Aynı olgu tekrar söylendi: yeni kayıt açma, kaynağı güçlendir.
                prev["hits"] = int(prev.get("hits") or 0) + 1
                prev.setdefault("quotes", []).append((str(quote), int(self.turn)))
                prev["quotes"] = prev["quotes"][-4:]
                prev["turn"] = int(self.turn)
                return prev
            # ÇELİŞKİ: eski değer SİLİNMEZ, önceki olarak saklanır.
            rec["onceki"] = prev.get("val")
            rec["onceki_quotes"] = list(prev.get("quotes") or [])[-2:]
        self.facts[key] = rec
        if len(self.facts) > self.MAX_RECORDS:
            # En eski kayıt düşer (kapasite sınırı); düşme sessiz değil: sayılır.
            oldest = min(self.facts.items(), key=lambda kv: kv[1].get("turn", 0))[0]
            self.facts.pop(oldest, None)
            self.evicted = int(getattr(self, "evicted", 0)) + 1
        return rec

    # ---------------------------------------------------------------- okuma
    def self_name(self):
        return self.facts.get(("benim", "ad"))

    def recall(self, text: str) -> dict | None:
        """Sorudan özne+ilişki çıkar ve kaydı döndür (yoksa None)."""
        low = tr_lower((text or "").strip(" .!?"))
        rel = None
        if any(k in low for k in ("adı ne", "adı neydi", "ismi ne", "adını", "adı")):
            rel = "ad"
        elif "yaş" in low:
            rel = "yaş"
        elif "sevdiğ" in low:
            rel = "sever"
        if rel is None:
            return None
        for key, rec in self.facts.items():
            subj, r = key
            if r != rel:
                continue
            if rel == "sever":
                # 'en sevdiğim renk ne?' -> niteleyici (renk) soruda geçmeli
                q = str(rec.get("q") or "")
                if "sevdiğ" in low and (not q or q in low):
                    rec["hits"] = int(rec.get("hits") or 0) + 1
                    return rec
                continue
            # Özne, sorunun içinde geçiyor mu? (ekli biçimler: kardeşim/kardeşimin)
            stem = subj.rstrip("im")
            if subj in low or (len(stem) >= 3 and stem in low):
                rec["hits"] = int(rec.get("hits") or 0) + 1
                return rec
        return None

    def state_dict(self) -> dict:
        return {"turn": int(self.turn),
                "facts": [dict(v) for v in self.facts.values()]}

    def load_state_dict(self, d: dict | None) -> None:
        self.facts = {}
        for rec in ((d or {}).get("facts") or []):
            if isinstance(rec, dict) and rec.get("subj") and rec.get("rel"):
                self.facts[(str(rec["subj"]).lower(), str(rec["rel"]).lower())] = rec
        self.turn = int((d or {}).get("turn") or 0)


#: Tanıklık kalıbı: "Ayşe bana toplantının saat üçte olduğunu söyledi."
_WITNESS = re.compile(r"^(\w+)\s+bana\s+(.+?)\s+(?:diye\s+)?(?:söyledi|dedi)\b", re.U)
#: Önerme kuyruğu: "… olduğunu / … dediğini"
_PROP_TAIL = re.compile(r"\s+(?:olduğunu|olduğunu|dediğini|diye)\s*$", re.U)
#: Değer eki: "üçte" -> "üç", "beşte" -> "beş"
_VAL_SUFFIX = re.compile(r"(?:'?[dt][ae]|'?[dt][ae]n|'?[yn][ae]|'?[dt][ae]r)$", re.U)


def _stem(word: str) -> str:
    """Basit gövde: tamlayan/çekim eklerini at ('toplantının' -> 'toplantı')."""
    w = _clean_word(word)
    for suf in ("nın", "nin", "nun", "nün", "ın", "in", "un", "ün",
                "ya", "ye", "da", "de", "ta", "te", "ı", "i", "u", "ü"):
        if len(w) - len(suf) >= 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _value(word: str) -> str:
    w = _clean_word(word)
    stripped = _VAL_SUFFIX.sub("", w)
    return stripped if len(stripped) >= 2 or stripped.isdigit() else w


class Testimony:
    """ÜÇÜNCÜ KİŞİ TANIKLIĞI: aynı olguya iki kaynak iki değer verirse İKİSİ de tutulur.

    Kişisel bellek (PersonalMemory) karşıdakinin KENDİ söylediğini tutar. Burada
    karşıdakinin BAŞKASINDAN aktardığı olgu tutulur: (kaynak, özne, ilişki, değer).
    Çelişki sessizce son değere indirgenmez — iki tanık yan yana söylenir, çünkü
    "hangi tanığa inanacağım" sorusunun cevabı veride yoktur.
    """

    MAX = 256

    def __init__(self):
        self.records: list[dict] = []
        self.turn = 0
        self.evicted = 0

    # ---------------------------------------------------------------- yazma
    def observe(self, text: str, turn: int | None = None) -> list[dict]:
        raw = (text or "").strip()
        if not raw or raw.endswith("?"):
            return []
        out = []
        for clause in re.split(r"[,;]|\s+ve\s+|\s+ama\s+", raw):
            c = clause.strip(" .!?")
            low = tr_lower(c)
            m = _WITNESS.match(low)
            if not m:
                continue
            src = _clean_word(m.group(1))
            prop = _PROP_TAIL.sub("", m.group(2)).strip()
            words = [w for w in re.split(r"\s+", prop) if w]
            if len(words) < 2 or not src:
                continue
            subj = _stem(words[0])
            val = _value(words[-1])
            rel = " ".join(_stem(w) for w in words[1:-1]) if len(words) > 2 else ""
            if not subj or not val or subj == val:
                continue
            out.extend(self.add(src, subj, rel, val, turn=turn, quote=c))
        return out

    def add(self, src: str, subj: str, rel: str, val: str,
            turn: int | None = None, quote: str = "") -> list[dict]:
        t = int(self.turn if turn is None else turn)
        self.records.append({"src": str(src).lower(), "subj": str(subj).lower(),
                             "rel": str(rel).lower(), "val": str(val).lower(),
                             "turn": t, "quote": str(quote)})
        if len(self.records) > self.MAX:
            self.records.pop(0)
            self.evicted += 1
        return self.records[-1:]

    # ---------------------------------------------------------------- okuma
    def query(self, text: str) -> dict | None:
        """Sorudaki özne/ilişkiye ait tanıklıkları getir (çelişki varsa işaretle)."""
        low = tr_lower((text or "").strip(" .!?"))
        if not low:
            return None
        hits = [r for r in self.records if r["subj"] in low]
        if not hits:
            return None
        # İlişki sorusu: soruda geçen ilişki sözcüğü kayıtta da varsa daralt.
        narrowed = [r for r in hits if r["rel"] and (r["rel"] in low or
                                                     any(w in low for w in r["rel"].split()))]
        if narrowed:
            hits = narrowed
        vals = []
        for r in hits:
            vals.append((r["src"], r["val"]))
        distinct = sorted({v for _s, v in vals})
        return {"subj": hits[0]["subj"], "vals": vals, "distinct": distinct,
                "conflict": len(distinct) > 1}

    def state_dict(self) -> dict:
        return {"turn": int(self.turn), "evicted": int(self.evicted),
                "records": self.records[-self.MAX:]}

    def load_state_dict(self, d: dict | None) -> None:
        d = d or {}
        self.records = [r for r in (d.get("records") or []) if isinstance(r, dict)]
        self.turn = int(d.get("turn") or 0)
        self.evicted = int(d.get("evicted") or 0)
