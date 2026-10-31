"""nora_apply.py — UYGULAMA: kuralı NESNE olarak bağla, yeni girdiye UYGULA.

Teşhis
------
Nora **retrieve** ediyordu, **apply** etmiyordu. Kavram ağında kenar vardı ama bir kuralı
nesne olarak bağlayıp yeni bir girdiye uygulayan operatör yoktu. Bu yüzden kural tümevarımı,
transfer, sayma, planlama — hepsi aynı noktadan düşüyordu: deyim kitabı, kompozisyon yok.

Bu organ üç primitif getirir:

    1. WorkingMemory    — 7 yuvalı çalışma belleği (plan, sayma, çok adım için tampon)
    2. Rule             — bağlanmış kural NESNESİ: özellik -> ölçüm -> eşleme -> etiket
    3. Interpreter      — decompose -> dene -> doğrula -> (olmazsa) AÇIKÇA "yapamadım"

Kural metni değil, kuralın KENDİSİ saklanır:

    "sesli harf sayısı tek ise mavi, çift ise kırmızı"
      -> Rule(feature="vowel_count", map={"tek":"mavi", "çift":"kırmızı"})

    apply(rule, "Bal arısı uçar")  ->  ("mavi", trace=[vowel_count=6, parity=çift, ...])

Yani yeni bir girdi geldiğinde Nora cümleyi hatırlamıyor; ölçüyor ve eşliyor. Transfer de
aynı nesnenin özelliğini değiştirmektir:

    transfer(rule, feature="word_count", map={"tek":"tuzlu", "çift":"tatlı"}, domain="mutfak")

Her adım iz bırakır (`trace`), böylece "nasıl yaptın" sorusu da retrieve değil kayıt okuma
olur.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

VOWELS = set("aeıioöuüAEIİOÖUÜ")

#: Kural ŞEKİLLERİNİN adları. Şekil, ölçümden etikete giden yolun biçimidir;
#: yeni bir şekil yeni bir CÜMLE değil, aynı makinenin yeni bir eşlemesidir.
SHAPE_NAMES = {"parity": "tek/çift", "threshold": "eşik", "rank": "sıra",
               "presence": "varlık"}

#: Ölçüm fonksiyonları: kural nesnesinin "özellik" alanı bunlardan birine çözülür.
def _tokens(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text or "", flags=re.UNICODE)


FEATURES = {
    "vowel_count": ("sesli harf", lambda s: sum(1 for ch in (s or "") if ch in VOWELS)),
    "char_count": ("harf", lambda s: sum(1 for ch in (s or "") if ch.isalpha())),
    "consonant_count": ("sessiz harf", lambda s: sum(1 for ch in (s or "") if ch.isalpha() and ch not in VOWELS)),
    "word_count": ("sözcük", lambda s: len(_tokens(s))),
    "letter_count": ("harf", lambda s: sum(1 for ch in (s or "") if ch.isalpha())),
    "syllable_count": ("hece", lambda s: max(1, sum(1 for ch in (s or "") if ch in VOWELS))),
}

#: Ölçüm adlarının KISA biçimi: bileşik ölçüm adı ("sessiz-sesli farkı") böyle kurulur.
SHORT_NAMES = {"vowel_count": "sesli", "consonant_count": "sessiz",
               "letter_count": "harf", "word_count": "sözcük", "syllable_count": "hece"}


def pos_name(spec: dict) -> str:
    """Konum ölçümünün okunur adı: 'ilk harf sesli', 'son harf sessiz', '2. harf'."""
    i = int((spec or {}).get("i") or 0)
    cls = str((spec or {}).get("cls") or "")
    n = {0: "ilk", 1: "2.", 2: "3.", 3: "4.", 4: "5."}.get(i, f"{i + 1}.")
    if i < 0:
        n = {-1: "son", -2: "sondan ikinci", -3: "sondan üçüncü"}.get(i, f"sondan {-i}.")
    ad = {"vowel": "harf sesli", "consonant": "harf sessiz",
          "letter": f"harf '{spec.get('arg')}'"}.get(cls, "harf")
    return f"{n} {ad}".strip()

#: Sayının OKUNUŞUNA göre ayrılma eki: 3 -> "ten" (üçten), 6 -> "dan" (altıdan),
#: 1 -> "den" (birden), 0 -> "dan" (sıfırdan). Sabit "'ten az" yazmak "-1'ten az"
#: gibi yanlış bir yüzey üretiyordu; ek artık sayıdan geliyor.
_ABL_UNIT = {0: "dan", 1: "den", 2: "den", 3: "ten", 4: "ten", 5: "ten",
             6: "dan", 7: "den", 8: "den", 9: "dan"}
_ABL_TEEN = {10: "dan", 11: "den", 12: "den", 13: "ten", 14: "ten", 15: "ten",
             16: "dan", 17: "den", 18: "den", 19: "dan"}
_ABL_TEN = {2: "den", 3: "dan", 4: "tan", 5: "den", 6: "dan", 7: "den",
            8: "den", 9: "dan"}


def num_abl(n: int) -> str:
    """`n` sayısı okunduğunda ayrılma eki ne olur: 3 -> 'ten', -1 -> 'den'."""
    n = abs(int(n))
    if n < 10:
        return _ABL_UNIT[n]
    if n < 20:
        return _ABL_TEEN[n]
    if n % 10 == 0:
        return _ABL_TEN.get(n // 10 % 10, "den")
    return _ABL_UNIT[n % 10]


def feature_spec_fn(spec: dict):
    """İCAT EDİLMİŞ ölçümü yeniden kur: kural kendi ölçümünü YANINDA taşır.

    `nora_forge` yeni bir ölçüm icat ettiğinde onu `FEATURES`'a kaydeder; ama bir kural
    başka bir süreçte yüklenirse o kayıt yoktur. Bu yüzden kural, ölçümün TARİFİNİ
    (`feature_spec`) taşır ve gerekirse buradan yeniden kurulur — sessizce başka bir
    ölçüme düşmez.
    """
    kind = str((spec or {}).get("kind") or "")
    arg = str((spec or {}).get("arg") or "").lower()
    if kind == "suffix":
        return lambda s: 1 if str(s or "").lower().endswith(arg) else 0
    if kind == "prefix":
        return lambda s: 1 if str(s or "").lower().startswith(arg) else 0
    if kind == "letter":
        return lambda s: 1 if arg and arg in str(s or "").lower() else 0
    if kind == "and":
        # BİLEŞİK (MANTIKSAL): iki ikili ölçümün İKİSİ de doğruysa. Tek başına hiçbiri
        # ayırmıyorsa, ikisi BİRLİKTE ayırabilir; kural artık bir birleşim.
        fa = feature_spec_fn(dict((spec or {}).get("a") or {}))
        fb = feature_spec_fn(dict((spec or {}).get("b") or {}))
        return lambda s: 1 if (fa(s) and fb(s)) else 0
    if kind == "diff":
        # BİLEŞİK ÖLÇÜM: iki ölçümün FARKI. Tek başına hiçbiri ayırmıyorsa aradaki fark
        # ayırabilir; kural ilişkisel olur ("sesli eksi sessiz 1'den az" gibi) ve hiçbir
        # yerleşik ölçüm bunu tek başına söyleyemez.
        fa = (FEATURES.get(str((spec or {}).get("a") or "")) or (None, lambda s: 0))[1]
        fb = (FEATURES.get(str((spec or {}).get("b") or "")) or (None, lambda s: 0))[1]
        return lambda s: int(fa(s)) - int(fb(s))
    if kind == "pos":
        # KONUM ÖLÇÜMÜ: sözcüğün i. harfi şu SINIFTAN mı? ("ilk harfi sesli mi")
        # Son ek/ön ek/harf varlığı konumu temsil edemez; bu ölçüm tam olarak konumu
        # ölçer. i negatifse sondan sayılır (-1 = son harf).
        i = int((spec or {}).get("i") or 0)
        cls = str((spec or {}).get("cls") or "vowel")
        arg = str((spec or {}).get("arg") or "").lower()

        def _f(s: str) -> int:
            t = str(s or "").lower()
            t = "".join(ch for ch in t if ch.isalpha())
            if not t:
                return 0
            idx = i if i >= 0 else len(t) + i
            if idx < 0 or idx >= len(t):
                return 0
            ch = t[idx]
            if cls == "vowel":
                return 1 if ch in VOWELS else 0
            if cls == "consonant":
                return 1 if (ch.isalpha() and ch not in VOWELS) else 0
            if cls == "letter":
                return 1 if (arg and ch == arg) else 0
            return 0
        return _f
    if kind == "suffix_pattern":
        # ÜNLÜ UYUMU KALIBI: 'lVk' -> son üç harf [sessiz][ünlü][sessiz] ise eşleşir.
        # Türkçe'de -lık/-lik/-luk/-lük AYNI ektir; kalıp, dört ayrı ek görmek yerine
        # tek bir morfemi ölçer.
        pat = arg
        def _f(s: str) -> int:
            t = str(s or "").lower()
            if not pat or len(t) < len(pat):
                return 0
            for a, b in zip(t[-len(pat):], pat):
                if b == "v":
                    if a not in VOWELS:
                        return 0
                elif a != b:
                    return 0
            return 1
        return _f
    return lambda s: 0


def register_feature(key: str, name: str, spec: dict) -> None:
    """Ölçümü kaydet: bundan sonra `Rule.measure` bu anahtarı çözebilir."""
    FEATURES[str(key)] = (str(name), feature_spec_fn(spec))


#: Özellik sözünü (kullanıcının söylediği) FEATURES anahtarına çevir.
FEATURE_WORDS = {
    "sesli harf": "vowel_count", "sesli": "vowel_count", "ünlü": "vowel_count",
    "sessiz harf": "consonant_count", "sessiz": "consonant_count",
    "harf": "letter_count", "sözcük": "word_count", "kelime": "word_count",
    "hece": "syllable_count",
    # Alana özgü "adet" sözleri: sayılan şey öğedir, harf değil.
    "malzeme": "word_count", "paket": "word_count", "nesne": "word_count",
    "öğe": "word_count", "adet": "word_count", "ürün": "word_count",
    "kişi": "word_count", "kutu": "word_count", "sipariş": "word_count",
}

PARITY_WORDS = {"tek": "odd", "çift": "even", "tek sayı": "odd", "çift sayı": "even"}


@dataclass
class Slot:
    """Çalışma belleğindeki tek yuva."""
    key: str
    value: object
    source: str = ""


class WorkingMemory:
    """7 yuvalı çalışma belleği. Plan, sayma ve çok adımlı iş için tampon.

    Bilinçli sınır: kapasite 7. Daha fazlası istenirse en eski yuva düşer (insan belleği
    gibi) — ve bu düşme `evicted` listesine yazılır, sessizce kaybolmaz.
    """

    def __init__(self, capacity: int = 7):
        self.capacity = int(capacity)
        self.slots: dict[str, Slot] = {}
        self.order: list[str] = []
        self.evicted: list[Slot] = []
        self.writes = 0

    # ------------------------------------------------------------------ yaz/oku
    def write(self, key: str, value, source: str = "") -> None:
        k = str(key)
        if k not in self.slots:
            self.order.append(k)
            if len(self.order) > self.capacity:
                old = self.order.pop(0)
                gone = self.slots.pop(old, None)
                if gone is not None:
                    self.evicted.append(gone)
        self.slots[k] = Slot(k, value, source)
        self.writes += 1

    def read(self, key: str, default=None):
        s = self.slots.get(str(key))
        return default if s is None else s.value

    def drop(self, key: str) -> None:
        k = str(key)
        if k in self.slots:
            del self.slots[k]
            if k in self.order:
                self.order.remove(k)

    def clear(self) -> None:
        self.slots.clear()
        self.order.clear()

    @property
    def load(self) -> int:
        return len(self.slots)

    def snapshot(self) -> dict:
        return {k: self.slots[k].value for k in self.order if k in self.slots}

    def state_dict(self) -> dict:
        from nora_state import encode
        return {"slots": [{"key": s.key, "value": encode(s.value), "source": s.source}
                          for s in (self.slots[k] for k in self.order if k in self.slots)],
                "evicted": [{"key": s.key, "value": encode(s.value), "source": s.source}
                            for s in self.evicted[-self.capacity:]],
                "writes": int(self.writes)}

    def load_state_dict(self, d: dict | None) -> None:
        if not isinstance(d, dict):
            return
        from nora_state import decode
        self.clear()
        self.evicted.clear()
        for rec in (d.get("evicted") or []):
            self.evicted.append(Slot(str(rec.get("key", "")), decode(rec.get("value")),
                                     str(rec.get("source", ""))))
        for rec in (d.get("slots") or [])[: self.capacity]:
            self.write(str(rec.get("key", "")), decode(rec.get("value")),
                       str(rec.get("source", "")))
        self.writes = int(d.get("writes", self.writes) or 0)


@dataclass
class Rule:
    """BAĞLANMIŞ KURAL NESNESİ — cümle değil, operatör.

    `feature` ölçüm fonksiyonunu, `mapping` ölçüm sonucunu etikete çevirmeyi söyler.
    `domain` kuralın hangi alana bağlandığını tutar (arıcılık, mutfak, lojistik …).
    """
    rid: str
    feature: str
    mapping: dict[str, str]
    domain: str = "genel"
    source_text: str = ""
    uses: int = 0
    derived_from: str = ""
    #: İSTİSNA: [{"pattern": "bal", "label": "kırmızı"}]. Ana kuraldan ÖNCE bakılır.
    #: Kural "bağlanan" bir şeyse, istisna da kuralın parçasıdır; ayrı bir metin değil.
    exceptions: list = field(default_factory=list)
    #: ŞEKİL: ölçümden ETİKETE giden yol. Kuralın kendisi değil, eşlemenin biçimi.
    #:   parity    — ölçüm tek/çift (varsayılan, tek şekilli geçmiş)
    #:   threshold — ölçüm eşiğin altında/üstünde ("3'ten az ise")
    #:   rank      — ölçüm küme içinde en büyük/en küçük ("en çok olan")
    shape: str = "parity"
    threshold: int = 0
    op: str = "lt"          # lt: <, le: <=, gt: >, ge: >=   (yalnız threshold)
    #: KİM YAZDI: "insan" (bize öğretilen bağlı kural) ya da "kendi" (nora_forge'un
    #: örneklerden türettiği operatör). Sayaç, operatör yazımının kime ait olduğunu
    #: gizlemesin diye kuralın kendisinde durur.
    origin: str = "insan"
    #: İCAT EDİLMİŞ ÖLÇÜMÜN TARİFİ: {"kind": "suffix", "arg": "lık"}. Boşsa ölçüm
    #: `FEATURES`'taki yerleşik ölçümlerden biridir. Dolu ise kural kendi ölçümünü taşır.
    feature_spec: dict = field(default_factory=dict)

    def measure(self, item: str) -> int:
        spec = getattr(self, "feature_spec", None)
        if spec and self.feature not in FEATURES:
            fn = feature_spec_fn(spec)          # icat edilmiş ölçüm kendini taşır
        else:
            fn = FEATURES.get(self.feature, FEATURES["letter_count"])[1]
        return int(fn(item))

    def bucket(self, n: int) -> str:
        """Ölçümü EŞLEME ANAHTARINA çevir. Tek fonksiyon, üç şekil.

        Eşik şekli anahtarı 'cond'/'else' olarak verir: eşik ifadesinin kendisi
        doğruysa 'cond' (cümlenin ilk etiketi), değilse 'else'.
        """
        if self.shape == "threshold":
            t, o = int(self.threshold), str(self.op)
            cond = {"lt": n < t, "le": n <= t, "gt": n > t, "ge": n >= t}.get(o, n < t)
            return "cond" if cond else "else"
        if self.shape == "rank":
            return "rank"     # küme gerekir: tek öğe üzerinde etiket YOK
        if self.shape == "presence":
            # İCAT EDİLMİŞ ölçüm: ölçüm 0/1'dir, eşleme "var/yok" üzerinden kurulur.
            return "var" if int(n) else "yok"
        return "odd" if n % 2 else "even"

    def apply(self, item: str) -> tuple[str | None, dict]:
        """Yeni girdiye uygula: İSTİSNA -> ölç -> eşle -> etiket. İz (trace) döner."""
        low = (item or "").lower()
        for ex in (self.exceptions or []):
            pat = str(ex.get("pattern") or "").lower()
            if pat and pat in low:
                self.uses += 1
                return str(ex.get("label")), {
                    "rule": self.rid, "feature": self.feature, "item": item,
                    "exception": pat, "mapped": str(ex.get("label")), "domain": self.domain,
                    "measured": None, "parity": "exception"}
        n = self.measure(item)
        key = self.bucket(n)
        label = None if key == "rank" else self.mapping.get(key)
        self.uses += 1
        trace = {"rule": self.rid, "feature": self.feature,
                 "feature_name": FEATURES.get(self.feature, ("harf",))[0],
                 "item": item, "measured": n, "bucket": key, "parity": key,
                 "mapped": label, "domain": self.domain, "shape": self.shape,
                 "threshold": self.threshold, "op": self.op}
        return label, trace

    def apply_set(self, items: list[str]) -> tuple[list[dict], dict]:
        """SIRA ŞEKLİ: etiket kümenin İÇİNDE belirlenir (en çok -> A, en az -> B)."""
        rows = [{"item": it, "measured": self.measure(it), "label": None} for it in items]
        if self.shape != "rank" or not rows:
            self.uses += 1
            return rows, {"rule": self.rid, "shape": self.shape, "kind": "apply_set",
                          "note": "sıra şekli değil"}
        hi = max(r["measured"] for r in rows)
        lo = min(r["measured"] for r in rows)
        tie = hi == lo
        for r in rows:
            if not tie and r["measured"] == hi:
                r["label"] = self.mapping.get("max")
            elif not tie and r["measured"] == lo:
                r["label"] = self.mapping.get("min")
        self.uses += 1
        trace = {"rule": self.rid, "shape": self.shape, "kind": "apply_set",
                 "rows": rows, "tie": tie, "domain": self.domain}
        return rows, trace

    def transfer(self, feature: str, mapping: dict[str, str], domain: str) -> "Rule":
        """Yapıyı koru, alanı değiştir: aynı operatörün yeni bağlanışı.

        ŞEKİL de taşınır: eşik kuralı başka alana gittiğinde eşik olarak kalır
        (eşik ve yön korunur); yalnız ölçülen özellik ve etiketler değişir.
        """
        return Rule(rid=f"{self.rid}->{domain}",
                    feature=feature, mapping=dict(mapping), domain=domain,
                    source_text=self.source_text, derived_from=self.rid,
                    shape=self.shape, threshold=self.threshold, op=self.op)


@dataclass
class RelationRule:
    """Örneklerden açılan, henüz kanıtlanmamış ilişki operatörü.

    Bu nesne bir cümle şablonu değildir. SVO gözleminin soyutlanmış hâlidir:
    operatör `(özne, nesne)` bağlanışını ilişki üçlüsüne uygular. ``probation``
    durumu sembolik grafın yön/çelişki ve transfer kanıtı olmadan kabul edilmiş
    dünya yasası sayılmayacağını açıkça taşır.
    """

    rid: str
    relation: str
    verb: str
    status: str = "probation"
    syntax: str = "SVO"
    examples: list[dict] = field(default_factory=list)
    uses: int = 0

    def apply(self, source: str, target: str) -> tuple[tuple[str, str, str] | None, dict]:
        source, target = str(source).strip(), str(target).strip()
        if self.status in {"rejected", "conflict"} or not source or not target:
            return None, {"rule": self.rid, "status": self.status, "applied": False}
        self.uses += 1
        triple = (source, self.relation, target)
        return triple, {
            "rule": self.rid, "relation": self.relation, "verb": self.verb,
            "source": triple[0], "target": triple[2], "status": self.status,
            "syntax": self.syntax, "applied": bool(triple[0] and triple[2]),
        }

    def state_dict(self) -> dict:
        return {
            "rid": self.rid, "relation": self.relation, "verb": self.verb,
            "status": self.status, "syntax": self.syntax,
            "examples": [dict(row) for row in self.examples[-64:]], "uses": self.uses,
        }


@dataclass
class ChainRule:
    """İKİ ADIMLI ZİNCİR: birinci kuralın ETİKETİ, ikinci kuralın GİRDİSİ olur.

    "önce sesli harf sayısını bul; tek ise A, çift ise B yaz. sonra A ise kısa, B ise uzun de."

        step1: metin -> ölçüm -> A/B          (Rule, parity -> sembol)
        step2: A/B   -> kısa/uzun             (sözlük eşlemesi, ölçüm yok)

    Tek adımlı uygulama burada yetmez: ara sonuç taşınmadan ikinci adım çalışmaz.
    """

    rid: str
    step1: Rule
    label_map: dict
    source_text: str = ""

    def apply(self, item: str) -> tuple[str | None, dict]:
        mid, t1 = self.step1.apply(item)
        final = (self.label_map or {}).get(str(mid).lower()) if mid is not None else None
        trace = {"rule": self.rid, "kind": "chain", "item": item,
                 "step1": t1, "mid": mid, "final": final}
        return final, trace


#: OPERATÖR REJİSTRİSİ — Nora "neyi yapabiliyorum" sorusunu buradan okur.
#: Rejistri VERİDİR (`nora_operators.json`): yeni yetenek eklemek bir KAYIT eklemektir,
#: koda yeni cümle yazmak değil. İskelet mandalı bu ayrımı ölçer.
def _data_path():
    """Rejistri dosyasının yolu (tek yerden: testler de buradan okur)."""
    import pathlib
    return pathlib.Path(__file__).with_name("nora_operators.json")


def _load_operators() -> tuple[dict, tuple]:
    import json
    import pathlib
    try:
        blob = json.loads(pathlib.Path(__file__).with_name("nora_operators.json")
                          .read_text(encoding="utf-8"))
        ops = {str(k): str(v) for k, v in (blob.get("operators") or {}).items()}
        miss = tuple(str(x) for x in (blob.get("missing_organ") or ()))
        return ops, miss
    except Exception:
        return {}, ()


def _load_labels() -> dict:
    import json
    import pathlib
    try:
        blob = json.loads(pathlib.Path(__file__).with_name("nora_operators.json")
                          .read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in (blob.get("unbound_labels") or {}).items()}
    except Exception:
        return {}


OPERATORS, MISSING_ORGAN = _load_operators()
UNBOUND_LABELS = _load_labels()


class Interpreter:
    """Küçük yorumlayıcı: kural bağla, uygula, planı adım adım yürüt, olmazsa AÇIKÇA söyle."""

    MAX_PLAN_STEPS = 12

    def __init__(self, wm: WorkingMemory | None = None):
        self.wm = wm or WorkingMemory()
        self.rules: dict[str, Rule] = {}
        self.last: Rule | None = None
        #: Sözdiziminden öğrenilen ilişki operatörleri; metin kurallarından ayrı
        #: defterdedir ve kabul durumları silinmeden izlenir.
        self.relation_rules: dict[str, RelationRule] = {}
        self.trace: list[dict] = []
        self.failures: list[str] = []
        #: YAPILMAMIŞ NOTASYON: "⊕ tek demek" -> {"⊕": "tek"}. Sembol, anlamı öğretilene
        #: kadar bağlanmaz; öğretilince kural metni çözülürken yerine konur.
        self.aliases: dict[str, str] = {}
        #: İKİ ADIMLI ZİNCİR: A'nın çıktısı B'nin girdisi.
        self.chain: "ChainRule | None" = None

    # ------------------------------------------------------------------ notasyon
    def teach_alias(self, symbol: str, meaning: str) -> bool:
        sym = (symbol or "").strip()
        val = (meaning or "").strip().lower()
        if not sym or not val:
            return False
        self.aliases[sym] = val
        self._trace("alias", {"symbol": sym, "means": val})
        return True

    def _expand(self, text: str) -> str:
        t = text or ""
        for sym, val in (self.aliases or {}).items():
            t = t.replace(sym, f" {val} ")
        return re.sub(r"\s+", " ", t)

    # ------------------------------------------------------------------ sıralama
    @staticmethod
    def sort_items(items: list[str], mode: str = "len_asc") -> list[str]:
        """SIRALAMA operatörü: sayı → parite → etiket DEĞİL, ölçüt → sıra.

        mode: num_asc/num_desc/len_asc/len_desc/alpha. Sayıya çevrilemeyen öğe
        sayısal sıralamada sona atılır (sessizce atılmaz, sıralamada görünür).
        """
        def as_num(x: str):
            m = re.fullmatch(r"\s*(-?\d+)\s*", str(x))
            return int(m.group(1)) if m else None

        if mode in ("num_asc", "num_desc"):
            nums = [(as_num(x), str(x)) for x in items]
            nums.sort(key=lambda p: (p[0] is None, p[0] if p[0] is not None else 0))
            if mode == "num_desc":
                nums = [p for p in nums if p[0] is not None][::-1] + [p for p in nums if p[0] is None]
            return [t for _n, t in nums]
        if mode == "alpha":
            return sorted(items, key=lambda x: x.lower())
        key = (lambda x: (len(x), x.lower())) if mode == "len_asc" else (lambda x: (-len(x), x.lower()))
        return sorted(items, key=key)

    # ------------------------------------------------------------------ bağlama
    def bind_relation(self, relation: str, verb: str, *, source: str, target: str,
                      sentence: str, status: str = "probation") -> RelationRule:
        """Bir SVO gözlemini yeni bağlanışlara uygulanabilir operatöre soyutla.

        Aynı ilişkinin yeni örneği eski operatörü ezmez; örnek defterine eklenir.
        Operatörün kendisi kavram çiftini ezberlemez, yalnız ilişki ve fiil kökünü
        taşır. Bu yüzden ``apply`` çağrısı öğretilmemiş özne/nesneyle de çalışır;
        sonucun dünya bilgisi olarak kabulü ise ayrıca graf kanıtı gerektirir.
        """
        rel = str(relation).strip()
        root = str(verb).strip()
        rid = f"relation::{rel}"
        rule = self.relation_rules.get(rid)
        if rule is None:
            rule = RelationRule(rid=rid, relation=rel, verb=root, status=status)
            self.relation_rules[rid] = rule
        observation = {
            "source": str(source).strip(), "target": str(target).strip(),
            "sentence": str(sentence).strip(), "status": str(status),
        }
        if observation not in rule.examples:
            rule.examples.append(observation)
            del rule.examples[:-64]
        if status in {"rejected", "conflict"}:
            rule.status = status
        self._trace("bind_relation", {
            "rule": rid, "relation": rel, "status": rule.status,
            "examples": len(rule.examples), "invented": len(rule.examples) == 1,
        })
        return rule

    def apply_relation(self, relation: str, source: str, target: str):
        rule = self.relation_rules.get(f"relation::{str(relation).strip()}")
        if rule is None:
            return None, {"error": "no_relation_rule", "relation": relation}
        result = rule.apply(source, target)
        self._trace("apply_relation", result[1])
        return result

    def bind(self, text: str) -> Rule | None:
        """Kullanıcının söylediği kuralı NESNEYE çevir.

        Anladığı biçimler (üretken, cümle listesi değil):

            "<özellik> sayısı tek ise <A>, çift ise <B>"
            "<özellik> tek ise <A>, çift ise <B>"
            "<özellik> çift ise <A>, tek ise <B>"
        """
        t = (text or "").strip()
        low = self._expand(t).lower()

        # özellik: kullanıcının söylediği ada en uzun eşleşme
        feature = None
        for word, key in sorted(FEATURE_WORDS.items(), key=lambda kv: -len(kv[0])):
            if word in low:
                feature = key
                break
        if feature is None:
            feature = "letter_count"

        # ŞEKİL 2 — EŞİK: "3'ten az ise kısa, değilse uzun" / "en az 4 ise uzun"
        th = self._parse_threshold(low)
        if th is not None:
            mapping, threshold, op = th
            rid = f"R{len(self.rules) + 1}"
            rule = Rule(rid=rid, feature=feature, mapping=mapping,
                        domain=self._domain_of(low), source_text=t,
                        shape="threshold", threshold=threshold, op=op)
            self.rules[rid] = rule
            self.last = rule
            self.chain = None
            self.wm.write("rule", rid, "bind")
            self.wm.write("rule_map", dict(mapping), "bind")
            self._trace("bind", {"rule": rid, "feature": feature, "map": mapping,
                                 "shape": "threshold", "threshold": threshold, "op": op})
            return rule

        # ŞEKİL 3 — SIRA: "sesli harfi en çok olan uzun, en az olan kısa"
        rk = self._parse_rank(low)
        if rk is not None:
            rid = f"R{len(self.rules) + 1}"
            rule = Rule(rid=rid, feature=feature, mapping=rk,
                        domain=self._domain_of(low), source_text=t, shape="rank")
            self.rules[rid] = rule
            self.last = rule
            self.chain = None
            self.wm.write("rule", rid, "bind")
            self.wm.write("rule_map", dict(rk), "bind")
            self._trace("bind", {"rule": rid, "feature": feature, "map": rk, "shape": "rank"})
            return rule

        # eşleme: "tek ise X", "çift ise Y" (sıra serbest)
        pairs = re.findall(r"(tek|çift)\s*(?:sayı\s*)?(?:ise|olursa|->|:)\s*([^,;.]+)", low)
        mapping: dict[str, str] = {}
        for par, lab in pairs:
            lab = re.sub(r"\s+", " ", lab).strip(" ,.;")
            # etiket çok uzunsa yalnız ilk sözcüğü al ("mavi" gibi)
            lab = lab.split()[0] if lab else lab
            if lab:
                mapping[PARITY_WORDS[par]] = lab
        if len(mapping) < 2:
            return None

        # İSTİSNA: "... ama içinde 'bal' geçiyorsa her zaman kırmızı"
        exceptions: list[dict] = []
        m_exc = re.search(r"\bama\b(.*)$", low)
        if m_exc:
            tail = m_exc.group(1)
            pm = (re.search(r"['\"“”]([\wçğıöşü]{2,})['\"“”]\s*(?:geç|olan|içeren)", tail)
                  or re.search(r"içinde\s+([\wçğıöşü]{2,})", tail))
            words = re.findall(r"[\wçğıöşü]+", tail)
            if pm and words:
                exceptions.append({"pattern": pm.group(1), "label": words[-1]})

        rid = f"R{len(self.rules) + 1}"
        rule = Rule(rid=rid, feature=feature, mapping=mapping, domain=self._domain_of(low),
                    source_text=t, exceptions=exceptions)
        self.rules[rid] = rule
        self.last = rule
        self.wm.write("rule", rid, "bind")
        self.wm.write("rule_map", dict(mapping), "bind")
        self._trace("bind", {"rule": rid, "feature": feature, "map": mapping,
                             "exceptions": exceptions})
        return rule

    @staticmethod
    def _parse_threshold(low: str) -> tuple[dict, int, str] | None:
        """Eşikli kuralı metinden çıkar: (eşleme, eşik, yön). Cümle listesi değil, biçim."""
        m = re.search(r"(\d+)\s*(?:'?[dt]en|'?[dt]an)\s+(az|küçük|aşağı|çok|fazla|büyük|yukarı)", low)
        if m:
            t, w = int(m.group(1)), m.group(2)
            op = "lt" if w in ("az", "küçük", "aşağı") else "gt"
        else:
            m2 = re.search(r"en\s+(az|çok|fazla)\s+(\d+)", low)
            if not m2:
                return None
            t = int(m2.group(2))
            op = "ge" if m2.group(1) == "az" else "le"
        after = low[m.end():] if m else low[m2.end():]
        cond = (re.search(r"\bise\b\s*([a-zçğıöşü]{1,12})", after)
                or re.search(r"\bise\b\s*([a-zçğıöşü]{1,12})", low))
        other = (re.search(r"(?:değilse|aksi\s+halde|yoksa)\s+([a-zçğıöşü]{1,12})", low))
        if not (cond and other):
            return None
        return {"cond": cond.group(1), "else": other.group(1)}, t, op

    @staticmethod
    def _parse_rank(low: str) -> dict | None:
        """Sıra şekli: en çok/en az olanın etiketi. Tek öğe etiketlenemez, küme gerekir."""
        if re.search(r"\d", low):
            return None
        hi = re.search(r"en\s+(?:çok|fazla|yüksek)\s+olan\w*\s+([a-zçğıöşü]{1,12})", low)
        lo = re.search(r"en\s+(?:az|düşük)\s+olan\w*\s+([a-zçğıöşü]{1,12})", low)
        if not (hi and lo):
            return None
        return {"max": hi.group(1), "min": lo.group(1)}

    # ------------------------------------------------------------------ zincir
    def bind_chain(self, text: str) -> "ChainRule | None":
        """Tek cümleden iki adımlı kural: ara sonuç ikinci adımın girdisi olur."""
        t = (text or "").strip()
        low = self._expand(t).lower()
        feature = None
        for word, key in sorted(FEATURE_WORDS.items(), key=lambda kv: -len(kv[0])):
            if word in low:
                feature = key
                break
        if feature is None:
            return None
        first = dict(re.findall(r"(tek|çift)\s*(?:sayı\s*)?(?:ise|olursa|->|:)\s*([a-zçğıöşü0-9]{1,12})", low))
        if len(first) < 2:
            return None
        symbols = {v.lower() for v in first.values()}
        second = {}
        for lhs, rhs in re.findall(r"([a-zçğıöşü0-9]{1,12})\s+ise\s+([a-zçğıöşü]{2,})", low):
            if lhs in symbols:
                second[lhs] = rhs
        if len(second) < 2:
            return None
        r1 = Rule(rid=f"R{len(self.rules) + 1}", feature=feature,
                  mapping={PARITY_WORDS[k]: v.lower() for k, v in first.items()},
                  domain=self._domain_of(low), source_text=t)
        self.rules[r1.rid] = r1
        ch = ChainRule(rid=f"{r1.rid}+chain", step1=r1, label_map=second, source_text=t)
        self.chain = ch
        self.last = r1
        self._trace("bind_chain", {"rule": ch.rid, "feature": feature,
                                   "map": r1.mapping, "second": second})
        return ch

    def relabel(self, mapping: dict[str, str], feature: str | None = None) -> "Rule | None":
        """Kuralın ETİKETLERİNİ değiştir: aynı ölçüm, yeni harf. Kural yeniden bağlanır."""
        if self.last is None:
            return None
        rid = f"{self.last.rid}~{len(self.rules) + 1}"
        rule = Rule(rid=rid, feature=feature or self.last.feature, mapping=dict(mapping),
                    domain=self.last.domain, source_text=self.last.source_text,
                    derived_from=self.last.rid, exceptions=list(self.last.exceptions or []),
                    shape=self.last.shape, threshold=self.last.threshold, op=self.last.op)
        self.rules[rid] = rule
        self.last = rule
        self.chain = None
        self._trace("relabel", {"from": rule.derived_from, "to": rid, "map": mapping})
        return rule

    @staticmethod
    def _domain_of(low: str) -> str:
        for d, words in (("mutfak", ("mutfak", "malzeme", "tuzlu", "tatlı", "yemek", "tarif")),
                         ("lojistik", ("lojistik", "paket", "kargo", "sevkiyat", "acil")),
                         ("arıcılık", ("arı", "kovan", "bal", "petek")),
                         ("dil", ("harf", "sesli", "sözcük", "cümle"))):
            if any(w in low for w in words):
                return d
        return "genel"

    def transfer(self, new_mapping: dict[str, str] | None = None,
                 feature: str | None = None, domain: str = "") -> Rule | None:
        """Son kuralın YAPISINI başka alana taşı."""
        if self.last is None:
            return None
        rule = self.last.transfer(feature or self.last.feature,
                                  new_mapping or dict(self.last.mapping),
                                  domain or self.last.domain)
        self.rules[rule.rid] = rule
        self.last = rule
        self._trace("transfer", {"from": rule.derived_from, "to": rule.rid,
                                 "feature": rule.feature, "domain": rule.domain})
        return rule

    # ------------------------------------------------------------------ uygulama
    def apply(self, item: str, rule: Rule | None = None) -> tuple[str | None, dict]:
        # Zincir kuruluysa iki adım birlikte yürür (ara sonuç taşınır).
        if rule is None and self.chain is not None:
            label, tr = self.chain.apply(item)
            self._trace("apply", tr)
            return label, tr
        r = rule or self.last
        if r is None:
            self.failures.append("apply:no_rule")
            return None, {"error": "no_rule"}
        label, tr = r.apply(item)
        self._trace("apply", tr)
        return label, tr

    def apply_set(self, items: list[str], rule: "Rule | None" = None) -> tuple[list[dict], dict]:
        """Küme üzerinde uygula: sıra şekli burada anlam kazanır ('en çok olan')."""
        r = rule or self.last
        if r is None:
            self.failures.append("apply_set:no_rule")
            return [], {"error": "no_rule"}
        rows, tr = r.apply_set(list(items))
        self._trace("apply_set", tr)
        return rows, tr

    def counterfactual_apply(self, item: str) -> dict:
        """KARŞIOLGUSAL ÖZ-ERİŞİM: "kural olmasaydı ne derdim?"

        Aynı girdi iki kez yürütülür: (1) bağlı kural/ zincir varken, (2) kural
        DEVRE DIŞI bırakılarak. İkinci koşuda ölçüm yine yapılır — çünkü ölçüm
        operatörü kuraldan bağımsızdır; ama ETİKET verilemez. Fark, kuralın
        katkısının tam olarak nerede olduğunu gösterir: ölçümde değil, eşlemede.
        """
        saved_last, saved_chain = self.last, self.chain
        with_label, with_trace = self.apply(item)
        self.last, self.chain = None, None
        try:
            without_label, without_trace = self.apply(item)
        finally:
            self.last, self.chain = saved_last, saved_chain
        measured = None
        for src in (with_trace or {}, without_trace or {}):
            if src.get("measured") is not None:
                measured = src.get("measured")
                break
            step = (src or {}).get("step1") or {}
            if step.get("measured") is not None:
                measured = step.get("measured")
                break
        feat = (with_trace or {}).get("feature")
        if feat is None:
            feat = ((with_trace or {}).get("step1") or {}).get("feature")
        name = FEATURES.get(feat or "", ("ölçüm",))[0]
        self._trace("counterfactual", {"item": item, "with": with_label,
                                       "without": without_label, "measured": measured})
        return {"item": item, "with": with_label, "without": without_label,
                "measured": measured, "feature_name": name,
                "rule": getattr(saved_last, "rid", ""), "ablatable": saved_last is not None}

    def count(self, feature: str, item: str) -> int:
        n = int(FEATURES.get(feature, FEATURES["letter_count"])[1](item))
        self._trace("count", {"feature": feature, "item": item, "n": n})
        return n

    # ------------------------------------------------------------------ plan
    def plan(self, goal: str, n_steps: int = 10) -> list[dict]:
        """Hedefi adımlara ayır. Adımlar cümle değil, (eylem, hedef, ölçüt) üçlüsüdür.

        Bu, hazır plan metni değildir: her adım hedefin kendisinden ve ölçülebilir
        bir kontrol noktasından türetilir. `revise()` hedef değişince aynı makineyi
        yeniden çalıştırır; eski adımlar `superseded` olarak işaretlenir.
        """
        g = (goal or "").strip() or "belirsiz"
        n = max(3, min(int(n_steps), self.MAX_PLAN_STEPS))
        steps = []
        for i in range(1, n + 1):
            if i == 1:
                act, crit = "durum", "başlangıç ölçüldü"
            elif i == n:
                act, crit = "doğrula", "hedef_metrigi"
            elif i == n - 1:
                act, crit = "toparla", "artık kalmadı"
            elif i % 3 == 0:
                act, crit = "ölç", "ilerleme sayısal"
            else:
                act, crit = "yürüt", f"adım {i} tamam"
            steps.append({"i": i, "act": act, "goal": g, "criterion": crit,
                          "status": "open"})
        self.wm.write("plan_goal", g, "plan")
        self.wm.write("plan_n", n, "plan")
        self._trace("plan", {"goal": g, "steps": n})
        return steps

    def revise(self, old: list[dict], new_goal: str, n_steps: int = 10) -> list[dict]:
        for s in old:
            s["status"] = "superseded"
        new = self.plan(new_goal, n_steps)
        self.wm.write("plan_goal", new_goal, "revise")
        self._trace("revise", {"from": old[0]["goal"] if old else "", "to": new_goal})
        return new

    # ------------------------------------------------------------------ iz
    def _trace(self, kind: str, rec: dict) -> None:
        r = {"kind": kind}
        r.update(rec)
        self.trace.append(r)
        if len(self.trace) > 200:
            del self.trace[:-200]

    def last_trace(self, n: int = 1) -> list[dict]:
        return self.trace[-int(n):]

    def state_dict(self) -> dict:
        return {"rules": [{"rid": r.rid, "feature": r.feature, "mapping": r.mapping,
                           "domain": r.domain, "source": r.source_text,
                           "derived_from": r.derived_from, "uses": r.uses}
                          for r in self.rules.values()],
                "relation_rules": [rule.state_dict() for rule in self.relation_rules.values()],
                "last": self.last.rid if self.last else "",
                "wm": self.wm.state_dict()}

    def load_state_dict(self, d: dict | None) -> None:
        if not isinstance(d, dict):
            return
        try:
            self.rules = {}
            for rec in (d.get("rules") or []):
                r = Rule(rid=str(rec.get("rid")), feature=str(rec.get("feature")),
                         mapping=dict(rec.get("mapping") or {}),
                         domain=str(rec.get("domain") or "genel"),
                         source_text=str(rec.get("source") or ""),
                         derived_from=str(rec.get("derived_from") or ""),
                         uses=int(rec.get("uses", 0) or 0))
                self.rules[r.rid] = r
            self.relation_rules = {}
            for rec in (d.get("relation_rules") or []):
                rule = RelationRule(
                    rid=str(rec.get("rid")), relation=str(rec.get("relation")),
                    verb=str(rec.get("verb")), status=str(rec.get("status") or "probation"),
                    syntax=str(rec.get("syntax") or "SVO"),
                    examples=[dict(row) for row in (rec.get("examples") or [])],
                    uses=int(rec.get("uses", 0) or 0),
                )
                self.relation_rules[rule.rid] = rule
            lid = str(d.get("last") or "")
            self.last = self.rules.get(lid) or (list(self.rules.values())[-1] if self.rules else None)
            self.wm.load_state_dict(d.get("wm") or {})
        except Exception:
            pass


def _fixture() -> list[str]:
    """Öz-test cümleleri VERİDEN gelir (`nora_priors.json` -> `apply.selftest`).

    Test metni de bir sözdür; koda gömülürse iskelet sayacı haklı olarak büyür.
    Veri yoksa test susar — uydurma cümle üretmez.
    """
    try:
        import json
        import pathlib
        blob = json.loads(pathlib.Path(__file__).with_name("nora_priors.json")
                          .read_text(encoding="utf-8"))
        return [str(p.get("text", "")) for p in
                ((blob.get("kinds") or {}).get("apply.selftest") or {}).get("priors") or []
                if str(p.get("text", "")).strip()]
    except Exception:
        return []


def _selftest() -> None:
    fx = _fixture()
    if len(fx) < 5:
        print("selftest: no data")
        return
    it = Interpreter()
    r = it.bind(fx[0])
    print("kural:", r)
    for s in fx[1:4]:
        print(s, "->", it.apply(s)[0])
    print("transfer:", it.transfer({"odd": "tuzlu", "even": "tatlı"},
                                   feature="word_count", domain="mutfak"))
    print("mutfak:", it.apply(fx[4])[0])
    print("trace:", it.last_trace(3))


if __name__ == "__main__":
    _selftest()
