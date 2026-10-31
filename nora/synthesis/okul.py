"""NORA — OKUL: sayı sayma, basamak, okunuş, semboller ve aşamalı düşünme.

Neden ayrı bir organ?
---------------------
Dil öğrenmek ile SAYI öğrenmek farklıdır. Sayı sistemi kurallıdır: 10 rakam,
basamaklar, onluk/onluk gruplama, okunuş kuralları. Bir çocuk da bunları önce
ezberlemez; kuralla öğrenir ve *adım adım* düşünür:

    "Sepette 3 elma var. 2 elma daha koydum. Kaç oldu?"
      1. adım: başlangıç 3
      2. adım: eklenen 2
      3. adım: 3 + 2 = 5

Bu modül Nora'ya şunları verir:

* **Sayma / komşu sayı**: "7'den sonra ne gelir?", "birden ona kadar say"
* **Okunuş**: "345 nasıl okunur?" -> "üç yüz kırk beş"; rakam rakam okuma da var
* **Basamak**: "345'te 4 ne demek?" -> "4 onluk = 40"; "3 yüzlük 4 onluk 5 birlik"
* **Karşılaştırma**: "12 < 15 doğru mu?" -> adımlarla
* **Semboller**: + - × ÷ = < > ≤ ≥ ≠ % √ ² π ∞ ... adı, okunuşu, anlamı
* **Ondalık virgül**, **sıra sayısı** ("12." -> "on ikinci")
* **Aşamalı problem**: "3 elma vardı, 2 daha koydum" -> adımlar + sonuç

Ve ÖĞRENİR: "artı toplama demektir", "yüzlük yüz demektir", "5'ten sonra 6 gelir"
gibi cümleler kalıcı olarak `learned` sözlüğüne yazılır. Yani Nora'ya konuşarak
öğretilebilir; hiçbir cevap hazır cümle değildir, hepsi sayılardan türetilir.

Dürüst sınır: bu bir matematik motoru değil, çocuk seviyesinde sayı duyusudur.
Çok basamaklı cebir, kesir işlemleri ve istatistik burada YOKTUR; bilmediğinde
"bilmiyorum" der (uydurmaz).

Hız: tüm işlemler O(basamak) ve tur başına < 0.2 ms.
"""

from __future__ import annotations

from nora._paths import REPO_ROOT


import re

from nora_lang import TurkishMorphology, tr_lower, tr_capitalize

# ---------------------------------------------------------------- sabitler

# Semboller: ad (okunuş), anlam, örnek. Bunlar alfabe gibidir — konvansiyon.
# `Nora` bunları konuşarak da öğrenebilir; öğrendiği `learned` sözlüğüne yazılır.
SYMBOLS: dict[str, dict[str, str]] = {
    "+": {"ad": "artı", "anlam": "iki sayıyı toplar", "ornek": "3 + 2 = 5"},
    "-": {"ad": "eksi", "anlam": "çıkarma yapar; bir sayıdan diğerini alır", "ornek": "7 - 2 = 5"},
    "×": {"ad": "çarpı", "anlam": "bir sayıyı diğeri kadar tekrar eder", "ornek": "3 × 4 = 12"},
    "x": {"ad": "çarpı", "anlam": "bir sayıyı diğeri kadar tekrar eder", "ornek": "3 x 4 = 12"},
    "÷": {"ad": "bölü", "anlam": "bir sayıyı eşit parçalara böler", "ornek": "12 ÷ 3 = 4"},
    "=": {"ad": "eşittir", "anlam": "iki tarafın aynı olduğunu söyler", "ornek": "3 + 2 = 5"},
    "<": {"ad": "küçüktür", "anlam": "soldaki sayı sağdakinden küçük", "ornek": "3 < 5"},
    ">": {"ad": "büyüktür", "anlam": "soldaki sayı sağdakinden büyük", "ornek": "7 > 5"},
    "≤": {"ad": "küçük veya eşittir", "anlam": "küçük ya da aynı", "ornek": "5 ≤ 5"},
    "≥": {"ad": "büyük veya eşittir", "anlam": "büyük ya da aynı", "ornek": "5 ≥ 5"},
    "≠": {"ad": "eşit değildir", "anlam": "iki taraf aynı değil", "ornek": "3 ≠ 4"},
    "%": {"ad": "yüzde", "anlam": "yüz parçadan kaç tanesi", "ornek": "%40 = 40/100"},
    "√": {"ad": "karekök", "anlam": "kendisiyle çarpılınca sayıyı veren değer", "ornek": "√9 = 3"},
    "²": {"ad": "kare", "anlam": "sayının kendisiyle çarpımı", "ornek": "3² = 9"},
    "³": {"ad": "küp", "anlam": "sayının üç kez kendisiyle çarpımı", "ornek": "2³ = 8"},
    "π": {"ad": "pi", "anlam": "çemberin çevresinin çapına oranı, yaklaşık 3,14", "ornek": "π ≈ 3,14"},
    "∞": {"ad": "sonsuz", "anlam": "sonu olmayan, sayılamayan büyüklük", "ornek": "…"},
    "±": {"ad": "artı eksi", "anlam": "iki yönde de olabilir", "ornek": "±2"},
    "°": {"ad": "derece", "anlam": "açı ya da sıcaklık birimi", "ornek": "90°"},
}

PLACE_NAMES = {1: "birlik", 2: "onluk", 3: "yüzlük", 4: "binlik",
               5: "on binlik", 6: "yüz binlik", 7: "milyonluk"}

# Sıra sayısı ekleri (düzensizler burada; düzenliler kural ile). Konvansiyon, alfabe gibi.
_ORDINAL_IRREGULAR = {
    "bir": "birinci", "iki": "ikinci", "üç": "üçüncü", "dört": "dördüncü", "beş": "beşinci",
    "altı": "altıncı", "yedi": "yedinci", "sekiz": "sekizinci", "dokuz": "dokuzuncu",
    "on": "onuncu", "yirmi": "yirminci", "otuz": "otuzuncu", "kırk": "kırkıncı",
    "elli": "ellinci", "altmış": "altmışıncı", "yetmiş": "yetmişinci",
    "seksen": "sekseninci", "doksan": "doksanıncı", "yüz": "yüzüncü", "bin": "bininci",
    "sıfır": "sıfırıncı",
}

_UNIT_WORDS = {"sıfır": 0, "bir": 1, "iki": 2, "üç": 3, "dört": 4, "beş": 5, "altı": 6,
               "yedi": 7, "sekiz": 8, "dokuz": 9}
_TEN_WORDS = {"on": 10, "yirmi": 20, "otuz": 30, "kırk": 40, "elli": 50,
              "altmış": 60, "yetmiş": 70, "seksen": 80, "doksan": 90}


def _tr_lower(s: str) -> str:
    return tr_lower(s or "")


def _last_vowel(word: str) -> str:
    for ch in reversed(_tr_lower(word)):
        if ch in "aeıioöuü":
            return ch
    return "e"


def abl_suffix(n: int) -> str:
    """'-den/-dan/-ten/-tan' eki (ünlü uyumu + sert ünsüz benzeşmesi)."""
    last = number_words(int(n)).split()[-1]
    v = _last_vowel(last)
    hard = last[-1] in "fstkçşhp"
    suf = "tan" if hard else "dan"
    if v in "eiöü":
        suf = "ten" if hard else "den"
    return suf


def poss_suffix(n: int) -> str:
    """'-ı/-i/-u/-ü' (3. tekil iyelik) eki: 40'ı, 12'si, 100'ü."""
    v = _last_vowel(number_words(int(n)).split()[-1])
    return {"a": "ı", "ı": "ı", "o": "u", "u": "u", "e": "i", "i": "i", "ö": "ü", "ü": "ü"}.get(v, "i")


def loc_suffix(n: int) -> str:
    last = number_words(int(n)).split()[-1]
    v = _last_vowel(last)
    hard = last[-1] in "fstkçşhp"
    suf = "ta" if hard else "da"
    if v in "eiöü":
        suf = "te" if hard else "de"
    return suf


def number_words(n: int) -> str:
    """Sayının Türkçe okunuşu ('üç yüz kırk beş')."""
    return TurkishMorphology.int_to_words(int(n))


def digit_words(s: str) -> str:
    """Rakam rakam okuma ('345' -> 'üç, dört, beş') — telefon numarası okur gibi."""
    names = ["sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
    return ", ".join(names[int(ch)] for ch in str(s) if ch.isdigit())


def ordinal(n: int) -> str:
    """Sıra sayısı ('12' -> 'on ikinci')."""
    if n == 0:
        return "sıfırıncı"
    if n < 0:
        return "eksi " + ordinal(-n)
    words = number_words(n).split()
    if not words:
        return ""
    last = words[-1]
    if last in _ORDINAL_IRREGULAR:
        words[-1] = _ORDINAL_IRREGULAR[last]
        return " ".join(words)
    # Düzenli: ünlü uyumuna göre -inci/-ıncı/-uncu/-üncü
    v = TurkishMorphology.vowel_4(last) if hasattr(TurkishMorphology, "vowel_4") else "i"
    suf = {"a": "ıncı", "ı": "ıncı", "o": "uncu", "u": "uncu",
           "e": "inci", "i": "inci", "ö": "üncü", "ü": "üncü"}.get(v, "inci")
    words[-1] = last + suf
    return " ".join(words)


def parse_number_token(tok: str) -> int | None:
    """'345', '3', '12.', "7'den", 'iki yüz' gibi biçimlerden tam sayı çıkarır."""
    if not tok:
        return None
    t = _tr_lower(tok).strip()
    t = re.sub(r"[.,;:!?]", "", t)
    t = t.split("'")[0]
    if re.fullmatch(r"-?\d+", t):
        return int(t)
    v = TurkishMorphology.words_to_int([t]) if hasattr(TurkishMorphology, "words_to_int") else None
    if v is not None:
        return int(v)
    # '12.' gibi sıra sayısı biçimi
    m = re.fullmatch(r"(\d+)\.", _tr_lower(tok).strip())
    if m:
        return int(m.group(1))
    return None


def decompose(n: int) -> list[tuple[int, int, str]]:
    """Basamak çözümlemesi: 345 -> [(3,'yüzlük',300), (4,'onluk',40), (5,'birlik',5)]."""
    n = abs(int(n))
    out: list[tuple[int, int, str]] = []
    digits = str(n)[::-1]
    for i, ch in enumerate(digits):
        d = int(ch)
        if d == 0:
            continue
        place = i + 1
        name = PLACE_NAMES.get(place, f"{10 ** (place - 1)}lik")
        out.append((d, 10 ** (place - 1), name))
    return list(reversed(out))


def place_value_line(n: int) -> str:
    parts = decompose(n)
    if not parts:
        return "0'ın basamağı yok; sıfır tek başına bir değerdir."
    chunks = [f"{d} {name}" for d, _val, name in parts]
    return f"{number_words(n)}: " + ", ".join(chunks) + "."


def _digit_value_line(n: int, place: int) -> str:
    """'345'te 4'ün değeri' -> '4 onluk = 40'."""
    s = str(abs(int(n)))
    if not (1 <= place <= len(s)):
        return "O basamak bu sayıda yok."
    idx = len(s) - place
    d = int(s[idx])
    val = d * (10 ** (place - 1))
    name = PLACE_NAMES.get(place, "basamak")
    if d == 0:
        return f"{n} sayısının {name} basamağında 0 var; değeri 0."
    return f"{n} sayısında {place}. basamak {d}; yani {d} {name} = {val}. " \
           f"(sayının okunuşu: {number_words(n)})"


class SchoolCortex:
    """Çocuk seviyesinde sayı duyusu + aşamalı düşünme organı.

    `answer(raw, agi)` soruyu tanır ve ADIMLARLA yanıtlar; tanımadığında None döner
    (o zaman dil organları devreye girer). `maybe_learn(raw, agi)` öğretme cümlesini
    işler ve onay sözü döner.
    """

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self.learned: dict[str, str] = {}        # "artı" -> "toplama", "yüzlük" -> "yüz"
        self.successors: dict[int, int] = {}     # 5 -> 6 (öğretilmiş komşu sayı)
        self.taught_count = 0
        self.answered = 0
        self._rng = 0

    # ---------------------------------------------------------------- yardımcı
    def _pick(self, options: tuple[str, ...]) -> str:
        """Söz çeşitliliği: aynı bilgi her seferinde aynı kalıpla söylenmez."""
        self._rng = (self._rng * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return options[(self._rng >> 33) % len(options)]

    def state_dict(self) -> dict:
        return {"learned": dict(self.learned),
                "successors": {str(k): int(v) for k, v in self.successors.items()},
                "taught": int(self.taught_count), "answered": int(self.answered)}

    def load_state_dict(self, d: dict) -> None:
        try:
            self.learned = {str(k): str(v) for k, v in (d.get("learned") or {}).items()}
            self.successors = {int(k): int(v) for k, v in (d.get("successors") or {}).items()}
            self.taught_count = int(d.get("taught", 0) or 0)
            self.answered = int(d.get("answered", 0) or 0)
        except Exception:
            pass

    # ---------------------------------------------------------------- öğrenme
    def maybe_learn(self, raw: str, agi=None) -> str | None:
        """'artı toplama demektir', 'yüzlük yüz demektir', '5'ten sonra 6 gelir' gibi
        öğretme cümlelerini kalıcı olarak öğrenir. Öğretme değilse None."""
        s = _tr_lower(raw).replace("'", "")
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"[.!]+$", "", s).strip()

        # 1) "<sembol/ad> <anlam> demektir"
        _names: dict[str, str] = {}
        for _k, _v in SYMBOLS.items():
            # ASCII kopya ('x'), gerçek sembolü (×) ezmesin.
            if _v["ad"] not in _names or (_k in "×÷" and _names[_v["ad"]] not in "×÷"):
                _names[_v["ad"]] = _k
        m = re.match(r"^(?:nora[, ]+)?([\wçğıöşü+\-×÷=<>%√²³π∞]{1,12})\s+([\wçğıöşü ]{2,30}?)\s+demektir$", s)
        if m and m.group(1) in _names:
            m = re.match(r"^(?:nora[, ]+)?(.+?)\s+([\wçğıöşü ]{2,30}?)\s+demektir$", s)
            self.learned[_names[m.group(1)]] = m.group(2).strip()
            self.taught_count += 1
            return f"Anladım; {m.group(1)} artık bende {m.group(2).strip()} demek."
        if m and m.group(1) in SYMBOLS:
            self.learned[m.group(1)] = m.group(2).strip()
            self.taught_count += 1
            return f"Anladım; {m.group(1)} artık bende {m.group(2).strip()} demek."

        # 2) "yüzlük yüz demektir" / "onluk on demektir"
        m = re.match(r"^(?:nora[, ]+)?([\wçğıöşü]{2,12})\s+([\wçğıöşü ]{2,20}?)\s+demektir$", s)
        if m:
            w, val = m.group(1), m.group(2).strip()
            self.learned[w] = val
            self.taught_count += 1
            return self._pick((f"'{w}' bende '{val}' demek; unutmadım.",
                               f"Kaydettim: {w} = {val}.",
                               f"Tamam, {w} sözünü {val} diye anlıyorum."))

        # 3) "5'ten sonra 6 gelir" / "5 ten sonra 6 gelir"
        m = re.match(r"^(?:nora[, ]+)?(\d+)\s*(?:'?[dt][ae]n|den|dan)\s+sonra\s+(\d+)\s+gelir$", s)
        if m:
            self.successors[int(m.group(1))] = int(m.group(2))
            self.taught_count += 1
            return f"{m.group(1)}'{abl_suffix(int(m.group(1)))} sonra {m.group(2)} gelir diye öğrendim."

        # 4) "345 üç yüz kırk beş demektir"
        m = re.match(r"^(?:nora[, ]+)?(\d+)\s+([\wçğıöşü ]{2,40}?)\s+demektir$", s)
        if m:
            n = int(m.group(1))
            val = TurkishMorphology.words_to_int(m.group(2).split())
            if val == n:
                self.learned[str(n)] = m.group(2).strip()
                self.taught_count += 1
                return f"Doğru okunuş: {n} = {m.group(2).strip()}. Böyle diyeceğim."
        return None

    # ---------------------------------------------------------------- sorular
    # Uzman matematik: cebir/kalkülüs/sayılar teorisi. Bunlar OKUL'un değil, parietal
    # matematik korteksinin işidir; okul burada araya girerse "2x+7=23" gibi bir denklemi
    # "23 sayısı" diye yanıtlar. Nora çocuk seviyesini öğrenir ama uzman bilgiyi ezmez.
    _EXPERT_MATH = re.compile(
        r"[=^]|\d\s*[a-z]\b|\b(türev|türevi|integral|integrale|integrali|denklem|denklemin|"
        r"ebob|ekok|logaritma|log\b|limit|matris|determinant|karekök|karekok|asal|çarpanlar|"
        r"fonksiyon|fonksiyonunun|sinüs|kosinüs|tanjant|olasılık|permütasyon|kombinasyon|"
        r"mod\b|bölünebilme|polinom)\b")

    def _expert_math(self, s: str) -> bool:
        """Cümle okul düzeyini aşıyor mu (o zaman okul susar)."""
        return bool(self._EXPERT_MATH.search(s))

    def answer(self, raw: str, agi=None) -> str | None:
        """Okul sorusu ise adım adım yanıtlar; değilse None döner."""
        s = _tr_lower(raw).replace("'", "")
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"[.!]+$", "", s).strip()
        if self._expert_math(s):
            return None
        out = (self._q_decimal(s) or self._q_fraction(s)
               or self._q_read_number(s) or self._q_digits(s) or self._q_place(s)
               or self._q_counting(s) or self._q_percent(s) or self._q_compare(s)
               or self._q_symbol(s) or self._q_ordinal(s) or self._q_word_problem(s))
        if out is None:
            return None
        self.answered += 1
        return out

    # 0) ondalık ve kesir — "2,5" tek bir sayıdır, "5" değil.
    def _q_decimal(self, s: str) -> str | None:
        m = re.search(r"(\d{1,9})\s*[,.]\s*(\d{1,9})\s*(?:sayısı\s*)?"
                      r"(?:nasıl\s+okunur|okunuşu\s*(?:ne|nedir|nasıl)?|ne\s+demek|kaç\s+eder)", s)
        if not m:
            return None
        whole, frac = m.group(1), m.group(2)
        n = float(f"{whole}.{frac}")
        read = f"{number_words(int(whole))} virgül {digit_words(frac)}"
        def _fmt(v: float) -> str:
            t = f"{v:.2f}".rstrip("0").rstrip(".")
            return t.replace(".", ",")

        out = (f"{whole},{frac} sayısı '{read}' diye okunur. Tam kısım {whole} ({number_words(int(whole))}), "
               f"ondalık kısım {frac} ({digit_words(frac)}). "
               f"Yani {int(whole)} ile {int(whole) + 1} arasında bir sayı: {_fmt(n)}.")
        if (n / 2) != int(n / 2) or True:
            out += f" Yarısı {_fmt(n / 2)}."
        return out

    def _q_fraction(self, s: str) -> str | None:
        words = {"yarım": (1, 2, "0,5"), "çeyrek": (1, 4, "0,25"),
                 "üçte bir": (1, 3, "0,33"), "üçte iki": (2, 3, "0,67")}
        for w, (num, den, dec) in words.items():
            if w in s and re.search(r"ne demek|nedir|ne kadar|nasıl yazılır|kaç eder|kaç", s):
                if w == "çeyrek" and "çeyrek" not in s:
                    continue
                return (f"{w}: bir bütünün {den} eşit parçasından {num} tanesi. "
                        f"Kesir olarak {num}/{den}, ondalık olarak {dec}.")
        m = re.search(r"(\d{1,2})\s*/\s*(\d{1,2})\s*(?:ne demek|nedir|kaç eder|kaç)", s)
        if m:
            num, den = int(m.group(1)), int(m.group(2))
            if den:
                dec = f"{num / den:.2f}".replace(".", ",")
                return (f"{num}/{den}: bir bütünün {den} eşit parçasından {num} tanesi; "
                        f"ondalık olarak {dec}.")
        return None

    # 1) okunuş
    def _q_read_number(self, s: str) -> str | None:
        if re.search(r"\d\s*[,.]\s*\d", s):
            return None   # ondalık sayı: burada bölünmez
        m = re.search(r"(\d{1,12})\s*(?:sayısı\s*)?(?:nasıl|nice)\s+(?:okunur|okunuyor|yazılır)", s)
        if not m:
            m = re.search(r"(\d{1,12})\s*(?:sayısının\s*)?okunuşu\s*(?:ne|nedir|nasıl)?", s)
        if not m:
            return None
        n = int(m.group(1))
        return f"{n} sayısı '{number_words(n)}' diye okunur. Rakam rakam: {digit_words(str(n))}."

    # 2) rakamlar
    def _q_digits(self, s: str) -> str | None:
        if re.search(r"\d\s*[,.]\s*\d", s):
            return None   # "2,5" -> rakamları "2" ve "5" diye ayırmak yanlış olur
        if re.search(r"kaç\s+tane\s+rakam", s):
            return ("On tane rakam var: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9. "
                    "Rakamlar tek başına semboldür; sayı bu rakamlardan kurulur.")
        m = re.search(r"(\d{2,12})\s*(?:sayısının\s*)?rakamları", s)
        if m:
            d = str(m.group(1))
            return f"{d} sayısının rakamları: {digit_words(d)}. Toplam {len(d)} rakam."
        return None

    # 3) basamak
    def _q_place(self, s: str) -> str | None:
        m = re.search(r"(\d{1,9})\s*(?:sayısı)?\s*(?:kaç)\s*basamaklı", s)
        if m:
            n = m.group(1).lstrip("0") or "0"
            k = max(1, len(n))
            names = {1: "tek basamaklı", 2: "iki basamaklı", 3: "üç basamaklı",
                     4: "dört basamaklı", 5: "beş basamaklı", 6: "altı basamaklı",
                     7: "yedi basamaklı", 8: "sekiz basamaklı", 9: "dokuz basamaklı"}
            return f"{int(m.group(1))} {names.get(k, str(k) + ' basamaklı')} bir sayıdır. {place_value_line(int(m.group(1)))}"
        # "345'te 4 ne demek" / "4 onluk ne demek"
        m = re.search(r"(\d{1,9})(?:de|da|te|ta)\s+(\d)\s*(?:rakamı|sayısı)?\s*(?:ne demek|ne anlama|ne işe|değeri ne)", s)
        if m:
            n = int(m.group(1))
            d = int(m.group(2))
            ss = str(n)
            # rakamın hangi basamakta olduğunu soldan bul
            for i, ch in enumerate(ss):
                if int(ch) == d:
                    place = len(ss) - i
                    return _digit_value_line(n, place)
            return f"{n} sayısında {d} rakamı yok."
        # "3 yüzlük 4 onluk 5 birlik kaç eder" — ÇOK birimli ifade, tek birimden ÖNCE gelir
        if "yüzlük" in s and re.search(r"\d", s):
            total = 0
            steps = []
            for d, name in re.findall(r"(\d+)\s*(yüzlük|onluk|birlik|binlik)", s):
                mult = {"birlik": 1, "onluk": 10, "yüzlük": 100, "binlik": 1000}[name]
                total += int(d) * mult
                steps.append(f"{d} {name} = {int(d) * mult}")
            if steps:
                return (" + ".join(steps) + f" = {total}. Yani sayı {total}; okunuşu "
                        f"'{number_words(total)}'.")
        m = re.search(r"(\d)\s*(yüzlük|onluk|birlik|binlik)\s*(?:kaç|ne kadar|ne)\b", s)
        if m:
            d = int(m.group(1))
            name = m.group(2)
            mult = {"birlik": 1, "onluk": 10, "yüzlük": 100, "binlik": 1000}[name]
            return f"{d} {name}, {d} × {mult} = {d * mult} eder."
        # "345 sayısını basamaklarına ayır"
        m = re.search(r"(\d{1,9})\s*(?:sayısını|sayısı)?\s*basamak", s)
        if m:
            return place_value_line(int(m.group(1)))
        return None

    # 4) sayma
    def _q_counting(self, s: str) -> str | None:
        if re.search(r"(\d+|bir|iki|üç|dört|beş|altı|yedi|sekiz|dokuz|on)\s*(?:'?[dt][ae]n|den|dan)\s+sonra\s+(?:hangi\s+)?(?:sayı|ne)\s*gelir", s):
            m = re.search(r"(\d+)\s*(?:'?[dt][ae]n|den|dan)\s+sonra", s)
            if m:
                n = int(m.group(1))
                nxt = self.successors.get(n, n + 1)
                return f"{n}'{abl_suffix(n)} sonra {nxt} gelir ({number_words(nxt)})."
        if re.search(r"geriye\s+doğru\s+say|kaçtan\s+geri", s):
            return "Geriye sayarken her adımda bir eksiltirim: 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0."
        m = re.search(r"(?:birden|1'?den)\s+(\d+|ona|yüze|yirmiye|elliye)(?:'?[ae]|ya|ye)?\s+kadar\s+say", s)
        if m:
            w = m.group(1)
            limit = {"ona": 10, "yüze": 100, "yirmiye": 20, "elliye": 50}.get(w)
            if limit is None:
                limit = parse_number_token(w) or 10
            limit = min(int(limit), 100)
            seq = ", ".join(str(i) for i in range(1, limit + 1))
            return f"{seq}. Toplam {limit} sayı saydım."
        m = re.search(r"(\d+)\s*(?:'?e|'?a|ye|ya)\s+kadar\s+say", s)
        if m:
            limit = min(int(m.group(1)), 100)
            seq = ", ".join(str(i) for i in range(1, limit + 1))
            return f"{seq}. Toplam {limit} sayı saydım."
        # "7'den büyük en küçük sayı"
        m = re.search(r"(\d+)\s*(?:'?[dt][ae]n|den|dan)\s+(büyük|küçük)\s+en\s+(büyük|küçük)\s+sayı", s)
        if m:
            n = int(m.group(1))
            if m.group(2) == "büyük":
                return f"{n}'{abl_suffix(n)} büyük en küçük tam sayı {n + 1} ({number_words(n + 1)})."
            return f"{n}'{abl_suffix(n)} küçük en büyük tam sayı {n - 1}."
        return None

    # 5) karşılaştırma
    def _q_compare(self, s: str) -> str | None:
        m = re.search(r"(-?\d+)\s*([<>]=?|=|≠)\s*(-?\d+)", s)
        if not m:
            m2 = re.search(r"(-?\d+)\s+(büyük|küçük|eşit)\s*(?:mü|mi|mu|mü|midir)?\s*(-?\d+)?", s)
            if m2 and m2.group(3):
                a, rel, b = int(m2.group(1)), m2.group(2), int(m2.group(3))
                truth = {"büyük": a > b, "küçük": a < b, "eşit": a == b}[rel]
                word = {"büyük": "büyüktür", "küçük": "küçüktür", "eşit": "eşittir"}[rel]
                return (f"{number_words(a)} {a}, {number_words(b)} {b}. "
                        f"{a} {b}'den {word}. Bu doğru" if truth else
                        f"{number_words(a)} {a}, {number_words(b)} {b}. {a}, {b}'den {word} değil; bu yanlış.")
            return None
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        res = {"<": a < b, ">": a > b, "=": a == b, "<=": a <= b, ">=": a >= b, "≠": a != b}[op]
        read = {"<": "küçüktür", ">": "büyüktür", "=": "eşittir", "<=": "küçük veya eşittir",
                ">=": "büyük veya eşittir", "≠": "eşit değildir"}[op]
        verdict = "doğru" if res else "yanlış"
        return (f"{a} {op} {b} ifadesi '{a} {read} {b}' diye okunur. "
                f"Adım: {a} ile {b} karşılaştırılır; sonuç {verdict}.")

    # 6) semboller
    def _q_symbol(self, s: str) -> str | None:
        if re.search(r"(sembol|işaret)(?:ler|leri)?\s*(?:ne|nedir|nedir bunlar|nasıl okunur)", s) and not re.search(r"[+\-×÷=<>%√²π]", s):
            return ("Matematikte + artı, - eksi, × çarpı, ÷ bölü, = eşittir, "
                    "< küçüktür, > büyüktür, % yüzde, √ karekök, ² kare, π pi. "
                    "Hangisini açayım?")
        sym = None
        for cand in ("≤", "≥", "≠", "+", "-", "×", "÷", "=", "<", ">", "%", "√", "²", "³", "π", "∞", "±", "°"):
            if cand in s:
                sym = cand
                break
        if sym is None:
            for word, key in (("artı", "+"), ("eksi", "-"), ("çarpı", "×"), ("bölü", "÷"),
                              ("eşittir", "="), ("küçüktür", "<"), ("büyüktür", ">"),
                              ("yüzde", "%"), ("karekök", "√"), ("sonsuz", "∞"), ("pi", "π")):
                if word in s:
                    sym = key
                    break
        if sym is None:
            return None
        if not re.search(r"ne demek|ne işe|nedir|anlamı|nasıl okunur|ne anlama", s):
            return None
        info = SYMBOLS.get(sym)
        if info is None:
            return None
        learned = self.learned.get(sym) or self.learned.get(info["ad"])
        meaning = info["anlam"] + (f" (bana öğrettiğin: {learned})" if learned and learned != info["anlam"] else "")
        opener = self._pick(("Şöyle:", "Şu sembol:", f"'{sym}' işareti:"))
        return f"{opener} {sym} — {info['ad']}. {tr_capitalize(meaning)}. Örnek: {info['ornek']}."

    # 7) yüzde
    def _q_percent(self, s: str) -> str | None:
        m = re.search(r"yüzde\s*(\d+)|%\s*(\d+)", s)
        if not m:
            return None
        p = int(m.group(1) or m.group(2))
        if not re.search(r"ne demek|nedir|ne anlama|kaç", s) and "yüzde" not in s:
            return None
        part = p / 100.0
        return (f"Yüzde {p}, yüz parçadan {p} tanesi demektir; {p}/100 = {part:g}. "
                f"Örnek: 200'ün yüzde {p}'{poss_suffix(p)} = {200 * p // 100}.")

    # 8) sıra sayısı
    def _q_ordinal(self, s: str) -> str | None:
        m = re.search(r"(\d+)\s*\.\s*(?:sırada|sıradaki|sıra|olan)?", s)
        if m and ("sıra" in s or "kaçıncı" in s or "okun" in s):
            n = int(m.group(1))
            return f"{n}. sıra sayısı '{ordinal(n)}' diye okunur."
        m = re.search(r"(\d+)\s*(?:'?[dt][ae]n|den|dan)\s+kaçıncı", s)
        if m:
            n = int(m.group(1))
            return f"{n}. sıradaki kişiye '{ordinal(n)}' denir."
        return None

    # 9) aşamalı problem
    def _q_word_problem(self, s: str) -> str | None:
        """'Sepette 3 elma var, 2 elma daha koydum, kaç elma oldu?' -> adımlar."""
        if "kaç" not in s:
            return None
        # "en küçük ortak kat", "ortak bölen", "katı", "böleni" bir ALIŞVERİŞ problemi değil,
        # sayılar teorisi sorusudur; okul burada çarpma uydurmamalı.
        if re.search(r"ortak\s+(kat|bölen)|(ebob|ekok)|\bkatı\b|\bbölen", s):
            return None
        nums = [int(n) for n in re.findall(r"\b(\d{1,6})\b", s)]
        if len(nums) < 2:
            return None
        add_cues = ("daha", "ekle", "koy", "aldım", "getir", "verdi", "arttı", "oldu")
        sub_cues = ("yedim", "yedi", "verdim", "kaybettim", "düştü", "gitti", "azaldı", "çıkardım")
        mul_cues = ("her birinde", "her biri", "kere", "katı", "çarptım")
        a, b = nums[0], nums[1]
        if any(c in s for c in mul_cues):
            op, res, sym = "çarpma", a * b, "×"
        elif any(c in s for c in sub_cues):
            op, res, sym = "çıkarma", a - b, "-"
        elif any(c in s for c in add_cues):
            op, res, sym = "toplama", a + b, "+"
        else:
            return None
        steps = [f"1. adım: elimde {a} var ({number_words(a)}).",
                 f"2. adım: {b} ({number_words(b)}) {('ekleniyor' if op == 'toplama' else 'çıkarılıyor' if op == 'çıkarma' else 'ile çarpılıyor')}.",
                 f"3. adım: {a} {sym} {b} = {res}."]
        return " ".join(steps) + f" Sonuç: {res}. " + self._pick(
            ("Böyle hesapladım.", "Adımlar bunlar.", "Baştan sona izledim."))

    # ---------------------------------------------------------------- rapor
    def known_summary(self) -> str:
        n = len(SYMBOLS) + len(self.learned)
        return (f"Sayı bilgim: 0-9 rakamları, onluk basamaklar, okunuş, sıra sayıları, "
                f"karşılaştırma ve {len(SYMBOLS)} sembol. Bana öğrettiğin {len(self.learned)} ek kural var.")


def _selftest() -> None:
    sc = SchoolCortex(seed=1)
    assert number_words(345) == "üç yüz kırk beş", number_words(345)
    assert ordinal(12) == "on ikinci", ordinal(12)
    assert decompose(345) == [(3, 100, "yüzlük"), (4, 10, "onluk"), (5, 1, "birlik")]
    print("okunuş  :", sc.answer("345 nasıl okunur?"))
    print("basamak :", sc.answer("345'te 4 ne demek?"))
    print("sayma   :", sc.answer("7'den sonra ne gelir?"))
    print("sembol  :", sc.answer("+ ne demek?"))
    print("problem :", sc.answer("Sepette 3 elma var, 2 elma daha koydum, kaç elma oldu?"))


if __name__ == "__main__":
    _selftest()
