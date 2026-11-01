# -*- coding: utf-8 -*-
"""Yapısal soru çözücüleri: takvim, sayı karşılaştırması, kelime problemi, boş kap sayımı,
'hangisi daha' karşılaştırması ve özellik yankısı.

Her çözücü (yanıt, gerekçe) çifti ya da None döndürür. Eşleşmeyen metin None alır ve
Nora'nın olağan yanıt yollarına geçer; bu modül yalnızca kalıbı açıkça tanınan soruları çözer.
"""

import re

from nora.language.lang import TurkishMorphology, tr_capitalize, tr_lower

WEEKDAYS = ("pazartesi", "salı", "çarşamba", "perşembe", "cuma", "cumartesi", "pazar")
DAY_OFFSET = {"dün": -1, "bugün": 0, "yarın": 1}
COND_SUFFIX = ("ysa", "yse", "ise", "sa", "se")
COND_WORDS = {"ise", "ysa", "yse"}
MI_SET = {"mı", "mi", "mu", "mü", "mıdır", "midir", "mudur", "müdür", "mısın", "misin", "musun", "müsün"}
ARTICLE_LIKE = ("bir", "ona", "onu", "onun", "onda", "ondan", "onlar", "onlara")

# Eksiltme: fiil kökü + koşul/istek ekleri ('yersem', 'çıkarırsak', 'harcarsa').
REMOVE_STEMS = ("yer", "ver", "çıkar", "harca", "kaybed", "sat", "eksil", "kır", "götür", "dağıt")
REMOVE_SUFFIX = {"", "ırsam", "irsem", "arsam", "ersem", "ırsan", "irsen", "arsan", "ersen",
                 "ırsak", "irsek", "arsak", "ersek", "ırsa", "irse", "arsa", "erse", "sak", "sek",
                 "sam", "sem", "san", "sen", "sa", "se"}
# Ekleme: açık ekleme fiilleri ('veriyor', 'aldı').
ADD_WORDS = {"veriyor", "verir", "verdi", "getiriyor", "getirdi", "aldı", "alıyor", "geldi", "geliyor",
             "ekledi", "ekleniyor", "kazandı", "buldu", "alırsan", "alırsa", "gelirse", "gelirsen",
             "eklenirse", "eklersen", "ekler"}

ANTONYMS = {"uzun": "kısa", "büyük": "küçük", "ağır": "hafif", "yüksek": "alçak", "hızlı": "yavaş",
            "geniş": "dar", "pahalı": "ucuz", "yaşlı": "genç", "kalın": "ince"}
ANTONYMS.update({v: k for k, v in list(ANTONYMS.items())})


def _words(raw: str) -> list[str]:
    """Küçük harfli sözcükler; liste numarası ('1.') ve yönlendirme işareti ('>') atılır."""
    s = re.sub(r"^[\s>]*\d+\s*[.)]\s+", "", tr_lower(raw.strip()))
    return re.findall(r"[\wçğıöşü]+", s)


def _numbers(words: list[str]) -> list[int]:
    """Sayılar sırasıyla. 'bir' belirteç, 'ona' zamir olduğundan sayılmaz."""
    out = []
    for w in words:
        if w in ARTICLE_LIKE:
            continue
        norm = TurkishMorphology.normalize_num_token(w)
        if norm is None:
            continue
        if re.fullmatch(r"[+-]?\d+", norm):
            out.append(int(norm))
        else:
            val = TurkishMorphology.words_to_int([norm])
            if val is not None:
                out.append(int(val))
    return out


def _is_removal(w: str) -> bool:
    return any(w.startswith(stem) and w[len(stem):] in REMOVE_SUFFIX for stem in REMOVE_STEMS)


def removal_any(words: list[str]) -> bool:
    return any(_is_removal(w) for w in words)


def weekday(raw: str) -> tuple[str, str] | None:
    """'Bugün pazartesiyse yarın hangi gündür?' ve 'Bugün pazartesi ise ...' kaydırmaları."""
    words = _words(raw)
    if "hangi" not in words:
        return None
    cond_idx, cond_day = -1, ""
    for i, w in enumerate(words):
        for name in sorted(WEEKDAYS, key=len, reverse=True):
            if w.startswith(name) and w[len(name):] in COND_SUFFIX:
                cond_idx, cond_day = i, name
                break
            if w == name and i + 1 < len(words) and words[i + 1] in COND_WORDS:
                cond_idx, cond_day = i, name
                break
        if cond_idx >= 0:
            break
    if cond_idx < 0:
        return None
    cond_ref = 0
    for w in words[:cond_idx]:
        if w in DAY_OFFSET:
            cond_ref = DAY_OFFSET[w]
    target = None
    for w in words[cond_idx + 1:]:
        if w in DAY_OFFSET:
            target = DAY_OFFSET[w]
    pos = WEEKDAYS.index(cond_day)
    if target is None:
        # 'Bugün çarşambaysa üç gün sonra hangi gün olur?': bugünden N gün ileri/geri.
        span = re.search(r"gün\s+(sonra|önce)", " ".join(words))
        nums = _numbers(words[cond_idx + 1:])
        if span is None or len(nums) != 1:
            return None
        delta = nums[0] if span.group(1) == "sonra" else -nums[0]
        res = WEEKDAYS[(pos - cond_ref + delta) % 7]
        expl = f"Bugünden {delta:+d} gün sonra takvim {res} gününe geldi"
        return f"{tr_capitalize(res)}.", expl
    res = WEEKDAYS[(pos + target - cond_ref) % 7]
    expl = f"{cond_day} gününe göre takvimde {target - cond_ref:+d} gün kaydırdım; sonuç {res}"
    return f"{tr_capitalize(res)}.", expl


def weekday_from(raw: str) -> tuple[str, str] | None:
    """'Çarşambadan (üç gün) sonra hangi gün gelir?' — adlandırılmış günden ileri/geri sayım."""
    words = _words(raw)
    if "hangi" not in words or not ({"sonra", "önce"} & set(words)):
        return None
    for i, w in enumerate(words):
        for name in sorted(WEEKDAYS, key=len, reverse=True):
            if w.startswith(name) and w[len(name):] in {"dan", "den", "tan", "ten"}:
                nums = _numbers(words[i + 1:])
                count = nums[0] if len(nums) == 1 else 1
                delta = count if "sonra" in words else -count
                res = WEEKDAYS[(WEEKDAYS.index(name) + delta) % 7]
                return (f"{tr_capitalize(res)}.",
                        f"{name} gününden {delta:+d} gün kaydırdım; sonuç {res}")
    return None


def word_problem(raw: str) -> tuple[str, str] | None:
    """İki sayılı basit kelime problemi: çıkarma, toplama ve 'bir X'in N Y varsa, M X'in kaç Y' çarpması."""
    s_clean = re.sub(r"^[\s>]*\d+\s*[.)]\s+", "", (raw or "").strip())
    words = _words(s_clean)
    if "kaç" not in words and not any(w.startswith("kaç") for w in words):
        return None
    wset = set(words)
    # 'Bir kedinin dört ayağı varsa, iki kedinin toplam kaç ayağı vardır?' → her birinin N'i, M kez.
    if "varsa" in words and "bir" in words:
        cut = words.index("varsa")
        per_unit, asked = _numbers(words[:cut]), _numbers(words[cut:])
        if len(per_unit) == 1 and len(asked) == 1:
            n, m = per_unit[0], asked[0]
            return (f"{m} × {n} = {m * n}; toplamda {m * n} olur.",
                    f"Her birinde {n} tane var; {m} tane için {m} × {n} = {m * n} buldum")
    # Çok cümleli veya birim/oran/hayvan ayağı içeren soruları story_math çözer; burada kör toplam yapılmaz.
    sents = [p.strip() for p in re.split(r"(?<=[.!?])\s+", s_clean) if p.strip()]
    if len(sents) > 3:
        return None
    if wset & {"her", "bacak", "bacağı", "ayak", "ayağı", "tekerlek", "tekerleği", "kanat", "kanadı",
               "bazı", "birkaç", "ondan", "fazla", "eksik", "geri"}:
        return None
    nums = _numbers(words)
    if "toplam" in wset and not removal_any(words) and len(nums) >= 2 and len(sents) <= 2:
        # Tüm sayılar ilk cümlede yan yana listelenmiş ve soru sadece "Toplam kaç ...?" olmalı
        sent_num_counts = [len(_numbers(_words(st))) for st in sents]
        q_words = _words(sents[-1])
        if sent_num_counts[0] == len(nums) and len(nums) >= 3 and "ve" not in q_words:
            total = sum(nums)
            parts = " + ".join(str(n) for n in nums)
            return f"{parts} = {total}; toplamda {total} olur.", f"Kelime problemi: {parts} = {total} buldum"
    if wset & {"kutu", "kutuda", "paket", "pakette", "torba", "torbada", "sepet", "sepette",
               "grup", "grupta", "sıra", "sırada", "rafa", "rafta"}:
        return None
    if len(nums) != 2:
        return None
    # "toplam 14 kitap varsa daha kaç..." gibi ters/fark sorularında toplama yapma
    if len(sents) >= 2 and ("toplam" in _words(sents[0]) or "toplamda" in _words(sents[0]) or "varsa" in _words(sents[-1])):
        return None
    a, b = nums
    asks_left = bool(wset & {"kalır", "kalan", "geriye", "kaldı", "kalacak"})
    asks_total = bool(wset & {"olur", "eder", "toplam", "toplamda", "toplamı", "toplamını", "toplamının", "toplamları"})
    asks_diff = bool(wset & {"farkı", "farkını", "farkının"})
    removal = any(_is_removal(w) for w in words) or bool(
        re.search(r"(?:arkadaşına|kardeşine|annesine|babasına)\s+ver", s_clean.lower()))
    if (asks_left or asks_diff) and (removal or asks_diff) and a >= b:
        return f"{a} - {b} = {a - b}; geriye {a - b} kalır." if asks_left else f"{a} - {b} = {a - b}; fark {a - b} olur.", f"Kelime problemi: {a} - {b} = {a - b} buldum"
    if asks_total and not removal and not asks_left and "hâlâ" not in wset and (wset & ADD_WORDS or wset & {"toplam", "toplamda", "toplamı", "toplamını"}):
        if len(sents) <= 2 or "ona" in wset:
            return f"{a} + {b} = {a + b}; toplamda {a + b} olur.", f"Kelime problemi: {a} + {b} = {a + b} buldum"
    return None


def which_number(raw: str) -> tuple[str, str] | None:
    """'Hangisi daha büyük: 17 mi, 9 mu?' → iki sayıdan büyüğü (ya da küçüğü)."""
    words = _words(raw)
    if "hangisi" not in words:
        return None
    rel = next((w for w in ("büyük", "küçük") if w in words), None)
    nums = _numbers(words)
    if rel is None or len(nums) != 2 or nums[0] == nums[1]:
        return None
    win = max(nums) if rel == "büyük" else min(nums)
    return f"{win} daha {rel}.", f"{nums[0]} ile {nums[1]} sayılarını karşılaştırdım; {win} daha {rel}"


def numeric_compare(raw: str) -> tuple[str, str] | None:
    """'5 sayısı 3 sayısından büyük müdür?' biçimli sayı karşılaştırması."""
    words = _words(raw)
    if "sayısından" not in words and "sayısı" not in words:
        return None
    nums = _numbers(words)
    rel = next((w for w in ("büyük", "küçük", "eşit") if w in words), None)
    if len(nums) != 2 or rel is None or not (set(words) & MI_SET):
        return None
    a, b = nums
    truth = {"büyük": a > b, "küçük": a < b, "eşit": a == b}[rel]
    if truth:
        word = {"büyük": "büyüktür", "küçük": "küçüktür", "eşit": "eşittir"}[rel]
        return f"Evet, {a}, {b}'den {word}.", f"{a} ile {b} sayılarını karşılaştırdım"
    return f"Hayır, {a}, {b}'den {rel} değildir.", f"{a} ile {b} sayılarını karşılaştırdım"


def empty_count(raw: str) -> tuple[str, str] | None:
    """'Bir kutu boşsa içinde kaç nesne vardır?' → boş kapta nesne sayısı sıfırdır."""
    words = _words(raw)
    if "kaç" not in words and not any(w.startswith("kaç") for w in words):
        return None
    cond = any(
        w.startswith("boş") and (w[3:] in COND_SUFFIX
                                 or (w == "boş" and i + 1 < len(words) and words[i + 1] in COND_WORDS))
        for i, w in enumerate(words)
    )
    if not cond:
        return None
    return "Boşsa içinde hiç nesne yoktur; sayı 0.", "Boş kapta nesne bulunmaz"


def _prop_form(word: str) -> str | None:
    """'kısadır' → 'kısa': sıfat kökü ve (varsa) yüklem eki."""
    for key in ANTONYMS:
        if word.startswith(key) and word[len(key):] in {"", "dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür"}:
            return key
    return None


def _comparisons(raw: str) -> list[tuple[str, str, str]]:
    """'A, B'ten PROP(dır)' kalıplı karşılaştırmalar: (A, B, PROP)."""
    found = []
    for sent in re.split(r"[.!?]", tr_lower(raw)):
        m = re.search(r"([\wçğıöşü]+),?\s+([\wçğıöşü]+)['’](?:ten|tan|den|dan)\s+([\wçğıöşü]+?)(?:dur|dır|dir|dür|tur|tür)\b",
                      sent.strip())
        if m and m.group(3) in ANTONYMS:
            found.append((m.group(1), m.group(2), m.group(3)))
    return found


def which_shorter(raw: str, memory: dict) -> tuple[str, str] | None:
    """'Ahmet, Mehmet'ten uzundur. Hangisi daha kısadır?' → karşılaştırılan iki kişiden biri.

    Karşılaştırma aynı girdide yoksa önceki turdan (memory) alınır.
    """
    if "hangisi" not in _words(raw):
        return None
    # Sorulan özellik yalnız soru cümlesinden alınır ('uzundur' gibi karşılaştırma sözcüklerinden değil).
    question = next(s for s in re.split(r"[.!?]", tr_lower(raw)) if "hangisi" in s)
    asked = next((p for p in (_prop_form(w) for w in _words(question)) if p), None)
    if asked is None or "comparison" not in memory:
        return None
    a, b, prop = memory["comparison"]
    if asked == prop:
        winner = a
    elif asked == ANTONYMS[prop]:
        winner = b
    else:
        return None
    return (f"{tr_capitalize(winner)} daha {asked}.",
            f"Karşılaştırma: {a}, {b}'ten {prop}; {asked} olan {winner}")


def attribute_echo(raw: str) -> tuple[str, str] | None:
    """'Kırmızı bir elma kırmızı mıdır?' — özellik sorunun öznesinde zaten verilmiş."""
    words = _words(raw)
    if len(words) != 5 or words[1] != "bir" or words[3] != words[0] or words[4] not in MI_SET:
        return None
    noun, adj = words[2], words[0]
    return f"Evet, verilen bilgiye göre {noun} {adj}.", f"Sorunun kendisi {noun} için {adj} özelliğini verdi"


SOLVERS = (weekday, weekday_from, empty_count, which_number, numeric_compare, which_shorter, attribute_echo,
           word_problem)


def solve(raw: str, memory: dict) -> tuple[str, str] | None:
    """İlk eşleşen çözücünün (yanıt, gerekçe) çifti; yoksa None.

    ``memory`` tur arası bağlamdır (son karşılaştırma); çözücüler ona yazabilir.
    """
    for comp in _comparisons(raw):
        memory["comparison"] = comp
    for fn in SOLVERS:
        res = fn(raw, memory) if fn is which_shorter else fn(raw)
        if res is not None:
            return res
    return None
