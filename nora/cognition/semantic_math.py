# -*- coding: utf-8 -*-
"""Anlamsal kelime problemi çözücüsü: miktarları kişi ve nesneye göre izler.

Her sayı bir (kişi, nesne) çiftine bağlanır; cümlenin fiili eklenme, eksilme ya da devir
olduğunu söyler; soru hangi kişiyi ve nesneyi sorduğuna göre yanıtlanır. Karşılaştırma
('A, B'den kaç fazla'), toplam ('ikisinin toplamı') ve kişi-başına sorular da burada.

Tanınmayan bir fiil, belirsiz bir cümle ya da ölçülemeyen bir soru varsa None döner.
Bu bir dil modeli değildir: cümle yapısı ve fiil sınıfları açıkça tanımlıdır.
"""

from __future__ import annotations

import re

from nora.language.lang import tr_lower

GAIN = ("aldı", "aldım", "aldık", "alıyor", "buldu", "buldum", "buluyor", "topladı", "topladım",
        "topluyor", "kazandı", "kazandım", "kazanıyor", "getirdi", "getirdim", "getiriyor",
        "ekledi", "eklendi", "ekliyor", "koydu", "koyuyor", "geldi", "geliyor", "katıldı",
        "katılıyor", "yaptı", "yapıyor", "üretti", "üretiyor", "pişirdi", "dikti", "ekti", "bindi")
LOSS = ("yedi", "yiyor", "yedim", "kaybetti", "kaybettim", "kaybediyor", "attı", "atıyor",
        "sattı", "sattım", "satıyor", "harcadı", "harcadım", "harcıyor", "kullandı", "kullanıyor",
        "kırdı", "kırıyor", "sildi", "çıkardı", "çıkarıyor", "gitti", "gidiyor", "uçtu",
        "kayboldu", "bozuldu", "kırıldı", "yenildi", "verdi", "verdim", "verildi", "veriyor",
        "çıkarıldı", "satıldı", "harcandı", "okudu", "okuyor", "indi", "iniyor", "dağıttı")
FILLER = {"tane", "tanesini", "tanesi", "tanesinden", "adet", "kadar", "tanesine", "tanesinin"}
NEUTRAL_STATE = ("vardı", "var", "varmış", "tane", "bulunuyor", "bulunuyordu", "oturuyordu", "biniyordu")
STOP_CAPS = {"kaç", "her", "bir", "sonra", "şimdi", "toplam", "hepsi", "ayrıca", "bugün", "dün",
             "daha", "onlara", "bunlardan", "bunların", "pazartesi", "salı", "çarşamba", "perşembe",
             "cuma", "cumartesi", "pazar", "sabah", "öğleden", "akşam", "gece", "ilk", "ikinci",
             "üçüncü", "ertesi", "başta", "geriye", "ancak", "ama", "fakat", "çünkü", "eğer",
             "ise", "veya", "ya", "hem", "ne", "bu", "şu", "o", "önce", "ardından", "sonrasında",
             "sonraki", "içeride", "dışarıda"}
QUANT = re.compile(r"(\d+)\s+([\wçğıöşü']+)")
POSS = re.compile(r"([A-Za-zÇĞİÖŞÜçğıöşü]+)'(?:nin|nın|nun|nün|in|ın|un|ün|de|da|ye|ya|a|e)\b")
CAP = re.compile(r"\b([A-ZÇĞİÖŞÜ][a-zçğıöşü]+)\b")
DATIVE = re.compile(r"([A-ZÇĞİÖŞÜ][a-zçğıöşü]+)'(?:ye|ya|e|a)\b")


def _stem(word: str) -> str:
    """Sondaki ünlüler ve iyelik/durum ekleri ('elması', 'elmasını', 'elmayı', 'sayfalık', 'tl'si') atılır, ilk dört harf."""
    w = tr_lower(word).split("'")[0]
    w = re.sub(r"(?:larını|lerini|larına|lerine|lardan|lerden|larıyla|leriyle)$", "", w)
    w = re.sub(r"(?:sını|sini|sunu|sünü|sına|sine|sunda|sünde|sından|sinden|sıyla|siyle)$", "", w)
    w = re.sub(r"(?:ını|ini|unu|ünü|ına|ine|unda|ünde|ından|inden)$", "", w)
    w = re.sub(r"(?:lık|lik|luk|lük|sız|siz|suz|süz|ları|leri|lar|ler|sı|si|su|sü|nın|nin|nun|nün|dan|den|tan|ten)$", "", w)
    base = re.sub(r"[aeıioöuü]+$", "", w)
    return base[:4] or w[:4]


def _sentences(raw: str) -> list[str]:
    s = re.sub(r"^[\s>]*\d+\s*[.)]\s+", "", (raw or "").strip())
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", s) if p.strip()]


def _event(sentence: str) -> str:
    words = re.findall(r"[\wçğıöşü]+", tr_lower(sentence))
    # Eğer başka bir özne ("babası", "annesi", "arkadaşı") ana özneye "veriyor/verdi" ise bu kazançtır.
    if any(w in ("verdi", "veriyor") for w in words) and any(
            w in ("babası", "annesi", "arkadaşı", "kardeşi", "dedesi", "ona") for w in words):
        if not any(w in ("arkadaşına", "kardeşine", "annesine", "babasına") for w in words):
            return "gain"
    if any(w in LOSS for w in words):
        return "loss"
    if any(w in GAIN for w in words):
        return "gain"
    if any(w in NEUTRAL_STATE for w in words):
        return "state"
    return "none"


def _names_with_pos(sentence: str) -> list[tuple[int, str]]:
    """(konum, ad) çiftleri: iyelik eki taşıyan ya da büyük harfle başlayan ilk sözcükler."""
    found = [(m.start(), tr_lower(m.group(1))) for m in POSS.finditer(sentence)
             if not m.group(1).isdigit() and tr_lower(m.group(1)) not in STOP_CAPS]
    for m in CAP.finditer(sentence):
        word = m.group(1)
        if word.lower() in STOP_CAPS or tr_lower(word) in STOP_CAPS:
            continue
        if tr_lower(word) not in [w for _, w in found]:
            found.append((m.start(), tr_lower(word)))
    return sorted(found)


def _nearest_name(names: list[tuple[int, str]], pos: int) -> str | None:
    before = [n for p, n in names if p <= pos]
    return before[-1] if before else None


def solve_semantic(raw: str) -> tuple[int, str] | None:
    sentences = _sentences(raw)
    if len(sentences) < 2:
        return None
    low_all = tr_lower(" ".join(sentences))
    # Ters problemler ('bazı/birkaç/biraz ... başta kaç'), karşılaştırma/oran ('fazla/eksik/az'), grup/bölme burada çözülmez.
    if re.search(r"\bbazı\b|\bbirkaç\b|\bbiraz\b|\bbaşta\s+kaç\b|\beşit\b|\bpaylaştır|\bgrup\b|"
                 r"\bher\b|\d+\s+kutu\b|\d+\s+paket\b|\d+\s+torba\b|\bfazla\b|\bfazlası\b|\bfazlaydı\b|\beksik\b|\beksiği\b|\bkalması\s+için\b", low_all):
        return None
    question = sentences[-1]
    low_q = tr_lower(question)
    if not re.search(r"\bkaç\b|ne kadar", low_q):
        return None
    # Alt-küme soruları ("yeşil ve sarı şeftali", "öğleden sonra ve akşam", "matematik ve biyoloji"):
    if " ve " in low_q or "kadar kaç" in low_q:
        return None

    # Soru: karşılaştırma ya da tek kişi/toplam.
    compare = re.match(r"\s*([\wçğıöşü]+),\s*([\wçğıöşü]+)'(?:den|dan|ten|tan)\s+(?:ne kadar\s+|kaç\s+)?"
                       r"(?:[\wçğıöşü]+\s+)?(?:daha\s+)?(fazla|çok|eksik|az)", low_q)
    qnoun_m = re.search(r"kaç\s+([\wçğıöşü']+)|ne kadar\s+([\wçğıöşü']+)", low_q)
    if qnoun_m is None:
        return None
    qnoun_raw = next(g for g in qnoun_m.groups() if g)
    if qnoun_raw in {"fazla", "az", "eksik", "çok", "tane", "kadar"}:
        return None
    qnoun = _stem(qnoun_raw)

    body = sentences[:-1]
    # Soru cümlesi de koşul/olay taşıyorsa ("10 kek sattıysa hâlâ kaç keki olur?") onu da gövdeye kat
    if re.search(r"\d+", low_q) and _event(question) in ("gain", "loss"):
        body = body + [question]
    amount: dict[str, dict[str, int]] = {}
    seen_nouns: set[str] = set()
    used_nums: list[int] = []
    last_subject: str | None = None
    last_noun: str | None = None
    steps_history: list[str] = []

    def add(person: str, noun: str, value: int) -> None:
        amount.setdefault(person, {}).setdefault(noun, 0)
        amount[person][noun] += value

    # Çok parçalı cümleleri (virgül, 'ardından/sonra' veya iki ayrı fiilli 've') alt-parçalara ayır.
    clauses: list[str] = []
    for s in body:
        parts = [p.strip() for p in re.split(r",\s*(?:ardından|sonra)\s*|\b(?:ardından|daha sonra)\b", s) if p.strip()]
        for p in parts:
            # Eğer cümlede 've' veya ',' bağlacı iki ayrı olayı bağlıyorsa ("13 bilye buldu, 5 bilye kaybetti")
            sub_parts = [sp.strip() for sp in re.split(r"\s+ve\s+|,\s+", p) if sp.strip()]
            if len(sub_parts) == 2 and _event(sub_parts[0]) != "none" and _event(sub_parts[1]) != "none":
                clauses.extend(sub_parts)
            else:
                clauses.append(p)

    for sentence in clauses:
        low = tr_lower(sentence)
        # Zaman/gün belirteçlerindeki sayıları ('2 günde', '1. gün') miktar sanma
        if re.search(r"\b\d+\s*(?:gün|günde|saat|saatte|hafta|haftada|ay|yıl)\b", low):
            return None
        event = _event(sentence)
        names = _names_with_pos(sentence)
        quants = [(m.start(), int(m.group(1)), tr_lower(m.group(2))) for m in QUANT.finditer(low)]
        if not quants:
            if event == "none" and not names:
                continue                       # bağlam cümlesi
            continue
        if event == "none":
            return None                        # sayı var ama olay bilinmiyor: tahmin yürütmem
        dative = DATIVE.search(sentence)
        receiver = tr_lower(dative.group(1)) if dative else None
        # "babası 12 TL veriyor" veya "Ayşe ona 3 elma veriyor" gibi ifadelerde asıl alıcı önceki ana öznedir
        giver_rel = bool(re.search(r"\b(?:babası|annesi|dedesi|abisi|ablası|ona)\b", low)) and event == "gain"
        for pos, value, noun in quants:
            if noun in {"ilk", "ikinci", "üçüncü", "gün", "saat", "kez", "kere", "defa"}:
                continue
            if noun in FILLER or noun in {"yeni", "eski", "farklı", "büyük", "küçük"}:
                # '170 yeni kek' -> sonraki kelime nesne mi?
                after = low[pos:].split()
                if len(after) >= 3 and after[1] in {"yeni", "eski", "farklı", "büyük", "küçük"} and after[2] not in {"olanı", "olanlardan"}:
                    noun = after[2]
                else:
                    noun = last_noun or noun
            # Alıcı (dative) ad, özne olamaz: özne cümledeki diğer ad ya da önceki özne.
            person = None
            if not giver_rel:
                for p_, n_ in names:
                    if p_ <= pos and n_ != receiver:
                        person = n_
            person = person or last_subject
            if person is None:
                return None
            last_subject, last_noun = person, noun          # ham (çekimli) biçim saklanır
            stemmed_noun = _stem(noun)
            seen_nouns.add(stemmed_noun)
            if stemmed_noun == qnoun:
                used_nums.append(value)
            if event == "state":
                add(person, stemmed_noun, value)
                steps_history.append(f"{person.capitalize()}'in başlangıçta {value} {noun} miktarı vardır.")
            elif event == "gain":
                prev = amount.get(person, {}).get(stemmed_noun, 0)
                add(person, stemmed_noun, value)
                now = amount[person][stemmed_noun]
                steps_history.append(f"{value} {noun} eklenince/alınınca: {prev} + {value} = {now} {noun} olur.")
            elif event == "loss":
                prev = amount.get(person, {}).get(stemmed_noun, 0)
                add(person, stemmed_noun, -value)
                now = amount[person][stemmed_noun]
                steps_history.append(f"{value} {noun} eksilince/verilince: {prev} - {value} = {now} {noun} kalır.")
                if receiver and receiver != person:
                    add(receiver, stemmed_noun, value)
    if not amount:
        return None
    if qnoun not in seen_nouns:
        return None
    # Tek nesneli bir problemde problemdeki sayıların tamamı kullanılmadıysa (örn. bir sayı atlandıysa) sus:
    all_digits = [int(x) for x in re.findall(r"\d+", low_all)]
    if len(seen_nouns) == 1 and len(used_nums) < len(all_digits):
        return None

    if compare:
        a, b, polarity = tr_lower(compare.group(1)), tr_lower(compare.group(2)), compare.group(3)
        va = amount.get(a, {}).get(qnoun)
        vb = amount.get(b, {}).get(qnoun)
        if va is None or vb is None:
            return None
        if polarity in {"fazla", "çok"} and va <= vb:
            return None
        if polarity in {"eksik", "az"} and va >= vb:
            return None
        diff = abs(va - vb)
        return diff, f"{a} {va}, {b} {vb}; fark {diff}"

    people = [p for p in amount if qnoun in amount[p]]
    asked = None
    asked_m = re.search(r"([\wçğıöşü]+)'(?:nin|nın|nun|nün|in|ın|un|ün)\s+kaç", low_q)
    if asked_m:
        asked = tr_lower(asked_m.group(1))
    if asked and asked in amount and qnoun in amount[asked]:
        value = amount[asked][qnoun]
        if value < 0:
            return None
        if len(steps_history) >= 2:
            return value, "; ".join(steps_history)
        steps = ", ".join(f"{p} {amount[p][qnoun]}" for p in people)
        return value, f"{asked} için {qnoun}: {value} ({steps})"
    if re.search(r"toplam|birlikte|ikisi|hepsi", low_q) and people:
        value = sum(amount[p][qnoun] for p in people)
        if value < 0:
            return None
        return value, " + ".join(f"{p} {amount[p][qnoun]}" for p in people) + f" = {value}"
    if len(people) == 1:
        value = amount[people[0]][qnoun]
        if value < 0:
            return None
        if len(steps_history) >= 2:
            return value, "; ".join(steps_history)
        return value, f"{people[0]} için {qnoun}: {value}"
    return None


def solve_inverse(raw: str) -> tuple[int, str] | None:
    """Tek bilinmeyenli ters problem: 'bazı X vardı / verdi / aldı ... şimdi N var' → bilinmeyeni çözer.

    start + Σ(işaretli olaylar) = son durum. Cümle başına en çok bir bilinmeyen (bazı/birkaç) ve
    tam bir son durum cümlesi (şimdi/geriye/kaldı + sayı) gerekir; aksi halde None.
    """
    sentences = _sentences(raw)
    if len(sentences) < 3:
        return None
    question = tr_lower(sentences[-1])
    if not re.search(r"\bkaç\b", question):
        return None
    start = None            # (değer | None)
    events: list[tuple[int, int | None]] = []
    final = None
    unknown_slots = 0
    raw_sents = sentences[:-1]
    proc_sents = []
    for s in raw_sents:
        parts = [p.strip() for p in re.split(r"(?:(?:dıktan|dikten|duktan|dükten|tıktan|tikten|tuktan|tükten)\s+sonra)", s, flags=re.I) if p.strip()]
        if len(parts) == 2:
            proc_sents.append(parts[0] + " harcadı")
            proc_sents.append("geriye " + parts[1])
        else:
            proc_sents.append(s)

    for sentence in proc_sents:
        low = tr_lower(sentence)
        nums = [int(x) for x in re.findall(r"\d+", low)]
        unknown = bool(re.search(r"\bbazı\b|\bbirkaç\b|\bbiraz\b|\bbir\s+miktar\b", low))
        ev = _event(sentence)
        is_final = bool(re.search(r"\bşimdi\b|\bgeriye\b|\bkaldı\b", low)) and len(nums) == 1 and not unknown
        if is_final:
            final = nums[0]
            continue
        if ev == "state":
            if start is not None:
                return None
            if unknown and not nums:
                start = None; unknown_slots += 1; start = -1  # işaret: bilinmeyen başlangıç
            elif len(nums) == 1:
                start = nums[0]
            else:
                return None
        elif ev in ("gain", "loss"):
            sign = 1 if ev == "gain" else -1
            if unknown and not nums:
                events.append((sign, None)); unknown_slots += 1
            elif len(nums) == 1:
                events.append((sign, nums[0]))
            else:
                return None
        elif unknown or nums:
            return None
    if final is None or start is None or unknown_slots != 1:
        return None
    known = sum(sg * v for sg, v in events if v is not None)
    if start == -1:                                   # başlangıç bilinmiyor
        value = final - known
        if len(events) == 1 and events[0][0] == -1 and events[0][1] is not None:
            spent = events[0][1]
            steps = [
                f"Harcama yapıldıktan sonra geriye {final} kalmıştır.",
                f"Başlangıçtaki miktarı bulmak için harcanan {spent} ile kalan {final} toplanır: {final} + {spent} = {value}."
            ]
            return value, "; ".join(steps)
        why = f"son {final}, olaylar {known:+d} → başlangıç {value}"
    else:
        sg = next(sg for sg, v in events if v is None)
        value = (final - start - known) * sg
        why = f"{start} ile başladı, son {final}; eksik olay {value}"
    if value < 0:
        return None
    return value, why


_NAME_COUNT = re.compile(r"([\wçğıöşü]+)'(?:nin|nın|nun|nün|in|ın|un|ün)\s+(?:(?:ise|de|da)\s+)?(\d+)")


def solve_relation(raw: str) -> tuple[int, str] | None:
    """Üç genel ilişki: türetilmiş karşılaştırma, grup sayısı (N ÷ K), grup çarpımı (G × E)."""
    sentences = _sentences(raw)
    if len(sentences) < 2:
        return None
    body, question = sentences[:-1], tr_lower(sentences[-1])
    if not re.search(r"\bkaç\b", question):
        return None
    low_body = [tr_lower(s) for s in body]
    # 1) 'B'nin A'dan d fazlası/eksiği var' + 'A'nın x var' → B = x ± d
    for s in low_body:
        m = re.search(r"([\wçğıöşü]+)'(?:nin|nın|nun|nün|in|ın|un|ün)\s+(?:[\wçğıöşü]+\s+sayısı\s+)?([\wçğıöşü]+)'(?:den|dan|ten|tan|inkinden|unkundan|nkinden|nkindan)\s+(\d+)\s*(?:[\wçğıöşü]+)?\s+(fazlası|eksiği|fazla|eksik|fazladır|eksiktir)", s)
        if not m:
            continue
        target, other, d, pol = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        base = None
        rel_noun = None
        for t in low_body:
            if t is s:
                continue
            if other in t:
                for qm_num, qm_noun in QUANT.findall(t):
                    base = int(qm_num)
                    rel_noun = _stem(qm_noun)
                    break
            if base is None:
                c = _NAME_COUNT.search(t)
                if c and c.group(1) == other and len(QUANT.findall(t)) == 1:
                    base = int(c.group(2))
                    rel_noun = _stem(QUANT.findall(t)[0][1])
        if base is None:
            continue
        target_val = base + d if pol.startswith("fazla") else base - d
        if re.search(r"\b(?:ikisinin\s+toplam|toplam\s+kaç|birlikte\s+kaç)\b", question):
            tot = base + target_val
            steps_desc = [
                f"{other.capitalize()}'in {base} nesnesi var.",
                f"{target.capitalize()}'nin sayısı {other.capitalize()}'inkinden {d} {pol}: {base} {'+' if pol.startswith('fazla') else '-'} {d} = {target_val}.",
                f"İkisinin toplamı: {base} + {target_val} = {tot}."
            ]
            return tot, "; ".join(steps_desc)
        qm = re.match(r"\s*([\wçğıöşü]+)'(?:nin|nın|nun|nün|in|ın|un|ün)\s+kaç\s+([\wçğıöşü']+)", question)
        if qm and qm.group(1) == target and (rel_noun is None or _stem(qm.group(2)) == rel_noun):
            return target_val, f"{other} {base}, fark {d}"
    # 2) grup sayısı: 'N var. Her kutuya K koyuyor. Kaç kutu?'
    if len(body) == 2 and re.search(r"kaç\s+(?:kutu|grup|torba|sepet|tabak|sıra|kişi|saksı|paket)", question):
        each = [s for s in low_body if re.search(r"\bher\b", s) and len(re.findall(r"\d+", s)) == 1]
        tot = [s for s in low_body if s not in each and len(re.findall(r"\d+", s)) == 1]
        if len(each) == 1 and len(tot) == 1:
            n, k = int(re.findall(r"\d+", tot[0])[0]), int(re.findall(r"\d+", each[0])[0])
            if k and n % k == 0:
                return n // k, f"{n} ÷ {k} = {n // k}"
    # 3) grup çarpımı: 'G kutusu var. Her kutuda E var. Toplam kaç?'
    if len(body) == 2 and re.search(r"toplam", question) and not re.search(r"kaç\s+(?:kutu|grup|torba|sepet|tabak|sıra|kişi|saksı|paket)", question):
        each = [s for s in low_body if re.search(r"\bher\b", s) and len(re.findall(r"\d+", s)) == 1]
        grp = [s for s in low_body if s not in each and len(re.findall(r"\d+", s)) == 1 and "toplam" not in s]
        if len(each) == 1 and len(grp) == 1:
            g, e = int(re.findall(r"\d+", grp[0])[0]), int(re.findall(r"\d+", each[0])[0])
            return g * e, f"{g} × {e} = {g * e}"
    return None
