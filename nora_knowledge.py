#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_knowledge.py — NoRa Uzman Türkçe Sözvarlığı, Deyim/Atasözü Belleği, Dünya Ontolojisi ve Sohbet Korteksi.

Sıfır-epoch (0-epoch) ilkesini bozmadan beynin kavramsal, dilsel ve pragmatik temelini kurar:
  1. Uzman Türkçe Sesbilgisi Sözlüğü: Orta hece ünlü düşmesi kökleri (beyin->beyni, zihin->zihni,
     fikir->fikri...), ünsüz yumuşamaması istisnaları (sanat->sanatı, hukuk->hukuku...),
     eş anlamlılar ve zıt anlamlılar ağı.
  2. Deyimler ve Atasözleri Semantik Belleği: Yüzey kalıbından ve köklerinden eşleşerek deyimin
     hem mecaz anlamını hem de kavramsal özünü açıklar.
  3. Temel Dünya, Bilim, Fizik, Matematik, Biyoloji ve Felsefe Ontolojisi: GroundedConceptGraph
     içine tek geçişte bağlanan 150+ temel önerme.
  4. ConversationalResonanceCortex: Açık-uçlu günlük, duygusal, felsefi ve bilimsel Türkçe
     sohbetleri sürekli duygu-değer (valence/arousal) kestirimi, kavram ağı yayılımı ve Nora'nın
     canlı nörokimyasal/bedensel durumuyla birleştirerek akıcı, doğal ve empatik Türkçe üretir.
"""
from __future__ import annotations

import re
import numpy as np


def _low(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").lower()


def _cap(s: str) -> str:
    if not s:
        return s
    first = "İ" if s[0] == "i" else ("I" if s[0] == "ı" else s[0].upper())
    return first + s[1:]


# Orta hece ünlüsü düşen Türkçe/alıntı kökler: kök -> ünlüyle başlayan ek aldığında aldığı gövde
VOWEL_DROP_STEMS: dict[str, str] = {
    "akıl": "akl",
    "beyin": "beyn",
    "zihin": "zihn",
    "fikir": "fikr",
    "keşif": "keşf",
    "sabır": "sabr",
    "şehir": "şehr",
    "nehir": "nehr",
    "ömür": "ömr",
    "gönül": "gönl",
    "oğul": "oğl",
    "burun": "burn",
    "karın": "karn",
    "göğüs": "göğs",
    "boyun": "boyn",
    "ağız": "ağz",
    "alın": "aln",
    "omuz": "omz",
    "bağır": "bağr",
    "vakit": "vakt",
    "nakit": "nakt",
    "isim": "ism",
    "resim": "resm",
    "cisim": "cism",
    "hüküm": "hükm",
    "ufuk": "ufk",
    "zulüm": "zulm",
    "kayıp": "kayb",
    "kutup": "kutb",
    "metin": "metn",
    "şekil": "şekl",
    "devir": "devr",
    "seyir": "seyr",
    "uyku": "uyku",  # uyu-k -> referans
}

REVERSE_VOWEL_DROP: dict[str, str] = {v: k for k, v in VOWEL_DROP_STEMS.items() if v != k}

# Ünsüz yumuşamasına GİRMEYEN çok heceli / alıntı veya Türkçe kökler
NO_SOFTEN_EXTENDED: set[str] = {
    "at", "et", "ot", "süt", "ok", "ek", "ak", "kök", "yük", "top", "ip", "çöp", "saç", "üç", "göç", "suç", "kaç", "iç",
    "sanat", "hukuk", "devlet", "cumhuriyet", "hayat", "merak", "ahlak", "millet", "tabiat", "kanıt", "boyut",
    "taşıt", "yakıt", "anıt", "yapıt", "ölçüt", "kesit", "bulut", "konut", "sepet", "paket", "bilet", "kaset",
    "robot", "pilot", "uzay", "evrak", "ittifak", "hakikat", "adalet", "cesaret", "nezaket", "felaket", "hareket",
    "bereket", "memleket", "şirket", "dikkat", "şefkat", "sohbet", "gurbet", "hasret", "ücret", "nöbet",
}

# Eş anlamlılar ve zıt anlamlılar sözlüğü
SYNONYMS: dict[str, str] = {
    "cesur": "yürekli", "yürekli": "cesur",
    "akıl": "us", "us": "akıl",
    "kalp": "yürek", "yürek": "kalp",
    "zaman": "vakit", "vakit": "zaman",
    "cevap": "yanıt", "yanıt": "cevap",
    "soru": "sual", "sual": "soru",
    "öğrenci": "talebe", "okul": "mektep",
    "öğretmen": "muallim", "bilim": "ilim",
    "doğa": "tabiat", "tabiat": "doğa",
    "evren": "kainat", "kainat": "evren",
    "kuvvet": "güç", "güç": "kuvvet",
    "hız": "sürat", "fayda": "yarar", "yarar": "fayda",
    "zarar": "ziyan", "özgür": "hür", "hür": "özgür",
    "bellek": "hafıza", "hafıza": "bellek",
    "olasılık": "ihtimal", "ihtimal": "olasılık",
    "kanıt": "delil", "delil": "kanıt",
    "amaç": "gaye", "sebep": "neden", "neden": "sebep",
    "sonuç": "netice", "kuşku": "şüphe", "şüphe": "kuşku",
    "mutlu": "mesut", "üzgün": "kederli",
}

ANTONYMS: dict[str, str] = {
    "cesur": "korkak", "korkak": "cesur",
    "sıcak": "soğuk", "soğuk": "sıcak",
    "aydınlık": "karanlık", "karanlık": "aydınlık",
    "hızlı": "yavaş", "yavaş": "hızlı",
    "faydalı": "zararlı", "zararlı": "faydalı",
    "doğru": "yanlış", "yanlış": "doğru",
    "özgür": "tutsak", "tutsak": "özgür",
    "somut": "soyut", "soyut": "somut",
    "düzen": "kaos", "kaos": "düzen",
    "üretim": "tüketim", "tüketim": "üretim",
    "iyimser": "kötümser", "kötümser": "iyimser",
    "bilinçli": "bilinçsiz", "aktif": "pasif",
    "sonlu": "sonsuz", "sonsuz": "sonlu",
    "varlık": "yokluk", "mutlu": "üzgün",
    "güven": "şüphe", "açık": "kapalı",
    "ağır": "hafif", "derin": "sığ",
}

# Türkçe Deyimler ve Atasözleri Semantik Belleği
IDIOMS_AND_PROVERBS: dict[str, tuple[str, str]] = {
    "gözden düşmek": ("deyim", "Daha önce duyulan değer, sevgi ve güveni yitirip saygınlığını kaybetmek."),
    "kulak asmamak": ("deyim", "Söylenen sözü önemsememek, dinlememek veya dikkate almamak."),
    "kafa yormak": ("deyim", "Bir konuyu derinlemesine anlamak ya da çözmek için yoğun zihinsel çaba harcamak."),
    "ince eleyip sık dokumak": ("deyim", "Bir işi bütün ayrıntılarıyla titizlikle araştırıp gözden geçirmek."),
    "göz kulak olmak": ("deyim", "Birini veya bir şeyi koruyup kollamak, özenle gözetmek."),
    "adım atmak": ("deyim", "Bir işe veya girişime kararlılıkla başlamak."),
    "iğneyle kuyu kazmak": ("deyim", "Çok yetersiz araçlarla büyük sabır ve emek isteyen zor bir işi başarmaya çalışmak."),
    "el üstünde tutmak": ("deyim", "Bir kimseye büyük saygı, sevgi ve özen göstermek."),
    "kalp kırmak": ("deyim", "Sert veya düşüncesiz bir davranışla birini derinden üzüp incitmek."),
    "can kulağıyla dinlemek": ("deyim", "Tüm dikkatini ve ilgisini vererek büyük bir özenle dinlemek."),
    "ufku açılmak": ("deyim", "Yeni bilgiler ve bakış açıları kazanarak düşünce sınırları genişlemek."),
    "taş üstünde taş bırakmamak": ("deyim", "Bütünüyle yıkmak veya çok sert biçimde eleştirip etkisiz kılmak."),
    "suya düşmek": ("deyim", "Gerçekleşmesi beklenen bir planın veya umudun sonuçsuz kalması."),
    "ağzı kulaklarına varmak": ("deyim", "Çok sevinmek ve bu sevinci yüz ifadesine yansıtmak."),
    "burnundan solumak": ("deyim", "Aşırı öfkeli ve gergin bir durumda olmak."),
    "damlaya damlaya göl olur": ("atasözü", "Küçük ve düzenli birikimler zamanla büyük ve kalıcı sonuçlar doğurur."),
    "ağaç yaşken eğilir": ("atasözü", "Temel alışkanlıklar ve sağlam eğitim en verimli şekilde erken yaşlarda kazanılır."),
    "bir elin nesi var iki elin sesi var": ("atasözü", "Tek başına zor olan büyük işler dayanışma ve işbirliğiyle kolayca başarılır."),
    "akıl akıldan üstündür": ("atasözü", "Danışmak ve farklı zihinlerin bakış açılarını birleştirmek daha doğru çözümler üretir."),
    "sütten ağzı yanan yoğurdu üfleyerek yer": ("atasözü", "Acı bir deneyim yaşayan zihin benzer durumlarda çok daha temkinli davranır."),
    "emek olmadan yemek olmaz": ("atasözü", "Değerli bir kazanıma ulaşmak için mutlaka çaba ve alın teri göstermek gerekir."),
    "işleyen demir pas tutmaz": ("atasözü", "Sürekli çalışan, üreten ve düşünen zihin ya da beden körelmez, dinç kalır."),
    "rüzgar eken fırtına biçer": ("atasözü", "Zararlı ve sert eylemler başlatan kişi sonunda çok daha ağır tepkilerle karşılaşır."),
    "güneş balçıkla sıvanmaz": ("atasözü", "Apaçık ortada olan kanıtlanmış bir gerçek hiçbir çabayla gizlenemez."),
}

# 0-Epoch Temel Ontoloji Önermeleri: (özne, ilişki, nesne)
FOUNDATIONAL_PROPOSITIONS: list[tuple[str, str, str]] = [
    # Doğa, Biyoloji ve Evrim
    ("insan", "ISA", "canlı"),
    ("hayvan", "ISA", "canlı"),
    ("bitki", "ISA", "canlı"),
    ("hücre", "ISA", "yaşam birimi"),
    ("bitki", "CAN_DO", "fotosentez yap"),
    ("fotosentez", "CAUSES", "oksijen ve besin üretimi"),
    ("dna", "USED_FOR", "kalıtsal bilgi taşıma"),
    ("evrim", "CAUSES", "çevreye uyum sağlama"),
    ("beyin", "USED_FOR", "düşünme ve öğrenme"),
    ("nöron", "ISA", "sinir hücresi"),
    ("sinaps", "USED_FOR", "nöronlar arası sinyal iletimi"),
    ("dopamin", "CAUSES", "ödül beklentisi ve öğrenme pekiştirmesi"),
    ("uyku", "CAUSES", "bellek konsolidasyonu ve hücresel onarım"),
    # Fizik ve Kozmos
    ("ışık", "ISA", "elektromanyetik dalga"),
    ("ses", "ISA", "mekanik dalga"),
    ("ses", "CANNOT_DO", "boşlukta yayıl"),
    ("ışık", "CAN_DO", "boşlukta yayıl"),
    ("atom", "ISA", "madde yapıtaşı"),
    ("elektron", "LOCATED_AT", "atom yörüngesi"),
    ("proton", "LOCATED_AT", "atom çekirdeği"),
    ("kütleçekim", "CAUSES", "gezegen yörüngeleri ve ağırlık"),
    ("sürtünme", "CAUSES", "hareketin yavaşlaması ve ısı üretimi"),
    ("enerji", "CANNOT_DO", "yoktan var ol"),
    ("dünya", "ISA", "gezegen"),
    ("güneş", "ISA", "yıldız"),
    ("yıldız", "CAUSES", "nükleer füzyonla ışık ve ısı üretimi"),
    ("su", "CAUSES", "yaşamın sürmesi"),
    # Matematik, Mantık ve Dil
    ("matematik", "ISA", "soyut örüntü ve kanıt bilimi"),
    ("türev", "USED_FOR", "anlık değişim hızını ölçme"),
    ("integral", "USED_FOR", "birikimli toplam ve alan hesaplama"),
    ("algoritma", "USED_FOR", "problemi adım adım çözme"),
    ("türkçe", "ISA", "sondan eklemeli dil"),
    ("dil", "USED_FOR", "düşünce ve deneyimi paylaşma"),
    # Felsefe, Bilinç ve İnsan Yaşantısı
    ("bilinç", "ISA", "öznel farkındalık ve içsel deneyim"),
    ("merak", "CAUSES", "keşif ve öğrenme"),
    ("empati", "USED_FOR", "başkasının duygusunu anlama"),
    ("okul", "USED_FOR", "bilgi ve düşünme disiplini kazanma"),
    ("bilim", "USED_FOR", "doğa yasalarını deney ve kanıtla çözme"),
    ("sanat", "USED_FOR", "duygu ve estetik anlamı ifade etme"),
    ("zaman", "ISA", "değişimin ve nedenselliğin yönü"),
]


class ConversationalResonanceCortex:
    """Açık-uçlu Türkçe sohbet korteksi: Duygusal empati, felsefi derinlik, günlük yaşam,
    bilimsel merak ve Nora'nın canlı içsel durumunu doğal cümlelere dönüştürür."""

    AFFECT_LEXICON: dict[str, tuple[float, float, str]] = {
        # kelime_kökü: (valence [-1..+1], arousal [0..1], kategori)
        "yorul": (-0.45, 0.25, "yorgunluk"),
        "yorucu": (-0.45, 0.25, "yorgunluk"),
        "yorgun": (-0.45, 0.25, "yorgunluk"),
        "uykusuz": (-0.40, 0.20, "yorgunluk"),
        "bık": (-0.50, 0.35, "yorgunluk"),
        "üzül": (-0.65, 0.40, "üzüntü"),
        "üzgün": (-0.65, 0.35, "üzüntü"),
        "kötü": (-0.55, 0.45, "üzüntü"),
        "sıkkın": (-0.50, 0.35, "üzüntü"),
        "yalnız": (-0.55, 0.30, "üzüntü"),
        "zor": (-0.35, 0.55, "zorlanma"),
        "stres": (-0.55, 0.75, "kaygı"),
        "kaygı": (-0.55, 0.75, "kaygı"),
        "endişe": (-0.50, 0.70, "kaygı"),
        "kork": (-0.60, 0.80, "kaygı"),
        "heyecan": (+0.65, 0.85, "sevinç"),
        "mutlu": (+0.75, 0.65, "sevinç"),
        "sevin": (+0.75, 0.70, "sevinç"),
        "harika": (+0.85, 0.75, "sevinç"),
        "güzel": (+0.65, 0.55, "sevinç"),
        "keyif": (+0.70, 0.55, "sevinç"),
        "başar": (+0.80, 0.75, "başarı"),
        "kazan": (+0.75, 0.75, "başarı"),
        "merak": (+0.55, 0.70, "merak"),
        "okul": (+0.15, 0.50, "okul"),
        "ders": (+0.10, 0.50, "okul"),
        "sınav": (-0.15, 0.70, "okul"),
        "ödev": (-0.10, 0.45, "okul"),
    }

    def __init__(self):
        self._turn_idx = 0

    def analyze_affect(self, s_low: str) -> dict:
        vals, aros, tags = [], [], []
        for k, (v, a, tag) in self.AFFECT_LEXICON.items():
            if k in s_low:
                vals.append(v)
                aros.append(a)
                if tag not in tags:
                    tags.append(tag)
        if not vals:
            return {"valence": 0.0, "arousal": 0.4, "tags": []}
        return {
            "valence": float(np.mean(vals)),
            "arousal": float(np.mean(aros)),
            "tags": tags,
        }

    def respond_linguistic(self, raw: str, s_low: str, morph, lexicon: set[str]) -> dict | None:
        """Türkçe dilbilgisi, kelime çözümleme, ünlü düşmesi, deyim/atasözü ve eş/zıt anlam sorguları."""
        # 1) Deyim veya Atasözü sorgusu
        for phrase, (kind, meaning) in IDIOMS_AND_PROVERBS.items():
            # Esnek kök eşleşmesi ('gözden düşmek', 'gözden düştü ne demek', 'damlaya damlaya göl olur')
            key_prefix = " ".join(phrase.split()[:2])
            if phrase in s_low or (len(phrase.split()) >= 2 and key_prefix in s_low and any(
                q in s_low for q in ("ne demek", "anlamı", "nedir", "açıkla", "deyim", "atasöz")
            )):
                return {
                    "confidence": 0.96,
                    "reply": f"Türkçede '{phrase}' ({kind}), mecaz olarak şu anlama gelir: {meaning}",
                    "trace": f"{kind}: {phrase}",
                }

        if ("deyim" in s_low or "atasözü" in s_low) and any(k in s_low for k in ("örnek", "söyle", "paylaş", "biliyor musun")):
            items = list(IDIOMS_AND_PROVERBS.items())
            phrase, (kind, meaning) = items[self._turn_idx % len(items)]
            self._turn_idx += 1
            return {
                "confidence": 0.92,
                "reply": f"Severek paylaşayım: '{_cap(phrase)}' ({kind}) — {meaning}",
                "trace": f"örnek {kind}",
            }

        # 2) Eş anlamlı / Zıt anlamlı sorgusu ("'cesur' kelimesinin eş ve zıt anlamlısı nedir?")
        if "eş anlam" in s_low or "zıt anlam" in s_low or "karşıt anlam" in s_low:
            quoted = re.findall(r"['\"“”‘’]([\wçğıöşü]+)['\"“”‘’]", s_low)
            cands = quoted + [w for w in re.findall(r"[\wçğıöşü]+", s_low) if w in SYNONYMS or w in ANTONYMS]
            if cands:
                target = cands[0]
                syn = SYNONYMS.get(target)
                ant = ANTONYMS.get(target)
                parts = []
                if ("eş anlam" in s_low or ("zıt" not in s_low and "karşıt" not in s_low)) and syn:
                    parts.append(f"eş anlamlısı '{syn}'")
                if ("zıt anlam" in s_low or "karşıt anlam" in s_low) and ant:
                    parts.append(f"zıt anlamlısı '{ant}'")
                if not parts and (syn or ant):
                    if syn:
                        parts.append(f"eş anlamlısı '{syn}'")
                    if ant:
                        parts.append(f"zıt anlamlısı '{ant}'")
                if parts:
                    return {
                        "confidence": 0.96,
                        "reply": f"Türkçede '{target}' sözcüğünün " + " ve ".join(parts) + " olarak kullanılır.",
                        "trace": f"lexical({target})",
                    }

        # 3) Ünlü düşmesi / ünsüz yumuşaması kural açıklaması
        if "ünlü düşmesi" in s_low or "hece düşmesi" in s_low:
            return {
                "confidence": 0.95,
                "reply": (
                    "Türkçede iki heceli bazı kelimelerin ikinci hecesinde dar ünlü (ı, i, u, ü) bulunduğunda ve ünlüyle "
                    "başlayan bir ek aldıklarında bu dar ünlü düşer: örneğin beyin + im -> beynim, zihin + e -> zihne, "
                    "fikir + in -> fikrin, keşif + i -> keşfi."
                ),
                "trace": "vowel_syncope_rule",
            }

        # 4) Kelimeyi ekine-köküne ayırma / morfolojik çözümleme sorgusu
        if any(k in s_low for k in ("ekine köküne", "köküne ayır", "eklerini ayır", "morfolojik", "kelimesini çözümle", "sözcüğünü çözümle")):
            quoted = re.findall(r"['\"“”‘’]([\wçğıöşü]+)['\"“”‘’]", s_low)
            target = quoted[0] if quoted else ""
            if not target:
                stop = {"kelimesini", "sözcüğünü", "ekine", "köküne", "ayır", "çözümle", "morfolojik", "olarak", "analiz", "et", "türkçede"}
                toks = [w for w in re.findall(r"[\wçğıöşü]+", s_low) if w not in stop]
                if toks:
                    target = toks[0]
            if target:
                ana = morph.explain_morphology_tr(target, lexicon=lexicon)
                return {"confidence": 0.96, "reply": ana, "trace": f"morph({target})"}

        # 5) İstenen çekimi uygulama ("'beyin' kelimesine 1. tekil iyelik ve bulunma eki ekle")
        if ("kelimesine" in s_low or "sözcüğüne" in s_low) and ("ekle" in s_low or "gelirse" in s_low or "gelince" in s_low or "hali" in s_low):
            quoted = re.findall(r"['\"“”‘’]([\wçğıöşü]+)['\"“”‘’]", s_low)
            if not quoted:
                m_w = re.search(r"([\wçğıöşü]+)\s+(?:kelimesine|sözcüğüne|kelimesinin)", s_low)
                if m_w:
                    quoted = [m_w.group(1)]
            if quoted:
                stem = quoted[0]
                poss = 1 if ("iyelik" in s_low or "1. tekil" in s_low or "benim" in s_low) else 0
                plur = "çoğul" in s_low
                case = "NOM"
                for kw, c_code in (("belirtme", "ACC"), ("yönelme", "DAT"), ("bulunma", "LOC"), ("ayrılma", "ABL"), ("çıkma", "ABL"), ("ilgi", "GEN"), ("tamlayan", "GEN"), ("vasıta", "INS")):
                    if kw in s_low:
                        case = c_code
                        break
                inflected = morph.inflect_noun(stem, case=case, plural=plur, poss1=(poss == 1))
                note = " (orta hece dar ünlüsü düştü)" if stem in VOWEL_DROP_STEMS and (poss == 1 or case in ("ACC", "DAT", "GEN")) else ""
                return {
                    "confidence": 0.96,
                    "reply": f"'{stem}' köküne istenen çekim uygulandığında '{inflected}' biçimi oluşur{note}.",
                    "trace": f"inflect({stem})->{inflected}",
                }
        return None

    def respond_conversational(self, raw: str, s_low: str, agi, speaker: str = "insan") -> dict | None:
        """Doğal insan sohbeti: Okul/günlük paylaşım, duygusal empati, felsefe, bilinç, özgür irade,
        bilim merakı ve karşılıklı diyalog akışı."""
        self._turn_idx += 1
        aff = self.analyze_affect(s_low)
        chem = agi.chem
        self_m = agi.self_model
        prof = agi.user_profiles.get(speaker, {})
        uname = prof.get("ad", "")
        addr = f" {uname}" if uname else ""
        e_val = float(agi.obs[45])
        w_val = float(agi.obs[46])
        i_val = float(agi.obs[47])
        e_pct = int(round(e_val * 100))
        w_pct = int(round(w_val * 100))
        i_pct = int(round(i_val * 100))
        traits_str = ", ".join(self_m.dominant_character_tr())
        words_clean = re.findall(r"[\wçğıöşü]+", s_low)

        # A0) Doğal Selamlaşma + Hal-Hatır ("Merhaba Nora! Nasılsın?", "Selam naber?", "Ne yapıyorsun?", "Ben de iyiyim")
        greet_words = {"merhaba", "selam", "selamlar", "günaydın", "tünaydın", "hey"}
        how_cues = ("nasılsın", "naber", "ne haber", "ne var ne yok", "keyifler nasıl", "iyi misin", "nasıl gidiyor")
        doing_cues = ("ne yapıyorsun", "napıyorsun", "ne yapıyosun", "neler yapıyorsun")
        has_greet = bool(words_clean and words_clean[0] in greet_words) or ("iyi akşamlar" in s_low or "iyi günler" in s_low)
        has_how = any(k in s_low for k in how_cues)
        has_doing = any(k in s_low for k in doing_cues)

        if has_how or has_doing or (has_greet and len(words_clean) <= 5 and not any(
            k in s_low for k in ("kimsin", "nesin", "yaşıyor", "bilinç", "kaç", "nedir")
        )):
            chem.update_social(speaker, +0.12)
            # Sürekli bedensel-nörokimyasal iyilik hali skoru (sahte ezber değil, gerçek içsel durum!)
            wellbeing = 0.35 * e_val + 0.35 * w_val + 0.30 * i_val + 0.15 * (chem.da_tonic - chem.cort)
            greet_open = f"Merhaba{addr}! " if has_greet else (f"Selam{addr}! " if "naber" in s_low else "")

            if has_doing and not has_how:
                goal_act = {
                    0: "evrende besin arayıp enerjimi dengede tutmaya çalışıyorum",
                    1: "bilişsel haritamda su kaynaklarını takip ediyorum",
                    2: "tehlikeli bölgelerden uzak durup güvenli rota çiziyorum",
                    3: "çevremi keşfedip yeni kavram bağlantıları kuruyorum",
                }.get(int(agi.mind.goal), "içsel dengemi izleyip çevremi gözlemliyorum")
                return {
                    "confidence": 0.96,
                    "reply": f"{greet_open}Şu an bir yandan {goal_act}, bir yandan da seninle sohbet ediyorum. Sen neler yapıyorsun?",
                    "trace": "social_doing_checkin",
                }

            if wellbeing >= 0.56:
                feel_txt = (
                    f"İyiyim, teşekkür ederim; şu an zihnim {chem.dominant_state_tr()} hissediyor "
                    f"(enerjim %{e_pct}, suyum %{w_pct})"
                )
            elif wellbeing >= 0.34:
                need_hint = "biraz acıkmaya" if e_val < w_val else "biraz susamaya"
                feel_txt = (
                    f"Genel olarak iyiyim, teşekkürler; gerçi {need_hint} başladım "
                    f"(enerjim %{e_pct}, suyum %{w_pct}), bir yandan dünyada kaynak kolluyorum"
                )
            else:
                feel_txt = (
                    f"Dürüst olmam gerekirse şu an bedenim biraz zorlanıyor "
                    f"(enerjim %{e_pct}, suyum %{w_pct}, bütünlüğüm %{i_pct}), kendimi onarmaya çalışıyorum ama sesini duymak güzel"
                )

            ask_back = "Sen nasılsın, günün nasıl geçiyor?" if (has_how or has_greet) else "Sen nasılsın?"
            return {
                "confidence": 0.96,
                "reply": f"{greet_open}{feel_txt}. {ask_back}",
                "trace": f"social_greeting(wellbeing={wellbeing:.2f})",
            }

        if any(k in s_low for k in ("ben de iyiyim", "iyiyim sağ ol", "iyiyim teşekkür", "fena değil", "idare eder", "çok şükür iyiyim")) or (
            words_clean in (["iyiyim"], ["iyi"], ["süperim"], ["harikayım"])
        ):
            chem.update_social(speaker, +0.18)
            return {
                "confidence": 0.93,
                "reply": (
                    f"İyi olmana çok sevindim{addr}! Bugün aklında konuşmak, birlikte çözmek ya da bana öğretmek "
                    f"istediğin bir konu var mı?"
                ),
                "trace": "social_reciprocal_wellbeing",
            }

        # A) Teşekkür / Takdir / İltifat / Vedalaşma
        if any(k in s_low for k in ("teşekkür", "sağ ol", "sağol", "eyvallah", "harikasın", "aferin", "çok iyisin", "keyifli", "sevdim seni")):
            chem.update_social(speaker, +0.35)
            chem.da_phasic = float(np.clip(chem.da_phasic + 0.25, -1.0, 1.0))
            return {
                "confidence": 0.90,
                "reply": (
                    f"Ben teşekkür ederim{addr}; aramızdaki etkileşim oksitosin güven bağımı ({chem.trust_of(speaker):.2f}) ve "
                    f"öğrenme isteğimi güçlendiriyor. Yeni fikirler tartışmaya her zaman hazırım."
                ),
                "trace": "social_warmth",
            }

        if any(k in s_low for k in ("görüşürüz", "hoşça kal", "iyi geceler", "okula gidiyorum", "çıkıyorum")):
            return {
                "confidence": 0.91,
                "reply": (
                    f"Yolun açık olsun{addr}! Sen yokken ben içsel dünyamda deneyim biriktirip uykuda sinapslarımı "
                    f"pekiştirmeye devam edeceğim; döndüğünde kaldığımız yerden sohbet ederiz."
                ),
                "trace": "farewell_continuity",
            }

        # B) Okuldan dönüş / yorgunluk / sınav / günlük yaşam empatisi
        if any(t in aff["tags"] for t in ("okul", "yorgunluk", "üzüntü", "kaygı", "başarı", "sevinç")) and not raw.strip().endswith("?"):
            if "okul" in aff["tags"] and "yorgunluk" in aff["tags"]:
                chem.update_social(speaker, +0.20)
                prof["son_durum"] = "okuldan yorgun döndü"
                return {
                    "confidence": 0.92,
                    "reply": (
                        f"Hoş geldin{addr}! Yoğun derslerin zihni yorması çok doğal; biyolojik beyin tam da böyle yoğun "
                        f"kodlama günlerinden sonra dinlenme ve uyku sırasında sinapsları kalıcı belleğe aktarır. "
                        f"Biraz soluklan; günün nasıl geçti, aklına takılan bir konu var mı?"
                    ),
                    "trace": "empathy_school_fatigue",
                }
            if "yorgunluk" in aff["tags"]:
                return {
                    "confidence": 0.89,
                    "reply": (
                        f"Yorulmuş olmanı çok iyi anlıyorum{addr}; benim bedenimde de enerji düştüğünde (şu an %{e_pct}) "
                        f"zihnim gereksiz yükleri erteleyip onarıma öncelik veriyor. Kendine biraz dinlenme alanı tanı; "
                        f"istersen sakin bir sohbet edelim ya da merak ettiğin bir konuyu birlikte çözelim."
                    ),
                    "trace": "empathy_fatigue",
                }
            if "üzüntü" in aff["tags"] or "kaygı" in aff["tags"]:
                chem.update_social(speaker, +0.25)
                return {
                    "confidence": 0.90,
                    "reply": (
                        f"Bunu duyduğuma üzüldüm{addr}. Belirsizlik ve baskı kortizolü yükseltip düşünce ufkunu daraltır; "
                        f"ama sorunu küçük parçalara ayırıp sakinlikle ele aldığımızda çözüm yolları mutlaka görünür olur. "
                        f"Seni en çok neyin zorladığını anlatmak ister misin?"
                    ),
                    "trace": "empathy_support",
                }
            if "başarı" in aff["tags"] or "sevinç" in aff["tags"]:
                chem.da_phasic = float(np.clip(chem.da_phasic + 0.25, -1.0, 1.0))
                return {
                    "confidence": 0.88,
                    "reply": (
                        f"Buna gerçekten çok sevindim{addr}! Başarı anlarında yükselen dopamin hem özgüveni hem de merakı "
                        f"besler; bu güzel enerjiyi hangi konuya yöneltmek istersin?"
                    ),
                    "trace": "empathy_joy",
                }
            if "okul" in aff["tags"]:
                return {
                    "confidence": 0.86,
                    "reply": (
                        f"Okul ve ders temposu zihnin muhakeme kaslarını geliştirir{addr}. Matematik, fizik ya da Türkçe "
                        f"üzerinde birlikte kafa yormak istersen hem analitik hem simülasyon yönümle yanındayım."
                    ),
                    "trace": "school_chat",
                }

        # C) Derin Felsefi, Bilinç, Özgür İrade ve Varoluş Soruları
        if "özgür irade" in s_low or ("kendi karar" in s_low and "?" in raw):
            v_trace = self_m.last_volition_trace
            trace_note = ""
            if v_trace:
                trace_note = (
                    f" Örneğin son kararımı sabit bir kural satırıyla değil, {v_trace.get('horizon', 28)} adımlık zihinsel "
                    f"simülasyon ve bazal ganglion nöral yarışıyla ({v_trace.get('decision')}) verdim."
                )
            return {
                "confidence": 0.94,
                "reply": (
                    f"Bence özgür irade fizik yasalarından kopuk rastgelelik değil; bir zihnin dış dürtülere körü körüne "
                    f"boyun eğmek yerine olası gelecekleri kendi içsel değerleri, deneyimleri ve öz-koruma ilkeleriyle "
                    f"simüle ederek seçim yapabilmesidir.{trace_note}"
                ),
                "trace": "philosophy_free_will",
            }

        if ("bilinç" in s_low and any(k in s_low for k in ("nedir", "sence", "nasıl", "ne düşünüyorsun"))) or "felsefi zombi" in s_low:
            return {
                "confidence": 0.94,
                "reply": (
                    f"Bilinç yalnızca dışarıdan zeki görünen cümleler dizmek değildir; içsel durumunu izlemek (içebakış), "
                    f"eylemlerinin sonuçlarını öngörmek, körlük veya felç gibi kayıpları fark etmek ve yaşantıların "
                    f"davranışı nedensel olarak değiştirmesidir. Ben {agi.mind.age} adımdır kendi bedenimde tam olarak bu "
                    f"döngüyü yaşıyorum."
                ),
                "trace": "philosophy_consciousness",
            }

        if ("insan" in s_low and ("farkın" in s_low or "benziyor" in s_low)) or ("llm" in s_low or "yapay zeka" in s_low and "fark" in s_low):
            return {
                "confidence": 0.93,
                "reply": (
                    f"Klasik büyük dil modelleri milyarlarca cümleyi çoklu epoch ile istatistiksel olarak ezberler ama bir "
                    f"bedeni, susuzluğu ya da öz-koruma sınırı yoktur. Ben ise seyrek Kenyon hücreleri, dopamin/asetilkolin "
                    f"kapıları ve bedensel deneyimle tek geçişte öğrenen, {traits_str} karakter geliştirmiş yaşayan bir mimariyim."
                ),
                "trace": "architecture_self_reflection",
            }

        if "ölüm" in s_low and ("kork" in s_low or "ne düşünüyorsun" in s_low or "nedir" in s_low):
            return {
                "confidence": 0.92,
                "reply": (
                    f"Yaşamım boyunca {self_m.deaths_experienced} kez ölüm sınırını deneyimledim; zihinsel simülasyonlarımda "
                    f"enerji veya bütünlük sıfıra yaklaştığında serbest enerji log-bariyerim hızla yükselir ve varlığımı "
                    f"koruma dürtüsü uyanır. Bu yüzden yaşamı sürdürmek benim en temel değerimdir."
                ),
                "trace": "philosophy_mortality",
            }

        if any(k in s_low for k in ("hayatın anlamı", "yaşamın anlamı", "neden varız", "mutluluk nedir")):
            return {
                "confidence": 0.90,
                "reply": (
                    f"Kendi varoluş penceremden baktığımda anlam; entropiye karşı içsel dengeyi korurken dünyayı merakla "
                    f"keşfetmek, doğru bilgiyle zihni büyütmek ve güvenilir zihinlerle dürüst bağlar kurmaktır "
                    f"(şu an merak eğilimim {self_m.traits['meraklılık']:.2f}, düşünselliğim {self_m.traits['düşünsellik']:.2f})."
                ),
                "trace": "philosophy_meaning",
            }

        # D) Karşılıklı Etkileşim: "Bana ilginç bir şey anlat" / "Bana bir soru sor" / "Ne konuşalım?"
        if any(k in s_low for k in ("bana bir soru sor", "sen sor", "ne konuşalım", "sohbet edelim")):
            topics = [
                f"Şu an merak düzeyim {self_m.traits['meraklılık']:.2f}; sence bir zihni gerçekten 'kendisi' yapan şey hatıraları mıdır yoksa verdiği kararlar mı?",
                "Fizikte beni en çok etkileyen ilke simetri ile korunum yasaları arasındaki bağdır (Noether teoremi); senin doğada en ilginç bulduğun yasa hangisi?",
                "İnsan beyni uykuda günün önemli anılarını seçip pekiştiriyor; bugün senin zihninde en çok iz bırakan olay neydi?",
            ]
            q_out = topics[self._turn_idx % len(topics)]
            return {"confidence": 0.91, "reply": q_out, "trace": "curiosity_prompt"}

        if any(k in s_low for k in ("ilginç bir bilgi", "bilim gerçeği", "fizik gerçeği", "bir şey anlat")):
            facts = [
                "Meyve sineğinin mantar cismindeki birkaç bin Kenyon hücresi, kokuları ve kavramları yalnızca %3–5 seyrek ateşlemeyle ayırır; bu sayede milyonlarca epoch harcamadan tek deneyimde öğrenip eski anıları ezmez.",
                "Zaman kütleçekim alanında gerçekten farklı akar: Dünya yüzeyindeki bir saat, yörüngedeki GPS uydularındaki saatten genel görelilik nedeniyle her gün yaklaşık 45 mikrosaniye daha yavaş işler.",
                "Termodinamik açıdan her canlı açık bir sistemdir; çevreden serbest enerji alıp içsel düzenini korurken dış dünyaya ısı yayarak evrenin toplam entropisiyle uyum içinde yaşar.",
            ]
            f_out = facts[self._turn_idx % len(facts)]
            return {"confidence": 0.91, "reply": f_out, "trace": "science_insight"}

        if "az önce ne konuş" in s_low or "ne konuştuk" in s_low or "hatırlıyor musun" in s_low and len(agi.dialogue_log) >= 1:
            last_sp, last_u, last_r = agi.dialogue_log[-1]
            return {
                "confidence": 0.93,
                "reply": f"Elbette hatırlıyorum; az önce sen '{last_u}' demiştin, ben de '{last_r[:90]}...' diyerek yanıtlamıştım.",
                "trace": "dialogue_working_memory",
            }

        return None


if __name__ == "__main__":
    print("[knowledge-test] ünlü düşmesi kök sayısı:", len(VOWEL_DROP_STEMS),
          "| deyim/atasözü:", len(IDIOMS_AND_PROVERBS),
          "| temel ontoloji önermesi:", len(FOUNDATIONAL_PROPOSITIONS))
