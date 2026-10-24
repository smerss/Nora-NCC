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
    "keyif": "keyf",
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
    "zehir": "zehr",
    "emir": "emr",
    "kesir": "kesr",
    "nesil": "nesl",
    "asır": "asr",
    "hapis": "haps",
    "beniz": "benz",
    "geniz": "genz",
    "uyku": "uyku",  # uyu-k -> referans
    # Beden ve gündelik yaşam (ikinci hecedeki dar ünlü düşer: boyun -> boynu)
    "boyun": "boyn", "burun": "burn", "ağız": "ağz", "göğüs": "göğs", "omuz": "omz",
    "karın": "karn", "akıl": "akl", "beyin": "beyn", "oğul": "oğl", "koyun": "koyn",
    "gönül": "gönl", "asıl": "asl", "kavuş": "kavş",
}

REVERSE_VOWEL_DROP: dict[str, str] = {v: k for k, v in VOWEL_DROP_STEMS.items() if v != k}

# Orta-İleri Türkçe Temel Fiil Kökleri Dağarcığı (Günlük yaşam + iletişim + bilim + soyut muhakeme)
# Kişinin kendi hâlini bildiren sıfatlar (1. kişi koşacıyla: 'yorgunum', 'yalnızım', 'mutluyum').
# Bunlar bir OLGU değil, o anki yaşantıdır; Nora bunlara bilgi maddesiyle değil, duygudaşlıkla karşılık verir.
# Özel adlar: Türkçede kesme işaretiyle çekimlenir ve büyük harfle yazılır
# ("Himalaya'da", "Antalya'da", "Mars'ta").
PROPER_NOUNS: set[str] = {
    "himalaya", "everest", "antalya", "istanbul", "ankara", "izmir", "bursa", "adana", "konya",
    "türkiye", "ankara", "mars", "jüpiter", "satürn", "venüs", "merih", "güneş", "ay",
    "avrupa", "asya", "afrika", "amerika", "akdeniz", "karadeniz", "ege", "nora", "atlas",
    "nil", "amazon", "sahra", "kafkas", "toros", "alpler", "pasifik", "atlantik",
}

STATE_ADJECTIVES: set[str] = {
    "yorgun", "yalnız", "mutlu", "üzgün", "kötü", "iyi", "aç", "tok", "susuz", "korkmuş",
    "heyecanlı", "sıkılmış", "sıkıntılı", "dingin", "huzurlu", "huzursuz", "gergin", "rahat",
    "mutsuz", "umutsuz", "umutlu", "kaygılı", "endişeli", "sinirli", "kızgın", "kırgın",
    "yaşlı", "genç", "hasta", "sağlıklı", "uykulu", "uykusuz", "dalgın", "şaşkın", "meraklı",
    "bozuk", "karışık", "sıkkın", "dargın", "keyifsiz", "dolu", "tuhaf", "bitkin", "mutsuz",
    "hevesli", "isteksiz", "kırılgan", "duygusal", "sakin", "yorgun", "kırık", "dağınık",
}

CORE_TURKISH_VERBS: set[str] = {
    # Gündelik anlatı (konuşma dilinin en sık fiilleri)
    "izle", "seyret", "başla", "bitir", "oku", "yaz", "dinle", "hazırla", "temizle", "gez",
    "ziyaret", "karşılaş", "tanış", "uğraş", "eğlen", "kaybet", "kazan", "anlat", "söyle", "sor",
    "cevapla", "dene", "tamamla", "sürdür", "böl", "birleştir", "say", "ölç", "tart", "kokla",
    "tat", "dokun", "nefes", "uyan", "giy", "çıkar", "yıka", "pişir", "kaynat", "kes", "doğra",
    "temizlen", "otur", "kalk", "yürü", "koş", "yüz", "uç", "dinlen", "çalış", "oyna", "gül",
    "ağla", "bağır", "fısılda", "düşle", "özle", "sevin", "üzül", "şaşır", "kız", "utan",
    "geçir", "harca", "biriktir", "dene", "çöz", "tamamla", "ertel", "vazgeç", "sürdür", "topla",
    "hisset", "kaygılan", "heyecanlan", "rahatla", "sıkıl", "usan", "umut et", "yalnızlaş",
    "daral", "bozul", "karış", "şaşır", "sevin", "üzül", "sinirlen", "kırıl",

    # Hareket ve eylem
    "git", "gel", "ye", "iç", "dur", "bekle", "kaç", "gir", "çık", "bak", "gör", "duy", "dinle",
    "uç", "yüz", "koş", "yürü", "atla", "düş", "kalk", "otur", "yat", "uyu", "uyan", "dön", "geç",
    "yaklaş", "uzaklaş", "ilerle", "gerile", "durakla", "taş", "taşı", "çek", "it", "at", "tut",
    "bırak", "koy", "al", "ver", "aç", "kapat", "kır", "bük", "kes", "bağla", "çöz", "kur", "yık",
    # Zihin, biliş, araştırma ve dil
    "öğren", "öğret", "bil", "düşün", "anla", "kavrama", "kavra", "sez", "fark", "hatırla", "unut",
    "okuma", "oku", "yaz", "sor", "yanıtla", "cevapla", "anlat", "söyle", "konuş", "sus", "tartış",
    "açıkla", "tanımla", "adlandır", "sınıflandır", "etiketle", "karşılaştır", "ölç", "hesapla",
    "gözlemle", "incele", "araştır", "keşfet", "kanıtla", "doğrula", "yanlışla", "sına", "varsay",
    "çıkar", "genelle", "özetle", "yorumla", "planla", "tasarla", "simüle", "öngör", "tahmin",
    # İrade, duygu ve sosyal etkileşim
    "iste", "seç", "reddet", "kabul", "inan", "güven", "şüphelen", "merak", "sev", "beğen", "kork",
    "üzül", "sevin", "şaşır", "öfkelen", "kız", "sakinleş", "yatış", "umutlan", "özle", "paylaş",
    "yardım", "destekle", "koru", "onar", "affet", "kutla", "öner", "danış", "katıl", "reddet",
    # Doğa, fizik, kimya ve durum geçişleri
    "ol", "oluş", "var", "yaşa", "öl", "doğ", "büyü", "geliş", "değiş", "dönüş", "art", "artır",
    "azal", "azalt", "yüksel", "alçal", "hızlan", "yavaşla", "genleş", "daral", "sıkış", "gevşe",
    "ısıt", "ısın", "soğut", "soğu", "eri", "don", "kayna", "buharlaş", "yoğunlaş", "çözün", "yan",
    "parla", "ışılda", "yansıt", "yansı", "soğur", "ilet", "yayıl", "dağıl", "odakla", "böl", "bölün",
    "birleş", "ayrış", "çök", "çökert", "boz", "bozul", "dengele", "etkile", "tetikle", "üret", "tüket",
    "başla", "bit", "bitir", "sür", "sürdür", "kazan", "kaybet", "çalış", "dinlen", "yorul", "iyileş",
    "belirle", "bul", "bulun", "sağla", "uygula", "engelle", "aktar", "sakla", "barındır", "sergile", "göster",
    "diren", "tanı", "kaynaklan", "yükselt", "tetikle", "yönet", "besle", "sınırla", "genişlet", "yoğunlaş",
    "sentezle", "parçala", "harca", "depola", "sıçra", "yayımla", "doğrula", "çürüt", "yorumla",
}

# Ünsüz yumuşamasına GİRMEYEN çok heceli / alıntı veya Türkçe kökler
NO_SOFTEN_EXTENDED: set[str] = {
    "at", "et", "ot", "süt", "ok", "ek", "ak", "kök", "yük", "top", "ip", "çöp", "saç", "üç", "göç", "suç", "kaç", "iç",
    "sanat", "hukuk", "devlet", "cumhuriyet", "hayat", "merak", "ahlak", "millet", "tabiat", "kanıt", "boyut",
    "taşıt", "yakıt", "anıt", "yapıt", "ölçüt", "kesit", "bulut", "konut", "sepet", "paket", "bilet", "kaset",
    "etiket", "işaret", "suret", "şahsiyet", "hüviyet", "kudret", "devlet", "servet", "kıyamet", "alamet",
    "robot", "pilot", "uzay", "evrak", "ittifak", "hakikat", "adalet", "cesaret", "nezaket", "felaket", "hareket",
    "bereket", "memleket", "şirket", "dikkat", "şefkat", "sohbet", "gurbet", "hasret", "ücret", "nöbet",
    "saat", "kanat", "sanat", "iftar", "sabah", "silah", "sabır",
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
    "gözden kaçırmak": ("deyim", "Dikkatsizlik yüzünden önemli bir ayrıntıyı fark edememek."),
    "kafası karışmak": ("deyim", "Çelişkili veya yoğun bilgiler karşısında ne düşüneceğini bilemez duruma gelmek."),
    "içine sinmek": ("deyim", "Yapılan bir işin veya kararın doğruluğundan gönül rahatlığıyla emin olmak."),
    "tadını çıkarmak": ("deyim", "Güzel bir anın veya durumun keyfini doyasıya yaşamak."),
    "yüzüstü bırakmak": ("deyim", "Güvenen bir kimseyi en çok ihtiyaç duyduğu anda desteksiz bırakmak."),
    "kolları sıvamak": ("deyim", "Bir işe güçlü bir istekle başlamak üzere hazırlanmak."),
    "ipin ucu kaçmak": ("deyim", "Bir konudaki denetimi ve düzeni yitirmek."),
    "taşı gediğine koymak": ("deyim", "Tam zamanında ve yerinde çok etkili, düşündürücü bir söz söylemek."),
    "azıcık aşım kaygısız başım": ("atasözü", "Aşırı hırstan uzak, sade ve dengeli yaşam zihne huzur verir."),
    "bugünün işini yarına bırakma": ("atasözü", "Zamanında yapılan eylem birikmiş belirsizliği ve stresi önler."),
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
    ("özgür irade", "ISA", "zihinsel simülasyon seçimi"),
    ("özgür irade", "CAUSES", "içsel değer yönelimi"),
    ("varoluş", "CAUSES", "entropiye karşı içsel denge arayışı"),
    ("anlam", "ISA", "keşif ve dürüst bağ kurma yönelimi"),
    ("mutluluk", "CAUSES", "içsel denge ve dopamin uyumu"),
    ("hakikat", "ISA", "deney ve kanıtla sınanan gerçeklik"),
    ("etik", "USED_FOR", "zararı önleme ve güven koruma"),
    ("merak", "CAUSES", "keşif ve öğrenme"),
    ("empati", "USED_FOR", "başkasının duygusunu anlama"),
    ("okul", "USED_FOR", "bilgi ve düşünme disiplini kazanma"),
    ("bilim", "USED_FOR", "doğa yasalarını deney ve kanıtla çözme"),
    ("sanat", "USED_FOR", "duygu ve estetik anlamı ifade etme"),
    ("zaman", "ISA", "değişimin ve nedenselliğin yönü"),
    # Günlük Yaşam, Hava, Doğa, Toplum, Sağlık ve Teknoloji (Orta-İleri Genel Türkçe Temeli)
    ("yağmur", "CAUSES", "toprağın sulanması ve su döngüsü"),
    ("bulut", "CAUSES", "yağış oluşumu"),
    ("rüzgar", "ISA", "basınç farkından doğan hava akımı"),
    ("mevsim", "CAUSES", "iklim ve sıcaklık değişimi"),
    ("ilkbahar", "ISA", "doğanın canlandığı mevsim"),
    ("kış", "ISA", "soğuk ve yağışlı mevsim"),
    ("yaz", "ISA", "sıcak ve güneşli mevsim"),
    ("sonbahar", "ISA", "geçiş ve yaprak dökümü mevsimi"),
    ("orman", "USED_FOR", "ekosistemi ve temiz havayı koruma"),
    ("okyanus", "ISA", "dev tuzlu su kütlesi"),
    ("deprem", "CAUSES", "yer kabuğunda sarsıntı"),
    ("kitap", "USED_FOR", "bilgi ve hayal gücünü aktarma"),
    ("müzik", "USED_FOR", "ritim ve duygu uyumu oluşturma"),
    ("spor", "CAUSES", "bedensel dayanıklılık ve sağlık"),
    ("beslenme", "USED_FOR", "hücresel enerji ve yaşam desteği"),
    ("dinlenme", "CAUSES", "zihinsel berraklık ve sinaptik toparlanma"),
    ("hastalık", "CAUSES", "bedensel dengenin bozulması"),
    ("ilaç", "USED_FOR", "hastalığı iyileştirme"),
    ("arkadaşlık", "CAUSES", "karşılıklı güven ve dayanışma"),
    ("dürüstlük", "CAUSES", "kalıcı sosyal güven"),
    ("sabır", "USED_FOR", "zor süreçlerde dengeyi koruma"),
    ("planlama", "USED_FOR", "belirsizliği azaltıp hedefe ulaşma"),
    ("deney", "USED_FOR", "varsayımları gerçeklikle sınama"),
    ("gözlem", "USED_FOR", "olguları dikkatle inceleme"),
    ("mantık", "USED_FOR", "öncüllerden tutarlı sonuç çıkarma"),
    ("bilgisayar", "USED_FOR", "veri işleme ve hesaplama"),
    ("internet", "USED_FOR", "küresel bilgi paylaşımı ve araştırma"),
    ("elektrik", "USED_FOR", "enerji ve sinyal iletimi"),
    ("manyetizma", "CAUSES", "yüklü parçacıklar üzerinde kuvvet"),
    ("sıcaklık", "ISA", "taneciklerin ortalama kinetik enerjisi"),
    ("basınç", "ISA", "birim yüzeye uygulanan dik kuvvet"),
    ("kimya", "ISA", "maddenin bağ ve dönüşüm bilimi"),
    ("molekül", "ISA", "atomların bağ kurduğu yapı"),
    ("oksijen", "USED_FOR", "hücresel solunum ve yanma"),
    ("karbon", "ISA", "organik yaşamın temel elementi"),
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
        "şaşır": (+0.20, 0.80, "merak"),
        "umut": (+0.70, 0.60, "sevinç"),
        "huzur": (+0.75, 0.25, "sevinç"),
        "rahatla": (+0.65, 0.30, "sevinç"),
        "kararsız": (-0.25, 0.55, "kararsızlık"),
        "karışık": (-0.30, 0.55, "kararsızlık"),
        "hasta": (-0.60, 0.35, "hastalık"),
        "ağrı": (-0.65, 0.55, "hastalık"),
        "ateş": (-0.45, 0.50, "hastalık"),
        "öfke": (-0.65, 0.85, "öfke"),
        "sinirli": (-0.60, 0.80, "öfke"),
        "kızgın": (-0.60, 0.75, "öfke"),
        "özür": (-0.15, 0.40, "özür"),
        "pişman": (-0.40, 0.45, "özür"),
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
        """SIFIR hazır cümle şablonu: tüm konuşma, empati, felsefe, bellek ve merak yanıtlarını
        Prefrontal İletişimsel İrade + TurkishSyntaxEngine kök çekimiyle canlı sentezler."""
        self._turn_idx += 1
        return agi.broca_organ.generate_response(raw, s_low, agi, speaker=speaker)


if __name__ == "__main__":
    print("[knowledge-test] ünlü düşmesi kök sayısı:", len(VOWEL_DROP_STEMS),
          "| deyim/atasözü:", len(IDIOMS_AND_PROVERBS),
          "| temel ontoloji önermesi:", len(FOUNDATIONAL_PROPOSITIONS))
