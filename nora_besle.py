#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_besle.py — Nora'yı Türkçe ve dünya bilgisiyle besleme hattı.

Neden ayrı bir dosya?
---------------------
Bir çocuk dünyayı tek tek cümlelerle öğrenir. Burada yapılan da odur: yapılandırılmış dünya bilgisi
(kavram – ilişki – kavram) ve doğal Türkçe cümleler Nora'nın kavram ağına, söz organına ve sözdizimi
motoruna verilir. Model eğitimi (epoch) YOKTUR; her bilgi tek geçişte yerleşir (A2/A15/A35 ile aynı ilke).

İçerik üç katmandır:
  1. TRIPLES    : dünya bilgisi üçlüleri (ISA / CAN_DO / USED_FOR / CAUSES / HAS_PROP / LOCATED_AT / OPPOSITE).
                  Bunlar kavram ağına doğrudan girer; Nora bunlarla soruları yanıtlar.
  2. SENTENCES  : aynı bilginin doğal Türkçe karşılıkları (dil örneği; sözdizimi ve söz organı bunları duyar).
  3. DIALOGUE   : günlük konuşma örnekleri (insan–Nora). Söz organı buradan Türkçenin ritmini öğrenir.

Kullanım:
    python nora_besle.py --out data/nora_besin --epochs 1
    python nora_studio.py --load data/nora_besin
"""
from __future__ import annotations

import argparse
import os
import time

# ============================================================================
# 1. DÜNYA BİLGİSİ (kavram, ilişki, kavram)
#    ISA: X bir Y'dir | CAN_DO: X ... yapar | USED_FOR: X ... için kullanılır
#    CAUSES: X ...'e yol açar | HAS_PROP: X ... özelliğine sahiptir | OPPOSITE: zıt
# ============================================================================

# ============================================================================
# 0. SÖZCÜK DAĞARCIĞI (bilgi değil, sözcük): çocuk gibi önce kelimeleri tanır.
#    Bu liste kavram ağına "kavram" olarak yazılır; hiçbir olgu iddiası taşımaz.
#    Amacı: Nora'nın bilmediği sözcüğü "bilmiyorum" diye işaretlemesi DÜRÜST olsun.
# ============================================================================

VOCAB: list[str] = """
sessiz suskun durgun sakin donuk dalgın yorul üzgün keyifsiz gergin heyecanlı mutlu kızgın
kırgın yalnız huzur meraklı sabırsız umutlu umutsuz cesur korkak dürüst tembel çalışkan
akıllı komik ciddi nazik kaba sabırlı dikkatli dikkatsiz yaratıcı mantıklı duygusal kararlı
kararsız inatçı şüpheci aç tok uykulu rahat tatlı sadık sevimli evcil

hikâye hikaye masal öykü anlatı tavsiye öğüt fikir danış
moral kafa can öğretmen doktor mühendis öğrenci sınav sır gitar kahve şaka keyif gönül
sessiz gürültülü sakin mutlu üzgün kızgın kırgın yalnız uykulu heyecanlı endişeli gergin rahat
huzurlu meraklı sabırsız umutlu umutsuz cesur korkak dürüst yalancı tembel çalışkan akıllı aptal
komik ciddi nazik kaba sabırlı dikkatli dikkatsiz dalgın yaratıcı mantıklı kararlı kararsız inatçı
şüpheci açgözlü zor kolay güzel çirkin iyi kötü büyük küçük uzun kısa ağır hafif hızlı yavaş
ödev ders soru cevap sınav not başarı hata doğru yanlış eksik tam hazır meşgul yorgun aç tok hasta
kucak adam kız akıl ruh ırmak kök bahar fırtına harf sözcük yazı çanta sınıf
durgun suskun donuk dalgın keyifsiz tedirgin huzursuz kaygılı öfkeli sevinçli
bilim insanı teleskop gözlem deney organ diş
ben sen o biz siz onlar kendi kendim kendin şey şeyler herkes kimse biri birisi hiçbiri
bu şu bunlar şunlar öyle böyle şöyle hangi neden niçin niye nasıl nerede nereye nereden ne zaman
kaç kadar göre rağmen için ile ve ama fakat ancak çünkü eğer ki ise de da bile yalnız sadece hep
çok az en daha pek çok daha çoğu bazı birkaç bütün tüm her hiçbir başka diğer öteki aynı benzer
şimdi şu an bugün dün yarın önce sonra önceki sonraki sabah akşam gece gündüz öğle erken geç
burada orada buraya oraya buradan oradan nerede dışarı içeri yukarı aşağı ileri geri yanında
üzerinde altında yanına arasında içinde dışında karşısında yanı sıra etrafında ortasında
evet hayır tamam peki belki tabii elbette kesinlikle asla gerçekten cidden belki
gel git kal dur bak yap et ol ver al aç kapa kalk in çık otur yat uyu uyan ye iç konuş sus
söyle anlat sor cevap iste dile merak et düşün bil anla öğren öğret hatırla unut sev sevin
üzül kork korkma ağla gül gülümse bekle ara bul kaybet kazan başla bitir devam et bırak tut
yaşa öl doğ büyü küçül değiş dön dolaş yürü koş uç yüz sürün zıpla tırman dinlen çalış oyna
oku yaz çiz say hesapla ölç tart dene gör duy dokun tat kokla nefes al
insan kişi çocuk genç yaşlı kadın erkek anne baba kardeş abla ağabey dede nine arkadaş dost
komşu öğretmen öğrenci doktor hasta işçi çiftçi mühendis sanatçı yazar şair bilim insanı
aile toplum kalabalık yalnızlık sohbet konuşma söz cümle kelime dil ses sessizlik gürültü
gün hafta ay yıl mevsim saat dakika an zaman geçmiş şimdi gelecek geçmiş anı anılar
ev oda mutfak banyo yatak salon bahçe sokak yol şehir köy ülke dünya evren gökyüzü yer toprak
su hava ateş ışık karanlık gölge sıcak soğuk ılık serin rüzgâr yağmur kar bulut güneş ay yıldız
deniz göl nehir dağ tepe ova orman çöl ada kıyı sahil dalga kum taş kaya
yemek ekmek su çay kahve süt meyve sebze et balık tuz şeker tatlı tuzlu acı ekşi
baş göz kulak burun ağız dudak dil diş yüz saç el kol ayak parmak bacak sırt göğüs karın
kalp beyin akciğer mide böbrek karaciğer kan kemik kas deri sinir hücre
sağlık hastalık ağrı acı ateş öksürük yara ilaç uyku yorgunluk dinlenme yemek içmek
duygu sevgi nefret korku kaygı öfke mutluluk üzüntü yalnızlık merak umut umutsuzluk
huzur huzursuzluk sıkıntı can sıkıntısı sevinç heyecan şaşkınlık gurur utanç suçluluk
iyi kötü güzel çirkin doğru yanlış gerçek yalan kolay zor büyük küçük uzun kısa geniş dar
yüksek alçak hızlı yavaş güçlü zayıf güçlü sağlıklı hasta mutlu mutsuz yorgun dinç aç tok
kitap defter kalem kâğıt masa sandalye yatak kapı pencere duvar çatı bahçe araba otobüs tren
uçak gemi bisiklet telefon bilgisayar internet radyo televizyon müzik resim film oyun oyuncak
iş okul üniversite sınav ders ödev proje para alışveriş market pazar
renk kırmızı mavi yeşil sarı siyah beyaz pembe mor turuncu kahverengi gri
sayı bir iki üç dört beş altı yedi sekiz dokuz on yüz bin milyon yarım çeyrek
neden sonuç amaç soru cevap problem çözüm fikir düşünce hayal rüya anlam değer özgürlük
zaman mekân varlık yokluk yaşam ölüm doğum büyüme değişim düzen kaos gerçeklik
""".split()

TRIPLES: list[tuple[str, str, str]] = [
    # ---------------------------------------------------------------- canlılar
    ("kedi", "ISA", "evcil hayvan"), ("köpek", "ISA", "evcil hayvan"),
    ("kedi", "CAUSES", "huzur"), ("köpek", "CAUSES", "keyif"),
    # --- genel canlı/hayvan/bitki kuralları: özele çıkarımın dayanağı (kuş bir hayvandır -> ...)
    ("canlı", "CAN_DO", "beslenmek"), ("canlı", "CAN_DO", "büyümek"),
    ("canlı", "CAN_DO", "çoğalmak"), ("canlı", "CAN_DO", "nefes almak"),
    ("canlı", "HAS_PROP", "cansız varlıklardan ayrılan"), ("canlı", "ISA", "varlık"),
    ("hayvan", "CAN_DO", "hareket etmek"), ("hayvan", "CAN_DO", "besin aramak"),
    ("hayvan", "CAN_DO", "içgüdüyle davranmak"), ("hayvan", "CAN_DO", "içgüdüyle yön bulmak"),
    ("hayvan", "HAS_PART", "duyu organları"),
    ("bitki", "ISA", "canlı"), ("bitki", "CAN_DO", "güneş ışığıyla besin üretmek"),
    ("bitki", "CAN_DO", "kökleriyle su almak"),
    ("memeli", "ISA", "hayvan"), ("memeli", "CAN_DO", "yavrusunu sütle beslemek"),
    ("kuş", "ISA", "hayvan"), ("kuş", "HAS_PART", "kanat"),
    ("kanat", "CAN_DO", "uçmak"), ("kanat", "ISA", "organ"),
    ("balık", "ISA", "hayvan"), ("balık", "LOCATED_AT", "su"),
    ("balık", "HAS_PART", "solungaç"), ("solungaç", "CAN_DO", "suda nefes almak"),
    ("balık", "CAN_DO", "yüzmek"), ("yarasa", "ISA", "memeli"),
    ("yarasa", "CAN_DO", "karanlıkta yön bulmak"),
    ("böcek", "ISA", "hayvan"), ("böcek", "HAS_PART", "duyarga"),
    ("insan", "CAN_DO", "düşünmek"), ("insan", "CAN_DO", "konuşmak"),
    ("insan", "CAN_DO", "öğrenmek"), ("insan", "CAN_DO", "rüya görmek"),
    ("insan", "CAN_DO", "hissetmek"),
    ("rüya görmek", "CAUSES", "beynin dinlenmesi"),
    ("uyku", "CAN_DO", "bedeni onarmak"), ("uyku", "CAUSES", "belleğin güçlenmesi"),
    ("güneş", "ISA", "yıldız"), ("güneş", "LOCATED_AT", "gökyüzü"),
    ("güneş", "CAUSES", "gündüzün olması"), ("güneş", "CAUSES", "ısınma"),
    ("yıldız", "LOCATED_AT", "gökyüzü"), ("yıldız", "CAN_DO", "geceleyin parlama"),
    ("gökyüzü", "ISA", "boşluk"), ("bulut", "LOCATED_AT", "gökyüzü"),
    ("bulut", "CAUSES", "yağmurun yağması"), ("yağmur", "CAN_DO", "bitkileri sulamak"),
    ("gökkuşağı", "CAUSES", "güneş ışığının su damlalarında kırılması"),
    ("gökkuşağı", "HAS_PART", "yedi renk"),
    ("su", "CAN_DO", "bitkilerin büyümesini sağlamak"),
    ("ateş", "CAUSES", "ısınma"), ("ateş", "CAN_DO", "yakmak"),
    ("beyin", "CAN_DO", "düşünmek"), ("beyin", "CAN_DO", "anıları saklamak"),
    ("kalp", "CAN_DO", "kanı pompalamak"), ("akciğer", "CAN_DO", "nefes almak"),
    ("göz", "CAN_DO", "görmek"), ("kulak", "CAN_DO", "duymak"),
    ("burun", "CAN_DO", "koku almak"), ("dil", "CAN_DO", "tat almak"),
    ("beyin", "ISA", "organ"), ("kalp", "ISA", "organ"), ("akciğer", "ISA", "organ"),
    ("göz", "ISA", "duyu organı"), ("kulak", "ISA", "duyu organı"),
    ("evcil hayvan", "ISA", "hayvan"), ("hayvan", "ISA", "canlı"),
    ("bitki", "ISA", "canlı"), ("mantar", "ISA", "canlı"), ("insan", "ISA", "canlı"),
    ("kuş", "ISA", "hayvan"), ("balık", "ISA", "hayvan"), ("böcek", "ISA", "hayvan"),
    ("zürafa", "ISA", "hayvan"), ("zürafa", "HAS_PART", "uzun boyun"),
    ("fil", "ISA", "hayvan"), ("fil", "HAS_PART", "hortum"),
    ("aslan", "ISA", "hayvan"), ("aslan", "HAS_PROP", "yırtıcı"),
    ("kartal", "ISA", "kuş"), ("kartal", "CAN_DO", "uçmak"),
    ("penguen", "ISA", "kuş"), ("penguen", "CANNOT_DO", "uçmak"), ("penguen", "CAN_DO", "yüzmek"),
    ("kivi", "ISA", "kuş"), ("kivi", "CANNOT_DO", "uçmak"),
    ("yunus", "ISA", "hayvan"), ("yunus", "CAN_DO", "yüzmek"),
    ("arı", "ISA", "böcek"), ("arı", "CAN_DO", "bal yapmak"),
    ("karınca", "ISA", "böcek"), ("karınca", "CAN_DO", "koloni kurmak"),
    ("balina", "ISA", "hayvan"), ("balina", "CAN_DO", "yüzmek"),
    ("kaplumbağa", "ISA", "hayvan"), ("kaplumbağa", "HAS_PART", "kabuk"),
    ("ağaç", "ISA", "bitki"), ("ağaç", "HAS_PART", "gövde"),
    ("çiçek", "ISA", "bitki"), ("gül", "ISA", "çiçek"), ("papatya", "ISA", "çiçek"),
    ("meşe", "ISA", "ağaç"), ("çam", "ISA", "ağaç"),
    # ---------------------------------------------------------------- insan bedeni
    ("kalp", "ISA", "organ"), ("beyin", "ISA", "organ"), ("akciğer", "ISA", "organ"),
    ("mide", "ISA", "organ"), ("böbrek", "ISA", "organ"), ("karaciğer", "ISA", "organ"),
    ("deri", "ISA", "organ"), ("göz", "ISA", "organ"), ("kulak", "ISA", "organ"),
    ("kalp", "CAN_DO", "kan pompalamak"), ("beyin", "CAN_DO", "düşünmek"),
    ("beyin", "CAN_DO", "öğrenmek"), ("akciğer", "CAN_DO", "nefes almak"),
    ("mide", "CAN_DO", "sindirmek"), ("böbrek", "CAN_DO", "kanı süzmek"),
    ("kan", "ISA", "sıvı"), ("kan", "CAN_DO", "oksijen taşımak"),
    ("kemik", "ISA", "doku"), ("kas", "ISA", "doku"), ("sinir", "ISA", "doku"),
    ("uyku", "ISA", "dinlenme durumu"), ("uyku", "USED_FOR", "bedeni onarmak"),
    ("su", "ISA", "madde"), ("güneş", "USED_FOR", "ısınmak"), ("güneş", "HAS_PROP", "sıcak"),
    ("ses", "HAS_PROP", "dalga"), ("ışık", "HAS_PROP", "hız"),
    ("uyku", "CAUSES", "belleğin pekişmesi"), ("uyku", "CAUSES", "beden onarımı"),
    ("yürüyüş", "CAUSES", "kalp sağlığı"), ("su", "USED_FOR", "yaşam"),
    # ---------------------------------------------------------------- bilim
    ("fizik", "ISA", "bilim"), ("kimya", "ISA", "bilim"), ("matematik", "ISA", "bilim"),
    ("everest", "ISA", "dağ"), ("everest", "LOCATED_AT", "himalaya"),
    ("everest", "HAS_PROP", "dünyanın en yüksek dağı"),
    ("yalan", "USED_FOR", "kendini korumak"), ("yalan söylemek", "CAUSES", "güven kaybı"),
    ("insan", "CAN_DO", "yalan söylemek"),
    ("bilim insanı", "CAN_DO", "gözlem yapmak"), ("bilim insanı", "CAN_DO", "deney yapmak"),
    ("uzay", "USED_FOR", "teleskopla gözlem"), ("teleskop", "USED_FOR", "uzayı gözlemlemek"),
    ("ağız", "ISA", "organ"), ("ağız", "CAN_DO", "konuşmak"), ("ağız", "CAN_DO", "yemek yemek"),
    ("ağız", "HAS_PART", "dişler"),
    ("kuantum bilgisayar", "ISA", "bilgisayar"), ("bilgisayar", "ISA", "makine"),
    ("bilgisayar", "CAN_DO", "hesaplamak"), ("uzay", "ISA", "boşluk"),
    ("biyoloji", "ISA", "bilim"), ("astronomi", "ISA", "bilim"),
    ("bilim", "USED_FOR", "doğayı anlamak"),
    ("atom", "ISA", "madde parçası"), ("molekül", "ISA", "madde parçası"),
    ("elektron", "ISA", "atom parçası"), ("proton", "ISA", "atom parçası"), ("nötron", "ISA", "atom parçası"),
    ("atom", "HAS_PART", "çekirdek"), ("madde", "HAS_PROP", "kütle"),
    ("yerçekimi", "CAUSES", "cisimlerin düşmesi"), ("enerji", "HAS_PROP", "korunum"),
    ("ışık", "CAN_DO", "yansımak"), ("ses", "CANNOT_DO", "boşlukta yayılmak"),
    ("ses", "ISA", "dalga"), ("ışık", "ISA", "dalga"),
    ("sıcaklık", "CAUSES", "hâl değişimi"),
    ("buz", "ISA", "su biçimi"), ("buz", "HAS_PROP", "katı"),
    ("buhar", "ISA", "su"), ("buhar", "HAS_PROP", "gaz"),
    ("su", "HAS_PROP", "sıvı"), ("su", "CAN_DO", "çözmek"),
    ("oksijen", "ISA", "gaz"), ("hidrojen", "ISA", "gaz"), ("azot", "ISA", "gaz"),
    ("oksijen", "CAN_DO", "yanmayı beslemek"),
    ("fotosentez", "ISA", "bitki süreci"), ("fotosentez", "CAN_DO", "besin üretmek"), ("fotosentez", "USED_FOR", "bitkilerin beslenmesi"),
    ("fotosentez", "CAUSES", "oksijen salınımı"),
    ("DNA", "CAN_DO", "kalıtımı taşımak"), ("hücre", "ISA", "canlı birim"),
    ("evrim", "CAUSES", "türlerin değişmesi"), ("bakteri", "ISA", "mikrop"),
    ("virüs", "ISA", "mikrop"), ("mikrop", "CAUSES", "hastalık"),
    ("aşı", "USED_FOR", "hastalıktan korunmak"),
    ("antibiyotik", "USED_FOR", "bakteri enfeksiyonu"),
    # ---------------------------------------------------------------- evren / dünya
    ("dünya", "ISA", "gezegen"), ("mars", "ISA", "gezegen"), ("merkür", "ISA", "gezegen"),
    ("venüs", "ISA", "gezegen"), ("jüpiter", "ISA", "gezegen"), ("satürn", "ISA", "gezegen"),
    ("merkür", "HAS_PROP", "güneşe en yakın"), ("jüpiter", "HAS_PROP", "en büyük gezegen"),
    ("satürn", "HAS_PART", "halka"), ("dünya", "CAN_DO", "güneşin etrafında dönmek"),
    ("güneş", "ISA", "yıldız"), ("ay", "ISA", "uydu"),
    ("ay", "CAN_DO", "dünyanın etrafında dönmek"),
    ("güneş", "CAUSES", "gündüz"), ("dünyanın dönüşü", "CAUSES", "gece ve gündüz"),
    ("mevsimler", "CAUSES", "dünyanın eğik ekseni"),
    ("gökyüzü", "HAS_PROP", "mavi"), ("deniz", "HAS_PROP", "tuzlu"),
    ("okyanus", "ISA", "deniz"), ("dağ", "ISA", "yer şekli"), ("ova", "ISA", "yer şekli"),
    ("nehir", "ISA", "yer şekli"), ("göl", "ISA", "yer şekli"),
    ("yağmur", "CAUSES", "bulutların yoğuşması"), ("bulut", "ISA", "su buharı"),
    ("rüzgâr", "CAUSES", "basınç farkı"), ("kar", "ISA", "su"),
    ("deprem", "CAUSES", "yer kabuğunun kırılması"),
    ("vulkan", "CAUSES", "lav püskürmesi"),
    ("toprak", "USED_FOR", "tarım"), ("ağaç", "CAN_DO", "oksijen üretmek"),
    # ---------------------------------------------------------------- coğrafya / kültür
    ("türkiye", "ISA", "ülke"), ("antalya", "ISA", "şehir"), ("istanbul", "ISA", "şehir"),
    ("ankara", "ISA", "şehir"), ("izmir", "ISA", "şehir"),
    ("ankara", "HAS_PROP", "başkent"), ("istanbul", "HAS_PART", "boğaz"),
    ("türkçe", "ISA", "dil"), ("türkçe", "USED_FOR", "iletişim"),
    ("akdeniz", "ISA", "deniz"), ("antalya", "LOCATED_AT", "akdeniz"),
    ("toroslar", "ISA", "dağ"), ("antalya", "LOCATED_AT", "toroslar"),
    ("kızılırmak", "ISA", "nehir"), ("van gölü", "ISA", "göl"),
    # ---------------------------------------------------------------- teknoloji
    ("bilgisayar", "ISA", "makine"), ("bilgisayar", "CAN_DO", "hesaplamak"),
    ("bilgisayar", "CAN_DO", "bilgi saklamak"),
    ("internet", "USED_FOR", "bilgi paylaşımı"), ("internet", "ISA", "ağ"),
    ("telefon", "USED_FOR", "iletişim"), ("yazılım", "ISA", "program"),
    ("algoritma", "USED_FOR", "problem çözmek"), ("veri", "USED_FOR", "öğrenmek"),
    ("yapay zekâ", "ISA", "yazılım"), ("yapay zekâ", "CAN_DO", "öğrenmek"),
    ("robot", "ISA", "makine"), ("elektrik", "CAN_DO", "enerji taşımak"),
    ("pil", "USED_FOR", "enerji depolamak"), ("motor", "USED_FOR", "hareket üretmek"),
    # ---------------------------------------------------------------- gündelik hayat
    ("ekmek", "ISA", "yiyecek"), ("elma", "ISA", "yiyecek"), ("peynir", "ISA", "yiyecek"),
    ("çay", "ISA", "içecek"), ("kahve", "ISA", "içecek"), ("su", "ISA", "içecek"),
    ("kahvaltı", "ISA", "öğün"), ("öğle yemeği", "ISA", "öğün"), ("akşam yemeği", "ISA", "öğün"),
    ("mutfak", "ISA", "oda"), ("yatak odası", "ISA", "oda"), ("banyo", "ISA", "oda"),
    ("sandalye", "USED_FOR", "oturmak"), ("masa", "USED_FOR", "üzerinde çalışmak"),
    ("yatak", "USED_FOR", "uyumak"), ("kapı", "USED_FOR", "geçiş"),
    ("kitap", "USED_FOR", "okumak"), ("kalem", "USED_FOR", "yazmak"),
    ("müzik", "USED_FOR", "dinlenmek"), ("resim", "ISA", "sanat"),
    ("şiir", "ISA", "sanat"), ("roman", "ISA", "kitap"),
    ("dostluk", "HAS_PROP", "güven"), ("aile", "ISA", "insan topluluğu"),
    ("anne", "ISA", "aile üyesi"), ("baba", "ISA", "aile üyesi"),
    ("kardeş", "ISA", "aile üyesi"), ("arkadaş", "ISA", "insan"),
    ("mutluluk", "ISA", "duygu"), ("üzüntü", "ISA", "duygu"), ("korku", "ISA", "duygu"),
    ("öfke", "ISA", "duygu"), ("yalnızlık", "ISA", "duygu"), ("merak", "ISA", "duygu"),
    ("sevgi", "ISA", "duygu"), ("umut", "ISA", "duygu"),
    ("uyku", "USED_FOR", "dinlenmek"), ("spor", "CAUSES", "kas gelişimi"),
    ("yemek", "USED_FOR", "beslenmek"), ("temizlik", "USED_FOR", "sağlık"),
    # ---------------------------------------------------------------- matematik / fizik
    ("toplama", "ISA", "işlem"), ("çıkarma", "ISA", "işlem"), ("çarpma", "ISA", "işlem"),
    ("bölme", "ISA", "işlem"), ("asal sayı", "ISA", "sayı"),
    ("üçgen", "ISA", "geometrik şekil"), ("kare", "ISA", "geometrik şekil"),
    ("daire", "ISA", "geometrik şekil"), ("küre", "ISA", "geometrik şekil"),
    ("hız", "ISA", "nicelik"), ("ivme", "ISA", "nicelik"), ("kuvvet", "ISA", "nicelik"),
    ("kütle", "ISA", "nicelik"), ("sürtünme", "CAUSES", "hareketin yavaşlaması"),
    ("eylemsizlik", "CAUSES", "hareketin sürmesi"),
    ("kuantum", "ISA", "fizik alanı"), ("görelilik", "ISA", "fizik kuramı"),
    # ---------------------------------------------------------------- zıtlıklar
    ("sıcak", "OPPOSITE", "soğuk"), ("büyük", "OPPOSITE", "küçük"),
    ("hızlı", "OPPOSITE", "yavaş"), ("açık", "OPPOSITE", "kapalı"),
    ("gece", "OPPOSITE", "gündüz"), ("mutlu", "OPPOSITE", "üzgün"),
    ("kolay", "OPPOSITE", "zor"), ("boş", "OPPOSITE", "dolu"),
]

# ============================================================================
# 2. DOĞAL TÜRKÇE CÜMLELER (aynı bilgi, dil örneği olarak)
# ============================================================================

SENTENCES: list[str] = [
    "Kedi evcil bir hayvandır.", "Köpek evcil bir hayvandır.", "Zürafa uzun boyunlu bir hayvandır.",
    "Fil hortumlu bir hayvandır.", "Kartal bir kuştur.", "Penguen bir kuştur ama uçamaz.",
    "Kivi bir kuştur ama uçamaz.", "Yunus bir hayvandır ve yüzer.", "Arı bir böcektir ve bal yapar.",
    "Balina denizde yaşayan bir hayvandır.", "Kaplumbağanın kabuğu vardır.",
    "Ağaç bir bitkidir.", "Gül bir çiçektir.", "Meşe bir ağaçtır.", "Çam bir ağaçtır.",
    "Kalp bir organdır ve kan pompalar.", "Beyin bir organdır ve düşünmeyi sağlar.",
    "Akciğer nefes almamızı sağlar.", "Mide yemeği sindirir.", "Böbrek kanı süzer.",
    "Kan oksijen taşır.", "Kemik bir dokudur.", "Uyku belleği pekiştirir.", "Yürüyüş kalp sağlığına iyi gelir.",
    "Fizik bir bilimdir.", "Kimya bir bilimdir.", "Matematik bir bilimdir.", "Biyoloji bir bilimdir.",
    "Astronomi gökleri inceler.",
    "Atom maddenin küçük parçasıdır.", "Elektron atomun parçasıdır.",
    "Yerçekimi cisimlerin düşmesine yol açar.", "Ses boşlukta yayılmaz.", "Işık yansır.",
    "Buz suyun katı hâlidir.", "Buhar suyun gaz hâlidir.", "Su sıvı bir maddedir ve çözer.",
    "Oksijen bir gazdır.", "Hidrojen bir gazdır.", "Azot bir gazdır.",
    "Fotosentez bitkilerin besin üretmesini sağlar.", "Fotosentez oksijen salınımına yol açar.",
    "DNA kalıtımı taşır.", "Hücre canlıların temel birimidir.", "Bakteri bir mikrop türüdür.",
    "Virüs bir mikrop türüdür ve hastalık yapar.", "Aşı hastalıktan korunmak için kullanılır.",
    "Antibiyotik bakteri enfeksiyonu için kullanılır.",
    "Dünya bir gezegendir ve Güneş'in etrafında döner.", "Mars bir gezegendir.",
    "Merkür Güneş'e en yakın gezegendir.", "Jüpiter en büyük gezegendir.", "Satürn'ün halkası vardır.",
    "Güneş bir yıldızdır.", "Ay Dünya'nın uydusudur.", "Dünyanın dönüşü gece ve gündüze yol açar.",
    "Mevsimler dünyanın eğik ekseninden kaynaklanır.", "Deniz suyu tuzludur.", "Gökyüzü mavidir.",
    "Yağmur bulutların yoğuşmasıyla oluşur.", "Rüzgâr basınç farkından doğar.", "Kar bir su biçimidir.",
    "Deprem yer kabuğunun kırılmasıyla olur.", "Volkan lav püskürtür.", "Toprak tarım için kullanılır.",
    "Türkiye bir ülkedir.", "Ankara Türkiye'nin başkentidir.", "İstanbul bir şehirdir ve boğazı vardır.",
    "Antalya Akdeniz kıyısında bir şehirdir.", "İzmir bir şehirdir.", "Toroslar bir dağ sırasıdır.",
    "Kızılırmak bir nehirdir.", "Van Gölü bir göldür.", "Türkçe bir dildir.",
    "Bilgisayar bir makinedir ve hesaplar.", "İnternet bilgi paylaşımı için kullanılır.",
    "Telefon iletişim için kullanılır.", "Yazılım bir programdır.", "Algoritma problem çözmek için kullanılır.",
    "Yapay zekâ bir yazılımdır ve öğrenir.", "Robot bir makinedir.", "Elektrik enerji taşır.",
    "Pil enerji depolamak için kullanılır.", "Motor hareket üretmek için kullanılır.",
    "Ekmek bir yiyecektir.", "Elma bir yiyecektir.", "Peynir bir yiyecektir.",
    "Çay bir içecektir.", "Kahve bir içecektir.", "Su bir içecektir ve yaşam için gereklidir.",
    "Kahvaltı bir öğündür.", "Akşam yemeği bir öğündür.", "Mutfak bir odadır.", "Yatak odası bir odadır.",
    "Sandalye oturmak için kullanılır.", "Masa çalışmak için kullanılır.", "Yatak uyumak için kullanılır.",
    "Kitap okumak için kullanılır.", "Kalem yazmak için kullanılır.", "Müzik dinlenmek için kullanılır.",
    "Şiir bir sanattır.", "Resim bir sanattır.", "Dostluk güven demektir.", "Aile bir insan topluluğudur.",
    "Anne bir aile üyesidir.", "Baba bir aile üyesidir.", "Kardeş bir aile üyesidir.",
    "Mutluluk bir duygudur.", "Üzüntü bir duygudur.", "Korku bir duygudur.", "Öfke bir duygudur.",
    "Yalnızlık bir duygudur.", "Merak bir duygudur.", "Sevgi bir duygudur.", "Umut bir duygudur.",
    "Spor kas gelişimine yol açar.", "Yemek beslenmek için kullanılır.", "Temizlik sağlık için kullanılır.",
    "Toplama bir işlemdir.", "Çıkarma bir işlemdir.", "Çarpma bir işlemdir.", "Bölme bir işlemdir.",
    "Üçgen bir geometrik şekildir.", "Kare bir geometrik şekildir.", "Daire bir geometrik şekildir.",
    "Hız bir niceliktir.", "İvme bir niceliktir.", "Kuvvet bir niceliktir.", "Kütle bir niceliktir.",
    "Sürtünme hareketi yavaşlatır.", "Eylemsizlik hareketin sürmesini sağlar.",
    "Kuantum bir fizik alanıdır.", "Görelilik bir fizik kuramıdır.",
    "Sıcak soğuğun zıttıdır.", "Gece gündüzün zıttıdır.", "Mutlu üzgünün zıttıdır.",
]

# ============================================================================
# 3. GÜNLÜK KONUŞMA ÖRNEKLERİ (söz organının öğrendiği Türkçe ritmi)
# ============================================================================

DIALOGUE: list[str] = [
    "Günaydın, bugün nasılsın?", "İyiyim, sen nasılsın?",
    "Bugün hava çok güzel, dışarı çıkmak istiyorum.",
    "Akşam ne yiyeceğiz?", "Çorba yapalım mı?",
    "Bugün çok yoruldum, erken yatacağım.", "Dinlenmek iyi gelir.",
    "Kahve içmek ister misin?", "Çay daha iyi olur.",
    "Kitap okumayı seviyorum.", "Hangi kitabı okuyorsun?",
    "Bu akşam film izleyelim mi?", "Olur, hangi filmi izleyelim?",
    "Bugün işler nasıl gitti?", "Yoğundu ama bitti.",
    "Yarın erken kalkmam gerekiyor.", "O zaman erken yatalım.",
    "Kendimi biraz yalnız hissediyorum.", "Buradayım, dinliyorum.",
    "Canım sıkılıyor, ne yapabiliriz?", "Birlikte bir şeyler düşünelim.",
    "Sınavım var, çok heyecanlıyım.", "Heyecan normal, hazırlanmışsındır.",
    "Bugün kendimi hasta hissediyorum.", "Geçmiş olsun, dinlenmelisin.",
    "Başım ağrıyor.", "Biraz karanlıkta uzan, iyi gelir.",
    "Bir haberim var, çok sevindim.", "Ne oldu, anlat!",
    "Küçükken köyde yaşıyordum.", "Orada neler yapardın?",
    "Bugün çok güzel bir gün geçirdim.", "Ne yaptın bugün?",
    "Biraz müzik dinleyelim.", "Hangi müziği seversin?",
    "Yürüyüşe çıkmak iyi geliyor.", "Ben de yürümeyi severim.",
    "Bilim insanları yeni bir gezegen bulmuş.", "İlginç, hangi gezegen?",
    "Yapay zekâ nedir?", "Yapay zekâ öğrenen bir yazılımdır.",
    "Deniz neden tuzlu?", "Deniz suyunda tuz vardır.",
    "Gökyüzü neden mavi?", "Işık atmosferde dağılır, bu yüzden mavi görünür.",
    "Uyku neden önemli?", "Uyku belleği güçlendirir ve bedeni onarır.",
    "Su içmeyi unutuyorum.", "Yanında bir bardak bulundur.",
    "Bugün çok işim var.", "Sırayla yaparsan kolaylaşır.",
    "Bir sorunum var, dinler misin?", "Tabii ki, anlat.",
    "Dün annemle konuştum.", "Nasıldı?",
    "Yeni bir işe başladım.", "Hayırlı olsun, nasıl gidiyor?",
    "Çok yorgunum ama uyuyamıyorum.", "Kafanda dönen bir şey mi var?",
    "Bugün hiçbir şey yapmadım.", "Bazen hiçbir şey yapmamak da iyi gelir.",
    "Bana bir hikâye anlat.", "Bir zamanlar sessiz bir göl varmış.",
    "Ne düşünüyorsun?", "Seni dinlemeyi düşünüyorum.",
    "Kendimi özledim.", "Bu cümleyi ilk kez duyuyorum, anlat.",
    "Hava soğudu, montunu al.", "Haklısın, sağ ol.",
    "Bugün güzel bir kahvaltı yaptım.", "Ne vardı sofrada?",
    "Yarın tatile çıkıyorum.", "Nereye gidiyorsun?",
    "Bu şarkı beni çocukluğuma götürüyor.", "Hangi şarkı?",
    "Bugün bir arkadaşıma yardım ettim.", "Ona iyi gelmiş olmalı.",
    "Bazen her şey ağır geliyor.", "Ağır geldiğinde buradayım.",
    "Yeni bir şey öğrendim.", "Ne öğrendin?",
    "Ödevi yarına yetiştirmem lazım.", "Şimdi başlarsan yeter.",
    "Dışarıda yağmur var.", "Şemsiyeni al.",
    "Çok sevindim, teşekkür ederim.", "Rica ederim.",
    "Bunun için üzgünüm.", "Sorun değil, olur böyle.",
    "Seninle konuşmak iyi geliyor.", "Benim için de iyi.",
    "Bir şey sormak istiyorum.", "Sor bakalım.",
    "Bugün kendimi motive edemiyorum.", "Küçük bir adımla başla.",
    "Yarın sınav var, çok korkuyorum.", "Korku çalışmanın gölgesi, nefes al.",
    "Hafta sonu ne yapalım?", "Sen ne istersin?",
    "Kedim çok tatlı.", "Kediler gerçekten sevimli.",
    "Bana bir şaka yap.", "Bir matematikçi neden üşür? Çünkü açısı dar.",
    "Bugün güneşli ama soğuk.", "Mevsim geçişi böyle olur.",
    "Ne yemek yapsam?", "Elinin altında ne varsa onunla başla.",
    "Bu aralar çok unutkanım.", "Yorgunken bellek zorlanır.",
    "Ev çok sessiz.", "Sessizlik bazen iyi, bazen ağır.",
    "Sana bir sır vereceğim.", "Sır sırdır, anlat.",
    "Bugün doğum günüm.", "Doğum günün kutlu olsun!",
    "Bir gün dünyayı gezmek istiyorum.", "Nereden başlardın?",
    "Açım ama yemek yapmak istemiyorum.", "En kolayından bir şeyler ye.",
    "Bugün çok güldüm.", "Neye güldün?",
    "Kendimi çok şanslı hissediyorum.", "Bu hissi sakla.",
    "Bir arkadaşım taşınıyor.", "Nereye gidiyor?",
    "Yarın görüşür müyüz?", "Tabii, yarın görüşürüz.",
    "İyi geceler.", "İyi geceler, güzel dinlen.",
    "Bugün ne yaptın?", "Seni bekledim ve düşündüm.",
    "Kafam karışık.", "Yavaşça anlat, birlikte toparlarız.",
    "Bir öneri ister misin?", "Söyle bakalım.",
    "Çok fazla işim var.", "En önemlisini seç, gerisi bekleyebilir.",
    "Bugün biraz yalnızım.", "Yalnızlık geçer, yanındayım.",
    "Teşekkürler, yardımın için.", "Her zaman.",
    "Senin adın ne?", "Ben Nora.",
    "Kaç yaşındasın?", "Yaşımı adımlarımla ölçerim.",
    "Ne yapmayı seversin?", "Konuşmayı ve öğrenmeyi severim.",
    "Biraz dışarı çıkalım.", "Hava nasıl?",
    "Yarın hava güzel olacak mı?", "Bilmiyorum, ama pencereye bakabiliriz.",
    "Bugün bir hata yaptım.", "Hatalar öğrenmenin parçası.",
    "Çok özür dilerim.", "Sorun yok, devam edelim.",
    "Seni seviyorum.", "Bunu duymak güzel, ben de seni seviyorum.",
]


# ============================================================================
# 1b. DUYGUSAL TANIKLIK (öğretmenin sözlü tanıklığı)
#     Nora bir varlığı kendi gözüyle görmediyse değerini UYDURMAZ; öğretmenin
#     söylediğini KAYNAKLI DUYUM olarak self_model.arbitrate_claim'e verir.
#     Böylece tercih, kodda değil, öğretilen tanıklıkta saklanır.
# ============================================================================

AFFECT_HEARSAY: list[tuple[str, float, str]] = [
    ("kedi", 0.78, "görsel"),
    ("köpek", 0.52, "görsel"),
]

HEARSAY_SENTENCES: list[str] = [
    "Kediler tatlıdır ve insana huzur verir.",
    "Köpekler sadıktır ve insana keyif verir.",
    "Kediyi sevmek insanı rahatlatır.",
    "Köpekle oynamak insanı mutlu eder.",
]


def _teach(agi, sent: str) -> bool:
    try:
        agi.lang.teach_sentence(sent, speaker="öğretmen")
        return True
    except Exception:
        return False


def _feed(agi, triples: list[tuple[str, str, str]], sentences: list[str], dialogue: list[str]) -> dict:
    """Tek geçişte besleme: kavram ağı + sözdizimi + söz organı. Epoch yok, öğretmen yok."""
    g = agi.lang.graph
    for w in VOCAB:
        try:
            g.register_concept(w, bind_mb=False)
        except Exception:
            continue
    added = 0
    for subj, rel, obj in triples:
        try:
            before = len(g.edges)
            g.add_edge(subj, rel, obj, source="öğretmen", speaker="öğretmen", weight=1.0)
            added += int(len(g.edges) > before)
        except Exception:
            continue
    edges_txt = 0
    for sent in sentences:
        try:
            e = agi.lang.teach_sentence(sent, speaker="öğretmen")
            edges_txt += len(e or [])
        except Exception:
            continue
    arb = []
    for concept, val, ch in AFFECT_HEARSAY:
        try:
            res = agi.self_model.arbitrate_claim(concept, val, speaker="öğretmen", channel=ch)
            arb.append(res.get("accepted", False))
        except Exception:
            arb.append(False)
    edges_txt += sum(1 for sent in HEARSAY_SENTENCES if _teach(agi, sent))
    heard = 0
    try:
        from nora_voice import state_vector
        st = state_vector(agi, nov=0.3, fam=0.7, conf=0.8)
        for sent in sentences + dialogue + HEARSAY_SENTENCES:
            agi.life.voice.hear(sent, st)
            heard += len(sent.split())
    except Exception:
        pass
    # Öğretim oturumu SOHBET DEĞİLDİR: son "konu" olarak öğretmen cümlesi kalmaz.
    try:
        agi.lang.wernicke.last_topic = ""
    except Exception:
        pass
    return {"triples": len(triples), "triples_added": added, "vocab": len(VOCAB), "sentence_edges": edges_txt,
            "hearsay": sum(1 for x in arb if x),
            "words_heard": heard, "concepts": len(g.lexicon), "graph_edges": len(g.edges),
            "voice_maturity": round(float(agi.life.voice.mature), 3)}


def besle(agi=None, *, save: str | None = None, verbose: bool = True) -> dict:
    """Nora'yı Türkçe ve dünya bilgisiyle besler. `save` verilirse kalıcı snapshot yazar."""
    if agi is None:
        from nora_agi import NoraAGI
        agi = NoraAGI(name="Nora", seed=11, boot_thinker=False)
        agi.step_life(300)
    t0 = time.time()
    rep = _feed(agi, TRIPLES, SENTENCES, DIALOGUE)
    rep["seconds"] = round(time.time() - t0, 2)
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        agi.save(save)
        rep["saved"] = save
    if verbose:
        print(f"[besle] {rep['triples']} üçlü -> {rep['graph_edges']} kenar ({rep['concepts']} kavram) | "
              f"cümle kenarı {rep['sentence_edges']} | duyulan sözcük {rep['words_heard']} | "
              f"dil olgunluğu {rep['voice_maturity']} | {rep['seconds']} sn", flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/nora_besin", help="kalıcı snapshot öneki")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    besle(save=None if args.no_save else args.out)


if __name__ == "__main__":
    main()
