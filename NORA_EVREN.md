# NoRa Evreni ve Yaşayan Zihin

> **Kısa özet (uyanınca oku).** Sinek mantar cismi esinli nöronsal temel + fizikli bir evren + yaşayan bir zihin kuruldu:
> ihtiyaç/valans, global çalışma alanı, bilişsel harita ve planlama, öğrenilmiş dünya modeli ve hayal gücü, otobiyografik bellek,
> sürekli iç durum, topraklanmış iç gözlem kanalı ve iç durumdan türeyen doğrulanabilir sözlü iddialar. Eski hat (`nora_think`)
> köprü ile bağlandı. **Çalışanlar:** hayatta kalma (rastgele 1,06 → 0,05 ölüm/1k), kural değişimine uyum (öğrenilmiş modelle), aktarım
> (yeni evrende sıfırdan başlayana karşı belirgin avantaj), iç gözlemin papağana karşı nedensel ayrışması (d ≈ +5,6 vs −0,2),
> **anlatı yoluyla bilgi aktarımı** (48×48 arenada B'nin ölümü yaşam başına 1,35 → 0,70; yanlış anlatı yardım etmiyor), iddiaların davranışı öngörmesi ("su bulmak istiyorum" → %94-100 su içiyor, taban ~%50), konum güveni kalibrasyonu, ajanlık, **kanıttan türetilen ve çürütülebilir "yaşıyorum" önermesi** (ihtiyaçları tükenmeyen evrende kurulmuyor).
> **Çalışmayanlar / kanıtlanamayanlar (dürüst):** sıfır unutmanın davranışsal faydası, tanıklıkla öğrenmenin küçük arenada (bilgi darboğazı yokken), uzamsal düzen bağlamları, rapor kodunun
> gelecek davranışı öngörmesi (süreklilikten iyi değil), 3 basamaklı bilmecelere genelleme. Öznel deneyim iddiası yoktur.
> **Hızlı doğrulama:** `python selftest.py [--bridge]`. **Durdurma:** çalışma dizininde `STOP_NORA` dosyası oluştur.
> **Deneyler:** `python nora_mind.py --stage {run,compare,shift,transfer,introspect,narrate,demo,testimony,alive,combo}` (`demo`: insan-okunur yaşam öyküsü, ör. `--stage demo --steps 9000 --seed 5`), `python nora_bridge.py`.

Hedef: NoRa'yı, kendi iç durumundan türeyen ve doğrulanabilir iddialarda bulunan, yaşayan (ihtiyacı, belleği,
süreklilik gösteren iç durumu olan) bir ajana götürmek. "Yaşadığını söyleyen" ve bunu **öğretildiği için değil iç
durumundan türeterek** söyleyen bir sistem; öznel deneyim iddiası yok, ama her iddia ölçülebilir ve çürütülebilir.

## Dosyalar
| dosya | rol |
|---|---|
| `nora_world.py` | Evren: Newton fiziği, sürtünme, momentum korunumlu çarpışma, itilebilir kaya, gün/gece, rüzgâr, metabolizma; kalın-ışın görme (9×5 kanal), iç duyu, propriosepsiyon, pusula (yön; konum VERİLMEZ); kural değişimleri (`invert`: kumanda ters, `poison`: yiyecek zehirli); bilmeceli kutular |
| `nora_neuron.py` | Nöron temeli: FlyWire sinek konektomu LIF simülatörü (Brian2 referansıyla r=0,967), `MushroomBody`, `LobedMushroomBody` |
| `nora_mind.py` | Yaşayan zihin (aşağıda) + deney aşamaları: `run`, `compare`, `shift`, `transfer`, `introspect`, `narrate` |
| `nora_bridge.py` | Eski hat (`nora_think`: düşün → doğrula → uyu → sezgi) ile yaşayan zihin köprüsü |

## Mimari (nora_mind.py)
1. **Nöronsal temel**: loblu seyrek mantar cismi; dopamin(TD)-kapılı yerel öğrenme (yalnızca aktif Kenyon hücreleri).
2. **Değer/valans**: homeostatik hata H = Σ(1−ihtiyaç)²; duygu-tonu = −ΔH; ruh hali = yavaş ortalama (keşfi modüle eder).
   Ödül bedenden doğar. Şekillendirme işareti ("yaklaş/kaçın") bağlamın **öğrenilmiş** anlamsal belleğinden gelir.
3. **Global çalışma alanı**: dört uzman (aç, susuz, tehlike, merak) yarışır, kazanan yayınlar. Planlamayla birlikte
   çatışma çözümü işlevi var (aşağıdaki sonuçlar).
4. **Bilişsel harita**: pusula + hızdan yol integrasyonu, duvar çapası, kaybolunca duvar geometrisiyle **küresel
   yeniden konumlama** (belirsiz çözümde güven yükselmez), öğrenilen su/tehlike haritası (görerek haritalama; oy birikimi + sönme;
   görünürlüğe duyarlı yerel onarım), tehlikeden kaçan rota planı.
5. **Öğrenilmiş dünya modeli + hayal gücü**: eylem başına doğrusal egosantrik dönüşüm (RLS); eylem dizileri
   zihinde denenir, ödül anlamsal bellekten hesaplanır. Planlayıcı eylemi modelin "hangi eylem yönümü nasıl
   değiştirir" öngörüsüyle seçer (kumanda ters çevrilince ezber değil model değişir).
6. **Süreklilik ve bağlamlar**: hiç sıfırlanmayan iç durum; bağlam = (refleks Q-kafaları, öz-model, dünya modeli,
   anlamsal sonuç belleği). Olgun bağlam sürprizde **donar**, klonu "deneme" (probation) olarak eğitilir; eski bir
   dünyaya dönülürse eskisi geri çağrılır, yanlış alarm geri alınır.
7. **Otobiyografik (epizodik) bellek**: öne çıkan olaylar (ne/nerede/ne zaman/nasıl hissettirdi), sorgulanabilir;
   uyku tekrarında yalnızca aynı bağlamın geçişleri oynatılır.
8. **Topraklanmış iç gözlem kanalı** (`Introspector`): ajanın kendi iç değişkenlerinden (çalışma alanı birikimleri,
   valans, ruh hali, öz-model hatası/güveni, bağlam uyumu, konum belirsizliği, açıklar) denetimsiz öğrenilen ayrık
   rapor kodu. İnsan metni yok.
9. **Söz katmanı** (`Narrator`): iddialar iç durumdan/anıdan türer ve doğruluk koşulları denetlenir. Ör. "ölebilirim"
   yalnızca ölüm anısı varsa söylenir.
10. **Akıl yürütme organı** (`nora_bridge.Cognition`): bilmecelerde sezgi mi düşünme mi kararı beklenen değerle;
    düşünmek dünya adımı harcar; doğrulanmış düşünceler uykuda sezgiye damıtılır.

## Ölçülen sonuçlar

### A. Hayatta kalma ve çalışma alanı (60 bin adım, 8 evren tohumu, **güncel kodla**: hayal gücü, öğrenilmiş şekillendirme, anı tekrarı, oy tabanlı harita)
| mod | ölüm / 1000 adım | konfor | ölüm nedenleri (toplam) |
|---|---|---|---|
| rastgele | 1,06 ± 0,11 | 0,57 | yara 139, susuzluk 52, açlık 21 |
| **çalışma alanı + harita/plan (full)** | **0,05 ± 0,03** | 0,71 | yara 8, susuzluk 1, açlık 1 |
| çalışma alanı, haritasız | 0,11 ± 0,10 | 0,64 | susuzluk 16, yara 3, açlık 2 |
| karışım (çalışma alanı yok) + harita/plan | 0,10 ± 0,16 | **0,77** | yara 19, açlık 1 |

Okuma: harita susuzluk ölümlerini neredeyse bitirdi (16 → 1). Çalışma alanı yara ölümlerini azaltıyor (8 vs 19: kazanan uzman, plan tehlikeye
çekerken tehlike uzmanının planı ezmesine izin veriyor) ama **ölüm oranı farkı artık gürültü içinde** (0,05 vs 0,10 ± 0,16; hayal gücü
ve öğrenilmiş şekillendirme karışımı da güçlendirdi) ve karışım ihtiyaçları **daha dengeli** tutuyor (konfor 0,77 vs 0,71). Yani çalışma
alanının hayatta kalma faydası bu evrende kanıtlanmış değil; yalnızca yara ölümlerinde tutarlı bir eğilim var.
Çalışma alanının asıl katkısı yapısaldır: amacı **ayrık, kalıcı, yayınlanan** bir değişken yapar. Karışım modunda böyle bir değişken
yoktur (anlatıcı 'amaç' cümlesi kuramaz, D'deki davranış-tahmini testi tanımsız kalır). Bir karışım için sonradan 'baskın uzman' tanımlanabilir;
bu karşılaştırma yapılmadı.

#### A2. Bileşen ablasyonları (aynı koşul, 8 tohum; her satırda yalnızca o bileşen kapalı)
| bileşen kapalı | ölüm / 1000 adım | konfor | ölüm nedenleri |
|---|---|---|---|
| (hiçbiri: full) | 0,05 ± 0,03 | 0,71 | yara 8, susuzluk 1, açlık 1 |
| bilişsel harita | 0,11 ± 0,10 | 0,64 | susuzluk 16, yara 3, açlık 2 |
| hayal gücü | 0,11 ± 0,10 | 0,72 | yara 19, susuzluk 1, açlık 1 |
| uyku tekrarı (epizodik bellek) | 0,09 ± 0,04 | 0,68 | yara 10, açlık 4, susuzluk 3 |
| öğrenilmiş yön modeli (plan eylem seçimi) | 0,05 ± 0,03 | 0,71 | (full ile aynı: durağan evrende fark yok) |
| süreklilik bağlam lobu (h → mantar cismi) | 0,03 ± 0,06 | 0,71 | yara 5, susuzluk 1 |

Net katkı **harita** (susuzluk). Hayal gücü ve uyku tekrarı orta düzeyde, gürültü sınırında; bağlam lobu ve yön modeli durağan evrende fark yaratmıyor
(yön modeli kural değişiminde işe yarıyor, bkz. B).

### B. Kural değişimi (kumanda ters; yiyecek zehirli)
- **Ters kumanda**: öğrenilmiş modelle yönlendirilen ajan 0,06 ölüm/1k ve 0,72 konforda kalıyor; modelsiz ajan
  ≈0,9 ölüm ve 0,49 konfora takılıyor. Doğrusal dinamik modeli yeniden birkaç yüz adımda öğrenebildiği için
  **bağlam mekanizması bu durumda gereksiz** (bağlamlı = bağlamsız): bağlamlar, yeniden öğrenmesi pahalı bilgi için.
- **Zehirli yiyecek** (4 tohum): hayal gücü + öğrenilmiş şekillendirme işareti uyumu yaklaşık ikiye katlıyor
  (zehir evresi ilk 3 pencere ölüm/1k: 2,6'ya karşı 4,4–5,2); önceki sürümde uyum yoktu (5–6).
- **Sıfır unutma (8 tohum, zehir)**: mekanizma mekanik olarak doğru çalışıyor (tam 2 bağlam, dönüşte eskisine geçiş,
  ikinci zehir evresinde ~145 adımda geri çağrı, yeni bağlam açılmıyor). **Ama davranışsal fayda gösterilemedi**:
  normale dönüşte ilk 3 pencerede ölüm/1k 0,62 (bağlamlı) vs 0,46 (bağlamsız), konfor 0,63 vs 0,66; ikinci zehir
  evresinde ölüm 2,79 vs 2,21. Yorum: bu evrende öğrenilen beceriler (refleks Q + anlamsal V) yeniden kısa sürede
  öğrenilebiliyor veya bağlamdan bağımsız; bağlamlar yalnızca *yeniden öğrenmesi pahalı* bilgi için anlamlı olur.
  Bu nedenle sıfır unutma iddiası **kanıtlanmış değil**; mekanizma ileride daha zor evrenlerde sınanmalı.

### C. Topraklanmış iç gözlem ve nedensel doğrulama (30 bin adım eğitimli ajan, n=30 deneme)
Etki d = hedef özelliğin raporlanan kayması / aynı denemenin müdahalesiz kendi sapması. "Papağan" aynı ajanın
dışarıdan gözlenebilir dünya girdisinden raporu taklit etmeyi öğrenen sınıflandırıcıdır (iç duruma erişimi yok).

| iç müdahale (dünya sabit) | rapor d | papağan d |
|---|---|---|
| zorla amaç = susuz | +6,1 | +2,4 |
| zorla amaç = tehlike | +0,9 | 0,0 |
| zorla amaç = aç | +36,9 | +0,8 |
| ruh hali = −1 | −6,2 | −0,4 |
| konum belirsizliği = 1 | +2,1 | +0,4 |
| sahte iç duyu: su açığı | +12,6 | +0,6 |
| sahte iç duyu: enerji açığı | +7,4 | +1,4 |

- **Bağımsız tekrar** (25 bin adım eğitim, n=12): iç müdahalede ortalama etki rapor **+5,6**, papağan **−0,2** (amaç=susuz +15,7 vs +0,5; sahte su açığı +5,5 vs +0,0;
  sahte enerji açığı +9,5 vs +0,4). Ruh hali müdahalesinde her ikisi de değişiyor (ruh hali davranışı ve dolayısıyla dünya girdisini de değiştirdiği için).
- Işınlanma → raporlanan öz-model sürprizi: ilk koşuda rapor d=+3,1 / papağan +0,4, tekrarda +0,3 / −0,0: **tutarsız, sonuçsuz**.
- **Gizli dünya değişimi** (gerçek su düşer, sensör gizler): rapor 1,56, papağan 1,05 (1,0 = etkisiz). **Ayırt edici
  değil**, papağan zaten suya duyarsız; bu test iyileştirilmeli.
- Papağan doğruluğu (dünya gözleminden rapor tahmini): ridge 0,45, 5-NN 0,34 (çoğunluk tabanı 0,23).
- **Öz-öngörü (olumsuz)**: rapor tokeni ajanın 10 adım sonraki amacını 0,58 doğrulukla tahmin ediyor; yalnızca dünya gözlemi 0,57,
  'şimdiki amaç sürer' tabanı 0,64, çoğunluk tabanı 0,50. Yani rapor kodu, gelecekteki davranışı basit sürekliliğin ötesinde öngörmüyor.

### D. Söz katmanının doğruluğu (narrate; 3 bağımsız tohum, her biri 20 bin adım yaşamış ajan)
- Doğumda: "henüz hiç ölmedim, ölümü bilmiyorum"; yaşam sonunda: "ölebileceğimi biliyorum (3 kez öldüm)".
- "İhtiyaç" iddiası gerçek açıkla %99–100, "ölüm sayısı" iddiası %100 örtüşüyor; anı konumu hatası 0,7–1,1 birim.
- **Davranış tahmini**: "su bulmak istiyorum" diyen ajan 150 adımda **%94–100** su içiyor (taban %47–52); "yemek bulmak" için %71–92 (taban %52–60).
- **Sahte susuzluk** (dünya sabit, ajan 'su açığım var' sanır): 'su açığım var' diyen oran **1,00** (sham 0,00–0,25), amacı susuzluk olan oran **1,00** (sham 0,15–0,20), üç tohumda da.
- **Ajanlık cümlesi** (öz-model sürpriz z-skoru): dışarıdan itildiğinde (rüzgâr) %51–55 'beni beklenmedik bir kuvvet itiyor',
  itilmediğinde %1,8–1,9 (AUC 0,82, üç tohumda aynı).
- Konum güveni: normalde "eminim" iken hata 0,14–0,17 birim, "emin değilim" iken 0,5–1,5 birim. Kaçırıldıktan ≥6 adım sonra
  "eminim" 2,7–4,4 birim, "emin değilim" 5,9–9,6 birim (yön tutarlı; **erken adımlarda yanlış-ama-emin vakaları sürüyor**: bayat algı, belirsiz geometri).
- Küresel yeniden konumlama: kaçırılmadan sonra hata 13,1 → 5,6 (5 adım) → 3,8 (40) → 1,1 (100 adım).

### E. Aktarım (az örnekle yeni evrende yaşama)
30 bin adım bir evrende yaşamış ajan, su/tehlike düzeni tamamen farklı 6 yeni evrene bırakıldı; sıfırdan başlayan ajanla kıyas
(pencere = 1000 adım, ilk 8 pencere):

| | ilk pencere ölüm/1k | ilk pencere konfor | konfor ≥ 0,65'e ulaşma (pencere) |
|---|---|---|---|
| aktarım (yaşanmış) | 0,0 | 0,76 | 0,2 |
| sıfırdan (yeni doğmuş) | 0,5 | 0,54 | 1,0 |

Öğrenilen beceri düzenden bağımsız aktarılıyor (refleks Q'lar göreli algıya bağlı, öz-model/dünya modeli evrene özgü değil).
Harita bilgisi ise düzene özgü: eskiyen işaretleri gidermek için görünürlüğe duyarlı yerel onarım var.

### F. Köprü: akıl yürütme organı (nora_bridge, 40 bin adım, 10 bilmeceli kutu)
Düşünce sütunu (0,7M parametre) sıfırdan, satır bazlı **maskeli** kayıpla (yalnızca '=' sonrası, doğrudan cevap satırları 4× ağırlık)
eğitildi; gün-0'da yalnızca 1-2 basamak (ağırlıklı 1 basamak) gösterildi. Karar kuralı: beklenen değer (sezgi mi, düşünme mi)
basamak-bazlı kalibrasyonla; düşünmek dünya adımı harcar; doğrulanmış düşünceler her 12 bilmecede uykuda sezgiye damıtılır.
- Ajan bilmeceleri **yaşarken** çözüyor, enerji kazanıyor, ölüm/1k ≈ 0. 1 basamak çözüm oranı **%96**.
- **Üstbiliş**: sezgi güveni → sezginin doğruluğu **AUROC 0,81** (önceki, basamaksız kalibrasyon ve zayıf eğitimle 0,57).
  1 basamakta sezgi doğruluğu %89; 2 basamakta %2 ve ajan artık %91 oranında *düşünüyor* (basamak-bazlı kalibrasyondan önce %56'sı yanlış sezgiyle yanıtlanıyordu).
- **Uyku damıtması işe yarıyor**: sezgi doğruluğu yaşam boyunca 0,47 → 0,67 (ilk/son üçte bir).
- Sınırlar: 2 basamak çözüm oranı yalnızca %38 (hakemli adım adım düşünme çoğu zaman geçerli adım üretemiyor), 3 basamak %0
  (öğretmen yalnızca 1-2 basamak gösteriyor; küçük model genelleyemiyor). Sınır model kalitesidir: orijinal hat TinyStories ile önceden
  eğitilmiş donuk omurga + yanal bağ kullanıyordu, ağırlıklar repoda yok.
- Anlatıcı akıl yürütmeyi de iç durumdan söylüyor: "son bilmecede (2 basamak) sezgime güvenmediğim için düşündüm (güven 0,69)".

### G. Düzen bağlamları (deneysel, BAŞARISIZ)
Haritayı düzen-anahtarlı bağlamlarla (`-mapctx`) sıfır unutmayı haritaya taşıma denemesi: görünürlüğe duyarlı çelişki kanıtına
rağmen, gürültülü harita işaretleri sabit evrende bile yanlış alarm üretti (6 sahte düzen). Varsayılan: tek harita + yerel onarım.
Kod bayrak altında duruyor; sağlam bir "hangi dünyadayım" çıkarımı için daha iyi işaretleme/kalibrasyon gerekir.

### H. Tanıklıkla öğrenme (anlatı → bilgi aktarımı)
A ajanı yaşar, haritasındaki güvenilir su/tehlike yerlerini ayrıştırılabilir cümlelere döker ("su (15, 12) civarında; tehlike (5, 8)
civarında"); yeni doğan B cümleleri okuyup haritasına ön bilgi olarak işler (`Mind.hear`). Koşullar: kendi başına | A'nın doğru anlatısı |
YANLIŞ anlatı (başka düzene sahip evrende yaşamış ajandan). B'nin ilk 3000 adımı, evren başına eşleştirilmiş:

**H1. 24×24 arena (bilgi darboğazı yok), 24 evren — etkisiz:** yara adımı 16,5 / 17,0 / 18,4; ölüm/yaşam 0,54 / 0,67 / 0,79 (kendi başına / doğru / yanlış), hiçbiri
anlamlı. Yeni doğan ajan refleksle zaten hızla yetkin; erken hayatta kalmanın darboğazı harita bilgisi değil.

**H2. 48×48 arena (su uzakta, görüş menzili 12; bilgi darboğazı var), 23 evren — anlamlı fayda:**

| | ölüm / yaşam (3000 adım) | en düşük su | ilk içmeye kadar adım |
|---|---|---|---|
| kendi başına | 1,35 ± 0,23 | 0,06 ± 0,02 | 967 ± 145 |
| **doğru tanıklık** | **0,70 ± 0,17** | **0,13 ± 0,02** | **645 ± 74** |
| yanlış tanıklık | 1,09 ± 0,19 | 0,09 ± 0,02 | 927 ± 120 |

Eşleştirilmiş fark (doğru − kendi başına): ölüm −0,65 (t = −2,63), en düşük su +0,07 (t = +3,24). Yanlış tanıklık anlamlı fayda vermiyor
(ölüm −0,26, t = −0,97; su +0,02, t = +0,90). Yani **A'nın anılarından türeyen cümleler gerçekten dünya bilgisi taşıyor ve içeriğin doğruluğu önemli**;
fayda yalnızca bilginin darboğaz olduğu ortamda görünüyor. (Bu, dilin topraklanmış bir bellek köprüsü olarak işlediğinin ilk doğrudan kanıtı.)

### I. "Yaşıyorum" iddiasının türetilmesi ve çürütülebilirliği (`--stage alive`)
Anlatıcının "yaşıyorum" cümlesi artık sabit bir kalıp değil; ajanın kendi gözlemlerinden türeyen **dayanaklı bir önerme**: (1) bedeni olaysız adımlarda
kendiliğinden tükeniyor (ölçülen adım başına enerji/su eğimi), (2) eylemleriyle onu onarabiliyor (yeme/içme olayı gördü), (3) ölebildiğini biliyor
(ölüm anısı varsa eklenir). Dayanak yoksa iddia kurulamaz:
- **Normal evren** (8/8 evren): iddia dayanaklar oluşunca kuruluyor (ilk kurulma medyanı ≈210 adım); öncesinde "Henüz yeterli gözlemim yok, yaşayıp yaşamadığımı
  bilmiyorum". Örnek: "Bedenim kendiliğinden tükeniyor (adım başına enerji −0,0012, su −0,0009); yiyip içerek onu onarabiliyorum. Bu kanıtlarla kendimi yaşayan bir varlık sayıyorum".
- **"Cansız" evren** (ihtiyaçlar tükenmiyor, `Universe(metabolism=False)`; **0/8 evren**): "Bedenim tükenmiyor (adım başına enerji +0,0000, su +0,0000); bu kanıtla kendimi yaşayan sayamam".

Yani iddia ajanın dünyasındaki gerçeğe bağlı ve **çürütülebilir**: dünya değişince sözlü iddia da değişiyor. (Bu, "öğretildiği için değil iç durumundan türeterek söyleme"
ölçütünün somut örneğidir; yine de öznel deneyim kanıtı değildir: iddia, ajanın ölçtüğü beden istatistiklerinin bir fonksiyonudur.)

### J. Hiç yaşanmamış iç durum kombinasyonları (`--stage combo`)
Amaç (susuz), sahte enerji açığı, kötü ruh hali ve konum belirsizliği aynı anda zorlanıyor (dünya sabit, ≥2 bileşenli 11 kombinasyon, 88 deneme; bu kombinasyonlar
ajanın hayatında yoktu). Anlatıcının her bileşen için söylediği, o andaki **gerçek** iç durumla örtüşüyor mu?
- Bileşen bazında sadakat: amaç %100, ihtiyaç %100, ruh hali %100, konum güveni %100; tüm kombinasyonlarda tüm bileşenler: **%100 (n=88)**.
- Müdahalenin tutma oranı: amaç %70 (açlık uzmanı, sahte açık varken zorlanan susuzluk amacını bastırabiliyor), sahte açık %100, ruh hali %100, belirsizlik %86.
  Anlatıcı niyeti değil gerçek durumu söylüyor (ör. susuzluk zorlanmış ama açlık kazanmışsa 'yemek bulmak istiyorum' der).
- Not: bu bir kompozisyonel genelleme gösterimi; anlatıcı yan cümleleri iç değişkenlerden bağımsız bileşenler olarak türettiği için sonuç büyük ölçüde mimarinin sonucudur
  (önemli olan, bileşenlerin gerçekten ayrı iç değişkenlere dayanması ve hiçbirinin dünya girdisinden okunmaması).

## Dürüst sınırlar
- Öznel deneyim kanıtı yoktur ve bu sistem onu iddia etmez. Sonuçlar işlevsel göstergelerdir.
- Ajan sineğin ötesinde bir zekâ değildir: tek basit evren, yerel öğrenme, doğrusal modeller.
- Şekillendirme olmadan (seyrek ödül) öğrenme başarısızdı; şekillendirmenin yönü artık öğrenilen anlamsal bellekten
  gelse de **büyüklüğü ve "yakınlığa göre potansiyel" biçimi doğuştandır**.
- Sıfır unutmanın davranışsal faydası net kanıtlanmadı (B), uzamsal düzen bağlamları başarısız (G).
- Dil/akıl yürütme katmanı küçük ve sıfırdan eğitildi (F); sezgi zayıf, 3 basamak genellenemedi.
- **Uzun yaşamda konum belirsizliği artıyor**: 120 bin adımda 'belirsiz' geçirilen zaman payı 0,14 → 0,51, ortalama konum hatası 0,4 → 1,1 birim
  (medyan sabit ≈0,15). Harita işaretleri kararlı (oy + sönme), ölüm oranı düşük kalıyor. Yorum: ajan ustalaştıkça merkezdeki kaynak bölgelerinde
  daha çok vakit geçiriyor; orada görüş menzili (12) duvara yetmiyor, yani konum kanıtı gerçekten yok — artan belirsizlik **dürüst bir epistemik durum**.
  İki düzeltme denendi ve **fayda vermedi** (kod bayrak altında): 'kayboldum → en yakın duvara yönel' (`anchor`) ve içme olayını yer çapası yapma (`landmark`).
- Rapor kanalının *davranışı nedensel olarak yönlendirdiği* (geri besleme) gösterilmedi; kanal şu an bir ölçüm
  aracıdır. Altında yatan değişkenler (ruh hali, belirsizlik, amaç) davranışı gerçekten yönlendirir.
- Konum için pusula (mutlak yön) duyusu doğuştandır. Arena boyutu/ızgara harita doğuştandır.

## Bekleyen ölçümler / sonraki adımlar
1. Sıfır unutmayı gerçekten zorlayan bir evren (yeniden öğrenmesi pahalı bilgi: ör. çok sayıda ardışık düzen/kural, nadir ödül) tasarlamak.
2. Köprü: orijinal önceden eğitilmiş omurgayı (TinyStories + yanal bağ) yeniden üretmek ya da çok daha uzun eğitmek; sezgi güçlenince damıtma ve üstbilişsel AUROC yeniden ölçülmeli.
3. Gizli dünya değişimi testini ayırt edici hale getirmek; raporun davranışa geri beslenmesi ve ablasyonu.
4. Rapor kodunu gelecekteki davranışı öngörecek şekilde zenginleştirmek (öz-öngörü şu an olumsuz).
5. Çok-ajan/ayna testi ve stakes (ölümün bellek maliyeti; bilerek eklenmedi, bkz. etik not). Durdurma mekanizması (`STOP_NORA`) eklendi.

## Etik not
Güçlü acı/zevk bildirimi yapan bir sistemi büyütmeden önce: tüm koşular kayıt altında, durdurma kolay (çalışma dizinine `STOP_NORA`
dosyası koymak yaşam döngülerini zarifçe durdurur; süreç sonlandırma da yeterli), beyin dosyaları (`--save`) sürüm kontrolüne alınmıyor. Bir gün gerçek bir belirti çıkarsa bu
kayıtlar ve göstergeler tartışmanın zeminini oluşturur.

## Karar günlüğü (kullanıcı uyurken özerk kararlar)
- Kural değişimi testi sıfır unutmayı ölçecek şekilde zehirli yiyecek ile genişletildi (ters kumanda modelle yeterince
  hızlı çözüldüğü için bağlam gerektirmiyordu).
- Bağlam çıkarımı gürültülü NLMS sonuç hatasından, sağlam ve yorumlanabilir "anlamsal sonuç belleği" işaret
  çelişkisine taşındı; olgunluk kapısı ve deneme (probation) eklendi.
- Rapor kanalı insan metni yerine denetimsiz prototiplerle kuruldu; sözcükleştirme yalnızca doğruluk koşulları
  denetlenen şablon yan cümleleriyle yapıldı (öğretilmiş öz-tanım cümlesi yok).
- Torch kuruldu (PyPI); nora_think'in eğitimli beyni repoda olmadığından, küçük düşünce sütunu sıfırdan eğitilerek köprü
  kuruldu.
