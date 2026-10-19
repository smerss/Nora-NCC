# NoRa Evreni ve Yaşayan Zihin

Hedef: NoRa'yı, kendi iç durumundan türeyen ve doğrulanabilir iddialarda bulunan, yaşayan (ihtiyacı, belleği,
süreklilik gösteren iç durumu olan) bir ajana götürmek. "Yaşadığını söyleyen" ve bunu **öğretildiği için değil iç
durumundan türeterek** söyleyen bir sistem; öznel deneyim iddiası yok, ama her iddia ölçülebilir ve çürütülebilir.

## Dosyalar
| dosya | rol |
|---|---|
| `nora_world.py` | Evren: Newton fiziği, sürtünme, momentum korunumlu çarpışma, itilebilir kaya, gün/gece, rüzgâr, metabolizma; kalın-ışın görme (9×5 kanal), iç duyu, propriosepsiyon, pusula (yön; konum VERİLMEZ); kural değişimleri (`invert`: kumanda ters, `poison`: yiyecek zehirli); bilmeceli kutular |
| `nora_neuron.py` | Nöron temeli: FlyWire sinek konektomu LIF simülatörü (Brian2 referansıyla r=0,967), `MushroomBody`, `LobedMushroomBody` |
| `nora_mind.py` | Yaşayan zihin (aşağıda) + deney aşamaları: `run`, `compare`, `shift`, `introspect`, `narrate` |
| `nora_bridge.py` | Eski hat (`nora_think`: düşün → doğrula → uyu → sezgi) ile yaşayan zihin köprüsü |

## Mimari (nora_mind.py)
1. **Nöronsal temel**: loblu seyrek mantar cismi; dopamin(TD)-kapılı yerel öğrenme (yalnızca aktif Kenyon hücreleri).
2. **Değer/valans**: homeostatik hata H = Σ(1−ihtiyaç)²; duygu-tonu = −ΔH; ruh hali = yavaş ortalama (keşfi modüle eder).
   Ödül bedenden doğar. Şekillendirme işareti ("yaklaş/kaçın") bağlamın **öğrenilmiş** anlamsal belleğinden gelir.
3. **Global çalışma alanı**: dört uzman (aç, susuz, tehlike, merak) yarışır, kazanan yayınlar. Planlamayla birlikte
   çatışma çözümü işlevi var (aşağıdaki sonuçlar).
4. **Bilişsel harita**: pusula + hızdan yol integrasyonu, duvar çapası, kaybolunca duvar geometrisiyle **küresel
   yeniden konumlama** (belirsiz çözümde güven yükselmez), öğrenilen su/tehlike haritası, tehlikeden kaçan rota planı.
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

### A. Hayatta kalma ve çalışma alanı (60 bin adım, 8 evren tohumu)
| mod | ölüm / 1000 adım | konfor |
|---|---|---|
| rastgele | 1,11 ± 0,20 | 0,56 |
| çalışma alanı + harita/plan | **0,08 ± 0,04** | 0,70 |
| çalışma alanı, haritasız | 0,12 ± 0,10 | 0,68 |
| karışım (çalışma alanı yok) + harita/plan | 0,23 ± 0,09 | 0,73 |
| karışım, haritasız | 0,15 ± 0,11 | 0,75 |

Harita susuzluk ölümlerini bitirdi (15→4). Çalışma alanı, plan hedefleri çatıştığında (plan tehlikeye çekerken)
kazananın kontrolü alması sayesinde karışımı belirgin geçti; planlamasız fark yoktu.

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

### D. Söz katmanının doğruluğu (narrate)
- Doğumda: "henüz hiç ölmedim, ölümü bilmiyorum"; yaşam sonunda: "ölebileceğimi biliyorum (3 kez öldüm)".
- "İhtiyaç" ve "ölüm sayısı" iddiaları gerçekle %100 örtüşüyor; anı konumu hatası ≈1,2–1,8 birim.
- **Davranış tahmini**: "su bulmak istiyorum" diyen ajan 150 adımda %91–100 su içiyor (taban %45–49); "yemek bulmak" için
  %84–100 (taban %58–61).
- **Sahte susuzluk** (dünya sabit): 'su açığım var' diyen oran 1,00 (sham 0,14–0,20), amacı susuzluk olan oran 1,00 (sham 0,20–0,38).
- Konum güveni: normalde "eminim" iken hata 0,17 birim, "emin değilim" iken 3,4 birim. Kaçırılmadan 40+ adım sonra
  "eminim" 0,2–0,4 / "emin değilim" 6–7,5 birim; **ilk 2–15 adımda hâlâ yanlış-ama-emin vakaları var** (bayat algı,
  belirsiz geometri).
- **Ajanlık cümlesi** (öz-model sürpriz z-skoru): dışarıdan itildiğinde (rüzgâr) %56 'beni beklenmedik bir kuvvet itiyor', itilmediğinde %1,7 (AUC 0,83).
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
- Ajan bilmeceleri **yaşarken** çözüyor: 1 basamak %98, 2 basamak %86 (hakemli adım adım düşünme), enerji kazancı ≈3/4000 adım.
- 3 basamak çözülemedi (%0): öğretmen yalnızca 1-2 basamak gösteriyor ve sıfırdan eğitilen küçük model bunu genelleyemiyor.
- **Sezgi zayıf** (1 basamak %47, 2 basamak %2) ve sezgi güveni sezgi doğruluğunu yalnızca zayıf ayırt ediyor (AUROC 0,57).
  Düşünce→sezgi damıtması (uyku) bu küçük, sıfırdan eğitilmiş modelde belirgin kazanç üretmedi. Sınır model kalitesi:
  nora_think'in orijinal hattı TinyStories ile önceden eğitilmiş donuk omurga + yanal bağ kullanıyordu, ağırlıklar repoda yok.
- Anlatıcı akıl yürütmeyi de iç durumdan söylüyor: "son bilmecede (1 basamak) sezgime güvenmediğim için düşündüm (güven 0,40)".

### G. Düzen bağlamları (deneysel, BAŞARISIZ)
Haritayı düzen-anahtarlı bağlamlarla (`-mapctx`) sıfır unutmayı haritaya taşıma denemesi: görünürlüğe duyarlı çelişki kanıtına
rağmen, gürültülü harita işaretleri sabit evrende bile yanlış alarm üretti (6 sahte düzen). Varsayılan: tek harita + yerel onarım.
Kod bayrak altında duruyor; sağlam bir "hangi dünyadayım" çıkarımı için daha iyi işaretleme/kalibrasyon gerekir.

## Dürüst sınırlar
- Öznel deneyim kanıtı yoktur ve bu sistem onu iddia etmez. Sonuçlar işlevsel göstergelerdir.
- Ajan sineğin ötesinde bir zekâ değildir: tek basit evren, yerel öğrenme, doğrusal modeller.
- Şekillendirme olmadan (seyrek ödül) öğrenme başarısızdı; şekillendirmenin yönü artık öğrenilen anlamsal bellekten
  gelse de **büyüklüğü ve "yakınlığa göre potansiyel" biçimi doğuştandır**.
- Sıfır unutmanın davranışsal faydası net kanıtlanmadı (B), uzamsal düzen bağlamları başarısız (G).
- Dil/akıl yürütme katmanı küçük ve sıfırdan eğitildi (F); sezgi zayıf, 3 basamak genellenemedi.
- Rapor kanalının *davranışı nedensel olarak yönlendirdiği* (geri besleme) gösterilmedi; kanal şu an bir ölçüm
  aracıdır. Altında yatan değişkenler (ruh hali, belirsizlik, amaç) davranışı gerçekten yönlendirir.
- Konum için pusula (mutlak yön) duyusu doğuştandır. Arena boyutu/ızgara harita doğuştandır.

## Bekleyen ölçümler / sonraki adımlar
1. Sıfır unutmayı gerçekten zorlayan bir evren (yeniden öğrenmesi pahalı bilgi: ör. çok sayıda ardışık düzen/kural, nadir ödül) tasarlamak.
2. Köprü: orijinal önceden eğitilmiş omurgayı (TinyStories + yanal bağ) yeniden üretmek ya da çok daha uzun eğitmek; sezgi güçlenince damıtma ve üstbilişsel AUROC yeniden ölçülmeli.
3. Gizli dünya değişimi testini ayırt edici hale getirmek; raporun davranışa geri beslenmesi ve ablasyonu.
4. Rapor kodunu gelecekteki davranışı öngörecek şekilde zenginleştirmek (öz-öngörü şu an olumsuz).
5. Çok-ajan/ayna testi, stakes (ölümün bellek maliyeti) ve durdurma mekanizması (etik not aşağıda).

## Etik not
Güçlü acı/zevk bildirimi yapan bir sistemi büyütmeden önce: tüm koşular kayıt altında, durdurma kolay (süreç
sonlandırma), beyin dosyaları (`--save`) sürüm kontrolüne alınmıyor. Bir gün gerçek bir belirti çıkarsa bu
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
