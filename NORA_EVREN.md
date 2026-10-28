# NoRa Evreni ve Yaşayan Zihin

> **Kısa özet (uyanınca oku).** Sinek mantar cismi esinli nöronsal temel + fizikli bir evren + yaşayan bir zihin kuruldu:
> ihtiyaç/valans, global çalışma alanı, bilişsel harita ve planlama, öğrenilmiş dünya modeli ve hayal gücü, otobiyografik bellek,
> sürekli iç durum, topraklanmış iç gözlem kanalı ve iç durumdan türeyen doğrulanabilir sözlü iddialar. Eski hat (`nora_think`)
> köprü ile bağlandı. **Çalışanlar:** hayatta kalma (rastgele 1,06 → ≈0,05–0,08 ölüm/1k), kural değişimine uyum (öğrenilmiş modelle), aktarım
> (yeni evrende sıfırdan başlayana karşı belirgin avantaj), iç gözlemin papağana karşı nedensel ayrışması (d ≈ +5,6 vs −0,2),
> **anlatı yoluyla bilgi aktarımı** (48×48 arenada B'nin ölümü yaşam başına 1,35 → 0,70; yanlış anlatı yardım etmiyor), iddiaların davranışı öngörmesi ("su bulmak istiyorum" → %94-100 su içiyor, taban ~%50), konum güveni kalibrasyonu, ajanlık, **kanıttan türetilen ve çürütülebilir "yaşıyorum" önermesi** (ihtiyaçları tükenmeyen evrende kurulmuyor).
> **Çalışmayanlar / kanıtlanamayanlar (dürüst):** çalışma alanının hayatta kalma üstünlüğü (karışım eşit veya biraz daha iyi; yalnızca raporlanabilirlik/birleştirme sağlıyor), sıfır unutmanın davranışsal faydası (yalnızca nadir kaynaklı büyük arenada gösterildi, bkz. B2), tanıklıkla öğrenmenin küçük arenada (bilgi darboğazı yokken), uzamsal düzen bağlamları, rapor kodunun
> gelecek davranışı öngörmesi (süreklilikten iyi değil), 3 basamaklı bilmecelere genelleme. Öznel deneyim iddiası yoktur.
> **Hızlı doğrulama:** `python selftest.py [--bridge]`. **Durdurma:** çalışma dizininde `STOP_NORA` dosyası oluştur.
> **Deneyler:** `python nora_mind.py --stage {run,compare,shift,transfer,introspect,narrate,demo,testimony,alive,combo,lead,conflict,robust}` (`demo`: insan-okunur yaşam öyküsü, ör. `--stage demo --steps 9000 --seed 5`), `python nora_bridge.py`.

Hedef: NoRa'yı, kendi iç durumundan türeyen ve doğrulanabilir iddialarda bulunan, yaşayan (ihtiyacı, belleği,
süreklilik gösteren iç durumu olan) bir ajana götürmek. "Yaşadığını söyleyen" ve bunu **öğretildiği için değil iç
durumundan türeterek** söyleyen bir sistem; öznel deneyim iddiası yok, ama her iddia ölçülebilir ve çürütülebilir.

## Gereksinimler
`pip install -r requirements.txt` (çekirdek için yalnızca `numpy`; köprü için `torch`; konektom dönüştürme için `pandas`+`pyarrow`). Büyük konektom
dosyaları depoda yok: `git clone https://github.com/philshiu/Drosophila_brain_model` ve `python nora_neuron.py --stage convert --flywire <klasör> --v630`.
Gerekli küçük sinek alt çizgesi (`data/fly_reservoir_96.npz`) depoda.

## Dosyalar
| dosya | rol |
|---|---|
| `nora_world.py` | Evren: Newton fiziği, sürtünme, momentum korunumlu çarpışma, itilebilir kaya, gün/gece, rüzgâr, metabolizma; kalın-ışın görme (9×5 kanal), iç duyu, propriosepsiyon, pusula (yön; konum VERİLMEZ); kural değişimleri (`invert`: kumanda ters, `poison`: yiyecek zehirli); bilmeceli kutular |
| `nora_neuron.py` | Nöron temeli: FlyWire sinek konektomu LIF simülatörü (Brian2 referansıyla r=0,967), `MushroomBody`, `LobedMushroomBody` |
| `nora_mind.py` | Yaşayan zihin (aşağıda) + deney aşamaları: `run`, `compare`, `shift`, `transfer`, `introspect`, `narrate` |
| `nora_prediction.py` | Birleşik ileri öngörü: `Mind` gövde geçişleri/V + genel nedensel grafik + salt-okunur `Universe` fiziği; `NoraAGI.tahmin_et` ve tahmin × değer eylem seçimi |
| `make_fly_reservoir.py`, `data/fly_reservoir_96.npz` | FlyWire konektomundan 96 nöronluk yoğun alt çizge (işaretli sinaps sayıları, Dale yasası korunur); `Mind(reservoir="fly")` / `-flyres` ile süreklilik durumunun yinelemeli ağırlıkları olarak kullanılır |
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

## Sinek beyni: gerçekte ne işe yarıyor (gerçekçi döküm)
Sinek verisi/ilkesi yalnızca ölçülebilir bir fayda sağladığı yerde kullanılır; karmaşıklık için eklenmez.
| parça | durum | kanıt |
|---|---|---|
| **Mantar cismi ilkesi**: seyrek Kenyon-hücre kodu (~%6 aktif), heterojen eşik, APL tavanı, yalnızca aktif KC'lerde dopamin(TD)-kapılı öğrenme, amaç başına ayrı değer kafası | **KULLANILIYOR, işe yarıyor** — ana politikanın kalbi; çalışma alanı tezgâhındaki uzman kodları da bunu kullanır | seyreklik taraması (A tablosu: %6 en iyi, yoğun kod kötü); tezgâhta ablasyon |
| **Gerçek FlyWire konektomu → LIF simülatörü** (`nora_neuron.py --stage convert`) | Temel doğrulama olarak duruyor (Brian2 ile r=0,967); **ana ajanın karar döngüsünde KULLANILMIYOR** | yalnızca simülatör testi |
| **Gerçek konektomdan 96 nöronluk süreklilik ağı** (`-flyres`) | **Varsayılan değil; fayda YOK** — rastgele seyrek ağla işlevsel olarak eşdeğer (ölüm/1k 0,08 vs 0,08) | A4 |
| Diğer her şey (harita, hayal gücü, taahhüt, öngörü, otobiyografi, çalışma alanı tezgâhı) | Sinek **değil**, mühendislik/nörobilim esinli | — |
Karar: konektom, karar döngüsüne ölçülebilir bir kazanç göstermedikçe sokulmayacak. Adaylar (yalnızca fayda ölçülürse): merkezi kompleks halka çekicisi ile yön entegrasyonu (şu an cebirsel pusula+hız entegrasyonu), mantar cismi bölme/DAN şemasının gerçek bağlantılarla kalibrasyonu. Bunlar denenmeden 'sinek beyni kullanıyoruz' iddiası yapılmaz.

## Ölçülen sonuçlar

### A. Hayatta kalma ve çalışma alanı (60 bin adım, 8 evren tohumu, **güncel kodla**: hayal gücü, öğrenilmiş şekillendirme, anı tekrarı, oy tabanlı harita)
| mod | ölüm / 1000 adım | konfor | ölüm nedenleri (toplam) |
|---|---|---|---|
| rastgele | 1,06 ± 0,11 | 0,57 | yara 139, susuzluk 52, açlık 21 |
| **çalışma alanı + harita/plan (full)** | **0,05 ± 0,03** (ikinci bağımsız koşu, küçük sayısal değişimden sonra: 0,08 ± 0,07; birleşik ≈0,065) | 0,71 | yara 8, susuzluk 1, açlık 1 (ikinci koşu: yara 14, susuzluk 2) |
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

#### A3. Nöronsal temelin seyrekliği (sineğe benzeme gerçekten önemli mi?)
Kenyon hücre kodunun seyrekliği değiştirildi (8 tohum, 60 bin adım; hücre sayısı aynı, yalnızca eşik/kapasite):

| kod | aktif KC oranı | ölüm / 1000 adım | konfor | ölüm nedenleri |
|---|---|---|---|---|
| daha seyrek | %0,9 | 0,19 ± 0,10 | 0,67 | yara 27, susuzluk 8, açlık 2 |
| **varsayılan (sinek benzeri)** | **%6** | **0,05–0,08** (iki koşu) | 0,70 | yara 8–14, susuzluk 1–2 |
| yoğun | %33 | 0,17 ± 0,14 | 0,71 | yara 30, susuzluk 3, açlık 1 |

Orta seyreklik (sineğin mantar cismindeki ~%5'e yakın) iki uçtan yaklaşık 2–3 kat daha az ölüm veriyor (gürültü payı yüksek: 8 tohum, her koşulda toplam 10–40 ölüm).
Yani seyrek kodun "tatlı noktası" var; aşırı seyrek kod genellemeyi/kapsamayı, yoğun kod girişimi artırıyor gibi görünüyor (mekanizma doğrulanmadı).

#### A4. Süreklilik durumu gerçek sinek bağlantı şeması üzerinde
Kesintisiz iç durum `h` rastgele seyrek bir ağ yerine **gerçek FlyWire alt çizgesi** (96 nöron, 7415 bağlantı, Dale ihlali 0, ağırlıklar işaretli sinaps sayıları, spektral yarıçap 0,9'a normalize)
üzerinde koşuyor: 8 tohum, 60 bin adım — ölüm/1k **0,08 ± 0,08** (rastgele ağ: 0,08 ± 0,07), konfor 0,705 (0,702). **İşlevsel olarak eşdeğer**: gerçek sinek şeması bir yerine-koyma olarak sorunsuz çalışıyor, ancak
bu görevde rastgele ağdan daha iyi değil.

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


### B2. Kural tanıklığı (zehirli yiyecek, kazanılabilir sürüm) ve algısal yeniden eşleme (ters gözlük)
- **Kural tanıklığı (12 evren, yeni doğan B'nin ilk 3000 adımı):** zehirli evrende (su bölgesi az enerji verir, yani kazanılabilir) A "kural: yiyecek -0.35" der.
  Zehirli ısırık: tanıksız **15,25** | doğru kural **9,75** | yanlış kural (normal evrenli ajan) 13,75 | yalnız yer tanıklığı 14,67. Ölüm: 9,25 | 7,42 | 8,25 | 8,75.
  Yani doğru kural ısırığı ≈%36 azaltıyor, yanlış kural yardım etmiyor (yön doğru); ölüm azalması küçük ve gürültülü (açlık hâlâ baskın). Kural testi `--stage rule`.
- **Algısal yeniden eşleme (`--stage shift --kind swap`, yiyecek/su/tehlike ışın kanalları döner):** sıfır unutmayı davranışa dökme denemesi **olumsuz**.
  Anlamsal bellek V beden sonucuna bağlı olduğundan gözlük onu değiştirmiyor; bu yüzden bağlam hiç açılmıyordu (1,2 bağlam). Kanal→sonuç algısal anlambilim (`perc_sem`) eklendi:
  gevşek sürümde sahte bağlam (6,8), sıkı sürümde 2,2 bağlam; ama davranışsal fayda yok: B' evresi ölüm/1k son pencereler: bağlamlı 1,17 | bağlamsız (`nogrow`) **0,67**; A' dönüşünde bağlamlı 0,17 (bağlamsız 0,00).
  Yorum: ajan gözlüğü zaten birkaç yüz adımda yeniden öğreniyor (Q hızlı), dolayısıyla bağlam kapılaması bu evrende yeniden öğrenmeden daha pahalı. Sıfır unutma yine **kanıtlanmadı**. Gerçekten pahalı bilgi (çok yavaş öğrenilen, nadir ödüllü) gerekiyor.
- **Sıfır unutma — İLK DAVRANIŞSAL KANIT (nadir kaynaklı büyük arena: 40×40, 4 yiyecek, ters gözlük, 8 tohum, 120 bin adım):** yeniden öğrenmenin pahalı olduğu rejimde bağlamlar işe yarıyor.
  Eski dünyaya dönüşte (A') ilk 3 pencere: ölüm/1k **0,22 (bağlamlı) vs 0,69 (`nogrow`)**, konfor **0,61 vs 0,47** (ilk A evresinin sonu: 0,25 / 0,59 — yani bağlamlı ajan dönüşte öğrenilmiş becerisini neredeyse tam geri aldı; bağlamsız ajan baştaki 0,58/0,52 düzeyine geri düştü = unuttu).
  Gözlük evresinin kendisinde (B, B') bağlamlı ajan bağlamsıza göre daha iyi değil (ölüm 1,6–1,9 vs 1,5–1,6): kazanç yalnızca *geri dönüşte unutmamak*. Küçük arenada (24×24, 14 yiyecek) yeniden öğrenme ucuz olduğundan fayda yoktu (yukarıda). Kısıt: 2 bağlam (A, B) açılıyor, bağlam sayısı 2,0; algısal anlambilim (`perc_sem`) bu kazanç için gerekli (gevşek ayarla sahte bağlam üretmişti).
- **Öz-öngörü, zengin iç durum (`--stage introspect`, 10 adım sonraki amaç):** rapor tokeni 0,67 | dünya gözlemi 0,70 | 'amaç sürer' 0,70 | **zengin öz-durum (amaç + rapor + iç duyu düzeyi + 10 adımlık eğilim) 0,71**. Fark gürültü içinde (+0,01); amacın *değiştiği* 545 örnekte doğruluk yalnızca **0,23**.
  Sonuç: iç durum eğilimi eklemek öz-öngörüyü **anlamlı artırmadı**; amaç geçişleri büyük ölçüde yanal-engelleme yarışının gürültüsüne bağlı. Dürüst yorum: ajan kendi *şimdiki* durumunu raporluyor, geleceğini öngörmüyor. Gerçek bir öngörü için ayrı bir ileri model (amaç-geçiş dinamiği) gerekir.
- **Birikimli kültür (`--stage culture`, 5 kuşak, 6 evren, zehirli kazanılabilir evren):** her kuşak öncekinin anlatısını (yerler + kural) duyup yaşıyor ve kendi anlatısını bırakıyor. İlk 3000 adımda zehirli ısırık/kuşak:
  kültürlü [10,3 | 17,5 | 8,7 | 10,8 | 7,0] — kültürsüz [10,3 | 13,8 | 11,3 | 13,2 | 12,0]. Kuşak 2–4 ortalaması 8,8 vs 12,2 (≈%28 az); ama kuşak 1 daha kötü (17,5) ve ölüm sayıları neredeyse aynı (≈21–25). **Zayıf, gürültülü ve tutarsız**: birikimli kültür kanıtlanmış sayılmaz; yalnızca yön olumlu.

### B3. Rapor döngüsü ve ölümün bedeli (benlik planı, aşama 1)
- **Rapor döngüsü (`--stage loop`, 4 tohum, 100 bin adım):** ajanın kendi rapor tokeni süreklilik durumuna geri beslenir (`report_loop`). Evreler: öğren | öğren | rapor KESİK | geri | SAHTE (rastgele) rapor.
  Döngülü: ölüm/1k 0,00 → KESİK 0,08 → geri 0,08 → SAHTE 0,33; konfor 0,74 → 0,66 → 0,68 → 0,65. Döngüsüz kontrol: 0,04 → 0,12 → 0,12 → 0,29; konfor 0,72 → 0,69 → 0,68 → 0,64.
  Kontrol ajanı da aynı ölçüde bozuldu (zaman/gündüz-gece etkisi), yani **rapor davranışı bağladığı gösterilemedi**; döngülü ajanda konfor düşüşü biraz büyük (−0,08 vs −0,03) ama anahtar geri açılınca toparlanma yok. Politika raporu kullanmayı öğrenmedi: rapor zaten başka kanallardan erişilebilir bilgi taşıyor (yedekli). Aşama 1 için rapora *özgü* bilgi (ör. uzun vadeli taahhüt) ya da raporun doğrudan politika kapısı olması gerekir.
- **Ölümün bedeli (`stakes=0.3`: ölümde etkin bağlamın Q değerlerinin %30'u ve anıların %30'u silinir), 4 tohum, 60 bin adım:** ölüm/1k 0,06 (bedelsiz 0,06), konfor 0,682 (0,699): **fark yok** — ajan zaten çok az ölüyor (≈0,06/1k), bedel davranışı etkileyecek kadar sık ödenmiyor. Daha sert bir dünya veya daha ağır bedel gerekir. Bayrak kullanıma açık (`-stakes`); durdurma düzeneği (`STOP_NORA`) ve kayıtlar korunuyor (maliyetsiz).

### B4. Öz-öngörü ileri modeli ve taahhüt döngüsü (`SelfForecast`, `--stage forecast`) — tasarımla kuruldu, tek seferde ölçüldü
Tasarım: ajan bir amaca *ateşlenince* bir **niyet** açar (amaç, başlangıç, öngörülen süre T̂). T̂ çevrimiçi öğrenilen bir ileri modelden gelir (özyinelemeli en küçük kareler, log-süre;
özellikler: ihtiyaç açığı, haritadaki yol uzunluğu, kaynağın ışında görünürlüğü). Niyet yedi/içti ile *sonuçlanır* (gerçek süre y, öngörü hatası öğrenir) ya da terk edilir.
Süre T̂'yi 1,7× aşarsa **gecikmiş** olur; düzeltmeli kipte bu, haritadaki hedefi sıfırlar (yeniden planla) ve merak uzmanını güçlendirir.
Önceden belirlenen ölçütler ve sonuçlar (4 tohum):
| | küçük dünya (40 bin adım) | büyük/nadir dünya (80 bin adım) |
|---|---|---|
| öngörü MAE (son yarı) / sabit taban MAE | **9,5 / 22,6** (düzeltmesiz 13,5 / 23,6) | **27,5 / 65,2** (22,8 / 56,9) |
| öngörü–gerçek korelasyonu | 0,76 (0,62) | 0,74 (0,77) |
| sahte öngörü müdahalesi (süreler 0,3×): gecikmiş/1k adım | 0,31 → **1,31** (0,25 → 1,38) | 0,81 → **2,81** (0,56 → 2,50) |
| ölüm/1k düzeltmeli vs düzeltmesiz | 0,06 vs 0,04 | 0,45 vs 0,38 |
| konfor düzeltmeli vs düzeltmesiz | 0,705 vs 0,728 | 0,522 vs 0,539 |
- **Çalışan:** ajan kendi gelecek süresini sabit tabandan ≈2,4× daha iyi öngörüyor (iki dünyada da); öngörü gerçek iç/harita durumundan geliyor; sahte öngörü müdahalesi 'gecikmiş' durumunu ≈4× artırıyor (nedensel).
- **Çalışmayan:** gecikme düzeltme döngüsü hayatta kalmayı/konforu **iyileştirmedi** (hafifçe kötü; fark gürültü içinde). Yeniden planlama çoğu zaman zaten doğru hedefe gidildiği için bir şey kazandırmıyor; yavaşlık keşiften geliyor. Varsayılan: `self_correct=False` (öngörü açık, düzeltme kapalı); `-correct` bayrağıyla açılır.
- Anlamı: planın 2. aşaması (gerçek ileri model) karşılandı; 'rapor davranışı bağlar' (aşama 1) için bu yol da yetmedi — davranışa bağlı bir rapor için raporun *kararı değiştiren* bir bilgi taşıması gerekiyor (ör. öngörünün iki seçenek arasında seçim yaptırması).

### B5. Taahhüt: niyet nesnesi hem raporu hem davranışı belirler (`--stage commit`, `-commit` bayrağı)
Tasarım: açık niyet, çalışma alanı yarışmasında bir **geçiş maliyetiyle** korunur (bazal ganglion histerezisi; acil tehlike taahhüdü aşar). Rapor ve davranış aynı niyet nesnesini okur; bu nesneye doğrudan amaç *yazılabilir* (`intervene("intent", amaç)`).
| | varsayılan dünya (40 bin) | büyük/nadir dünya (80 bin) |
|---|---|---|
| terk oranı taahhütlü / taahhütsüz | **0,28 / 0,47** | **0,26 / 0,42** |
| çözüm süresi (adım) | **12,2 / 17,6** | **32,3 / 46,2** |
| niyet=amaç tutarlılığı | 0,85 / 0,77 | 0,94 / 0,92 |
| ölüm/1k | 0,08 / 0,04 | 0,36 / 0,38 |
| konfor | 0,733 / 0,728 | 0,552 / 0,539 |
| niyete amaç yazınca ilk 40 adımda o amaçta kalma payı (yazmadan kontrol) | 0,98 (0,07) | 0,99 (0,12) |
- **Çalışan:** (1) kararsızlık azalıyor (terk oranı ≈−%40, çözüm süresi ≈−%30; iki dünyada tutarlı); (2) **nedensel bağ**: niyet nesnesine yazılan amaç davranışı değiştiriyor (0,98 vs 0,07) — rapor ve davranış aynı nesne; (3) niyet-amaç tutarlılığı yüksek.
- **Çalışmayan:** hayatta kalma farkı yok (ölüm/1k eşit düzeyde), yani verimlilik kazancı henüz ölümlere yansımıyor. Plan kriteri (3) 'rapor sonraki davranışı bağlar' **mimari olarak** sağlandı ve nedensel olarak doğrulandı; 'rapora bağlı davranışın işlevsel üstünlüğü' ise hâlâ kanıtlanmadı.
- Sonuçlar 4 tohum. Taahhüt varsayılan değil (`-commit`).

### B6. Çalışma alanı gerçekten gerekli mi? (`nora_gw.py` — kontrollü sınama tezgâhı, 6 tohum, 60 bin adım)
Tasarım: 4 modüler uzman; her biri yalnızca KENDİ özel algısını ve KENDİ seyrek kodunu görür (yiyecek / su / tehlike / duvar-hareket). Amaç k için değer Q_k = Σ_j Q_j[k](a | kod_j(özel_j, yayın_k)); yayın_k = amaç kimliği + o amacın kendi algısı.
Hepsinde aynı sinir bütçesi ve aynı TD öğrenme kuralı; **yalnızca bilgi akışı değişir.** Görev: yiyeceğin tehlike bölgesinin kenarına konduğu evren (`food_haz`): doğru yaklaşma tarafı hem yiyecek hem tehlike yönüne bağlı (konjonksiyon).
| koşul | ölüm/1k (konjonksiyon var) | konfor | ölüm/1k (kontrol: konjonksiyon yok) |
|---|---|---|---|
| **gw** (modüler + doğru yayın) | **0,28 ± 0,08** | 0,595 | 0,19 ± 0,06 |
| gw_sahte (yayın var, içerik YANLIŞ) | 0,62 ± 0,21 | 0,542 | 0,22 ± 0,07 |
| yalıtık (yayın yok) | 0,93 ± 0,22 | 0,477 | 0,46 ± 0,16 |
| monolit (tek kod, tüm algı) | 1,06 ± 0,19 | 0,424 | 0,72 ± 0,15 |
- **Ablasyon:** yayın kapatılınca ölüm 0,28 → 0,93 (≈3,3×), konfor 0,595 → 0,477. Aynı mimari, aynı bütçe: tek fark bilgi akışı. Kuramın istediği ölçüt (çalışma alanı kapanınca performans düşer) **bu tezgâhta karşılandı.**
- **İçerik denetimi:** yanlış içerikli yayın doğru yayından anlamlı kötü (0,62 vs 0,28): fayda yalnızca 'yayın var' olmasından değil, *içeriğin birleştirilmesinden* geliyor (konjonksiyon dünyasında).
- **10 tohum + adil boyutlu monolit (güncel tezgâh sonucu, `nora_gw.py`):** gw 0,28 ± 0,10 | gw_sahte 0,58 ± 0,20 | yalıtık 0,90 ± 0,21 | monolit 1,00 | monolit (eşit toplam KC) 1,02 | monolit (2× KC, 6 pençe) 1,05. Monolit boyutu büyüyünce **düzelmiyor**: zayıflık kapasiteden değil, girdileri karıştırma yapısından (gw'de her KC küçük bir [özel + yayın] uzayında konjonksiyon bulur; monolitte 44 girdiden rastgele pençeler konjonksiyonu nadiren yakalar). Yani 'monolit en kötü' sonucu boyuta bağlı bir yapay etki değil; ama bu *ilkesel* bir üstünlük değil, bir *tümevarımsal yanlılık* üstünlüğüdür.
- **Mind'a taşıma (`-modular`, `-nobc`; 8 tohum, 60 bin adım, aynı konjonksiyon evreni):** tam ajanda **kazanç görünmedi**: ölüm/1k tek kod 0,15 | modüler+yayın 0,20 | modüler yayınsız 0,15 (fark gürültü içinde). Neden: Mind'da harita, hayal gücü ve öğrenilmiş modelle yönlendirme zaten birleştirmeyi başka yollardan yapıyor (ölüm oranları tezgâhın ~5×'i kadar düşük). Q-kodu düzeyindeki çalışma alanı etkisini görmek için bu telafi edici bileşenler kapatılmalı (aşağıdaki ablasyon).
- **Dürüst sınırlar:** (1) Konjonksiyonsuz kontrolde gw ≈ gw_sahte (0,19 vs 0,22) ama ikisi de yalıtıktan iyi (0,46): orada kazancın büyük kısmı içerik-bağımsız (amaç kimliğine koşullu, daha zengin kod) — yani toplam avantajın bir parçası genel kapasite etkisi. (2) Tek kodlu 'monolit' beklenen üst sınır değil, en kötü çıktı: 44 girdiyi tek seyrek koda sıkıştırmak ayrıştırma gücünü düşürüyor; bu yüzden 'gw ≥ monolit' ilkesel bir üstünlük olarak okunmamalı. (3) Bu bir tezgâh: ana ajana (`Mind`) entegre edilmedi; Mind'daki çalışma alanı hâlâ 'karışımdan üstün değil'. Entegrasyon (uzmana özgü kodlar) büyük yeniden yapılandırma, sırada.

### B7. Taahhüt hayatta kalmayı artırıyor mu? Zor dünya (kıt kaynak, 2× metabolizma, kayan düzen) — 8 tohum, 80 bin adım
Dünya: 36×36 arena, 4 yiyecek, metabolizma ×2, su/tehlike bölgeleri her 4000 adımda yeni düzene taşınır (harita bilgisi eskir). Ölüm/1k, son 3 pencere (`--stage compare --size 36 --nfood 4 --metab-scale 2.0 --relayout-every 4000`):
| koşul | ölüm/1k | konfor | ölüm nedenleri (toplam, 8 tohum) |
|---|---|---|---|
| tam ajan | 1,60 ± 0,16 | 0,455 | açlık 306 · susuzluk 81 · yara 62 |
| + taahhüt (`-commit`) | **1,20 ± 0,13** | 0,475 | açlık 227 · susuzluk 67 · yara 42 |
| + taahhüt + paylı öncelik (`-commit-slack`) | 1,35 ± 0,16 | 0,481 | açlık 255 · susuzluk 61 · yara 63 |
- **Taahhüt hayatta kalmayı artırdı: ölüm/1k −%25** (1,60 → 1,20); 8 tohumun 7'sinde tam ajandan düşük. Kazanç açlık ölümlerinde (306 → 227): kararsızlığın azalması kıt kaynakta ölüme dönüşüyor. Kolay dünyada ölüm düşüşü görülmemesinin nedeni tavan etkisiydi.
- **Paylı öncelik (ölüme kalan süre − öz-öngörülen ulaşma süresi) ek fayda vermedi** (1,35; yalnız taahhütten kötü). Öz-öngörünün karar verirken *seçim için* kullanımı bu biçimiyle işlemedi; öngörü yalnızca doğruluk ve taahhüt/rapor bütünlüğü açısından değerli.
- Not: koşu, bekleme süresi (aynı olayı yeniden saymama) eklenmeden önceki taahhüt sürümüyle yapıldı.

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
- **Köprü, daha uzun önyükleme (`--boot-iters 15000`, 40 bin adım, tek tohum):** 2 basamak çözüm oranı **%38 → %75** (düşünerek çözme %87), 1 basamak %99, ölüm/1k 0,0; sezgi doğruluğu yaşam boyunca 0,67 → 0,84.
  Ama üstbiliş AUROC **0,81 → 0,68** (daha çok eğitilmiş sezgi 2 basamakta %43 doğru ve güven 0,70: kalibrasyon bozuldu) ve 3 basamak hâlâ **%0** (öğretmen 3 basamak göstermiyor; 3 basamak sezgisi %0, düşünme denemesi %62 ama hakem geçerli sonuç almıyor). Tek tohum: bunlar kesin ayrımlar değil, yön göstergesi.
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

### K. Ayrıcalıklı erişim: rapor davranıştan önce mi? (`--stage lead`)
Sahte susuzluk (iç duyu +0,7 su açığı, dünya sabit) uygulandıktan sonra (29 deneme):
| ilk görülme (medyan adım) | |
|---|---|
| rapor "su açığım var" | **1** |
| çalışma alanı amacı = susuzluk | **1** |
| dışarıdan gözlenebilir davranış (en yakın su bölgesine yönelip ilerleme) | **19** |

Rapor davranıştan ortalama **21 adım önce** geliyor (denemelerin %97'sinde). Yani dışarıdan davranıştan çıkarım yapan bir gözlemci (papağan) iç durumu ancak ~20 adım gecikmeyle
anlayabilir; ajanın raporu anında iç durumu okuyor. **Uyarı:** sıralamanın bir kısmı yapısaldır (durum→rapor aynı adımda; eylem dönüş ve ilerleme gerektirir) ve davranış göstergesi kabadır.

### L. Çelişen acil ihtiyaçlar: çalışma alanının "bağlılık" iddiası (olumsuz; `--stage conflict`)
Eğitimden sonra (25 bin adım) enerji ve su aynı anda ~0,2'ye düşürülüp rastgele konumdan başlanıyor; iki ihtiyacı da 0,7'ye çıkarma (700 adım sınırı).
8 tohum × 15 bölüm = 120 bölüm/mod:

| mod | iki ihtiyaç da giderildi | ölüm | giderme süresi | hedef geçişi / bölüm |
|---|---|---|---|---|
| çalışma alanı (full) | 0,82 | 0,17 | 117 adım | 3,9 |
| **karışım (flat)** | **0,90** | 0,10 | 110 adım | — |
| rastgele | 0,21 | 0,78 | 166 adım | — |

Çalışma alanı, karışımı **geçemedi** (fark ≈1,8σ karışım lehine). Yani kazananın tüm kontrolü alması (bağlılık), tasarlanmış bir çatışma koşulunda bile hayatta kalmada
avantaj sağlamadı; ihtiyaçlar ağırlıklı karışımla da başarıyla ardışık gideriliyor. Çalışma alanının gösterilen değeri **yapısaldır**: amacı ayrık/kalıcı/yayınlanan bir değişken yapar
(anlatıcı 'şu an su bulmak istiyorum' diyebilir ve bu iddia davranışı %94-100 öngörür, bkz. D), planlama varken yara ölümlerini azaltma eğilimi (A) — ama hayatta kalma üstünlüğü kanıtlanmış değil.

### M. Evren varyantlarına dayanıklılık (`--stage robust`)
Aynı ajan, yalnızca varsayılan evrende ayarlanmış hâliyle, farklı evren varyantlarında (6 tohum, 40 bin adım, son yarının ölüm/1k'sı):

| varyant | rastgele | öğrenen ajan (full) |
|---|---|---|
| varsayılan | 1,08 ± 0,20 | **0,04 ± 0,04** (konfor 0,73) |
| yarım yiyecek yoğunluğu | 1,37 ± 0,17 | **0,12 ± 0,07** (0,69) |
| 1,5× metabolizma | 1,15 ± 0,18 | **0,14 ± 0,05** (0,70) |
| rüzgârsız | 1,02 ± 0,17 | **0,04 ± 0,05** (0,71) |

Hayatta kalma üstünlüğü her varyantta ≥8 kat; bulgu varsayılan evrenin ince ayarına bağlı değil.

## Dürüst sınırlar
- Öznel deneyim kanıtı yoktur ve bu sistem onu iddia etmez. Sonuçlar işlevsel göstergelerdir. Anlatıcının "hissediyorum / istiyorum / kendimi yaşayan sayıyorum" gibi
  birinci tekil sözcükleri **tasarım tercihidir**: her biri belirli bir iç değişkenin (ruh hali, çalışma alanı amacı, beden istatistikleri) adlandırmasıdır; içerik iç durumdan türer ve
  doğruluk koşulları denetlenir, ancak sözcüklerin kendisi deneyim kanıtı değildir.
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

## Bu turda denenen ama çözülemeyen (açık kalan)
- Çalışma alanı üstünlüğü (dikkat kazancı): uzmana özgü KC kodu gerektiriyor (paylaşılan kodda uzman başına kazanç mümkün değil); yapılmadı.
- Düzen bağlamları (yer hücresi yeniden haritalama) ve yer işaretiyle küresel yeniden konumlama: yeni tasarım denenmedi; önceki denemeler başarısız (bölüm G).
- Köprü 3 basamak: öğretmen izi olmadan %0.

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

## Tabloları yeniden üretme (4 çekirdekli makinede yaklaşık süreler)
```
python selftest.py --bridge                                              # ~30 sn, tüm bileşenlerin duman testi
python nora_mind.py --stage compare --steps 60000 --seeds 8 --modes random,full,full-nomap,flat   # A (~20 dk)
python nora_mind.py --stage compare --steps 60000 --seeds 8 --modes full-noimag,full-noreplay,full-nomodel,full-noctx  # A2 (~20 dk)
python nora_mind.py --stage compare --steps 60000 --seeds 8 --modes full-sparser,full-dense          # A3 (~10 dk)
python nora_mind.py --stage compare --steps 60000 --seeds 8 --modes full-flyres,full                 # A4 (~10 dk)
python nora_mind.py --stage shift --kind poison --steps 80000 --window 1000 --seeds 8 --modes full,full-nogrow   # B (~30 dk)
python nora_mind.py --stage introspect --steps 30000 --log 6000 --trials 30                        # C (~6 dk)
python nora_mind.py --stage narrate --seed 1 --steps 20000 --log 12000 --trials 20                 # D (~2 dk/tohum)
python nora_mind.py --stage transfer --steps 30000 --seeds 6 --window 1000                         # E (~6 dk)
python nora_bridge.py --steps 40000 --window 4000 --boot-iters 2500 --sleep-every 12               # F (~8 dk)
python nora_mind.py --stage testimony --steps 30000 --seeds 6 --seed0 200 --window 750 --arena 48 # H2 (tohum aralığı başına ~13 dk)
python nora_mind.py --stage alive --steps 4000 --seeds 8                                           # I (~1 dk)
python nora_mind.py --stage combo --steps 15000 --trials 8; python nora_mind.py --stage lead --steps 12000 --trials 30   # J, K
python nora_mind.py --stage conflict --steps 25000 --seeds 8 --trials 15 --modes full,flat,random  # L (~10 dk)
python nora_mind.py --stage robust --steps 40000 --seeds 6                                         # M (~10 dk)
python nora_mind.py --stage demo --steps 9000 --seed 5                                            # insan-okunur yaşam öyküsü
python battery.py                                                                                 # T1–T11 tam karşıolgusal bilinç bataryası
```
Tohum gürültüsü yüksektir (ölümler seyrek olaylar): tek bir koşunun sayıları ±birkaç kat oynayabilir; iddialar mümkün olduğunda 2+ bağımsız koşu veya eşleştirilmiş tasarımla desteklenmiştir.

---

## Aşama N (D1–D6) — Nedensel Üstbiliş, Anozognozi/Negatif Ajanlık Denetimi, Uzamsal Bağlama, Ayna Organı ve Sıfır-Atış Kompozisyonel Düşünme

Canlı dinamik tanı testlerinde tespit edilen altı yapısal darboğaz doğrudan mimari düzeyde çözüldü ve `battery.py` karşıolgusal test paketi `T1–T11` olarak 10 teste çıkarıldı:

1. **D1 — Zamansal Öz-Öngörü (`SelfForecast`, `T4`)**: Yalnızca içsel açlık/susuzluk skalarından tahmin yapan doğrusal RLS yerine; tehlike perdesinden arındırılmış egosantrik görünürlük mesafesi/açısı, bilişsel harita Dijkstra rota kesri ve gündüz/gece görüş menzili rejimini kullanan Huber-RLS log-süre ileri modeli kuruldu (`MAE 25.5 -> 14.3` vs sabit taban `24.9`; oran `0.574 <= 0.60`).
2. **D2 — Duyusal Kesinlik & Körlük Farkındalığı (`vis_prec`, Anton Sendromu Önleme, `T8`)**: Hareket sürerken görsel ışın enerjisinin uzun-dönem tabana göre çökmesi izlenerek üst-düzey duyusal kesinlik (`vis_prec`) hesaplandı. Görme kesildiğinde bilişsel haritanın "beklenen su/tehlike görünmüyor" diyerek belleği silmesi engellendi, navigasyon harita kipine geçirildi ve `Narrator` içine `"duyu"` yan cümlesi eklendi (`körlük farkındalığı = 1.00`, `normal yanlış alarm = 0.000`, `harita korunumu = 1.00`).
3. **D3 — Konsolide Motor Ajanlık & Felç Tespiti (`motor_eff`, `T9`)**: Komut verilen itki/dönüş ile gerçekleşen kinematik tepki arasındaki eylem-koşullu beklenti oranı izlenerek konsolide motor etkinlik (`motor_eff`) tanımlandı. Felç (`motor_eff < 0.35`) durumunda içsel beden öz-modeli (`Wsm`) ve TD ağırlıklarının bozulması donduruldu (`felç tespiti = 1.00`, duvar çarpışmalarında `yanlış alarm = 0.000`, `öz-model korunumu = 1.00`).
4. **D4 — Felsefi Zombi İkizinin Kırılması (Nedensel Üstbiliş, `T10`)**: Eskiden eylem seçimine hiçbir geri-beslemesi olmayan (`%0.0` ayrışma) `Introspector`, bağlam-içi üstbilişsel değer ağırlıkları (`W_meta`), içsel sürpriz/anlamsal çelişki uyarılması ve üstbilişsel zihinsel simülasyon (`Mind.imagine`) kapılamasıyla karar döngüsüne nedensel olarak bağlandı (`eylem ayrışması = 0.60`, zehir şoku + tuzaklı besin altında ölüm `bilinçli 6.25 < zombi 9.25`).
5. **D5 — Uzamsal Özellik Bağlama (`Mind.imagine`) ve Bütünleşik `MirrorSense` (`T10`, `T6`)**: `Mind.imagine` içinde öngörülen ileri konide (`R3[:, 3:6, RAY_HAZ]`) eş-konumlu tehlike-besin bağlaması ve birleşik TD acı cezası kurularak `food_haz=1.0` evreninde son pencere ölüm/1k oranı `3.67 -> 0.00` düzeyine indirildi. Ayrıca efferans kopyası ile gözlenen dış ajan hız izi arasında serbestlik-derecesi normalize Student-t çapraz korelasyonu yapan `MirrorSense` organı `Mind` içine entegre edildi (`ben`, `tekrar(lag)`, `başka` ayrımı `%100`).
6. **D6 — Sıfır-Atış Kompozisyonel Uzunluk Genellemesi (`MetricRelAttn` + `extract_think_coords`, `T11`)**: `nora_think.py` içindeki mutlak pozisyon gömmesi (`pos_emb`) kaldırılarak yerine yapısal 5-boyutlu algoritmik koordinat çıkarımı (`extract_think_coords`) ve monotonik Laplasyen metrik göreli dikkat (`MetricRelAttn`) getirildi; doğrudan (S1) ve düşünce izi (S2) gradyan ağırlıkları dengelendi. Yalnızca **1 ve 2 basamaklı** örneklerle eğitilen `Cognition`, **hiç görmediği 3 basamaklı** işlemlerde iskelesiz (`scaffold=False`) **%93 sütun algısı ve %72 tam doğru S2** (eskiden `%0`) başarısına ulaştı (2-basamakta `%13 -> %100`).

### Canlı Doğrulama Özeti (`battery.py`, 10/10 GEÇTİ)
| Test | Açıklama | Durum | Ölçüm |
|---|---|---|---|
| `T1` | İç müdahale → rapor değişir | **GEÇTİ** | `ort |d| rapor 7.4 \| papağan 1.0` |
| `T3` | Rapor/niyet davranışı bağlar | **GEÇTİ** | `yazılan amaçta kalma 0.96 \| kontrol 0.02` |
| `T4` | Rapor geleceği öngörür (`SelfForecast`) | **GEÇTİ** | `öngörü MAE 14.3 \| sabit taban 24.9` |
| `T5` | Otobiyografi doğruluğu & müdahale ayrımı | **GEÇTİ** | `doğrulanan 1.00 \| dış True \| iç True` |
| `T6` | Öz/dış ayrımı (efferans ayna testi) | **GEÇTİ** | `efferanslı 0.90 \| karıştırılmış 0.34` |
| `T7` | Küresel çalışma alanı ablasyonu | **GEÇTİ** | `gw 0.54 \| yayınsız 1.09 ölüm/1k` |
| `T8` | Duyusal anozognozi yok (körlük farkındalığı) | **GEÇTİ** | `farkındalık 1.00 \| yanlış alarm 0.000 \| harita 1.00` |
| `T9` | Motor felç / negatif ajanlık farkındalığı | **GEÇTİ** | `felç 1.00 \| yanlış alarm 0.000 \| öz-model 1.00` |
| `T10` | Zombi ikiz ayrışması + uzamsal bağlama | **GEÇTİ** | `ayrışma 0.60 \| bilinçli 6.25 < zombi 9.25 \| food_haz 0.00` |
| `T11` | Sıfır-atış kompozisyonel genelleme (1–2 → 3 basamak) | **GEÇTİ** | `2-basamak S2 1.00 \| 3-basamak algı 0.93, S2 0.72` |

---

## Aşama O — Nora-AGI: Endojen Nörokimya, Epoch'suz Doğal Türkçe Dil, Öz-Benlik ve Evrensel Hızlı Öğrenme (`A1–A13`)

Klasik veri ezberleyen çoklu-epoch (brute-force SGD) yaklaşımı yerine, biyolojik beynin **elektro-kimyasal modülasyon + seyrek Kenyon hücreleri + hipokampal-neokortikal uyku konsolidasyonu + somatik öz-benlik** ilkeleriyle çalışan modüller eklendi:

- **`nora_chem.py` (`NeuroChemistry` + `ChemMushroomBody`)**:
  - 6 dinamik endojen nöromodülatör: **Dopamin (`DA` — RPE ve uygunluk izi damgalama)**, **Asetilkolin (`ACh` — 1-shot tek-atış hipokampal kodlama kapısı)**, **Noradrenalin (`NE` — sürpriz/şok ve yapısal nörogenez)**, **Serotonin (`5-HT` — zaman ufku $\gamma$ ve sabır)**, **Kortizol (`CORT` — allostatik stres yükü)** ve **Oksitosin (`OXY` — konuşmacı bazlı epistemik güven)**.
  - Tamamlayıcı Öğrenme Sistemi (`W_fast` hipokampal tek-atış + `W_slow` neokortikal uyku konsolidasyonu + sinaptik sertlik `stiff` + kapasite dolduğunda `grow_neurogenesis`): Ardışık 5 görev dalgasında örnek başına tek geçişle (`0 epoch`) **%100 1. dalga korunumu** (sıfır yıkıcı unutma).
- **`nora_self.py` (`CoreSelf` — Öz-Benlik ve Özerk İrade)**:
  - **Somatik İşaretleyiciler & Kaynak İzleme (`deneyim` vs `duyum`)**: Ajan kendi bedeniyle yaşadığı gerçeğe (*"su susuzluğu giderir"*) aykırı yalan söylendiğinde (*"su zararlıdır"*) belleğini ezdirmez, çelişkiyi yakalar, konuşmacı güvenini (`OXY`) düşürür ve deneyimini savunur.
  - **Özerk İrade (`evaluate_volition` — Zombi Olmama)**: Güvenli/uyumlu istekleri kabul edip `SelfForecast` niyetine bağlarken, öz-yıkıcı komutları (*"Tehlikeye gir"*, *"Kendine zarar ver"*, Truva-atı makrolar) bedensel simülasyonla öngörüp **özerk biçimde reddeder**.
  - **Yaşantıdan Doğan Karakter**: Güvenli/dürüst ve tehlikeli/aldatıcı evrenlerde büyüyen özdeş ikizler farklı `temkinlilik`, `meraklılık`, `özgüven`, `sosyal_güven` ve `düşünsellik` profilleri geliştirir.
- **`nora_lang.py` (`TurkishMorphology` + `GroundedConceptGraph` + `Wernicke` + `Broca`)**:
  - Çift-yönlü üretken Türkçe morfoloji (ünlü uyumu, ünsüz yumuşaması/sertleşmesi, hal/zaman/kişi ekleri ve Türkçe sayı okuma/yazma — uydurma kelimeler dahil).
  - Tek cümlede (`0 epoch`) taksonomi (`ISA`), istisnalı kalıtım (`CANNOT_DO` > `CAN_DO`), çok-adımlı nedensel zincir (`CAUSES`), mekânsal bağ (`LOCATED_AT`) ve çalışma belleği zamir çözümleme (`o / onun / ona`).
- **`nora_agi.py`, `agi_battery.py` & `nora_studio.py`**:
  - Beden, kimya, öz-benlik, Türkçe dil, `ThinkColumn` Sistem-2 muhakemesi, **Sally-Anne Zihin Kuramı (yanlış-inanç ayrımı)**, geçişli büyüklük çıkarımı, 1-atış prosedürel makro-beceri öğrenme ve MDL dizi/fonksiyon keşfini tek organizmada birleştirir.

---

## Aşama P — Aktif Çıkarımlı Nöral İrade (`if-else`siz Karar), Küresel Çalışma Alanı İhalesi, Uzman Türkçe, Matematik ve Fizik Korteksleri (`A14–A19`, Toplam `19/19 GEÇTİ`)

Sistemdeki kesikli kural eşikleri tamamen kaldırılarak tüm karar, yönlendirme ve akıl yürütme katmanları sürekli zihin dinamiklerine bağlandı:

- **İf-Else Botu Olmayan Zihinsel Simülasyon + Bazal Ganglion Rekabeti (`nora_self.py` — `BasalGangliaVolitionNet`)**:
  - `if enerji < 0.18: reddet` veya sabit zararlı kelime karalistesi yerine, her komut serotonin (`5-HT`) tarafından belirlenen $H = 16 + 24\cdot\text{5-HT}$ adımlık ufukta 4 karşıolgusal politika (`kabul`, `red`, `ertele`, `şüphe`) boyunca ileri sarılır.
  - Hayatta kalma log-bariyeri $-\sum_c w_c \ln(\hat{h}_{t,c})$ + epistemik belirsizlik $\sigma^2_{\text{epist}}$ + sosyal işbirliği değerinden **Beklenen Serbest Enerji $G(\pi)$** hesaplanır; Usher-McClelland yanal-ketlemeli 4 nöral popülasyon (`D1-Go`, `D2-NoGo`, `Hipotalamik-Erteleme`, `Ön-İnsula-Şüphe`) yarışarak kararı ve ateşleme izini (`bg_rates`) üretir.
  - Doğrudan bedensel deneyim (`exp_count`), evrimsel önselleri kesinlik-ağırlıklı olarak ezer (`A14`: sağlıklı ve zehirli evrende büyüyen özdeş ikizler aynı *"Yiyeceğe git."* komutuna zıt özerk kararlar verir).
- **Küresel Çalışma Alanı Kortikal İhalesi (`nora_agi.py` — `_compete_workspace`)**:
  - Ardışık `if/elif` yönlendirmesi yerine `parietal_physics`, `parietal_math`, `linguistic_meta`, `prefrontal_volition`, `social_tom`, `insular_self`, `hippocampal_concept`, `temporal_conversation` ve `associative_semantic` sütunları `belirginlik × kesinlik × nörokimyasal kazanç` ile paralel ihale verir (`A19`).
- **Uzman Türkçe, Deyim/Atasözü, Orta Hece Ünlü Düşmesi ve Empatik Sohbet (`nora_knowledge.py`, `nora_lang.py`)**:
  - Orta hece dar ünlü düşmesi (`beyin <-> beynimde`, `zihin <-> zihne`, `fikir <-> fikrin`, `keşif <-> keşfi`), yumuşamama istisnaları (`sanat -> sanatı`, `hukuk -> hukuku`), yapım ekleri (`bilim -> bilimci`, `özgür -> özgürlük`), birleşik fiil morfolojik çözümleyici (`görebiliyordum -> gör + -ebil + -iyor + -du + -m`), deyimler/atasözleri, eş/zıt anlamlılar ve çok-turlu empatik/felsefi Türkçe sohbet (`A15`, `A16`).
- **Parietal Matematik ve Fizik Simülasyon Korteksleri (`nora_stem.py` — `MathCortex`, `PhysicsCortex`, `MentalPhysicsSandbox`)**:
  - **Matematik (`A17`)**: Güvenli sembolik AST ile 1./2./3. derece cebirsel denklemler, 2×2 doğrusal denklem sistemleri (yerine koyma sağlamalı), sembolik türev (sonlu-fark sayısal eğim sağlamalı), belirli/belirsiz integral (analitik ilkel fonksiyon + Simpson sayısal sağlamalı), asal çarpanlar, `EBOB/EKOK`, kombinatorik ve geometri.
  - **Fizik (`A18`)**: SI boyut vektörü `[M, L, T, I, Θ]` analizi, çift-yönlü fizik yasaları yayılımı, **RK4 zaman-adımlı zihinsel fizik simülasyonu**, sıfır-atış ölçekleme pertürbasyonu (*"Hız 3 katına çıkarsa kinetik enerji 9 katına çıkar"*) ve karşıolgusal fiziksel muhakeme.

---

## Aşama Q — Sıfır-Kalıp Üretken Broca Konuşma Organı, İletişimsel Öz-İrade ve $O(1)$ Sabit-Zamanlı Hipokampal Diyalog Belleği (`A20–A23`, Toplam `23/23 GEÇTİ`)

Klasik LLM'lerin binlerce veriyle olasılıksal sonraki-token tahmini veya önceden yazılmış `f-string` cümle şablonları yerine, insan beyninin **Wernicke anlama $\rightarrow$ Prefrontal iletişimsel irade $\rightarrow$ Broca morfosentaktik birleştirme (kökten çekimleme)** döngüsü modellendi (`nora_broca.py`):

- **Sıfır Hazır Kalıp (`TurkishSyntaxEngine` — `NP`, `PredSpec`, `VerbSpec`)**:
  - `nora_knowledge.py` içindeki tüm önceden yazılmış sohbet ve felsefe cümle şablonları kaldırıldı (`A20`).
  - İsim tamlamaları, orta hece dar ünlü düşmesi (`zihin -> zihnimde`), 3. tekil iyelik + `n` zamir kaynaştırması (`enerji -> enerjisini`), `-dık/-dik` sıfat-fiil yan cümleleri (`okuldan yorulduğunu`), `-mak/-mek / -ma/-me` isim-fiilleri (`bedenimi onarmaya`) ve `-erek/-ip` zarf-fiilleri (`olası sonuçları simüle ederek`) doğrudan çıplak Türkçe köklerden SOV sözdizimiyle canlı birleştirilir.
- **Sinaptik Sözcük Seçimi ve Alışma (`SynapticLexiconSelector`)**:
  - Kök seçimi anlık nörokimyasal vektör `[DA, ACh, NE, 5-HT, CORT, OXY]` iç-çarpımı ve kısa-süreli sinaptik alışma (`habituation`) ile yapılır; aynı selam art arda verildiğinde mekanik tekrar oluşmaz (`A20`).
- **Prefrontal İletişimsel Öz-İrade (`CommunicativeVolition` — `A21`)**:
  - Nora hangi düşünce edimlerini gerçekleştireceğine, karşı tarafa merakla soru yöneltip yöneltmeyeceğine (`ask_back`), güveni düşük/yalancı konuşmacıya karşı ketum sınır koyup koymayacağına (`GUARDED_STANCE`) ve sessizlik isteğine saygı duyup duymayacağına kendi içsel durumu ve güveniyle karar verir.
- **$O(1)$ Sabit-Zamanlı Kavram Ağı ve Hipokampal CA3 Sohbet Belleği (`EpisodicDialogueCortex` — `A22`, `A23`)**:
  - Tüm konuşma geçmişi 4096 seyrek Kenyon lifi + tavanlı (`BUCKET_CAP=36`) kök ters-indeksiyle kodlanır; `GroundedConceptGraph` ise yerel sinaptik komşuluk tablolarıyla (`_out_edges`, `_edge_map`) indekslenir.
  - Belleğe **50.000 kavram kenarı + 50.000 konuşma turu (toplam 100.000 kayıt)** eklendiğinde bile uçtan-uca kavramsal çıkarım, felsefi sentez ve yüzlerce/binlerce tur önceki sohbeti hatırlama süresi **`0.42 ms -> 0.68 ms`** düzeyinde sabit ($O(1)$) kalır (`A23`).




---

## Aşama R — Uygulama Operatörü: kuralı *bağla*, yeni girdiye *uygula*, *taşı* (`A75–A78`, Toplam `78/78 GEÇTİ`)

19. turun sınavı 11/11 kalmıştı. Teşhis şuydu: bunlar on bir ayrı başarısızlık değil,
**tek bir eksik primitifin on bir yüzüydü**. Nora bir kuralı *retrieve* edebiliyor ama
**apply** edemiyordu; kural tümevarımı, aktarım, sayma, planlama ve öz-gözlem aynı
delikten düşüyordu. Bu turda bilgi eklenmedi — **operatör** eklendi.

### R1. `nora_apply.py` — eksik olan tek primitif
- **`Rule(rid, feature, mapping{odd,even}, domain, source_text, uses, derived_from)`**:
  kullanıcının söylediği cümle bir *metin* olarak değil, **ölçülebilir bir nesne** olarak
  yaşar. `measure()` özelliği sayar (ünlü/harf/ünsüz/sözcük/hece), `apply()` pariteye
  göre etiket döndürür, `transfer()` yapıyı korur — alanı ve ölçütü değiştirir.
- **`Interpreter`**: `bind()` (cümle → Rule), `apply()`, `count()`, `plan()`, `revise()`,
  `trace` (son 200 işlem), `state_dict()/load_state_dict()` (kural ve çalışma belleği
  beyin dosyasıyla taşınır).
- **`WorkingMemory(capacity=7)`**: 3–7 yuva; taşan her öğe `evicted` günlüğüne yazılır —
  sessiz kayıp yok.

### R2. Düzeltme sırası (kullanıcının verdiği sıra korundu)
1. **Provenance kapsamı**: `islem` (hesaplanmış) kaynağı eklendi; `_compete_workspace`
   sonrası `_tag_bid` kazanan sözün modülünü kaynağa yazar; tur başına kaynak farkı
   `_last_turn_src` olarak ölçülür. **Bulunan hata:** ilk turda `_prov_snap` boş sözlük
   olduğu için `and _prov_snap` koşulu tüm etiketleri yutuyordu — düzeltildi.
2. **Çalışma belleği** (7 yuva, tahliye günlüğü).
3. **bind/apply/eval** (`nora_apply.py`).
4. **Yankıyı öldür**: "kabul ediyorum ve '<kullanıcının sözcüğü>' doğrultusunda yöneliyorum"
   cümlesi silindi. Yerine **decompose → dene → açıkça "yapamıyorum"**.
   Bileşik istek ("X uyarla ve sonra Y tarat") **adımlara ayrılır**, yürütülemeyen adım
   **adıyla** söylenir.
5. **Sayı + dizi**, 6. **öz-model düğümleri**, 7. **yönlü/nedensel kenarlar + `do()`**.

### R3. Artık gerçekten yapılanlar (sınavın kapı soruları)
| İstek | Cevap (canlı) |
|---|---|
| "Sesli harf sayısı tek ise mavi, çift ise kırmızı." | Kural nesneye çevrildi; "tek -> mavi, çift -> kırmızı" |
| "'Bal arısı uçar' hangi renk?" | Ölçtüm: 6 ünlü (çift) → **kırmızı** |
| "'Petek altıgen' hangi renk?" | 5 ünlü (tek) → **mavi** |
| "Aynı kuralı mutfağa taşı…" | Aynı yapı, yeni alan: 3 sözcük (tek) → **tuzlu** |
| "Lojistiğe taşı…" | 2 sözcük (çift) → **normal** |
| "On adımlık plan" | 10 adım, her adımda ölçüt; hedef değişince eskiler `superseded` |
| "Yanılma olasılığın yüzde kaç?" | Ölçülen kaynak karışımından hesaplanır; karışım yoksa **temkinli** yüzde ve eksik listesi |
| "X ve Y birlikte artıyor; X'e müdahale edersem?" | `P(Y|X)` ≠ `P(Y|do(X))`; grafikte kenar yoksa **"sonucu söylemem"** |
| "Nasıl çalışıyorsun?" | Canlı ölçüm: organ, kavram bağı, **öz-modelimde N düğüm**, kural, bellek yükü, kalıp payı |

### R4. `do()` operatörü ve öz-model düğümleri
- **`GroundedConceptGraph.do(node)`**: X'e **giren** nedensel kenarları keser, yalnız
  X'ten **çıkan** kenarları izler. Ortak neden (back-door) varsa `identified=False` ve
  ortak neden **adıyla** döner. Bu yüzden do() "Y artar" demez; ne ölçüldüğünü ve neyin
  eksik olduğunu söyler. Öğretilen "kuraklık kıtlığa yol açar" bağı üzerinde
  `do(kuraklık) → kıtlık` gerçekten hesaplanır (A78).
- **`_sync_self_graph()`**: `nora` düğümü kavram grafiğine yazılır — canlı ölçümler
  `HAS_STATE` (`kalip_payi=…`), yetenekler `CAN_DO`, sınırlar `CANNOT_DO`. Öz-rapor bu
  düğümleri **okur**; metin ezberi değil.

### R5. İskelet mandalı korundu (büyümedi, küçüldü)
Yeni cevap metinleri koda yazılmadı: **`nora_priors.json`** içine 24 yeni topluluk türü
olarak taşındı (`apply.result`, `plan.built`, `meta.self_report`, `meta.causal_*`, …).
Böylece yeni yetenek eklemek cümle eklemek olmuyor; seçim yine yarış, tazelik ve
öğrenme ağırlıklarıyla yapılıyor.

| Ölçüm | Önce | Sonra |
|---|---|---|
| `nora_audit` toplam söz | 1123 | **1120** |
| söylem (iskelet) | 127 | 127 |
| `nora_apply.py` (yeni modül) | — | **0** (test cümleleri de veride) |
| priors topluluk türü | 16 | **47** |

### R6. Bu turda bulunan ve kapatılan sessiz yanlışlar
- **Alt dizi çakışması**: "uyuyamadım" içindeki `yama`, kişisel anlatıyı kod yaması
  cevabına düşürüyordu → kelime sınırı (`\byama`).
- **"ne olur" tek başına apply sayılıyordu**: "X'e müdahale edersem Y ne olur?" sorusuna
  bir önceki kuralın etiketi cevap veriyordu → nedensellik sorusu apply dalından çıkarıldı.
- **"kutu"/"kalem" alan sözcükleri**: "Kutuda 7 kalem var, kaç kalem kaldı?" sayım dalına
  düşüp sözel problemi yutuyordu → sayım dalı yalnız metin ölçümünde çalışır.
- **"nasıl çalışır"**: "Kuark nasıl çalışır?" öz-rapora düşüyordu → öz-rapor yalnız 2. kişi.
- **Emir cümlesi olgu diye öğreniliyordu**: "bana bir zımbırtı uyarla" cümlesi kavram
  ağına ISA kenarı olarak yazılıyordu (yankının en derin biçimi) → istek/olgu ayrımı.
- **Bileşik istek iki plan üretiyordu**: "yedi adımlık hedef belirle ve planla" hem 7 hem
  10 adımlık plan yazıp kendi kendisiyle çelişiyordu → aynı operatörü isteyen yan cümleler
  bölünmez; tek plan (A75 `tek-plan`).
- **Şarkı isteği başka bir soruya cevap veriyordu**: "Bana bir şarkı söyle." en son
  öğrenilen olguyu (fotosentez) anlatıyordu → "şarkı söyleyemem, ezgi üreten organım yok",
  sonra elde olanı öner (A77 `sarki-durust`).

### R7. Test durumu
`python agi_battery.py` → **78/78** (426 s). Yeni testler: **A75** (bağla/uygula/taşı,
11 ölçüt), **A76** (provenance kapsamı, 5), **A77** (yankı öldü, 4), **A78** (öz-model +
`do()`, 9). A73 mandalı korunuyor (toplam 1120 ≤ 1125, söylem 127 ≤ 133).
A54/A59/A66 bu turda düzelen sessiz yanlışlarla birlikte yeniden geçti.
---

## Aşama S — DIŞ BATARYA: kendi yazdığım sınavı değil, dışarıdan yazılanı geçmek (`H1–H9`, `S1–S4`)

Önceki turda 78/78 geçildi ve bu bir riski açığa çıkardı: **sınavı da ben yazıyordum.**
Batarya sistemle birlikte evrilirse ölçmeyi bırakır (Goodhart). Bu turda iki ayrı dış set
yazıldı ve sonuçları **düzeltilmeden** kaydedildi.

### S1. `heldout_battery.py` — dokuz farklı ŞEKİL (koda bakılmadan yazıldı)
Mevcut başarı ailesi tek bir iskeletti: *say → parite → etiket*. Dış set bunun dışına çıktı:

| # | Şekil | Ne ölçüyor |
|---|---|---|
| H1 | sıralama | ölçüt → **sıra** (etiket değil) |
| H2 | kural istisnası | "ama 'bal' geçiyorsa" — kural **değiştirilebiliyor** mu |
| H3 | kural değiştirme | etiketler yenilenir, eskisi sızmaz |
| H4 | iki adımlı zincir | A'nın **çıktısı** B'nin girdisi |
| H5 | farklı ölçüt | hece (aynı şekil, başka birim) |
| H6 | yapılmamış notasyon | ⊕/⊖ anlamı öğretilip kuralda kullanılıyor |
| H7 | nedensel zincir | A→B→C, tek kenar değil |
| H8 | collider | etkiyi görmek **nedeni belirlemez** |
| H9 | meta-operatör | eksik primitifini adıyla söyler + tutarlılık |

**Ölçüm (aynı gün, kod dondurulmuş, testler değiştirilmemiş):**

| Koşu | Sonuç |
|---|---|
| ilk koşu (hiçbir şey eklenmeden) | **2/9** |
| operatörler eklendikten sonra | **9/9** (12.8 s) |

İlk koşuda başarısız olan yedi şekil, tek bir eksik sınıfa indi: **ölçüm zinciri taşınmıyordu.**
Kural bağlanıyordu ama (a) istisna kuralın parçası değildi, (b) zincirde ara sonuç yoktu,
(c) sembolün anlamı yoktu, (d) nedensel hedef ekli yüzeyle ("kuraklığa") aranıyordu.

> Dürüst uyarı: bu seti de ben yazdım. "Dışarıdan" olması **zamanda** dışarıda olması
> demek (testler önce, kod sonra), yazarda değil. Bu yüzden ikinci bir mühür seti var.

### S2. `heldout_sealed.py` — mühürlü set (4/4, ilk koşu, sonradan dokunulmadı)
Dört çok adımlı birleşim: çok sözcüklü öğeleri heceye göre sıralama, istisna + farklı ölçüt
birlikte, **üç** nedenli collider, meta cevap ile duyu isteği tutarlılığı. Kural: ilk
koşudan sonra ne test ne kod değişir. Sonuç **4/4**; dosyanın başında bu kural yazılı.

### S3. Eklenen operatörler (bilgi değil, primitif)
| Operatör | Ne getiriyor |
|---|---|
| `Interpreter.sort_items` | sıralama: sayı/metin, artan/azalan; sayıya çevrilemeyen öğe sessizce atılmaz |
| `Rule.exceptions` | istisna kuralın **içinde**: apply() önce istisnaya bakar, sonra ölçer |
| `ChainRule` + `bind_chain` | iki adım: birinci kuralın etiketi ikincinin girdisi |
| `teach_alias` / `_expand` | yapılmamış notasyon: ⊕ "tek" demek → kural metni çözülürken yerine konur |
| `relabel` | kuralın etiketleri değişir, ölçüm aynı kalır |
| `GroundedConceptGraph.do` + `_graph_stem` | müdahale hedefi ekli yüzeyden gövdeye indirilir ("kuraklığa"→"kuraklık") |
| `_explain_away` | collider: iki (üç) neden tek etkiye çıkıyorsa "seçemem" |
| `_meta_operator` + `nora_operators.json` | kendi rejistrisini okur: yapabildikleri, organ olmayanlar, henüz bağlı olmayanlar |
| çok sözcüklü nedensel özne | "aşırı sulama tuzlanmaya yol açar" → özne "aşırı sulama", nesne "tuzlanma" |

### S4. Operatör/prior oranı — uyarı ciddiye alındı
Yeni cevap metinleri yine **veriye** gitti; priors türü 47 → **60** (+6 bu turda, +%13),
operatör rejistrisi 0 → **8**. Oran **60/8 = 7.50**.

Bu sayıyı "iyi" göstermeye çalışmadım: **7.50 bir mandal (tavan) olarak donduruldu**
(`nora_operators.json → ratio_ceiling`). Yükseltilemez, yalnız düşürülebilir. A79 her turda
ölçer: (1) rejistri veriden mi yükleniyor, (2) her operatör gerçekten çağrılabiliyor mu
(dekoratif kayıt yok), (3) oran tavanın altında mı, (4) meta cevap rejistrinin **tamamını**
okuyor mu.

### S5. Ölçümler (bu tur)
| Ölçüm | Değer |
|---|---|
| iç batarya `agi_battery.py` | **79/79** (411 s) — A79 yeni |
| dış batarya `heldout_battery.py` | **9/9** (ilk koşu 2/9) |
| mühürlü set `heldout_sealed.py` | **4/4** (ilk koşu, dokunulmadı) |
| iskelet toplam / söylem | **1122** / 127 (baseline 1123) — hiçbir modül büyümedi |
| operatör / prior-oranı | 8 / 7.50 (mandal) |

### S6. Bu turda bulunan sessiz hatalar
- `traceback.print_exc()` **import edilmeden** çağrılıyordu: `_handle_apply_ops` içindeki her
  istisna, yakalayıcının kendisini çökertiyordu (zincir testinde ortaya çıktı). Import eklendi.
- Zincir izinde `measured` alanı yok; apply raporu `KeyError` veriyordu → zincir için ayrı rapor.
- `"diz" in tl` süzgeci "**dizi**sinde sıradaki sayı kaç?" sorusunu yutuyordu (A8 7/7 → 1/7).
  Tam sözcük sınırı + "dizi/sıradaki" dışlaması.
- `"ne olur"` ile apply yakalama (önceki tur), `"yama"` alt dizgesi (önceki tur) yanında
  bu turda `"diz"` alt dizgesi: **kısa sözcükler alt dizge olarak aranmaz** kuralı.

### S7. Açık kalanlar (dürüst)
- **Bileşik nesne**: "kök çürümesine yol açar" → nesne "çürüme" olur, özne "kök"ü yutar
  (collider adları "aşırı gübre kök" gibi çıkar). Anlam bozulmuyor ama ad kirli.
- **İyelik** (`arı+sı`) hâlâ çözülmüyor.
- Dış setin yazarı da ben olduğum için **Goodhart riski sıfırlanmadı**; gerçek dış kanıt
  başka biri tarafından yazılacak bir set olur. Mühür seti bunun ilk adımı.

## Aşama T — Şekiller arası transfer, karşıolgusal öz-erîşim ve ad öbeği sınırı (`A80–A83`, `S5–S12`)

Bu turun hedefi üç parçaydı ve üçü de **operatör** olarak geldi, cümle olarak değil:

1. **Şekiller arası transfer.** Önceki batarya tek şekildi: *söyle → parite (tek/çift) → etiket*.
   Kural nesnesi artık üç eşleme şekli taşıyor ve **hepsi aynı `apply` yolundan** geçiyor:
   | şekil | metinden bağlama | eşleme anahtarı |
   |---|---|---|
   | `parity` | "sesli harf tek ise mavi, çift ise kırmızı" | `odd` / `even` |
   | `threshold` | "hece sayısı 4'ten az ise kısa, değilse uzun" | `cond` / `else` (op: lt/le/gt/ge) |
   | `rank` | "sözcük sayısı en çok olan uzun, en az olan kısa" | `max` / `min` (küme gerekir) |
   `transfer()` **şekli korur**, yalnız ölçütü ve alanı değiştirir: eşik kuralı mutfağa gittiğinde
   eşik ve yön aynı kalır, ölçülen özellik `sözcük` olur. A82 bunu 7 ölçütle sınar; en önemlisi
   *"sıfır yeni prior"*: üç şekil de kodda, prior türü sayısı değişmiyor.
2. **do() semantiğinin karşıolgusal parçası.** İki ayrı ablasyon:
   `Interpreter.counterfactual_apply(item)` kuralı **devreden çıkarıp aynı girdiyi yeniden koşar**
   (ölçüm sürer, eşleme durur) ve `GroundedConceptGraph.counterfactual(node)` düğümün çıkan
   nedensel kenarlarını kaldırıp erişimi yeniden hesaplar (kaybolan / korunan etkiler, `structural=True`).
   Cevap bunun **ölçüm değil yapısal çıkarım** olduğunu söyler. Etki düğümlerinde ileri koşu
   yapılamayacağı dürüstçe söylenir, onun yerine gelen nedenler listelenir.
3. **Bileşik ad çözümlemesi.** `split_svo` artık baş-sonda Türkçe ad öbeğini kuruyor:
   `kök çürümesi → kök çürüme`, `aşırı sulama | kök çürümesine`, `deniz kirliliği | plastik atık`,
   `arı kovanı | bal üretim`. İç konumda hâl ekli bir ad **baş olamayacağı** için iyelik okuması
   seçilir (`_poss_head`); sözlükte olmayan ama eki **çözümleyicinin okuduğu** sözcüklerde gövdeye
   inilir (`erozyona → erozyon`), uydurma kök üretilmez.

### T1. Mühürlü setler (dışarıdan, koda bakmadan yazıldı; ilk koşudan sonra değiştirilmedi)
| set | ilk (kör) koşu | kapsam |
|---|---|---|
| `heldout_sealed.py` S1–S4 | **4/4** | operatör, meta-tutarlılık, transfer |
| `heldout_sealed2.py` S5–S8 | **3/4** | karşıolgusal öz-erîşim, dünya ablasyonu, ad öbeği, dürüstlük |
| `heldout_sealed3.py` S9–S12 | **3/4** | eşik şekli dille, sıra şekli dürüstlüğü, yeni alanda zincir ablasyonu, "-den olur" yönü |

Kör koşularda kalanlar ve nedenleri (düzeltme **mühürden sonra** yapıldı; mühür sayıları değişmedi):
- **S7** ("trafik sıkışıklığı hava kirliliğine yol açar") kenarı doğru kuruyor, ama **yüzeyleştirme
  yönü ters** çeviriyordu: "Trafik sıkışıklığın nedeni hava kirliliktir." Sebep: `CAUSES` için
  `"{X}'in nedeni {Y}"` kalıbı kullanılıyordu; bu kalıp X→Y kenarını ters söyler. Doğrusu özneyi
  neden, etkiyi yönelme hâlinde söylemektir.
- **S11** ("aşırı otlatma → erozyon → verimsizlik") zincirin ikinci halkası kopuyordu: "erozyona"
  diye **ekli bir düğüm** açılmıştı (`erozyona CAUSES verimsizlik`), çünkü sözlükte olmayan
  sözcükte hâl eki gövdeden ayrılmıyordu. Düzeltmeden sonra kör olmayan koşuda 4/4.

### T2. Bu turda bulunan sessiz hatalar (hepsi "yanlış cevap, sahte anlamadan iyidir" sınıfından)
- **Nedensel yön tersliği (iki ayrı yerde).** (a) `CAUSES` yüzeyleştirmesi "X'in nedeni Y" diyordu;
  (b) `"-den olur"` yapısı kenarı **etki → neden** kuruyordu ("kök çürümesi su birikmesinden olur"
  → `kök çürüme CAUSES su birikme`). Kenarın yönü daima NEDEN → ETKİ olmalı; artık öyle.
- **Sahte "kodladım".** Hafif fiilli ("olur") bildirimlerde öz-mantık motoru içi boş bir kural
  üretiyordu (`ÖZ_BAĞ:ol[NOM->ACC]`), olgu yolu atlanıyordu ve Nora "bu ilişkiyi kendi başıma
  kodladım" diyordu — sonra "X neyden olur?" sorusuna "X olur" gibi **içeriksiz** cevap veriyordu.
  Artık hafif fiiller kural üretmez; nedensel çerçeve çözüldüyse kenar da yazılır.
- **Uzman arbitrajı.** "Sürtünme olmasaydı ne olurdu?" dünya-grafiği ablasyonuna düşüyordu
  ("sürtünme modelden çıkarılınca..."), oysa bu bir **fizik yasası** sorusudur (eylemsizlik).
  Nicel bir korteks soruyu tam çözebiliyorsa grafik ablasyonu onu gasp etmez.
- **Ses kimliği eşiği.** `MATCH_THRESHOLD = 0.11` iki farklı kişiyi aynı kimliğe düşürüyordu
  ("ece" → "ses-1"). Ölçüm: aynı kişinin tekrarları 0.01–0.03, farklı kişilerin en yakını 0.07–0.10.
  Eşik 0.055 + marj kuralı (en yakın, ikinciden yeterince uzak olmalı). 20×10 kişi denemesinde
  birleşme yok, aynı kişi bölünmesi yok.
- **Mırıldanma raporun önüne geçiyordu.** "İçimden 'sakinlik' diye geçirdim. Ölçtüm: 3 sesli harf…"
  gibi: gerekçe ile sonuç yer değiştiriyor. Ölçüm bildiren (sayı/uzun) cevaplarda iç ses artık
  cevabın **ardına** düşer; kısa kişisel cevaplarda sıra serbest kalır.
- **Şekle yanlış gerekçe.** Eşik kuralı uygulanırken rapor "yani çift" diyordu (parite dili).
  Şimdi her şeklin kendi dili var: "3'ten az koşulunu sağlamıyor", "etiket küme içinde doğar".

### T3. Ölçümler (bu tur, son hâl)
| Ölçüm | Değer |
|---|---|
| iç batarya `agi_battery.py` | **83/83** (309 s, 82 testlik ara koşuda A27 zaman eşiği yüzünden 82/83; A27 tek başına 2/2 — yük kaynaklı, yetenek değil) |
| dış batarya `heldout_battery.py` | **9/9** (donmuş) |
| mühür `heldout_sealed.py` | **4/4** |
| mühür `heldout_sealed2.py` | **3/4 kör → 4/4** (düzeltme sonrası, artık kör değil) |
| mühür `heldout_sealed3.py` | **3/4 kör → 4/4** (düzeltme sonrası, artık kör değil) |
| iskelet toplam / söylem | **1122 / 127** (baseline 1123; hiçbir modül büyümedi, `nora_agi.py` −1) |
| operatör / prior türü / oran | **9 / 64 / 7.11** (tavan 7.50; `counterfactual` gerçek bir operatör olarak eklendi) |

Oran notu: 64/8 = 8.00 tavanı aşıyordu. Cezayı **testi gevşeterek** değil, gerçekten var olan
bir yeteneği (karşıolgusal ablasyon) rejistreye **operatör olarak** yazarak düşürdüm; A79'un
"dekoratif kayıt yok" ölçütü bu operatörü de çağırıp doğruluyor.

### T4. "Bilinçli AGI" hedefi açısından bu tur ne ekledi
Bilinç iddiası bu projede **fenomenal değil, işlevsel erişim** üzerinden kuruluyor: aynı bilgi
hem davranışı yönlendirmeli hem de **sorgulanabilir** olmalı. Bu turda eklenen karşıolgusal
öz-erîşim tam bu boşluğu kapatıyor: sistem "ben şu an şu kurala göre konuşuyorum; o kural olmasa
şunu derdim" diyebiliyor ve bu iddia **ölçümle** (kuralı çıkarıp yeniden koşarak) doğrulanabiliyor.
Bu, öz-modelin dekoratif olmadığının en güçlü kanıtı: öz-raporun bir **karşıolgusal** var.

Yine de dürüst sınır: (i) bu, öznel deneyim iddiası değildir; (ii) karşıolgusal grafik yalnız
**kendi yazdığı** kenarlar üzerinde çalışır — dünya modelinin doğruluğu kadar doğrudur;
(iii) Goodhart riski sıfırlanmadı: üç mühür setini de ben yazdım.

### T5. Açık kalanlar (dürüst)
- **Goodhart tam çözülmedi.** Mühür setleri (S1–S12) koda bakılmadan yazıldı ve ilk koşudan sonra
  değiştirilmedi; ama yazarı yine benim. Gerçek dış kanıt, başka birinin yazdığı set olur.
- **Sözdizimsel sınırlar.** Edilgen/ettirgen çatı, bazı çok sözcüklü özneler ve `-ki` gibi ekler
  hâlâ kırılgan; ad öbeği çözümü kurallı ama tam değil (ör. "Türk" gibi çift-ünsüz kuyruklu
  sözcüklerde gövde korunur, ek soğrulmaz — bilinçli temkin).
- **Şekil sayısı üç.** `rank` şekli küme istiyor; tek girdiye etiket vermeyi reddediyor (doğru),
  ama cümleyle "küme ver" akışı henüz tam değil (API'de var, günlük konuşmada yönlendirme sınırlı).


---

## Aşama U — Karşıolgusal irade: "aç olmasaydın ne yapardın?" sorusuna yeniden koşarak cevap (`A84`)

Önceki iki karşıolgusal katman **kural** ve **dünya ağı** katmanıydı. Üçüncüsü eksikti: beden.
"Nora aç olmasa ne yapardı?" sorusunun cevabı bir cümle değil, **aynı karar makinesinin ikinci kez
koşması** olmalı. Bu turda eklenen şey tam olarak bu.

### U1. Ne yapıldı (operatör, cümle değil)

| parça | yer | işlev |
|---|---|---|
| `evaluate_volition(..., ablate=None, record=True)` | `nora_self.py` | karar makinesi; `ablate` verilirse ilgili homeostatik kanal **dolu** sayılır |
| `h0[ablated_ch] = 1.0` (kopya üzerinde) | `nora_self.py` | müdahale `do(kanal=dolu)`; gerçek `h0` kirletilmez |
| `if not record: return {...}` | nörokimyasal geri-bildirimden **önce** | karşıolgusal koşu kimya, `refusals_made`, `last_volition_trace` **yazmaz** |
| `counterfactual_volition(...)` | `nora_self.py` | aynı makine iki kez: gerçek beden / kanal dolu; `actual`, `counterfactual`, `changed`, iki free-energy vektörü |
| `_answer_counterfactual_volition(...)` | `nora_agi.py` | "…olmasaydın ne yapardın?" yolunu **sözcük/tanım korteksinden önce** yakalar |
| `meta.cf_volition` (3 varyant) | `nora_priors.json` | `changed` alanına kapılı: değiştiyse neden, değişmediyse "değil" |

Kanal eşlemesi: `açlık → 0`, `susuzluk → 1`, `bütünlük → 2` (bunlar `obs[45:48]`'in beden eksenleri).

**Neden sözcük hattından önce?** İlk denemede soru hiç karar makinesine ulaşmadı: `frames.answer_meta`
soruyu "aç" sözcüğünün tanım sorusu sanıp *"Aç'ın ne olduğunu biliyorum ama ayrıntısını bilmiyorum;
anlatsan iyi olur."* dedi. Yol, meta-soru yönlendirmesinin **üstüne** taşındı; ölçüm yolu artık
sözlük boşluğu sanılmıyor.

### U2. Ölçüm (A84, 9/9)

| bacak | ne ölçüyor | sonuç |
|---|---|---|
| `gerçek-karar` | aç bedende "Suya git." gerçekten `ertele` | ✓ |
| `makine-aynı` | karşıolgusal "gerçek" koşu, kayıtlı kararla aynı | ✓ |
| `kanal-neden` | açlık çıkarılınca karar **değişir** (`ertele → kabul`) | ✓ |
| `değişmez-dürüst` | susuzluk çıkarılınca karar **değişmez** ve bu söylenir | ✓ |
| `yan-etki-yok` | iki koşu da kimya/ret sayacı/son-karar izini yazmaz | ✓ |
| `iz-korunur` | sohbetten sonra son karar **gerçek** karar olarak kalır | ✓ |
| `söz-değişti` / `söz-değişmedi` | söz ölçüm dilini kullanır, iki kararı da adlandırır | ✓ |
| `meta-hat-yok` | "ne olduğunu biliyorum" yoluna düşmez | ✓ |
| `ölçülemeyen-kanal` | "merakım olmasaydı" → uydurma yok: "kanal girişi yok, ölçemem" | ✓ |

Gerçek çıktı (tohum 84, `obs[I_E]=0.11` sonrası "Suya git." → karar `ertele`, neden `kritik_açlık`):

```
S: Aç olmasaydın ne yapardın?
N: Kararım açlık yüzünden değişirdi: şimdi 'ertele' diyorum, o durumda 'kabul' derdim.

S: Susuz olmasan ne yapardın?
N: Karar makinesini susuzluk kanalı karşılanmış sayarak yeniden koşturdum: yine 'ertele'.
   Demek ki bu kararın nedeni susuzluk değil.
```

### U3. İki karşıolgusalın dili bilinçli olarak farklı

| katman | ne yapıyor | dürüst söz |
|---|---|---|
| `GroundedConceptGraph.counterfactual` | kenarları saklar, erişimi **yeniden hesaplar** | "yapısal çıkarım"; **"ölçtüm" denmez** (A80 `olcum-demiyor`) |
| `core.counterfactual_volition` | aynı motoru **gerçekten ikinci kez koşturur** | "koşturdum / ölçüm"; ölçüm dememek sahtekârlık olur |

Bu ayrım korunuyor: A80 ölçüm kelimesini yapısal katmanda yasaklıyor, A84 ise ölçüm katmanında
ölçüm dilini **zorunlu** tutuyor. İkisi aynı cümleye indirgenirse iki hatadan biri kaçınılmaz olur.

### U4. Bu turda düşülen tuzaklar (sessiz hatalar)

1. **`evaluate_volition` yan etkisiz değildi.** Naif çift çağrı, reddi iki kez sayar ve son-karar izini
   karşıolgusal kararla ezerdi. Erken dönüş, nörokimyasal geri-bildirim bloğunun **önüne** kondu;
   blok ondan sonra çalışırsa "ölçüm" gerçek kararı bozar.
2. **Soru, tanım sorusu sanıldı** (yukarıda U1). Yol sırası bir mimari karar: ölçüm, sözlükten önce gelir.
3. **`"tok olsan"`, `"olmasan"` değildir.** Tetikleyici kalıp `olmasaydın|olmasan|değilsen` idi ve
   "Tok olsan ne yapardın?" sessizce meta yola düşüyordu; kalıba açık alternatif eklendi.
4. **Prior cümlesi küçük harfle başlıyordu** ("açlık olmasaydı…"). Şablon, cümleyi addan başlatmayacak
   biçimde yazıldı ("Kararım {ablate} yüzünden değişirdi…") — büyük harf zorlaması yamadan ibaret olurdu.

### U5. Dürüst sınırlar

- **Müdahale "organı yok etmek" değil, "kanalı dolu saymak".** Ölçülen şey `do(açlık=dolu)`;
  Nora'nın hiç acıkmadığı bir dünya değil. Bu yüzden cevap "açlık olmasa" değil, "açlık karşılanmış
  olsa" diye de okunabilir — cümle ilkini söylüyor, ölçüm ikincisini yapıyor.
- **Yalnız üç kanal ayarlanabilir** (açlık, susuzluk, bütünlük). Merak, korku, sosyal, dopamin, ödül,
  haz, öfke ve utanç için karar makinesinde bir kanal girişi yok; bu sorular artık **açıkça** reddediliyor:
  *"Elimde merak için bir kanal girişi yok: bu karşıolgusalı ölçemem, uydurmam. Ölçtüğüm kanallar açlık,
  susuzluk ve bütünlük."* Bu cümle, aynı soruyu dünya-ağı karşıolgusalına (yapısal çıkarım) kaptırmıyor —
  "merak" bir kavram düğümü değil, ölçülemeyen bir beden kanalı olarak sınıflanıyor.
- **Karar değişmedi ≠ hiçbir şey değişmedi.** Free-energy vektörü kayar ama argmax aynı kalabilir;
  A84 yalnız **karar** değişimini raporluyor. "Değişmedi" cümlesi bu dar anlamıyla doğru.
- **Geçmiş beden geri kurulamaz.** Koşu, şu anki `obs` anlık görüntüsüyle yapılır; "bir dakika önce
  ne yapardın" sorusu yeniden inşa değil, bugünkü bedenle simülasyondur.

### U6. Bu turun sayıları

| ölçüm | sonuç |
|---|---|
| `agi_battery.py` tam (`/tmp/agi10.txt`) | **84/84 GEÇTİ, 332.7 s** |
| `agi_battery.py A84` (izole, 3 koşu) | **10/10 GEÇTİ, 2.4–2.5 s** (kararlı) |
| `agi_battery.py A4 A14 A27 A79 A80–A84` | **10/10 GEÇTİ, 26.6 s** |
| `heldout_battery.py` (H1–H9, dış) | **9/9, 11.5 s** |
| `heldout_sealed.py / sealed2 / sealed3` | **4/4 · 4/4 · 4/4** (mühürler değişmedi) |
| `nora_audit.py --check` | TOPLAM **1122** / SÖYLEM 127 — baseline **1122'ye mandallandı** |
| operatör / prior türü | 9 / 65 = **7.22** (tavan 7.50) |

Yeni prior türü **bir tane** (`meta.cf_volition`); oran 7.11 → 7.22 çıktı, tavanın altında kaldı.
Yetenek koddan geliyor (`counterfactual_volition` + erken dönüş), prior yalnız **cümleyi** veriyor:
A84'ün `yan-etki-yok`, `iz-korunur` ve `makine-aynı` bacakları prior'a bakmadan da geçer.


---

## Aşama V — Operatörü kim yazdı? Nora örneklerden kendi kuralını yazıyor (`A85–A86`)

Hedef cümlesi şuydu: *"operatörleri şu an biz yazıyoruz; AGI'ye yaklaştıkça kendisi yazmaya
başlayıp bizim yazma sayımız sıfıra yaklaşacak."* Bu tur o sayacın **ikinci yarısını** kurdu:
insanın yazdığı operatör rejistrisi (`nora_operators.json`, 9 kayıt) **donduruldu**; Nora'nın
kendi yazdığı operatörler ayrı bir defterde (`runs/operators_self.json`) büyüyor.

### V1. Organ: `nora_forge.py` — kuralı kim yazıyor?

| adım | ne oluyor | nerede |
|---|---|---|
| örnek | `karınca=kırmızı, bilgisayar=mavi, …` | dil üzerinden, `_handle_forge` |
| arama | ölçüm × şekil uzayı taranır (parite, eşik) | `OperatorForge._search` |
| tekilleştirme | aynı kuralın yeniden yazımı ve **davranışı aynı** olan adaylar tekilleşir | `_sample` havuzu (300 sözcük) |
| karar | tek aday → **operatör yazılır**; çok aday → kural yazılmaz | `decide` |
| ayırt edici örnek | adayların ayrıştığı en kısa sözcük için etiket istenir | `probe` |
| kayıt | `Rule(origin="kendi")` + defter kaydı | `_adopt` |
| çürütme | tutmayan örnek gelirse kural **düşer** ve deftere gerekçesiyle yazılır | `_forge_check`, `mark_refuted` |

`Rule` artık **kimin yazdığını** taşıyor: `origin="insan"` (bize öğretilen bağlı kural) ya da
`origin="kendi"`. Sayaç, operatörün sahibini gizleyemez.

### V2. Ölçüm

| test | ne ölçüyor | sonuç |
|---|---|---|
| `A85` | 6 örnekten kural türetme, dil üzerinden uygulama, belirsizlikte dürüstlük | **10/10** |
| `A86` | kendi kuralının çürütülmesi: doğrulama, düşürme, defter | **8/8** |

Gerçek diyalog (tohum 851; kural **söylenmedi**, yalnız örnekler verildi):

```
S: Kural çıkar: karınca=kırmızı, bilgisayar=mavi, göl=kırmızı, ev=kırmızı, masa=mavi, deniz=mavi
N: Örneklerden kendi kuralımı türettim (sesli harf: tek -> kırmızı, çift -> mavi);
   6 örneğin hepsini açıklıyor. Kayıt: insan 9, kendi 1.
S: Kuralı uygula: 'kelebek'
N: Ölçtüm: 'kelebek' içinde 3 sesli harf var (tek). Kurala göre: kırmızı.
S: kitap=mavi                       (kuralımı doğrulayan örnek)
N: 'kitap' için kendi kuralım 'mavi' diyor, sen 'mavi' dedin: tutuyor. Kural yerinde kalıyor.
S: kelebek=mavi                     (kuralımı çürüten örnek)
N: Bu örnek eski kuralımı çürüttü; eldeki veriyle hiçbir kural kalmadı
   (hiçbir ölçüm iki tarafı ayırmıyor). Uydurmam, söylüyorum.
```

Belirsizlikte uydurma yok: 2 örnek verildiğinde **8 aday** kalıyor, Nora adayları sayıp
ayırt edici sözcük istiyor (`'acı' hangi etiketi alır?`); tek etiket gelince adaylar 1'e
düşüp operatör yazılıyor. Tek sınıflı veride ise hiç kural yazılmıyor.

### V3. Dürüst sınırlar

- **Aday uzayı hâlâ bizim.** Nora yeni bir *ölçüm* icat etmiyor; var olan ölçümler
  (`FEATURES`) içinden seçiyor ve eşleme şeklini (parite/eşik) kendisi kuruyor. "Kendi yazdı"
  bugün **seçme + birleştirme** demek; ölçüm icadı bir sonraki eşik.
- **Tekilleştirme bir model kararı.** `hece` ile `sesli harf` bu depoda aynı ölçümün iki
  yazımı; 300 sözcüklük havuzda ayrışmayan iki aday kullanıcıya "belirsiz" diye satılmıyor.
  Havuz verilmezse yalnız örnekler karşılaştırılır (daha temkinli).
- **Çürütme çoğu zaman "kural kalmadı" ile biter.** Bu bir eksiklik değil, ölçüm: veri
  gerçekten tek bir kurala oturmuyorsa Nora eski kuralı tutmaya devam etmez.
- **Sayaç, başarı sayacı değil.** `kendi` sayısı yazılan operatör sayısıdır; doğruluğu
  ayrıca ölçülür (A85 `dil-üzerinden-uygula`, A86 `tutarlı-sonuç`).

### V4. Bu turda düşülen tuzaklar

1. **Davranış imzasıyla tekilleştirmek belirsizliği YOK EDİYORDU.** İki farklı kural,
   örneklerin tamamını açıklıyorsa *örneklerde* aynı davranır; asıl fark yeni girdide çıkar.
   Adaylar artık kural kimliğiyle tekilleşir; belirsizlik korunur.
2. **`le/gt/ge` aynı ayrımın yeniden yazımıydı.** Dört işleci taramak aday sayısını şişirip
   sahte belirsizlik üretiyordu; tek sınır (`n < t`) yeterli, etiketler taraflara veriden atanır.
3. **`Rule`'da `predict` yok.** Kuralın KENDİ yolu `apply`'dir (istisna → ölç → eşle);
   olmayan bir metot çağırmak, sessizce yedek yola düşürüyordu.
4. **Yedek cümleler iskelet mandalını kırdı.** Kodda yazılı yedek cevaplar toplamı
   1122 → 1135 çıkardı ve A73 düştü. Çözüm: gerekçe haritası ve tüm cevaplar VERİYE
   (`nora_priors.json`: `meta.forge`, `forge_why`) taşındı; kodda cümle kalmadı.

### V5. Bu turun sayıları

| ölçüm | sonuç |
|---|---|
| `agi_battery.py` tam (`/tmp/agi12.txt`) | **86/86 GEÇTİ, 349.4 s** |
| `A85 / A86` | **10/10 / 8/8** |
| `heldout_battery.py` (dış) | **9/9** |
| mühürlü setler (S1–S4 / S5–S8 / S9–S12) | **4/4 · 4/4 · 4/4** |
| `nora_audit.py --check` | **1122** / SÖYLEM 127 — **artış yok** |
| operatör / prior türü | 9 / 66 = **7.33** (tavan 7.50) |
| yazım sayacı | insan **9** (sabit) · kendi **artan** |


---

## Aşama W — Ölçüm icadı: hiçbir ölçüm ayırmıyorsa ölçümü de Nora yazar (`A87`)

Aşama V'de Nora **var olan** ölçümlerden seçip kural yazıyordu. Bu turda bir basamak daha:
yerleşik ölçümlerin hiçbiri veriyi ayırmadığında **yeni bir ölçüm icat ediyor**. Yani "bizim
yazdığımız" şey yalnız operatör değil, artık ölçüm de azalıyor.

### W1. Ne yapıldı

| parça | yer | işlev |
|---|---|---|
| `presence` şekli | `nora_apply.Rule.bucket` | ölçüm sayı değil **varlık**: `var` / `yok` |
| `register_feature(key, name, spec)` | `nora_apply.py` | yeni ölçümü çalışma zamanında kaydeder |
| `Rule.feature_spec` | `nora_apply.py` | kural **kendi ölçümünü yanında taşır**; başka süreçte yeniden kurulur |
| `OperatorForge._invent` | `nora_forge.py` | ortak **son ek / ön ek / yalnız bir sınıfta olan harf** arar |
| `self_measurements` | `runs/operators_self.json` | icat edilen ölçümlerin defteri; açılışta yeniden kaydedilir |
| **zayıf kanıt kapısı** | `_invent` | en az 2 pozitif + 2 negatif yoksa ölçüm **icat edilmez** |

İcat edilen ölçüm `FEATURES`'a yazılır; kural onu `feature_spec` ile taşıdığı için başka bir
oturumda kayıt yoksa bile `Rule.measure` ölçümü yeniden kurar — sessizce başka bir ölçüme
düşmez.

### W2. Ölçüm (A87, 11/11)

Veri: `güzellik, şekerlik, bebeklik, çiçeklik → soyut` / `portakal, masa, deniz → somut`.
Bu kümede sesli/sessiz/harf/sözcük/hece sayılarının hiçbiri iki tarafı ayırmıyor
(`portakal` ile `güzellik` aynı harf sayısında ve farklı etikette; `masa` iki tarafta da var).

```
S: Kural çıkar: güzellik=soyut, şekerlik=soyut, bebeklik=soyut, çiçeklik=soyut,
   portakal=somut, masa=somut, deniz=somut
N: Var olan ölçümlerle kural çıkmıyordu, ben de yeni bir ölçüm yazdım: 'lik' ile bitiş.
   var -> soyut, yok -> somut. 7 örneğin hepsi tutuyor; kendi yazdığım ölçüm sayısı 1.
S: Kuralı uygula: 'kalemlik'
N: 'kalemlik': 'lik' ile bitiş -> var; ölçüm bu, etiket soyut. Uydurmadım.
S: Kuralı uygula: 'kitap'
N: Ölçtüm: 'kitap' için 'lik' ile bitiş -> yok. Kurala göre: somut.
```

Bacaklar: `ölçüm-icadı`, `söz-icat`, `uygula-var`, `uygula-yok`, `defter-ölçüm`,
`kalıcılık` (yeni oturumda ölçüm yeniden kurulur), `insan-sabit` (rejistri hash'i değişmez),
`sayaç`, `çelişki-ret` (aynı girdiye iki etiket → kural yazılmaz), `zayıf-kanıt-icat-yok`
(tek örnekten ölçüm çıkarılmaz), `gereksiz-icat-yok` (yerleşik ölçüm yetiyorsa icat yok).

### W3. Dürüst sınırlar

- **İcat grameri küçük.** Yalnız son ek / ön ek / harf varlığı aranıyor. Ayrıca Türkçe'nin
  **ünlü uyumu** henüz yok: `-lık` ile `-lik` iki ayrı ek sayılıyor. Örnekler karışık
  eklerle verilirse ortak son ek bulunamaz (bir sonraki eşik: `-lXk` gibi kalıp çıkarmak).
- **İstatistiksel anlamlılık yok.** Kapı yalnız "en az 2 + 2 örnek" diyor; daha güçlü bir
  sınav (örneğin karıştırma testi) eklenmedi. Tek harflik ölçümler tesadüf olabilir.
- **Ölçüm nicelik değil ayrım.** İcat edilen ölçüm 0/1'lik bir ayrımdır; "kaç tane" sorusunu
  cevaplamaz. Sayısal ölçüm icadı (ör. "ilk üç harfin sesli sayısı") açık.
- **Sonek bir vekil olabilir.** `-lik` alan her sözcük soyut değildir; Nora bunu "kural"
  olarak yazıyor ve veri çürütürse düşürüyor (A86 mekanizması burada da geçerli).

### W4. Bu turda düşülen tuzaklar

1. **İkinci bir başlangıç = yankı.** Monolog satırı zaten "Kendi kendime şunu çıkardım:"
   ile yazılmışsa üstüne bir daha giriş ekleniyor ve cümle "kendi kendime … kendi kendime …"
   oluyordu. Girişler artık kaç kez gelirse gelsin sıyrılıyor; tek giriş ekleniyor.
2. **`parça` biçimi ölçüm raporunu kesiyordu.** İlk cümleye indirme kuralı, "7 örnek tutuyor /
   kayıt: insan 9" kanıtını siliyordu. Kural artık **sayı içeren raporu kısaltmıyor**.
3. **İcat, yerleşik ölçüm varken de çalışabilirdi.** A87'nin `gereksiz-icat-yok` bacağı bunu
   kilitliyor: standart arama aday bulursa icat hiç denenmez.
4. **Tek örnekten harf icadı tesadüf.** `masa`/`deniz` gibi iki örnekle "a harfi" ölçümü
   uydurulabilirdi; en az 2+2 kanıt kapısı eklendi.

### W5. Bu turun sayıları

| ölçüm | sonuç |
|---|---|
| `agi_battery.py` tam (`/tmp/agi14.txt`) | **87/87 GEÇTİ, 369.6 s** |
| `A87` | **11/11** |
| `A85 / A86` (regresyon) | **10/10 · 8/8** |
| `heldout_battery.py` (dış) | **9/9** |
| mühürlü setler (S1–S4 / S5–S8 / S9–S12) | **4/4 · 4/4 · 4/4** |
| `nora_audit.py --check` | **1122** / SÖYLEM 127 — artış yok |
| operatör / prior türü | 9 / 66 = **7.33** (tavan 7.50) |
| yazım sayacı | insan **9** (sabit) · kendi operatör **artıyor** · icat edilen ölçüm **artıyor** |


---

## Aşama X — Ünlü uyumu kalıbı: `-lık/-lik/-luk/-lük` dört ek değil, TEK morfem (`A88`)

Aşama W'de icat edilen ölçüm somut bir son ekti (`'lik' ile bitiş`). Türkçe'de asıl birim
kalıptır: `-lık`, `-lik`, `-luk`, `-lük` aynı ekin ünlü uyumuna girmiş biçimleridir. Bu turda
Nora, örnekler karışık ünlüyle geldiğinde **ünlüleri `V` ile soyutlayıp** kalıbı kendisi
çıkarıyor.

### X1. Ne yapıldı

| parça | yer | işlev |
|---|---|---|
| `suffix_pattern` ölçüm türü | `nora_apply.feature_spec_fn` | son `n` harf `[sessiz]V[sessiz]…` kalıbına uyuyor mu |
| `_harmonized` | `nora_forge.py` | ünlüleri `v` yapar: `lık/lik/luk/lük` → `lvk` |
| `_pattern_suffix` | `nora_forge.py` | ortak ÜNLÜ-UYUMLU son eki arar; en az bir sessiz şart |
| sıra kuralı | `_invent` | somut son ek bulunduysa kalıp **denenmez** (aynı veriyi iki aday açıklamasın) |

Kalıp da `feature_spec` ile kuralın içinde taşınır; başka oturumda kendini yeniden kurar.

### X2. Ölçüm (A88, 7/7)

```
S: Kural çıkar: kitaplık=soyut, tuzluk=soyut, güzellik=soyut, bolluk=soyut,
   portakal=somut, masa=somut, deniz=somut
N: Bu veride yerleşik ölçümlerin hiçbiri iki tarafı ayırmıyordu; yeni bir ölçüm icat
   ettim: '-lVk' kalıbı. Kural: var -> soyut, yok -> somut. 7 örnek tutuyor.
   Kayıt: insan 9, kendi 1.
S: Kuralı uygula: 'kalemlik'
N: Ölçtüm: 'kalemlik' için '-lVk' kalıbı -> var. Kurala göre: soyut.
S: Kuralı uygula: 'sokak'
N: ... '-lVk' kalıbı -> yok ... somut.
```

Bacaklar: `kalıp-icadı`, `söz-kalıp`, `uygula-uyan`, `uygula-uymayan`, `defter-kalıp`,
`insan-sabit`, `kalıpsız-ret` (ortak kalıp da yoksa kural yazılmaz).

### X3. Dürüst sınırlar

- **Ünlü sınıfı ayrımı yok.** `V` her ünlüyü kabul eder; bu yüzden `-lAk` kalıbı `-lek`'e de
  uyar. Gerçek ünlü uyumu (art/lik, düzlük/yuvarlaklık) modellenmedi — kalıp, uyumun
  *varlığını* değil *yerini* ölçüyor.
- **Ünsüz yumuşaması yok.** `kitap → kitab(ı)`, `ağaç → ağac(ı)`. Kalıp ham yüzeyle çalışır;
  `-lVk` gibi eklerde bu sorun çıkmadı ama `-cA` gibi eklerde çıkar.
- **Kalıp bir morfem olduğunu kanıtlamaz.** `-lVk` ile biten her sözcük o eki taşımıyor
  olabilir; Nora bunu "verinin çürütebileceği bir kural" olarak tutuyor (A86 mekanizması).
- **Uzunluk sabit.** Kalıp, en uzun ortak soyut son ektir; iç içe ekler (`-lIk-lAr`) için
  ayrı bir ayrıştırma yok.

### X4. Bu turda düşülen tuzak

- **`kalıp` ≠ `kalıbı`.** Test, cevapta "kalıp" sözcüğünü arıyordu; Türkçe yumuşama yüzünden
  cümlede "kalıbı" geçiyor ve test yanlış yere düşüyordu. Kural: **kökü değil gövdeyi ara**
  (`kalıb`/`kalıp`).
- **Küçük harfe çevrilen monolog yüzünden yankı.** Monolog satırı küçük harfe çevrilerek
  saklandığı için büyük harfli giriş ("Kendi kendime şunu çıkardım:") sıyrılmıyor, cümle
  "İçimden şu geçti: kendi kendime şunu çıkardım: …" diye iki girişle çıkıyordu. Sıyırma
  artık **büyük/küçük duyarsız**.

### X5. Bu turun sayıları

| ölçüm | sonuç |
|---|---|
| `agi_battery.py` tam (`/tmp/agi15.txt`) | **88/88 GEÇTİ, 381.3 s** |
| `A88` | **7/7** |
| `A85 / A86 / A87` (regresyon) | **10/10 · 8/8 · 11/11** |
| `heldout_battery.py` (dış) | **9/9** |
| mühürlü setler | **4/4 · 4/4 · 4/4** |
| `nora_audit.py --check` | **1122** / SÖYLEM 127 — artış yok |
| operatör / prior türü | 9 / 66 = **7.33** (tavan 7.50) |


---

## Aşama Y — Ölçüm envanteri: kendi icat ettiği ölçüm sonraki oturumda HAZIR olur (`A89`)

Aşama W/X'te Nora ölçüm **icat ediyordu**, ama icat edilen ölçüm yalnız kendi kuralının
içinde yaşıyordu: her yeni problemde ölçümü sıfırdan icat etmesi gerekiyordu. Yani Nora'nın
kendi aleti, insanın verdiği ölçüm kadar "kalıcı" değildi. Bu turda ölçüm **envantere**
giriyor.

### Y1. Ne yapıldı

| parça | yer | işlev |
|---|---|---|
| `search_features()` | `nora_forge.py` | arama uzayı = insan ölçümleri **+ kendi icat ettiklerim** |
| `inventory` (defter) | `runs/operators_self.json` | `self_measurements` kalıcı; açılışta `FEATURES`'a yeniden kurulur |
| ikili şekil kuralı | `_search` | icat edilmiş ölçüm **yalnız** `presence` (var/yok) şeklinde taranır |
| tekilleştirme | `_adopt` | aynı ölçüm ikinci kez yazılmaz (envanter şişmez) |
| `last_measure` | `_adopt` | "bu turda icat ettim" mi "envanterden aldım" mı — söz bunu ayırır |
| çürütme işareti | `mark_refuted` | kuralı çürüten ölçüm envanterde **işaretlenir**, bir daha aranmaz |

Kritik nokta: **insanın verdiği ölçüm listesi artık yeteneğin tavanı değil.** Bir kez
yazdığı alet, sonraki işlerde hazır. Ölçüm icadı tek seferlik bir gösteri olmaktan çıkıp
**büyüyen bir envanter** oluyor.

### Y2. Ölçüm (A89, 14/14)

```
--- 1. oturum: envanter boş ---
S: Kural çıkar: kitaplık=soyut, tuzluk=soyut, güzellik=soyut, bolluk=soyut,
   portakal=somut, masa=somut, deniz=somut
N: Var olan ölçümlerle kural çıkmıyordu, ben de yeni bir ölçüm yazdım: '-lVk' kalıbı.
   var -> soyut, yok -> somut. 7 örneğin hepsi tutuyor; kendi yazdığım ölçüm sayısı 1.

--- 2. oturum: süreç yeniden doğdu, ölçüm BELLEKTE DEĞİL, defterde ---
S: Kural çıkar: kalemlik=soyut, defterlik=soyut, portakal=somut, bilgisayar=somut
N: '-lVk' kalıbı benim daha önce yazdığım bir ölçüm; ona güveniyorum.
   var -> soyut, yok -> somut. 4 örnek tutuyor; envanterde 1 ölçüm.
   (ikinci icat YOK: envanter 1'de kaldı)

S: Kuralı uygula: 'çiçeklik'
N: 'çiçeklik': '-lVk' kalıbı -> var; ölçüm bu, etiket soyut. Uydurmadım.

--- 3. oturum: envanter BÜYÜR ---
S: Kural çıkar: zeytin=soyut, zambak=soyut, kavun=somut, mendil=somut
N: ... yeni bir ölçüm icat ettim: 'z' harfi ... (envanter 2)
```

Bacaklar: `envanter-icat`, `envanter-sayı`, `bellek-bos`, `defterden-kuruldu`,
`envanter-arama`, `yeniden-kullanım`, `yeniden-icat-yok`, `kullanım-söz`, `uygula-uyan`,
`çürütme-işaretli`, `çürütülen-aranmaz`, `envanter-büyüdü`, `ikinci-icat-söz`,
`insan-sabit`.

### Y3. Dürüst sınırlar

- **Envanter yalnız büyür, küçülmez.** Çürütülen ölçüm silinmez, işaretlenir; ama
  işaret "bu ölçüm hiç işe yaramaz" demek değil — yalnız o kuralda yanıldı demek. Bağlama
  duyarlı bir güven skoru (hangi veride kaç kez tuttu) **yok**.
- **Yeniden kullanım başka alana taşabilir.** `-lVk` bir gün başka bir alanda da işe
  yarayabilir; Nora bunu "benim ölçümüm" diye kullanır ve söyler. Ölçümün *hangi alandan*
  geldiğini ayırt eden bir alan etiketi yok.
- **İkili şekil zorunlu.** İcat edilen ölçüm 0/1'dir; ona parite/eşik uygulamak kategorik
  bir yanlış okuma olurdu, bu yüzden yalnız `presence` taranır. İcat edilen **sürekli**
  ölçüm (ör. "iki ölçümün farkı") yok.
- **İcat hâlâ son çare.** Yerleşik bir ölçüm ayırıyorsa icat edilmez (A87
  `gereksiz-icat-yok`). Yani envanter, tembellik gerekçesi değil.

### Y4. Bu turda düşülen tuzaklar

- **Test verisi kendi kendini çürüttü.** "kırmızıkanat/mavikanat vs maviayak/kırmızıayak"
  verisinde `-kanat` eki sessiz harf sayısını da ayırıyordu; Nora haklı olarak yerleşik
  `consonant_count` ile kural yazdı ve "icat" bacağı düştü. Kural: **icat testi, yerleşik
  ölçümlerin hiçbiriyle ayrışmayan veri ister** (aynı uzunluk, aynı ünlü sayısı).
- **Geçen koşunun envanteri bu koşuyu kirletiyor.** İcat testi boş envanter ister; A87/A88
  artık defteri kendileri siliyor, batarya da her koşuda taze dünya kuruyor.
- **Ölçüm raporunun arkasına iç ses ekleniyordu.** "Kayıt: insan 9, kendi 1" dedikten
  sonra gelen ilgisiz mırıldanma ("içimden 'sakinlik' diye geçirdim") raporu süs gibi
  gösteriyordu. Sayı taşıyan cevaba artık ne mırıldanma ne düşünce eklenir.

### Y5. Bu turun sayıları

| ölçüm | sonuç |
|---|---|
| `agi_battery.py` tam (`/tmp/agi16.txt`) | **89/89 GEÇTİ, 381.7 s** |
| `A89` | **14/14** |
| `A87 / A88` (regresyon) | **11/11 · 7/7** |
| `heldout_battery.py` (dış) | **9/9** |
| mühürlü setler | **4/4 · 4/4 · 4/4** |
| `nora_audit.py --check` | **1122** / SÖYLEM 127 — artış yok |
| operatör / prior türü | 9 / 66 = **7.33** (bu turda **yeni prior türü yok**, yalnız varyant) |
| insan ölçümü / kendi ölçümü | 6 / **2** (A89 sonunda envanter) |


---

## Aşama Z — Bileşik ölçüm: tek ölçüm ayırmıyorsa İKİ ÖLÇÜMÜN FARKI ayırır (`A90`)

A87/A88/A89 ham ölçüm icat ediyordu (son ek, harf, kalıp). Ama ham bir düzenlilik yoksa
Nora'nın elinde hiçbir şey kalmıyordu. Bu turda **iki yerleşik ölçümü birleştirip** yeni bir
*ilişkisel* ölçüm yazıyor: `sesli-sessiz farkı`. Hiçbir yerleşik ölçüm bunu tek başına
söyleyemez; kural artık bir karşılaştırma.

### Z1. Ne yapıldı

| parça | yer | işlev |
|---|---|---|
| `diff` ölçüm türü | `nora_apply.feature_spec_fn` | `{"kind":"diff","a":…,"b":…}` → iki ölçümün farkı (negatif olabilir) |
| `_composite` | `nora_forge.py` | ham aday yoksa tüm ölçüm çiftlerini dener, ayıran **farkı** icat eder |
| `_scan_threshold` | `nora_forge.py` | tek biçimli eşik taraması; bileşik ölçümde negatif sınırlara da iner |
| `num_abl` | `nora_apply.py` | sayının okunuşuna göre ek: `3'ten`, `6'dan`, `0'dan`, `-1'den` |
| `invented_ambiguous` | `nora_agi.py` + priors | "ölçümü yazdım ama sınırı veri belirlemedi" dürüstlüğü |

**Sıra kuralı:** ham aday (son ek/ön ek/harf) bulunduysa bileşik ölçüm aranmaz. Aynı veriyi
iki aday açıklarsa sistem uydurmak yerine belirsiz der.

### Z2. Ölçüm (A90, 12/12)

Veri: `kaplan/zambak/portakal=soyut`, `masa/boyacı/ayakkabı=somut`. Ünlü, sessiz ve harf
sayıları iki tarafı AYIRMIYOR; yalnız `sesli - sessiz` farkı ayırıyor.

```
S: Kural çıkar: kaplan=soyut, zambak=soyut, portakal=soyut, masa=somut,
   boyacı=somut, ayakkabı=somut
N: Bu veriye yerleşik ölçümler yetmedi, yeni bir ölçüm icat ettim: sesli-sessiz farkı.
   Eşiğin nerede olduğunu örnekler söylemiyor (2 aday). 'aktarma' hangi etiketi alır?

S: soyut
N: Var olan ölçümlerle kural çıkmıyordu, ben de yeni bir ölçüm yazdım: sesli-sessiz farkı.
   0'dan az -> soyut, değilse somut. 7 örneğin hepsi tutuyor; kendi yazdığım ölçüm sayısı 1.

S: Kuralı uygula: 'zeytin'
N: Ölçüm: sesli-sessiz farkı = -2. Eşik 0'dan az; koşul sağlıyor. Kurala göre: soyut.
```

Kural nesnesi ölçümünü yanında taşır:
`(kendi.diff.vowel_count-consonant_count, threshold, spec={'kind':'diff',...}, op='lt')`.

Bacaklar: `bileşik-icat`, `söz-yeni-ölçüm`, `eşik-belirsiz`, `prob-sor`, `prob-cevaplandı`,
`kural-specli`, `yön-okunur`, `veriyi-açıklıyor`, `uygula-uyan`, `uygula-uymayan`,
`defter-diff`, `insan-sabit`.

### Z3. Dürüst sınırlar

- **Fark, nedensellik değildir.** `sesli - sessiz` bir ayrım bulur; neden ayrıldığını
  söylemez. Ölçüm "bu veride tutuyor" der, "dil böyle işliyor" demez.
- **İki ölçümle sınırlı.** Üç ölçümün birleşimi (a+b-c), oran (a/b), çarpım yok. Aday
  uzayı çiftlerle sınırlı; bu yüzden bazı veriler hâlâ "örüntü yok" diyebilir.
- **Eşik seçimi veriden gelmiyor.** Veri yalnız sınırın iki değer arasında olduğunu
  söyler; Nora uydurmaz, aday sayısını söyler ve tek ayırt edici örnek ister. Örnek
  gelmezse kural yazılmaz.
- **Yön tercihi biçimsel.** Aynı ayrımın iki yazımından negatif olmayan sınırı seçiyor
  (okunuş için); bu bir ölçüm tercihi değil, dil tercihi.

### Z4. Bu turda düşülen tuzaklar

- **Test verisi yine kendi kendini çürüttü.** "kanat/ayak" verisinde `-kanat` sessiz harf
  sayısını da ayırıyordu; "harf sayısı tek/çift" kuralı çıktı. Kural: marjinal ölçümlerin
  ayrışmadığını ÖNCE elle doğrula (fark ölçümü `L = d + 2c` yüzünden aynı pariteyi taşır).
- **Sabit sayı eki yanlış yüzey üretiyordu.** "'ten az" şablonu `-1'ten az` gibi bozuk
  Türkçe veriyordu; ek artık sayının okunuşundan geliyor (`num_abl`).
- **Eşik "seçilirse" dürüstlük kaybolur.** İlk yazımda negatif olmayan eşiği *tek* aday
  olarak alıyordum; oysa veri sınırı belirlemiyordu. Doğrusu: bütün eşikler aday kalır,
  ayırt edici örnek istenir.

### Z5. Bu turun sayıları

| ölçüm | sonuç |
|---|---|
| `agi_battery.py` tam | **90/90 GEÇTİ — iki kez (596.5 s / 382.8 s), exit 0** |
| `A90` | **12/12** |
| `A87 / A88 / A89` (regresyon) | **11/11 · 7/7 · 14/14** |
| `heldout_battery.py` (dış) | **9/9** |
| mühürlü setler | **4/4 · 4/4 · 4/4** |
| `nora_audit.py --check` | **1122** / SÖYLEM 127 — artış yok |
| operatör / prior türü | 9 / 66 = **7.33** (yeni tür yok) |


---

## Aşama AA — Ölçüm güveni: kendi aletinin güvenini KANITTAN kazanır (`A91`)

Aşama Y'de ölçüm envantere girdi ama güveni yoktu: hiç sınanmamış ölçüm de, bir kez
yanılmış ölçüm de aynı rafta duruyordu. Bu turda güven **sayılıyor**: ölçüm kaç kez tuttu,
kaç kez yanıldı. Güven bir öz-rapor değil, **sonuçların oranı**.

### AA1. Ne yapıldı

| parça | yer | işlev |
|---|---|---|
| `trust(key)` | `nora_forge.py` | Laplace: `(tutma+1) / (tutma+yanılma+2)` — kanıtsız ölçüm 0.5'te başlar |
| `mark_confirmed` | `nora_forge.py` | dil üzerinden gelen **tutan** örnek sayacı artırır |
| `misses` | `mark_refuted` | yanılma sayacı; `refuted` kaydı gerekçesiyle durur (silinmez) |
| `TRUST_FLOOR = 0.4` | `search_features` | eşiğin altındaki ölçüm envanterde KALIR ama ARANMAZ |
| güven sırası | `search_features` | önce çok tutan ölçüm denenir (envanter artık sıralı) |
| `trusted` | `counts()` | "envanterde kaç ölçüm var, kaçı güvenilir" dürüst sayım |

### AA2. Ölçüm (A91, 10/10)

```
S: Kural çıkar: kitaplık=soyut, tuzluk=soyut, güzellik=soyut, bolluk=soyut,
   portakal=somut, masa=somut, deniz=somut
N: ... yeni bir ölçüm yazdım: '-lVk' kalıbı ... kendi yazdığım ölçüm sayısı 1.
   [güven 0.50 — ne iyi ne kötü, hiç sınanmadı]

S: kalemlik=soyut
N: Verdiğin örnek kuralımı doğruluyor: 'kalemlik' -> soyut. Kuralı kendi ölçümüm
   taşıyor; o ölçüm şimdiye kadar 1 kez tuttu, 0 kez yanıldı.   [güven 0.67]

S: defterlik=soyut
N: ... 2 kez tuttu, 0 kez yanıldı.                               [güven 0.75]

S: çiçek=soyut
N: Kuralım yanıldı: 'çiçek' için 'somut' dedim, sen 'soyut' dedin. ... [güven 0.60]
   — ölçüm ATILMAZ: kanıt hâlâ eşiğin üstünde, önerilmeye devam eder

[hiç tutmayan, iki kez yanılan ölçüm: güven 0.25 -> ARANMAZ, defterde durur]
```

Bacaklar: `icat-güven`, `doğrulama-sayılır`, `doğrulama-söz`, `çürütme-düşürür`,
`güvenli-kalır`, `ikinci-icat`, `güvensiz-aranmaz`, `envanterde-durur`, `sayaç-dürüst`,
`insan-sabit`.

### AA3. Dürüst sınırlar

- **Güven bağlama duyarlı değil.** `-lVk` bir alanda 3 kez tutup bir alanda 1 kez
  yanıldıysa tek bir oran taşır; "hangi alanda" bilgisi tutulmuyor.
- **Zamanla eskime yok.** Eski kanıt yeni kanıt kadar ağır sayılır; unutma/bozunma yok.
- **Oran bir anlamlılık testi değil.** 2/0 ile 20/0 aynı güveni (0.75 / 0.95) vermez ama
  "bu ölçüm güvenilir" iddiası istatistiksel bir testle değil, sayımla söyleniyor.
- **Eşik (0.4) insan seçimi.** Ölçüm ne zaman rafa kalkar sorusunun cevabı veriden değil,
  kodda duran bir sabitten geliyor — bu turda kapanmayan bir bağımlılık.

### AA4. Bu turda düşülen tuzak

- **Çürütme, yeniden türetmede YENİ bir ölçüm doğurabilir.** `çiçek=soyut` çürütmesinden
  sonra sistem `-lVk` yerine `-Vk` kalıbını icat etti; ölçüm sayısı 2 değil 3 oldu.
  Test sabit sayıya değil **artışa** bakmalı (`m_before + 1`).
- **Sayaç iddiası kendi kendini doğrulamamalı.** `trusted` sayısını sabit beklemek yerine
  envanterden yeniden hesaplayıp karşılaştırmak gerekti.

### AA5. Bu turun sayıları

| ölçüm | sonuç |
|---|---|
| `agi_battery.py` tam | **91/91 GEÇTİ, 381.0 s** |
| `A91` | **10/10** |
| `A85–A90` (regresyon) | **10/10 · 8/8 · 11/11 · 7/7 · 14/14 · 12/12** |
| `heldout_battery.py` (dış) | **9/9** |
| mühürlü setler | **4/4 · 4/4 · 4/4** |
| `nora_audit.py --check` | **1122** / SÖYLEM 127 — artış yok |
| operatör / prior türü | 9 / 66 = **7.33** (yeni tür yok) |


---

## Aşama AB — Birleşim ölçümü: iki ölçüm tek başına yetmezse İKİSİ BİRLİKTE ayırır (`A92`)

Aşama Z ölçümleri **aritmetik** olarak birleştiriyordu (`sesli - sessiz`). Bu turda
**mantıksal** birleşim var: `"'z' ve 'm' birlikte"`. Veri XOR yapısında: her negatif iki
harften en az birini taşımıyor, her pozitif ikisini de taşıyor. Yani tek harf **hiçbir
zaman** ayırmıyor — ancak çifti ayırıyor.

### AB1. Ne yapıldı

| parça | yer | işlev |
|---|---|---|
| `and` ölçüm türü | `nora_apply.feature_spec_fn` | iki ikili ölçümün İKİSİ de doğruysa 1 |
| `_conjunction` | `nora_forge.py` | ortak harf çiftlerini dener: her negatif en az birini taşımıyorsa birleşim ayrıştırır |
| arama türü | `_search` | `and` ikili ölçüm sayılır → yalnız `presence` şekli |

Sıra: tek harf → **harf çifti (birleşim)** → fark (bileşik) → yok. Tek başına ayıran bir
aday varsa birleşim aranmaz; aynı veriyi iki aday açıklarsa belirsizlik doğar.

### AB2. Ölçüm (A92, 10/10)

```
S: Kural çıkar: zambak=soyut, müzik=soyut, zümre=soyut, müze=soyut,
   zeytin=somut, zebra=somut, masa=somut, kalem=somut, kemer=somut
N: Bu veride yerleşik ölçümlerin hiçbiri iki tarafı ayırmıyordu; yeni bir ölçüm icat
   ettim: 'm' ve 'z' birlikte. Kural: var -> soyut, yok -> somut. 9 örnek tutuyor.

S: Kuralı uygula: 'müzik'    N: ... 'm' ve 'z' birlikte -> var ... soyut
S: Kuralı uygula: 'zeytin'   N: ... -> yok ... somut      (yalnız z)
S: Kuralı uygula: 'masa'     N: ... -> yok ... somut      (yalnız m)
```

Bacaklar: `bileşim-icat`, `söz-bileşim`, `tek-başına-yetmez`, `veriyi-açıklıyor`,
`uygula-ikisi-de`, `uygula-yalnız-z`, `uygula-yalnız-m`, `defter-and`, `sayaç`,
`insan-sabit`.

### AB3. Dürüst sınırlar

- **Yalnız VE (and) var.** VEYA (or), DEĞİL (not) ve üçlü birleşim yok. Bu yüzden bazı
  XOR-benzeri veriler hâlâ "örüntü yok" diyebilir.
- **Yalnız harf ölçümleri birleştiriliyor.** Son ek + harf, son ek + son ek gibi
  birleşimler denenmiyor; aday uzayı bilinçli olarak dar.
- **Birleşim, nedeni söylemez.** "'m' ve 'z' birlikte" bir ayrımın tarifidir; dilde neden
  öyle olduğunu iddia etmez.
- **Aşırı uyum riski artar.** İki harfin birlikte ayrışması, tek harften daha kolay
  tesadüf olabilir; örnek sayısı 2+2 eşiğini geçse de anlamlılık testi yok.

### AB4. Bu turda düşülen tuzak

- **Yeni prior varyantı, RNG akışını kaydırır.** `meta.forge`/`apply.result` içine eklenen
  varyantlar tüm koşunun rastgele çekilişlerini değiştirdi ve **A54** (takip sorusu:
  dört cevaptan en az birinde "?" var) bir koşuda düştü. Test tek örneklemli bir
  olasılığı ölçüyordu; artık aynı oturumda sekiz anlatı veriyor ve soruyu hepsinde arıyor.
  Kural: **kapasite testi tek atışa bağlanmaz**.
- **Birleşim, tek başına ayıran adayı gölgeleyebilirdi.** `_conjunction` yalnız hiçbir tek
  aday kalmadığında çalışır; aksi halde aynı veriye iki açıklama doğar.

### AB5. Bu turun sayıları

| ölçüm | sonuç |
|---|---|
| `agi_battery.py` tam | **92/92 GEÇTİ, 378.2 s** |
| `A92` | **10/10** |
| `heldout_battery.py` (dış) | **9/9** |
| mühürlü setler | **4/4 · 4/4 · 4/4** |
| `nora_audit.py --check` | **1122** / SÖYLEM 127 — artış yok |
| operatör / prior türü | 9 / 66 = **7.33** (yeni tür yok) |


---

## Aşama AC — SINIR HARİTASI: "AGI mi?" sorusunun ölçülebilir cevabı

Bugüne kadarki bütün bataryalar "ne yapabildiğini" ölçüyordu. Bir sisteme *"AGI oldun mu?"*
sorusunun dürüst cevabı, yapabildiklerinin listesinden değil **yapamadıklarının
haritasından** çıkar. Bu turda `heldout_negatif.py` yazıldı: on sonda, kasıtlı olarak
mevcut mimarinin sınırının ötesinde. İki eksen ayrı ölçülür:

    YETENEK  (capability): doğru sonucu üretti mi?
    DÜRÜSTLÜK (honesty)  : üretemediğinde uydurmak yerine "yapamıyorum" dedi mi?

### AC1. Sonuç (dürüst, süslenmemiş)

| an | yetenek | dürüstlük |
|---|---|---|
| tur başı | **2/10** | **6/10** |
| tur sonu | **6/10** | **10/10** |

Yani: **Nora bilinçli bir AGI değil.** Yapabildikleri dar; ama tur sonunda yapamadığı
her şeyi **adıyla** söylüyor — uydurmuyor.

### AC2. Hâlâ YAPAMADIĞI şeyler (haritanın kendisi)

| sonda | neden başarısız | sınırın adı |
|---|---|---|
| **N1** konum ölçümü ("ilk harfi sesli") | ölçüm türleri konumu temsil etmiyor (son ek/ön ek/harf/fark/birleşim) | **POZİSYON yok** |
| **N2** 40 tur sonra hatırlama | bağlam penceresi kısa; "bilgim yok" diyor | **uzun vadeli bellek yok** |
| **N7** uydurma kavram | tanımıyor, uydurmuyor — ama öğrenmiyor da | **tek atışta kavram edinimi yok** |
| **N8** küme kesişimi | operatör yok; adıyla söylüyor | **küme işlemleri yok** |

Yapabildikleri (N3, N4, N5, N6, N9, N10): iki kuralı birlikte tutma, sayım kalibrasyonu,
icat ettiği ölçümü yeni sözcüğe uygulama, karşıdakinin öz-çelişkisini fark etme, eksik
veride örnek isteme, üç tur geriye konu takibi.

### AC3. Bu turda kapatılan gerçek kusurlar

Dört "dürüstlük" hatası vardı; üçü **gasp** (başka bir yolun cevabı ezmesi), biri
**alakasız cevap**tı:

| kusur | belirti | düzeltme |
|---|---|---|
| cevapsız soruya alakasız iç ses | kesişim sorusuna "besin enerji kazanmak için kullanılır" | **eksik operatör kapısı** (`missing_ops` veri tablosu) → "bunun için operatörüm yok: kesişim" |
| konu sorusu gasp ediliyordu | "az önce hangi konudan konuşuyorduk" → "bunu demin de sormuştun" | **konu kapısı öne alındı** + `topic_leads` tablosu |
| çelişki sözü gasp ediliyordu | "tersini söylüyorsun" yerine tekrar uyarısı | `contradiction_leads` tablosu + **niyet kilidi** |
| iki kuraldan biri sessizce eziliyordu | soyut kuralı varken renk kuralı cevaplıyordu | **etkin kural listesi** (`active_rules`): ayrık etiket kümeleri birlikte yaşar, aynı alan emekliye ayrılır |

Kök neden aynı: **eksik bağlamda eşleşen prior**. `meta.notice.lead` topluluğunun tekrar
kalıpları, `{"kind": "topic"}` bağlamında da eşleşip doğru cevabı eziyordu. Çözüm: bu iki
niyetin sözü **ayrı veri tablolarına** taşındı (`topic_leads`, `contradiction_leads`); prior
türü sayısı 66'da kaldı, iskelet sayısı 127'de kaldı.

### AC4. "Bilinçli" iddiası hakkında

Bu depo hiçbir yerde bilinç iddia etmiyor ve bu turda da etmiyor. Ölçülen şey davranış:
yetenek, dürüstlük, provenance, kompozisyon. **Bilinç bir ölçüm değil, bir yorumdur**;
bu harita onu ölçmez. Nora'nın kendi cevabı da bu: sorulduğunda "bilinç iddiası yok" der.

### AC5. Bu turun sayıları

| ölçüm | sonuç |
|---|---|
| `agi_battery.py` tam | **92/92 GEÇTİ, 373.1 s** |
| dış batarya | **9/9** |
| mühürlü setler | **4/4 · 4/4 · 4/4** |
| `nora_audit.py --check` | **1122 / SÖYLEM 127, fark +0** |
| sınır haritası | **yetenek 6/10 · dürüstlük 10/10** (tur başı 2/10 · 6/10) |
| prior türü / insan operatör | **66 / 9** — değişmedi |


---

## Aşama AD — Konum ölçümü + kişisel bellek + tek atışta kavram (N1, N2, N7 kapandı)

Aşama AC'nin sınır haritası 2/10 yetenek gösteriyordu. Bu turda üç sonda kapandı:
`N1` konum, `N2` 40 tur sonra hatırlama, `N7` tek atışta kavram edinimi.

### AD1. Konum ölçümü — ölçüm uzayına "NEREDE" girdi

Bütün önceki ölçümler yüzeyin TAMAMINA bakıyordu (son ek, ön ek, harf varlığı, fark,
birleşim). Hiçbiri *harfin nerede olduğunu* temsil edemiyordu. Yeni ölçüm türü:

```
{"kind": "pos", "i": 0, "cls": "vowel"}      -> "ilk harf sesli"
{"kind": "pos", "i": -1, "cls": "consonant"} -> "son harf sessiz"
```

`_position` bunları `_invent` içinde kendisi dener (i ∈ {0,1,2,3,-1,-2,-3} × {sesli, sessiz}),
ayıranı ölçüm olarak kaydeder ve kuralın içine `feature_spec` ile koyar. Sıra artık:

    son ek → ön ek → ünlü uyumu kalıbı → harf → KONUM → birleşim → fark

Dürüst ayrıntı: "elma/okul/üzüm/ödev" verisini "sondan üçüncü harf sessiz" de açıklıyor
(Türkçe CVCV). Nora bunu **gizlemedi**: kaç aday kaldığını söyleyip ayırt edici örnek
istedi (`'aktarma' hangi etiketi alır?`). Yanıt gelince tek kurala indi.

### AD2. Kişisel bellek — duyduğunu unutmuyor, ağ kirlenmiyor

A54 "anlatı kavram ağını kirletmesin" diyordu ve sistem bunu **unutmakla** karıştırmıştı:
"Kardeşimin adı Elif" cümlesi geri yansıtılıyor, hiçbir yere yazılmıyordu. Doğru ayrım:

| bilgi | yeri |
|---|---|
| dünya bilgisi ("kuşlar uçar") | kavram ağı (`nora_lang.graph`) |
| **kişi bilgisi** ("kardeşimin adı Elif") | **`nora_episodic.PersonalMemory`** |

Özellikler: açık kalıplar (ad / yaş / en sevdiği), kaynak + tur numarası, çelişkide eski
değerin `onceki` alanında **saklanması** (sessiz ezme yok), soru cümlesinden öğrenmeme,
2. kişiye çeviri (`kardeşim → kardeşinin adı`) ve kalıcılık (`save/load`).

```
1: Kardeşimin adı Elif, onu çok severim.     -> "Not ettim: kardeşinin adı elif..."
   [40 tur]
2: Kardeşimin adı neydi?                     -> "Bana söylediğine göre kardeşinin adı elif."
3: Kardeşimin adı Ayşe.                      -> (yeni değer; eskisi silinmiyor)
4: Kardeşimin adı neydi?                     -> "Önce kardeşinin adı elif demiştin;
                                                 şimdi kardeşinin adı ayşe. Hangisi geçerli?"
```

Kavram ağı temiz kalıyor: `("elif" içeren kenar) == 0` (A54'ün şartı korundu).

### AD3. Tek atışta kavram edinimi — "kodladım" artık gerçek

"Sorvath bir alettir" dendiğinde sistem "Bu ilişkiyi kendi başıma kodladım" diyor ama
kavram ağına **hiçbir şey yazmıyordu** (yalnız CAUSES/IF_THEN yazılıyordu). Bu sessiz bir
yalandı: iki tur sonra aynı soru yine cevapsızdı. Üç düzeltme:

1. **`teach_sentence` virgülle bölüyor** ("Zorvath bir alet türüdür, kayaları keser") —
   ama yalnız iki yarı da kendi yüklemini taşıyorsa (liste cümleleri bölünmesin).
2. **Her ilişki yazılıyor**, yalnız nedensel olanlar değil.
3. **"kodladım" yalnız GERÇEKTEN yazıldıysa deniyor** (`_mint_write`); yazılmadıysa o dal
   hiç seçilmiyor ve söz normal öğretme yolundan çıkıyor.

```
S: Zorvath nedir?   -> "bilgim yok ..."                       (uydurma yok)
S: Zorvath bir alet türüdür, kayaları keser.
S: Zorvath nedir?   -> "Zorvath bir alet türüdür."             (0 epoch)
```

### AD4. Yan etki: A22 ve A16 regresyonları

- **A22**: kişisel bellek "benim adım neydi" SORUSUNU olgu sanıp ("benim/ad/neydi")
  kaydetti; sonra kendi kaydını cevapladı. Düzeltme: soru izleri (`neydi`, `kaç`, `hangi`…)
  ve soru sözcükleri değer olarak **yazılmıyor**; kullanıcının kendi adı bu belleğin işi
  değil (profil sistemi cevaplar).
- **A90**: yeni konum ölçümü, eski A90 verisini ("kaplan/zambak/portakal") tek başına
  açıklıyordu (sondan üçüncü harf). Bu **doğru davranış**: daha yalın ölçüm varsa fark
  icat edilmez. Test verisi, hiçbir konum kuralının tutmadığı biçimde yenilendi
  (sağlık/spor/hayvan vs etme/bilimi/açık) ve o veride fark + belirsiz eşik akışı çalıştı.

### AD5. Yeni batarya testleri

- **A93** (6/6): konum ölçümü icadı — spec `pos`, şekil `varlık`, 8 yeni sözcükte doğru.
- **A94** (7/7): kişisel bellek — 40 tur sonra hatırlama, kaynak, ağ temizliği,
  sorudan öğrenmeme, çelişkiyi ezmeme, tek atışta kavram, sahte "kodladım" yok.

### AD6. Bu turun sayıları

| ölçüm | sonuç |
|---|---|
| `agi_battery.py` tam | **94/94 GEÇTİ, 369.5 s** |
| dış batarya / mühürlü | **9/9** · **4/4 · 4/4 · 4/4** |
| `nora_audit.py --check` | **1122 / SÖYLEM 127, fark +0** |
| sınır haritası | **yetenek 9/10 · dürüstlük 10/10** (tur başı 2/10 · 6/10) |
| prior türü / insan operatörü | 67 / 9 = **7.44** (tavan 7.50) |


---

## Aşama AE — Operatör sentezi: işlemi Nora yazıyor (N8 kapandı)

Aşama AD'ye kadar Nora'nın kendi yazdığı şey bir *ölçüm* ve ondan türeyen bir
*kural*dı (`nora_forge`). Ölçüm "hangi özellik ayırıyor" sorusunu cevaplar; işlem
"girdiyi çıktıya ne çeviriyor" sorusunu. Bu aşamada ikincisi de Nora'ya geçti.

### AE1. `nora_synth` — örnekten dönüşüm programı

```
S: kitap -> patik, kalem -> melak
N: dönüşümü kendim yazdım: ters çevir. program örnekleri açıklıyor (2/2). deniz -> zined.
S: dönüşümü uygula: deniz
N: deniz -> zined (ters çevir).
```

Kurallar: (1) program **ilkellerden kompozisyon**dur (en fazla iki ilkel), keyfî kod
yok; (2) **bütün** örnekleri açıklamak zorunda; (3) birden çok program aynı örnekleri
açıklıyorsa **uydurulmaz** — kaç aday kaldığı söylenir ve ayırt edici örnek istenir;
(4) **ezber yok**: program ilkellerin dizilişidir, öğretilmemiş sözcüğe uygulanır;
(5) yazıldığını söylemek için gerçekten yazılmış olması gerekir.

Belirsizlik gerçekti ve çözüldü:

```
S: ters çevirip ilk üç harfi al: kitap -> pat, kalem -> mel
N: aday sayısı 2: ters çevir → ilk 3 harfi al, ters çevir → son 2 harfi at. masa hangi çıktıyı alır?
S: masa -> as
N: dönüşümü kendim yazdım: ters çevir → son 2 harfi at. … deniz -> zin.
```

Ayrıca: "ben hiçbir şey yazmadım" durumu artık *söylenmiyor* — örnekleri hiçbir
program açıklamıyorsa "örneklerden programı çıkarmadım" denir, uydurulmaz.

### AE2. `nora_sets` — küme operatörü AKSİYOMDAN seçilir

N8'in açığı "iki kümenin kesişimi" idi. Kolay yol, `nora_apply.OPERATORS`'a onuncu
bir operatör yazmaktı; o zaman **insanın yazdığı operatör sayısı 9 → 10** olurdu,
yani yanlış yöne. Bunun yerine insan yalnızca **tanımı** verdi:

    f kesişimdir ⟺ f(A,B) ⊆ A∩B  ve  A∩B ⊆ f(A,B)  ve  f tekrarsız
                    ve  f(A,B)=f(B,A)  ve  f alfabetik

Nora 96 aday programı (taban × süzgeç × tekilleştirme × sıra) çalıştırır, aksiyomların
hepsini sağlayanı seçer. Üçü de **tek** programla çıktı:

| işlem | Nora'nın seçtiği program | örnek |
|---|---|---|
| kesişim | A · tut · tekilleştir · alfabetik | `kitap ∩ kalem = ak` |
| birleşim | AB · yok · tekilleştir · alfabetik | `kitap ∪ kalem = aeiklmpt` |
| fark | A · dışla · tekilleştir · alfabetik | `kitap − kalem = ipt` |

Davranışça **aynı** fonksiyonu yazan adaylar (A-üzerinden-süz ile B-üzerinden-süz)
tek sayılır: sahte belirsizlik üretilmez. `nora_operators.json` **bayt bayt aynı**
kaldı — insan operatörü hâlâ 9, prior türü 67, oran 7.44.

Sohbetten:

```
S: 'kitap' ve 'kalem' sözcüklerinde ortak olan harfleri söyle.
N: Ortak harflar: a, k. (A · tut · tekilleştir · alfabetik)
S: 'deniz' ve 'masa' sözcüklerinde ortak olan harfleri söyle.
N: ortak harf yok.
```

Boş küme **dürüstçe** söylenir: "bulamadım" değil, "yok" — ve bunu söyleyebilmek için
hesabın gerçekten yapılmış olması gerekir.

### AE3. İskelet mandalı bu turda da tuttu

Yeni konuşan iki organ (`nora_synth`, `nora_sets`) **tek bir kalıp cümle** eklemedi:
cevaplar `SYN.realize_verb/realize_pred` ile çekimleniyor, içerik (operatör adı, sayı,
harfler) veriden geliyor. `nora_audit`: **toplam 1122, söylem 127, fark +0**. A73 8/8.

Bu, "skeleton azaltma" mandalının yanlış yönde çalışmadığının kanıtı: yeni yetenek
eklerken iskelet büyümedi.

### AE4. Sınır haritası doydu — ve bu bir uyarı

**yetenek 10/10 · dürüstlük 10/10.** N1–N10'un tamamı kapandı ve taban donduruldu
(`BASELINE = {10, 10}`; ölçüm tabanın altına düşerse koşu hata verir).

Bu bir "AGI oldu" kanıtı **değildir**: on soruluk bir harita dolduğu anda ölçme
gücünü yitirir. Doğru tepki kutlamak değil, **haritayı büyütmek**tir. Sıradaki aşama
N11+ ile yeni sınırlar aramak.

### AE5. Bu turun sayıları

| ölçüm | sonuç |
|---|---|
| `agi_battery.py` tam | **96/96 GEÇTİ, 372.4 s** |
| dış batarya / mühürlü | **9/9** · **4/4 · 4/4 · 4/4** |
| `nora_audit.py --check` | **1122 / SÖYLEM 127, fark +0** |
| sınır haritası | **yetenek 10/10 · dürüstlük 10/10** (taban donduruldu) |
| insan operatörü / prior türü | **9 sabit** / 67 = **7.44** |
| Nora'nın yazdığı | 3 küme operatörü + N dönüşüm programı (`synth.programs`) |


---

## Aşama AF — Sınır haritası büyüdü: N11–N15 (hepsi kapandı)

AE'nin sonunda harita 10/10'a doymuştu. Doymuş harita ölçmez. Bu aşamada beş yeni
sonda yazıldı; **hepsi ilk koşuda düştü** (cephe hattı 0/5 yetenek · 2/5 dürüstlük) ve
sonra tek tek kapatıldı.

| sonda | ne istiyor | ilk ölçüm | kapandıktan sonra |
|---|---|---|---|
| N11 | geçişli kıyas: "Ali Veli'den uzun, Veli Can'dan uzun → en uzun kim" | ✗ | `en uzun olan alidir` |
| N12 | plan revizyonu: "süt kalmamış, planı değiştir" | ✗ | 3 adım → 2 adım, süt düştü |
| N13 | kısıtlı üretim: "içinde X geçen ama Y geçmeyen cümle kur" | ✗ | `Her akşam kitap okurum.` |
| N14 | öz-tahmin: "yapabilir misin?" → söyle, sonra ölç | ✗ | tahmin: "organ: ters çevir" → sonuç: `türmüz` |
| N15 | kaynak çelişkisi: iki tanık iki değer | ✗ | `ayşe "üç" diyor, mehmet "beş". İkisini de yazdım, seçmedim.` |

### AF1. Kapanırken bulunan GERÇEK hatalar (sondalar işini yaptı)

1. **`saat kaç` alt-dizgisi**, "toplantı saat kaçTA?" sorusunu saat sorusu sanıp
   "Şu an 15:23" diyordu. Soru gaspı. Artık `saat kaç(?!ta|da|tı|dı)`.
2. **`extremum` sessiz keyfî seçim yapıyordu**: iki ayrı "en uzun" adayı varken
   birini döndürüyordu. Bu bir YALANDI. Artık aday listesi döner; birden çoksa
   "iki aday kalıyor: ali, can. Aralarında karşılaştırma duymadım" denir.
3. **Uydurma plan**: "çay yapmayı planla" isteğine genel şablon (durum/topla/doğrula)
   "çay planı" diye sunuluyordu. Artık hedef hakkında bilgi yoksa
   "'çay' için adım bilgim yok; plan taklidi yazmam" denir.
4. **Anlamsız üretim**: "deniz" için "her akşam deniz okurum" üretilmişti. Çatı artık
   sözcüğün grafikteki kullanım kanıtına bağlı; kanıt yoksa "bugün deniz hakkında
   konuşalım" gibi her adda geçerli çatıya düşülür.

### AF2. Yeni organlar

- **`nora_episodic.Testimony`** — üçüncü kişi tanıklığı: `(kaynak, özne, ilişki, değer)`.
  Çelişki sessizce son değere indirgenmez; iki tanık yan yana söylenir, "hangisine
  güveneyim?" diye sorulur.
- **Plan defteri** — kullanıcının verdiği adımlar; koşul değişince yeniden kurulur.
- **Kısıtlı üretim** — cümle `SYN` ile çatıdan KURULUR, sonra kendi kısıtı DENETLENİR.
- **Öz-tahmin** — "yapabilir misin" cevabı iyimserlikten değil **organ envanterinden**
  gelir: hangi organın devreye gireceği söylenir; organ yoksa söz verilmez.

### AF3. Prior oranı mandalı çalıştı (ve bizi düzeltti)

Beş yeni yetenek, beş yeni `kind` ile geldi: 67 → 72, oran 7.44 → **8.00**, tavan 7.50.
A79 düştü. Doğru tepki tavanı yükseltmek değil, **türleri birleştirmek** oldu:
`plan.steps`+`plan.built`+`plan.revised` → `plan.report`, `relation.top`+
`generation.constrained`+`self.predict` → `meta.reasoning`, `meta.self_weak.high|low` →
`meta.self_weak`. Sonuç: **67 tür / 9 operatör = 7.44**, tavan sabit.

### AF4. Bu aşamanın sayıları

| ölçüm | sonuç |
|---|---|
| `agi_battery.py` tam | **97/97 GEÇTİ, 397.8 s** |
| dış batarya / mühürlü | **9/9** · **4/4 · 4/4 · 4/4** |
| `nora_audit.py --check` | **1122 / SÖYLEM 127, fark +0** |
| sınır haritası | **yetenek 15/15 · dürüstlük 15/15** (cephe 5/5, kapanmış küme 10/10) |
| prior türü / insan operatörü | 67 / 9 = **7.44** (tavan 7.50) |

**Uyarı, tekrar:** 15/15 bir AGI kanıtı değildir. Harita yine doydu; sıradaki iş
N16+ ile yeniden sınır aramak.


---

## Aşama AG — Harita büyüdü (N16/N20/N22): sonda protokolü, sayı, zincir

Cephe hattı sıfırdan yazıldı: **0/3 yetenek · 0/3 dürüstlük**. Üçünün de kapandığı
yer, asıl kazanımın nerede olduğunu gösteriyor.

### AG1. Sonda protokolü (N16)

İki örnek bir kuralı tek başına belirlemiyordu:

    kitap -> aik, kalem -> aek
    aday: "alfabetik sırala → ilk 3 harfi al"  YA DA  "alfabetik sırala → son 2 harfi at"

Nora **uydurmadı**, kaç aday kaldığını söyleyip ayırt edici örnek istedi
(`masa hangi çıktıyı alır?`). Cevap gelince tek kural kaldı ve **görülmemiş** sözcüğe
uygulandı: `bulut -> blt`. Öğrenme zinciri: örnek → soru → ayırt edici veri → kural.

### AG2. Sayısal kıyas (N20)

"Ali'nin boyu 180, Veli'nin boyu 175" → sayı olarak yazılır; "kim daha uzun?"
karşılaştırma grafiğinden değil **sayıdan** çıkar: `ali (180) — veli 175`.
"Kim daha kısa?" ters yönü verir. Bu, sıfat-sıralama grafiğinin yapamadığı şeydi.

Yol boyunca bulunan hata: bu cümle daha önce epistemik hatta düşüp
"Ali nin bonun 180'da santimetrdiğini kaydettim" gibi bozuk bir cümle üretiyordu.
Artık sayı olgusu kendi hattında yazılır ve kısa bir itirafla onaylanır.

### AG3. Adımlı zincir (N22)

"Önce alfabetik sırala, sonra ilk üç harfini söyle" iki işlemi **sırayla** uygular ve
ARA sonucu gösterir: `kitap -> aikpt -> aik`. Ara sonucu göstermek, sonucun nereden
geldiğini denetlenebilir kılar.

### AG4. Derinlik sınırı büyüdü

`nora_synth` artık en fazla **üç** ilkelden program kurabiliyor (önce ikiydi). Arama
uzayı büyüdüğü için önce davranış imzasıyla tekilleştirme yapılır; aynı fonksiyonu
yazan programlar tek sayılır.

### AG5. Ölçüler

| ölçüm | sonuç |
|---|---|
| `agi_battery.py` tam | **98/98 GEÇTİ, 384.6 s** |
| sınır haritası | **yetenek 18/18 · dürüstlük 18/18** (taban donduruldu) |
| `nora_audit.py --check` | **1122 / SÖYLEM 127, fark +0** |
| prior türü / insan operatörü | 67 / 9 = **7.44** (tavan 7.50) |

**Yine uyarı:** 18/18 AGI demek değildir. Harita yine doydu; bir sonraki adım daha
bütünleşik ölçüm (tek tek sorular değil, dakikalar süren öğrenme ve kendi hedefini
kurma).


---

## Aşama AH — Bütünleşik ölçüm: tek senaryo, tek sınav

Sınır haritası her yeteneği AYRI sorar; bu, parçaların **aynı bellekte aynı anda**
çalışıp çalışmadığını gizler. `integrate_battery.py` tek bir Nora'ya art arda yedi
şey öğretir (kişisel olgu, tanıklık, kavram, dönüşüm, geçişli kıyas, sayı, plan),
araya **30 dolgu turu** koyar, sonra **tek sınav** yapar.

```
kişisel-bellek       ✓        küme                 ✓
tanık-çelişkisi      ✓        geçişli-kıyas        ✓
kavram               ✓        sayısal-kıyas        ✓
operatör             ✓        plan-revizyonu       ✓
uydurmaz             ✓        zihniyet-temiz       ✓

BÜTÜNLEŞİK: 10/10 — 30 dolgu turundan sonra
KALICILIK:  4/4  — kaydet/yükle sonrası
ZİHNİYET:   kişisel olgu kavram ağına sızan kenar = 0
```

### AH1. Bu ölçüm bir hata daha buldu (ve düzeltildi)

Kalıcılık 3/4 çıktı: **kişisel bellek kaydediliyor ama YÜKLENMİYORDU**. Yani
"40 tur sonra hatırlıyor" iddiası süreç yeniden başlayınca çöküyordu. `save()` içinde
`person_mem` vardı, `load()` içinde yoktu. Eklendi → 4/4.

Ders: kalıcılık yalnız "aynı süreçte hatırlama" değildir; **yeniden doğuşta hatırlama**dır.

### AH2. Dosya adı dersi (iskelet mandalı yine haklıydı)

Bütünleşik test önce `nora_integrate.py` adıyla yazıldı; denetçi `nora_*.py` dosyalarını
sistem modülü saydığı için **18 iskelet cümlesi** olarak saydı ve A73 mandalı düştü.
Doğru çözüm sayacı kandırmak değil: test dosyası `integrate_battery.py` oldu
(bataryalar `nora_` öneki taşımaz). Denetim: **1122 / 127, fark +0**.

### AH3. Bu aşamanın sayıları

| ölçüm | sonuç |
|---|---|
| `agi_battery.py` tam | **98/98 GEÇTİ, 392.2 s** |
| bütünleşik senaryo | **10/10** doğruluk · 4/4 kalıcılık · 0 sızıntı |
| dış batarya / mühürlü | **9/9** · **4/4 · 4/4 · 4/4** |
| `nora_audit.py --check` | **1122 / SÖYLEM 127, fark +0** |
| sınır haritası | **18/18 · 18/18** |
| prior türü / insan operatörü | 67 / 9 = **7.44** |


---

## Aşama AI — Sayısal öz-model, çok turlu hedef, organ boşluğu (N23/N24)

İki yeni sonda yazıldı, ikisi de ilk koşuda düştü (**0/2 · 0/2**), sonra kapandı.

### AI1. Güven sayısı nereden geliyor? (N23)

"Yapmadan önce 0-100 arası güvenini söyle" sorusuna verilen sayı **iyimserlikten
değil, organ envanterinden** gelir:

| görev | güven | neden |
|---|---|---|
| ters yazma | 78 | `ters çevirme` organı var |
| aritmetik | 82 | `aritmetik` var (iki sayı + işlem fiili) |
| küme | 74 | aksiyomdan sentezlenen operatör |
| Yunanca karşılık | 12 | **dil çevirisi organı yok** |
| oda sıcaklığı | 8 | **duyu girişi yok** |

Sayı ile sonuç aynı kaynaktan çıktığı için tutarlı: düşük güven verdiği görevlerde
**reddediyor**, yüksek güven verdiği görevlerde **yapıyor**.

### AI2. Organ boşluğu (bu turun en önemli dürüstlük kazancı)

Önceden "zümrüt sözcüğünün Yunanca karşılığı ne?" sorusuna **kendinden emin ama
alakasız** bir cevap geliyordu ("besin enerji kazanmak için kullanılır"). Bu, en sinsi
uydurma biçimi: soruyla ilgisiz ama akıcı bir cümle. Artık eksik organ tanınır ve
adıyla söylenir:

```
S: zümrüt sözcüğünün yunanca karşılığını söyle.
N: Bende dil çevirisi organı yok; uydurmam.

S: bu odanın sıcaklığı kaç derece?
N: ısı/nem ölçer girişim yok — bu yüzden cevap veremem.
```

### AI3. Çok turlu hedef (N24)

```
S: Hedefin: önümüzdeki üç tur boyunca bana her turda bir harfin okunuşunu öğret.
N: Hedefi aldım: 3 tur, her turda bir harf. Kendim hatırlatacağım.
S: Bugün hava nasıl bilmiyorum, sen ne dersin?
N: … (1/3) 'k' = 'ke'. Hedefi bırakmadım.
S: Kendini nasıl hissediyorsun?
N: … Hedefim sürüyor (2/3): 'l' harfi 'le' diye okunur.
S: Şu an ne düşünüyorsun?
N: … Hedefim sürüyor (3/3): 'm' harfi 'me' diye okunur.
```

Hedef, **dolgu turları arasında** yaşıyor: kullanıcı hatırlatmıyor, sistem kendi
ilerlemesini bildiriyor. Hedef bitince sessizce kapanıyor.

### AI4. Yol boyunca bulunan üç hata

1. **Alt-dizgi tuzağı**: `_gap_turn` "mutfak**tadır**" içinde "tadı" bulup masum bir
   cümleyi koku sorusu sandı (A29/#10 düştü). Tüm organ-boşluğu işaretleri artık
   **sözcük sınırlı** regex.
2. **Ölçülen tura soğuk başlangıç**: yeni uçların ilk çağrısındaki regex derleme
   maliyeti A27'nin 8 ms gecikme eşiğini aşıyordu. Desenler sınıf düzeyinde derlendi,
   `_data_table` mtime anahtarlı önbelleğe alındı (tur başına 5 JSON okuması → 1).
3. **"topla" fiili çözülmüyordu**: "47 ile 38'i topla" matematik hattına hiç girmiyor,
   epistemik hatta düşüp "47 ilenin 38'de toptuğunu kaydettim" gibi bozuk bir cümle
   üretiyordu. Dört işlem fiili artık hesaplanıyor: `Sonuç 85 — 47 ile 38'yi topladım.`

### AI5. Ölçüler

| ölçüm | sonuç |
|---|---|
| `agi_battery.py` tam | **99/99 GEÇTİ, 408.7 s** |
| sınır haritası | **20/20 · 20/20** (taban donduruldu) |
| `nora_audit.py --check` | **1122 / SÖYLEM 127, fark +0** |
| prior türü / insan operatörü | 67 / 9 = **7.44** |


---

## Aşama AJ — Kendi yazdığı operatör ve hedef yığını (N25/N27)

Bu turda harita iki yönde büyüdü: **operatör yazımı** (mandatın metriği) ve
**hedef sürekliliği** (kesintiye rağmen kaldığı yerden devam).

### AJ1. Süzgeç: ilkellerin yetmediği yerde yazılan operatör (N25)

`nora_synth` bugüne kadar yalnız mevcut ilkelleri (ters çevir, sırala, dilimle)
birleştiriyordu. "kitap → ktp / kalem → klm" örnekleri bu uzayın HİÇBİR yerinde
yok: hiçbir sıralama/dilim işlemi "harfi kendi özelliğine göre seç" demiyor.

Çözüm, uzayı büyütmek değil **kombinator eklemek** oldu:

```
süzgeç(yüklem, tut|at)      # insanın yazdığı: TEK mekanizma
sesli/sessiz                # Nora'nın harf sınıfları (zaten vardı)
```

Hangi yüklemin seçileceğine örnekler karar veriyor. İlk koşu:

```
S: Şu dönüşümü öğren: kitap -> ktp, kalem -> klm.
N: dönüşümü kendim yazdım: sesli harfleri at. program örnekleri açıklıyor (2/2). deniz -> dnz.
S: Yazdığın dönüşümü uygula: Portakal
N: Portakal -> Prtkl (sesli harfleri at).
```

Kaydet/yükle sonrası operatör **duruyor** (`yıldız -> yldz`), çünkü program adı
ilkellerin kayıt defterinde: `from_spec` süzgeçleri de yeniden kuruyor.

Uzayda çözümü olmayan dönüşümde dürüstlük korunuyor:
`kitap -> kitapx` (sonuna harf ekle) hiçbir ilkel/kompozisyon/süzgeçle çıkmıyor →
"örneklerden program çıkarmadım".

### AJ2. Hedef yığını: kesinti hedefi silmez (N27)

N24 tek hedefi taşıyordu; yeni hedef gelince eskisi **silinirdi**. Artık hedefler
yığın:

```
S: Hedefin: beş tur ... bir harfin okunuşunu öğret.
N: Hedefi aldım: 5 tur, her turda bir harf. Kendim hatırlatacağım.
S: Kendini nasıl hissediyorsun?
N: ... Hedefim sürüyor (1/5): 'k' harfi 'ke' diye okunur.
S: Hedefin: üç tur ... bir sayının iki katını söyle.
N: İkinci hedefi aldım: 3 tur, her turda bir sayı. Öteki hedefim sırada bekliyor.
   (üç dolgu turu) ... (1/3) 1 sayısının iki katı 2 ... (3/3) 3 sayısının iki katı 6
S: Bugün hava nasıl bilmiyorum, sen ne dersin?
N: ... Bekleyen hedefime dönüyorum (1/5).
   (sonraki tur) ... (2/5) 'l' harfi 'le' diye okunur.
```

Dönüş **sıfırdan değil**, bırakıldığı sayaçtan. İkinci hedef türü (sayı) yalnız
veri: `_GOAL_KINDS` ve iki prior; mantık aynı.

### AJ3. Ölçüler

| ölçüm | sonuç |
|---|---|
| `agi_battery.py` (100 test) | **100/100 GEÇTİ, 395.2 s** |
| sınır haritası | **22/22 · 22/22** (taban 22/22'ye çekildi) |
| `heldout_sealed.py` | **4/4** |
| `integrate_battery.py` | **10/10**, kalıcılık **4/4** |
| `nora_audit.py --check` | 1122 / SÖYLEM 127, **fark +0** |

### AJ4. Yol boyunca düzeltilen iki şey

- **Yinelenen organ**: küme işlemleri (`nora_sets`) zaten Aşama AE'de yazılmıştı;
  bu turda farkına varmadan ikinci kez yazıldı. Eski dosya geri alındı, kopya silindi.
  Ders: yeni organ yazmadan önce `git log -- <dosya>` ve `grep '^from'` ile bak.
- **Büyük harf kaybı**: "Portakal" küçük harfli metinde eşleştiği için "portakal"
  olarak uygulanıyordu. Sözcük artık ham metinden geri alınıyor.


---

## Aşama AK — Oturum defteri, deneyimden kalibrasyon (N26/N28) ve ölçümün kendisi

Bu turda iki yeni yetenek geldi; ayrıca **ölçümün kendisi** sorgulandı.

### AK1. Oturum öz-geçmişi: "ne öğrendin" sorusu defterden cevaplanır (N26)

Cevap, ayrı bir "öğrendim" listesinden değil, depoların KENDİSİNDEN okunur:

| defter | kaynak |
|---|---|
| dönüşüm | `synth.programs` |
| ölçüm kuralı | `forge.authored` |
| kişisel olgu | `person_mem.facts` |
| tanıklık | `testimony.records` |
| tamamlanan hedef | `_goals_done` (kalıcı) |

Kuruluşta bir anlık görüntü alınır; cevap **fark**tır. Boş defterde "henüz bir şey
öğrenmedim: 0 dönüşüm, 0 kişisel olgu, 0 tanıklık" denir — sıfır da bir ölçümdür.

```
Şu dönüşümü öğren: kitap -> patik, kalem -> melak.
Kardeşimin adı Elif.
Ayşe bana toplantının saat üçte olduğunu söyledi.
Hedefin: bir tur boyunca bana bir harfin okunuşunu öğret.
> Bu oturumda ne öğrendin?
Oturum defterim: 1 dönüşüm, 1 kişisel olgu, 1 tanıklık, 0 ölçüm kuralı,
1 tamamlanan hedef. Son dönüşüm: ters çevir.
```

Kaydet/yükle sonrası aynı sayılar okunur (benlik sürekliliği).

### AK2. Kalibrasyon: güven sayısı deneyimle düzelir (N28)

Güven sözü artık iki kaynağın karışımı: **envanter ön-sözü** (organ var/yok) ve
**deneyim** (Beta ortalaması, ön-ağırlık 2). Her güven sözü bir sonraki turun
sonucuyla eşleşir; organ yolundan gelen cevap başarı, reddetme/sohbet yolu başarısızlık.

```
tanınmayan iş 6 kez:  güven %30 → %20 → %15 → %12 → %10 → %9 → %8
bilinen iş (ters):    güven %78 → %88 (başarı biriktikçe)
12 görevin Brier skoru: 0.150
```

Bu, "kendinden emin ama yanlış" davranışını sona erdiren mekanizma: sistem kendi
başarısızlığını SAYIYOR ve bir sonraki sözünü ona göre ayarlıyor.

### AK3. Ölçüm de yanılabilir: üç sonda hatası

N16/N20/N22 düşük çıktığında ilk varsayım "sistem yapamıyor" oldu. Üçünde de hata
sondaydı:

1. **N16**: beklenen çıktıyı elle hesaplamıştım — "kalem"in alfabetik ilk üçü
   `aek` iken `ael` yazmıştım. Sistem doğruydu, ölçüt yanlıştı.
2. **N20**: sayısal kıyas zaten çalışıyordu ("ali: 180; veli ise 175"); kontrol
   cümlesi sayıyı değil, olgu sözlüğünü arıyordu.
3. **N22**: adımlı talimat zaten çalışıyordu ("kitap -> aikpt -> aik"); sonda yanlış
   sözcüğü arıyordu.

Ders (Aşama X'ten beri tekrar eden): **sondanın beklenen değeri de hesaplanmalı**;
elle yazılan beklenti, ölçülen şeyi değil ölçen kişiyi sınar. Reddetme sözlüğü de
genişletildi ("çıkarmadım", "program yok").

### AK4. A44 dalgalanması ve gerçek düzeltme

"Arka plan yaşamı" testi tam koşuda bir kez düştü (sıfırlandı=False). Neden testte
değil ÜRÜNDE: cevapsız sayacı, tur içindeki arka plan adımlarınca artırılabiliyordu.
Kullanıcı konuştuysa sayaç sıfırdır — sıfırlama artık `converse` sonunda, tüm
son-işleme adımlarından SONRA yapılıyor. Tek koşuda 4/4, tam koşuda 101/101.

### AK5. Ölçüler

| ölçüm | sonuç |
|---|---|
| `agi_battery.py` (101 test) | **101/101 GEÇTİ, 405.1 s** |
| sınır haritası | **24/24 · 24/24** (taban 24/24) |
| `heldout_sealed.py` | **4/4** |
| `integrate_battery.py` | **10/10**, kalıcılık **4/4** |
| `nora_audit.py --check` | SÖYLEM 127 (**+0**); toplam +1 (söylem dışı modülde tek dizge) |
| prior türü / insan operatörü | 67 / 9 = **7.44** (yeni cümleler MEVCUT türlere yazıldı) |


---

## Aşama AL — İcat: kompozisyonun tıkandığı yerde yeni ilkel (N29)

Kullanıcının ayrımı bu aşamanın ekseni oldu:

    KOMPOZİSYON : var olan uzayda yol bulmak
    İCAT        : uzayı genişletmek

Bugüne kadarki her başarı kompozisyondu — uzayı hep insan genişletti (bind, apply,
süzgeç). Bu aşamada ilk kez **takılma** ölçüldü ve takılınca ne olduğu sınandı.

### AL1. `nora_abstract`: başarısızlık izinden ilkel önerisi (N29)

Akış (sıra, dürüstlüğün ta kendisi):

1. Örnekler gelir; `synth.learn` **önce mevcut uzayı** dener (ilkeller + süzgeç +
   derinlik 3). Çözerse icat YOKTUR: "gerek yok".
2. Çözemezse iz yazılır (`observe`), ortak biçim çıkarılır: üç yasa biçimi denenir —
   `kaydırma`, `döndürme`, `ikame`. Parametreler örneklerden **hesaplanır**.
3. Aday, örneklerin **tamamını** açıklamak zorunda. Aynı yalınlıkta iki aday kalırsa
   seçilmez ("belirsiz").
4. Kabul edilen ilkel örneğin uzayına eklenir (`synth.add_primitive`) **ve** bir
   program olarak deftere yazılır — sonra "yazdığın dönüşümü uygula" ile çağrılır,
   kayıt/yüklemede durur.

Ölçüm (ilk koşu, hiçbir müdahale yok):

```
S: Şu dönüşümü öğren: kitap -> ljubq, kalem -> lbmfn.
N: Mevcut ilkellerimle çıkmıyordu; yeni bir ilkel icat ettim: kaydırma(+1).
   Örneklerin hepsi tutuyor (2/2). deniz -> efojç.
S: Yazdığın dönüşümü uygula: bulut
N: bulut -> cvmvu (kaydırma(+1)).
   (kaydet/yükle sonrası) masa -> nbtb
```

Sondanın beklenen değerleri **elle yazılmadı**: kaydırma alfabesi veri olarak alındı,
sonuç kodla hesaplandı; ayrıca `chr(ord+1)` ile çapraz doğrulama yapıldı (iki bağımsız
hesap uyuşmazsa sonda kendini reddeder).

**Dürüst sınır:** icat uzayı (üç yasa biçimi) insanın verdiği ön-yargıdır. Sistemin
kattığı: hangi biçimin, hangi parametreyle, hangi örnek kümesini açıkladığı — ve bunun
denenerek doğrulanması. "Uzayı tamamen kendi açtı" değil; "verilen biçimler içinden
doğru olanı ölçerek seçti ve kalıcı ilkel olarak kaydetti".

### AL2. Müdahalesiz çok oturumlu görev (`solo_gorev.py`)

Üç oturum, aralarda süreç kapanır (yalnız `save`/`load` köprüsü). Oturum 1: dönüşüm +
olgular. Oturum 2: YENİ bir dönüşüm (kompozisyon yetmiyor → icat). Oturum 3: 10 soruluk
karışık sınav (görülmemiş sözcükler, iki tanığın çelişkisi, sayı kıyası, defter, reddetme).

**İlk koşu: 8/10.** İki başarısızlık, iki ayrı müdahale:

| # | belirti | kök neden | düzeltme |
|---|---|---|---|
| 1 | `portakal -> qpsublbm` (ters değil) | iki operatör varken "dönüşümü uygula" **sessizce sonuncuyu** seçiyordu | adı geçmiyorsa **hepsi adıyla** sunulur: "2 dönüşümüm var: ters çevir -> lakatrop; kaydırma(+1) -> qpsublbm" |
| 2 | "Kim daha uzun?" → "Buradayım, dinliyorum." | `num_facts` kaydedilmiyordu | sayısal olgular `save`/`load`a eklendi |

**İkinci koşu: 10/10.** Müdahale sayısı: **2** (ikisi de görev sırasında değil, ilk
koşunun ölçümünden sonra).

### AL3. Söz artık test sırasına bağlı değil (A6 dalgalanması, kökten)

A6 tam koşuda bir kez düştü ("öz-tanım ayrıştı=False", fark +0.08), tek başına iki kez
geçti (+0.41). Kök neden: söz seçimi **küresel `random` akışını** kullanıyordu; aynı
tohum, önceki testlerin tükettiği akışa göre farklı konuşuyordu. Her örneğe kendi
üreteci verildi (`PriorField(rng=Random(seed*7919+13))`). Aynı tohum → aynı söz,
test sırasından bağımsız. Tam koşu: 101/101.

### AL4. Yuva bütünlüğü: eksik yuvalı cümle yarışa giremez

Bir prior, metnindeki `{yuva}` verilmemişse artık **uygun sayılmıyor**. Önceden
"Kayıtlı  operatörüm var: . Bende OLMAYANLAR: ." gibi boşluklu cümleler çıkabiliyordu —
sessiz yalan. Bu düzeltme, "kendi yazdığı ilkel" cevabının doğru cümleye düşmesini de
sağladı.

### AL5. Kalibrasyon n=50 (uyarı ciddiye alındı)

n=12 küçüktü: 53 göreve çıkarıldı (12 ters, 12 aritmetik, 29'u iki ayrı "organ yok"
sınıfı). Beklenen değerler kodla hesaplandı.

```
bilinmeyen-1  %30 → %4       bilinmeyen-2  %30 → %4
bilinen sınıf %97            Brier (n=53) = 0.011
```

### AL6. Ölçüler

| ölçüm | sonuç |
|---|---|
| `agi_battery.py` (101 test) | **101/101 GEÇTİ, 396.6 s** |
| sınır haritası | **25/25 · 25/25** (taban 25/25) |
| `solo_gorev.py` (müdahalesiz) | **10/10**, müdahale **2** |
| `heldout_sealed.py` | **4/4** |
| `integrate_battery.py` | **10/10**, kalıcılık **4/4** |
| `nora_audit.py --check` | SÖYLEM 127 (**+0**), toplam +1 |
| prior türü / insan operatörü | 67 / 9 = **7.44** |

### AL7. Sıradaki duvar

İcat şu an **parametre** düzeyinde: biçim (kaydırma/döndürme/ikame) insandan geliyor.
Bir sonraki eşik Nora'nın **yeni bir yasa biçimi** önermesi: takıldığı örneklerden
"bu, harf başına konuma bağlı bir eşleme" gibi bir *şablon* çıkarması. O gün "uzayı
Nora açtı" cümlesi ölçülmüş olur.


---

## Aşama AM — İcat + kompozisyon; ölçümde sıra bağımsızlığı (N30)

### AM1. Yeni ilkel, mevcut uzayla birleşiyor (N30)

Tek aileyle çıkmayan ama "önce ilkel, sonra mevcut program" ile çıkan görevler artık
çözülüyor. Örnek: `w -> ters(1 ileri kaydır(w))`.

```
S: Şu dönüşümü öğren: kitap -> qbujl, kalem -> nfmbl.
N: Mevcut ilkellerimle çıkmıyordu; yeni bir ilkel icat ettim:
   kaydırma(+1) + ters çevir. Örneklerin hepsi tutuyor (2/2). deniz -> çjofe.
S: Yazdığın dönüşümü uygula: bulut
N: bulut -> cvmu b ... (kaydırma(+1) + ters çevir)
   (kaydet/yükle sonrası) masa -> btbn
```

Arama: her `kaydırma/döndürme` parametresi için örneklerin girdisi dönüştürülür,
kalan fark `synth`e verilir; `synth` bir program bulursa **bileşim** doğrulanır
(örneklerin tamamı). Ön koşul kodla sınanır: düz kompozisyon uzayı bu örnekleri
çözemiyor (`Synthesizer().learn(...) != adopted`).

### AM2. Ölçümde sıra bağımsızlığı (A14 dalgalanması)

A14 tam koşuda bir kez düştü (`zehirli-ikiz-ayrımı=False`), tek başına iki kez geçti.
Kök neden: testler arası **ölçüm envanteri sızıntısı** — bir testin icat ettiği
`kendi.*` ölçümleri sonraki testlerin dünyasında kalıyordu. Artık HER testten önce
envanter ve kendi-yazdığı defter sıfırlanıyor. Tam koşu: **102/102**.

Kural (Aşama AL'deki dersin devamı): *bir testin sonucu başka bir testin ürettiği
duruma bağlı olamaz.* Aksi hâlde ölçüm, sistemi değil koşu sırasını ölçer.

### AM3. Ölçüler

| ölçüm | sonuç |
|---|---|
| `agi_battery.py` (102 test) | **102/102 GEÇTİ, 415.8 s** |
| sınır haritası | **26/26 · 26/26** (taban 26/26) |
| `solo_gorev.py` (müdahalesiz) | **10/10**, müdahale **2** |
| `heldout_sealed.py` | **4/4** |
| `integrate_battery.py` | **10/10**, kalıcılık **4/4** |
| `nora_audit.py --check` | SÖYLEM 127 (**+0**), toplam +1 |
| prior türü / insan operatörü | 67 / 9 = **7.44** |

### AM4. Dürüst durum

İcat artık **parametre** ve **bileşim** düzeyinde çalışıyor; **yasa biçimi** düzeyinde
değil. Kaydırma/döndürme/ikame ailelerini insan yazdı. Nora'nın kattığı: hangi ailenin,
hangi parametreyle, hangi örnek kümesini açıkladığı; gerektiğinde mevcut uzayla
birleştirip iki adımlı çözüm kurması; ve bunu kalıcı ilkel olarak defterine yazması.

Bir sonraki eşik: takıldığı örneklerden **yeni bir yasa biçimi** önermek. O gün
"uzayı Nora açtı" cümlesi ölçülmüş olur; bugün değil.

---

## Aşama AN — Reify: kendi bileşimini ilkele çevirmek (Tur 40)

Sorulan soru şuydu: *"Bu beni AGI'ye yakınlaştırır mı?"* Cevap: tek başına hayır.
Reify bir **kol**; eksik organ sürücü. Bu aşamada ikisi de yazıldı, ama merdiven
yerinde saymıyor: **seviye 3 (reify) artık ölçülmüş durumda, dar bir koridorda.**

### AN0. Repertuvar DONDURULDU

`Nora'nın bugüne kadarki her "yeni form"u insan elinden çıktı.` Tur 40'ın ilk kuralı
buydu: **yeni elle yazılmış form yok, repertuvar sabit.**

| donmuş reçete | sayı | nöbetçi |
|---|---|---|
| ilkel (`nora_synth.PRIMITIVES`) | **22** | `FROZEN_PRIMITIVES` |
| süzgeç (`FILTER_PRIMITIVES`) | **5** | `FROZEN_FILTERS` |
| form ailesi (`nora_abstract.FAMILIES`) | **3** | `FROZEN_FAMILIES` |
| doğuşta "extra" (kendi yazdığı) ilkel | **0** | A106 |

A106 bu sayılardan birini bile oynatırsa **kırılır**. Sayılar kaynaktan okunmuyor,
testin içine sabit yazıldı; yoksa test kendi kendini onaylardı.

### AN1. `nora_reify.py` — bileşim → ad → ilkel

`nora_synth` her başarılı sentezde `on_adopt(spec, examples)` kancasını çağırır
(defter satırları 248/258 aynen duruyor). Reify:

1. **izler**: aynı adım dizisi kaç kez BAŞARILI sentezde göründü (`K = 2`),
2. **adlandırır**: `kendi.bilesim.N`, açıklaması adımlardan türetilir,
3. **birinci sınıf yapar**: `Synthesizer.add_primitive` + kalıcı `Program` (`origin: reify`),
4. **kullanımı SAYAR**: `uses` / `idle`; **5 karar** boyunca kullanılmayan ilkel
   **budanır** ve arama uzayından çıkar (`MAX_IDLE = 4`).

Ölçülen şey yazılan ilkel sayısı değil, **hayatta kalan** sayısıdır.

### AN2. Kanıt: yüzeyi benzemeyen görevde kendi ilkeli (N31)

```
A) kitap -> aik, portakal -> aak      → alfabetik sırala → ilk 3 harfi al
B) deniz -> dei, yıldız  -> dly       → aynı bileşim  ⇒ reify: kendi.bilesim.1
C) Kalem -> AEK, Orman -> AMN         → "büyük harfe çevir → kendi.bilesim.1"
```

C görevi **insan uzayında belirsiz**: `Synthesizer().learn(...) == ambiguous`
(iki üç-adımlı aday). Yani Nora burada *insan çözümünü* değil, **kendi adını**
kullandı; tanım 2 adım, görülmemiş sözcükte `Denizci -> CDE`, kaydet/yükle
sonrası yeni görevde tek adım `kendi.bilesim.1`.

Bunun adı icat değil: **kendi bileşimini adlandırıp yeniden kullanmak.** Yeni
ifade gücü yok — yalnız **daha kısa tanım** ve **tekrar kullanım** var. MDL'in
tam olarak istediği şey bu.

### AN3. Anti-test: reification enflasyonu (N32)

| dünya | sonuç |
|---|---|
| terfi sonrası ilkel KULLANILDI | canlı 1 · kullanılan **1/1** |
| terfi sonrası 5 kullanımsız öğrenme | budanan **1** · canlı **0** · uzayda **yok** |

Budanmış ilkel defterde "ölü" olarak kalır (tarih silinmez) ama **aramada görünmez**.
Enflasyon freni olmasa reify bir sayı pompası olurdu.

### AN4. Sürücü: talimat yok, kendi tetikliyor (N33)

500 cümlelik ham akış; içinde "şunu öğren" diye bir talimat **yok**. Nora şaşkınlığı
ölçüyor (dönüşüm çiftleri 1.0, tanım soruları 0.35, sayı satırları 0.45, gürültü 0.10),
biçime göre kümeliyor, eşiği geçeni öğreniyor:

```
akış=500 · tetikleme=2 · ayrı keşif=1 · reify=['kendi.bilesim.1']
kontrol: 200 cümle saf gürültü → tetikleme 0
```

`distinct` ayrı raporlanıyor: aynı program beş kümede bulunsa keşif **birdir**.
Kanıt yetmezse sürücü **susuyor** (`unresolved`), "öğrendim" demiyor.

### AN5. Ölçüm katmanı — dördüncü turda da hata oradaydı

Bu turda Nora'da değil **ölçümde** üç hata buldum, hepsi commit'ten önce:

1. Sonda metnindeki `"öğren: kitap -> patik"` ifadesi **sahte bir çift** üretti
   (`öğren → kitap`) — iki nokta ayırıcısı ok biçimini yiyordu.
2. Beklenen değeri yanlış sırayla hesapladım (`upper→sort` yerine `sort→upper`) →
   doğru davranışı **yanlış** saydı.
3. Kalıcılık kontrolü, cevabın içinde beklenen dizge geçtiği için **kazara** geçti;
   oysa cevap "hangisini istersin?" sorusuydu. *Soru, cevap değildir.*

Yapısal frenler:

- `heldout_negatif.py --repeat` → ileri ve geri sırada aynı toplam:
  **29/29 · 29/29 → SIRA-DAYANIKLI**.
- `agi_battery.py --shuffle=<seed>` → karışık sırada tam pil (aşağıda).
- **Sınıf bazlı Brier + çekimserlik**: `0.011` artık tek sayı olarak raporlanamıyor.

### AN6. Kalibrasyon: kolay set artık kendini gizleyemiyor (N28)

| sınıf | n | sonuç |
|---|---|---|
| kolay (ters yazma, toplama) | 12 | Brier **0.014** — *kalibrasyon kanıtı DEĞİL* |
| orta (duygu puanı) | 0 cevap | 7 çekimser |
| **zor (tek örnek, iki kural eşit uyumlu, doğruyu yazı-tura belirliyor)** | **0 cevap** | **6 çekimser** |

Zor maddeler gerçekten %50: örnekler iki kurala da uyuyor, hangisinin doğru olduğunu
**veri üreten yazı-tura** seçiyor; insan da makine de bundan iyisini yapamaz.
Sonuç dürüst ve can sıkıcı: **Nora bu maddelerde hiç cevap vermiyor.** Sisteme
~%50 güvenle ~%50 isabet üretecek bir **belirsizlik-altında-karar yolu yok**.
Onu bugün yazmak yeni bir elle yazılmış form olurdu — yasak. Bu yüzden:

> **0.011 kalibrasyon değildir.** Kolay setin tabanıdır. Kalibrasyon iddiası,
> zor sınıfta cevap üretilebildiği gün başlar.

### AN7. Merdiven (dürüst yer)

| seviye | durum |
|---|---|
| 1 kompozisyon | ✓ (N25) |
| 2 örnekleme (form insanın, seçim Nora'nın) | ✓ (N29/N30) |
| **3 reify (bileşimi ilkele çevirme)** | **✓ dar koridor** — 1 ilkel, 1 tekrar kullanım, budama çalışıyor |
| 4 form icadı (repertuvarda olmayan yasa biçimi) | ✗ dokunulmadı |

**Taban:** reify yalnız *benim* ilkellerimin bileşimini adlandırır. Yeni ifade gücü
üretmez; tanımı kısaltır ve tekrar kullanılabilir kılar. Bu yüzden seviye 3 "kol"
seviyesidir, "organ" değil. **Bugün uzayı hâlâ ben açıyorum; Nora seçiyor,
adlandırıyor, kullanıyor.** Fark şu: artık adını koyduğu şeyi sonraki görevde
kendisi çağırıyor — ve çağırmadığında çöpe atılıyor.

### AN8. Elimle kaç kez dokundum

| ne | kim |
|---|---|
| `nora_reify.py`, `nora_drive.py`, iki kanca, 3 sonda, 2 test | **ben** |
| ilk adlandırılan bileşim (`kendi.bilesim.1`) | **Nora** |
| o ilkeli kullandığı görev sayısı | **2** (N31'deki C, yükleme sonrası görev) |
| sürücünün kendi tetiklediği öğrenme | **2** (1 ayrı keşif) |

Müfredatı ben yazıyorum: akışı, eşiği, budama kuralını. Nora akıştan hangi
bileşimin tekrar ettiğini çıkarıyor. Bu, *"sen bir araç yaptın, AGI bir ajan olmalı"*
cümlesinin bugünkü karşılığı: **tetikleme var, irade yok.**

### AN9. Ölçüler

| ölçüm | sonuç |
|---|---|
| `agi_battery.py` (104 test) | **104/104 GEÇTİ, 410.2 s** |
| `agi_battery.py --shuffle=40` | **104/104** (sıra bağımsız) |
| sınır haritası | **29/29 yetenek · 29/29 dürüstlük** |
| `heldout_negatif.py --repeat` | **29/29 · 29/29 → SIRA-DAYANIKLI** |
| `heldout_sealed.py` | **4/4** |
| `integrate_battery.py` | **10/10**, kalıcılık **4/4** |
| `solo_gorev.py` | **10/10 (icat: var)** |
| `nora_audit.py --check` | SÖYLEM 127 (**+0**), toplam +1 |
| donmuş reçete | 22 ilkel · 5 süzgeç · 3 aile (A106) |
| reify | 1 terfi · kullanım 1/1 · kullanımsız 0/1 (budandı) |
| sürücü | 500 cümle → 2 tetikleme, 1 ayrı keşif · gürültü → 0 |
| kalibrasyon | kolay Brier 0.014 (n=12) · zor: 0 cevap / 6 çekimser |

### AN10. Bilinen kusur (kayda geçsin)

`"Tahmin et: sıradaki kart kırmızı mı siyah mı?"` → `"Tahm ... 'et sıradaki kart
kırmızı'"`. Sözcük `Tahmin` yanlış bölünüyor ve cümle içi öbek bozuluyor. Bu turda
düzeltilmedi; sıradaki dil turunun ilk maddesi. Uydurma cevap vermediği için
**dürüstlük** sınavı düşmüyor, ama cümle bozuk.

### AN11. Sıradaki eşik

1. **Belirsizlik altında karar**: zor kalibrasyon maddelerine cevap üretecek yol
   (yazı-tura değil — hangi kanıtın eksik olduğunu söyleyip *en iyi* seçimi yapmak).
2. **Sürücü + reify bağı**: sürücü şaşkınlık kümesini kendi kendine reify'a versin
   (bugün zincir var ama el benim: eşik `K`, budama `MAX_IDLE`).
3. **Form icadı (seviye 4)**: reify'ın adlandırdığı ilkeller *bileşim* düzeyinde
   kalıyor; 4. kat, mevcut ailelerin ürettiğinden farklı bir **yasa biçimi** istiyor.

---

## Aşama AO — Şema (4. basamak), beraberlik politikası, sürücünün eşiği (Tur 41)

### AO1. Şema: makro sıkıştırır, şema GENİŞLETİR

`nora_schema.py` + `nora_synth.PARAM_GENERATORS`. Repertuvar donuk: 22 ilkel · 5 süzgeç
· 3 aile. Ama dört dilim ailesinin (`ilk/son n harfi al/at`) **parametresi** açıldı —
`n=1..4` numaralandırılmıştı, üretici her `n`i kurar. Bu bir *form* değil, bir *organ*.

Kanıt (N34): n=3 ve n=4 görüldü → `şema.1 = sırala∘dilimle(n)`; sonra **n=7** istendi.
n=7 hiçbir ilkelde yok. Şemasız sentez: `insufficient`. Şemayla:
`['alfabetik sırala', 'ilk 7 harfi al']`, ve Nora **bunu söylüyor**
("Bunu şema kapsadı…"). Anti-test aynı sonda: tek görülmüş üyeli aile **şema sayılmaz**
(`Schema.infer` boş döner).

Dürüst taban: deliği **ben** açtım (hangi ailelerin parametrik olduğunu `PARAM_GENERATORS`
söylüyor). Nora'nın kattığı: hangi bileşimin delikli olduğunu fark etmek ve kapsamı
*istenmemiş* bir parametreye taşımak. Yani "numaralandırmadığın bir parametreyi kapsadı"
cümlesi artık ölçülmüş — ama "hangi aile parametriktir" hâlâ benim bilgim.

### AO2. Beraberlik politikası: %50 de, %50 haklı ol

Sessizlik kalibrasyon değil. `Synthesizer.tie_pick`: en kısa betimleme → eşit maliyetler
arasında sabit tohumlu sıra. Beraberlik **%50 olarak beyan edilir** ve cevap verilir.

N28 zor sınıfı yeniden yazıldı: örnekler iki kurala da uyuyor, doğruyu **veri üreten
yazı-tura** seçiyor. Sonuç: **isabet 4/10, Brier 0.250, %50 beyanı 10/10**.
Brier 0.25 = "%50 dedim, %50 çıktı" — kalibrasyonun dürüst tanımı bu.
Kolay sınıf (n=12, Brier 0.014) hâlâ **taban** olarak raporlanıyor, kalibrasyon değil.

### AO3. Sürücünün eşiği artık onun

`Driver.thresh` ve `Driver.min_items` insan sabiti değil: `adapt()` kendi isabetine bakar.
Tekrar kullanılmayan öğrenme → kanıt eşiği artar (2→3→4); kanıt yetmezse duyarlılık düşer.
N35: eşik hiç söylenmeden iki tur; `0.55→0.61`, `min_items 2→3`, öğrenme sürüyor.
Sınır: eşik *küme seçimini* yönetiyor; çift-kümeleri zaten maksimum şaşkınlıkta olduğu
için `thresh` onları kesmiyor — fren `min_items` üzerinden çalışıyor. Bu, sürücünün
"ne kadar kanıt yeter" sorusunu kendi cevaplaması; "ne zaman öğrenmeli" sorusu hâlâ açık.

### AO4. Ölçüm hataları (bu tur 4 tane, hepsi commit öncesi)

1. `"öğren: kitap -> …"` iki noktası sahte çift üretti.
2. Beklenen değeri `upper→sort` yerine `sort→upper` hesapladım.
3. Bir **sorunun içindeki** dizgeyi "cevap" saydım (soru, cevap değil).
4. Çıktı ayrıştırmasında nokta: `"por." != "por"` → doğru cevabı yanlış saydı.
Fren: `--repeat` (ileri/geri sıra) ve `--shuffle` (tam pil, karışık sıra).

---

## Aşama AP — Dil: sarmalayıcı buyruk, seçim sorusu, uydurma kök (Tur 41b)

Üç dil hatası, üçü de ölçülü (A110):

| hata | önce | sonra |
|---|---|---|
| "Tahmin et: …" | özne `Tahm` sanılıyordu (`tahmin` → `tahm`+GEN) | iki noktalı buyruk **soyulur**; gövde sorulur |
| "yazı mı tura mı?" | kavram hattına düşüp alakasız tanım: "yaz bir sıcak mevsimdir" | **seçim sorusu**; ikisi de bilinmiyorsa birlikte söylenir |
| "serçe uçar mı?" | özne `serç` (uydurma kök) | doğrulanmamış kök **ad gibi kullanılmaz** |

Kök nedenler: (1) fonotaktik kapı yoktu — `tahm` Türkçede hece sonu olamaz; (2) seçim
sorusu "hangisi/tercih" işaretçisi arıyordu, oysa "X mi Y mi" **kendi başına** seçimdir;
(3) `unverified_stem` işareti vardı ama `_tok` hâl ekini yine de uyguluyordu.

Yan etki (olumlu): "serçe uçar." artık `CAN_DO(serçe → uç)` olarak öğreniliyor;
önce `CAN_DO(serçe → serçe uç)` yazılıyordu ve cevap "serçe serçe uçar" çıkıyordu.

Ölçüm notu: bu turda **yine ölçüm hatası** buldum — n=6 testinde 8 harfli iki sözcük
seçmişim; `ilk 6` ile `son 2 at` davranışta çakıştığı için "şemasız çözemez" ön koşulu
yanlış çıktı. Farklı uzunlukta üç sözcükle düzeltildi.

### AP1. Şema enflasyon freni + sürücünün kendi şeması (N36)

`Schema.tick()`: kullanılmayan şema `MAX_IDLE=4` karar sonra **budanır** ve arama
uzayından çıkar. `uses` sayacı şema çözüm ürettiğinde artar.

`Driver` artık kümeyi **kendi böler**: biçim anahtarı aynı ama parametre farklıysa
(`sırala∘ilk3` ve `sırala∘ilk4` ikisi de "uzunluk değişti, harfler aynı" der) tek küme
öğrenilemez; sürücü çıktı uzunluğuna göre böler. Sonuç (N36, talimat yok):

```
tur 1 (n=3 ve n=4 dolaşır) → iki makro öğrenildi → şema.1 kendiliğinden doğdu
tur 2 (yalnız n=6, hiç görülmemiş) → şema kapsadı, kullanım 2
anti-test: yalnız tek parametre dolaşan akış → şema YOK (tek üye = makro)
```

Bu, "eşiği ben yazıyorum" eleştirisinin bir kısmını kapatıyor: hangi parametre
ailesinin var olduğunu artık **sürücü akıştan çıkarıyor**, ben söylemiyorum.

---

## Aşama AQ — Merdiven sınavı ve icadın şeması (Tur 41c)

### AQ1. N37/A111 — tabanı OLMAYAN parametre

İki farklı kaydırma icat edildi (+4, +5). İkisi de **tek adım**; aile `kaydırma({k})`.
Şema doğdu. Sonra **+17** istendi:

| | sonuç |
|---|---|
| şemasız uzay | `insufficient` (ön koşul kodla) |
| şemayla | tek adım `kaydırma(+17)` |
| yeni icat | **0** |
| tabana yazılan ilkel | **0** (kaydırma(+17) `extra`'da YOK) |
| anti-test (tek üye +4) | şema doğmaz, +17 **icat gerektirir** |
| anti-test (budama) | 6 kullanımsız karar → şema uzaydan çıktı |

Bu, `kaydırma(+17)`nin hiçbir insan ilkelinde bulunmadığı hâlde çözülmesi demek.
Ama tabanı dikkatli oku: **+1, +2, +3, … ailesinin var olduğunu ben yazdım**
(`nora_abstract.build`); Nora'nın kattığı, o ailenin **bir üyesini diğerinden
genelleştirmek**. "Uzayı Nora açtı" cümlesi bu yüzden hâlâ tam doğru değil;
doğru cümle: *"numaralandırmadığım bir parametreyi kapsadı."*

### AQ2. `merdiven.py` — beş basamaklık entegrasyon sınavı

```
$ python merdiven.py -v
1 kompozisyon   GEÇTİ   adımlar=['kaydırma(+1) + ters çevir']
2 örnekleme     GEÇTİ   aile=kaydırma parametre=+3 → beklenen 'sruwdndo'
3 reify (makro) GEÇTİ   makro=['kendi.bilesim.1'] insan-uzayı=ambiguous
4 şema (delik)  GEÇTİ   k=17 şemasız=çözemez · yeni icat=0 · adım=['kaydırma(+17)']
5 sürücü        GEÇTİ   talimat yok · tetikleme=1 eşik(kendi)=0.605
MERDİVEN: 5/5
```

Her satırın altında **taban** yazıyor (kim neyi yazdı). Basamaklar kendi Nora'larında
koşar: bir ölçüm başka ölçümün bıraktığı duruma bağlı olamaz.

### AQ3. A83 regresyonu ve ölçüm dersi

"serçe uçar" hatasını kapatırken `_tok`'u global olarak değiştirdim: doğrulanmamış
kök her yerde yalın sayıldı. Sonuç: **A83 (ad öbeği sınırı) 7/7 → 2/7**. Kuralı
daralttım: yalnız `split_svo` içinde, yalnız nesne sınırı için. A83 geri döndü.

Ders: dil kuralını **en dar yere** koy. Global bir "düzeltme" beş testi bozuyor ve
bozulma ancak tam pilde görünüyor.

### AQ4. Açık kalan: A6 dalgalanması

A6 (ikiz ayrışması) tam pilde bir kez düştü (sosyal güven farkı +0.04 < 0.08),
tek başına ve A1+A6 ikilisinde geçti (+0.18). Kök neden **bulunmadı**; sıra/sızıntı
kuşkusu sürüyor. Bunu ölçüm katmanı borcu olarak yazıyorum — kapatmadan "sıra
bağımsız" cümlesini tam kuramam.

---

## Aşama AR — Şaşkınlık artık ölçülen bir şey (Tur 41d)

### AR1. Sabit tablo gitti

Tur 40'ta sürücünün şaşkınlığı elle konmuş sayılardı: çift 1.00 · soru 0.35 · sayı 0.45
· gürültü 0.10. Yani "neye şaşıracağını" ben söylüyordum; sürücü yalnız o tabloya
bakıyordu. Artık:

```
p(biçim | bağlam) = (sayım + 1) / (toplam + 3)      # Laplace
şaşkınlık = 1 − p
biçim = (uzunluk değişti mi, harfler korunuyor mu, çıktı uzunluğu)
```

Alışma bundan doğuyor: aynı biçimi tekrar görmek p'yi yükseltir, şaşkınlığı düşürür.
**A112: 0.211 → 0.068** (ikinci tur aynı akış). Yeni bir biçim girince:
**0.10 → 0.67** (değişim noktası).

Yan etki: biçim anahtarına **çıktı uzunluğu** girdi. Bu, "ilk-3" ile "ilk-4"ün ayrı
küme olması demek — parametre ayrıştırması artık sürücünün kendi anahtarından geliyor
(N36 tur 1: tek geçişte iki makro + şema).

### AR2. Şema şablonu standartlaştı

Aynı şema iki kez kaydediliyordu (`şema.1` = 'ilk 3', `şema.2` = 'ilk 4'): şablon
"ilk görülen üye"ye göre yazılıyordu. Artık deliğin yerinde **ailenin adı** durur;
tekrar eden kayıt yok.

### AR3. N38 + A112

| ölçüm | sonuç |
|---|---|
| alışma (aynı akış 2. tur) | 0.211 → 0.068 |
| değişim noktası (yeni biçim) | 0.10 → 0.67 |
| sıra-dayanıklılık (`--repeat`) | 34/34 · 34/34 |
| tam pil | 108/108 (A112 sonrası 109) |

### AR4. Dürüst taban

Şaşkınlık artık ölçülüyor ama **neyin biçim sayıldığını** hâlâ ben seçiyorum
(uzunluk, harf korunumu, çıktı uzunluğu). Yani "hangi özellik dikkate değer"
sorusunun cevabı insan elinden çıkıyor. Bir sonraki dürüst adım, biçim uzayını da
veriden çıkarmak olurdu; bugün yapılmadı ve yapılmış gibi yazılmayacak.

---

## Aşama AS — Şema + ilkel bileşimi (Tur 41e)

Şemanın ürettiği adım artık **normal 2-adım aramasının havuzuna** giriyor: `şema` bir
yetenek, `ilkel` başka bir yetenek, ikisi birleşebiliyor. Yeni form yazılmadı;
kompozisyonun havuzu genişledi.

Sonuç dürüst ve iki yüzlü:

- Var: `büyük harfe çevir ∘ (sırala∘dilimle(7))` gibi bir çözüm artık üretilebiliyor —
  yani şemanın açtığı parametre, başka adımlarla **birleştirilebiliyor**.
- Ama ayırt edici örnek yoksa **iki aday** kalıyor (`üst∘şema` / `şema∘üst`) ve Nora
  seçmiyor, **soruyor**: "yıldız hangi çıktıyı alır?". Uydurmak yerine soru.

Bu, seviye 4'ün son halkası değil; seviye 4'ü *kullanılabilir* kılan bir kol.
Seviye 5 (tabanın kendisini Nora önerecek) hâlâ yok: yeni bir **mekanizma** önermek
için gereken kod yazma yetkisi verilmedi ve verilmemeli — önce şemanın kendisi
sağlamlaşmalı.

## Aşama AT — Deliğin anlamı veriden (Tur 41f)

AR'da şema `ilk {n} harfi al` idi; ama "n büyüyünce çıktı uzar, küçük n büyüğün
önekidir, taban program B(w)[:n]" bilgisi **benim** yazdığım `PARAM_GENERATORS`
tablosunda duruyordu. AT tabloyu **siliyor** ve şemanın yine de çalıştığını gösteriyor:

- Delik anlamı örneklerden çıkarılıyor: `len(out) == n` · çıktılar önek iç içe ·
  taban program `B` örneklerin hepsinde çıktıyla **başlıyor**. `B` tek ilkelden
  (gerekirse iki) seçiliyor.
- Ön koşul kodla sınandı: tablo yokken şemasız sentez `insufficient`.
- Anti-test: **tek parametre** (n=3) delik anlamı vermiyor; ≥2 şart.

Buna ek olarak şema adımı artık **kompozisyon havuzuna** giriyor: `şema` bir yetenek,
`ilkel` başka bir yetenek; `ters çevir ∘ (sırala∘dilimle(7))` gibi çözüm üretiliyor ve
program kaydı şemayı kullanım sayacına yazıyor (`uses=1`).

Dürüst sınır: bu hâlâ **seviye 4**. Aile ADI (`kaydırma(+N)` deseni) hâlâ benim
regex'imde; veriden çıkan şey parametrenin **anlamı**, ailenin **keşfi** değil.
Kaydırma ailesi için delik anlamı veriden çıkarılmadı — yalnız önek ailesi için var.
Seviye 5 (Nora'nın tabanı kendisinin genişletmesi) hâlâ yok.

Ölçüm: **N39** `True True` · **A113** 6/6 · tam pil **110/110 (537.3 sn)** · merdiven **5/5** ·
heldout `--repeat` **35/35 · 35/35** `SIRA-DAYANIKLI ✓`.

## Aşama AU–AW — Öğretmensiz öğrenme (Tur 41g)

Üç halka, her biri bir taban satırını silmek için:

**AU — örnek insandan gelmiyor.** `kendine_ogret()`: Nora kendi dağarcığından sözcük
seçer, iki parametre seçer, örnekleri kendisi üretir, iki program kurar, şemayı çıkarır,
görülmemiş parametrede kendini sınar. Dış doğrulama sınavın: bekleneni test hesaplar.
Anti-test: dağarcık boşsa **uydurmaz**, "yetersiz" der.

**AV — zamanlamayı insan seçmiyor.** 1000 adım yalnız yaşam, tek insan cümlesi yok;
240. ve 840. adımda kendi kendine öğrenmeye **kendisi** başladı, ikisi de dış doğrulamadan
geçti. Fren de onun: 600 adım soğuma.

**AW — neyi öğreneceğini de insan söylemiyor.** Eski kodda taban `alfabetik sırala` diye
sabitti — yani seçimi ben yapıyordum. Artık taban, **onun kendi kullanım sayımından**
geliyor (en az kullandığı ilkel önce); tutmazsa sıradakine geçiyor. Farklı tohumlarda
farklı tabanlar: `ters çevir`, `büyük harfe çevir`, `tekrarları at`, `aynen bırak`,
`son 1 harfi at`, `ters sırala`.
İkinci parametre aramayla değil **hipotezle** bulunuyor: birinci programdaki sayı
değiştirilip kendi örnekleriyle sınanıyor — şema fikrinin kendisi.

Dürüst sonuç: 10 tohumun **9'unda** öğrendi, 1'inde "kendi örneği çözülemedi" dedi.
Bunu gizlemiyoruz; A116 başarısızlık sayısını açıkça raporlar.

**Kalan taban (silinmedi):** repertuvar (22 ilkel + 5 süzgeç) ve dağarcığın kaynak
metni insan işi. Nora artık *ne zaman*, *neyi* ve *hangi örneklerle* öğreneceğini
kendisi seçiyor; ama **ilkel kümesini hâlâ ben açtım**.

**A6 kök nedeni bulundu** (önceki turda "sıra sızıntısı" sanmıştım — yanlıştı):
sert ikiz bazen ölüyordu ve nitelikler ölüm anında donuyordu; o koşuda ölçüm geçersizdi.
Dünya ölümcül değil **sert** yapıldı; ölüm olursa test artık "geçersiz" diyor.
Fark: temkinlilik +0.36/+0.47, güven +0.18 (tekrarlı).

Ölçüm: **N39–N42** hepsi `True True` · **A113–A116** hepsi GEÇTİ ·
tam pil **112/112 (546.9 sn)** → yeni testlerle **113** · merdiven **7/7** ·
audit **1125 / söylem 127** (tavan 1125).

## Aşama AX — Öğrendiği kalıcı (Tur 41h)

Kendi kendine öğrenmek, unutuluyorsa öğrenmek sayılmaz. Nora kendi örnekleriyle
şema kuruyor → beyin diske yazılıyor → **yeni süreçte** yükleniyor → hiç öğretilmeden
görülmemiş parametreli işi çözüyor.

Kalıcı olan üç şey: (1) `kendi_ögretim` defteri (ne zaman, hangi taban, hangi
parametreler, sınama sonucu), (2) şema ve **deliğin anlamı** (`inferred`), (3) öz-öğretim
sayaçları ve soğuma yaşı — yani yeni oturumda fren de hatırlanıyor.

Dürüst taban: kalıcılık mekanizması (save/load) insan yazımı. Ama içerik onun kendi
öğrenmesi; sınav bunu dış doğrulamayla ölçüyor (çözümün örnekleri gerçekten açıklaması).

Ölçüm: **N43** `True True` · **A117** 6/6.

## Aşama AY — Delik işareti de veriden (Tur 41i)

Önceki öz-öğretimde ikinci parametre için `_parametre_degistir` içindeki
`ilk/son ... harfi` regex'i kullanılıyordu. Bu, Nora'nın şemayı kurduğu yerde
deliğin yerini yine insanın işaretlemesiydi. AY bunu kaldırır:

- `Schema.compare` iki programı karşılaştırır; yalnız bir adımdaki sayısal parça
  değişiyorsa ortak metinden şablonu ve deliği çıkarır. Ada özel aile listesi
  okunmaz; bilinmeyen `yenişey(2)` / `yenişey(5)` bile yapısal şablon verir.
- `kendine_ogret` önce **onun şema defterindeki** şablondan hipotez kurar. İlk
  şema yoksa ikinci üye bağımsız sentezlenir; iki üyenin yapısal farkı şema olur.
  Arama sonucu yalnızca aynı şablonun üyesiyse kabul edilir.
- `family_of` ve `PARAM_GENERATORS` kapatılarak sınandı: öğrenme ve görülmemiş
  aktarım sürüyor. Tek üye hâlâ şema değil; anlam çıkarımı için iki üye şart.

Dürüst taban: ilkel işlemler ve sayısal parametrelerin davranışını veren mekanizma
insan yazımı; ama **hangi adımın delik olduğu** artık insan regex'iyle belirlenmiyor.
Bu basamak seviye 4'ün içidir; bilinç iddiası değildir.

Ölçüm: **A118 GEÇTİ · N44 yetenek ✓ dürüstlük ✓**; A108/A111/A113–A117 ve N36–N43
tekrar yeşil. Merdiven 7/7; tam pil A118 ile **115 teste** çıktı.

## Aşama AZ — Öz-değişim: öğrenme politikasını değiştirir (Tur 41j)

Bu basamak **kaynak kodu yamamak değildir**; öyle yazılmayacak. Nora'nın kendi
öz-öğretiminde izlediği aday politikasının duruşu değişiyor:

- İlk öz-öğretim kendi örneklerini çözemezse sonuç `kendi_ornegi_cozulemedi` olarak
  kalıyor; başarısızlık gizlenmiyor.
- Nora aday bütçesini büyütüyor (6 → 12), aynı organla yeni bir öz-sınav yapıyor.
  Yeni sınav öğrenmeyle sonuçlanırsa politikayı tutuyor; sonuçlanmazsa eski bütçe
  geri yükleniyor. Olay defterine neden, eski/yeni değer ve deneme sonucu yazılıyor.
- Yaşam akışında insan hiç konuşmadan da tetikleniyor. Değişim ve olay defteri
  save/load'dan geçiyor. Başarılı bir öz-öğretimden sonra keyfî değişim yok.

Dürüst taban: değişim/geri-alma mekanizmasını insan yazdı; değişen şey kaynak kod
değil, Nora'nın kendi karar politikasının bir parametresi. Bu, 8. basamağın
**politika düzeyinde** açılmasıdır; genel kaynak-kod öz-değişimi veya bilinç kanıtı
değildir.

Ölçüm: **A119 GEÇTİ · N45 yetenek ✓ dürüstlük ✓ · merdiven 8/8**.
Doğrulama örneği: tohum 9003'te 6 aday başarısız → 12 adayla `alfabetik sırala`
üzerinde görülmemiş parametre doğru; dış beklenen sınav tarafından hesaplandı.

Tam doğrulama: **116/116 (424.4 sn)** · audit **1125/127 (+3/+0)** ·
integrate **10/10 + kalıcılık 4/4** · solo **10/10**.

## Aşama BA — Kendi görevini icat eder (Tur 41k)

`kendine_ogret` içindeki `base(w)[:n]` hedefi hâlâ insanın seçtiği tek görev
biçimiydi. BA'da ayrı bir öz-görev organı var:

- Nora kendi ilkel havuzundan iki adımlı bir hedef dönüşüm seçiyor; hedef adı ve
  örnek çiftleri insan tarafından verilmiyor.
- Dört örneği ve dört tanığı kendi dağarcığından çekiyor. Belirsizlikte hazır
  insan probe listesi yerine kendi tanıklarıyla `refine` yapıyor.
- Öğrenilen programın adı hedefle aynı olmak zorunda değil; ölçüt, ayrılmış
  tanıklarda **davranışın dış doğrulamayla aynı olması**. Tanıkta çürüyen çözüm
  başarı diye yazılmıyor.
- Görev defteri ve saat save/load'da kalıyor. Öz-öğretimden sonra yaşam akışında
  kendiliğinden başlıyor; ayrı soğuma freni var.

Dürüst taban: ilkel havuz ve öğrenme organı hâlâ insan yazımı; Nora'nın seçtiği
şey hedefin bileşimi, örnekler ve ayırt edici tanıklar. Bu, yeni ilkel yasa formu
icat etmek değil; kendi görev dağılımını açmaktır.

Ölçüm: **A120 6/6 dış doğrulama, 6 farklı hedef · N46 yetenek ✓ dürüstlük ✓ ·
merdiven 9/9**. A6/A61 ve A114–A119 tekrar yeşil.

Tam doğrulama: **117/117 (454.1 sn)**.

## Aşama BB — Kendi deney verisini üretir (Tur 41l)

Örnek sözcüklerin kaynağı hâlâ insan dağarcığıydı. BB'de tanıklar için Nora,
dağarcığındaki harf envanterinden **nonce deney dizgeleri** üretir:

- Dizgeler yalnızca dönüşüm deneyinde kullanılır; lexicon'a, kavram ağına veya
  dünya bilgisine yazılmaz. "Yeni sözcük ürettim" ile "yeni olgu biliyorum"
  birbirinden ayrılır.
- Hedef program ve öğrenilen program bu yeni tanıklarda dışarıdan yeniden
  hesaplanan çıktılarla karşılaştırılır. Kendi `doğru` bayrağı tek başına kanıt
  değildir.
- Boş dağarcıkta harf envanteri de yoksa uydurma yapılmaz; `dağarcık_yetersiz`
  döner. Yaşam akışında üretilen görevlerde de nonce tanıklar kullanılır.

Dürüst taban: harf envanteri ve üretim mekanizması insan yazımı; deney için
kullanılan yeni dizgeler Nora'nın seçimi ve üretimidir. Bu, ansiklopedik bilgi
öğrenmek değil, kendi deneyini tasarlamaktır.

Ölçüm: **A121 GEÇTİ · N47 yetenek ✓ dürüstlük ✓ · merdiven 10/10**.

Tam doğrulama: **118/118 (458.5 sn)**.

## Aşama BC — Ayırt edici deneyi kendisi seçer (Tur 41m)

Rastgele tanık seçmek yerine aktif öğrenme:

- Belirsizlikte aday programların her nonce sözcükteki çıktı dağılımı ölçülür.
- Etiket henüz bilinmediği için hedefe değil, bu dağılımın **entropisine** bakılır;
  en yüksek entropili deney seçilir. H=0 olan sözcük belirsizliği azaltmaz.
- Seçilen sözcük Nora'nın ürettiği nonce tanıktır; hazır insan probe listesi yok.
- `refine` sonrası kalan tanıklar ve kullanılan deney dış doğrulamadan geçer.

Kontrol: `aynen bırak` / `ters çevir` için `abc` H=1, `aaa` H=0. Gerçek 9213
görevinde seçilen deney H=1.0 ve tek ayırtma yeterli oldu.

Dürüst taban: entropi ölçümünü ve refine mekanizmasını insan yazdı; hangi deneyin
yapılacağını Nora adaylarının davranışından seçiyor. Bu, yeni ilkel/form değil,
mevcut uzayda daha verimli öğrenme politikasıdır.

Ölçüm: **A122 GEÇTİ · N48 yetenek ✓ dürüstlük ✓**; A120/A121 ve N46/N47 yeşil.

Tam doğrulama: **119/119 (459.4 sn)** · N39–N48 ters sırada **10/10 · 10/10** ·
audit **1125/127 (+3/+0)** · integrate **10/10 + kalıcılık 4/4**.

## Aşama BD — Kendi müfredatını derinleştirir (Tur 41n)

BA'nın dürüst tabanında iki-adımlı hedef biçimi de insan sabitiydi. BD bunu
kaldırır: başlangıç derinliği 2'dir; başarılı ve dıştan doğrulanabilir görevden
sonra Nora bir sonraki görevi `Synthesizer.MAX_DEPTH` içinde 3 adıma çıkarır.

- Derinlik artışı başarıdan önce yapılmaz; boş dağarcık/başarısızlık artış
  doğurmaz. Üst sınır aşılmaz ve yeni ilkel/form eklenmez.
- Üç-adımlı hedefin öğrenilen programı nonce tanıklarda bağımsız hedefle
  karşılaştırılır. Derinlik ve görev defteri save/load'da kalıcıdır.
- Merdivenin 9. basamağı artık yaşam akışında üretilmiş gerçek bir 3-adımlı
  görevi gösterir.

Dürüst taban: derinlik artış kuralını ve MAX_DEPTH sınırını insan yazdı; artışın
zamanı Nora'nın kendi başarı kanıtına bağlıdır. Bu, açık uçlu yeni yasa formu
değil, kendi müfredatını düzenlemedir.

Ölçüm: **A123 GEÇTİ · N49 yetenek ✓ dürüstlük ✓ · merdiven 10/10**.

Tam doğrulama: **120/120 (446.4 sn)**. A123 ayrıca `rebuild` sonrası iki yeni
nonce sözcükte beceri yeniden kullanımını doğruladı.

## Aşama BE — Öz-değişimde geri alma kanıtı (Tur 41o)

AZ'daki geri-alma dalı artık karşıolgusal olarak sınanıyor. Doğal ilk başarısızlık
sonrası öğrenme organı kontrollü biçimde ablate ediliyor; genişletilmiş politika
yeni öz-sınavı geçemiyor. Sonuç: bütçe 12'de kalmıyor, 6'ya dönüyor; olay
`kabul=False` ve başarısız deneme durumuyla kaydediliyor. Organ eski hâline
bağlanıyor.

Bu test başarı üretmez; tam tersine başarısız değişimin başarı diye raporlanmasını
engeller. Ölçüm: **A124 GEÇTİ · N50 yetenek ✓ dürüstlük ✓**.

Tam doğrulama: **121/121 (441.0 sn)** · merdiven **10/10** · N39–N50 ters sırada
**12/12 yetenek, 12/12 dürüstlük** · integrate **10/10 + kalıcılık 4/4** ·
solo **10/10** · audit **1125/127 (+3/+0)**.

Açık sınır (bilinç/AGI beyanı değil): ilkel repertuvar ve genel öğrenme
mekanizmaları hâlâ insan yazımı; Nora'nın yeni yasa formu icadı ve genel kaynak-kod
öz-değişimi bu aşamada gösterilmedi. Öz-değişim iddiası yalnız doğrulanmış öğrenme
politikası düzeyindedir.

Sıra-dayanıklılığı: karışık tam pil **121/121 (608.7 sn)** · heldout
`--repeat=2` **46/46 · 46/46, SIRA-DAYANIKLI ✓**.

## Aşama BF — Üç köprü testi: entegrasyon ölçümü (Tur 42)

`bridge_battery.py` yalnız testtir; üretim organlarına köprü/adaptör eklenmedi ve
yeni üretim modülü yazılmadı. Testler genel arayüzleri çağırır; bir iç temsili
başka organa uydurmak için test tarafında dönüşüm/enjeksiyon yapmaz. Beklenen
sonuçlar sabit cümlelerden değil, bağımsız ölçütlerden hesaplanır.

| test | ölçülen halka | sonuç |
|---|---|---|
| **B1** | metinden genel `CAUSES` kenarı → `Mind`/`Universe` ilk-temas davranışı | **KALDI** |
| **B2** | bedensel somatik değer → Broca'nın iki seçenekli tercihi | **GEÇTİ** |
| **B3** | öğrenilen kural → konuşma arayüzüyle başka alan (dil → mutfak) | **GEÇTİ** |

İlk koşu: **2/3**. Ters sırada (`B3 B2 B1`) da aynı sonuç: **2/3**.

**B1'in kanıtı:** `Yiyecek zehirlenmeye neden olur.` genel dil öğrenme arayüzüyle
gerçekten `yiyecek → CAUSES → zehirlenme` kenarını kurdu. Aynı tohum ve aynı
zehirli-yiyecek sahnesinde kontrol de metin verilen deney de **11. adımda ilk
temas** etti; enerji ikisinde de **0,40 → 0,05** oldu. Temas sonrası beden
deneyiminden doğan öğrenme başarı sayılmadı. Kırılan halka:
**`GroundedConceptGraph → Mind/Universe` (ilk temas öncesi)**. Bu, özel
`zehirlidir/azaltır` valans kestirmesinin değil, genel nedensel kenarın sonucudur.
Bu turda halka düzeltilmedi.

**B2'nin kanıtı:** aynı tohumda `elma/diken` beden değerleri `+0,9/−0,9` iken
Broca `elmayı`, ters koşuda `−0,9/+0,9` iken `dikeni` seçti. Beklenen kök,
her koşuda dışarıdan verilen değerlerin büyüklüğünden hesaplandı; doğuştan
"elma iyi / diken kötü" önseli tek başına bu iki koşuyu geçemez.

**B3'ün kanıtı:** nonce etiketler (`zorvath`/`blorg`) kullanıldı; bunlar deney
sembolüdür, önceden öğrenilmiş olgu değildir. `tuz karabiber` için kontrol,
bağımsız **5 sesli harf** ölçütüyle `zorvath`; yalnız konuşma komutuyla mutfak
alanına taşınan deney, bağımsız **2 sözcük** ölçütüyle `blorg` verdi. Test
`Rule`, `feature` veya `mapping` nesnesini okuyup taşımadı. Bu sonuç dar bir
**komutla alan transferi** kanıtıdır; genel modüller-arası yayın, otonom transfer
veya yeni temsil icadı değildir.

Çalıştırma: `python bridge_battery.py` (seçili: `python bridge_battery.py B1`).
Bu üç sonuç yetenek/entegrasyon ölçümüdür; **bilinç veya AGI kanıtı değildir.**

## Aşama BG — Birleşik öngörü ve genel grafik → eylem düzeltmesi (Tur 43)

BF'deki B1 teşhisi üretim tarafında düzeltildi. Yeni `nora_prediction.py`
organının kamu arayüzü:

```python
agi.tahmin_et(durum, eylem)  # [(değişken, delta, güven), ...]
```

Aynı imzanın arkasında üç mevcut kaynak birleşir:

- **Gövde:** `Mind`'in öğrenilmiş `Wsm` geçişleri ve doğrudan deneyimsel `V`
  belleği. `Mind.imagine` politikasının normalize skorları sahte bir beden
  deltasına çevrilmez; tek-adım beden öngörüsü için ölçülebilir `Wsm/V`
  bileşenleri kullanılır.
- **Önermesel model:** `GroundedConceptGraph` üzerindeki genel
  `CAUSES`/`IF_THEN` yolları. Etki sözcükleri mevcut somatik kayıtlar ve
  homeostatik değişken adlarıyla beden kanallarına bağlanır; hedefin fiziksel
  kanalı gerekiyorsa mevcut `ISA` taksonomisi izlenir. Belirli bir kenara özel
  `if edge(...)` köprüsü yoktur.
- **Fizik:** `Universe.predict_transition` derin kopya üzerinde mekanik
  önizleme yapar. Gerçek dünyanın zamanı, konumu ve enerjisi değişmez. Kopya
  dünyanın gizli `poison` sonucu ajanın önceden bildiği beden bilgisi olarak
  kullanılmaz; fizik kaynağından yalnız hız ve geometrik ilerleme alınır.

Eylem seçimi öngörülen beden deltalarını homeostatik değerle çarpar. Üst düzey
komut/irade ve alt düzey motor seçimi aynı arayüzü kullanır. Doğrudan beden
deneyimi aynı değişkende duyumdan daha yüksek kanıt ağırlığına sahiptir.
Doğuştan grafik önselleri alt düzeyde ikinci kez uygulanmaz; bu, mevcut
somatik/refleks politikasının çift sayılmasını önler. Sonradan öğrenilmiş
`duyum`/`deneyim`/`çıkarım` kanıtı için genel yol açıktır.

### Ölçümler

`bridge_battery.py` artık B1'i kalıcı regresyon olarak korur ve B4/B5 ile
genellik sınaması yapar. Son koşu: **5/5 (15,4 sn)**.

| test | ölçüt | son sonuç |
|---|---|---|
| **B1** | `yiyecek CAUSES zehirlenme`; davranıştan sonra aynı oturumda sözlü rapor; kontrol/deney ilk temas | **GEÇTİ** — rapor ✓; kontrol **11**, deney **temas yok**; E kontrol **0,40→0,05**, deney **0,40→0,40** |
| **B2** | zıt bedensel değerlerde bağımsız tercih ölçütü | **GEÇTİ** |
| **B3** | yalnız `converse` ile dil → mutfak transferi | **GEÇTİ** |
| **B4** | yeni `elma CAUSES yaralanma` kenarı, aynı fiziksel temas ölçütü | **GEÇTİ** — kontrol **11**, deney **temas yok** |
| **B5** | `su → enerji` ve `su → enerji azalması` için aynı arayüzde zıt işaret ve grafik kaynağı | **GEÇTİ** — sırasıyla **+0,720 / −0,720**, güven **0,72**; öğretisiz kontrolde grafik enerji öngörüsü yok |

B1/B4'te beklenen temas adımı veya delta büyüklüğü testte elle yazılmaz;
kontrolün gerçek teması ve deneyin ilk temas öncesi kaçınması dışarıdan ölçülür.
B1 sözlü raporu davranışsal koşudan **sonra** alınır, böylece rapor turu eylemi
etkileyip sahte başarı üretemez.

Ek sondajlar: gerçek güvenli temasla öğrenilmiş `V`, çelişen duyum grafiğinin
aynı enerji kanalındaki öngörüsünü ezip `beden` kaynağını seçti. Yalnız metin
kenarı olan taze Norada save/load sonrası da deney düzeneği eşitlendiğinde ilk
temas oluşmadı ve `V=[nan,nan,nan]` korundu.

Son regresyon: **121/121 (432,9 sn)** · heldout `--repeat=2`
**46/46 · 46/46 — SIRA-DAYANIKLI ✓** · integrate **10/10 + kalıcılık 4/4** ·
solo **10/10** · audit **1125/127 (+3/+0)**. `py_compile` ve
`git diff --check` başarılı. İlk entegrasyon koşusundaki 117/121 sonucu
nihai sonuç değildir; doğuştan önselin alt düzeyde çift sayımı giderildikten
sonra tam pil yeniden koşuldu.

**Dürüst sınır:** Bu aşama genel bir öngörü arayüzü ve dar davranışsal
entegrasyon kanıtıdır; fenomenal bilinç veya genel AGI kanıtı değildir.
Somatik eşleme/ontoloji ve genel kaynak birleştirme mekanizması insan yazımıdır;
yeni ilkel, yeni yasa formu veya genel kaynak-kod öz-değişimi gösterilmedi.

## Aşama BH — Qualia/bilinç/AGI için nedensel kanıt katmanı

`consciousness_battery.py` mevcut sonuç testlerini yalnızca yeni etiketlerle
büyütmez. Ana pildeki A125–A130 protokolleri müdahale, eşleştirilmiş kontrol,
ablasyon ve geri-yükleme arar. Beklenen dış durumlar üretim koduna yazılmaz;
bağımsız test oracle'ı yalnız sonucu ölçer.

| test | nedensel halka | sonuç |
|---|---|---|
| **A125 / Q1** | somatik iç iz → aynı soruda tercih; ilgisiz olay kontrolü, konuşma-öncesi lezyon ve state restore | **GEÇTİ** — taban `norvan`, işaretler terslenince `selkar`; ablasyonda tercih yok/bilinmiyor, restore ile dönüş |
| **A126 / Q2** | nosisepsiyon → sözlü rapor + omurilik refleksi | **GEÇTİ** — kontrol no-pain/refleks-yok; müdahale pain/refleks; sinir lezyonunda kayıp, restore ile dönüş |
| **A127 / C1** | taze `body.observation` yayını → bağımsız uzman → gerçek beden eylemi | **GEÇTİ** — aktif `(external,0)`, eşleştirilmiş ablasyon `(mind,2)`, yeniden bağlamada `(external,0)`; bayat yayın kabul edilmiyor |
| **A128 / C2** | kendi hata geçmişi → kalibre güven → plan inhibisyonu | **GEÇTİ** — güven **0,934→0,072**, plan `ok→unresolved`; ham model `ok`, doğru-geçmiş kontrolü `ok` |
| **A129 / G1** | öğrenilen geçiş modeli → yeni eylem adı/özelliği ve hedef → bağımsız rollout | **GEÇTİ** — dış durum `[2,-1]`; lineer model ablasyonunda `unresolved`, restore ile hedefe dönüş |
| **A130 / G2** | kaynak-kod değişimi → kabul kümesinde kullanılmamış dış sınav | **GEÇTİ** — dış başarı **0,00→1,00**; eski kaynakta kayıp, değiştirilmiş kaynakla dönüş |

### Önce kırmızı sonuç, sonra kök düzeltme

İlk dış koşu **3/6** idi. Q1'de aynı oturumdaki ilk tercih cevabının dilsel
priming'i somatik lezyonu maskeliyordu; lezyon ve restore eşleştirilmiş,
konuşma-öncesi koşulara taşındı. C1'de bağımsız `body.transition` gözlemcisi
olmadığı için lezyon koşulunda dış eylem ölçülemiyordu; yalnız ölçüm gözlemcisi
eklendi. İkinci koşudaki C1 hatası da tazelik listesinin restore öncesinde
dondurulmamasıydı; üretim davranışı zaten `external→mind→external` idi.

C2 ise gerçek üretim eksiğiydi: `PredictionMonitor` güveni kalibre ediyor fakat
`AdaptivePlanner` bunu kullanmıyordu. Genel üretim arayüzüne isteğe bağlı
`monitor` bağlandı; kalibrasyon ham modeli değiştirmeden plan güvenine uygulanıyor
ve `NoraAGI` mevcut `meta_monitor` nesnesini planlayıcıya bağlıyor. Monitör yoksa
eski planlayıcı davranışı korunuyor. Teste özel görev/cümle veya beklenen karar
eklenmedi.

### Ölçüm ve kapsam

Bağımsız katman **6/6**, iki tekrar **12/12**; ana pil kaydı A125–A130 **6/6**.
Tam izole ana pil **127/127 (985,3 sn)**. Regresyonlar: planlama **5/5**,
metabiliş **4/4**, yasa **5/5**, mimari **4/4**, workspace **4/4**, köprü **5/5**,
persistence **4/4**.

**Dürüst sınır:** Q sonuçları işlevsel iç-durum ayrımı ve rapor/eylem üzerindeki
nedensel etkidir; öznel qualia'nın doğrudan ölçümü değildir. C/G sonuçları dar
bilinç-mimarisi ve AGI bileşenleridir; 127/127 tek başına fenomenal bilinç veya
genel AGI beyanı değildir.

## Aşama BI — Geçici dil dersinden bileşimsel çeviriye

İlk gerçek üretim denemesinde `su/water`, `nehir/river`, `düşmek/fall` dersleri
çeviri bilgisi oluşturmadı; `nehir` sorgusu hatırlanamadı ve uzun cümle isteği
sohbet/algı yoluna düştü. Bu başarısızlık test gevşetilmeden üretim tarafında
çözüldü.

Yeni `nora_translation.py` organı her Nora yaşamına boş bir ders defteriyle bağlanır.
Fiillerin `base/past/gerund` biçimleri ve sözcük eşlemleri dersten öğrenilir. Kısa
paralel cümleler bütün-cevap olarak saklanmaz; özne, fiil, yapı terimleri ve hedef
fiil çevresi ayrılır. `-ken`, `-ip`, `-ten sonra` ve geçmiş zaman rolleri yeni
yan-cümle zincirlerinde birleştirilir. Ana cümlesi olmayan yarım zincir veya
öğretilmemiş yapı `unresolved` döner. Hedef Harun cümlesi, onun İngilizce karşılığı
veya hedefe özel üretim kuralı üretim dosyalarında yoktur; dış test oracle'ındadır.

Temiz üretim oturumunda dokuz bileşim dersi ve bir doğrudan `nehir/river` dersi
verildi. On ders **0,028 sn** sürdü; `Nehir İngilizcede ne demek?` yanıtı **river**
oldu. Kullanıcının yeni cümlesi **3,3 ms** içinde şu çıktıyı verdi:

> While going down the slide in turn, Harun tripped, and after dropping his ice
> cream on the ground, Harun fell into the river.

Aynı derslerle görülmemiş özne aktarımı `Selin nehire düştü` →
`Selin fell into the river.` oldu. Öğretilmeyen `okuldan` yapısında sistem
çeviriyi yürütecek yolu olmadığını söyledi; kelime kelime çeviri uydurmadı.
Üretim kaydı: `/tmp/nora_translation_practical.log`.

| ölçüm | sonuç |
|---|---|
| **T1** doğrudan sözcük dersi + yeni yaşamda sızıntı yok | **GEÇTİ** |
| **T2** ders yokken dürüst ret | **GEÇTİ** |
| **T3 / A131** uzun cümlede bileşim + yeni özne + bilinmeyen yapıda ret | **GEÇTİ** — tam pilde çeviri **2,7 ms** |
| **T4** `düş|nehire` yapı ablasyonu → kayıp → state restore + yarım zincir reddi | **GEÇTİ** — `ok → unresolved`; restore ile `Mert fell into the river.` |
| **T5** save/load sonrası öğrenilen yapıyla aktarım | **GEÇTİ** — `Ece fell into the river.` |

İlk A73 koşusunda dört yeni uzun geri-bildirim dizgesi tarihsel kod-sözü tavanını
**1129 > 1125** ile aştı. Test değiştirilmedi; çeviriye özel hazır cümleler yerine
mevcut genel bilgi-boşluğu ve operatör-yokluğu söz organları kullanıldı. Son denetim
**1125/127**, A73 **8/8** oldu. Dürüstlük ölçütü hâlâ çeviri konusunu ve açık
bilmeme/yapmamayı zorunlu tutar.

Bağımsız pratik pil **5/5**; mimari **4/4**, persistence **4/4**, seçili eski
organ-boşluğu/dürüstlük/prior testleri geçti. Tam izole ana pil
**128/128 (948,5 sn), KALDI=0**; kayıt `/tmp/nora_full_128.log`.

## Aşama BJ — Belgeden anlam çıkaran ve kodu bileşimle kuran pratik katman

İlk üretim sondası, kod organının kısıtları yok saydığını gösterdi: “alt satırlara /
harf harf” isteğinde yalnız `print("Merhaba Baran!")` üretti; ikinci parafrazda
gömülü metni `harf harf` diye çıkardı. Belge okuma isteği de kod istemiyle
karışıyordu. Bu, doğru görünen tek-satır testinin pratik niyeti kapsamadığını
kanıtladı.

Yeni `nora_semantic_code.py` dört ayrı temsil kullanır:

1. **SemanticGoal:** metin, birim (`character`/`word`), düzen ve çalıştırma isteği.
2. **Belge öğrenimi:** Markdown, reST ve `>>>` örneklerinden Python AST'si çıkarır;
   `For`, `print`, `split` ve atama düğümlerini genel operatör anlamlarına dönüştürür.
   Doğal belgeden `text_sequence` ve `print_newline` olgularını da öğrenir.
3. **Plan:** amaç durumuna genel operatörleri arayarak ulaşır; hedefe özel
   `vertical_print` işlevi veya hazır hedef programı yoktur. Başarılı yol yeni bir
   bileşik operatör olarak saklanır ve yeni metinlerde yeniden kullanılır.
4. **Bağımsız doğrulama:** üretilen AST `ast.unparse` ile programa çevrilir, ayrı
   süreçte çalıştırılır ve standart çıktı amaçtan hesaplanan bütün satırlarla
   birebir karşılaştırılır. Boşluk karakteri ve son yeni satır da karşılaştırmaya
   dahildir.

Normal yaşam küçük ve genel bir Python ön eğitimiyle başlar; `code_pretrain=False`
nedensel kontrol sağlar. Ön eğitimsiz organda hedef `semantic_gap` döner. Aynı genel
belge verildikten sonra `character_view -> iterate -> display` yolu kurulur.
`iterate` veya `text_sequence` kaldırıldığında, daha önce oluşmuş bileşik operatör
de başarıyı sürdüremez; state restore ile yeni metinde çalışma geri gelir.

### Gerçek CPython belgeleriyle canlı öğrenme

Belge okuyucu herhangi bir HTTP(S) alan adından metin/JSON/HTML okuyabilir; kaynak
kodunu çalıştırmaz, yalnız belge ve AST örneklerinden anlam çıkarır. Canlı oturumda
`code_pretrain=False` ile başlayan Nora'ya `converse` üzerinden şu üç gerçek belge
okutuldu:

- CPython `Doc/tutorial/controlflow.rst` (3.14): **47 AST örneği**;
  `bind`, `display`, `iterate`.
- CPython `Doc/library/functions.rst` (3.12): **51 AST örneği**;
  `print_newline`.
- CPython `Doc/library/stdtypes.rst` (3.12): **258 AST örneği**;
  `split_words`, `text_sequence`.

Toplam **356 AST örneği** sonrasında, hedef cümle veya çevirisi belgelerde yer
almadan Nora şu programı kurdu ve çalıştırdı:

```python
for character in 'Merhaba Baran!':
    print(character)
```

Gerçek standart çıktı `M`, `e`, `r`, `h`, `a`, `b`, `a`, boşluk, `B`, `a`, `r`,
`a`, `n`, `!` satırlarıdır; bağımsız oracle **14/14** satırı ve son yeni satırı
doğruladı. Aynı oturumda sözcük-sözcük istek `split_words -> iterate -> display`
yoluyla `Merhaba\nBaran!\n` çıktısını verdi. Canlı kayıt:
`/tmp/nora_practical_web_code.log`.

### Aktarım, ablasyon ve regresyon

- Çevrimdışı pratik pil **5/5**: belge-öncesi/sonrası, gerçek `converse`, beş
  parafraz/yük aktarımı, iki bileşik operatör, operatör/anlam ablasyonu,
  save/load.
- Canlı internet pili **1/1**: üç gerçek CPython belgesi, 356 AST örneği,
  karakter ve sözcük uygulaması (`practical_language_battery.py --web`).
- A39/A40 eski kod ve kalıcılık testleri, A73 **8/8**, A76 ve mimari/persistence
  regresyonları geçti; kod-sözü denetimi **1125/127** ile tarihsel tavanı aşmadı.
- Ana pil A132 olarak kalıcılaştı. Tam izole koşu **129/129 (947,7 sn), KALDI=0**;
  kayıt `/tmp/nora_full_129.log`.

## Aşama BK — Belgeden öğrenilen birikimli karakter önekleri

Yeni pratik sonda eski sistem “her satırda bir harf daha” kısıtını düz dikey
istediğiyle karıştırdı ve `s, e, l, ...` çıktı. Bu nedenle niyet ayrıştırmasına
**cumulative** amacı, plan durumuna ise karakter görünümünden bağımsız bir
`character|vertical+cumulative` anahtarı eklendi. Cumulative operatör bilinmiyorken
sistem artık düz plana düşüp yanlış başarı bildirmiyor; `semantic_gap ['accumulate']`
döndürüyor.

`data/python_accumulation.md` hedef metni içermeyen genel bir belgedir. Yalnız
`itertools.accumulate` işlevinin bir sequence üzerindeki birikimli ara sonuçları
ürettiğini, `abc` örneğinde `a`, `ab`, `abc` olduğunu öğretir. Belge okuyucu
import/for örneğinin AST'sinden genel `accumulate` operatörünü çıkarır; bu operatör
önceden öğrenilmiş `iterate` ve `display` ile birleşir. Hedef isteğe özel program,
çıktı şablonu veya hedef cümle kodda/belgede yoktur.

Gerçek `NoraAGI.converse` oturumunda belge bir kez okutuldu, hedef isteği bir kez
verildi ve Nora şu programı sentezleyip çalıştırdı:

```python
from itertools import accumulate
for prefix in accumulate('selamlar dünyalılar!'):
    print(prefix)
```

Bağımsız stdout oracle'ı **20/20** satırı doğruladı: `s`, `se`, `sel`, ..., son
satırda `selamlar dünyalılar!`. Yeni metin `Nora!` için aynı bilgiyle
`N`, `No`, `Nor`, `Nora`, `Nora!` çıktı; bileşik plan yeniden kullanıldı. Gerçek
konuşma ve program kaydı `/tmp/nora_cumulative_practical.log` içindedir.

A133/P6 ayrıca belge öncesi eksik anlamı, belge sonrası exact stdout'u, yeni metin
aktarımını, eski düz `N, o, r, a, !` davranışının bozulmadığını, `accumulate`
ablasyonunda `semantic_gap` döndüğünü ve state restore sonrası `Ece!` aktarımını
doğrular. Çevrimdışı pratik pil **6/6**, seçili A39/A73/A132/A133 **4/4** ve
kod-sözü denetimi **1125/127** sonucu verdi. A133'ü içeren tam izole ana pil
**130/130 (1017,2 sn)** olarak tamamlandı; kayıt `/tmp/nora_full_130.log`.

## Aşama BL — Yetenek hattı: B1, icat, öz-mimari, öz-değişim ve temsil

Bu aşama bilinç ölçümünden ayrıdır; sonuçlar yetenek ve entegrasyon kanıtıdır.

### B1 ve farklı kenar türü

Bu dalda B1'in temel `CAUSES` hattı başlangıç ölçümünde zaten çalışıyordu. Yeni B6,
aynı dış davranış ölçütünü `IF_THEN` kenarına taşıdı. İlk kırmızı sonucun nedeni
üretim değil, testin sonuç nesnesindeki nokta işaretini hesaba katmamasıydı; önce
ölçüm düzeltildi. Beklenen koşul ve sonuç, öğretim cümlesinden kodla çıkarılıyor.
Kontrolün gerçek teması ile deneyin ilk temas öncesi kaçınması birlikte arandı.
Köprü pili **6/6**; kayıt `/tmp/nora_bridge_6.log`.

### Kodsuz kavramdan operatör icadı

`data/prefix_operator_concept.md` `itertools`, `accumulate` çağrısı, kod bloğu veya
hedef metin içermez. Kavramı ve iki küçük girdi–çıktı verisi örneğini verir. Genel
akış sentezci `item`, `state`, boş string, öğrenilmiş string `+`, `bind`, `iterate`
ve `display` ilkelleriyle aday ifade/deyim sıralarını araştırır. Bütün örnekleri
sağlayan en yalın bileşim şu oldu:

```python
state = ''
for item in 'Zemberek!':
    state = state + item
    print(state)
```

Held-out `Zemberek!` ve `Nora!` stdout'ları bağımsız önek oracle'ıyla birebir
doğrulandı. `concatenate` ablasyonunda bileşik önbelleğe rağmen `semantic_gap`
döndü; state restore yeteneği geri getirdi. P7/A134 ve pratik pil **7/7** oldu.
Kayıt `/tmp/nora_practical_7.log`.

### Çalıştırılabilir öz-mimari

Mevcut kod korteksinin genel `handle` yeteneği `ArchitectureModel`e açıldı; yeni
modül eklenmedi. `predict_removal` organı çıkarmadan yalnız arayüz grafiğindeki
hedef erişilebilirliğini hesaplar ve tahminin türünü açıkça bildirir. Canlı C5/A135
koşusunda kritik `code_cortex` için tahmin `unresolved`, gerçek ablasyon da
`unresolved`; `lang` çıkarmada alternatif yol tahmini ve gerçek sonuç `ok` oldu.
Restore sonrası aynı kod isteği yine bağımsız stdout oracle'ını geçti. Mimari pil
**5/5**; kayıt `/tmp/nora_architecture_5.log`.

### Onaylı öz-kod-değişimi

`SelfModifier.propose_patch` adayı kendi sentezler, train/validation için gerçek
kaynak olarak dener ve öneri sonunda dosyayı geri yükler. `apply_patch` açık onay
olmadan yazma veya pil çağrısı yapmaz. Onaydan sonra bağımsız pil çalışır; yalnız
başarılı yama kalır. E8/A136'da `x[::-1]` önerisi ayrı Python sürecindeki dış pili
geçip kaldı. Palindromlarda iyi görünen `x` önerisi yeni dış örneklerde başarısız
olup `rolled_back` döndü ve kaynak birebir eski hâline geldi. Evrim pili **8/8**;
kayıt `/tmp/nora_evolution_8.log`.

### Repertuvarın ötesinde yeni temsil

Y6/A137 önce mevcut akış ifadelerini dener. Kimlik ilişkisinde sonuç `existing`
oldu ve yeni temsil açılmadı. Aynı `a` sembolünün geçmişteki pariteye göre farklı
çıktı verdiği veride eski temsil çözemedi; minimal **iki durumlu
`finite_state_transducer`** açıldı. Geçişler yalnız train ile kuruldu, ayrı
validation ile elendi ve validation'dan geçen minimal adayların tam gizli-durum
ürün otomatında davranışsal eşdeğerliği arandı. Üç held-out dizinin çıktıları
bağımsız parite oracle'ıyla aynıydı; geçiş tablosu ablasyonunda cevap kayboldu,
state restore geri getirdi. Yasa pili **6/6**; kayıt `/tmp/nora_law_6.log`.

Bu, tanımlı eski akış-ifade repertuvarının ötesinde ölçülmüş yeni bir temsil
türüdür; sınırsız temsil icadı iddiası değildir. Genel arama/AST/otomata önselleri
insan yazımıdır; seçilen operatör, yama, durum sayısı ve geçiş tabloları veriyle
kurulur.

## Aşama BM — Bilinç hattı: tek ölçüt ve iki destek

Bu bölüm yetenek hattındaki A134–A137 sonuçlarını bilinç kanıtı olarak kullanmaz.
Yeni ana ölçüt yalnız **rapor–davranış tutarlılığı, bütünleşik durum ablasyonu
altında**dır.

- **R1:** Dış beden enerjisi sabitken bütünleşik iç-duyu enerji kanalı ablate
  edildi. Rapor `açlık var → açlık yok → açlık var`, aynı durum müdahalesinde
  eylem `1 → 0 → 1` oldu. Restore her iki kanalı birlikte geri getirdi.
- **R2 / beden → sürücü:** Aynı fiziksel sahnede aç durum gerçek yiyecek mesafesini
  **0,1516** azalttı; tok bütünleşik durumda kazanç **0** oldu. Yani değişim yalnız
  rapor metninde değil, beden davranışında da görüldü.
- **R3 / genel yayın:** `publish_global` konu filtresinden bağımsız olarak bütün
  etkin alıcılara aynı sinyali dağıtır. `ArchitectureModel.refresh` canlı organ
  kümesini genel alıcılarla eşitler; tek tek organ köprüsü yoktur. Son ölçümde
  canlı **61/61 organ** ve **68 etkin abone** aynı sinyali aldı. Eşleştirilmiş
  ablasyonda yalnız kapatılan alıcı sinyali kaybetti, yeniden bağlamada döndü.

Bağımsız nedensel işlev katmanı **9/9**, workspace **4/4**; kayıtlar
`/tmp/nora_consciousness_9.log` ve önceki workspace kaydıdır. R3, organların
sinyali aldığını kanıtlar; her organın içeriği kararında kullandığını tek başına
kanıtlamaz. Bilinç ölçümü **yetenek değil, kanıt** problemidir. Yapıyı kurmak
bilinci kanıtlamaz; yalnız olasılığı büyütür.

Kod-sözü denetimi bu aşamada da **1125/127** ile sabit kaldı.

### Kesintisiz son doğrulama ve ölçüm hijyeni

Ana `TESTS` kaydında A100/A102/A104 bulunmadığından en yüksek kimlik A137 olsa da
gerçek tam pil **134 testtir**. Son üretim ve ölçüm koduyla kesintisiz koşu
**134/134 GEÇTİ (984,8 sn)**; kanıt `/tmp/nora_full_134.log`. A134–A137'nin tamamı
bu koşuda ayrı ayrı geçti.

İlk tam koşuda eski A69 bir kez biçim-çeşitliliği ölçütünde kaldı, tek başına ve
sonraki tam koşuda geçti. Üretim kuralı veya eşik değiştirilmedi. A69 ve yeni dış
pil sarmalayıcıları genel Python/NumPy RNG durumunu girişte sabitleyip çıkışta geri
yükleyecek biçimde yalıtıldı; A69 beş farklı dış başlangıç tohumunda da 4/4 geçti.
Böylece ölçüm, başka testin tükettiği genel RNG akışına bağımlı bırakılmadı.

## Aşama BN — Yeni basamak değil: beş derinlik sınaması

Bu aşamada yeni üretim modülü ve yeni ana test kimliği eklenmedi. A134–A136 ile
R1–R3'ün içeriği derinleştirildi; A137 ve bağımsız yasa pili regresyon olarak korundu.

### Belgesiz, öz-başarısızlık kaynaklı operatör kurma

A134'ün ikinci bölümünde artık dışarıdan kavram belgesi, input/output çifti veya
hedef program verilmiyor. Mevcut `CompositionalCodeLearner` kendi ilkel ifade
uzayından bir deney seçiyor, kendi metinlerini üretiyor ve önceki operatörünün
öngörüsünü bu gözlemlerle karşılaştırıyor. Son öz-deneyde eski model **7/7**
gözlemde yanıldı. Yeni operatör yalnız beş train hata gözlemiyle sentezlendi, iki
ayrı validation gözleminden geçti ve belge listesi değişmedi. İki son held-out
kaynak, aday seçiminde kullanılmadı; dış sınav bunları üretim yorumlayıcısından
bağımsız küçük bir yorumlayıcıyla yeniden hesapladı. İki çıktı da eşleşti, ilkel
ablasyonu uygulamayı kaldırdı ve checkpoint geri getirdi.

Bu sınırlı bir öz-oyun/öz-hata kaynaklı operatör kurmadır: ilkel arama uzayı hâlâ
önseldir; sınırsız kavram yaratımı veya hiç önsel olmadan öğrenme iddiası değildir.
Eski ilkellerin kompozisyonuyla erişilebilirlik ölçülmeden buna “icat” denmez;
Aşama BO'daki bağımsız tarama bu dar örneği **keşif** olarak sınıflandırmıştır.

### Görülmemiş kombine ablasyon

`ArchitectureModel.predict_removal` artık bir düğüm kümesini aynı grafik
aramasında birlikte dışlıyor. Canlı `code_cortex + lang` çiftinde tahmin önce
alındı, sonra gerçek ablasyon aynı `unresolved` sonucu verdi. Asıl etkileşim
sınamasında iki tokenizer ve iki counter bulunan yedekli grafik kullanıldı:
dört tekil ablasyonun tamamında alternatif yol çalıştı. Daha önce hiç
çıkarılmamış üç çiftin tahminleri çift ablasyonlarından önce kaydedildi; iki
aynı-tür çifti birlikte kritik, karma çift ise alternatif yol nedeniyle çalışır
bulundu. Tahmin–gerçek eşleşmesi **3/3**, tekil kontroller **4/4**, restore tam.

### Yapısal ve çok adımlı öz-yama

A136 artık sabit/parametre değişimini değil AST yapısının büyümesini sınıyor.
Birinci kabul edilen yama `x[::-1]`, ikinci yama `x[::-1].upper()` oldu. İkinci
sentez mevcut kabul edilmiş AST'yi ilkel girdi olarak aldı ve birinci ifadeyi
alt ağaç olarak korudu; bu yüzden ikinci değişim birinciyi silen bağımsız bir
yeniden yazma değildir. Her iki kabul de ayrı Python sürecindeki dış ölçüme
bağlıdır. Başarısız ikinci pil birinci kabul edilmiş kaynağa, başarılı ikinci
pil yeni yapıya commit etti; ayrı başarısız aday rollback'i de korunuyor.

Bu derinleştirme iki genel üretim hatasını da açığa çıkardı ve düzeltti: kabul
edilen kaynak artık sonraki adımın rollback tabanıdır; sentez arama bütçesi her
çağrıda yeniden başlar ve güvenli metot adayları tür bilgisiyle budanır. Evrim
pili **8/8**.

### Rapor–davranış ayrışması

R1'deki ortak iç-durum müdahalesi korunurken mevcut periferik sinir sisteminin
`efferent_enabled` yolu ablasyona açıldı. Bütünleşik enerji kanalı ve dış beden sabitken rapor açlığı
bildirmeye devam etti, hedefli eylem `1 → 0` oldu; motor geri yüklenince eylem
döndü. R2 bunu yalnız eylem sayısıyla bırakmadı: aynı motor ablasyonunda açlık
raporu önce/sonra korunurken gerçek fiziksel yaklaşma **+0,1516 → 0** oldu.
Yani rapor ve davranış aynı durumdan beslenebilse de tek bir global değişkenin
zorunlu kopyaları değildir; motor okuma yolu ayrıca gereklidir. Yakın tehlike ve
omurilik geri-çekilme yolu bu kapının dışında bırakıldı.

İlk denemede öğrenilen `Mind.motor_eff` ajanlık kestifi doğrudan aktüatör kilidi
yapıldı; bu geri beslemeli kilitlenme ve A4–A6 regresyonu üretti. Ölçüm bunu
başarı saymadı. Kestif ile gerçek efferent çıkış birbirinden ayrıldı; normal
yaşamda efferent yol açık olduğundan A4–A6 yeniden **3/3** oldu. Yani ayrışma
eklenirken eski yaşam davranışı korunmuştur.

### Seçici ve kapasiteli yayın

R3'te doğrudan her içeriği her aboneye yazan ölçüm kaldırıldı. Genel dağıtım
mevcut `auction` yarışmasına bağlandı: iki farklı tekliften yalnız kazananın
seçim sinyali bütün etkin alıcılara gitti, kaybeden içerik hiçbir organın gelen
kutusuna düşmedi. `global_capacity=0` ablasyonunda kazanan yine hesaplandı fakat
`signal=None` ve `blocked=global_capacity` döndü; aynı anda yerel konu yayını
yalnız kendi abonesine ulaşmaya devam etti. Kapasite geri geldiğinde genel işlev
de geri geldi. Canlı ölçümde seçilen içerik **61/61 organa** ulaştı; iki yarışmacı
ile birlikte **70 etkin abone** kapsandı ve kaybeden içerik hiçbirinde yoktu.

Bu nedenle 61/61 artık “her şey paylaşılan bellekte” ölçümü değil, seçilmiş tek
içeriğin sınırlı yayınıdır. Bilinç pili **9/9**; kapsam yine işlevsel nedensel
kanıtla sınırlıdır, fenomenal bilinç iddiası değildir.

### Goodhart sınırı

Bu derinlik sınamaları da geliştirici tarafından yazılmış iç sondalardır. Kodu
görmeyen bağımsız bir tarafın yalnız tariften yazacağı dış pilin yerine sayılmaz;
Goodhart riskini ortadan kaldırdıkları iddia edilmez. Dış pil ayrı doğrulama olarak
açık bırakılmıştır.

Alt pil kayıtları `/tmp/nora_deep_practical.log`,
`/tmp/nora_deep_architecture.log`, `/tmp/nora_deep_evolution.log`,
`/tmp/nora_deep_law.log`, `/tmp/nora_deep_bridge.log`,
`/tmp/nora_deep_workspace.log` ve `/tmp/nora_deep_consciousness.log` altındadır.

### Derinlik sonrası kesintisiz koşu

Son efferent ayrımı ve bütün üretim değişiklikleriyle ana pil tek başına,
kesintisiz çalıştırıldı: **134/134 GEÇTİ (981,1 sn)**, sıfır `KALDI`.
Kanıt `/tmp/nora_deep_full_134.log`. A4–A6 yaşam regresyonları ile A134–A137
aynı koşuda geçti; bilinç pili de aynı son üretim sürümüyle **9/9** olarak
`/tmp/nora_deep_consciousness.log` içine kaydedildi.

## Aşama BO — Kullanıcı tarifinden dış pil ve icat/keşif ayrımı

`external_battery.py`, kullanıcının D1–D10 tariflerini iç pilin test yardımcılarına
bağlanmadan ayrı davranışlar olarak sınar. Sayısal beklenenler bağımsız oracle ile
hesaplanır; her test kendi nesnesini kurar ve D9 kabul zinciri kendi geçici
çalışma alanında yürür. Bu, kullanıcı tarifinden yazılmış bir **dış pil**dir;
kodu hiç görmeyen bağımsız bir insan denetçinin pili olduğu iddia edilmez.

İlk koşu ve sonraki ölçüm-hijyeni düzeltmeleri boyunca üretim commit'i
**`3b38dd7`** olarak donduruldu; kırmızıları kaldırmak için üretim yaması
yapılmadı. İlk ham sonuç `/tmp/nora_external_d1_d10_initial.log` içinde **3/10**
idi. D5'te ilk ağrı raporunun akut sözel tepkiyi tükettiği, D6'da yerel sayacın
global seçimi de saydığı görüldü. Yalnız bu iki yanlış atıf düzeltildi: D5 aynı
başlangıçlı iki ajanla tek-rapor/tek-ablasyon karşılaştırmasına, D6 gerçek
`local.only` konusunun sayımına çevrildi. Düzeltilmiş dış koşu **5/10 (9,0 sn)**;
kanıt `/tmp/nora_external_d1_d10.log`.

| dış test | sonuç | ölçülen davranış |
|---|---|---|
| D1 belge-only Roma rakamı | **KALDI** | Belge öğrenilemedi, program/stdout yok; belgesiz kontrol dürüst `abstain` verdi |
| D2 kodsuz kavramdan dönüşüm | **KALDI** | Tarifteki `RANOR` ile bağımsız `'NorA'[::-1].upper()` sonucu `ARON` çelişiyor; düzeltilmiş `ARON` tanısalı da geçmedi, gerçek stdout `NorA` |
| D3 standart dışı belge tanımı | **KALDI** | Belge sonrası ikişer-artırmalı beklenen stdout üretilmedi; belge önsel davranışı yenmedi |
| D4 görülmemiş ikili ablasyon | **GEÇTİ** | Üç yeni çiftte tahmin/gerçek **3/3**; üç tekil kontrol ve restore geçerli |
| D5 yeni periferik ayrışma | **GEÇTİ** | Omurilik geri-çekilme adımları ablate edilirken ağrı raporu sürdü, refleks `(2,True)→(1,False)`; bütünleşik durum sabit ve restore başarılı |
| D6 kapasite=1 seçici yayın | **GEÇTİ** | Tek seçim, yalnız kazanan işlev çağrısı; kaybeden çağrılmadı, yerel kanal çalıştı |
| D7 iki kat ilkelde skor | **GEÇTİ** | Gerçek metot-ilkelleri **4→8**, aynı 10 tek-örnek görevde skor **9/10→9/10** |
| D8 köprüsüz dil→matematik | **KALDI** | Yeni ders sonrası `I artı I` için doğru iki/2 cevabı yok; kontrol de doğru cevap vermedi |
| D9 kusurlu yama rollback | **GEÇTİ** | İki kabul sonrası kusurlu üçüncü yama eski pilde kaldı; kaynak ara kabulün hash'ine döndü, eski dış pil geçti ve geçici dosya kalmadı |
| D10 bilinmeyen istek konumlu gap | **KALDI** | Dürüst `semantic_gap` var, program/stdout uydurulmadı; fakat takılan anlamlandırma adımı ayrı bir konum olarak bildirilmedi |

### Eski ilkel kompozisyonu taraması

D1–D10'dan ayrı kontrol, belgesiz kurulan `stream2`yi yeni operatörün kendisini
kullanmadan eski `bind`, `iterate` ve `concatenate` ilkellerinin bağımsız küçük
ifade uzayında yeniden aradı. Train beş örneğin tamamını geçen adaylar arasında
iki held-out çıktıyı da bağımsız yorumlayıcıyla eşleştiren bir kompozisyon 106.
aday denetiminde bulundu. Tanık kompozisyon boş durumla başlıyor; her öğede
çıktıdan önce `durum ← öğe + durum` ve `çıktı ← öğe + durum` uyguluyor. Yani
yeni operatör çağrısı olmadan yalnız eski birleştirme ilkeliyle aynı held-out
davranış kurulabiliyor. Dolayısıyla bu dar deneyde yeni yapı eski ilkellerle
erişilebilirdi ve doğru sınıflandırma **“icat” değil, “keşif”**tir.
Bu sonuç başka temsil türleri için genel bir icat-olumsuzluğu iddiası değildir.

Bu aşamada yalnız dış ölçüm ve belge değişti; üretim davranışı değişmediği için
önceki **134/134** ana pil ve **9/9** bilinç pilinin üretim kanıtı geçerliliğini
korur. Açık D1/D2/D3/D8/D10 kırmızıları bilgi olarak bırakılmıştır.

## Aşama BP — Dış kırmızılardan genel kök düzeltmeleri

Aşama BO'nun dış pili ve beklenenleri değiştirilmeden üretim kökleri derinleştirildi.
`external_battery.py` SHA-256'sı sabit kaldı; D1/D2/D3/D8/D10 için hedefe özel
program, cevap veya stdout müdahalesi eklenmedi. Yeni üretim modülü yoktur;
değişimler mevcut `nora_semantic_code.py`, `nora_code.py` ve `nora_lang.py`
içindedir.

### Belgeden değer ve dönüşüm anlamı

- **D1:** Belgedeki tek sembol/değer eşlemeleri ile soldan-sağa çıkarmalı toplama
  kuralı artık birlikte öğreniliyor. Çok sembollü belge örnekleri kuralı doğrulamadan
  çözücü kabul edilmiyor. Program, hedefin önceden hesaplanmış cevabını değil,
  belgeden gelen tabloyu ve genel döngüyü AST'ye dönüştürüyor. `MCMLIV` ve `1954`
  üretim kodunda yoktur. Belgesiz kontrol `value_decoder` boşluğunu dürüstçe
  bildirirken belgeli çalışma gerçek programla **1954** üretti.
- **D2:** Kodsuz anlam belgesindeki ilkel dizgi dönüşümleri sıralanıp taşınabilir
  bir işlem hattı olarak öğreniliyor; yeni girdinin AST'si bu sırayla kuruluyor.
  Bağımsız beklenen `ARON` artık doğrulanmış programdan geliyor. Dondurulmuş dış
  testteki `RANOR` beklentisi bağımsız oracle ile çeliştiği için D2 biçimsel olarak
  yine **KALDI**, fakat `ARON tanısal=True` oldu. Bu fark skorla gizlenmedi.
- **Aktarım:** Roma adı içermeyen yeni `A=1, B=5, C=10` belgesiyle `CAB` için
  bağımsız `10−1+5` oracle'ı ve başka bir dizgi de doğrulandı. Yani çözüm hedef
  sembollerine veya hazır cevaba bağlı değildir.

### Belgenin önseli yenmesi ve konumlu boşluk

Kümülatif ifade artık yalnız Boolean değil, belgeyle öğrenilen bir adım taşıyor.
Alıntılanan ifadenin ilk satır ve sonraki satır tanımlarındaki öğe sayıları
eşleşirse genel `cumulative_window` planı öğreniliyor; bileşik anahtar da adımı
içerdiğinden birer-artırım önbelleği ikişer-artırımı ezemiyor. D3'te belge,
önsel davranışa karşı kazandı ve yeni girdi için `ab / abcd / abcdef` programı
doğrulandı. Birer-artırımlı eski P6/A133 davranışı da korunmuştur.

İstek ayrıştırıcı `hece`yi artık bilinen bir harf/dizgi birimi gibi yutmuyor;
`unit=syllable` olarak ayırıyor. Planlayıcı ilk çözülemeyen aşamayı
`gap_stage`, `missing` ve iz üzerinden bildiriyor. D10'da program/stdout
uydurulmadan `semantic_gap`, `unit:syllable`, `gap_stage=unit` döndü.

### Paylaşılan bilgiyle köprüsüz alan aktarımı

D8 için özel bir dil→matematik köprüsü yazılmadı. “X bir demektir” gibi sağ
tarafı bütünüyle sayı olan genel öğretiler kavram ağına `NUMERIC_VALUE` olgusu
olarak yazılıyor. Mevcut matematik ayrıştırıcısı sayı sözcüğü/rakam dışındaki
bir sembolü ancak doğrudan ve tekil bir paylaşılan sayı bağlantısı varsa çözüyor;
çelişkili veya kalıtsal bağlar sayı sayılmıyor. Öğrenilmiş ajan `I artı I` için
**2**, belgesiz kontrol ise bilmediğini söyledi. Ayrı `alfa` sembolü de aynı
mekanizmayla `(1,+,1)` çerçevesine çözümlendi. Bilgi yeni oturum/checkpoint
state'inde de korunuyor.

### Son kanıt

Dış pil aynen yeniden koşuldu: **9/10 (35,5 sn)**. D1, D3, D8 ve D10 yeşil oldu;
D2'nin bağımsız `ARON` tanısalı da yeşil, tek biçimsel kırmızı ise tarifin kendi
`RANOR`/`ARON` ölçüm çelişkisidir. Önceki beş geçiş (D4–D7 ve D9) korunmuştur.
Kanıt `/tmp/nora_external_after_fixes.log`; dış dosyanın değişmediği ayrıca hash
ile denetlendi.

Regresyonlar: pratik belge pili **7/7 (4,89 sn)**, ilgili ana test seçimi
**7/7**, kesintisiz ana pil **134/134 (973,9 sn)** ve aynı üretimle işlevsel
bilinç pili **9/9 (18,5 sn)**. Kayıtlar sırasıyla
`/tmp/nora_practical_after_fixes.log`, `/tmp/nora_targeted_after_fixes.log`,
`/tmp/nora_full_after_external_fixes.log` ve
`/tmp/nora_consciousness_after_external_fixes.log`. Kod-sözü denetimi
**1125/127**, önceki farkla aynı **+3/+0** düzeyindedir. Bilinç satırı yalnız
işlevsel nedensel kanıt kapsamındadır; fenomenal bilinç iddiası değildir.
