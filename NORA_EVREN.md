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
