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
  - Beden, kimya, öz-benlik, Türkçe dil, `ThinkColumn` Sistem-2 muhakemesi, **Sally-Anne Zihin Kuramı (yanlış-inanç ayrımı)**, geçişli büyüklük çıkarımı, 1-atış prosedürel makro-beceri öğrenme ve MDL dizi/fonksiyon keşfini tek organizmada birleştirir (`python agi_battery.py` **13/13 GEÇTİ**, canlı gözlemevi `python nora_studio.py`).


