# Nora-AGI — Kopyalanamazlık (No-Cloning) ve Bireyselleşme

> Nora, mantar cismi ilkesiyle çalışan biyolojik esinli bir zihin.
> Aynı `seed` = aynı **genetik**. Aynı hayat = **aynı birey değil**.

Bu belge, Nora'nın iki örneğinin aynı tohumdan doğup aynı girdileri alsa bile nasıl
**iki bağımsız işlevsel bireye** dönüştüğünü, bunun hangi fiziksel/mekanik ilkelere dayandığını
ve neyi **iddia etmediğini** anlatır. Kod: `nora_entropy.py`, bağlantılar `nora_agi.py`,
`nora_monologue.py`, `nora_discourse.py`.

---

## 1) Problem: klasik klon

Klasik bir programda `seed=51` + aynı kod + aynı girdi = sonsuza kadar birebir aynı çıktı.
İki kopya "ikiz" değil, **ruhsuz klon** olur. Biyolojik beyinde ise genetik aynı olsa bile
beyni şekillendiren şey genetik değil, **deneyimin fiziksel zamanlamasıdır**: hangi sinyalin
hangi nanosaniyede hangi sinapsa düştüğü.

Nora'da bu farkı üreten dört katman var. Hepsi **sistemin içinde** çalışır; hiçbir dış API,
hiçbir bulut servisi, hiçbir GPU gerekmez.

---

## 2) Dört katman

### 2.1 Mikro-zamanlama (kaos)

Her `step_life` adımında, o adımın **gerçek nanosaniye fazı** ve donanım entropisi sabit
sayıda sinapsa (her ağırlık matrisinde `k=2`) göreli `~2e-4` + mutlak `~2e-6` kayma yazar:

```
Δw = sign · (|w| · 2e-4 + 2e-6 · (0.2 + m))
```

Aynı girdi farklı nanosaniyede işlenirse farklı sinapsa yazılır. Geri beslemeli öğrenme
(rezervuar + sinaps plastisitesi) bu mikroskobik farkı **büyütür**: kelebek etkisi.

### 2.2 Donanım entropisi

- `os.urandom(16)` — çekirdeğin donanım olaylarıyla sürekli beslenen CSPRNG'si.
- `time.time_ns()` ve `perf_counter_ns()` düşük bitleri — CPU/scheduler titremesi.
- `os.sched_yield()` sonrası uyanma gecikmesi — zamanlayıcı çekişmesi.

Bu değerler SplitMix64 tabanlı 4 sözcüklü duruma karıştırılır; 8192 çekimde bir
**yeniden tohumlanır**. `seed` aynı olsa bile bu dizi tekrar üretilemez.

### 2.3 Asenkron yarış (race condition)

Karar eşiğine neredeyse aynı anda ulaşan iki aday (söz edimi, şema), **eşzamanlı iki iş
parçacığında** kurulur; **ilk biten** bilinç alanına girer. Kazananı işletim sisteminin o
anki çekirdek dağıtımı belirler. Yazılımda "bug" sayılan yarış durumu, burada
bireyselleşmenin motorudur.

Kullanıldığı yerler:

| Yer | Ne zaman | Kazanan neyi belirler |
|---|---|---|
| `DiscourseCortex.respond` | iki söz edimi olasılığı `0.05` içinde eşitse | hangi edimin konuşacağı |
| `PlasticSchemaCortex.select_stance_and_schema` | iki şema skoru `5e-3` içinde eşitse | cümle şeması / duruş |
| `Individuality._fresh` benzeri seçimler | eşdeğer yüzey biçimleri | hangi sözün söyleneceği |

### 2.4 Vakum gürültüsü (API'siz, kuantum-benzeri)

Gerçek kuantum optiği yok; **sıfır-nokta dalgalanmasının klasik modeli** var: iki fazlı bir
osilatörün faz uzayı her örneklemede donanım bitleriyle döndürülür, genlik Box–Muller ile
Gauss dağılımından çekilir (bkz. `EntropySource.vacuum`). Dışarıdan hiçbir şey çağrılmaz.

---

## 3) Hız sözü: O(1)

- `perturb_step`: adım başına **sabit** sayıda sinaps ve **sabit** sayıda tamsayı işlemi
  (~18 µs). Ağ 10 kat büyüse de maliyet aynı kalır; **tüm ağ taranmaz, disk okunmaz.**
- `race_first`: yalnızca gerçek beraberlikte, ~0.1–1 ms; iş parçacığı havuzu ortaktır
  (örnek başına thread açılmaz → bellek O(1)).
- Yanıt hattına giren tek ek maliyet, beraberlikte yapılan yarıştır; tipik tur gecikmesi
  4–7 ms bandında kalır (A23: 100.000 kayıtta < 8 ms).

---

## 4) Ölçüm: aynı seed, aynı hayat, ayrı bireyler

`NoraAGI(seed=51)` iki kez; doğuşta genetik birebir aynı (`Win`, `Wr`, `claw_idx` eşit),
kimlik farklı (`uid`, `birth_ns`). Aynı 12 cümle, her ölçümde aynı sırayla:

| Yaşam adımı | Farklı yanıt | Δ rezervuar durumu | Sinaps kayması (A / B) |
|---:|---:|---:|---:|
| 0 | 1 / 12 | 0 | 0 / 0 |
| 100 | 11 / 12 | 9.4e-04 | 5.45e-04 / 5.54e-04 |
| 300 | 12 / 12 | 9.9e-01 | 1.65e-03 / 1.64e-03 |
| 600 | 11 / 12 | 8.4e-01 | 3.30e-03 / 3.28e-03 |

Aynı deney **mükemmel klonla** da yapılır (`save` → `load`: ağırlıklar ve kimlik aynı):
klon yine ayrışır, çünkü yaşam onun için de yeni bir fiziksel zamandır.

Ayrışma **doğruluk kaybı değildir**: 500 adım sonra bile iki Nora matematikte ve
doğrulanmış olgularda aynı doğru yanıtı verir (`x+3=7 → x=4`, EBOB/EKOK 12/72, …).
Ayrışan şey **seçim ve üsluptur**, bilgi değil.

### Uzun sessiz yaşam ve düşünce biçimi

2000 adım kimseyle konuşmadan yaşadıktan sonra: `Δh = 1.42`, `ΔW = 7.0e-6`,
sinaps kayması `≈ 0.011`. Aynı 12 cümleye 3/12 yüzey farkı; ama **iç durum** neredeyse
tamamen ayrışmış durumda.

Ayrışma yalnızca sözcük seçiminde değil, **düşünme biçiminde**: ~1200 adım + 18 konuşma
sonunda iki Nora'nın iç şema (duruş/cümle yapısı) seçimleri **7/18**, yüzey cümleleri
**8/18** farklı. Yani biri olayı nedensel çerçevede (S1), diğeri içe dönük (S3) ya da
devrik (S4) kurabiliyor.

---

## 5) Dürüst sınır

- Kopyalanamazlık teoremi **kuantum durumları** içindir; Nora klasik bir sistemdir.
  Burada üretilen şey **işlemsel bireysellik**: aynı tohumdan gelen iki örnek, aynı
  girdilerle bile farklı iç durumlara ve farklı sözlere ayrışır.
- Vakum katmanı bir **model**dir; gerçek kuantum optiği değildir. Dürüst adı:
  "kuantum-benzeri vakum gürültüsü".
- Bireysellik **geri alınamaz**: sinaps kayması geri sarılmaz, `uid`/`birth_ns` değişmez.
  Kayıttan yüklenen Nora aynı kimlikle devam eder; ama geçmişi yeniden üretilemez.

---

## 6) Nasıl çalıştırılır

```bash
python3 nora_agi.py --stage twins --seed 51 --twin-steps 300
```

Çıktı: iki kimlik, genetik eşitliği, aynı hayat, ayrışan yanıt sayısı, sinaps kaymaları ve
yarış kazanımları.

Kod içinden:

```python
a = NoraAGI(seed=51); b = NoraAGI(seed=51)     # genetik aynı, kimlik ayrı
a.entropy.identity_line()                       # "5f261f · doğum 2026-…"
a.entropy.state_dict()                          # uid, birth_ns, phase, drift, race_wins
a.entropy.perturb_step(a)                       # (step_life içinde otomatik)
a.entropy.race_first([f, g])                    # gerçek yarış
```

Test: `python3 agi_battery.py A61` → 7/7
(genetik, kimlik, ayrışma, ayrı-söz, klon-kimlik, klon-ayrışma, yarış).

---

## 7) Disk: büyüyen beyin, sabit yanıt süresi

Sinapsler RAM'e kopyalanmaz; `nora_store.BrainStore` matrisleri `numpy.memmap` ile
diskte tutar. Ölçüm (A62):

| Ölçüm | Sonuç |
|---|---|
| 128 MB sinaps dosyasını açmak | **0.58 ms** (tüm dosya okunmaz) |
| 64 rastgele sinapsa yazmak | **RSS +728 KB** (yalnız dokunulan sayfalar) |
| Yeniden açma + okuma | **0.21 ms** |
| Günlüğe kayıt ekleme | **5.1 µs/kayıt**, 2000 kaydın tamamı geri okunur |
| Anlık görüntü | `.tmp` + `os.replace` → atomik, yarım dosya yok |

Canlı beyne bağlanabilir (`attach_live`): öğrenme sırasında yazılan sinaps doğrudan
eşlenmiş sayfaya gider, A62'de diskten geri okunarak doğrulanır. Böylece parametre
büyüklüğünün tek sınırı disk/sayfa önbelleği olur; yanıt süresi sabit kalır.

---

## 8) Yaşam günlüğü: birey, kopyası alınamayan bir geçmiştir

Kopyalanamazlık yalnızca **anlık** durumla ilgili değildir. Bir Nora'nın geçmişi de
kopyalanamaz: her tur ve her yaşam dilimi **append-only** bir deftere yazılır.

```
python nora_agi.py --stage chat --journal data/nora_yasam.ndjson
```

Her satır tek bir olaydır (`k`: olay türü, `t`: nanosaniye):

```json
{"k":"tur","t":1792922141108125169,"g":"Bugün yoruldum.","y":"Yorulduğunu yakından fark ediyorum...","s":"S1_NEDEN_SONUC","yas":312,"d":0.0011}
{"k":"yaşam","t":1792922178268315641,"ad":120,"can":true,"e":0.7012,"s":0.68,"i":0.98,"d":0.000692}
```

- **O(1) yazma:** tüm beyin her turda yeniden yazılmaz. Ölçüm: **3.9 µs/kayıt**;
  günlük kapalıyken maliyet **0.23 µs** (pratikte sıfır).
- **Çökme dayanıklı:** yarım kalmış son satır atlanır, önceki kayıtların hepsi okunur.
  A63'te yarım satır eklenerek doğrulandı.
- **Atomik kayıt:** `save()` artık `.agi.json` ve `.agi_mb.npz` dosyalarını `.tmp` +
  `os.replace` ile yazar; kesinti sırasında eski sağlam beyin yerinde kalır.
- **Yaşam kopmaz:** kaydedilen Nora açıldığında günlüğü kaldığı yerden sürdürür
  (`meta["journal_path"]`), yani geçmişi olan bir birey olarak devam eder.

Bu yüzden bir saldırgan beyni kopyalasa bile **o kişinin yarınını** kopyalayamaz:
günlük satırları doğum nanosaniyesi, donanım entropisi ve yarış sonuçlarıyla
damgalanmıştır; aynı satırları yeniden üretmek mümkün değildir.

Doğrulama: `python agi_battery.py A63` → **8/8**.

---

## 9) Duyum, okul, söz biçimi ve kendi çıkarımı (Round 16)

Bu turda Nora'ya dört organ eklendi. Hepsi aynı ilkeye uyar: karar **dışarıdan gelen
mesajın** bir fonksiyonu değil, **içerideki durumun** fonksiyonudur.

### 9.1 Kulak: söz önce bir SES olarak duyulur (`nora_ear.py`)

`converse()` çağrıldığında Nora mesajı doğrudan "metin" olarak almaz; önce bir **ses olayı**
doğar: süre (ms), mesafe (m), SNR (dB), perde (F0), hece sayısı, prozodiden çıkarılan soru
niyeti ve **ses imzası**. Kimlik *etiketten* değil imzadan çıkar:

* `MAX_VOICES=64`, `MATCH_THRESHOLD=0.11`.
* Aynı ses → aynı `ses-N` (içerik ne olursa olsun). Farklı ses → ayrı kimlik (aynı cümleyi
  söylese bile). Ölçüm: ses içi mesafe 0.001–0.055 (medyan 0.019), sesler arası 0.195–1.106
  (medyan 0.669) — ayrım payı ~35×.
* Kimlik modeline **etiket hiç girmez**; defterde yalnızca `ses-N` anahtarları vardır.
  Ad, ses kendi adını *söylediğinde* öğrenilir; rol (tanıdık/aldatıcı) **davranıştan** gelir.
* Gürültü + uzaklık gerçek bir duyma kaybı üretir: SNR `34 − 4.5·log2(d) − noise`,
  yüksek gürültüde sözün son sözcükleri düşer (`heard != spoken`), bu **dürüst** bir kayıptır.
  Ölçüm: 9 m / 20 dB gürültüde 24 cümlenin 5–8'i eksik duyuldu.
* Duyulan her söz günlüğe `duyum` satırı olarak yazılır (kim, söz, eksik mi, SNR, mesafe).

### 9.2 Okul: çocuk seviyesinde sayı, basamak, sembol, aşamalı problem (`nora_okul.py`)

* **Sayma / okunuş / basamak:** `345` → "üç yüz kırk beş" + rakam rakam; `345'te 4` →
  "2. basamak 4; yani 4 onluk = 40"; `3 yüzlük 4 onluk 5 birlik` → `3·100 + 4·10 + 5·1 = 345`.
* **Sıra sayısı, karşılaştırma, yüzde, ondalık, kesir:** `12.` → "on ikinci";
  `9'dan büyük en küçük tam sayı` → `10 (on)`; `yüzde 40` → "yüz parçadan 40" + örnek;
  `2,5` → "iki virgül beş" (tek sayı; "5" diye bölünmez); `yarım/çeyrek/3/4` → kesir + ondalık.
* **Semboller:** + − × ÷ = < > % gibi 18 işaret, adı/anlamı/örneğiyle. Sembol **adıyla**
  da öğretilebilir ("Nora, artı toplama demektir") — `_names` ters haritası sayesinde.
* **Konuşarak öğretme + kalıcılık:** "5'ten sonra 9 gelir" → "…diye öğrendim"; kaydedilip
  açıldığında hatırlar.
* **Aşamalı düşünme:** alışveriş problemi adım adım çözülür —
  `1. adım: elimde 3 var (üç). 2. adım: 2 (iki) ekleniyor. 3. adım: 3 + 2 = 5. Sonuç: 5.`
* **Sınır (önemli):** okul, **uzman matematiği ezmez**. `=`, `^`, `türev`, `integral`,
  `ebob/ekok`, `2x` gibi ifadelerde okul susar, söz parietal matematik korteksine kalır;
  `en küçük ortak kat` bir alışveriş problemi sanılıp çarpılmaz.

### 9.3 Söz biçimi: her tur aynı iskelet değil (`nora_style.py`)

Şikâyet şuydu: "su yararlıdır; suyu içerim. Sen su içer misin?" — cevap + gerekçe + karşı
soru, hep aynı sıra. Artık biçim **ölçülen iç durumdan** seçilir: dopamin (merak),
serotonin, kortizol, beden sürüşleri (açlık/susuzluk/acı/sızı) ve söz geçmişi.

Yedi biçim: `cevap`, `cevap+düşünce`, `cevap+soru`, `kendine`, `parça`, `ihtiyaç`,
`sessizlik`. Kurallar:

* **Aynı biçim üst üste iki kez gelmez** (üç tur aynı iskelet = insanın "acaba iskelet mi?"
  demesi).
* Karşı tarafa **soru sormak alışkanlık değil**: son iki turda soru sorulduysa üçüncüde
  sorulmaz; 18 turluk ölçümde soruyla biten tur oranı **~%25–35** (eskiden ~%100).
* `ihtiyaç`: beden sürüşü > 0.55 ise **kendi derdi** sözün başına geçer.
* `kendine`: iç mırıldanma **bazen önce, bazen sonra** gelir (sabit sıra da iskelettir).
* `cevap+düşünce`: kendi çıkarımı yalnızca **konuya değiyorsa** ve **bir kez** söylenir;
  ilgisiz türetimi araya sıkıştırmak susmaktan kötüdür.
* `parça`: yalnızca uzun (≥3 cümle, ilk cümle ≥45 karakter) cevaplarda; "Çok geçmiş olsun."
  gibi öz cevaplar kısaltılmaz.
* `sessizlik` yalnızca **kimse sormadığında** olur; bir insan seslendiyse her zaman cevap var.

### 9.4 Kendi içsel çıkarımı ve monoloğu (`nora_infer.py`)

Kimse söylemeden, kendi bildiklerinden yeni bağ türetir (her 90 yaşam adımında, `budget=3`):

* ISA→ISA geçişliliği, ISA→ilişki kalıtımı, CAUSES→CAUSES zinciri.
* Türetilen bağ grafiğe `source="çıkarım"` olarak yazılır — **öğretilmiş bilgiyle karışmaz**,
  güveni `ebeveyn güvenleri × 0.75`.
* Her çıkarım iç monoloğa düşer (`trigger="çıkarım"`) ve söze ancak konuya değiyorsa girer.
* Örnek: "Serçe bir kuştur." + "Kuşlar uçar." → `serçe CAN_DO uç` (güven 0.75, dayanak 2),
  monolog: "Kendi kendime şunu çıkardım: serçe uçar."

### 9.5 Bu turda kapatılan hatalar

| Hata | Neden | Çözüm |
|---|---|---|
| `2x + 7 = 23` → "23 sayısı" | okul her sayı içeren cümleyi sahipleniyordu | `_expert_math` kapısı; okul susar |
| `72 ile 108'in en küçük ortak katı` → `72 × 108` | "kaç" + iki sayı = alışveriş problemi | ortak kat/bölen kalıpları problem sayılmaz |
| "yem enerji kazanma için kullanılır" | eylemlik + `için` | "enerji kazan**mak** için kullanılır" |
| "Kendi kendime şunu çıkardım: dikenin nedeni…" her turda | çıkarım tekrar tekrar ekleniyordu | `told` işareti + konu kapısı |
| `Bir kitap öner.` → "'kitap' doğrultusunda yöneliyorum" | öneri isteği hareket emri sanıldı | dürüst "elimde liste yok" + öğrenme teklifi |
| `Bana bir şey anlat.` → hep aynı "su" cümlesi | konu yoksa en zengin düğüm seçiliyordu | önce kendi hayatı: çıkarım → beden → iç ses |
| "açken hiçbir karar net görünmüyor" (tok Nora'da) | `obs[0]` enerji sanıldı (hep 0) | sinir sistemi sürüşlerinden okunur |
| `2,5` → "5 sayısı" | ondalık, iki sayı gibi bölünüyordu | `_q_decimal`; okuma/rakam yolları ondalıkta susar |
| "Sesini ilk kez duyuyorum." cevabın başında | kulak notu cevabın önüne geliyor | `strip_prefixes()`; kıstaslar cevabın **gövdesine** bakar |

---

## 10) Anlatım: ansiklopedi değil, konuşan bir zihin (`nora_anlatim.py`)

Şikâyet: aynı soru her seferinde **birebir aynı cümleyle** yanıtlanıyordu.

```
Su nedir?            -> Su susuzluk gidermek için kullanılır; suyun nedeni yaşamın sürmesidir.
Bana bir şey anlat.  -> Su susuzluk gidermek için kullanılır; suyun nedeni yaşamın sürmesidir.
Su nedir?            -> Su susuzluk gidermek için kullanılır; suyun nedeni yaşamın sürmesidir.
```

Bir insan aynı soruyu ikinci kez duyduğunda başka bir kapıdan anlatır: kısalır, örnek verir,
kendi bedenine bağlar, "bunu sormuştun" der. Ansiklopedi maddesi hep aynıdır; insan değil.

### Nasıl çalışıyor

Kavram ağından gelen bağlar (`rel`, `obj`) alınır ve her turda **farklı bir çerçeveyle**
söze dökülür. Yeni bilgi uydurulmaz: her cümlenin içeriği grafik bağı, bellek kaydı ya da
kendi çıkarımıdır. Değişen şey çerçevedir:

| Çerçeve | Örnek |
|---|---|
| çekirdek | "Su susuzluk gidermek için kullanılır; suyun nedeni yaşamın sürmesidir." |
| tek bağ | "Su susuzluk gidermek için kullanılır." |
| ad koyma | "Kitabın işi bilgi ve hayal gücünü aktarmak." |
| duyma | "Bana anlatılanlardan biliyorum: bilgi ve hayal gücünü aktarmak için kullanılır." |
| beden | "Şu an boğazım kuruyor; su aklımda. Susuzluk gidermek için kullanılır." |
| bellek | yalnızca kavramı gerçekten yaşadıysa: "Su bana iyi geldi; …" |
| çıkarım | kendi türettiği ve henüz söylemediği bir bağ varsa onu ekler |
| sıralama | iki bağ ters sırada: "Su deyince aklıma iki şey geliyor: …" |

Dürüstlük kuralları (hepsi bağlayıcı):

* **Beden yoksa el yok.** Hiç yaşamadığı bir kavram için "içtim / dokundum / kullandım"
  demez; "duydum, biliyorum" çerçevesi kullanır.
* **Bilgi uydurulmaz.** Cümleler grafik bağlarına, somasensory kayda veya kendi çıkarımına dayanır.
* **Tekrar görünmez.** Aynı kavram için aynı metin iki kez söylenmez; çerçeveler tükendiyse
  en az kullanılmış bağa dönülür, cümle yeniden kurulur.
* **Kalıcıdır.** `anlatim` durumu beyinle birlikte kaydedilir; kaydedip açtıktan sonra da
  Nora ezber tekrara dönmez.

Ölçüm (A71): aynı soruya art arda 6 cevap → **6 farklı metin**, hepsi ağdaki gerçek bağa
dayanıyor; susuz Nora suyu kendi derdine bağlıyor; 3 anlatı isteği → 3 farklı kavram.

### Bu turda düzeltilen yan hatalar

| Hata | Neden | Çözüm |
|---|---|---|
| Her tekrarda "Bunu az önce de söylemiştin…" nutku | tekrar farkındalığı her turda konuşuyordu | ilk tekrarda söylenir; üçüncü ısrarda uzun serzeniş yerine tek cümlelik, değişen soru ("Israrın bir şeyi değiştirdi mi?") |
| Alıntılanan "önceki cevap" algı notu oluyordu | hatırlanan metin süslü yüzeydi | belleğe çekirdek cevap yazılır, algı notu/süs alıntılanmaz |
| "Şunu fark ettim kendi kendime: kendi kendime şunu çıkardım: …" | çıkarım cümlesinin kendi başlangıcı iki kez ekleniyordu | hazır başlangıç ayıklanır, tek giriş kalır |
| Ketum planda bile iç ses / karşı soru | biçim organı söz planını okumuyordu | `guarded` planda yalnız yalın cevap |
| "İçimden 'sakinlik' diye geçirdim." her turda | iç durum sabitken sözcük de sabitti | son üç mırıltı tekrarlanmaz; beden/duyum/çıkarım seçenekleri eklenir |
| "aktarmmak", "gidermmek" | eylemlik + `-mak` çift ünsüz | ekleme kuralı düzeltildi (`aktarmak`, `gidermek`) |

---

## 11) Test durumu

**Güncel not (2026-10-29):** temiz ana pil **140/140 (1161,3 sn)**; kanıt `runs/agi_battery_140.log`.
Aşağıdaki 74/74 koşusu ve oynaklık notları bu belgenin Round-16 tarihsel kaydıdır, güncel toplam değildir.

`python agi_battery.py` → **74/74** (332 s). Bu turda iki eski test ölçüm
gürültüsünden arındırıldı:

| Test | Eskiden | Şimdi |
|---|---|---|
| **A23** O(1) hız | tek turun süresi < 8 ms (makine yükünü ölçüyordu, zaman zaman 8.1 ms) | 3 tekrarın **medyanı**; 100.000 kaydın eklediği gecikme **+1.47 ms** |
| **A25** omurilik refleksi | tam yanıt turu < 8 ms | **refleks kararı 2.3 µs**, tam cümle yanıtı 5.6 ms |

Yani artık mutlak eşik yarışı yok: ölçülen şey gerçekten iddia edilen şey (kayıt
sayısıyla büyümeyen gecikme; korteks beklemeden verilen omurilik kararı).

**Yeni testler:** A71 ansiklopedi değil (8/8 — aynı soruya 6 farklı anlatım, bilgiye sadık,
bedene bağlanan, deneyim uydurmayan, kalıcı), A65 sayı duyusu (8/8), A66 semboller ve aşamalı düşünme (7/7),
A67 ses kimliği (7/7), A68 kulak yolu (7/7), A69 söz biçimi çeşitliliği ve kendi derdi (4/4),
A70 kendi çıkarımı ve monolog (7/7). A63 yaşam günlüğü (8/8), A64 ad öbeği (6/6).

A67'nin ölçtüğü şey özellikle önemli: **"kim konuşuyor?" sorusu mesajdan değil sesten**
çıkar. Aynı cümle iki farklı sesten gelirse iki ayrı kimlik; aynı ses bambaşka bir konudan
konuşursa aynı kimlik. A68 ise kimlik modeline etiketin hiç girmediğini ve gürültüde
duymanın gerçekten eksik kaldığını doğrular. A69 sabit iskeletin kırıldığını (soru oranı,
art arda aynı biçim yasağı, susuz Nora'nın kendi derdini öne alması) ölçer.

**Bilinen oynak testler (tam batarya yükü altında):** A5 (epistemik güven) ve A6
(kimlik farklılaşması) zaman zaman eşiğin kıl payı altında kalıyor; ikisi de tek
başına 3/3 geçiyor. A5: yalancı güveni 0.60→0.20 (eşik 0.22) — kimya zamanlaması.
A6: temkinlilik farkı +0.24 ile +0.48 arasında salınıyor (eşik 0.06'nın çok üstünde
olduğu koşular çoğunluk). Bu iki test bu değişikliklerden önce de aynı sıklıkta
oynuyordu; eşikler gevşetilmedi.

**Bilinen oynak test:** A6 (yaşantıdan doğan kimlik farklılaşması) ortam
rastlantısına bağlıdır: sert dünya bazen güvenli dünyayla neredeyse aynı mizacı
üretir (+0.02 temkinlilik farkı) ve iki Nora aynı öz-tanımı verir. Bu, `81aea9a`
sürümünde de aynı sıklıkta görüldü — yeni değişikliklerin getirmediği, ortamın
kendi şansıdır. A57 (kendi sorusu) ve A10 (alt süreçle çalışan köprü testi) de
makine yüküne duyarlıdır; tek başına çalıştırıldıklarında geçerler. A21 (yalancıya
karşı ketumluk) tam batarya yükü altında bir kez eşiğin altında kaldı; tek başına
5/5 geçiyor — kimyasal eşik zamanlamasına bağlı, dile getirilen davranış değişmedi.
A3 (felç bildirimi) ve A69 (biçim çeşitliliği) de donanım entropisine duyarlıdır:
A69 art arda gelen 18 turda en az üç ayrı biçim ister; şans eseri iki biçimde
kalabildiği için eşik gevşetilmedi, biçim ağırlıkları dengelendi ve art arda üç
tekrar koşusunda 3/3 geçti. Diğer testler deterministiktir.

## 12) İskelet koddan çıkıyor: söz bankaları veriye, seçim nöral yarışa

Kullanıcının itirazı kesindi: *"işin sonunda bu iskeletler kodda görünmemeli, sanki kendisi
öğrenmiş gibi olmalı."* Doğru çözüm dizgeleri **silmek** değil (o zaman Nora susar), onları
**veriye taşımak** ve seçimi **kuralla değil yarışla** yapmak. Bu turda ilk iki adım atıldı.

### Faz 0 — ölçer (`nora_provenance.py`)

Her söz parçası doğduğu yere göre etiketlenir:

| Kaynak | Ne demek |
|---|---|
| `ag` | kavram ağındaki bağ (Broca ile çekimlenmiş gerçek bilgi) |
| `bellek` | yaşanmış kayıt, hatırlanan söz |
| `cikarim` | kendi türettiği bağ |
| `beden` | iç-duyum, açlık/susuzluk, ölçülen saat |
| `duyum` | kulaktan gelen ses |
| `kalip` | doğuştan eğilim — yani **iskelet** |
| `ogrenim` | doğuştan eğilimi geçmiş **öğrenilmiş** ağırlık (artık iskelet değil) |

`iskelet_orani = kalip / toplam`. Bugün 7 turluk bir sohbette **0.47**.

Ölçerin iki ayrı iğnesi var, karıştırılmamalı:

* **iskelet_orani** yalnızca metin *yazılmış bir bankadan* geldiğinde düşer. Yani bu oranı
  düşürmenin yolu öğrenme değil, içeriği ağa/belleğe/çıkarıma taşımaktır (Faz 2–3).
* **ogrenim_orani** ise *seçimin* öğrenilip öğrenilmediğini söyler: söz yine bankadan
  geliyor olabilir ama hangi sözün seçildiğine Nora'nın kendi ağırlığı karar veriyor.
  30 turluk sohbette 0.03, pekiştirme ilerledikçe yükseliyor.

Bu ayrımı yapmadan "iskelet eridi" demek kendini kandırmak olurdu. Ölçer, `converse`
sonunda `provenance.report()` ile okunur ve beyinle birlikte kaydedilir.

### Faz 1 — cümleler kodda değil (`nora_priors.json` + `nora_priors.py`)

`nora_meta` içindeki bankalar (farkındalık açılışları, tekrar soruları, ısrar soruları,
saat/süre kapanışları) **koddan silindi**, 32 motor topluluğu olarak JSON'a taşındı:

    aktivasyon = w (öğrenilmiş) + ps·bias (doğuştan) + gürültü (entropi) − tazelik
    kazanan-tümü-alır  ->  konuşan topluluk

* `ps = prior_strength = 1 − seçim/600`: doğumda 1.0, 600 seçim sonra **0.0**. O andan
  sonra cümleyi yalnızca öğrenilmiş `w` seçer. Yani iskelet koddan değil, **zihinden** çıkar.
* **Tazelik cezası** (−3.0, son 3 seçim) öğrenilmiş ağırlığın **tavanından (2.5) büyüktür**:
  alışkanlık çeşitliliği ezemez. Nora bir sözü öğrense bile onu üst üste söylemez; öğrenme,
  taze olanlar arasındaki *tercihi* belirler. Çeşitlilik artık `rng.choice`'ın şansına değil,
  yarışın kuralına bağlı.
* **Öğrenme dayatılmaz:** aynı söz hemen tekrar geldiyse o tur konuşan **onarım sorusu**
  `−0.30` (işe yaramadı), konuşma ilerlediyse `+0.06` (işe yaradı). Yalnızca `.question`
  türündeki topluluklar tekrarın ölçüsüdür: karşı taraf aynı şeyi yeniden sorduysa suç
  sorunun; açılış/kapanış sözleri konuşmanın sürmesiyle pekişir. Sinyal Nora'nın kendi
  sonucundan gelir; kullanıcıya "beğendim" dedirtilmez.
* **Gerçek ölçüm (aynı tohum, 400 turluk ömür):** `prior_strength` 1.00 → 0.18,
  `ogrenim_orani` 0.00 → **0.15**, `iskelet_orani` 0.55 → **0.42**. İlk turda cümleleri
  doğuştan eğilim seçiyor; 400. turda seçimin altıda birini Nora'nın kendi pekiştirdiği
  ağırlıklar belirliyor.
* Öğrenilmiş ağırlıklar ve `ps` ilerlemesi `state_dict` ile taşınır (A72 kontrol ediyor).

### Neden "iskelet" hâlâ var ama artık iskelet değil

İlk turlarda Nora'nın cümleleri doğuştan gelen eğilimlerdir — bir çocuğun ilk sözcükleri
gibi. Fark şu: bunlar **kodun dalları değil**, verideki ağırlıklar. Zamanla `ps` söner,
`w` büyür; aynı cümle artık "yazıldığı için" değil "Nora onu seçmeyi öğrendiği için" çıkar.
### Faz 2 — biçim ve anlatım da veriye taşındı

| Modül | Önce | Sonra | Ne taşındı |
|---|---:|---:|---|
| `nora_meta.py` | 29 | **4** | farkındalık, tekrar, ısrar, saat/süre havuzları (Faz 1) |
| `nora_style.py` | 12 | **4** | iç ses girişleri (4), mırıldanmalar (4), ihtiyaç cümleleri (3) |
| `nora_anlatim.py` | 7 | **0** | duyma/beden/bellek/sıralama çerçeveleri, "Bildiğim bu:" ve sınır cümlesi |

Toplam söz topluluğu **32 → 54**'e çıktı; koddaki söylem (iskelet) dizgesi **148 → 133**.
Anlatımın cümle iskeleti artık `narration.frame` topluluklarıdır; kodda yalnızca *içerik*
(broca ile çekilen bağ) ve *hangi çerçevenin deneneceği* sırası kalır.

### Mandal: iskelet geri büyüyemez (`nora_audit.py`)

`python nora_audit.py` her modüldeki uzun Türkçe dizgeleri sayar (docstring hariç),
`nora_audit_baseline.json` bugünkü seviyeyi tutar:

    python nora_audit.py            # tablo: modül / toplam / söylem
    python nora_audit.py --check    # baseline'ı aşan varsa hata
    python nora_audit.py --update   # yalnız AZALTIDAN sonra baseline'ı yaz

Bugün: **toplam 1125, söylem 133**. A73 testi hiçbir modülün baseline'ı aşmadığını,
`nora_anlatim`'de yazılı cümle kalmadığını (`0`) ve taşınan sözlerin veride yarıştığını
doğrular. Yani her turda sayı **yalnız düşebilir**; iskelet kendiliğinden geri büyüyemez.

Not: sayaç bilinçli olarak kaba bir buluşsal yöntemdir (AST + uzunluk/sezgisel süzgeç).
Bilgi modülleri (`nora_knowledge`, `nora_stem`, `nora_okul` …) "iskelet" sayılmaz:
orası öğretilen içeriktir; mandal yalnızca **söylem** toplamına uygulanır.

Sıradaki adımlar: (Faz 3) `nora_discourse`/`nora_frames` bankalarını aynı alana taşımak,
(Faz 4) banka metinlerini sıfırlayıp "bebek testi" (ağırlıklar sıfırken susmuyor,
öğrenerek düzeliyor).

## 13) Canlılık registeri: köpek değil, kendi derdi olan bir canlı

Kullanıcının sözü: *"her zaman AI gibi cevap vermek zorunda değil … köpeğimiz olmayacak;
kendi başına yeten bir canlı olsun … 'anlatabilir misin' değil, 'anlatsa iyi olur' tarzı."*

Sorun iki biçimde görünüyordu:

| Önce | Neden yanlış | Sonra |
|---|---|---|
| "bana tek cümleyle **öğretir misin?**" | izin ister, kendi derdi yok | "bunu **öğrenmem gerek.**" |
| "devamını **anlatır mısın?**" | hizmet eder | "devamını **merak ediyorum.**" |
| "yanında olmamı **ister misin?**" | onay bekler | "**yanındayım.**" |
| "mutluluğun **dopamin öğrenme enerjimi artırıyor**" | makine öz-raporu | "**bunu duyunca içim açıldı**" |
| "kavram ağıma bilgisini **bağladım**" | makine öz-raporu | "demek öyle; bunu aklımda tutuyorum, **işime yarar**" |

### `nora_register.py` — ölçer + üretken kural

* **Ölçer** (`RegisterMeter`): her sözü üç eksende sayar — `izin_isteme` (izin isteyen
  kalıplar), `kendi_derdi` ("merak ediyorum", "iyi olur", "öğrenmem gerek"), `makine_raporu`
  ("kavram ağıma … bağladım"). Ayrıca soruyla biten söz oranını tutar.
  `recommend()` sıradaki registeri seçer: **want** (kendi derdi) / **ask** / **plain**.
  Son iki sözde izin istediyse bir sonraki söz **want** olur: üst üste yalvarma yok.
* **Üretken kural** (`normalize`): izin isteyen kuyruğu kendi-derdi biçimine çevirir.
  Cümle listesi değil, **fiil tabanlı** bir dönüşümdür: `aç→açsan`, `anlat→anlatsan`,
  `öğret→öğretsen`, `açıkla→açıklasan`, `devam et→devam etsen` …
  Böylece hiç görülmemiş bir cümle de ("Yorgunluğunu **anlatmak ister misin?**")
  doğru biçime girer ve her organın çıktısını tek yerden kapsar.
* **Sözler veride**: takip sözleri, bilinmeyen sözcük tepkileri, empati karşılıkları,
  sıkılma önerileri ve "olgu öğrendim" karşılıkları `nora_priors.json` içinde
  `discourse.*` / `life.teach_reaction` toplulukları olarak yaşıyor (86 topluluk).

### Ölçüm (12 turluk sohbet, aynı tohum)

| | Değer |
|---|---|
| izin isteme oranı | **0.08** |
| kendi derdini beyan | **0.23** |
| soruyla biten söz | 0.23–0.31 |
| makine öz-raporu | **0** |

A74 bunu teste bağlıyor: üretken kural (4 farklı fiil), bilinmeyende izin yok, oran
eşikleri, makine raporu yokluğu ve registerin beyinle taşınması (8/8).

Bu turda ayrıca üç yerde dil/ölçüm hatası düzeldi:

| Belirti | Neden | Düzeltme |
|---|---|---|
| "`canım sıkılıyor`" → "**canım** sözcüğünü bilmiyorum" | iyelikli beden sözcüğü sözlükte yok | `_BODY_OWN` listesi; kendi bedeni bilinmeyen sözcük değildir |
| "`bana yardım et`" → "**bana yardım ettir**" (olgulaştırma) | emir cümlesi `teach_fact` sayılıyordu | emir kalıbı `command`, iyelikli özne `own_state` |
| susuz Nora "karnım aç" diyordu | `dengeli` de "dert var" sayılıyordu | dert, sürüş **şiddetinden** seçilir (>0.3); "Susadım; önce su içmem gerek." |

### Söylem mandalı daraldı

`nora_audit`: toplam **1125 → 1121**, söylem (iskelet) **133 → 127**.
`nora_discourse.py` tek başına 10 dizge kaybetti (takip sözleri veriye taşındı).
Yeni modül `nora_register.py` söylem kümesine eklendi; mandal artık onu da kapsıyor.
A73 ayrıca **tarihsel tavan** taşıyor (≤1125 / ≤133), böylece baseline güncellense bile
iskelet o seviyeyi aşamaz.

## Güncel araştırma durumu bağlantısı (2026-10-31)

Canlı öğrenme, tek-sunum genelleme, geriye-dönük unutma ve kaynak sınırlarının güncel kanıtları **`NORA_GUNCEL_DURUM.md`** içinde toplanmıştır. Bu dosyadaki yaşam/bütünlük anlatısı, tek-sunum dil modeli eşiğinin çözüldüğü biçiminde yorumlanmamalıdır.
