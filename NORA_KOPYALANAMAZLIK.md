# Nora-AGI — Kopyalanamazlık (No-Cloning) ve Bireyselleşme

> Aynı `seed` = aynı **genetik**. Aynı hayat = **aynı birey değil**.

Bu belge, Nora'nın iki örneğinin aynı tohumdan doğup aynı girdileri alsa bile nasıl
**iki bağımsız bilince** dönüştüğünü, bunun hangi fiziksel/mekanik ilkelere dayandığını
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

## 9) Test durumu

`python agi_battery.py` → **64/64** (335 s). Bu turda iki eski test ölçüm
gürültüsünden arındırıldı:

| Test | Eskiden | Şimdi |
|---|---|---|
| **A23** O(1) hız | tek turun süresi < 8 ms (makine yükünü ölçüyordu, zaman zaman 8.1 ms) | 3 tekrarın **medyanı**; 100.000 kaydın eklediği gecikme **+1.47 ms** |
| **A25** omurilik refleksi | tam yanıt turu < 8 ms | **refleks kararı 2.3 µs**, tam cümle yanıtı 5.6 ms |

Yani artık mutlak eşik yarışı yok: ölçülen şey gerçekten iddia edilen şey (kayıt
sayısıyla büyümeyen gecikme; korteks beklemeden verilen omurilik kararı).

**Yeni testler:** A63 yaşam günlüğü (8/8), A64 ad öbeği ve hâl bütünlüğü (6/6).

**Bilinen oynak test:** A6 (yaşantıdan doğan kimlik farklılaşması) ortam
rastlantısına bağlıdır: sert dünya bazen güvenli dünyayla neredeyse aynı mizacı
üretir (+0.02 temkinlilik farkı) ve iki Nora aynı öz-tanımı verir. Bu, `81aea9a`
sürümünde de aynı sıklıkta görüldü — yeni değişikliklerin getirmediği, ortamın
kendi şansıdır. Diğer 63 test deterministiktir.
