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
