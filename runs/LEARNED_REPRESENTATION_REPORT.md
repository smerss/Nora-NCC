# Öğrenilen CCG ve ilişkisel önerici: bağımsız karşılaştırma raporu

Tarih: 2026-10-29
Kapsam: mevcut `nora_semantic_code.py`, `nora_broca_net.py` ve `law_battery.py` içindeki öğrenme yolları. Bu belge genel AGI, genel amaçlı LLM veya gerçek dünya modeli kanıtı iddiası değildir.

## 1. Kısa sonuç

- Final kodla iki yeni veri/model tohumunda, yedi görevin toplamında CCG 734/896, scratch Transformer 566/896, pretrained MiniGPT 479/896 tam dizi üretti. Bu pooled sayı yalnız aşağıdaki aynı 14 görevin özetidir; genel üstünlük iddiası değildir.
- Counter24 sonuçları CCG 57 ve 64, scratch Transformer 57 ve 56 oldu. Bir split berabere, bir split CCG lehine büyük fark var. **Bu iki final koşuda auto counter24 için rotation değil linear-summary seçti.** Dolayısıyla counter24 kazancı rotation’a atfedilmez.
- A142’nin yeni dört eğitim tohumunda nesne-uç benzerliği top-1’i 54/136’dan 115/136’ya, doğru-dayanaklı top-1’i 50/136’dan 109/136’ya çıkardı. Object-top-1 59/136’dan 136/136’ya geldi.
- Weighted çözülmüş değil: final bağımsız sonuçlar CCG 43/64 ve 43/64; pozitif sonuçlar 2/16 ve 0/16. Scratch ve pretrained bu iki splitte P0 olsa da CCG’nin ikinci P0 sonucu açık başarısızlıktır.

## 2. Final bağımsız CCG ↔ scratch Transformer ↔ pretrained MiniGPT

### Protokol

- Her model aynı `train=64`, `validation=16`, `held-out=64` çiftlerini ve aynı train/validation/held-out kaynak ayrımını kullandı.
- Her split 16 pozitif-olay ve 48 negatif dizi içerir. Tablodaki `P` ve `N` tam dizi sayılarıdır.
- Veri tohumları `727272`, `727273`; model tohumları `717171`, `717172`.
- CCG auto üç aday çalıştırır: baseline GRU, linear-summary ve learned rotation. Transformer örnek-sunumu üst sınırı üç CCG adayının toplamına eşittir. Erken duran model daha az tüketebilir.
- Örnek sunumu/gradyan adımı eşitliği FLOP, parametre veya duvar süresi eşitliği değildir. Raporlanan scratch model 2 katman, 4 head, d64; pretrained model 4 katman, 4 head, d128’dir.
- Pretrained yol ayrıca Tiny Shakespeare ön-eğitimi ve kayıtlı 5000 adım taşır; scratch ile aynı öğrenme/veri geçmişine sahip değildir. Checkpoint `strict=True` ile yüklenir.
- Pretrained checkpoint: `NaveedShk/miniGPT_Assignment-`, commit `5063ea23d04f596ee6ddbb258cd2b0bbb245d0b3`, SHA256 `64e8d66a04d23a54af1a9ad7022a3ec0c86e3438fc849aa979675c8598744f02`. Bu, karakter seviyesinde küçük bir Tiny Shakespeare modelidir; genel LLM diye sunulmaz. Kaynakta lisans bilgisi bulunmadığından dağıtım varlığı olarak eklenmemiştir.

| Görev | Model | Exact /64, iki tohum | Ort. | P /16, iki tohum | N /48, iki tohum |
|---|---|---:|---:|---:|---:|
| counter mod12 | CCG | 55 / 56 | 55.5 | 7 / 8 | 48 / 48 |
| counter mod12 | Scratch | 55 / 57 | 56.0 | 7 / 9 | 48 / 48 |
| counter mod12 | Pretrained | 33 / 48 | 40.5 | 0 / 0 | 33 / 48 |
| counter mod16 | CCG | 62 / 62 | 62.0 | 15 / 14 | 47 / 48 |
| counter mod16 | Scratch | 63 / 60 | 61.5 | 15 / 12 | 48 / 48 |
| counter mod16 | Pretrained | 48 / 48 | 48.0 | 0 / 0 | 48 / 48 |
| counter mod24 | CCG | 57 / 64 | 60.5 | 9 / 16 | 48 / 48 |
| counter mod24 | Scratch | 57 / 56 | 56.5 | 9 / 8 | 48 / 48 |
| counter mod24 | Pretrained | 48 / 48 | 48.0 | 0 / 0 | 48 / 48 |
| reset mod12 | CCG | 57 / 54 | 55.5 | 9 / 6 | 48 / 48 |
| reset mod12 | Scratch | 54 / 52 | 53.0 | 6 / 4 | 48 / 48 |
| reset mod12 | Pretrained | 49 / 49 | 49.0 | 2 / 3 | 47 / 46 |
| affine mod12 | CCG | 33 / 43 | 38.0 | 2 / 5 | 31 / 38 |
| affine mod12 | Scratch | 20 / 0 | 10.0 | 1 / 0 | 19 / 0 |
| affine mod12 | Pretrained | 12 / 9 | 10.5 | 0 / 0 | 12 / 9 |
| ternary-reset mod12 | CCG | 53 / 52 | 52.5 | 7 / 4 | 46 / 48 |
| ternary-reset mod12 | Scratch | 45 / 27 | 36.0 | 0 / 0 | 45 / 27 |
| ternary-reset mod12 | Pretrained | 44 / 43 | 43.5 | 0 / 0 | 44 / 43 |
| ternary-weighted mod12 | CCG | 43 / 43 | 43.0 | 2 / 0 | 41 / 43 |
| ternary-weighted mod12 | Scratch | 3 / 17 | 10.0 | 0 / 0 | 3 / 17 |
| ternary-weighted mod12 | Pretrained | 0 / 0 | 0.0 | 0 / 0 | 0 / 0 |

Ham kanıt: `runs/transformer_final_independent_727272.json`, `runs/transformer_final_independent_727273.json` ve `runs/transformer_final_independent.log`.

### Auto seçimleri ve sınıf çöküşü

- Counter12/16/24: iki final tohumda da `causal_summary` seçildi.
- Reset: `baseline`; affine: `baseline`.
- Ternary-reset: ilk tohumda `causal_summary`, ikinci tohumda `rotation`.
- Weighted: iki tohumda da `rotation`.
- Final koşuların hiçbirinde zero-recall coverage rescue gerekmedi. Ayrı yeniden-üretimde (`data_seed=515152`, doğru `model_seed=424249`) summary pozitif recall 0, rotation 0.333 ölçüldü; rescue rotation seçti. Held-out toplam 41 → 43 oldu fakat pozitif sonuç 0/16 olarak kaldı. Bu, rescue’nun genellemeyi kanıtlamadığının açık kaydıdır.

## 3. CCG’deki teknik değişiklikler

### Train-underfit erken durma düzeltmesi

Eski kural validation skoru bayat ve patience dolu olduğunda train henüz tam öğrenilmemiş olsa bile durabiliyordu. Yeni kural:

- train ve validation tamamsa dur;
- patience yalnız train tamamsa durdurur;
- train tam değilse kalan bütçe kullanılır.

Bu, patience değerini büyütmekten farklıdır: yarım fit durumunu erken-durma kararıyla gizlemez. Final counter24 adaylarında baseline 48–50/64 train’de kalırken linear-summary 64/64 train ve 16/16 validation’a ulaştı.

### Learned rotation

Rotation her token için öğrenilen bir açı artışı kullanır:

```text
theta_t = theta_(t-1) + W x_t
z_t = [sin(theta_t), cos(theta_t)]
logit_t = GRU_readout(h_t) + rotation_readout([h_t, z_t])
```

Token ağırlıkları veya modulus koda yazılmaz. Periyodiklik açık bir indüktif önyargıdır; açılar ve okuma gradyanla öğrenilir. Son katman sıfır başlangıçlıdır, böylece eski GRU başlangıcı korunur.

Auto önce eski baseline/linear-summary seçimini yapar. Rotation yalnız validation exact daha yüksek ve makro sınıf doğruluğu gerilemiyorsa, ya da exact eşitken çoklu-token ve makro kazanç varsa seçilir. Son kontrol, seçilen model bir validation sınıfında sıfır recall’a sahipken başka aday tüm sınıfları en az bir kez doğru yapabiliyorsa zero-recall modelini eler. Held-out seçimde kullanılmaz.

### Sınıf bazlı ölçüm

Fit raporu artık train/validation sınıf token doğruluğunu ve sınıf token sayılarını içerir. Böylece 11/16 exact fakat pozitif recall 0 olan bir model, yalnız toplam exact ile başarılı görünmez.

## 4. A142 ilişkisel önerici

### Değişiklik

Nesne skoruna sıfır başlangıçlı, öğrenilen bir artık eklendi. Beş genel kosinüs benzerliği kullanır:

- aday nesne ↔ query;
- aday nesne ↔ ilk dayanağın subject/object uçları;
- aday nesne ↔ ikinci dayanağın subject/object uçları.

Bunlar kavram etiketi veya beklenen cevap değildir. Nonce adları altında taşınabilen işaretçi özellikleridir; skor ağırlıkları düzeltmelerden öğrenilir. Sembolik graf yalnız bağımsız doğrulayıcıdır.

### Fresh eşleştirilmiş sonuç

Eğitim tohumları `1001/1002/1003/1004`; sınav tohumları `20182/20283/20384`. Her model 300 düzeltme aldı.

| Metrik | Eski nesne işaretçisi | Uç-benzerlik işaretçisi |
|---|---:|---:|
| Sorgu top-1 | 54/136 | **115/136** |
| Doğru-dayanaklı top-1 | 50/136 | **109/136** |
| Sorgu top-5 | 128/136 | **134/136** |
| Doğru-dayanaklı top-5 | 113/136 | **124/136** |
| Object-top-1 | 59/136 | **136/136** |

| Eğitim tohumu | Eski top-1 /34 | Yeni top-1 /34 | Yeni grounded /34 |
|---|---:|---:|---:|
| 1001 | 14 | **30** | 29 |
| 1002 | 17 | **32** | 30 |
| 1003 | 12 | **20** | 18 |
| 1004 | 11 | **33** | 32 |

Ham kanıt: `runs/relational_similarity_confirmation.json/.log`.

### Görev tasarımı ile öğrenme varyansının ayrılması

- Önceki beş-tohum matrisinde 44 sorgunun yalnız 3’ü çoklu geçerli hedef içeriyordu; sorgu-başı top-1 ile eski hedef-top-1 sayıları aynıydı. Çoklu-hedef metriği sorunu gerçek bir ölçüm riski olmakla birlikte bu matristeki ana varyansı açıklamıyordu.
- Eski modelde eğitim tohumuna göre top-1 11–19/44 arasındaydı. Sınav tohumu etkisi de vardı; dolayısıyla tek bir yeşil A142 koşusu yeterli kanıt değildir.
- Bileşen tanısı relation-top-1’i güçlü, pair sıralamasını ikincil darboğaz, object pointer’ı ana darboğaz gösterdi. Yeni özellik bu darboğazı hedefledi; başarı yalnız toplam skora değil object-top-1’e de yansıdı.

## 5. Reddedilen deneyler

| Deney | Sonuç | Karar |
|---|---|---|
| Nesne elementwise interaction | Fresh top-1 58 → 59/144; object-top-1 49 → 49 | Varsayılana alınmadı, kod kaldırıldı |
| Listwise candidate ranking loss | Aynı üç tohumda eski 18/19/16, yeni 18/16/15 top-1 | Geriledi, kod kaldırıldı |
| Grounded-bootstrap semantiği | Beş tohumda top-1 76 → 62; grounded 71 → 57 | Geri alındı |
| Object loss weight 2/4 | Weight2 toplamı çözmedi; weight4 top-1 65 → 50/188 | Kör ağırlık artırımı reddedildi |
| Phase/additive/prefix CCG | Counter24 exact validation’ı yükseltmedi; weighted pozitif recall çoğunlukla 0 kaldı | Üretim adayı yapılmadı, kod kaldırıldı |
| 300→600→1200→2400 A142 düzeltme | Ortalama top-1 65/188 → 56 → 54 → 59 | Daha uzun öğretim tek başına çözüm değil |

Başarısız deneylerin ham günlükleri `runs/relational_*.log` ve `runs/ccg_*.log` altındadır.

## 6. Sınırlar

- Yedi sentetik dizi ailesi ve dört düğümlü nonce grafikler genel zekâ veya geniş dünya modeli kanıtı değildir.
- `nora_world.py` 2D sandbox ve A143 sınırlı yetenek envanteri bu raporun kapsamı dışındadır; gerçek dünya modeli diye sunulmaz.
- Weighted ve bazı ternary-reset splitlerinde pozitif genelleme hâlâ zayıf veya sıfırdır.
- Rotation periyodik, nesne işaretçisi benzerlik tabanlı tasarlanmış indüktif önyargılardır; “hiç yapı priors yok” iddiası yoktur. Beklenen cevaplar veya modulus/token ağırlıkları elle yazılmaz.
- Pretrained MiniGPT’nin ek ön-eğitim bütçesi scratch/CCG’den farklıdır. Sonuçlar hesap, veri ve FLOP eşitliği değil, açık örnek-bütçe üst sınırıyla okunmalıdır.
- Geliştirme tohumlarında varyant seçimi yapılan eski koşular final bağımsız kanıt olarak kullanılmamıştır.

## 7. Yeniden üretim

```bash
cd /home/user/Nora-NCC
PYTHONPATH=/home/user/Nora-NCC .venv/bin/python - <<'PY'
import json, torch
from law_battery import transformer_representation_comparison
torch.set_num_threads(2)
for seed, data_seed in ((717171, 727272), (717172, 727273)):
    report = transformer_representation_comparison(
        seed=seed,
        data_seed=data_seed,
        pretrained_checkpoint=".cache/baselines/pretrained-mini-gpt/checkpoints/tiny_gpt.pt",
    )
    with open(f"runs/reproduce_{data_seed}.json", "w") as handle:
        json.dump(report, handle, indent=2)
    assert report["measurement_complete"]
PY
```

A142 için `NeuralRelationalProposer(seed=..., object_similarity=False/True)`, `bootstrap_relational_proposer(..., examples=300)` ve `measure_relational_transfer(..., graphs=40)` yukarıdaki tohumlarla eşleştirilmiş çalıştırılır.

## 8. Doğrulama

- Eski CCG baseline/summary fixture prediction roundtrip: geçti.
- Yeni rotation save/load prediction roundtrip: geçti.
- Eski relational checkpoint’in yeni varsayılan modele strict yüklenmesi: geçti.
- Hedefli regresyon: A73, A137, A140, A141, A142, A143 geçti.
- Tam `law_battery.py`: **15/15, 35,2 sn** (`runs/law_battery_after_learning.log`).
- Tam `agi_battery.py`: **140/140, 1610,6 sn** (`runs/full_battery_after_learning.log`).
