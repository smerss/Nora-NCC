# Türkçe tek-sunum öğrenme ve korunum — geliştirme raporu

Tarih: 2026-10-31

## Karar

**Ortak kabul eşiği henüz aşılmadı.** Aynı adayda fold-1 için en az 181/401, fold-2 için en az 189/419 top-1 ve geriye-dönük unutma testinde kabul edilebilir korunum bulunmuyor. Epoch/replay ile elde edilen eski 181/174 gibi sonuçlar bu raporda başarı sayılmaz.

Bütün varyant seçimleri fold-1/2 development üzerindedir. Ortak aday bulunmadığı için fold-3/4 confirmation başlatılmadı; buradan bağımsız genelleme sonucu çıkarılmaz.

## Öne çıkan ölçümler

| Tek-sunum kol | Fold-1 | Fold-2 | Korunum ölçümü | Karar |
|---|---:|---:|---|---|
| Full-pass taban + full Adam, LR 0,001 | 164/401 | 174/419 | 128-cümle probe: −95/−78 hit1 | Kalite ve korunum yetersiz |
| Clean1000 + sampled, batch 1, K 512 | 160/401 | 173/419 | 128-cümle probe: −63/−51 hit1 | Kalite ve korunum yetersiz |
| Full-pass taban + donuk 1024-merkez RBF | 166/401 | 161/419 | Global okuma değil; ortak kalite yok | Çözüm değil |
| Uniform sampled + taze haricî akış | 158/401 | 160/419 | Tam probe aşağıda | En iyi korunum noktası, kalite yetersiz |
| Sabit 2048-KC `ChemMushroomBody` | 159/401 | 159/419 | Hızlı sinaps, replay yok | Çözüm değil |

### Tam geriye-dönük unutma denetimi

Dondurulmuş development tarifi: clean1000 taban; her hedef cümle batch=1 ve yalnız bir kez, uniform sampled K=2048, hedef LR=0,0005; her hedefin ardından daha önce okunmamış 12 haricî cümle, full loss, LR=0,0001. Hedef cycle/replay yok.

| Fold | Held top-1 | Chatbot hit1 değişimi | Chatbot PPL değişimi | XQuAD hit1 değişimi | XQuAD PPL değişimi |
|---|---:|---:|---:|---:|---:|
| 1 | 158/401 | −569 / 154.678 | +15,266 | −28 / 37.097 | +50,328 |
| 2 | 160/419 | −575 / 154.678 | +15,223 | −49 / 37.097 | +62,770 |

Kayıp sınırlıdır fakat sıfır değildir. Bu kol korunumu çözülmüş veya %45 adayı olarak etiketlenmez.

Kanıt: `runs/turkish_retention_candidate_audit.json`.

## Haricî tek-geçiş taban

Clean1000 checkpoint’inin okuduğu ilk 32.000 cümle tekrar edilmeden akış tamamlandı:

- 286.656/286.656 haricî cümle, toplam 8.958 adım;
- devam bölümünde 2.965.776 token, hedef satırı 0, exact-skip 0;
- son checkpoint: `runs/models/external_onepass_full_step8958.pt`;
- devam optimizer’ı clean1000 sonrası yeni Adam’dır; bu durum ham kayıtta açıkça yazılıdır.

Final tabanın kendisi fold-1/2’de 126/130 verdi. Daha uzun haricî akış PPL’yi iyileştirdi fakat tek başına top-1 eşiğini çözmedi.

Kanıt: `runs/turkish_external_onepass_full_dev.json`, `runs/turkish_fullbase_onepass_dev.json`.

## Kapatılan tek-sunum kolları

- Tam fine-tune, donuk gömme/GRU, output-only, untied output, SGD/Adam, dropout, LR çizelgeleri ve kelime-başı ağırlığı: ortak eşik yok.
- PA, ridge, kNN-LM, fast-weight ve 256/1024-merkez RBF: ortak kalite/korunum yok.
- Fisher/EWC: düşük ölçek etkisiz; yüksek ölçek korurken öğrenmeyi dondurdu. Yüksek-LR taraması da ortak nokta vermedi. Deneysel önem haritası ana hattan çıkarıldı; oracle ve sonuçlar korundu.
- Taze haricî gradyanla PCGrad/conflict projection: en yüksek 159/162; birleşik-gradyan kontrolü 156/162. Çakışma çıkarma ortak çözüm vermedi.
- Altı bağımsız tek-sunum modeliyle probability ensemble ve weight soup: 159/165; hata korelasyonu yüksek.
- Taze haricî batch’leri karıştırma/dönüşümlü kullanma: korunumu iyileştirdi, hedef top-1’i %45’e taşımadı.
- Chunk’lı online GRU: sıfır-LR tam-forward eşdeğerliği 0,0; öğrenme sonuçları 154/162 ve altı.
- Gerçek k-WTA/`learn_oneshot`: otomatik nörogenezli ilk deneme RSS yaklaşık 2,4 GB’a ulaştığı için durduruldu ve sonuç sayılmadı. Sabit 2048-KC tekrarı tamamlandı; en yüksek 159/159.

Ayrıntılı ham kayıtlar `runs/turkish_onepass_*_dev.json`, `runs/turkish_fullbase_rbf_dev.json`, `runs/turkish_ewc_onepass*_dev.json`, `runs/turkish_rbf*_dev.json` ve `runs/turkish_chem_onepass_fixed_dev.json` içindedir.

## Kod doğrulaması

- EWC deneyinin bağımsız küçük-örnek ceza oracle’ı mutlak hata **4,20e-7** verdi; sonuç kabul edilmediği için deneysel `parameter_importance` kodu ana hattan çıkarıldı. Legacy/seeded varsayılan eğitim regresyonu eşit.
- `nora_chem.py` skor hesabı artık bütün hızlı+yavaş matrisi her token’da kopyalamak yerine yalnız aktif KC sütunlarını birleştirir. Bağımsız eski-yeni skor oracle’ı **0,0**; mevcut tek-atış duman testi **5/5**.
- `py_compile` ve `git diff --check` temiz.

## Sıradaki teknik eşik

Yeni bir LR/epoch/replay grid’i değil; **tek sunumda 181/401 ve 189/419 ile tam chatbot+XQuAD korunumunu aynı adayda** sağlayan öğrenme mekanizması. Ancak böyle ortak bir aday dondurulduktan sonra fold-3/4 confirmation ve çoklu-tohum karşılaştırması anlamlı olacaktır.

## Canlı model ve felsefe notu (2026-10-31)

Bu rapordaki eğitimler ayrı checkpoint kopyalarında yapılmıştır; canlı kullanılan modelin üzerine yazılmamıştır. “Dondurma” yalnız ölçüm snapshot’ıdır ve canlı öğrenmeyi durdurmaz.

Canlı `hear()`/`learn_online()` yolu her cümleyi bir kez işler. Buna karşılık ana next-subword motoru hâlâ GRU/TokenMoE’dir; mantar cismi yardımcı öğrenme yollarındadır. `sleep()` içinde 20–48 replay adımlı konsolidasyon vardır; bu yol tek-sunum skor kanıtı olarak kullanılmamıştır. Dolayısıyla canlı öğrenme iskeleti hedef felsefeyle uyumludur, fakat ana öğrenme motorunun tek-okuyuş genellemesi ve unutmama kanıtı tamamlanmamıştır.

Bütün proje alanlarının birleşik güncel durumu `NORA_GUNCEL_DURUM.md` içindedir.
