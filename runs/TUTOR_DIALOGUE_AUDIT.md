# Nora öğretici diyalog ve nöral-aday denetimi — 30 Ekim 2026

## Sonuç

**Bu eşik aşıldı:** Nora'nın öğrenilmiş grafik bilgisini durumlu bir öğrenci–öğretmen oturumunda kullanması ve gerçek `TokenMoEOrgan` adaylarını grafik/morfoloji denetiminden geçirerek konuşmaya katması sağlandı. Son ürün kodunun birleşik SHA-256 değeri: `62ec753c419e8dce7972e2d542a56925cce125229f39d41f3f514edac7b8f095`.

- **30/30 tur:** açıklama, takip, örnek, karşılaştırma, iki farklı quiz, değerlendirme, öğrenci özeti, yanılgı düzeltme, dört maddeli özet ve doğal kapanış.
- **Bağımsız oracle:** 58/58 ilişki izi doğrudan kenar veya bağımsız ISA-BFS ile kanıtlı; 13/13 genel kontrol geçti.
- **Aritmetik oracle:** `2+3=5` iki ayrı örnekte ve öğrencinin verdiği `4+7=11` bağımsız Python toplamıyla doğrulandı.
- **Gerçek MoE:** `seen_tokens=41.234.454`; son koşuda 57 token dizisi üretildi, 3'ü doğrulanıp kullanıldı, 54'ü reddedildi; 8 turda nöral metin ve 22 önbellek kullanımı var.
- **Bağımsız paraphrase confirmation:** taze V4 seti 15 tur ve **19/19 kontrol**; yeni `12+4=16` örneği geçti.
- **Kalıcı regresyon:** A144 **27/27**, A73 **8/8**, law battery **15/15 (35,4 sn)**.
- **Tam AGI pili:** **141/141 (932,8 sn)**; kırmızı test yok.

Kanıt dosyaları:

- Diyalog: `runs/tutor_dialogue_final.log`, `runs/tutor_dialogue_final.json`
- Yeniden üretim: `runs/tutor_dialogue_final.py`
- Bağımsız oracle: `runs/tutor_dialogue_oracle.py`, `runs/tutor_dialogue_oracle.json`
- Taze paraphrase confirmation: `runs/tutor_paraphrase_confirmation_v4.py`, `runs/tutor_paraphrase_confirmation_v4.json`
- Dış kaynak/kapsam defteri: `runs/tutor_dialogue_source_audit.json`
- Regresyon özeti: `runs/tutor_regression_summary.json`

## Nöral aday → grafik doğrulama yolu

1. `TokenMoEOrgan.generate`, yalnız grafik öznesi/nesnesi çapaları ve ilişkinin işlev sözcükleriyle token üretir.
2. Aday grafik öznesiyle başlamalıdır; nesne köklerinin kapsamı yalnız **özne sonrası yüklemde** ölçülür.
3. `USED_FOR` için “için kullanılır” bağı; `ISA` için geçerli ek-fiil ve mevcut Türkçe morfolojisiyle üretilen son biçim zorunludur.
4. Grafik kökleri dışında içerik, tek harfli apostrof gürültüsü, soru/ünlem veya ters özne-nesne yönü varsa aday reddedilir.
5. Kabul edilmeyen aday konuşmaya çıkmaz; mevcut kompozisyonel grafik anlatımı kullanılır.

Bu nedenle sonuç “bütün metni MoE yazdı” iddiası değildir. Ölçülen sonuç, **MoE'nin gerçekten ürettiği uygun adayların öğretici akışa girdiği**, uygun olmayan üretimin ise grafik ve dil denetimiyle engellendiğidir. `CAN_DO` yetenek kapsamı kompozisyonel “öğrenebilir” biçiminde kalır.

## Sohbette kanıtlanan işlevler

- Konu açma ve grafikten temel açıklama
- Önceki odağı taşıyan takip soruları
- Öğrencinin verdiği sayılarla yeni örnek
- Algoritma/yazılım, AI/yazılım, internet/bilgisayar ve kuantum/klasik bilgisayar karşılaştırmaları
- “Bence ikisi aynı” yanılgısını öğrenilmiş ilişkilerle düzeltme
- Aynı kazanımları yinelemeyen iki quiz ve cevap değerlendirme
- Öğrenci özetini değerlendirme; nesnede geçen kavramı haksız biçimde ayrı özet konusu saymama
- Kesinlik sınırını söyleme ve oturumu doğal biçimde kapatma
- “aynı → ay” gibi meta-gövde konu sıçramasını ve bağlamsız “beyin öğrenebilir” eklemesini engelleme

Öğretici yüzeyler `PriorField` yarışındaki **44 genel çerçeveden** gelir; `nora_priors.json` içinde konu-cevabı çiftleri yoktur. `nora_deliberation.py` denetiminde yalnız 5 genel söylem kalıbı kalmıştır; A73'ün tarihsel mandalları yükseltilmemiştir.

## Başarısız denemeler ve bağımsızlık

Hiçbir kırmızı paraphrase turu başarı sayılmadı:

- **V1:** “örneği” morfolojisi, açık karşılaştırma başlığı ve “Kuantum bilgisayar birdır” sahte-adayı sorunlarını gösterdi.
- **V2:** “aynı” yüzeyinden `ay` konu sıçramasını ve bağlamsız “beyin öğrenebilir” eklemesini yakaladı.
- **V3:** öğrenci özetindeki nesne kavramının ayrı özet konusu sayılması ve “kesin bildiklerini” parafrazının tanınmaması sorunlarını gösterdi.

İlk hatalı JSON'lar `runs/tutor_paraphrase_confirmation_failed.json`, `runs/tutor_paraphrase_confirmation_recheck_failed.json`, `runs/tutor_paraphrase_confirmation_v2_failed.json` ve `runs/tutor_paraphrase_confirmation_v3_failed.json` olarak korunur. V1–V3 düzeltmede kullanıldığı için genelleme kanıtı değildir. **V4 hiç kullanılmamış sorularla bir kez çalıştırılmış ve yeşil kalmıştır.**

## Bağımsız dış kaynak eşleşmesi

| İddia | Denetim | Kaynak |
|---|---|---|
| Bilgisayar bilgi işleyen/hesaplayan makinedir | Desteklendi | [Britannica — Computer](https://www.britannica.com/technology/computer) |
| Algoritma sonlu adımlı problem çözme prosedürüdür | Desteklendi | [Britannica — Algorithm](https://www.britannica.com/science/algorithm) |
| Veri, model öğrenmesinde kullanılır | Makine öğrenmesi kapsamıyla desteklendi | [IBM — Training data](https://www.ibm.com/think/topics/training-data) |
| AI modeli programdır; öğrenen sistemler öğrenebilir | AI modeli ve öğrenen sistemler kapsamıyla desteklendi | [IBM — AI model](https://www.ibm.com/think/topics/ai-model), [IBM — AI](https://www.ibm.com/think/topics/artificial-intelligence) |
| İnternet ağlar ağı ve bilgi altyapısıdır | Desteklendi | [Internet Society — How it works](https://www.internetsociety.org/internet/how-it-works/) |
| Kuantum bilgisayar hesaplama aygıtıdır | Üst düzey taksonomiyle desteklendi; QPU ayrıntısı iddia edilmedi | [Britannica — Quantum computer](https://www.britannica.com/technology/quantum-computer), [IBM Quantum context](https://quantum.cloud.ibm.com/learning/en/courses/use-a-qc-today/quantum-computing-context) |

“Yazılım bir programdır” cümlesi tekil/giriş düzeyi kullanım olarak kapsamlıdır; yazılımın tamamının tek program olduğu genellemesi yapılmaz. Nora'nın “öğrenebilir” ifadesi, “öğrenme kaydı olan sistemlerle sınırlıyorum” cümlesiyle bütün AI sistemlerine genellemez. Kaynaklar yalnız dış doğruluk denetimi içindir; diyalog sırasında Nora'ya yeni bilgi verilmemiştir.

## Ölçüm tanımları

- `graph_edges=590`
- `concepts=603`: kenarlara katılan özne ve nesne kavramlarının birleşimi
- `subject_concepts=301`: en az bir çıkan kenarı olan kavram
- `registered_concepts=1085`: toplam kayıtlı kavram
- 30 `converse` çağrısının toplamı son koşuda **1,67 sn**; başlangıç `besle → attach_broca → step_life(1500)` ayrıca çalıştırılmıştır.

## Sonraki eşik

Öğretici diyalog ve nöral-aday doğrulama eşiği kapandı. Sıradaki teknik eşik, aynı görev/split/bütçe ve çoklu tohumla Transformer karşılaştırmasını koruyarak **fold-1/fold-2 top-1 sonuçlarını %45 üzerine taşımak** ve **0,45B disk-MoE ölçeğini gerçek eğitilmiş uzmanlarla birleştirmek**tir.
