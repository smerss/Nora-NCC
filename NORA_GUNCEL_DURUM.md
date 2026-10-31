# Nora — Güncel Teknik ve Araştırma Durumu

**Tarih:** 2026-10-31

**Kapsam:** Canlı model felsefesi, sürekli öğrenme, Türkçe dil modeli, CCG ve ilişkisel öğrenme, diyalog, dış korunum, disk-MoE ve açık teknik sorunlar.

Bu belge güncel ana durum dosyasıdır. Ayrıntılı tarihsel kanıtlar `runs/LEARNED_REPRESENTATION_REPORT.md`, `runs/TURKISH_MOE_FINAL_REPORT.md`, `runs/TURKISH_ONEPASS_CONTINUATION_REPORT.md` ve `runs/TUTOR_DIALOGUE_AUDIT.md` içindedir.

## 1. Kısa karar

Nora canlı çalışan ve yeni girdileri tek sunumda işleyen bir sistemdir. **31 Ekim HDC/MB güncellemesi, dar A/B/C tek-sunum protokolünü ve dinamik ilişki icadı/transfer protokolünü geçti; genel “tek okuyuşta genelleyen ve hiç unutmayan zihin” iddiası ise hâlâ kanıtlanmış değildir.**

- Anlamsal olgu ve ilişki kararları artık HDC/MB aktivasyonuyla adaylandırılıp sembolik grafça doğrulanıyor; grafın ham BFS listesi cevap motoru değildir. GRU/TokenMoE tarafındaki üretim rolü artikülasyondur.
- Canlı öğrenme iskeleti vardır ve deneyler canlı checkpoint’in üzerine yazılmamıştır.
- Yeni C6–C10 sonuçları genel next-subword dil modeli sonucu değildir; genel diyalog ve açık uçlu öğrenme için daha geniş bağımsız sınav gerekir.
- Fold-1/fold-2 için ortak %45 top-1 ile geriye-dönük korunum aynı adayda sağlanmamıştır.
- Fold-3/4 confirmation, ortak development adayı bulunmadığı için başlatılmamıştır.
- Eski epoch/replay sonuçları tek-sunum başarı kabulünü karşılamaz.
- 0,45B disk-MoE yalnız kaynak stresidir; rastgele genişletilmiş uzmanlarla kalite kanıtı değildir.

### 1.1 HDC/MB ve dinamik ilişki güncellemesi

Yeni çekirdek `nora_chem.py::HDCMushroomBody`, her olayı rol–dolgu HDC bağlarıyla bir kez yazar; k-WTA hedefi **%6**dır. Özne ve tam özne–ilişki–nesne erişim anahtarları `W_fast/W_slow` üzerinden okunur. Nörogenezden sonra eski kodun aktivasyonu yeni toplam KC sayısına bölünmez; her olay kendi yazım anındaki aktif-KC normunu kullanır. Bu son nokta, A16’da yakalanan erken-bilgi erişim kaybını düzeltmiştir; aktivasyon eşiği düşürülmemiştir.

| Sınav | Sonuç | Kanıtlanan dar kapsam |
|---|---:|---|
| Mimari C1–C10 | **10/10** | Yayın/ablasyon + yeni MB ve ilişki sınavları |
| C6 A/B/C | **1/1** | Üç öğretim birer sunum; üç soru uyanıkken; uyku ve replay sıfır; üç MB izi |
| C7 yeni ilişki | **1/1** | Probation `YENI_OPERATOR_VORL`; iki held-out transfer; kontrol `None`; ters yön `conflict` |
| C8 beş tohum | **5/5** | MB ablasyonunda cevap kaybı, restorasyonda öğretimsiz dönüş; 40 yeni bilgi ve bakım sonrası korunum; bütün eski bootstrap kenarlarında eksik **0** |
| C9 uyku/kalıcılık | **1/1** | 501 yaşam adımı ve açık uykuda `replay=0`; yasak eski `mind._sleep` çağrısı **0**; save/load sonrası eski bilgi, probation, operatör ve canlı transfer |
| C10 çok-köklü icat | **3 tohum × 5 kök** | 15/15 ayrı operatör ve transfer; bilinen fiil/koşaç yeni operatöre dönüşmedi |
| Dış held-out H1–H9 | **9/9** | Önceki dış işlevler; H7 nedensel zincir 3/3 |
| A142 nöral öneri + sembolik kanıt | **13/13** | Nonce top-5 18/18; dayanaklı kabul 16/17; nöral/doğrulayıcı ablasyonu `unresolved` |
| A140 Broca/corpus | **18/18; corpus 73/73** | Öğrenilen artikülasyon yolu gerilemedi |

A94 **7/7**, A95 **10/10**, A141 **25/25**, A75 **14/14** ve A16 **4/4** da yeniden çalıştırıldı. Tam `agi_battery.py` koşusu 30 dakikalık sınırda A22 sonrasında tamamlanamadı; dolayısıyla bu güncelleme için “bütün işlevsel batarya geçti” denmez.

Bilinmeyen SVO yüklemi artık otomatik ISA/HAS_PROP sınıfına zorlanmaz. Wernicke özne/nesne/yüklem rollerini çıkarır; `GroundedConceptGraph.invent_relation()` yeni ilişkiyi probation olarak yazar. Aynı ilişkinin yeni bir alanda görülmesi `discovery`, daha önce bulunmayan ilişkinin açılması `invention` olarak deftere geçer. Nöral proposer yalnız aday verir; kabul yön/çelişki ve kanıt denetiminden geçer. Operatör koda gömülmez, `nora_forge.py`/`nora_apply.py` defterlerinde ve save/load meta kaydında taşınır.

**Sınır:** C6–C10 belirli, bağımsız oracle’lı mimari protokolleridir. Sınırsız Türkçe ayrıştırma, her olası ilişkinin doğru icadı, bütün alanlarda genelleme veya yıkıcı unutmanın her koşulda imkânsızlığı kanıtlanmamıştır.

## 2. Hedef felsefe ve kabul ölçütleri

1. **Canlı ve sürekli öğrenme:** Yeni bilgi tek sunumda edinilmeli; epoch veya replay ile ezberleme ana çözüm olmamalıdır.
2. **Genelleme:** Ezberlenmiş cümleyi geri almak yerine görülmemiş bağlama ve parafraza aktarım ölçülmelidir.
3. **Geriye-dönük korunum:** Yeni öğrenme önceki bilgiyi yıkıcı biçimde silmemelidir; chatbot ve bağımsız XQuAD probe’ları ayrı ölçülmelidir.
4. **Mantar cismi ilkesi:** Seyrek kodlama, hızlı sinaps, sinaptik sertlik ve sınırlı kaynakla canlı öğrenme araştırılır; sinek/konektom pazarlaması yapılmaz.
5. **Nöral üretim, sembolik doğrulama:** Hazır cevap sıralamak öğrenme kanıtı değildir. Nöral model aday üretir; graf yalnız bağımsız doğrular.
6. **Dürüst karşılaştırma:** Aynı görev/split/bütçe ve çoklu tohum kullanılır. Pretrained sistemle scratch Transformer aynı başlangıç bütçesindeymiş gibi sunulmaz.
7. **Kaynak sınırı:** Veri ve uzmanlar diskte kalır; aktif uzman ve batch sınırlıdır. Veri büyüdükçe bellek büyük ölçüde sabit kalmalıdır.
8. **Öncelik:** Önce Türkçe öğrenme ve korunum; ardından gerçek eğitimli uzmanlarla yaklaşık 0,45B disk-MoE kalite birleşimi.

## 3. “Dondurma” ve “adım” ne demektir?

- **Dondurma:** Canlı modeli kapatmak değildir. Yalnız ölçüm anındaki ağırlıkların değişmeyen bir snapshot kopyası değerlendirilir. Böylece model test sırasında değişmez ve sonuçlar karşılaştırılabilir.
- **Adım:** Yeni veriyle yapılan bir optimizer güncellemesidir. Epoch, tekrar veya canlı öğrenmenin bitirilmesi değildir.
- **Canlı üretim:** Öğrenme sürer. Deney snapshot’ları canlı ağırlıkların yerine geçirilmez.
- **Confirmation:** Development sırasında seçilen tarifin bağımsız fold’larda değişmeden sınanmasıdır; canlı modelin dondurulmuş kalacağı anlamına gelmez.

## 4. Canlı model felsefe denetimi

### Uyumlu olan yapı

- `nora_voice.py::hear` metni cümlelere ayırır ve seçilen cümleleri bir kez `learn_online` yoluna verir; epoch yoktur.
- Canlı turda `neural_limit` varsayılanı 8 cümledir; çevrimdışı dosya okumada `None` ile bütün cümleler tek geçiş işlenebilir.
- `TokenMoEOrgan.learn_online` her cümleyi nedensel prefix kararıyla hedef veya haricî uzmana bir kez yönlendirir.
- `BrocaOrgan.learn_online` küçük öğrenme oranıyla (`3e-4`) tek-geçiş güncelleme yapar.
- Mantar cismi modülleri k-WTA seyrek kod, hızlı/yavaş sinaps ve stiffness mekanizmalarını gerçekten uygular.
- Anlamsal graf sorgusu, nedensel zincir ve analoji adayları HDC/MB’den gelir; MB ablasyonu graf kenarları yerindeyken cevabı kaldırır.
- Yeni bilgi için `learn_oneshot()` gradyan/epoch kullanmaz; çakışan rakip sınıfın ağırlığını silen contrastive güncelleme kaldırılmıştır.
- `sleep()` yalnız sinaptik bakım/homeostaz yapar; eski `mind._sleep(64)`, S2 ve MB replay çağrıları kaldırılmıştır. Ana Nora’da 500-adım periyodik `mind._sleep(48)` de kapalıdır ve eski save/load kaydı bunu yeniden açamaz. Pozitif `replay_steps` açık `ValueError` üretir.

### Tamamlanmamış uyum

- Genel next-subword kalitesi ve fold-1/fold-2 ortak %45 eşiği çözülmemiştir; Broca’nın gradyanlı dil öğrenmesi için bütün dış korunum iddiası hâlâ ayrı bir problemdir.
- C6–C10 dışındaki bütün olası cümle biçimleri ve alanlar için genelleme kanıtı yoktur; intransitif, çok-öbekli ve karmaşık Türkçe yapılar daha geniş held-out sınav ister.
- Tam `agi_battery.py` bu güncellemede sonuna kadar tamamlanmamıştır; yalnız yukarıda adı ve sonucu verilen koşular kanıt sayılır.
- Dolayısıyla dar tek-sunum ilişki protokolü çözülmüş, genel zihin hedefi çözülmemiştir.

## 5. Güncel Türkçe tek-sunum sonucu

Kabul sınırı: fold-1 için en az **181/401**, fold-2 için en az **189/419** top-1 ve aynı adayda kabul edilebilir dış korunum.

| Tek-sunum kol | Fold-1 | Fold-2 | Korunum özeti | Karar |
|---|---:|---:|---|---|
| Full-pass taban + full Adam, LR 0,001 | 164/401 | 174/419 | 128-cümle probe: −95/−78 hit1 | Ortak aday değil |
| Clean1000 + sampled, batch 1, K 512 | 160/401 | 173/419 | 128-cümle probe: −63/−51 hit1 | Ortak aday değil |
| Full-pass taban + donuk 1024-merkez RBF | 166/401 | 161/419 | Ortak kalite yok | Çözüm değil |
| Uniform sampled + taze haricî akış | 158/401 | 160/419 | En iyi ölçülen korunum noktası | Kalite yetersiz |
| Sabit 2048-KC `ChemMushroomBody` | 159/401 | 159/419 | Hızlı sinaps, replay yok | Çözüm değil |
| PCGrad, en iyi kol | 159/401 | 162/419 | 128-cümle probe: −19/−22 hit1 | Çözüm değil |

**Ortak %45 eşiği aşılmamıştır.** En iyi ortak full-pass sonucu 164/174’tür; bu sonucun dış kaybı da vardır.

## 6. Tam geriye-dönük unutma denetimi

En iyi korunumlu development tarifi:

- clean1000 başlangıcı;
- her hedef cümle batch=1 ve bir kez;
- uniform sampled K=2048, hedef LR=0,0005;
- ardından daha önce okunmamış 12 haricî cümle, full loss, LR=0,0001;
- hedef cycle veya replay yok.

| Fold | Held top-1 | Chatbot hit1 değişimi | Chatbot PPL değişimi | XQuAD hit1 değişimi | XQuAD PPL değişimi |
|---|---:|---:|---:|---:|---:|
| 1 | 158/401 | −569 / 154.678 | +15,266 | −28 / 37.097 | +50,328 |
| 2 | 160/419 | −575 / 154.678 | +15,223 | −49 / 37.097 | +62,770 |

Bu, sınırlı fakat sıfır olmayan kayıptır. Tarif “unutmama çözüldü” veya “%45 adayı” diye etiketlenmez.

Canlı kullanılan checkpoint bu deneylerle değiştirilmemiştir; ölçülen kayıplar ayrı deney kopyalarına aittir.

## 7. Haricî tek-geçiş dil akışı

Clean1000 checkpoint’inin okuduğu ilk 32.000 cümle tekrar edilmeden akış tamamlanmıştır:

- **286.656/286.656 haricî cümle**;
- toplam **8.958 adım**;
- devam bölümünde **2.965.776 token**;
- hedef satırı kullanımı **0**, exact-skip **0**;
- devam optimizer’ı clean1000 sonrası yeni Adam’dır;
- final taban fold-1/fold-2’de **126/130** vermiştir.

Daha uzun haricî akış PPL’yi iyileştirmiş, tek başına top-1 eşiğini çözmemiştir.

Kanıt: `runs/turkish_external_onepass_full_dev.json`, `runs/turkish_fullbase_onepass_dev.json`.

## 8. Önceki yüksek skorların durumu

- Replay12: fold-1/2 **181/174**.
- Replay28 tohum çiftleri: ensemble sonuçları yaklaşık **174–179 / 167–175**.
- 16 uzman + kısa-diyalog position: **182/174**.
- Fold-2 %45 sınırı **189/419** olduğundan bu eski sonuçlar da ortak eşiği geçmemiştir.
- Bunlar epoch/replay içerdiği için güncel tek-sunum kabulünün çözümü değildir.

Hiçbir eski skor yeni tek-sunum ve korunum koşulları sağlanmış gibi sunulmaz.

## 9. Kapatılan veya çözüm olmayan kollar

- Tam fine-tune, donuk gömme/GRU, output-only, untied output.
- SGD/Adam, momentum, dropout, LR çizelgeleri, kelime-başı ve frekans ağırlıkları.
- Pasif-agresif çıktı, ridge readout, fast-weight ve kNN-LM.
- 256/1024 merkezli RBF okuma.
- Fisher/EWC: düşük ölçek etkisiz, yüksek ölçek öğrenmeyi dondurdu; yüksek-LR ortak nokta vermedi. Deneysel kod ana hatta bırakılmadı.
- Altı tek-sunum modeliyle ensemble/soup: **159/165**; hata korelasyonu yüksek.
- Taze haricî veriyle mixed/alternating akış: korunumu iyileştirdi, kaliteyi %45’e çıkarmadı.
- Chunk’lı online GRU: sıfır-LR tam-forward eşdeğerliği **0,0**; öğrenme sonucu çözüm olmadı.
- Gerçek k-WTA/`learn_oneshot`: sabit 2048-KC ile en yüksek **159/159**.
- PCGrad/conflict projection: en yüksek **159/162**.

Otomatik nörogenezli ilk mantar-cismi denemesi RSS yaklaşık 2,4 GB’a ulaştığı için durdurulmuş ve sonuç sayılmamıştır. Sabit-KC deneyi ayrı tamamlanmıştır.

## 10. Mantar cismi performans düzeltmesi

`nora_chem.py::ChemMushroomBody.scores` artık her çağrıda bütün `W_slow + W_fast` matrisini kopyalamak yerine yalnız aktif KC sütunlarını birleştirir.

- Bağımsız eski-yeni skor oracle’ı: **0,0 hata**.
- Mevcut tek-atış duman testi: **5/5**.
- Bu değişiklik hesap sonucunu değil, bellek tahsisi ve çalışma maliyetini iyileştirir.
- Türkçe top-1 başarısı olarak sunulmaz.

Kanıt: `runs/turkish_chem_score_oracle.json`.

## 11. CCG ve ilişkisel öğrenme

Ayrıntılar `runs/LEARNED_REPRESENTATION_REPORT.md` içindedir.

- Son bağımsız iki veri/model tohumu toplamında CCG **734/896**, scratch Transformer **566/896**, pretrained MiniGPT **479/896** tam dizi üretmiştir.
- Counter24: CCG **57 ve 64**, scratch **57 ve 56**. Bu koşularda auto seçimi rotation değil linear-summary olmuştur; kazanç rotation’a atfedilmez.
- Weighted açık sorundur: bağımsız CCG **43/64 ve 43/64**, pozitif sonuçlar **2/16 ve 0/16**.
- A142 dört eğitim tohumunda nesne-uç benzerliği top-1’i **54/136 → 115/136**, doğru-dayanaklı top-1’i **50/136 → 109/136** çıkarmıştır.
- Pretrained MiniGPT ek Tiny Shakespeare ön-eğitimi taşır; scratch ile aynı öğrenme geçmişine sahip değildir.

Bu sentetik sonuçlar Türkçe next-subword %45 eşiğinin yerine geçmez.

## 12. Diyalog, yaşantı ve öğretici

Ayrıntılar `runs/TUTOR_DIALOGUE_AUDIT.md` ve `runs/TURKISH_MOE_FINAL_REPORT.md` içindedir.

- Karşıtlıklı yaşantı, parafrazlı geri çağırma, kişi rolleri, belge kapsamı ve doğal anı anlatımı için hedefli regresyonlar eklenmiştir.
- “Seninle konuşmak bana nasıl hissettirdi?” sorusunda Nora kullanıcıyı ve kendisini ayırır; yanıt “Benimle konuşman sana iyi geldi.” biçiminde kurulur.
- Makale/dış anlatı Nora’nın yaşantısı sayılmaz; alıntı kapsamı kişisel olay belleğine yazılmaz.
- Öğretici akışta nöral adaylar sembolik graf kanıtıyla doğrulanır; hazır cevap veya yalnız istatistiksel bellek anlama sayılmaz.
- Son kayıtlı tam işlevsel koşular **140/140**, law battery **15/15** olmuştur. Bunlar ilgili tarihsel kod koşularıdır; bu belge için yeniden tam pil çalıştırıldığı iddia edilmez.
- D2 görünür known-issue’dur; D1 ve D3–D10 dondurulmuş davranış sözleşmeleri korunur.

Akıcı diyalog, tek-sunum next-subword genellemesinin kanıtı değildir.

## 13. 0,45B disk-MoE stresi

- Toplam parametre: **451.887.104**.
- Aktif parametre: **7.060.736**.
- Maksimum RSS: **884.867.072 bayt**.
- Scratch GRU üç tohum: **81/81/81**.
- 5,17M Transformer: **118/117/117**.

Bu stres diskte büyük model ve sınırlı aktif uzman çalışmasını gösterir. Rastgele genişletilmiş checkpoint’lerle yapıldığı için kalite veya öğrenme kanıtı değildir. Sonraki aşamada gerçek eğitim geçmişi ve kalite ölçümü olan uzmanlar gerekir.

## 14. Veri ve bağımsız kanıt düzeni

- Decontamine corpus: **170.457 wiki + 116.199 chat** cümlesi.
- Target exact-match: **0/286.656**.
- Dış korunum: `chatbot_confirmation.jsonl` ve `xquad_rebuilt_v2.jsonl`.
- Kaynak commit/hash ve split rolleri `.cache/corpora/processed/provenance.json` içinde tutulur; önbellek Git’e eklenmez.
- Fold-1/2 varyant seçimi development’tır; bağımsız genelleme iddiası için dondurulmuş tarifin fold-3/4 confirmation’ı gerekir.
- Olumsuz sonuçlar da JSON ve raporlarda saklanır.

## 15. Doğrulama durumu

Son tek-sunum çalışma düğümünde:

- legacy/seeded varsayılan eğitim regresyonu eşit;
- EWC küçük-örnek ceza oracle’ı mutlak hata **4,20e-7**; deney kabul edilmediği için kod ana hattan çıkarıldı;
- PCGrad küçük-vektör oracle’ı geçti;
- chunk’lı forward eşdeğerliği **0,0**;
- mantar-cismi skor oracle’ı **0,0** ve duman testi **5/5**;
- `py_compile` ve `git diff --check` temiz;
- çalışan deney süreci yok.

Bunlar ilgili hedefli kontrollerdir; tam AGI/law battery’nin her yeni deneyden sonra tekrarlandığı anlamına gelmez.

## 16. Sıradaki teknik eşikler

1. **Aynı tek-sunum adayda** fold-1 ≥181/401, fold-2 ≥189/419 ve tam chatbot+XQuAD korunumu.
2. Dondurulmuş tarifle fold-3/4 confirmation; epoch, tohum, mimari veya ağırlık seçimi yapılmadan.
3. Çoklu tohumda A142 varyansı ve görev zorluğunu ayrı raporlamak.
4. Ana next-subword öğrenmesini mantar cismi ilkelerine yaklaştırırken canlı öğrenmeyi ve önceki bilgiyi korumak.
5. Gerçek eğitimli uzmanlarla 0,45B disk-MoE kalite ve kaynak birleşimi.
6. Weighted ve D2 gibi açık sorunları ayrı tutmak; başarılı alanlarla örtmemek.

## 17. Kanıt haritası

- Ana tek-sunum raporu: `runs/TURKISH_ONEPASS_CONTINUATION_REPORT.md`
- Türkçe MoE tarihsel raporu: `runs/TURKISH_MOE_FINAL_REPORT.md`
- CCG/Transformer/A142: `runs/LEARNED_REPRESENTATION_REPORT.md`
- Diyalog ve öğretici: `runs/TUTOR_DIALOGUE_AUDIT.md`
- Tam korunum: `runs/turkish_retention_candidate_audit.json`
- Full-pass taban: `runs/turkish_external_onepass_full_dev.json`, `runs/turkish_fullbase_onepass_dev.json`
- PCGrad: `runs/turkish_pcgrad_onepass_dev.json`
- Gerçek mantar-cismi deneyi: `runs/turkish_chem_onepass_fixed_dev.json`
- Skor oracle’ı: `runs/turkish_chem_score_oracle.json`
- Diğer olumsuz kollar: `runs/turkish_onepass_*`, `runs/turkish_*_dev.json`

Yeni büyük deney checkpoint’leri, corpus önbelleği, loglar ve geçici dosyalar Git’e eklenmez; önceki `data/checkpoints/` referans uzmanları kendi model commit’lerinde bulunur. Küçük kanıt JSON’ları ve repo içindeki reprodüksiyon script’leri korunur. Bazı erken `/tmp` pilot script’leri kalıcı çalışma ağacında bulunmadığından onların ana tarifleri ve sonuçları MD raporlarında açıklanmıştır.
