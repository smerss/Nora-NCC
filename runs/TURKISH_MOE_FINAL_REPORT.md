# Türkçe next-subword ve diskteki uzmanlar — nihai dondurulmuş sonuç

Tarih: 2026-10-30. Ölçüm, aynı subword tokenizer ve çapasız next-subword protokolüyle yapıldı.
Birincil hedef A140/fold-0 development metriğiydi; dış-alan ve yeni FAQ sonuçları bu metriğin yerine
kullanılmadı.

## 1. Aşılan eşik ve bağımsız sonuçlar

| Veri | Model | Token | Top-1 | Top-5 | PPL |
|---|---|---:|---:|---:|---:|
| A140 fold-0 | taban `data/broca_tr.pt` | 393 | 117 = %29,7710 | %35,6234 | 386,1220 |
| A140 fold-0 | saf position-149 + wide-147 | 393 | 178 = %45,2926 | %50,1272 | 88,8259 |
| A140 fold-0 | haricî uzman | 393 | 128 = %32,5700 | %39,6947 | 172,8854 |
| A140 fold-0 | **nihai token-MoE** | 393 | **177 = %45,0382** | %49,3639 | 89,5512 |
| XQuAD-v2 | taban | 37.097 | 3.345 = %9,0169 | %16,7102 | 5.951,4498 |
| XQuAD-v2 | haricî uzman | 37.097 | 4.197 = %11,3136 | %21,7457 | 978,2862 |
| XQuAD-v2 | **nihai token-MoE** | 37.097 | **3.548 = %9,5641** | %18,3842 | 2.577,8992 |
| Yeni FAQ confirmation | taban | 199.307 | 17.647 = %8,8542 | %14,9453 | 7.933,1603 |
| Yeni FAQ confirmation | saf hedef ensemble | 199.307 | 15.634 = %7,8442 | %13,4346 | 10.335,2345 |
| Yeni FAQ confirmation | haricî uzman | 199.307 | 22.174 = %11,1256 | %19,5156 | 1.233,0090 |
| Yeni FAQ confirmation | **nihai token-MoE** | 199.307 | **19.586 = %9,8271** | %17,2528 | 3.046,3111 |

Birincil sonuç: tabana göre aynı 393 tokenda **+60 doğru**, +15,2671 yüzde puanı; %45 eşiği aşıldı.
Nihai MoE, saf hedef ensemble'ın dış alandaki gerilemesini her iki bağımsız veri kümesinde de tabanın
üzerine taşıdı. Haricî uzmanın dış-alan katkısı kullanılırken hedef splitte %45 eşiği korundu.

Kanıt dosyaları:

- `runs/moe_router_final_development.json`
- `runs/moe_router_final_xquad_v2.json`
- `runs/turkish_unseen_faq_final_router.json`
- `runs/turkish_unseen_faq_base.json`, `runs/turkish_unseen_faq_external.json`,
  `runs/turkish_unseen_faq_pair2_ensemble.json`

## 2. Seçim ve confirmation disiplini

- Fold-0 mimari/tohum/ensemble geliştirme splitidir; genelleme kanıtı olarak tek başına sunulmadı.
- Router calibration'ı yalnız fold-0 güvenlik kısıtı (`hit1 >= 177`) ve haricî akışın **son 1.000**
  cümlesiyle yapıldı. Bu son dilim, 3.500 adımlık haricî eğitimin ve ilk 1.000 cümlelik router
  eğitiminin dışındadır.
- FAQ hiçbir seçimde kullanılmadı; her adayın kendi XQuAD ölçümü, o adayın calibration'ı bittikten
  sonra yapıldı. Seçim kuralı: önce haricî calibration top-1, eşitlikte hedef top-1 ve haricî PPL.
  İkinci uzman çifti bu kuralla seçildi: calibration %22,0485; birinci çift %20,4292 idi.
- XQuAD-v2 daha sonra ikincil dış-alan tanısı olarak ölçüldü.
- FAQ kaynağı yalnız nihai onam için indirildi; hiçbir eğitim, router veya eşik seçimine girmedi.
  8.467 cümle ve 199.307 token ölçüldü. Ham wiki/tüm chat ve external-train ile tam-cümle kesişimi
  **0**dır. Kaynak MIT, commit `3ff5a943569883d86a4b63ecd88fbfc13891f9f7`.
- Eski işlenmiş 37.086-token XQuAD dosyası resette kaybolduğu için v2 açıkça yeniden tanımlandı:
  1.230 cümle/37.097 token; taban bu dosyada yeniden ölçüldü. Eski ve yeni XQuAD oranları
  birebir aynı dosya gibi karşılaştırılmadı.

Calibration kanıtı: `runs/turkish_router_calibration_pair2.json`.
Veri kanıtı: `.cache/corpora/processed/provenance.json` ve yeniden üretici
`runs/turkish_corpus_rebuild.py`.

## 3. Gerçek token üretimi ve nedensel router

Nihai yapılandırma:

- position-149: 5.696.256 parametre; wide-147: 8.835.072; haricî uzman: 5.679.616.
- Hedef ensemble ağırlığı `%50 position + %50 wide`.
- Token güven oranı `0,85`.
- `nll_prefix` gate: yalnız ilk üç **gözlenmiş** parçanın teacher-forced NLL farkı;
  eşik `3,7105026245117188`.
- OOD reddi dördüncü hedef parçadan itibaren etkindir. İlk üç hedefin tahmininde ilk üç gerçek
  parçayı kullanmak suretiyle gelecek bilgisi sızdırılmaz.
- Toplam aktif parametre **20.210.944**; checkpoint'lerin toplam boyutu **82.661.136 bayt** (78,83 MiB).

`TokenMoEOrgan`, `nora_broca_net.py` içindedir. Üç uzmandan hazır cümle üretip sıralamaz; her adımda
next-subword dağılımını seçer ve örneklemeyi bu dağılımdan yapar. `NoraAGI.attach_broca` artık tek
`.pt` yanında `data/checkpoints/turkish_moe.json` manifestini de yükler. Çevrimiçi öğrenme yeni metni
nedensel prefix kararıyla hedef/haricî uzmana yazabilir; MoE kaydı uzmanları ayrı disk dosyalarında
ve atomik manifestte saklar.

Çalışma-zamanı kanıtı `runs/turkish_moe_final_runtime_smoke.json`: gerçek token üretimi, sonlu
log-prob ve 20.210.944 aktif parametre doğrulandı.

## 4. Çoklu tohum ve açık negatif sonuçlar

- İki bağımsız eğitilmiş uzman çifti de fold-0'da 178 saf-ensemble ve 177 kalibre MoE sonucuna
  ulaştı; dış calibration ve XQuAD davranışları aynı olmadı. Bu nedenle tek bir yeşil tohum yerine
  seçim kuralı ve her iki sonuç da saklandı (`runs/turkish_router_calibration.json`,
  `runs/turkish_router_calibration_pair2.json`).
- Aynı dağılım cross-fold hâlâ %45'in altındadır. Nihai çiftin epoch'u yeni fold'da seçilmeden
  yapılan ölçümleri: fold-1 **170/401 = %42,3940**, fold-2 **165/419 = %39,3795**.
  Birinci çift de sırasıyla %41,8953 ve %39,8568 verdi. Bu sonuçlar fold-0 genellemesi diye
  sunulmuyor (`runs/turkish_ensemble_pair2_crossfold.json`, `runs/turkish_ensemble178_crossfold.json`).
- Saf hedef ensemble FAQ'da tabanın altındadır (%7,8442); başarı yalnız fold-0'a bakılarak ilan
  edilmedi. Haricî uzman ve nedensel router bu açığı kapattı.
- Dropout açık eski 177 ensemble kaydı geçersizdir; hiçbir nihai tabloya alınmadı.

## 5. Transformer/GRU karşılaştırmasının bütçe ayrımı

Aynı fold-0 split, 20 epoch, full-softmax ve scratch başlangıçla yapılan kontrolde GRU üç tohumda
en iyi 81/81/81, 5,17M parametreli Transformer ise 118/117/117 verdi. Bu, aynı-bütçe scratch A/B
kanıtıdır (`runs/turkish_transformer_comparison.json`). Nihai 177 sonucu ise 20M-token önceden
eğitilmiş tabandan fine-tune edilen üç uzmanlı sistemdir; scratch Transformer ile aynı başlangıç veya
parametre bütçesindeymiş gibi sunulmaz. Mimari üstünlük iddiası yalnız aynı-bütçe scratch A/B'ye,
177/393 sonucu ise ayrı fine-tuned sistem ölçümüne aittir.

## 6. Süre, bellek ve çıkarım maliyeti

- Haricî uzmanın tam 3.500 adımı: **814,89 sn**, 1.302.178 yeni token; development tarafından seçilen
  1.000-adım checkpoint'ine ulaşma süresi **235,70 sn**.
- Fold-0 nihai MoE değerlendirmesi: **1,65 sn**. XQuAD-v2: **37,89 sn**. Tam FAQ, batch-64:
  **232,41 sn**.
- Aynı 256 FAQ cümlesi/5.753 token üzerinde batch-16 ve batch-64 aynı hit sayılarını verdi.
  Batch-16 RSS **1.214,94 MiB**, batch-64 RSS **1.866,26 MiB**; süre 7,24 sn / 7,96 sn.
  Bellek veri büyüklüğüyle değil sabit batch ve aktif uzman sayısıyla sınırlıdır
  (`runs/turkish_router_resource_batch16.json`, `runs/turkish_router_resource_batch64.json`).
- Kalite ölçülen bu sistem 20,21M aktif parametredir. Önceki **451.887.104 toplam / 7.060.736 aktif**
  disk-MoE stresi ayrı bir kaynak testidir; onun kalite kanıtı olmadığı ve nihai kaliteyi 0,45B'ye
  bağlamadığı açıkça korunur (`runs/moe_scale_stress.json`).

## 7. Kalıcı üretim dosyaları

- `data/checkpoints/turkish_external.pt`
- `data/checkpoints/turkish_position149.pt`
- `data/checkpoints/turkish_wide147.pt`
- `data/checkpoints/turkish_router_final.pt`
- `data/checkpoints/turkish_moe.json`

Çalıştırma: `NoraAGI.attach_broca("data/checkpoints/turkish_moe.json")`.

## 8. Nihai işlevsel regresyon

- `agi_battery.py`: **140/140**, 936,4 sn; başarısız ölçüt yok.
- `law_battery.py`: **15/15**, 35,9 sn.
- A41 kodda Türkçe sabit denetimini sıfır ihlalle, A42 dengeli Türkçe araştırma akışını ve A73
  iskelet mandalını **8/8** geçti. A73 toplamı **1.124 ≤ 1.125**, söylem toplamı **126 ≤ 133**.
- Ara koşularda yakalanan A41 ve A73 hataları kaynak kodda giderildi; test, baseline veya tarihsel
  tavan gevşetilmedi. Yukarıdaki sonuçlar düzeltmelerden sonra sıfırdan çalıştırılan son koddan alındı.

## 9. Canlı diyalog takip düzeltmesi

Dondurulmuş MoE'nin ardından yapılan gerçek kullanıcı sohbeti üç yönlendirme hatasını görünür kıldı:
“şu sıralar” sıralama komutuna, “bunu” bilinmeyen nesneye, “beni dinlerken” de dış konuya
dönüşüyordu. Kaynak kodda komut sınırı, zamirli takip ve kişi/fiil rolü düzeltildi; hazır diyalog
cevabı eklenmedi.

Son canlı dört tur: “Merhaba, nasılsın?” → “Seni düşünüyorum.” → “Seni merak ettim.” →
“İçim dingin.” Yanıtlar önceki turu takip etti; alakasız sıralama ve “Beni görmedim” çıktısı kalktı.

- A75 **14/14**: gerçek sıralama korunur, zaman ifadesi komut sayılmaz.
- A139 **17/17**: düşünce → zamirli takip zinciri ve iki kişi-rolü ayrımı.
- A73 **8/8**, law battery **15/15**.
- Bu düzeltmenin tam battery koşusu **139/140** oldu (964,1 sn). Tek kırmızı A6 idi; tam koşudaki
  temkinlilik farkı -0,08'di. A6 üç ayrı izole koşuda **3/3** geçti ve fark her seferinde +0,13'tü.
  Tam koşu bu nedenle “140/140” diye raporlanmaz; yaşantı varyansı açıkça korunur.

## 10. Sonraki eşik

**Bu turda aşılan eşik:** fold-0 top-1 > %45 ve aynı dondurulmuş MoE ile iki dış veri kümesinde
taban üstü. **Sıradaki eşik:** fold-1/2 dahil aynı-dağılım çapraz-fold sonuçlarını %45'e taşımak;
ardından 0,45B disk-MoE stresini aynı kaliteli, gerçek eğitilmiş uzmanlarla birleştirmek.
