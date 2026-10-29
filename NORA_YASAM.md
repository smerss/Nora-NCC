# Nora: arka plan yaşamı, öğrenen söz, eksik tamamlama

**Konumlandırma:** Nora, mantar cismi ilkesiyle çalışan biyolojik esinli bir zihin.

İlke: altyapıyı kurmak, içeriği Nora'nın yaşayarak/okuyarak edinmesine bırakmak. Hata düzeltmek için hedefe özel
cevap kalıbı eklenmez. Bununla birlikte mevcut sistemde elle yazılmış ontoloji etiketleri, Türkçe sözlük/çekim
öncülleri, eşikler ve bilişsel iskelet vardır; bunlar **bootstrap ve mühendislik bağımlılığıdır**, kendi kendine
öğrenme veya bilinç kanıtı değildir.

| dosya | rol |
|---|---|
| `nora_voice.py` | `LearnedVoice`: söz organının çerçevesi. Alt-sözcük (BPE) ayrıştırıcıyı veriden öğrenir; sözcük–iç durum çağrışımı ve n-gram geçişleri yalnızca duyulan metinden gelir. Dil duyulmadıysa sessizdir. Sinirsel organ bağlıysa söz onunla kurulur, n-gram yedektir. |
| `nora_broca_net.py` | `BrocaOrgan`: küçük (5,7M) GRU artikülasyon motoru; tek geçiş ve çevrimiçi küçük adımlarla öğrenir. A142 hattında ayrıca kapsamı sınırlı **nöral ilişki önerici** serbest top-5 aday üretir; sembolik graf kanıt bulmadan bilgi yazılmaz. Bu, genel serbest yanıt veya tam öğrenilmiş zekâ değildir. |
| `make_corpus_dump.py`, `nora_egit.py` | Veri ve eğitim: Vikipedi akışı + Tatoeba konuşma cümleleri; `tokenizer / oku / birlestir / broca / kisa` aşamaları. |
| `nora_life.py` | `BackgroundLife`: arka plan döngüsü (`step_life` içinde döner). A22 CA3 tarama + özet, zihinsel gerilim (RK4), 5 popülasyonlu LCA ihalesi (dinlen / pekiştir / prova / araştır / seslen), `spontaneous_action` (outbox). |
| `nora_life.HypothesisPool` | Çapraz doğrulama: internet iddiası önce hipotez; ≥2 bağımsız kurum (alan adı) + kesinlik ile kavram ağına terfi. |
| `nora_life.ReplyRepertoire` | Girdiyi tanıyıp karşılığını bilmeme boşluğu; karşılık iki bağımsız kaynakta görülünce yeterlilik doğar. |
| `nora_life_experiment.py` | Yoğun girdi → sessizlik → kararlılık deneyi. |
| `nora_prediction.py` | Gövdesel RLS modeli, genel nedensel grafik ve salt-okunur sandbox fiziğiyle sınırlı ileri öngörü; genel gerçek-dünya modeli değildir. |
| `nora_drive.py` | İnsan hedefi olmadan hedef kurma, üç hatada terk, başarısız hedeflerden ortak alt-hedef ve yetenek yoklaması. |
| `nora_semantic_code.py` | FST sentezi ile paralel sürekli nedensel grafik (CCG); CCG'de öğrenilen sıfır-durumlu sürekli biriktirici ve train/validation ile otomatik mimari seçimi. |

## Metabolizma
Her düşünsel eylem dünya enerjisinden pay alır (`THOUGHT_COST`). Açlık/ağrı sürüşleri araştırma ve seslenmeyi
bastırır (`calm`). Yani arka plan yaşamı bedenin kapasitesiyle sınırlıdır; kural değil fizyolojidir.

## Sonsuz döngü / mesaj yağmuru nasıl önlenir
Kanal alışması (habituation) + cevapsız her seslenmede sürüşün yarıya inmesi (`0.5**unanswered`). Kullanıcı yazınca sıfırlanır.
Tüm tablolar sınırlıdır; tick maliyeti sabittir (`nora_life_experiment.py` ve A45 ölçer).

## Çalıştırma
```
python nora_agi.py --stage chat --corpus metin1.txt metin2.txt --internet
python nora_life_experiment.py --corpus NORA_EVREN.md --silence 24000
python agi_battery.py A41 A42 A43 A44 A45
```

## Rol ayrımı (Broca)
Karar, merak, duygu ve duruş zihinden gelir (Küresel Çalışma Alanı, A24, A22/CA3, gerilim defteri). Genel konuşmada
sinirsel organ ağırlıkla bunları Türkçeye döker; A142'nin kapsamlı hattında ise ilişki adayını nöral model üretir,
graf yalnız kanıt/doğrulama sağlar. Dolayısıyla güncel mimari “yalnız çeviren Broca” değil, **öğrenen parçalar +
elle kurulmuş bilişsel iskeletin hibritidir**. Duruş (bilmediğinde soru, bildiğinde bildirim) etiket ya da ödül
hilesiyle değil, verinin kendi ilişkisinden öğrenilir: soru işaretiyle biten sözler "bilmiyor/merak ediyor" iç
durumunda okunur (`nora_egit.py kisa`).
Bir ödül-pekiştirme denemesi (`durust_net`) mod çöküşüne (`"??`) yol açtığı için varsayılan yoldan çıkarıldı.

Ölçüm (görülmemiş sohbet cümlelerinde sonraki parçayı tahmin, çapalı): n-gram top-1 %24 → sinirsel %41 (şaşkınlık 44).
Ansiklopedik metinde top-1 %21 (n-gram) → %19 (sinirsel, çapasız %16); yani bilgi metninde sinirsel organ n-gram'dan iyi değil, sohbette belirgin iyi.

## Eğitilmiş ağırlık
`data/broca_tr.pt` (+ `data/tok_tr.json`): `python nora_agi.py --stage chat` otomatik yükler. Yeniden üretmek için:
```
python make_corpus_dump.py --chars 80000000 --out data/text/tr_dump.txt   # Tatoeba: downloads.tatoeba.org/exports/per_language/tur/tur_sentences.tsv.bz2 (3. sütun), Tom/Mary geçen satırlar süzülmüş
python nora_egit.py tokenizer --corpus data/text/tr_dump.txt data/text/tr_tatoeba_c.txt --out runs/tok.json
python nora_egit.py broca --tokenizer runs/tok.json --wiki data/text/tr_dump.txt --chat data/text/tr_tatoeba_c.txt --out runs/broca.pt --minutes 100
python nora_egit.py kisa --out runs/broca.pt --chat data/text/tr_tatoeba_c.txt
```

## Dil olgunluğu kapısı (birleşme sonrası eklendi)
Söz organı yalnızca `.mature ≥ 0.35` (≈1300 duyulmuş parça) ya da eğitilmiş `data/broca_tr.pt` bağlıyken kendi sözünü kurar.
Bunun altında söz dizgi gürültüsü olacağı için Nora uydurmaz: `speak_or()` dürüst yedeğe düşer ("… hakkında henüz bilgim yok; merak listeme aldım").
Ayrıca `_voice_usable()` denetimi vardır: <4 parçalı, çapayı taşımayan ya da aynı parçayı tekrarlayan söz söylenmez.
Eğitilmiş organ bağlıyken bilinmeyen kavramda iki parçalı dürüst duruş kullanılır: "X hakkında henüz bilgim yok; merak listeme aldım. X mi?".

## Dürüst sınırlar
* Sinirsel Broca uzun ve tutarlı genel konuşmada sınırlıdır (5,7M parametre, tek geçiş). A140 sonrası held-out top-1 %29,77'dir; A142'nin ilişki önerisi başarısı bu sınırı genel dil akıcılığına dönüştürmez.
* Eğitim verisi (Tatoeba) çeviri cümleleridir; doğal Türkçe sohbet değildir.
* Eski çekirdekte hâlâ elle yazılmış parçalar var (reddetme gerekçeleri, selamlaşma/empati kortekslerinin sözcük seçimi, Türkçe ayrıştırıcı). Bunlar öğrenilmiş katmana taşınmadı.
* Öğrenilmiş sözün kalitesi duyulan dille sınırlı: hiç metin okutulmadan organ konuşmaz (bilinçli sessizlik), eğitilmiş organ ise kısa sorular kurar.
* Hedef-eşiği gibi birkaç geliştirici sabiti kaldı (ör. terfi için 2 bağımsız kaynak, kesinlik 0.74, `IDLE_TAU`). Bunlar ileride Nora'nın kendi deneyiminden ayarlanmalı.
* ATP/DNA/RNA gibi kısaltmaların yüzeyini koruyan liste dil normalizasyonudur; kavramın işlevini veya nedenselliğini öğretmez. Kuantum gibi yeni ve ağır bir alanı insan müdahalesi olmadan kavrama iddiası yoktur.
* Sınırlı sandbox ileri modelleri gerçek dünya fiziği değildir. Aynı görev/veri/bütçeyle küçük Transformer/LLM karşılaştırması yapılmadı; “Transformer alternatifi” veya genel zekâ iddiası kurulmaz.
* `A23`/`A25` süre eşikleri donanıma bağlıdır; bu makinede değişikliklerden önce de geçmiyordu.

## Güncel işlevsel durum (2026-10-29)

- Temiz ana pil **140/140 (1161,3 sn)**; A73 8/8 ve tarihsel tavan 1124≤1125. Kanıt `runs/agi_battery_140.log`.
- A141 **25/25**: Aves/hücre/mitokondri öğrenimi, kapsam, gerçek boşluk kapanışı ve save/load sonrası bağımsız kullanım.
- A142 **13/13**: nöral top-5 öneri → sembolik kanıt; kabul edilen mitokondri konumu öneri sırasında 4., nöral-off aday yok, validator-off bilgi yazımı yok.
- A143 **12/12**: 1000 adımda 20 hedef üretildi, 9 çözüldü, 10 terk edildi, 1 ortak alt-hedef birleşmesi.
- CCG confirmation-2: counter16 **64/64**, counter24 **62/64 (P16)**. Confirmation-1'in counter24 başarısızlığı kayıtlıdır; weighted P0 ve FST candidate-limit/bütçe patlaması açık sorunlardır. Kanıtlar `runs/ccg_auto_fresh_confirmation*.json` ve `runs/continuous_auto_v2.json`.
- B2'de sıfır-unutmanın yararı yalnız nadir kaynaklı büyük arenada gösterildi; uzun yaşamda konum belirsizliği artışı (0,14→0,51) hâlâ açık.
