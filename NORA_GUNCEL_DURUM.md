# Nora — Güncel Teknik ve Araştırma Durumu

**Tarih:** 2026-10-31

**Kapsam:** Canlı model felsefesi, sürekli öğrenme, Türkçe dil modeli, CCG ve ilişkisel öğrenme, diyalog, dış korunum, disk-MoE ve açık teknik sorunlar.

Bu belge güncel ana durum dosyasıdır. Ayrıntılı tarihsel kanıtlar `runs/LEARNED_REPRESENTATION_REPORT.md`, `runs/TURKISH_MOE_FINAL_REPORT.md`, `runs/TURKISH_ONEPASS_CONTINUATION_REPORT.md` ve `runs/TUTOR_DIALOGUE_AUDIT.md` içindedir.

## 1. Kısa karar

Nora canlı çalışan ve yeni girdileri tek sunumda işleyen bir sistemdir. **31 Ekim HDC/MB güncellemesi, dar A/B/C tek-sunum protokolünü, dinamik ilişki icadı/transfer protokolünü, epistemik-Broca/pasif-bellek sınavlarını, kanıtlı System-2 birleşik sorgu protokollerini ve yeni HDC müdahale/vektör/bağlam/öz-soru sınavlarını geçti; genel “tek okuyuşta genelleyen ve hiç unutmayan zihin” iddiası ise hâlâ kanıtlanmış değildir.**

- Anlamsal olgu ve ilişki kararları artık HDC/MB aktivasyonuyla adaylandırılıp sembolik grafça doğrulanıyor; grafın ham BFS listesi cevap motoru değildir. GRU/TokenMoE tarafındaki üretim rolü artikülasyondur.
- Bilinmezlik `(kavram,HAS_STATE,unknown)` epistemik olayı olarak Graph’ta tutulur ve sembolik Broca’nın çekim motoruyla söylenir; ana anlam sorgusu bu olayı dünya bilgisi saymaz. Kanıtsız tanımda nöral üretim ve çevrimiçi öğrenme sıfırdır.
- ISA mirası ve nedensel zincir kanıt yolunu gösterir; birleşik ters sorgu ve karşılaştırma üçüncü HDC `relation_object` adresiyle çalışır. Aktif çalışma kümesi dışındaki olaylar bilinmiyor sayılmaz, pasif sinapslardan geri çağrılır.
- Canlı öğrenme iskeleti vardır ve deneyler canlı checkpoint’in üzerine yazılmamıştır.
- Yeni C6–C18 sonuçları genel next-subword dil modeli sonucu değildir; genel diyalog ve açık uçlu öğrenme için daha geniş bağımsız sınav gerekir.
- Fold-1/fold-2 için ortak %45 top-1 ile geriye-dönük korunum aynı adayda sağlanmamıştır.
- Fold-3/4 confirmation, ortak development adayı bulunmadığı için başlatılmamıştır.
- Eski epoch/replay sonuçları tek-sunum başarı kabulünü karşılamaz.
- 0,45B disk-MoE yalnız kaynak stresidir; rastgele genişletilmiş uzmanlarla kalite kanıtı değildir.

### 1.1 HDC/MB ve dinamik ilişki güncellemesi

Yeni çekirdek `nora_chem.py::HDCMushroomBody`, her olayı rol–dolgu HDC bağlarıyla bir kez yazar; k-WTA hedefi **%6**dır. Aynı tek sunumun özne, tam özne–ilişki–nesne ve ters erişim için ilişki–nesne (`relation_object`) içerik adresleri `W_fast/W_slow` üzerinden okunur; üç adres üç öğretim/epoch değildir ve `presentations` yine olay başına bir artar. Nörogenezden sonra eski kodun aktivasyonu yeni toplam KC sayısına bölünmez; her olay kendi yazım anındaki aktif-KC normunu kullanır. Aktivasyon eşiği düşürülmemiştir.

| Sınav | Sonuç | Kanıtlanan dar kapsam |
|---|---:|---|
| Mimari C1–C18 | **18/18, 78,4 sn** | Yayın/ablasyon + MB, ilişki, epistemik-Broca, graph-corpus, disk-pasif bellek, System-2, HDC vektör/bağlam/do() ve Graph öz-soruları |
| C6 A/B/C | **1/1** | Üç öğretim birer sunum; üç soru uyanıkken; uyku ve replay sıfır; üç MB izi |
| C7 yeni ilişki | **1/1** | Probation `YENI_OPERATOR_VORL`; iki held-out transfer; kontrol `None`; ters yön `conflict` |
| C8 beş tohum | **5/5** | MB ablasyonunda cevap kaybı, restorasyonda öğretimsiz dönüş; 40 yeni bilgi ve bakım sonrası korunum; bütün eski bootstrap kenarlarında eksik **0** |
| C9 uyku/kalıcılık | **1/1** | 501 yaşam adımı ve açık uykuda `replay=0`; yasak eski `mind._sleep` çağrısı **0**; save/load sonrası eski bilgi, probation, operatör ve canlı transfer |
| C10 çok-köklü icat | **3 tohum × 5 kök** | 15/15 ayrı operatör ve transfer; bilinen fiil/koşaç yeni operatöre dönüşmedi |
| C11 epistemik kapı | **1/1** | Gerçek `broca_tr.pt`; üç soğuk bilinmezlik **1,33/0,85/0,83 ms**; `generate=0`, `learn_online=0`; bilinmeyen dinamik ilişkide de sıfır nöral çağrı |
| C12 graph-corpus | **1/1** | İki cümle → iki kenar/iki HDC sunumu; son tam koşuda tanım **4,1 ms**, ayrı üç soğuk tekrar **9,0/8,1/7,3 ms**; tanımda nöral çağrı **0**, tek anlam boyutunda yetersizlik açık |
| C13 disk-pasif HDC | **1/1** | Bağımsız SQLite satır kanıtı; aktif küme **4**; 24 kayıt korunur; normal/pickle/ilk-nesne okuması doğru; yeni sunum **0** |
| C14 epistemik olay/Broca | **1/1** | Üç bilinmezlik cevabı sembolik Broca’dan; nöral çağrı **0**; ana anlam kenarı yok; save/load’da `unknown` korunur, accepted öğreti `known` olayı ekler |
| C15 System-2 birleşik anlam | **1/1** | Doğrudan kenar `used_s2=False`; ISA/nedensel zincir ve başarılı birleşik sorgu `True`; tam/kısmi ablasyon, olumsuz koşul, üç+ doğrudan kenarda yetersizlik ve kapsamlı-parça kanıtı; sorguda yazım **0** |
| C16 sınırlı tutma/genelleme | **3 tohum × 96 olay** | Her tohumda canlı/bakım/pickle okumaları **96/96 + 96/96 + 96/96**; aktif küme **3**, pasif kayıt **216**, replay **0**, kalıtım yeni kenar ezberlemeden çalışır |
| C17 HDC vektör/bağlam/do() | **23/23** | Bağımsız yol ve bipolar vektör oracle’ı; `Kral−Erkek+Kadın→Kraliçe` skor **1,000**; Organ/Sayı bağlam seçimi; MB ablasyonları; alternatifli karşıolgusal `lost/kept` |
| C18 Graph öz-soruları | **9/9** | Ortak sınıf/özellik ve iki-halkalı nedensel öz-test; dinamik `sebebiyet verir`; HDC/Graph kanıtı, provisional ayrımı ve sorguda yazım **0** |
| Dış held-out H1–H9 | **9/9** | Önceki dış işlevler; H7 nedensel zincir 3/3 |
| A142 nöral öneri + sembolik kanıt | **13/13** | Nonce top-5 18/18; dayanaklı kabul 16/17; nöral/doğrulayıcı ablasyonu `unresolved` |
| A140 Broca/corpus | **19/19** | Corpus `organ_seen=0`; nöral yüzey skorları corpus boyunca sabit; sosyal/çevrimiçi artikülasyon yolu ayrı ve canlı |

Son kaynakta iki ayrı işlevsel grup çalıştı: A140 **19/19**, A141 **25/25**, A142 **13/13**, A144 **27/27**, A51 ve A47 ile **6/6, 398,2 sn**; A94 **7/7**, A50 **12/12 kapsam**, A16 **4/4**, A110 **5/5**, A57 ve A54 ile **6/6, 104,2 sn**. A78 **11/11** (HDC ablasyonu ve alternatifli karşıolgusal), A143 **16/16** (1000 adım ve 8 öz-soru) geçti. Chem duman testi **5/5**, `selftest.py --bridge` **118 sn** ile geçti. Tam `agi_battery.py` bu güncellemede sonuna kadar çalıştırılmadı; dolayısıyla “bütün işlevsel batarya geçti” denmez.

Bilinmeyen SVO yüklemi artık otomatik ISA/HAS_PROP sınıfına zorlanmaz. Wernicke özne/nesne/yüklem rollerini çıkarır; `GroundedConceptGraph.invent_relation()` yeni ilişkiyi probation olarak yazar. Aynı ilişkinin yeni bir alanda görülmesi `discovery`, daha önce bulunmayan ilişkinin açılması `invention` olarak deftere geçer. Nöral proposer yalnız aday verir; kabul yön/çelişki ve kanıt denetiminden geçer. Operatör koda gömülmez, `nora_forge.py`/`nora_apply.py` defterlerinde ve save/load meta kaydında taşınır.

**System-2 artık ölçülebilir bir hesap yolu:** `relation_object` HDC okuması ters adayları getirir, canlı Graph her iki koşuldaki kenarları ayrı ayrı doğrular ve aday kümeleri kesiştirilir. Karşılaştırmada doğrudan kenarlar farklıysa iki tarafın HDC kanıtı da zorunludur; tek taraflı aktivasyon veya yalnız sembolik küme kesişimi yeterli değildir. Başarılı sonuç `used_s2=True` olur; çözümleme hatası, eksik kavram ve boş sonuç bilişsel başarı sayılmaz. Son gerçek diyalog probunda:

- “Kedi ile köpeğin ortak noktası nedir?” → “Kedi ve köpek bir hayvandır; kedi ve köpek bir evcil varlıktır; kedi ve köpek evcildir; kedi ve köpek bir canlıdır.”
- “Kanadı olan ve uçan şey nedir?” → “Kuş; çünkü kuşun kanadı var ve kuş uçar.”
- “Kanadı olan ve havlayan şey nedir?” → “Bu ölçütleri birlikte karşılayan kavram bulmadım.” ve `used_s2=False`.
- Yalnız parça+eylem öğretilen “Kuş nedir?” → “Kuşun kanadı var. Kuş uçar.”; tek boyutlu C12 örneğinde ise eksik parça/nedensellik açıkça söylenir. Bütün bu sorgularda yeni HDC sunumu **0**dır.

**Sınır:** C6–C18 belirli, bağımsız oracle’lı mimari protokolleridir. Sınırsız Türkçe ayrıştırma, her olası ilişkinin doğru icadı, bütün alanlarda genelleme veya yıkıcı unutmanın her koşulda imkânsızlığı kanıtlanmamıştır. C16’daki sıfır kayıp yalnız üç tohum × 96 olay ve canlı/bakım/pickle protokolüne aittir.

### 1.1.1 Etiketlemeden düşünmeye: müdahale, vektör, bağlam ve öz-soru

Bu güncellemede yeni modül eklenmedi; mevcut `GroundedConceptGraph`, `NoraAGI`, `HypothesisPool` ve otonom sürücü derinleştirildi.

- **Geçişli çıkarım:** `Kedi ISA Memeli` ve `Memeli HAS_PROP Sıcakkanlı` için cevap yalnız “Evet” değil, HDC/MB’den doğrulanan `Kedi → Memeli → Sıcakkanlı` kanıt yolunu Broca ile gerekçelendirir. Doğrudan `(kedi,HAS_PROP,sıcakkanlı)` kenarı yazılmaz; MB ablasyonunda sonuç `unresolved` olur.
- **Gerçek `do(X)`:** Her nedensel halka `query()` üzerinden HDC/MB adayı ve canlı Graph yön doğrulamasıyla izlenir. X’e giren kenarlar etki olarak izlenmez; MB kapalıyken ham `_out_edges` üzerinden cevap üretilmez. Karşıolgusal hesap X’siz modelde alternatif nedensel kökleri de tarar; bu yüzden `ortak→Y` varken Y `kept`, alternatifsiz yol `lost` olur. Bilgi silinmez, kenarlar geçici maskeyle devre dışı bırakılır.
- **HDC vektör cebiri:** `vector_analogy()` öğrenilen A→B rolünü bulur, bipolar `role×filler` bağlarıyla `v(A) − bind(rol,B) + bind(rol,C)` hesaplar ve adayları kosinüs/MB seyrek koduyla sıralar. Copula/taksonomi (`ISA`) ve özellik (`HAS_PROP`) yalnız vektör rolünde `ATTRIBUTE` olarak normalize edilir; özgün Graph kanıtı değiştirilmez. Eski `solve_analogy()` API’si klasik çağrıları koruyup çözülemeyen aritmetik analojide bu yola düşer. Hedefe özel `Kraliçe` şablonu yoktur; C17 bağımsız numpy oracle’ı aynı skoru hesaplar.
- **Bağlamla anlam seçimi:** Birden çok doğrudan ISA anlamı olan yüzeylerde bağlam ve her anlamın öğrenilmiş önermeleri rol–dolgu bağlamasıyla seyrek KC koduna çevrilir. `Denizde yüzen adamın yüzü` → `organ`, `Matematikte yüz` → `sayı`; bağlam kanıtı yoksa `ambiguous` denir. Tek anlamlı tanım soruları ve sayısal aritmetik bu özel yola zorlanmaz.
- **Kendi soruları:** `HypothesisPool.probe_graph()` ortak sınıf, ortak özellik/kalıtım ve çok-halkalı nedensel soruları üretip hemen HDC/Graph sorgusuyla çalıştırır. `BackgroundLife.tick()` ve `Driver.run_autonomous()` her 20 adımda bu organı çağırır. Kayıtlar soru, yanıt, kanıt yolu, `presentations_delta`, `graph_edges_delta` ve `accepted/provisional` düzeyini taşır; çözülemeyen boşluk gizlenmez, çıkarım yeni olgu diye yazılmaz. A143’te insan hedefi verilmeden 1000 adımda 8 öz-soru oluştu.
- **Dinamik nedensel yapı:** `Çöl kıtlığa sebebiyet verir` DAT nesneli CAUSES öğretisi olarak aynı zincire katılır; havuz `Çöl → Kıtlık → Göç` öz-testini de üretip sınar. Bilinmeyen SVO ilişkilerinin probation/icat yolu korunur.

**Sınır:** Vektör analojisi öğrenilmiş kompozisyonel önermelerle sınırlıdır; genel kelime-gömme analojisi iddiası değildir. Bağlam çözümü ayırt edici kanıt yoksa seçim yapmaz. Öz-soruların çözülmesi yeni gerçek öğrenmek değil, mevcut kanıtı sınamaktır. Fold-1/2 kalite eşiği ve genel next-subword hedefi bu sonuçlarla değişmedi.

### 1.2 Epistemik kapı, graph-corpus ve disk-pasif bellek

- **Bilinmezlik ana yoldur:** Bağımsız tek-sözcüklü “X nedir / ne demek?” sorusu önce Graph/HDC’ye gider. Kanıt yoksa `speak_unknown()` `(kavram,HAS_STATE,unknown)` olayını ayrı epistemik deftere yazar; `BrocaGenerator.realize_epistemic()` çekimli cümleyi üretir, hazır `nora_priors` cevabı veya nöral dünya bilgisi kullanılmaz. Tekrarlanan soru kavramı “bilinen” yapmaz; accepted öğreti eski olayı silmeden `known` olayı ekler. Hızlı nesnel cevap, gerçek-zaman yaşam yakalama sırasındaki iç-konuşma/nöral üretimden önce tamamlanır. Doğrudan kenar sayısından bağımsız olarak işlev/parça/nedensellikten en az ikisi aranır; yoksa kapsamlı-parça kanıtı da korunarak yetersizlik söylenir. `speak_from_state()` kavram çapalı her nöral üretimde ayrıca Graph/HDC kanıtı arar. Aktif öğretici oturumu, kişisel bellek ve uzman yollar erken sözlük kapısıyla kesilmez.
- **Broca bilgi kaynağı değildir:** `read_corpus()`, canlı dünya-olgusu dinleme, araştırma ve `nora_besle` genel kültürü nöral `learn_online` hattından çıkarır. Anlam Wernicke → Graph/HDC tek-sunum hattına gider; sosyal dil ve soru biçimleri sözdizimi/artikülasyon deneyimi olarak kalır. `broca_tr.pt` yeniden eğitilmemiştir; değişen, onun ana sistemde doğrulanmamış içerik kaynağı olmasını engelleyen çalışma-zamanı sözleşmesidir.
- **Aktif olmayan bilgi yok değildir:** `HDCMushroomBody` varsayılan 64 satırlık LRU çalışma kümesi kullanır. Pasife inen `W_fast/W_slow` satırları RAM sözlüğünde değil, `.cache/nora_hdc` altındaki SQLite dosyasındadır. HDC digest’i aktif ve pasif tarafta yalnız ölçülecek sinapsları adresler; her adayın aktivasyonu yeniden ölçülür, eşik ve Graph yön/kenar doğrulaması zorunludur. Okuma yolunda karar için kullanılmayan bütün-matris enerji normları kaldırılmış, açık `weight_energy()` tanısına taşınmıştır. Sembolik depo cevap bypass’ı değildir.
- **Kalıcılık ve temizlik:** SQLite geçici çalışma deposudur; pickle snapshot’a pasif satırlar ve erişim anahtarları eklenir, yüklenen nesne bağımsız yeni depo açar. Normal AGI save/load kenarlardan HDC bağlarını yeniden kurar. Nesne kapanınca geçici dosyalar temizlenir; `.cache` Git’e eklenmez.

**Canlı besleme/checkpoint probu:** 324 üçlüden 302 yeni kenar ve cümlelerden 171 kenar çıktı; sosyal diyalogun 529 sözcüğü/126 metni artikülasyona giderken `SENTENCES` genel-kültür öğelerinin nöral öğrenmeye sızması 0’dı. Save/load sonrası 505 kenarın 501’i pasifleştirildi; `(kedi, ISA, evcil hayvan)` diskten doğru geri çağrıldı ve sorgu yeni HDC sunumu üretmedi.

**Ölçüm sınırı:** ≤20 ms sonucu bağımsız tek-sözcüklü tanım sorularına aittir; bütün karmaşık soru biçimlerine genellenmez. C13 ve canlı prob pasif ağırlıkların gerçek SQLite satırlarından geri geldiğini kanıtlar; mimari ayrıca pasif-ağırlık RAM sözlüğü tutmaz. Sıfırlanmış satır depolaması, graf ve diğer organlar dâhil toplam RAM’in veriyle sabit kaldığı iddia edilmez; büyük ölçekte disk I/O ve RAM ayrı stres testi ister.

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

- `nora_voice.py::hear` sosyal/sözdizimi metnini cümlelere ayırır ve seçilen cümleleri bir kez `learn_online` yoluna verir; epoch yoktur. Dünya-olgusu ve genel-kültür corpusu ana hatta bu çağrıdan çıkarılmıştır.
- Canlı turda `neural_limit` varsayılanı 8 cümledir. `read_corpus()` artık `hear_file` çağırmaz; her uygun cümleyi bir kez Graph/HDC ve yapı/şema kortekslerinden geçirir, `organ_seen=0` ölçülür.
- `TokenMoEOrgan.learn_online` kendisine verilen dilbiçimi örneğini nedensel prefix kararıyla hedef veya haricî uzmana bir kez yönlendirir; semantik kanıt kaynağı değildir.
- `BrocaOrgan.learn_online` küçük öğrenme oranıyla (`3e-4`) tek-geçiş güncelleme yapar.
- Mantar cismi modülleri k-WTA seyrek kod, hızlı/yavaş sinaps ve stiffness mekanizmalarını gerçekten uygular.
- Anlamsal graf sorgusu, nedensel zincir ve analoji adayları HDC/MB’den gelir; MB ablasyonu graf kenarları yerindeyken cevabı kaldırır.
- Yeni bilgi için `learn_oneshot()` gradyan/epoch kullanmaz; çakışan rakip sınıfın ağırlığını silen contrastive güncelleme kaldırılmıştır.
- `sleep()` yalnız sinaptik bakım/homeostaz yapar; eski `mind._sleep(64)`, S2 ve MB replay çağrıları kaldırılmıştır. Ana Nora’da 500-adım periyodik `mind._sleep(48)` de kapalıdır ve eski save/load kaydı bunu yeniden açamaz. Pozitif `replay_steps` açık `ValueError` üretir.

### Tamamlanmamış uyum

- Genel next-subword kalitesi ve fold-1/fold-2 ortak %45 eşiği çözülmemiştir; Broca’nın gradyanlı dil öğrenmesi için bütün dış korunum iddiası hâlâ ayrı bir problemdir.
- C6–C18 dışındaki bütün olası cümle biçimleri ve alanlar için genelleme kanıtı yoktur; intransitif, çok-öbekli ve karmaşık Türkçe yapılar daha geniş held-out sınav ister.
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
6. Disk-pasif HDC’de olay sayısı büyürken SQLite I/O, RAM sabitliği ve erişim gecikmesini ayrı stres testiyle ölçmek.
7. Weighted ve D2 gibi açık sorunları ayrı tutmak; başarılı alanlarla örtmemek.

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
