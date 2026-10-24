# NoRa_AGI - eklemeli sürekli öğrenme
- Tüm veri/kod D:\NoRa_AGI altında (C dolu, 0.7GB boş). Data: D:\NoRa_AGI\data\<dataset>, beyin: D:\NoRa_AGI\beyin.npz
- Sistem: Pentium G5400 2-çekirdek, RX580 -> CPU-only NumPy, <5sn yerleşme hedefi
- v3: --fast (prototip kafa, geri yayılım yok, ~0.07sn/görev), --sleep (prototip tazeleme), --router proto (varsayılan)
- Sonuç Fashion 5-görev: fast 1.3sn toplam, task-known 0.93, BWT 0.0, task-unknown proto 0.65
- Felsefe: reset yok, dondur + üzerine ekle (extra-if), uyku fazı yıkıcı değil

v3 inceleme düzeltmeleri (8 madde) uygulandı: _train tek-softmax (NLL yalnızca raporda), reuse_hist çift koruma + plato bypass kaldırıldı, ensemble temiz z-skor (DENEYSEL), --fast=hızlı-NCM taban çizgisi + eğitim-acc kapısı, PROJECT_ROOT taşınabilir (Win D:/NoRa_AGI), sleep gerçek ms + aday etiketi, growth_map npz'de kalıcı. D-disiplini: .venv + .cache D'de, D_kullan.ps1 kilidi, DPY ile çalıştır. Fashion 5-görev yavaş: 0.979/BWT 0.0, proto-unknown 0.621.

micro_gpt.py: 4L/4H/256e block128 batch32 ~3.2M char-GPT, torch-CPU D .venv'de. Veri: tiny_slice.txt 42M karakter TinyStories (HF Range ile ilk 42MB). Eğitim: 800 iter total, val 4.02→1.25 (~45dk). --resume ile kaldığı yerden devam, --curriculum ilk %20 kolay satırlar. ckpt: D:\NoRa_AGI\models\micro_tiny.pt. Sonraki: grow_gpt (dondur+adaptör blok, unutmasız dil).

v4 AGI yönü: --stream ile etiketsiz akışta 5/5 sınır otomatik (referans-çapalı seviye+değişim kuralı), 2.4sn, BWT 0.0, tek blok. Parça maliyeti sabit (0.5/0.1sn) — veri 10x artsa adım maliyeti aynı. surprise()/recall() eklendi (çok-boyutlu kafa uyumlu). Ders: EMA-çapa kayması ve inf-önyükleme hataları düzeltildi. Sonraki: grow_gpt adaptör + micro-GPT dünya-modeli geri-beslemesi.

grow_gpt.py: donuk 3.2M omurga + 57k adaptör (256-64-256 sıfır-init) + B kafası. Dağılım kaymasında (TinyStories→Shakespeare): adapt F=0.000/L_B=2.045/4dk, full F=+0.724/L_B=1.773/11dk. Aynı-dağılım kontrolünde unutma yok (F=-0.077). Açık: plastisite farkı 0.27 — sonraki iş adaptör kapasitesi/büyüme eşiği. B-diliminde 52 görülmemiş karakter boşluğa eşlendi (kayıt).

Adaptör-büyüme: plato (W=3 B-val, eps=0.02) → eskileri dondur + sıfır-init yeni adaptör. 300 iter Shakespeare: plato tetiklenmedi (istikrarlı iyileşme 2.17→2.01), L_B=2.009, F=0.0. Mekanik kanıt: yeni adaptör identity (fark 0.0), eskiler donuk, 57k eğitilebilir. Kural silahlı bekliyor; tetikleme daha uzun/plato-yakın rejimde.

Dil-şaşkınlık kapısı (grow_gpt --stage detect): 8 parça [A,A,B,B,S,S,A,B] akışında 3/3 isabet, 0 yanlış-alarm (ort+p90, referans-çapa, seviye+değişim). Aynı-dağılım A→B sessiz (özgüllük), S→A dönüşü ötüyor. İki hat birleşti: hangi cümle yeni bilgi sorusu etiketsiz çözülüyor. Sonraki: detect→adaptör zinciri (otomatik büyü-kapan döngüsü).

Kapalı döngü (grow_gpt --stage auto) çalışıyor: [S×4,A×2] akışında insansız 2 segment, yönlendirici 20/20 (yenilik→segment, tanıdık→omurga), 335sn. A-dönüşü ateşledi ama gereksiz büyüme YOK (birikim tanıdık→atıldı). Bilinen kusur: önyükleme+birikim çift S segmenti (hafif aşırı-bölütleme), ateşleme-nedeni hizası kusurlu ama segmentler saf. grow_auto.pt kayıtlı.

Uyku-birleştirme (grow_gpt --stage sleep): işlevsel eşdeğerlik (eps 0.025, sabit partiler) + kosinüs ön-eleme. Çift S: cos=0.050, fark=0.046 → birleşme DOĞRU ŞEKİLDE REDDEDİLDİ (farklı çözüm, benzer kayıp). Router 20/20 korundu, grow_merged.pt kayıtlı. Ders: aynı kayıp ≠ aynı bilgi; işlevsel test yanlış-birleşmeyi önledi.

BLOKAJ: commit 0.3GB (hoi4 3GB+steam yiyor, pagefile C dolu büyüyemiyor). torch import 310MB (threads=1 ile 183MB). Önlemler kodlandı (uint8 depo, kapaklı yükleme, del ck, batch/thread CLI, D_kullan.ps1) ama commit bitikken eğitim imkansız — eskiden çalışan config bile ölüyor. Çözüm: hoi4/steam kapatılıp commit 2GB+ olunca devam. Resume: . D:\NoRa_AGI\D_kullan.ps1; DPY grow_gpt.py --stage auto --pool codemath --batch 8

Kapalı-döngü codemath BAŞARILI: [C×4,M×2] akışında insansız ['C','M'] segmentleri; C-val 2.90 (sıfır-atım 4.66), M-val 1.46 (sıfır-atım ~7.2); yönlendirici 20/20; 348sn. Anahtar: asimetrik vektör-şaşkınlık (aday-yayılımlı, kf=3) — skaler kayıp M'yi C-bulutunda kaybediyordu. Pekleşmiş birikim atılıyor (verimlilik), A-dönüşü büyüme yapmıyor. Blokaj dersi: commit darlığında uint8 depo + kapaklı yükleme + batch 8 + tek-çekirdek.

Full-codemath tavanı (150iter/faz): CODE-sonu L_A=2.37/L_C=1.82; MATH-sonu L_A=6.27/L_C=7.07/L_M=0.96. Hüküm tablosu — adapt: C 2.90/M 1.46/F 0.00; full: C 1.82/M 0.96/F +5.08 ve kod-tam-yıkım 7.07. Plastisite farkı C 1.08/M 0.50 (full ~2x iter gördü, not edildi). Sıradaki: farkı F=0 ile kapatma (adaptör kapasite/iter/plato-büyüme).

Grid hükmü: G1 2.445/G2 2.417/G3 2.271/G4 2.207 (kod-val, hepsi F=0). Genişlik 128 değmez (+58% param, -0.03). İter işgücü, büyüme taçlandırma (2 kez tetiklendi). Kazanan-konfig MATH: 1.248. Final: adapt C 2.21/M 1.25/F 0.00 vs full C 1.82/M 0.96/F +5.08. Fark C 1.08→0.39, M 0.50→0.29. Önerilen varsayılan: bn64 + plato-büyüme (eps~0.1, max-adapt 3) + bütçece iter.

Döngü-içi istifleme DÜZGÜN çalışıyor: seg1 altına adaptör2 istiflendi, M-val 1.364→1.250, ['C','M'], C 2.90, router 20/20. Ders kaydedildi: yeni-kafa açmak sıfırlardı (router 6/20'ye çöktü), istifleme korur. Final: adapt C 2.90/M 1.25/F 0.00 vs full C 1.82/M 0.96/F +5.08. Mimari olgun: detect→segment-büyüme→döngü-içi-istifleme→uyku→yönlendirici.

3-havuz döngü TAMAM: ['C','M','T'], val 2.84/1.32/2.51, kf=2.5 ile T yakalandı (3.0'da kaçmıştı), istifleme C+M+T hepsinde tetiklendi, router 20/20, uyku 3→3 (doğru ret), 123 sembol. Üretim sondası: M formatı kapmış (12+34=58), C zayıf, T ekli-kök uyduruyor. Sıradaki (kullanıcı dönünce soracak): C segmentine daha iter veya 4. havuz.

System-2 üçlüsü CANLI: C) tool_calc (ast-whitelist) + kapı 46'yı düzeltti; B) ToT-lite geri-dönüşlü seçim; A) iz-eğitimi 400iter (M 1.18→1.01) iz ŞEKLİ öğretti ama sadakat yok ([81+9=27] tipi halüsinasyon) — P(rakam|'=')≈0, ödül örneklenmeyeni seçemez. C-refine: 2.84→2.25→2.14 (<2.0 kaçtı, tavan ~2.1). Ders: altyapı hazır, getiri kafa gücüne kilitli.

Müfredat sonucu: 1digit 350iter (0.78→0.74) kopyalama filizlendi (sadık alt-izler, '= [2' bağlam-kopyası), mezuniyet think_math 200iter (1.17→0.97). 2-basamak iz-sadakati YOK ([87+6=44]) — transfer bekliyor. Araç kapısı her demoda düzeltiyor (46). Ölçüm dersi: prob konumu iz-içi olmalı ('=' sonrası boşluk+köşeli DOĞRU davranış). r1 beyinde M 4 adaptör.

RFT soğuk-başlatma SIFIR isabet (katı eşleşme + bağlama yok) → tek-basamak müfredat + 750 iz-iteri: İLK operand bağlanması filizlendi ([2+9.., [2-2.., [3+3..), val 0.73. İkinci operand + sonuç hâlâ rastgele. Araç kapısı her demoda düzeltiyor (5, 46). Hüküm: yön doğru, tam sadakat daha iter ister; correctness ŞİMDİDEN araçta garanti.

NÖRON TEMELİ + EVREN (yeni hat): nora_neuron.py = FlyWire sinek beyni (Shiu et al., philshiu/Drosophila_brain_model) saf-NumPy LIF; sürüm 630 üzerinde Brian2 referansıyla r=0.967 (aktif 457 vs 448; MN9 122 vs 93 Hz — oranlar ~%30 yüksek, kalibre edilmedi), 1sn=12dk (yavaş). MushroomBody/LobedMushroomBody = öğrenen modül. nora_world.py = fizikli evren (Newton, momentum, gün/gece, rüzgâr, metabolizma). nora_mind.py = çalışma alanı + valans + süreklilik + öz-model, dopamin-kapılı Q. Bkz NORA_EVREN.md. Sonuç: ölüm/1k 1.2→~0.15; çalışma alanı karışıma üstün DEĞİL (dürüst); öz-model beceri +0.5, ajanlık AUC 0.82. Ders: ödül şekillendirmesi olmadan seyrek ödülde öğrenme yok; kalın ışın gerekliydi (ince ışınlar yiyeceği kaçırıyordu).

Bilişsel harita+plan eklendi (CognitiveMap: pusula+hız yol integrasyonu, duvar çapası, görerek tehlike haritası, Bellman rota). 8 tohum/60k: full 0.08±0.04, full-nomap 0.12, flat 0.23, flat-nomap 0.15, random 1.11 ölüm/1k. Harita susuzluk ölümünü bitirdi; çalışma alanı planlama varken karışıma açık üstün (çatışma çözümü), planlamasız fark yoktu. Konum hatası ~0.5. Sonraki: öğrenilmiş dünya modeli/hayal, değişen evren.

ÖZERK OTURUM (kullanıcı uyurken): kural değişimi evreni (invert/poison), öğrenilmiş yön modeliyle plan, bağlam mekanizması v2 (olgunluk+dondur+probation+anlamsal sonuç belleği V), RLS dünya modeli + hayal gücü, epizodik bellek + uyku tekrarı, Introspector (denetimsiz prototipli iç gözlem kodu) + nedensel doğrulama (iç müdahalede rapor d~+6..37 vs papağan ~0..2.4; ışınlanma sürpriz raporu +3.1 vs 0.4), Narrator (doğrulanabilir iddialar; 'su bulmak' deyince %91-100 su içiyor taban ~%45), kaçırılma tespiti + küresel yeniden konumlama (hata 13->1.1 /100 adım), nora_bridge.py (torch kuruldu: bilmeceli kutular, sezgi/düşünme kararı, uyku). Detay: NORA_EVREN.md. Açık: sıfır unutmanın davranışsal faydası zayıf; gizli dünya değişimi testi ayırt edici değil.

ÖZERK OTURUM SONU DURUMU: NORA_EVREN.md tam sonuç raporu (A-G + dürüst sınırlar). Güncel nihai: full 0.05±0.03 ölüm/1k (rastgele 1.06), karışım 0.10±0.16 (konfor 0.77>0.71) — çalışma alanı faydası kanıtlanmış DEĞİL (yalnız yara ölümlerinde eğilim). Aktarım güçlü (yeni evrende ilk pencere konfor 0.76 vs 0.54). İç gözlem papağana karşı nedensel ayrışma iki bağımsız koşuda (d +5.6 vs -0.2). Narrator 3 tohumda tutarlı: 'su bulmak istiyorum'->%94-100 su içiyor (taban ~%50). Köprü: torch kuruldu (PyPI); maskeli+ağırlıklı satır eğitimi sezgiyi 1 basamakta %89'a çıkardı, üstbiliş AUROC 0.81, uyku damıtması sezgi doğruluğunu 0.47->0.67 yaptı; 2 basamak %38, 3 basamak %0. Olumsuz sonuçlar: sıfır unutma davranışsal faydası yok (8 tohum), düzen-bağlamları (mapctx) yanlış alarm, öz-öngörü rapor kodu süreklilikten iyi değil, duvara yönelme/içme çapası uzun yaşam belirsizliğini azaltmadı. Araçlar: selftest.py [--bridge], STOP_NORA dosyası ile durdurma. Önemli ders: negatif kanıt 'görünmüyor' görüş alanı (150°) yüzünden 'yok' demek değildir; kalibrasyon bağlama (basamak) özgü olmalı.
Ek bulgular (özerk oturum sonu): tanıklıkla öğrenme 48x48 arenada ölümü yarıya indirir (A'nın anlatısını B okur; yanlış anlatı etkisiz) ama 24x24'te etkisiz (bilgi darboğazı gerekli). 'Yaşıyorum' önermesi kanıttan türetilir ve çürütülebilir (ihtiyaçları tükenmeyen evrende kurulmaz). Anlatıcı yaşanmamış 88 iç durum kombinasyonunda %100 sadık. Rapor davranıştan ~21 adım önce. Seyreklik tatlı noktası ~%6 aktif KC (daha seyrek/yoğun 2-3x kötü). FlyWire 96 nöron alt çizgesi süreklilik ağı olarak rastgele ağla eşdeğer. Çalışma alanı hayatta kalmada karışımı geçemedi (çelişen acil ihtiyaçta 0.82 vs 0.90): değeri yapısal/raporlanabilirlik. Aşamalar: run,compare,shift,transfer,introspect,narrate,demo,testimony,alive,combo,lead,conflict.

## Güncel (tur: Çalışmayanlar)
- Kural tanıklığı işe yarıyor (ısırık 15,25→9,75, 12 evren, kazanılabilir zehir evreni). Kültür zinciri zayıf pozitif.
- Sıfır unutma ilk davranışsal kanıt: `--stage shift --kind swap --size 40 --nfood 4` (A' ölüm 0,22 vs 0,69, 8 tohum); küçük arenada fayda yok. `perc_sem` (kanal→sonuç) gerekli.
- Olumsuz: öz-öngörü (0,71 vs 0,70), köprü AUROC 0,68 ile 2 basamak %75, 3 basamak %0; çalışma alanı dikkat kazancı ve düzen bağlamları yapılmadı.
- HABER_ADAYLARI.md: Clausewitz-MCP en güçlü haber adayı.

## Güncel (tur: epistemik katman + Python sentezi)
- `nora_episteme.py`: inanç deposu (Jeffreys α/β=0.5), kaynak güven modeli (alan önseli + Beta öğrenme, karışım n/(n+4)), çelişki izi, eş ad birleştirme (0.80 + 2 bağımsız kaynak), `unify_after_alias` (hipotez kapanışı + inanç birleştirme + kanonik anahtar indeksi), System-1 terfi (güven ≥0.80 + kullanım/doğrulama), halüsinasyon denetçisi, allowlist'li `TrustedResearcher` (wikipedia/who/tubitak/arxiv/python/.gov/.edu). `absorb` soruları asla iddia diye kaydetmez; yalnız yapısal iddialar (olay/ölçü/eş ad) inanç defterine girer.
- `nora_code.py`: plastik operatör korteksi (DISPLAY/CONCAT/ASSIGN/LOOP; softmax seçim + çalıştırma geri bildirimiyle ağırlık) + `RequestGrounder` (tırnaklı/tırnaksız, kanal sözcüğünden sonra gelen metin, düz tek tırnak) + `ExecutionVerifier` (subprocess, stdout birebir) + System-1 beceri belleği. Doğrulanmamış program asla söylenmez; belirsiz istekte dürüst abstain.
- `nora_agi.py`: iki yeni kortikal sütun — `epistemic_prefrontal` (0.966–0.972) ve `python_motor_cortex` (0.988). Kod istemi cümle bölmesine uğramaz ("Merhaba! Ey Ahali!" içindeki ünlem metnin parçası). Kalıcılık: `episteme` + `python_cortex` state_dict.
- `nora_broca.py`: özel ad/kısaltma yüzeyi okunuşa göre ek alır (`COVID-19'un`, `TBMM'ye`, `F-16'yı`); IMP (emir) kipi eklendi; `saat/sanat` gibi alıntılarda yumuşama yok.
- Testler: A36 (kör kabul yok → 2 hipotez), A37 (güvenilirlikle öğrenme, çelişki ezmez, System-1 terfi/iptal), A38 (yalnız güvenilir kaynak; güvenilmezden öğrenmeme), A39 (çalıştırarak doğrulanan kod + doğal tırnaksız istem + abstain), A40 (epistemik+kod kalıcılığı). Tam paket: **40/40, 168.7 sn**; A23 O(1) 3.89→5.21 ms.
- Canlı kanıt: gözlemevinde "Merhaba! Ey Ahali! yazdıracak tek satırlık bir Python kodu yaz." → `print("Merhaba! Ey Ahali!")`, çalıştırıldı, çıktı birebir doğrulandı; ikinci istemde System-1 (12 ms).
- Bilinen açık: durumsal nezaket korteksi hâlâ sözcük ölçütleriyle yönlendiriliyor (beyin-benzeri plastik sınıflandırıcıya taşınmalı); bu ortamda Wikipedia'ya ağ çıkışı yok (araştırmacı dürüstçe başarısız oluyor); eş ad birleştirme iki bağımsız güvenilir kaynak ister, tek kaynakla bilinçli olarak birleşmez.

## Güncel (tur: claude/clever-franklin-sz4mmr incelemesi + main'e birleştirme)
- İncelenen dal: `claude/clever-franklin-sz4mmr` (ed028e5 + 2018492, ata: fc6331c). Getirdikleri:
  `nora_voice.py` (LearnedVoice: dilden bağımsız alt-sözcük/n-gram söz organı, sözcük–iç durum çağrışımı),
  `nora_broca_net.py` (5,7M GRU artikülasyon motoru, tek geçiş çevrimiçi öğrenme, çözümlemede **yapısal sözcük maskesi** → uydurma imkânsız),
  `nora_life.py` (arka plan yaşamı: CA3 tarama, RK4 zihinsel gerilim, 5 popülasyonlu LCA ihalesi, Hipotez Havuzu=çapraz doğrulama,
  ReplyRepertoire=eksik tamamlama iradesi, metabolik THOUGHT_COST, outbox), `nora_egit.py` + `make_corpus_*.py` (eğitim hattı),
  `data/broca_tr.pt` (23 MB eğitilmiş ağırlık). Ayrıca `nora_lang/nora_broca`'da **kararlı FNV-1a hash** (PYTHONHASHSEED'e bağlı olmayan kalıcı bellek) ve OPT kipi düzeltmesi.
  Testler: A41–A47 (dilsiz organ sessiz / EN-TR taklidi yok, kodda Türkçe sabit yok, tek kaynak ağa girmiyor, 2 bağımsız kaynak terfi,
  eksik tamamlama, arka plan yaşamı, O(1) tick kalıcılık, Broca tek geçiş + maske, eğitilmiş Broca'nın dürüst duruşu).
- Birleştirme (merge commit `e0f0c58`). Birleşme sonrası **gerçek konuşma testi regresyon gösterdi**: dil hiç duyulmamışken
  (`data/broca_tr.pt` bağlı değil) söz organı birkaç turdan öğrendiği dizgilerle konuşuyordu →
  "Zürafa nedir?" → *"Benim adım ali."*, "Karadeniz hakkında ne biliyorsun?" → **girdinin yankısı**, "Kediler uçar mı?" → bağlantısız cümle.
  (main'de aynı sorular dürüst yedeği veriyordu: "X hakkında henüz bilgim yok; merak listeme aldım…")
- Düzeltme (bu tur): (1) `NoraAGI.voice_ready()` dil olgunluğu kapısı (`VOICE_READY_MATURE=0.35`, eğitilmiş organ varsa açık);
  (2) `_voice_usable()` kullanılabilirlik denetimi (boş/çok kısa söz, çapayı taşımayan söz, tekrar eden söz söylenmez);
  (3) `speak_or()` — organ hazır değilse *dürüst yedeğe* düşer; bilinmeyen kavramda organ sorusu varsa "bilgim yok" + kendi merak sorusu birleşir;
  (4) `nora_lang`/`nora_broca`: t>d yumuşaması **sözlüksel** oldu (`SOFTEN_T_VERBS`, `soften_verb_t`): hisset→hissediyorum, affet→affediyorum, kaydet→kaydediyorum; tüket/yönet/ilet yumuşamaz. Önceki hâlde "hissetiyorsun" üretiliyordu.
- Ölçüm: varsayılan yapılandırmada bilinmeyen soru 4–22 ms'de dürüst yedek; `data/broca_tr.pt` bağlıyken 130–260 ms ve cevap
  "Mitokondri hakkında henüz bilgim yok; merak listeme aldım. Mitokondri mi?" (dürüst + meraklı). İç ses: dil yoksa sessiz, organ varsa kısa ama gerçek.
- Tam paket: **A1–A47 47/47 GEÇTİ (277 sn)**; A23/A25 hız eşikleri donanıma bağlı (yük altında ara sıra düşebiliyor — bir koşuda 46/47, aynı koşuda torch indirmesi arka planda çalışıyordu; tekil tekrarda A44/A45/A46/A30 dört kez üst üste geçti).
- Gözlemevi `nora_studio.py` birleşik kodla çalışıyor: /api/chat 5 ms, arka plan yaşamı tick'i studio içinde de dönüyor.

