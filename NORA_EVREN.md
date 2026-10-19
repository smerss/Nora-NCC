# NoRa Evreni ve Yaşayan Zihin

Hedef: AGI'ye giden yolda, bilinç kuramlarının **ortak** aradığı dört bileşeni tek kapalı döngüde kurmak
ve her birini ölçülebilir bir göstergeye bağlamak. Bu bir simülasyon oyunu değil, NoRa'nın yaşadığı ortam:
beyin ölümler arasında kesintisiz sürer, öğrenme asla sıfırlanmaz.

## Dosyalar
| dosya | rol |
|---|---|
| `nora_world.py` | Evren: Newton fiziği, sürtünme, momentum korunumlu çarpışma, itilebilir kaya, gün/gece, rüzgâr, metabolizma. Kalın-ışın görme (9×5 kanal), iç duyu, propriosepsiyon. `python nora_world.py` fizik öz-sınaması |
| `nora_neuron.py` | Nöron temeli: FlyWire konektomu için LIF simülatörü (Brian2 referansıyla doğrulandı), `MushroomBody`, `LobedMushroomBody` |
| `nora_mind.py` | Zihin: lobülü mantar cismi + uzman Q-kafaları + global çalışma alanı + valans/ruh hali + sürekli iç durum + öz-model |

## Dört bileşen nasıl kuruldu
1. **Kapalı döngü**: algı → karar → eylem → evren → algı. Eylem kopyası (efference) iç duruma geri beslenir.
2. **Değer/valans**: homeostatik hata H = Σ(1−ihtiyaç)². Duygu-tonu = −ΔH (ihtiyaç giderilince iyi, yara
   kötü). Ruh hali = yavaş ortalama; kötü ruh hali keşfi artırır. Ödül dışarıdan verilmez, bedenden doğar.
3. **Global çalışma alanı**: dört uzman (aç, susuz, tehlike, merak) aynı seyrek kodu okur, yanal baskılamalı
   sızıntılı toplayıcılarda yarışır, kazanan eşik+marj aşarsa ateşler ve amacını tüm sisteme yayınlar.
4. **Süreklilik/öz-model**: hiç sıfırlanmayan tekrarlayan durum `h` (hem mantar cismine bağlam lobu olarak
   girer hem öz-modele). Öz-model eyleme koşullu hareket değişimini öngörür; hatadan şaşkınlık, ajanlık
   (öz/dış ayrımı) ve üstbilişsel güven çıkar.

Öğrenme: dopamin(TD hatası)-kapılı, yalnızca aktif Kenyon hücreleri değişir (sinek mantar cismi kuralı).
Nöron temeli: loblu seyrek kod; her duyusal kanalın kendi KC popülasyonu ve heterojen eşikleri var.

## Ölçülen sonuçlar (60.000 adım, 8 evren tohumu, son %40 pencere)
| mod | ölüm / 1000 adım | konfor | ölüm nedenleri (toplam) |
|---|---|---|---|
| rastgele | 1,11 ± 0,20 | 0,56 | yara 138, susuzluk 62, açlık 22 |
| **full** (çalışma alanı + harita/plan) | **0,08 ± 0,04** | 0,70 | yara 10, susuzluk 4, açlık 2 |
| full, haritasız | 0,12 ± 0,10 | 0,68 | susuzluk 15, açlık 6, yara 3 |
| flat (çalışma alanı yok) + harita/plan | 0,23 ± 0,09 | 0,73 | yara 35, açlık 9, susuzluk 2 |
| flat, haritasız | 0,15 ± 0,11 | 0,75 | yara 20, susuzluk 6, açlık 3 |

Okuma:
- **Harita susuzluk ölümlerini bitiriyor** (15 → 4 çalışma alanlıda): suyun yerini öğrenip görünmediğinde rota çiziyor.
- **Çalışma alanı, planlama varken belirgin üstün** (0,08 vs 0,23, ~3 sd): hedefler çatıştığında (plan tehlikeye
  doğru çekerken) kazananın kontrolü alması tehlike uzmanının planı ezmesini sağlıyor; karışım bunu yapamıyor
  (flat'ta harita eklemek ölüm oranını 0,15 → 0,23 kötüleştirdi). Yani çalışma alanının işlevsel değeri
  *çatışma çözümü*; planlama olmadan görünmüyordu.
- Haritasız ablasyonun etkisi ölüm toplamında gürültü içinde; nedenlerde (susuzluk) net.
- Konum bilgisi VERİLMİYOR: pusula + hızdan yol integrasyonu + duvar çapası ile konum hatası ≈0,5 birim
  (arena 24). Rastgele politikada bu bile 1,6 (çapa az duvar görüyor).
- Öz-model hareket becerisi ≈+0,5–0,68 (doğrusal tavan ≈+0,5), güven kalibrasyonu r≈+0,5, ajanlık AUC≈0,8.

## Dürüst sınırlar
- Öğrenme potansiyel-tabanlı ödül şekillendirme (koku gradyanı benzeri doğuştan duyusal potansiyel) olmadan
  çalışmadı: 40 bin adımda ~60 yem olayı öğrenmek için yetersiz. Şekillendirme en iyi politikayı değiştirmez
  ama doğuştan bir ön bilgidir; ileride kaldırılmalı/öğrenilmeli.
- Çalışma alanı karışıma göre konforda hâlâ biraz geride (0,70 vs 0,73): kazanan-hepsini-alır geçiş maliyeti var.
- "Ateşleme" şu an hemen her adımda gerçekleşiyor (≈0,99): ayırt edici bir gösterge değil.
- Planlama yalnızca haritada rota arama; öğrenilmiş bir dünya modeli (algı öngörüsü, hayal gücü) yok.
- Harita ızgarası ve arena boyutu (24, 12x12) ajana doğuştan verilmiş; ölçek öğrenilmiyor.
- Bunlar işlevsel göstergelerdir; öznel deneyim kanıtı değildir.

## Sonraki adımlar (AGI'ye katkıya göre)
1. Öğrenilmiş dünya modeli + hayal gücü (algı öngörüsü), değişen evrende (kaynaklar taşınır) harita güncelleme.
2. Ateşleme göstergesini seçici yapmak (nonlineer, sürdürülen yayın) ve çalışma alanının değerini ölçen testler.
3. Şekillendirmenin öğrenilmesi (içsel koku değeri), uyku/tekrar oynatma (`nora_think` ile birleşme).
4. Dil: `nora_think`/GPT hattını bu zihnin çalışma alanına bağlamak.
