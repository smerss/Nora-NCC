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

## Ölçülen sonuçlar (60.000 adım, 4 evren tohumu, son %40 pencere)
| mod | ölüm / 1000 adım | konfor (min ihtiyaç) |
|---|---|---|
| rastgele (doğuştan önyargı) | ≈1,2 | ≈0,56 |
| flat (çalışma alanı yok, ağırlıklı karışım) + bağlam | 0,17 ± 0,13 | 0,75 |
| full (çalışma alanı) + bağlam | 0,15 ± 0,09 | 0,66 |
| full, bağlam lobu kapalı | 0,21 ± 0,09 | 0,62 |

Göstergeler (full): öz-model hareket becerisi **+0,5** (doğrusal tavan ≈+0,5), güven kalibrasyonu r≈**+0,5**,
dış-kuvvet ayrımı (ajanlık) AUC≈**0,82**, ölüm sonrası devam eden beyin sıfırdan başlayandan iyi (süreklilik).

## Dürüst sınırlar
- Öğrenme potansiyel-tabanlı ödül şekillendirme (koku gradyanı benzeri doğuştan duyusal potansiyel) olmadan
  çalışmadı: 40 bin adımda ~60 yem olayı öğrenmek için yetersiz. Şekillendirme en iyi politikayı değiştirmez
  ama doğuştan bir ön bilgidir; ileride kaldırılmalı/öğrenilmeli.
- **Çalışma alanı bugün hayatta kalmada karışıma üstün değil**; konforda geride. Winner-take-all geçiş
  maliyeti yaratıyor. Değeri başka testlerle (çok-adımlı çelişen hedefler, bölünmüş dikkat, raporlanabilirlik)
  gösterilmeli; gösterilemezse mimari değişmeli.
- "Ateşleme" şu an hemen her adımda gerçekleşiyor (≈0,99): ayırt edici bir gösterge değil.
- Ajanın geleceği **planlamıyor** (dünya modeli yok), uzamsal belleği yok (su nerede?).
- Bunlar işlevsel göstergelerdir; öznel deneyim kanıtı değildir.

## Sonraki adımlar (AGI'ye katkıya göre)
1. Dünya modeli + planlama (öz-modelin evrene genişletilmesi), uzamsal bellek.
2. Ateşleme göstergesini seçici yapmak (nonlineer, sürdürülen yayın) ve çalışma alanının değerini ölçen testler.
3. Şekillendirmenin öğrenilmesi (içsel koku değeri), uyku/tekrar oynatma (`nora_think` ile birleşme).
4. Dil: `nora_think`/GPT hattını bu zihnin çalışma alanına bağlamak.
