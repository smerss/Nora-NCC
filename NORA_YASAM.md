# Nora: arka plan yaşamı, öğrenen söz, eksik tamamlama

İlke: altyapıyı kurmak, içeriği Nora'nın yaşayarak/okuyarak edinmesine bırakmak. Kodda söz, kalıp ya da
"şunu derse şunu de" kuralı yoktur. Hata düzeltmek için de kalıp eklenmez; düzeltme ancak öğrenme yoluyla gelir.

| dosya | rol |
|---|---|
| `nora_voice.py` | `LearnedVoice`: dilden bağımsız söz organı. Sözcük–iç durum çağrışımı ve geçişleri yalnızca duyulan metinden öğrenilir (`corpus=` / konuşmalar). Dil duyulmadıysa sessizdir. |
| `nora_life.py` | `BackgroundLife`: arka plan döngüsü (`step_life` içinde döner). A22 CA3 tarama + özet, zihinsel gerilim (RK4), 5 popülasyonlu LCA ihalesi (dinlen / pekiştir / prova / araştır / seslen), `spontaneous_action` (outbox). |
| `nora_life.HypothesisPool` | Çapraz doğrulama: internet iddiası önce hipotez; ≥2 bağımsız kurum (alan adı) + kesinlik ile kavram ağına terfi. |
| `nora_life.ReplyRepertoire` | Girdiyi tanıyıp karşılığını bilmeme boşluğu; karşılık iki bağımsız kaynakta görülünce yeterlilik doğar. |
| `nora_life_experiment.py` | Yoğun girdi → sessizlik → kararlılık deneyi. |

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

## Dürüst sınırlar
* Söz organı küçük/teknik bir metinle bebek gibi konuşur (anlamsız ya da bozuk cümleler normaldir); kaliteyi veri ve yaşantı belirler. Düzgün düzyazı metinle çok daha iyi olur.
* Eski çekirdekte hâlâ elle yazılmış parçalar var (reddetme gerekçeleri, selamlaşma/empati kortekslerinin sözcük seçimi, Türkçe ayrıştırıcı). Bunlar öğrenilmiş katmana taşınmadı.
* Hedef-eşiği gibi birkaç geliştirici sabiti kaldı (ör. terfi için 2 bağımsız kaynak, kesinlik 0.74, `IDLE_TAU`). Bunlar ileride Nora'nın kendi deneyiminden ayarlanmalı.
* `A23`/`A25` süre eşikleri donanıma bağlıdır; bu makinede değişikliklerden önce de geçmiyordu.
