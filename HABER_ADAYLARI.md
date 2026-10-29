# Haber çıkarabilecek proje adayları (smerss hesabı)

Bakılanlar: `clausewitz-mcp`, `bloodmarch` (+ `bloodmarch-game`), `aoh3-multiplayer`, `ncc`, `wrath`, `again`, bu proje (NoRa). Özel görünen kişisel bir depoya bakılmadı.

## 1. Clausewitz-MCP — en güçlü aday
"Yapay zekâ, Hearts of Iron IV'ü senin bastığın düğmelere basarak oynasın." 107 MCP aracı; süreç belleği okuma + kendi komut kuyruğu üzerinden emir verme; muharebe/arazi/kuşatma planlayıcısı; MIT; Linux.
- **Neden haber:** "Bir LLM bir büyük strateji oyununu baştan sona kendi başına oynayıp kazandı" tek cümlelik, görsel, paylaşılabilir bir sonuç. Çıta düşük ama etki yüksek.
- **Haber olması için gereken:** (a) kayıtlı, tekrarlanabilir bir tam oyun (ör. belirli bir senaryoda X yılında zafer) ve zaman çizelgesi videosu; (b) insan oyuncu/yerleşik YZ ile ölçülebilir karşılaştırma; (c) yol haritasındaki Windows portu (çoğu oyuncu orada) — "50 yıldız" eşiği için ilk lansman haberi önemli.
- **Riskler:** oyunun EULA'sı ve bellek okuma/yazma anti-hile sayılabilir; yalnızca tek oyunculu kullanımı vurgulayıp çok oyunculu/başarım (achievement) etkisini açıkça dışlamak gerekir. Hukuki iddia yok, yalnızca dikkat notu.

## 2. NoRa (bu proje) — en derin ama en zor anlatılan
Nora, mantar cismi ilkesiyle çalışan biyolojik esinli bir zihin: seyrek Kenyon-hücre temsili ve dopamin-kapılı yerel öğrenme + homeostatik değer + kendi iç durumundan türetilen "yaşıyorum" raporu; nedensel müdahale ile papağana karşı doğrulanmış.
- **Haber olabilecek cümle:** "Yaşadığını *öğretilmediği hâlde* iç durumundan türeten, nedensel olarak sınanabilir bir ajan." Bilinç iddiası YOK; bu kasıtlı ve doğru olan tek çerçeve.
- **Gereken:** küçük, tekrar edilebilir bir demo + dürüst olumsuz sonuçlar bölümü (zaten `NORA_EVREN.md` içinde) + tek komutla yeniden üretim. Akademik ilgi (arXiv) ticari haberden daha olası.
- **Güncel teknik dayanak:** ana pil 140/140; A141 kavram/kalıcılık, A142 nöral top-5→sembolik kanıt, A143 hedef yaşam döngüsü ve CCG'de counter16/24 genellemesi. FST arama maliyeti ve weighted P0 hâlâ açık; genel zekâ veya bilinç iddiası yok.

## 3. Age of Bloodmarch (Unity 6, ~175 bin satır C#)
Büyük emek, ayrıntılı README. Haber değeri oyunun yayınlanması/oynanabilir demo ve çıkış videosuna bağlı; bugünkü hâliyle "haber" değil, "ürün". Tüm hakları saklıdır, açık kaynak değil.

## 4. aoh3-multiplayer
Resmî olmayan çok oyunculu yama. İlgi çekici ama telif/ToS riski yüksek; öne çıkarmadan önce hukuki durum netleştirilmeli. Önerim: şimdilik haber hedefi yapma.

## 5. ncc, wrath, again
`ncc`: 7 küçük PyTorch dosyası (NoRa'nın öncül hattı). `wrath`: yalnızca iki .exe (bilinmeyen ikili dosyaları paylaşmak/öne çıkarmak önerilmez). `again`: boş.

## Öneri
Kısa vadede: **Clausewitz-MCP** için tek bir tam, kayıtlı oyun + video + Windows port duyurusu. Uzun vadede: **NoRa** için arXiv raporu. İkisi birbirini besler: "LLM kendi emirlerini ve bellek durumunu okuyarak oynuyor" ↔ "ajan iç durumunu okuyarak kendini raporluyor".
