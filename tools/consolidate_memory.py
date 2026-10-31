"""
Sözcü 2014 haber metinlerindeki temel olguları ve tanımları Nora'nın semantik/biyolojik hafızasına (Kenyon Cells / Knowledge Graph)
tek seferde işler ve 'state/nora_sozcu' checkpoint'i olarak kaydeder.
"""
import os
import sys
import re
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from nora.cognition.agi import NoraAGI
from nora.synthesis.egit import sentences_of


def main():
    data_path = os.path.join(_REPO_ROOT, "data", "text", "sozcu_2014.txt")
    out_dir = os.path.join(_REPO_ROOT, "state")
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(out_dir, "nora_sozcu")
    broca_path = os.path.join(_REPO_ROOT, "data", "broca_tr_sozcu.pt")

    print(f"Haber cümleleri taranıyor: {data_path} ...")
    raw_sents = sentences_of([data_path])
    print(f"Toplam {len(raw_sents)} cümle bulundu.")

    # Tanım ve temel bildirim/olgu bildiren cümleleri filtrele (-dır, -dir, -tir, -tur, vb. veya temel yüklemler)
    # Nora'nın epistemik motoru özellikle tanım ve eylem bildiren net cümleleri güçlü kaydeder.
    print("Hafızaya aktarılacak temel olgu ve tanım cümleleri süzülüyor...")
    facts = []
    seen = set()

    # Öncelikli çekirdek olgular (haber bağlamını sabitlemek için)
    core_facts = [
        "Türkiye bir cumhuriyettir.",
        "Hükümet devleti yöneten yürütme organıdır.",
        "Meclis yasaları çıkaran kurumdur.",
        "Seçim halkın yöneticilerini belirlemesidir.",
        "Ekonomi üretim, tüketim ve ticaret düzenidir.",
        "Enflasyon fiyatların genel düzeyindeki artıştır.",
        "Anayasa devletin temel kanunudur.",
        "Bakanlar Kurulu hükümetin icra kuruludur.",
        "Yargı bağımsız mahkemelerce kullanılan adalettir.",
        "Vatandaş devletin yurttaşıdır.",
        "Parti siyasi görüşleri temsil eden kuruluştur.",
        "Muhalefet iktidarı denetleyen siyasi güçtür.",
    ]
    for cf in core_facts:
        facts.append(cf)
        seen.add(cf)

    # Haber külliyatından tanım, nedensellik ve güçlü olgu cümlelerini topla
    pattern = re.compile(r"(?:d[ıiırruü]r|t[ıiırruü]r|yönetir|sağlar|belirler|oluşturur|kabul\s+etti|karar\s+verdi)\b", re.IGNORECASE)
    
    for s in raw_sents:
        s = s.strip()
        words = s.split()
        if 4 <= len(words) <= 18 and pattern.search(s):
            if s not in seen and not any(ch in s for ch in "()[]{}\"/'”’"):
                seen.add(s)
                facts.append(s)
                if len(facts) >= 1200:  # Hızlı ve temiz bir konsolidasyon için en kaliteli 1200 olgu
                    break

    print(f"Toplam {len(facts)} adet net olgu ve tanım cümlesi hafızaya işlenmek üzere seçildi.")

    print("\nNora bilişsel mimarisi başlatılıyor...")
    agi = NoraAGI(
        name="Nora",
        seed=42,
        boot_thinker=False,
        single_life=True,
        broca=broca_path if os.path.exists(broca_path) else None,
        code_pretrain=False,
    )

    t0 = time.time()
    learned_edges = 0
    print("Biyolojik hafıza aktarımı (Kenyon Hücreleri & Knowledge Graph) yapılıyor...")

    for i, fact in enumerate(facts):
        try:
            edges = agi.lang.teach_sentence(fact, speaker="kaynak")
            if edges:
                learned_edges += len(edges)
        except Exception:
            pass
        if (i + 1) % 200 == 0 or (i + 1) == len(facts):
            print(f"İlerleme: {i + 1}/{len(facts)} cümle | Eklenen Bilişsel Bağlantı: {learned_edges}")

    print(f"\nHafıza aktarımı tamamlandı! Süre: {time.time() - t0:.2f}s")
    print(f"Toplam Graftaki Kenar Sayısı: {len(agi.lang.graph.edges)}")

    print(f"\nZihin durumu '{checkpoint_prefix}' olarak donduruluyor (checkpoint)...")
    agi.save(checkpoint_prefix)
    print("Checkpoint başarıyla kaydedildi!")

    print("\n--- Test Soruları ---")
    test_queries = [
        "Türkiye nedir?",
        "Hükümet nedir?",
        "Seçim nedir?",
        "Meclis nedir?",
        "Ekonomi nedir?",
        "Enflasyon nedir?"
    ]
    for q in test_queries:
        ans = agi.converse(q, speaker="insan")
        print(f"Soru: {q}\nNora > {ans}\n")


if __name__ == "__main__":
    main()
