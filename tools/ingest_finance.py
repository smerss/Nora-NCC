"""
umarigan/turkiye_finance_qa veri setini çeker,
soru-cevap çiftlerindeki finansal tanımları ve olguları Nora'nın semantik zihnine (state/nora_sozcu) işler ve kalıcı kaydeder.
Ayrıca Broca artikülasyon modeline finans terimlerinin gramerini akıtır.
"""
import os
import sys
import re
import time
import pandas as pd
from huggingface_hub import hf_hub_download

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from nora.cognition.agi import NoraAGI
from nora.language.voice import state_vector
from nora.language.lang import tr_lower

DATASET_REPO = "umarigan/turkiye_finance_qa"
DATASET_FILE = "data/train-00000-of-00001-edba223bc265f4b0.parquet"
CKPT_PATH = os.path.join(_REPO_ROOT, "state", "nora_sozcu")
BROCA_PATH = os.path.join(_REPO_ROOT, "data", "broca_tr_sozcu.pt")
OUT_TEXT = os.path.join(_REPO_ROOT, "data", "text", "finance_qa.txt")


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text


def main():
    print(f"Hugging Face üzerinden '{DATASET_REPO}' indiriliyor...")
    file_path = hf_hub_download(repo_id=DATASET_REPO, filename=DATASET_FILE, repo_type="dataset")
    df = pd.read_parquet(file_path)
    print(f"Veri seti yüklendi: {len(df)} adet finansal Soru-Cevap çifti.")

    # 1. Metinleri cümlelere ayır
    sentences = []
    plain_texts = []
    for _, row in df.iterrows():
        ans = clean_text(row.get("cevap", ""))
        q = clean_text(row.get("soru", ""))
        if ans:
            plain_texts.append(ans)
            # Cümlelere böl
            for s in re.split(r"(?<=[.!?…])\s+", ans):
                s = s.strip()
                if len(s.split()) >= 3:
                    sentences.append(s)

    # Düz metin olarak da kaydet
    os.makedirs(os.path.dirname(OUT_TEXT), exist_ok=True)
    with open(OUT_TEXT, "w", encoding="utf-8") as f:
        for pt in plain_texts:
            f.write(pt + "\n")
    print(f"Toplam {len(sentences)} adet cümle çıkarıldı ve {OUT_TEXT} dosyasına yazıldı.")

    # 2. Kalıcı Zihni Yükle
    print(f"\nNora zihni yükleniyor: {CKPT_PATH}...")
    t0 = time.time()
    agi = NoraAGI.load(CKPT_PATH)
    print(f"Zihin hazır ({time.time()-t0:.2f}s). Mevcut kenar sayısı: {len(agi.lang.graph.edges)}")

    # 3. Finansal Olguları Semantik Hafızaya Öğret (1-shot)
    print("\nFinansal tanımlar ve olgular semantik grafiğe işleniyor...")
    learned_edges = 0
    t_start = time.time()
    for i, s in enumerate(sentences):
        try:
            edges = agi.lang.teach_sentence(s, speaker="finans_kaynagi")
            learned_edges += len(edges)
        except Exception:
            pass

        if (i + 1) % 100 == 0 or (i + 1) == len(sentences):
            print(f"İlerleme: {i + 1}/{len(sentences)} cümle | Yeni Bilişsel Bağlantı: {learned_edges}")

    print(f"Semantik aktarım tamamlandı! Eklenen bağlantı: {learned_edges} (Süre: {time.time()-t_start:.2f}s)")
    print(f"Toplam Güncel Kenar Sayısı: {len(agi.lang.graph.edges)}")

    # 4. Broca Organına da Finans Dilini Akıt (Eğer varsa)
    organ = getattr(agi.life.voice, "organ", None)
    if organ is not None:
        print("\nBroca konuşma organına finans terminolojisi akıtılıyor...")
        st = state_vector(agi)
        B = 32
        for i in range(0, len(sentences), B):
            batch = sentences[i : i + B]
            try:
                organ.learn_batch(batch, st, anchor_dropout=0.1, loss_mode="sampled", negative_samples=512)
            except Exception:
                pass
        organ.update_common()
        if os.path.exists(BROCA_PATH):
            organ.save(BROCA_PATH)
            print(f"Güncel Broca organı kaydedildi: {BROCA_PATH}")

    # 5. Kalıcı Olarak Kaydet
    print(f"\nZihin güncellenerek kalıcı kaydediliyor: {CKPT_PATH}...")
    agi.save(CKPT_PATH)
    print("Kaydedildi!")

    # 6. Test Çıktıları
    print("\n--- Finans Test Soruları ---")
    test_queries = [
        "FED nedir?",
        "Borsa nedir?",
        "Nasdaq nedir?",
        "Enflasyon nedir?",
        "Hisse senedi nedir?",
        "Tahvil nedir?",
        "Likidite nedir?",
        "Faiz nedir?",
    ]
    for q in test_queries:
        ans = agi.converse(q, speaker="insan")
        print(f"Soru: {q}\nNora > {ans}\n")


if __name__ == "__main__":
    main()
