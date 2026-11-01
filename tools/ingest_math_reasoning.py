"""tools/ingest_math_reasoning.py

erayalp/medium_turkish_math_reasoning veri setini Hugging Face'den indirir,
matematiksel akıl yürütme adımlarını ve tanımlarını Nora'nın:
  1. Broca Konuşma / Artikülasyon Organına (Türkçe adım adım matematiksel akıl yürütme üslubu)
  2. Bilişsel Semantik Ağına (Kavramlar, formüller ve temel matematiksel tanımlar)
yüksek hızda işler ve 'state/nora_brain.nora' birleşik beyin dosyasına kalıcı olarak kaydeder.
"""
from __future__ import annotations

import argparse
import os
import sys
import re
import time
from pathlib import Path
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nora.cognition.agi import NoraAGI
from nora.language.voice import state_vector
from nora.language.lang import tr_lower

DATASET_REPO = "erayalp/medium_turkish_math_reasoning"
BRAIN_PATH = _REPO_ROOT / "state" / "nora_brain.nora"
OUT_TEXT_DIR = _REPO_ROOT / "data" / "text"
OUT_MATH_FILE = OUT_TEXT_DIR / "math_reasoning.txt"


def clean_text(text: str) -> str:
    text = re.sub(r"\r\n|\r", "\n", str(text or ""))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def download_dataset() -> pd.DataFrame:
    """Hugging Face'den parquet veya csv/json dosyasını bulup DataFrame olarak yükler."""
    print(f"Hugging Face Hub üzerinde '{DATASET_REPO}' taranıyor...")
    api = HfApi()
    files = api.list_repo_files(DATASET_REPO, repo_type="dataset")

    parquet_files = [f for f in files if f.endswith(".parquet")]
    if parquet_files:
        target = parquet_files[0]
        print(f"Parquet veri dosyası indiriliyor: {target}")
        local_path = hf_hub_download(repo_id=DATASET_REPO, filename=target, repo_type="dataset")
        return pd.read_parquet(local_path)

    data_files = [f for f in files if f.endswith(".csv") or f.endswith(".json") or f.endswith(".jsonl")]
    if data_files:
        target = data_files[0]
        print(f"Veri dosyası indiriliyor: {target}")
        local_path = hf_hub_download(repo_id=DATASET_REPO, filename=target, repo_type="dataset")
        if target.endswith(".csv"):
            return pd.read_csv(local_path)
        return pd.read_json(local_path, lines=target.endswith(".jsonl"))

    raise RuntimeError(f"Desteklenen formatta dosya bulunamadı. Repo dosyaları: {files}")


def parse_args():
    parser = argparse.ArgumentParser(description="Nora Math Reasoning Dataset Ingestion")
    parser.add_argument(
        "--max-facts",
        type=int,
        default=600,
        help="Semantik grafiğe işlenecek matematiksel kural/tanım sayısı (varsayılan: 600, 0 = tümü)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Broca artikülasyon modeli için batch büyüklüğü (varsayılan: 64)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="PyTorch iş parçacığı sayısı (varsayılan: 4)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)

    print("=" * 65)
    print(" Nora NCC - Türkçe Matematiksel Akıl Yürütme Entegrasyonu")
    print(f" Hedef: {DATASET_REPO}")
    print("=" * 65)

    # 1. Veri setini indir ve yükle
    df = download_dataset()
    print(f"Veri seti başarıyla yüklendi: {len(df)} adet matematik problemi ve çözümü.")

    q_col = next((c for c in ["question", "soru", "problem", "prompt"] if c in df.columns), df.columns[0])
    a_col = next((c for c in ["answer", "cevap", "solution", "cozum"] if c in df.columns), df.columns[1])
    print(f"Kullanılan sütunlar: Soru -> '{q_col}', Çözüm/Akıl Yürütme -> '{a_col}'")

    # 2. Akıl yürütme adımlarını ve tanımları ayıkla
    chat_turns: list[str] = []
    factual_candidates: list[str] = []
    seen = set()

    # Matematiksel tanım, kural ve bağıntı kalıpları
    math_rule_pattern = re.compile(
        r"(?: bir .*?(?:d[ıiırruü]r|t[ıiırruü]r)|eşittir|toplamıdır|farkıdır|katıdır|alanıdır|çevresidir|formülüdür|sayıdır|bağıntısıdır|özelliğidir)\b",
        re.IGNORECASE,
    )

    for _, row in df.iterrows():
        question = clean_text(row.get(q_col, ""))
        answer = clean_text(row.get(a_col, ""))

        if question:
            chat_turns.append(question)
        if answer:
            chat_turns.append(answer)
            # Çözüm içerisindeki kural ve adımları tara
            for s in re.split(r"(?<=[.!?…])\s+|\n+", answer):
                s = s.strip()
                words = s.split()
                # 4-16 kelimelik net ve odaklı yargı cümleleri
                if not (4 <= len(words) <= 16):
                    continue
                if s in seen or any(ch in s for ch in "()[]{}\"/'”’*#"):
                    continue
                if s.count(",") > 1 or ":" in s or ";" in s or s.startswith(("-", "•")):
                    continue

                if math_rule_pattern.search(s):
                    seen.add(s)
                    factual_candidates.append(s)

    OUT_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_MATH_FILE, "w", encoding="utf-8") as f:
        for turn in chat_turns:
            f.write(turn + "\n")

    print(f"Diyalog ve çözüm adımları kaydedildi: {OUT_MATH_FILE}")
    print(f"Toplam matematik metin parçası (Broca için): {len(chat_turns)}")
    print(f"Ayıklanan matematiksel tanım/kural adayı (Grafik için): {len(factual_candidates)}")

    if args.max_facts > 0 and len(factual_candidates) > args.max_facts:
        facts_to_teach = factual_candidates[:args.max_facts]
    else:
        facts_to_teach = factual_candidates

    # 3. Nora Kalıcı Zihnini Yükle
    print(f"\nNora kalıcı zihni yükleniyor: {BRAIN_PATH} ...")
    if BRAIN_PATH.is_file():
        agi = NoraAGI.load(str(BRAIN_PATH), attach_broca_async=False)
        print("Mevcut 'nora_brain.nora' başarıyla yüklendi.")
    else:
        print("Yeni Nora zihni oluşturuluyor...")
        agi = NoraAGI(name="Nora", seed=42, boot_thinker=False, single_life=True)

    print(f"Başlangıç semantik kenar sayısı: {len(agi.lang.graph.edges)}")

    # 4. Matematiksel Kuralları Semantik Hafızaya Öğret (1-shot)
    if facts_to_teach:
        print(f"\n[1/2] Matematik kuralları semantik hafızaya aktarılıyor ({len(facts_to_teach)} cümle)...")
        learned_edges = 0
        t0 = time.time()

        rmb = getattr(agi.lang.graph, "relation_mb", None)
        if rmb is not None:
            rmb._auto_commit = False

        try:
            for i, s in enumerate(facts_to_teach):
                try:
                    edges = agi.lang.teach_sentence(s, speaker="matematik_kaynagi")
                    if edges:
                        learned_edges += len(edges)
                except Exception:
                    pass

                if (i + 1) % 100 == 0 or (i + 1) == len(facts_to_teach):
                    elapsed = time.time() - t0
                    rate = (i + 1) / max(0.1, elapsed)
                    remaining = (len(facts_to_teach) - (i + 1)) / max(0.1, rate)
                    print(
                        f"  İlerleme: {i + 1:4d}/{len(facts_to_teach)} | "
                        f"Yeni Kenar: {learned_edges:4d} | "
                        f"Hız: {rate:.1f} c/s | Kalan: {remaining:.1f}s"
                    )
        finally:
            if rmb is not None:
                rmb._db.commit()
                rmb._auto_commit = True

        print(f"Semantik aktarım tamamlandı! Süre: {time.time() - t0:.1f}s | Toplam Kenar: {len(agi.lang.graph.edges)}")
    else:
        print("\n[1/2] Semantik kural süzgecinden geçen cümle bulunmadı, Broca adımına geçiliyor.")

    # 5. Broca Artikülasyon Organına Matematiksel Akıl Yürütme Dilini Akıt
    organ = getattr(agi.life.voice, "organ", None)
    if organ is not None:
        print(f"\n[2/2] Broca konuşma organına adım adım matematik dili aktarılıyor ({len(chat_turns)} metin)...")
        t_broca = time.time()
        st = state_vector(agi)
        B = args.batch_size
        for i in range(0, len(chat_turns), B):
            batch = chat_turns[i : i + B]
            try:
                organ.learn_batch(batch, st, anchor_dropout=0.1, loss_mode="sampled", negative_samples=512)
            except Exception:
                pass
            if (i // B + 1) % 10 == 0 or (i + B) >= len(chat_turns):
                progress = min(i + B, len(chat_turns))
                print(f"  Broca ilerleme: {progress:5d}/{len(chat_turns)} ({progress*100//len(chat_turns)}%)", flush=True)

        organ.update_common()
        print(f"Broca eğitimi tamamlandı! Süre: {time.time() - t_broca:.1f}s")

    # 6. Beyni Kalıcı Olarak Kaydet (state/nora_brain.nora)
    print(f"\nBütün beyin güncellenerek kalıcı olarak kaydediliyor: {BRAIN_PATH} ...")
    agi.save(str(BRAIN_PATH))
    print(f"Başarıyla kaydedildi ({BRAIN_PATH})! Matematik akıl yürütme kabiliyeti kalıcı hale geldi.")

    # 7. Örnek Matematik Test Soruları
    print("\n--- Örnek Matematik Muhakeme Testi ---")
    test_prompts = [
        "15 ile 27'yi toplarsak kaç eder?",
        "Bir sınıfta 12 kız ve 18 erkek öğrenci varsa toplam kaç öğrenci vardır?",
        "Bir dikdörtgenin alanı nasıl hesaplanır?",
    ]
    for q in test_prompts:
        ans = agi.converse(q, speaker="insan")
        print(f"Soru: {q}\nNora > {ans}\n")


if __name__ == "__main__":
    main()
