"""tools/ingest_sixfinger.py

sixfingerdev/chatbot-turkish-dataset-sixfinger-2b veri setini Hugging Face'den çeker,
diyalogları ve olguları Nora'nın:
  1. Bilişsel Semantik Ağına (Kenyon Hücreleri & Knowledge Graph - olgusal/tanımsal bilgi)
  2. Broca Konuşma / Artikülasyon Organına (Türkçe sohbet akışı, üslup ve kelime haznesi)
tek seferde (online/continual, epoch-free) işler ve 'state/nora_brain.nora' birleşik
beyin dosyasına kalıcı (permanent) olarak kaydeder.
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

DATASET_REPO = "sixfingerdev/chatbot-turkish-dataset-sixfinger-2b"
BRAIN_PATH = _REPO_ROOT / "state" / "nora_brain.nora"
OUT_TEXT_DIR = _REPO_ROOT / "data" / "text"
OUT_CHAT_FILE = OUT_TEXT_DIR / "sixfinger_chat.txt"


def clean_text(text: str) -> str:
    text = re.sub(r"\r\n|\r", "\n", str(text or ""))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()

def download_dataset() -> pd.DataFrame:
    """Hugging Face'den parquet veya json dosyasını bulup pandas DataFrame olarak yükler."""
    cached_candidates = list(Path.home().glob(".cache/huggingface/hub/datasets--sixfingerdev--chatbot-turkish-dataset-sixfinger-2b/**/*.json"))
    if cached_candidates:
        print(f"Yerel önbellekten veri dosyası yükleniyor: {cached_candidates[0]}")
        return pd.read_json(cached_candidates[0])

    print(f"Hugging Face Hub üzerinde '{DATASET_REPO}' taranıyor...")
    api = HfApi()
    files = api.list_repo_files(DATASET_REPO, repo_type="dataset")
    
    parquet_files = [f for f in files if f.endswith(".parquet")]
    if parquet_files:
        target = parquet_files[0]
        print(f"Parquet veri dosyası indiriliyor: {target}")
        local_path = hf_hub_download(repo_id=DATASET_REPO, filename=target, repo_type="dataset")
        return pd.read_parquet(local_path)

    data_files = [f for f in files if f.endswith(".json") or f.endswith(".jsonl") or f.endswith(".csv")]
    if data_files:
        target = data_files[0]
        print(f"Veri dosyası indiriliyor: {target}")
        local_path = hf_hub_download(repo_id=DATASET_REPO, filename=target, repo_type="dataset")
        if target.endswith(".csv"):
            return pd.read_csv(local_path)
        return pd.read_json(local_path, lines=target.endswith(".jsonl"))

    raise RuntimeError(f"Desteklenen formatta veri dosyası bulunamadı. Mevcut dosyalar: {files}")


def parse_args():
    parser = argparse.ArgumentParser(description="Nora Sixfinger Dataset Ingestion")
    parser.add_argument(
        "--max-facts",
        type=int,
        default=1200,
        help="Semantik grafiğe işlenecek en kaliteli net olgu/tanım sayısı (varsayılan: 1200, 0 = tümü)",
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
    print(" Nora NCC - Hızlı Sixfinger Türkçe Chatbot Entegrasyonu")
    print(f" Ayarlar: Max Olgu = {args.max_facts} | Broca Batch = {args.batch_size} | Threads = {args.threads}")
    print("=" * 65)

    # 1. Veri setini yükle
    df = download_dataset()
    print(f"Veri seti başarıyla yüklendi: {len(df)} satır.")

    input_col = next((c for c in ["input", "instruction", "soru", "prompt"] if c in df.columns), df.columns[0])
    output_col = next((c for c in ["output", "response", "cevap", "completion"] if c in df.columns), df.columns[1])

    # 2. Cümleleri ve diyalogları ayıkla
    chat_turns: list[str] = []
    factual_candidates: list[str] = []
    seen = set()

    # Sohbet dolgu kelimeleri (semantik hafızayı kirletmemesi gereken birinci/ikinci şahıs ifadeleri)
    chat_stopwords = {
        "ben", "bana", "benim", "sen", "sana", "senin", "biz", "bizim", "siz", "sizin",
        "merhaba", "selam", "teşekkür", "lütfen", "yardımcı", "ederim", "edebilirim",
        "üzgünüm", "anladım", "tabii", "elbette", "harika", "güzel bir", "zevk",
    }

    # Net tanım ve nedensellik kalıpları ("X bir Y'dir", "... sağlar", "... oluşturur")
    def_pattern = re.compile(
        r"(?: bir .*?(?:d[ıiırruü]r|t[ıiırruü]r)|sağlar|belirler|oluşturur|yönetir|türüdür|parçasıdır|içerir|demektir)\b",
        re.IGNORECASE,
    )

    for _, row in df.iterrows():
        user_in = clean_text(row.get(input_col, ""))
        asst_out = clean_text(row.get(output_col, ""))

        if user_in:
            chat_turns.append(user_in)
        if asst_out:
            chat_turns.append(asst_out)
            # Yanıtlardaki cümleleri süz
            for s in re.split(r"(?<=[.!?…])\s+|\n+", asst_out):
                s = s.strip()
                words = s.split()
                # 4 ila 16 kelime arası özlü cümleler (Wernicke ve epistemik motor için en yüksek verim)
                if not (4 <= len(words) <= 16):
                    continue
                if s in seen or any(ch in s for ch in "()[]{}\"/'”’*#"):
                    continue

                lowered = tr_lower(s)
                # Sohbet dolgusu içeren cümleleri semantik olgu listesinden ele
                if any(re.search(rf"\b{re.escape(sw)}\b", lowered) for sw in chat_stopwords):
                    continue

                # Çoklu virgül, iki nokta veya madde imleri Wernicke'nin kombinatoryal dallanmasını tetikler.
                # En temiz ve hızlı 1-shot öğrenim tek yargılı doğrudan cümlelerden elde edilir.
                if s.count(",") > 1 or ":" in s or ";" in s or s.startswith(("-", "•")):
                    continue

                if def_pattern.search(s):
                    seen.add(s)
                    factual_candidates.append(s)

    OUT_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CHAT_FILE, "w", encoding="utf-8") as f:
        for turn in chat_turns:
            f.write(turn + "\n")

    print(f"Toplam diyalog parçası: {len(chat_turns)}")
    print(f"Ayıklanan yüksek verimli tanım/olgu adayı: {len(factual_candidates)}")

    # Hedef olgu sayısını sınırla (Hızlı ve yüksek yoğunluklu semantik aktarım)
    if args.max_facts > 0 and len(factual_candidates) > args.max_facts:
        facts_to_teach = factual_candidates[:args.max_facts]
        print(f"En kaliteli {len(facts_to_teach)} olgu işleme alındı (hızlı mod).")
    else:
        facts_to_teach = factual_candidates

    # 3. Nora Zihnini Yükle
    print(f"\nNora kalıcı zihni yükleniyor: {BRAIN_PATH} ...")
    if BRAIN_PATH.is_file():
        agi = NoraAGI.load(str(BRAIN_PATH), attach_broca_async=False)
        print("Mevcut 'nora_brain.nora' başarıyla yüklendi.")
    else:
        print("Yeni Nora zihni oluşturuluyor...")
        agi = NoraAGI(name="Nora", seed=42, boot_thinker=False, single_life=True)

    print(f"Başlangıç semantik kenar sayısı: {len(agi.lang.graph.edges)}")

    # 4. Semantik Bilişsel Hafızaya Öğret (1-shot / Epistemic Graph & Kenyon Cells)
    print(f"\n[1/2] Olgu ve tanımlar semantik hafızaya aktarılıyor (Hedef: {len(facts_to_teach)} cümle)...")
    learned_edges = 0
    t0 = time.time()

    # Toplu aktarım sırasında SQLite disk fsync (commit) döngüsünü durdur, bellekte hızlandır
    rmb = getattr(agi.lang.graph, "relation_mb", None)
    if rmb is not None:
        rmb._auto_commit = False

    try:
        for i, s in enumerate(facts_to_teach):
            try:
                edges = agi.lang.teach_sentence(s, speaker="sixfinger_corpus")
                if edges:
                    learned_edges += len(edges)
            except Exception:
                pass

            if (i + 1) % 200 == 0 or (i + 1) == len(facts_to_teach):
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

    # 5. Broca Konuşma Organına Diyalog Akışını Eğit (Online/Continual)
    organ = getattr(agi.life.voice, "organ", None)
    if organ is not None:
        print(f"\n[2/3] Broca konuşma organına Türkçe diyalog yapısı aktarılıyor ({len(chat_turns)} parça)...")
        t_broca = time.time()
        st = state_vector(agi)
        B = args.batch_size
        for i in range(0, len(chat_turns), B):
            batch = chat_turns[i : i + B]
            try:
                organ.learn_batch(batch, st, anchor_dropout=0.1, loss_mode="sampled", negative_samples=512)
            except Exception:
                pass
            if (i // B + 1) % 20 == 0 or (i + B) >= len(chat_turns):
                progress = min(i + B, len(chat_turns))
                print(f"  Broca ilerleme: {progress:5d}/{len(chat_turns)} diyalog ({progress*100//len(chat_turns)}%)", flush=True)

        organ.update_common()
        print(f"Broca eğitimi tamamlandı! Süre: {time.time() - t_broca:.1f}s")

    # 6. Epizodik Diyalog Hafızasına (CA3 RAG/Retrieval) Soru-Cevap Çiftlerini İndeksle
    print(f"\n[3/3] Soru-cevap çiftleri epizodik diyalog hafızasına indeksleniyor ({len(df)} kayıt)...")
    t_epi = time.time()
    epi = getattr(agi.broca_organ, "episodic", None)
    indexed_qa = 0
    if epi is not None:
        stop_words = {"ne", "nasıl", "nedir", "neydi", "kim", "kimdir", "hangi", "hangisi", "kaç", "nerede", "neresi", "yapılır", "et", "ol", "bir", "mi", "mı", "mu", "mü", "ve", "veya", "ile", "için"}
        for _, row in df.iterrows():
            q_text = clean_text(row.get(input_col, ""))
            a_text = clean_text(row.get(output_col, ""))
            if not q_text or not a_text or len(a_text) < 5:
                continue
            q_low = tr_lower(q_text)
            words = re.findall(r"\b[\wçğıöşü]+\b", q_low)
            stems = [w for w in words if len(w) > 2 and w not in stop_words]
            if not stems:
                continue
            topic = stems[0] if stems else "bilgi"
            epi.store(
                age=int(agi.mind.age),
                speaker="sixfinger_bilgi",
                user_text=q_text,
                stems=stems,
                topic=topic,
                affect_val=0.0,
                chosen_acts=["QA_EXPERT"],
                reply_text=a_text,
            )
            indexed_qa += 1
    print(f"Epizodik diyalog indeksleme tamamlandı! Süre: {time.time() - t_epi:.1f}s | İndekslenen Soru-Cevap: {indexed_qa}")

    # 7. Beyni Kalıcı Olarak Kaydet (Tek dosya .nora)
    print(f"\nBütün beyin güncellenerek kalıcı olarak kaydediliyor: {BRAIN_PATH} ...")
    agi.save(str(BRAIN_PATH))
    print(f"Başarıyla kaydedildi ({BRAIN_PATH})! Artık './chat.py' bu zihinle açılacaktır.")

    # 7. Hızlı Sohbet Doğrulaması
    print("\n--- Örnek Sohbet Doğrulaması ---")
    test_prompts = [
        "Merhaba, nasılsın?",
        "Bana kendini tanıtır mısın?",
        "Bilgi nedir?",
    ]
    for q in test_prompts:
        ans = agi.converse(q, speaker="insan")
        print(f"Sen: {q}\nNora > {ans}\n")


if __name__ == "__main__":
    main()
