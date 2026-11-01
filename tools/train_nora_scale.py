"""tools/train_nora_scale.py

Nora NCC - Geniş Ölçekli Bilişsel ve Nöral Veri Eğitimi.
Hugging Face 51k Türkçe veri setinden 5 ana disiplinde binlerce yüksek kaliteli
talimat paketini çıkarır, Nora'nın:
1. Semantik Bilgi Grafiğine (Knowledge Graph - Wernicke & HDC)
2. Broca Nöral Dil Ağına (PyTorch GRU - learn_batch)
3. CA3 Epizodik Diyalog Belleğine (Associative Kenyon Fibers & Postings)
4. Söylem ve Çerçeve Korteksine (Schema Cortex & Frames)
aktarır ve kalıcı olarak 'state/nora_brain.nora' ile 'data/broca_tr.pt'ye mühürler.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nora.cognition.agi import NoraAGI
from nora.language.voice import state_vector
from nora.language.lang import tr_lower, tr_capitalize
from nora.language.broca_net import BrocaOrgan

BRAIN_PATH = _REPO_ROOT / "state" / "nora_brain.nora"
DATA_DIR = _REPO_ROOT / "data"
RAW_CSV_PATH = DATA_DIR / "hf_turkish_instructions_raw.csv"
BASE_BROCA_PATH = DATA_DIR / "broca_tr.pt"

DOMAIN_KEYWORDS = {
    "kodlama_teknoloji": [
        r"\bkod\b", r"\byazılım\b", r"\bprogramlama\b", r"\bpython\b", r"\bhtml\b",
        r"\bcss\b", r"\bjavascript\b", r"\bweb\b", r"\bveri tabanı\b", r"\balgoritma\b",
        r"\bapi\b", r"\bfonksiyon\b", r"\bsınıf\b", r"\bhata\b", r"\bdizi\b"
    ],
    "bilim_doga": [
        r"\bbilim\b", r"\bfizik\b", r"\bkimya\b", r"\bbiyoloji\b", r"\bgezegen\b",
        r"\batom\b", r"\bhücre\b", r"\benerji\b", r"\bdoğa\b", r"\bevren\b",
        r"\bgüneş\b", r"\bdünya\b", r"\bışık\b", r"\byerçekimi\b", r"\bmolekül\b"
    ],
    "yaratici_sanat": [
        r"\bhikaye\b", r"\bşiir\b", r"\byaratıcı\b", r"\bfikir\b", r"\btasarım\b",
        r"\bdiyalog\b", r"\bsenaryo\b", r"\broman\b", r"\bmüzik\b", r"\bsanat\b",
        r"\bmetafor\b", r"\bbesle\b", r"\bkurgu\b", r"\bkarakter\b"
    ],
    "mantik_matematik": [
        r"\bmantık\b", r"\bmatematik\b", r"\bhesapla\b", r"\bproblem\b", r"\bçözüm\b",
        r"\bneden\b", r"\bsonuç\b", r"\boran\b", r"\bolasılık\b", r"\bdenklem\b",
        r"\bgeometri\b", r"\btoplam\b", r"\bçıkarma\b", r"\bkanıt\b"
    ],
    "dil_kultur": [
        r"\bcümle\b", r"\bkelime\b", r"\btürkçe\b", r"\banlam\b", r"\beş anlam\b",
        r"\bzıt anlam\b", r"\bözet\b", r"\bbaşlık\b", r"\btarih\b", r"\bfelsefe\b",
        r"\bkültür\b", r"\bdeyim\b", r"\batasözü\b", r"\btanım\b"
    ]
}

def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def collect_diverse_data(raw_csv_path: Path, target_total: int = 5000) -> list[dict]:
    print(f"\n[1/5] Hugging Face veri seti taranıyor: {raw_csv_path} ...")
    t0 = time.time()
    df = pd.read_csv(raw_csv_path)

    inst_col = df.columns[1]
    inp_col = df.columns[2]
    out_col = df.columns[3]

    per_domain_target = target_total // len(DOMAIN_KEYWORDS)
    domain_buckets: dict[str, list[dict]] = {k: [] for k in DOMAIN_KEYWORDS}
    seen_queries = set()

    # Regex derle
    domain_res = {k: re.compile("|".join(pats), re.IGNORECASE) for k, pats in DOMAIN_KEYWORDS.items()}

    for _, row in df.iterrows():
        inst = str(row[inst_col] or "").strip()
        out = str(row[out_col] or "").strip()
        inp = str(row[inp_col] or "").strip() if pd.notna(row[inp_col]) else ""

        if len(inst) < 10 or len(out) < 20 or len(out) > 2200:
            continue

        q = clean_text(inst if not inp else f"{inst} ({inp})")
        q_key = tr_lower(q)
        if q_key in seen_queries:
            continue

        full_search_text = (q + " " + out).lower()

        # En uygun alanı bul
        matched_domain = None
        for dom, rgx in domain_res.items():
            if len(domain_buckets[dom]) < per_domain_target and rgx.search(full_search_text):
                matched_domain = dom
                break

        if matched_domain is None:
            # Eğer hala dolmamış bir alan varsa genel olarak değerlendir
            for dom in domain_buckets:
                if len(domain_buckets[dom]) < per_domain_target:
                    matched_domain = dom
                    break

        if matched_domain is not None:
            seen_queries.add(q_key)
            out_clean = clean_text(out)
            
            # Cümleleri ayıkla
            sentences = [
                s.strip() for s in re.split(r"(?<=[.!?])\s+", out_clean)
                if 12 <= len(s.strip()) <= 250 and not s.strip().startswith("<")
            ][:4]
            if not sentences:
                sentences = [out_clean[:200]]

            domain_buckets[matched_domain].append({
                "domain": matched_domain,
                "query": q,
                "output": out_clean,
                "sentences": sentences
            })

        if all(len(b) >= per_domain_target for b in domain_buckets.values()):
            break

    total_curated = sum(len(b) for b in domain_buckets.values())
    print(f"Veri seçimi tamamlandı ({time.time() - t0:.2f}s). Toplam seçilen paket: {total_curated}")
    for dom, items in domain_buckets.items():
        print(f"  - {dom}: {len(items)} paket")

    all_items = []
    for b in domain_buckets.values():
        all_items.extend(b)
    return all_items

def train_nora_scale(items: list[dict], train_broca: bool = True, max_broca_sentences: int = 12000):
    print("\n" + "=" * 65)
    print(f" NORA BÜYÜK ÖLÇEKLİ VERİ EĞİTİMİ BAŞLIYOR ({len(items)} PAKET)")
    print("=" * 65)

    # 1. Zihni Yükle
    print(f"\n[2/5] Kalıcı zihin yükleniyor: {BRAIN_PATH} ...")
    agi = NoraAGI.load(str(BRAIN_PATH), attach_broca_async=False)
    print(f"Mevcut Durum: {len(agi.lang.graph.edges)} Kenar | {len(agi.lang.graph.concept_id)} Kavram | {len(agi.broca_organ.episodic.episodes)} Anı")

    # 2. Semantik Bilgi Grafiğine Öğretim (Wernicke ile tek geçişte 0-epoch öğrenme)
    print(f"\n[3/5] Semantik Bilgi Grafiği ve Çerçeve Belleği eğitiliyor...")
    t_kg = time.time()
    kg_added = 0
    fact_candidates = []
    
    # Tanım ve olgusal cümleleri topla
    for it in items:
        for s in it["sentences"]:
            if any(cop in s.lower() for cop in ("dir.", "dır.", "dur.", "dür.", "tir.", "tır.", "tur.", "tür.", "sağlar.", "kullanılır.")):
                if 15 <= len(s) <= 120:
                    fact_candidates.append(s)

    fact_candidates = fact_candidates[:1200]
    print(f"Seçilen {len(fact_candidates)} olgusal cümle Wernicke üzerinden kavrama dönüştürülüyor...")
    
    for sent in fact_candidates:
        try:
            edges = agi.lang.teach_sentence(sent, speaker="hf_corpus")
            if edges:
                kg_added += len(edges)
            if getattr(agi, "frames", None) is not None:
                agi.frames.observe(sent)
        except Exception:
            pass

    print(f"Bilgi grafiği güncellendi: +{kg_added} yeni kenar ({time.time() - t_kg:.1f}s). Toplam: {len(agi.lang.graph.edges)}")

    # 3. Broca Nöral Dil Ağı Eğitimi (PyTorch GRU learn_batch)
    if train_broca:
        print(f"\n[4/5] Broca Nöral Dil Ağı eğitiliyor...")
        organ = getattr(agi.life.voice, "organ", None)
        if organ is None and BASE_BROCA_PATH.is_file():
            organ = BrocaOrgan.load(str(BASE_BROCA_PATH))
            agi.life.voice.organ = organ

        if organ is not None:
            t_broca = time.time()
            all_sentences = []
            for it in items:
                all_sentences.extend(it["sentences"])
            
            # Cümleleri sınırla ve karıştır
            all_sentences = all_sentences[:max_broca_sentences]
            st = state_vector(agi)
            batch_size = 32
            losses = []
            
            print(f"{len(all_sentences)} akıcı cümle batchler halinde Broca organına öğretiliyor...")
            for i in range(0, len(all_sentences), batch_size):
                batch = all_sentences[i : i + batch_size]
                try:
                    loss = organ.learn_batch(batch, st, anchor_dropout=0.1, loss_mode="sampled", negative_samples=512)
                    losses.append(loss)
                except Exception:
                    pass

            organ.update_common()
            if BASE_BROCA_PATH.is_file():
                organ.save(str(BASE_BROCA_PATH))
            avg_loss = sum(losses) / max(1, len(losses))
            print(f"Broca nöral dil eğitimi tamamlandı! Ortalama Loss: {avg_loss:.4f} ({time.time() - t_broca:.1f}s)")

    # 4. CA3 Epizodik Diyalog Hafızasına İndeksleme
    print(f"\n[5/5] Soru-Cevap çiftleri CA3 Epizodik Belleğe indeksleniyor...")
    t_epi = time.time()
    epi = getattr(agi.broca_organ, "episodic", None)
    stop_words = {"ne", "nasıl", "nedir", "neydi", "kim", "kimdir", "hangi", "hangisi", "kaç", "nerede", "bir", "mi", "mı", "ve", "ile", "için", "olan", "bu", "şu", "şunlar"}
    indexed_qa = 0

    if epi is not None:
        for it in items:
            q_text = it["query"]
            a_text = it["output"]
            q_low = tr_lower(q_text)
            words = re.findall(r"\b[\wçğıöşü]+\b", q_low)
            stems = [w for w in words if len(w) > 2 and w not in stop_words]
            if not stems:
                continue
            topic = stems[0]
            epi.store(
                age=int(agi.mind.age),
                speaker="hf_corpus",
                user_text=q_text,
                stems=stems,
                topic=topic,
                affect_val=0.0,
                chosen_acts=["KNOWLEDGE_BASE"],
                reply_text=a_text,
            )
            indexed_qa += 1

        print(f"Epizodik bellek indekslendi: +{indexed_qa} yeni QA hafızaya alındı ({time.time() - t_epi:.1f}s). Toplam Anı: {len(epi.episodes)}")

    # 5. Kalıcı Olarak Mühürle
    print(f"\nNora kalıcı beyin dosyası mühürleniyor: {BRAIN_PATH} ...")
    t_save = time.time()
    agi.save(str(BRAIN_PATH))
    print(f"Beyin dosyası mühürlendi ({time.time() - t_save:.1f}s).")
    print(f"\nTÜM EĞİTİM BAŞARIYLA TAMAMLANDI!")

def main():
    parser = argparse.ArgumentParser(description="Nora NCC Geniş Ölçekli Veri Eğitimi")
    parser.add_argument("--samples", type=int, default=5000, help="Eğitilecek talimat paketi sayısı (varsayılan: 5000)")
    parser.add_argument("--no-broca", action="store_true", help="Broca nöral eğitimini atla")
    args = parser.parse_args()

    items = collect_diverse_data(RAW_CSV_PATH, target_total=args.samples)
    train_nora_scale(items, train_broca=not args.no_broca)

if __name__ == "__main__":
    main()
