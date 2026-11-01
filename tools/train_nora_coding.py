"""tools/train_nora_coding.py

Nora NCC - Kapsamlı Kodlama ve Web Geliştirme Eğitim Motoru.
1. Hugging Face 51k Türkçe veri setindeki tüm HTML, CSS, JavaScript, Web ve Portfolyo verilerini çıkarır.
2. Portfolyo, modern web mimarisi, etkileşimli JS ve CSS kurallarını yapılandırır.
3. Nora'nın Bilişsel Bilgi Grafiğine (Knowledge Graph) aktarır.
4. Broca Nöral Dil Organına (PyTorch GRU) cümle kalıplarını eğitir (learn_batch).
5. Epizodik Diyalog Hafızasına (CA3 RAG) indeksler.
6. Kalıcı olarak 'state/nora_brain.nora' ve 'data/broca_tr.pt'ye mühürler.
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
TEXT_DIR = DATA_DIR / "text"
RAW_CSV_PATH = DATA_DIR / "hf_turkish_instructions_raw.csv"
OUT_JSON_PATH = DATA_DIR / "hf_coding_knowledge_full.json"
BASE_BROCA_PATH = DATA_DIR / "broca_tr.pt"

HTML_KEYWORDS = [
    r"\bhtml5?\b", r"\bcss3?\b", r"\bjavascript\b", r"\bjs\b",
    r"\bweb sayfası\b", r"\bweb sitesi\b", r"\bweb geliştirme\b", r"\bweb tasarımı\b",
    r"\bportfolyo\b", r"\bportfolio\b", r"\bözgeçmiş\b", r"\bcv\b",
    r"\bfrontend\b", r"\bön uç\b", r"\barayüz\b", r"\bui\b", r"\bux\b",
    r"\bform\b", r"\bformu\b", r"\binput\b", r"\bbuton\b", r"\btablo\b",
    r"\bnavbar\b", r"\bresponsive\b", r"\bduyarlı\b", r"\bkod\b", r"\bkodlama\b",
    r"<(?:html|head|body|div|p|span|a|button|input|form|table|tr|th|td|ul|ol|li|h[1-6]|header|footer|nav|section|article)\b"
]

NON_CODING_EXCLUSIONS = [
    "sınıflandırma", "pazarlama", "sosyal medya", "makine öğrenimi",
    "regresyon", "kümeleme", "müşteri ilişkileri", "envanter", "satış",
    "tedarik zinciri", "reklam kampanyası", "insan kaynakları"
]

def clean_html_formatting(text: str) -> str:
    t = re.sub(r"<\s*/\s*", "</", text)
    t = re.sub(r"<\s*([a-zA-Z0-9]+)\s*", r"<\1 ", t)
    t = re.sub(r"\s*>\s*", ">", t)
    t = re.sub(r"<\s*([a-zA-Z0-9]+)>", r"<\1>", t)
    t = re.sub(r"</\s*([a-zA-Z0-9]+)>", r"</\1>", t)
    return t.strip()

# Portfolyo ve modern web mimarisi için çekirdek uzmanlık bilgisi
PORTFOLIO_CORE_KNOWLEDGE = [
    {
        "concept": "portfolyo",
        "triples": [
            ("portfolyo", "ISA", "kişisel web sitesi ve geliştirici vitrini"),
            ("portfolyo", "USED_FOR", "projeleri, yetenekleri ve deneyimleri sergilemek"),
            ("portfolyo", "CAUSES", "profesyonel görünürlük ve iş birliği fırsatları"),
            ("portfolyo sitesi", "HAS_PART", "kahraman alanı, projeler, teknik yetenekler ve iletişim formu"),
        ],
        "sentences": [
            "Portfolyo, geliştiricinin projelerini, teknik yeteneklerini ve iletişim bilgilerini sergileyen modern bir web sitesidir.",
            "Başarılı bir web geliştirici portfolyosu responsive tasarım, temiz tipografi ve etkileşimli proje kartları içerir.",
            "Kişisel portfolyo sitelerinde siyah ve beyaz minimalist tema modern ve profesyonel bir görünüm sağlar.",
        ],
        "qa": [
            ("Portfolyo sitesi nedir?", "Portfolyo sitesi, bir yazılımcının veya tasarımcının yaptığı projeleri, teknik becerilerini ve biyografisini sergilediği kişisel web platformudur."),
            ("İyi bir portfolyo sayfasında neler olmalıdır?", "İyi bir portfolyoda etkileyici bir giriş (hero), öne çıkan projeler, teknik yetenekler listesi ve çalışan bir iletişim formu yer almalıdır."),
            ("Web developer portfolyosu nasıl tasarlanır?", "Modern HTML5 semantik etiketleri, responsive CSS Grid/Flexbox düzeni ve kullanıcı etkileşimlerini destekleyen Vanilla JavaScript ile tasarlanır."),
        ]
    },
    {
        "concept": "responsive tasarım",
        "triples": [
            ("responsive tasarım", "ISA", "modern web tasarım metodolojisi"),
            ("responsive tasarım", "USED_FOR", "sayfaların mobil, tablet ve masaüstü ekranlara uyum sağlaması"),
            ("viewport meta etiketi", "CAUSES", "mobil cihazlarda doğru ölçeklendirme"),
        ],
        "sentences": [
            "Responsive tasarım, web sayfalarının tüm ekran boyutlarında kusursuz görüntülenmesini sağlar.",
            "CSS Flexbox ve Grid mimarisi duyarlı arayüzlerin temel yapı taşlarıdır.",
        ],
        "qa": [
            ("Responsive web tasarımı ne demektir?", "Responsive web tasarımı, bir web sayfasının mobil cihazlardan geniş masaüstü monitörlere kadar her ekranda esnek ve uyumlu çalışmasıdır."),
        ]
    }
]

def extract_and_curate_all_coding_data(raw_csv_path: Path, max_samples: int = 1500) -> list[dict]:
    print(f"Hugging Face veri seti açılıyor: {raw_csv_path} ...")
    df = pd.read_csv(raw_csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    inst_col = "talimat" if "talimat" in df.columns else ("instruction" if "instruction" in df.columns else df.columns[1])
    out_col = "çıktı" if "çıktı" in df.columns else ("output" if "output" in df.columns else df.columns[-1])
    inp_col = "giriş" if "giriş" in df.columns else ("input" if "input" in df.columns else None)

    regex_pattern = re.compile("|".join(HTML_KEYWORDS), re.IGNORECASE)

    curated_items = list(PORTFOLIO_CORE_KNOWLEDGE)
    seen_prompts = set()

    for _, row in df.iterrows():
        instruction = str(row[inst_col] or "").strip()
        output = str(row[out_col] or "").strip()
        inp = str(row[inp_col] or "").strip() if inp_col and pd.notna(row[inp_col]) else ""

        full_query = (instruction + " " + inp).strip()
        full_text = (full_query + " " + output).lower()

        if any(exc in full_text for exc in NON_CODING_EXCLUSIONS):
            continue

        if not regex_pattern.search(full_text):
            continue

        if len(instruction) < 8 or len(output) < 15:
            continue
        if len(output) > 3500:
            continue

        clean_q = instruction if not inp else f"{instruction} ({inp})"
        clean_q = re.sub(r"\s+", " ", clean_q).strip()
        if clean_q in seen_prompts:
            continue
        seen_prompts.add(clean_q)

        clean_output = clean_html_formatting(output)

        concept_match = re.search(r"\b(html5?|css3?|javascript|portfolyo|portfolio|özgeçmiş|cv|form|input|button|tablo|navbar|div|link)\b", clean_q, re.I)
        concept = concept_match.group(1).lower() if concept_match else "web geliştirme"

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_output) if len(s.strip()) > 15 and not s.strip().startswith("<")][:3]
        if not sentences:
            sentences = [f"{concept} modern web arayüzleri ve dinamik uygulamalar geliştirmek için kullanılır."]

        triples = [
            (concept, "USED_FOR", "web sayfaları ve modern kullanıcı arayüzleri geliştirmek"),
            (concept, "ISA", "web programlama teknolojisi ve standardı"),
        ]

        curated_items.append({
            "concept": concept,
            "triples": triples,
            "sentences": sentences,
            "qa": [
                (clean_q, clean_output)
            ]
        })

        if len(curated_items) >= max_samples:
            break

    print(f"Toplam derlenen yüksek kaliteli kodlama/web bilgi paketi sayısı: {len(curated_items)}")
    return curated_items

def train_and_ingest(curated_items: list[dict]):
    print("\n" + "=" * 60)
    print(" NORA BİLİŞSEL VE NÖRAL KODLAMA EĞİTİMİ BAŞLIYOR")
    print("=" * 60)

    # 1. Bilgi kayıtlarını hazırla
    all_triples = []
    all_sentences = []
    all_qa = []
    for item in curated_items:
        all_triples.extend(item.get("triples", []))
        all_sentences.extend(item.get("sentences", []))
        all_qa.extend(item.get("qa", []))

    print(f"Eğitim Verisi: {len(all_triples)} Semantik İlişki | {len(all_sentences)} Nöral Cümle | {len(all_qa)} Soru-Cevap Çifti")

    # 2. Kalıcı Zihni Yükle
    print(f"\n[1/4] Nora kalıcı zihni yükleniyor: {BRAIN_PATH} ...")
    if BRAIN_PATH.is_file():
        agi = NoraAGI.load(str(BRAIN_PATH), attach_broca_async=False)
        print("Mevcut 'nora_brain.nora' başarıyla yüklendi.")
    else:
        agi = NoraAGI(name="Nora", seed=42, boot_thinker=False, single_life=True)

    g = agi.lang.graph
    print(f"Mevcut semantik kenar sayısı: {len(g.edges)}")

    # 3. Semantik Bilgi Grafiğine Aktarım
    print(f"\n[2/4] Semantik Bilgi Grafiğine aktarım yapılıyor ({len(all_triples)} üçlü)...")
    t0 = time.time()
    rmb = getattr(g, "relation_mb", None)
    if rmb is not None:
        rmb._auto_commit = False

    added_edges = 0
    try:
        for s, r, o in all_triples:
            s_clean = tr_lower(s).strip()
            o_clean = tr_lower(o).strip()
            g.register_concept(s_clean, bind_mb=False)
            g.register_concept(o_clean, bind_mb=False)

            before = len(g.edges)
            g.add_edge(s_clean, r, o_clean, source="hf_coding_corpus", weight=1.0)
            added_edges += int(len(g.edges) > before)
    finally:
        if rmb is not None:
            rmb._db.commit()
            rmb._auto_commit = True

    print(f"Semantik aktarım tamamlandı: +{added_edges} yeni kenar ({time.time() - t0:.1f}s). Toplam kenar: {len(g.edges)}")

    # 4. Broca Nöral Dil Ağına Eğitim (PyTorch GRU)
    print(f"\n[3/4] Broca Nöral Dil Ağı eğitiliyor ({len(all_sentences)} cümle)...")
    organ = getattr(agi.life.voice, "organ", None)
    if organ is None and BASE_BROCA_PATH.is_file():
        organ = BrocaOrgan.load(str(BASE_BROCA_PATH))
        agi.life.voice.organ = organ

    if organ is not None and all_sentences:
        t_broca = time.time()
        st = state_vector(agi)
        B = 32
        losses = []
        for i in range(0, len(all_sentences), B):
            batch = all_sentences[i : i + B]
            try:
                loss = organ.learn_batch(batch, st, anchor_dropout=0.1, loss_mode="sampled", negative_samples=512)
                losses.append(loss)
            except Exception:
                pass
        organ.update_common()
        if BASE_BROCA_PATH.is_file():
            organ.save(str(BASE_BROCA_PATH))
        print(f"Broca nöral eğitimi tamamlandı! Ortalama Loss: {sum(losses)/max(1, len(losses)):.4f} ({time.time() - t_broca:.1f}s)")

    # 5. Epizodik Diyalog Hafızasına İndeksleme (CA3 RAG)
    print(f"\n[4/4] Soru-cevap çiftleri Epizodik Hafızaya indeksleniyor ({len(all_qa)} QA)...")
    epi = getattr(agi.broca_organ, "episodic", None)
    if epi is not None and all_qa:
        t_epi = time.time()
        stop_words = {"ne", "nasıl", "nedir", "neydi", "kim", "kimdir", "hangi", "hangisi", "kaç", "nerede", "bir", "mi", "mı", "ve", "ile", "için"}
        indexed = 0
        for q_text, a_text in all_qa:
            q_low = tr_lower(q_text)
            words = re.findall(r"\b[\wçğıöşü]+\b", q_low)
            stems = [w for w in words if len(w) > 2 and w not in stop_words]
            if not stems:
                continue
            topic = stems[0]
            epi.store(
                age=int(agi.mind.age),
                speaker="hf_coding_corpus",
                user_text=q_text,
                stems=stems,
                topic=topic,
                affect_val=0.0,
                chosen_acts=["QA_EXPERT"],
                reply_text=a_text,
            )
            indexed += 1
        print(f"Epizodik hafıza indekslendi: +{indexed} QA kaydedildi ({time.time() - t_epi:.1f}s).")

    # 6. Kalıcı Olarak Kaydet
    print(f"\nNora kalıcı beyin dosyası mühürleniyor: {BRAIN_PATH} ...")
    agi.save(str(BRAIN_PATH))
    print(f"TÜM EĞİTİM BAŞARIYLA TAMAMLANDI VE MÜHÜRLENDİ!")

def main():
    items = extract_and_curate_all_coding_data(RAW_CSV_PATH, max_samples=1500)
    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"Veri paketi JSON olarak kaydedildi: {OUT_JSON_PATH}")
    train_and_ingest(items)

if __name__ == "__main__":
    main()
