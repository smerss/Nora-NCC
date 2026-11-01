"""tools/fetch_hf_coding_data.py

Hugging Face Hub üzerindeki 'merve/turkish_instructions' veri setinden
HTML, web geliştirme ve kodlama ile ilgili zengin soru-cevap ve talimatları çeker,
temizler, Nora'nın bilişsel formatına dönüştürür ve beyne entegre eder.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DATA_DIR = _REPO_ROOT / "data"
RAW_CSV_PATH = DATA_DIR / "hf_turkish_instructions_raw.csv"
OUT_JSON_PATH = DATA_DIR / "hf_coding_knowledge.json"

HF_CSV_URL = "https://huggingface.co/datasets/merve/turkish_instructions/resolve/main/instructions.csv"

CODING_KEYWORDS = [
    r"\bhtml\b", r"\bhtml5\b", r"\bcss\b", r"\bjavascript\b", r"\bjs\b",
    r"\bweb sayfası\b", r"\bweb sitesi\b", r"\bweb geliştirme\b",
    r"\betiket\b", r"\betiketi\b", r"\betiketleri\b",
    r"\bkod\b", r"\bkodu\b", r"\bkodlama\b", r"\bprogramlama\b",
    r"\bform\b", r"\bformu\b", r"\binput\b", r"\bbuton\b", r"\btablo\b",
    r"\bnavbar\b", r"\bfrontend\b", r"\bbackend\b", r"\bbağlantı\b",
    r"\bdiv\b", r"\bspan\b", r"\blink\b", r"\bhref\b", r"\bsrc\b"
]


def download_hf_dataset(dest_path: Path) -> Path:
    """Hugging Face'ten ham CSV veri dosyasını indirir."""
    if dest_path.is_file() and dest_path.stat().st_size > 10000:
        print(f"Mevcut veri dosyası bulundu: {dest_path} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")
        return dest_path

    print(f"Hugging Face'ten indiriliyor: {HF_CSV_URL} -> {dest_path} ...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # User-Agent ile güvenli indirme
    req = urllib.request.Request(
        HF_CSV_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NoraNCC/0.1"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as f:
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk:
                break
            f.write(chunk)
            
    print(f"İndirme tamamlandı! Boyut: {dest_path.stat().st_size / (1024*1024):.2f} MB")
    return dest_path


HTML_PRIORITY_KEYWORDS = [
    r"\bhtml5?\b", r"\bcss3?\b", r"\bweb sayfası\b", r"\bweb sitesi\b",
    r"\bweb geliştirme\b", r"\bweb tasarımı\b", r"\bmarkup\b", r"\bön uç\b", r"\bfrontend\b",
    r"<(?:html|head|body|div|p|span|a|button|input|form|table|tr|th|td|ul|ol|li|h[1-6]|header|footer|nav|section|article)\b"
]

NON_CODING_EXCLUSIONS = [
    "özgeçmiş", "cv", "resume", "sınıflandırma", "pazarlama", "sosyal medya", "makine öğrenimi",
    "regresyon", "kümeleme", "müşteri ilişkileri", "envanter", "satış",
    "tedarik zinciri", "reklam kampanyası", "insan kaynakları"
]

def clean_html_formatting(text: str) -> str:
    """Hugging Face çıktılarındaki hatalı etiket boşluklarını (< / div>, < a href>) onarır."""
    t = re.sub(r"<\s*/\s*", "</", text)
    t = re.sub(r"<\s*([a-zA-Z0-9]+)\s*", r"<\1 ", t)
    t = re.sub(r"\s*>\s*", ">", t)
    t = re.sub(r"<\s*([a-zA-Z0-9]+)>", r"<\1>", t)
    t = re.sub(r"</\s*([a-zA-Z0-9]+)>", r"</\1>", t)
    return t.strip()

def filter_and_curate_coding_data(raw_csv_path: Path, max_samples: int = 150) -> list[dict]:
    """Ham talimat verilerini okur, doğrudan HTML ve web geliştirme odaklı kaliteli örnekleri süzüp yapılandırır."""
    print("Veri seti taranıyor ve doğrudan HTML/Web filtrelemesi yapılıyor...")
    df = pd.read_csv(raw_csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"Toplam ham satır sayısı: {len(df)}")

    # Gerekli sütunları bul
    inst_col = "talimat" if "talimat" in df.columns else ("instruction" if "instruction" in df.columns else df.columns[1])
    out_col = "çıktı" if "çıktı" in df.columns else ("output" if "output" in df.columns else df.columns[-1])
    inp_col = "giriş" if "giriş" in df.columns else ("input" if "input" in df.columns else None)

    regex_pattern = re.compile("|".join(HTML_PRIORITY_KEYWORDS), re.IGNORECASE)

    curated_items = []
    seen_prompts = set()

    for _, row in df.iterrows():
        instruction = str(row[inst_col] or "").strip()
        output = str(row[out_col] or "").strip()
        inp = str(row[inp_col] or "").strip() if inp_col and pd.notna(row[inp_col]) else ""

        full_query = (instruction + " " + inp).strip()
        full_text = (full_query + " " + output).lower()

        # Alakasız veya CV/pazarlama konularını ele
        if any(exc in full_text for exc in NON_CODING_EXCLUSIONS):
            continue

        # Doğrudan HTML / Web anahtar sözcüğü veya etiket geçiyor mu?
        if not regex_pattern.search(full_text):
            continue

        # Çok kısa veya anlamsız satırları ele
        if len(instruction) < 8 or len(output) < 15:
            continue
        # Çok devasa kod bloklarını hafıza dengesi için sınırla
        if len(output) > 2800:
            continue

        clean_q = instruction if not inp else f"{instruction} ({inp})"
        clean_q = re.sub(r"\s+", " ", clean_q).strip()
        if clean_q in seen_prompts:
            continue
        seen_prompts.add(clean_q)

        clean_output = clean_html_formatting(output)

        # Temel kavram çıkarımı
        concept_match = re.search(r"\b(html5?|css|javascript|form[ua]?|input|button|buton|tablo|table|navbar|menü|div|span|link|etiket[i|leri]?)\b", clean_q, re.I)
        concept = concept_match.group(1).lower() if concept_match else "html ve web"

        # Broca organı için doğal cümleler
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_output) if len(s.strip()) > 15 and not s.strip().startswith("<")][:3]
        if not sentences:
            sentences = [f"{concept} web sayfalarını yapılandırmak için kullanılan temel bir HTML standardıdır."]

        triples = [
            (concept, "USED_FOR", "web sayfaları ve kullanıcı arayüzü geliştirmek"),
            (concept, "ISA", "web işaretleme ve geliştirme teknolojisi"),
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

    print(f"Hugging Face'ten filtrelenmiş yüksek kaliteli HTML/Web örneği sayısı: {len(curated_items)}")
    return curated_items


def main():
    parser = argparse.ArgumentParser(description="Hugging Face Coding Data Ingestor")
    parser.add_argument("--max-samples", type=int, default=80, help="Maksimum işlenecek örnek sayısı")
    parser.add_argument("--ingest", action="store_true", help="Doğrudan Nora beynine mühürle")
    args = parser.parse_args()

    raw_path = download_hf_dataset(RAW_CSV_PATH)
    items = filter_and_curate_coding_data(raw_path, max_samples=args.max_samples)

    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"Yapılandırılmış bilgi paketi kaydedildi: {OUT_JSON_PATH}")

    if args.ingest:
        from tools.ingest_knowledge import ingest_knowledge_domain
        ingest_knowledge_domain("hf_coding_knowledge", items)


if __name__ == "__main__":
    main()
