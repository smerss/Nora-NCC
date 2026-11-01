"""tools/ingest_capitals.py

Hugging Face Hub üzerindeki 'finnianx/capitals' veri setini çeker,
197 dünya ülkesi ve başkentini Türkçe morfolojik uyumla zenginleştirerek:
  1. Bilişsel Semantik Ağına (Knowledge Graph & HDC - ISA, LOCATED_AT, HAS_PROP, CAPITAL)
  2. Broca Konuşma / Artikülasyon Organına (Türkçe cümle kalıpları ve akıcı artikülasyon)
  3. Epizodik Diyalog Hafızasına (CA3 QA RAG - Anlık, kesin soru-cevap indekslemesi)
işler ve 'state/nora_brain.nora' birleşik beyin dosyasına kalıcı olarak kaydeder.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
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

DATASET_REPO = "finnianx/capitals"
BRAIN_PATH = _REPO_ROOT / "state" / "nora_brain.nora"
OUT_CSV_FILE = _REPO_ROOT / "data" / "dunya_baskentleri.csv"
OUT_TEXT_DIR = _REPO_ROOT / "data" / "text"
OUT_TEXT_FILE = OUT_TEXT_DIR / "dunya_baskentleri.txt"

TR_COUNTRY_MAP: dict[str, tuple[str, str]] = {
    "Afghanistan": ("Afganistan", "Kabil"),
    "Albania": ("Arnavutluk", "Tiran"),
    "Algeria": ("Cezayir", "Cezayir"),
    "Andorra": ("Andorra", "Andorra la Vella"),
    "Angola": ("Angola", "Luanda"),
    "Antigua and Barbuda": ("Antigua ve Barbuda", "St. John's"),
    "Argentina": ("Arjantin", "Buenos Aires"),
    "Armenia": ("Ermenistan", "Erivan"),
    "Australia": ("Avustralya", "Canberra"),
    "Austria": ("Avusturya", "Viyana"),
    "Azerbaijan": ("Azerbaycan", "Bakü"),
    "Bahamas": ("Bahamalar", "Nassau"),
    "Bahrain": ("Bahreyn", "Manama"),
    "Bangladesh": ("Bangladeş", "Dakka"),
    "Barbados": ("Barbados", "Bridgetown"),
    "Belarus": ("Belarus", "Minsk"),
    "Belgium": ("Belçika", "Brüksel"),
    "Belize": ("Belize", "Belmopan"),
    "Benin": ("Benin", "Porto-Novo"),
    "Bhutan": ("Butan", "Thimphu"),
    "Bolivia": ("Bolivya", "Sucre"),
    "Bosnia and Herzegovina": ("Bosna-Hersek", "Saraybosna"),
    "Botswana": ("Botsvana", "Gaborone"),
    "Brazil": ("Brezilya", "Brasília"),
    "Brunei": ("Brunei", "Bandar Seri Begawan"),
    "Bulgaria": ("Bulgaristan", "Sofya"),
    "Burkina Faso": ("Burkina Faso", "Ouagadougou"),
    "Burundi": ("Burundi", "Gitega"),
    "Cabo Verde": ("Yeşil Burun Adaları", "Praia"),
    "Cambodia": ("Kamboçya", "Phnom Penh"),
    "Cameroon": ("Kamerun", "Yaoundé"),
    "Canada": ("Kanada", "Ottawa"),
    "Central African Republic": ("Orta Afrika Cumhuriyeti", "Bangui"),
    "Chad": ("Çad", "N'Djamena"),
    "Chile": ("Şili", "Santiago"),
    "China": ("Çin", "Pekin"),
    "Colombia": ("Kolombiya", "Bogotá"),
    "Comoros": ("Komorlar", "Moroni"),
    "Costa Rica": ("Kosta Rika", "San José"),
    "Croatia": ("Hırvatistan", "Zagrep"),
    "Cuba": ("Küba", "Havana"),
    "Cyprus": ("Kıbrıs", "Lefkoşa"),
    "Czechia": ("Çekya", "Prag"),
    "Democratic Republic of the Congo": ("Demokratik Kongo Cumhuriyeti", "Kinşasa"),
    "Republic of the Congo": ("Kongo Cumhuriyeti", "Brazzaville"),
    "Denmark": ("Danimarka", "Kopenhag"),
    "Djibouti": ("Cibuti", "Cibuti"),
    "Dominica": ("Dominika", "Roseau"),
    "Dominican Republic": ("Dominik Cumhuriyeti", "Santo Domingo"),
    "Ecuador": ("Ekvador", "Quito"),
    "Egypt": ("Mısır", "Kahire"),
    "El Salvador": ("El Salvador", "San Salvador"),
    "Equatorial Guinea": ("Ekvator Ginesi", "Malabo"),
    "Eritrea": ("Eritre", "Asmara"),
    "Estonia": ("Estonya", "Tallinn"),
    "Eswatini": ("Esvatini", "Mbabane"),
    "Ethiopia": ("Etiyopya", "Addis Ababa"),
    "Fiji": ("Fiji", "Suva"),
    "Finland": ("Finlandiya", "Helsinki"),
    "France": ("Fransa", "Paris"),
    "Gabon": ("Gabon", "Libreville"),
    "Gambia": ("Gambiya", "Banjul"),
    "Georgia": ("Gürcistan", "Tiflis"),
    "Germany": ("Almanya", "Berlin"),
    "Ghana": ("Gana", "Akra"),
    "Greece": ("Yunanistan", "Atina"),
    "Grenada": ("Grenada", "St. George's"),
    "Guatemala": ("Guatemala", "Guatemala"),
    "Guinea": ("Gine", "Konakri"),
    "Guinea-Bissau": ("Gine-Bissau", "Bissau"),
    "Guyana": ("Guyana", "Georgetown"),
    "Haiti": ("Haiti", "Port-au-Prince"),
    "Honduras": ("Honduras", "Tegucigalpa"),
    "Hungary": ("Macaristan", "Budapeşte"),
    "Iceland": ("İzlanda", "Reykjavik"),
    "India": ("Hindistan", "Yeni Delhi"),
    "Indonesia": ("Endonezya", "Cakarta"),
    "Iran": ("İran", "Tahran"),
    "Iraq": ("Irak", "Bağdat"),
    "Ireland": ("İrlanda", "Dublin"),
    "Israel": ("İsrail", "Kudüs"),
    "Italy": ("İtalya", "Roma"),
    "Ivory Coast": ("Fildişi Sahili", "Yamoussoukro"),
    "Jamaica": ("Jamaika", "Kingston"),
    "Japan": ("Japonya", "Tokyo"),
    "Jordan": ("Ürdün", "Amman"),
    "Kazakhstan": ("Kazakistan", "Astana"),
    "Kenya": ("Kenya", "Nairobi"),
    "Kiribati": ("Kiribati", "Tarawa"),
    "Kosovo": ("Kosova", "Priştine"),
    "Kuwait": ("Kuveyt", "Kuveyt"),
    "Kyrgyzstan": ("Kırgızistan", "Bişkek"),
    "Laos": ("Laos", "Vientiane"),
    "Latvia": ("Letonya", "Riga"),
    "Lebanon": ("Lübnan", "Beyrut"),
    "Lesotho": ("Lesotho", "Maseru"),
    "Liberia": ("Liberya", "Monrovia"),
    "Libya": ("Libya", "Trablus"),
    "Liechtenstein": ("Lihtenştayn", "Vaduz"),
    "Lithuania": ("Litvanya", "Vilnius"),
    "Luxembourg": ("Lüksemburg", "Lüksemburg"),
    "Madagascar": ("Madagaskar", "Antananarivo"),
    "Malawi": ("Malavi", "Lilongwe"),
    "Malaysia": ("Malezya", "Kuala Lumpur"),
    "Maldives": ("Maldivler", "Malé"),
    "Mali": ("Mali", "Bamako"),
    "Malta": ("Malta", "Valletta"),
    "Marshall Islands": ("Marshall Adaları", "Majuro"),
    "Mauritania": ("Moritanya", "Nuakşot"),
    "Mauritius": ("Mauritius", "Port Louis"),
    "Mexico": ("Meksika", "Mexico City"),
    "Micronesia": ("Mikronezya", "Palikir"),
    "Moldova": ("Moldova", "Kişinev"),
    "Monaco": ("Monako", "Monako"),
    "Mongolia": ("Moğolistan", "Ulan Batur"),
    "Montenegro": ("Karadağ", "Podgorica"),
    "Morocco": ("Fas", "Rabat"),
    "Mozambique": ("Mozambik", "Maputo"),
    "Myanmar": ("Myanmar", "Naypyidaw"),
    "Namibia": ("Namibya", "Windhoek"),
    "Nauru": ("Nauru", "Yaren"),
    "Nepal": ("Nepal", "Katmandu"),
    "Netherlands": ("Hollanda", "Amsterdam"),
    "New Zealand": ("Yeni Zelanda", "Wellington"),
    "Nicaragua": ("Nikaragua", "Managua"),
    "Niger": ("Nijer", "Niamey"),
    "Nigeria": ("Nijerya", "Abuja"),
    "North Korea": ("Kuzey Kore", "Pyongyang"),
    "North Macedonia": ("Kuzey Makedonya", "Üsküp"),
    "Norway": ("Norveç", "Oslo"),
    "Oman": ("Umman", "Maskat"),
    "Pakistan": ("Pakistan", "İslamabad"),
    "Palau": ("Palau", "Ngerulmud"),
    "Palestine": ("Filistin", "Doğu Kudüs"),
    "Panama": ("Panama", "Panama"),
    "Papua New Guinea": ("Papua Yeni Gine", "Port Moresby"),
    "Paraguay": ("Paraguay", "Asunción"),
    "Peru": ("Peru", "Lima"),
    "Philippines": ("Filipinler", "Manila"),
    "Poland": ("Polonya", "Varşova"),
    "Portugal": ("Portekiz", "Lizbon"),
    "Qatar": ("Katar", "Doha"),
    "Romania": ("Romanya", "Bükreş"),
    "Russia": ("Rusya", "Moskova"),
    "Rwanda": ("Ruanda", "Kigali"),
    "Saint Kitts and Nevis": ("Saint Kitts ve Nevis", "Basseterre"),
    "Saint Lucia": ("Saint Lucia", "Castries"),
    "Saint Vincent and the Grenadines": ("Saint Vincent ve Grenadinler", "Kingstown"),
    "Samoa": ("Samoa", "Apia"),
    "San Marino": ("San Marino", "San Marino"),
    "Sao Tome and Principe": ("Sao Tome ve Principe", "Sao Tome"),
    "Saudi Arabia": ("Suudi Arabistan", "Riyad"),
    "Senegal": ("Senegal", "Dakar"),
    "Serbia": ("Sırbistan", "Belgrad"),
    "Seychelles": ("Seyşeller", "Victoria"),
    "Sierra Leone": ("Sierra Leone", "Freetown"),
    "Singapore": ("Singapur", "Singapur"),
    "Slovakia": ("Slovakya", "Bratislava"),
    "Slovenia": ("Slovenya", "Ljubljana"),
    "Solomon Islands": ("Solomon Adaları", "Honiara"),
    "Somalia": ("Somali", "Mogadişu"),
    "South Africa": ("Güney Afrika Cumhuriyeti", "Pretoria"),
    "South Korea": ("Güney Kore", "Seul"),
    "South Sudan": ("Güney Sudan", "Cuba"),
    "Spain": ("İspanya", "Madrid"),
    "Sri Lanka": ("Sri Lanka", "Kolombo"),
    "Sudan": ("Sudan", "Hartum"),
    "Suriname": ("Surinam", "Paramaribo"),
    "Sweden": ("İsveç", "Stockholm"),
    "Switzerland": ("İsviçre", "Bern"),
    "Syria": ("Suriye", "Şam"),
    "Taiwan": ("Tayvan", "Taipei"),
    "Tajikistan": ("Tacikistan", "Duşanbe"),
    "Tanzania": ("Tanzanya", "Dodoma"),
    "Thailand": ("Tayland", "Bangkok"),
    "Timor-Leste": ("Doğu Timor", "Dili"),
    "Togo": ("Togo", "Lomé"),
    "Tonga": ("Tonga", "Nukuʻalofa"),
    "Trinidad and Tobago": ("Trinidad ve Tobago", "Port of Spain"),
    "Tunisia": ("Tunus", "Tunus"),
    "Turkey": ("Türkiye", "Ankara"),
    "Turkmenistan": ("Türkmenistan", "Aşkabat"),
    "Tuvalu": ("Tuvalu", "Funafuti"),
    "Uganda": ("Uganda", "Kampala"),
    "Ukraine": ("Ukrayna", "Kiev"),
    "United Arab Emirates": ("Birleşik Arap Emirlikleri", "Abu Dabi"),
    "United Kingdom": ("Birleşik Krallık", "Londra"),
    "United States": ("Amerika Birleşik Devletleri", "Washington, D.C."),
    "Uruguay": ("Uruguay", "Montevideo"),
    "Uzbekistan": ("Özbekistan", "Taşkent"),
    "Vanuatu": ("Vanuatu", "Port Vila"),
    "Vatican City": ("Vatikan", "Vatikan"),
    "Venezuela": ("Venezuela", "Karakas"),
    "Vietnam": ("Vietnam", "Hanoi"),
    "Yemen": ("Yemen", "Sana"),
    "Zambia": ("Zambiya", "Lusaka"),
    "Zimbabwe": ("Zimbabve", "Harare"),
}


def make_genitive(noun: str) -> str:
    """Türkçe ilgi (iyelik) eki ekler: Türkiye'nin, Fransa'nın, Rusya'nın, Irak'ın vb."""
    vowels = "aeıioöuüAEIİOÖUÜ"
    clean = noun.strip().rstrip("'\"")
    last_char = clean[-1]
    last_vowel = next((ch for ch in reversed(clean) if ch in vowels), "e")
    is_vowel_ending = last_char in vowels
    if last_vowel in "eiEİ":
        suffix = "nin" if is_vowel_ending else "in"
    elif last_vowel in "öüÖÜ":
        suffix = "nün" if is_vowel_ending else "ün"
    elif last_vowel in "aıAI":
        suffix = "nın" if is_vowel_ending else "ın"
    else:
        suffix = "nun" if is_vowel_ending else "un"
    return f"{clean}'{suffix}"


def make_copula(noun: str) -> str:
    """Türkçe bildirme eki ekler: Ankara'dır, Paris'tir, Berlin'dir, Tokyo'dur vb."""
    vowels = "aeıioöuüAEIİOÖUÜ"
    voiceless = "fstkçşhpFSTKÇŞHP"
    clean = noun.strip().rstrip("'\"")
    last_char = clean[-1]
    last_vowel = next((ch for ch in reversed(clean) if ch in vowels), "e")
    lead = "t" if last_char in voiceless else "d"
    if last_vowel in "eiEİ":
        v = "ir"
    elif last_vowel in "öüÖÜ":
        v = "ür"
    elif last_vowel in "aıAI":
        v = "ır"
    else:
        v = "ur"
    return f"{clean}'{lead}{v}"


def download_dataset() -> pd.DataFrame:
    """Hugging Face Hub'dan 'finnianx/capitals' veri setini çeker."""
    cache_glob = list(Path.home().glob(".cache/huggingface/hub/datasets--finnianx--capitals/**/*.parquet"))
    if cache_glob:
        print(f"Yerel önbellekten veri dosyası yükleniyor: {cache_glob[0]}")
        return pd.read_parquet(cache_glob[0])

    print(f"Hugging Face Hub üzerinde '{DATASET_REPO}' indiriliyor...")
    file_path = hf_hub_download(repo_id=DATASET_REPO, filename="data/train-00000-of-00001.parquet", repo_type="dataset")
    return pd.read_parquet(file_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Nora Capitals Dataset Ingestion & Training")
    parser.add_argument("--batch-size", type=int, default=32, help="Broca için batch büyüklüğü (varsayılan: 32)")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch iş parçacığı sayısı (varsayılan: 4)")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)

    print("=" * 65)
    print(" Nora NCC - Dünya Başkentleri Entegrasyonu ve Model Eğitimi")
    print(f" Kaynak: Hugging Face ({DATASET_REPO})")
    print("=" * 65)

    # 1. Veri setini indir
    df_raw = download_dataset()
    print(f"Veri seti başarıyla yüklendi: {len(df_raw)} ülke ve başkent eşleşmesi.")

    # 2. Türkçe ve İngilizce zenginleştirilmiş kayıtları üret
    enriched_records = []
    text_corpus_lines = []
    qa_pairs = []

    for _, row in df_raw.iterrows():
        c_en = str(row["country"]).strip()
        cap_en = str(row["capital"]).strip()

        if c_en in TR_COUNTRY_MAP:
            c_tr, cap_tr = TR_COUNTRY_MAP[c_en]
        else:
            c_tr, cap_tr = c_en, cap_en

        u_gen = make_genitive(c_tr)
        c_cop = make_copula(cap_tr)

        # Doğal Türkçe cümle
        tr_sentence = f"{u_gen} başkenti {c_cop}."
        en_sentence = f"The capital of {c_en} is {cap_en}."

        enriched_records.append({
            "country_en": c_en,
            "capital_en": cap_en,
            "country_tr": c_tr,
            "capital_tr": cap_tr,
            "sentence_tr": tr_sentence,
            "sentence_en": en_sentence,
        })

        text_corpus_lines.append(tr_sentence)
        text_corpus_lines.append(f"{c_tr} bir ülkedir.")
        text_corpus_lines.append(f"{cap_tr} bir başkenttir.")
        text_corpus_lines.append(f"{cap_tr}, {c_tr}'de bulunan bir şehirdir.")

        # Soru-Cevap varyasyonları (Epizodik QA için)
        a_primary = tr_sentence
        qa_pairs.append((f"{u_gen} başkenti neresidir?", a_primary))
        qa_pairs.append((f"{u_gen} başkenti neresi?", a_primary))
        qa_pairs.append((f"{u_gen} başkenti nedir?", a_primary))
        qa_pairs.append((f"{u_gen} başkenti hangi şehirdir?", a_primary))
        qa_pairs.append((f"{cap_tr} nerenin başkentidir?", f"{cap_tr}, {u_gen} başkentidir."))
        qa_pairs.append((f"{cap_tr} hangi ülkenin başkentidir?", f"{cap_tr}, {u_gen} başkentidir."))
        qa_pairs.append((f"{cap_tr} neresidir?", f"{cap_tr}, {u_gen} başkentidir."))

    # CSV ve TXT dosyalarını kaydet
    df_enriched = pd.DataFrame(enriched_records)
    OUT_CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_enriched.to_csv(OUT_CSV_FILE, index=False, encoding="utf-8")
    print(f"İki dilli başkent tablosu kaydedildi: {OUT_CSV_FILE}")

    OUT_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_TEXT_FILE, "w", encoding="utf-8") as f:
        for line in text_corpus_lines:
            f.write(line + "\n")
    print(f"Eğitim metin korpusu kaydedildi: {OUT_TEXT_FILE} ({len(text_corpus_lines)} cümle)")

    # 3. Nora Kalıcı Zihnini Yükle
    print(f"\nNora kalıcı zihni yükleniyor: {BRAIN_PATH} ...")
    if BRAIN_PATH.is_file():
        agi = NoraAGI.load(str(BRAIN_PATH), attach_broca_async=False)
        print("Mevcut 'nora_brain.nora' başarıyla yüklendi.")
    else:
        print("Yeni Nora zihni oluşturuluyor...")
        agi = NoraAGI(name="Nora", seed=42, boot_thinker=False, single_life=True)

    # 4. [1/3] Bilişsel Semantik Ağına (Knowledge Graph) Aktarım
    print(f"\n[1/3] Başkent ilişkileri semantik ağa aktarılıyor...")
    g = agi.lang.graph
    rmb = getattr(g, "relation_mb", None)
    if rmb is not None:
        rmb._auto_commit = False

    t0_graph = time.time()
    added_triples = 0

    try:
        for rec in enriched_records:
            u = tr_lower(rec["country_tr"])
            c = tr_lower(rec["capital_tr"])

            g.register_concept(u, bind_mb=False)
            g.register_concept(c, bind_mb=False)

            # Temel ontolojik ilişkiler
            g.add_edge(u, "ISA", "ülke", source="başkent_atlası", weight=1.0)
            g.add_edge(c, "ISA", "başkent", source="başkent_atlası", weight=1.0)
            g.add_edge(c, "LOCATED_AT", u, source="başkent_atlası", weight=1.0)
            g.add_edge(u, "CAPITAL", c, source="başkent_atlası", weight=1.0)
            g.add_edge(u, "HAS_PROP", f"başkenti {c}", source="başkent_atlası", weight=1.0)
            added_triples += 5

            # Ayrıca cümle olarak Wernicke / Graph işleme
            try:
                agi.lang.teach_sentence(rec["sentence_tr"], speaker="başkent_atlası")
            except Exception:
                pass
    finally:
        if rmb is not None:
            rmb._db.commit()
            rmb._auto_commit = True

    print(f"Semantik grafik aktarımı tamamlandı: {added_triples} yeni ilişki eklendi ({time.time() - t0_graph:.1f}s). Toplam Kenar: {len(g.edges)}")

    # 5. [2/3] Broca Konuşma / Artikülasyon Organına Eğitim (Nöral Dil Modeli)
    organ = getattr(agi.life.voice, "organ", None)
    if organ is not None:
        print(f"\n[2/3] Broca konuşma organına başkent cümleleri öğretiliyor ({len(text_corpus_lines)} cümle)...")
        t_broca = time.time()
        st = state_vector(agi)
        B = args.batch_size
        for i in range(0, len(text_corpus_lines), B):
            batch = text_corpus_lines[i : i + B]
            try:
                organ.learn_batch(batch, st, anchor_dropout=0.1, loss_mode="sampled", negative_samples=512)
            except Exception:
                pass

        organ.update_common()
        print(f"Broca eğitimi tamamlandı! Süre: {time.time() - t_broca:.1f}s")

    # 6. [3/3] Epizodik Diyalog Hafızasına (CA3 QA RAG) İndeksleme
    print(f"\n[3/3] Soru-cevap çiftleri epizodik diyalog hafızasına indeksleniyor ({len(qa_pairs)} soru-cevap)...")
    t_epi = time.time()
    epi = getattr(agi.broca_organ, "episodic", None)
    stop_words = {"ne", "nasıl", "nedir", "neydi", "kim", "kimdir", "hangi", "hangisi", "hangisidir", "kaç", "kaçtır", "nerede", "neresi", "neresidir", "nerededir", "yapılır", "et", "ol", "bir", "mi", "mı", "mu", "mü", "ve", "veya", "ile", "için", "dan", "den", "tan", "ten"}
    indexed_count = 0

    if epi is not None:
        for q_text, a_text in qa_pairs:
            q_low = tr_lower(q_text)
            words = re.findall(r"\b[\wçğıöşü]+\b", q_low)
            stems = [w for w in words if len(w) > 2 and w not in stop_words]
            if not stems:
                continue
            topic = stems[0] if stems else "başkent"
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
            indexed_count += 1

    print(f"Epizodik hafıza indekslemesi tamamlandı: {indexed_count} QA kaydı eklendi ({time.time() - t_epi:.1f}s).")

    # 7. Kalıcı Beyin Dosyasını Kaydet
    print(f"\nBirleşik beyin güncellenerek kaydediliyor: {BRAIN_PATH} ...")
    agi.save(str(BRAIN_PATH))
    print(f"Başarıyla kaydedildi ({BRAIN_PATH})! Başkent bilgileri kalıcı hale getirildi.")

    # 8. Canlı Test ve Doğrulama
    print("\n" + "=" * 65)
    print(" Canlı Doğrulama Testi (Nora Soru-Cevap)")
    print("=" * 65)

    test_queries = [
        "Türkiye'nin başkenti neresidir?",
        "Fransa'nın başkenti neresidir?",
        "Almanya'nın başkenti nedir?",
        "Japonya'nın başkenti neresidir?",
        "İtalya'nın başkenti neresi?",
        "Azerbaycan'ın başkenti neresidir?",
        "Tokyo hangi ülkenin başkentidir?",
        "Paris nerenin başkentidir?",
    ]

    for q in test_queries:
        ans = agi.converse(q, speaker="insan")
        print(f"Soru : {q}")
        print(f"Nora : {ans}\n")


if __name__ == "__main__":
    main()
