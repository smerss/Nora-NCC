"""tools/ingest_knowledge.py

Nora NCC - Evrensel ve Otomatik Bilgi Entegrasyon Motoru (Universal Knowledge Ingestion Engine).

Herhangi bir alandaki yapılandırılmış bilgiyi (bilim, neden-sonuç, coğrafya, tıp, günlük hayat):
  1. Bilişsel Semantik Ağa (Knowledge Graph - Çift yönlü ontolojik ilişkiler)
  2. Broca Artikülasyon Organına (Nöral dil ağırlıklarına cümle kalıpları)
  3. Epizodik Diyalog Hafızasına (CA3 RAG - Soru-Cevap indeksi)
otomatik olarak işler ve 'state/nora_brain.nora' dosyasına mühürler.
Yeni eklenen her alan için agi/lang kodunu değiştirmeye gerek bırakmaz!
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

BRAIN_PATH = _REPO_ROOT / "state" / "nora_brain.nora"
DATA_DIR = _REPO_ROOT / "data"
TEXT_DIR = DATA_DIR / "text"

# -----------------------------------------------------------------------------
# 3. Kategori: Neden-Sonuç ve Bilimsel Sağduyu Veri Tabanı (Causal & Scientific Commonsense)
# -----------------------------------------------------------------------------
SCIENCE_CAUSAL_DATA = [
    {
        "concept": "gökkuşağı",
        "triples": [
            ("gökkuşağı", "ISA", "optik doğa olayı"),
            ("gökkuşağı", "CAUSES_BY", "güneş ışığının yağmur damlacıklarında kırılması ve yansıması"),
            ("güneş ışığının yağmur damlacıklarında kırılması ve yansıması", "CAUSES", "gökkuşağı"),
        ],
        "sentences": [
            "Gökkuşağı, güneş ışığının yağmur damlacıklarında kırılması ve yansımasıyla oluşur.",
            "Gökkuşağı gökyüzünde beliren rengârenk bir optik doğa olayıdır.",
        ],
        "qa": [
            ("Gökkuşağı nasıl oluşur?", "Gökkuşağı, güneş ışığının yağmur damlacıklarında kırılması ve yansımasıyla oluşur."),
            ("Gökkuşağı neden olur?", "Gökkuşağı, güneş ışığının yağmur damlacıklarında kırılması ve yansımasıyla meydana gelir."),
            ("Gökkuşağı nedir?", "Gökkuşağı, yağmur sonrası ışığın kırılmasıyla oluşan bir optik doğa olayıdır."),
        ],
    },
    {
        "concept": "fotosentez",
        "triples": [
            ("fotosentez", "ISA", "biyolojik süreç"),
            ("fotosentez", "USED_FOR", "bitkilerin güneş ışığıyla besin ve oksijen üretmesi"),
            ("fotosentez", "CAUSES", "oksijen ve glikoz üretimi"),
            ("fotosentez", "CAUSES_BY", "klorofilin güneş ışığı, su ve karbondioksiti birleştirmesi"),
        ],
        "sentences": [
            "Fotosentez, bitkilerin güneş enerjisini kullanarak besin ve oksijen üretmesini sağlar.",
            "Fotosentez yeryüzündeki oksijen döngüsünün temel kaynağıdır.",
        ],
        "qa": [
            ("Fotosentez ne işe yarar?", "Fotosentez, bitkilerin güneş ışığıyla besin ve oksijen üretmesini sağlar."),
            ("Fotosentez ne için kullanılır?", "Fotosentez, bitkilerde organik besin ve oksijen üretimi için kullanılır."),
            ("Fotosentez neye yol açar?", "Fotosentez, oksijen ve glikoz üretimine yol açar."),
            ("Fotosentez nedir?", "Fotosentez, bitkilerin ışık enerjisiyle besin ürettiği biyolojik bir süreçtir."),
        ],
    },
    {
        "concept": "deprem",
        "triples": [
            ("deprem", "ISA", "jeolojik doğa olayı"),
            ("deprem", "CAUSES_BY", "yer kabuğundaki fay hatlarının kırılması ve enerji boşalması"),
            ("fay hatlarının kırılması", "CAUSES", "deprem"),
            ("deprem", "CAUSES", "yer kabuğunda sarsıntılar ve sismik dalgalar"),
        ],
        "sentences": [
            "Deprem, yer kabuğundaki fay hatlarının kırılması ve biriken enerjinin aniden boşalmasıyla oluşur.",
            "Deprem yer kabuğunda şiddetli sarsıntılara ve sismik dalgalara yol açar.",
        ],
        "qa": [
            ("Deprem neden olur?", "Deprem, yer kabuğundaki fay hatlarının kırılması ve biriken enerjinin açığa çıkmasıyla olur."),
            ("Deprem nasıl oluşur?", "Deprem, tektonik plakaların hareketi ve fay hatlarının kırılmasıyla oluşur."),
            ("Deprem neye yol açar?", "Deprem, yer kabuğunda sarsıntılara ve sismik dalgalara yol açar."),
            ("Deprem nedir?", "Deprem, yer kabuğunda meydana gelen ani sarsıntılardır."),
        ],
    },
    {
        "concept": "yağmur",
        "triples": [
            ("yağmur", "ISA", "yağış türü"),
            ("yağmur", "CAUSES_BY", "su buharının atmosferde soğuyarak yoğunlaşması"),
            ("su buharının yoğuşması", "CAUSES", "yağmur"),
            ("yağmur", "USED_FOR", "tatlı su kaynaklarının yenilenmesi ve toprağın sulanması"),
            ("yağmur", "CAUSES", "toprağın nemlenmesi ve su döngüsünün sürmesi"),
        ],
        "sentences": [
            "Yağmur, buharlaşan suyun atmosferde soğuyup yoğuşarak su damlalarına dönüşmesiyle oluşur.",
            "Yağmur tatlı su kaynaklarını besler ve ekosistemin su ihtiyacını karşılar.",
        ],
        "qa": [
            ("Yağmur nasıl oluşur?", "Yağmur, su buharının atmosferde soğuyarak su damlacıklarına yoğuşmasıyla oluşur."),
            ("Yağmur neden yağar?", "Yağmur, bulutlardaki su damlacıklarının ağırlaşarak yerçekimi etkisiyle düşmesiyle yağar."),
            ("Yağmur ne işe yarar?", "Yağmur, toprağın sulanması ve tatlı su döngüsünün sürmesi için kullanılır."),
        ],
    },
    {
        "concept": "rüzgâr",
        "triples": [
            ("rüzgâr", "ISA", "meteorolojik hava hareketi"),
            ("rüzgâr", "CAUSES_BY", "yüksek basınç ile alçak basınç alanları arasındaki basınç farkı"),
            ("basınç farkı", "CAUSES", "rüzgâr"),
            ("rüzgâr", "USED_FOR", "hava sirkülasyonu ve rüzgâr türbinleriyle elektrik üretmek"),
        ],
        "sentences": [
            "Rüzgâr, havadaki sıcaklık ve basınç farkı nedeniyle yüksek basınçtan alçak basınca doğru esen hava akımıdır.",
        ],
        "qa": [
            ("Rüzgâr nasıl oluşur?", "Rüzgâr, atmosferdeki yüksek basınç ile alçak basınç alanları arasındaki basınç farkından doğar."),
            ("Rüzgâr neden olur?", "Rüzgâr, sıcaklık ve basınç farklarının hava kütlelerini harekete geçirmesiyle oluşur."),
            ("Rüzgâr ne işe yarar?", "Rüzgâr, hava sirkülasyonunu sağlar ve rüzgâr enerjisi üretmek için kullanılır."),
        ],
    },
    {
        "concept": "güneş tutulması",
        "triples": [
            ("güneş tutulması", "ISA", "astronomik olay"),
            ("güneş tutulması", "CAUSES_BY", "Ay'ın Dünya ile Güneş arasına girerek Güneş'i örtmesi"),
            ("Ay'ın Dünya ile Güneş arasına girmesi", "CAUSES", "güneş tutulması"),
        ],
        "sentences": [
            "Güneş tutulması, Ay'ın Dünya ile Güneş arasına girmesi ve Güneş ışınlarını gölgelemesiyle gerçekleşir.",
        ],
        "qa": [
            ("Güneş tutulması nasıl gerçekleşir?", "Güneş tutulması, Ay'ın Dünya ile Güneş arasına girerek Güneş'i gölgelemesiyle gerçekleşir."),
            ("Güneş tutulması nasıl oluşur?", "Güneş tutulması, Ay'ın yörünge hareketi sırasında Güneş ile Dünya arasına girmesiyle oluşur."),
        ],
    },
    {
        "concept": "ay tutulması",
        "triples": [
            ("ay tutulması", "ISA", "astronomik olay"),
            ("ay tutulması", "CAUSES_BY", "Dünya'nın Güneş ile Ay arasına girip Ay'ın üzerine gölge düşürmesi"),
        ],
        "sentences": [
            "Ay tutulması, Dünya'nın Güneş ile Ay arasına girmesi ve gölgesinin Ay üzerine düşmesiyle oluşur.",
        ],
        "qa": [
            ("Ay tutulması nasıl gerçekleşir?", "Ay tutulması, Dünya'nın Güneş ile Ay arasına girerek Ay'a gelen ışığı kesmesiyle gerçekleşir."),
        ],
    },
    {
        "concept": "aşı",
        "triples": [
            ("aşı", "ISA", "biyolojik koruyucu tıp aracı"),
            ("aşı", "USED_FOR", "bağışıklık sistemini mikroplara karşı eğitmek ve hastalıklardan korumak"),
            ("aşı", "CAUSES", "antikor üretimi ve bağışıklık hafızasının oluşması"),
        ],
        "sentences": [
            "Aşı, zayıflatılmış mikroplarla vücudun bağışıklık sistemini önceden eğiterek hastalıklara karşı korur.",
        ],
        "qa": [
            ("Aşı ne işe yarar?", "Aşı, bağışıklık sistemini mikroplara karşı eğiterek hastalıklardan korunmak için kullanılır."),
            ("Aşı ne için kullanılır?", "Aşı, bulaşıcı hastalıklara karşı vücutta antikor ve bağışıklık oluşturmak için kullanılır."),
        ],
    },
    {
        "concept": "antibiyotik",
        "triples": [
            ("antibiyotik", "ISA", "tedavi edici ilaç"),
            ("antibiyotik", "USED_FOR", "bakteriyel enfeksiyonları tedavi etmek ve bakterileri yok etmek"),
            ("antibiyotik", "PREVENTS", "bakterilerin çoğalması"),
        ],
        "sentences": [
            "Antibiyotik, vücutta hastalığa neden olan zararlı bakterileri yok etmek veya çoğalmasını durdurmak için kullanılır.",
        ],
        "qa": [
            ("Antibiyotik ne işe yarar?", "Antibiyotik, bakteri kaynaklı enfeksiyonları tedavi etmek ve bakterileri yok etmek için kullanılır."),
            ("Antibiyotik ne için kullanılır?", "Antibiyotik, bakteriyel hastalıkların tedavisinde kullanılır."),
        ],
    },
    {
        "concept": "erozyon",
        "triples": [
            ("erozyon", "ISA", "toprak aşınması"),
            ("erozyon", "CAUSES_BY", "rüzgâr ve su akıntılarının verimli üst toprağı süpürmesi"),
            ("erozyon", "CAUSES", "verimli tarım arazilerinin kaybı ve çölleşme"),
        ],
        "sentences": [
            "Erozyon, bitki örtüsünün yetersiz olduğu alanlarda su ve rüzgârın toprağı aşındırmasıyla meydana gelir.",
        ],
        "qa": [
            ("Erozyon neden olur?", "Erozyon, bitki örtüsü zayıf olan toprağın rüzgâr ve su tarafından süpürülmesiyle meydana gelir."),
            ("Erozyon neye yol açar?", "Erozyon, verimli tarım arazilerinin kaybına ve çölleşmeye yol açar."),
        ],
    },
    {
        "concept": "sera etkisi",
        "triples": [
            ("sera etkisi", "ISA", "atmosferik ısınma mekanizması"),
            ("sera etkisi", "CAUSES_BY", "atmosferdeki karbondioksit ve metan gibi sera gazlarının ısıyı hapsetmesi"),
            ("sera etkisi", "CAUSES", "küresel sıcaklık artışı ve iklim değişikliği"),
        ],
        "sentences": [
            "Sera etkisi, atmosferdeki sera gazlarının yeryüzünden yansıyan ısıyı tutarak gezegenin ısınmasına yol açmasıdır.",
        ],
        "qa": [
            ("Sera etkisi nedir?", "Sera etkisi, atmosferdeki gazların ısıyı tutarak Dünya'nın sıcaklığını koruması ve artırmasıdır."),
            ("Sera etkisi neye yol açar?", "Sera etkisi, küresel ısınmaya ve iklim değişikliklerine yol açar."),
        ],
    },
    {
        "concept": "yerçekimi",
        "triples": [
            ("yerçekimi", "ISA", "temel kütleçekim kuvveti"),
            ("yerçekimi", "CAUSES", "kütlelerin birbirini çekmesi ve cisimlerin yere düşmesi"),
            ("yerçekimi", "USED_FOR", "gezegenlerin yörüngede kalması ve atmosferin uzaya dağılmaması"),
        ],
        "sentences": [
            "Yerçekimi, kütlesi olan cisimlerin birbirini çekmesini sağlayan temel bir doğa kuvvetidir.",
        ],
        "qa": [
            ("Yerçekimi ne işe yarar?", "Yerçekimi, gezegenleri yörüngede tutar ve atmosferin uzaya dağılmasını engeller."),
            ("Yerçekimi neye yol açar?", "Yerçekimi, cisimlerin yere düşmesine ve kütlelerin birbirine çekilmesine yol açar."),
        ],
    },
    {
        "concept": "gelgit",
        "triples": [
            ("gelgit", "ISA", "okyanus ve deniz su hareketi"),
            ("gelgit", "CAUSES_BY", "Ay ve Güneş'in Dünya üzerindeki sulara uyguladığı kütleçekim kuvveti"),
        ],
        "sentences": [
            "Gelgit, Ay ve Güneş'in kütleçekim kuvveti etkisiyle deniz seviyesinin periyodik olarak yükselip alçalmasıdır.",
        ],
        "qa": [
            ("Gelgit nasıl oluşur?", "Gelgit, Ay ve Güneş'in çekim kuvvetinin deniz ve okyanus sularını çekmesiyle oluşur."),
        ],
    },
    {
        "concept": "volkan patlaması",
        "triples": [
            ("volkan patlaması", "ISA", "jeolojik doğa olayı"),
            ("volkan patlaması", "CAUSES_BY", "magmanın yer kabuğundaki yüksek basınç nedeniyle yüzeye püskürmesi"),
            ("volkan patlaması", "CAUSES", "lav, kül ve volkanik gazların yayılması"),
        ],
        "sentences": [
            "Volkan patlaması, yerin altındaki sıcak magmanın yüksek gaz basıncıyla yer kabuğundan dışarı püskürmesidir.",
        ],
        "qa": [
            ("Volkan patlaması neden olur?", "Volkan patlaması, yer kabuğunun altındaki erimiş magmanın yüksek gaz basıncıyla yüzeye çıkmasıyla oluşur."),
            ("Volkan patlaması neye yol açar?", "Volkan patlaması, çevreye lav, kül ve zehirli gazların yayılmasına yol açar."),
        ],
    },
    {
        "concept": "buharlaşma",
        "triples": [
            ("buharlaşma", "ISA", "fiziksel hâl değişimi"),
            ("buharlaşma", "CAUSES_BY", "sıvı moleküllerinin ısı enerjisi alarak gaz fazına geçmesi"),
        ],
        "sentences": [
            "Buharlaşma, sıvının yüzeyindeki moleküllerin ısı alarak gaz hâline geçmesidir.",
        ],
        "qa": [
            ("Buharlaşma nasıl gerçekleşir?", "Buharlaşma, sıvı moleküllerinin ısı enerjisi kazanarak gaz hâline geçmesiyle gerçekleşir."),
        ],
    },
    {
        "concept": "yoğuşma",
        "triples": [
            ("yoğuşma", "ISA", "fiziksel hâl değişimi"),
            ("yoğuşma", "CAUSES_BY", "gaz moleküllerinin ısı verip soğuyarak sıvı fazına geçmesi"),
        ],
        "sentences": [
            "Yoğuşma, gaz hâlindeki bir maddenin ısısını kaybedip soğuyarak sıvı hâle dönüşmesidir.",
        ],
        "qa": [
            ("Yoğuşma nasıl gerçekleşir?", "Yoğuşma, gaz moleküllerinin soğuyarak sıvı hâle geçmesiyle gerçekleşir."),
        ],
    },
]


def ingest_knowledge_domain(domain_name: str, items: list[dict], batch_size: int = 32, threads: int = 4):
    """Herhangi bir yapılandırılmış bilgi paketini Nora'nın bilişsel ve nöral organlarına işler."""
    torch.set_num_threads(threads)

    print("=" * 70)
    print(f" Nora NCC - Evrensel Bilgi Entegrasyonu: [{domain_name.upper()}]")
    print("=" * 70)

    # 1. Veri Hazırlığı
    all_triples = []
    all_sentences = []
    all_qa = []
    all_records = []

    for item in items:
        concept = item.get("concept", "")
        triples = item.get("triples", [])
        sentences = item.get("sentences", [])
        qa = item.get("qa", [])

        all_triples.extend(triples)
        all_sentences.extend(sentences)
        all_qa.extend(qa)

        for s, r, o in triples:
            all_records.append({
                "concept": concept,
                "subject": s,
                "relation": r,
                "object": o,
            })

    # Verileri yerel CSV ve TXT dosyalarına kaydet
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    csv_file = DATA_DIR / f"{domain_name}.csv"
    txt_file = TEXT_DIR / f"{domain_name}.txt"

    pd.DataFrame(all_records).to_csv(csv_file, index=False, encoding="utf-8")
    with open(txt_file, "w", encoding="utf-8") as f:
        for s in all_sentences:
            f.write(s + "\n")

    print(f"Veri kayıtları oluşturuldu: {len(all_triples)} üçlü, {len(all_sentences)} cümle, {len(all_qa)} S&C.")
    print(f"Kayıt dosyaları: {csv_file} ve {txt_file}")

    # 2. Kalıcı Zihni Yükle
    print(f"\nNora kalıcı zihni yükleniyor: {BRAIN_PATH} ...")
    if BRAIN_PATH.is_file():
        agi = NoraAGI.load(str(BRAIN_PATH), attach_broca_async=False)
        print("Mevcut 'nora_brain.nora' başarıyla yüklendi.")
    else:
        print("Yeni Nora zihni oluşturuluyor...")
        agi = NoraAGI(name="Nora", seed=42, boot_thinker=False, single_life=True)

    g = agi.lang.graph
    print(f"Başlangıç semantik kenar sayısı: {len(g.edges)}")

    # 3. [1/3] Bilişsel Semantik Ağa (Knowledge Graph) Aktar
    print(f"\n[1/3] Semantik grafiğe aktarım başlıyor ({len(all_triples)} üçlü)...")
    t0_graph = time.time()
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
            g.add_edge(s_clean, r, o_clean, source=f"bilgi_{domain_name}", weight=1.0)
            added_edges += int(len(g.edges) > before)

            # Doğal cümle olarak Wernicke çözümlemesine de ver
            try:
                agi.lang.teach_sentence(f"{s_clean} bir {o_clean} türüdür.", speaker=f"bilgi_{domain_name}")
            except Exception:
                pass
    finally:
        if rmb is not None:
            rmb._db.commit()
            rmb._auto_commit = True

    print(f"Semantik aktarım tamamlandı: {added_edges} yeni kenar ({time.time() - t0_graph:.1f}s). Toplam: {len(g.edges)}")

    # 4. [2/3] Broca Konuşma Organına Nöral Cümle Eğitimi
    organ = getattr(agi.life.voice, "organ", None)
    if organ is not None and all_sentences:
        print(f"\n[2/3] Broca konuşma organına cümle kalıpları öğretiliyor ({len(all_sentences)} cümle)...")
        t_broca = time.time()
        st = state_vector(agi)
        for i in range(0, len(all_sentences), batch_size):
            batch = all_sentences[i : i + batch_size]
            try:
                organ.learn_batch(batch, st, anchor_dropout=0.1, loss_mode="sampled", negative_samples=512)
            except Exception:
                pass
        organ.update_common()
        print(f"Broca eğitimi tamamlandı! Süre: {time.time() - t_broca:.1f}s")

    # 5. [3/3] Epizodik Diyalog Hafızasına (CA3 QA RAG) İndeksleme
    epi = getattr(agi.broca_organ, "episodic", None)
    if epi is not None and all_qa:
        print(f"\n[3/3] Soru-cevap çiftleri epizodik hafızaya indeksleniyor ({len(all_qa)} QA)...")
        t_epi = time.time()
        stop_words = {"ne", "nasıl", "nedir", "neydi", "kim", "kimdir", "hangi", "hangisi", "hangisidir", "kaç", "kaçtır", "nerede", "neresi", "neresidir", "nerededir", "yapılır", "et", "ol", "bir", "mi", "mı", "mu", "mü", "ve", "veya", "ile", "için", "dan", "den", "tan", "ten"}
        indexed_qa = 0
        for q_text, a_text in all_qa:
            q_low = tr_lower(q_text)
            words = re.findall(r"\b[\wçğıöşü]+\b", q_low)
            stems = [w for w in words if len(w) > 2 and w not in stop_words]
            if not stems:
                continue
            topic = stems[0]
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
        print(f"Epizodik indeksleme tamamlandı: {indexed_qa} QA eklendi ({time.time() - t_epi:.1f}s).")

    # 6. Kalıcı Olarak Kaydet
    print(f"\nBirleşik beyin güncellenerek kaydediliyor: {BRAIN_PATH} ...")
    agi.save(str(BRAIN_PATH))
    print(f"Başarıyla kaydedildi ({BRAIN_PATH})! [{domain_name.upper()}] kalıcı hale getirildi.")

    # 7. Doğrulama Testi
    print("\n" + "=" * 70)
    print(f" Canlı Doğrulama Testi: [{domain_name.upper()}]")
    print("=" * 70)

    test_samples = [item["qa"][0][0] for item in items if item.get("qa")]
    for q in test_samples[:8]:
        ans = agi.converse(q, speaker="insan")
        print(f"Soru : {q}")
        print(f"Nora : {ans}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Nora Universal Knowledge Ingestion Engine")
    parser.add_argument("--domain", default="science_causal", choices=["science_causal"], help="İşlenecek bilgi alanı")
    parser.add_argument("--file", default="", help="İsteğe bağlı harici JSON bilgi dosyası")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.file and os.path.isfile(args.file):
        with open(args.file, "r", encoding="utf-8") as f:
            items = json.load(f)
        domain = Path(args.file).stem
    elif args.domain == "science_causal":
        items = SCIENCE_CAUSAL_DATA
        domain = "science_causal"
    else:
        raise ValueError(f"Geçersiz alan: {args.domain}")

    ingest_knowledge_domain(domain, items, batch_size=args.batch_size, threads=args.threads)


if __name__ == "__main__":
    main()
