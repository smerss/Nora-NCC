"""Sözce öz-sınıflandırıcısının çapraz-tohum (seed) dayanıklılığını ölçer."""

import sys, os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import nora  # noqa: F401  (eski nora_xxx import'ları için uyumluluk kancası)
from nora_autonomous_logic import NeuralUtteranceCortex
from nora_chem import NeuroChemistry
from nora_lang import TurkishMorphology

PROBES = [
    # (metin, beklenen_etiket_ailesi, soru_mu)
    ("elektron aynı anda hem dalga hem parçacık olabilir mi", ("SORU_",), True),
    ("enerji korunumlu bir büyüklük müdür", ("SORU_",), True),
    ("bunu bana biraz açar mısın", ("SORU_",), True),
    ("bilinç fiziksel süreç olarak açıklanabilir mi", ("SORU_",), True),
    ("sence bu doğru mu", ("SORU_",), True),
    ("elektron ölçülürse ne olur", ("SORU_",), True),
    ("sıcaklık artarsa basınç nasıl değişir", ("SORU_",), True),
    ("gözlem yapılmazsa hangi durumda kalır", ("SORU_",), True),
    ("kuantum dolanıklık nedir", ("SORU_",), True),
    ("entropi nasıl tanımlanır", ("SORU_",), True),
    ("bu olay neden gerçekleşir", ("SORU_",), True),
    ("bunun sebebi nedir", ("SORU_",), True),
    ("hangi durumda süperpozisyon bozulur", ("SORU_",), True),
    ("çok kararsız kaldım ne önerirsin", ("TAVSIYE_KARAR",), True),
    ("bana bir yol gösterir misin", ("TAVSIYE_KARAR",), True),
    ("günaydın nora", ("SOSYAL_NEZAKET",), False),
    ("kolay gelsin sana", ("SOSYAL_NEZAKET",), False),
    ("çok geçmiş olsun", ("SOSYAL_NEZAKET",), False),
    ("özür dilerim geciktim", ("SOSYAL_NEZAKET",), False),
    ("bugün biraz hastayım başım ağrıyor", ("DUYGUSAL_PAYLASIM",), False),
    ("kafam çok karışık odaklanamıyorum", ("DUYGUSAL_PAYLASIM",), False),
    ("çok stresliyim içim daralıyor", ("DUYGUSAL_PAYLASIM",), False),
    ("sınavı kazandım çok mutluyum", ("DUYGUSAL_PAYLASIM",), False),
    ("elektron gözlemlenmeden önce hem dalga hem parçacık durumundadır", ("IKILI_SUPERPOZISYON",), False),
    ("foton aynı anda hem dalga hem tanecik özelliği gösterir", ("IKILI_SUPERPOZISYON",), False),
    ("konum belirsizliği azaldıkça momentum belirsizliği artar", ("ORANTI_BAGINTI",), False),
    ("dalga boyu kısaldıkça frekans yükselir", ("ORANTI_BAGINTI",), False),
    ("basınç ile hacim ters orantılıdır", ("ORANTI_BAGINTI",), False),
    ("gözlem yapılırsa süperpozisyon bozulur", ("KOSULLU_YASA",), False),
    ("sıcaklık sıfırın altına düşerse su donar", ("KOSULLU_YASA",), False),
    ("ölçüm dalga fonksiyonunu tek bir özduruma çökertir", ("OLGU_EYLEM_YASASI", "KOSULLU_YASA"), False),
    ("enzimler kimyasal tepkimeleri hızlandırır", ("OLGU_EYLEM_YASASI", "KOSULLU_YASA"), False),
    ("bağışıklık sistemi patojeni tanır", ("OLGU_EYLEM_YASASI", "KOSULLU_YASA"), False),
    ("bence bu konuda haklısın katılıyorum", ("GORUS_TARTISMA", "SOSYAL_NEZAKET", "DUYGUSAL_PAYLASIM"), False),
    ("hayır bence tam öyle değil yanılıyorsun", ("GORUS_TARTISMA", "SOSYAL_NEZAKET", "DUYGUSAL_PAYLASIM"), False),
]


def evaluate(seed: int) -> tuple[float, float, list[str]]:
    chem = NeuroChemistry()
    morph = TurkishMorphology()
    uc = NeuralUtteranceCortex(chem, morph, seed=seed)
    lab_ok = 0
    q_ok = 0
    misses = []
    for text, allowed, is_q in PROBES:
        r = uc.classify(text)
        if r["label"] in allowed or (allowed == ("SORU_",) and r["label"].startswith("SORU_")):
            lab_ok += 1
        else:
            misses.append(f"{r['label']} <- {text}")
        q_ok += int(bool(r["is_question"]) == is_q)
    return lab_ok / len(PROBES), q_ok / len(PROBES), misses


if __name__ == "__main__":
    tot_l = tot_q = 0.0
    seeds = [7, 11, 47, 104, 105, 106, 107, 2024]
    for sd in seeds:
        l, q, misses = evaluate(sd)
        tot_l += l
        tot_q += q
        print(f"seed={sd:<5} etiket={l*100:5.1f}%  soru={q*100:5.1f}%")
        if l < 0.9:
            for m in misses[:6]:
                print("    -", m)
    print(f"\nORTALAMA: etiket={tot_l/len(seeds)*100:.1f}%  soru={tot_q/len(seeds)*100:.1f}%")
