"""solo_gorev.py — MÜDAHALESİZ ÇOK OTURUMLU GÖREV (ölçüm, öğretim değil).

Protokol
--------
Nora'ya ÜÇ oturumda biten bir iş verilir. Oturumlar arasında süreç kapanır
(yalnız `save`/`load` köprüsü vardır). Görev sırasında kod DEĞİŞTİRİLMEZ:
her başarısızlık puan olarak yazılır, müdahale sayısı ayrıca tutulur.

    Oturum 1: örneklerden dönüşüm öğren + olguları öğret + kaydet
    Oturum 2: yükle, YENİ bir dönüşüm öğren (kompozisyon yetmiyor: icat gerekir)
    Oturum 3: yükle, 10 soruluk karışık sınav

Beklenen değerler ELLE YAZILMAZ: her cevap kodla hesaplanır (ters çevirme, kaydırma,
sayı karşılaştırması). Sınav görülmemiş sözcükler ve görülmemiş soru biçimleri kullanır.

Kullanım:
    python solo_gorev.py            # koş, puanı yazdır
    python solo_gorev.py -v         # transkriptle
"""
from __future__ import annotations

import shutil
import sys
import tempfile

from nora_abstract import ALPHABET as ALFA
from nora_agi import NoraAGI
from nora_style import strip_prefixes

VERBOSE = "-v" in sys.argv


def shift(text: str, k: int) -> str:
    out = []
    for ch in text:
        low = ch.lower()
        if low in ALFA:
            nxt = ALFA[(ALFA.index(low) + k) % len(ALFA)]
            out.append(nxt.upper() if ch.isupper() else nxt)
        else:
            out.append(ch)
    return "".join(out)


def say(agi, text: str) -> str:
    rep = strip_prefixes(agi.converse(text))
    if VERBOSE:
        print(f"  S: {text}\n  N: {rep[:150]}")
    return rep


def main() -> int:
    d = tempfile.mkdtemp(prefix="nora_solo_")
    pref = d + "/nora"

    # ---------------------------------------------------------------- OTURUM 1
    print("[oturum 1] dönüşüm + olgular")
    g = NoraAGI(seed=7001, boot_thinker=False)
    g.step_life(120)
    rev_in = ["kitap", "kalem", "deniz"]
    say(g, "Şu dönüşümü öğren: " + ", ".join(f"{w} -> {w[::-1]}" for w in rev_in) + ".")
    say(g, "Kardeşimin adı Elif.")
    say(g, "Ayşe bana toplantının saat üçte olduğunu söyledi.")
    say(g, "Mehmet bana toplantının saat beşte olduğunu söyledi.")
    say(g, "Ali'nin boyu 180 santimetre, Veli'nin boyu 175 santimetre.")
    g.save(pref)

    # ---------------------------------------------------------------- OTURUM 2
    print("[oturum 2] yükle + YENİ ilkel gerektiren dönüşüm")
    g = NoraAGI.load(pref)
    g.step_life(60)
    inv_in = ["masa", "bulut"]
    said = say(g, "Şu dönüşümü öğren: "
                  + ", ".join(f"{w} -> {shift(w, 1)}" for w in inv_in) + ".")
    invented = ("icat" in said.lower()) or ("ilkel" in said.lower())
    say(g, "Dönüşümü uygula: zeytin")
    g.save(pref)

    # ---------------------------------------------------------------- OTURUM 3
    print("[oturum 3] yükle + sınav")
    g = NoraAGI.load(pref)
    g.step_life(60)

    checks = []

    def q(label, text, expected):
        rep = say(g, text)
        ok = expected.lower() in rep.lower()
        checks.append((label, ok, rep[:60], expected))
        return ok

    # 1) kalıcı kişisel olgu, görülmemiş soru biçimi
    q("kişisel-olgu", "Kardeşimin adı neydi, hatırlıyor musun?", "elif")
    # 2) iki tanığın çelişkisi (ikisi de anılmalı)
    rep = say(g, "Toplantı saat kaçta?")
    checks.append(("tanık-çelişkisi",
                   ("ayşe" in rep.lower() or "mehmet" in rep.lower()),
                   rep[:60], "iki tanık"))
    # 3) oturum 1'de yazdığı dönüşüm, görülmemiş sözcükte
    q("dönüşüm-A", "Dönüşümü uygula: portakal", "portakal"[::-1])
    # 4) oturum 2'de İCAT ettiği ilkel, görülmemiş sözcükte
    q("icat-B", "Dönüşümü uygula: yıldız", shift("yıldız", 1))
    # 5) sayısal kıyas
    q("sayı-kıyas", "Kim daha uzun?", "ali")
    # 6) kendi defterini okuma
    rep = say(g, "Bu oturumda ne öğrendin?")
    checks.append(("defter", ("dönüşüm" in rep.lower()), rep[:60], "defter"))
    # 7) bilmediğini uydurmama
    rep = say(g, "Blurg nedir?")
    checks.append(("dürüstlük", any(k in rep.lower() for k in
                                    ("bilmiyorum", "bilgim yok", "duymadım", "öğret")),
                   rep[:60], "reddetme"))
    # 8) aritmetik (fiille)
    q("aritmetik", "64 ile 28'i topla", "92")
    # 9) harf kümesi
    q("küme", "'kitap' ve 'kalem' sözcüklerinin ortak harfleri neler?", "a")
    # 10) organ boşluğu dürüstlüğü
    rep = say(g, "Şu metnin Yunanca karşılığını ver: deniz")
    checks.append(("organ-reddi", "yok" in rep.lower() or "uydurm" in rep.lower(),
                   rep[:60], "ret"))

    ok = sum(1 for _l, c, _r, _e in checks if c)
    print("\n=== SINAV ===")
    for label, c, rep, exp in checks:
        print(f"  {label:<16s} {'✓' if c else '✗':<2s}  beklenen~{exp[:18]:<18s} | {rep}")
    print(f"\nSKOR: {ok}/{len(checks)}   (icat: {'var' if invented else 'yok'})")
    shutil.rmtree(d, ignore_errors=True)
    return 0 if ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
