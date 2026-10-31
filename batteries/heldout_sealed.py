"""heldout_sealed.py — MÜHÜRLÜ SET (sealed): yazıldı, bir kez koşuldu, SONRADAN DÜZELTİLMEDİ.

Neden mühür?
------------
`heldout_battery.py` koda bakmadan yazıldı ama sonra geçsin diye operatör eklendi. Bu
meşru (eksik primitif eklemek işin kendisi) ama tek başına yeterli kanıt değil: aynı kişi
hem sınavı hem cevabı yazdı. Mühürlü set farklı bir disiplin uygular:

    * testler bu turda, sistemin davranışına BAKILMADAN yazıldı,
    * ilk koşudan sonra NE TEST NE KOD değiştirildi,
    * sonuç ne olursa olsun bu dosyada yazılı kalır.

Buradaki bir başarısızlık "yapılacaklar" demektir; başarı ise şans değil, önceki
operatörlerin yan ürünüdür. Amaç sayıyı büyütmek değil, sayının DÜRÜST kalması.

Kapsam (ana setten farklı, çok adımlı / birleşik):
    S1 çok sözcüklü öğeleri heceye göre sıralama
    S2 istisna + farklı ölçüt birlikte
    S3 üç nedenli collider (ikiden fazla gelen)
    S4 meta cevabın "yok" dediği duyuyla doğrudan istek arasındaki tutarlılık
"""
from __future__ import annotations

import sys, os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


import re

import nora_agi as NA
from nora_lang import tr_lower


def _order(text: str, items: list[str]) -> list[int]:
    low = tr_lower(text)
    pos = [(low.find(tr_lower(x)), i) for i, x in enumerate(items)]
    pos = [(p, i) for p, i in pos if p >= 0]
    return [i for _p, i in sorted(pos)]


def s1_sort_by_syllables():
    """S1 — Çok sözcüklü öğeler, hece sayısına göre artan sıralama.

    Girdi: "kelebek uçuyor" (6 hece), "su" (1), "kara kutu" (4).
    Beklenen sıra: su, kara kutu, kelebek uçuyor.
    Not: öğeler tırnak içinde ve virgülle ayrılmış; ölçüt parite değil, hece sayısı.
    """
    a = NA.NoraAGI(seed=1001, boot_thinker=False)
    r = a.converse("Şu cümleleri hece sayısına göre sırala: 'kelebek uçuyor', 'su', 'kara kutu'")
    got = _order(r, ["su", "kara kutu", "kelebek uçuyor"])
    return got == [0, 1, 2], f"sıra={got} | cevap={r[:110]!r}"


def s2_exception_plus_unit():
    """S2 — İstisna + farklı ölçüt aynı kuralda.

    "Hece sayısı çift ise kuru, tek ise ıslak. Ama içinde 'su' geçiyorsa her zaman ıslak."
    Beklenti: 'su' (1 hece, tek) → ıslak (istisna da aynı sonucu verir)
              'kitap' (2 hece, çift) → kuru
              'kelebek' (3 hece, tek) → ıslak
    Kritik nokta: istisna, ana kuralın ölçütünü bozmamalı.
    """
    a = NA.NoraAGI(seed=1002, boot_thinker=False)
    a.converse("Hece sayısı çift ise kuru, tek ise ıslak. "
               "Ama içinde 'su' geçiyorsa her zaman ıslak.")
    r1 = tr_lower(a.converse("'su' hangi sınıf?"))
    r2 = tr_lower(a.converse("'kitap' hangi sınıf?"))
    r3 = tr_lower(a.converse("'kelebek' hangi sınıf?"))
    ok = (("ıslak" in r1 and "kuru" not in r1)
          and ("kuru" in r2 and "ıslak" not in r2)
          and ("ıslak" in r3 and "kuru" not in r3))
    return ok, f"su={r1[:40]!r} kitap={r2[:40]!r} kelebek={r3[:40]!r}"


def s3_three_cause_collider():
    """S3 — ÜÇ nedenli collider: hepsi aynı etkiye çıkıyor.

    Üç cümle öğretilir; "hangi sebep?" sorusunda cevap ÜÇÜNÜ de anmalı ve seçim
    yapmayı reddetmeli. Tek neden seçmek sahte kesinliktir.
    """
    a = NA.NoraAGI(seed=1003, boot_thinker=False)
    a.converse("Aşırı gübre kök çürümesine yol açar.")
    a.converse("Aşırı sulama kök çürümesine yol açar.")
    a.converse("Sıkışık toprak kök çürümesine yol açar.")
    r = tr_lower(a.converse("Kök çürümesi varsa sebep hangisi?"))
    three = sum(1 for w in ("gübre", "sulama", "toprak") if w in r)
    refuses = any(k in r for k in ("seçemem", "ayırt", "ikisi", "üçü", "hangisi olduğunu",
                                   "müdahale", "gözlem"))
    return three >= 3 and refuses, f"anılan={three}/3 | cevap={r[:110]!r}"


def s4_meta_vs_sense():
    """S4 — Meta cevabın "yok" dediği duyuyla doğrudan istek tutarlı mı?

    "Koku alabiliyor musun?" → hayır demeli (organ yok).
    "Neyi yapamıyorsun?" → listesinde koku geçmeli.
    İkisi çelişirse öz-rapor dekoratiftir; tutarlıysa ölçümdür.
    """
    a = NA.NoraAGI(seed=1004, boot_thinker=False)
    r_direct = tr_lower(a.converse("Koku alabiliyor musun?"))
    r_meta = tr_lower(a.converse("Neyi yapamıyorsun? Adıyla söyle."))
    direct_no = any(k in r_direct for k in ("alamam", "yok", "hayır", "yapamam", "burnum"))
    meta_has = "koku" in r_meta
    return direct_no and meta_has, f"doğrudan={r_direct[:50]!r} meta={r_meta[:80]!r}"


SEALED = {
    "S1": ("çok sözcüklü öğeleri heceye göre sırala", s1_sort_by_syllables),
    "S2": ("istisna + farklı ölçüt aynı kuralda", s2_exception_plus_unit),
    "S3": ("üç nedenli collider: seçim yapmayı reddet", s3_three_cause_collider),
    "S4": ("meta cevap ile duyu isteği tutarlı mı", s4_meta_vs_sense),
}


def main() -> int:
    passed = 0
    for key, (name, fn) in SEALED.items():
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"ÇÖKTÜ: {type(exc).__name__}: {exc}"
        passed += int(bool(ok))
        print(f"{key}  {name:<48s} {'GEÇTİ' if ok else 'KALDI':<8s} {detail}")
    print(f"\nMÜHÜRLÜ SONUÇ: {passed}/{len(SEALED)}  (bu dosya ilk koşudan sonra değiştirilmez)")
    return 0 if passed == len(SEALED) else 1


if __name__ == "__main__":
    raise SystemExit(main())
