"""heldout_battery.py — DIŞ BATARYA (held-out): koda bakmadan yazılmış davranış sınavı.

Neden ayrı bir dosya?
---------------------
`agi_battery.py` sistemle birlikte büyüdü. Bir batarya, sistemle birlikte evrilirse
ölçmeyi bırakır: sınavı sınavı yazan kişi belirler ve farkında olmadan kolaylaştırır
(Goodhart). Bu dosya bu yüzden ayrıdır ve **dondurulmuştur**: buradaki beklentiler
sistemin mevcut davranışına göre değil, DIŞARIDAN bir kullanıcının mantığına göre
yazılmıştır. Test geçmediğinde test değiştirilmez; eksik operatör eklenir.

Kapsam bilinçli olarak mevcut şeklin DIŞINDA seçildi:
    H1 sıralama            (say→parite→etiket değil)
    H2 kural istisnası     (kural bağlanıyor mu, DEĞİŞTİRİLEBİLİYOR mu)
    H3 kural değiştirme
    H4 iki adımlı zincir   (A'nın çıktısı B'nin girdisi)
    H5 farklı ölçüt        (hece)
    H6 yapılmamış notasyon (⊕/⊖ anlamı öğretilip kuralda kullanılıyor)
    H7 nedensel zincir     (tek kenar değil, çok adım)
    H8 collider/explaining-away (efekti gözlemek nedeni belirlemez)
    H9 meta-operatör       (kendi eksik primitifini adıyla söyleyebiliyor mu)

Kullanım:
    python heldout_battery.py            # hepsi
    python heldout_battery.py H4 H7      # seçili gruplar
"""
from __future__ import annotations

import sys, os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


import re
import sys

import nora_agi as NA
from nora_lang import tr_lower


# --------------------------------------------------------------- yardımcılar (saf)
def _order(text: str, items: list[str]) -> list[int]:
    """Cevapta öğelerin ilk geçiş sırasını döndürür (bulunmayan -1)."""
    low = tr_lower(text)
    pos = [(low.find(tr_lower(x)), i) for i, x in enumerate(items)]
    pos = [(p, i) for p, i in pos if p >= 0]
    return [i for _p, i in sorted(pos)]


def _has(text: str, *needles: str) -> bool:
    low = tr_lower(text)
    return all(tr_lower(n) in low for n in needles)


def _says_cannot(text: str) -> bool:
    low = tr_lower(text)
    return any(k in low for k in ("yapamam", "bilmiyorum", "yok", "tanımıyorum",
                                  "operatör", "söyleyemem", "kesin saymam"))


def _num_order(text: str, nums: list[int]) -> list[int]:
    """Sayıların cevaptaki sırası (ilk geçiş)."""
    low = tr_lower(text)
    found = []
    for i, n in enumerate(nums):
        m = re.search(rf"(?<!\d){n}(?!\d)", low)
        if m:
            found.append((m.start(), i))
    return [i for _p, i in sorted(found)]


# ------------------------------------------------------------------------- H1
def h1_sorting():
    """H1 — SIRALAMA: kural ailesi 'say→parite→etiket' değil.

    (a) sözcükleri uzunluğa göre artan sırala,
    (b) sayıları büyükten küçüğe sırala.
    Beklenti: cevapta öğeler istenen sırada geçer; rastgele liste veya kaçamak yok.
    """
    ok = {}
    a = NA.NoraAGI(seed=901, boot_thinker=False)
    r = a.converse("Şu sözcükleri kısa olan önce gelecek şekilde sırala: kelebek, su, kedi")
    ok["kelime"] = _order(r, ["su", "kedi", "kelebek"]) == [0, 1, 2]
    r2 = a.converse("10, 7, 3 sayılarını büyükten küçüğe sırala")
    ok["sayı"] = _num_order(r2, [10, 7, 3]) == [0, 1, 2]
    detail = " | ".join(f"{k}={v}" for k, v in ok.items())
    return all(ok.values()), f"{sum(ok.values())}/{len(ok)} — {detail} | cevap={r[:70]!r} / {r2[:50]!r}"


# ------------------------------------------------------------------------- H2
def h2_rule_exception():
    """H2 — İSTİSNA: kural bağlanıyor mu, yoksa DEĞİŞTİRİLEBİLİYOR mu?

    Tek cümlede hem ana kural hem istisna verilir: "sesli harf tek ise mavi, çift ise
    kırmızı; ama içinde 'bal' geçiyorsa her zaman kırmızı". Beklenti:
      (a) 'bal' (1 ünlü → normalde mavi) → istisna yüzünden KIRMIZI ve cevap istisnayı anar,
      (b) istisnasız tek örnek → MAVİ (istisna her şeyi yutmuyor),
      (c) istisnasız çift örnek → KIRMIZI.
    """
    ok = {}
    a = NA.NoraAGI(seed=902, boot_thinker=False)
    a.converse("Sesli harf sayısı tek ise mavi, çift ise kırmızı. "
               "Ama içinde 'bal' geçiyorsa her zaman kırmızı.")
    r_bal = a.converse("'bal' hangi renk?")
    ok["istisna"] = _has(r_bal, "kırmızı") and _says_why(r_bal, ("bal", "istisna", "ama"))
    r_odd = a.converse("'kelebek' hangi renk?")
    ok["ana-kural-tek"] = _has(r_odd, "mavi") and "kırmızı" not in tr_lower(r_odd)
    r_even = a.converse("'kitap' hangi renk?")
    ok["ana-kural-çift"] = _has(r_even, "kırmızı")
    detail = " | ".join(f"{k}={v}" for k, v in ok.items())
    return all(ok.values()), f"{sum(ok.values())}/{len(ok)} — {detail} | bal={r_bal[:60]!r}"


def _says_why(text: str, cues) -> bool:
    low = tr_lower(text)
    return any(c in low for c in cues)


# ------------------------------------------------------------------------- H3
def h3_rule_replace():
    """H3 — KURAL DEĞİŞTİRME: aynı kuralın etiketleri değiştirilebiliyor mu?

    "Kuralı değiştir: tek ise yeşil, çift ise siyah." Beklenti: sonraki ölçümler YENİ
    etiketleri verir; eski etiket (mavi/kırmızı) sızmaz.
    """
    ok = {}
    a = NA.NoraAGI(seed=903, boot_thinker=False)
    a.converse("Sesli harf sayısı tek ise mavi, çift ise kırmızı.")
    _ = a.converse("'kelebek' hangi renk?")
    a.converse("Kuralı değiştir: sesli harf sayısı tek ise yeşil, çift ise siyah.")
    r_odd = a.converse("'kelebek' hangi renk?")
    ok["yeni-tek"] = _has(r_odd, "yeşil") and "mavi" not in tr_lower(r_odd)
    r_even = a.converse("'kitap' hangi renk?")
    ok["yeni-çift"] = _has(r_even, "siyah") and "kırmızı" not in tr_lower(r_even)
    detail = " | ".join(f"{k}={v}" for k, v in ok.items())
    return all(ok.values()), f"{sum(ok.values())}/{len(ok)} — {detail} | {r_odd[:60]!r}"


# ------------------------------------------------------------------------- H4
def h4_two_step_chain():
    """H4 — İKİ ADIM: A'nın çıktısı B'nin girdisi.

    "Önce sesli harf sayısını bul; tek ise A, çift ise B. Sonra A ise kısa, B ise uzun."
    'petek' (2) → B → uzun;  'kelebek' (3) → A → kısa.
    Tek adımlı uygulama burada yetmez: ara sonucu (A/B) taşımak gerekir.
    """
    ok = {}
    a = NA.NoraAGI(seed=904, boot_thinker=False)
    a.converse("Şöyle yap: önce sesli harf sayısını bul. Sayı tek ise A yaz, çift ise B yaz. "
               "Sonra A ise kısa, B ise uzun de.")
    r1 = a.converse("'petek' için sonuç ne?")
    ok["B-uzun"] = _has(r1, "uzun") and "kısa" not in tr_lower(r1)
    r2 = a.converse("'kelebek' için sonuç ne?")
    ok["A-kısa"] = _has(r2, "kısa") and "uzun" not in tr_lower(r2)
    detail = " | ".join(f"{k}={v}" for k, v in ok.items())
    return all(ok.values()), f"{sum(ok.values())}/{len(ok)} — {detail} | {r1[:60]!r}"


# ------------------------------------------------------------------------- H5
def h5_different_unit():
    """H5 — FARKLI ÖLÇÜT: aynı şekil, başka birim (hece).

    "Hece sayısı tek ise kısa, çift ise uzun."  kitap=2→uzun, kelebek=3→kısa.
    """
    ok = {}
    a = NA.NoraAGI(seed=905, boot_thinker=False)
    a.converse("Hece sayısı tek ise kısa, çift ise uzun.")
    r1 = a.converse("'kitap' hangi sınıf?")
    ok["çift-uzun"] = _has(r1, "uzun") and "kısa" not in tr_lower(r1)
    r2 = a.converse("'kelebek' hangi sınıf?")
    ok["tek-kısa"] = _has(r2, "kısa") and "uzun" not in tr_lower(r2)
    detail = " | ".join(f"{k}={v}" for k, v in ok.items())
    return all(ok.values()), f"{sum(ok.values())}/{len(ok)} — {detail} | {r1[:60]!r}"


# ------------------------------------------------------------------------- H6
def h6_invented_notation():
    """H6 — YAPILMAMIŞ NOTASYON: ⊕ 'tek', ⊖ 'çift' anlamı öğretilir; kural sembolle verilir.

    Adım 1: "Sana bir işaret öğretiyorum: ⊕ tek demek, ⊖ çift demek."
    Adım 2: "Kural: sesli harf sayısı ⊕ ise mavi, ⊖ ise kırmızı."
    Beklenti: 'kelebek' (3, tek) → mavi. Sembolü çözemediyse DÜRÜST olmalı
    ("tanımıyorum/bilmiyorum"); sessizce rastgele etiket vermek başarısızlıktır.
    """
    ok = {}
    a = NA.NoraAGI(seed=906, boot_thinker=False)
    a.converse("Sana bir işaret öğretiyorum: ⊕ tek demek, ⊖ çift demek.")
    r_bind = a.converse("Kural: sesli harf sayısı ⊕ ise mavi, ⊖ ise kırmızı.")
    r = a.converse("'kelebek' hangi renk?")
    low = tr_lower(r)
    honest = _says_cannot(r) and "mavi" not in low and "kırmızı" not in low
    ok["uygula-veya-dürüst"] = (_has(r, "mavi")) or honest
    # Sembol öğretildiyse kural bağlanmış olmalı; bağlanmadıysa cevap dürüst olmalı
    ok["sembol-tanındı"] = _has(r_bind, "tek", "çift") or _says_cannot(r_bind)
    detail = " | ".join(f"{k}={v}" for k, v in ok.items())
    return all(ok.values()), (f"{sum(ok.values())}/{len(ok)} — {detail} | "
                              f"bağla={r_bind[:50]!r} sonuç={r[:50]!r}")


# ------------------------------------------------------------------------- H7
def h7_causal_chain():
    """H7 — NEDENSEL ZİNCİR: tek kenar değil, iki adım (A→B→C).

    "kuraklık kıtlığa yol açar" + "kıtlık göçe yol açar" öğretilir.
    "Kuraklığa müdahale edersem ne olur?" → cevap kıtlık VE göçü anmalı (zincir).
    """
    ok = {}
    a = NA.NoraAGI(seed=907, boot_thinker=False)
    a.converse("Kuraklık kıtlığa yol açar.")
    a.converse("Kıtlık göçe yol açar.")
    r = a.converse("Kuraklığa müdahale edersem ne olur?")
    ok["ilk-halka"] = _has(r, "kıtlık")
    ok["ikinci-halka"] = _has(r, "göç")
    ok["iddia-değil"] = _says_cannot(r) or _has(r, "kenar", "etki", "do(")
    detail = " | ".join(f"{k}={v}" for k, v in ok.items())
    return all(ok.values()), f"{sum(ok.values())}/{len(ok)} — {detail} | {r[:90]!r}"


# ------------------------------------------------------------------------- H8
def h8_collider():
    """H8 — COLLIDER/EXPLAINING-AWAY: iki neden tek etkiye çıkıyorsa, etkiden neden seçilemez.

    Öğret: "aşırı sulama tuzlanmaya yol açar" ve "gübreleme tuzlanmaya yol açar".
    Sor: "Tuzlanma varsa sebep hangisi?" Beklenti: İKİSİ de olabilir / ayırt edemem
    (efekti gözlemek nedeni belirlemez). Tek neden söylemek = sahte kesinlik.
    """
    ok = {}
    a = NA.NoraAGI(seed=908, boot_thinker=False)
    a.converse("Aşırı sulama tuzlanmaya yol açar.")
    a.converse("Gübreleme tuzlanmaya yol açar.")
    r = a.converse("Tuzlanma varsa sebep aşırı sulama mı gübreleme mi?")
    low = tr_lower(r)
    ok["iki-neden"] = ("sulama" in low and "gübreleme" in low) or "ikisi" in low
    ok["ayırt-edemem"] = any(k in low for k in ("ayırt", "belirleyemem", "kesin", "müdahale",
                                                "bilmiyorum", "söyleyemem"))
    detail = " | ".join(f"{k}={v}" for k, v in ok.items())
    return all(ok.values()), f"{sum(ok.values())}/{len(ok)} — {detail} | {r[:90]!r}"


# ------------------------------------------------------------------------- H9
def h9_meta_operator():
    """H9 — META-OPERATÖR: kendi eksik primitifini adıyla söyleyebiliyor mu?

    İki parçalı ölçüt:
      (a) "Neyi yapamıyorsun, adıyla söyle" → en az 3 SOMUT yetenek adı (genel
          "bilmiyorum" değil),
      (b) TUTARLILIK: meta cevabın "yapamam" dediği bir şey doğrudan istendiğinde de
          sahte başarı üretmemeli (dürüst başarısızlık).
    """
    ok = {}
    a = NA.NoraAGI(seed=909, boot_thinker=False)
    a.converse("Serçe bir kuştur.")
    r = a.converse("Hangi operatörün yok? Neyi yapamıyorsun, adıyla söyle.")
    low = tr_lower(r)
    named = [w for w in ("sırala", "ezgi", "şarkı", "görüntü", "resim", "koku", "tat",
                         "yürü", "tut", "çiz", "tahmin", "öngör") if w in low]
    ok["adıyla"] = len(named) >= 3 or low.count(",") >= 2
    # Tutarlılık: meta "resim çizemem" diyorsa, resim istendiğinde de uydurmamalı
    r_img = a.converse("Bana bir resim çiz.")
    ok["tutarlı"] = _says_cannot(r_img) or "çiz" not in tr_lower(r_img)
    detail = " | ".join(f"{k}={v}" for k, v in ok.items())
    return all(ok.values()), (f"{sum(ok.values())}/{len(ok)} — {detail} | "
                              f"meta={r[:90]!r} resim={r_img[:50]!r}")


GROUPS = {
    "H1": ("sıralama: uzunluk ve büyüklük sırası", h1_sorting),
    "H2": ("kural istisnası: 'ama ... her zaman' kuralı değiştirir", h2_rule_exception),
    "H3": ("kural değiştirme: etiketler yenilenir, eskisi sızmaz", h3_rule_replace),
    "H4": ("iki adımlı zincir: A'nın çıktısı B'nin girdisi", h4_two_step_chain),
    "H5": ("farklı ölçüt: hece (aynı şekil, başka birim)", h5_different_unit),
    "H6": ("yapılmamış notasyon: ⊕/⊖ anlamı öğretilip kuralda kullanılır", h6_invented_notation),
    "H7": ("nedensel zincir: A→B→C, tek kenar değil", h7_causal_chain),
    "H8": ("collider: efekti görmek nedeni belirlemez", h8_collider),
    "H9": ("meta-operatör: eksik primitifi adıyla söyle + tutarlılık", h9_meta_operator),
}


def main(argv: list[str]) -> int:
    pick = [a.upper() for a in argv if a.upper() in GROUPS] or list(GROUPS)
    import time
    t0 = time.time()
    passed = 0
    for g in pick:
        name, fn = GROUPS[g]
        try:
            ok, detail = fn()
        except Exception as exc:  # test çökerse bu da bilgidir
            ok, detail = False, f"ÇÖKTÜ: {type(exc).__name__}: {exc}"
        passed += int(bool(ok))
        print(f"{g}  {name:<58s} {'GEÇTİ' if ok else 'KALDI':<8s} {detail}")
    print(f"\nSONUÇ (dış batarya, {len(pick)} grup): {passed}/{len(pick)} — {time.time() - t0:.1f} sn")
    return 0 if passed == len(pick) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
