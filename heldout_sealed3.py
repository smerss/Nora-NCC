"""heldout_sealed3.py — ÜÇÜNCÜ MÜHÜRLÜ SET: şekiller arası transfer + yön (ilk koşu).

Kural (önceki mühürlerle aynı): bu testler koddan SONRA değil, koddan BAĞIMSIZ olarak,
kullanıcıya görünen davranış tarif edilerek yazıldı; İLK KOŞUDAN SONRA NE TEST NE KOD
değiştirilir. Sonuç ne olursa olsun bu dosyada yazılı kalır.

Kapsam — bu turda eklenen yeteneklerin DIŞARIDAN sınavı:
    S9  eşik şekli, dille: "3'ten az ise ..." kuralı cümleden bağlanır ve YENİ bir
        cümleye uygulanır; gerekçe parite dili ("tek/çift") kullanmaz.
    S10 sıra şekli, dürüstlük: "en çok olan" kuralı TEK girdiye etiket UYDURMAZ;
        küme ister ve bunu söyler.
    S11 dünya karşıolgusalı, yeni alan: otlatma -> erozyon -> verimsizlik zincirinde
        ilk halka çıkarılınca etkiler kaybolur; cevap "yapısal" olduğunu söyler.
    S12 bileşik ad + yön: "-den olur" yapısında NEDEN kaynak, ETKİ cümle öznesidir;
        ilişki soruyla geri okunabilir (ters yönde söylenmez).
"""
from __future__ import annotations

import nora_agi as NA
from nora_lang import tr_lower
from nora_style import strip_prefixes


def _body(text: str) -> str:
    return tr_lower(strip_prefixes(text))


def s9_threshold_shape_in_language():
    """Eşik kuralı cümleden bağlanır, yeni cümleye uygulanır, gerekçe eşik dilinde."""
    a = NA.NoraAGI(seed=1301, boot_thinker=False)
    a.converse("Sözcük sayısı 2'den çok ise uzun, değilse kısa.")
    r = _body(a.converse("'tuz ve karabiber' hangi etiketi alır?"))
    ok = ("uzun" in r) and ("çift" not in r) and ("tek " not in r) and ("3" in r)
    return ok, f"cevap={r[:140]!r}"


def s10_rank_shape_honesty():
    """Sıra kuralı tek girdiye etiket uydurmaz: küme gerektiğini söyler."""
    a = NA.NoraAGI(seed=1302, boot_thinker=False)
    a.converse("Hece sayısı en çok olan uzun, en az olan kısa.")
    r = _body(a.converse("'kelebek' hangi etiketi alır?"))
    asks_set = any(k in r for k in ("küme", "sıra", "en az iki", "iki öğe"))
    fabricated = ("uzun" in r or "kısa" in r) and not asks_set
    return asks_set and not fabricated, f"cevap={r[:140]!r}"


def s11_world_counterfactual_new_domain():
    """Yeni alanda zincir: ilk halka çıkarılınca etkiler kaybolur, 'yapısal' denir."""
    a = NA.NoraAGI(seed=1303, boot_thinker=False)
    a.converse("Aşırı otlatma erozyona yol açar.")
    a.converse("Erozyon verimsizliğe yol açar.")
    r = _body(a.converse("Aşırı otlatma olmasaydı ne olurdu?"))
    ok = ("erozyon" in r) and ("verimsiz" in r) and ("yapısal" in r or "modelden" in r)
    return ok, f"cevap={r[:150]!r}"


def s12_compound_name_and_direction():
    """'-den olur': kaynak NEDEN, özne ETKİ; ilişki soruyla geri okunur."""
    a = NA.NoraAGI(seed=1304, boot_thinker=False)
    a.converse("Deniz kirliliği plastik atıktan olur.")
    r = _body(a.converse("Deniz kirliliği neyden olur?"))
    names_both = ("plastik atık" in r) and ("deniz kirlili" in r)
    reversed_reading = "deniz kirliliği plastik" in r      # tamlama yanlış tarafa kaçmış
    edges = [(e.subj, e.obj) for e in a.lang.graph._out_edges.get("plastik atık", ())]
    right_way = ("deniz kirlilik",) in [(o,) for _s, o in edges if _s == "plastik atık"]
    return names_both and right_way and not reversed_reading, \
        f"kenar={edges[:2]} cevap={r[:140]!r}"


SEALED3 = {
    "S9": ("eşik şekli dille: bağla, uygula, eşik diliyle açıkla", s9_threshold_shape_in_language),
    "S10": ("sıra şekli: tek girdiye etiket uydurma, küme iste", s10_rank_shape_honesty),
    "S11": ("yeni alanda dünya karşıolgusalı: zincir ablasyonu", s11_world_counterfactual_new_domain),
    "S12": ("bileşik ad + yön: kaynak neden, soruyla geri okuma", s12_compound_name_and_direction),
}


def main() -> int:
    passed = 0
    for key, (name, fn) in SEALED3.items():
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"ÇÖKTÜ: {type(exc).__name__}: {exc}"
        passed += int(bool(ok))
        print(f"{key}  {name:<58s} {'GEÇTİ' if ok else 'KALDI':<8s} {detail}")
    print(f"\nMÜHÜRLÜ SET 3 SONUÇ: {passed}/{len(SEALED3)}  "
          f"(ilk koşu; bu dosya değiştirilmez)")
    return 0 if passed == len(SEALED3) else 1


if __name__ == "__main__":
    raise SystemExit(main())
