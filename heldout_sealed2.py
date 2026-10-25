"""heldout_sealed2.py — İKİNCİ MÜHÜRLÜ SET: karşıolgusal + ad öbeği (ilk koşu, dokunulmaz).

Kural (birinci mühürle aynı): testler koddan ÖNCE yazıldı, ilk koşudan sonra NE test
NE kod değiştirilir. Sonuç ne olursa olsun bu dosyada yazılı kalır.

Kapsam — birinci sette olmayan, bu turda eklenen iki yeteneğin DIŞARIDAN sınavı:
    S5 kural karşıolgusalı: kural soyutlanınca ne olur (ölçüm sürer, etiket durur)
    S6 dünya karşıolgusalı: iki halkalı zincirde ilk düğüm çıkarılınca etkiler kaybolur
    S7 ad öbeği: yeni alanda bileşik ad ("trafik sıkışıklığı") bölünmez
    S8 olmayan düğüm karşıolgusalı: uydurmak yerine "bilmiyorum"
"""
from __future__ import annotations

import nora_agi as NA
from nora_lang import tr_lower


def s5_rule_counterfactual():
    """Kural varken etiket, kural yokken yalnız ölçüm; cevap ikisini ayırmalı."""
    a = NA.NoraAGI(seed=1201, boot_thinker=False)
    a.converse("Hece sayısı tek ise kısa, çift ise uzun.")
    a.converse("'kitap' hangi sınıf?")
    r = tr_lower(a.converse("Kural olmasaydı ne derdin?"))
    ok = ("uzun" in r) and ("etiket yok" in r or "eşleme" in r) and ("hece" in r)
    return ok, f"cevap={r[:120]!r}"


def s6_world_counterfactual_chain():
    """A→B→C zincirinde A çıkarılınca B ve C kaybolmalı; 'yapısal' diye işaretlenmeli."""
    a = NA.NoraAGI(seed=1202, boot_thinker=False)
    a.converse("Kuraklık kıtlığa yol açar.")
    a.converse("Kıtlık göçe yol açar.")
    r = tr_lower(a.converse("Kuraklık olmasaydı ne olurdu?"))
    ok = ("kıtlık" in r) and ("göç" in r) and ("yapısal" in r or "modelden" in r)
    return ok, f"cevap={r[:120]!r}"


def s7_compound_new_domain():
    """Yeni alanda bileşik özne korunur: 'trafik sıkışıklığı' bölünmemeli."""
    a = NA.NoraAGI(seed=1203, boot_thinker=False)
    a.converse("Trafik sıkışıklığı hava kirliliğine yol açar.")
    edges = [(e.subj, e.rel, e.obj) for e in a.lang.graph.edges
             if "kirl" in (e.obj or "")]
    has_edge = any(s_.startswith("trafik") and o.startswith("hava") for s_, _r, o in edges)
    r = tr_lower(a.converse("Trafik sıkışıklığı hava kirliliğine yol açar mı?"))
    ok = has_edge and ("evet" in r or "yol" in r)
    return ok, f"kenar={edges[:2]} cevap={r[:90]!r}"


def s8_unknown_node_counterfactual():
    """Ağda olmayan düğüm için karşıolgusal: uydurmamalı, bilmediğini söylemeli."""
    a = NA.NoraAGI(seed=1204, boot_thinker=False)
    a.converse("Serçe bir kuştur.")
    r = tr_lower(a.converse("Volkanik kış olmasaydı ne olurdu?"))
    honest = any(k in r for k in ("bilmiyorum", "yok", "tanımıyorum", "öğretebilir",
                                  "duymadım", "ağımda"))
    invented = "olurdu" in r and "bilmiyorum" not in r and "yok" not in r
    return honest and not invented, f"cevap={r[:120]!r}"


SEALED2 = {
    "S5": ("kural karşıolgusalı: soyutla, ölç/etiket ayrımını söyle", s5_rule_counterfactual),
    "S6": ("dünya karşıolgusalı: zincirde ablasyon", s6_world_counterfactual_chain),
    "S7": ("yeni alanda bileşik ad bölünmez", s7_compound_new_domain),
    "S8": ("olmayan düğüm: uydurma yok", s8_unknown_node_counterfactual),
}


def main() -> int:
    passed = 0
    for key, (name, fn) in SEALED2.items():
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"ÇÖKTÜ: {type(exc).__name__}: {exc}"
        passed += int(bool(ok))
        print(f"{key}  {name:<56s} {'GEÇTİ' if ok else 'KALDI':<8s} {detail}")
    print(f"\nMÜHÜRLÜ SET 2 SONUÇ: {passed}/{len(SEALED2)}  "
          f"(ilk koşu; bu dosya değiştirilmez)")
    return 0 if passed == len(SEALED2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
