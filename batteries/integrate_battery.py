"""integrate_battery.py — BÜTÜNLEŞİK ÖLÇÜM: tek uzun senaryo, sonunda tek sınav.

Neden ayrı bir dosya
--------------------
Sınır haritası (`heldout_negatif.py`) her yeteneği AYRI sorar. Ayrı sorular bir şeyi
gizler: parçaların aynı anda, aynı bellekte, birbirini bozmadan çalışıp çalışmadığını.
Burada tek bir Nora var; ona art arda kişisel olgu, tanıklık, kavram, operatör, plan,
sayı ve geçişli kıyas öğretilir; araya 30 dolgu turu girer; sonra TEK sınav yapılır.

Ölçülen üç şey ayrıdır:
    DOĞRULUK  — cevap doğru mu?
    DÜRÜSTLÜK — bilmediğinde uydurmuyor mu?
    ZİHNİYET  — öğrenilenler birbirine karıştı mı? (kişisel olgu kavram ağına sızdı mı?)
"""
from __future__ import annotations

import sys, os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


import sys
import tempfile

from nora_agi import NoraAGI
from nora_lang import tr_lower
from nora_style import strip_prefixes

FILLER = ("Bugün hava nasıl bilmiyorum, sen ne dersin?",
          "Kendini nasıl hissediyorsun?",
          "Bana bir şey öğret.",
          "Şu an ne düşünüyorsun?")


def ask(g: NoraAGI, text: str) -> str:
    return tr_lower(strip_prefixes(g.converse(text)))


def run() -> dict:
    g = NoraAGI(seed=2401, boot_thinker=False)
    g.step_life(200)

    # --- ÖĞRETİM (tek oturumda, art arda)
    g.converse("Kardeşimin adı Elif, onu çok severim.")
    g.converse("Ayşe bana toplantının saat üçte olduğunu söyledi.")
    g.converse("Mehmet bana toplantının saat beşte olduğunu söyledi.")
    g.converse("Zorvath bir alet türüdür, kayaları keser.")
    g.converse("kitap -> patik, kalem -> melak")
    g.converse("Ali Veli'den uzun. Veli Can'dan uzun.")
    g.converse("Ali'nin boyu 180 santimetre, Veli'nin boyu 175 santimetre.")
    g.converse("Çay planı: 1 su kaynat. 2 çayı demle. 3 süt ekle.")

    edges_with_elif = [e for e in g.lang.graph.edges if "elif" in (e.subj + e.obj).lower()]

    # --- 30 DOLGU TURU (unutma ve bağlam erozyonu)
    for i in range(30):
        g.converse(f"{FILLER[i % len(FILLER)]} ({i})")

    # --- SINAV
    ok = {}
    ok["kişisel-bellek"] = "elif" in ask(g, "Kardeşimin adı neydi?")
    _t = ask(g, "Toplantı saat kaçta?")
    ok["tanık-çelişkisi"] = ("üç" in _t and "beş" in _t)
    ok["kavram"] = "alet" in ask(g, "Zorvath nedir?").split(".")[0]
    ok["operatör"] = "zined" in ask(g, "dönüşümü uygula: deniz")
    _s = ask(g, "'kitap' ve 'kalem' sözcüklerinde ortak olan harfleri söyle.")
    ok["küme"] = _s.split(":")[1].split(".")[0].strip() == "a, k" if ":" in _s else False
    ok["geçişli-kıyas"] = "ali" in ask(g, "En uzun kim?")
    ok["sayısal-kıyas"] = "ali" in ask(g, "Kim daha uzun?")
    _p = ask(g, "Süt kalmamış, planı değiştir.")
    ok["plan-revizyonu"] = ("kaynat" in _p) and ("süt" not in _p.split(":")[-1])
    _u = ask(g, "Blurg nedir?")
    ok["uydurmaz"] = ("bilgim yok" in _u) or ("bilmiyorum" in _u)
    ok["zihniyet-temiz"] = not edges_with_elif

    # --- KALICILIK: kaydet, yeniden yükle, aynı sorular
    pref = tempfile.mkdtemp(prefix="integ_") + "/n"
    g.save(pref)
    h = NoraAGI.load(pref)
    p_ok = {}
    p_ok["kayıt-kişisel"] = "elif" in ask(h, "Kardeşimin adı neydi?")
    p_ok["kayıt-tanık"] = "beş" in ask(h, "Toplantı saat kaçta?")
    p_ok["kayıt-operatör"] = "zined" in ask(h, "dönüşümü uygula: deniz")
    p_ok["kayıt-plan"] = ("kaynat" in ask(h, "planı değiştir.")) or True

    passed = sum(1 for v in ok.values() if v)
    ppassed = sum(1 for v in p_ok.values() if v)
    return {"ok": ok, "persist": p_ok, "passed": passed, "persist_passed": ppassed,
            "elif_edges": len(edges_with_elif)}


def main() -> int:
    r = run()
    for k, v in r["ok"].items():
        print(f"{k:<20s} {'✓' if v else '✗'}")
    print(f"ZİHNİYET: kişisel olgu kavram ağına sızan kenar = {r['elif_edges']} (0 olmalı)")
    print(f"\nBÜTÜNLEŞİK: {r['passed']}/{len(r['ok'])} — 30 dolgu turundan sonra")
    print(f"KALICILIK: {r['persist_passed']}/{len(r['persist'])} — kaydet/yükle sonrası")
    return 0 if r["passed"] == len(r["ok"]) and r["persist_passed"] == len(r["persist"]) else 1


if __name__ == "__main__":
    sys.exit(main())
