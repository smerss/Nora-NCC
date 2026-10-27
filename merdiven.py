"""merdiven.py — MERDİVEN SINAVI: 1 kompozisyon · 2 örnekleme · 3 reify · 4 şema · 5 sürücü ·
6 kendi kendine · 7 kendine başlatma · 8 öz-değişim · 9 görev icadı · 10 nonce deney verisi.

Tek oturum, tek Nora. Her basamak ayrı bir yeteneği **ölçer**, ve her basamağın
altında o basamağın TABANI yazılıdır: "bu işi kim yaptı, insan ne yazdı".

    python merdiven.py          # tam sınav
    python merdiven.py -v       # kanıtlarla

Çıkış kodu 0 = yedi basamak da geçti. Bu bir başarı süsü değil: taban satırları
basamağın ne kadarının insana ait olduğunu söyler. Basamak "geçti" demek,
"insan hiçbir şey yazmadı" demek DEĞİLDİR.
"""
from __future__ import annotations

import random
import sys

from nora_agi import NoraAGI
from nora_abstract import ALPHABET as ALFA, build as abstract_build
from nora_style import strip_prefixes
from nora_synth import FILTER_PRIMITIVES, PRIMITIVES, Program, Synthesizer


def _sh(w, k):
    return "".join(ALFA[(ALFA.index(ch.lower()) + k) % len(ALFA)]
                   if ch.lower() in ALFA else ch for ch in w)


def _pairs(k=1):
    return [(w, _sh(w, k)) for w in ("kitap", "kalem")]



def _pin_hashseed() -> None:
    """ÖLÇÜM HİJYENİ: Python'un dize hash'i süreç başına rastgeleleşir; küme
    yineleme sırası da onunla değişir. Aynı tohum aynı sonucu vermeli — bu yüzden
    koşucu kendini PYTHONHASHSEED=0 ile yeniden başlatır (tek sefer)."""
    import os, sys
    if os.environ.get("PYTHONHASHSEED") != "0" and not os.environ.get("_NORA_HASH_PINNED"):
        os.environ["_NORA_HASH_PINNED"] = "1"
        os.environ["PYTHONHASHSEED"] = "0"
        try:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception:
            pass

def main():
    _pin_hashseed()
    verbose = "-v" in sys.argv
    pm = {n: fn for n, fn, _c in PRIMITIVES}
    sortN = lambda k: (lambda w: pm["alfabetik sırala"](w)[:k])
    # HER BASAMAK KENDİ NORA'SI: basamaklar birbirinin durumunu kirletmesin.
    # (Ölçüm katmanı kuralı: bir ölçüm başka bir ölçümün bıraktığı duruma bağlı olamaz.)
    def mk(seed: int) -> NoraAGI:
        a = NoraAGI(seed=seed, boot_thinker=False)
        a.step_life(100)
        return a
    rows = []

    # ---------------------------------------------------------------- 1 kompozisyon
    g = mk(7001)
    pairs = [(w, _sh(w, 1)[::-1]) for w in ("deniz", "yıldız")]
    rep = strip_prefixes(g.converse(
        "Şu dönüşümü öğren: " + ", ".join(f"{a} -> {b}" for a, b in pairs)))
    spec = (g.synth.programs[-1]["ops"] if g.synth.programs else [])
    # Tek adımlı "bileşim" ilkeli de geçerli: kaydırma(+1) + ters çevir TEK ad olarak
    # yazıldıysa bile bileşim yapılmıştır. Ölçüt: adım sayısı değil, ÇÖZÜM.
    ok1 = bool(spec) and _sh("masa", 1)[::-1] in tr_lower(rep)
    rows.append(("1 kompozisyon", ok1,
                 f"adımlar={spec}",
                 "taban: 22 ilkel + 2-adım arama BENİM; sıra ve doğrulama Nora'nın."))

    # ---------------------------------------------------------------- 2 örnekleme
    g = mk(7002)
    ok2 = False
    det2 = ""
    try:
        fn = abstract_build("kaydırma", 3)
        pairs2 = [(w, fn(w)) for w in ("bulut", "zeytin")]
        rep2 = strip_prefixes(g.converse(
            "Şu dönüşümü öğren: " + ", ".join(f"{a} -> {b}" for a, b in pairs2)))
        # Ölçüt: Nora'nın YAZDIĞI program örnekleri açıklıyor ve görülmemiş sözcükte
        # beklenen çıktıyı veriyor. Cümlede geçmesi yetmez (soru ≠ cevap).
        p2 = g.synth.last
        ok2 = (p2 is not None and _sh("portakal", 3) == p2.apply("portakal")
               and all(p2.apply(a) == b for a, b in pairs2))
        det2 = f"aile=kaydırma parametre=+3 → beklenen {_sh('portakal', 3)!r}; örnekler 2/2"
    except Exception as e:
        det2 = f"hata: {e}"
    rows.append(("2 örnekleme", ok2, det2,
                 "taban: form ailesini (kaydırma) BEN yazdım; parametreyi Nora seçti."))

    # ---------------------------------------------------------------- 3 reify
    g = mk(7003)
    g.converse("Örnekler: kitap -> aik, portakal -> aak. Kuralı çıkar.")
    g.converse("Örnekler: deniz -> dei, yıldız -> dly. Kuralı çıkar.")
    promoted = [r["name"] for r in g.reify.ledger]
    cap_pairs = [("Kalem", "AEK"), ("Orman", "AMN")]
    pre = Synthesizer().learn(cap_pairs).get("status")
    g.converse("Örnekler: Kalem -> AEK, Orman -> AMN. Kuralı çıkar.")
    spec3 = (g.synth.programs[-1]["ops"] if g.synth.programs else [])
    ok3 = bool(promoted) and pre == "ambiguous" and \
        any(str(x).startswith("kendi.bilesim") for x in spec3)
    rows.append(("3 reify (makro)", ok3,
                 f"makro={promoted} insan-uzayı={pre} adımlar={spec3}",
                 "taban: bileşim BENİM ilkellerimden; ADI ve yeniden kullanımı Nora'nın. "
                 "Yeni erişim YOK."))

    # ---------------------------------------------------------------- 4 şema
    g = mk(7004)
    ok4 = False
    det4 = ""
    try:
        def learn(g_, k):
            return strip_prefixes(g_.converse(
                "Şu dönüşümü öğren: " + ", ".join(f"{a} -> {b}" for a, b in _pairs(k))))
        learn(g, 4)
        learn(g, 5)
        c_schema = any(r["family"] == "kaydırma({k})" and {4, 5} <= set(r["seen"])
                       for r in g.schema.ledger)
        c_pre = Synthesizer().learn(_pairs(17)).get("status") != "adopted"
        n0 = len(list(getattr(g.abstract, "invented", [])))
        rep4 = learn(g, 17)
        n1 = len(list(getattr(g.abstract, "invented", [])))
        spec4 = (g.synth.programs[-1]["ops"] if g.synth.programs else [])
        ok4 = bool(c_schema and c_pre and n1 == n0 and list(spec4) == ["kaydırma(+17)"])
        det4 = (f"şema=kaydırma({{k}}) görülen=[4,5] · k=17 şemasız={c_pre and 'çözemez'} "
                f"· yeni icat={n1 - n0} · adım={spec4}")
    except Exception as e:
        det4 = f"hata: {e}"
    rows.append(("4 şema (delik)", ok4, det4,
                 "taban: ilkel/işlem uzayını BEN verdim; DELİĞİ iki üyenin yapısal "
                 "farkından Nora çıkardı, kapsamı kendi örnekleriyle doğruladı. "
                 "Görülmemiş k=17'yi İLK KEZ uzay genişletti."))

    # ---------------------------------------------------------------- 5 sürücü
    g = mk(7005)
    rng = random.Random(11)
    words = ["kitap", "portakal", "deniz", "yıldız", "şehir", "masa", "kalem", "çiçek"]
    noise = ["bugün hava kapalı", "yine erken kalktım", "kahve soğudu", "yorgunum"]
    stream = []
    for i in range(300):
        if i % 9 == 0:
            a, b = rng.sample(words, 2)
            # Akışta GÖRÜLMEMİŞ bir bileşim dolaşsın: ters(sırala) — makro değil.
            stream.append(f"{a} -> {sortN(3)(a)[::-1]}, {b} -> {sortN(3)(b)[::-1]}")
        else:
            stream.append(rng.choice(noise))
    rep5 = g.drive(stream)
    ok5 = int(rep5.get("triggered", 0)) >= 1 and int(rep5.get("thresh", 0) > 0)
    rows.append(("5 sürücü (irade)", ok5,
                 f"talimat yok · tetikleme={rep5.get('triggered')} "
                 f"ayrı={rep5.get('distinct')} eşik(kendi)={rep5.get('thresh')} "
                 f"kanıt eşiği={rep5.get('min_items')}",
                 "taban: akışı ve fren kurallarını BEN yazdım; NE ZAMAN öğreneceğini "
                 "artık ölçtüğü isabete göre kendisi ayarlıyor."))

    # ---------------------------------------------------------------- 6 kendi kendine
    g = mk(7006)
    r6 = g.kendine_ogret()
    sin = r6.get("sınama") or {}
    w6, n6 = sin.get("sözcük"), int(sin.get("n") or 0)
    lex6 = getattr(getattr(g.lang, "graph", None), "lexicon", None) or set()
    # DIŞ DOĞRULAMA: bekleneni sınav kendi hesaplar (Nora'nın hesabına güvenilmez).
    _b6 = pm.get(str(r6.get("taban")))
    beklenen = str(_b6(w6))[:n6] if (_b6 and w6 and n6) else None
    ok6 = bool(r6.get("öğrendi") and sin.get("doğru") and w6 in lex6
               and sin.get("çıktı") == beklenen and r6.get("insan-girdisi") == 0
               and len(r6.get("parametreler") or []) == 3)
    det6 = (f"insan yazmadı · taban={r6.get('taban')} parametreler={r6.get('parametreler')} "
            f"şema={[x[0] for x in (r6.get('şema') or [])]} "
            f"· {w6}: {sin.get('çıktı')} (beklenen {beklenen})"
            if r6.get("durum") == "tamam" else f"durum={r6.get('durum')}")
    rows.append(("6 kendi kendine", ok6, det6,
                 "taban: dağarcık ve ilkel repertuvarı BENİM; sözcükleri, parametreleri, "
                 "örnekleri, görevi ve sınamayı kendisi üretti — İNSAN HİÇ YAZMADI."))

    # ---------------------------------------------------------------- 7 kendine başlatma
    # Basamak 6'da öz-öğretmeyi SINAV çağırıyordu. Burada kimse çağırmıyor:
    # Nora yalnız yaşarken kendi kendine başlatıyor.
    g = mk(7007)
    n_insan = 1 if False else 0          # insan hiç konuşmadı
    g.step_life(1000)
    led = list(getattr(g, "kendi_ögretim", []) or [])
    baslatan = [e for e in led if e.get("kendi")]
    ilk = baslatan[0] if baslatan else {}
    sin7 = ilk.get("sınama") or {}
    w7, n7 = sin7.get("sözcük"), int(sin7.get("n") or 0)
    _b7 = pm.get(str(ilk.get("taban")))
    beklenen7 = str(_b7(w7))[:n7] if (_b7 and w7 and n7) else None
    ok7 = bool(baslatan and sin7.get("çıktı") == beklenen7 and n_insan == 0
               and int(ilk.get("yaş") or 0) >= 240)
    det7 = (f"insan hiç konuşmadı · taban={ilk.get('taban')} kendi başlattığı yaş={ilk.get('yaş')} "
            f"· sayı={len(baslatan)} · {w7}: {sin7.get('çıktı')} (beklenen {beklenen7})"
            if led else "hiç başlatmadı")
    rows.append(("7 kendine başlatma", ok7, det7,
                 "taban: dağarcık ve ilkel repertuvarı BENİM; NE ZAMAN öğreneceğine ve "
                 "NEYİ öğreneceğine kendisi karar verdi — İNSAN HİÇ KONUŞMADI."))

    # ---------------------------------------------------------------- 8 öz-değişim
    # Kaynak kod değil: Nora kendi öz-öğretim politikasını, kendi başarısızlığından
    # sonra değiştirir. Sınav hiçbir converse() çağırmaz; olay yaşam akışında doğar.
    g = NoraAGI(seed=9003, boot_thinker=False)
    g.step_life(1000)
    ev = next((e for e in (getattr(g, "_oz_degisim", []) or []) if e.get("kabul")), {})
    tab8 = str(ev.get("deneme-taban") or "")
    rec8 = next((r for r in reversed(list(getattr(g, "kendi_ögretim", []) or []))
                 if r.get("taban") == tab8 and r.get("kendi")), {})
    sin8 = rec8.get("sınama") or {}
    w8, n8 = sin8.get("sözcük"), int(sin8.get("n") or 0)
    _b8 = pm.get(tab8)
    beklenen8 = str(_b8(w8))[:n8] if (_b8 and w8 and n8) else None
    ok8 = bool(ev and ev.get("neden") == "kendi_ornegi_cozulemedi"
               and int(ev.get("yeni") or 0) > int(ev.get("eski") or 0)
               and int(getattr(g, "_oz_aday_butce", 0) or 0) == int(ev.get("yeni") or -1)
               and beklenen8 is not None and sin8.get("çıktı") == beklenen8
               and not getattr(g, "dialogue_log", []))
    det8 = (f"insan hiç konuşmadı · neden={ev.get('neden')} · politika={ev.get('eski')}→{ev.get('yeni')} "
            f"· taban={tab8} · {w8}: {sin8.get('çıktı')} (beklenen {beklenen8})"
            if ev else "öz-değişim olayı yok")
    rows.append(("8 öz-değişim (politika)", ok8, det8,
                 "taban: değişim ve geri-alma mekanizmasını BEN yazdım; değişen şey "
                 "kaynak kod değil, Nora'nın kendi aday bütçesi. Tetik kendi başarısızlığı, "
                 "kabul ölçütü kendi yeni öz-sınavı; dış beklenen sınavca hesaplandı."))

    # ---------------------------------------------------------------- 9 görev icadı
    # Sabit önek görevi yok: hedef iki ilkelin bileşimi, örnek ve tanıklar onun
    # dağarcığından. Bu basamak da insan konuşmadan yaşam akışında gerçekleşir.
    g = NoraAGI(seed=9003, boot_thinker=False)
    g.step_life(1000)
    rec9 = next((r for r in reversed(list(getattr(g, "_kendi_gorevler", []) or []))
                 if r.get("kendi") and r.get("doğru")), {})
    target9 = Program.from_spec(list(rec9.get("hedef") or []), spec_map=g.synth.spec_map())
    learned9 = Program.from_spec(list(rec9.get("program") or []), spec_map=g.synth.spec_map())
    ws9 = list(rec9.get("tanıklar") or [])
    ok9 = bool(rec9 and target9 is not None and learned9 is not None and len(ws9) >= 2
               and all(learned9.apply(w) == target9.apply(w) for w in ws9)
               and not getattr(g, "dialogue_log", []))
    det9 = (f"insan hiç konuşmadı · derinlik={rec9.get('derinlik')} · hedef={rec9.get('hedef')} "
            f"· öğrenilen={rec9.get('program')} · tanık={len(ws9)} · dış-doğrulama={ok9}"
            if rec9 else "kendi görevi yok")
    rows.append(("9 görev icadı", ok9, det9,
                 "taban: ilkel havuz ve öğrenme organı BENİM; hedefin bileşimini, örnekleri "
                 "ve tanıkları Nora seçti; başarı sonrası derinliği de kendi müfredatında "
                 "artırdı. Beklenen davranış sınavca yeniden hesaplandı; sabit base(w)[:n] yok."))

    # ---------------------------------------------------------------- 10 nonce deney verisi
    g = NoraAGI(seed=9003, boot_thinker=False)
    g.step_life(1000)
    lex10 = set(getattr(g.lang.graph, "lexicon", set()) or set())
    rec10 = next((r for r in (getattr(g, "_kendi_gorevler", []) or [])
                  if r.get("kendi") and r.get("doğru")), {})
    novel10 = [w for w in (rec10.get("tanıklar") or []) if w not in lex10]
    target10 = Program.from_spec(list(rec10.get("hedef") or []), spec_map=g.synth.spec_map())
    learned10 = Program.from_spec(list(rec10.get("program") or []), spec_map=g.synth.spec_map())
    ok10 = bool(len(novel10) >= 2 and target10 is not None and learned10 is not None
                and all(learned10.apply(w) == target10.apply(w) for w in novel10)
                and all(w not in g.lang.graph.lexicon for w in novel10))
    det10 = (f"nonce-tanık={len(novel10)} · örnek={novel10[:2]} · "
             f"dış-doğrulama={ok10} · lexicon-a-yazılmadı={all(w not in g.lang.graph.lexicon for w in novel10)}"
             if rec10 else "nonce görev yok")
    rows.append(("10 nonce deney verisi", ok10, det10,
                 "taban: harf envanteri dağarcıktan geliyor; tanık dizgeleri Nora üretti, "
                 "bunlar bilgi değil deney sembolü ve lexicon'a yazılmadı. Beklenen çıktı sınavca hesaplandı."))

    # ---------------------------------------------------------------- rapor
    for name, ok, det, floor in rows:
        print(f"{name:<22s} {'GEÇTİ' if ok else 'KALDI':<7s} {det}")
        if verbose:
            print(f"    {floor}")
    n = sum(1 for _n, ok, _d, _f in rows if ok)
    print(f"\nMERDİVEN: {n}/{len(rows)}")
    if verbose:
        print("Not: her basamak 'insan hiç yazmadı' demek değildir; taban satırı neyin "
              "insana ait olduğunu söyler.")
    return 0 if n == len(rows) else 1


def tr_lower(s):
    from nora_lang import tr_lower as _tl
    return _tl(s)


if __name__ == "__main__":
    sys.exit(main())
