"""nora_forge.py — OPERATÖR OCAĞI: kuralı insan yazmaz, Nora ÖRNEKLERDEN kendi yazar.

Neden bu organ var
------------------
Bugüne kadar operatörleri biz yazdık (`nora_apply`: bind, apply, count, sort, chain,
plan, revise, transfer, counterfactual). AGI'ye yaklaşmanın ölçülebilir tanımı şudur:

    insanın yazdığı operatör sayısı SABİT kalırken
    Nora'nın kendi yazdığı operatör sayısı ARTAR.

Bu organ o sayacın ikinci yarısını üretir. Girdi: etiketlenmiş örnekler. Çıktı: bir
`Rule` nesnesi — yani uygulanabilir bir operatör — ve kimin yazdığının kaydı.

Neyi YAPMAZ (dürüstlük sınırları)
---------------------------------
1. **Ezber yok.** Aday uzayında "her girdiyi ayrı eşle" hipotezi yoktur. Kural,
   ölçülebilir bir özellik + bir eşleme ŞEKLİ olmak zorundadır.
2. **Birden çok kural aynı örnekleri açıklıyorsa seçmez.** Kaç aday kaldığını söyler
   ve AYIRT EDİCİ bir örnek ister (`probe`). Uydurma yerine veri talebi.
3. **Tek sınıflı veriden kural çıkmaz.** Bütün etiketler aynıysa hiçbir şekil
   ayırt edilemez; "sabit etiket" bir kural değildir, susmaktır.
4. **Aynı davranışı veren iki yazım tek aday sayılır.** `syllable_count` ile
   `vowel_count` bu depoda aynı ölçümü verir; bunları iki ayrı aday gibi göstermek
   sahte bir belirsizlik üretirdi. Adaylar DAVRANIŞ imzasıyla tekilleştirilir ve
   en yalın yazım (parite < eşik) tutulur.

Sözlüğün tamamı burada değil: cümleler `nora_priors.json` içinde (`meta.forge`).
Bu dosya yalnız ÖLÇER, TÜRETİR ve KAYDEDER.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

from nora_apply import (FEATURES, OPERATORS, SHORT_NAMES, Interpreter, Rule,
                        num_abl, register_feature)

#: İNSANIN verdiği ölçümler. Tümevarım bunlarla başlar; ama tavan değildir: Nora'nın
#: kendi icat ettiği ölçümler de (envanter) arama uzayına girer — bkz. `search_features`.
BASE_FEATURES = ("vowel_count", "consonant_count", "letter_count", "word_count",
                 "syllable_count")
#: Geriye dönük ad (eski çağıranlar bu adı kullanıyor).
SEARCH_FEATURES = BASE_FEATURES

#: Eşik taramasının üst sınırı (ölçüm birimi: harf/sözcük). Verinin üstüne kadar +
#: 2 pay taranır; daha büyük eşikler aynı ayrımı verir, aday şişirir.
MAX_T = 14

#: Şekil sırası: aynı davranışı veren yazımlarda EN YALIN olan tutulur.
_SHAPE_ORDER = {"parity": 0, "threshold": 1, "presence": 2}


@dataclass(frozen=True)
class Hypothesis:
    """Tek bir kural ADAYI: ölçüm + eşleme şekli + eşleme."""

    feature: str
    shape: str
    mapping: tuple          # (("odd", "mavi"), ("even", "kırmızı")) — sıralı, karma değil
    threshold: int = 0
    op: str = "lt"
    #: İCAT EDİLMİŞ ölçümün tarifi ({"kind": "suffix", "arg": "lık"}); boşsa yerleşik ölçüm.
    spec: dict | None = None

    # ------------------------------------------------------------------ ölç/uygula
    def measure(self, text: str) -> int:
        fn = FEATURES.get(self.feature, FEATURES["letter_count"])[1]
        return int(fn(text or ""))

    def bucket(self, n: int) -> str:
        if self.shape == "presence":
            return "var" if int(n) else "yok"
        if self.shape == "threshold":
            t, o = int(self.threshold), str(self.op)
            cond = {"lt": n < t, "le": n <= t, "gt": n > t, "ge": n >= t}.get(o, n < t)
            return "cond" if cond else "else"
        return "odd" if n % 2 else "even"

    def predict(self, text: str):
        return dict(self.mapping).get(self.bucket(self.measure(text)))

    def signature(self, examples) -> tuple:
        """Örnekler üzerindeki DAVRANIŞ imzası: iki aday aynı davranıyorsa aynıdır."""
        return tuple(self.predict(t) for t, _ in examples)

    def desc(self) -> str:
        m = dict(self.mapping)
        name = FEATURES.get(self.feature, ("ölçüm",))[0]
        if self.shape == "presence":
            return f"{name}: var -> {m.get('var')}, yok -> {m.get('yok')}"
        if self.shape == "threshold":
            _a = num_abl(self.threshold)
            cond = {"lt": f"{self.threshold}'{_a} az", "le": f"en çok {self.threshold}",
                    "gt": f"{self.threshold}'{_a} çok", "ge": f"en az {self.threshold}"}
            return f"{name}: {cond.get(self.op, self.threshold)} -> {m.get('cond')}, değilse {m.get('else')}"
        return f"{name}: tek -> {m.get('odd')}, çift -> {m.get('even')}"

    def to_rule(self, rid: str, source: str = "", domain: str = "kendi") -> Rule:
        return Rule(rid=rid, feature=self.feature, mapping=dict(self.mapping),
                    domain=domain, source_text=source, shape=self.shape,
                    threshold=int(self.threshold), op=str(self.op), origin="kendi",
                    feature_spec=dict(self.spec or {}))


class OperatorForge:
    """Örneklerden operatör türeten ocak + kimin yazdığının defteri."""

    MAX_PROBE_SCAN = 4000

    def __init__(self, interpreter: Interpreter | None = None,
                 ledger_path: str | pathlib.Path = "runs/operators_self.json",
                 step: int = 0):
        self.eng = interpreter
        self.ledger_path = pathlib.Path(ledger_path)
        self.step = int(step)
        self.examples: list[tuple[str, str]] = []
        self.cands: list[Hypothesis] = []
        self.adopted: Rule | None = None
        self.why: str = ""
        self.probe_info: dict | None = None
        self.authored: list[dict] = self._load_ledger()
        self.refuted: list[dict] = self._load_refuted()
        self.measurements: list[dict] = self._load_measurements()
        self._reload_measurements()

    # ================================================================= defter
    def _load_ledger(self) -> list[dict]:
        try:
            blob = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            recs = blob.get("self_authored") or []
            return [r for r in recs if isinstance(r, dict)]
        except Exception:
            return []

    def _load_measurements(self) -> list[dict]:
        try:
            blob = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            return [r for r in (blob.get("self_measurements") or []) if isinstance(r, dict)]
        except Exception:
            return []

    def _reload_measurements(self) -> None:
        """Önceki oturumlarda icat edilen ölçümler yeniden kurulur: kural ölçümsüz kalmasın."""
        for rec in self.measurements:
            try:
                key = str(rec.get("key") or "")
                spec = rec.get("spec") or {}
                if key and key not in FEATURES:
                    register_feature(key, str(rec.get("name") or key), spec)
            except Exception:
                pass

    def _load_refuted(self) -> list[dict]:
        try:
            blob = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            return [r for r in (blob.get("refuted") or []) if isinstance(r, dict)]
        except Exception:
            return []

    def mark_refuted(self, rule: Rule, item: str, actual: str, claimed) -> None:
        """Kendi operatörüm ÇÜRÜTÜLDÜ: sessizce silinmez, deftere gerekçesiyle yazılır."""
        if rule is None:
            return
        # Kural bir ÖLÇÜM üzerinde duruyorsa ve kural yanlışsa, o ölçüm de yanlış
        # olabilir: envanterde işaretlenir (silinmez), bir daha önerilmez.
        feat = str(getattr(rule, "feature", "") or "")
        if feat:
            for rec in self.measurements:
                if str(rec.get("key") or "") == feat:
                    rec["misses"] = int(rec.get("misses") or 0) + 1
                    rec.setdefault("refuted", {"item": str(item), "actual": str(actual),
                                               "claimed": claimed, "step": int(self.step)})
        self.refuted.append({"id": getattr(rule, "rid", ""),
                             "desc": getattr(rule, "source_text", ""),
                             "refuted_by": {"item": str(item), "claimed": claimed,
                                            "actual": str(actual)},
                             "step": int(self.step)})
        self._save_ledger()

    def _save_ledger(self) -> None:
        try:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            blob = {"authorship": {"human": self.human_count(), "self": len(self.authored)},
                    "_doc": ("insan: nora_operators.json'daki dondurulmuş rejistri; "
                             "self: Nora'nın örneklerden kendi türettiği operatörler."),
                    "self_authored": self.authored[-200:],
                    "refuted": self.refuted[-200:],
                    "self_measurements": self.measurements[-100:]}
            self.ledger_path.write_text(json.dumps(blob, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
        except Exception:
            pass

    # ================================================================= envanter
    def search_features(self) -> tuple:
        """Taranacak ölçümler: insanın verdikleri + KENDİ icat ettiklerim.

        A87/A88'de icat edilen ölçüm yalnız kendi kuralının içinde yaşıyordu; yani her
        yeni problemde ölçümü sıfırdan icat etmek gerekiyordu. Envanter bunu bitirir:
        bir kez yazdığım ölçüm sonraki işlerde HAZIR olur. Böylece insanın verdiği
        ölçüm listesi yeteneğin tavanı olmaktan çıkar — Nora kendi aletleriyle çalışır.

        Çürütülmüş ölçüm envanterde KALIR (gizlenmez) ama ARANMAZ: yanlış çıkmış bir
        aleti tekrar önermek, çürütmeyi duymamış olmak demektir.
        """
        keys = list(BASE_FEATURES)
        mine = [str(r.get("key") or "") for r in self.measurements
                if str(r.get("key") or "") in FEATURES and str(r.get("key")) not in keys]
        # GÜVEN SIRASI: çok tutan ölçüm önce denenir. Güven EŞİĞİN ALTINDAYSa hiç
        # önerilmez — yanlış çıkmış bir aleti tekrar önermek, çürütmeyi duymamaktır.
        mine.sort(key=lambda k: -self.trust(k))
        keys.extend(k for k in mine if self.trust(k) >= self.TRUST_FLOOR)
        return tuple(keys)

    #: Güven, bildirilen bir inanç değil KAZANILMIŞ bir orandır: kaç kez tuttu / kaç kez
    #: yanıldı. Laplace düzeltmesi (artı bir) kanıtsız ölçümü 0.5'te başlatır: ne iyi ne
    #: kötü. Bu eşiğin altındaki ölçüm envanterde durur ama ARANMAZ.
    TRUST_FLOOR = 0.4

    def trust(self, key: str) -> float:
        rec = next((r for r in self.measurements if str(r.get("key")) == str(key)), None)
        if rec is None:
            return 0.0
        h = int(rec.get("hits") or 0)
        m = int(rec.get("misses") or 0)
        return round((h + 1) / (h + m + 2), 4)

    def mark_confirmed(self, rule) -> None:
        """Kendi ölçümüm bir örnekte TUTTU: bu kanıt sayılır (söz değil, sonuç)."""
        feat = str(getattr(rule, "feature", "") or "")
        if not feat:
            return
        for rec in self.measurements:
            if str(rec.get("key")) == feat:
                rec["hits"] = int(rec.get("hits") or 0) + 1
                self._save_ledger()
                return

    @staticmethod
    def human_count() -> int:
        return len(OPERATORS)

    def counts(self) -> dict:
        h, s = self.human_count(), len(self.authored)
        return {"human": h, "self": s, "refuted": len(self.refuted),
                "measured": len(self.measurements),
                "measured_refuted": sum(1 for r in self.measurements if r.get("refuted")),
                "trusted": sum(1 for r in self.measurements
                               if self.trust(str(r.get("key") or "")) >= self.TRUST_FLOOR),
                "ratio": round(s / max(1, h + s), 4) if (h + s) else 0.0}

    # ================================================================= tümevarım
    def observe(self, pairs, pool=None) -> dict:
        """Örnekleri al, adayları yeniden hesapla, karar ver."""
        self.examples = [(str(t).strip(), str(l).strip().lower())
                         for t, l in pairs if str(t).strip() and str(l).strip()]
        self.adopted = None
        self.last_measure = {}
        self.cands = self._search(pool)
        return self.decide(pool)

    def add_example(self, text: str, label: str, pool=None) -> dict:
        self.examples.append((str(text).strip(), str(label).strip().lower()))
        self.cands = self._search(pool)
        self.adopted = None
        return self.decide(pool)

    @staticmethod
    def _sample(pool, n: int = 300) -> list[str]:
        """Karşılaştırma havuzu: deterministik sıralı, kısa ve yalın sözcükler."""
        out: list[str] = []
        for w in sorted({str(x).strip() for x in (pool or ()) if str(x or "").strip()}):
            if 2 <= len(w) <= 14 and w.replace(" ", "").isalpha():
                out.append(w)
            if len(out) >= n:
                break
        return out

    def _search(self, pool=None) -> list[Hypothesis]:
        ex = self.examples
        if len(ex) < 2:
            return []
        found: list[Hypothesis] = []
        feats = self.search_features()
        # İCAT EDİLMİŞ ölçümün TÜRÜ, hangi ŞEKİLDE taranacağını belirler:
        #   son ek / ön ek / harf / kalıp -> İKİLİ, yalnız "varlık" (var/yok)
        #   fark (bileşik)               -> SÜREKLİ, yalnız "eşik"
        # Bir son ekin "tek/çift"i olmaz, bir farkın "var/yok"u olmaz: kategorik yanlış
        # okuma, uydurma kadar kötüdür.
        kind_of = {str(r.get("key") or ""): str((r.get("spec") or {}).get("kind") or "")
                   for r in self.measurements}
        spec_of = {str(r.get("key") or ""): dict(r.get("spec") or {})
                   for r in self.measurements}
        for feat in feats:
            meas = [(int(FEATURES[feat][1](t)), lab) for t, lab in ex]
            hi = max(n for n, _ in meas)
            lo = min(n for n, _ in meas)
            kind = kind_of.get(feat, "")

            if kind in ("suffix", "prefix", "letter", "suffix_pattern"):
                b: dict[str, set] = {}
                for n, lab in meas:
                    b.setdefault("var" if int(n) else "yok", set()).add(lab)
                if len(b) == 2 and all(len(v) == 1 for v in b.values()):
                    lv, ly = next(iter(b["var"])), next(iter(b["yok"]))
                    if lv != ly:
                        found.append(Hypothesis(feat, "presence",
                                                (("var", lv), ("yok", ly)),
                                                spec=spec_of.get(feat)))
                continue

            if kind:      # bileşik (fark) ölçüm: yalnız eşik, negatif değerler dahil
                found.extend(self._scan_threshold(feat, meas, lo, hi + 1,
                                                  spec=spec_of.get(feat)))
                continue

            # --- parite: her iki taraf da GÖRÜLMÜŞ ve tek etiketli olmalı
            sides: dict[str, set] = {}
            for n, lab in meas:
                sides.setdefault("odd" if n % 2 else "even", set()).add(lab)
            if len(sides) == 2 and all(len(v) == 1 for v in sides.values()):
                lo_lab, le = next(iter(sides["odd"])), next(iter(sides["even"]))
                if lo_lab != le:
                    found.append(Hypothesis(feat, "parity",
                                            (("even", le), ("odd", lo_lab))))

            # --- eşik: YALNIZ "n < t" biçimi. `le/gt/ge` aynı ayrımın yeniden
            #     yazımıdır (etiketler taraflara zaten veriden atanıyor); hepsini
            #     taramak sahte belirsizlik üretir. Tek sınır, tek yazım.
            found.extend(self._scan_threshold(feat, meas, 1, min(MAX_T, hi + 2)))

        # --- AYNI KURALIN yeniden yazımını tekilleştir: anahtar (şekil, ölçüm,
        #     sınır). DAVRANIŞ imzasıyla tekilleştirmek YANLIŞ olurdu: iki farklı kural
        #     örneklerin TAMAMINI açıklayabilir (o yüzden ikisi de aday) ama yeni
        #     girdide ayrışır — belirsizlik tam olarak budur.
        # --- HİÇBİR YERLEŞİK ÖLÇÜM AYIRMIYORSA: yeni bir ÖLÇÜM icat et --------------
        # Bu, "operatör yazımı"nın bir üst basamağı: ölçüm de bizim değil, veriden.
        self.invented_now: list[Hypothesis] = []
        if not found:
            self.invented_now = self._invent(ex)
            found.extend(self.invented_now)

        #     "AYNI DAVRANIŞ" örneklerle değil GENİŞ HAVUZLA ölçülür: `hece` ile
        #     `sesli harf` bu depoda aynı ölçümün iki yazımıdır; 6 örnekte ayrışmayan
        #     ama 300 sözcükte de ayrışmayan iki aday kullanıcıya "belirsiz" diye
        #     satılmaz. Havuz verilmezse yalnız örnekler karşılaştırılır (daha temkinli).
        ref = list(ex) + [(w, None) for w in self._sample(pool)]
        uniq: dict[tuple, Hypothesis] = {}
        def _order(x):
            # Ölçüm tanım sırası (vowel_count önce) belirleyici: davranışı aynı olan
            # iki yazımdan ADI DAHA TEMEL olan tutulur ("hece" değil "sesli harf").
            try:
                fi = feats.index(x.feature)
            except ValueError:
                fi = 99
            return (fi, _SHAPE_ORDER.get(x.shape, 9), int(x.threshold), str(x.op))

        for h in sorted(found, key=_order):
            sig = h.signature(ref)
            key = sig if len(sig) > len(ex) else (h.shape, h.feature, int(h.threshold), str(h.op))
            uniq.setdefault(key, h)
        return list(uniq.values())

    @staticmethod
    def _scan_threshold(feat: str, meas, lo: int, hi_end: int, spec: dict | None = None):
        """Eşik taraması: YALNIZ "n < t". Aralık, ölçümün gerçek değerlerinden gelir.

        Bileşik ölçümler NEGATİF olabilir (sesli - sessiz), bu yüzden alt sınır sabit 1
        değil, verinin en küçük değeridir.
        """
        out: list[Hypothesis] = []
        for t in range(int(lo), int(hi_end) + 1):
            hs = Hypothesis(feat, "threshold", (("cond", None), ("else", None)), t, "lt",
                            spec=spec)
            buckets: dict[str, set] = {}
            for n, lab in meas:
                buckets.setdefault(hs.bucket(n), set()).add(lab)
            if len(buckets) == 2 and all(len(v) == 1 for v in buckets.values()):
                lc, lel = next(iter(buckets["cond"])), next(iter(buckets["else"]))
                if lc != lel:
                    out.append(Hypothesis(feat, "threshold",
                                          (("cond", lc), ("else", lel)), t, "lt",
                                          spec=spec))
        return out

    # ================================================================= ölçüm icadı
    @staticmethod
    def _common_suffix(words: list[str]) -> str:
        if not words:
            return ""
        ws = [w.lower() for w in words]
        first = ws[0]
        for n in range(len(first), 1, -1):
            suf = first[-n:]
            if len(suf) >= 2 and all(w.endswith(suf) for w in ws[1:]):
                return suf
        return ""

    @staticmethod
    def _common_prefix(words: list[str]) -> str:
        if not words:
            return ""
        ws = [w.lower() for w in words]
        first = ws[0]
        for n in range(len(first), 1, -1):
            pre = first[:n]
            if len(pre) >= 2 and all(w.startswith(pre) for w in ws[1:]):
                return pre
        return ""

    @staticmethod
    def _harmonized(s: str) -> str:
        """Ünlüleri 'v' ile değiştir: -lık/-lik/-luk/-lük aynı kalıba insin."""
        from nora_apply import VOWELS
        return "".join("v" if ch in VOWELS else ch for ch in str(s or "").lower())

    def _pattern_suffix(self, pos: list[str], neg: list[str]) -> str:
        """Ortak ÜNLÜ-UYUMLU son ek kalıbı ('lVk'). En az bir sessiz içermeli."""
        hs = [self._harmonized(w) for w in pos]
        if not hs:
            return ""
        first = hs[0]
        # Somut son ek zaten denendi; kalıp ancak ONU GEÇEN bir şeyse anlamlı.
        for n in range(min(len(first), 8), 1, -1):
            pat = first[-n:]
            if not any(ch not in "v" for ch in pat):
                continue     # yalnız ünlülerden oluşan "kalıp" ölçüm değildir
            if all(h.endswith(pat) for h in hs[1:]) and \
                    not any(self._harmonized(w).endswith(pat) for w in neg):
                return pat
        return ""

    def _composite(self, ex) -> list[Hypothesis]:
        """BİLEŞİK ÖLÇÜM İCADI: iki ölçümün FARKI sınıfları ayırıyor mu?

        Tek bir ölçüm ayırmıyorsa, iki ölçümün ARASINDAKİ fark ayırabilir. Bu, ölçüm
        icadının ikinci basamağı: yeni bir HAM ölçüm (son ek, harf) yoksa var olan iki
        ölçüm birleştirilir. Kural ilişkiseldir ("sesli eksi sessiz 1'den az") ve hiçbir
        yerleşik ölçüm bunu tek başına söyleyemez — yani insanın verdiği ölçüm listesi
        burada da tavandır değildir.

        Adaylar ÖLÇÜM SIRASINA göre tek yönlüdür (a, b) ve a<b; ters yön aynı ayrımın
        yeniden yazımı olurdu (sahte belirsizlik).
        """
        out: list[Hypothesis] = []
        for i, a in enumerate(BASE_FEATURES):
            for b in BASE_FEATURES[i + 1:]:
                # İKİ YÖN denenir (a-b ve b-a): aynı ayrımın iki yazımıdır. Hangisi
                # NEGATİF OLMAYAN bir sınır veriyorsa o tutulur — "-1'den az" yerine
                # "1'den çok" gibi okunur bir kural. Davranış aynı, söz daha temiz.
                best = None
                for x, y in ((a, b), (b, a)):
                    vals = [(int(FEATURES[x][1](t)) - int(FEATURES[y][1](t)), lab)
                            for t, lab in ex]
                    lo_v = min(n for n, _ in vals)
                    hi_v = max(n for n, _ in vals)
                    if lo_v == hi_v:
                        continue          # fark sabit: hiçbir şey ayırmaz
                    spec = {"kind": "diff", "a": x, "b": y}
                    hits = self._scan_threshold(f"kendi.diff.{x}-{y}", vals,
                                                lo_v, hi_v + 1, spec=spec)
                    if not hits:
                        continue
                    # EŞİK SEÇİLMEZ: veri sınırı belirlemiyorsa bütün eşikler aday kalır
                    # (belirsizlik dürüsttür, ayırt edici örnek istenir). Yalnız YÖN
                    # seçilir, o da okunuşu temiz olsun diye: negatif olmayan sınır.
                    nonneg = any(int(h.threshold) >= 0 for h in hits)
                    if best is None or (nonneg and not best[2]):
                        best = (hits, spec, nonneg)
                    if nonneg:
                        break
                if best is None:
                    continue
                hits, spec, _nn = best
                key, nm = hits[0].feature, (f"{SHORT_NAMES.get(spec['a'], spec['a'])}-"
                                            f"{SHORT_NAMES.get(spec['b'], spec['b'])} farkı")
                if key not in FEATURES:
                    try:
                        register_feature(key, nm, spec)
                    except Exception:
                        continue
                out.extend(hits)
        return out

    def _invent(self, ex) -> list[Hypothesis]:
        """Veriden YENİ ÖLÇÜM türet: ortak son ek / ön ek / yalnız bir sınıfta olan harf.

        Ölçüt: icat edilen ölçüm POZİTİF örneklerin hepsinde var, NEGATİFLERİN hiçbirinde
        yok. Aksi halde ölçüm ayrım yapmıyor demektir ve aday üretilmez.
        """
        labels = [l for _, l in ex]
        if len(set(labels)) < 2:
            return []
        pos_lab = labels[0]
        other = next(l for l in labels if l != pos_lab)
        pos = [t for t, l in ex if l == pos_lab]
        neg = [t for t, l in ex if l != pos_lab]
        if not pos or not neg:
            return []
        # ZAYIF KANIT: tek örnekten "ortak harf" çıkarmak tesadüf üretir. En az 2 pozitif
        # ve 2 negatif yoksa ölçüm İCAT EDİLMEZ (uydurmak yerine veri istenir).
        if len(pos) < 2 or len(neg) < 2:
            return []
        out: list[Hypothesis] = []

        def add(kind: str, arg: str, name: str) -> None:
            key = f"kendi.{kind}.{arg}"
            try:
                if key not in FEATURES:
                    register_feature(key, name, {"kind": kind, "arg": arg})
            except Exception:
                return
            out.append(Hypothesis(key, "presence", (("var", pos_lab), ("yok", other)),
                                  spec={"kind": kind, "arg": arg}))

        suf = self._common_suffix(pos)
        if suf and not any(t.lower().endswith(suf) for t in neg):
            add("suffix", suf, f"'{suf}' ile bitiş")
        pre = self._common_prefix(pos)
        if pre and not any(t.lower().startswith(pre) for t in neg):
            add("prefix", pre, f"'{pre}' ile başlangıç")
        # ÜNLÜ UYUMU: somut son ek tutmuyorsa KALIBI dene (`lık|lik|luk|lük` -> `lVk`).
        # Somut ek zaten eklendiyse kalıp onun genellemesidir; ikisi AYNI veriyi açıklarsa
        # aday ikiye çıkar ve sistem dürüstçe belirsiz der. Bu yüzden kalıp yalnız somut
        # ek BULUNAMADIĞINDA denenir.
        pat = "" if suf else self._pattern_suffix(pos, neg)
        if pat:
            add("suffix_pattern", pat, "'-" + pat.replace("v", "V") + "' kalıbı")

        letters = set("".join(pos).lower())
        for ch in sorted(letters):
            if not ch.isalpha():
                continue
            if all(ch in t.lower() for t in pos) and not any(ch in t.lower() for t in neg):
                add("letter", ch, f"'{ch}' harfi")
        # SON BASAMAK: ham ölçüm yoksa BİLEŞİK ölçüm (iki ölçümün farkı). İkili bir aday
        # bulunduysa bileşik aranmaz: aynı veriyi iki aday açıklarsa belirsizlik doğar.
        if not out:
            out.extend(self._composite(ex))
        return out

    # ================================================================= karar
    def decide(self, pool=None) -> dict:
        if self.adopted is not None:
            return {"status": "adopted", "rule": self.adopted, "desc": self.adopted.source_text,
                    "counts": self.counts()}
        if len(self.examples) < 2:
            self.why = "az-veri"
            return {"status": "insufficient", "why": self.why,
                    "counts": self.counts()}
        if not self.cands:
            self.why = self._why_empty()
            return {"status": "insufficient", "why": self.why,
                    "counts": self.counts()}
        if len(self.cands) == 1:
            return {"status": "adopted", "rule": self._adopt(self.cands[0]),
                    "desc": self.cands[0].desc(), "counts": self.counts()}
        self.probe_info = self.probe(pool) if pool else None
        return {"status": "ambiguous", "n": len(self.cands),
                "candidates": [h.desc() for h in self.cands[:6]],
                "probe": (self.probe_info or {}).get("word"),
                "counts": self.counts()}

    def _why_empty(self) -> str:
        """Aday yoksa NEDEN yok: uydurmak yerine adlandırılmış gerekçe."""
        labels = {lab for _, lab in self.examples}
        if len(labels) == 1:
            return "tek-sınıf"
        # Aynı girdi iki farklı etiketle verilmişse veri kendi içinde TUTARSIZDIR.
        by_text: dict[str, set] = {}
        for t, lab in self.examples:
            by_text.setdefault(t.strip().lower(), set()).add(lab)
        if any(len(v) > 1 for v in by_text.values()):
            return "çelişki"
        # Girdiler çelişmiyor ama hiçbir ölçüm iki tarafı AYIRAMIYOR: kural yok, veri az.
        # (Bunu "çelişki" diye adlandırmak yanlış teşhis olurdu: aynı girdiye iki etiket
        # verilmemiş; yalnızca hiçbir ölçüm ayırmıyor.)
        return "örüntü-yok"

    def _adopt(self, h: Hypothesis) -> Rule:
        rid = f"kendi.{len(self.authored) + 1}"
        rule = h.to_rule(rid, source=h.desc())
        rule.forge_meta = {"n_examples": len(self.examples),
                           "step": self.step, "origin": "kendi"}
        if self.eng is not None:
            try:
                self.eng.rules[rid] = rule
                self.eng.last = rule
            except Exception:
                pass
        # ÖLÇÜM KAYDI: aynı ölçüm ikinci kez kaydedilmez (envanter şişmez). İlk kez
        # yazıldıysa "icat", envanterde zaten varsa "yeniden kullanım" — söz bunu
        # saklamaz (nora_agi `stage`).
        now = {getattr(x, "feature", "") for x in (getattr(self, "invented_now", []) or [])}
        self.last_measure = {"key": (h.feature if h.spec else ""),
                             "invented": bool(h.spec) and h.feature in now,
                             "reused": bool(h.spec) and h.feature not in now}
        if h.spec:
            try:
                if not any(str(m.get("key")) == str(h.feature) for m in self.measurements):
                    self.measurements.append({"key": h.feature,
                                              "name": (FEATURES.get(h.feature) or (h.feature,))[0],
                                              "spec": dict(h.spec), "origin": "kendi",
                                              "for_rule": rid, "step": int(self.step)})
            except Exception:
                pass
        self.authored.append({
            "id": rid, "feature": h.feature, "shape": h.shape,
            "threshold": int(h.threshold), "op": str(h.op),
            "mapping": dict(h.mapping), "examples": len(self.examples),
            "desc": h.desc(), "origin": "kendi", "step": self.step,
        })
        self._save_ledger()
        self.adopted = rule
        return rule

    # ================================================================= ayırt edici örnek
    def probe(self, pool) -> dict | None:
        """Adayların AYRIŞTIĞI en kısa girdiyi bul: hangi etiketi alırsa kural çıkar."""
        if len(self.cands) < 2:
            return None
        known = {t for t, _ in self.examples}
        seen: set[str] = set()
        n = 0
        for w in pool or ():
            w = str(w or "").strip()
            if not w or w in known or w in seen or not w.replace(" ", "").isalpha():
                continue
            seen.add(w)
            n += 1
            if n > self.MAX_PROBE_SCAN:
                break
            preds = {h.predict(w) for h in self.cands}
            if len(preds) > 1:
                return {"word": w,
                        "options": {h.desc(): h.predict(w) for h in self.cands[:6]}}
        return None

    # ================================================================= durum
    def state(self) -> dict:
        return {"examples": len(self.examples), "candidates": len(self.cands),
                "measurements": len(self.measurements),
                "status": ("adopted" if self.adopted is not None else
                           ("ambiguous" if len(self.cands) > 1 else
                            ("insufficient" if self.examples else "boş"))),
                "counts": self.counts()}
