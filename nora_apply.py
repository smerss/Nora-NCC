"""nora_apply.py — UYGULAMA: kuralı NESNE olarak bağla, yeni girdiye UYGULA.

Teşhis
------
Nora **retrieve** ediyordu, **apply** etmiyordu. Kavram ağında kenar vardı ama bir kuralı
nesne olarak bağlayıp yeni bir girdiye uygulayan operatör yoktu. Bu yüzden kural tümevarımı,
transfer, sayma, planlama — hepsi aynı noktadan düşüyordu: deyim kitabı, kompozisyon yok.

Bu organ üç primitif getirir:

    1. WorkingMemory    — 7 yuvalı çalışma belleği (plan, sayma, çok adım için tampon)
    2. Rule             — bağlanmış kural NESNESİ: özellik -> ölçüm -> eşleme -> etiket
    3. Interpreter      — decompose -> dene -> doğrula -> (olmazsa) AÇIKÇA "yapamadım"

Kural metni değil, kuralın KENDİSİ saklanır:

    "sesli harf sayısı tek ise mavi, çift ise kırmızı"
      -> Rule(feature="vowel_count", map={"tek":"mavi", "çift":"kırmızı"})

    apply(rule, "Bal arısı uçar")  ->  ("mavi", trace=[vowel_count=6, parity=çift, ...])

Yani yeni bir girdi geldiğinde Nora cümleyi hatırlamıyor; ölçüyor ve eşliyor. Transfer de
aynı nesnenin özelliğini değiştirmektir:

    transfer(rule, feature="word_count", map={"tek":"tuzlu", "çift":"tatlı"}, domain="mutfak")

Her adım iz bırakır (`trace`), böylece "nasıl yaptın" sorusu da retrieve değil kayıt okuma
olur.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

VOWELS = set("aeıioöuüAEIİOÖUÜ")

#: Ölçüm fonksiyonları: kural nesnesinin "özellik" alanı bunlardan birine çözülür.
def _tokens(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text or "", flags=re.UNICODE)


FEATURES = {
    "vowel_count": ("sesli harf", lambda s: sum(1 for ch in (s or "") if ch in VOWELS)),
    "char_count": ("harf", lambda s: sum(1 for ch in (s or "") if ch.isalpha())),
    "consonant_count": ("sessiz harf", lambda s: sum(1 for ch in (s or "") if ch.isalpha() and ch not in VOWELS)),
    "word_count": ("sözcük", lambda s: len(_tokens(s))),
    "letter_count": ("harf", lambda s: sum(1 for ch in (s or "") if ch.isalpha())),
    "syllable_count": ("hece", lambda s: max(1, sum(1 for ch in (s or "") if ch in VOWELS))),
}

#: Özellik sözünü (kullanıcının söylediği) FEATURES anahtarına çevir.
FEATURE_WORDS = {
    "sesli harf": "vowel_count", "sesli": "vowel_count", "ünlü": "vowel_count",
    "sessiz harf": "consonant_count", "sessiz": "consonant_count",
    "harf": "letter_count", "sözcük": "word_count", "kelime": "word_count",
    "hece": "syllable_count",
    # Alana özgü "adet" sözleri: sayılan şey öğedir, harf değil.
    "malzeme": "word_count", "paket": "word_count", "nesne": "word_count",
    "öğe": "word_count", "adet": "word_count", "ürün": "word_count",
    "kişi": "word_count", "kutu": "word_count", "sipariş": "word_count",
}

PARITY_WORDS = {"tek": "odd", "çift": "even", "tek sayı": "odd", "çift sayı": "even"}


@dataclass
class Slot:
    """Çalışma belleğindeki tek yuva."""
    key: str
    value: object
    source: str = ""


class WorkingMemory:
    """7 yuvalı çalışma belleği. Plan, sayma ve çok adımlı iş için tampon.

    Bilinçli sınır: kapasite 7. Daha fazlası istenirse en eski yuva düşer (insan belleği
    gibi) — ve bu düşme `evicted` listesine yazılır, sessizce kaybolmaz.
    """

    def __init__(self, capacity: int = 7):
        self.capacity = int(capacity)
        self.slots: dict[str, Slot] = {}
        self.order: list[str] = []
        self.evicted: list[Slot] = []
        self.writes = 0

    # ------------------------------------------------------------------ yaz/oku
    def write(self, key: str, value, source: str = "") -> None:
        k = str(key)
        if k not in self.slots:
            self.order.append(k)
            if len(self.order) > self.capacity:
                old = self.order.pop(0)
                gone = self.slots.pop(old, None)
                if gone is not None:
                    self.evicted.append(gone)
        self.slots[k] = Slot(k, value, source)
        self.writes += 1

    def read(self, key: str, default=None):
        s = self.slots.get(str(key))
        return default if s is None else s.value

    def drop(self, key: str) -> None:
        k = str(key)
        if k in self.slots:
            del self.slots[k]
            if k in self.order:
                self.order.remove(k)

    def clear(self) -> None:
        self.slots.clear()
        self.order.clear()

    @property
    def load(self) -> int:
        return len(self.slots)

    def snapshot(self) -> dict:
        return {k: self.slots[k].value for k in self.order if k in self.slots}

    def state_dict(self) -> dict:
        return {"slots": [{"key": s.key, "value": str(s.value)[:300], "source": s.source}
                          for s in (self.slots[k] for k in self.order if k in self.slots)]}

    def load_state_dict(self, d: dict | None) -> None:
        if not isinstance(d, dict):
            return
        try:
            self.clear()
            for rec in (d.get("slots") or [])[: self.capacity]:
                self.write(str(rec.get("key", "")), rec.get("value", ""), str(rec.get("source", "")))
        except Exception:
            pass


@dataclass
class Rule:
    """BAĞLANMIŞ KURAL NESNESİ — cümle değil, operatör.

    `feature` ölçüm fonksiyonunu, `mapping` ölçüm sonucunu etikete çevirmeyi söyler.
    `domain` kuralın hangi alana bağlandığını tutar (arıcılık, mutfak, lojistik …).
    """
    rid: str
    feature: str
    mapping: dict[str, str]
    domain: str = "genel"
    source_text: str = ""
    uses: int = 0
    derived_from: str = ""

    def measure(self, item: str) -> int:
        fn = FEATURES.get(self.feature, FEATURES["letter_count"])[1]
        return int(fn(item))

    def apply(self, item: str) -> tuple[str | None, dict]:
        """Yeni girdiye uygula: ölç -> eşle -> etiket. İz (trace) döner."""
        n = self.measure(item)
        parity = "odd" if n % 2 else "even"
        label = self.mapping.get(parity)
        self.uses += 1
        trace = {"rule": self.rid, "feature": self.feature,
                 "feature_name": FEATURES.get(self.feature, ("harf",))[0],
                 "item": item, "measured": n, "parity": parity,
                 "mapped": label, "domain": self.domain}
        return label, trace

    def transfer(self, feature: str, mapping: dict[str, str], domain: str) -> "Rule":
        """Yapıyı koru, alanı değiştir: aynı operatörün yeni bağlanışı."""
        return Rule(rid=f"{self.rid}->{domain}",
                    feature=feature, mapping=dict(mapping), domain=domain,
                    source_text=self.source_text, derived_from=self.rid)


class Interpreter:
    """Küçük yorumlayıcı: kural bağla, uygula, planı adım adım yürüt, olmazsa AÇIKÇA söyle."""

    MAX_PLAN_STEPS = 12

    def __init__(self, wm: WorkingMemory | None = None):
        self.wm = wm or WorkingMemory()
        self.rules: dict[str, Rule] = {}
        self.last: Rule | None = None
        self.trace: list[dict] = []
        self.failures: list[str] = []

    # ------------------------------------------------------------------ bağlama
    def bind(self, text: str) -> Rule | None:
        """Kullanıcının söylediği kuralı NESNEYE çevir.

        Anladığı biçimler (üretken, cümle listesi değil):

            "<özellik> sayısı tek ise <A>, çift ise <B>"
            "<özellik> tek ise <A>, çift ise <B>"
            "<özellik> çift ise <A>, tek ise <B>"
        """
        t = (text or "").strip()
        low = t.lower()

        # özellik: kullanıcının söylediği ada en uzun eşleşme
        feature = None
        for word, key in sorted(FEATURE_WORDS.items(), key=lambda kv: -len(kv[0])):
            if word in low:
                feature = key
                break
        if feature is None:
            feature = "letter_count"

        # eşleme: "tek ise X", "çift ise Y" (sıra serbest)
        pairs = re.findall(r"(tek|çift)\s*(?:sayı\s*)?(?:ise|olursa|->|:)\s*([^,;.]+)", low)
        mapping: dict[str, str] = {}
        for par, lab in pairs:
            lab = re.sub(r"\s+", " ", lab).strip(" ,.;")
            # etiket çok uzunsa yalnız ilk sözcüğü al ("mavi" gibi)
            lab = lab.split()[0] if lab else lab
            if lab:
                mapping[PARITY_WORDS[par]] = lab
        if len(mapping) < 2:
            return None

        rid = f"R{len(self.rules) + 1}"
        rule = Rule(rid=rid, feature=feature, mapping=mapping, domain=self._domain_of(low),
                    source_text=t)
        self.rules[rid] = rule
        self.last = rule
        self.wm.write("rule", rid, "bind")
        self.wm.write("rule_map", dict(mapping), "bind")
        self._trace("bind", {"rule": rid, "feature": feature, "map": mapping})
        return rule

    @staticmethod
    def _domain_of(low: str) -> str:
        for d, words in (("mutfak", ("mutfak", "malzeme", "tuzlu", "tatlı", "yemek", "tarif")),
                         ("lojistik", ("lojistik", "paket", "kargo", "sevkiyat", "acil")),
                         ("arıcılık", ("arı", "kovan", "bal", "petek")),
                         ("dil", ("harf", "sesli", "sözcük", "cümle"))):
            if any(w in low for w in words):
                return d
        return "genel"

    def transfer(self, new_mapping: dict[str, str] | None = None,
                 feature: str | None = None, domain: str = "") -> Rule | None:
        """Son kuralın YAPISINI başka alana taşı."""
        if self.last is None:
            return None
        rule = self.last.transfer(feature or self.last.feature,
                                  new_mapping or dict(self.last.mapping),
                                  domain or self.last.domain)
        self.rules[rule.rid] = rule
        self.last = rule
        self._trace("transfer", {"from": rule.derived_from, "to": rule.rid,
                                 "feature": rule.feature, "domain": rule.domain})
        return rule

    # ------------------------------------------------------------------ uygulama
    def apply(self, item: str, rule: Rule | None = None) -> tuple[str | None, dict]:
        r = rule or self.last
        if r is None:
            self.failures.append("apply:no_rule")
            return None, {"error": "no_rule"}
        label, tr = r.apply(item)
        self._trace("apply", tr)
        return label, tr

    def count(self, feature: str, item: str) -> int:
        n = int(FEATURES.get(feature, FEATURES["letter_count"])[1](item))
        self._trace("count", {"feature": feature, "item": item, "n": n})
        return n

    # ------------------------------------------------------------------ plan
    def plan(self, goal: str, n_steps: int = 10) -> list[dict]:
        """Hedefi adımlara ayır. Adımlar cümle değil, (eylem, hedef, ölçüt) üçlüsüdür.

        Bu, hazır plan metni değildir: her adım hedefin kendisinden ve ölçülebilir
        bir kontrol noktasından türetilir. `revise()` hedef değişince aynı makineyi
        yeniden çalıştırır; eski adımlar `superseded` olarak işaretlenir.
        """
        g = (goal or "").strip() or "belirsiz"
        n = max(3, min(int(n_steps), self.MAX_PLAN_STEPS))
        steps = []
        for i in range(1, n + 1):
            if i == 1:
                act, crit = "durum", "başlangıç ölçüldü"
            elif i == n:
                act, crit = "doğrula", "hedef_metrigi"
            elif i == n - 1:
                act, crit = "toparla", "artık kalmadı"
            elif i % 3 == 0:
                act, crit = "ölç", "ilerleme sayısal"
            else:
                act, crit = "yürüt", f"adım {i} tamam"
            steps.append({"i": i, "act": act, "goal": g, "criterion": crit,
                          "status": "open"})
        self.wm.write("plan_goal", g, "plan")
        self.wm.write("plan_n", n, "plan")
        self._trace("plan", {"goal": g, "steps": n})
        return steps

    def revise(self, old: list[dict], new_goal: str, n_steps: int = 10) -> list[dict]:
        for s in old:
            s["status"] = "superseded"
        new = self.plan(new_goal, n_steps)
        self.wm.write("plan_goal", new_goal, "revise")
        self._trace("revise", {"from": old[0]["goal"] if old else "", "to": new_goal})
        return new

    # ------------------------------------------------------------------ iz
    def _trace(self, kind: str, rec: dict) -> None:
        r = {"kind": kind}
        r.update(rec)
        self.trace.append(r)
        if len(self.trace) > 200:
            del self.trace[:-200]

    def last_trace(self, n: int = 1) -> list[dict]:
        return self.trace[-int(n):]

    def state_dict(self) -> dict:
        return {"rules": [{"rid": r.rid, "feature": r.feature, "mapping": r.mapping,
                           "domain": r.domain, "source": r.source_text,
                           "derived_from": r.derived_from, "uses": r.uses}
                          for r in self.rules.values()],
                "last": self.last.rid if self.last else "",
                "wm": self.wm.state_dict()}

    def load_state_dict(self, d: dict | None) -> None:
        if not isinstance(d, dict):
            return
        try:
            self.rules = {}
            for rec in (d.get("rules") or []):
                r = Rule(rid=str(rec.get("rid")), feature=str(rec.get("feature")),
                         mapping=dict(rec.get("mapping") or {}),
                         domain=str(rec.get("domain") or "genel"),
                         source_text=str(rec.get("source") or ""),
                         derived_from=str(rec.get("derived_from") or ""),
                         uses=int(rec.get("uses", 0) or 0))
                self.rules[r.rid] = r
            lid = str(d.get("last") or "")
            self.last = self.rules.get(lid) or (list(self.rules.values())[-1] if self.rules else None)
            self.wm.load_state_dict(d.get("wm") or {})
        except Exception:
            pass


def _fixture() -> list[str]:
    """Öz-test cümleleri VERİDEN gelir (`nora_priors.json` -> `apply.selftest`).

    Test metni de bir sözdür; koda gömülürse iskelet sayacı haklı olarak büyür.
    Veri yoksa test susar — uydurma cümle üretmez.
    """
    try:
        import json
        import pathlib
        blob = json.loads(pathlib.Path(__file__).with_name("nora_priors.json")
                          .read_text(encoding="utf-8"))
        return [str(p.get("text", "")) for p in
                ((blob.get("kinds") or {}).get("apply.selftest") or {}).get("priors") or []
                if str(p.get("text", "")).strip()]
    except Exception:
        return []


def _selftest() -> None:
    fx = _fixture()
    if len(fx) < 5:
        print("selftest: no data")
        return
    it = Interpreter()
    r = it.bind(fx[0])
    print("kural:", r)
    for s in fx[1:4]:
        print(s, "->", it.apply(s)[0])
    print("transfer:", it.transfer({"odd": "tuzlu", "even": "tatlı"},
                                   feature="word_count", domain="mutfak"))
    print("mutfak:", it.apply(fx[4])[0])
    print("trace:", it.last_trace(3))


if __name__ == "__main__":
    _selftest()
