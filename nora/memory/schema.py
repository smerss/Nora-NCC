"""nora_schema.py — ŞEMA: makroda bir delik soyutla, aileyi tarafla.

Merdiven
--------
    3  makro (reify) : sırala∘dilimle(3)          → aynı davranışın kısa adı
    4  ŞEMA          : sırala∘dilimle(n)          → İNSANIN SAYMADIĞI üyeler
    5  taban uzantısı: Nora önerir, insan onaylar
    6  kendini değiştirme yetkisi

Makro sıkıştırır, genişletmez. Şema genişletir: `n=3` ve `n=4` görülmüşse
`sırala∘dilimle(n)` şeması `n=7`yi de kapsar — ve `n=7` **hiçbir ilkelde yok**.
Yeni erişimin ölçülebilir tanımı budur.

Kurallar (şema mı, makro mu)
----------------------------
1. En az **2 farklı görülmüş parametre** (tek üyeli aile şema değil makrodur).
2. Delik **tek** bileşende; iki üyenin yapısal karşılaştırmasıyla bulunur. Aile
   üreticisi varsa kullanılır; yoksa örneklerden çıkarılan anlam (ör. önek) yeterlidir.
3. Betimleme **kısalmalı**: şema 1 adım sayılır; somut makro 2+ adım.
4. Yeni form yazılmaz: şema bir *organ*, form değil. Repertuvar donuk kalır.
   `family_of` gibi ada özel çözücüler çıkarımın parçası değildir.
"""
from __future__ import annotations

from nora._paths import REPO_ROOT


from nora_synth import Program, instantiate


class Schema:
    """Delikli makroları bulan ve uzaya kaydeden ocak."""

    MIN_SEEN = 2

    MAX_IDLE = 4      # kullanılmayan şema kaç karar sonra budanır

    def __init__(self):
        self.ledger: list[dict] = []
        self.created = 0
        self.pruned = 0

    # ------------------------------------------------------------------ delik anlamı
    @staticmethod
    def infer_prefix_hole(members) -> dict | None:
        """DELİĞİN ANLAMINI VERİDEN ÇIKAR: "çıktı, bir taban dizgenin ilk n harfidir".

        `members`: [(n, [(girdi, çıktı), …]), …] — aynı ailenin farklı üyeleri.
        Koşullar (hepsi veriden, tablodan değil):
          (1) her üyenin çıktısı, parametresi kadar uzun (len(out) == n),
          (2) küçük parametrenin çıktısı büyük parametrenin ÖNEKİdir (iç içe).
        Taban program `B` aranır: B(girdi) her örnekte çıktıyla BAŞLAMALI.

        Bulunursa, görülmemiş n için f_n(w) = B(w)[:n] — PARAM_GENERATORS tablosuna
        hiç bakmadan. Yani "hangi aile parametriktir" bilgisi de veriden gelebilir.
        """
        # Anti-test: TEK parametre deliğin anlamını vermez (örnek: "ilk 3" tek
        # başına hem "sırala∘dilimle" hem "ters∘dilimle" olabilir). En az 2 şart.
        if len({int(n) for n, _ex in members}) < 2:
            return None
        pairs_all = []
        for n, exs in members:
            if not exs:
                return None
            for a, b in exs:
                if len(str(b)) != int(n):
                    return None
                pairs_all.append((str(a), str(b), int(n)))
        if not pairs_all:
            return None
        # iç içe önek koşulu: aynı girdinin farklı n çıktıları önek olmalı
        by_in: dict[str, list[tuple[int, str]]] = {}
        for a, b, n in pairs_all:
            by_in.setdefault(a, []).append((n, b))
        for a, lst in by_in.items():
            lst.sort()
            for (_n1, b1), (_n2, b2) in zip(lst, lst[1:]):
                if not str(b2).startswith(str(b1)):
                    return None
        # taban program arayışı: tek ilkel (ve gerekirse iki ilkel) — ucuz ve okunur
        from nora_synth import PRIMITIVES
        pool = list(PRIMITIVES)
        for nm, fn, _c in pool:
            if all(str(fn(a)).startswith(b) for a, b, _n in pairs_all):
                return {"kind": "prefix", "base": [nm]}
        for nm1, f1, _c1 in pool:
            for nm2, f2, _c2 in pool:
                if all(str(f2(f1(a))).startswith(b) for a, b, _n in pairs_all):
                    return {"kind": "prefix", "base": [nm1, nm2]}
        return None

    # ------------------------------------------------------------- yapısal karşılaştırma
    @staticmethod
    def _tokens(text: str) -> list[str]:
        """Metni, sayı kümelerini ayrı parçalar yaparak böler (regex yok).

        İşaret sayıya aittir: `kaydırma(+4)` → `kaydırma(`, `+4`, `)`.
        Böylece hangi adın parametrik olduğunu önceden bilmek gerekmez; iki örneğin
        farkına bakmak yeter.
        """
        s = str(text)
        out, buf, i = [], "", 0
        while i < len(s):
            signed = s[i] in "+-" and i + 1 < len(s) and s[i + 1].isdigit()
            if s[i].isdigit() or signed:
                if buf:
                    out.append(buf)
                    buf = ""
                j = i + 1
                while j < len(s) and s[j].isdigit():
                    j += 1
                out.append(s[i:j])
                i = j
            else:
                buf += s[i]
                i += 1
        if buf:
            out.append(buf)
        return out

    @classmethod
    def compare(cls, a, b) -> dict | None:
        """İki programı karşılaştır; tek fark sayısal ise deliği VERİDEN çıkar.

        Aile adı sözlükte aranmaz. Diğer bütün adımlar ve adın sayı dışı parçaları
        aynı kalmalıdır. Örneğin `ilk 3 harfi al` / `ilk 4 harfi al` karşılaştırması
        `ilk {n} harfi al` şablonunu üretir; bu şablon iki adın kendisinden türetilir.
        """
        a, b = [str(x) for x in (a or [])], [str(x) for x in (b or [])]
        if not a or len(a) != len(b):
            return None
        diffs = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        if len(diffs) != 1:
            return None
        hole = diffs[0]
        ta, tb = cls._tokens(a[hole]), cls._tokens(b[hole])
        if len(ta) != len(tb):
            return None
        varying = [i for i, (x, y) in enumerate(zip(ta, tb)) if x != y]
        if len(varying) != 1:
            return None
        k = varying[0]
        try:
            na, nb = int(ta[k]), int(tb[k])
        except (TypeError, ValueError):
            return None
        if na == nb:
            return None
        # Parantez içindeki/işaretli parametreler işaretli ailedir; diğerleri n.
        # Bu, ada özel bir liste değil, karşılaştırmadan çıkan sözdizimidir.
        pre, post = "".join(ta[:k]), "".join(ta[k + 1:])
        marker = "{k}" if (ta[k].startswith(("+", "-")) or tb[k].startswith(("+", "-"))
                            or (pre.endswith("(") and post.startswith(")"))) else "{n}"
        family = pre + marker + post
        steps = list(a)
        steps[hole] = family
        return {"steps": steps, "hole": hole, "family": family,
                "seen": sorted({na, nb})}

    @staticmethod
    def parameter_of(name: str, family: str) -> int | None:
        """Somut adın, daha önce ÇIKARILMIŞ ailedeki parametresini oku."""
        marker = next((m for m in ("{n}", "{k}") if m in str(family)), None)
        if marker is None:
            return None
        pre, post = str(family).split(marker, 1)
        name = str(name)
        if not name.startswith(pre) or not name.endswith(post):
            return None
        middle = name[len(pre):len(name) - len(post) if post else None]
        try:
            return int(middle)
        except (TypeError, ValueError):
            return None

    @classmethod
    def member_parameter(cls, spec, sc: dict) -> int | None:
        """Program bu şemanın hangi somut üyesi? (eşleşmiyorsa None)."""
        spec = [str(x) for x in (spec or [])]
        steps = list(sc.get("steps") or [])
        hole = int(sc.get("hole", -1))
        if len(spec) != len(steps) or not (0 <= hole < len(steps)):
            return None
        if any(spec[i] != steps[i] for i in range(len(steps)) if i != hole):
            return None
        return cls.parameter_of(spec[hole], str(sc.get("family") or ""))

    @staticmethod
    def spec_for(sc: dict, n: int) -> list[str] | None:
        """Şemanın KENDİ şablonundan yeni üye kur; ada özel regex kullanma."""
        steps = list(sc.get("steps") or [])
        hole = int(sc.get("hole", -1))
        family = str(sc.get("family") or "")
        if not (0 <= hole < len(steps)):
            return None
        hit = instantiate(family, int(n))
        if hit is not None:
            steps[hole] = hit[0]
            return steps
        marker = next((m for m in ("{n}", "{k}") if m in family), None)
        if marker is None:
            return None
        value = f"{int(n):+d}" if marker == "{k}" and int(n) >= 0 else str(int(n))
        steps[hole] = family.replace(marker, value)
        return steps

    def hypothesis(self, spec, n: int) -> list[str] | None:
        """Önce ONUN defterindeki şemalara bak; uygun şema yoksa uydurma."""
        for sc in reversed(self.ledger):
            if not sc.get("alive", True):
                continue
            if self.member_parameter(spec, sc) is not None:
                return self.spec_for(sc, n)
        return None

    def infer_items(self, items) -> list[dict]:
        """`items`: [(spec, [(girdi, çıktı), …]), …] — delik anlamı da çıkarılır."""
        items = [(list(sp), list(ex or [])) for sp, ex in items]
        out = self.infer([sp for sp, _ex in items])
        for sc in out:
            by_n: dict[int, list] = {}
            for sp, exs in items:
                n = self.member_parameter(sp, sc)
                if n is not None:
                    by_n.setdefault(int(n), []).extend(exs)
            members = [(n, by_n.get(n, [])) for n in sc["seen"]]
            inf = self.infer_prefix_hole(members)
            if inf:
                sc["inferred"] = inf
        return out

    # ------------------------------------------------------------------ çıkarım
    def infer(self, specs) -> list[dict]:
        """Delik, ada özel sözlükten değil ÜYELERİN KARŞILAŞTIRILMASINDAN doğar."""
        specs = [[str(x) for x in sp] for sp in (specs or []) if sp]
        groups: dict[tuple, dict] = {}
        for i, a in enumerate(specs):
            for b in specs[i + 1:]:
                sc = self.compare(a, b)
                if sc is None:
                    continue
                key = (tuple(sc["steps"]), int(sc["hole"]), sc["family"])
                g = groups.setdefault(key, {"steps": list(sc["steps"]),
                                            "hole": int(sc["hole"]),
                                            "family": sc["family"], "seen": set()})
                g["seen"].update(sc["seen"])
        out = []
        for g in groups.values():
            if len(g["seen"]) >= self.MIN_SEEN:
                out.append({"steps": list(g["steps"]), "hole": int(g["hole"]),
                            "family": g["family"], "seen": sorted(g["seen"])})
        return out

    def register(self, sc: dict, synth) -> dict | None:
        """Şemayı kalıcı deftere ve sentez uzayına yaz."""
        if not sc or len(sc.get("seen") or []) < self.MIN_SEEN:
            return None
        if any(r["steps"] == sc["steps"] and r["hole"] == sc["hole"]
               and r["family"] == sc["family"] for r in self.ledger):
            return None
        item = {"steps": list(sc["steps"]), "hole": int(sc["hole"]),
                "family": sc["family"], "seen": list(sc["seen"]),
                "inferred": sc.get("inferred"),
                "name": f"şema.{self.created + 1}", "uses": 0, "idle": 0,
                "alive": True}
        self.created += 1
        self.ledger.append(item)
        if synth is not None:
            synth.add_schema(item["steps"], item["hole"], item["family"], item["seen"],
                             inferred=item.get("inferred"))
        return item

    # ------------------------------------------------------------------ kullanım
    def note_use(self, family: str) -> None:
        for r in self.ledger:
            if r["family"] == str(family) and r.get("alive", True):
                r["uses"] = int(r.get("uses", 0)) + 1
                r["idle"] = 0
                return

    def tick(self) -> list[str]:
        """Bir karar geçti: kullanılmayan şema sayaç artırır, sınırı geçen BUDANIR.

        Şema enflasyonu da enflasyondur: iki parametre gördüm diye on şema açmak
        betimlemeyi KISALTMAZ. Ölçü yazılan şema değil, kullanılan şemadır.
        """
        pruned = []
        for r in self.ledger:
            if not r.get("alive", True):
                continue
            r["idle"] = int(r.get("idle", 0)) + 1
            if int(r["idle"]) > self.MAX_IDLE and int(r.get("uses", 0)) == 0:
                r["alive"] = False
                self.pruned += 1
                pruned.append(str(r["name"]))
        return pruned

    def survival(self) -> dict:
        alive = [r for r in self.ledger if r.get("alive", True)]
        used = [r for r in alive if int(r.get("uses", 0)) > 0]
        total = len(self.ledger)
        return {"total": total, "alive": len(alive), "used": len(used),
                "pruned": self.pruned,
                "survival": (len(used) / total) if total else 1.0}

    @staticmethod
    def covers(item: dict, n: int) -> bool:
        """Şema bu parametreyi kapsıyor mu? (görülmüş üyeler dahil)"""
        return int(n) >= 1

    def state(self) -> dict:
        return {"created": self.created, "pruned": self.pruned,
                "ledger": self.ledger[-16:]}

    def load_state(self, d, synth=None) -> None:
        d = d or {}
        self.created = int(d.get("created", 0) or 0)
        self.pruned = int(d.get("pruned", 0) or 0)
        self.ledger = [r for r in (d.get("ledger") or []) if isinstance(r, dict)]
        if synth is not None:
            for r in self.ledger:
                if not r.get("alive", True):
                    continue
                synth.add_schema(r.get("steps") or [], int(r.get("hole", 0)),
                                 r.get("family") or "", r.get("seen") or [],
                                 inferred=r.get("inferred"))
