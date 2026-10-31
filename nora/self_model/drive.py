"""nora_drive.py — SÜRÜCÜ: talimat olmadan, şaşkınlığa göre kendi öğrenmesini tetikler.

Neden bu organ
--------------
Bugüne kadar tetikleyen insandı: "şu dönüşümü öğren", "şu örneklerden kural çıkar".
Nora cevap veriyordu ama **sürücüsü yoktu**. Bu organ akışı kendi izler, şaşkınlığı
ölçer, kümeler ve en yüksek şaşkınlıkta kendi kendine öğrenme çağırır.

    sürücüsüz sistem : araç  (sen tetiklersin, o cevaplar)
    sürücülü sistem  : ajan  (kendi tetikler)

Kararlılık (fren)
-----------------
Her şaşkınlık öğrenilmeye değmez; gürültü kovalanırsa sistem her cümleye bir kural
uydurur. Kural: **yalnız akışın betimleme uzunluğunu KISALTAN** öğrenme kabul edilir
(MDL). Bir küme en az `MIN_ITEMS` maddede aynı dönüşümü gösteriyorsa, tek adım/tek ad
`MIN_ITEMS` sürümden kısadır; aksi hâlde öğrenme yoktur.

Dürüstlük
---------
1. Şaşkınlık ölçümü veriden: dönüşüm çiftleri, bilinmeyen kavramlar, sayı olguları.
2. Öğrenme iddiası yalnız **doğrulanmış** kayıtla: `synth` programı benimsedi ya da
   `abstract` ilkel önerdi ve örneklerin hepsi tuttu.
3. Hiçbir şey öğrenilmediyse "öğrenmedim" denir; sürücü boş rapor uydurmaz.
"""
from __future__ import annotations

from nora._paths import REPO_ROOT


import copy

from nora_lang import tr_lower


class Driver:
    """Ham akışı izleyen, şaşkınlığı ölçen ve öğrenmeyi KENDİ tetikleyen sürücü."""

    MIN_ITEMS = 2          # bir kümenin öğrenme sayılması için en az madde
    MIN_SURPRISE = 0.55    # bu eşiğin altındaki şaşkınlık öğrenme tetiklemez
    MAX_ITEMS = 512

    def __init__(self, agi=None):
        self.agi = agi
        self.items: list[dict] = []      # {"text","kind","surprise","key"}
        #: BEKLENTİ MODELİ: sabit şaşkınlık tablosu DEĞİL. Aynı biçimi tekrar görmek
        #: şaşkınlığı düşürür (alışma), yeni biçim yükseltir. Olasılık kendi
        #: sayımlarından gelir; sabit 0.10/0.45 gibi sayılar insanın koyduğu eğilimdir.
        self._seen_shape: dict[tuple, int] = {}
        self._seen_total: dict[str, int] = {}
        self._prev_kind = ""
        self.seen = 0
        self.triggered = 0               # KENDİ tetiklediği öğrenme sayısı
        self.learned: list[dict] = []    # {"kind","name","n","keys"}
        self.rejected = 0                # MDL freni: kazancı olmayan öneriler
        self.unresolved = 0              # kanıt yetmedi (belirsiz/bilinmeyen) — sessiz kalır
        #: EŞİK ARTIK SABİT DEĞİL: kendi isabetini ölçüp ayarlar (irade).
        self.thresh = float(self.MIN_SURPRISE)
        #: Kanıt eşiği: kaç maddelik küme öğrenmeye değer. Gürültüde artar.
        self.min_items = int(self.MIN_ITEMS)
        self._specs: dict[tuple, int] = {}
        self._run = {"trig": 0, "unres": 0, "reuse": 0}
        # OTONOM HEDEF YAŞAM DÖNGÜSÜ: insan hedef listesi vermez. Adaylar akış
        # şaşkınlığı, kullanılmamış yetenek merakı ve gerçek envanterden çıkar.
        self.goals: list[dict] = []
        self.goal_history: list[dict] = []
        self.goal_serial = 0
        self.goal_step = 0
        self.goal_idle = 0
        self.goal_capability: dict[str, dict] = {}

    # ------------------------------------------------------------------ algı
    def _surprise_of(self, kind: str, shape: tuple) -> float:
        """Beklenti modeline göre şaşkınlık: 1 - p(biçim | bağlam). Alışma buradan."""
        key = (self._prev_kind, kind, shape)
        n = int(self._seen_total.get(f"{self._prev_kind}|{kind}", 0))
        c = int(self._seen_shape.get(key, 0))
        p = (c + 1.0) / (n + 3.0)      # Laplace: model boşken de sıfıra bölünmez
        return max(0.0, 1.0 - p)

    def _learn_shape(self, kind: str, shape: tuple) -> None:
        key = (self._prev_kind, kind, shape)
        self._seen_shape[key] = int(self._seen_shape.get(key, 0)) + 1
        k2 = f"{self._prev_kind}|{kind}"
        self._seen_total[k2] = int(self._seen_total.get(k2, 0)) + 1
        self._prev_kind = kind

    def ingest(self, text: str) -> dict:
        """Bir cümleyi izle: şaşkınlığını ÖLÇ ve kaydet.

        Şaşkınlık sabit bir tablodan gelmez: kendi biçim sayımlarına göre
        "bu biçimi bu bağlamda ne sıklıkta gördüm" sorusunun cevabıdır.
        """
        raw = str(text or "").strip()
        self.seen += 1
        rec = {"text": raw, "kind": "noise", "surprise": 0.10, "key": ()}
        if not raw:
            self.items.append(rec)
            return rec
        try:
            from nora_synth import parse_pairs
            pairs = parse_pairs(raw)
        except Exception:
            pairs = []
        if pairs:
            a, b = pairs[0]
            explained = self._explained(a, b)
            rec = {"text": raw, "kind": "pair", "pairs": pairs,
                   # BİÇİM anahtarı: (uzunluk değişti mi, harfler korunuyor mu, ÇIKTI
                   # UZUNLUĞU). Girdi uzunluğu DEĞİL: aynı parametre farklı uzunlukta
                   # girdilerle aynı kümede kalmalı. Çıktı uzunluğu ise PARAMETREYİ
                   # ayırır (ilk-3 ile ilk-4 ayrı küme) — ayrıştırma buradan doğuyor.
                   "key": (len(a) == len(b), sorted(a) == sorted(b), len(b)),
                   # AÇIKLANAN çift şaşkınlık üretmez: uzay onu zaten kapsıyor.
                   "surprise": 0.05 if explained else 1.0}
        else:
            low = tr_lower(raw)
            if " nedir" in low or " ne demek" in low:
                rec = {"text": raw, "kind": "question", "key": ("soru",)}
            elif any(ch.isdigit() for ch in raw):
                rec = {"text": raw, "kind": "number", "key": ("sayı",)}
        rec["surprise"] = round(self._surprise_of(rec["kind"], rec.get("key") or ()), 3)
        self._learn_shape(rec["kind"], rec.get("key") or ())
        self.items.append(rec)
        if len(self.items) > self.MAX_ITEMS:
            del self.items[:-self.MAX_ITEMS]
        return rec

    def ingest_stream(self, lines) -> int:
        for ln in (lines or []):
            self.ingest(ln)
        return self.seen

    def _explained(self, a: str, b: str) -> bool:
        """Bu çift şu an HALLEDİLİYOR mu? (kayıtlı programlardan biri uygulanabiliyorsa)"""
        agi = self.agi
        sy = getattr(agi, "synth", None) if agi is not None else None
        if sy is None:
            return False
        for rec in (sy.programs or [])[-8:]:
            try:
                p = sy.rebuild(rec.get("id"))
            except Exception:
                p = None
            if p is not None and p.apply(a) == b:
                return True
        return False

    # ------------------------------------------------------------------ kümeler
    def clusters(self) -> list[dict]:
        """Aynı biçimi paylaşan yüksek-şaşkınlık maddelerini grupla."""
        out: dict[tuple, dict] = {}
        for it in self.items:
            if it.get("kind") != "pair" or it.get("consumed"):
                continue
            if float(it.get("surprise", 0)) < self.thresh:
                continue
            key = it.get("key") or ()
            c = out.setdefault(key, {"key": key, "pairs": [], "surprise": 0.0})
            for pr in it.get("pairs") or []:
                if pr not in c["pairs"]:
                    c["pairs"].append(pr)
            c["surprise"] = max(c["surprise"], float(it["surprise"]))
        res = list(out.values())
        res.sort(key=lambda c: -(c["surprise"] * len(c["pairs"])))
        return res

    # ------------------------------------------------------------------ karar
    def step(self) -> dict | None:
        """En yüksek şaşkınlıklı kümeyi öğren. MDL freni burada."""
        agg = self.agi
        if agg is None:
            return None
        best = None
        for c in self.clusters():
            pairs = [tuple(p) for p in c["pairs"]]
            if len(pairs) < self.min_items:
                continue          # yetersiz kanıt: ne öğrenme ne ret sayılır
            best = (c, pairs)
            break
        if best is None:
            return None
        c, pairs = best
        sy = getattr(agg, "synth", None)

        # 0) KÜME BÖLME: biçim anahtarı aynı ama PARAMETRE farklı olabilir
        #    ('sırala∘ilk3' ve 'sırala∘ilk4' ikisi de "uzunluk değişti, harfler aynı").
        #    Tek küme olarak öğrenilemiyorsa çıktı uzunluğuna göre BÖL ve yeniden dene.
        #    Böylece çok parametreli aile, sürücünün kendiliğinden ayrıştırdığı şey olur.
        tries = [pairs]
        by_out: dict[int, list] = {}
        for a_, b_ in pairs:
            by_out.setdefault(len(b_), []).append((a_, b_))
        if len(by_out) > 1:
            tries += [g for _k, g in sorted(by_out.items(), key=lambda kv: -len(kv[1]))
                      if len(g) >= self.min_items]
        res = {"status": "insufficient"}
        for cand_pairs in tries:
            try:
                r_try = sy.learn(cand_pairs) if sy is not None else {"status": "insufficient"}
            except Exception:
                r_try = {"status": "insufficient"}
            if r_try.get("status") == "adopted":
                res, pairs = r_try, cand_pairs
                break
            if r_try.get("status") == "ambiguous":
                res = r_try
        if res.get("status") in ("ambiguous", "insufficient"):
            # Kanıt yetmedi: uydurma yok. Küme TÜKETİLMEZ; akış devam ederse yeniden
            # denenir. "Ret" sayılmaz: ret, denenip başarısız olan öneriye aittir.
            self.unresolved += 1
            self._run["unres"] += 1
            return None
        if res.get("status") == "adopted":
            self._consume(c)
            self.triggered += 1
            spec = tuple(getattr(res.get("program"), "spec", lambda: [])())
            self._run["reuse"] += int(self._specs.get(spec, 0) > 0)
            self._specs[spec] = self._specs.get(spec, 0) + 1
            self._run["trig"] += 1
            rec = {"kind": "kompozisyon", "name": res["program"].desc(),
                   "n": len(pairs), "keys": list(c["key"])}
            self.learned.append(rec)
            if hasattr(agg, "_after_synth"):
                try:
                    agg._after_synth(res["program"])
                except Exception:
                    pass
            return rec
        # 2) uzay yetmiyorsa: soyutlama (icat) — sürücü kendi çağırır
        ab = getattr(agg, "abstract", None)
        if ab is None:
            return None
        try:
            prop = ab.propose(pairs, composition_exists=False,
                              register=(lambda nm, fn: sy.add_primitive(nm, fn)) if sy else None)
        except Exception:
            prop = {"status": "insufficient"}
        if prop.get("status") != "invented":
            self.rejected += 1
            return None
        # MDL FRENİ: icat en az MIN_ITEMS maddeyi tek adla açıklamalı.
        fn = None
        try:
            from nora_abstract import build as _build
            fn = _build(prop["family"], prop["params"])
        except Exception:
            fn = None
        if fn is None or len(pairs) < self.MIN_ITEMS:
            self.rejected += 1
            return None
        if not all(fn(a) == b for a, b in pairs):
            self.rejected += 1
            return None
        self._consume(c)
        self.triggered += 1
        rec = {"kind": "icat", "name": prop["name"], "n": len(pairs),
               "keys": list(c["key"])}
        self.learned.append(rec)
        return rec

    def _consume(self, c: dict) -> None:
        """Öğrenilen kümeyi tüket: aynı şaşkınlık ikinci kez öğrenilmez."""
        key = c.get("key")
        for it in self.items:
            if it.get("kind") == "pair" and (it.get("key") or None) == key:
                it["consumed"] = True

    def run(self, max_steps: int = 6) -> list[dict]:
        """Akışı kendi kendine işle, sonunda eşiği KENDİ isabetine göre ayarla."""
        self._run = {"trig": 0, "unres": 0, "reuse": 0}
        out = []
        for _ in range(int(max_steps)):
            rec = self.step()
            if rec is None:
                break
            out.append(rec)
        self.adapt()
        return out

    def adapt(self) -> float:
        """Kendi isabetini ölç, eşiği oynat. Eşik artık insanın sabiti değil.

        - **Kanıt yetmiyor** (unresolved baskın): eşik DÜŞER — daha çok kanıt topla.
          (Gürültüye tetikleme riski, kanıtsızlıkta susmaktan iyidir; ama sınırlı.)
        - **Öğrendiği tekrar kullanılmıyor** (reuse=0): eşik YÜKSELİR — bu akışta
          öğrenilecek bir şey yok, boşuna tetikleme.
        """
        r = self._run
        if r["unres"] > max(1, 2 * r["trig"]):
            self.thresh = max(0.15, self.thresh * 0.90)
        elif r["trig"] > 0 and r["reuse"] == 0:
            self.thresh = min(0.95, self.thresh * 1.10)
            # Tekrar kullanılmayan öğrenme = kanıt zayıf: daha çok madde iste.
            self.min_items = min(4, self.min_items + 1)
        return self.thresh

    # ---------------------------------------------------------- otonom hedefler
    def _experience_version(self) -> tuple:
        agi = self.agi
        sy = getattr(agi, "synth", None) if agi is not None else None
        schema = getattr(agi, "schema", None) if agi is not None else None
        return (
            int(self.seen), len(getattr(sy, "programs", []) or []),
            len(getattr(agi, "_kendi_gorevler", []) or []),
            len(getattr(agi, "kendi_ögretim", []) or []),
            len(getattr(schema, "ledger", []) or []),
            int(getattr(agi, "_oz_aday_butce", 0) or 0),
        )

    def _capability_inventory(self) -> dict[str, dict]:
        agi = self.agi
        available = {
            "stream_learn": agi is not None and getattr(agi, "synth", None) is not None,
            "self_task": callable(getattr(agi, "kendine_gorev", None)),
            "self_teach": callable(getattr(agi, "kendine_ogret", None)),
            "self_change": callable(getattr(agi, "kendini_degistir", None)),
        }
        out = {}
        for name, present in available.items():
            stat = self.goal_capability.setdefault(
                name, {"attempts": 0, "successes": 0, "uses": 0},
            )
            attempts = int(stat.get("attempts", 0))
            successes = int(stat.get("successes", 0))
            uses = int(stat.get("uses", 0))
            out[name] = {
                "available": bool(present), "attempts": attempts,
                "successes": successes, "uses": uses,
                # Laplace yalnız seçim önceliğidir; başarı kanıtı değildir.
                "competence": (successes + 1.0) / (attempts + 2.0),
                "curiosity": 1.0 / (1.0 + uses),
            }
        return out

    @staticmethod
    def _blocker_of(kind: str, result: dict) -> str:
        status = str(result.get("durum") or result.get("status") or
                     result.get("neden") or "doğrulanmamış-sonuç")
        # "tamam" yalnız sürecin bittiğini söyleyebilir; öğrenme bayrağı false
        # ise blokajı "tamam" diye etiketlemek ölçüm hatasıdır.
        if status == "tamam" and (result.get("öğrendi") is False or
                                  result.get("değişti") is False):
            status = "self_validation"
        # İki farklı yüzey durumu aynı ölçülebilir organ blokajına indirgenir;
        # hedefin başarılı olduğu varsayılmaz, yalnız ortak eksik adlandırılır.
        if status in {"görev_kurulamadı", "kendi_ornegi_cozulemedi", "self_validation",
                      "ambiguous", "insufficient", "unresolved"}:
            return "program_synthesis" if kind != "stream_learn" else "evidence_insufficient"
        return status

    def _new_goal(self, kind: str, capability: str, *, surprise: float,
                  blocker: str = "", parents: list[str] | None = None,
                  signature: str = "", depth: int = 0) -> dict | None:
        inventory = self._capability_inventory()
        if not inventory.get(capability, {}).get("available"):
            return None
        version = self._experience_version()
        if not signature and kind == "expand":
            # İç başarıyı sonsuz yeni dış deneyim gibi sayma: aynı akışta bir
            # yetenek bir keşif hedefi açar. Yeni ham akış yeni hedef açabilir.
            signature = f"expand|{capability}|external:{self.seen}"
        sig = signature or f"{kind}|{capability}|{blocker}|{version!r}"
        # Deneyim değişmeden aynı terminal hedefi yeniden üretmek otonomi değil,
        # başarısızlığı bin kez saymaktır. Açık hedef zaten varsa da kopya yok.
        for old in self.goals:
            if old.get("signature") == sig:
                return old if old.get("status") == "active" else None
        if any(g.get("status") == "active" and g.get("kind") == kind
               and g.get("capability") == capability and kind == "expand"
               for g in self.goals):
            return None
        self.goal_serial += 1
        gid = f"G{self.goal_serial}"
        info = inventory[capability]
        goal = {
            "id": gid, "kind": kind, "capability": capability,
            "signature": sig, "status": "active", "created_step": self.goal_step,
            "attempts": 0, "failures": 0, "surprise": float(surprise),
            "curiosity": float(info["curiosity"]),
            "competence_at_birth": float(info["competence"]),
            "blocker": str(blocker), "parents": list(parents or []),
            "depth": int(depth), "history": [],
            "reason": "şaşkınlık+merak+envanter",
        }
        self.goals.append(goal)
        self.goal_history.append({"step": self.goal_step, "goal": gid,
                                  "event": "generated", "kind": kind,
                                  "capability": capability, "blocker": blocker,
                                  "parents": list(parents or [])})
        return goal

    def _ensure_autonomous_goals(self) -> None:
        # Akıştan gelen hedef, insanın "şunu öğren" demesi değildir; küme ve
        # şaşkınlık Driver'ın kendi algısından oluşur.
        for cluster in self.clusters():
            if len(cluster.get("pairs") or []) < self.min_items:
                continue
            key = repr(cluster.get("key"))
            self._new_goal("stream", "stream_learn",
                           surprise=float(cluster.get("surprise", 0.0)),
                           signature=f"stream|{key}")
        inventory = self._capability_inventory()
        latest_surprise = max([float(item.get("surprise", 0.0))
                               for item in self.items[-8:]] or [0.0])
        # Başlangıçta birden çok açık hedef bulunması ortak blokajları gerçekten
        # birleştirebilmek için gereklidir; tek hedefi yapay olarak bölmeyiz.
        for capability in ("self_task", "self_teach", "self_change"):
            if inventory.get(capability, {}).get("available"):
                self._new_goal("expand", capability, surprise=latest_surprise)

    def _select_goal(self) -> dict | None:
        inventory = self._capability_inventory()
        active = [goal for goal in self.goals if goal.get("status") == "active"]
        scored = []
        for goal in active:
            info = inventory.get(goal["capability"], {})
            surprise = float(goal.get("surprise", 0.0))
            curiosity = float(info.get("curiosity", 0.0))
            competence = float(info.get("competence", 0.0)) if info.get("available") else 0.0
            failures = int(goal.get("failures", 0))
            waiting = min(1.0, max(0, self.goal_step - int(goal.get("created_step", 0))) / 8.0)
            # Başarısızlık merakı, bekleyen hedefe dönme baskısı da seçime girer;
            # sürekli başarılı tek organ diğer açık hedefleri sonsuzca aç bırakamaz.
            score = (surprise + 0.65 * curiosity * (1.0 + 0.2 * failures)
                     + 0.35 * competence + 0.50 * waiting)
            scored.append((score, goal["id"], goal, {
                "surprise": surprise, "curiosity": curiosity,
                "competence": competence, "waiting": waiting,
                "available": bool(info.get("available")), "score": score,
            }))
        if not scored:
            return None
        scored.sort(key=lambda row: (-row[0], row[1]))
        _score, _gid, goal, rationale = scored[0]
        goal["last_rationale"] = rationale
        return goal

    def _derive_unblock(self, parents: list[dict], blocker: str) -> dict | None:
        parents = [goal for goal in parents if goal.get("depth", 0) < 2]
        if not parents or not blocker:
            return None
        depth = 1 + max(int(goal.get("depth", 0)) for goal in parents)
        # Program sentezi tıkandığında mevcut politika-değiştirme yeteneği gerçek
        # bir alternatif; diğer blokajlarda en az kullanılan yetenek seçilir.
        inventory = self._capability_inventory()
        if blocker == "program_synthesis" and inventory["self_change"]["available"]:
            capability = "self_change"
        else:
            choices = [(info["uses"], name) for name, info in inventory.items()
                       if info["available"] and name != "stream_learn"]
            if not choices:
                return None
            capability = min(choices)[1]
        surprise = max(float(goal.get("surprise", 0.0)) for goal in parents)
        child = self._new_goal(
            "unblock", capability, surprise=surprise, blocker=blocker,
            parents=[goal["id"] for goal in parents], depth=depth,
            signature=f"unblock|{blocker}|{capability}|{self._experience_version()!r}",
        )
        if child is not None:
            child["probe_capabilities"] = sorted({goal["capability"] for goal in parents})
        return child

    def _merge_blocked(self, goal: dict) -> bool:
        if goal.get("kind") == "unblock" or not goal.get("blocker"):
            return False
        # Önce var olan ortak alt hedefi kullan; aynı blokaj için ikinci bir
        # politika-değiştirme yarışı açmak birleşme değil çoğaltma olurdu.
        existing = [old for old in self.goals if old["id"] != goal["id"]
                    and old.get("status") == "active" and old.get("kind") == "unblock"
                    and old.get("blocker") == goal.get("blocker")]
        if existing:
            child = existing[0]
            parents = [goal]
        else:
            others = [old for old in self.goals if old["id"] != goal["id"]
                      and old.get("status") == "active" and old.get("kind") != "unblock"
                      and old.get("blocker") == goal.get("blocker")]
            if not others:
                return False
            parents = [goal, others[0]]
            child = self._derive_unblock(parents, str(goal["blocker"]))
            if child is None:
                return False
        for parent in parents:
            parent["status"] = "merged"
            parent["merged_into"] = child["id"]
            parent["reason"] = f"ortak alt hedefe indirgendi: {goal['blocker']}"
            child["parents"] = sorted(set(child.get("parents", []) + [parent["id"]]))
            child["probe_capabilities"] = sorted(set(
                child.get("probe_capabilities", []) + [parent["capability"]]))
            self.goal_history.append({"step": self.goal_step, "goal": parent["id"],
                                      "event": "merged", "into": child["id"],
                                      "blocker": goal["blocker"]})
        return True

    def _execute_goal(self, goal: dict) -> tuple[bool, dict]:
        agi = self.agi

        def call(capability: str) -> tuple[bool, dict]:
            if capability == "stream_learn":
                learned = self.step()
                return bool(learned), dict(learned or {"status": "unresolved"})
            method = getattr(agi, {
                "self_task": "kendine_gorev", "self_teach": "kendine_ogret",
                "self_change": "kendini_degistir",
            }[capability], None)
            if method is None:
                return False, {"durum": "yetenek-yok"}
            result = dict(method() or {})
            success = bool(result.get("öğrendi") or result.get("değişti"))
            return success, result

        try:
            success, result = call(goal["capability"])
            if not success or goal.get("kind") != "unblock":
                return success, result
            # Yardımcı yeteneğin başarısı blokajın kendisi değildir. Asıl başarısız
            # yetenekleri bağımsız olarak yeniden çalıştır; hepsi geçmeden alt
            # hedef "çözüldü" olamaz.
            probes = list(goal.get("probe_capabilities") or [])
            last = dict(result, helper_succeeded=True)
            for capability in probes:
                if capability == goal["capability"]:
                    probe_ok, probe = success, result
                else:
                    probe_ok, probe = call(capability)
                last = dict(probe, helper_succeeded=True,
                            probe_capability=capability)
                if not probe_ok:
                    return False, last
            return True, last
        except Exception as exc:
            return False, {"durum": f"exception:{type(exc).__name__}", "error": str(exc)}

    def autonomous_step(self) -> dict:
        """Bir hedef seçip gerçek yeteneği çalıştır; sonuç uydurma sayaç değildir."""
        self.goal_step += 1
        self._ensure_autonomous_goals()
        goal = self._select_goal()
        if goal is None:
            self.goal_idle += 1
            return {"step": self.goal_step, "event": "idle", "active": 0}
        success, result = self._execute_goal(goal)
        stat = self.goal_capability[goal["capability"]]
        stat["attempts"] += 1
        stat["uses"] += 1
        stat["successes"] += int(success)
        goal["attempts"] += 1
        summary = {key: result.get(key) for key in
                   ("durum", "status", "öğrendi", "değişti", "neden", "hedef", "taban",
                    "helper_succeeded", "probe_capability")
                   if key in result}
        event = {"step": self.goal_step, "goal": goal["id"],
                 "capability": goal["capability"], "success": success,
                 "result": summary, "rationale": dict(goal.get("last_rationale") or {})}
        goal["history"].append(event)
        self.goal_history.append(dict(event, event="attempt"))
        if success:
            goal["status"] = "solved"
            goal["solved_step"] = self.goal_step
            goal["failures"] = 0
            goal["reason"] = "yetenek sonucu"
            self.goal_history.append({"step": self.goal_step, "goal": goal["id"],
                                      "event": "solved", "result": summary})
        else:
            goal["failures"] += 1
            observed_blocker = self._blocker_of(goal["capability"], result)
            goal["last_blocker"] = observed_blocker
            # Alt hedefin çözmeye çalıştığı blokaj, yardımcı yeteneğin yeni
            # hatasıyla sessizce değişmez; iki durum ayrı alanlarda izlenir.
            if not goal.get("blocker"):
                goal["blocker"] = observed_blocker
            blocker = str(goal["blocker"])
            goal["reason"] = f"başarısızlık {goal['failures']}/3: {observed_blocker}"
            if not self._merge_blocked(goal) and goal["failures"] >= 3:
                goal["status"] = "abandoned"
                goal["abandoned_step"] = self.goal_step
                goal["reason"] = (f"üç hata; terk: {blocker}; "
                                  f"son hata: {observed_blocker}")
                self.goal_history.append({"step": self.goal_step, "goal": goal["id"],
                                          "event": "abandoned", "blocker": blocker,
                                          "last_blocker": observed_blocker})
                self._derive_unblock([goal], blocker)
        # Defter sınırlı; sayaçlar ve bütün hedeflerin özeti korunur.
        if len(self.goal_history) > 2048:
            del self.goal_history[:-2048]
        return dict(event, event="solved" if success else "failed",
                    goal_status=goal["status"])

    def run_autonomous(self, steps: int = 1000, experiences: dict | None = None) -> dict:
        """İnsan hedefi olmadan çalış; isteğe bağlı girdi yalnız ham gözlemdir.

        ``experiences={adım: metin}`` hedef/çıktı anahtarı değildir. Metin normal
        ``ingest`` yolundan geçer; başarı yine gerçek yetenek sonucundan ölçülür.
        """
        schedule = experiences or {}
        for step in range(max(0, int(steps))):
            incoming = schedule.get(step, ())
            if isinstance(incoming, str):
                incoming = (incoming,)
            for text in incoming:
                self.ingest(text)
            self.autonomous_step()
            # Otonom yaşam yalnız dış hedef kovalamaz: mevcut Graph/HDC
            # bilgisinden kendi karşılaştırma ve nedensel öz-testlerini de üretir.
            if self.agi is not None and (step + 1) % 20 == 0:
                try:
                    self.agi.life.pool.probe_graph(self.agi, limit=3)
                except Exception as exc:
                    # Arka-plan merakı ana hedef döngüsünü düşürmez; fakat hata
                    # da kaybolmaz. Bu kayıt resolved değildir ve sınavda yeşil
                    # sayılmaz.
                    try:
                        self.agi.life.pool.self_questions.append({
                            "kind": "graph_probe", "question": "Graph öz-sorgusu",
                            "status": "error", "answer": None, "proof": [],
                            "error": f"{type(exc).__name__}: {exc}",
                            "presentations_delta": 0, "graph_edges_delta": 0,
                        })
                    except Exception:
                        pass
        return self.goal_report()

    def goal_report(self) -> dict:
        counts = {status: sum(goal.get("status") == status for goal in self.goals)
                  for status in ("active", "solved", "abandoned", "merged")}
        return {
            "steps": self.goal_step, "idle_steps": self.goal_idle,
            "generated": len(self.goals), **counts,
            "human_goals_supplied": 0,
            "three_failure_abandons": sum(goal.get("status") == "abandoned"
                                           and goal.get("failures") == 3 for goal in self.goals),
            "capability_inventory": self._capability_inventory(),
            "goals": [{key: value for key, value in goal.items() if key != "history"}
                      for goal in self.goals],
            "history": list(self.goal_history[-80:]),
        }

    # ------------------------------------------------------------------ rapor
    def summary(self) -> dict:
        kinds = {}
        for r in self.learned:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        # DÜRÜST SAYI: aynı program beş kümede beş kez bulunsa da KEŞİF birdir.
        # Ölçü "kaç kez tetiklendi" değil, "kaç AYRI şey öğrenildi" olmalı.
        names = [r["name"] for r in self.learned]
        return {"seen": self.seen, "triggered": self.triggered,
                "distinct": len(set(names)), "rejected": self.rejected,
                "unresolved": self.unresolved, "thresh": round(self.thresh, 3),
                "min_items": int(self.min_items),
                "kinds": kinds, "names": names[-6:]}

    def state(self) -> dict:
        return {"seen": self.seen, "triggered": self.triggered,
                "rejected": self.rejected, "unresolved": self.unresolved,
                "thresh": round(self.thresh, 3), "min_items": int(self.min_items),
                "learned": self.learned[-24:],
                "autonomous": {
                    "goals": copy.deepcopy(self.goals), "history": copy.deepcopy(self.goal_history[-256:]),
                    "serial": self.goal_serial, "step": self.goal_step,
                    "idle": self.goal_idle, "capability": copy.deepcopy(self.goal_capability),
                }}

    def load_state(self, d: dict | None) -> None:
        d = d or {}
        self.seen = int(d.get("seen", 0) or 0)
        self.triggered = int(d.get("triggered", 0) or 0)
        self.rejected = int(d.get("rejected", 0) or 0)
        self.unresolved = int(d.get("unresolved", 0) or 0)
        try:
            self.thresh = float(d.get("thresh", self.MIN_SURPRISE) or self.MIN_SURPRISE)
            self.min_items = int(d.get("min_items", self.MIN_ITEMS) or self.MIN_ITEMS)
        except Exception:
            self.thresh = float(self.MIN_SURPRISE)
        self.learned = [r for r in (d.get("learned") or []) if isinstance(r, dict)]
        autonomous = d.get("autonomous") or {}
        self.goals = copy.deepcopy(autonomous.get("goals") or [])
        self.goal_history = copy.deepcopy(autonomous.get("history") or [])
        self.goal_serial = int(autonomous.get("serial", 0) or 0)
        self.goal_step = int(autonomous.get("step", 0) or 0)
        self.goal_idle = int(autonomous.get("idle", 0) or 0)
        self.goal_capability = copy.deepcopy(autonomous.get("capability") or {})
