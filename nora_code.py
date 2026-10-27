#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_code.py — Nora'nın Çalıştırarak Doğrulayan, Plastik Olarak Öğrenen Kod Sentez Korteksi.

Tasarım ilkesi
--------------
Nora bir dil modeli gibi "kulağa doğru gelen" kod UYDURMAZ. Bir programı ancak ve ancak
  (1) isteği köklerden çözümleyip bir niyet vektörüne indirgediyse,
  (2) bu niyeti, plastik sinaptik ağırlıklarla seçilmiş bir OPERATÖR + gömülü metin üzerinden
      bir AST'ye dönüştürdüyse,
  (3) ve ürettiği programı gerçekten ÇALIŞTIRIP çıktısını istenen metinle karşılaştırdıysa
söyler. Doğrulama geçmezse cevap "bilmiyorum / şu an üretemiyorum" olur. Bu, halüsinasyonu
mimari düzeyde imkânsız kılar: doğrulanmamış hiçbir satır dışarı çıkmaz.

Öğrenme (if/else kural tablosu değil):
  * Her Türkçe ipucu kökü ("yazdır", "ekrana", "bastır"...) ile her operatör (DISPLAY, CONCAT,
    ASSIGN, LOOP) arasındaki bağ bir sinaptik ağırlıktır. Operatör seçimi bu ağırlıkların
    yumuşak maksimumudur (softmax); başarılı çalıştırma ağırlığı güçlendirir, başarısızlık zayıflatır.
  * Çalıştığı kanıtlanmış bir niyet, beceri belleğine (System-1) yazılır: bir dahaki sefere
    hiç arama yapmadan, anında aynı program üretilir.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nora_broca import NP, VerbSpec, TurkishSyntaxEngine
from nora_lang import tr_capitalize, tr_lower
from nora_semantic_code import CompositionalCodeLearner, SemanticGoal, ProgramPlan


# ============================================================================
# 1. PLASTİK OPERATÖR AĞI
# ============================================================================

@dataclass
class Operator:
    name: str
    arity: int
    cues: tuple[str, ...]          # bu operatörü çağıran Türkçe kök ipuçları
    py_name: str


class PlasticOperatorCortex:
    """Türkçe kök ipucu -> Python operatörü sinaptik çağrışım matrisi.

    Başlangıç ağırlıkları dilsel önseldir (bir insanın 'yazdır'ı bilmesi gibi); sonrasında
    tamamen çalıştırma geri bildirimiyle değişir. Bu bir kural tablosu değil, öğrenilen bir
    ilişkilendirme matrisidir: yeni bir ipucu eklenirse ağırlığı 0'dan başlar ve denemeyle öğrenir."""

    OPERATORS: tuple[Operator, ...] = (
        Operator("DISPLAY", 1, ("yazdır", "yazdir", "yaz", "ekrana", "bastır", "bas", "print", "çıktı", "göster",
                                "satır", "python", "kod", "program", "betik"), "print"),
        Operator("CONCAT", 2, ("birleştir", "ekle", "yan yana", "concat", "topla"), "concat"),
        Operator("ASSIGN", 2, ("ata", "değişken", "tut", "sakla"), "assign"),
        Operator("LOOP", 2, ("tekrarla", "döngü", "defa", "kez", "loop"), "loop"),
    )

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.W: dict[str, dict[str, float]] = {op.name: {} for op in self.OPERATORS}
        self.op_by_name = {op.name: op for op in self.OPERATORS}
        self.learning_events = 0
        # Dilsel önsel: ipucu -> kendi operatörü (küçük gürültüyle, böylece tek tohum kader değil)
        for op in self.OPERATORS:
            for cue in op.cues:
                self.W[op.name][cue] = 0.85 + 0.05 * float(self.rng.standard_normal())
        for other in self.OPERATORS:
            if other.name == "DISPLAY":
                continue
            for cue in other.cues:
                for opn in self.W:
                    self.W[opn].setdefault(cue, 0.0)

    def presence_features(self, text: str) -> dict[str, float]:
        s = tr_lower(text)
        words = re.findall(r"[\wçğıöşü]+", s)
        feats: dict[str, float] = {}
        for cue in self.W["DISPLAY"].keys():
            if cue in s:
                # Türkçe kök + ek: 'yazdıran', 'yazdıracak', 'yazdırır' hepsi 'yazdır' kökünü içerir
                feats[cue] = 1.0
            else:
                for w in words:
                    if w.startswith(cue) and len(cue) >= 3:
                        feats[cue] = 0.85
                        break
        return feats

    def scores(self, text: str) -> dict[str, float]:
        feats = self.presence_features(text)
        out = {}
        for opn, wmap in self.W.items():
            out[opn] = float(sum(wmap.get(c, 0.0) * v for c, v in feats.items()))
        return out

    def select(self, text: str, threshold: float = 0.30) -> tuple[str | None, dict]:
        sc = self.scores(text)
        if not sc:
            return None, {}
        names = list(sc.keys())
        vals = np.array([sc[n] for n in names], dtype=np.float64)
        if float(np.max(vals)) < threshold:
            return None, {n: round(float(v), 3) for n, v in sc.items()}
        z = np.exp((vals - np.max(vals)) * 3.0)
        probs = z / max(1e-9, float(z.sum()))
        best = names[int(np.argmax(probs))]
        return best, {n: round(float(p), 3) for n, p in zip(names, probs)}

    def reinforce(self, text: str, op_name: str, good: bool, lr: float = 0.12):
        """Çalıştırma sonucuyla sinaptik ağırlığı güncelle (üç-faktörlü öğrenme: ipucu x operatör x ödül)."""
        feats = self.presence_features(text)
        if not feats or op_name not in self.W:
            return
        sign = 1.0 if good else -1.0
        for cue, presence in feats.items():
            cur = self.W[op_name].get(cue, 0.0)
            self.W[op_name][cue] = float(np.clip(cur + sign * lr * presence, -1.0, 2.0))
        self.learning_events += 1


# ============================================================================
# 2. İSTEK ÇÖZÜMLEME (gömülü metin + tek satır kısıtı)
# ============================================================================

@dataclass
class CodeIntent:
    verb: str                     # seçilen operatör
    payload: str = ""
    single_line: bool = True
    language: str = "python"
    raw: str = ""
    scores: dict = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)


class RequestGrounder:
    """Türkçe isteği (operatör, gömülü metin, kısıtlar) üçlüsüne indirger."""

    QUOTE_PATTERNS = (
        r'"([^"]+)"',
        r"“([^”]+)”",
        r"«([^»]+)»",
        r"‘([^’]+)’",
        r"`([^`]+)`",
        # Düz tek tırnak: Türkçe iyelik kesmesiyle ('Python'da) karışmasın diye
        # yalnızca sözcük sınırındaki tırnaklar kabul edilir.
        r"(?<![\wçğıöşü])'([^']{2,})'(?![\wçğıöşü])",
    )

    CUE_ROOTS = ("yazdır", "yazdir", "ekrana", "bastır", "print", "çıktı ver", "göster")
    # Hitap/kanal sözcükleri: gömülü metin DEĞİLDİR, yalnızca nereye yazılacağını söyler
    ADDRESS_ROOTS = ("ekrana", "ekran", "konsola", "konsol", "terminale", "terminal", "çıktı olarak", "alt alta")

    def __init__(self, cortex: PlasticOperatorCortex):
        self.cortex = cortex

    def ground(self, text: str) -> CodeIntent:
        raw = text.strip()
        s = tr_lower(raw)

        payload = ""
        for pat in self.QUOTE_PATTERNS:
            m = re.search(pat, raw)
            if m and m.group(1).strip():
                payload = m.group(1)
                break

        if not payload:
            # Tırnaksız istek: 'Merhaba! Ey Ahali! yazdıran python satırı' / 'ekrana Merhaba! yaz'
            def _cut_payload(span: str) -> str:
                cut = len(span)
                low_span = tr_lower(span)
                for cue in self.CUE_ROOTS + ("yaz", "bas", "yazdır"):
                    j = low_span.find(cue)
                    if j > 0:
                        cut = min(cut, j)
                cand = span[:cut].strip(" ,.:;-\n\t")
                cand = re.sub(r"^(?:bana|lütfen|acaba|hemen|şimdi)\s+", "", cand, flags=re.IGNORECASE)
                if any(k in tr_lower(cand) for k in ("python", "satır", "kod", "program", "betik")):
                    return ""
                return cand

            # (a) Kanal adı ('ekrana', 'konsola') verilmişse gömülü metin ONDAN SONRA gelir
            addr_hits = [(s.rfind(a), a) for a in self.ADDRESS_ROOTS if s.rfind(a) >= 0]
            if addr_hits:
                idx, addr = max(addr_hits)
                cand = _cut_payload(raw[idx + len(addr):])
                if cand:
                    payload = cand
            # (b) Klasik sıra: 'METİN yazdıran python satırı'
            if not payload:
                payload = _cut_payload(raw)

        verb, probs = self.cortex.select(raw)
        single = bool(re.search(r"\b(tek|bir)\s+satır", s)) or ("satır" in s)
        intent = CodeIntent(verb=verb or "", payload=payload, single_line=single, raw=raw, scores=probs)
        intent.trace.append(f"ipuçları={list(self.cortex.presence_features(raw).keys())}")
        intent.trace.append(f"operatör olasılıkları={probs}")
        intent.trace.append(f"gömülü metin={payload!r}")
        return intent


# ============================================================================
# 3. AST + YÜZEY GERÇEKLEŞTİRME + ÇALIŞTIRARAK DOĞRULAMA
# ============================================================================

class PythonEmitter:
    """Soyut sözdizimi ağacını Python metnine çevirir (karar vermez, yalnızca biçimlendirir)."""

    @staticmethod
    def quote(text: str, style: str = '"') -> str:
        if style == '"':
            return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
        return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"

    @classmethod
    def display(cls, text: str, style: str = '"', variant: int = 0) -> str:
        lit = cls.quote(text, style)
        if variant == 0:
            return f"print({lit})"
        if variant == 1:
            return f"import sys; sys.stdout.write({lit} + chr(10))"
        if variant == 2:
            # f-string yolu (kaçış davranışını farklı bir kod yolundan sınar)
            safe = text.replace("{", "{{").replace("}", "}}")
            return "print(f" + cls.quote(safe, style) + ")"
        return f"print({lit})"


class ExecutionVerifier:
    """Programı ayrı bir süreçte çalıştırır; çıktıyı beklenen metinle karşılaştırır."""

    def __init__(self, timeout: float = 6.0):
        self.timeout = float(timeout)
        self.runs = 0
        self.successes = 0
        self.last_stdout = ""

    def run(self, program: str) -> dict:
        self.runs += 1
        t0 = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory() as td:
                p = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True,
                                   timeout=self.timeout, cwd=td)
            out = p.stdout
            self.last_stdout = out
            return {"ok": p.returncode == 0, "stdout": out, "stderr": p.stderr[-300:],
                    "ms": round((time.perf_counter() - t0) * 1000.0, 2), "code": p.returncode}
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "stderr": "zaman_aşımı", "ms": self.timeout * 1000, "code": -1}
        except Exception as e:
            return {"ok": False, "stdout": "", "stderr": type(e).__name__, "ms": 0.0, "code": -2}

    def verify(self, program: str, expected: str) -> dict:
        res = self.run(program)
        out = res["stdout"]
        matched = res["ok"] and out.strip() == expected.strip() and (expected.strip() in out)
        if matched:
            self.successes += 1
        res["expected"] = expected
        res["matched"] = bool(matched)
        return res


# ============================================================================
# 4. KOD SENTEZ KORTEKSİ (cephe)
# ============================================================================

class CodeSynthesisCortex:
    """Nora'nın kod üretme organı: niyet -> aday programlar -> ÇALIŞTIRARAK DOĞRULAMA -> beyan.
    Doğrulanmamış program asla dışarı verilmez; bilinmeyen istekte dürüstçe 'yapamam' der."""

    MAX_CANDIDATES = 6
    # Her ipucu kökünün kanıt gücü: "çıktı" tek başına kod istemi sayılmaz (Türkçede 'ortaya çıktı'
    # anlamına da gelir); "yazdır/print/kod" ise tek başına yeter. Bu bir kural tablosu değil,
    # duyusal özellik ağırlığıdır — toplam kanıt eşiği geçilirse korteks ateşlenir.
    CUE_STRENGTH = {
        "yazdır": 1.0, "yazdir": 1.0, "print": 1.0, "kod": 0.9, "python": 1.0, "program": 0.8,
        "satır": 0.7, "bastır": 0.9, "ekrana": 0.6, "göster": 0.5, "çıktı ver": 0.6,
    }
    REQUEST_THRESHOLD = 1.0

    def requested_evidence(self, raw: str) -> float:
        feats = self.cortex.presence_features(raw)
        return float(sum(self.CUE_STRENGTH.get(c, 0.5) * v for c, v in feats.items()))

    def is_requested(self, raw: str) -> bool:
        if self.requested_evidence(raw) < self.REQUEST_THRESHOLD:
            return False
        op, probs = self.cortex.select(raw)
        return op == "DISPLAY" and float(probs.get("DISPLAY", 0.0)) >= 0.45

    def __init__(self, seed: int = 0, pretrain: bool = True):
        self.cortex = PlasticOperatorCortex(seed=seed)
        self.grounder = RequestGrounder(self.cortex)
        self.verifier = ExecutionVerifier()
        self.syn = TurkishSyntaxEngine
        self.skills: dict[str, dict] = {}      # niyet imzası -> doğrulanmış program (System-1)
        self.feedback_log: list[dict] = []
        self.min_score = 0.30
        self.semantic = CompositionalCodeLearner()
        if pretrain:
            primer = Path(__file__).resolve().parent / "data" / "python_semantics_primer.md"
            try:
                self.semantic.load_primer(primer)
            except (OSError, UnicodeError):
                pass

    # --------------------------------------------------------------- yardımcılar
    @staticmethod
    def _signature(intent: CodeIntent) -> str:
        return f"{intent.verb}|{'tek' if intent.single_line else 'cok'}|{intent.payload}"

    def _candidates(self, intent: CodeIntent) -> list[str]:
        if intent.verb != "DISPLAY" or not intent.payload:
            return []
        styles = ['"', "'"]
        variants = [0, 1] if intent.single_line else [0, 1, 2]
        out: list[str] = []
        for style in styles:
            for v in variants:
                prog = PythonEmitter.display(intent.payload, style=style, variant=v)
                if intent.single_line and "\n" in prog:
                    continue
                if prog not in out:
                    out.append(prog)
        return out[: self.MAX_CANDIDATES]

    def _clarify(self, missing: str) -> str:
        syn = self.syn
        if missing == "payload":
            c1 = syn.realize_verb(VerbSpec(
                verb_root="söyle", obj=syn.nominalize_dik("yazdır", poss=2, case="ACC"),
                neg=True, tense="PAST", person="2SG",
            ))
            c2 = syn.realize_verb(VerbSpec(verb_root="söyle", obj=NP("metin", case="ACC"), tense="IMP", person="2SG"))
            c3 = syn.realize_verb(VerbSpec(
                verb_root="ver", obj=NP("satır", case="ACC"),
                advs=("ancak çalıştırıp doğruladıktan sonra",), tense="AOR", person="1SG",
            ))
            return f"{tr_capitalize(c1)}. {tr_capitalize(c2)}; {c3}."
        c1 = syn.realize_verb(VerbSpec(verb_root="çözümle", obj=NP("istek", poss=2, case="ACC"), tense="PAST", person="1SG"))
        c2 = syn.realize_verb(VerbSpec(verb_root="anla", advs=("hangi işlemi istediğini",),
                                       neg=True, tense="PAST", person="1SG"))
        return f"{tr_capitalize(c1)}, ama {c2}. Ne yapmamı istediğini tek cümleyle söyle; uydurmak yerine soruyorum."

    # --------------------------------------------------------------- anlamsal bileşim
    def _handle_semantic(self, goal: SemanticGoal, request: str, started: float) -> dict:
        if goal.layout != "vertical":
            solved = {"ok": False, "reason": "layout_gap", "missing": [goal.layout],
                      "trace": list(goal.trace)}
        else:
            solved = self.semantic.solve(goal)
        if isinstance(solved, dict):
            missing = [str(value) for value in solved.get("missing") or ()]
            c_plan = self.syn.realize_verb(VerbSpec(
                verb_root="kur", obj=NP("plan", case="ACC"), neg=True,
                tense="PAST", person="1SG",
            ))
            return {"ok": False, "path": "semantic_gap", "program": "", "verified": False,
                    "confidence": 0.0, "reason": solved.get("reason", "semantic_gap"),
                    "missing": missing, "reply_hint": tr_capitalize(c_plan) + ": " + ", ".join(missing),
                    "ms": round((time.perf_counter() - started) * 1000, 2),
                    "trace": solved.get("trace") or list(goal.trace), "intent": goal}
        plan: ProgramPlan = solved
        ver = self.verifier.run(plan.program)
        matched = bool(ver.get("ok") and ver.get("stdout") == plan.expected)
        trace = list(plan.trace) + [f"exact_output={matched}"]
        if not matched:
            return {"ok": False, "path": "unverified", "program": "", "verified": False,
                    "confidence": 0.0, "reason": "doğrulanamadı", "reply_hint": self._clarify("intent"),
                    "ms": round((time.perf_counter() - started) * 1000, 2),
                    "trace": trace + [f"stderr={ver.get('stderr', '')}"], "intent": goal}
        self.verifier.successes += 1
        self.semantic.adopt(goal, plan)
        self.cortex.reinforce(request, "DISPLAY", good=True)
        if "iterate" in plan.operators:
            self.cortex.reinforce(request, "LOOP", good=True)
        self.feedback_log.append({"sig": "semantic:" + "|".join(goal.key),
                                  "program": plan.program, "ok": True})
        return {"ok": True, "path": "composed", "program": plan.program,
                "stdout": ver["stdout"], "expected": plan.expected, "verified": True,
                "confidence": 1.0, "operators": list(plan.operators), "run_ms": ver["ms"],
                "ms": round((time.perf_counter() - started) * 1000, 2),
                "trace": trace, "intent": goal}

    @staticmethod
    def is_document(request: str) -> bool:
        low = tr_lower(request or "")
        has_url = bool(re.search(r"https?://", low))
        return (bool(re.match(r"^\s*(?:doküman|belge|document)\s*:", low))
                or ("```" in low and any(cue in low for cue in ("öğren", "oku", "belge")))
                or (has_url and any(cue in low for cue in ("öğren", "oku", "belge", "doküman"))))

    def handle_document(self, request: str) -> dict:
        url = re.search(r"https?://[^\s<>]+", request)
        if url:
            result = self.semantic.learn_url(url.group(0).rstrip(".,;"))
        else:
            body = re.sub(r"^\s*(?:doküman|belge|document)\s*:\s*", "", request,
                          count=1, flags=re.IGNORECASE)
            result = self.semantic.learn_document(body, source="conversation")
        result.update({"path": "document", "program": "", "verified": False})
        if not result.get("ok"):
            c_read = self.syn.realize_verb(VerbSpec(
                verb_root="oku", obj=NP("belge", case="ACC"), neg=True,
                tense="PAST", person="1SG",
            ))
            result["reply_hint"] = tr_capitalize(c_read) + ": " + str(result.get("error", "0"))
        return result

    # --------------------------------------------------------------- ana giriş
    def handle(self, request: str) -> dict:
        t0 = time.perf_counter()
        goal = self.semantic.request(request)
        if goal is not None:
            return self._handle_semantic(goal, request, t0)
        intent = self.grounder.ground(request)
        trace = list(intent.trace)

        # 1) System-1: aynı niyet daha önce çalıştırılarak doğrulandıysa anında dön
        sig = self._signature(intent)
        cached = self.skills.get(sig)
        if cached is not None:
            ver = self.verifier.verify(cached["program"], intent.payload)
            if ver["matched"]:
                cached["hits"] = cached.get("hits", 0) + 1
                return {
                    "ok": True, "path": "system1", "program": cached["program"],
                    "stdout": ver["stdout"], "confidence": 1.0, "verified": True,
                    "ms": round((time.perf_counter() - t0) * 1000, 2), "trace": trace + ["System-1 beceri belleğinden"],
                    "intent": intent,
                }

        # 2) Niyet belirsizse: uydurma, açıkça söyle
        if intent.verb != "DISPLAY":
            self.cortex.reinforce(request, intent.verb or "DISPLAY", good=False, lr=0.02)
            return {
                "ok": False, "path": "abstain", "program": "", "verified": False, "confidence": 0.0,
                "reason": "belirsiz_niyet", "reply_hint": self._clarify("intent"),
                "ms": round((time.perf_counter() - t0) * 1000, 2), "trace": trace, "intent": intent,
            }
        if not intent.payload.strip():
            return {
                "ok": False, "path": "abstain", "program": "", "verified": False, "confidence": 0.0,
                "reason": "metin_yok", "reply_hint": self._clarify("payload"),
                "ms": round((time.perf_counter() - t0) * 1000, 2), "trace": trace, "intent": intent,
            }

        # 3) Adayları üret ve ÇALIŞTIRARAK doğrula
        candidates = self._candidates(intent)
        for prog in candidates:
            ver = self.verifier.verify(prog, intent.payload)
            trace.append(f"aday={prog} -> çıktı={ver['stdout']!r} eşleşti={ver['matched']}")
            if ver["matched"]:
                self.cortex.reinforce(request, "DISPLAY", good=True)
                self.skills[sig] = {"program": prog, "payload": intent.payload, "hits": 1,
                                    "verified_at": time.time()}
                self.feedback_log.append({"sig": sig, "program": prog, "ok": True})
                return {
                    "ok": True, "path": "verified", "program": prog, "stdout": ver["stdout"],
                    "confidence": 1.0, "verified": True, "run_ms": ver["ms"],
                    "ms": round((time.perf_counter() - t0) * 1000, 2), "trace": trace, "intent": intent,
                }

        # 4) Hiçbir aday doğrulanmadıysa: başarısızlığı gizleme, öğren ve dürüst ol
        self.cortex.reinforce(request, "DISPLAY", good=False)
        self.feedback_log.append({"sig": sig, "program": candidates[0] if candidates else "", "ok": False})
        return {
            "ok": False, "path": "unverified", "program": "", "verified": False, "confidence": 0.0,
            "reason": "doğrulanamadı", "reply_hint": self._clarify("intent"),
            "ms": round((time.perf_counter() - t0) * 1000, 2), "trace": trace, "intent": intent,
        }

    # --------------------------------------------------------------- beyan
    def speak(self, res: dict) -> str:
        """Doğrulanmış programı, kendi kök çekimiyle kurulmuş Türkçe bir beyana dönüştürür."""
        syn = self.syn
        if not res.get("ok"):
            return str(res.get("reply_hint") or "")
        if res.get("path") == "document":
            learned = list(res.get("new_operators") or ()) + list(res.get("new_facts") or ())
            if not learned:
                learned = list(res.get("operators") or ()) + list((res.get("facts") or {}).keys())
            c_read = syn.realize_verb(VerbSpec(verb_root="oku", obj=NP("belge", case="ACC"),
                                               tense="PAST", person="1SG"))
            return tr_capitalize(c_read) + ": " + (", ".join(learned) or "0") + "."
        if res.get("path") == "composed":
            prog = str(res.get("program") or "")
            out = str(res.get("stdout") or "").rstrip("\n")
            c_write = syn.realize_verb(VerbSpec(verb_root="yaz", obj=NP("program", case="ACC"),
                                                tense="PAST", person="1SG"))
            c_run = syn.realize_verb(VerbSpec(verb_root="çalıştır", obj="onu",
                                              tense="PAST", person="1SG"))
            head = tr_capitalize(c_write) + "."
            return (head + "\n```python\n" + prog + "\n```\n"
                    + tr_capitalize(c_run) + "; çıktı:\n```text\n" + out + "\n```")
        prog = res.get("program", "")
        out = str(res.get("stdout", "")).strip()
        c_run = syn.realize_verb(VerbSpec(verb_root="çalıştır", obj="onu", advs=("hemen",),
                                          tense="PAST", person="1SG"))
        c_chk = syn.realize_verb(VerbSpec(verb_root="karşılaştır", obj=NP("çıktı", poss=3, case="ACC"),
                                          advs=("istediğin metinle",), tense="PAST", person="1SG"))
        if res.get("path") == "system1":
            c1 = syn.realize_verb(VerbSpec(verb_root="hatırla", obj="bunu", advs=("düşünmeden",),
                                           tense="PRES", person="1SG"))
            head = f"{tr_capitalize(c1)}; bu satırı daha önce çalıştırıp doğrulamıştım."
        else:
            c1 = syn.realize_verb(VerbSpec(verb_root="yaz", obj=NP("satır", case="ACC"),
                                           advs=("istediğin gibi",), tense="PAST", person="1SG"))
            head = f"{tr_capitalize(c1)}."
        verify = f"{tr_capitalize(c_run)} ve {c_chk}; çıktısı birebir şu: {out}"
        stop = "" if out.endswith(("!", "?", ".")) else "."
        return f"{head} `{prog}` — {verify}{stop} Doğrulanmamış hiçbir kodu sana vermem."

    def snapshot(self) -> dict:
        return {
            "skills": len(self.skills), "learning_events": self.cortex.learning_events,
            "verified_runs": self.verifier.runs, "successes": self.verifier.successes,
            "semantic_operators": len(self.semantic.operators),
            "composite_operators": len(self.semantic.composites),
            "weights": {op: {k: round(v, 2) for k, v in list(w.items())[:5]} for op, w in self.cortex.W.items()},
        }

    def state_dict(self) -> dict:
        return {"W": {k: dict(v) for k, v in self.cortex.W.items()},
                "skills": {k: {"program": v["program"], "payload": v["payload"], "hits": v.get("hits", 1)}
                           for k, v in list(self.skills.items())[-400:]},
                "events": self.cortex.learning_events,
                "semantic": self.semantic.state_dict()}

    def load_state_dict(self, d: dict):
        for opn, wmap in (d.get("W") or {}).items():
            if opn in self.cortex.W:
                self.cortex.W[opn].update({k: float(v) for k, v in wmap.items()})
        self.skills = {k: dict(v, payload=v.get("payload", "")) for k, v in (d.get("skills") or {}).items()}
        self.cortex.learning_events = int(d.get("events", 0))
        if d.get("semantic") is not None:
            self.semantic.load_state_dict(d.get("semantic") or {})
