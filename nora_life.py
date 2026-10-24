"""
NORA ARKA PLAN YAŞAMI — kullanıcı yazmasa da dönen otonom iç döngü.

Bu dosyada doğal dil cümlesi, sözcük ya da "şunu derse şunu de" kuralı YOKTUR. Nora'nın dışarıya
söylediği her şey `nora_voice.LearnedVoice` (duyduğu metinden öğrenilmiş) üzerinden çıkar; dili yoksa
söyleyecek sözü de yoktur ve bunu dürüstçe `text=None` olarak kaydeder.

Kararlar bir `if eşik` ile değil, beş nöral popülasyonun sızdıran yarışmalı akümülatör (LCA) ihalesiyle
verilir (Küresel Çalışma Alanı): dinlen | pekiştir (uyku) | zihinde prova | araştır | seslen.
Sürüş akımları sürekli değişkenlerden gelir: zihinsel gerilim, nörokimya, A24 iç-duyum, sessizlik süresi,
uyku basıncı. Her kanalın alışması (habituation) ve cevapsız kalan her seslenmenin üstel geri-çekilmesi
sonsuz döngüyü / mesaj yağmurunu biyolojik olarak engeller.

Bileşenler
  * Gap / GapLedger     : "girdiyi biliyor, çıktısını/karşılığını bilmiyor" boşlukları (Zihinsel Gerilim T).
                          dT/dt = a·pull·(1−T) − b·T  → RK4 ile ilerletilir; zihinde ileri-yürütme (simulate)
                          çözülmezse gerilimin nereye varacağını önceden hesaplar (A14 RK4 zihinsel simülasyonu).
  * ReplyRepertoire     : sosyal kalıp girdileri için ÖĞRENİLMİŞ karşılık bilgisi (iki bağımsız kaynakta
                          görülen karşılık "doğrulanmış" sayılır).
  * HypothesisPool      : internetten gelen iddialar önce geçici hipotez; ≥2 bağımsız alan adı + kesinlik
                          eşiğinde kavram ağına (Sistem 1) terfi eder (çapraz doğrulama).
  * BackgroundLife.tick : A22 CA3 anı tarama + özet, gerilim ilerletme, ihale, eylem (spontaneous_action).
Her yapı sınırlıdır: tick maliyeti sabittir (O(1)).
"""
from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from nora_lang import tr_lower
from nora_voice import LearnedVoice, state_vector, tokenize

# Düşünmenin metabolik bedeli (dünya enerjisinden/adım): bilişsel iş bedene yük bindirir, Nora'nın kendi
# açlığı arka plan yaşamını doğal olarak frenler. Kural değil, fizyoloji.
THOUGHT_COST = np.array([0.0, 0.0008, 0.0003, 0.0010, 0.0005])
CHANNELS = ("rest", "consolidate", "rehearse", "research", "reach")
R_REST, R_CONS, R_REH, R_RES, R_REACH = range(5)


def rk4_step(f: Callable[[float], float], y: float, dt: float) -> float:
    """Klasik dördüncü derece Runge–Kutta (MentalPhysicsSandbox ile aynı şema)."""
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return float(y + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0)


# ============================================================================
# 1. ZİHİNSEL GERİLİM: eksik tamamlama dürtüsü
# ============================================================================

@dataclass
class Gap:
    key: str                  # "c:<kavram>" | "s:<niyet>"
    kind: str                 # concept | social
    topic: str
    pull: float = 0.0         # boşluğun çekim gücü [0,1]
    tension: float = 0.0      # Zihinsel gerilim T [0,1]
    born: int = 0
    attempts: int = 0         # araştırma denemeleri
    rehearsals: int = 0
    voiced: int = 0
    resolved: bool = False


class GapLedger:
    MAX_GAPS = 48
    A_IN, B_OUT, C_RES = 0.020, 0.0016, 0.060     # birim: 1/adım

    def __init__(self):
        self.gaps: OrderedDict[str, Gap] = OrderedDict()

    def touch(self, key: str, kind: str, topic: str, pull: float, age: int, closed: bool = False) -> Gap:
        g = self.gaps.get(key)
        if g is None:
            if len(self.gaps) >= self.MAX_GAPS:
                victim = min(self.gaps.values(), key=lambda x: (not x.resolved, x.tension))
                self.gaps.pop(victim.key, None)
            g = self.gaps[key] = Gap(key=key, kind=kind, topic=topic, born=age)
        g.pull = float(np.clip(max(g.pull * 0.9, pull), 0.0, 1.0))
        g.resolved = closed
        return g

    def _rhs(self, g: Gap) -> Callable[[float], float]:
        pull, res = g.pull, float(g.resolved)
        return lambda T: self.A_IN * pull * (1.0 - T) - self.B_OUT * T - self.C_RES * res * T

    def step(self, dt: float = 3.0):
        for g in self.gaps.values():
            g.tension = float(np.clip(rk4_step(self._rhs(g), g.tension, dt), 0.0, 1.0))

    def simulate(self, g: Gap, horizon: int = 240, dt: float = 6.0) -> float:
        """Zihinde ileri-yürütme: bu boşluk çözülmeden `horizon` adım geçerse gerilim nereye varır?"""
        f, T = self._rhs(g), g.tension
        for _ in range(int(horizon / dt)):
            T = rk4_step(f, T, dt)
        return float(np.clip(T, 0.0, 1.0))

    def top(self, kind: str | None = None) -> Gap | None:
        c = [g for g in self.gaps.values() if not g.resolved and (kind is None or g.kind == kind)]
        return max(c, key=lambda g: g.tension) if c else None

    def max_tension(self, kind: str | None = None) -> float:
        g = self.top(kind)
        return g.tension if g else 0.0

    def state_dict(self) -> list[dict]:
        return [g.__dict__.copy() for g in self.gaps.values()]

    def load_state_dict(self, items: list[dict]):
        self.gaps.clear()
        for it in items:
            g = Gap(**{k: it[k] for k in Gap.__dataclass_fields__ if k in it})
            self.gaps[g.key] = g


# ============================================================================
# 2. ÖĞRENİLMİŞ KARŞILIK BİLGİSİ (sosyal kalıplar)
# ============================================================================

class ReplyRepertoire:
    MAX_PATTERNS = 24

    def __init__(self):
        self.patterns: dict[str, OrderedDict[str, set]] = {}   # niyet -> {karşılık metni: kaynak kümesi}
        self.exposures: Counter = Counter()

    def expose(self, intent: str, w: float = 1.0):
        self.exposures[intent] += w

    def learn(self, intent: str, reply: str, source_id: str):
        d = self.patterns.setdefault(intent, OrderedDict())
        key = " ".join(tokenize(reply))
        if not key:
            return
        d.setdefault(key, set()).add(source_id)
        if len(d) > self.MAX_PATTERNS:
            d.popitem(last=False)

    def competence(self, intent: str) -> float:
        d = self.patterns.get(intent, {})
        n_ver = sum(len(s) >= 2 for s in d.values())
        n_one = len(d) - n_ver
        return float(1.0 - np.exp(-(1.0 * n_ver + 0.3 * n_one)))

    def verified(self, intent: str) -> list[str]:
        return [k for k, s in self.patterns.get(intent, {}).items() if len(s) >= 2]

    def pull(self, intent: str) -> float:
        """Girdi sık gelip karşılık bilgisi yoksa boşluğun çekimi artar."""
        return float((1.0 - self.competence(intent)) * (1.0 - np.exp(-self.exposures[intent] / 2.0)))

    DASH_RE = re.compile(r"(?:^|\n)\s*[-–—]\s*([^\n]+)")
    QUOTE_RE = re.compile(r"[“\"']([^”\"'\n]{2,80}[?!.]?)[”\"']\s*[,.]?\s*[“\"']([^”\"'\n]{2,80})[”\"']")

    @classmethod
    def extract_exchanges(cls, text: str) -> list[tuple[str, str]]:
        """Metindeki ardışık konuşma çiftlerini (tire satırları veya iki ardışık tırnaklı söz) çıkarır."""
        lines = [m.group(1).strip() for m in cls.DASH_RE.finditer(text)]
        pairs = [(a, b) for a, b in zip(lines[::2], lines[1::2])]
        pairs += [(m.group(1).strip(), m.group(2).strip()) for m in cls.QUOTE_RE.finditer(text)]
        return pairs[:32]

    def state_dict(self) -> dict:
        return {"patterns": {i: {k: sorted(s) for k, s in d.items()} for i, d in self.patterns.items()},
                "exposures": dict(self.exposures)}

    def load_state_dict(self, d: dict):
        self.patterns = {i: OrderedDict((k, set(v)) for k, v in p.items()) for i, p in (d.get("patterns") or {}).items()}
        self.exposures = Counter({k: float(v) for k, v in (d.get("exposures") or {}).items()})


# ============================================================================
# 3. HİPOTEZ HAVUZU: çapraz doğrulama
# ============================================================================

def registrable_domain(domain: str) -> str:
    """tr.wikipedia.org ve en.wikipedia.org aynı kurumdur (bağımsız değil); tdk.gov.tr ayrı bir kurumdur."""
    labels = domain.split(".")
    keep = 3 if len(labels) >= 3 and labels[-2] in {"com", "org", "gov", "edu", "net", "co"} else 2
    return ".".join(labels[-keep:])


class HypothesisPool:
    """İnternet/okuma kaynaklı iddialar Sistem 1'e (kavram ağı) doğrudan girmez. Önce `EpistemicEngine`
    inançlarına kaynak izleriyle yazılır; ≥PROMOTE_SOURCES bağımsız alan adından destek ve
    ≥PROMOTE_CREDENCE kesinlik oluşunca kavram ağına terfi eder."""

    PROMOTE_SOURCES = 2
    PROMOTE_CREDENCE = 0.74
    MAX_PENDING = 2048

    def __init__(self):
        self.pending: OrderedDict[str, dict] = OrderedDict()
        self.by_topic: dict[str, set[str]] = {}
        self.rejected = 0
        self.promoted_total = 0

    @staticmethod
    def extract_triples(agi, sentence: str) -> list[tuple[str, str, str]]:
        wern = agi.lang.wernicke
        saved = (wern.last_topic, wern.turn_count)
        try:
            fr = wern.parse(sentence)
        finally:
            wern.last_topic, wern.turn_count = saved
        if fr.kind == "teach_fact":
            if fr.rel == "ADJ_ISA":
                adj, noun = fr.obj.split("|", 1)
                return [(fr.subj, "ISA", f"{adj} {noun}"), (f"{adj} {noun}", "ISA", noun), (fr.subj, "HAS_PROP", adj)]
            return [(fr.subj, fr.rel, fr.obj)]
        if fr.kind == "teach_val":
            return [(fr.subj, fr.rel, fr.obj)]
        return [(a.subj, a.rel, a.obj) for a in agi.auto_net.open_world_triples(tr_lower(sentence), agi.lang.graph, agi.lang.morph)]

    def ingest(self, agi, text: str, source_id: str, topic: str) -> dict:
        ep = agi.episteme
        rel_src = ep.trust.reliability(source_id)
        staged, rejected = [], []
        for sent in [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) >= 6]:
            wern = agi.lang.wernicke
            saved = (wern.last_topic, wern.turn_count)
            fr = wern.parse(sent)
            wern.last_topic, wern.turn_count = saved
            if fr.kind == "teach_val":
                arb = agi.self_model.arbitrate_claim(fr.subj, fr.valence, speaker=source_id, channel=fr.sub_kind)
                if arb.get("conflict") or arb.get("low_trust"):
                    rejected.append(sent)
                    self.rejected += 1
                    continue
            for subj, rel, obj in self.extract_triples(agi, sent):
                b = ep.store.upsert(subj, rel, obj, source_id, rel_src, raw=sent)
                if b.key not in self.pending and len(self.pending) >= self.MAX_PENDING:
                    old_key, old = self.pending.popitem(last=False)
                    self.by_topic.get(old["topic"], set()).discard(old_key)
                self.pending[b.key] = {"triple": (subj, rel, obj), "topic": topic, "raw": sent}
                self.by_topic.setdefault(topic, set()).add(b.key)
                staged.append(b.key)
        return {"staged": staged, "rejected": rejected}

    def promote_ready(self, agi) -> list[dict]:
        out = []
        for key in list(self.pending):
            b = agi.episteme.store.get(key)
            if b is None:
                self.pending.pop(key, None)
                continue
            domains = {registrable_domain(agi.episteme.trust.domain_of(s)) for s in b.sources}
            if len(domains) >= self.PROMOTE_SOURCES and b.credence >= self.PROMOTE_CREDENCE:
                rec = self.pending.pop(key)
                s, r, o = rec["triple"]
                agi.lang.graph.add_edge(s, r, o, source="cross_verified", speaker="+".join(sorted(domains))[:60],
                                        weight=float(b.credence))
                agi.episteme.system1.maybe_promote(b)
                agi.auto_net.stimulate([s, o], valence=0.25, resolved_relations=1)
                self.by_topic.get(rec["topic"], set()).discard(key)
                self.promoted_total += 1
                out.append({"triple": (s, r, o), "topic": rec["topic"], "credence": float(b.credence),
                            "domains": sorted(domains), "quote": rec["raw"]})
        return out

    def trace_for(self, agi, concept: str) -> tuple[float, int] | None:
        """Bu kavram hakkında elde yarım bir iz (hipotez) var mı? (en yüksek kesinlik, bağımsız alan sayısı)"""
        best = None
        for key in self.by_topic.get(concept, ()):
            b = agi.episteme.store.get(key)
            if b is not None:
                n = len({registrable_domain(agi.episteme.trust.domain_of(s)) for s in b.sources})
                if best is None or b.credence > best[0]:
                    best = (float(b.credence), n)
        return best

    def state_dict(self) -> dict:
        return {"pending": {k: {"triple": list(v["triple"]), "topic": v["topic"], "raw": v["raw"]}
                            for k, v in self.pending.items()}, "promoted": self.promoted_total}

    def load_state_dict(self, d: dict):
        self.pending = OrderedDict((k, {"triple": tuple(v["triple"]), "topic": v["topic"], "raw": v["raw"]})
                                   for k, v in (d.get("pending") or {}).items())
        self.by_topic = {}
        for k, v in self.pending.items():
            self.by_topic.setdefault(v["topic"], set()).add(k)
        self.promoted_total = int(d.get("promoted", 0))


# ============================================================================
# 4. KÜRESEL ÇALIŞMA ALANI İHALESİ (LCA)
# ============================================================================

class WorkspaceAuction:
    """tau·dr/dt = −r + ReLU(I − W_lat·r). Eşik yok: en güçlü sürüş yanal ketlemeyle kazanır."""

    def __init__(self, n: int = len(CHANNELS), inhibition: float = 0.55):
        self.W = inhibition * (np.ones((n, n)) - np.eye(n))

    def compete(self, drive: np.ndarray, steps: int = 20, dt: float = 0.25) -> tuple[int, np.ndarray]:
        r = np.zeros_like(drive)
        for _ in range(steps):
            r = np.clip(r + dt * (-r + np.maximum(0.0, drive - self.W @ r)), 0.0, 3.0)
        return int(np.argmax(r)), r


# ============================================================================
# 5. ARKA PLAN YAŞAMI
# ============================================================================

class BackgroundLife:
    IDLE_TAU = 900.0          # sessizlik süresi ölçeği (adım)
    SLEEP_TAU = 4000.0        # uyku basıncı ölçeği (adım)
    HAB_DECAY = 0.985

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(30_011 + int(seed))
        self.voice = LearnedVoice(seed=seed + 5)
        self.ledger = GapLedger()
        self.repertoire = ReplyRepertoire()
        self.pool = HypothesisPool()
        self.auction = WorkspaceAuction()
        self.hab = np.zeros(len(CHANNELS))
        self.outbox: deque[dict] = deque(maxlen=32)
        self.summaries: deque[dict] = deque(maxlen=32)
        self.news: deque[dict] = deque(maxlen=16)
        self.rates_log: deque[tuple[int, list[float]]] = deque(maxlen=256)
        self.sources: list[tuple[str, Callable[[str, str], str]]] = []
        self.stride = 3
        self.enabled = True
        self.unanswered = 0
        self.last_interaction_age = 0
        self.last_sleep_age = 0
        self.last_reach_age = -10_000      # son seslenmeden bu yana geçen süre (refrakter dönem)
        self.prev_oxy = 0.5
        self.t = 0
        self.tick_seconds = 0.0
        self.stats: Counter = Counter()

    # ------------------------------------------------------------ etkileşim
    def on_interaction(self, agi, raw: str, speaker: str):
        """Kullanıcı bir şey yazdığında: sessizlik sıfırlanır, söz organı duyar ve ödülle pekişir,
        sosyal kalıp girdisi karşılık bilgisiyle karşılaştırılıp boşluk gerilimi doğurur."""
        age = int(agi.mind.age)
        st = state_vector(agi, conf=agi.chem.trust_of(speaker))
        valence = float(agi.chem.oxy - self.prev_oxy) + (0.5 if self.unanswered > 0 else 0.0)
        self.voice.reinforce(valence, st)
        self.voice.hear(raw, st)
        self.prev_oxy = float(agi.chem.oxy)
        self.unanswered = 0
        self.hab[R_REACH] *= 0.25          # biri konuştu: seslenme alışkanlığı sıfırlanır
        self.last_interaction_age = age
        self._register_phatic(agi, raw, 1.0, age)

    def _register_phatic(self, agi, raw: str, w: float, age: int):
        scores = agi.plastic_discourse.infer_intents(tr_lower(raw))
        intent = max(scores, key=scores.get)
        strength = float(scores[intent]) ** 2            # sürekli ağırlık: eşik yok, belirsiz girdi zayıf iz bırakır
        self.repertoire.expose(intent, w * strength)
        self.ledger.touch(f"s:{intent}", "social", " ".join(tokenize(raw)[:3]), self.repertoire.pull(intent) * strength, age,
                          closed=self.repertoire.competence(intent) >= 0.6)

    def note_unknown_concept(self, agi, concept: str, uncertainty: float = 0.9):
        self.ledger.touch(f"c:{concept}", "concept", concept, uncertainty, int(agi.mind.age))

    # ------------------------------------------------------------ CA3 tarama
    def replay_ca3(self, agi, m: int = 3) -> dict:
        """Hipokampal CA3 anılarından yakınlık ağırlıklı örnek alır (O(m)); eksik kavramları ve cevapsız
        kalıpları gerilim defterine yazar, kısa bir durum özeti çıkarır."""
        eps = agi.broca_organ.episodic.episodes
        n = len(eps)
        topics: list[str] = []
        for _ in range(min(m, n)):
            idx = n - 1 - min(n - 1, int(self.rng.exponential(25.0)))
            ep = eps[idx]
            self._register_phatic(agi, ep.user_text, 0.0, int(agi.mind.age))   # tekrar yeni deneyim değildir: maruziyet artmaz
            for stem in ep.stems[:3]:
                node = agi.auto_net.assemblies.get(stem)
                edges = len(agi.lang.graph._out_edges.get(stem, ()))
                if node is not None and node.uncertainty > 0.6 and edges < 2:
                    self.ledger.touch(f"c:{stem}", "concept", stem, node.uncertainty, int(agi.mind.age))
                topics.append(stem)
        top = self.ledger.top()
        rec = {"age": int(agi.mind.age), "replayed": min(m, n), "topics": list(dict.fromkeys(topics))[:4],
               "top_gap": top.key if top else None, "tension": round(top.tension, 3) if top else 0.0}
        self.summaries.append(rec)
        return rec

    # ------------------------------------------------------------ internet
    def enable_internet(self, timeout: float = 3.0):
        """Gerçek ağ kaynaklarını ekler: Vikipedi özeti + Vikiveri tanımı (iki ayrı alan adı = bağımsız kaynak)."""
        def _get(url: str) -> str:
            req = urllib.request.Request(url, headers={"User-Agent": "NoraAGI-Curiosity/2.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", errors="ignore")

        def wiki(topic: str, _q: str) -> str:
            import json
            try:
                return str(json.loads(_get("https://tr.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(topic.replace(" ", "_")))).get("extract", ""))
            except Exception:
                return ""

        def wikidata(topic: str, _q: str) -> str:
            import json
            try:
                d = json.loads(_get("https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&language=tr&limit=1&search=" + urllib.parse.quote(topic)))
                hit = (d.get("search") or [{}])[0]
                return f"{hit.get('label', topic)} {hit.get('description', '')}".strip() + "." if hit.get("description") else ""
            except Exception:
                return ""

        self.sources = [("web:tr.wikipedia.org", wiki), ("web:www.wikidata.org", wikidata)]

    def _research(self, agi, g: Gap) -> dict:
        g.attempts += 1
        got = 0
        for sid, fetch in self.sources:
            text = str(fetch(g.topic, g.topic) or "")
            if not text.strip():
                continue
            got += 1
            if g.kind == "social":
                for prompt, reply in self.repertoire.extract_exchanges(text):
                    sc = agi.plastic_discourse.infer_intents(tr_lower(prompt))
                    intent = max(sc, key=sc.get)
                    self.repertoire.learn(intent, reply, sid)
            else:
                self.pool.ingest(agi, text, sid, g.topic)
        promoted = self.pool.promote_ready(agi)
        for p in promoted:
            self.news.append(p)
        if g.kind == "concept":
            g.resolved = len(agi.lang.graph._out_edges.get(g.topic, ())) >= 2 or any(p["topic"] == g.topic for p in promoted)
        else:
            intent = g.key.split(":", 1)[1]
            g.resolved = self.repertoire.competence(intent) >= 0.6
        return {"sources_reached": got, "promoted": len(promoted)}

    # ------------------------------------------------------------ sürüşler ve ihale
    def _drives(self, agi, idle: float) -> np.ndarray:
        c = agi.chem
        nv = agi.nervous
        body = max(float(nv.hunger_drive), float(nv.thirst_drive), float(nv.adelta_pain), float(nv.cfiber_ache))
        calm = 1.0 - 0.7 * body
        age = int(agi.mind.age)
        sleep_p = float(np.clip((age - self.last_sleep_age) / self.SLEEP_TAU, 0.0, 1.5)) ** 1.5
        g_any = self.ledger.top()
        t_any = g_any.tension if g_any else 0.0
        t_pred = self.ledger.simulate(g_any) if g_any else 0.0
        attempts = g_any.attempts if g_any else 0
        social_need = 1.0 - np.exp(-idle * self.IDLE_TAU / 600.0)
        news = 0.6 * float(len(self.news) > 0)
        reach_t = max(self.ledger.max_tension("social"), self.ledger.max_tension("concept") * 0.8, news,
                      0.25 * float(len(self.summaries) > 0) * float(self.stats["sleeps"] > self.stats["summaries_voiced"]))
        d = np.zeros(len(CHANNELS))
        d[R_REST] = 0.30 + 0.30 * c.sht * (1.0 - c.cort)
        d[R_CONS] = sleep_p * (0.5 + 0.5 * c.sht) * (1.0 - 0.5 * t_any)
        d[R_REH] = t_any * (0.55 + 0.45 * c.ach) * calm
        d[R_RES] = t_pred * (0.4 + 0.6 * c.da_tonic) * (1.0 - 0.6 * c.cort) * calm * (0.5 ** attempts) * float(len(self.sources) > 0)
        # Yalnızlık kendi başına bir sürüştür: içinden geçeni paylaşma isteği. Kimse yazmasa da yaşayan
        # varlık ses çıkarır; cevapsız kaldıkça üstel geri çekilir (mesaj yağmuru yok).
        loneliness = 1.15 * social_need
        # Refrakter dönem: az önce konuştuysa hemen tekrar konuşmaz (biyolojik susma süresi).
        refractory = float(np.clip((age - self.last_reach_age) / 900.0, 0.0, 1.0))
        # Cevapsız kaldıkça seslenme isteği yavaşça geri çekilir (0.72**n): iki-üç denemeden sonra susar,
        # kullanıcı yazınca sıfırlanır. Ne spam ne de ölü sessizlik.
        d[R_REACH] = ((reach_t + loneliness) * (0.25 + 0.75 * c.oxy) * calm
                      * (0.72 ** self.unanswered) * refractory)
        return np.maximum(0.0, d - 0.35 * np.tanh(self.hab)) * (1.0 + 0.04 * self.rng.standard_normal(len(d)))

    # ------------------------------------------------------------ ana döngü
    def tick(self, agi) -> dict:
        t_begin = time.perf_counter()
        self.t += 1
        age = int(agi.mind.age)
        idle = float(np.clip((age - self.last_interaction_age) / self.IDLE_TAU, 0.0, 1.0))
        self.ledger.step()
        win, rates = self.auction.compete(self._drives(agi, idle))
        self.rates_log.append((age, [round(float(x), 3) for x in rates]))
        self.stats[CHANNELS[win]] += 1
        agi.world.e = max(0.0, float(agi.world.e) - float(THOUGHT_COST[win]))   # düşünmenin metabolik bedeli
        self.hab *= self.HAB_DECAY
        self.hab[win] += float(win != R_REST)          # dinlenme alışma yaratmaz
        out: dict = {"action": CHANNELS[win], "age": age}
        if win == R_REST:
            self.replay_ca3(agi, 2)
        elif win == R_CONS:
            summ = self.replay_ca3(agi, 4)
            agi.sleep()
            self.last_sleep_age = age
            self.stats["sleeps"] += 1
            out["summary"] = summ
        elif win == R_REH:
            g = self.ledger.top()
            g.rehearsals += 1
            out["rehearsed"] = g.key
            out["predicted_tension"] = round(self.ledger.simulate(g), 3)
            if g.kind == "social":
                g.pull = max(g.pull * 0.8, self.repertoire.pull(g.key.split(":", 1)[1]))
            else:
                node = agi.auto_net.assemblies.get(g.topic)
                g.pull = float(node.uncertainty) if node is not None else g.pull
                if g.topic not in agi.auto_net.curiosity_queue and g.topic not in agi.auto_net.researched_concepts:
                    agi.auto_net.curiosity_queue.append(g.topic)
        elif win == R_RES:
            out["researched"] = self._research(agi, self.ledger.top())
        else:
            out["spontaneous_action"] = self._reach(agi, idle)
        self.tick_seconds += time.perf_counter() - t_begin
        return out

    def _reach(self, agi, idle: float) -> dict:
        """Dışarıya kendiliğinden mesaj: kimin için, neyi paylaşmak üzere olduğunu gerilimler belirler,
        sözü ise yalnızca öğrenilmiş söz organı kurar."""
        age = int(agi.mind.age)
        g_soc, g_con = self.ledger.top("social"), self.ledger.top("concept")
        cand = {"discovery": 0.9 * float(len(self.news) > 0),
                "gap_social": g_soc.tension if g_soc else 0.0,
                "gap_concept": 0.8 * (g_con.tension if g_con else 0.0),
                "summary": 0.25 * float(len(self.summaries) > 0)}
        kind = max(cand, key=cand.get)
        topic, quote, T = "", "", 0.0
        if kind == "discovery":
            item = self.news.popleft()
            topic, quote, T = item["topic"], item["quote"], 0.5
        elif kind == "gap_social":
            topic, T = g_soc.topic, g_soc.tension
            g_soc.voiced += 1
        elif kind == "gap_concept":
            topic, T = g_con.topic, g_con.tension
            g_con.voiced += 1
        else:
            s = self.summaries[-1]
            topic = (s["topics"] or [""])[0]
            self.stats["summaries_voiced"] = self.stats["sleeps"]
        # Söz, zihnin inisiyatif korteksinden çıkar (plastik tür seçimi + gerçek iç durum + Türkçe çekim).
        # Öğrenilmiş söz organı yalnızca o da olgunlaştıysa ek bir kaynak olarak denenir.
        text = ""
        move = ""
        try:
            res = agi.proactive_line(kind, topic, T, idle, self.unanswered)
            if res:
                text, move = str(res.get("text") or ""), str(res.get("move") or "")
                T = max(T, float(res.get("features", {}).get("tension", T)))
        except Exception:
            text = ""
        if not text:
            st = state_vector(agi, nov=0.5, fam=0.5, conf=0.5, tension=T, idle=idle)
            text = self.voice.speak(st, anchors=[topic] if topic else None) or ""
            move = "voice"
        if quote:
            text = f"{text} {quote}".strip() if text else quote
        if not text:
            text = "…"
        if not text or text.strip() in ("…", "..."):
            # Söyleyecek sözü yoksa susar (suskunluk da bir cevaptır); boş balon üretmez.
            self.stats["silent_reach"] += 1
            self.last_reach_age = age
            return {"age": age, "kind": kind, "text": "", "move": "silent"}
        msg = {"age": age, "kind": kind, "topic": topic, "text": text, "tension": round(T, 3),
               "idle": round(idle, 3), "move": move}
        self.outbox.append(msg)
        self.last_reach_age = age
        self.unanswered += 1
        self.hab[R_REACH] += 2.5
        self.stats["spontaneous"] += 1
        return msg

    def snapshot(self) -> dict:
        return {"t": self.t, "actions": dict(self.stats), "gaps": {k: round(g.tension, 3) for k, g in self.ledger.gaps.items()},
                "pending_hypotheses": len(self.pool.pending), "promoted": self.pool.promoted_total,
                "outbox": len(self.outbox), "unanswered": self.unanswered, "voice_maturity": round(self.voice.mature, 3)}

    def state_dict(self) -> dict:
        return {"voice": self.voice.state_dict(), "ledger": self.ledger.state_dict(), "repertoire": self.repertoire.state_dict(),
                "pool": self.pool.state_dict(), "last_sleep_age": self.last_sleep_age}

    def load_state_dict(self, d: dict):
        self.voice.load_state_dict(d.get("voice") or {})
        self.ledger.load_state_dict(d.get("ledger") or [])
        self.repertoire.load_state_dict(d.get("repertoire") or {})
        self.pool.load_state_dict(d.get("pool") or {})
        self.last_sleep_age = int(d.get("last_sleep_age", 0))
