"""
NORA ÖĞRENEN SÖYLEM ORGANI (LearnedVoice) — dilden bağımsız, sıfır sözcük.

Bu dosyada tek bir Türkçe (ya da başka dilde) sözcük, kök, ek veya cümle çerçevesi YOKTUR.
Nora'nın konuşabileceği her şey duyduğu/okuduğu metinden öğrenilir:

  * hear(metin, iç_durum): her sözcük için "hangi iç durumda duyuldu" ortalaması (12 boyutlu durum),
    sözcükten sözcüğe geçiş sayaçları (ileri + geri, sınırlı), son-ek (karakter kuyruğu) geçişleri;
  * speak(iç_durum, çapalar): çapa sözcüklerden (konu) başlayıp öğrenilmiş geçişlerle iki yana büyür;
    adaylar iç durum uyumu + akıcılık + çapa kapsama ile yarıştırılır (Gumbel-softmax, nörokimyasal sıcaklık);
  * reinforce(valence): az önce söylenen sözcüklerin durum çağrışımı ödüle göre pekişir/söner (dopamin kapılı).

Dil bilgisi ya da anlam elle verilmez; İngilizce metin okunursa İngilizce, Türkçe okunursa Türkçe konuşur.
Hiç metin duymamış bir organ sessiz kalır (babıldama bile yok): dürüstçe `None` döner.

Hafıza O(1): tüm tablolar sınırlıdır (sözcük sayısı, geçiş başına aday sayısı, cümle belleği).
"""
from __future__ import annotations

import re
from collections import Counter, deque

import numpy as np

FEATURES = ("da", "ach", "ne", "sht", "cort", "oxy", "nov", "fam", "conf", "tension", "body", "idle")
F_IDX = {f: i for i, f in enumerate(FEATURES)}
NF = len(FEATURES)

TOKEN_RE = re.compile(r"[^\W\d_]+|[,.!?…;:]", re.UNICODE)   # yalnızca harfli sözcükler + temel noktalama
END_MARKS = frozenset(".!?…")
BOS, EOS = "<s>", "</s>"


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def state_vector(agi, nov: float = 0.0, fam: float = 0.0, conf: float = 0.5,
                 tension: float = 0.0, idle: float = 0.0) -> np.ndarray:
    """İç durum: nörokimya (6) + A24 iç-duyum + bağlam sayıları; hepsi [0,1]."""
    c, nv = agi.chem, agi.nervous
    body = max(float(nv.hunger_drive), float(nv.thirst_drive), float(nv.adelta_pain), float(nv.cfiber_ache))
    v = np.zeros(NF)
    for name, val in (("da", c.da_tonic), ("ach", c.ach), ("ne", c.ne), ("sht", c.sht), ("cort", c.cort),
                      ("oxy", c.oxy), ("nov", nov), ("fam", fam), ("conf", conf), ("tension", tension),
                      ("body", body), ("idle", idle)):
        v[F_IDX[name]] = _clip01(val)
    return v


class LearnedVoice:
    MAX_VOCAB = 20000
    MAX_SUCC = 24            # her sözcük için en fazla bu kadar ardıl/öncül aday (O(1) bellek)
    MAX_SENT = 2048
    MAX_LEN = 16

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(20_011 + int(seed))
        self.count: Counter[str] = Counter()
        self.mean_state: dict[str, np.ndarray] = {}
        self.fwd: dict[str, Counter[str]] = {}
        self.bwd: dict[str, Counter[str]] = {}
        self.suf_fwd: dict[str, Counter[str]] = {}
        self.suf_bwd: dict[str, Counter[str]] = {}
        self.recent_out: deque[str] = deque(maxlen=64)
        self._last_tokens: list[str] = []
        self.heard_tokens = 0
        self._memo: dict = {}

    # ---------------------------------------------------------------- öğrenme
    @staticmethod
    def _suf(tok: str) -> str:
        return "~" + tok[-3:]

    def _bump(self, table: dict[str, Counter], key: str, nxt: str):
        c = table.get(key)
        if c is None:
            c = table[key] = Counter()
        c[nxt] += 1
        if len(c) > self.MAX_SUCC * 2:
            for k, _ in c.most_common()[self.MAX_SUCC:]:
                del c[k]

    def hear(self, text: str, state: np.ndarray):
        """Metni cümle cümle duyar: sözcük–durum çağrışımı ve geçişler güncellenir."""
        for sent in re.split(r"(?<=[.!?…])\s+|\n+", text):
            toks = tokenize(sent)
            words = [t for t in toks if t not in END_MARKS and t not in ",;:"]
            if len(words) < 3 or len(words) < 0.7 * len(re.findall(r"\S+", sent)):
                continue   # kod/tablo/işaret kalabalığı dil değildir
            seq = [BOS] + toks + ([] if toks[-1] in END_MARKS else [EOS])
            if toks[-1] in END_MARKS:
                seq.append(EOS)
            for a, b in zip(seq, seq[1:]):
                self._bump(self.fwd, a, b)
                self._bump(self.bwd, b, a)
                if a not in (BOS, EOS):
                    self._bump(self.suf_fwd, self._suf(a), b)
                if b not in (BOS, EOS):
                    self._bump(self.suf_bwd, self._suf(b), a)
            for t in toks:
                n = self.count[t]
                m = self.mean_state.get(t)
                self.mean_state[t] = state.copy() if m is None else m + (state - m) / (n + 1.0)
                self.count[t] = n + 1
                self.heard_tokens += 1
            if len(self.count) > self.MAX_VOCAB:
                for k, _ in self.count.most_common()[self.MAX_VOCAB:]:
                    self.count.pop(k, None)
                    self.mean_state.pop(k, None)
                    self.fwd.pop(k, None)
                    self.bwd.pop(k, None)

    def hear_file(self, path: str, state: np.ndarray, max_chars: int = 400_000):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            self.hear(f.read(max_chars), state)

    @property
    def mature(self) -> float:
        """Dil olgunluğu [0,1]: duyulan sözcük sayısıyla doyar. Hiç duymadıysa 0."""
        return float(1.0 - np.exp(-self.heard_tokens / 3000.0))

    # ---------------------------------------------------------------- üretim
    def _match(self, w: str, state: np.ndarray) -> float:
        """Sözcüğün öğrenilmiş durum çağrışımı ile mevcut durumun uyumu (bir konuşma süresince önbellekli)."""
        key = (w, id(state))
        v = self._memo.get(key)
        if v is None:
            m = self.mean_state.get(w)
            if m is None:
                v = -0.15
            else:
                d = m - state
                v = -float(d @ d) / NF
            self._memo[key] = v
        return v

    def _next_dist(self, table, suf_table, tok: str, state: np.ndarray, exclude: set[str]) -> tuple[list[str], np.ndarray]:
        c = table.get(tok)
        if not c and tok not in (BOS, EOS):
            c = suf_table.get(self._suf(tok))
        if not c:
            return [], np.zeros(0)
        cands = [w for w, _ in c.most_common(self.MAX_SUCC) if w not in exclude]
        if not cands:
            return [], np.zeros(0)
        logp = np.log(np.array([c[w] for w in cands], dtype=np.float64) + 1.0)
        match = np.array([self._match(w, state) for w in cands])
        return cands, logp + 6.0 * match

    def _grow(self, table, suf_table, start: str, state: np.ndarray, temp: float, used: set[str], limit: int) -> list[str]:
        out, cur = [], start
        for _ in range(limit):
            cands, sc = self._next_dist(table, suf_table, cur, state, set(out) | used)
            if not cands:
                break
            sc = sc + temp * self.rng.gumbel(size=len(sc))
            nxt = cands[int(np.argmax(sc))]
            if nxt in (BOS, EOS):
                break
            out.append(nxt)
            cur = nxt
        return out

    def speak(self, state: np.ndarray, anchors: list[str] | None = None, n_candidates: int = 4) -> str | None:
        """Çapalardan iki yana büyüyen aday cümleler üretir, en iyisini seçer. Dil yoksa None."""
        if self.mature < 0.002 or not self.count:
            return None
        self._memo.clear()
        anchors = [a for a in (tokenize(" ".join(anchors or [])) if anchors else []) if a not in END_MARKS]
        temp = 0.25 + 0.5 * float(state[F_IDX["ne"]]) + 0.2 * (1.0 - float(state[F_IDX["sht"]]))
        best, best_sc = None, -1e18
        for _ in range(n_candidates):
            if anchors:
                core = [anchors[int(self.rng.integers(len(anchors)))]]
            else:
                cands, sc = self._next_dist(self.fwd, self.suf_fwd, BOS, state, set())
                if not cands:
                    continue
                core = [cands[int(np.argmax(sc + temp * self.rng.gumbel(size=len(sc))))]]
            used = set(core)
            right = self._grow(self.fwd, self.suf_fwd, core[-1], state, temp, used, self.MAX_LEN // 2)
            left = self._grow(self.bwd, self.suf_bwd, core[0], state, temp, used | set(right), self.MAX_LEN // 2)
            sent = list(reversed(left)) + core + right
            if not sent:
                continue
            cover = sum(a in sent for a in anchors) / max(1, len(anchors))
            fit = float(np.mean([self._match(w, state) for w in sent]))
            flu = np.mean([np.log(self.count.get(w, 0) + 1.0) for w in sent]) / 6.0
            rep = len(set(sent)) / len(sent)
            score = 2.0 * cover + 4.0 * fit + flu + rep + 0.3 * self.rng.gumbel()
            if score > best_sc:
                best, best_sc = sent, score
        if best is None:
            return None
        self.recent_out.extend(best)
        self._last_tokens = list(best)
        return self._detok(best)

    @staticmethod
    def _detok(toks: list[str]) -> str:
        s = ""
        for t in toks:
            s += t if (not s or t in END_MARKS or t in ",;:)") else " " + t
        s = s.strip()
        return (s[:1].upper() + s[1:]) if s else s

    def reinforce(self, valence: float, state: np.ndarray, lr: float = 0.12):
        """İyi karşılanan söylenenlerin durum çağrışımı mevcut duruma doğru çekilir (kötüyse uzaklaşır)."""
        for t in set(self._last_tokens):
            m = self.mean_state.get(t)
            if m is not None:
                self.mean_state[t] = np.clip(m + lr * float(valence) * (state - m), 0.0, 1.0)
        self._last_tokens = []

    # ---------------------------------------------------------------- kalıcılık
    def state_dict(self) -> dict:
        top = [w for w, _ in self.count.most_common(4000)]
        return {
            "heard": self.heard_tokens,
            "count": {w: self.count[w] for w in top},
            "mean": {w: self.mean_state[w].round(4).tolist() for w in top if w in self.mean_state},
            "fwd": {w: dict(self.fwd[w]) for w in top if w in self.fwd},
            "bwd": {w: dict(self.bwd[w]) for w in top if w in self.bwd},
        }

    def load_state_dict(self, d: dict):
        self.heard_tokens = int(d.get("heard", 0))
        self.count = Counter({k: int(v) for k, v in (d.get("count") or {}).items()})
        self.mean_state = {k: np.asarray(v, dtype=np.float64) for k, v in (d.get("mean") or {}).items()}
        self.fwd = {k: Counter(v) for k, v in (d.get("fwd") or {}).items()}
        self.bwd = {k: Counter(v) for k, v in (d.get("bwd") or {}).items()}
        self.suf_fwd, self.suf_bwd = {}, {}
        for table, suf in ((self.fwd, self.suf_fwd), (self.bwd, self.suf_bwd)):
            for k, c in table.items():
                if k not in (BOS, EOS):
                    for w, n in c.items():
                        suf.setdefault(self._suf(k), Counter()).update({w: n})
