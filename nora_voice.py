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
import gzip
import os
import pickle
import heapq
import json
import random
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


def _seeded_bpe_word(word, merge_ranks, dropout, rng):
    """BPE merge kuyruğu; dropout kararları yalnız verilen RNG'den gelir."""
    values = list(word)
    count = len(values)
    if count < 2:
        return values
    previous = list(range(-1, count - 1))
    following = list(range(1, count + 1))
    following[-1] = -1
    alive = [True] * count
    heap = []
    order = 0

    def offer(left):
        nonlocal order
        if left < 0:
            return
        right = following[left]
        if right < 0:
            return
        rank = merge_ranks.get((values[left], values[right]))
        if rank is not None:
            heapq.heappush(heap, (rank, order, left, right,
                                  values[left] + values[right]))
            order += 1

    for index in range(count - 1):
        offer(index)
    skipped = []
    while heap:
        entry = heapq.heappop(heap)
        # Rust tokenizers önce dropout kararını verir; atlanan adayları
        # sonraki dropout'suz adayda kuyruğa geri koyar. Geçersiz/eski
        # kayıtlar da bu RNG kararını tüketir.
        if dropout and rng.random() < dropout:
            skipped.append(entry)
            continue
        if skipped:
            for skipped_entry in skipped:
                heapq.heappush(heap, skipped_entry)
            skipped.clear()
        _, _, left, right, merged = entry
        if (not alive[left] or not alive[right] or following[left] != right
                or values[left] + values[right] != merged):
            continue
        values[left] = merged
        alive[right] = False
        new_right = following[right]
        following[left] = new_right
        if new_right >= 0:
            previous[new_right] = left
        offer(previous[left])
        offer(left)
    result = []
    index = 0
    while index >= 0:
        if alive[index]:
            result.append(values[index])
        index = following[index]
    return result


class SubwordTokenizer:
    """Veriden öğrenilen BPE alt-sözcük ayrıştırıcı (dilden bağımsız; sözlük metinden çıkar).
    Ek alma/çekim bolluğu olan dillerde (ör. Türkçe) hiç görülmemiş sözcükleri de parçalayabilmek için."""

    def __init__(self):
        self.tk = None

    def _new(self):
        from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders
        t = Tokenizer(models.BPE(unk_token="<unk>"))
        t.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase(), normalizers.Replace("\u0307", "")])
        t.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Metaspace(), pre_tokenizers.Punctuation()])
        t.decoder = decoders.Metaspace()
        return t

    def train(self, texts, vocab_size: int = 16000):
        from tokenizers import trainers
        t = self._new()
        t.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>"], min_frequency=3))
        self.tk = t

    def encode(self, text: str) -> list[str]:
        return self.tk.encode(text).tokens

    def decode(self, pieces: list[str]) -> str:
        return self.tk.decode_tokens(pieces) if hasattr(self.tk, "decode_tokens") else \
            "".join(pieces).replace("\u2581", " ").strip()

    def _seeded_bpe_spec(self):
        cached = getattr(self, "_seeded_bpe_spec_cache", None)
        if cached is not None:
            return cached
        model = json.loads(self.to_str())["model"]
        if model.get("type") != "BPE" or model.get("byte_fallback"):
            raise ValueError("tohumlu subword dropout yalnız BPE/byte_fallback=False destekler")
        if model.get("continuing_subword_prefix") or model.get("end_of_word_suffix"):
            raise ValueError("tohumlu subword dropout için affix'siz BPE gerekli")
        ranks = {}
        if not model.get("ignore_merges"):
            for rank, merge in enumerate(model["merges"]):
                left, right = merge.split(" ", 1) if isinstance(merge, str) else merge
                ranks[(left, right)] = rank
        vocab = model["vocab"]
        unknown = vocab[model["unk_token"]]
        fuse_unknown = bool(model.get("fuse_unk"))
        cached = (vocab, ranks, unknown, fuse_unknown)
        self._seeded_bpe_spec_cache = cached
        return cached

    def encode_batch_bpe_dropout(self, texts, dropout: float, seed: int):
        """Aynı BPE sözlüğüyle (ids, tokens); sonuç yalnız metin/tohum/p'ye bağlıdır."""
        dropout = float(dropout)
        if not 0.0 <= dropout < 1.0:
            raise ValueError("BPE dropout [0,1) aralığında olmalı")
        vocab, ranks, unknown, fuse_unknown = self._seeded_bpe_spec()
        rng = random.Random(int(seed))
        encoded = []
        for text in texts:
            normalized = self.tk.normalizer.normalize_str(text)
            pieces = self.tk.pre_tokenizer.pre_tokenize_str(normalized)
            ids, tokens = [], []
            for piece, _ in pieces:
                current = _seeded_bpe_word(piece, ranks, dropout, rng)
                for token in current:
                    token_id = vocab.get(token, unknown)
                    if fuse_unknown and token_id == unknown and ids and ids[-1] == unknown:
                        continue
                    tokens.append(token)
                    ids.append(token_id)
            encoded.append((ids, tokens))
        return encoded

    def to_str(self) -> str:
        return self.tk.to_str() if self.tk is not None else ""

    @classmethod
    def from_str(cls, s: str):
        from tokenizers import Tokenizer
        o = cls()
        o.tk = Tokenizer.from_str(s) if s else None
        return o


def tokenize(text: str) -> list[str]:
    # casefold büyük İ'yi 'i'+birleşik nokta yapar; noktayı atarak 'i' kalır (dilden bağımsız Unicode düzeltmesi)
    return TOKEN_RE.findall(text.casefold().replace("\u0307", ""))


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
    """Sözcük dizilerinin yüksek-dereceli (4-gram'a kadar) geçiş istatistiği + sözcük–iç durum çağrışımı.
    Her şey duyulan metinden öğrenilir; sınırlı bellek (bağlam tabloları budanır)."""

    MAX_VOCAB = 40000
    MAX_SUCC = 40            # bir bağlam için en fazla bu kadar ardıl aday
    MAX_CTX = 1_500_000       # derece başına en fazla bağlam (aşılırsa tek görülenler budanır)
    ORDER = 4                # ileri bağlam uzunluğu (parça sayısı)
    BWD_ORDER = 2            # geri bağlam (yalnızca çapanın soluna büyütmek için)
    MAX_LEN = 24

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(20_011 + int(seed))
        self.count: Counter[str] = Counter()
        self.mean_state: dict[str, np.ndarray] = {}
        # fwd[k][bağlam(k sözcük)] -> Counter(sonraki);  bwd[k][bağlam] -> Counter(önceki)
        self.fwd: list[dict[tuple, Counter]] = [dict() for _ in range(self.ORDER + 1)]
        self.bwd: list[dict[tuple, Counter]] = [dict() for _ in range(self.BWD_ORDER + 1)]
        self.tok = SubwordTokenizer()
        self.suf_fwd: dict[str, Counter] = {}
        self.suf_bwd: dict[str, Counter] = {}
        self._last_tokens: list[str] = []
        self.heard_tokens = 0
        self._memo: dict = {}
        self.organ = None            # nora_broca_net.BrocaOrgan (varsa söz bununla kurulur; n-gram yedek)
        self._last_organ = None

    def _tokens(self, text: str) -> list[str]:
        return self.tok.encode(text) if self.tok.tk is not None else tokenize(text)

    # ---------------------------------------------------------------- öğrenme
    @staticmethod
    def _suf(tok: str) -> str:
        return "~" + tok[-3:]

    @staticmethod
    def _add(table: dict, key, nxt: str):
        c = table.get(key)
        if c is None:
            c = table[key] = Counter()
        c[nxt] += 1

    def _prune(self, tables: list[dict]):
        for t in tables:
            if len(t) > self.MAX_CTX:
                for k in [k for k, c in t.items() if sum(c.values()) <= 1]:
                    del t[k]
            for c in t.values():
                if len(c) > self.MAX_SUCC * 3:
                    for w, _ in c.most_common()[self.MAX_SUCC * 2:]:
                        del c[w]

    def hear(self, text: str, state: np.ndarray, neural_limit: int | None = 8):
        """Metni cümle cümle duyar ve öğrenir.

        Canlı turda sinirsel güncelleme ilk ``neural_limit`` cümleyle sınırlıdır;
        çevrimdışı dosya okumada ``None`` bütün cümleleri tek geçiş işler. Bu ayrım
        yanıt gecikmesini büyütmeden corpus eğitiminin sessizce ilk 8 cümlede
        kalmasını önler.
        """
        n_sent = 0
        if self.organ is not None:
            sents = [x for x in re.split(r"(?<=[.!?…])\s+|\n+", text) if len(re.findall(r"[^\W\d_]+", x)) >= 3]
            selected = sents if neural_limit is None else sents[:max(0, int(neural_limit))]
            for start in range(0, len(selected), 32):
                self.organ.learn_online(selected[start:start + 32], state)  # her cümle bir kez, epoch yok
        for sent in re.split(r"(?<=[.!?…])\s+|\n+", text):
            n_words = len(re.findall(r"[^\W\d_]+", sent))
            if n_words < 3 or n_words < 0.7 * len(re.findall(r"\S+", sent)):
                continue   # kod/tablo/işaret kalabalığı dil değildir
            toks = self._tokens(sent)
            if not toks:
                continue
            seq = [BOS] + toks + ([] if toks[-1] in END_MARKS else [EOS]) + ([EOS] if toks[-1] in END_MARKS else [])
            for i in range(1, len(seq)):
                for k in range(1, self.ORDER + 1):
                    if i - k >= 0:
                        self._add(self.fwd[k], tuple(seq[i - k:i]), seq[i])
                # geriye: seq[i] bağlamı = sonraki k sözcük, hedef = önceki sözcük
                for k in range(1, self.BWD_ORDER + 1):
                    if i + k < len(seq) + 0 and i - 1 >= 0:
                        self._add(self.bwd[k], tuple(seq[i:i + k]), seq[i - 1])
                if seq[i - 1] not in (BOS, EOS):
                    self._add(self.suf_fwd, self._suf(seq[i - 1]), seq[i])
                if seq[i] not in (BOS, EOS) and i - 1 >= 0:
                    self._add(self.suf_bwd, self._suf(seq[i]), seq[i - 1])
            for t in toks:
                n = self.count[t]
                m = self.mean_state.get(t)
                self.mean_state[t] = state.copy() if m is None else m + (state - m) / (n + 1.0)
                self.count[t] = n + 1
                self.heard_tokens += 1
            n_sent += 1
            if n_sent % 5000 == 0:
                self._prune(self.fwd + self.bwd)
        if len(self.count) > self.MAX_VOCAB:
            for k, _ in self.count.most_common()[self.MAX_VOCAB:]:
                self.count.pop(k, None)
                self.mean_state.pop(k, None)

    def hear_file(self, path: str, state: np.ndarray, max_chars: int = 400_000_000):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                chunk = f.read(2_000_000)
                if not chunk:
                    break
                tail = f.readline()          # cümleyi ortasından bölme
                # Dosya eğitimi canlı sohbet turu değildir: bütün cümleler tek
                # geçiş öğrenilir; n-gram ve sinirsel organ aynı metni görür.
                self.hear(chunk + tail, state, neural_limit=None)

    @property
    def mature(self) -> float:
        """Dil olgunluğu [0,1]: duyulan sözcük sayısıyla doyar. Hiç duymadıysa 0."""
        n = self.heard_tokens + (self.organ.seen_tokens if self.organ is not None else 0)
        return float(1.0 - np.exp(-n / 3000.0))

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

    def _dist(self, tables, suf_table, ctx: list[str]) -> Counter | None:
        """Geri-çekilmeli (backoff) dağılım: en uzun bağlamdan başlar, yetersizse kısaltır, sonunda son-ek sınıfı."""
        for k in range(min(len(tables) - 1, len(ctx)), 0, -1):
            c = tables[k].get(tuple(ctx[-k:]) if tables is self.fwd else tuple(ctx[:k]))
            if c and (sum(c.values()) >= 2 or k == 1):
                return c
        last = ctx[-1] if tables is self.fwd else ctx[0]
        if last not in (BOS, EOS):
            return suf_table.get(self._suf(last))
        return None

    def _step(self, tables, suf_table, ctx: list[str], state: np.ndarray, temp: float, used: Counter) -> str | None:
        c = self._dist(tables, suf_table, ctx)
        if not c:
            return None
        items = c.most_common(self.MAX_SUCC)
        cands = [w for w, _ in items]
        logp = np.log(np.array([n for _, n in items], dtype=np.float64) + 0.5)
        sc = logp + 6.0 * np.array([self._match(w, state) for w in cands]) \
            - 1.2 * np.array([used.get(w, 0) for w in cands]) + temp * self.rng.gumbel(size=len(cands))
        return cands[int(np.argmax(sc))]

    def _grow_right(self, ctx: list[str], state, temp, used: Counter, limit: int) -> list[str]:
        out = []
        for _ in range(limit):
            w = self._step(self.fwd, self.suf_fwd, ctx + out, state, temp, used)
            if w is None or w in (BOS, EOS):
                break
            out.append(w)
            used[w] += 1
        return out

    def _grow_left(self, ctx: list[str], state, temp, used: Counter, limit: int) -> list[str]:
        out = []
        for _ in range(limit):
            w = self._step(self.bwd, self.suf_bwd, list(reversed(out)) + ctx, state, temp, used)
            if w is None or w in (BOS, EOS):
                break
            out.append(w)
            used[w] += 1
        return list(reversed(out))

    def speak_organ(self, state: np.ndarray, anchors: list[str] | None, extra_allowed: list[str] | None = None,
                    n: int = 8, explore: float = 1.0) -> str | None:
        """Sinirsel Broca ile söz: içerik yalnızca çapa/bilgi sözcüklerinden (yapısal halüsinasyon engeli)."""
        org = self.organ
        temp = explore * (0.55 + 0.35 * float(state[F_IDX["ne"]]))
        seqs = [q for q in org.generate(state, anchors or [], extra_allowed, temperature=temp, n=n)
                if len(q) >= 2]
        if not seqs:
            return None
        # Üretilen adayları yalnız çapa kapsamıyla değil, aynı öğrenilmiş modelin
        # dizi olasılığıyla da sırala. İçerik maskesi ve anlam çapaları korunur.
        try:
            raw_fluency = np.asarray(org.logprob(seqs, state, anchors or []), dtype=np.float64)
            fluency = np.clip((raw_fluency - raw_fluency.mean()) / (raw_fluency.std() + 1e-6), -2.0, 2.0)
            self._last_organ_fluency = {
                "active": True, "logprob": [float(x) for x in raw_fluency],
            }
        except Exception as exc:
            fluency = np.zeros(len(seqs), dtype=np.float64)
            self._last_organ_fluency = {"active": False, "error": type(exc).__name__}
        best, best_sc = None, -1e18
        anchor_ids = set()
        for a in anchors or []:
            anchor_ids |= set(org.tok.tk.encode(a).ids)
        for q, learned in zip(seqs, fluency):
            cover = len(anchor_ids & set(q)) / max(1, len(anchor_ids))
            rep = len(set(q)) / len(q)
            ends = float(org.tok.tk.id_to_token(q[-1]) in END_MARKS)
            sc = (2.0 * cover + 1.0 * rep + 0.6 * ends + 0.45 * float(learned)
                  + 0.02 * min(len(q), 12) + 0.1 * self.rng.gumbel())
            if sc > best_sc:
                best, best_sc = q, sc
        if best is None:
            return None
        self._last_organ = (best, list(anchors or []), state.copy())
        return org.detok(best)

    def speak(self, state: np.ndarray, anchors: list[str] | None = None, n_candidates: int = 6,
              n_free: int = 2, explore: float = 1.0, return_all: bool = False, extra_allowed: list[str] | None = None):
        if self.organ is not None and not return_all:
            return self.speak_organ(state, anchors, extra_allowed, n=max(6, n_candidates), explore=explore)
        return self._speak_ngram(state, anchors, n_candidates, n_free, explore, return_all)

    def _speak_ngram(self, state: np.ndarray, anchors: list[str] | None = None, n_candidates: int = 6,
                     n_free: int = 2, explore: float = 1.0, return_all: bool = False):
        """Çapalardan (yoksa cümle başından) büyüyen adaylar üretir, en iyisini seçer. Dil yoksa None."""
        if self.mature < 0.002 or not self.count:
            return None
        self._memo.clear()
        anchor_words = [a for a in (anchors or []) if a and a.strip()]
        anchors = [t for t in (self._tokens(a) for a in anchor_words) if t]
        temp = explore * (0.15 + 0.35 * float(state[F_IDX["ne"]]) + 0.15 * (1.0 - float(state[F_IDX["sht"]])))
        best, best_sc = None, -1e18
        pool: list[list[str]] = []
        for ci in range(n_candidates + (n_free if anchors else 0)):
            used: Counter = Counter()
            if anchors and ci < n_candidates:
                core = list(anchors[int(self.rng.integers(len(anchors)))])
                for w in core:
                    used[w] += 1
                right = self._grow_right(core, state, temp, used, self.MAX_LEN)
                left = self._grow_left(core + right, state, temp, used, self.MAX_LEN // 3)
                sent = left + core + right
            else:
                sent = self._grow_right([BOS], state, temp, used, self.MAX_LEN)
            if not sent:
                continue
            pool.append(sent)
            cover = sum(all(p in sent for p in a) for a in anchors) / max(1, len(anchors))
            fit = float(np.mean([self._match(w, state) for w in sent]))
            ends = float(sent[-1] in END_MARKS)
            rep = len(set(sent)) / len(sent)
            flu = self._fluency(sent)
            score = 2.0 * cover + 6.0 * fit + 1.0 * rep + 0.8 * ends + 0.02 * min(len(sent), 12) + 1.5 * flu + 0.1 * self.rng.gumbel()
            if score > best_sc:
                best, best_sc = sent, score
        if best is None:
            return None
        self._last_tokens = list(best)
        if return_all:
            return pool
        return self._detok(best)

    def _fluency(self, sent: list[str]) -> float:
        """Cümlenin kendi öğrenilmiş dizi modeline göre ortalama log-olasılığı ([0,1]'e sıkıştırılmış)."""
        seq = [BOS] + sent
        tot, n = 0.0, 0
        for i in range(1, len(seq)):
            c = self._dist(self.fwd, self.suf_fwd, seq[max(0, i - self.ORDER):i])
            if c:
                tot += float(np.log((c.get(seq[i], 0) + 0.05) / (sum(c.values()) + 0.05 * len(c) + 1.0)))
            else:
                tot += -7.0
            n += 1
        return float(np.exp(tot / max(1, n)))

    def _detok(self, toks: list[str]) -> str:
        if self.tok.tk is not None:
            s = "".join(toks).replace("\u2581", " ")
            s = re.sub(r"\s+([,.;:!?…)])", r"\1", s).strip()
            return (s[:1].upper() + s[1:]) if s else s
        s = ""
        for t in toks:
            s += t if (not s or t in END_MARKS or t in ",;:)") else " " + t
        s = s.strip()
        return (s[:1].upper() + s[1:]) if s else s

    def common_pieces(self, n: int = 1500) -> set:
        """Dilin işlev parçaları: en sık duyulan n parça (içerik olmayan; ek, bağlaç, noktalama...)."""
        if getattr(self, "_common_n", -1) != self.heard_tokens // 100000:
            self._common = {w for w, _ in self.count.most_common(n)}
            self._common_n = self.heard_tokens // 100000
        return self._common

    def groundedness(self, sent: list[str], known: set) -> float:
        """Cümledeki İÇERİK parçalarının ne kadarı bilinen şeylere (çapa/bilgi) dayanıyor? İşlev parçaları sayılmaz."""
        com = self.common_pieces()
        content = [p for p in sent if p not in com and p not in END_MARKS]
        return 1.0 if not content else sum(p in known for p in content) / len(content)

    def reinforce_tokens(self, tokens: list[str], valence: float, state: np.ndarray, lr: float = 0.15):
        for t in set(tokens):
            m = self.mean_state.get(t)
            if m is not None:
                self.mean_state[t] = np.clip(m + lr * float(valence) * (state - m), 0.0, 1.0)

    def reinforce(self, valence: float, state: np.ndarray, lr: float = 0.12):
        """İyi karşılanan söylenenlerin durum çağrışımı mevcut duruma doğru çekilir (kötüyse uzaklaşır)."""
        if self.organ is not None and getattr(self, "_last_organ", None) is not None and valence > 0:
            seq, anc, st = self._last_organ
            self.organ.reinforce([seq], [float(valence)], st, anc, solo=True)    # iyi karşılanan söz bir adım pekişir
        self._last_organ = None
        for t in set(self._last_tokens):
            m = self.mean_state.get(t)
            if m is not None:
                self.mean_state[t] = np.clip(m + lr * float(valence) * (state - m), 0.0, 1.0)
        self._last_tokens = []

    # ---------------------------------------------------------------- değerlendirme
    def topk_accuracy(self, text: str, k: int = 5) -> dict:
        """Görülmemiş metinde bir sonraki sözcüğü tahmin başarısı (top-1 / top-k) — dil bilgisi ölçüsü."""
        hit1 = hitk = n = 0
        for sent in re.split(r"(?<=[.!?…])\s+|\n+", text):
            toks = self._tokens(sent)
            if len(toks) < 4:
                continue
            seq = [BOS] + toks
            for i in range(1, len(seq)):
                c = self._dist(self.fwd, self.suf_fwd, seq[max(0, i - self.ORDER):i])
                if not c:
                    n += 1
                    continue
                top = [w for w, _ in c.most_common(k)]
                hit1 += int(top[0] == seq[i])
                hitk += int(seq[i] in top)
                n += 1
        return {"n": n, "top1": hit1 / max(1, n), f"top{k}": hitk / max(1, n)}

    # ---------------------------------------------------------------- kalıcılık (sıkıştırılmış, ayrı dosya)
    def merge(self, other: "LearnedVoice"):
        """İki ayrı okumadan gelen sayaçları toplar (aynı alt-sözcük ayrıştırıcıyla eğitilmiş olmalı)."""
        for mine, theirs in ((self.fwd, other.fwd), (self.bwd, other.bwd)):
            for k in range(len(mine)):
                for ctx, c in theirs[k].items():
                    t = mine[k].get(ctx)
                    if t is None:
                        mine[k][ctx] = Counter(c)
                    else:
                        t.update(c)
        for mine, theirs in ((self.suf_fwd, other.suf_fwd), (self.suf_bwd, other.suf_bwd)):
            for k, c in theirs.items():
                mine.setdefault(k, Counter()).update(c)
        for w, n in other.count.items():
            a, b = self.count.get(w, 0), n
            ms_a, ms_b = self.mean_state.get(w), other.mean_state.get(w)
            if ms_a is None:
                self.mean_state[w] = ms_b
            elif ms_b is not None:
                self.mean_state[w] = (ms_a * a + ms_b * b) / max(1, a + b)
            self.count[w] = a + b
        self.heard_tokens += other.heard_tokens
        self._prune(self.fwd + self.bwd)

    def save_file(self, path: str):
        tmp = path + ".tmp"          # atomik yazım: yarım kalan kayıt eskisini bozmaz
        with gzip.open(tmp, "wb") as f:
            pickle.dump({"tok": self.tok.to_str(), "heard": self.heard_tokens, "count": dict(self.count), "mean": self.mean_state,
                         "fwd": self.fwd, "bwd": self.bwd, "suf_fwd": self.suf_fwd, "suf_bwd": self.suf_bwd}, f, protocol=4)
        os.replace(tmp, path)

    def load_file(self, path: str):
        with gzip.open(path, "rb") as f:
            d = pickle.load(f)
        self.tok = SubwordTokenizer.from_str(d.get("tok", ""))
        self.heard_tokens = int(d["heard"])
        self.count = Counter(d["count"])
        self.mean_state = d["mean"]
        self.fwd, self.bwd, self.suf_fwd, self.suf_bwd = d["fwd"], d["bwd"], d["suf_fwd"], d["suf_bwd"]

    def state_dict(self) -> dict:       # hafif özet (ana belleğe ayrı dosyada yazılır)
        return {"heard": self.heard_tokens}

    def load_state_dict(self, d: dict):
        pass
