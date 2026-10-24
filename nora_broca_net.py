"""
NORA BROCA ARTİKÜLASYON ORGANI (sinirsel) — zihin karar verir, bu organ yalnızca dile döker.

Rol ayrımı
  * Ne söyleneceği (niyet, kavram çapası, epistemik duruş, gerilim) zihinden gelir: Küresel Çalışma Alanı,
    A24 iç-duyum, A22 CA3 anıları. Bu organ KENDİ içeriğini üretmez.
  * Girdi: iç durum vektörü (12 boyut) + çapa parçaları (zihnin vermek istediği kavramlar).
  * Halüsinasyon engeli ÖĞRENMEYE bırakılmaz, çözümlemede yapısaldır: dilin işlev parçaları (en sık duyulanlar)
    serbesttir; içerik parçaları yalnızca zihnin verdiği çapa/bilgi kümesinden seçilebilir (sözcük maskesi).
  * Öğrenme: tek geçiş (epoch yok), akıştan çevrimiçi. Eğitim çifti kendiliğinden kurulur: cümle + o cümleden
    rastgele seçilen birkaç içerik sözcüğü (çapa) + okunduğu iç durum. Etiket/öğretmen yoktur.
  * Çevrimiçi pekiştirme: kendi sözü ödüllendirilirse (dayanaklı, akıcı) o söz bir adım daha öğrenilir.

Küçük bir GRU'dur (~7M parametre); mimarinin zihni DEĞİL, yalnızca artikülasyon motorudur.
"""
from __future__ import annotations

import os
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nora_voice import NF, SubwordTokenizer

N_SPECIAL = 3          # PAD, BOS, EOS (sözlüğün sonuna eklenir)


class BrocaNet(nn.Module):
    def __init__(self, vocab: int, d: int = 256, h: int = 512):
        super().__init__()
        self.V = vocab + N_SPECIAL
        self.pad, self.bos, self.eos = vocab, vocab + 1, vocab + 2
        self.emb = nn.Embedding(self.V, d)
        self.cond = nn.Sequential(nn.Linear(NF + d, h), nn.Tanh())
        self.cond_in = nn.Linear(h, d, bias=False)
        self.gru = nn.GRU(d, h, num_layers=1, batch_first=True)
        self.proj = nn.Linear(h, d)
        self.h = h

    def encode_cond(self, state: torch.Tensor, anchor_bag: torch.Tensor) -> torch.Tensor:
        """state: (B,NF); anchor_bag: (B,V) çok-sıcak -> (B,h)"""
        n = anchor_bag.sum(1, keepdim=True).clamp(min=1.0)
        a = (anchor_bag @ self.emb.weight) / n
        return self.cond(torch.cat([state, a], dim=1))

    def hidden(self, ids: torch.Tensor, c: torch.Tensor, h0: torch.Tensor | None = None):
        x = self.emb(ids) + self.cond_in(c).unsqueeze(1)
        out, hn = self.gru(x, c.unsqueeze(0) if h0 is None else h0)
        return self.proj(out), hn                              # (B,L,d)

    def forward(self, ids: torch.Tensor, c: torch.Tensor, h0: torch.Tensor | None = None):
        z, hn = self.hidden(ids, c, h0)
        return z @ self.emb.weight.t(), hn                     # gömme bağlama (tied): tam sözlük üzerinden


class BrocaOrgan:
    MAX_LEN = 40

    def __init__(self, tokenizer: SubwordTokenizer, seed: int = 0, d: int = 256, h: int = 512):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed + 90_001)
        self.tok = tokenizer
        V = self.tok.tk.get_vocab_size()
        self.net = BrocaNet(V, d, h)
        self.lr = 2.5e-3
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.V = V
        self.freq = torch.zeros(V)
        self.seen_tokens = 0
        self.pos = {"wiki": 0, "chat": 0}          # akışta nereye kadar okundu (devam ederken tekrar okunmaz: epoch yok)
        self.common_ids: torch.Tensor | None = None
        self.punct_ids: set[int] = set()

    # ------------------------------------------------------------ veri
    def _encode(self, sentences: list[str]) -> list[list[int]]:
        return [e.ids[: self.MAX_LEN] for e in self.tok.tk.encode_batch(sentences)]

    def _is_content_start(self, piece: str) -> bool:
        return piece.startswith("▁") and any(ch.isalpha() for ch in piece)

    def _pseudo_anchor_bag(self, ids: list[int], pieces: list[str]) -> list[int]:
        """Cümleden 1-3 içerik sözcüğü seç (çapa); sözcüğün tüm parçaları çapa kümesine girer."""
        starts = [i for i, p in enumerate(pieces) if self._is_content_start(p)
                  and (self.common_ids is None or ids[i] not in self._common_set)]
        if not starts:
            return []
        k = int(self.rng.integers(1, min(3, len(starts)) + 1))
        chosen = self.rng.choice(starts, size=k, replace=False)
        bag: list[int] = []
        for s in chosen:
            j = s + 1
            while j < len(pieces) and not pieces[j].startswith("▁") and pieces[j].isalpha():
                j += 1
            bag += ids[s:j]
        return bag

    def update_common(self, n: int = 1500):
        top = torch.topk(self.freq, min(n, self.V)).indices
        self.common_ids = top
        self._common_set = set(top.tolist())

    # ------------------------------------------------------------ çevrimiçi (tek geçiş) öğrenme
    def learn_batch(self, sentences: list[str], state: np.ndarray, lr: float | None = None) -> float:
        """Tek bir gradyan adımı: bu akış parçası görüldükten sonra bir daha okunmaz (epoch yok)."""
        enc = self.tok.tk.encode_batch(sentences)
        rows, bags = [], []
        for e in enc:
            ids, pcs = e.ids[: self.MAX_LEN], e.tokens[: self.MAX_LEN]
            if len(ids) < 3:
                continue
            for i in ids:
                self.freq[i] += 1
            self.seen_tokens += len(ids)
            rows.append(ids)
            bags.append(self._pseudo_anchor_bag(ids, pcs))
        if not rows:
            return 0.0
        if self.common_ids is None or self.seen_tokens % 2_000_000 < 4000:
            self.update_common()
        L = max(len(r) for r in rows) + 1
        B = len(rows)
        net = self.net
        inp = torch.full((B, L), net.pad, dtype=torch.long)
        tgt = torch.full((B, L), -100, dtype=torch.long)
        for b, r in enumerate(rows):
            inp[b, 0] = net.bos
            inp[b, 1:len(r) + 1] = torch.tensor(r)
            tgt[b, :len(r)] = torch.tensor(r)
            tgt[b, len(r)] = net.eos
        bag = torch.zeros(B, net.V)
        for b, g in enumerate(bags):
            if g:
                bag[b, g] = 1.0
        st = torch.from_numpy(np.tile(state.astype(np.float32), (B, 1)))
        c = net.encode_cond(st, bag)
        z, _ = net.hidden(inp, c)
        loss = self._sampled_loss(z, tgt)
        for g in self.opt.param_groups:
            g["lr"] = self.lr if lr is None else lr
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        self.opt.step()
        return float(loss.detach())

    def _sampled_loss(self, z: torch.Tensor, tgt: torch.Tensor, K: int = 2048) -> torch.Tensor:
        """Örneklemeli softmax (log-q düzeltmeli): tam sözlük yerine hedefler + K örneklenmiş negatif."""
        net = self.net
        mask = tgt != -100
        zt, yt = z[mask], tgt[mask]
        q_full = torch.full((net.V,), 1.0 / net.V)
        q = (self.freq + 1.0) ** 0.75
        q_full[: self.V] = q / q.sum()
        q_full = q_full / q_full.sum()
        neg = torch.multinomial(q_full, K, replacement=True)
        cand = torch.unique(neg)
        W = net.emb.weight[cand]
        logits = zt @ W.t() - torch.log(K * q_full[cand])[None, :]
        logits = logits.masked_fill(cand[None, :] == yt[:, None], -1e9)        # kazara isabet
        true = (zt * net.emb.weight[yt]).sum(1) - torch.log(K * q_full[yt])
        all_l = torch.cat([true[:, None], logits], dim=1)
        return F.cross_entropy(all_l, torch.zeros(len(yt), dtype=torch.long))

    def learn_online(self, sentences: list[str], state: np.ndarray) -> float:
        """Yaşarken duyulan cümleler: tek geçiş, küçük adım (önceki bilgiyi bozmadan)."""
        return self.learn_batch(sentences, state, lr=3e-4)

    @torch.no_grad()
    def evaluate(self, sentences: list[str], state: np.ndarray, k: int = 5, max_sent: int = 800, use_anchor: bool = True) -> dict:
        """Görülmemiş cümlelerde bir sonraki parçayı tahmin (çapa verilerek; n-gram ile aynı ölçü)."""
        net = self.net
        net.eval()
        hit1 = hitk = n = 0
        nll = 0.0
        enc = self.tok.tk.encode_batch(sentences[:max_sent])
        for i in range(0, len(enc), 64):
            chunk = enc[i:i + 64]
            rows = [(e.ids[: self.MAX_LEN], e.tokens[: self.MAX_LEN]) for e in chunk if len(e.ids) >= 4]
            if not rows:
                continue
            L = max(len(r[0]) for r in rows) + 1
            B = len(rows)
            inp = torch.full((B, L), net.pad, dtype=torch.long)
            tgt = torch.full((B, L), -100, dtype=torch.long)
            bag = torch.zeros(B, net.V)
            for b, (r, pcs) in enumerate(rows):
                inp[b, 0] = net.bos
                inp[b, 1:len(r) + 1] = torch.tensor(r)
                tgt[b, :len(r)] = torch.tensor(r)
                tgt[b, len(r)] = net.eos
                g = self._pseudo_anchor_bag(r, pcs)
                if g and use_anchor:
                    bag[b, g] = 1.0
            st = torch.from_numpy(np.tile(state.astype(np.float32), (B, 1)))
            logits, _ = net(inp, net.encode_cond(st, bag))
            mask = tgt != -100
            top = logits.topk(k, dim=-1).indices
            hit1 += int(((top[..., 0] == tgt) & mask).sum())
            hitk += int(((top == tgt.unsqueeze(-1)).any(-1) & mask).sum())
            n += int(mask.sum())
            nll += float(F.cross_entropy(logits[mask], tgt[mask], reduction="sum"))
        net.train()
        return {"n": n, "top1": hit1 / max(1, n), f"top{k}": hitk / max(1, n), "ppl": float(np.exp(nll / max(1, n)))}

    # ------------------------------------------------------------ üretim (yapısal halüsinasyon engeli)
    @torch.no_grad()
    def generate(self, state: np.ndarray, anchors: list[str], extra_allowed: list[str] | None = None,
                 temperature: float = 0.7, top_k: int = 30, max_len: int = 28, n: int = 1,
                 cover_bonus: float = 4.0) -> list[list[int]]:
        """İçerik parçaları YALNIZCA çapa + zihnin verdiği bilgi sözcüklerinden seçilebilir. İşlev parçaları serbest."""
        net = self.net
        net.eval()
        # Çapa sözcükler ATOMİKTİR: yalnızca ilk parçaları seçilebilir; seçilince kalan parçaları sırayla zorla eklenir.
        words = [self.tok.tk.encode(a).ids for a in list(anchors) + list(extra_allowed or []) if a and a.strip()]
        words = [w for w in words if w]
        starts = {w[0]: w[1:] for w in words}
        anchor_starts = {self.tok.tk.encode(a).ids[0] for a in anchors if a and a.strip() and self.tok.tk.encode(a).ids}
        allowed = torch.zeros(net.V, dtype=torch.bool)
        if self.common_ids is not None:
            allowed[self.common_ids] = True
        allowed[list(starts)] = True
        allowed[net.eos] = True
        bag = torch.zeros(n, net.V)
        for a in anchors:
            for t in self.tok.tk.encode(a).ids:
                bag[:, t] = 1.0
        st = torch.from_numpy(np.tile(state.astype(np.float32), (n, 1)))
        c = net.encode_cond(st, bag)
        h = c.unsqueeze(0)
        cur = torch.full((n, 1), net.bos, dtype=torch.long)
        out = [[] for _ in range(n)]
        done = [False] * n
        queue: list[list[int]] = [[] for _ in range(n)]
        remaining = [set(anchor_starts) for _ in range(n)]
        for _ in range(max_len):
            logits, h = net(cur, c, h)
            lg = logits[:, -1, :] / max(1e-3, temperature)
            lg[:, ~allowed] = -1e9
            lg[:, net.pad] = lg[:, net.bos] = -1e9
            for b in range(n):
                for t in remaining[b]:
                    lg[b, t] += cover_bonus
                for t in set(out[b]):
                    lg[b, t] -= (3.0 if t in starts else 1.5) * out[b].count(t)
            kth = lg.topk(min(top_k, net.V), dim=-1).values[:, -1:]
            lg[lg < kth] = -1e9
            nxt = torch.multinomial(F.softmax(lg, dim=-1), 1)
            for b in range(n):
                if queue[b]:                          # sürmekte olan sözcüğün kalan parçaları
                    nxt[b, 0] = queue[b].pop(0)
                t = int(nxt[b])
                if done[b]:
                    continue
                if t == net.eos:
                    done[b] = True
                else:
                    if t in starts and not queue[b]:
                        queue[b] = list(starts[t])         # sözcüğün kalan parçaları sırayla gelir
                    out[b].append(t)
                    remaining[b].discard(t)
            if all(done) and not any(queue):
                break
            cur = nxt
        net.train()
        return out

    @torch.no_grad()
    def logprob(self, sequences: list[list[int]], state: np.ndarray, anchors: list[str]) -> list[float]:
        """Her söz için ortalama token log-olasılığı (akıcılık ölçüsü; çapa-koşullu)."""
        net = self.net
        net.eval()
        anchor_ids: list[int] = []
        for a in anchors:
            anchor_ids += self.tok.tk.encode(a).ids
        B = len(sequences)
        L = max(len(x) for x in sequences) + 1
        inp = torch.full((B, L), net.pad, dtype=torch.long)
        tgt = torch.full((B, L), -100, dtype=torch.long)
        for b, r in enumerate(sequences):
            inp[b, 0] = net.bos
            if r:
                inp[b, 1:len(r) + 1] = torch.tensor(r)
                tgt[b, :len(r)] = torch.tensor(r)
            tgt[b, len(r)] = net.eos
        bag = torch.zeros(B, net.V)
        if anchor_ids:
            bag[:, anchor_ids] = 1.0
        st = torch.from_numpy(np.tile(state.astype(np.float32), (B, 1)))
        logits, _ = net(inp, net.encode_cond(st, bag))
        lp = -F.cross_entropy(logits.reshape(-1, net.V), tgt.reshape(-1), ignore_index=-100, reduction="none").view(B, L)
        net.train()
        return [float(x) for x in (lp.sum(1) / (tgt != -100).sum(1).clamp(min=1))]

    def detok(self, ids: list[int]) -> str:
        pieces = [self.tok.tk.id_to_token(i) for i in ids if i < self.V]
        s = "".join(pieces).replace("▁", " ")
        s = re.sub(r"\s+([,.;:!?…)])", r"\1", s).strip()
        return (s[:1].upper() + s[1:]) if s else s

    # ------------------------------------------------------------ çevrimiçi pekiştirme (ödüllü tek adım)
    def reinforce(self, sequences: list[list[int]], rewards: list[float], state: np.ndarray, anchors: list[str],
                  solo: bool = False, extra_words: list[str] | None = None) -> float:
        """Ödülü yüksek sözler bir adım daha öğrenilir (ödül-ağırlıklı olabilirlik); düşük ödül ağırlığı sıfırdır."""
        net = self.net
        anchor_ids: list[int] = []
        for a in anchors:
            anchor_ids += self.tok.tk.encode(a).ids
        rs = np.array(rewards, dtype=np.float64)
        w = np.ones(len(rs)) if solo else np.maximum(0.0, rs - rs.mean())
        if w.sum() <= 1e-9:
            return 0.0
        w = torch.tensor(w / w.sum(), dtype=torch.float32)
        rows = [s for s in sequences if s]
        if not rows:
            return 0.0
        B = len(sequences)
        L = max(len(s) for s in sequences) + 1
        inp = torch.full((B, L), net.pad, dtype=torch.long)
        tgt = torch.full((B, L), -100, dtype=torch.long)
        for b, r in enumerate(sequences):
            inp[b, 0] = net.bos
            if r:
                inp[b, 1:len(r) + 1] = torch.tensor(r)
                tgt[b, :len(r)] = torch.tensor(r)
            tgt[b, len(r)] = net.eos
        bag = torch.zeros(B, net.V)
        if anchor_ids:
            bag[:, anchor_ids] = 1.0
        st = torch.from_numpy(np.tile(state.astype(np.float32), (B, 1)))
        logits, _ = net(inp, net.encode_cond(st, bag))
        ce = F.cross_entropy(logits.reshape(-1, net.V), tgt.reshape(-1), ignore_index=-100, reduction="none").view(B, L)
        per_seq = ce.sum(1) / (tgt != -100).sum(1).clamp(min=1)
        loss = (per_seq * w).sum()
        for g in self.opt.param_groups:
            g["lr"] = 5e-4
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        self.opt.step()
        return float(loss.detach())

    # ------------------------------------------------------------ kalıcılık
    def save(self, path: str):
        tmp = path + ".tmp"
        torch.save({"net": self.net.state_dict(), "freq": self.freq, "seen": self.seen_tokens,
                    "tok": self.tok.to_str(), "pos": self.pos, "dims": (self.net.emb.embedding_dim, self.net.h)}, tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "BrocaOrgan":
        d = torch.load(path, weights_only=False)
        o = cls(SubwordTokenizer.from_str(d["tok"]), d=d["dims"][0], h=d["dims"][1])
        o.net.load_state_dict(d["net"])
        o.freq = d["freq"]
        o.seen_tokens = int(d["seen"])
        o.pos = d.get("pos", {"wiki": 0, "chat": 0})
        o.update_common()
        return o
