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

from nora._paths import REPO_ROOT


import os
import re
import json
import hashlib
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nora_voice import NF, SubwordTokenizer

N_SPECIAL = 3          # PAD, BOS, EOS (sözlüğün sonuna eklenir)

# Bunlar cevap/şablon değil, yalnız açıkça yarım kalmış yüzeyleri elemek için
# kullanılan Türkçe bağlaç/ilgeç sonu denetimidir. Anlam kapsamı plandan gelir.
_ARTICULATION_NONFINAL = frozenset({
    "ve", "veya", "ile", "ama", "fakat", "ancak", "çünkü", "için",
    "gibi", "kadar", "de", "da", "ki", "bir", "bu", "şu", "o",
})


class NeuralRelationalProposer(nn.Module):
    """Dağıtık bellekten iki dayanak seçip yapısal ilişki adayı üreten küçük ağ.

    Çıktı katmanı sabit kavram sözlüğü değildir. İlişki ve nesne, o anki bellekteki
    simgelere işaretçi skorlarıyla seçilir; böylece eğitimde görülmemiş kavram
    adları da yeni bir sınıf eklenmeden taşınır. Grafik burada yoktur: modelin
    kendi doğrudan gözlem belleği ve öğrenilen ağırlıkları kullanılır.

    ``learn_correction`` yalnız bağımsız doğrulayıcının kabul ettiği bir düzeltmeyi
    öğrenir. Son üretimde top-k adayları sembolik kanıt yerine geçmez. Nesne
    işaretçisindeki beş kosinüs benzerliği kavram etiketi taşımaz; adayın query
    ve iki dayanağın dört ucuyla eşleşmesini yeni nonce adlarına taşır.
    """

    def __init__(self, dim: int = 32, seed: int = 0, learning_rate: float = 0.01,
                 object_similarity: bool = True):
        super().__init__()
        self.dim = max(16, int(dim))
        self.seed = int(seed)
        torch.manual_seed(self.seed)
        d = self.dim
        self.symbol = nn.Linear(d, d)
        self.encoder = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.Tanh(), nn.Linear(2 * d, d), nn.Tanh())
        self.pair_transform = nn.Sequential(
            nn.Linear(4 * d + 8, 2 * d), nn.Tanh(), nn.Linear(2 * d, d), nn.Tanh(),
        )
        self.pair_score = nn.Linear(d, 1)
        self.relation_pointer = nn.Sequential(nn.Linear(2 * d, d), nn.Tanh(), nn.Linear(d, 1))
        self.object_pointer = nn.Sequential(nn.Linear(2 * d, d), nn.Tanh(), nn.Linear(d, 1))
        self.object_similarity = None
        self._configure_object_similarity(object_similarity)
        self.opt = torch.optim.Adam(self.parameters(), lr=float(learning_rate))
        self.memory: list[tuple[str, str, str]] = []
        self._memory_set: set[tuple[str, str, str]] = set()
        self._vectors: dict[str, torch.Tensor] = {}
        self.enabled = True
        self.corrections = 0
        self.failures = 0
        self.successful_proposals = 0
        self.last_loss = 0.0

    def _configure_object_similarity(self, enabled: bool) -> None:
        """Nonce uç eşleşmelerini açık gören sıfır-artık nesne işaretçisi."""
        if enabled:
            self.object_similarity = nn.Sequential(
                nn.Linear(5, self.dim), nn.Tanh(), nn.Linear(self.dim, 1),
            )
            nn.init.zeros_(self.object_similarity[-1].weight)
            nn.init.zeros_(self.object_similarity[-1].bias)
        else:
            self.object_similarity = None

    def _object_scores(self, pair_vector: torch.Tensor, query_vector: torch.Tensor,
                       concept_vectors: torch.Tensor, *, context=None) -> torch.Tensor:
        pair = pair_vector.unsqueeze(0).expand(len(concept_vectors), -1)
        base = self.object_pointer(torch.cat([pair, concept_vectors], dim=1))
        if self.object_similarity is not None:
            if context is None:
                raise ValueError("object similarity requires evidence context")
            memory, cindex, first, second = context
            references = torch.stack([
                query_vector,
                concept_vectors[cindex[memory[first][0]]],
                concept_vectors[cindex[memory[first][2]]],
                concept_vectors[cindex[memory[second][0]]],
                concept_vectors[cindex[memory[second][2]]],
            ])
            similarities = F.cosine_similarity(
                concept_vectors.unsqueeze(1), references.unsqueeze(0), dim=-1)
            base = base + self.object_similarity(similarities)
        return base.reshape(-1)

    def _vector(self, symbol: str) -> torch.Tensor:
        key = str(symbol)
        cached = self._vectors.get(key)
        if cached is not None:
            return cached
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little") + self.seed)
        raw = rng.standard_normal(self.dim).astype(np.float32)
        raw /= max(1e-6, float(np.linalg.norm(raw)))
        vector = torch.from_numpy(raw)
        self._vectors[key] = vector
        return vector

    def observe(self, subj: str, rel: str, obj: str) -> None:
        triple = (str(subj), str(rel), str(obj))
        if triple not in self._memory_set:
            self._memory_set.add(triple)
            self.memory.append(triple)

    @staticmethod
    def _cos(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cosine_similarity(left, right, dim=0).reshape(1)

    def _encode(self, memory, query: str):
        memory = [tuple(row) for row in memory]
        concepts = sorted({str(query)} | {row[0] for row in memory} | {row[2] for row in memory})
        relations = sorted({row[1] for row in memory})
        if not memory or not relations:
            raise ValueError("empty relational memory")
        cindex = {symbol: i for i, symbol in enumerate(concepts)}
        rindex = {symbol: i for i, symbol in enumerate(relations)}
        cv = self.symbol(torch.stack([self._vector(symbol) for symbol in concepts]))
        rv = self.symbol(torch.stack([self._vector(symbol) for symbol in relations]))
        encoded = []
        for subj, rel, obj in memory:
            encoded.append(self.encoder(torch.cat([cv[cindex[subj]], rv[rindex[rel]], cv[cindex[obj]]])))
        return memory, concepts, relations, cindex, rindex, cv, rv, torch.stack(encoded)

    def _pairs(self, encoded, query: str):
        memory, concepts, relations, cindex, rindex, cv, rv, mem = encoded
        qv = self.symbol(self._vector(str(query)))
        vectors, scores = [], []
        n = len(memory)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                left, right = memory[i], memory[j]
                # Kavram eşleşmeleri dağıtık benzerlik olarak modele girer;
                # ilişki adı için ayrı bir kural özelliği yoktur.
                s_i, o_i = cv[cindex[left[0]]], cv[cindex[left[2]]]
                s_j, o_j = cv[cindex[right[0]]], cv[cindex[right[2]]]
                similarities = torch.cat([
                    self._cos(s_i, s_j), self._cos(s_i, o_j),
                    self._cos(o_i, s_j), self._cos(o_i, o_j),
                    self._cos(qv, s_i), self._cos(qv, o_i),
                    self._cos(qv, s_j), self._cos(qv, o_j),
                ])
                pair = self.pair_transform(torch.cat([
                    mem[i], mem[j], mem[i] * mem[j], qv, similarities,
                ]))
                vectors.append(pair)
                scores.append(self.pair_score(pair).reshape(()))
        return (torch.stack(vectors) if vectors else torch.empty(0, self.dim),
                torch.stack(scores) if scores else torch.empty(0), n)

    @staticmethod
    def _scoped(logits: torch.Tensor, allowed: set[int]) -> torch.Tensor:
        mask = torch.full_like(logits, -1e9)
        indices = sorted(int(index) for index in allowed)
        if indices:
            mask[indices] = logits[indices]
        return mask

    @staticmethod
    def _pair_indices(index: int, n: int) -> tuple[int, int]:
        # _pairs köşegenleri atlar: her satırda n-1 öğe vardır.
        row = index // (n - 1)
        offset = index % (n - 1)
        col = offset if offset < row else offset + 1
        return row, col

    def learn_correction(self, memory, query: str, target, evidence_pair) -> float:
        if not self.enabled:
            return 0.0
        target = tuple(target)
        memory = [tuple(row) for row in memory]
        if evidence_pair[0] == evidence_pair[1] or not 0 <= evidence_pair[0] < len(memory) \
                or not 0 <= evidence_pair[1] < len(memory):
            raise ValueError("invalid evidence pair")
        if target[0] != str(query) or target[1] not in {row[1] for row in memory}:
            raise ValueError("target is not pointer-decodable")
        if target[2] not in ({str(query)} | {row[0] for row in memory} | {row[2] for row in memory}):
            raise ValueError("target object is not in memory")
        encoded = self._encode(memory, str(query))
        memory, concepts, relations, cindex, rindex, cv, rv, mem = encoded
        vectors, pair_logits, n = self._pairs(encoded, str(query))
        first, second = evidence_pair
        pair_index = first * (n - 1) + second - int(second > first)
        pair_loss = torch.nn.functional.cross_entropy(
            pair_logits.reshape(1, -1), torch.tensor([pair_index], dtype=torch.long),
        )
        pair_vector = vectors[pair_index]
        rel_logits = self.relation_pointer(torch.cat([
            pair_vector.unsqueeze(0).expand(len(relations), -1), rv,
        ], dim=1)).reshape(-1)
        obj_logits = self._object_scores(
            pair_vector, self.symbol(self._vector(str(query))), cv,
            context=(memory, cindex, first, second))
        rel_scope = {rindex[memory[first][1]], rindex[memory[second][1]]}
        obj_scope = {cindex[str(query)]} | {
            cindex[memory[index][position]] for index in (first, second) for position in (0, 2)
        }
        if rindex[target[1]] not in rel_scope or cindex[target[2]] not in obj_scope:
            raise ValueError("target escapes evidence pointer scope")
        rel_logits = self._scoped(rel_logits, rel_scope)
        obj_logits = self._scoped(obj_logits, obj_scope)
        rel_loss = torch.nn.functional.cross_entropy(
            rel_logits.reshape(1, -1), torch.tensor([rindex[target[1]]], dtype=torch.long),
        )
        obj_loss = torch.nn.functional.cross_entropy(
            obj_logits.reshape(1, -1), torch.tensor([cindex[target[2]]], dtype=torch.long),
        )
        loss = pair_loss + rel_loss + obj_loss
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.opt.step()
        self.corrections += 1
        self.last_loss = float(loss.detach())
        return self.last_loss

    @torch.no_grad()
    def diagnose(self, memory, query: str, target, evidence_pair) -> dict:
        """Top-1 hatasını pair/relation/object olarak davranışı değiştirmeden ölç.

        Hedef ve kanıt yalnız ölçüm fonksiyonundan gelir; üretim önerisi bu
        metodu çağırmaz. Böylece top-5 içinde görünen bir hedefin hangi nöral
        bileşen yüzünden birinci sıraya çıkamadığı ayrılabilir.
        """
        target = tuple(target)
        memory = [tuple(row) for row in memory]
        encoded = self._encode(memory, str(query))
        memory, concepts, relations, cindex, rindex, cv, rv, mem = encoded
        vectors, pair_logits, n = self._pairs(encoded, str(query))
        first, second = (int(evidence_pair[0]), int(evidence_pair[1]))
        pair_index = first * (n - 1) + second - int(second > first)
        pair_order = sorted(range(len(pair_logits)),
                            key=lambda index: (-float(pair_logits[index]), index))
        pair_rank = pair_order.index(pair_index) + 1
        pair_vector = vectors[pair_index]
        rel_logits = self.relation_pointer(torch.cat([
            pair_vector.unsqueeze(0).expand(len(relations), -1), rv,
        ], dim=1)).reshape(-1)
        obj_logits = self._object_scores(
            pair_vector, self.symbol(self._vector(str(query))), cv,
            context=(memory, cindex, first, second))
        rel_scope = {rindex[memory[first][1]], rindex[memory[second][1]]}
        obj_scope = {cindex[str(query)]} | {
            cindex[memory[index][position]] for index in (first, second) for position in (0, 2)
        }
        rel_logits = self._scoped(rel_logits, rel_scope)
        obj_logits = self._scoped(obj_logits, obj_scope)
        rel_order = sorted(rel_scope, key=lambda index: (-float(rel_logits[index]), index))
        obj_order = sorted(obj_scope, key=lambda index: (-float(obj_logits[index]), index))
        relation_rank = rel_order.index(rindex[target[1]]) + 1
        object_rank = obj_order.index(cindex[target[2]]) + 1
        return {
            "pair_rank": pair_rank, "relation_rank": relation_rank,
            "object_rank": object_rank,
            "pair_top1": pair_rank == 1, "pair_top5": pair_rank <= 5,
            "relation_top1": relation_rank == 1, "object_top1": object_rank == 1,
            "all_components_top1": pair_rank == relation_rank == object_rank == 1,
        }

    @torch.no_grad()
    def propose(self, query: str, *, memory=None, k: int = 5) -> list[dict]:
        if not self.enabled:
            return []
        memory = self.memory if memory is None else [tuple(row) for row in memory]
        if len(memory) < 2:
            return []
        encoded = self._encode(memory, str(query))
        memory, concepts, relations, cindex, rindex, cv, rv, mem = encoded
        vectors, pair_logits, n = self._pairs(encoded, str(query))
        if not len(pair_logits):
            return []
        pair_logprob = torch.log_softmax(pair_logits, dim=0)
        pair_order = torch.topk(pair_logprob, min(max(int(k) * 2, 6), len(pair_logits))).indices.tolist()
        direct = set(memory)
        candidates: dict[tuple, dict] = {}
        for pair_index in pair_order:
            first, second = self._pair_indices(pair_index, n)
            pair_vector = vectors[pair_index]
            rel_logits = self.relation_pointer(torch.cat([
                pair_vector.unsqueeze(0).expand(len(relations), -1), rv,
            ], dim=1)).reshape(-1)
            obj_logits = self._object_scores(
            pair_vector, self.symbol(self._vector(str(query))), cv,
            context=(memory, cindex, first, second))
            rel_scope = {rindex[memory[first][1]], rindex[memory[second][1]]}
            obj_scope = {cindex[str(query)]} | {
                cindex[memory[index][position]] for index in (first, second) for position in (0, 2)
            }
            rel_logits = self._scoped(rel_logits, rel_scope)
            obj_logits = self._scoped(obj_logits, obj_scope)
            rel_lp = torch.log_softmax(rel_logits, dim=0)
            obj_lp = torch.log_softmax(obj_logits, dim=0)
            rel_top = torch.topk(rel_lp, min(3, len(rel_scope)))
            obj_top = torch.topk(obj_lp, min(3, len(obj_scope)))
            for ri, rs in zip(rel_top.indices.tolist(), rel_top.values.tolist()):
                for oi, os in zip(obj_top.indices.tolist(), obj_top.values.tolist()):
                    triple = (str(query), relations[ri], concepts[oi])
                    if triple in direct:       # istenen şey hatırlama değil yeni öneri
                        continue
                    score = float(pair_logprob[pair_index] + rs + os)
                    row = {"subj": triple[0], "rel": triple[1], "obj": triple[2],
                           "score": score, "neural_pair": [memory[first], memory[second]],
                           "neural": True}
                    if triple not in candidates or score > candidates[triple]["score"]:
                        candidates[triple] = row
        ordered = sorted(candidates.values(), key=lambda row: (-row["score"], row["rel"], row["obj"]))
        return ordered[:max(0, int(k))]

    def export(self) -> dict:
        return {
            "kind": "neural_relational_proposer", "dim": self.dim, "seed": self.seed,
            "object_similarity": self.object_similarity is not None,
            "weights": {key: value.detach().cpu().tolist() for key, value in self.state_dict().items()},
            "memory": [list(row) for row in self.memory], "enabled": bool(self.enabled),
            "corrections": self.corrections, "failures": self.failures,
            "successful_proposals": self.successful_proposals,
        }

    def load_export(self, data: dict) -> None:
        if not data or data.get("kind") != "neural_relational_proposer":
            raise ValueError("invalid relational proposer")
        self._configure_object_similarity(bool(data.get("object_similarity", False)))
        self.opt = torch.optim.Adam(self.parameters(), lr=float(self.opt.defaults["lr"]))
        weights = {key: torch.tensor(value, dtype=self.state_dict()[key].dtype)
                   for key, value in (data.get("weights") or {}).items()}
        self.load_state_dict(weights, strict=True)
        self.memory = []
        self._memory_set = set()
        for row in data.get("memory") or ():
            if len(row) == 3:
                self.observe(*row)
        self.enabled = bool(data.get("enabled", True))
        self.corrections = int(data.get("corrections", 0) or 0)
        self.failures = int(data.get("failures", 0) or 0)
        self.successful_proposals = int(data.get("successful_proposals", 0) or 0)


def bootstrap_relational_proposer(proposer: NeuralRelationalProposer, *,
                                  examples: int = 300, seed: int = 0,
                                  max_attempts: int | None = None) -> dict:
    """Kavram adı vermeden, kendi başarısız önerilerinden genel aktarım öğren.

    Rastgele nonce çizgeler kurulur; bağımsız sembolik doğrulayıcı geçerli iki-
    dayanak sonuçları denetler. Doğrulayıcı biyoloji veya hedef kavram görmez.
    Ağın top-1'i doğru sonucu kaçırırsa kanıt çifti bir düzeltme olarak öğrenilir;
    doğruysa örnek tekrar edilip yapay epoch oluşturulmaz.
    """
    from nora_lang import GroundedConceptGraph
    rng = np.random.default_rng(int(seed) + 81_773)
    graph = GroundedConceptGraph(chem=None, seed=int(seed))
    graph.clear_all_edges()
    relations = ["ISA", "PART_OF", "CAUSES", "IF_THEN", "COMPOSED_OF",
                 "LOCATED_AT", "HAS_PART", "CAN_DO"]
    rule_counts: dict[str, int] = {}
    attempts = failures = already = 0
    max_attempts = int(max_attempts or max(1000, examples * 40))
    alphabet = np.asarray(list("abcdefghijklmnpqrstuvwxyz"))

    def nonce(index: int) -> str:
        letters = "".join(rng.choice(alphabet, size=5).tolist())
        return f"n{index}{letters}"

    while proposer.corrections < int(examples) and attempts < max_attempts:
        attempts += 1
        nodes = [nonce(i) for i in range(4)]
        rng.shuffle(nodes)
        memory = []
        for index in range(3):
            left, right = nodes[index], nodes[index + 1]
            if rng.random() < 0.5:
                left, right = right, left
            rel = str(rng.choice(relations))
            memory.append((left, rel, right))
        # Bazı bağlantısız/aykırı gözlemler dikkat seçimini kolay ezber olmaktan çıkarır.
        if rng.random() < 0.7:
            memory.append((str(rng.choice(nodes)), str(rng.choice(relations)),
                           str(rng.choice(nodes))))
        memory = list(dict.fromkeys(memory))
        if len(memory) < 2:
            continue
        graph.clear_all_edges()
        for subj, rel, obj in memory:
            graph.add_edge(subj, rel, obj, source="doğuştan")
        index_of = {triple: i for i, triple in enumerate(memory)}
        valid = []
        candidate_relations = sorted({row[1] for row in memory})
        for query in nodes:
            for rel in candidate_relations:
                for obj in nodes:
                    if obj == query or (query, rel, obj) in index_of:
                        continue
                    proof = graph.prove_candidate(query, rel, obj)
                    if proof.get("status") != "proved" or len(proof.get("proof") or []) != 2:
                        continue
                    pair = []
                    for edge in proof["proof"]:
                        pair.append(index_of[(edge.subj, edge.rel, edge.obj)])
                    if pair[0] == pair[1]:
                        continue
                    valid.append((rule_counts.get(proof["rule"], 0), rng.random(),
                                  query, rel, obj, tuple(pair), proof["rule"]))
        if not valid:
            continue
        valid.sort(key=lambda row: (row[0], row[1]))
        _count, _tie, query, rel, obj, pair, rule = valid[0]
        target = (query, rel, obj)
        before = proposer.propose(query, memory=memory, k=5)
        if before and (before[0]["subj"], before[0]["rel"], before[0]["obj"]) == target:
            already += 1
            proposer.successful_proposals += 1
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
            continue
        failures += 1
        proposer.failures += 1
        proposer.learn_correction(memory, query, target, pair)
        rule_counts[rule] = rule_counts.get(rule, 0) + 1
    return {
        "attempts": attempts, "corrections": proposer.corrections,
        "failed_before_correction": failures, "already_top1": already,
        "rule_coverage": dict(rule_counts),
        "complete": proposer.corrections >= int(examples),
        "external_concepts": 0, "heldout_concepts_used": False,
    }


def measure_relational_transfer(proposer: NeuralRelationalProposer, *,
                                graphs: int = 40, seed: int = 0) -> dict:
    """Hiç düzeltme yapmadan yeni nonce grafiklerde top-1/top-5 aktarımı ölç.

    Üçlünün rastgele üretilmesiyle, nöral dikkatin seçtiği iki dayanağın gerçek
    kanıt çifti olması ayrılır. Ayrıca üretim sırasındaki ilk kanıtlı/yeni aday
    ve onun dayanak eşleşmesi raporlanır. Bu fonksiyon ağırlık, bellek veya
    ``corrections`` sayacını değiştirmez; held-out grafikler eğitime geri yazılmaz.
    """
    from nora_lang import GroundedConceptGraph
    before_corrections = int(proposer.corrections)
    before_memory = list(proposer.memory)
    rng = np.random.default_rng(int(seed) + 181_243)
    graph = GroundedConceptGraph(chem=None, seed=int(seed))
    graph.clear_all_edges()
    relations = ["ISA", "PART_OF", "CAUSES", "IF_THEN", "COMPOSED_OF",
                 "LOCATED_AT", "HAS_PART", "CAN_DO"]
    alphabet = np.asarray(list("abcdefghijklmnpqrstuvwxyz"))

    def nonce(index: int) -> str:
        letters = "".join(rng.choice(alphabet, size=5).tolist())
        return f"n{index}{letters}"

    stats = {"attempted_graphs": 0, "graphs": 0, "queries": 0, "targets": 0, "top1": 0,
             "top5": 0, "grounded_top1": 0, "grounded_top5": 0,
             "accepted_queries": 0, "grounded_accepted_queries": 0,
             "pair_top1": 0, "pair_top5": 0, "relation_top1": 0, "object_top1": 0,
             "all_components_top1": 0, "pair_rank_sum": 0,
             "relation_rank_sum": 0, "object_rank_sum": 0,
             "query_top1_valid": 0, "query_top1_grounded": 0,
             "query_top5_valid": 0, "query_top5_grounded": 0,
             "multi_target_queries": 0}
    by_rule: dict[str, dict[str, int]] = {}
    for _graph_index in range(max(0, int(graphs))):
        stats["attempted_graphs"] += 1
        nodes = [nonce(i) for i in range(4)]
        rng.shuffle(nodes)
        memory = []
        for index in range(3):
            left, right = nodes[index], nodes[index + 1]
            if rng.random() < 0.5:
                left, right = right, left
            memory.append((left, str(rng.choice(relations)), right))
        if rng.random() < 0.7:
            memory.append((str(rng.choice(nodes)), str(rng.choice(relations)),
                           str(rng.choice(nodes))))
        memory = list(dict.fromkeys(memory))
        graph.clear_all_edges()
        for subj, rel, obj in memory:
            graph.add_edge(subj, rel, obj, source="doğuştan")
        index_of = {triple: i for i, triple in enumerate(memory)}
        valid = []
        for query in nodes:
            for rel in sorted({row[1] for row in memory}):
                for obj in nodes:
                    if obj == query or (query, rel, obj) in index_of:
                        continue
                    proof = graph.prove_candidate(query, rel, obj)
                    if proof.get("status") != "proved" or len(proof.get("proof") or []) != 2:
                        continue
                    pair = [index_of[(edge.subj, edge.rel, edge.obj)]
                            for edge in proof["proof"]]
                    if pair[0] == pair[1]:
                        continue
                    valid.append({
                        "target": (query, rel, obj), "rule": proof["rule"],
                        "pair": tuple(pair),
                        "proof_set": frozenset((edge.subj, edge.rel, edge.obj)
                                               for edge in proof["proof"]),
                    })
        if valid:
            stats["graphs"] += 1
        for query in sorted({row["target"][0] for row in valid}):
            targets = [row for row in valid if row["target"][0] == query]
            candidates = proposer.propose(query, memory=memory, k=5)
            stats["queries"] += 1
            stats["targets"] += len(targets)
            stats["multi_target_queries"] += int(len(targets) > 1)

            def valid_choice(row, target, grounded=False):
                if (row["subj"], row["rel"], row["obj"]) != target["target"]:
                    return False
                return (not grounded or frozenset(map(tuple, row["neural_pair"]))
                        == target["proof_set"])

            if candidates:
                stats["query_top1_valid"] += int(any(
                    valid_choice(candidates[0], target) for target in targets))
                stats["query_top1_grounded"] += int(any(
                    valid_choice(candidates[0], target, grounded=True) for target in targets))
            stats["query_top5_valid"] += int(any(
                valid_choice(row, target) for row in candidates for target in targets))
            stats["query_top5_grounded"] += int(any(
                valid_choice(row, target, grounded=True)
                for row in candidates for target in targets))
            for target in targets:
                rule = by_rule.setdefault(target["rule"], {
                    "targets": 0, "top1": 0, "top5": 0, "grounded_top5": 0})
                rule["targets"] += 1
                diagnosis = proposer.diagnose(memory, query, target["target"], target["pair"])
                for key in ("pair_top1", "pair_top5", "relation_top1", "object_top1",
                            "all_components_top1"):
                    stats[key] += int(bool(diagnosis[key]))
                for key in ("pair_rank", "relation_rank", "object_rank"):
                    stats[key + "_sum"] += int(diagnosis[key])
                matches = [row for row in candidates
                           if (row["subj"], row["rel"], row["obj"]) == target["target"]]
                if matches:
                    stats["top5"] += 1
                    rule["top5"] += 1
                if candidates and (candidates[0]["subj"], candidates[0]["rel"],
                                   candidates[0]["obj"]) == target["target"]:
                    stats["top1"] += 1
                    rule["top1"] += 1
                grounded = [row for row in matches
                            if frozenset(map(tuple, row["neural_pair"])) == target["proof_set"]]
                if grounded:
                    stats["grounded_top5"] += 1
                    rule["grounded_top5"] += 1
                if grounded and grounded[0] is candidates[0]:
                    stats["grounded_top1"] += 1

            accepted_any = False
            grounded_accepted_any = False
            for row in candidates:
                proof = graph.prove_candidate(row["subj"], row["rel"], row["obj"])
                if proof.get("status") == "proved" and proof.get("novel"):
                    accepted_any = True
                    accepted_proof_set = frozenset(
                        (edge.subj, edge.rel, edge.obj) for edge in proof["proof"])
                    if frozenset(map(tuple, row["neural_pair"])) == accepted_proof_set:
                        grounded_accepted_any = True
            stats["accepted_queries"] += int(accepted_any)
            stats["grounded_accepted_queries"] += int(grounded_accepted_any)
    stats["by_rule"] = by_rule
    stats["learning_updates"] = int(proposer.corrections) - before_corrections
    stats["memory_unchanged"] = proposer.memory == before_memory
    stats["top5_rate"] = stats["top5"] / max(1, stats["targets"])
    stats["grounded_top5_rate"] = stats["grounded_top5"] / max(1, stats["targets"])
    stats["accepted_query_rate"] = stats["accepted_queries"] / max(1, stats["queries"])
    for key in ("pair_top1", "pair_top5", "relation_top1", "object_top1",
                "all_components_top1"):
        stats[key + "_rate"] = stats[key] / max(1, stats["targets"])
    for key in ("pair_rank", "relation_rank", "object_rank"):
        stats["mean_" + key] = stats[key + "_sum"] / max(1, stats["targets"])
    for key in ("query_top1_valid", "query_top1_grounded",
                "query_top5_valid", "query_top5_grounded"):
        stats[key + "_rate"] = stats[key] / max(1, stats["queries"])
    stats["mean_targets_per_query"] = stats["targets"] / max(1, stats["queries"])
    return stats


class BrocaNet(nn.Module):
    def __init__(self, vocab: int, d: int = 256, h: int = 512, dropout: float = 0.0,
                 temporal: int = 0, positional: bool = False, output_bias: bool = False,
                 untied_output: bool = False):
        super().__init__()
        self.V = vocab + N_SPECIAL
        self.pad, self.bos, self.eos = vocab, vocab + 1, vocab + 2
        self.emb = nn.Embedding(self.V, d)
        self.out_weight = nn.Parameter(self.emb.weight.detach().clone()) if untied_output else None
        self.out_bias = nn.Parameter(torch.zeros(self.V)) if output_bias else None
        self.pos_emb = nn.Embedding(65, d) if positional else None
        if self.pos_emb is not None:
            nn.init.zeros_(self.pos_emb.weight)
        self.cond = nn.Sequential(nn.Linear(NF + d, h), nn.Tanh())
        self.cond_in = nn.Linear(h, d, bias=False)
        self.gru = nn.GRU(d, h, num_layers=1, batch_first=True)
        self.proj = nn.Linear(h, d)
        self.drop = nn.Dropout(float(dropout))
        self.temporal = nn.ModuleList(
            nn.GRU(h, h, num_layers=1, batch_first=True) for _ in range(int(temporal)))
        self.temporal_proj = nn.ModuleList(
            nn.Linear(h, d, bias=False) for _ in range(int(temporal)))
        for projection in self.temporal_proj:
            nn.init.zeros_(projection.weight)  # ek derinlik başlangıçta eski fonksiyonu korur
        self.h = h

    def encode_cond(self, state: torch.Tensor, anchor_bag: torch.Tensor) -> torch.Tensor:
        """state: (B,NF); anchor_bag: (B,V) çok-sıcak -> (B,h)"""
        n = anchor_bag.sum(1, keepdim=True).clamp(min=1.0)
        a = (anchor_bag @ self.emb.weight) / n
        return self.cond(torch.cat([state, a], dim=1))

    def hidden(self, ids: torch.Tensor, c: torch.Tensor, h0: torch.Tensor | None = None,
               position_offset: int = 0):
        x = self.emb(ids) + self.cond_in(c).unsqueeze(1)
        if self.pos_emb is not None:
            positions = torch.arange(position_offset, position_offset + ids.shape[1],
                                     device=ids.device)
            x = x + self.pos_emb(positions).unsqueeze(0)
        initial = c.unsqueeze(0) if h0 is None else h0
        out, hn = self.gru(x, initial)
        z = self.proj(self.drop(out))
        for layer, projection in zip(self.temporal, self.temporal_proj):
            out, _ = layer(out, initial)
            z = z + projection(self.drop(out))
        return z, hn                                           # (B,L,d); dropout yalnız eğitim kipinde

    def output_weight(self):
        return self.emb.weight if self.out_weight is None else self.out_weight

    def forward(self, ids: torch.Tensor, c: torch.Tensor, h0: torch.Tensor | None = None,
                position_offset: int = 0):
        z, hn = self.hidden(ids, c, h0, position_offset=position_offset)
        logits = z @ self.output_weight().t()
        if self.out_bias is not None:
            logits = logits + self.out_bias
        return logits, hn                                      # bağlı gömme + isteğe bağlı öğrenilen önyargı


class BrocaOrgan:
    MAX_LEN = 40

    def __init__(self, tokenizer: SubwordTokenizer, seed: int = 0, d: int = 256, h: int = 512,
                 dropout: float = 0.0, temporal: int = 0, positional: bool = False,
                 output_bias: bool = False, untied_output: bool = False):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed + 90_001)
        # Üretim ve sampled-loss rastgeleliği global torch RNG'sini tüketmesin;
        # bu sayede organın devam durumu tek checkpoint içinde taşınabilir.
        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(int(seed) + 190_001)
        self.tok = tokenizer
        V = self.tok.tk.get_vocab_size()
        self.net = BrocaNet(V, d, h, dropout=dropout, temporal=temporal, positional=positional,
                            output_bias=output_bias, untied_output=untied_output)
        self.lr = 2.5e-3
        self._opt = None
        self.V = V
        self.freq = torch.zeros(V)
        self.seen_tokens = 0
        self.pos = {"wiki": 0, "chat": 0}          # akışta nereye kadar okundu (devam ederken tekrar okunmaz: epoch yok)
        self.common_ids: torch.Tensor | None = None
        self.punct_ids: set[int] = set()
        self.token_memory: dict | None = None  # açıkça train metninden kurulan episodic retrieval

    @property
    def opt(self):
        if self._opt is None:
            self._opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        return self._opt

    @opt.setter
    def opt(self, value):
        self._opt = value

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
    def learn_batch(self, sentences: list[str], state: np.ndarray, lr: float | None = None,
                    anchor_dropout: float = 0.0, loss_mode: str = "sampled",
                    negative_samples: int = 2048, bpe_dropout: float = 0.0,
                    parameter_anchor: dict[str, torch.Tensor] | None = None,
                    parameter_anchor_strength: float = 0.0,
                    focal_gamma: float = 0.0,
                    bpe_dropout_seed: int | None = None,
                    early_token_weight: float = 1.0,
                    early_token_count: int = 0,
                    frequency_alpha: float = 0.0,
                    frequency_cap: float = 8.0,
                    word_start_weight: float = 1.0) -> float:
        """Tek bir gradyan adımı: bu akış parçası görüldükten sonra bir daha okunmaz (epoch yok).

        ``anchor_dropout`` çapalı ve çapasız dil koşullarını aynı modelde öğrenir.
        Çapasız değerlendirme, yalnız çapayla eğitilmiş bir koşul katmanına dış-dağılım
        sınavı olmamalıdır. Sıfır değeri eski akışın RNG ve gradyan davranışını aynen korur.
        """
        anchor_dropout = float(anchor_dropout)
        if not 0.0 <= anchor_dropout <= 1.0:
            raise ValueError("anchor_dropout [0,1] aralığında olmalı")
        if loss_mode not in {"sampled", "full"}:
            raise ValueError("loss_mode 'sampled' veya 'full' olmalı")
        negative_samples = int(negative_samples)
        if negative_samples < 1:
            raise ValueError("negative_samples en az 1 olmalı")
        bpe_dropout = float(bpe_dropout)
        if not 0.0 <= bpe_dropout < 1.0:
            raise ValueError("bpe_dropout [0,1) aralığında olmalı")
        focal_gamma = float(focal_gamma)
        if focal_gamma < 0 or (focal_gamma and loss_mode != "full"):
            raise ValueError("focal_gamma yalnız full kayıpla sıfır veya pozitif olmalı")
        early_token_weight = float(early_token_weight)
        early_token_count = int(early_token_count)
        if early_token_weight <= 0 or early_token_count < 0:
            raise ValueError("early_token_weight pozitif, early_token_count negatif olamaz")
        if early_token_count and loss_mode != "full":
            raise ValueError("early token weighting yalnız full kayıpla kullanılabilir")
        frequency_alpha = float(frequency_alpha)
        frequency_cap = float(frequency_cap)
        if frequency_alpha < 0 or frequency_cap < 1:
            raise ValueError("frequency_alpha >=0 ve frequency_cap >=1 olmalı")
        if frequency_alpha and loss_mode != "full":
            raise ValueError("frequency weighting yalnız full kayıpla kullanılabilir")
        word_start_weight = float(word_start_weight)
        if word_start_weight <= 0:
            raise ValueError("word_start_weight pozitif olmalı")
        if word_start_weight != 1.0 and loss_mode != "full":
            raise ValueError("word-start weighting yalnız full kayıpla kullanılabilir")
        parameter_anchor_strength = float(parameter_anchor_strength)
        if parameter_anchor_strength < 0 or (parameter_anchor_strength and parameter_anchor is None):
            raise ValueError("pozitif L2-SP cezası için parameter_anchor gerekli")
        if bpe_dropout_seed is not None:
            enc = self.tok.encode_batch_bpe_dropout(sentences, bpe_dropout, bpe_dropout_seed)
        else:
            previous_bpe_dropout = self.tok.tk.model.dropout
            try:
                if bpe_dropout:
                    self.tok.tk.model.dropout = bpe_dropout
                enc = [(e.ids, e.tokens) for e in self.tok.tk.encode_batch(sentences)]
            finally:
                self.tok.tk.model.dropout = previous_bpe_dropout
        rows, bags, word_starts = [], [], []
        for raw_ids, raw_pcs in enc:
            ids, pcs = raw_ids[: self.MAX_LEN], raw_pcs[: self.MAX_LEN]
            if len(ids) < 3:
                continue
            for i in ids:
                self.freq[i] += 1
            self.seen_tokens += len(ids)
            rows.append(ids)
            word_starts.append([piece.startswith("▁") for piece in pcs])
            bag_ids = self._pseudo_anchor_bag(ids, pcs)
            if anchor_dropout and self.rng.random() < anchor_dropout:
                bag_ids = []
            bags.append(bag_ids)
        if not rows:
            return 0.0
        if self.common_ids is None or self.seen_tokens % 2_000_000 < 4000:
            self.update_common()
        L = max(len(r) for r in rows) + 1
        B = len(rows)
        net = self.net
        inp = torch.full((B, L), net.pad, dtype=torch.long)
        tgt = torch.full((B, L), -100, dtype=torch.long)
        word_weights = torch.ones(B, L) if word_start_weight != 1.0 else None
        for b, r in enumerate(rows):
            if word_weights is not None:
                starts = torch.tensor(word_starts[b], dtype=torch.bool)
                word_weights[b, :len(r)] = torch.where(
                    starts, torch.tensor(word_start_weight), torch.tensor(1.0))
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
        if loss_mode == "full":
            mask = tgt != -100
            # Değerlendirilen nesnelle birebir aynı tam-softmax; küçük partilerde
            # örneklemeli kaybın negatif dağılımı/tekilleştirme sapmasını ayırt eder.
            logits = z[mask] @ net.output_weight().t()
            if net.out_bias is not None:
                logits = logits + net.out_bias
            if focal_gamma or early_token_count or frequency_alpha or word_weights is not None:
                token_loss = F.cross_entropy(logits, tgt[mask], reduction="none")
                if focal_gamma:
                    token_loss = ((1.0 - torch.exp(-token_loss)) ** focal_gamma) * token_loss
                weights = word_weights[mask] if word_weights is not None else None
                if early_token_count:
                    positions = torch.arange(tgt.shape[1], device=tgt.device).unsqueeze(0)
                    early_weights = torch.where(positions < early_token_count,
                                                torch.tensor(early_token_weight, device=tgt.device),
                                                torch.tensor(1.0, device=tgt.device)).expand_as(tgt)[mask]
                    weights = early_weights if weights is None else weights * early_weights
                if frequency_alpha:
                    target_ids = tgt[mask]
                    freq_table = self.freq.to(device=target_ids.device, dtype=token_loss.dtype)
                    frequencies = torch.ones_like(target_ids, dtype=token_loss.dtype)
                    valid = target_ids < freq_table.numel()
                    frequencies[valid] = freq_table[target_ids[valid]]
                    rare_weights = (frequencies + 1.0).pow(-frequency_alpha)
                    # EOS'un frekansı tabloda yok; onu partinin ortalama ağırlığıyla nötrle.
                    rare_weights = torch.where(target_ids == net.eos,
                                               rare_weights.mean(), rare_weights)
                    rare_weights = rare_weights / rare_weights.mean().clamp_min(1e-8)
                    rare_weights = rare_weights.clamp(max=frequency_cap)
                    weights = rare_weights if weights is None else weights * rare_weights
                if weights is not None:
                    loss = (token_loss * weights).sum() / weights.sum().clamp_min(1e-8)
                else:
                    loss = token_loss.mean()
            else:
                loss = F.cross_entropy(logits, tgt[mask])
        else:
            loss = self._sampled_loss(z, tgt, K=negative_samples)
        if parameter_anchor_strength:
            squared = None
            count = 0
            for name, parameter in net.named_parameters():
                reference = parameter_anchor[name]
                value = (parameter - reference).square().sum()
                squared = value if squared is None else squared + value
                count += parameter.numel()
            loss = loss + parameter_anchor_strength * squared / max(1, count)
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
        neg = torch.multinomial(q_full, K, replacement=True, generator=self.torch_rng)
        cand = torch.unique(neg)
        output_weight = net.output_weight()
        W = output_weight[cand]
        logits = zt @ W.t() - torch.log(K * q_full[cand])[None, :]
        true = (zt * output_weight[yt]).sum(1) - torch.log(K * q_full[yt])
        if net.out_bias is not None:
            logits = logits + net.out_bias[cand]
            true = true + net.out_bias[yt]
        logits = logits.masked_fill(cand[None, :] == yt[:, None], -1e9)        # kazara isabet
        all_l = torch.cat([true[:, None], logits], dim=1)
        return F.cross_entropy(all_l, torch.zeros(len(yt), dtype=torch.long))

    def learn_online(self, sentences: list[str], state: np.ndarray) -> float:
        """Yaşarken duyulan cümleler: tek geçiş, küçük adım (önceki bilgiyi bozmadan)."""
        return self.learn_batch(sentences, state, lr=3e-4)

    @torch.no_grad()
    def build_token_memory(self, training_sentences: list[str], state: np.ndarray,
                           k: int = 4, temperature: float = 0.1, weight: float = 0.1,
                           forbidden_sentences: list[str] | None = None,
                           domain_gate: bool = True) -> dict:
        """Train gizli durum→sonraki-parça kayıtlarından küçük kNN-LM belleği kur.

        Bu bellek anlama yerine geçmez ve saf nöral sonuç diye raporlanmaz. Held-out
        metin ``forbidden_sentences`` ile verildiğinde tek bir tam-cümle sızıntısı bile
        kurulumu reddeder. Değerlendirme durumu, kayıt durumuyla aynı olmak zorundadır.
        """
        if not 0.0 <= float(weight) <= 1.0 or float(temperature) <= 0 or int(k) < 1:
            raise ValueError("geçersiz token-bellek k/temperature/weight değeri")
        forbidden = set(forbidden_sentences or [])
        overlap = forbidden.intersection(training_sentences)
        if overlap:
            raise ValueError(f"token memory has {len(overlap)} held-out sentence leaks")
        net = self.net
        was_training = net.training
        net.eval()
        enc = self.tok.tk.encode_batch(training_sentences)
        zs, ys = [], []
        st = torch.from_numpy(np.tile(state.astype(np.float32), (64, 1)))
        for i in range(0, len(enc), 64):
            chunk = enc[i:i + 64]
            rows = [e.ids[:self.MAX_LEN] for e in chunk if len(e.ids) >= 3]
            if not rows:
                continue
            L = max(map(len, rows)) + 1
            B = len(rows)
            inp = torch.full((B, L), net.pad, dtype=torch.long)
            tgt = torch.full((B, L), -100, dtype=torch.long)
            for b, r in enumerate(rows):
                inp[b, 0] = net.bos
                inp[b, 1:len(r) + 1] = torch.tensor(r)
                tgt[b, :len(r)] = torch.tensor(r)
                tgt[b, len(r)] = net.eos
            bag = torch.zeros(B, net.V)
            z, _ = net.hidden(inp, net.encode_cond(st[:B], bag))
            mask = tgt != -100
            zs.append(F.normalize(z[mask].float(), dim=-1).cpu())
            ys.append(tgt[mask].cpu())
        net.train(was_training)
        if not zs:
            raise ValueError("no eligible training sentence for token memory")
        payload = "\0".join(training_sentences).encode("utf-8")
        z_all = torch.cat(zs)
        nearest = []
        for start in range(0, len(z_all), 256):
            query = z_all[start:start + 256]
            similarity = query @ z_all.t()
            similarity[torch.arange(len(query)), torch.arange(start, start + len(query))] = -2.0
            nearest.append(similarity.max(dim=-1).values)
        gate_threshold = float(torch.cat(nearest).median()) if domain_gate else None
        self.token_memory = {
            "z": z_all, "target": torch.cat(ys),
            "state": np.asarray(state, dtype=np.float32).copy(),
            "k": int(k), "temperature": float(temperature), "weight": float(weight),
            "domain_gate": bool(domain_gate), "gate_threshold": gate_threshold,
            "source_sha256": hashlib.sha256(payload).hexdigest(),
            "source_sentences": len(training_sentences),
        }
        return {"records": int(self.token_memory["target"].numel()),
                "source_sentences": len(training_sentences),
                "source_sha256": self.token_memory["source_sha256"],
                "k": int(k), "temperature": float(temperature), "weight": float(weight),
                "domain_gate": bool(domain_gate), "gate_threshold": gate_threshold}

    @torch.no_grad()
    def evaluate(self, sentences: list[str], state: np.ndarray, k: int = 5, max_sent: int = 800,
                 use_anchor: bool = True, use_token_memory: bool = False) -> dict:
        """Görülmemiş cümlelerde bir sonraki parçayı tahmin (çapa verilerek; n-gram ile aynı ölçü)."""
        net = self.net
        memory = self.token_memory if use_token_memory else None
        if use_token_memory and memory is None:
            raise ValueError("use_token_memory=True için önce train belleği kurulmalı")
        if memory is not None and not np.array_equal(
                np.asarray(state, dtype=np.float32), memory["state"]):
            raise ValueError("token memory requires its construction-time internal state")
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
            z, _ = net.hidden(inp, net.encode_cond(st, bag))
            logits = z @ net.output_weight().t()
            if net.out_bias is not None:
                logits = logits + net.out_bias
            mask = tgt != -100
            yt = tgt[mask]
            scores = logits[mask]
            if memory is not None:
                query = F.normalize(z[mask].float(), dim=-1)
                similarity = query @ memory["z"].t()
                values, indices = similarity.topk(min(memory["k"], similarity.shape[1]), dim=-1)
                weights = F.softmax(values / memory["temperature"], dim=-1)
                retrieved = torch.zeros_like(scores, dtype=torch.float32)
                retrieved.scatter_add_(1, memory["target"][indices], weights)
                effective_weight = memory["weight"]
                if memory.get("domain_gate"):
                    threshold = memory.get("gate_threshold")
                    if threshold is not None:
                        effective_weight = effective_weight * (
                            values.max(dim=-1, keepdim=True).values >= float(threshold)).float()
                scores = ((1.0 - effective_weight) * F.softmax(scores.float(), dim=-1)
                          + effective_weight * retrieved + 1e-30).log()
            top = scores.topk(k, dim=-1).indices
            hit1 += int((top[..., 0] == yt).sum())
            hitk += int((top == yt.unsqueeze(-1)).any(-1).sum())
            n += int(mask.sum())
            nll += float(-scores.gather(1, yt[:, None]).sum()) if memory is not None else float(
                F.cross_entropy(scores, yt, reduction="sum"))
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
        for position in range(max_len):
            logits, h = net(cur, c, h, position_offset=position)
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
            nxt = torch.multinomial(F.softmax(lg, dim=-1), 1, generator=self.torch_rng)
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

    # ------------------------------------------------------------ anlam-koşullu artikülasyon
    @staticmethod
    def _articulation_terms(value) -> list[str]:
        """Plan alanındaki terimleri güvenli, tekrarsız yüzey listesine çevir."""
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        try:
            iterator = iter(value)
        except TypeError:
            iterator = iter([value])
        terms: list[str] = []
        for item in iterator:
            text = str(item or "").strip()
            if text and text not in terms:
                terms.append(text)
        return terms

    @staticmethod
    def _articulation_words(text: str) -> list[str]:
        normalized = str(text or "").casefold().replace("\u0307", "")
        return [w for w in re.findall(r"[\wçğıöşü]+", normalized, flags=re.UNICODE)
                if any(ch.isalpha() for ch in w)]

    @classmethod
    def _articulation_contains(cls, text: str, term: str) -> bool:
        """Çekimli yüzeyde plan terimini ara; kısa ekleri sınırlı tut."""
        words = cls._articulation_words(text)
        wanted = cls._articulation_words(term)
        if not words or not wanted:
            return False
        if len(wanted) > 1:
            return " ".join(wanted) in " ".join(words)
        needle = wanted[0]
        return any(
            word == needle
            or (len(needle) >= 4 and word.startswith(needle))
            # Kısa Türkçe çapaların yaygın çekimleri (su→suyu/suya)
            # yüzey denetiminde kaybolmasın; uzak birleşik sözcükler alınmaz.
            or (len(needle) >= 2 and word.startswith(needle)
                and 0 < len(word) - len(needle) <= 3)
            for word in words
        )

    @classmethod
    def _articulation_similarity(cls, left: str, right: str) -> float:
        a, b = cls._articulation_words(left), cls._articulation_words(right)
        if not a or not b:
            return 0.0
        return float(SequenceMatcher(None, a, b).ratio())

    @classmethod
    def _articulation_ngram_overlap(cls, left: str, right: str, n: int = 3) -> float:
        a, b = cls._articulation_words(left), cls._articulation_words(right)
        if len(a) < n or len(b) < n:
            return 0.0
        aa = {tuple(a[i:i + n]) for i in range(len(a) - n + 1)}
        bb = {tuple(b[i:i + n]) for i in range(len(b) - n + 1)}
        return len(aa & bb) / max(1, len(aa))

    @classmethod
    def _articulation_surface_valid(cls, text: str, words: list[str]) -> bool:
        """Öğrenilmiş adayın açık yüzey bozukluklarını ele; olgu denetimi yapmaz."""
        stripped = str(text or "").strip()
        if not stripped or len(words) < 2:
            return False
        if stripped.startswith((",", ";", ":", ".", "!", "?")):
            return False
        if stripped.endswith((",", ";", ":")):
            return False
        # ``…`` ve ``...`` meşru; diğer ardışık noktalama üretim artığıdır.
        normalized = stripped.replace("…", ".").replace("...", ".")
        if re.search(r"[,;:!?]\s*[\.,;:!?]|\.\s*[\.,;:!?]", normalized):
            return False
        if any(words[i] == words[i - 1] for i in range(1, len(words))):
            return False
        if len(words) >= 4 and max(words.count(w) for w in set(words)) > 2:
            return False
        if not stripped.endswith("?") and words[-1] in _ARTICULATION_NONFINAL:
            return False
        return True

    @torch.no_grad()
    def articulate(self, state: np.ndarray, anchors: list[str] | None = None,
                   extra_allowed: list[str] | None = None, *, speech_plan: dict | None = None,
                   recent_surfaces=(), n: int = 8, explore: float = 1.0,
                   temperature: float | None = None, top_k: int = 30,
                   max_len: int = 28) -> dict:
        """Zihin planını koruyarak öğrenilmiş dağılımdan bir yüzey seç.

        Bu katman yeni anlam üretmez. Adaylar planın zorunlu/yasaklı terimleri
        ve önceki yüzeylerle benzerlik bakımından yarışır. Zorunlu terimler
        karşılanmıyorsa ``text=None`` döner; üst katman sembolik yedeğe düşer.
        Plan alanları: ``anchors``/``preferred``, ``required``, ``allowed``,
        ``forbidden``, ``avoid``, ``strict`` ve ``temperature``.
        """
        plan = dict(speech_plan or {})
        base_anchors = self._articulation_terms(
            plan.get("anchors", anchors if anchors is not None else []))
        preferred = self._articulation_terms(plan.get("preferred", base_anchors))
        required = self._articulation_terms(
            plan.get("required", plan.get("required_anchors", [])))
        forbidden = self._articulation_terms(plan.get("forbidden", []))
        strict = bool(plan.get("strict", bool(required)))
        # Tercih edilen veya zorunlu bir terim çapa listesinde yoksa bile
        # yalnızca zihnin verdiği bu plan kapsamında üretilebilsin. Bunları
        # ``anchors``a katmayız: tercih edilen terimler zorla tüketilmez.
        extra = (self._articulation_terms(extra_allowed)
                 + self._articulation_terms(plan.get("allowed"))
                 + preferred + required)
        extra = list(dict.fromkeys(extra))
        temp = float(temperature if temperature is not None
                     else plan.get("temperature", 0.7) * max(0.05, float(explore)))
        temp = max(0.05, temp)
        count = max(8, min(24, int(n) + 4))
        sequences = [list(q) for q in self.generate(
            state, base_anchors, extra, temperature=temp, top_k=top_k,
            max_len=max_len, n=count,
        ) if len(q) >= 2]
        rejected = {"empty": 0, "surface": 0, "required": 0, "forbidden": 0}
        rows: list[dict] = []
        for sequence in sequences:
            text = str(self.detok(sequence) or "").strip()
            words = self._articulation_words(text)
            if len(words) < 2:
                rejected["empty"] += 1
                continue
            if not self._articulation_surface_valid(text, words):
                rejected["surface"] += 1
                continue
            coverage = (sum(self._articulation_contains(text, term) for term in required)
                        / max(1, len(required)))
            if strict and required and coverage < 1.0:
                rejected["required"] += 1
                continue
            if any(self._articulation_contains(text, term) for term in forbidden):
                rejected["forbidden"] += 1
                continue
            rows.append({"sequence": sequence, "text": text, "words": words,
                         "required_coverage": float(coverage)})
        result = {
            "text": None, "sequence": [], "grounded": False,
            "coverage": 0.0, "novelty": 0.0, "fluency": None,
            "candidate_count": len(sequences), "accepted_count": len(rows),
            "rejected": rejected,
            "plan": {"act": str(plan.get("act", "")), "required": required,
                     "preferred": preferred, "strict": strict},
            "model_seen": int(getattr(self, "seen_tokens", 0) or 0),
        }
        if not rows:
            return result

        try:
            raw_fluency = np.asarray(
                self.logprob([row["sequence"] for row in rows], state, base_anchors),
                dtype=np.float64,
            )
            if raw_fluency.shape != (len(rows),) or not np.isfinite(raw_fluency).all():
                raise ValueError("nonfinite articulation fluency")
            fluency = np.clip((raw_fluency - raw_fluency.mean()) /
                              (raw_fluency.std() + 1e-6), -2.0, 2.0)
        except Exception:
            raw_fluency = np.zeros(len(rows), dtype=np.float64)
            fluency = raw_fluency.copy()

        recent = [str(x) for x in list(recent_surfaces) + self._articulation_terms(plan.get("avoid"))
                  if str(x).strip()]
        best = None
        for index, row in enumerate(rows):
            text = row["text"]
            words = row["words"]
            preferred_coverage = (sum(self._articulation_contains(text, term) for term in preferred)
                                  / max(1, len(preferred)))
            similarity = max((self._articulation_similarity(text, old) for old in recent),
                             default=0.0)
            overlap = max((self._articulation_ngram_overlap(text, old) for old in recent),
                          default=0.0)
            exact = any(" ".join(words) == " ".join(self._articulation_words(old))
                        for old in recent)
            unique = len(set(words)) / max(1, len(words))
            ending = float(text.rstrip().endswith((".", "!", "?", "…")))
            novelty = 1.0 - similarity
            # Akıcılık birincil ölçüt; çeşitlilik yalnız yakın yüzey
            # tekrarını cezalandırır. Unique/novelty uğruna düşük olasılıklı,
            # bozuk bir yüzeyi ödüllendirmek anlam-artikülasyon ayrımını bozar.
            score = (2.0 * preferred_coverage + 0.35 * unique + 0.6 * ending
                     + 1.05 * float(fluency[index]) + 0.025 * min(len(words), 12)
                     + 0.60 * novelty - 1.0 * overlap - (2.0 if exact else 0.0)
                     + 0.05 * float(self.rng.gumbel()))
            if required:
                score += 2.0 * row["required_coverage"]
            row.update({"score": float(score), "preferred_coverage": float(preferred_coverage),
                        "novelty": float(novelty), "fluency": float(raw_fluency[index]),
                        "exact_repeat": bool(exact)})
            if best is None or row["score"] > best["score"]:
                best = row
        assert best is not None
        result.update({
            "text": best["text"], "sequence": best["sequence"], "grounded": True,
            "coverage": (best["required_coverage"] if required else best["preferred_coverage"]),
            "novelty": best["novelty"], "fluency": best["fluency"],
            "accepted_count": len(rows),
            "candidate_texts": [row["text"] for row in rows],
            "fluency_values": [float(x) for x in raw_fluency],
            "selected_score": best["score"],
        })
        return result

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

    # ------------------------------------------------------------ kapasite / kalıcılık
    def add_untied_output(self) -> dict:
        """Çıktı gömmesini girdinin birebir kopyasıyla ayır; başlangıç fonksiyonu korunur."""
        if self.net.out_weight is not None:
            return {"added": 0, "untied_output": True}
        before = sum(p.numel() for p in self.net.parameters())
        self.net.out_weight = nn.Parameter(self.net.emb.weight.detach().clone())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        after = sum(p.numel() for p in self.net.parameters())
        return {"before": before, "after": after, "added": after - before,
                "untied_output": True}

    def add_output_bias(self) -> dict:
        """Sıfır başlangıçlı öğrenilebilir sözlük önyargısı ekler; ilk logitler korunur."""
        if self.net.out_bias is not None:
            return {"added": 0, "output_bias": True}
        before = sum(p.numel() for p in self.net.parameters())
        self.net.out_bias = nn.Parameter(torch.zeros(self.net.V))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        after = sum(p.numel() for p in self.net.parameters())
        return {"before": before, "after": after, "added": after - before,
                "output_bias": True}

    def add_positional_embedding(self) -> dict:
        """Erken-parça modellemesi için sıfır-başlangıçlı konum gömmesi ekle."""
        if self.net.pos_emb is not None:
            raise ValueError("positional embedding already exists")
        before = sum(p.numel() for p in self.net.parameters())
        self.net.pos_emb = nn.Embedding(65, self.net.emb.embedding_dim)
        nn.init.zeros_(self.net.pos_emb.weight)
        after = sum(p.numel() for p in self.net.parameters())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        return {"before": before, "after": after, "added": after - before}

    def add_temporal_layer(self, seed: int = 0) -> dict:
        """Sıfır-izdüşümlü artık temporal katman ekle; mevcut logitler korunur."""
        net = self.net
        rng_state = torch.get_rng_state()
        torch.manual_seed(int(seed))
        layer = nn.GRU(net.h, net.h, num_layers=1, batch_first=True)
        projection = nn.Linear(net.h, net.emb.embedding_dim, bias=False)
        torch.set_rng_state(rng_state)
        nn.init.zeros_(projection.weight)
        before = sum(p.numel() for p in net.parameters())
        net.temporal.append(layer)
        net.temporal_proj.append(projection)
        after = sum(p.numel() for p in net.parameters())
        self.opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        return {"before": before, "after": after, "added": after - before,
                "temporal_layers": len(net.temporal)}

    def expand_hidden(self, h: int, seed: int = 0) -> dict:
        """GRU genişliğini fonksiyonu koruyarak büyüt; yeni birimler eğitimle açılır.

        Eski→yeni gizli tekrar bağları ve yeni birimlerin giriş/çıkış izdüşümleri
        sıfırlanır. Böylece genişleme anındaki logitler eski modelle aynıdır; ek
        parametreler yalnız sayı şişirmek için değil, sonraki gradyanlarda kullanılır.
        """
        old = self.net
        h = int(h)
        if len(old.temporal):
            raise ValueError("hidden expansion does not support temporal layers")
        if h <= old.h:
            raise ValueError("new hidden size must exceed current size")
        rng_state = torch.get_rng_state()
        torch.manual_seed(int(seed))
        new = BrocaNet(old.V - N_SPECIAL, old.emb.embedding_dim, h, dropout=old.drop.p,
                       positional=old.pos_emb is not None, output_bias=old.out_bias is not None,
                       untied_output=old.out_weight is not None)
        torch.set_rng_state(rng_state)
        oh = old.h
        with torch.no_grad():
            new.emb.weight.copy_(old.emb.weight)
            if old.out_weight is not None:
                new.out_weight.copy_(old.out_weight)
            if old.out_bias is not None:
                new.out_bias.copy_(old.out_bias)
            if old.pos_emb is not None:
                new.pos_emb.weight.copy_(old.pos_emb.weight)
            new.cond[0].weight[:oh].copy_(old.cond[0].weight)
            new.cond[0].bias[:oh].copy_(old.cond[0].bias)
            new.cond_in.weight[:, :oh].copy_(old.cond_in.weight)
            new.cond_in.weight[:, oh:].zero_()
            new.proj.weight[:, :oh].copy_(old.proj.weight)
            new.proj.weight[:, oh:].zero_()
            new.proj.bias.copy_(old.proj.bias)
            for gate in range(3):
                ns, os_ = gate * h, gate * oh
                new.gru.weight_ih_l0[ns:ns + oh].copy_(old.gru.weight_ih_l0[os_:os_ + oh])
                new.gru.weight_hh_l0[ns:ns + oh, :oh].copy_(old.gru.weight_hh_l0[os_:os_ + oh])
                new.gru.bias_ih_l0[ns:ns + oh].copy_(old.gru.bias_ih_l0[os_:os_ + oh])
                new.gru.bias_hh_l0[ns:ns + oh].copy_(old.gru.bias_hh_l0[os_:os_ + oh])
                new.gru.weight_hh_l0[ns:ns + oh, oh:].zero_()
        new.train(old.training)
        before = sum(p.numel() for p in old.parameters())
        after = sum(p.numel() for p in new.parameters())
        self.net = new
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        return {"before": before, "after": after, "added": after - before,
                "old_hidden": oh, "new_hidden": h}

    def save(self, path: str):
        tmp = path + ".tmp"
        # Ağırlıklar kadar tek-geçiş öğrenmenin nereden süreceği de önemlidir.
        # Optimizer ve RNG, ayrı checkpoint istemeden aynı organ yükünün içinde
        # taşınır; eski kayıtlarda alan yoksa load geriye uyumlu kalır.
        optimizer = self._opt.state_dict() if self._opt is not None else None
        torch.save({"net": self.net.state_dict(), "freq": self.freq, "seen": self.seen_tokens,
                    "tok": self.tok.to_str(), "pos": self.pos, "dims": (self.net.emb.embedding_dim, self.net.h),
                    "dropout": float(self.net.drop.p), "temporal": len(self.net.temporal),
                    "positional": self.net.pos_emb is not None,
                    "output_bias": self.net.out_bias is not None,
                    "untied_output": self.net.out_weight is not None,
                    "token_memory": self.token_memory,
                    "optimizer": optimizer,
                    "rng": self.rng.bit_generator.state,
                    "torch_rng": self.torch_rng.get_state()}, tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "BrocaOrgan":
        d = torch.load(path, weights_only=False)
        o = cls(SubwordTokenizer.from_str(d["tok"]), d=d["dims"][0], h=d["dims"][1],
                dropout=d.get("dropout", 0.0), temporal=d.get("temporal", 0),
                positional=d.get("positional", "pos_emb.weight" in d["net"]),
                output_bias=d.get("output_bias", "out_bias" in d["net"]),
                untied_output=d.get("untied_output", "out_weight" in d["net"]))
        o.net.load_state_dict(d["net"])
        o.freq = d["freq"]
        o.seen_tokens = int(d["seen"])
        o.pos = d.get("pos", {"wiki": 0, "chat": 0})
        o.token_memory = d.get("token_memory")
        if d.get("rng"):
            o.rng.bit_generator.state = d["rng"]
        if d.get("torch_rng") is not None:
            o.torch_rng.set_state(d["torch_rng"].cpu())
        if d.get("optimizer") is not None:
            # opt özelliği burada mevcut parametrelerle kurulur; state_dict
            # içindeki momentler ve adım sayısı da geri yüklenir.
            o.opt.load_state_dict(d["optimizer"])
        o.update_common()
        return o


class TokenMoEOrgan(BrocaOrgan):
    """Üç gerçek Broca uzmanını token düzeyinde birleştiren çalışma-zamanı organı.

    Hazır cümle/aday sıralamaz: her üretim adımında üç uzmanın next-subword dağılımı
    hesaplanır, güven ve nedensel OOD reddi ile tek dağılım seçilir ve örnekleme bu
    dağılımdan yapılır. OOD kararı ilk üç parça gerçekten üretildikten/gözlendikten
    sonra etkinleşir; ilk üç hedefin kendisine gelecek bilgisi sızdırılmaz.
    """

    def __init__(self, external: BrocaOrgan, position: BrocaOrgan, wide: BrocaOrgan,
                 router: nn.Module, mean: torch.Tensor, std: torch.Tensor,
                 ood_threshold: float, token_ratio: float = 0.75,
                 position_weight: float = 0.65, prefix: int = 3,
                 gate: str = "embedding", nll_threshold: float = 0.0):
        if not (external.V == position.V == wide.V):
            raise ValueError("incompatible MoE expert vocabularies")
        if int(prefix) != 3:
            raise ValueError("the trained prefix router requires three pieces")
        self.external, self.position, self.wide = external, position, wide
        self.router, self.mean, self.std = router, mean, std
        self.ood_threshold = float(ood_threshold)
        self.token_ratio = float(token_ratio)
        self.position_weight = float(position_weight)
        self.prefix = int(prefix)
        if gate not in {"embedding", "nll_prefix"}:
            raise ValueError("MoE gate embedding veya nll_prefix olmalı")
        self.gate = gate
        self.nll_threshold = float(nll_threshold)
        self.tok = position.tok
        self.V = position.V
        # Artikülasyon sıralamasındaki küçük keşif gürültüsü de uzmanın
        # öğrenme RNG'siyle aynı kayıt içinde geri yüklenir.
        self.rng = position.rng
        self.torch_rng = position.torch_rng
        self.common_ids = position.common_ids
        self.token_memory = None

    @classmethod
    def from_manifest(cls, path: str) -> "TokenMoEOrgan":
        with open(path, encoding="utf8") as handle:
            manifest = json.load(handle)
        if manifest.get("kind") != "token_moe_broca":
            raise ValueError("invalid token-MoE manifest")
        root = os.path.join(os.path.dirname(os.path.abspath(path)), manifest.get("directory", ""))

        def resolve(name):
            value = manifest[name]
            return value if os.path.isabs(value) else os.path.join(root, value)

        external, position, wide = [BrocaOrgan.load(resolve(key))
                                    for key in ("external", "position", "wide")]
        saved = torch.load(resolve("router"), weights_only=False)
        router = nn.Linear(saved["router"]["weight"].shape[1], 1)
        router.load_state_dict(saved["router"])
        return cls(external, position, wide, router, saved["mean"], saved["std"],
                   saved["ood_threshold"], token_ratio=saved.get("token_ratio", 0.75),
                   position_weight=saved.get("position_weight", 0.65),
                   prefix=saved.get("prefix", 3), gate=saved.get("gate", "embedding"),
                   nll_threshold=saved.get("nll_threshold", 0.0))

    @property
    def seen_tokens(self):
        return int(self.position.seen_tokens + self.external.seen_tokens)

    @seen_tokens.setter
    def seen_tokens(self, value):
        # BrocaOrgan.save/load mirasının beklediği yazma işlemi; MoE manifestinde kullanılmaz.
        self.position.seen_tokens = int(value)

    @property
    def active_parameters(self):
        return sum(p.numel() for organ in (self.external, self.position, self.wide)
                   for p in organ.net.parameters())

    def _train_mode(self):
        for organ in (self.external, self.position, self.wide):
            organ.net.train()

    def _eval_mode(self):
        for organ in (self.external, self.position, self.wide):
            organ.net.eval()

    @torch.no_grad()
    def _external_prefixes(self, prefixes: list[list[int]]) -> torch.Tensor:
        raw = torch.tensor(prefixes, dtype=torch.long)
        embedded = self.external.net.emb.weight[raw].detach()
        features = ((embedded - self.mean) / self.std).flatten(1)
        return torch.sigmoid(self.router(features)).squeeze(1) < self.ood_threshold

    @torch.no_grad()
    def _nll_external_prefixes(self, prefixes: list[list[int]], bag: torch.Tensor,
                               state: torch.Tensor) -> torch.Tensor:
        """Üç parçalık gözlemin öğretmen-zorlamalı NLL farkı; dördüncü parçaya bakılmaz."""
        self._eval_mode()
        batch = len(prefixes)
        net = self.position.net
        inp = torch.full((batch, self.prefix + 1), net.pad, dtype=torch.long)
        inp[:, 0] = net.bos
        inp[:, 1:] = torch.tensor(prefixes, dtype=torch.long)
        target = torch.tensor(prefixes, dtype=torch.long)
        prefix_probs = []
        for organ in (self.external, self.position, self.wide):
            logits, _ = organ.net(inp, organ.net.encode_cond(state, bag))
            prefix_probs.append(torch.softmax(logits[:, :self.prefix].float(), dim=-1))

        def nll(probs):
            return -probs.clamp_min(1e-30).log().gather(
                -1, target[..., None]).squeeze(-1).sum(1)

        target_p = (self.position_weight * prefix_probs[1]
                    + (1.0 - self.position_weight) * prefix_probs[2])
        return (nll(target_p) - nll(prefix_probs[0])) > self.nll_threshold

    def _select_probability(self, external_logits, position_logits, wide_logits,
                            sentence_external):
        external_p = torch.softmax(external_logits.float(), dim=-1)
        position_p = torch.softmax(position_logits.float(), dim=-1)
        wide_p = torch.softmax(wide_logits.float(), dim=-1)
        target_p = self.position_weight * position_p + (1.0 - self.position_weight) * wide_p
        use_target = (target_p.max(dim=-1).values
                      >= self.token_ratio * external_p.max(dim=-1).values)
        use_target = use_target & ~sentence_external
        return torch.where(use_target[..., None], target_p, external_p) + 1e-30

    @torch.no_grad()
    def generate(self, state: np.ndarray, anchors: list[str], extra_allowed: list[str] | None = None,
                 temperature: float = 0.7, top_k: int = 30, max_len: int = 28, n: int = 1,
                 cover_bonus: float = 4.0) -> list[list[int]]:
        """Her tokenı MoE dağılımından üret; uzman çıktısı hazır cümle olarak sıralanmaz."""
        self._eval_mode()
        net = self.position.net
        words = [self.tok.tk.encode(a).ids for a in list(anchors) + list(extra_allowed or [])
                 if a and a.strip()]
        words = [w for w in words if w]
        starts = {w[0]: w[1:] for w in words}
        anchor_starts = {self.tok.tk.encode(a).ids[0] for a in anchors
                         if a and a.strip() and self.tok.tk.encode(a).ids}
        allowed = torch.zeros(net.V, dtype=torch.bool)
        if self.common_ids is not None:
            allowed[self.common_ids] = True
        allowed[list(starts)] = True
        allowed[net.eos] = True
        bag = torch.zeros(n, net.V)
        for anchor in anchors:
            for token in self.tok.tk.encode(anchor).ids:
                bag[:, token] = 1.0
        st = torch.from_numpy(np.tile(state.astype(np.float32), (n, 1)))
        conditions = [organ.net.encode_cond(st, bag)
                      for organ in (self.external, self.position, self.wide)]
        hidden = [condition.unsqueeze(0) for condition in conditions]
        cur = torch.full((n, 1), net.bos, dtype=torch.long)
        out = [[] for _ in range(n)]
        done = [False] * n
        queue: list[list[int]] = [[] for _ in range(n)]
        remaining = [set(anchor_starts) for _ in range(n)]
        ood_decided = [False] * n
        sentence_external = torch.zeros(n, dtype=torch.bool)
        for position in range(max_len):
            if position >= self.prefix:
                eligible = [b for b in range(n) if not ood_decided[b] and len(out[b]) >= self.prefix]
                if eligible:
                    prefixes = [out[b][:self.prefix] for b in eligible]
                    decisions = (self._nll_external_prefixes(prefixes, bag[eligible], st[eligible])
                                 if self.gate == "nll_prefix" else self._external_prefixes(prefixes))
                    for b, decision in zip(eligible, decisions):
                        sentence_external[b] = bool(decision)
                        ood_decided[b] = True
            logits = []
            for index, organ in enumerate((self.external, self.position, self.wide)):
                value, hidden[index] = organ.net(cur, conditions[index], hidden[index],
                                                 position_offset=position)
                logits.append(value[:, -1, :])
            probability = self._select_probability(*logits, sentence_external)
            lg = probability.log() / max(1e-3, temperature)
            lg[:, ~allowed] = -1e9
            lg[:, net.pad] = lg[:, net.bos] = -1e9
            for b in range(n):
                for token in remaining[b]:
                    lg[b, token] += cover_bonus
                for token in set(out[b]):
                    lg[b, token] -= (3.0 if token in starts else 1.5) * out[b].count(token)
            kth = lg.topk(min(top_k, net.V), dim=-1).values[:, -1:]
            lg[lg < kth] = -1e9
            nxt = torch.multinomial(F.softmax(lg, dim=-1), 1, generator=self.torch_rng)
            for b in range(n):
                if queue[b]:
                    nxt[b, 0] = queue[b].pop(0)
                token = int(nxt[b])
                if done[b]:
                    continue
                if token == net.eos:
                    done[b] = True
                else:
                    if token in starts and not queue[b]:
                        queue[b] = list(starts[token])
                    out[b].append(token)
                    remaining[b].discard(token)
            if all(done) and not any(queue):
                break
            cur = nxt
        self._train_mode()
        return out

    @torch.no_grad()
    def logprob(self, sequences: list[list[int]], state: np.ndarray, anchors: list[str]) -> list[float]:
        if not sequences:
            return []
        self._eval_mode()
        net = self.position.net
        anchor_ids = [token for anchor in anchors for token in self.tok.tk.encode(anchor).ids]
        batch = len(sequences)
        length = max(len(x) for x in sequences) + 1
        inp = torch.full((batch, length), net.pad, dtype=torch.long)
        tgt = torch.full((batch, length), -100, dtype=torch.long)
        for b, row in enumerate(sequences):
            inp[b, 0] = net.bos
            if row:
                inp[b, 1:len(row) + 1] = torch.tensor(row)
                tgt[b, :len(row)] = torch.tensor(row)
            tgt[b, len(row)] = net.eos
        bag = torch.zeros(batch, net.V)
        if anchor_ids:
            bag[:, anchor_ids] = 1.0
        st = torch.from_numpy(np.tile(state.astype(np.float32), (batch, 1)))
        logits = []
        for organ in (self.external, self.position, self.wide):
            value, _ = organ.net(inp, organ.net.encode_cond(st, bag))
            logits.append(value)
        sentence_external = torch.zeros(batch, dtype=torch.bool)
        eligible = [b for b, row in enumerate(sequences) if len(row) >= self.prefix]
        if eligible:
            prefixes = [sequences[b][:self.prefix] for b in eligible]
            decisions = (self._nll_external_prefixes(prefixes, bag[eligible], st[eligible])
                         if self.gate == "nll_prefix" else self._external_prefixes(prefixes))
            for b, decision in zip(eligible, decisions):
                sentence_external[b] = bool(decision)
        observed = torch.arange(length)[None, :] >= self.prefix
        external_mask = (sentence_external[:, None] & observed).expand(batch, length)
        probability = self._select_probability(*logits, external_mask)
        safe_target = tgt.clamp_min(0)
        token_lp = probability.log().gather(-1, safe_target[..., None]).squeeze(-1)
        token_lp[tgt == -100] = 0.0
        self._train_mode()
        return [float(x) for x in token_lp.sum(1) / (tgt != -100).sum(1).clamp(min=1)]

    def save(self, path: str):
        """Uzmanları ayrı disk dosyalarında, küçük manifesti ``path`` altında atomik sakla."""
        root = path + ".moe"
        os.makedirs(root, exist_ok=True)
        for name, organ in (("external", self.external), ("position", self.position),
                            ("wide", self.wide)):
            organ.save(os.path.join(root, name + ".pt"))
        router_path = os.path.join(root, "router.pt")
        torch.save({"router": self.router.state_dict(), "mean": self.mean, "std": self.std,
                    "ood_threshold": self.ood_threshold, "prefix": self.prefix,
                    "token_ratio": self.token_ratio, "position_weight": self.position_weight,
                    "gate": self.gate, "nll_threshold": self.nll_threshold},
                   router_path + ".tmp")
        os.replace(router_path + ".tmp", router_path)
        manifest = {"kind": "token_moe_broca", "directory": os.path.basename(root),
                    "external": "external.pt", "position": "position.pt",
                    "wide": "wide.pt", "router": "router.pt"}
        with open(path + ".tmp", "w", encoding="utf8") as handle:
            json.dump(manifest, handle, indent=2)
        os.replace(path + ".tmp", path)

    def learn_online(self, sentences: list[str], state: np.ndarray) -> float:
        """Yeni metni nedensel prefix kararıyla hedef veya haricî uzmana tek-geçiş öğret."""
        target_rows, external_rows, pending = [], [], []
        for text in sentences:
            ids = self.tok.tk.encode(text).ids[:self.prefix]
            if len(ids) >= self.prefix:
                pending.append((text, ids))
            else:
                target_rows.append(text)
        if pending:
            prefixes = [ids for _, ids in pending]
            if self.gate == "nll_prefix":
                bag = torch.zeros(len(pending), self.position.net.V)
                st = torch.from_numpy(np.tile(state.astype(np.float32), (len(pending), 1)))
                decisions = self._nll_external_prefixes(prefixes, bag, st)
            else:
                decisions = self._external_prefixes(prefixes)
            for (text, _), decision in zip(pending, decisions):
                (external_rows if bool(decision) else target_rows).append(text)
        losses = []
        if target_rows:
            self.position.net.train()
            losses.append(self.position.learn_online(target_rows, state))
        if external_rows:
            self.external.net.train()
            losses.append(self.external.learn_online(external_rows, state))
        return float(np.mean(losses)) if losses else 0.0

    def reinforce(self, sequences: list[list[int]], rewards: list[float], state: np.ndarray,
                  anchors: list[str], solo: bool = False, extra_words: list[str] | None = None) -> float:
        # Ödül, konuşma-dağılımındaki hedef uzmanı günceller; soğuk genel uzman korunur.
        self.position.net.train()
        return self.position.reinforce(sequences, rewards, state, anchors, solo=solo,
                                      extra_words=extra_words)
