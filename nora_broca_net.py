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
import hashlib
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nora_voice import NF, SubwordTokenizer

N_SPECIAL = 3          # PAD, BOS, EOS (sözlüğün sonuna eklenir)


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
