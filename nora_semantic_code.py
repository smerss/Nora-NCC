#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Document-grounded semantics and compositional Python synthesis.

The request parser produces constraints, not a program name.  The document learner
acquires operator meanings from natural-language statements and Python AST examples.
A small planner composes those operators for unseen payloads.  No complete task
solution is stored in the request parser or in the document primer.
"""
from __future__ import annotations

import ast
import base64
import collections
import copy
from dataclasses import dataclass, field
import html
import itertools
import json
import math
from pathlib import Path
import random
import re
import time
from typing import Any
import urllib.request

from nora_lang import TurkishMorphology, tr_lower


_QUOTES = (
    r'"([^"\n]+)"',
    r"“([^”\n]+)”",
    r"«([^»\n]+)»",
    r"‘([^’\n]+)’",
    r"`([^`\n]+)`",
    r"(?<![\wçğıöşü])'([^'\n]{2,})'(?![\wçğıöşü])",
)
_NOMINALS = ("yazısını", "metnini", "ifadesini", "dizesini", "cümlesini")
_DISPLAY = ("yazdır", "yazdir", "bastır", "print", "göster", "çıktı")
_VERTICAL = ("alt alta", "alt satır", "satır satır", "dikey", "her satır")
_CUMULATIVE = ("bir harf daha", "1 harf daha", "bir karakter daha", "1 karakter daha",
               "birer harf", "harf eklen", "artan önek", "birikimli", "kümülatif")


@dataclass(frozen=True)
class SemanticGoal:
    payload: str
    unit: str
    layout: str
    execute: bool
    raw: str
    trace: tuple[str, ...] = field(default_factory=tuple)
    cumulative: bool = False
    cumulative_step: int = 1

    @property
    def key(self) -> tuple[str, str]:
        suffix = "+cumulative" if self.cumulative else ""
        if self.cumulative and self.cumulative_step != 1:
            suffix += f":{self.cumulative_step}"
        return self.unit, self.layout + suffix

    def units(self) -> list[str]:
        if self.cumulative and self.unit == "character":
            step = max(1, int(self.cumulative_step))
            return [self.payload[:index] for index in range(step, len(self.payload) + 1, step)]
        if self.unit == "character":
            return list(self.payload)
        if self.unit == "word":
            return self.payload.split()
        return [self.payload]

    def expected_output(self) -> str:
        if self.layout != "vertical":
            raise ValueError("unsupported output layout")
        units = self.units()
        return "\n".join(units) + ("\n" if units else "")

    def state_dict(self) -> dict:
        return {"payload": self.payload, "unit": self.unit, "layout": self.layout,
                "execute": self.execute, "raw": self.raw, "trace": list(self.trace),
                "cumulative": self.cumulative, "cumulative_step": int(self.cumulative_step)}


@dataclass(frozen=True)
class ProgramPlan:
    operators: tuple[str, ...]
    program: str
    expected: str
    trace: tuple[str, ...]


@dataclass(frozen=True)
class TransformationGoal:
    """Belgede öğrenilmiş bir değer/dizgi dönüşümünün yeni girdiye uygulanması."""

    payload: str
    operation: str
    raw: str
    reference: str = ""
    trace: tuple[str, ...] = field(default_factory=tuple)

    @property
    def key(self) -> tuple[str, str]:
        return "transformation", self.operation


class _FiniteObservationEvidence:
    """Örnek öneklerinden FST adayı ve exhaustif minimalite kanıtı.

    Bu bir temsil sınıfı değildir; yalnız mevcut ``finite_state_transducer``
    ailesi için arama verisidir. İki önek aynı gelecekte farklı çıktı veriyorsa
    aynı gizli durumda olamaz. Gözlem tabloları aday kurmak için bu kısıtı,
    klik/çapa analizi ise bütün minimal makineleri saymak için kullanılır.
    """

    def __init__(self, examples: list[tuple[str, str]], alphabet: list[str]):
        self.examples = list(examples)
        self.alphabet = tuple(alphabet)
        self.output_alphabet = tuple(sorted({
            output for _source, expected in self.examples for output in expected
        }))
        self.residuals: dict[str, dict[str, str]] = {}
        for source, expected in self.examples:
            for depth in range(len(source) + 1):
                prefix, suffix = source[:depth], source[depth:]
                row = self.residuals.setdefault(prefix, {})
                residual = expected[depth:]
                if suffix in row and row[suffix] != residual:
                    raise ValueError("conflicting_duplicate_example")
                row[suffix] = residual
        self.vertices = sorted(self.residuals, key=lambda text: (len(text), text))
        self.index = {text: pos for pos, text in enumerate(self.vertices)}
        self.levels: dict[int, list[str]] = {}
        for text in self.vertices:
            self.levels.setdefault(len(text), []).append(text)
        self.steps: dict[str, dict[str, tuple[str, str]]] = {}
        for prefix in self.vertices:
            row = {}
            for symbol in self.alphabet:
                # Kısa bir önek için tek karakterlik örnek bulunmayabilir.
                # Daha uzun herhangi bir geleceğin ilk çıktısı aynı adımı verir.
                outputs = {
                    residual[0] for suffix, residual in self.residuals[prefix].items()
                    if suffix.startswith(symbol) and residual
                }
                if len(outputs) > 1:
                    raise ValueError("inconsistent_single_step_outputs")
                if outputs and prefix + symbol in self.residuals:
                    row[symbol] = (next(iter(outputs)), prefix + symbol)
            self.steps[prefix] = row
        self.adjacency = [set() for _ in self.vertices]
        self.constraint_checks = 0
        for left in range(len(self.vertices)):
            for right in range(left + 1, len(self.vertices)):
                self.constraint_checks += 1
                if not self.compatible(self.residuals[self.vertices[left]],
                                       self.residuals[self.vertices[right]]):
                    self.adjacency[left].add(right)
                    self.adjacency[right].add(left)
        self.adjacency_bits = [
            sum(1 << other for other in row) for row in self.adjacency
        ]

    @staticmethod
    def compatible(left: dict[str, str], right: dict[str, str]) -> bool:
        small, other = (left, right) if len(left) <= len(right) else (right, left)
        return not any(suffix in other and residual != other[suffix]
                       for suffix, residual in small.items())

    @staticmethod
    def state_count(table: dict[tuple, tuple]) -> int:
        return len({state for state, _symbol in table})

    def run(self, table: dict[tuple, tuple], source: str, *, trace: bool = False):
        state = 0
        states = [0]
        output = []
        for symbol in source:
            transition = table.get((state, symbol))
            if transition is None:
                return None
            state, emitted = transition
            states.append(state)
            output.append(emitted)
        return (states, "".join(output)) if trace else "".join(output)

    def canonical(self, table: dict[tuple, tuple]) -> dict[tuple, tuple] | None:
        """Başlangıçtan BFS etiketle; erişilemeyen/eksik makineleri ele."""
        if not table:
            return None
        states = {state for state, _symbol in table}
        states.update(next_state for next_state, _output in table.values())
        mapping = {0: 0}
        queue = [0]
        result = {}
        for state in queue:
            for symbol in self.alphabet:
                transition = table.get((state, symbol))
                if transition is None:
                    return None
                next_state, output = transition
                if next_state not in mapping:
                    mapping[next_state] = len(mapping)
                    queue.append(next_state)
                result[mapping[state], symbol] = (mapping[next_state], output)
        if len(mapping) != len(states) or len(result) != len(table):
            return None
        return result

    def candidates(self, depth: int, state_cap: int, expansion_limit: int):
        """Bir gözlem derinliğindeki küçük geçiş belirsizliklerini say."""
        prefixes = self.levels.get(int(depth), [])
        if not prefixes or not self.alphabet or expansion_limit <= 0:
            return [], 0, False
        common = set(self.residuals[prefixes[0]])
        for prefix in prefixes[1:]:
            common.intersection_update(self.residuals[prefix])
        if not common:
            return [], 0, False
        suffixes = sorted(common, key=lambda text: (len(text), text))
        groups = {}
        base_class = {}
        base_members = []
        for prefix in prefixes:
            row = tuple(self.residuals[prefix][suffix] for suffix in suffixes)
            if row not in groups:
                groups[row] = len(groups)
                base_members.append([])
            base_class[prefix] = groups[row]
            base_members[groups[row]].append(prefix)
        if len(base_members) > state_cap:
            return [], 0, False

        keys, outputs, options = [], [], []
        total_choices = 1
        for source_class, members in enumerate(base_members):
            for symbol in self.alphabet:
                choices = None
                emitted = None
                for prefix in members:
                    step = self.steps[prefix].get(symbol)
                    if step is None:
                        continue
                    output, child = step
                    if emitted is None:
                        emitted = output
                    elif emitted != output:
                        return [], 0, False
                    possible = {
                        target_class
                        for target_class, targets in enumerate(base_members)
                        if all(self.compatible(self.residuals[child],
                                               self.residuals[target])
                               for target in targets)
                    }
                    choices = possible if choices is None else choices & possible
                    if not choices:
                        return [], 0, False
                if emitted is None or not choices:
                    return [], 0, False
                keys.append((source_class, symbol))
                outputs.append(emitted)
                options.append(sorted(choices))
                total_choices *= len(choices)
                if total_choices > expansion_limit:
                    return [], 0, True

        found = {}
        expansions = 0
        for targets in itertools.product(*options):
            if expansions >= expansion_limit:
                return list(found.values()), expansions, True
            expansions += 1
            transitions = {
                key: (target, output)
                for key, target, output in zip(keys, targets, outputs)
            }
            prefix_class = dict(base_class)
            members = [list(row) for row in base_members]
            invalid = False
            # Bilinen sınıfları başlangıca doğru katla. Yeni sınıf yalnız
            # mevcut hiçbir deterministik imzaya uymuyorsa açılır.
            for level in range(depth - 1, -1, -1):
                signatures = {}
                for source_class in range(len(members)):
                    signature = tuple(
                        (symbol, transitions[source_class, symbol][1],
                         transitions[source_class, symbol][0])
                        for symbol in self.alphabet
                    )
                    if signature in signatures and signatures[signature] != source_class:
                        invalid = True
                        break
                    signatures[signature] = source_class
                if invalid:
                    break
                for prefix in self.levels.get(level, []):
                    if len(self.steps[prefix]) != len(self.alphabet):
                        invalid = True
                        break
                    signature = tuple(
                        (symbol, self.steps[prefix][symbol][0],
                         prefix_class[self.steps[prefix][symbol][1]])
                        for symbol in self.alphabet
                    )
                    if signature not in signatures:
                        source_class = len(members)
                        if source_class >= state_cap:
                            invalid = True
                            break
                        signatures[signature] = source_class
                        members.append([prefix])
                        for symbol, output, target in signature:
                            transitions[source_class, symbol] = (target, output)
                    prefix_class[prefix] = signatures[signature]
                if invalid:
                    break
            if invalid:
                continue
            initial = prefix_class[""]
            mapping = {initial: 0}
            queue = [initial]
            try:
                for source_class in queue:
                    for symbol in self.alphabet:
                        target = transitions[source_class, symbol][0]
                        if target not in mapping:
                            mapping[target] = len(mapping)
                            queue.append(target)
            except KeyError:
                continue
            if len(mapping) != len(members):
                continue
            table = {
                (mapping[source], symbol): (mapping[target], output)
                for (source, symbol), (target, output) in transitions.items()
            }
            canonical = self.canonical(table)
            if canonical is None or self.state_count(canonical) > state_cap:
                continue
            if any(self.run(canonical, source) != expected
                   for source, expected in self.examples):
                continue
            found[tuple(sorted(canonical.items()))] = canonical
        return list(found.values()), expansions, False

    def _strongly_connected(self, table: dict[tuple, tuple], states: int) -> bool:
        adjacency = {state: set() for state in range(states)}
        for (state, _symbol), (target, _output) in table.items():
            adjacency[state].add(target)
        for initial in range(states):
            seen = {initial}
            queue = [initial]
            for state in queue:
                for target in adjacency[state]:
                    if target not in seen:
                        seen.add(target)
                        queue.append(target)
            if len(seen) != states:
                return False
        return True

    def clique_anchors(self, states: int, expansion_limit: int):
        """Greedy-coloring üst sınırlı, tam geri izlemeli clique araması.

        Dönüş: ``(çapalar, arama_düğümü, coloring_ziyareti, bütçe_aşıldı)``.
        Coloring ziyaretleri pairwise constraint-check türündendir ve arama
        expansion'larından ayrı raporlanır. Bütçe aşılmadan clique bulunamazsa
        geri izleme tamamlandığı için sonuç eksiksizdir.
        """
        states = int(states)
        if states <= 0 or states > len(self.vertices):
            return None, 0, 0, False
        adjacency = self.adjacency_bits
        nodes = visits = 0
        found = None
        capped = False

        def color_sort(candidates):
            nonlocal visits
            order, bounds = [], []
            color = 0
            remaining = candidates
            while remaining:
                color += 1
                available = remaining
                while available:
                    visits += 1
                    bit = available & -available
                    vertex = bit.bit_length() - 1
                    available -= bit
                    order.append(vertex)
                    bounds.append(color)
                    remaining -= bit
                    available &= ~adjacency[vertex]
            return order, bounds

        def expand(candidates, clique):
            nonlocal nodes, found, capped
            if found is not None or capped:
                return
            if len(clique) == states:
                found = clique
                return
            order, bounds = color_sort(candidates)
            # Aynı greedy-color sınırı içindeki sıralama tamlığı değiştirmez.
            # Derin/tek-residual çapalar yerine kanıtı zengin prefixleri önce dene.
            decorated = sorted(zip(order, bounds), key=lambda row: (
                row[1], len(self.residuals[self.vertices[row[0]]]),
                len(self.steps[self.vertices[row[0]]]), -len(self.vertices[row[0]]),
                self.vertices[row[0]],
            ))
            order = [vertex for vertex, _bound in decorated]
            bounds = [bound for _vertex, bound in decorated]
            for index in range(len(order) - 1, -1, -1):
                if found is not None or capped:
                    return
                if len(clique) + bounds[index] < states:
                    return
                vertex = order[index]
                bit = 1 << vertex
                if not candidates & bit:
                    continue
                nodes += 1
                if nodes > expansion_limit:
                    capped = True
                    return
                expand(candidates & adjacency[vertex], clique + [vertex])
                candidates &= ~bit

        expand((1 << len(self.vertices)) - 1, [])
        return found, nodes, visits, capped

    def prove(self, initial_table: dict[tuple, tuple] | None, expansion_limit: int,
              *, anchor_vertices: list[int] | None = None,
              selection_strategy: str = "recurrent", selection_seed: int = 0,
              exhaustive_labelings: bool = False, return_candidates: bool = False) -> dict:
        """Bütün aynı-boyutlu makineleri sabit clique çapaları altında say.

        Klik, daha az durumun mümkün olmadığını kanıtlar. Zorunlu geçişler
        yayılır; kalan geçişler ve çapa permütasyonu exhaustively denenir.
        Birden çok minimal çözümde seçim yalnız açık stratejiyle yapılır.
        """
        if selection_strategy not in {"recurrent", "uniform-random", "lexicographic", "mdl"}:
            raise ValueError("unknown_selection_strategy")
        table = self.canonical(initial_table) if initial_table is not None else None
        if initial_table is not None and table is None:
            return {"status": "unresolved", "expansions": 0}
        prefix_states = []
        if table is not None:
            states = self.state_count(table)
            prefix_states = [self.run(table, prefix, trace=True)[0][-1]
                             for prefix in self.vertices]
            by_state = [[] for _ in range(states)]
            for vertex, state in enumerate(prefix_states):
                by_state[state].append(vertex)
            order = sorted(range(states), key=lambda state: len(by_state[state]))

            def clique(position, anchors):
                if position == states:
                    return anchors
                for vertex in by_state[order[position]]:
                    if all((self.adjacency_bits[vertex] >> anchor) & 1
                           for anchor in anchors):
                        found = clique(position + 1, anchors + [vertex])
                        if found is not None:
                            return found
                return None

            found = clique(0, [])
        else:
            found = list(anchor_vertices or [])
            states = len(found)
        if not found:
            return {"status": "unresolved", "reason": "no-clique", "expansions": 0}
        if (len(set(found)) != states or any(
                not (self.adjacency_bits[left] >> right) & 1
                for i, left in enumerate(found) for right in found[i + 1:])):
            return {"status": "unresolved", "reason": "invalid-clique", "expansions": 0}
        full_domain = (1 << states) - 1
        anchors = [None] * states
        if table is not None:
            for vertex in found:
                anchors[prefix_states[vertex]] = vertex
        else:
            anchors = found

        valid = set()
        permutations = completions = completion_nodes = validations = max_missing = 0
        # Sabit clique çapalarında yalnız başlangıcın hangi çapa olduğu önemlidir.
        # Kalan durum adları kanonik BFS ile normalize edilir; N! yerine en fazla
        # N etiketleme bütün izomorfizm sınıflarını eksiksiz kapsar.
        if exhaustive_labelings:
            # Bağımsız denetim kapısı: kanonik azaltmanın aday kümesini N! ile
            # karşılaştırmak için kullanılır; üretim varsayılanı değildir.
            labelings = itertools.permutations(range(states))
            labeling_space = "all-anchor-permutations"
        else:
            root_vertex = self.index[""]
            labelings = []
            for root_anchor in range(states):
                if (self.adjacency_bits[root_vertex] >> anchors[root_anchor]) & 1:
                    continue
                permutation = [None] * states
                permutation[root_anchor] = 0
                others = [index for index in range(states) if index != root_anchor]
                for index, state in zip(others, range(1, states)):
                    permutation[index] = state
                labelings.append(tuple(permutation))
            labeling_space = "root-anchor-canonical"
        validation_trie = [{}]
        for source, expected in self.examples:
            node = 0
            for symbol, emitted in zip(source, expected):
                key = (symbol, emitted)
                if key not in validation_trie[node]:
                    validation_trie[node][key] = len(validation_trie)
                    validation_trie.append({})
                node = validation_trie[node][key]
        for permutation in labelings:
            permutations += 1
            if permutations + completion_nodes > expansion_limit:
                return {"status": "budget-exceeded",
                        "expansions": permutations + completion_nodes}
            domains = []
            inconsistent = False
            for vertex in range(len(self.vertices)):
                forbidden = 0
                for candidate_state, anchor in enumerate(anchors):
                    if (self.adjacency_bits[vertex] >> anchor) & 1:
                        forbidden |= 1 << permutation[candidate_state]
                domain = full_domain & ~forbidden
                if vertex == self.index[""]:
                    domain &= 1  # FST başlangıcı her zaman durum 0.
                if not domain:
                    inconsistent = True
                    break
                domains.append(domain)
            if inconsistent:
                continue

            forced = {}
            changed = True
            while changed and not inconsistent:
                changed = False
                for vertex, prefix in enumerate(self.vertices):
                    domain = domains[vertex]
                    if not domain or domain & (domain - 1):
                        continue
                    source_state = domain.bit_length() - 1
                    for symbol, (output, child) in self.steps[prefix].items():
                        child_vertex = self.index[child]
                        key = (source_state, symbol)
                        if key in forced:
                            target = forced[key][0]
                            narrowed = domains[child_vertex] & (1 << target)
                            if not narrowed:
                                inconsistent = True
                                break
                            if narrowed != domains[child_vertex]:
                                domains[child_vertex] = narrowed
                                changed = True
                        elif not (domains[child_vertex] & (domains[child_vertex] - 1)):
                            target = domains[child_vertex].bit_length() - 1
                            forced[key] = (target, output)
                            changed = True
                    if inconsistent:
                        break
            if inconsistent:
                continue

            missing = [
                (state, symbol) for state in range(states) for symbol in self.alphabet
                if (state, symbol) not in forced
            ]
            max_missing = max(max_missing, len(missing))
            options = []
            support = []
            for state, symbol in missing:
                definite = [
                    vertex for vertex, prefix in enumerate(self.vertices)
                    if domains[vertex] == 1 << state and symbol in self.steps[prefix]
                ]
                sources = definite or [
                    vertex for vertex, prefix in enumerate(self.vertices)
                    if (domains[vertex] >> state) & 1 and symbol in self.steps[prefix]
                ]
                if not sources:
                    choices = {
                        (target, output) for target in range(states)
                        for output in self.output_alphabet
                    }
                else:
                    choices = None
                    for vertex in sources:
                        output, child = self.steps[self.vertices[vertex]][symbol]
                        child_domain = domains[self.index[child]]
                        current = {
                            (target, output) for target in range(states)
                            if (child_domain >> target) & 1
                        }
                        if choices is None:
                            choices = current
                        elif definite:
                            choices.intersection_update(current)
                        else:
                            choices.update(current)
                if not choices:
                    inconsistent = True
                    break
                options.append(sorted(choices))
                support.append(len(sources))
            if inconsistent:
                continue

            option_map = dict(zip(missing, options))
            support_map = dict(zip(missing, support))
            capped = False

            def inspect(candidate, remaining):
                """Bilinen hataları ve ilk-eksik geçişlerin yol desteğini bul."""
                pending = [(0, 0)]  # (girdi-trie düğümü, FST durumu)
                missing_support = {}
                while pending:
                    node, state = pending.pop()
                    for (symbol, expected), child in validation_trie[node].items():
                        key = (state, symbol)
                        transition = candidate.get(key)
                        if transition is None:
                            if key in remaining:
                                missing_support[key] = missing_support.get(key, 0) + 1
                            continue  # Bu dalın devamı henüz bilinmiyor; yanlış sayma.
                        target, emitted = transition
                        if emitted != expected:
                            return False, missing_support
                        pending.append((child, target))
                return True, missing_support

            def complete(remaining, candidate):
                nonlocal completions, completion_nodes, validations, capped
                if capped:
                    return
                matches, missing_support = inspect(candidate, remaining)
                if not matches:
                    return
                if remaining:
                    key = min(remaining, key=lambda item: (
                        -missing_support.get(item, 0), -support_map[item], item,
                    ))
                    following = tuple(item for item in remaining if item != key)
                    for value in option_map[key]:
                        completion_nodes += 1
                        if permutations + completion_nodes > expansion_limit:
                            capped = True
                            return
                        candidate[key] = value
                        complete(following, candidate)
                        candidate.pop(key, None)
                        if capped:
                            return
                    return
                completions += 1
                canonical = self.canonical(candidate)
                if canonical is None:
                    return
                anchor_ok = True
                for candidate_state, anchor in enumerate(anchors):
                    traced = self.run(candidate, self.vertices[anchor], trace=True)
                    if traced is None or traced[0][-1] != permutation[candidate_state]:
                        anchor_ok = False
                        break
                if not anchor_ok:
                    return
                validations += 1
                witnessed = set()
                matches = True
                for source, expected in self.examples:
                    state = 0
                    output = []
                    for symbol in source:
                        key = (state, symbol)
                        transition = candidate.get(key)
                        if transition is None:
                            matches = False
                            break
                        witnessed.add(key)
                        state, emitted = transition
                        output.append(emitted)
                    if not matches or "".join(output) != expected:
                        matches = False
                        break
                # Eski DFS de yalnız örneklerde kurulabilen tam tabloları kabul
                # eder; hiç görülmeyen geçişlerle sahte alternatif üretme.
                if not matches or witnessed != set(candidate):
                    return
                valid.add(tuple(sorted(canonical.items())))

            complete(tuple(missing), dict(forced))
            if capped:
                return {"status": "budget-exceeded",
                        "expansions": permutations + completion_nodes,
                        "permutations": permutations, "completions": completions,
                        "completion_nodes": completion_nodes, "validations": validations,
                        "max_missing": max_missing}

        tables = [dict(items) for items in sorted(valid)]
        recurrent = [
            candidate for candidate in tables
            if self._strongly_connected(candidate, states)
        ]
        state_bits = math.log2(states) if states > 1 else 0.0
        output_bits = math.log2(len(self.output_alphabet)) if len(self.output_alphabet) > 1 else 0.0
        mdl_bits = states * len(self.alphabet) * (state_bits + output_bits)
        if len(tables) == 1:
            selected, selection = tables[0], "unique-minimal"
        elif selection_strategy == "recurrent" and len(recurrent) == 1:
            selected, selection = recurrent[0], "unique-minimal-recurrent"
        elif selection_strategy == "uniform-random" and tables:
            selected = random.Random(int(selection_seed)).choice(tables)
            selection = "uniform-random"
        elif selection_strategy == "lexicographic" and tables:
            selected, selection = tables[0], "lexicographic"
        else:
            selection = "mdl-tie" if selection_strategy == "mdl" and tables else "ambiguous"
            selected = None
        stats = {
            "expansions": permutations + completion_nodes,
            "permutations": permutations, "completions": completions,
            "completion_nodes": completion_nodes,
            "validations": validations, "max_missing": max_missing,
            "candidates": len(tables), "recurrent_candidates": len(recurrent),
            "selection": selection, "selection_strategy": selection_strategy,
            "selection_seed": int(selection_seed), "minimal_clique": states,
            "labeling_space": labeling_space,
            "mdl_bits_per_candidate": mdl_bits,
        }
        result = {"status": "ambiguous" if selected is None and tables else
                  ("unresolved" if selected is None else "proved"), **stats}
        if selected is not None:
            result["table"] = selected
        if return_candidates:
            result["candidate_tables"] = tables
        return result


def _active_run_finite(candidate, source: str) -> str | None:
    """İç aday tablosunu veya kanonik FST temsilini bağımsız yürüt."""
    if isinstance(candidate, dict) and "transitions" in candidate:
        table = {(int(state), symbol): (int(target), output)
                 for state, symbol, target, output in candidate["transitions"]}
    else:
        table = candidate
    state = 0
    output = []
    for symbol in source:
        transition = table.get((state, symbol))
        if transition is None:
            return None
        state, emitted = transition
        output.append(emitted)
    return "".join(output)


def _active_table_representation(table, alphabet) -> dict:
    states = 1 + max([state for state, _symbol in table]
                     + [target for target, _output in table.values()])
    return {
        "kind": "finite_state_transducer", "states": states,
        "alphabet": list(alphabet), "initial": 0,
        "transitions": [[state, symbol, target, output]
                        for (state, symbol), (target, output) in sorted(table.items())],
    }


def _active_conformance_probes(candidate, alphabet: str, lengths,
                               max_states: int) -> list[str]:
    """Durum kapsama + W + ek-durum orta parçalarından sonda üret."""
    if isinstance(candidate, dict) and "transitions" in candidate:
        table = {(int(state), symbol): (int(target), output)
                 for state, symbol, target, output in candidate["transitions"]}
    else:
        table = candidate
    states = sorted({state for state, _symbol in table}
                    | {target for target, _output in table.values()})
    cover = {0: ""}
    queue = collections.deque([0])
    while queue:
        state = queue.popleft()
        for symbol in alphabet:
            target = table[(state, symbol)][0]
            if target not in cover:
                cover[target] = cover[state] + symbol
                queue.append(target)
    characterization = {""}
    for left, right in itertools.combinations(states, 2):
        queue = collections.deque([((left, right), "")])
        seen = {(left, right)}
        suffix = None
        while queue and suffix is None:
            (a_state, b_state), path = queue.popleft()
            for symbol in alphabet:
                next_a, output_a = table[(a_state, symbol)]
                next_b, output_b = table[(b_state, symbol)]
                if output_a != output_b:
                    suffix = path + symbol
                    break
                pair = (next_a, next_b)
                if pair not in seen:
                    seen.add(pair)
                    queue.append((pair, path + symbol))
        if suffix is None:
            raise ValueError("non-minimal-candidate")
        characterization.add(suffix)
    extra = max(0, int(max_states) - len(states))
    middles = ["".join(row) for width in range(extra + 2)
               for row in itertools.product(alphabet, repeat=width)]
    probes = {}
    allowed = sorted(set(map(int, lengths)))
    for state in states:
        for middle in middles:
            for suffix in characterization:
                base = cover[state] + middle + suffix
                target_lengths = [length for length in allowed if length >= len(base)]
                if not target_lengths:
                    continue
                target_length = target_lengths[0]
                source = base + alphabet[0] * (target_length - len(base))
                row = (len(middle), len(suffix), len(base), base, source)
                previous = probes.get(source)
                if previous is None or row[:4] > previous[:4]:
                    probes[source] = row
    ordered = sorted(probes.values(), key=lambda row: (-row[0], -row[1], -row[2], row[3]))
    return [row[4] for row in ordered]


class ContinuousCausalGraph:
    """Geleceği görmeyen, dağıtık gizli durumlu küçük bir sinirsel transdüser.

    FST'nin sonlu ayrık durumları yerine sürekli bir ``h_t`` vektörü vardır.
    ``h_t = GRU(x_t, h_{t-1})`` ve ``y_t`` yalnız ``h_t``den okunur; dolayısıyla
    çıktı gelecekteki girdilere erişemez. Model train ile gradyan öğrenir,
    validation yalnız erken durdurma/kabul içindir, held-out bu sınıfa hiç
    verilmez. Validation tam değilse temsil ``provisional`` olarak işaretlenir.

    Bu sınıf FST'nin başka adı değildir ve "daha iyi" olduğu varsayılmaz. Örnek
    sayısı, gradyan adımı, parametre, süre ve held-out sonuçları ayrı raporlanır.
    """

    KIND = "continuous_causal_graph"

    def __init__(self, *, hidden: int = 48, seed: int = 0,
                 learning_rate: float = 0.01, max_epochs: int = 240,
                 batch_size: int = 32, patience: int = 60,
                 class_balance: bool = True, prefix_augmentation: int = 0,
                 prefix_warmup_epochs: int = 0,
                 checkpoint_selection: str = "train_first",
                 causal_summary: bool = False):
        self.hidden = max(8, int(hidden))
        self.seed = int(seed)
        self.learning_rate = float(learning_rate)
        self.max_epochs = max(1, int(max_epochs))
        self.batch_size = max(1, int(batch_size))
        self.patience = max(1, int(patience))
        self.class_balance = bool(class_balance)
        self.prefix_augmentation = max(0, int(prefix_augmentation))
        self.prefix_warmup_epochs = max(0, int(prefix_warmup_epochs))
        if checkpoint_selection not in {"train_first", "validation_macro"}:
            raise ValueError("unknown checkpoint selection")
        self.checkpoint_selection = str(checkpoint_selection)
        self.causal_summary = bool(causal_summary)
        self.model = None
        self.input_symbols: list[str] = []
        self.output_symbols: list[str] = []
        self.report: dict[str, Any] = {}

    @staticmethod
    def _rows(examples) -> list[tuple[str, str]]:
        rows = CompositionalCodeLearner._relation_examples(examples)
        if not rows or any(not source for source, _expected in rows):
            raise ValueError("nonempty equal-length string pairs required")
        return rows

    @staticmethod
    def _prefix_rows(rows, per_row: int) -> list[tuple[str, str]]:
        """Train çiftlerinden, çıktı sınıfı geçişlerini kapsayan ek önekler.

        Nedensel transdüserde bir çiftin her öneği de geçerli bir çifttir. Yeni
        etiket oracle'dan veya validation/held-out'tan alınmaz; ``expected``'in
        ilgili öneği kullanılır. Sınıf değişimleri nadir olayın görüldüğü yerleri
        de örneklemeye sokar; tamamen düz çıktılarda eşit aralıklı önekler seçilir.
        """
        count = max(0, int(per_row))
        if count == 0:
            return list(rows)
        augmented = list(rows)
        seen = set(rows)
        for source, expected in rows:
            length = len(source)
            if length <= 1:
                continue
            transitions = [index + 1 for index in range(1, length)
                           if expected[index] != expected[index - 1]]
            fractions = [max(1, (length * part) // (count + 1))
                         for part in range(1, count + 1)]
            candidates = sorted({size for size in transitions + fractions
                                 if 0 < size < length})
            if len(candidates) > count:
                # İlk/son geçişi de kaybetmeden, geçiş ekseninde eşit seyretme.
                indices = (list(range(count)) if count == 1 else
                           [round(i * (len(candidates) - 1) / (count - 1))
                            for i in range(count)])
                candidates = [candidates[index] for index in sorted(set(indices))]
            for size in candidates:
                row = (source[:size], expected[:size])
                if row not in seen:
                    seen.add(row)
                    augmented.append(row)
        return augmented

    @staticmethod
    def _batch(model, rows, input_id, output_id, class_weights=None):
        import torch
        lengths = [len(source) for source, _expected in rows]
        width = max(lengths)
        xs = torch.zeros((len(rows), width), dtype=torch.long)
        ys = torch.zeros((len(rows), width), dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.bool)
        for row, (source, expected) in enumerate(rows):
            length = len(source)
            xs[row, :length] = torch.tensor([input_id[s] for s in source])
            ys[row, :length] = torch.tensor([output_id[s] for s in expected])
            mask[row, :length] = True
        logits = model(xs)                       # (B,L,O); padded girdiler maskede yok
        loss = torch.nn.functional.cross_entropy(
            logits[mask], ys[mask], reduction="mean", weight=class_weights,
        )
        return loss, logits, ys, mask

    def fit(self, train, validation) -> dict:
        import torch
        import numpy as np
        train_rows = self._rows(train)
        validation_rows = self._rows(validation)
        fit_rows = self._prefix_rows(train_rows, self.prefix_augmentation)
        self.input_symbols = sorted({s for source, _ in train_rows for s in source})
        self.output_symbols = sorted({s for _source, out in train_rows for s in out})
        if any(s not in self.input_symbols for source, _ in validation_rows for s in source):
            raise ValueError("validation-alphabet")
        if any(s not in self.output_symbols for _source, out in validation_rows for s in out):
            raise ValueError("validation-output-alphabet")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed % (2 ** 32 - 1))
        embedding = max(8, self.hidden // 2)
        # Sequential yerine açık hücre: sıfır başlangıç durumu her bağımsız
        # dizide yeniden kurulur; örnekler arasında gizli bilgi sızıntısı yoktur.
        class CausalCell(torch.nn.Module):
            def __init__(self, n_in, n_out, d, h, summary=False):
                super().__init__()
                self.embed = torch.nn.Embedding(n_in, d)
                self.gru = torch.nn.GRU(d, h, num_layers=1, batch_first=True)
                self.readout = torch.nn.Linear(h, n_out)
                self.with_summary = bool(summary)
                if self.with_summary:
                    self.summary_input = torch.nn.Linear(d, h)
                    self.summary_reset = torch.nn.Linear(d, h)
                    self.summary_decay = torch.nn.Parameter(torch.zeros(h))
                    # Başlangıçta nötr reset; hangi girdinin durumu sıfırladığı
                    # veya sönümü değiştirdiği yalnız gradyanla öğrenilir.
                    torch.nn.init.zeros_(self.summary_reset.weight)
                    torch.nn.init.zeros_(self.summary_reset.bias)
                    # Eski GRU+readout başlangıcı aynen korunur. Sıfır adaptör
                    # başlangıçta hiçbir tahmini değiştirmez; faydasız özet yolu
                    # diğer ailelerin çalışan çözümünü bozmak zorunda kalmaz.
                    self.summary_readout = torch.nn.Sequential(
                        torch.nn.Linear(2 * h, h), torch.nn.SiLU(),
                        torch.nn.Linear(h, n_out),
                    )
                    torch.nn.init.zeros_(self.summary_readout[-1].weight)
                    torch.nn.init.zeros_(self.summary_readout[-1].bias)

            def forward(self, xs):
                embedded = self.embed(xs)
                recurrent = self.gru(embedded)[0]
                if not self.with_summary:
                    return self.readout(recurrent)
                # Doygun tanh durumuna paralel, modulusü bilmeyen sürekli lineer
                # biriktirici. decay=2*sigmoid(...) sıfır-reset, bir-sayım ve
                # iki-affine rejimlerini aynı genel hücrede erişilebilir kılar.
                state = embedded.new_zeros((embedded.shape[0], self.gru.hidden_size))
                summaries = []
                for index in range(embedded.shape[1]):
                    current = embedded[:, index]
                    decay = 2.0 * torch.sigmoid(
                        self.summary_decay + self.summary_reset(current))
                    state = decay * state + self.summary_input(current)
                    summaries.append(state)
                summary = torch.stack(summaries, dim=1)
                joined = torch.cat((recurrent, summary), dim=-1)
                return self.readout(recurrent) + self.summary_readout(joined)

        model = CausalCell(len(self.input_symbols), len(self.output_symbols),
                           embedding, self.hidden, self.causal_summary)
        self.model = model
        input_id = {s: i for i, s in enumerate(self.input_symbols)}
        output_id = {s: i for i, s in enumerate(self.output_symbols)}
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        # Tokenları çok dengesiz olan akışlarda (ör. seyrek 1, bol 0) ağırlıksız
        # kayıp hep-negatif çöküşü ödüllendirir. Sınıf adları veya görev kuralı
        # bilinmez; ağırlıklar yalnız train tokenlarının frekansından çıkar.
        # Warmup+ince ayarda asıl amaç özgün train dağılımıdır; warmup boyunca
        # bile sınıf ağırlıkları kısa öneklerin dağılımına kaydırılmaz.
        weight_rows = train_rows if self.prefix_warmup_epochs else fit_rows
        class_counts = torch.zeros(len(self.output_symbols), dtype=torch.float64)
        for _source, expected in weight_rows:
            for symbol in expected:
                class_counts[output_id[symbol]] += 1
        class_weights = torch.zeros_like(class_counts, dtype=torch.float32)
        present = class_counts > 0
        class_weights[present] = (
            class_counts[present].sum() /
            (present.sum() * class_counts[present])
        ).to(torch.float32)
        if not self.class_balance:
            class_weights.fill_(1.0)
        rng = np.random.default_rng(self.seed + 71_039)
        best_score = (-1, -1, -float("inf"), -float("inf"),
                      -float("inf"), -float("inf"))
        best_weights = None
        best_epoch = 0
        stale = 0
        gradient_steps = 0
        started = time.perf_counter()
        example_presentations = 0
        model.train()
        for epoch in range(1, self.max_epochs + 1):
            warmup = (self.prefix_augmentation > 0 and
                      (self.prefix_warmup_epochs == 0 or
                       epoch <= self.prefix_warmup_epochs))
            epoch_rows = fit_rows if warmup else train_rows
            example_presentations += len(epoch_rows)
            order = rng.permutation(len(epoch_rows))
            for start in range(0, len(order), self.batch_size):
                rows = [epoch_rows[int(i)] for i in order[start:start + self.batch_size]]
                optimizer.zero_grad()
                loss, _logits, _ys, _mask = self._batch(
                    model, rows, input_id, output_id, class_weights=class_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                gradient_steps += 1
            if epoch == 1 or epoch % 5 == 0 or epoch == self.max_epochs:
                metrics = self.evaluate(validation_rows)
                training_metrics = self.evaluate(train_rows)
                if self.checkpoint_selection == "validation_macro":
                    # Ezberlenmiş train exact'i, validation'daki pozitif sınıfın
                    # tamamen kaybolmasını gizlemesin. Kabul kapısı ayrıdır ve
                    # aşağıda hâlâ iki kümenin de exact olmasını ister.
                    score = (int(metrics["exact"]),
                             float(metrics["macro_token_accuracy"]),
                             int(training_metrics["exact"]),
                             float(training_metrics["macro_token_accuracy"]),
                             float(metrics["token_accuracy"]),
                             float(training_metrics["token_accuracy"]))
                else:
                    score = (int(metrics["exact"]), int(training_metrics["exact"]),
                             float(metrics["macro_token_accuracy"]),
                             float(training_metrics["macro_token_accuracy"]),
                             float(metrics["token_accuracy"]),
                             float(training_metrics["token_accuracy"]))
                if score > best_score:
                    best_score = score
                    best_weights = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    stale = 0
                else:
                    stale += 5
                if (metrics["exact"] and training_metrics["exact"]) or stale >= self.patience:
                    break
        if best_weights is not None:
            model.load_state_dict(best_weights)
        validation_metrics = self.evaluate(validation_rows)
        train_metrics = self.evaluate(train_rows)
        self.report = {
            "kind": self.KIND,
            "status": "accepted" if validation_metrics["exact"] and train_metrics["exact"] else "provisional",
            "class_balance": self.class_balance,
            "checkpoint_selection": self.checkpoint_selection,
            "causal_summary": self.causal_summary,
            "hidden": self.hidden, "parameters": sum(p.numel() for p in model.parameters()),
            "train_examples": len(train_rows), "validation_examples": len(validation_rows),
            "optimization_examples": len(fit_rows),
            "prefix_augmentation": self.prefix_augmentation,
            "prefix_augmented_examples": len(fit_rows) - len(train_rows),
            "prefix_warmup_epochs": self.prefix_warmup_epochs,
            "epochs": epoch, "best_epoch": best_epoch, "gradient_steps": gradient_steps,
            "example_presentations": example_presentations,
            "seconds": time.perf_counter() - started,
            "train_token_accuracy": train_metrics["token_accuracy"],
            "train_macro_token_accuracy": train_metrics["macro_token_accuracy"],
            "train_exact_sequences": train_metrics["exact_sequences"],
            "validation_token_accuracy": validation_metrics["token_accuracy"],
            "validation_macro_token_accuracy": validation_metrics["macro_token_accuracy"],
            "validation_exact_sequences": validation_metrics["exact_sequences"],
            "class_weights": {symbol: float(class_weights[index])
                              for index, symbol in enumerate(self.output_symbols)},
            "validation_exact": validation_metrics["exact"],
            "heldout_passed_to_learner": False,
        }
        return dict(self.report)

    def evaluate(self, examples) -> dict:
        """Decorator gerektirmeden inference_mode ile bağımsız metrikler."""
        if self.model is None:
            raise RuntimeError("continuous model is not fitted")
        import torch
        rows = self._rows(examples)
        input_id = {s: i for i, s in enumerate(self.input_symbols)}
        output_id = {s: i for i, s in enumerate(self.output_symbols)}
        correct_tokens = total_tokens = exact_sequences = 0
        class_correct = [0 for _symbol in self.output_symbols]
        class_total = [0 for _symbol in self.output_symbols]
        losses = []
        with torch.inference_mode():
            self.model.eval()
            for start in range(0, len(rows), self.batch_size):
                chunk = rows[start:start + self.batch_size]
                loss, logits, ys, mask = self._batch(self.model, chunk, input_id, output_id)
                predicted = logits.argmax(-1)
                correct_tokens += int(((predicted == ys) & mask).sum())
                total_tokens += int(mask.sum())
                exact_sequences += sum(int(bool((predicted[i] == ys[i])[mask[i]].all()))
                                       for i in range(len(chunk)))
                for class_index in range(len(self.output_symbols)):
                    selected = mask & (ys == class_index)
                    class_total[class_index] += int(selected.sum())
                    class_correct[class_index] += int(((predicted == ys) & selected).sum())
                losses.append(float(loss))
        class_accuracy = {symbol: class_correct[index] / class_total[index]
                          for index, symbol in enumerate(self.output_symbols)
                          if class_total[index] > 0}
        return {
            "token_accuracy": correct_tokens / max(1, total_tokens),
            "macro_token_accuracy": (sum(class_accuracy.values()) /
                                      max(1, len(class_accuracy))),
            "class_token_accuracy": class_accuracy,
            "exact_sequences": exact_sequences, "total_sequences": len(rows),
            "exact": exact_sequences == len(rows), "loss": sum(losses) / max(1, len(losses)),
        }

    def apply(self, source: str) -> str | None:
        if self.model is None:
            return None
        import torch
        input_id = {s: i for i, s in enumerate(self.input_symbols)}
        if any(symbol not in input_id for symbol in str(source)):
            return None
        ids = torch.tensor([[input_id[s] for s in str(source)]], dtype=torch.long)
        with torch.inference_mode():
            self.model.eval()
            output = self.model(ids).argmax(-1)[0].tolist()
        return "".join(self.output_symbols[i] for i in output)

    def state_dict(self) -> dict:
        if self.model is None:
            raise RuntimeError("continuous model is not fitted")
        return {
            "kind": self.KIND, "hidden": self.hidden, "seed": self.seed,
            "causal_summary": self.causal_summary,
            "input_symbols": list(self.input_symbols),
            "output_symbols": list(self.output_symbols),
            "weights": {key: value.detach().cpu().tolist()
                        for key, value in self.model.state_dict().items()},
            "evidence": copy.deepcopy(self.report),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContinuousCausalGraph":
        import torch
        evidence = dict(data.get("evidence") or {})
        out = cls(hidden=int(data.get("hidden", 48)), seed=int(data.get("seed", 0)),
                  causal_summary=bool(data.get("causal_summary", False)))
        out.input_symbols = [str(s) for s in data.get("input_symbols") or ()]
        out.output_symbols = [str(s) for s in data.get("output_symbols") or ()]
        if not out.input_symbols or not out.output_symbols:
            raise ValueError("invalid continuous representation")
        # Fit'in içindeki küçük hücreyle aynı mimariyi kur; eğitim çalıştırılmaz.
        embedding = max(8, out.hidden // 2)
        class CausalCell(torch.nn.Module):
            def __init__(self, n_in, n_out, d, h, summary=False):
                super().__init__()
                self.embed = torch.nn.Embedding(n_in, d)
                self.gru = torch.nn.GRU(d, h, num_layers=1, batch_first=True)
                self.readout = torch.nn.Linear(h, n_out)
                self.with_summary = bool(summary)
                if self.with_summary:
                    self.summary_input = torch.nn.Linear(d, h)
                    self.summary_reset = torch.nn.Linear(d, h)
                    self.summary_decay = torch.nn.Parameter(torch.zeros(h))
                    torch.nn.init.zeros_(self.summary_reset.weight)
                    torch.nn.init.zeros_(self.summary_reset.bias)
                    self.summary_readout = torch.nn.Sequential(
                        torch.nn.Linear(2 * h, h), torch.nn.SiLU(),
                        torch.nn.Linear(h, n_out),
                    )
                    torch.nn.init.zeros_(self.summary_readout[-1].weight)
                    torch.nn.init.zeros_(self.summary_readout[-1].bias)

            def forward(self, xs):
                embedded = self.embed(xs)
                recurrent = self.gru(embedded)[0]
                if not self.with_summary:
                    return self.readout(recurrent)
                state = embedded.new_zeros((embedded.shape[0], self.gru.hidden_size))
                summaries = []
                for index in range(embedded.shape[1]):
                    current = embedded[:, index]
                    decay = 2.0 * torch.sigmoid(
                        self.summary_decay + self.summary_reset(current))
                    state = decay * state + self.summary_input(current)
                    summaries.append(state)
                summary = torch.stack(summaries, dim=1)
                joined = torch.cat((recurrent, summary), dim=-1)
                return self.readout(recurrent) + self.summary_readout(joined)

        out.model = CausalCell(len(out.input_symbols), len(out.output_symbols),
                               embedding, out.hidden, out.causal_summary)
        weights = {key: torch.tensor(value, dtype=out.model.state_dict()[key].dtype)
                   for key, value in (data.get("weights") or {}).items()}
        out.model.load_state_dict(weights, strict=True)
        out.report = evidence
        return out


class CompositionalCodeLearner:
    """Learn general Python operators and compose them from semantic constraints."""

    def __init__(self):
        self.operators: set[str] = set()
        self.facts: dict[str, Any] = {}
        self.composites: dict[str, tuple[str, ...]] = {}
        # Belgedeki veri örneklerinden, ilkel ifade/akış aramasıyla kurulan
        # operatörler. Bunlar hazır operatör adları değil, çalıştırılabilir
        # küçük programların taşınabilir temsilleridir.
        self.stream_operators: dict[str, dict[str, Any]] = {}
        self.representations: dict[str, dict[str, Any]] = {}
        # Canlı model önbelleği kalıcı kanıt değildir; save/load sonrası aynı
        # ağırlıklardan yeniden kurulur ve roundtrip ile denetlenir.
        self._continuous_models: dict[str, ContinuousCausalGraph] = {}
        self.value_decoders: dict[str, dict[str, Any]] = {}
        self.string_transforms: dict[str, dict[str, Any]] = {}
        self.cumulative_rules: dict[str, int] = {}
        self.documents: list[str] = []
        self.learning_events = 0
        self.failure_experiments = 0
        self.last_failure: dict[str, Any] | None = None

    # -------------------------------------------------------------- request meaning
    @staticmethod
    def _payload(raw: str) -> tuple[str, int, int]:
        for pattern in _QUOTES:
            match = re.search(pattern, raw)
            if match and match.group(1).strip():
                return match.group(1), match.start(1), match.end(1)
        marker = re.search(r"\s+(?:" + "|".join(_NOMINALS) + r")\b", raw, re.IGNORECASE)
        if marker:
            candidate = raw[:marker.start()]
            start = 0
            polite = re.match(r"(?:bana|lütfen|acaba|hemen|şimdi)\s+", candidate, re.IGNORECASE)
            if polite:
                start = polite.end()
            payload = candidate[start:].strip(" ,.:;-\n\t")
            start += len(candidate[start:]) - len(candidate[start:].lstrip(" ,.:;-\n\t"))
            if payload:
                return payload, start, start + len(payload)
        return "", 0, 0

    @staticmethod
    def _unit(text: str) -> str:
        character = re.search(r"\b(?:harf|karakter)[\wçğıöşü]*", text)
        word = re.search(r"\b(?:sözcük|kelime)[\wçğıöşü]*", text)
        syllable = re.search(r"\bhece[\wçğıöşü]*", text)
        if character and (not word or character.start() > word.start()):
            return "character"
        if word:
            return "word"
        if syllable:
            return "syllable"
        return "whole"

    def _transformation_request(self, raw: str) -> TransformationGoal | None:
        """Açık dönüşüm isteklerini gömülü metin ve işlem referansıyla ayır."""
        decimal = re.match(
            r"^\s*(.+?)(?:'[\wçğıöşü]*)?\s+"
            r"(?:ondalığa|ondalık|sayıya|decimal)\s+çevir\b",
            raw, flags=re.IGNORECASE,
        )
        if decimal and decimal.group(1).strip():
            payload = decimal.group(1).strip(" ,.:;-\n\t")
            return TransformationGoal(
                payload, "decimal", raw,
                trace=(f"payload_length={len(payload)}", "operation=decimal"),
            )

        payload, _start, _end = self._payload(raw)
        if not payload:
            # Dönüşüm hedefi belirliyken sondaki açık payload da geçerlidir:
            # "... öğrendiğin işlemi uygula: 01001". Bu kural genel display
            # çözümlemesine değil, yalnız işlem referanslı isteklere uygulanır.
            trailing = re.search(r":\s*([^\n]+?)\s*$", raw)
            if trailing:
                payload = trailing.group(1).strip()
        if not payload:
            return None
        low = tr_lower(raw)
        applies = bool(re.search(r"(?:işlemden\s+geçir|dönüştür|transform|uygula)", low))
        anaphoric = bool(re.search(
            r"\b(?:bu|aynı|kendi|öğren\w*|kur\w*|bul\w*|geliştir\w*)\s+"
            r"(?:işlem|operatör)", low,
        ))
        representation_reference = ""
        for name, record in self.representations.items():
            alias = tr_lower(str(record.get("name") or ""))
            named = re.search(r"\b" + re.escape(tr_lower(name)) + r"\b", low)
            if named or (alias and re.search(r"\b" + re.escape(alias) + r"\b", low)):
                representation_reference = name
                break
        if applies and self.representations and (
                representation_reference or (anaphoric and not self.string_transforms)):
            return TransformationGoal(
                payload, "representation", raw, representation_reference,
                (f"payload_length={len(payload)}", "operation=representation",
                 f"reference={representation_reference or 'latest'}"),
            )

        reference = ""
        for name, record in self.string_transforms.items():
            alias = tr_lower(str(record.get("name") or ""))
            if alias and re.search(r"\b" + re.escape(alias) + r"\b", low):
                reference = name
                break
        if applies and (anaphoric or reference):
            return TransformationGoal(
                payload, "string", raw, reference,
                (f"payload_length={len(payload)}", "operation=string",
                 f"reference={reference or 'latest'}"),
            )
        return None

    def request(self, raw: str) -> SemanticGoal | TransformationGoal | None:
        """Ground constrained transformations and display requests."""
        transformed = self._transformation_request(raw)
        if transformed is not None:
            return transformed
        payload, start, end = self._payload(raw)
        if not payload:
            return None
        constraints = (raw[:start] + " " + raw[end:]).strip()
        low = tr_lower(constraints)
        if not any(cue in low for cue in _DISPLAY):
            return None
        unit = self._unit(low)
        layout = "vertical" if any(cue in low for cue in _VERTICAL) else "inline"
        if unit == "whole" and layout == "inline":
            return None
        execute = any(cue in low for cue in ("çalıştır", "yürüt", "run"))
        cumulative = any(cue in low for cue in _CUMULATIVE)
        cumulative_step = 1
        for phrase, step in sorted(self.cumulative_rules.items(), key=lambda row: -len(row[0])):
            if phrase in low:
                cumulative = True
                cumulative_step = max(1, int(step))
                break
        trace = (f"payload_length={len(payload)}", f"unit={unit}", f"layout={layout}",
                 f"execute={execute}", f"cumulative={cumulative}",
                 f"cumulative_step={cumulative_step}")
        return SemanticGoal(payload, unit, layout, execute, raw, trace, cumulative, cumulative_step)

    # -------------------------------------------------------------- document learning
    @staticmethod
    def _snippets(text: str) -> list[str]:
        blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        lines = text.splitlines()
        current: list[str] = []
        for line in lines + [""]:
            stripped = line.lstrip()
            if stripped.startswith(">>>") or (current and stripped.startswith("...")):
                content = stripped[3:]
                # reST prompt consumes one space, not the Python indentation.
                current.append(content[1:] if content.startswith(" ") else content)
            else:
                if current:
                    blocks.append("\n".join(current))
                    current = []
        if not blocks:
            try:
                ast.parse(text)
                blocks.append(text)
            except SyntaxError:
                pass
        return blocks

    def learn_document(self, text: str, source: str = "document") -> dict:
        before_ops = set(self.operators)
        before_facts = set(self.facts)
        low = tr_lower(text)
        if (re.search(r"(?s)\bstrings?\b.{0,100}\bsequences?\b.{0,100}\b(?:characters?|code points?)\b", low)
                or re.search(r"(?s)\b(?:characters?|code points?)\b.{0,100}\bstrings?\b.{0,100}\bsequences?\b", low)
                or ("metin" in low and "dizi" in low and "karakter" in low)):
            self.facts["text_sequence"] = True
        if (re.search(r"(?s)\bprint\b.{0,140}(?:newline|new line|yeni satır)", low)
                or re.search(r"\bend\s*=\s*['\"]\\n['\"]", text)):
            self.facts["print_newline"] = True

        parsed = 0
        for snippet in self._snippets(text):
            try:
                tree = ast.parse(snippet)
            except SyntaxError:
                continue
            parsed += 1
            for node in ast.walk(tree):
                if isinstance(node, ast.For):
                    self.operators.add("iterate")
                elif isinstance(node, ast.Assign):
                    self.operators.add("bind")
                elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                    # Sayısal toplamayı string birleştirme sanma: anlamı
                    # belgedeki genel string/concatenation açıklaması kurar.
                    if ("string" in low or "metin" in low) and "concat" in low:
                        self.operators.add("concatenate")
                elif isinstance(node, ast.Call):
                    name = node.func
                    if isinstance(name, ast.Name) and name.id == "print":
                        self.operators.add("display")
                    elif isinstance(name, ast.Name) and name.id == "accumulate":
                        self.operators.add("accumulate")
                    elif isinstance(name, ast.Attribute) and name.attr == "split" and not node.args:
                        self.operators.add("split_words")
        invented = self._learn_stream_relations(text)
        decoders = self._learn_value_decoders(text)
        transforms = self._learn_string_transforms(text)
        cumulative = self._learn_cumulative_rules(text)
        self.learning_events += 1
        self.documents.append(str(source))
        self.documents = self.documents[-40:]
        new_ops = sorted(self.operators - before_ops)
        new_facts = sorted(set(self.facts) - before_facts)
        learned = ([f"invented:{name}" for name in invented]
                   + [f"decoder:{name}" for name in decoders]
                   + [f"transform:{name}" for name in transforms]
                   + [f"cumulative:{phrase}" for phrase in cumulative])
        return {"ok": bool(parsed or new_facts or learned), "source": source, "parsed": parsed,
                "new_operators": new_ops + learned,
                "new_facts": new_facts, "operators": sorted(self.operators),
                "invented": invented, "decoders": decoders, "transforms": transforms,
                "cumulative_rules": cumulative, "facts": dict(self.facts)}

    @staticmethod
    def _decode_with_symbols(symbols: dict[str, int], text: str, rule: str) -> int | None:
        if not text or any(symbol not in symbols for symbol in text):
            return None
        total = 0
        for index, symbol in enumerate(text):
            value = int(symbols[symbol])
            following = symbols.get(text[index + 1]) if index + 1 < len(text) else None
            if rule == "subtractive_sum" and following is not None and value < following:
                total -= value
            else:
                total += value
        return total

    def _learn_value_decoders(self, text: str) -> list[str]:
        """Belgedeki sembol/değer eşlemesi ve toplama kuralını birlikte öğren.

        Hedef girdi veya ondalık cevap burada yoktur. Çok sembollü belge örnekleri
        kuralın bağımsız doğrulamasıdır; onları açıklayamayan çözücü kabul edilmez.
        """
        low = tr_lower(text)
        subtractive = bool(
            ("küçük" in low and "çıkar" in low and "topla" in low and "önce" in low)
            or ("smaller" in low and "subtract" in low and "before" in low)
        )
        if not subtractive:
            return []
        rows = [(match.group(1), int(match.group(2))) for match in re.finditer(
            r"(?<![\wçğıöşü])([\wçğıöşü]+)\s*=\s*(-?\d+)", text,
        )]
        symbols: dict[str, int] = {}
        for symbol, value in rows:
            if len(symbol) != 1:
                continue
            if symbol in symbols and symbols[symbol] != value:
                return []
            symbols[symbol] = value
        if len(symbols) < 2:
            return []
        rule = "subtractive_sum"
        checks = [(symbol, value) for symbol, value in rows if len(symbol) > 1]
        if not checks or any(self._decode_with_symbols(symbols, symbol, rule) != value
                             for symbol, value in checks):
            return []
        signature = repr((sorted(symbols.items()), rule))
        if any(record.get("signature") == signature for record in self.value_decoders.values()):
            return []
        serial = 1 + max(
            [int(name[7:]) for name in self.value_decoders
             if name.startswith("decoder") and name[7:].isdigit()] or [0],
        )
        name = f"decoder{serial}"
        self.value_decoders[name] = {
            "symbols": dict(symbols), "rule": rule, "signature": signature,
            "evidence": len(checks),
        }
        return [name]

    STRING_OPERATION_CUES = (
        ("reverse", r"ters(?:ine)?\s+çevir|reverse"),
        ("uppercase", r"büyük\s+harf(?:e)?(?:\s+çevir)?|uppercase|upper"),
        ("lowercase", r"küçük\s+harf(?:e)?(?:\s+çevir)?|lowercase|lower"),
        ("titlecase", r"baş\s+harf(?:ler)?(?:ini|\s+harf)?\s+büyüt|title\s*case|title"),
    )

    def _learn_string_transforms(self, text: str) -> list[str]:
        """Kodsuz anlam tanımındaki ilkel dönüşümleri sırasıyla çıkar."""
        learned: list[str] = []
        segments = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n\s*\n", text)
                    if part.strip()]
        for segment in segments:
            low = tr_lower(segment)
            if not any(cue in low for cue in ("işlem", "dönüşüm", "dönüştür", "transformation", "operation")):
                continue
            hits = []
            for operation, pattern in self.STRING_OPERATION_CUES:
                hits.extend((match.start(), match.end(), operation)
                            for match in re.finditer(pattern, low))
            hits.sort(key=lambda row: (row[0], -(row[1] - row[0])))
            operations: list[str] = []
            occupied = -1
            for start, end, operation in hits:
                if start < occupied:
                    continue
                occupied = end
                if not operations or operations[-1] != operation:
                    operations.append(operation)
            if not operations:
                continue
            alias_match = re.search(r"([\wçğıöşü]+)\s+işlemi\b", low)
            alias = alias_match.group(1) if alias_match else ""
            signature = repr((alias, operations))
            if any(record.get("signature") == signature for record in self.string_transforms.values()):
                continue
            if alias:
                for key, record in list(self.string_transforms.items()):
                    if record.get("name") == alias:
                        del self.string_transforms[key]
            serial = 1 + max(
                [int(name[9:]) for name in self.string_transforms
                 if name.startswith("transform") and name[9:].isdigit()] or [0],
            )
            name = f"transform{serial}"
            self.string_transforms[name] = {
                "name": alias, "operations": operations, "signature": signature,
            }
            learned.append(name)
        return learned

    def _learn_cumulative_rules(self, text: str) -> list[str]:
        """Belge bir kümülatif ifadeyi yeniden tanımlarsa adımını güncelle."""
        low = tr_lower(text)
        learned: list[str] = []
        quoted = re.findall(r"['\"“]([^'\"”\n]{3,})['\"”]", text)
        for raw_phrase in quoted:
            phrase = tr_lower(raw_phrase).strip()
            if not phrase:
                continue
            location = low.find(phrase)
            window = low[max(0, location): location + len(phrase) + 420] if location >= 0 else low
            initial = re.search(
                r"ilk\s+satır[^.\n]*?\bilk\s+([\wçğıöşü]+)\s+(?:öğe|karakter|harf)", window,
            )
            increment = re.search(
                r"\b([\wçğıöşü]+)\s+yeni\s+(?:öğe|karakter|harf)", window,
            )
            if not initial or not increment:
                continue
            first = TurkishMorphology.words_to_int([initial.group(1)])
            step = TurkishMorphology.words_to_int([increment.group(1)])
            if first is None or step is None or first != step or step < 1:
                continue
            if self.cumulative_rules.get(phrase) == step:
                continue
            self.cumulative_rules[phrase] = int(step)
            self.operators.add("cumulative_window")
            learned.append(phrase)
        return learned

    def load_primer(self, path: str | Path) -> dict:
        source = Path(path)
        return self.learn_document(source.read_text(encoding="utf-8"), source=str(source))

    @staticmethod
    def read_url(url: str, timeout: float = 10.0, max_bytes: int = 2_000_000) -> tuple[str, dict]:
        """Read a document from any HTTP host; never execute the fetched source."""
        if not re.match(r"^https?://", str(url), flags=re.IGNORECASE):
            raise ValueError("only_http_documents")
        request = urllib.request.Request(
            url, headers={"User-Agent": "NoraAGI-DocumentLearner/1.0",
                          "Accept": "application/json, text/plain, text/html;q=0.8"},
        )
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            raw = response.read(int(max_bytes) + 1)
            content_type = str(response.headers.get("Content-Type") or "")
            charset = response.headers.get_content_charset() or "utf-8"
        if len(raw) > max_bytes:
            raise ValueError("document_too_large")
        if "json" in content_type or raw.lstrip().startswith((b"{", b"[")):
            payload = json.loads(raw.decode(charset, errors="replace"))
            if isinstance(payload, dict) and payload.get("content"):
                encoding = str(payload.get("encoding") or "").lower()
                content = payload["content"]
                raw = base64.b64decode(content) if encoding == "base64" else str(content).encode()
            else:
                raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            text = raw.decode("utf-8", errors="replace")
        else:
            text = raw.decode(charset, errors="replace")
        if "html" in content_type or re.search(r"<(?:html|body|article|div)\b", text[:2000], re.I):
            text = re.sub(r"(?is)<(?:script|style).*?>.*?</(?:script|style)>", " ", text)
            text = html.unescape(re.sub(r"(?s)<[^>]+>", " ", text))
        return text, {"url": url, "bytes": len(raw), "content_type": content_type}

    def learn_url(self, url: str) -> dict:
        try:
            text, meta = self.read_url(url)
        except Exception as exc:
            return {"ok": False, "source": url, "error": type(exc).__name__,
                    "new_operators": [], "new_facts": []}
        result = self.learn_document(text, source=url)
        result.update(meta)
        result["fetched"] = True
        return result

    # -------------------------------------------------------------- operatör icadı
    @staticmethod
    def _stream_examples(text: str) -> list[tuple[str, list[str]]]:
        """Kod olmayan input/outputs veri satırlarını ilişki örneklerine çevir."""
        examples: list[tuple[str, list[str]]] = []
        pattern = re.compile(
            r'(?im)^\s*input\s*:\s*("[^"\n]*")\s*\n'
            r'\s*outputs?\s*:\s*([^\n]+)',
        )
        for match in pattern.finditer(text):
            try:
                source = json.loads(match.group(1))
                outputs = [json.loads(part.strip()) for part in match.group(2).split("|")]
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if (isinstance(source, str) and outputs
                    and all(isinstance(value, str) for value in outputs)):
                examples.append((source, outputs))
        return examples[:8]

    @staticmethod
    def _expression_space() -> list[tuple]:
        """Genel ilkel ifade uzayı; hedef ilişkiye özel bir düğüm içermez."""
        base: list[tuple] = [("name", "item"), ("name", "state"), ("const", "")]
        combined = [("add", left, right) for left in base for right in base]
        return base + combined

    @classmethod
    def _expression_cost(cls, expression: tuple) -> int:
        if expression[0] == "add":
            return 1 + cls._expression_cost(expression[1]) + cls._expression_cost(expression[2])
        return 1

    @classmethod
    def _expression_uses_concat(cls, expression: tuple) -> bool:
        return expression[0] == "add" or any(
            cls._expression_uses_concat(part) for part in expression[1:]
            if isinstance(part, (tuple, list))
        )

    @classmethod
    def _eval_expression(cls, expression: tuple, environment: dict[str, str]) -> str | None:
        kind = expression[0]
        if kind == "const":
            return str(expression[1])
        if kind == "name":
            return environment.get(str(expression[1]))
        if kind == "add":
            left = cls._eval_expression(expression[1], environment)
            right = cls._eval_expression(expression[2], environment)
            if isinstance(left, str) and isinstance(right, str):
                return left + right
        return None

    @classmethod
    def _run_stream(cls, construction: dict[str, Any], source: str) -> list[str] | None:
        state = cls._eval_expression(construction["initial"], {})
        outputs: list[str] = []
        for item in source:
            environment = {"item": item, "state": state}
            if construction.get("after"):
                state = cls._eval_expression(construction["update"], environment)
                environment["state"] = state
                value = cls._eval_expression(construction["emit"], environment)
            else:
                value = cls._eval_expression(construction["emit"], environment)
                state = cls._eval_expression(construction["update"], environment)
            if not isinstance(value, str) or not isinstance(state, str):
                return None
            outputs.append(value)
        return outputs

    @staticmethod
    def _construction_uses(construction: dict[str, Any]) -> set[str]:
        uses = {"bind", "iterate"}
        expressions = (construction["initial"], construction["update"], construction["emit"])
        if any(CompositionalCodeLearner._expression_uses_concat(expr) for expr in expressions):
            uses.add("concatenate")
        return uses

    def _synthesize_stream(self, examples: list[tuple[str, list[str]]]) -> dict[str, Any] | None:
        expressions = self._expression_space()
        # Küçük, genel ve deterministik bir enumeratif sentez uzayı. Adayın
        # adı veya hedef kavramı burada yok; yalnız ilkel ifadeler ve iki
        # genel deyim sırası vardır.
        candidates: list[tuple[tuple, dict[str, Any]]] = []
        for update in expressions:
            for emit in expressions:
                for after in (False, True):
                    construction = {
                        "initial": ("const", ""), "update": update,
                        "emit": emit, "after": after,
                    }
                    uses = self._construction_uses(construction)
                    if not uses.issubset(self.operators):
                        continue
                    if all(self._run_stream(construction, source) == expected
                           for source, expected in examples):
                        rank = (self._expression_cost(update) + self._expression_cost(emit),
                                int(not after), repr(update), repr(emit))
                        candidates.append((rank, construction))
        if not candidates:
            return None
        candidates.sort(key=lambda row: row[0])
        return candidates[0][1]

    def _learn_stream_relations(self, text: str) -> list[str]:
        examples = self._stream_examples(text)
        if not examples:
            return []
        chosen = self._synthesize_stream(examples)
        if chosen is None:
            return []
        signature = repr((chosen["initial"], chosen["update"], chosen["emit"], chosen["after"]))
        if any(existing.get("signature") == signature for existing in self.stream_operators.values()):
            return []
        serial = 1 + max(
            [int(name[6:]) for name in self.stream_operators if name.startswith("stream")
             and name[6:].isdigit()] or [0],
        )
        name = f"stream{serial}"
        chosen.update({
            "signature": signature,
            "examples": [[source, expected] for source, expected in examples],
            "uses": sorted(self._construction_uses(chosen)),
        })
        self.stream_operators[name] = chosen
        return [name]

    def apply_stream(self, name: str, source: str) -> list[str] | None:
        """Kurulmuş akış operatörünü yeni metne uygula; ilkel yoksa başarı uydurma."""
        construction = self.stream_operators.get(str(name))
        if not construction or not self._construction_uses(construction).issubset(self.operators):
            return None
        return self._run_stream(construction, str(source))

    def invent_from_failure(self, *, budget: int = 64) -> dict[str, Any]:
        """Belgesiz öz-deney: eski modelin kendi deneyindeki hatasından operatör kur.

        Dışarıdan kavram, örnek veya hedef program alınmaz. Nora ilkel ifade
        uzayından bir deney seçer, kendi metinlerini üretir ve mevcut akış
        modellerinin gözlemleri açıklayamamasını öğrenme verisi yapar. Sentez
        yalnız train gözlemlerini görür; validation ve held-out kaynaklar ayrıdır.
        """
        if not self.stream_operators:
            return {"status": "no-baseline", "reason": "failure requires an existing model"}
        alphabet = "abcxyz"
        for _ in range(max(1, int(budget))):
            self.failure_experiments += 1
            rng = random.Random(7919 + 104729 * self.failure_experiments)
            expressions = self._expression_space()
            experiment = {
                "initial": ("const", ""),
                "update": rng.choice(expressions),
                "emit": rng.choice(expressions),
                "after": bool(rng.randrange(2)),
            }
            uses = self._construction_uses(experiment)
            if "concatenate" not in uses or not uses.issubset(self.operators):
                continue

            sources: list[str] = []
            while len(sources) < 7:
                source = "".join(rng.choice(alphabet) for _ in range(rng.randint(2, 5)))
                if source not in sources:
                    sources.append(source)
            train_sources, validation_sources = sources[:5], sources[5:]
            train = [(source, self._run_stream(experiment, source)) for source in train_sources]
            validation = [(source, self._run_stream(experiment, source)) for source in validation_sources]
            observed = train + validation
            if any(outputs is None or not any(outputs) or len(set(outputs)) < 2
                   for _, outputs in observed):
                continue

            # Mevcut modellerden hiçbiri bütün öz-deneyi açıklamıyorsa bu gerçek
            # bir öngörü başarısızlığıdır. Doğrudan karakter görünümü de eski
            # repertuvarda sayılır; sığ bir "yeni" kimlik operatörü kabul edilmez.
            baseline_scores = {}
            for name, construction in self.stream_operators.items():
                if not self._construction_uses(construction).issubset(self.operators):
                    continue
                baseline_scores[name] = sum(
                    self._run_stream(construction, source) == outputs
                    for source, outputs in observed
                )
            identity_known = bool(self.facts.get("text_sequence")) and all(
                list(source) == outputs for source, outputs in observed
            )
            if identity_known or any(score == len(observed) for score in baseline_scores.values()):
                continue
            best_model = max(baseline_scores, key=baseline_scores.get, default="")
            best_score = baseline_scores.get(best_model, 0)
            self.last_failure = {
                "kind": "stream_prediction", "model": best_model,
                "mismatches": len(observed) - best_score, "trials": len(observed),
                "experiment": self.failure_experiments,
            }

            chosen = self._synthesize_stream(train)
            if chosen is None or not all(self._run_stream(chosen, source) == outputs
                                         for source, outputs in validation):
                continue
            signature = repr((chosen["initial"], chosen["update"], chosen["emit"], chosen["after"]))
            if any(existing.get("signature") == signature
                   for existing in self.stream_operators.values()):
                continue
            serial = 1 + max(
                [int(name[6:]) for name in self.stream_operators if name.startswith("stream")
                 and name[6:].isdigit()] or [0],
            )
            name = f"stream{serial}"
            chosen.update({
                "signature": signature,
                "examples": [[source, expected] for source, expected in train],
                "uses": sorted(self._construction_uses(chosen)),
                "origin": "self_failure",
                "evidence": {"validation": len(validation), **self.last_failure},
            })
            self.stream_operators[name] = chosen
            self.learning_events += 1
            heldout_sources = []
            while len(heldout_sources) < 2:
                source = "".join(rng.choice(alphabet) for _ in range(rng.randint(3, 6)))
                if source not in sources and source not in heldout_sources:
                    heldout_sources.append(source)
            return {
                "status": "invented", "id": name, "kind": "stream_expression",
                "origin": "self_failure", "failure": dict(self.last_failure),
                "train": len(train), "validation": len(validation),
                "heldout": [{"source": source,
                             "outputs": self._run_stream(experiment, source)}
                            for source in heldout_sources],
                # Açıklanabilir deney düzeneği; sentezci bu yapıyı değil yalnız
                # train input/output çiftlerini gördü.
                "experiment": experiment,
            }
        return {"status": "unresolved", "experiments": self.failure_experiments,
                "failure": self.last_failure}

    def _matching_streams(self, goal: SemanticGoal) -> list[str]:
        if goal.unit != "character" or not self.facts.get("text_sequence"):
            return []
        expected = goal.units()
        matches = []
        for name, construction in self.stream_operators.items():
            if self._construction_uses(construction).issubset(self.operators):
                if self._run_stream(construction, goal.payload) == expected:
                    matches.append(f"invent:{name}")
        return matches

    # -------------------------------------------------------------- yeni temsil türü
    def representation_kinds(self) -> set[str]:
        kinds = {str(rec.get("kind")) for rec in self.representations.values()}
        if self.stream_operators:
            kinds.add("stream_expression")
        return kinds

    @staticmethod
    def _relation_examples(examples) -> list[tuple[str, str]]:
        out = []
        for source, expected in examples or ():
            if not isinstance(source, str) or not isinstance(expected, str):
                return []
            if len(source) != len(expected):
                return []
            out.append((source, expected))
        return out

    @staticmethod
    def _finite_table(representation: dict[str, Any]) -> dict[tuple, tuple]:
        return {
            (int(state), str(symbol)): (int(next_state), str(output))
            for state, symbol, next_state, output in representation.get("transitions") or ()
        }

    @classmethod
    def _run_finite(cls, representation: dict[str, Any], source: str) -> str | None:
        table = cls._finite_table(representation)
        state = int(representation.get("initial", 0))
        output = []
        for symbol in source:
            transition = table.get((state, symbol))
            if transition is None:
                return None
            state, emitted = transition
            output.append(emitted)
        return "".join(output)

    @classmethod
    def _finite_equivalent(cls, left: dict[str, Any], right: dict[str, Any],
                           alphabet: list[str]) -> bool:
        """İki tam transdüserin bütün erişilebilir gizli-durum çiftlerini karşılaştır."""
        a = cls._finite_table(left)
        b = cls._finite_table(right)
        queue = [(0, 0)]
        seen = set()
        while queue:
            pair = queue.pop()
            if pair in seen:
                continue
            seen.add(pair)
            for symbol in alphabet:
                ta = a.get((pair[0], symbol))
                tb = b.get((pair[1], symbol))
                if ta is None or tb is None or ta[1] != tb[1]:
                    return False
                queue.append((ta[0], tb[0]))
        return True

    @classmethod
    def _finite_candidates(cls, examples: list[tuple[str, str]], states: int,
                           alphabet: list[str], limit: int = 512,
                           max_expansions: int = 100000) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Kanonik, özyinelemesiz gizli-durum araması.

        Durum etiketleri yalnız ilk kullanım sırasıyla açılır; aynı makinenin
        permütasyonları tekrar aranmaz. Bir iş, bütün örneklerde derinlemesine
        ilerlediği için kolay ilk örneklerde binlerce kısmi makine biriktirilmez.
        ``max_expansions`` tükenmemiş aramayı ``unresolved`` diye etiketlememek
        için ayrıca döndürülür.
        """
        # example index, symbol position, current state, immutable table, used states
        stack: list[tuple[int, int, int, dict[tuple, tuple], int]] = [(0, 0, 0, {}, 1)]
        solutions: list[dict[str, Any]] = []
        expansions = 0
        peak_stack = 1
        while stack and len(solutions) < limit and expansions < max_expansions:
            peak_stack = max(peak_stack, len(stack))
            index, position, state, table, used = stack.pop()
            if index >= len(examples):
                if used != states or len(table) != states * len(alphabet):
                    continue
                transitions = [
                    [source_state, symbol, next_state, output]
                    for (source_state, symbol), (next_state, output) in sorted(table.items())
                ]
                solutions.append({
                    "kind": "finite_state_transducer", "states": states,
                    "alphabet": list(alphabet), "initial": 0,
                    "transitions": transitions,
                })
                continue
            source, expected = examples[index]
            if position >= len(source):
                stack.append((index + 1, 0, 0, table, used))
                continue
            symbol = source[position]
            emitted = expected[position]
            key = (state, symbol)
            known = table.get(key)
            if known is not None:
                if known[1] == emitted:
                    stack.append((index, position + 1, known[0], table, used))
                continue
            expansions += 1
            # Mevcut durumlar veya yalnız sıradaki tek yeni durum. Bu restricted-
            # growth canonicalization hedef ilişkisinden bağımsızdır.
            choices = range(min(states, used + 1))
            for next_state in reversed(list(choices)):
                branch = dict(table)
                branch[key] = (next_state, emitted)
                stack.append((index, position + 1, next_state, branch,
                              used + int(next_state == used)))
        candidate_capped = len(solutions) >= limit
        expansion_capped = expansions >= max_expansions and bool(stack)
        stats = {
            "expansions": expansions, "peak_stack": peak_stack,
            # 512 aday bulmak ile arama bütçesini tüketmek aynı olay değildir.
            # Eski ortak 'capped' bayrağı ör. 2575 expansion'ı yanlışlıkla
            # budget-exceeded diye raporluyordu.
            "budget_exceeded": expansion_capped,
            "candidate_limit_reached": candidate_capped,
            "candidate_limit": int(limit),
            "expansion_limit": int(max_expansions),
        }
        return solutions, stats

    def _finite_characteristic(self, constraints, alphabet, max_states, expansion_limit,
                               *, min_states=1, selection_strategy="recurrent", selection_seed=0,
                               return_candidates=False):
        """Gözlem/klik hızlı yolu; kanıt kurulamazsa eski DFS'e dön."""
        try:
            evidence = _FiniteObservationEvidence(constraints, alphabet)
        except ValueError:
            return {"status": "unresolved", "expansions": 0, "peak_stack": 0}
        total_expansions = 0
        for depth in range(max(evidence.levels, default=0)):
            tables, used, capped = evidence.candidates(
                depth, max_states, max(0, expansion_limit - total_expansions),
            )
            total_expansions += used
            common = {
                "kind": "finite_state_transducer", "expansions": total_expansions,
                "peak_stack": 0, "algorithm": "characteristic-observation-table",
                "constraint_checks": evidence.constraint_checks,
                "prefixes": len(evidence.vertices), "characteristic_depth": depth,
            }
            if capped:
                # Cartesian seçenek sayısı daha dal açılmadan bütçeyi aşabilir.
                # Kalan bütçe varsa aynı graph'ın global clique kanıtına geç.
                if total_expansions < expansion_limit:
                    break
                return {"status": "budget-exceeded", **common}
            if not tables:
                continue
            tables = [table for table in tables
                      if int(min_states) <= evidence.state_count(table) <= int(max_states)]
            if not tables:
                continue
            minimal = min(evidence.state_count(table) for table in tables)
            ordered = sorted(
                (table for table in tables if evidence.state_count(table) == minimal),
                key=lambda table: tuple(sorted(table.items())),
            )
            for table in ordered:
                proof = evidence.prove(
                    table, max(0, expansion_limit - total_expansions),
                    selection_strategy=selection_strategy, selection_seed=selection_seed,
                    return_candidates=return_candidates,
                )
                total_expansions += int(proof.get("expansions", 0))
                common.update({
                    "states": minimal, "expansions": total_expansions,
                    "candidates": proof.get("candidates", 0),
                    "recurrent_candidates": proof.get("recurrent_candidates", 0),
                    "selection": proof.get("selection", "unresolved"),
                    "selection_strategy": proof.get("selection_strategy", selection_strategy),
                    "selection_seed": proof.get("selection_seed", int(selection_seed)),
                    "mdl_bits_per_candidate": proof.get("mdl_bits_per_candidate", 0.0),
                    "proof_permutations": proof.get("permutations", 0),
                    "proof_completions": proof.get("completions", 0),
                    "minimal_clique": proof.get("minimal_clique", 0),
                    "labeling_space": proof.get("labeling_space", ""),
                })
                if return_candidates and "candidate_tables" in proof:
                    common["candidate_tables"] = proof["candidate_tables"]
                if proof.get("status") == "budget-exceeded":
                    return {"status": "budget-exceeded", **common}
                if proof.get("status") == "ambiguous":
                    return {"status": "ambiguous", **common}
                if proof.get("status") != "proved":
                    continue
                selected = proof["table"]
                representation = {
                    "kind": "finite_state_transducer",
                    "states": evidence.state_count(selected),
                    "alphabet": list(alphabet), "initial": 0,
                    "transitions": [
                        [state, symbol, target, output]
                        for (state, symbol), (target, output) in sorted(selected.items())
                    ],
                }
                return {"status": "proved", "representation": representation, **common}
        # Derinlik tabloları uyumlu satırları gereğinden fazla birleştirebilir.
        # Bu durumda aday uydurmak yerine aynı incompatibility graph'tan bağımsız
        # bir clique çapası ara; bütün makine uzayını yine exhaustive prove sayar.
        anchor_used = anchor_color_visits = 0
        anchor_attempts = []
        common = {
            "kind": "finite_state_transducer", "expansions": total_expansions,
            "peak_stack": 0, "algorithm": "characteristic-global-clique",
            "constraint_checks": evidence.constraint_checks,
            "prefixes": len(evidence.vertices), "selection_strategy": selection_strategy,
            "selection_seed": int(selection_seed),
        }
        for states in range(max(1, int(max_states)), max(1, int(min_states)) - 1, -1):
            anchors, used, color_visits, anchor_capped = evidence.clique_anchors(
                states, max(0, expansion_limit - total_expansions),
            )
            total_expansions += used
            anchor_used += used
            anchor_color_visits += color_visits
            anchor_attempts.append({
                "states": states, "expansions": used,
                "color_visits": color_visits, "found": bool(anchors),
            })
            common.update({
                "expansions": total_expansions, "states": states,
                "anchor_expansions": anchor_used,
                "anchor_color_visits": anchor_color_visits,
                "anchor_attempts": anchor_attempts,
            })
            if anchor_capped:
                return {"status": "budget-exceeded", **common}
            if not anchors:
                continue
            proof = evidence.prove(
                None, max(0, expansion_limit - total_expansions),
                anchor_vertices=anchors, selection_strategy=selection_strategy,
                selection_seed=selection_seed, return_candidates=return_candidates,
            )
            total_expansions += int(proof.get("expansions", 0))
            common.update({
                "expansions": total_expansions,
                "candidates": proof.get("candidates", 0),
                "recurrent_candidates": proof.get("recurrent_candidates", 0),
                "selection": proof.get("selection", "unresolved"),
                "mdl_bits_per_candidate": proof.get("mdl_bits_per_candidate", 0.0),
                "proof_permutations": proof.get("permutations", 0),
                "proof_completions": proof.get("completions", 0),
                "minimal_clique": proof.get("minimal_clique", 0),
                "labeling_space": proof.get("labeling_space", ""),
            })
            if return_candidates and "candidate_tables" in proof:
                common["candidate_tables"] = proof["candidate_tables"]
            if proof.get("status") in {"ambiguous", "budget-exceeded"}:
                return {"status": proof["status"], **common}
            if proof.get("status") == "proved":
                selected = proof["table"]
                representation = {
                    "kind": "finite_state_transducer",
                    "states": evidence.state_count(selected),
                    "alphabet": list(alphabet), "initial": 0,
                    "transitions": [
                        [state, symbol, target, output]
                        for (state, symbol), (target, output) in sorted(selected.items())
                    ],
                }
                return {"status": "proved", "representation": representation, **common}
            # N-clique daha küçük makineyi dışlar; N kanıtı yoksa N−1'e inmek
            # geçerli bir çözüm üretmez, sonucu unresolved olarak koru.
            return {"status": "unresolved", **common}
        common["states"] = 0
        return {"status": "unresolved", "reason": "no-clique", **common}

    def invent_representation(self, train, validation, *, max_states: int = 3,
                              min_states: int = 1, max_expansions: int = 100000,
                              selection_strategy: str = "recurrent",
                              selection_seed: int = 0,
                              return_candidates: bool = False,
                              allow_dfs_fallback: bool = True) -> dict:
        """Eski temsil yetmezse minimal gizli-durum transdüseri kur.

        Önce mevcut akış ifadeleri denenir. Yeni tür yalnız başarısızlıkta açılır;
        validation arama kısıtıdır, held-out ise bu yönteme hiç verilmez.
        Validation'dan geçen minimal adaylar eşdeğer değilse varsayılan olarak
        seçim yapılmaz; tek istisna, exhaustively sayılmış alternatifler arasında
        yalnız bir güçlü-bağlı minimal makine bulunmasıdır. Deney amaçlı açık
        uniform-random/lexicographic/MDL stratejileri de aynı aday kanıtını kullanır.
        Her seçim ``selection`` alanında matematiksel teklikten ayrı bildirilir.
        ``min_states`` daha küçük modellerin önceki exhaustive kanıtla elendiği
        aktif-sorgu devamları içindir. ``return_candidates`` ham iç tabloları
        yalnız aday-filtreleme sürücüsüne açar; normal JSON sonucunu büyütmez.
        ``allow_dfs_fallback=False`` characteristic recovery deneyinde pahalı
        son çare aramasını devre dışı bırakır; varsayılan davranış korunur.
        """
        train_rows = self._relation_examples(train)
        validation_rows = self._relation_examples(validation)
        if not train_rows or not validation_rows:
            return {"status": "invalid-protocol"}
        if not 1 <= int(min_states) <= max(1, int(max_states)):
            return {"status": "invalid-protocol", "reason": "state-bound"}
        stream = self._synthesize_stream([(s, list(y)) for s, y in train_rows])
        if stream is not None and all(self._run_stream(stream, s) == list(y)
                                     for s, y in validation_rows):
            return {"status": "existing", "kind": "stream_expression"}
        alphabet = sorted({symbol for source, _ in train_rows for symbol in source})
        if any(symbol not in alphabet for source, _ in validation_rows for symbol in source):
            return {"status": "unresolved", "kind": "finite_state_transducer",
                    "reason": "validation-alphabet"}
        total_expansions = 0
        peak_stack = 0
        expansion_limit = max(1, int(max_expansions))
        constraints = train_rows + validation_rows
        characteristic = self._finite_characteristic(
            constraints, alphabet, max(1, int(max_states)), expansion_limit,
            min_states=max(1, int(min_states)),
            selection_strategy=selection_strategy, selection_seed=selection_seed,
            return_candidates=return_candidates,
        )
        total_expansions += int(characteristic.get("expansions", 0))
        if characteristic.get("status") == "proved":
            canonical = characteristic["representation"]
            serial = 1 + max(
                [int(name[4:]) for name in self.representations if name.startswith("repr")
                 and name[4:].isdigit()] or [0],
            )
            name = f"repr{serial}"
            canonical["evidence"] = {
                "train": len(train_rows), "validation": len(validation_rows),
                "validated_candidates": characteristic.get("candidates", 0),
                "recurrent_candidates": characteristic.get("recurrent_candidates", 0),
                "selection": characteristic.get("selection"),
                "selection_strategy": characteristic.get("selection_strategy", selection_strategy),
                "selection_seed": characteristic.get("selection_seed", int(selection_seed)),
                "mdl_bits_per_candidate": characteristic.get("mdl_bits_per_candidate", 0.0),
                "algorithm": characteristic.get("algorithm"),
                "constraint_checks": characteristic.get("constraint_checks", 0),
                "proof_permutations": characteristic.get("proof_permutations", 0),
                "proof_completions": characteristic.get("proof_completions", 0),
                "minimal_clique": characteristic.get("minimal_clique", 0),
                "labeling_space": characteristic.get("labeling_space", ""),
                "search_expansions": total_expansions,
            }
            self.representations[name] = canonical
            public = {key: value for key, value in characteristic.items()
                      if key not in {"status", "representation", "candidate_tables"}}
            return {"status": "invented", "id": name, **public}
        if characteristic.get("status") in {"ambiguous", "budget-exceeded"}:
            return characteristic
        if not allow_dfs_fallback:
            return characteristic
        for states in range(max(1, int(min_states)), max(1, int(max_states)) + 1):
            candidates, stats = self._finite_candidates(
                constraints, states, alphabet, max_expansions=expansion_limit,
            )
            total_expansions += int(stats["expansions"])
            peak_stack = max(peak_stack, int(stats["peak_stack"]))
            common = {"kind": "finite_state_transducer", "states": states,
                      "expansions": total_expansions, "peak_stack": peak_stack,
                      "candidate_limit_reached": bool(stats.get("candidate_limit_reached")),
                      "candidate_limit": int(stats.get("candidate_limit", 0))}
            if stats["budget_exceeded"]:
                return {"status": "budget-exceeded", **common}
            if stats.get("candidate_limit_reached"):
                return {"status": "candidate-limit", "candidates": len(candidates), **common}
            # Aramada kullanılan validation ayrıca yürütülür; tablo kurma hatası
            # veya kısıt atlaması bu bağımsız kontrolde başarıya dönüşemez.
            validated = [candidate for candidate in candidates
                         if all(self._run_finite(candidate, source) == expected
                                for source, expected in validation_rows)]
            if not validated:
                continue
            canonical = validated[0]
            if any(not self._finite_equivalent(canonical, other, alphabet)
                   for other in validated[1:]):
                result = {"status": "ambiguous", "candidates": len(validated), **common}
                if return_candidates:
                    result["candidate_representations"] = validated
                return result
            serial = 1 + max(
                [int(name[4:]) for name in self.representations if name.startswith("repr")
                 and name[4:].isdigit()] or [0],
            )
            name = f"repr{serial}"
            canonical["evidence"] = {
                "train": len(train_rows), "validation": len(validation_rows),
                "equivalent_candidates": len(validated),
                "search_expansions": total_expansions,
            }
            self.representations[name] = canonical
            return {"status": "invented", "id": name,
                    "candidates": len(validated), **common}
        return {"status": "unresolved", "kind": "finite_state_transducer",
                "expansions": total_expansions, "peak_stack": peak_stack}

    def learn_representation_active(
            self, train, validation, pool, oracle, *, alphabet: str,
            lengths, max_states: int = 3, min_states: int = 1,
            budget: int = 64, cost_budget: float | None = None,
            query_cost=1.0, repetitions: int = 1,
            adaptive_voting: bool = False, trusted_oracle: bool = True,
            secondary_oracle=None, secondary_trusted_oracle: bool = False,
            candidate_sample: int = 32, recovery_batch: int = 24,
            max_expansions: int = 100000) -> dict:
        """Sorgu seçerek FST öğren; maliyet ve kanıt düzeyini ayrı raporla.

        ``oracle`` yalnız seçilen kaynağı görür. ``budget`` gerçek oracle
        çağrılarını, ``cost_budget`` ise çağrı maliyetlerinin toplamını sınırlar.
        ``query_cost(source, attempt)`` veya sabit maliyet kullanılabilir.
        ``repetitions`` tek sayıdır; cevaplar modelden bağımsız olarak konum
        bazlı çoğunlukla birleştirilir. ``adaptive_voting`` açıkken gerekli oy
        sayısına daha önce ulaşılırsa kalan çağrılar yapılmaz; üçlü oyda iki
        cevabın bütün konumları uyuşuyorsa üçüncü çağrıdan kaçınılır.
        ``trusted_oracle=False`` bağımsız doğrulaması olmayan etiketleri kalıcı
        kabul için yeterli saymaz; teknik durma ``technical_stop`` alanında
        korunarak ``oracle-inconclusive`` döner. ``secondary_oracle`` verilirse
        birincil sorgu kaynaklarını ve son adayın conformance sondalarını ayrı
        bir acceptance/diagnostic kaydıyla yeniden sorar. Kalıcı kabul için
        ikinci kaynağın açıkça ``secondary_trusted_oracle=True`` olması, bütün
        kontrollerin tamamlanması ve sıfır mismatch gerekir; kısmen güvenilir
        kaynakta agreement tek başına yeterli değildir. İki kaynağın ortak
        korele yanlışını ayırt edebildiği iddia edilmez. Yalnız güvenilir ve
        conformance-complete aday kalıcı temsil olur. Bütçe/maliyet/oracle
        durmalarında dönen diğer adaylar
        ``provisional``dır ve ``self.representations`` içine yazılmaz.
        """
        if not callable(oracle):
            return {"status": "invalid-protocol", "accepted": False,
                    "reason": "oracle-not-callable"}
        if secondary_oracle is not None and not callable(secondary_oracle):
            return {"status": "invalid-protocol", "accepted": False,
                    "reason": "secondary-oracle-not-callable"}
        if secondary_trusted_oracle and secondary_oracle is None:
            return {"status": "invalid-protocol", "accepted": False,
                    "reason": "trusted-secondary-missing"}
        lengths = list(lengths)
        if (int(budget) < 0 or int(repetitions) < 1
                or int(repetitions) % 2 == 0 or int(candidate_sample) < 2
                or int(recovery_batch) < 0 or not lengths
                or (adaptive_voting and int(repetitions) < 3)):
            return {"status": "invalid-protocol", "accepted": False,
                    "reason": "active-budget-or-repetition"}
        if cost_budget is not None and float(cost_budget) < 0:
            return {"status": "invalid-protocol", "accepted": False,
                    "reason": "cost-budget"}
        train, validation, pool = list(train), list(validation), list(pool)
        used = set()
        queries, syntheses, filters = [], [], []
        selected = candidates = None
        last_unique_candidate = None
        last_key = None
        verification_queue = []
        minimum_states = max(1, int(min_states))
        recovery_reference = None
        recovery_attempted = set()
        oracle_calls = 0
        total_cost = 0.0
        oracle_seconds = 0.0
        secondary_queries = []
        secondary_oracle_calls = 0
        secondary_total_cost = 0.0
        secondary_oracle_seconds = 0.0
        secondary_check_ran = False
        secondary_mismatches = 0
        secondary_primary_mismatches = 0
        secondary_errors = 0
        train_rows = len(train)

        def report(stop: str) -> dict:
            nonlocal selected
            technical_stop = stop
            recovered_candidate = False
            if (selected is None and last_unique_candidate is not None and
                    (technical_stop == "state-cap" or technical_stop.startswith("synthesis-"))):
                # Geçici etiket reddi veya yeni sentez başarısızlığı son tekil
                # adayın held-out değerini yok etmez. Adayı kabul etme; reddin
                # kendisi de güvenilmez olabileceğinden yalnız provisional geri ver.
                selected = copy.deepcopy(last_unique_candidate)
                recovered_candidate = True
            if secondary_oracle is not None and not secondary_check_ran and (queries or selected):
                run_secondary_check()
            secondary_finished = bool(
                secondary_check_ran and secondary_oracle_calls > 0 and not secondary_errors
            )
            if secondary_oracle is not None:
                accepted = bool(
                    secondary_trusted_oracle and secondary_finished
                    and not secondary_mismatches and not secondary_primary_mismatches
                    and selected is not None
                    and technical_stop == "unique-conformance-complete"
                )
            else:
                accepted = (trusted_oracle and technical_stop == "unique-conformance-complete"
                            and selected is not None)
            if ((not trusted_oracle and not accepted and (
                    stop == "unique-conformance-complete" or (oracle_calls and (
                        stop in {"state-cap", "no-candidates", "verification-reduced",
                                 "no-pool-disagreement"} or stop.startswith("synthesis-")))))
                    or (secondary_oracle is not None and not accepted
                        and technical_stop == "unique-conformance-complete")):
                # Tek taraflı güvenilmez etiketle veya başarısız bağımsız
                # acceptance kontrolüyle model sınırı kalıcı kanıt sayılmaz.
                stop = "oracle-inconclusive"
            representation = copy.deepcopy(selected) if selected is not None else None
            if representation is not None and not accepted:
                representation["evidence"] = {
                    **(representation.get("evidence") or {}),
                    "quarantine": technical_stop, "persistent": False,
                    "accepted_conformance": False,
                }
            for query in queries:
                query["evidence_status"] = "accepted" if accepted else "quarantined"
            for query in secondary_queries:
                query["evidence_status"] = "accepted" if accepted else "quarantined"
            out = {
                "status": "invented" if accepted else "unresolved",
                "stop": stop, "technical_stop": technical_stop,
                "accepted": accepted, "trusted_oracle": bool(trusted_oracle),
                "secondary_trusted_oracle": bool(secondary_trusted_oracle),
                "secondary_check_ran": bool(secondary_check_ran),
                "secondary_complete": secondary_finished,
                "secondary_mismatches": int(secondary_mismatches),
                "secondary_primary_mismatches": int(secondary_primary_mismatches),
                "secondary_errors": int(secondary_errors),
                "secondary_queries": secondary_queries,
                "secondary_query_count": len(secondary_queries),
                "secondary_oracle_calls": secondary_oracle_calls,
                "secondary_total_cost": secondary_total_cost,
                "secondary_oracle_seconds": secondary_oracle_seconds,
                "combined_oracle_calls": oracle_calls + secondary_oracle_calls,
                "combined_total_cost": total_cost + secondary_total_cost,
                "provisional": selected is not None and not accepted,
                "representation": representation,
                "quarantined": not accepted,
                "quarantined_queries": 0 if accepted else len(queries),
                "quarantined_secondary_queries": 0 if accepted else len(secondary_queries),
                "persistent_writes": int(accepted),
                "candidate_recovered": recovered_candidate,
                "queries": queries, "syntheses": syntheses, "filters": filters,
                "train_rows": train_rows, "validation_rows": len(validation),
                "conformance_probe_count": len(verification_queue),
                "query_count": len(queries), "oracle_calls": oracle_calls,
                "total_cost": total_cost, "oracle_seconds": oracle_seconds,
                "cost_budget": cost_budget, "repetitions": int(repetitions),
                "adaptive_voting": bool(adaptive_voting),
            }
            if accepted:
                canonical = copy.deepcopy(selected)
                canonical["evidence"] = {
                    **(canonical.get("evidence") or {}),
                    "active_conformance": "complete-within-state-cap",
                    "active_queries": len(queries), "oracle_calls": oracle_calls,
                    "oracle_cost": total_cost, "repetitions": int(repetitions),
                    "adaptive_voting": bool(adaptive_voting),
                    "secondary_acceptance": bool(secondary_oracle is not None),
                    "secondary_oracle_calls": secondary_oracle_calls,
                    "secondary_mismatches": secondary_mismatches,
                    "secondary_primary_mismatches": secondary_primary_mismatches,
                }
                serial = 1 + max(
                    [int(name[4:]) for name in self.representations
                     if name.startswith("repr") and name[4:].isdigit()] or [0],
                )
                name = f"repr{serial}"
                self.representations[name] = canonical
                out["id"] = name
                out["representation"] = copy.deepcopy(canonical)
            return out

        def ask(source: str, reason: str, **metadata):
            """Tek kaynak için çağrıları ve bağımsız çoğunluğu hesapla."""
            nonlocal oracle_calls, total_cost, oracle_seconds
            responses, errors, costs, timings = [], [], [], []
            stop = None
            consensus = None
            agreement = 0.0

            def resolve(required_votes: int) -> bool:
                nonlocal consensus, agreement
                emitted = []
                minimum_votes = len(responses)
                for position in range(len(source)):
                    counts = collections.Counter(response[position] for response in responses)
                    character, votes = max(counts.items(), key=lambda row: row[1])
                    if votes < required_votes:
                        return False
                    emitted.append(character)
                    minimum_votes = min(minimum_votes, votes)
                consensus = "".join(emitted)
                agreement = minimum_votes / len(responses)
                return True

            for attempt in range(int(repetitions)):
                if oracle_calls >= int(budget):
                    stop = "budget"
                    break
                try:
                    price = (float(query_cost(source, attempt)) if callable(query_cost)
                             else float(query_cost))
                except Exception:
                    stop = "invalid-cost"
                    break
                if not math.isfinite(price) or price < 0:
                    stop = "invalid-cost"
                    break
                if (cost_budget is not None
                        and total_cost + price > float(cost_budget) + 1e-12):
                    stop = "cost-budget"
                    break
                started = time.perf_counter()
                oracle_calls += 1
                total_cost += price
                costs.append(price)
                try:
                    answer = oracle(source)
                except Exception as exc:
                    errors.append(type(exc).__name__)
                    timings.append(time.perf_counter() - started)
                    oracle_seconds += timings[-1]
                    stop = "oracle-unavailable"
                    break
                timings.append(time.perf_counter() - started)
                oracle_seconds += timings[-1]
                if not isinstance(answer, str):
                    errors.append("non-string-response")
                    stop = "oracle-unavailable"
                    break
                if len(answer) != len(source):
                    responses.append(answer)
                    errors.append("response-length")
                    stop = "oracle-invalid"
                    break
                responses.append(answer)
                required = int(repetitions) // 2 + 1
                if adaptive_voting and len(responses) >= required and resolve(required):
                    break
            if stop is None and consensus is None:
                majority = int(repetitions) // 2 + 1
                if not resolve(majority):
                    stop = "oracle-inconclusive"
            if costs or responses or errors:
                queries.append({
                    "source": source, "output": consensus, "reason": reason,
                    "responses": responses, "errors": errors,
                    "calls": len(costs), "call_costs": costs, "cost": sum(costs),
                    "call_ms": [1000 * value for value in timings],
                    "response_status": stop or "answered",
                    "adaptive_voting": bool(adaptive_voting),
                    "used_for_learning": False,
                    "consensus_agreement": agreement, **metadata,
                })
            if stop is not None:
                return None, stop
            used.add(source)
            return consensus, None

        def run_secondary_check() -> None:
            """Bağımsız kaynağı öğrenmeye değil, son kanıt denetimine bağla."""
            nonlocal secondary_check_ran, secondary_oracle_calls
            nonlocal secondary_total_cost, secondary_oracle_seconds
            nonlocal secondary_mismatches, secondary_primary_mismatches, secondary_errors
            secondary_check_ran = True
            primary_outputs = {
                query["source"]: query.get("output") for query in queries
                if query.get("output") is not None
            }
            sources = set(primary_outputs)
            if selected is not None:
                sources.update(verification_queue)
            for source in sorted(sources):
                primary_output = primary_outputs.get(source)
                try:
                    expected = (_active_run_finite(selected, source)
                                if selected is not None else primary_output)
                except Exception:
                    expected = primary_output
                if expected is None:
                    continue
                error = None
                answer = None
                price = 0.0
                attempted = False
                try:
                    price = (float(query_cost(source, 0)) if callable(query_cost)
                             else float(query_cost))
                    if not math.isfinite(price) or price < 0:
                        error = "invalid-cost"
                except Exception:
                    error = "invalid-cost"
                if (error is None and cost_budget is not None and
                        total_cost + secondary_total_cost + price > float(cost_budget) + 1e-12):
                    error = "cost-budget"
                started = time.perf_counter()
                if error is None:
                    attempted = True
                    secondary_oracle_calls += 1
                    secondary_total_cost += price
                    try:
                        answer = secondary_oracle(source)
                        if not isinstance(answer, str) or len(answer) != len(source):
                            error = "response-length"
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                    secondary_oracle_seconds += time.perf_counter() - started
                matched = bool(error is None and answer == expected)
                primary_matched = bool(
                    error is None and primary_output is not None and answer == primary_output
                )
                if error is not None:
                    secondary_errors += 1
                elif not matched:
                    secondary_mismatches += 1
                if error is None and primary_output is not None and not primary_matched:
                    secondary_primary_mismatches += 1
                secondary_queries.append({
                    "source": source, "output": answer, "prediction": expected,
                    "primary_output": primary_output, "prediction_matched": matched,
                    "primary_matched": primary_matched,
                    "reason": "independent-acceptance", "used_for_learning": False,
                    "calls": int(attempted), "cost": price if attempted else 0.0,
                    "errors": [error] if error else [],
                    "response_status": error or "answered",
                })
                if error == "cost-budget":
                    break

        stop = "budget"
        while True:
            if candidates is None:
                # Aday sentezleri geçicidir; yalnız son conformance-complete
                # temsil report() tarafından kalıcı learner'a aktarılır.
                learner = copy.deepcopy(self)
                learner.representations = {}
                synthesis = learner.invent_representation(
                    train, validation, min_states=minimum_states, max_states=max_states,
                    max_expansions=max_expansions,
                    return_candidates=True, allow_dfs_fallback=False,
                )
                tables = synthesis.pop("candidate_tables", None)
                representations = synthesis.pop("candidate_representations", None)
                syntheses.append(synthesis)
                if synthesis.get("status") == "invented":
                    selected = learner.representations[synthesis["id"]]
                    candidates = [selected]
                elif synthesis.get("status") == "ambiguous":
                    selected = None
                    candidates = tables if tables is not None else (representations or [])
                else:
                    status = str(synthesis.get("status"))
                    recovery_key = None
                    reference = None
                    if recovery_reference is not None:
                        reference = (recovery_reference if "transitions" in recovery_reference
                                     else _active_table_representation(recovery_reference, alphabet))
                        recovery_key = (minimum_states, json.dumps(
                            reference["transitions"], sort_keys=True,
                        ))
                    if (status in {"budget-exceeded", "unresolved"} and minimum_states > 1
                            and recovery_key is not None
                            and recovery_key not in recovery_attempted
                            and oracle_calls < int(budget)):
                        recovery_attempted.add(recovery_key)
                        probes = _active_conformance_probes(
                            reference, alphabet, lengths, max_states,
                        )
                        extra = [source for source in probes if source not in used]
                        extra = extra[:int(recovery_batch)]
                        if extra:
                            for source in extra:
                                if oracle_calls >= int(budget):
                                    break
                                prediction = _active_run_finite(reference, source)
                                query_index = len(queries)
                                expected, query_stop = ask(
                                    source, "recovery-conformance", candidates_before=0,
                                    prediction=prediction, prediction_matched=None,
                                )
                                if len(queries) > query_index:
                                    queries[-1]["prediction_matched"] = (
                                        prediction == expected if expected is not None else None
                                    )
                                if query_stop:
                                    return report(query_stop)
                                used.add(source)
                                if queries and queries[-1]["source"] == source:
                                    queries[-1]["used_for_learning"] = True
                                validation.append((source, expected))
                            candidates = selected = None
                            continue
                    stop = "synthesis-" + status
                    break
                if not candidates:
                    stop = "no-candidates"
                    break
            if candidates:
                recovery_reference = candidates[0]
            if len(candidates) == 1:
                if "transitions" not in candidates[0]:
                    selected = _active_table_representation(candidates[0], alphabet)
                    candidates = [selected]
                else:
                    selected = candidates[0]
                last_unique_candidate = copy.deepcopy(selected)
                key = json.dumps(selected.get("transitions", []), sort_keys=True)
                if key != last_key:
                    verification_queue = _active_conformance_probes(
                        selected, alphabet, lengths, max_states,
                    )
                    last_key = key
            else:
                selected = last_key = None
                verification_queue = []
            verification_mismatch = False
            while len(candidates) == 1 and oracle_calls < int(budget):
                unused = [source for source in verification_queue if source not in used]
                if not unused:
                    return report("unique-conformance-complete")
                source = unused[0]
                prediction = _active_run_finite(selected, source)
                query_index = len(queries)
                expected, query_stop = ask(
                    source, "conformance", candidates_before=1, prediction=prediction,
                    prediction_matched=None,
                )
                if len(queries) > query_index:
                    queries[-1]["prediction_matched"] = (
                        prediction == expected if expected is not None else None
                    )
                if query_stop:
                    return report(query_stop)
                used.add(source)
                if queries and queries[-1]["source"] == source:
                    queries[-1]["used_for_learning"] = True
                validation.append((source, expected))
                if prediction != expected:
                    verification_mismatch = True
                    break
            if verification_mismatch:
                minimum_states = max(minimum_states, int(selected["states"]) + 1)
                candidates = selected = None
                if minimum_states > int(max_states):
                    stop = "state-cap"
                    break
                if oracle_calls >= int(budget):
                    stop = "budget"
                    break
                continue
            if len(candidates) == 1:
                stop = "budget" if oracle_calls >= int(budget) else "verification-reduced"
                break
            if oracle_calls >= int(budget):
                stop = "budget"
                break
            unused = [source for source in pool if source not in used]
            if not unused:
                stop = "pool-exhausted"
                break
            count = min(int(candidate_sample), len(candidates))
            indices = {0, len(candidates) - 1}
            if count > 2:
                indices.update((i * (len(candidates) - 1)) // (count - 1)
                               for i in range(count))
            sample = [candidates[index] for index in sorted(indices)]
            best = None
            for length in lengths:
                tier = [source for source in unused if len(source) == int(length)]
                for source in tier:
                    outputs = [_active_run_finite(candidate, source) for candidate in sample]
                    counts = collections.Counter(outputs)
                    distinct = len(counts)
                    if distinct <= 1:
                        continue
                    entropy = -sum((number / len(outputs)) * math.log(number / len(outputs))
                                   for number in counts.values())
                    score = (distinct, entropy)
                    if best is None or score > best[0]:
                        best = (score, source, distinct, entropy)
                if best is not None:
                    break
            if best is None:
                stop = "no-pool-disagreement"
                break
            _score, source, distinct, entropy = best
            before = len(candidates)
            expected, query_stop = ask(
                source, "disagreement", candidates_before=before,
                sample_distinct_outputs=distinct, sample_entropy=entropy,
                prediction=None, prediction_matched=None,
            )
            if query_stop:
                return report(query_stop)
            candidates = [candidate for candidate in candidates
                          if _active_run_finite(candidate, source) == expected]
            filters.append({"before": before, "after": len(candidates), "expansions": 0})
            queries[-1]["candidates_after"] = len(candidates)
            used.add(source)
            if queries and queries[-1]["source"] == source:
                queries[-1]["used_for_learning"] = True
            validation.append((source, expected))
            if not candidates:
                minimum_states = max(minimum_states, int(synthesis.get("states", max_states)) + 1)
                candidates = None
                if minimum_states > int(max_states):
                    stop = "state-cap"
                    break
        return report(stop)

    def invent_continuous_representation(
            self, train, validation, *, hidden: int = 48, seed: int = 0,
            learning_rate: float = 0.01, max_epochs: int = 240,
            batch_size: int = 32, patience: int = 60,
            class_balance: bool = True, prefix_augmentation: int = 0,
            prefix_warmup_epochs: int = 0,
            checkpoint_selection: str = "train_first",
            causal_summary: bool | str = "auto") -> dict:
        """FST'den bağımsız sürekli nedensel-grafik adayını aynı veriyle öğren.

        Dönen ``provisional`` kimlik yalnız araştırma/held-out ölçümünde
        ``allow_provisional=True`` ile çalıştırılabilir. Normal üretim çağrısı
        kabul edilmemiş modeli çalıştırmaz; bu ayrım FST'nin exhaustive kanıtıyla
        sinirsel validation skorunu birbirine karıştırmaz. ``auto`` iki mimariyi
        yalnız train/validation kanıtıyla seçer; held-out bu karara katılmaz.
        """
        if isinstance(causal_summary, str) and causal_summary != "auto":
            raise ValueError("unknown causal summary mode")

        def fit_candidate(with_summary: bool):
            candidate = ContinuousCausalGraph(
                hidden=hidden, seed=seed, learning_rate=learning_rate,
                max_epochs=max_epochs, batch_size=batch_size, patience=patience,
                class_balance=class_balance, prefix_augmentation=prefix_augmentation,
                prefix_warmup_epochs=prefix_warmup_epochs,
                checkpoint_selection=checkpoint_selection,
                causal_summary=with_summary,
            )
            candidate_report = candidate.fit(train, validation)
            return candidate, candidate_report

        if causal_summary == "auto":
            baseline, baseline_report = fit_candidate(False)
            summary, summary_report = fit_candidate(True)
            # Held-out burada yoktur. Önce özgün validation'daki tam dizi
            # sayısı; eşitlikte yalnız bir tokenlik gürültüyü aşan token ve
            # makro kazanç. Kanıt yoksa küçük baseline kalır; aile/modulus yok.
            summary_exact = summary_report["validation_exact_sequences"]
            baseline_exact = baseline_report["validation_exact_sequences"]
            validation_tokens = sum(len(str(expected)) for _source, expected in validation)
            token_gain = (summary_report["validation_token_accuracy"] -
                          baseline_report["validation_token_accuracy"])
            token_correct_gain = (
                round(summary_report["validation_token_accuracy"] * validation_tokens) -
                round(baseline_report["validation_token_accuracy"] * validation_tokens))
            macro_gain = (summary_report["validation_macro_token_accuracy"] -
                          baseline_report["validation_macro_token_accuracy"])
            use_summary = (summary_exact > baseline_exact or
                           (summary_exact == baseline_exact and
                            token_correct_gain > 1 and macro_gain > 0.0))
            model, report = ((summary, dict(summary_report)) if use_summary
                             else (baseline, dict(baseline_report)))
            report["selected_gradient_steps"] = report["gradient_steps"]
            report["selected_example_presentations"] = report["example_presentations"]
            report["selected_seconds"] = report["seconds"]
            report["gradient_steps"] = (baseline_report["gradient_steps"] +
                                        summary_report["gradient_steps"])
            report["example_presentations"] = (
                baseline_report["example_presentations"] +
                summary_report["example_presentations"])
            report["seconds"] = baseline_report["seconds"] + summary_report["seconds"]
            report["candidate_selection"] = {
                "rule": "validation_exact_or_multi_token_macro_else_baseline",
                "selected_causal_summary": bool(use_summary),
                "validation_tokens": validation_tokens,
                "validation_token_gain": token_gain,
                "validation_token_correct_gain": token_correct_gain,
                "validation_macro_gain": macro_gain,
                "heldout_used": False,
                "candidates": {
                    "baseline": copy.deepcopy(baseline_report),
                    "causal_summary": copy.deepcopy(summary_report),
                },
            }
            model.report = copy.deepcopy(report)
        else:
            model, report = fit_candidate(bool(causal_summary))
        serial = 1 + max(
            [int(name[4:]) for name in self.representations if name.startswith("repr")
             and name[4:].isdigit()] or [0],
        )
        name = f"repr{serial}"
        representation = model.state_dict()
        representation["acceptance"] = "accepted" if report["status"] == "accepted" else "provisional"
        self.representations[name] = representation
        self._continuous_models[name] = model
        accepted = representation["acceptance"] == "accepted"
        return {
            **report, "status": "invented" if accepted else "provisional",
            "id": name, "kind": ContinuousCausalGraph.KIND,
            "acceptance": representation["acceptance"],
        }

    def apply_representation(self, name: str, source: str, *,
                             allow_provisional: bool = False) -> str | None:
        representation = self.representations.get(str(name))
        if not representation:
            return None
        if representation.get("acceptance") == "provisional" and not allow_provisional:
            return None
        if representation.get("kind") == "finite_state_transducer":
            return self._run_finite(representation, str(source))
        if representation.get("kind") == ContinuousCausalGraph.KIND:
            model = self._continuous_models.get(str(name))
            if model is None:
                try:
                    model = ContinuousCausalGraph.from_dict(representation)
                except (ValueError, KeyError, RuntimeError, TypeError):
                    return None
                self._continuous_models[str(name)] = model
            return model.apply(str(source))
        return None

    # -------------------------------------------------------------- planning
    def _transitions(self, goal: SemanticGoal, stage: int) -> list[tuple[str, int]]:
        if stage == 0:
            invented = [(name, 1) for name in self._matching_streams(goal)]
            if goal.cumulative:
                direct = []
                if goal.unit == "character" and self.facts.get("text_sequence"):
                    if goal.cumulative_step > 1 and "cumulative_window" in self.operators:
                        direct.append(("cumulative_window", 1))
                    elif goal.cumulative_step == 1 and "accumulate" in self.operators:
                        direct.append(("accumulate", 1))
                return direct + invented
            if goal.unit == "character" and self.facts.get("text_sequence"):
                return [("character_view", 1)] + invented
            if goal.unit == "word" and "split_words" in self.operators:
                return [("split_words", 1)]
            return []
        if stage == 1:
            return [("iterate", 2)] if "iterate" in self.operators else []
        if stage == 2:
            if "display" in self.operators and self.facts.get("print_newline"):
                return [("display", 3)]
        return []

    def _search(self, goal: SemanticGoal) -> tuple[str, ...] | None:
        frontier = [(0, ())]
        seen = {0}
        for stage, path in frontier:
            if stage == 3:
                return path
            for operator, next_stage in self._transitions(goal, stage):
                if next_stage in seen:
                    continue
                seen.add(next_stage)
                frontier.append((next_stage, path + (operator,)))
        return None

    def _path_available(self, path: tuple[str, ...]) -> bool:
        for operator in path:
            if operator in {"character_view", "accumulate", "cumulative_window"} and not self.facts.get("text_sequence"):
                return False
            if operator.startswith("invent:"):
                construction = self.stream_operators.get(operator.split(":", 1)[1])
                if not construction or not self._construction_uses(construction).issubset(self.operators):
                    return False
                continue
            if operator == "display" and not self.facts.get("print_newline"):
                return False
            if operator not in {"character_view"} and operator not in self.operators:
                return False
        return True

    def _missing(self, goal: SemanticGoal) -> list[str]:
        missing = []
        if goal.unit == "character" and not self.facts.get("text_sequence"):
            missing.append("text_sequence")
        if goal.cumulative and not self._matching_streams(goal):
            required = "cumulative_window" if goal.cumulative_step > 1 else "accumulate"
            if required not in self.operators:
                missing.append(required)
        if goal.unit == "word" and "split_words" not in self.operators:
            missing.append("split_words")
        if goal.unit not in {"character", "word"}:
            missing.append(f"unit:{goal.unit}")
        if "iterate" not in self.operators:
            missing.append("iterate")
        if "display" not in self.operators:
            missing.append("display")
        if not self.facts.get("print_newline"):
            missing.append("print_newline")
        return missing

    @staticmethod
    def _ast_expression(expression: tuple) -> ast.expr:
        kind = expression[0]
        if kind == "const":
            return ast.Constant(value=str(expression[1]))
        if kind == "name":
            return ast.Name(id=str(expression[1]), ctx=ast.Load())
        if kind == "add":
            return ast.BinOp(
                left=CompositionalCodeLearner._ast_expression(expression[1]),
                op=ast.Add(),
                right=CompositionalCodeLearner._ast_expression(expression[2]),
            )
        raise ValueError("unknown primitive expression")

    def _emit(self, goal: SemanticGoal, path: tuple[str, ...]) -> str:
        source: ast.expr = ast.Constant(value=goal.payload)
        statements: list[ast.stmt] = []
        invented = next((name for name in path if name.startswith("invent:")), "")
        if invented:
            construction = self.stream_operators[invented.split(":", 1)[1]]
            state = ast.Name(id="state", ctx=ast.Store())
            statements.append(ast.Assign(
                targets=[state], value=self._ast_expression(construction["initial"]),
            ))
            update = ast.Assign(
                targets=[ast.Name(id="state", ctx=ast.Store())],
                value=self._ast_expression(construction["update"]),
            )
            display = ast.Expr(value=ast.Call(
                func=ast.Name(id="print", ctx=ast.Load()),
                args=[self._ast_expression(construction["emit"])], keywords=[],
            ))
            body = [update, display] if construction.get("after") else [display, update]
            statements.append(ast.For(
                target=ast.Name(id="item", ctx=ast.Store()), iter=source,
                body=body, orelse=[],
            ))
        else:
            if "cumulative_window" in path:
                step = max(1, int(goal.cumulative_step))
                end = ast.Name(id="end", ctx=ast.Load())
                iterator = ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[ast.Constant(value=step),
                          ast.BinOp(left=ast.Call(func=ast.Name(id="len", ctx=ast.Load()),
                                                  args=[source], keywords=[]),
                                    op=ast.Add(), right=ast.Constant(value=1)),
                          ast.Constant(value=step)], keywords=[],
                )
                body = ast.Expr(value=ast.Call(
                    func=ast.Name(id="print", ctx=ast.Load()),
                    args=[ast.Subscript(value=source,
                                        slice=ast.Slice(lower=None, upper=end, step=None),
                                        ctx=ast.Load())], keywords=[],
                ))
                statements.append(ast.For(
                    target=ast.Name(id="end", ctx=ast.Store()), iter=iterator,
                    body=[body], orelse=[],
                ))
                module = ast.Module(body=statements, type_ignores=[])
                ast.fix_missing_locations(module)
                return ast.unparse(module)
            if "accumulate" in path:
                statements.append(ast.ImportFrom(
                    module="itertools", names=[ast.alias(name="accumulate")], level=0,
                ))
                source = ast.Call(func=ast.Name(id="accumulate", ctx=ast.Load()),
                                  args=[source], keywords=[])
            elif "split_words" in path:
                source = ast.Call(func=ast.Attribute(value=source, attr="split", ctx=ast.Load()),
                                  args=[], keywords=[])
            variable = "prefix" if goal.cumulative else {"character": "character", "word": "word"}.get(goal.unit, "item")
            body = ast.Expr(value=ast.Call(
                func=ast.Name(id="print", ctx=ast.Load()),
                args=[ast.Name(id=variable, ctx=ast.Load())], keywords=[],
            ))
            loop = ast.For(target=ast.Name(id=variable, ctx=ast.Store()), iter=source,
                           body=[body], orelse=[])
            statements.append(loop)
        module = ast.Module(body=statements, type_ignores=[])
        ast.fix_missing_locations(module)
        return ast.unparse(module)

    @staticmethod
    def _emit_decoder(payload: str, decoder: dict[str, Any]) -> str:
        # Yalnız belgeden gelen sembol tablosu ve genel soldan-sağa kuralı.
        # Hedef girdinin önceden hesaplanmış cevabı programa gömülmez.
        program = "\n".join([
            f"text = {payload!r}",
            f"values = {dict(decoder['symbols'])!r}",
            "total = 0",
            "for index in range(len(text)):",
            "    value = values[text[index]]",
            "    if index + 1 < len(text) and value < values[text[index + 1]]:",
            "        total -= value",
            "    else:",
            "        total += value",
            "print(total)",
        ])
        return ast.unparse(ast.parse(program))

    @staticmethod
    def _emit_representation(payload: str, representation: dict[str, Any]) -> str:
        """Öğrenilmiş transdüseri, cevabı gömmeden çalıştırılabilir AST'ye çevir."""
        if representation.get("kind") != "finite_state_transducer":
            raise ValueError("unsupported representation")
        table = {
            (int(state), str(symbol)): (int(next_state), str(output))
            for state, symbol, next_state, output in representation.get("transitions") or ()
        }
        program = "\n".join([
            f"text = {payload!r}",
            f"table = {table!r}",
            f"state = {int(representation.get('initial', 0))!r}",
            "outputs = []",
            "for item in text:",
            "    state, emitted = table[(state, item)]",
            "    outputs.append(emitted)",
            "print(''.join(outputs))",
        ])
        return ast.unparse(ast.parse(program))

    @staticmethod
    def _apply_string_operations(payload: str, operations: list[str]) -> str | None:
        value = str(payload)
        for operation in operations:
            if operation == "reverse":
                value = value[::-1]
            elif operation == "uppercase":
                value = value.upper()
            elif operation == "lowercase":
                value = value.lower()
            elif operation == "titlecase":
                value = value.title()
            else:
                return None
        return value

    @staticmethod
    def _emit_string_transform(payload: str, operations: list[str]) -> str:
        expression: ast.expr = ast.Constant(value=str(payload))
        for operation in operations:
            if operation == "reverse":
                expression = ast.Subscript(
                    value=expression,
                    slice=ast.Slice(lower=None, upper=None,
                                    step=ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1))),
                    ctx=ast.Load(),
                )
            else:
                method = {"uppercase": "upper", "lowercase": "lower", "titlecase": "title"}[operation]
                expression = ast.Call(
                    func=ast.Attribute(value=expression, attr=method, ctx=ast.Load()),
                    args=[], keywords=[],
                )
        module = ast.Module(body=[ast.Expr(value=ast.Call(
            func=ast.Name(id="print", ctx=ast.Load()), args=[expression], keywords=[],
        ))], type_ignores=[])
        ast.fix_missing_locations(module)
        return ast.unparse(module)

    def _solve_transformation(self, goal: TransformationGoal) -> ProgramPlan | dict:
        if goal.operation == "decimal":
            candidates = [(name, record) for name, record in self.value_decoders.items()
                          if goal.payload and all(symbol in record.get("symbols", {})
                                                  for symbol in goal.payload)]
            if not candidates:
                return {"ok": False, "reason": "semantic_gap", "missing": ["value_decoder"],
                        "gap_stage": "decode", "trace": list(goal.trace) + ["gap_stage=decode"]}
            if len(candidates) != 1:
                return {"ok": False, "reason": "ambiguous_decoder", "missing": ["decoder:ambiguous"],
                        "gap_stage": "decode", "trace": list(goal.trace) + ["gap_stage=decode"]}
            name, decoder = candidates[0]
            value = self._decode_with_symbols(decoder["symbols"], goal.payload, decoder["rule"])
            if value is None:
                return {"ok": False, "reason": "unknown_symbol", "missing": ["decoder:symbol"],
                        "gap_stage": "decode", "trace": list(goal.trace) + ["gap_stage=decode"]}
            program = self._emit_decoder(goal.payload, decoder)
            return ProgramPlan(
                (f"decode:{name}", "iterate", "display"), program, f"{value}\n",
                tuple(goal.trace) + (f"decoder={name}", "rule=subtractive_sum"),
            )

        if goal.operation == "representation":
            name = goal.reference or next(reversed(self.representations), "")
            if not name or (goal.reference == "" and len(self.representations) != 1):
                return {"ok": False, "reason": "ambiguous_representation",
                        "missing": ["representation:reference"], "gap_stage": "transform",
                        "trace": list(goal.trace) + ["gap_stage=transform"]}
            representation = self.representations.get(name)
            value = self.apply_representation(name, goal.payload) if representation else None
            if value is None:
                return {"ok": False, "reason": "representation_input_gap",
                        "missing": ["representation:domain"], "gap_stage": "transform",
                        "trace": list(goal.trace) + ["gap_stage=transform"]}
            program = self._emit_representation(goal.payload, representation)
            return ProgramPlan(
                (f"represent:{name}", "iterate", "display"), program, value + "\n",
                tuple(goal.trace) + (f"representation={name}",
                                     f"states={representation.get('states', 0)}"),
            )

        name = goal.reference or next(reversed(self.string_transforms), "")
        record = self.string_transforms.get(name)
        if not record:
            return {"ok": False, "reason": "semantic_gap", "missing": ["string_transformation"],
                    "gap_stage": "transform", "trace": list(goal.trace) + ["gap_stage=transform"]}
        operations = [str(value) for value in record.get("operations") or ()]
        value = self._apply_string_operations(goal.payload, operations)
        if value is None:
            return {"ok": False, "reason": "unknown_transform", "missing": ["transform:operation"],
                    "gap_stage": "transform", "trace": list(goal.trace) + ["gap_stage=transform"]}
        program = self._emit_string_transform(goal.payload, operations)
        return ProgramPlan(
            (f"transform:{name}", "display"), program, value + "\n",
            tuple(goal.trace) + (f"transform={name}", "operations=" + "->".join(operations)),
        )

    def _gap_stage(self, goal: SemanticGoal) -> str:
        if not self._transitions(goal, 0):
            return "unit" if goal.unit not in {"character", "word"} else "transformation"
        if "iterate" not in self.operators:
            return "iteration"
        if "display" not in self.operators or not self.facts.get("print_newline"):
            return "display"
        return "plan"

    def solve(self, goal: SemanticGoal | TransformationGoal) -> ProgramPlan | dict:
        if isinstance(goal, TransformationGoal):
            return self._solve_transformation(goal)
        key = "|".join(goal.key)
        cached = self.composites.get(key)
        path = cached if cached and self._path_available(cached) else self._search(goal)
        if path is None:
            stage = self._gap_stage(goal)
            return {"ok": False, "reason": "semantic_gap", "missing": self._missing(goal),
                    "gap_stage": stage, "trace": list(goal.trace) + [f"gap_stage={stage}"]}
        program = self._emit(goal, path)
        trace = list(goal.trace) + ["plan=" + " -> ".join(path)]
        if cached == path:
            trace.append("composite=reused")
        return ProgramPlan(path, program, goal.expected_output(), tuple(trace))

    def adopt(self, goal: SemanticGoal | TransformationGoal, plan: ProgramPlan) -> None:
        if isinstance(goal, TransformationGoal):
            return
        self.composites["|".join(goal.key)] = tuple(plan.operators)

    # -------------------------------------------------------------- persistence
    def state_dict(self) -> dict:
        return {"operators": sorted(self.operators), "facts": dict(self.facts),
                "composites": {key: list(value) for key, value in self.composites.items()},
                "stream_operators": copy.deepcopy(self.stream_operators),
                "representations": copy.deepcopy(self.representations),
                "value_decoders": copy.deepcopy(self.value_decoders),
                "string_transforms": copy.deepcopy(self.string_transforms),
                "cumulative_rules": dict(self.cumulative_rules),
                "documents": list(self.documents), "events": self.learning_events,
                "failure_experiments": int(self.failure_experiments),
                "last_failure": copy.deepcopy(self.last_failure)}

    def load_state_dict(self, data: dict | None) -> None:
        data = data or {}
        self.operators = {str(value) for value in data.get("operators") or ()}
        self.facts = dict(data.get("facts") or {})
        self.composites = {str(key): tuple(value) for key, value in (data.get("composites") or {}).items()}
        self.stream_operators = {
            str(key): dict(value) for key, value in (data.get("stream_operators") or {}).items()
            if isinstance(value, dict) and {"initial", "update", "emit"}.issubset(value)
        }
        self.representations = copy.deepcopy({
            str(key): value for key, value in (data.get("representations") or {}).items()
            if isinstance(value, dict) and value.get("kind") in {
                "finite_state_transducer", ContinuousCausalGraph.KIND,
            }
        })
        self._continuous_models = {}
        self.value_decoders = {}
        for key, value in (data.get("value_decoders") or {}).items():
            if not isinstance(value, dict) or value.get("rule") != "subtractive_sum":
                continue
            try:
                symbols = {str(symbol): int(number)
                           for symbol, number in (value.get("symbols") or {}).items()}
            except (TypeError, ValueError):
                continue
            if symbols:
                self.value_decoders[str(key)] = {
                    "symbols": symbols, "rule": "subtractive_sum",
                    "signature": repr((sorted(symbols.items()), "subtractive_sum")),
                    "evidence": int(value.get("evidence", 0) or 0),
                }
        allowed = {"reverse", "uppercase", "lowercase", "titlecase"}
        self.string_transforms = {}
        for key, value in (data.get("string_transforms") or {}).items():
            if not isinstance(value, dict):
                continue
            operations = [str(operation) for operation in value.get("operations") or ()]
            if operations and all(operation in allowed for operation in operations):
                alias = str(value.get("name") or "")
                self.string_transforms[str(key)] = {
                    "name": alias, "operations": operations,
                    "signature": repr((alias, operations)),
                }
        self.cumulative_rules = {}
        for phrase, step in (data.get("cumulative_rules") or {}).items():
            try:
                number = int(step)
            except (TypeError, ValueError):
                continue
            if number >= 1:
                self.cumulative_rules[str(phrase)] = number
        if self.cumulative_rules:
            self.operators.add("cumulative_window")
        self.documents = [str(value) for value in data.get("documents") or ()][-40:]
        self.learning_events = int(data.get("events", 0) or 0)
        self.failure_experiments = int(data.get("failure_experiments", 0) or 0)
        failure = data.get("last_failure")
        self.last_failure = copy.deepcopy(failure) if isinstance(failure, dict) else None


__all__ = ["CompositionalCodeLearner", "ContinuousCausalGraph", "SemanticGoal", "TransformationGoal", "ProgramPlan"]
