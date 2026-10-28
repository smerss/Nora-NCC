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
import copy
from dataclasses import dataclass, field
import html
import itertools
import json
from pathlib import Path
import random
import re
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

    def prove(self, initial_table: dict[tuple, tuple], expansion_limit: int) -> dict:
        """Bütün aynı-boyutlu makineleri çapa etiketleri altında say.

        Klik, daha az durumun mümkün olmadığını kanıtlar. Zorunlu geçişler
        yayılır; kalan az sayıdaki geçiş ve çapa permütasyonu exhaustively
        denenir. Birden çok minimal çözüm varsa yalnız tek güçlü-bağlı çözüm
        olması durumunda açık bir recurrent Occam tercihi uygulanır.
        """
        table = self.canonical(initial_table)
        if table is None:
            return {"status": "unresolved", "expansions": 0}
        states = self.state_count(table)
        full_domain = (1 << states) - 1
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
        if found is None:
            return {"status": "unresolved", "reason": "no-clique", "expansions": 0}
        anchors = [None] * states
        for vertex in found:
            anchors[prefix_states[vertex]] = vertex

        valid = set()
        permutations = completions = validations = max_missing = 0
        for permutation in itertools.permutations(range(states)):
            permutations += 1
            if permutations + completions > expansion_limit:
                return {"status": "budget-exceeded", "expansions": permutations + completions}
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
                            choices.union_update(current)
                if not choices:
                    inconsistent = True
                    break
                options.append(sorted(choices))
            if inconsistent:
                continue

            for values in itertools.product(*options):
                completions += 1
                if permutations + completions > expansion_limit:
                    return {"status": "budget-exceeded",
                            "expansions": permutations + completions}
                candidate = dict(forced)
                candidate.update(zip(missing, values))
                canonical = self.canonical(candidate)
                if canonical is None:
                    continue
                anchor_ok = True
                for candidate_state, anchor in enumerate(anchors):
                    traced = self.run(candidate, self.vertices[anchor], trace=True)
                    if traced is None or traced[0][-1] != permutation[candidate_state]:
                        anchor_ok = False
                        break
                if not anchor_ok:
                    continue
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
                    continue
                valid.add(tuple(sorted(canonical.items())))

        tables = [dict(items) for items in valid]
        recurrent = [
            candidate for candidate in tables
            if self._strongly_connected(candidate, states)
        ]
        if len(tables) == 1:
            selected, selection = tables[0], "unique-minimal"
        elif len(recurrent) == 1:
            selected, selection = recurrent[0], "unique-minimal-recurrent"
        else:
            selected, selection = None, "ambiguous"
        stats = {
            "expansions": permutations + completions,
            "permutations": permutations, "completions": completions,
            "validations": validations, "max_missing": max_missing,
            "candidates": len(tables), "recurrent_candidates": len(recurrent),
            "selection": selection, "minimal_clique": states,
        }
        if selected is None:
            return {"status": "ambiguous" if tables else "unresolved", **stats}
        return {"status": "proved", "table": selected, **stats}


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
            return None
        low = tr_lower(raw)
        applies = bool(re.search(r"(?:işlemden\s+geçir|dönüştür|transform|uygula)", low))
        anaphoric = bool(re.search(
            r"\b(?:bu|öğrendiğin|öğrendiğim|kurduğun|kurduğum)\s+(?:işlem|operatör)", low,
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
        capped = len(solutions) >= limit or (expansions >= max_expansions and bool(stack))
        stats = {
            "expansions": expansions, "peak_stack": peak_stack,
            "budget_exceeded": capped, "candidate_limit": int(limit),
            "expansion_limit": int(max_expansions),
        }
        return solutions, stats

    def _finite_characteristic(self, constraints, alphabet, max_states, expansion_limit):
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
                return {"status": "budget-exceeded", **common}
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
                )
                total_expansions += int(proof.get("expansions", 0))
                common.update({
                    "states": minimal, "expansions": total_expansions,
                    "candidates": proof.get("candidates", 0),
                    "recurrent_candidates": proof.get("recurrent_candidates", 0),
                    "selection": proof.get("selection", "unresolved"),
                    "proof_permutations": proof.get("permutations", 0),
                    "proof_completions": proof.get("completions", 0),
                    "minimal_clique": proof.get("minimal_clique", 0),
                })
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
        return {
            "status": "unresolved", "expansions": total_expansions, "peak_stack": 0,
            "constraint_checks": evidence.constraint_checks,
            "prefixes": len(evidence.vertices),
        }

    def invent_representation(self, train, validation, *, max_states: int = 3,
                              max_expansions: int = 100000) -> dict:
        """Eski temsil yetmezse minimal gizli-durum transdüseri kur.

        Önce mevcut akış ifadeleri denenir. Yeni tür yalnız başarısızlıkta açılır;
        validation arama kısıtıdır, held-out ise bu yönteme hiç verilmez.
        Validation'dan geçen minimal adaylar eşdeğer değilse seçim yapılmaz;
        tek istisna, exhaustively sayılmış alternatifler arasında yalnız bir
        güçlü-bağlı minimal makine bulunmasıdır. Bu durum ``selection`` alanında
        matematiksel teklikten ayrı ve açık bir Occam tercihi olarak bildirilir.
        """
        train_rows = self._relation_examples(train)
        validation_rows = self._relation_examples(validation)
        if not train_rows or not validation_rows:
            return {"status": "invalid-protocol"}
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
                "algorithm": characteristic.get("algorithm"),
                "constraint_checks": characteristic.get("constraint_checks", 0),
                "proof_permutations": characteristic.get("proof_permutations", 0),
                "proof_completions": characteristic.get("proof_completions", 0),
                "minimal_clique": characteristic.get("minimal_clique", 0),
                "search_expansions": total_expansions,
            }
            self.representations[name] = canonical
            public = {key: value for key, value in characteristic.items()
                      if key not in {"status", "representation"}}
            return {"status": "invented", "id": name, **public}
        if characteristic.get("status") in {"ambiguous", "budget-exceeded"}:
            return characteristic
        for states in range(1, max(1, int(max_states)) + 1):
            candidates, stats = self._finite_candidates(
                constraints, states, alphabet, max_expansions=expansion_limit,
            )
            total_expansions += int(stats["expansions"])
            peak_stack = max(peak_stack, int(stats["peak_stack"]))
            common = {"kind": "finite_state_transducer", "states": states,
                      "expansions": total_expansions, "peak_stack": peak_stack}
            if stats["budget_exceeded"]:
                return {"status": "budget-exceeded", **common}
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
                return {"status": "ambiguous", "candidates": len(validated), **common}
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

    def apply_representation(self, name: str, source: str) -> str | None:
        representation = self.representations.get(str(name))
        if not representation:
            return None
        if representation.get("kind") == "finite_state_transducer":
            return self._run_finite(representation, str(source))
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
            if isinstance(value, dict) and value.get("kind") == "finite_state_transducer"
        })
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


__all__ = ["CompositionalCodeLearner", "SemanticGoal", "TransformationGoal", "ProgramPlan"]
