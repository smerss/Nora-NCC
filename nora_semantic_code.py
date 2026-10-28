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
import json
from pathlib import Path
import re
from typing import Any
import urllib.request

from nora_lang import tr_lower


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

    @property
    def key(self) -> tuple[str, str]:
        return self.unit, self.layout + ("+cumulative" if self.cumulative else "")

    def units(self) -> list[str]:
        if self.cumulative and self.unit == "character":
            return [self.payload[:index] for index in range(1, len(self.payload) + 1)]
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
                "cumulative": self.cumulative}


@dataclass(frozen=True)
class ProgramPlan:
    operators: tuple[str, ...]
    program: str
    expected: str
    trace: tuple[str, ...]


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
        self.documents: list[str] = []
        self.learning_events = 0

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
        if character and (not word or character.start() > word.start()):
            return "character"
        if word:
            return "word"
        return "whole"

    def request(self, raw: str) -> SemanticGoal | None:
        """Ground only constrained display requests; simple requests use the legacy path."""
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
        trace = (f"payload_length={len(payload)}", f"unit={unit}", f"layout={layout}",
                 f"execute={execute}", f"cumulative={cumulative}")
        return SemanticGoal(payload, unit, layout, execute, raw, trace, cumulative)

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
        self.learning_events += 1
        self.documents.append(str(source))
        self.documents = self.documents[-40:]
        new_ops = sorted(self.operators - before_ops)
        new_facts = sorted(set(self.facts) - before_facts)
        return {"ok": bool(parsed or new_facts or invented), "source": source, "parsed": parsed,
                "new_operators": new_ops + [f"invented:{name}" for name in invented],
                "new_facts": new_facts, "operators": sorted(self.operators),
                "invented": invented, "facts": dict(self.facts)}

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
                           alphabet: list[str], limit: int = 512) -> list[dict[str, Any]]:
        table: dict[tuple, tuple] = {}
        solutions: list[dict[str, Any]] = []

        def finish():
            if len(table) != states * len(alphabet) or len(solutions) >= limit:
                return
            transitions = [
                [state, symbol, next_state, output]
                for (state, symbol), (next_state, output) in sorted(table.items())
            ]
            solutions.append({
                "kind": "finite_state_transducer", "states": states,
                "alphabet": list(alphabet), "initial": 0,
                "transitions": transitions,
            })

        def visit_example(index: int):
            if len(solutions) >= limit:
                return
            if index >= len(examples):
                finish()
                return
            source, expected = examples[index]

            def step(position: int, state: int):
                if len(solutions) >= limit:
                    return
                if position >= len(source):
                    visit_example(index + 1)
                    return
                symbol = source[position]
                emitted = expected[position]
                key = (state, symbol)
                known = table.get(key)
                if known is not None:
                    if known[1] == emitted:
                        step(position + 1, known[0])
                    return
                for next_state in range(states):
                    table[key] = (next_state, emitted)
                    step(position + 1, next_state)
                    del table[key]
                    if len(solutions) >= limit:
                        return

            step(0, 0)

        visit_example(0)
        return solutions

    def invent_representation(self, train, validation, *, max_states: int = 3) -> dict:
        """Eski temsil yetmezse minimal gizli-durum transdüseri kur.

        Önce mevcut akış ifadeleri denenir. Yeni tür yalnız başarısızlıkta açılır;
        validation'dan geçen minimal adaylar eşdeğer değilse seçim yapılmaz.
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
            return {"status": "unresolved", "reason": "validation-alphabet"}
        for states in range(1, max(1, int(max_states)) + 1):
            candidates = self._finite_candidates(train_rows, states, alphabet)
            if len(candidates) >= 512:
                return {"status": "budget-exceeded", "states": states}
            validated = [candidate for candidate in candidates
                         if all(self._run_finite(candidate, source) == expected
                                for source, expected in validation_rows)]
            if not validated:
                continue
            canonical = validated[0]
            if any(not self._finite_equivalent(canonical, other, alphabet)
                   for other in validated[1:]):
                return {"status": "ambiguous", "states": states,
                        "candidates": len(validated)}
            serial = 1 + max(
                [int(name[4:]) for name in self.representations if name.startswith("repr")
                 and name[4:].isdigit()] or [0],
            )
            name = f"repr{serial}"
            canonical["evidence"] = {"train": len(train_rows), "validation": len(validation_rows),
                                     "equivalent_candidates": len(validated)}
            self.representations[name] = canonical
            return {"status": "invented", "id": name, "kind": canonical["kind"],
                    "states": states, "candidates": len(validated)}
        return {"status": "unresolved"}

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
                if (goal.unit == "character" and self.facts.get("text_sequence")
                        and "accumulate" in self.operators):
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
            if operator in {"character_view", "accumulate"} and not self.facts.get("text_sequence"):
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
        if goal.cumulative and "accumulate" not in self.operators and not self._matching_streams(goal):
            missing.append("accumulate")
        if goal.unit == "word" and "split_words" not in self.operators:
            missing.append("split_words")
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

    def solve(self, goal: SemanticGoal) -> ProgramPlan | dict:
        key = "|".join(goal.key)
        cached = self.composites.get(key)
        path = cached if cached and self._path_available(cached) else self._search(goal)
        if path is None:
            return {"ok": False, "reason": "semantic_gap", "missing": self._missing(goal),
                    "trace": list(goal.trace)}
        program = self._emit(goal, path)
        trace = list(goal.trace) + ["plan=" + " -> ".join(path)]
        if cached == path:
            trace.append("composite=reused")
        return ProgramPlan(path, program, goal.expected_output(), tuple(trace))

    def adopt(self, goal: SemanticGoal, plan: ProgramPlan) -> None:
        self.composites["|".join(goal.key)] = tuple(plan.operators)

    # -------------------------------------------------------------- persistence
    def state_dict(self) -> dict:
        return {"operators": sorted(self.operators), "facts": dict(self.facts),
                "composites": {key: list(value) for key, value in self.composites.items()},
                "stream_operators": copy.deepcopy(self.stream_operators),
                "representations": copy.deepcopy(self.representations),
                "documents": list(self.documents), "events": self.learning_events}

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
        self.documents = [str(value) for value in data.get("documents") or ()][-40:]
        self.learning_events = int(data.get("events", 0) or 0)


__all__ = ["CompositionalCodeLearner", "SemanticGoal", "ProgramPlan"]
