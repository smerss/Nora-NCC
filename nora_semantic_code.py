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
                elif isinstance(node, ast.Call):
                    name = node.func
                    if isinstance(name, ast.Name) and name.id == "print":
                        self.operators.add("display")
                    elif isinstance(name, ast.Name) and name.id == "accumulate":
                        self.operators.add("accumulate")
                    elif isinstance(name, ast.Attribute) and name.attr == "split" and not node.args:
                        self.operators.add("split_words")
        self.learning_events += 1
        self.documents.append(str(source))
        self.documents = self.documents[-40:]
        new_ops = sorted(self.operators - before_ops)
        new_facts = sorted(set(self.facts) - before_facts)
        return {"ok": bool(parsed or new_facts), "source": source, "parsed": parsed,
                "new_operators": new_ops, "new_facts": new_facts,
                "operators": sorted(self.operators), "facts": dict(self.facts)}

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

    # -------------------------------------------------------------- planning
    def _transitions(self, goal: SemanticGoal, stage: int) -> list[tuple[str, int]]:
        if stage == 0:
            if goal.cumulative:
                if (goal.unit == "character" and self.facts.get("text_sequence")
                        and "accumulate" in self.operators):
                    return [("accumulate", 1)]
                return []
            if goal.unit == "character" and self.facts.get("text_sequence"):
                return [("character_view", 1)]
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
            if operator == "display" and not self.facts.get("print_newline"):
                return False
            if operator not in {"character_view"} and operator not in self.operators:
                return False
        return True

    def _missing(self, goal: SemanticGoal) -> list[str]:
        missing = []
        if goal.unit == "character" and not self.facts.get("text_sequence"):
            missing.append("text_sequence")
        if goal.cumulative and "accumulate" not in self.operators:
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
    def _emit(goal: SemanticGoal, path: tuple[str, ...]) -> str:
        source: ast.expr = ast.Constant(value=goal.payload)
        statements: list[ast.stmt] = []
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
                "documents": list(self.documents), "events": self.learning_events}

    def load_state_dict(self, data: dict | None) -> None:
        data = data or {}
        self.operators = {str(value) for value in data.get("operators") or ()}
        self.facts = dict(data.get("facts") or {})
        self.composites = {str(key): tuple(value) for key, value in (data.get("composites") or {}).items()}
        self.documents = [str(value) for value in data.get("documents") or ()][-40:]
        self.learning_events = int(data.get("events", 0) or 0)


__all__ = ["CompositionalCodeLearner", "SemanticGoal", "ProgramPlan"]
