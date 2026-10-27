#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_translation.py — küçük derslerden bileşimsel, örnek-tabanlı çeviri.

Sözcük/fiil eşlemleri ve kısa paralel cümleler çalışma zamanında öğrenilir. Paralel
cümleler birer bütün-cevap belleği olarak kullanılmaz: özne, fiil ve yapı parçaları
ayrılır; öğrenilen yapı yeni özne ve yan-cümle zincirlerine uygulanır. Bilinmeyen
yapıda kelime kelime uydurma yapılmaz, açık ``unresolved`` döner.

İnsan-yazımı prior yalnız ders/istek sözdizimi, Türkçe yan-cümle rolleri ve İngilizce
birleştirme düzenidir. Dil eşlemleri, fiil biçimleri ve çeviri yapıları kullanıcı
derslerinden gelir. Bütün hedef cümleler veya hedefe özel sözlük burada tutulmaz.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


_WORD = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşüâîû0-9]+", re.UNICODE)
_PAST_SUFFIXES = ("ti", "tı", "tu", "tü", "di", "dı", "du", "dü")
_CHAIN_SUFFIXES = ("ip", "ıp", "up", "üp")


def _plain(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def _unquote(text: Any) -> str:
    return _plain(text).strip("‘’“”\"'")


def _words(text: Any) -> list[str]:
    return _WORD.findall(str(text or ""))


def _infinitive_root(text: Any) -> str:
    value = _unquote(text).lower().replace(" ", "")
    if value.endswith(("mek", "mak")) and len(value) > 4:
        return value[:-3]
    return value


@dataclass(frozen=True)
class VerbForms:
    base: str
    past: str
    gerund: str


@dataclass(frozen=True)
class Clause:
    terms: tuple[str, ...]
    verb: str
    role: str


@dataclass(frozen=True)
class Construction:
    verb: str
    terms: tuple[str, ...]
    before: tuple[str, ...]
    after: tuple[str, ...]
    past: str

    @property
    def key(self) -> str:
        return self.verb + "|" + " ".join(self.terms)


class LearnedTranslator:
    """Örneklerden yapı çıkaran sınırlı TR→EN çeviri organı."""

    def __init__(self):
        self.verbs: dict[str, VerbForms] = {}
        self.lexicon: dict[str, str] = {}
        self.constructions: dict[str, Construction] = {}
        self.lessons = 0

    # ------------------------------------------------------------------ dersler
    def teach_verb(self, source: str, base: str, past: str, gerund: str) -> dict:
        root = _infinitive_root(source)
        forms = VerbForms(_plain(base).lower(), _plain(past).lower(), _plain(gerund).lower())
        if not root or not all((forms.base, forms.past, forms.gerund)):
            return {"status": "invalid", "reason": "empty verb form"}
        self.verbs[root] = forms
        self.lessons += 1
        return {"status": "learned", "kind": "verb", "source": root,
                "summary": f"{root} = {forms.base}/{forms.past}/{forms.gerund}"}

    def teach_lexicon(self, source: str, target: str) -> dict:
        src, dst = _unquote(source).lower(), _plain(target).lower()
        if not src or not dst:
            return {"status": "invalid", "reason": "empty lexical pair"}
        self.lexicon[src] = dst
        self.lessons += 1
        return {"status": "learned", "kind": "lexicon", "source": src,
                "summary": f"{src} = {dst}"}

    def teach(self, statement: str) -> dict | None:
        text = _plain(statement)
        verb = re.match(
            r"^(?P<src>[^\s]+)\s+(?:İngilizcede|ingilizcede)\s+"
            r"['\"](?P<base>[^'\"]+)['\"]\s*,\s*geçmişi\s+"
            r"['\"](?P<past>[^'\"]+)['\"]\s*,\s*eylemlisi\s+"
            r"['\"](?P<gerund>[^'\"]+)['\"]\s+demektir\.?$",
            text,
        )
        if verb:
            return self.teach_verb(verb.group("src"), verb.group("base"),
                                   verb.group("past"), verb.group("gerund"))
        pair = re.match(
            r"^['\"](?P<src>.+?)['\"]\s+cümlesinin\s+(?:İngilizcesi|ingilizcesi)\s+"
            r"['\"](?P<tgt>.+?)['\"]\.?$",
            text,
        )
        if pair:
            return self.teach_pair(pair.group("src"), pair.group("tgt"))
        lexical = re.match(
            r"^(?P<src>.+?)\s+(?:İngilizcede|ingilizcede)\s+"
            r"['\"](?P<tgt>[^'\"]+)['\"]\s+demektir\.?$",
            text,
        )
        if lexical:
            return self.teach_lexicon(lexical.group("src"), lexical.group("tgt"))
        return None

    def teach_pair(self, source: str, target: str) -> dict:
        subject, clauses = self._parse(source)
        if not subject or len(clauses) != 1 or clauses[0].role != "main":
            return {"status": "invalid", "reason": "one explicit-subject past clause required"}
        clause = clauses[0]
        forms = self.verbs.get(clause.verb)
        if forms is None:
            return {"status": "unresolved", "reason": f"verb forms not learned: {clause.verb}"}
        target_words = _words(target)
        if len(target_words) < 2:
            return {"status": "invalid", "reason": "target subject and verb required"}
        rest = target_words[1:]
        past_indices = [i for i, word in enumerate(rest) if word.lower() == forms.past]
        if len(past_indices) != 1:
            return {"status": "unresolved", "reason": f"target past form not aligned: {forms.past}"}
        index = past_indices[0]
        construction = Construction(
            verb=clause.verb,
            terms=clause.terms,
            before=tuple(word.lower() for word in rest[:index]),
            after=tuple(word.lower() for word in rest[index + 1:]),
            past=forms.past,
        )
        self.constructions[construction.key] = construction
        self.lessons += 1
        return {"status": "learned", "kind": "construction", "source": construction.key,
                "summary": f"{subject} + {construction.key}"}

    # ------------------------------------------------------------------ çözümleme
    def _surface_verb(self, token: str, next_token: str = "") -> tuple[str, str] | None:
        low = token.lower()
        for root in sorted(self.verbs, key=len, reverse=True):
            if not low.startswith(root) or len(low) <= len(root):
                continue
            if low.endswith("ken"):
                return root, "while"
            if low.endswith(_CHAIN_SUFFIXES):
                return root, "chain"
            if next_token.lower() == "sonra" and low.endswith(("ten", "tan")):
                return root, "after"
            if low.endswith(_PAST_SUFFIXES):
                return root, "main"
        return None

    def _parse(self, sentence: str) -> tuple[str, list[Clause]]:
        words = _words(sentence)
        if not words:
            return "", []
        subject = ""
        if words[0][:1].isupper():
            subject = words.pop(0)
        clauses: list[Clause] = []
        terms: list[str] = []
        index = 0
        while index < len(words):
            token = words[index]
            following = words[index + 1] if index + 1 < len(words) else ""
            found = self._surface_verb(token, following)
            if found is None:
                terms.append(token.lower())
                index += 1
                continue
            root, role = found
            clauses.append(Clause(tuple(terms), root, role))
            terms = []
            index += 2 if role == "after" else 1
        if terms:
            return subject, []
        return subject, clauses

    # ------------------------------------------------------------------ uygulama
    @staticmethod
    def _required_positions(actual: tuple[str, ...], required: tuple[str, ...]) -> list[int] | None:
        if not required:
            return []
        positions: list[int] = []
        start = 0
        for term in required:
            try:
                found = actual.index(term, start)
            except ValueError:
                return None
            positions.append(found)
            start = found + 1
        return positions

    def _apply(self, clause: Clause, subject: str, include_subject: bool) -> dict:
        candidates = []
        for construction in self.constructions.values():
            if construction.verb != clause.verb:
                continue
            positions = self._required_positions(clause.terms, construction.terms)
            if positions is None:
                continue
            extras = [term for i, term in enumerate(clause.terms) if i not in positions]
            if all(term in self.lexicon for term in extras):
                candidates.append((len(construction.terms), construction, extras))
        if not candidates:
            return {"status": "unresolved", "reason": f"construction not learned: {clause.verb}",
                    "terms": list(clause.terms), "role": clause.role}
        _score, chosen, extras = max(candidates, key=lambda row: row[0])
        forms = self.verbs[chosen.verb]
        verb = forms.gerund if clause.role in {"while", "after"} else forms.past
        words = ([subject] if include_subject and subject else [])
        words += list(chosen.before) + [verb] + list(chosen.after)
        words += [self.lexicon[term] for term in extras]
        return {"status": "ok", "text": " ".join(words), "role": clause.role,
                "construction": chosen.key}

    def translate(self, sentence: str) -> dict:
        subject, clauses = self._parse(sentence)
        if not clauses:
            return {"status": "unresolved", "reason": "no learned verb structure"}
        finite = [i for i, clause in enumerate(clauses) if clause.role in {"chain", "main"}]
        if not finite or not any(clause.role == "main" for clause in clauses):
            return {"status": "unresolved", "reason": "no finite main clause"}
        first_finite = finite[0]
        translated = []
        for index, clause in enumerate(clauses):
            # Bir -ip zinciri "after" ile yeni bir eşgüdüme açıldıysa ana cümle
            # özneyi yeniden söyler; böylece virgüller iki bağımsız cümleyi ayırır.
            repeat_after = (clause.role == "main" and index > 0
                            and clauses[index - 1].role == "after"
                            and any(item.role == "chain" for item in clauses[:index]))
            result = self._apply(
                clause, subject,
                include_subject=(index == first_finite or repeat_after),
            )
            if result["status"] != "ok":
                return result
            translated.append(result)

        output = ""
        chain_open = False
        coordinated = False
        used = []
        for index, result in enumerate(translated):
            role = result["role"]
            prefix = "while " if role == "while" else "after " if role == "after" else ""
            if role == "after" and chain_open and not coordinated:
                prefix = "and " + prefix
                coordinated = True
            elif role == "main" and chain_open and not coordinated:
                prefix = "and "
                coordinated = True
            piece = prefix + result["text"]
            output += (" " if output else "") + piece
            if index + 1 < len(translated):
                output += ","
            if role == "chain":
                chain_open = True
            used.append(result["construction"])
        output = output[:1].upper() + output[1:] + "."
        return {"status": "ok", "text": output, "constructions": used,
                "lessons": self.lessons}

    def translate_lexicon(self, source: str) -> dict:
        key = _unquote(source).lower()
        value = self.lexicon.get(key)
        if value is None:
            return {"status": "unresolved", "reason": f"lexical pair not learned: {key}"}
        return {"status": "ok", "text": value}

    # ------------------------------------------------------------------ kalıcılık
    def state_dict(self) -> dict:
        return {
            "version": 1,
            "lessons": self.lessons,
            "verbs": {key: [value.base, value.past, value.gerund]
                      for key, value in self.verbs.items()},
            "lexicon": dict(self.lexicon),
            "constructions": [
                {"verb": item.verb, "terms": list(item.terms),
                 "before": list(item.before), "after": list(item.after), "past": item.past}
                for item in self.constructions.values()
            ],
        }

    def load_state_dict(self, data: dict | None) -> None:
        data = data or {}
        self.verbs = {
            str(key): VerbForms(str(value[0]), str(value[1]), str(value[2]))
            for key, value in (data.get("verbs") or {}).items() if len(value) >= 3
        }
        self.lexicon = {str(key): str(value) for key, value in (data.get("lexicon") or {}).items()}
        self.constructions = {}
        for raw in data.get("constructions") or []:
            item = Construction(str(raw["verb"]), tuple(raw.get("terms") or ()),
                                tuple(raw.get("before") or ()), tuple(raw.get("after") or ()),
                                str(raw["past"]))
            self.constructions[item.key] = item
        self.lessons = int(data.get("lessons", 0) or 0)


__all__ = ["LearnedTranslator", "VerbForms", "Construction", "Clause"]
