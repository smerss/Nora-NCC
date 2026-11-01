#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SVAMP ölçümü: Nora'nın kelime problemi cevaplarını altın yanıtla karşılaştırır.

Kullanım:
    ./.venv/bin/python -m batteries.svamp_battery --limit 50
    ./.venv/bin/python -m batteries.svamp_battery --data data/benchmarks/svamp.json

Kurallar:
- Bu test MÜHÜRLÜ bir ölçüm: sonuçlara bakarak Nora'nın matematik kodu ayarlanmaz.
- Veri yalnızca okunur; checkpoint'e hiçbir şey yazılmaz (kaydetme çağrısı yok).
- "Bilmiyorum" cevabı yanlış sayılmaz; ayrı bir kategoridir (cevapsız).
- Sayı karşılaştırması: cevaptaki herhangi bir sayı altın yanıtla eşleşirse doğru sayılır.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "data" / "benchmarks" / "svamp.json"
DEFAULT_OUT = REPO_ROOT / "data" / "benchmarks"

ABSTAIN_MARKERS = (
    "bilmiyorum", "bilgim yok", "çıkaramıyorum", "doğrulanmış", "tanımım yok",
    "öğretirsen", "öğretebilirsin", "bilgisini",
)
NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def numbers_in(text: str) -> list[float]:
    out: list[float] = []
    for token in NUMBER_RE.findall(text or ""):
        try:
            out.append(float(token.replace(",", ".")))
        except ValueError:
            continue
    return out


def grade(answer: str, gold: float) -> str:
    """'correct', 'wrong' ya da 'abstain' döndürür."""
    if not answer or any(m in answer.lower() for m in ABSTAIN_MARKERS):
        return "abstain"
    return "correct" if any(abs(n - gold) < 1e-6 for n in numbers_in(answer)) else "wrong"


def load_problems(path: Path, limit: int | None) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data[:limit] if limit else data


def build_prompt(item: dict) -> str:
    """Türkçe çeviri (Body_tr/Question_tr) varsa onu, yoksa İngilizce orijinali kullanır."""
    body = item.get("Body_tr", item.get("Body", ""))
    question = item.get("Question_tr", item.get("Question", ""))
    return f"{str(body).strip()} {str(question).strip()}".strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SVAMP ölçümü (mühürlü, salt okunur)")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--limit", type=int, default=None, help="ilk N problem (hızlı deneme için)")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "state" / "nora_brain.nora",
                        help="Yüklenecek beyin dosyası; ana dosya değişmez, kopya kullanılması önerilir")
    parser.add_argument("--out", type=Path, default=None, help="sonuç JSONL yolu")
    args = parser.parse_args(argv)

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from nora.cognition.agi import NoraAGI

    problems = load_problems(args.data, args.limit)
    agi = NoraAGI.load(str(args.checkpoint), attach_broca_async=False)
    agi.refresh_chance_streams()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = args.out or (DEFAULT_OUT / f"svamp_results_{stamp}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    verdicts: Counter[str] = Counter()
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    with out_path.open("w", encoding="utf-8") as fh:
        for item in problems:
            prompt = build_prompt(item)
            gold = float(item["Answer"])
            agi.last_real_time = time.time()
            answer = str(agi.converse(prompt, speaker="insan") or "")
            verdict = grade(answer, gold)
            verdicts[verdict] += 1
            by_type[str(item.get("Type", ""))][verdict] += 1
            fh.write(json.dumps({
                "id": item.get("ID"), "type": item.get("Type"), "gold": gold,
                "verdict": verdict, "answer": answer, "path": getattr(agi, "_turn_path", None),
            }, ensure_ascii=False) + "\n")

    total = sum(verdicts.values()) or 1
    print(f"SVAMP: {sum(verdicts.values())} problem")
    for key in ("correct", "wrong", "abstain"):
        print(f"  {key:8s} {verdicts[key]:5d}  ({100.0 * verdicts[key] / total:5.1f}%)")
    for typ, counts in sorted(by_type.items()):
        n = sum(counts.values())
        print(f"  {typ:18s} doğru {counts['correct']:4d}/{n:<4d} yanlış {counts['wrong']:4d} cevapsız {counts['abstain']:4d}")
    print(f"sonuçlar: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
