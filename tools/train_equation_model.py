"""Denklem şablonu modeli: problem metninden işlem şablonunu (ör. n1-n2, n1*n2+n3) öğrenir.

Eğitim verisi: ASDiv'in Türkçe çevirisi (asdiv_tr_train_*.json; formül denetimi için kullanılır).
Sayılar metindeki sıraya göre n1, n2, ... yuvalarına eşlenir; model yalnızca şablonu tahmin eder,
aritmetiği kod yapar. Uygulama: aynı sayıların metinde birden çok kez geçtiği örnekler atılır.
"""
from __future__ import annotations

import glob
import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NUM = re.compile(r"\d+(?:[.,]\d+)?")


def numbers(text: str) -> list[float]:
    return [float(x.replace(",", ".")) for x in NUM.findall(text)]


def template_of(text: str, formula: str) -> str | None:
    """Formüldeki sayıları metindeki sıralarına göre yuvalara çevirir; belirsizse None."""
    nums = numbers(text)
    if not nums:
        return None
    expr = formula.split("=")[0].replace(" ", "")
    f_nums = [float(m.group(0)) for m in NUM.finditer(expr)]
    if len(nums) == len(f_nums) and sorted(nums) == sorted(f_nums):
        used = [False] * len(nums)
        out, last = [], 0
        for m in NUM.finditer(expr):
            value = float(m.group(0))
            idx = next((i for i, v in enumerate(nums) if v == value and not used[i]), None)
            if idx is None:
                return None
            used[idx] = True
            out.append(expr[last:m.start()] + f"n{idx + 1}"); last = m.end()
        out.append(expr[last:])
        return "".join(out)
    if len(set(nums)) != len(nums):
        return None
    slots = {n: f"n{i + 1}" for i, n in enumerate(nums)}
    out, last = [], 0
    for m in NUM.finditer(expr):
        value = float(m.group(0))
        if value not in slots:
            return None
        out.append(expr[last:m.start()] + slots[value]); last = m.end()
    out.append(expr[last:])
    return "".join(out)


CANON_2 = {"+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV"}


def canonical(tpl: str) -> str:
    """Değişme/sıra farklarını tek etikete indirger: 'n2+n1' ve 'n1+n2' ikisi de ADD.

    İki sayılı çıkarma/bölmede büyük sayı önce gelir (SUB: büyük−küçük, DIV: büyük/küçük).
    Üç sayılı toplam ADD3 ve eşdeğer çok adımlı şablonlar birleştirilir; diğerleri olduğu gibi kalır.
    """
    m = re.fullmatch(r"n(\d)([+\-*/])n(\d)", tpl)
    if m and m.group(1) in ("1", "2") and m.group(3) in ("1", "2"):
        return CANON_2[m.group(2)]
    if re.fullmatch(r"(?:\(?n[123]\+n[123]\)?\+n[123]|n[123]\+\(n[123]\+n[123]\))", tpl):
        return "ADD3"
    if re.fullmatch(r"(?:\(?n1-n2\)?-n3|n1-\(n2\+n3\)|n1-n3-n2)", tpl):
        return "SUB_2_3"
    if re.fullmatch(r"(?:\(?n1-n2\)?\+n3|n1\+n3-n2|\(?n1\+n3\)?-n2)", tpl):
        return "n1-n2+n3"
    if re.fullmatch(r"(?:\(?n1\+n2\)?-n3|n1\+n2-n3)", tpl):
        return "n1+n2-n3"
    if re.fullmatch(r"\(n\d\+n\d\)\*n\d|n\d\*\(n\d\+n\d\)", tpl):
        return "SUMMUL"
    return tpl


def apply_label(label: str, nums: list[float]) -> float | None:
    """Etiketi sayılara uygular; uymayan durumda None."""
    try:
        if label == "ADD": a, b = nums[:2]; return a + b
        if label == "SUB": a, b = sorted(nums[:2], reverse=True); return a - b
        if label == "MUL": a, b = nums[:2]; return a * b
        if label == "DIV": a, b = sorted(nums[:2], reverse=True); return a / b
        if label == "ADD3": return sum(nums[:3])
        if label == "SUB_2_3": return nums[0] - nums[1] - nums[2]
        if label == "SUMMUL": return (nums[0] + nums[1]) * nums[2]
        env = {f"n{i + 1}": v for i, v in enumerate(nums)}
        if re.fullmatch(r"[n\d+\-*/() .]+", label):
            return float(eval(label, {"__builtins__": {}}, env))
    except Exception:
        return None
    return None


def load_pairs() -> list[tuple[str, str]]:
    pairs = []
    for path in sorted(glob.glob(str(ROOT / "data/benchmarks/asdiv_tr_train_*.json"))):
        for x in json.load(open(path, encoding="utf-8")):
            tpl = template_of(x["text_tr"], x["formula"])
            if tpl:
                pairs.append((x["text_tr"], canonical(tpl)))
    return pairs


def featurize_text(text: str) -> str:
    """Sayıları yer tutucuya çevirir: model sayı değerlerini değil, dili öğrenir."""
    return NUM.sub(" SAYI ", text.lower())


def grams(text: str) -> list[str]:
    w = re.findall(r"[a-zçğıöşüâîû]+", featurize_text(text))
    return w + [f"{a}_{b}" for a, b in zip(w, w[1:])]


def build_vocab(texts: list[str], min_count: int = 2) -> dict[str, int]:
    c = Counter(g for t in texts for g in set(grams(t)))
    return {g: i for i, g in enumerate(sorted(g for g, n in c.items() if n >= min_count))}


def vectorize(texts: list[str], vocab: dict[str, int]):
    import torch
    X = torch.zeros(len(texts), len(vocab))
    for i, t in enumerate(texts):
        for g in grams(t):
            j = vocab.get(g)
            if j is not None:
                X[i, j] = 1.0
    return X


def train(texts: list[str], labels: list[str], epochs: int = 300):
    import torch
    vocab = build_vocab(texts)
    classes = sorted(set(labels)); cid = {c: i for i, c in enumerate(classes)}
    X = vectorize(texts, vocab); y = torch.tensor([cid[l] for l in labels])
    torch.manual_seed(0)
    lin = torch.nn.Linear(len(vocab), len(classes))
    opt = torch.optim.Adam(lin.parameters(), lr=0.05, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad(); loss = torch.nn.functional.cross_entropy(lin(X), y); loss.backward(); opt.step()
    return {"vocab": vocab, "classes": classes, "W": lin.weight.detach(), "b": lin.bias.detach()}


def predict(model: dict, text: str) -> tuple[str, float]:
    import torch
    x = vectorize([text], model["vocab"])
    probs = torch.softmax(x @ model["W"].T + model["b"], dim=-1)[0]
    k = int(probs.argmax())
    return model["classes"][k], float(probs[k])


def answer(model: dict, text: str, min_prob: float = 0.5) -> tuple[float, str] | None:
    """Etiketi tahmin eder, sayılara uygular; güven düşükse None döner."""
    label, prob = predict(model, text)
    nums = numbers(text)
    if prob < min_prob or len(set(nums)) != len(nums):
        return None
    value = apply_label(label, nums)
    return (value, label) if value is not None else None


def main() -> None:
    import random
    pairs = load_pairs()
    counts = Counter(t for _, t in pairs)
    data = [(x, t) for x, t in pairs if counts[t] >= 2]
    print("examples", len(pairs), "templates", len(counts), "usable (template seen >=2)", len(data))
    rng = random.Random(0); rng.shuffle(data)
    cut = int(len(data) * 0.8)
    tr, te = data[:cut], data[cut:]
    m = train([x for x, _ in tr], [t for _, t in tr])
    if te:
        ok = sum(predict(m, x)[0] == t for x, t in te)
        print(f"held-out template accuracy: {ok}/{len(te)}")
    m = train([x for x, _ in data], [t for _, t in data])
    out = ROOT / "data/equation_model.pt"
    import torch
    torch.save(m, out); print("saved", out)


if __name__ == "__main__":
    main()
