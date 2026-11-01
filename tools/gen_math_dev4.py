"""Dördüncü geliştirme seti: önceki setlerde görülmemiş yapılar (SVAMP'tan bağımsız).

Yapılar: iki öznenin ikinci cümlede olayı, eşit paylaştırma (kişiye), 'daha çok' karşılaştırması,
bir olayın iki kez tekrarlanması ve kayıp (kaybetti/yedi) ile eklemenin birlikte olduğu problemler.
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "math_dev4.json"
NAMES = ["Deniz", "Onur", "Pelin", "Yusuf", "Işık", "Berk", "Tuğba", "Volkan"]
OBJECTS = ["fındık", "lale", "bilet", "fincan", "anahtar", "jeton"]


def item(kind, i, text, gold):
    body, question = text.rsplit(". ", 1)
    return {"ID": f"dev4-{kind}{i}", "kind": kind, "Body_tr": body + ".", "Question_tr": question, "gold": gold}


def share(rng, i):
    n, obj = rng.randint(12, 80), rng.choice(OBJECTS)
    k = rng.choice([2, 3, 4, 5, 6])
    if n % k:
        n = n - n % k
    name = rng.choice(NAMES)
    text = f"{name} {n} {obj}i {k} kişiye eşit olarak paylaştırdı. Her kişi kaç {obj} aldı?"
    return item("share", i, text, n // k)


def more_than(rng, i):
    a, b = rng.sample(NAMES, 2)
    obj = rng.choice(OBJECTS)
    y = rng.randint(4, 50)
    d = rng.randint(2, 30)
    text = (f"{a} {y + d} {obj} topladı. {b} {y} {obj} topladı. "
            f"{a}, {b}'den ne kadar daha çok {obj} topladı?")
    return item("more", i, text, d)


def two_events(rng, i):
    name, obj = rng.choice(NAMES), rng.choice(OBJECTS)
    start = rng.randint(20, 90)
    gain = rng.randint(3, 25)
    loss = rng.randint(1, min(19, start))
    text = (f"{name}'in {start} {obj}i vardı. Birinden {gain} {obj} aldı. "
            f"Sonra {loss} {obj} kaybetti. Şu an kaç {obj} var?")
    return item("two", i, text, start + gain - loss)


def main() -> None:
    import random as _r
    rng = _r.Random(91011)
    items = []
    for i in range(60):
        items.append(share(rng, i))
        items.append(more_than(rng, i))
        items.append(two_events(rng, i))
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", len(items), OUT)


if __name__ == "__main__":
    main()
