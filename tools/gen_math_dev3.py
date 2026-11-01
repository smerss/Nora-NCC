"""Üçüncü geliştirme seti: dev1/dev2 şablonlarından FARKLI ifadelerle (genelleme ölçümü).

Fiiller ve cümle sırası bilerek farklı; çözücü bu kalıpları görmeden önce yazıldı.
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "math_dev3.json"
NAMES = ["Mert", "Gül", "Hakan", "Nazlı", "Ece", "Kerem", "Sena", "Tolga"]
OBJECTS = ["meyve", "defter", "boncuk", "madalya", "oyuncak", "mum", "tohum", "rozet"]


def add_sentence(rng, i):
    name, obj = rng.choice(NAMES), rng.choice(OBJECTS)
    n = rng.randint(5, 80)
    m = rng.randint(3, 60)
    text = (f"Sepette {n} {obj} vardı. {name} sepete {m} {obj} daha koydu. "
            f"Son durumda kaç {obj} var?")
    return {"ID": f"dev3-a{i}", "kind": "add", "Body_tr": text.rsplit(". ", 1)[0] + ".",
            "Question_tr": text.rsplit(". ", 1)[1], "gold": n + m}


def remove_sentence(rng, i):
    name, obj = rng.choice(NAMES), rng.choice(OBJECTS)
    n = rng.randint(10, 90)
    m = rng.randint(2, n - 1)
    text = (f"{name}'de {n} {obj} vardı. Bunların {m} tanesi arkadaşına verildi. "
            f"Geriye kaç {obj} kaldı?")
    return {"ID": f"dev3-r{i}", "kind": "remove", "Body_tr": text.rsplit(". ", 1)[0] + ".",
            "Question_tr": text.rsplit(". ", 1)[1], "gold": n - m}


def compare_sentence(rng, i):
    a, b = rng.sample(NAMES, 2)
    obj = rng.choice(OBJECTS)
    y = rng.randint(5, 60)
    d = rng.randint(1, 40)
    x = y + d
    text = (f"{a}'nın {x} {obj}i var, {b}'nin ise {y} {obj}i var. "
            f"{a}, {b}'den kaç {obj} fazla?")
    return {"ID": f"dev3-c{i}", "kind": "compare", "Body_tr": text.rsplit(". ", 1)[0] + ".",
            "Question_tr": text.rsplit(". ", 1)[1], "gold": d}


def main() -> None:
    rng = random.Random(4242)
    items = []
    for i in range(60):
        items.append(add_sentence(rng, i))
        items.append(remove_sentence(rng, i))
        items.append(compare_sentence(rng, i))
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", len(items), OUT)


if __name__ == "__main__":
    main()
