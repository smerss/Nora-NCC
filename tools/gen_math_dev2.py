"""İkinci geliştirme seti: karşılaştırma, bölme ve çok sayılı cümleler (SVAMP'tan bağımsız).

Bu set şablonlarımızdan üretilir; çözücüyü geliştirmek için kullanılır. SVAMP mühürlüdür.
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "math_dev2.json"
NAMES = ["Ali", "Ayşe", "Mehmet", "Zeynep", "Can", "Elif", "Burak", "Selin", "Emre", "Derya"]
OBJECTS = ["elma", "kalem", "kurabiye", "bilye", "kitap", "balon", "şeker", "kart", "çiçek", "kutu"]


def compare(rng, i):
    a, b = rng.sample(NAMES, 2)
    obj = rng.choice(OBJECTS)
    x = rng.randint(10, 90)
    d = rng.randint(1, x - 1)
    y = x - d
    text = f"{a}'nin {x} {obj}i var. {b}'nin {y} {obj}i var. {a}, {b}'den kaç {obj} fazla?"
    return {"ID": f"dev2-c{i}", "kind": "compare", "Body_tr": text.rsplit(". ", 1)[0] + ".",
            "Question_tr": text.rsplit(". ", 1)[1], "gold": d}


def divide(rng, i):
    name, obj = rng.choice(NAMES), rng.choice(OBJECTS)
    groups = rng.randint(2, 9)
    each = rng.randint(2, 12)
    total = groups * each
    text = f"{name}'nin {total} {obj}i var. Hepsini eşit olarak {groups} kutuya koydu. Her kutuda kaç {obj} var?"
    return {"ID": f"dev2-d{i}", "kind": "divide", "Body_tr": text.rsplit(". ", 1)[0] + ".",
            "Question_tr": text.rsplit(". ", 1)[1], "gold": each}


def multi_first(rng, i):
    a, b = rng.sample(NAMES, 2)
    obj = rng.choice(OBJECTS)
    x, y = rng.randint(5, 60), rng.randint(5, 60)
    m = rng.randint(1, x)
    text = f"{a} {x} {obj} ve {b} {y} {obj} topladı. {a} {m} {obj} verdi. Şimdi toplam kaç {obj} var?"
    return {"ID": f"dev2-m{i}", "kind": "multi", "Body_tr": text.rsplit(". ", 1)[0] + ".",
            "Question_tr": text.rsplit(". ", 1)[1], "gold": x + y - m}


def main() -> None:
    rng = random.Random(777)
    items = []
    for i in range(100):
        items.append(compare(rng, i))
        items.append(divide(rng, i))
        items.append(multi_first(rng, i))
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", len(items), OUT)


if __name__ == "__main__":
    main()
