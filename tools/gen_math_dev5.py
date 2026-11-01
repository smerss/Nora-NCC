"""Beşinci geliştirme seti: önceki setlerde yer almayan yapılar (SVAMP'a hiç bakılmadı).

Yapılar: iki katı / yarısı, 'her birine N' çarpması, iki kişinin toplamı (ikinci cümlede),
ve 'geriye ne kadar kaldı' biçiminde kayıp sorusu.
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "math_dev5.json"
NAMES = ["Kaan", "Lale", "Furkan", "Büşra", "Oğuz", "Nisa", "Taner", "İpek"]
OBJECTS = ["kitap", "rozet", "kart", "çanak", "tencere", "silgi"]


def item(kind, i, text, gold):
    body, question = text.rsplit(". ", 1)
    return {"ID": f"dev5-{kind}{i}", "kind": kind, "Body_tr": body + ".", "Question_tr": question, "gold": gold}


def double(rng, i):
    name, obj = rng.choice(NAMES), rng.choice(OBJECTS)
    n = rng.randint(3, 40)
    text = f"{name}'nin {n} {obj}i var. Arkadaşının {obj}i bundan iki kat fazla. Arkadaşının kaç {obj}i var?"
    return item("double", i, text, 2 * n)


def per_unit(rng, i):
    name, obj = rng.choice(NAMES), rng.choice(OBJECTS)
    boxes, each = rng.randint(3, 12), rng.randint(2, 15)
    text = f"Her kutuya {each} {obj} konuyor. {name} {boxes} kutu doldurdu. Toplam kaç {obj} konmuş oldu?"
    return item("perunit", i, text, boxes * each)


def pair_sum(rng, i):
    a, b = rng.sample(NAMES, 2)
    obj = rng.choice(OBJECTS)
    x, y = rng.randint(5, 70), rng.randint(5, 70)
    text = f"{a}'nin {x} {obj}i var. {b}'nin de {y} {obj}i var. İkisinin toplam kaç {obj}i var?"
    return item("pair", i, text, x + y)


def left_after(rng, i):
    name, obj = rng.choice(NAMES), rng.choice(OBJECTS)
    n = rng.randint(20, 90)
    used = rng.randint(3, 18)
    text = f"{name}'nin {n} {obj}i vardı. Bunların {used} tanesini kullandı. Geriye ne kadar {obj} kaldı?"
    return item("left", i, text, n - used)


def main() -> None:
    import random as _r
    rng = _r.Random(55443)
    items = []
    for i in range(60):
        items += [double(rng, i), per_unit(rng, i), pair_sum(rng, i), left_after(rng, i)]
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", len(items), OUT)


if __name__ == "__main__":
    main()
