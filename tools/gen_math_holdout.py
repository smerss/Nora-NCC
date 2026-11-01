"""Tutma seti (hold-out): geliştirme sırasında HİÇ açılmadı, yalnız tek ölçüm için.

Yeni isimler, nesneler ve karışık yapılar: devir, çoklu kişi, ilgisiz cümleler, 'tane' dolguları.
Bu dosyanın sonucuna bakarak çözücü ayarlanmaz.
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "math_holdout.json"
NAMES = ["Zehra", "Kemal", "Yasemin", "Önder", "Gamze", "Cem"]
OBJECTS = ["mısır", "pul", "tabak", "bant", "fındık", "çorap"]
FILLER = ["Güneş doğmuştu.", "Sokakta sessizlik vardı.", "Kimse konuşmuyordu."]


def item(kind, i, text, gold):
    body, question = text.rsplit(". ", 1)
    return {"ID": f"ho-{kind}{i}", "kind": kind, "Body_tr": body + ".", "Question_tr": question, "gold": gold}


def transfer(rng, i):
    a, b = rng.sample(NAMES, 2)
    obj = rng.choice(OBJECTS)
    x = rng.randint(20, 90)
    m = rng.randint(3, x - 2)
    text = f"{a}'nın {x} {obj}i vardı. {rng.choice(FILLER)} {a} {b}'ye {m} {obj} verdi. {a}'nın kaç {obj}i kaldı?"
    return item("transfer", i, text, x - m)


def gains_two(rng, i):
    a, b = rng.sample(NAMES, 2)
    obj = rng.choice(OBJECTS)
    x, y = rng.randint(5, 40), rng.randint(5, 40)
    g = rng.randint(2, 15)
    text = f"{a} {x} {obj} buldu, {b} {y} {obj} buldu. {a} {g} {obj} daha aldı. İkisinin toplam kaç {obj}i var?"
    return item("gains", i, text, x + y + g)


def per_person_after(rng, i):
    a = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    x = rng.randint(30, 90)
    y = rng.randint(1, 25)
    text = (f"{a}'nın {x} {obj}i vardı. {rng.choice(FILLER)} {a} {y} tane {obj} yedi. "
            f"Şimdi {a}'nın kaç {obj}i var?")
    return item("after", i, text, x - y)


def main() -> None:
    rng = random.Random(202605)
    items = []
    for i in range(60):
        items += [transfer(rng, i), gains_two(rng, i), per_person_after(rng, i)]
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", len(items), OUT)


if __name__ == "__main__":
    main()
