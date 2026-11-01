"""Altıncı geliştirme seti: SVAMP tarzı yapılar (SVAMP'tan bağımsız, elle yazılmış şablonlar).

Yapılar: ilgisiz (distractor) cümleler, 'X, Y'den N fazla', iki kişinin farklı nesne sayıları
ve sonuç sorusu öncesinde ek bilgi. Bu set yalnızca geliştirme içindir.
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "math_dev6.json"
NAMES = ["Sibel", "Okan", "Rüya", "Hasan", "Melike", "Tarık", "Aylin", "Serkan"]
OBJECTS = ["bilye", "kart", "kurdele", "mum", "elma"]
DISTRACT = ["Hava çok güzeldi.", "Bahçede kuşlar ötüyordu.", "Okul bugün erken kapandı.",
            "Herkes çok yorgundu."]


def item(kind, i, text, gold):
    body, question = text.rsplit(". ", 1)
    return {"ID": f"dev6-{kind}{i}", "kind": kind, "Body_tr": body + ".", "Question_tr": question, "gold": gold}


def distract(rng, i):
    a = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    x = rng.randint(10, 60)
    m = rng.randint(2, 20)
    d = rng.choice(DISTRACT)
    text = f"{a}'nın {x} {obj}i vardı. {d} {a} {m} {obj} daha aldı. Şimdi kaç {obj}i var?"
    return item("distract", i, text, x + m)


def than(rng, i):
    a, b = rng.sample(NAMES, 2)
    obj = rng.choice(OBJECTS)
    y = rng.randint(5, 50)
    d = rng.randint(2, 30)
    text = f"{a} {y + d} {obj} topladı. {b} ise {y} {obj} topladı. {b}, {a}'dan kaç {obj} eksik?"
    return item("than", i, text, d)


def distract_remove(rng, i):
    a, b = rng.sample(NAMES, 2)
    obj = rng.choice(OBJECTS)
    x = rng.randint(15, 80)
    m = rng.randint(3, x - 1)
    d = rng.choice(DISTRACT)
    text = f"{a}'nın {x} {obj}i vardı. {d} {b}'ye {m} {obj} verdi. {a}'nın kaç {obj}i kaldı?"
    return item("remove", i, text, x - m)


def main() -> None:
    import random as _r
    rng = _r.Random(66778)
    items = []
    for i in range(60):
        items += [distract(rng, i), than(rng, i), distract_remove(rng, i)]
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", len(items), OUT)


if __name__ == "__main__":
    main()
