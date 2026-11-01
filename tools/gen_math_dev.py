"""Geliştirme seti üretici: SVAMP'tan BAĞIMSIZ, kendi şablonlarımızla kelime problemleri.

Amaç: çözücünün iç tutarlılığını ölçmek. Bu set SVAMP değildir; sonuçları yalnızca
geliştirme için kullan, SVAMP'ı mühürlü son ölçüm olarak sakla.
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "math_dev.json"
NAMES = ["Ali", "Ayşe", "Mehmet", "Zeynep", "Can", "Elif", "Burak", "Selin", "Emre", "Derya"]
OBJECTS = ["elma", "kalem", "kurabiye", "bilye", "kitap", "balon", "şeker", "kart", "çiçek", "kutu"]
REMOVE = ["yedi", "verdi", "kaybetti", "attı", "sattı", "harcadı"]
ADD = ["aldı", "buldu", "topladı", "kazandı", "getirdi"]


def one(rng: random.Random, i: int) -> dict:
    name, obj = rng.choice(NAMES), rng.choice(OBJECTS)
    start = rng.randint(10, 99)
    if rng.random() < 0.5:
        m = rng.randint(1, start - 1)
        verb = rng.choice(REMOVE)
        text = f"{name}'nin {start} {obj}i vardı. {m} tanesini {verb}. Kaç {obj} kaldı?"
        gold = start - m
        kind = "sub"
    else:
        m = rng.randint(1, 60)
        verb = rng.choice(ADD)
        text = f"{name}'nin {start} {obj}i vardı. {m} {obj} daha {verb}. Kaç {obj} oldu toplam?"
        gold = start + m
        kind = "add"
    return {"ID": f"dev-{i}", "kind": kind, "Body_tr": text.rsplit(". ", 1)[0] + ".",
            "Question_tr": text.rsplit(". ", 1)[1], "gold": gold}


def main() -> None:
    rng = random.Random(20260)
    items = [one(rng, i) for i in range(300)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", len(items), OUT)


if __name__ == "__main__":
    main()
