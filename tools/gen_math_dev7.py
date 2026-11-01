"""Yedinci geliştirme seti: ters problemler (bilinmeyen başlangıç ya da olay). SVAMP'tan bağımsız."""
import json, random
from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "math_dev7.json"
NAMES = ["Naz", "Orhan", "Pınar", "Rıza", "Seda", "Umut"]
OBJ = ["kalem", "bilye", "kitap", "kart", "balon"]

def it(kind, i, body, q, gold):
    return {"ID": f"dev7-{kind}{i}", "kind": kind, "Body_tr": body, "Question_tr": q, "gold": gold}

def main():
    rng = random.Random(31337); items = []
    for i in range(60):
        n, o = rng.choice(NAMES), rng.choice(OBJ)
        a, b = rng.randint(10, 60), rng.randint(2, 9)
        items.append(it("start_gain", i, f"{n}'nin bazı {o}i vardı. {n} {b} {o} daha aldı. Şimdi {a + b} {o}i var.", f"Başta kaç {o}i vardı?", a))
        a, b = rng.randint(20, 80), rng.randint(2, 15)
        items.append(it("start_loss", i, f"{n}'nin bazı {o}i vardı. {n} {b} {o} verdi. Geriye {a} {o} kaldı.", f"Başta kaç {o}i vardı?", a + b))
        a, b = rng.randint(30, 90), rng.randint(5, 25)
        items.append(it("event_loss", i, f"{n}'nin {a + b} {o}i vardı. {n} bazı {o} verdi. Geriye {a} {o} kaldı.", f"Kaç {o} verdi?", b))
        a, b = rng.randint(5, 40), rng.randint(5, 30)
        items.append(it("event_gain", i, f"{n}'nin {a} {o}i vardı. {n} bazı {o} daha aldı. Şimdi {a + b} {o}i var.", f"Kaç {o} aldı?", b))
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8"); print(len(items))
main()
