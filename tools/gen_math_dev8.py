"""Sekizinci geliştirme seti: türetilmiş karşılaştırma, grup sayısı, grup çarpımı. SVAMP'tan bağımsız."""
import json, random
from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "math_dev8.json"
NAMES = ["Ceren", "Doruk", "Esra", "Fikret", "Gülay", "Hüseyin"]
OBJ = ["fincan", "pul", "rozet", "mum", "top"]

def it(kind, i, body, q, gold):
    return {"ID": f"dev8-{kind}{i}", "kind": kind, "Body_tr": body, "Question_tr": q, "gold": gold}

def main():
    rng = random.Random(8080); items = []
    for i in range(60):
        a, b = rng.sample(NAMES, 2); o = rng.choice(OBJ)
        x, d = rng.randint(10, 70), rng.randint(2, 9)
        items.append(it("more", i, f"{a}'nın {x} {o}i var. {b}'nin {a}'dan {d} {o} fazlası var.", f"{b}'nin kaç {o}i var?", x + d))
        items.append(it("less", i, f"{a}'nın {x} {o}i var. {b}'nin {a}'dan {d} {o} eksiği var.", f"{b}'nin kaç {o}i var?", x - d))
        g, e = rng.randint(3, 9), rng.randint(2, 8)
        items.append(it("groups", i, f"{a}'nın {g * e} {o}i var. Her kutuya {e} {o} koyuyor.", f"Kaç kutu gerekir?", g))
        items.append(it("product", i, f"{a}'nın {g} kutusu var. Her kutuda {e} {o} var.", f"Toplam kaç {o} var?", g * e))
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8"); print(len(items))
main()
