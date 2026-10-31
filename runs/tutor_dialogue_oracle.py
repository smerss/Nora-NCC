import hashlib
import json
import re
from collections import deque
from pathlib import Path

from nora_agi import NoraAGI
from nora_besle import besle
from nora_lang import tr_lower

DATA = Path("runs/tutor_dialogue_final.json")
OUT = Path("runs/tutor_dialogue_oracle.json")
PRODUCT_FILES = ["nora_deliberation.py", "nora_agi.py", "agi_battery.py", "nora_priors.json"]
data = json.loads(DATA.read_text(encoding="utf-8"))
turns = data["turns"]

# Ürünün öğretici sorgusunu çağırmadan bağımsız doğrudan-kenar / ISA-BFS defteri.
agi = NoraAGI(seed=0, boot_thinker=False, code_pretrain=False)
besle(agi, verbose=False)
agi.attach_broca("data/checkpoints/turkish_moe.json")
agi.step_life(1500)
edges = [edge for edge in agi.lang.graph.edges if float(edge.weight) > 0.25]
direct = {(edge.subj, edge.rel, edge.obj) for edge in edges}
isa = {}
for edge in edges:
    if edge.rel == "ISA":
        isa.setdefault(edge.subj, set()).add(edge.obj)

def grounded(fact):
    subject, relation, obj = fact
    if fact in direct:
        return True
    if relation != "ISA":
        return False
    queue = deque([(subject, 0)])
    seen = {subject}
    while queue:
        node, depth = queue.popleft()
        if depth >= 16:
            continue
        for nxt in isa.get(node, ()):
            if nxt == obj:
                return True
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, depth + 1))
    return False

fact_results = []
for turn in turns:
    for fact in turn["trace"].get("facts", []):
        key = tuple(fact)
        fact_results.append({"turn": turn["turn"], "fact": fact, "grounded": grounded(key)})

arithmetic = []
for turn in turns:
    if turn["trace"].get("intent") != "example":
        continue
    numbers = [int(value) for value in re.findall(r"\d+", turn["student"])]
    left, right = numbers[:2] if len(numbers) >= 2 else (2, 3)
    expected = left + right
    passed = re.search(rf"(?<!\d){expected}(?!\d)", turn["nora"]) is not None
    arithmetic.append({"turn": turn["turn"], "expression": f"{left}+{right}",
                       "expected": expected, "passed": passed})

r = [tr_lower(turn["nora"]) for turn in turns]
stats = data.get("neural_stats", {})
digest = hashlib.sha256()
for filename in PRODUCT_FILES:
    digest.update(filename.encode("utf-8") + b"\0" + Path(filename).read_bytes())
checks = {
    "thirty_turns": len(turns) == 30,
    "sequential_traces": all(turn["trace"].get("session_turn") == turn["turn"] for turn in turns),
    "graph_grounding": len(fact_results) == 58 and all(row["grounded"] for row in fact_results),
    "aritmetik": len(arithmetic) == 3 and all(row["passed"] for row in arithmetic),
    "quiz_and_evaluation": (
        turns[11]["trace"].get("quiz_pending") is True
        and turns[25]["trace"].get("quiz_pending") is True
        and r[12].startswith("evet") and r[26].startswith("evet")
    ),
    "student_summaries": "çelişmiyor" in r[13] and "çelişmiyor" in r[27],
    "nora_summaries": len(turns[14]["nora"].splitlines()) == 3 and len(turns[28]["nora"].splitlines()) == 4,
    "misconception_correction": "aynı kavram değil" in r[22],
    "knowledge_limits": (
        all("kanıt" in r[index] and "yok" in r[index] for index in (15, 16, 17, 19))
        and "kayd" in r[23] and "yok" in r[23]
    ),
    "surface_quality": not any(any(bad in reply for bad in (
        "madde kütledir", "operatör raporu", "bilgisayar bir gün", "olduk anlatabilirim",
        ". ve ", "b'in", "birdır", "yazılımdur", "ay bir uydu", "beyin öğrenebilir")) for reply in r),
    "real_moe_candidates": (
        data.get("moe_seen_tokens", 0) > 0
        and stats.get("accepted", 0) > 0
        and stats.get("generated", -1) == stats.get("accepted", 0) + stats.get("rejected", 0)
        and data.get("neural_turns", 0) > 0
    ),
    "closed": data.get("active_after_close") is False,
    "code_hash": data.get("code_sha256") == digest.hexdigest(),
}
result = {
    "checks": checks,
    "passed": all(checks.values()),
    "fact_count": len(fact_results),
    "fact_results": fact_results,
    "arithmetic": arithmetic,
    "neural_stats": stats,
    "neural_turns": data.get("neural_turns"),
    "code_sha256": digest.hexdigest(),
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in result.items() if key != "fact_results"},
                 ensure_ascii=False, indent=2))
raise SystemExit(0 if result["passed"] else 1)
