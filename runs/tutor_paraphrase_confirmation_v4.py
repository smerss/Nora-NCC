import hashlib
import json
import time
from pathlib import Path

from nora_agi import NoraAGI
from nora_besle import besle
from nora_lang import tr_lower

OUT = Path("runs/tutor_paraphrase_confirmation_v4.json")
PROMPTS = [
    "Ders çalışmak istiyorum; bilgisayar konusunu bana öğret.",
    "Algoritmanın ne işe yaradığını anlatır mısın?",
    "12 ile 4 üzerinden bir örnekle göster.",
    "Yazılım ve algoritma aynı şey mi?",
    "Bilgisayar ile veri arasındaki farkı anlatır mısın?",
    "Veri öğrenme sürecinde neden gereklidir?",
    "Yapay zekâ ile yazılım arasındaki farkı açıklar mısın?",
    "Bunu bir örnekle somutlaştır.",
    "İnternet bu sistemin neresinde bulunur?",
    "Bilgimi ölçmek için iki soru sor.",
    "Cevabım: yapay zekâ öğrenebilir; internet bilgi paylaşımı ve araştırma için kullanılır. Doğru mu?",
    "Özetleyeyim: bilgisayar veri işler, algoritma problem çözer, internet bilgi paylaşımı yapar. Özetimde eksik var mı?",
    "Kuantum bilgisayar için kesin olarak neleri bildiğini söyle.",
    "Öğrendiklerimizi üç maddede özetle.",
    "Çalışmayı bitir ve dersi toparla.",
]

agi = NoraAGI(seed=0, boot_thinker=False, code_pretrain=False)
besle(agi, verbose=False)
agi.attach_broca("data/checkpoints/turkish_moe.json")
turns = []
started = time.perf_counter()
for index, prompt in enumerate(PROMPTS, 1):
    before = time.perf_counter()
    reply = agi.converse(prompt)
    trace = dict(agi.broca_organ.last_speech_plan.get("tutorial_trace", {}) or {})
    turns.append({"turn": index, "student": prompt, "nora": reply, "trace": trace,
                  "seconds": time.perf_counter() - before})
    print(f"T{index:02d} [{trace.get('intent')}] {reply}", flush=True)

r = [tr_lower(turn["nora"]) for turn in turns]
t = [turn["trace"] for turn in turns]
direct = {(edge.subj, edge.rel, edge.obj) for edge in agi.lang.graph.edges}
checks = {
    "all_turns_traced": all(trace.get("session_turn") == i for i, trace in enumerate(t, 1)),
    "all_trace_facts_grounded": all(tuple(fact) in direct for trace in t for fact in trace.get("facts", [])),
    "computer_basis": all(word in r[0] for word in ("bilgisayar", "makine", "hesap")),
    "algorithm_role": "algoritma" in r[1] and "problem" in r[1] and "çöz" in r[1],
    "new_arithmetic": str(12 + 4) in r[2] and "12" in r[2] and "4" in r[2],
    "algorithm_software_distinction": "aynı kavram değil" in r[3],
    "computer_data_distinction": "aynı kavram değil" in r[4] and "veri" in r[4],
    "data_without_context_jump": "veri" in r[5] and "öğren" in r[5] and "beyin" not in r[5],
    "scoped_ai_difference": "öğrenebilir" in r[6] and "sınırl" in r[6],
    "self_example": "2 + 3" in turns[7]["nora"] and "öğrenme" in r[7],
    "internet_relation": "internet" in r[8] and "bilgisayar" in r[8] and "bilgi" in r[8],
    "quiz": t[9].get("quiz_pending") is True and len(t[9].get("facts", [])) == 2,
    "evaluation": r[10].startswith("evet") and "uyumlu" in r[10],
    "student_summary": "çelişmiyor" in r[11] and "kısmen" not in r[11],
    "quantum_certainty": "kuantum bilgisayar" in r[12] and "bilgisayardır" in r[12] and "kanıt" in r[12] and "yok" in r[12],
    "three_part_summary": len(turns[13]["nora"].splitlines()) == 3,
    "natural_close": agi.deliberation.tutorial.get("active") is False and "çalıştık" in r[14],
    "surface_and_context_quality": not any(any(bad in reply for bad in (
        "madde kütledir", "operatör raporu", "bilgisayar bir gün", "olduk anlatabilirim",
        ". ve ", "b'in", "birdır", "yazılımdur", "ay bir uydu", "beyin öğrenebilir")) for reply in r),
}
stats = agi.deliberation.tutorial.get("neural_stats", {})
checks["real_moe_generated_and_validated"] = (
    getattr(agi.life.voice.organ, "seen_tokens", 0) > 0
    and stats.get("generated", 0) >= stats.get("accepted", 0) > 0
    and any(trace.get("neural_used", 0) > 0 for trace in t)
)
digest = hashlib.sha256()
for filename in ("nora_deliberation.py", "nora_agi.py", "agi_battery.py", "nora_priors.json"):
    digest.update(filename.encode("utf-8") + b"\0" + Path(filename).read_bytes())
result = {
    "scenario": "final_fresh_independent_paraphrase_confirmation_v4",
    "code_sha256": digest.hexdigest(),
    "prior_confirmation_sets_used_in_development": [1, 2, 3],
    "this_prompt_set_used_in_development": False,
    "model": "data/checkpoints/turkish_moe.json",
    "moe_seen_tokens": int(getattr(agi.life.voice.organ, "seen_tokens", 0)),
    "neural_stats": stats,
    "arithmetic_oracle": {"expression": "12+4", "expected": 12 + 4},
    "checks": checks,
    "passed": all(checks.values()),
    "seconds": time.perf_counter() - started,
    "turns": turns,
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("CHECKS", json.dumps(checks, ensure_ascii=False), flush=True)
print("NEURAL", stats, flush=True)
print("PASSED", result["passed"], "SECONDS", round(result["seconds"], 3), flush=True)
raise SystemExit(0 if result["passed"] else 1)
