import json
import time
from pathlib import Path

from nora_agi import NoraAGI
from nora_besle import besle
from nora_lang import tr_lower

OUT = Path("runs/tutor_paraphrase_confirmation_v3.json")
PROMPTS = [
    "Konu çalışıyorum. Önce bilgisayarın temel işlevini öğret.",
    "Bilgisayarın veriyle ne yaptığını açıklayabilir misin?",
    "Algoritma bu işin neresinde devreye girer?",
    "10 ile 3 için örnek bir işlem göster.",
    "Algoritma ve yazılım arasındaki farkları sıralar mısın?",
    "Peki yazılım ne yapar?",
    "Yapay zekâ ile yazılım arasında nasıl bir fark var?",
    "Bu farkı bir örnekle açıklayalım.",
    "İnternet bu yapının neresinde?",
    "Veri ile internet ne için kullanılır?",
    "Bilgimi değerlendirmek için bana iki soru sor.",
    "Cevabım: internet bilgi paylaşımı ve araştırma için kullanılır; veri öğrenmek için kullanılır. Doğru mu?",
    "Özetleyeyim: bilgisayar veri işler, algoritma problem çözer, yazılım programdır. Özetimde eksik var mı?",
    "Bence internet ile bilgisayar aynı şey; sence bu öğrendiklerimizle çelişiyor mu?",
    "Kuantum bilgisayar hakkında yalnız kesin bildiklerini anlat.",
    "Öğrendiklerimizi dört maddede özetler misin?",
    "Son bir karşılaştırma: algoritma ile bilgisayar arasındaki fark nedir?",
    "Dersi bitir; bugünkü çalışmayı toparla.",
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
    "computer_basis": all(word in r[0] for word in ("bilgisayar", "veri", "hesap")),
    "computer_data_use": "bilgisayar" in r[1] and "veri" in r[1],
    "algorithm_role": "algoritma" in r[2] and "problem" in r[2] and "çöz" in r[2],
    "new_arithmetic": str(10 + 3) in r[3] and "10" in r[3] and "3" in r[3],
    "algorithm_software_distinction": "aynı kavram değil" in r[4] and "yazılım" in r[4],
    "software_ability_limit": "yazılım" in r[5] and "program" in r[5] and "kayd" in r[5] and "yok" in r[5],
    "scoped_ai_difference": "öğrenebilir" in r[6] and "sınırl" in r[6],
    "self_example": "2 + 3" in turns[7]["nora"] and "öğrenme" in r[7],
    "internet_relation": "internet" in r[8] and "bilgisayar" in r[8] and "bilgi" in r[8],
    "data_internet_functions": "veri" in r[9] and "internet" in r[9] and "öğren" in r[9],
    "quiz": t[10].get("quiz_pending") is True and len(t[10].get("facts", [])) == 2,
    "evaluation": r[11].startswith("evet") and "uyumlu" in r[11],
    "student_summary": "çelişmiyor" in r[12],
    "misconception_correction": "aynı kavram değil" in r[13] and "internet" in r[13],
    "quantum_certainty": "kuantum bilgisayar" in r[14] and "bilgisayardır" in r[14] and "kanıt" in r[14] and "yok" in r[14],
    "four_part_summary": len(turns[15]["nora"].splitlines()) == 4,
    "algorithm_computer_distinction": "aynı kavram değil" in r[16] and "algoritma" in r[16] and "bilgisayar" in r[16],
    "natural_close": agi.deliberation.tutorial.get("active") is False and "çalıştık" in r[17],
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
result = {
    "scenario": "fresh_independent_paraphrase_confirmation_v3",
    "prior_confirmation_sets_used_in_development": [1, 2],
    "this_prompt_set_used_in_development": False,
    "model": "data/checkpoints/turkish_moe.json",
    "moe_seen_tokens": int(getattr(agi.life.voice.organ, "seen_tokens", 0)),
    "neural_stats": stats,
    "arithmetic_oracle": {"expression": "10+3", "expected": 10 + 3},
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
