import json
import time
from pathlib import Path

from nora_agi import NoraAGI
from nora_besle import besle
from nora_lang import tr_lower

OUT = Path("runs/tutor_paraphrase_confirmation_v2.json")
PROMPTS = [
    "Ders çalışıyorum; bana bilgisayar ve algoritma konusunu öğret.",
    "Algoritmanın işlevini kendi cümlelerinle açıklayabilir misin?",
    "7 ile 6 üzerinden bir örneğe geçelim; bu adımları göster.",
    "İnternet ile bilgisayar arasındaki farkı açıklar mısın?",
    "Peki internet ne için kullanılır?",
    "Veri öğrenmek için neden gereklidir?",
    "Yapay zekâ ile yazılım aynı şey midir, yoksa bir ayrım var mı?",
    "Bunu başka bir örnekle açıklayabilir misin?",
    "Kuantum bilgisayar ile bilgisayar arasındaki ilişkiyi anlat.",
    "Bu konuda kesin bildiğin nedir; emin olmadığın ayrıntıları ekleme.",
    "Öğrendiklerimi sınamak için bana iki soru sor.",
    "Cevabım: veri öğrenmek için kullanılır; yapay zekâ öğrenebilir. Doğru mu?",
    "Özetleyeyim: algoritma problem çözer, internet bilgi paylaşımı ve araştırma için kullanılır, veri öğrenmeyi destekler. Özetimde eksik var mı?",
    "Sanırım algoritma ve yazılım aynı şey; sence bu düşüncem öğrendiklerimizle çelişiyor mu?",
    "Şu ana kadarki öğrendiklerimizi üç maddede özetle.",
    "Bilgisayar ile veri arasındaki farkı da açıklar mısın?",
    "Bugünkü çalışmayı toparlayalım; dersi bitir.",
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
    "new_arithmetic": str(7 + 6) in r[2] and "7" in r[2] and "6" in r[2],
    "internet_computer_distinction": "aynı kavram değil" in r[3] and "internet" in r[3] and "bilgisayar" in r[3],
    "internet_function": "internet" in r[4] and "bilgi" in r[4] and "kullanılır" in r[4],
    "data_learning": "veri" in r[5] and "öğren" in r[5],
    "scoped_ai_difference": "aynı kavram değil" in r[6] and "öğrenebilir" in r[6] and "sınırl" in r[6],
    "self_example": "2 + 3" in turns[7]["nora"] and "öğrenme" in r[7],
    "quantum_link": "kuantum bilgisayar" in r[8] and "bilgisayardır" in r[8],
    "certainty_scope": "kanıt" in r[9] and "yok" in r[9],
    "quiz": t[10].get("quiz_pending") is True and len(t[10].get("facts", [])) == 2,
    "evaluation": r[11].startswith("evet") and "uyumlu" in r[11],
    "student_summary": "çelişmiyor" in r[12],
    "misconception_correction": "aynı kavram değil" in r[13],
    "three_part_summary": len(turns[14]["nora"].splitlines()) == 3,
    "computer_data_distinction": "aynı kavram değil" in r[15] and "veri" in r[15],
    "natural_close": agi.deliberation.tutorial.get("active") is False and "çalıştık" in r[16],
    "surface_quality": not any(any(bad in reply for bad in (
        "madde kütledir", "operatör raporu", "bilgisayar bir gün", "olduk anlatabilirim",
        ". ve ", "b'in", "birdır", "yazılımdur")) for reply in r),
}
stats = agi.deliberation.tutorial.get("neural_stats", {})
checks["real_moe_generated_and_validated"] = (
    getattr(agi.life.voice.organ, "seen_tokens", 0) > 0
    and stats.get("generated", 0) >= stats.get("accepted", 0) > 0
    and any(trace.get("neural_used", 0) > 0 for trace in t)
)
result = {
    "scenario": "fresh_independent_paraphrase_confirmation_v2",
    "prior_confirmation_used_in_development": True,
    "this_prompt_set_used_in_development": False,
    "model": "data/checkpoints/turkish_moe.json",
    "moe_seen_tokens": int(getattr(agi.life.voice.organ, "seen_tokens", 0)),
    "neural_stats": stats,
    "arithmetic_oracle": {"expression": "7+6", "expected": 7 + 6},
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
