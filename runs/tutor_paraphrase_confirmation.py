import json
import time
from pathlib import Path

from nora_agi import NoraAGI
from nora_besle import besle
from nora_lang import tr_lower

OUT = Path("runs/tutor_paraphrase_confirmation.json")
PROMPTS = [
    "Konu çalışmaya yeni başlıyorum; bilgisayarı temelden öğretir misin?",
    "Algoritmanın bu sistemdeki rolünü açıklayabilir misin?",
    "6 ve 5 ile kısa bir işlem örneği gösterir misin?",
    "Yazılım ile algoritmayı karşılaştırır mısın? Temel farkı merak ediyorum.",
    "Bilgisayarda algoritma nasıl kullanılır?",
    "Veri neden gereklidir?",
    "Yapay zekâyı sıradan yazılımdan ayıran fark nedir?",
    "Bu ayrımı bir örnekle somutlaştırır mısın?",
    "İnternet bu sistemin neresinde yer alır?",
    "Yalnızca kesin olarak bildiklerini söyle; kuantum bilgisayar konusunda kanıtın ne?",
    "Bana iki soru sor da bilgimi ölç.",
    "Cevabım: yapay zekâ öğrenebilir; internet bilgi paylaşımı ve araştırma için kullanılır. Doğru mu?",
    "Özetleyeyim: bilgisayar hesaplar, algoritma problem çözer, veri öğrenmeyi destekler. Özetimde eksik var mı?",
    "Bence yazılım ve algoritma aynı şey; sence öğrendiklerimizle çelişiyor mu?",
    "Öğrendiklerimizi dört maddede özetler misin?",
    "Çalışmayı bitir ve bugünkü dersi toparla.",
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
    turns.append({
        "turn": index, "student": prompt, "nora": reply, "trace": trace,
        "seconds": time.perf_counter() - before,
    })
    print(f"T{index:02d} [{trace.get('intent')}] {reply}", flush=True)

r = [tr_lower(turn["nora"]) for turn in turns]
t = [turn["trace"] for turn in turns]
checks = {
    "all_turns_traced": all(trace.get("session_turn") == i for i, trace in enumerate(t, 1)),
    "computer_basis": all(word in r[0] for word in ("bilgisayar", "makine", "hesap")),
    "algorithm_role": "algoritma" in r[1] and "problem" in r[1] and "çöz" in r[1],
    "independent_arithmetic": str(6 + 5) in r[2] and "6" in r[2] and "5" in r[2],
    "software_algorithm_distinction": "aynı kavram değil" in r[3] and "yazılım" in r[3] and "algoritma" in r[3],
    "relation_followup": "bilgisayar" in r[4] and "algoritma" in r[4],
    "data_learning": "veri" in r[5] and "öğren" in r[5],
    "scoped_ai_difference": "öğrenebilir" in r[6] and "sınırl" in r[6],
    "example_continuation": "2 + 3" in turns[7]["nora"] and "5" in r[7] and "öğrenme" in r[7],
    "internet_relation": "internet" in r[8] and "bilgi" in r[8],
    "quantum_evidence_scope": ("kuantum bilgisayar" in r[9] and "bilgisayardır" in r[9]
                                and "kanıt" in r[9] and "yok" in r[9]),
    "quiz": t[10].get("quiz_pending") is True and len(t[10].get("facts", [])) == 2,
    "evaluation": r[11].startswith("evet") and "uyumlu" in r[11],
    "student_summary": "çelişmiyor" in r[12],
    "misconception_correction": "aynı kavram değil" in r[13],
    "four_part_summary": len(turns[14]["nora"].splitlines()) == 4,
    "natural_close": agi.deliberation.tutorial.get("active") is False and "çalıştık" in r[15],
    "no_known_discontinuity": not any(any(bad in reply for bad in (
        "madde kütledir", "operatör raporu", "bilgisayar bir gün", "olduk anlatabilirim",
        ". ve ", "b'in")) for reply in r),
}
stats = agi.deliberation.tutorial.get("neural_stats", {})
checks["real_moe_generated_and_validated"] = (
    getattr(agi.life.voice.organ, "seen_tokens", 0) > 0
    and stats.get("generated", 0) >= stats.get("accepted", 0) > 0
    and any(trace.get("neural_used", 0) > 0 for trace in t)
)
result = {
    "scenario": "independent_paraphrase_confirmation",
    "development_prompts_reused": False,
    "model": "data/checkpoints/turkish_moe.json",
    "moe_seen_tokens": int(getattr(agi.life.voice.organ, "seen_tokens", 0)),
    "neural_stats": stats,
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
