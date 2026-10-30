import hashlib
import json
import time
from pathlib import Path

from nora_agi import NoraAGI
from nora_besle import besle

PROMPTS = [
    "Bir konu çalışıyorum. Bana bilgisayarın ne olduğunu öğretir misin?",
    "Peki algoritma bunun neresinde?",
    "Bunu günlük hayattan bir örnekle açıklar mısın?",
    "Algoritma ile yazılım aynı şey mi?",
    "Bilgisayar algoritmayı nasıl kullanır?",
    "Veri neden gerekli?",
    "Yapay zekâ ile normal bir program arasındaki fark ne?",
    "Bunu kısa bir örnekle karşılaştırır mısın?",
    "Az önce anlattıklarını unutmadan devam edelim: internet bu sistemin neresinde?",
    "Bilgisayar tek başına öğrenebilir mi, yoksa veri mi gerekir?",
    "Bu cevabın önceki veri bilginle tutarlı mı?",
    "Şimdi bana iki soru sor; bilgimi ölç.",
    "Cevabım: yapay zekâ öğrenir; internet bilgi paylaşımı ve araştırma için kullanılır. Doğru mu?",
    "Ben özetleyeyim: bilgisayar hesaplar, algoritma çözüm adımlarını verir, veri öğrenmeyi destekler. Eksik var mı?",
    "Öğrendiklerimizi üç maddede özetler misin?",
    "Son olarak kuantum bilgisayarla normal bilgisayar arasındaki ilişkiyi anlatır mısın?",
    "Bunu yalnız kesin bildiğin kısımlarla anlat; emin olmadığın yerde dur.",
    "Kuantum bilgisayarın hangi özelliğini kesin olarak bildiğini bir kez daha söyler misin?",
    "Peki internet ve veri bu sistemde ne için kullanılır?",
    "Bu bağ hakkında emin olmadığın bir nokta var mı?",
    "Algoritmayı bu kez 4 ile 7 üzerinden kısa bir örnekle açıklar mısın?",
    "Algoritma ile veri arasındaki farkı iki cümlede anlatır mısın?",
    "Bence algoritma ile yazılım aynı şey; sence bu öğrendiklerimizle çelişiyor mu?",
    "Yeni bir soruyla devam edelim: yazılım ne yapar?",
    "Bunu yapay zekâyla karşılaştırınca temel fark ne oluyor?",
    "Şimdi bana iki soru sor; bu kez öğrendiklerimi ölç.",
    "Cevabım: algoritma problemi adım adım çözer; veri öğrenmek için kullanılır. Doğru mu?",
    "Ben de dört bağ kuruyorum: bilgisayar hesaplar, algoritma çözüm verir, veri öğrenmeyi destekler, yapay zekâ öğrenir. Özetimde eksik var mı?",
    "Bugün öğrendiklerimizi dört maddede özetler misin?",
    "Bugünkü çalışmamızı doğal bir sonla toparla.",
]
PRODUCT_FILES = ["nora_deliberation.py", "nora_agi.py", "agi_battery.py", "nora_priors.json"]

agi = NoraAGI(seed=0, boot_thinker=False, code_pretrain=False)
besle(agi, verbose=False)
agi.attach_broca("data/checkpoints/turkish_moe.json")
agi.step_life(1500)
turns = []
started = time.perf_counter()
for index, prompt in enumerate(PROMPTS, 1):
    before = time.perf_counter()
    reply = agi.converse(prompt)
    trace = dict(agi.broca_organ.last_speech_plan.get("tutorial_trace", {}) or {})
    turns.append({"turn": index, "student": prompt, "nora": reply,
                  "trace": trace, "latency_s": time.perf_counter() - before})
    print(f"T{index:02d} [{trace.get('intent')}] {reply}", flush=True)

digest = hashlib.sha256()
for filename in PRODUCT_FILES:
    digest.update(filename.encode("utf-8") + b"\0" + Path(filename).read_bytes())
stats = dict(agi.deliberation.tutorial.get("neural_stats", {}))
result = {
    "turns": turns,
    "elapsed_s": time.perf_counter() - started,
    "graph_edges": len(agi.lang.graph.edges),
    "concepts": len(set(agi.lang.graph._out_edges) | set(agi.lang.graph._in_edges)),
    "subject_concepts": len(agi.lang.graph._out_edges),
    "registered_concepts": len(agi.lang.graph.id_concept),
    "moe_seen_tokens": int(getattr(agi.life.voice.organ, "seen_tokens", 0)),
    "active_after_close": agi.deliberation.tutorial.get("active"),
    "neural_stats": stats,
    "neural_turns": sum(turn["trace"].get("neural_used", 0) > 0 for turn in turns),
    "code_sha256": digest.hexdigest(),
    "setup": "besle -> attach_broca(turkish_moe) -> step_life(1500) -> 30 converse turns",
}
Path("runs/tutor_dialogue_final.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("SUMMARY", json.dumps({key: value for key, value in result.items() if key != "turns"},
                            ensure_ascii=False), flush=True)
