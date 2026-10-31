"""
Sözcü 2014 haber metinleri üzerinde Nora Broca Artikülasyon Organını tek geçişte (online/continual) eğitir.
Orijinal ağırlıklar korunur, yeni model 'data/broca_tr_sozcu.pt' olarak kaydedilir.
"""
import os
import sys
import time
import random
import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from nora.cognition.agi import NoraAGI
from nora.language.voice import SubwordTokenizer, state_vector, F_IDX
from nora.language.broca_net import BrocaOrgan
from nora.synthesis.egit import sentences_of


def main():
    torch.set_num_threads(4)
    data_path = os.path.join(_REPO_ROOT, "data", "text", "sozcu_2014.txt")
    base_broca = os.path.join(_REPO_ROOT, "data", "broca_tr.pt")
    out_broca = os.path.join(_REPO_ROOT, "data", "broca_tr_sozcu.pt")
    tok_path = os.path.join(_REPO_ROOT, "data", "tok_tr.json")

    print(f"Cümleler yükleniyor: {data_path} ...")
    sents = sentences_of([data_path])
    random.Random(42).shuffle(sents)
    
    # 2000 cümleyi değerlendirme (validation) için ayıralım
    val_sents = sents[:2000]
    train_sents = sents[2000:]
    print(f"Toplam cümle: {len(sents)} (Eğitim: {len(train_sents)}, Doğrulama: {len(val_sents)})")

    print("Mevcut Broca organı yükleniyor...")
    org = BrocaOrgan.load(base_broca)
    agi = NoraAGI(seed=42)
    st_info = state_vector(agi)
    
    # Başlangıç değerlendirmesi
    eval_pre = org.evaluate(val_sents[:500], st_info)
    print(f"Eğitim Öncesi Doğrulama Skoru: {eval_pre}")

    B = 64
    step = 0
    t0 = time.time()
    losses = []

    print("\n--- Tek Geçişli (Continual) Haber Eğitimi Başlıyor ---")
    for i in range(0, len(train_sents), B):
        batch = train_sents[i : i + B]
        if not batch:
            continue
        
        # LR ısınma ve sönümleme
        org.lr = 1e-4 / (1.0 + step / 300.0) ** 0.5
        loss = org.learn_batch(batch, st_info, anchor_dropout=0.1, loss_mode="sampled", negative_samples=1024)
        losses.append(loss)
        step += 1

        if step % 50 == 0:
            avg_loss = float(np.mean(losses[-50:]))
            elapsed = time.time() - t0
            print(f"Adım: {step:4d} | Görülen Token: {org.seen_tokens:7d} | Son 50 Kayıp (Loss): {avg_loss:.4f} | Süre: {elapsed:.1f}s", flush=True)

    print(f"\nEğitim tamamlandı! Toplam süre: {time.time() - t0:.1f}s")
    
    eval_post = org.evaluate(val_sents[:500], st_info)
    print(f"Eğitim Sonrası Doğrulama Skoru: {eval_post}")
    
    org.update_common()
    org.save(out_broca)
    print(f"Yeni Broca organı kaydedildi: {out_broca}")

    # Örnek üretimler
    print("\n--- Örnek Cümle Üretimleri (Yeni Model) ---")
    anchors = [["türkiye"], ["hükümet"], ["seçim"], ["ekonomi"], ["meclis", "karar"], ["enflasyon"]]
    for anc in anchors:
        seqs = org.generate(st_info, anc, n=1)
        print(f"Çapa: {anc} -> {[org.detok(x) for x in seqs]}")


if __name__ == "__main__":
    main()
