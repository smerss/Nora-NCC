#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_fly_reservoir.py — FlyWire konektomundan 96 nöronluk yoğun bir alt çizge çıkarır (süreklilik durumunun yinelemeli ağırlıkları).
Kaynak: nora_neuron.py --stage convert ile üretilen flywire npz. Çıktı: data/fly_reservoir_96.npz (işaretli sinaps sayıları, post×pre).
Seçim: en yüksek toplam bağlantıya sahip çekirdekten başlayıp, kümeye en güçlü bağlı nöronu ekleyerek büyütür (kümenin aynı bölgeden olması için)."""

import sys, os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import nora  # noqa: F401  (eski nora_xxx import'ları için uyumluluk kancası)
import sys, numpy as np
sys.path.insert(0, '.')
from nora_neuron import Connectome

def main(path, out, n=96, seed=0):
    c = Connectome.load(path)
    deg = np.diff(c.indptr)
    rng = np.random.default_rng(seed)
    start = int(rng.choice(np.argsort(-deg)[200:400]))             # çok bağlı ama aşırı uç olmayan çekirdek
    S = [start]
    score = {}
    def add_neighbors(i):
        lo, hi = c.indptr[i], c.indptr[i + 1]
        for j, w in zip(c.indices[lo:hi], c.weight[lo:hi]):
            score[int(j)] = score.get(int(j), 0.0) + abs(float(w))
    add_neighbors(start)
    while len(S) < n:
        cand = [(v, j) for j, v in score.items() if j not in S]
        if not cand:
            break
        j = max(cand)[1]
        S.append(j)
        add_neighbors(j)
    idx = {j: k for k, j in enumerate(S)}
    W = np.zeros((len(S), len(S)), np.float32)                        # W[post, pre]
    for pre in S:
        lo, hi = c.indptr[pre], c.indptr[pre + 1]
        for post, w in zip(c.indices[lo:hi], c.weight[lo:hi]):
            if int(post) in idx:
                W[idx[int(post)], idx[pre]] += float(w)
    np.savez(out, W=W, flywire_index=np.array(S, np.int64), flywire_id=(c.ids[S] if c.ids is not None else np.array(S)))
    sgn = np.sign(W)
    mixed = int(sum(len(set(sgn[:, j][sgn[:, j] != 0])) > 1 for j in range(W.shape[1])))
    print("alt çizge: %d nöron, %d bağlantı (yoğunluk %.2f), Dale ihlali (karışık işaretli sütun) %d, |W| ort %.1f" %
          (len(S), int((W != 0).sum()), (W != 0).mean(), mixed, np.abs(W[W != 0]).mean()))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "data/fly_reservoir_96.npz")
