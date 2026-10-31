import sys, os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

"""Şaşkınlık-vektör probu: donuk omurgada A/C/M gizli-ortalama uzaklıkları."""
import sys
sys.path.insert(0, "D:\\NoRa_AGI")
import torch
from micro_gpt import MicroGPT
from grow_gpt import load_ids

ck = torch.load("D:\\NoRa_AGI\\models\\micro_tiny.pt", map_location="cpu", weights_only=False)
chars, cfg = ck["chars"], ck["cfg"]
stoi = {c: i for i, c in enumerate(chars)}
bb = MicroGPT(len(chars), cfg["n_layer"], cfg["n_head"], cfg["n_embd"], cfg["block"])
bb.load_state_dict(ck["state"])
bb.eval()


def mean_h(ten, seed, nw=6):
    rg = torch.Generator().manual_seed(seed)
    vs = []
    B, T = 8, cfg["block"]
    for _ in range(nw):
        s0 = int(torch.randint(0, len(ten) - T - 1, (1,), generator=rg))
        x = ten[s0:s0 + T].unsqueeze(0).repeat(B, 1).long()
        with torch.no_grad():
            mask = torch.triu(torch.full((T, T), float("-inf")), 1)
            h = bb.tok(x) + bb.pos(torch.arange(T))
            for l in bb.layers:
                h = l(h, mask)
            vs.append(bb.ln(h).mean(dim=(0, 1)))
    import numpy as _np
    M = torch.stack(vs)
    return M.mean(0), float(((M - M.mean(0)) ** 2).sum(1).mean())


A, _ = load_ids("D:\\NoRa_AGI\\data\\text\\tiny_slice.txt", stoi, 300000)
C, _ = load_ids("D:\\NoRa_AGI\\data\\text\\code_py.txt", stoi, 300000)
Mt, _ = load_ids("D:\\NoRa_AGI\\data\\text\\math_synth.txt", stoi, 300000)
pa, sa = mean_h(A, 1)
pc, sc = mean_h(C, 2)
pm, sm = mean_h(Mt, 3)
print("A spread: %.2f | C spread: %.2f | M spread: %.2f" % (sa, sc, sm), flush=True)
print("d(A,C): %.2f | d(A,M): %.2f | d(C,M): %.2f" % (
    float(((pa - pc) ** 2).sum()), float(((pa - pm) ** 2).sum()), float(((pc - pm) ** 2).sum())), flush=True)
