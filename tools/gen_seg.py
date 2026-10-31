import sys, os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

"""Segment kafalarından üretim sondası: her bilgi kendi dilinde konuşuyor mu?"""
import sys
sys.path.insert(0, "D:\\NoRa_AGI")
import torch
import torch.nn.functional as FF
from micro_gpt import MicroGPT
from grow_gpt import AutoBrain

ck = torch.load("D:\\NoRa_AGI\\models\\micro_tiny.pt", map_location="cpu", weights_only=False)
chars, cfg = ck["chars"], ck["cfg"]
bb = MicroGPT(len(chars), cfg["n_layer"], cfg["n_head"], cfg["n_embd"], cfg["block"])
bb.load_state_dict(ck["state"])
src = torch.load("D:\\NoRa_AGI\\models\\grow_auto.pt", map_location="cpu", weights_only=False)
n_ext = len(src.get("ext_chars", []))
brain = AutoBrain(bb, n_ext, src.get("bn", 64))
raw = src["adapts"] if src.get("nested") else [[s] for s in src["adapts"]]
for seg, h_sd in zip(raw, src["heads"]):
    brain.open_segment()
    brain.adapts[-1] = torch.nn.ModuleList()
    for a_sd in seg:
        na = brain._make_adapter()
        na.load_state_dict(a_sd)
        brain.adapts[-1].append(na)
    brain.heads[-1].load_state_dict(h_sd)
if n_ext:
    brain.emb_ext.load_state_dict(src["emb_ext"])
    brain.head_ext.load_state_dict(src["head_ext"])
brain.eval()
ext = src.get("ext_chars", [])
allc = chars + ext
stoi = {c: i for i, c in enumerate(allc)}
prompts = {"C": "def fib(n):", "M": "12 + 34 =", "T": "Ben eve"}
for si, tg in enumerate(src["tags"]):
    pr = prompts.get(tg, "Once ")
    idx = torch.tensor([[stoi.get(c, 0) for c in pr]], dtype=torch.long)
    out = list(pr)
    with torch.no_grad():
        for _ in range(120):
            lo = brain.forward_seg(idx[:, -cfg["block"]:], si)[0, -1]
            nxt = torch.multinomial(FF.softmax(lo / 0.7, dim=-1), 1).item()
            out.append(allc[nxt])
            idx = torch.cat([idx, torch.tensor([[nxt]])], dim=1)
    print("--- seg%d [%s] <%s>" % (si, tg, pr))
    print("".join(out))
    print()
