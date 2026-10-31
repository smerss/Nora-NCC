"""M kafasında '2 + 3 = ' sonrası ilk-token dağılımı: doğru cevap örneklenebilir mi?"""

import sys, os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import nora  # noqa: F401  (eski nora_xxx import'ları için uyumluluk kancası)
import sys
sys.path.insert(0, "D:\\NoRa_AGI")
import torch
import torch.nn.functional as FF
from micro_gpt import MicroGPT
from grow_gpt import rebuild_brain

ck = torch.load("D:\\NoRa_AGI\\models\\micro_tiny.pt", map_location="cpu", weights_only=False)
chars, cfg = ck["chars"], ck["cfg"]
bb = MicroGPT(len(chars), cfg["n_layer"], cfg["n_head"], cfg["n_embd"], cfg["block"])
bb.load_state_dict(ck["state"])
src = torch.load(sys.argv[1] if len(sys.argv) > 1 else "D:\\NoRa_AGI\\models\\grow_auto.pt",
                   map_location="cpu", weights_only=False)
brain = rebuild_brain(src, bb)
brain.eval()
si = list(src["tags"]).index("M")
allc = chars + src.get("ext_chars", [])
stoi = {c: i for i, c in enumerate(allc)}
for pr in ["2 + 3 =", "12 + 34 =", "What is 2 plus 3? It is"]:
    idx = torch.tensor([[stoi.get(c, 0) for c in pr]], dtype=torch.long)
    with torch.no_grad():
        lo = brain.forward_seg(idx, si)[0, -1]
    p = FF.softmax(lo, -1)
    top = torch.topk(p, 8)
    show = " ".join("'%s':%.3f" % (allc[i], float(p[i])) for i in top.indices.tolist())
    print("%-28s -> %s" % (pr, show), flush=True)
