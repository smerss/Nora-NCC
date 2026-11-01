"""~500M parametreli Türkçe decoder-only dil modelinin mimarisini kurar ve parametre sayımını doğrular.

Eğitim yapmaz. Ağırlıklar meta cihazda oluşturulur; bu yüzden bellek harcamadan sayım yapılır.

    python tools/build_tr_lm_500m.py --out models/tr_lm_500m_config.json
"""

import argparse, json, os, sys

import torch
import torch.nn as nn

# tok_tr.json içindeki alt-sözcük sözlüğü (Broca ile aynı boyut).
VOCAB = 16003


class Block(nn.Module):
    def __init__(self, d, heads, ffn):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class TrLM(nn.Module):
    def __init__(self, vocab, d, layers, heads, ffn, ctx):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads, ffn) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight  # girişle çıkış ağırlıkları paylaşılır
        self.register_buffer("causal", torch.triu(torch.full((ctx, ctx), float("-inf")), 1), persistent=False)

    def forward(self, ids):
        t = ids.shape[1]
        x = self.emb(ids) + self.pos(torch.arange(t, device=ids.device))
        mask = self.causal[:t, :t]
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln(x))


def count(model):
    return sum(p.numel() for p in model.parameters())  # paylaşılan ağırlık bir kez sayılır


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=1280)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--heads", type=int, default=20)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ffn = 4 * args.d
    with torch.device("meta"):
        model = TrLM(VOCAB, args.d, args.layers, args.heads, ffn, args.ctx)
    total = count(model)
    config = {"vocab": VOCAB, "d_model": args.d, "layers": args.layers, "heads": args.heads,
              "ffn": ffn, "context": args.ctx, "tied_embeddings": True, "total_parameters": total,
              "trained": False, "note": "mimari ve sayım; eğitim yapılmadı"}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(json.dumps(config, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
