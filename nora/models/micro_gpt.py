from nora._paths import REPO_ROOT
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""micro_gpt.py — Bu PC'de (2 çekirdek CPU, 16GB) eğitilebilir minik karakter-GPT.
Felsefe: az ama temiz veri + müfredat + tekrar = token başına max kazanç.
Tüm yollar D diskinde. Kullanım:
  DPY D:\\NoRa_AGI\\micro_gpt.py --train --iters 400            # duman testi (tinyshakespeare)
  DPY D:\\NoRa_AGI\\micro_gpt.py --train --data D:\\NoRa_AGI\\data\\text\\tiny_slice.txt --n_embd 256 --iters 3000
  DPY D:\\NoRa_AGI\\micro_gpt.py --gen --prompt "Once upon a " --n 300
"""
import argparse, math, os, sys, time, urllib.request

PROJECT_ROOT = os.path.join("D:" + os.sep, "NoRa_AGI") if os.name == "nt" else os.path.join(os.path.expanduser("~"), "NoRa_AGI")
TEXT_DIR = os.path.join(PROJECT_ROOT, "data", "text")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_text(args):
    os.makedirs(TEXT_DIR, exist_ok=True)
    if args.data and os.path.exists(args.data):
        path = args.data
    else:
        path = os.path.join(TEXT_DIR, "tinyshakespeare.txt")
        if not os.path.exists(path):
            print("  indiriliyor: tinyshakespeare (1MB)")
            urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    return open(path, encoding="utf-8").read()


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd, bias=False)
        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd, bias=False)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd, bias=False), nn.GELU(),
                                 nn.Linear(4 * n_embd, n_embd, bias=False))

    def forward(self, x, mask):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class MicroGPT(nn.Module):
    def __init__(self, vocab, n_layer, n_head, n_embd, block):
        super().__init__()
        self.block_size = block
        self.tok = nn.Embedding(vocab, n_embd)
        self.pos = nn.Embedding(block, n_embd)
        self.layers = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd, bias=False)
        self.head = nn.Linear(n_embd, vocab, bias=False)
        self.apply(lambda m: nn.init.normal_(m.weight, 0, 0.02) if isinstance(m, (nn.Linear, nn.Embedding)) else None)

    def forward(self, idx):
        idx = idx.long()  # uint8 depolama uyumu (long'da no-op)
        B, T = idx.shape
        mask = torch.triu(torch.full((T, T), float("-inf")), 1)
        x = self.tok(idx) + self.pos(torch.arange(T))
        for l in self.layers:
            x = l(x, mask)
        return self.head(self.ln(x))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def batchify(ids, block, batch, rng):
    ix = torch.randint(0, len(ids) - block - 1, (batch,), generator=rng)
    # .long(): uint8 depolamayla uyumluluk (long girdide no-op, kopyasız)
    x = torch.stack([ids[i:i + block] for i in ix]).long()
    y = torch.stack([ids[i + 1:i + block + 1] for i in ix]).long()
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--prompt", default="Once upon a ")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=192)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--curriculum", action="store_true", help="ilk %%20 adımda kısa satırlardan örnekle")
    ap.add_argument("--ckpt", default=os.path.join(MODEL_DIR, "micro.pt"))
    ap.add_argument("--resume", default=None, help="egitime bu ckpt'den devam et (kelime dagarcigi ayni olmali)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    os.makedirs(MODEL_DIR, exist_ok=True)

    if args.gen and not args.data:
        ck0 = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        chars = ck0["chars"]
        text = None
        print(f"uretken kip: vocab ckpt'den ({len(chars)}), cfg={ck0['cfg']['n_layer']}L/{ck0['cfg']['n_embd']}e")
    else:
        text = get_text(args)
        chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    if args.train:
        ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        n = int(0.9 * len(ids))
        tr, va = ids[:n], ids[n:]
        print(f"veri: {len(text)/1e6:.1f}M karakter, vocab={len(chars)}, train={len(tr)}, val={len(va)}")

    if args.train:
        model = MicroGPT(len(chars), args.n_layer, args.n_head, args.n_embd, args.block)
        print(f"parametre: {model.n_params()/1e6:.2f}M")
        if args.resume:
            r = torch.load(args.resume, map_location="cpu", weights_only=False)
            assert r["chars"] == chars, "resume vocab uyusmuyor"
            model.load_state_dict(r["state"])
            print(f"devam: {args.resume}")
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        rng = torch.Generator().manual_seed(args.seed)
        lines = sorted(text.split("\n"), key=len)
        easy = "".join(l + "\n" for l in lines[:len(lines) // 3])
        easy_ids = torch.tensor([stoi.get(c, 0) for c in easy], dtype=torch.long)
        t0 = time.time()
        model.train()
        for it in range(1, args.iters + 1):
            # müfredat: ilk %20 adımda kısa-satır diliminden örnekle
            pool = easy_ids if (args.curriculum and it < args.iters // 5) else tr
            x, y = batchify(pool, args.block, args.batch, rng)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, len(chars)), y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it % 50 == 0 or it == 1:
                with torch.no_grad():
                    vx, vy = batchify(va, args.block, args.batch, rng)
                    vl = F.cross_entropy(model(vx).view(-1, len(chars)), vy.view(-1)).item()
                print(f"  iter {it}/{args.iters}  train={loss.item():.3f}  val={vl:.3f}  {time.time()-t0:.0f}sn")
        torch.save({"state": model.state_dict(), "chars": chars, "cfg": vars(args)}, args.ckpt)
        print(f"kaydedildi: {args.ckpt}")

    if args.gen:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        chars, model = ck["chars"], MicroGPT(len(ck["chars"]), ck["cfg"]["n_layer"], ck["cfg"]["n_head"], ck["cfg"]["n_embd"], ck["cfg"]["block"])
        model.load_state_dict(ck["state"])
        model.eval()
        stoi = {c: i for i, c in enumerate(chars)}
        idx = torch.tensor([[stoi.get(c, 0) for c in args.prompt]], dtype=torch.long)
        out = list(args.prompt)
        with torch.no_grad():
            for _ in range(args.n):
                logits = model(idx[:, -model.block_size:])
                nxt = torch.multinomial(F.softmax(logits[0, -1] / 0.8, dim=-1), 1).item()
                out.append(chars[nxt])
                idx = torch.cat([idx, torch.tensor([[nxt]])], dim=1)
        print("".join(out))


if __name__ == "__main__":
    main()
