from nora._paths import REPO_ROOT
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_bridge.py — Eski hat (nora_think: düşün → doğrula → uyu → sezgi) ile yeni hat (yaşayan zihin) köprüsü.

Evrende 'bilmeceli kutular' vardır: yiyecek gibi görünür; dokununca aritmetik soru çıkar, doğru cevap enerji verir.
Ajan her bilmecede KARAR verir: sezgiyle mi (hızlı, güveni kalibre), yoksa düşünerek mi (adım adım, hakemli,
pahalı: her örnek bir dünya adımı = metabolizma)? Karar beklenen değere dayanır:
    EV_sezgi = P(doğru | güven) - fiyat*1        EV_düşün = P_başarı(basamak) - fiyat*maliyet(basamak)
P(doğru | güven) ajanın KENDİ geçmişinden kalibre edilir (öz-denetim: ara sıra güvenli cevap da düşünülür).
Doğrulanmış düşünceler periyodik 'uykuda' sezgiye damıtılır. Akıl yürütmenin adımları nora_think.verify_step
(hakem) ve safe_eval (araç kapısı) ile denetlenir: yanlış adım kabul edilmez.

Kullanım:  python nora_bridge.py --steps 40000 --seed 0
"""
import argparse, math, os, random, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nora_think as nt
from nora_mind import Mind, Universe, I_E, I_W, I_I, Narrator, live as _live_unused  # noqa: F401

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CHARS = list("0123456789+-=[]|>\n ")


class _NullBrain:
    def eval(self): return self
    def train(self): return self
    def parameters(self): return []


def make_thinker(seed=0, d=96, layers=3, block=160):
    """Donuk omurgasız, sıfırdan eğitilen küçük düşünce sütunu (nora_think.Mind arayüzü)."""
    torch.manual_seed(seed)
    m = object.__new__(nt.Mind)
    m.src, m.bb, m.si, m.cfg = {}, None, 0, {"block": block}
    m.brain = _NullBrain()
    m.allc, m.stoi, m.block, m.V0 = CHARS, {c: i for i, c in enumerate(CHARS)}, block, len(CHARS)
    m.cortex, m.col_cfg = "column", {}
    m.col = nt.ThinkColumn(len(CHARS), block, d=d, n_layer=layers, lateral=False, stoi=m.stoi)
    return m


def train_masked(mind, lines, iters, batch, lr, seed, block=160):
    """Satır bazlı, MASKELİ eğitim: kayıp yalnızca '=' sonrası karakterlere (sezgi cevabı / düşünce izi) uygulanır.
    Soru kısmı (rastgele sayılar) öğrenilemez gürültüdür; sözdizimsel göreli koordinatlar bir kez önceden hesaplanır."""
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(seed)
    enc = [[mind.stoi[c] for c in (l + "\n")][:block] for l in lines]
    L = max(len(e) for e in enc)
    X = torch.zeros(len(enc), L, dtype=torch.long)
    M = torch.zeros(len(enc), L, dtype=torch.bool)
    Wt = torch.ones(len(enc), L)
    for i, (e, l) in enumerate(zip(enc, lines)):
        X[i, :len(e)] = torch.tensor(e)
        eq = l.index("=")
        M[i, eq:len(e) - 1] = True                      # t konumundaki girdi t+1'i öngörür
        if "[" not in l:
            Wt[i] = 1.2                                 # doğrudan cevap (sezgi) satırları dengeli ağırlıkta (S2 izini boğmadan)
    all_coords = mind.col.precompute_coords(X[:, :-1]) if hasattr(mind.col, "precompute_coords") else None
    for p in mind.col.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(mind.col.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters, pct_start=0.1)
    mind.col.train()
    loss = None
    for it in range(iters):
        idx = torch.randint(0, len(enc), (batch,), generator=g)
        inp, tgt, m, wt = X[idx, :-1], X[idx, 1:], M[idx, :-1], Wt[idx, :-1]
        sub_c = tuple(t[idx] for t in all_coords) if all_coords is not None else None
        logits = mind.col(inp, None, coords=sub_c) if sub_c is not None else mind.col(inp, None)
        ce = F.cross_entropy(logits[m], tgt[m], reduction="none")
        loss = (ce * wt[m]).sum() / wt[m].sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    mind.col.eval()
    for p in mind.col.parameters():
        p.requires_grad = False
    return float(loss.detach())


class Cognition:
    """Yaşayan zihnin akıl yürütme organı: sezgi ↔ düşünme kararı, kalibrasyon, uyku."""

    def __init__(self, seed=0, boot_iters=3000, boot_n=3000, K=8, retries=3, sleep_every=30, sleep_iters=120,
                 price=0.004, audit=0.1, verbose=True, direct_every=1, lr=6e-3, batch=32, use_cache=True):
        torch.set_num_threads(2)
        self.mind, self.rng = make_thinker(seed), random.Random(seed)
        self.cur = nt.Curiosity(None, "auto")
        self.K, self.retries, self.price, self.audit = K, retries, price, audit
        self.sleep_every, self.sleep_iters = sleep_every, sleep_iters
        self.calib, self.tau = [], float("inf")
        self.p_think = {n: 0.8 for n in range(1, 5)}            # P(düşünme başarılı | basamak)
        self.c_think = {n: 6.0 * n for n in range(1, 5)}        # düşünme maliyeti (dünya adımı)
        self.memory, self.today = [], []
        self.n_puz = self.n_ok = self.n_int = self.n_think = self.n_int_ok = 0
        self.teach, self.lr, self.batch = [], lr, batch
        trng = random.Random(7)
        # gün-0 öğretmen gösterimi: yalnızca 1-2 basamak (3+ basamak HİÇ gösterilmez; sıfır-atış genelleme)
        for i in range(boot_n):
            n = 1 if i < int(0.45 * boot_n) else 2
            a, op, b = nt.make_problem(trng, nt.OPS[i % 2], n)
            self.teach.append(nt.think_line(a, op, b, nt.teacher_steps(a, op, b), nt.answer_of(a, op, b)))
            if (n == 1 and i % direct_every == 0) or (n == 2 and i % max(4, direct_every) == 0):
                self.teach.append(nt.direct_line(a, op, b, nt.answer_of(a, op, b)))
        t0 = time.time()
        self.mind.grow(5)
        cache_dir = str(REPO_ROOT / "models")
        cache_path = os.path.join(
            cache_dir,
            f"think_col_s{seed}_i{boot_iters}_n{boot_n}_d{direct_every}_b{batch}_lr{lr:g}.pt",
        )
        loaded_cache = False
        if use_cache and os.path.exists(cache_path):
            try:
                ckpt = torch.load(cache_path, map_location="cpu", weights_only=True)
                self.mind.col.load_state_dict(ckpt["col_state"])
                self.mind.col.eval()
                for p in self.mind.col.parameters():
                    p.requires_grad = False
                loss = float(ckpt.get("loss", 0.0))
                loaded_cache = True
            except Exception:
                loaded_cache = False
        if not loaded_cache:
            loss = train_masked(self.mind, self.teach, boot_iters, self.batch, self.lr, seed)
            if use_cache and boot_iters > 0:
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    tmp_path = f"{cache_path}.{os.getpid()}.tmp"
                    torch.save({"col_state": self.mind.col.state_dict(), "loss": loss}, tmp_path)
                    os.replace(tmp_path, cache_path)
                except Exception:
                    pass
        if verbose:
            print("[düşünce sütunu] gün-0 öğretmen gösterimi (yalnızca 1-2 basamak): %d satır, %d iter, kayıp %.3f, %.0fs%s" %
                  (len(self.teach), boot_iters, loss, time.time() - t0, " (önbellekten)" if loaded_cache else ""), flush=True)
        self.last = None                                         # son bilmecenin kaydı (anlatıcı için)
        self.log = []

    # ---- kalibrasyon: P(sezgi doğru | güven)
    def p_correct(self, conf, n=None):
        """Bağlama (basamak sayısına) özgü kalibrasyon: aynı zorlukta, benzer güvenli geçmiş sezgilerin doğruluk oranı."""
        pool = [t for t in self.calib if n is None or t[2] == n]
        if len(pool) < 10:
            pool = self.calib
        if len(pool) < 12:
            return 0.5
        near = sorted(pool, key=lambda t: abs(t[0] - conf))[:15]
        return float(np.mean([t[1] for t in near]))

    def new_problem(self):
        op, n = self.cur.pick(self.rng)
        return nt.make_problem(self.rng, op, n), n

    def solve(self, a, op, b, n, urgency=0.0):
        """Döner: (cevap, dünya adımı maliyeti, kayıt)."""
        g, conf = self.mind.intuition(a, op, b)
        truth = nt.answer_of(a, op, b)
        pi = self.p_correct(conf, min(n, 4)) if g is not None else 0.0
        price = self.price * (1.0 + urgency)
        ev_int = pi - price * 1
        ev_think = self.p_think[min(n, 4)] - price * self.c_think[min(n, 4)]
        think = ev_think > ev_int or conf < self.tau * 0.0 or self.rng.random() < self.audit
        rec = dict(a=a, op=op, b=b, n=n, g=g, conf=conf, p_int=pi, ev_int=ev_int, ev_think=ev_think, truth=truth)
        if not think:
            rec.update(mode="sezgi", ans=g if g is not None else 0, cost=1)
            return rec["ans"], 1, rec
        d = self.mind.deliberate(a, op, b, self.K, self.retries, scaffold=self.cur.scaffold(op, min(n, 4)))
        cost = 1 + int(math.ceil(d["samples"] / 4.0))
        if d["ok"]:
            rec.update(mode="düşünce", ans=d["ans"], cost=cost, steps=d["steps"], perc=d["perc"])
        else:
            rec.update(mode="düşünce-başarısız", ans=g if g is not None else 0, cost=cost, perc=d.get("perc"))
        return rec["ans"], cost, rec

    def outcome(self, rec, correct):
        """Dünyadan geri bildirim: öğrenme kayıtları."""
        self.n_puz += 1
        self.n_ok += int(correct)
        n = min(rec["n"], 4)
        if rec["mode"] == "sezgi":
            self.n_int += 1
            self.n_int_ok += int(correct)
            self.calib.append((rec["conf"], int(correct), n))
        else:
            self.n_think += 1
            ok = rec["mode"] == "düşünce"
            self.p_think[n] += 0.1 * (int(ok) - self.p_think[n])
            self.c_think[n] += 0.1 * (rec["cost"] - self.c_think[n])
            if ok:                                              # doğrulanmış cevap: sezginin doğruluğu da bilinir
                self.calib.append((rec["conf"], int(rec["g"] == rec["ans"]), n))
                self.today.append(nt.think_line(rec["a"], rec["op"], rec["b"], rec["steps"], rec["ans"]))
                self.today.append(nt.direct_line(rec["a"], rec["op"], rec["b"], rec["ans"]))
                self.memory.append([rec["a"], rec["op"], rec["b"]])
            self.cur.update(rec["op"], n, int(ok), (rec["g"] == rec["ans"]) if ok else None, rec.get("perc"))
        self.calib = self.calib[-400:]
        self.tau = nt.calibrate([(c, ok) for c, ok, _ in self.calib], 0.95)
        rec["correct"] = bool(correct)
        self.last = rec
        self.log.append(dict(n=rec["n"], conf=rec["conf"], g_ok=(rec["g"] == rec["truth"]), mode=rec["mode"], correct=bool(correct), cost=rec["cost"], p_int=rec["p_int"]))
        if self.n_puz % self.sleep_every == 0:
            self.sleep()

    def sleep(self):
        """UYKU: bugünün doğrulanmış düşünceleri + eski anılar + öğretmen tekrarı -> sezgiye damıtma."""
        replay = []
        for a, op, b in self.rng.sample(self.memory, min(len(self.memory), max(40, len(self.today)))) if self.memory else []:
            ans = nt.answer_of(a, op, b)
            replay += [nt.think_line(a, op, b, nt.teacher_steps(a, op, b), ans), nt.direct_line(a, op, b, ans)]
        lines = self.today * 2 + replay + self.rng.sample(self.teach, min(200, len(self.teach)))
        self.rng.shuffle(lines)
        train_masked(self.mind, lines, self.sleep_iters, self.batch, self.lr * 0.6, self.n_puz)
        self.today = []


def live_bridge(mind, cog, world, steps, window=5000, tag="köprü"):
    obs, died = world.observe(), False
    W = dict(steps=0, deaths=0, comfort=0.0, puz=0, ok=0, sez=0, sez_ok=0, thk=0, thk_steps=0, e_gain=0.0)
    t0 = time.time()
    rows = []
    for t in range(1, steps + 1):
        if t % 500 == 0 and os.path.exists("STOP_NORA"):
            print("STOP_NORA bulundu: durduruldu (adım %d)" % t)
            break
        a = mind.act(obs, died)
        obs, info = world.step(a)
        died = info["died"] is not None
        W["steps"] += 1
        W["deaths"] += died
        W["comfort"] += float(obs[I_E:I_I + 1].min())
        if info["puzzle"] > 0 and not died:
            (aa, op, bb), n = cog.new_problem()
            urgency = float(np.clip(1 - obs[I_E], 0, 1))
            ans, cost, rec = cog.solve(aa, op, bb, n, urgency)
            for _ in range(cost):                                # düşünürken dünya akar: durur ve yaşar
                a2 = mind.act(obs, died)
                obs, info2 = world.step(0)
                died = died or info2["died"] is not None
                W["steps"] += 1
                W["deaths"] += info2["died"] is not None
            correct = ans == rec["truth"]
            if correct:
                world.grant(info["puzzle"])
                W["e_gain"] += info["puzzle"]
            cog.outcome(rec, correct)
            W["puz"] += 1
            W["ok"] += int(correct)
            if rec["mode"] == "sezgi":
                W["sez"] += 1; W["sez_ok"] += int(correct)
            else:
                W["thk"] += 1; W["thk_steps"] += cost
        if t % window == 0:
            n = max(1, W["steps"])
            row = dict(t=t, per1k=1000.0 * W["deaths"] / n, comfort=W["comfort"] / n, puz=W["puz"],
                       ok=W["ok"] / max(1, W["puz"]), sez=W["sez"] / max(1, W["puz"]),
                       sez_acc=W["sez_ok"] / max(1, W["sez"]), thk_cost=W["thk_steps"] / max(1, W["thk"]),
                       tau=cog.tau, e_gain=W["e_gain"])
            rows.append(row)
            print("%s t=%6d ölüm/1k=%4.1f konfor=%.2f | bilmece %2d çözüm oranı %.2f | sezgiyle %.2f (sezgi doğruluğu %.2f) "
                  "düşünme maliyeti %.1f adım | τ=%.2f | enerji+%.2f | %.0fs" %
                  (tag, t, row["per1k"], row["comfort"], row["puz"], row["ok"], row["sez"], row["sez_acc"], row["thk_cost"],
                   cog.tau, row["e_gain"], time.time() - t0), flush=True)
            W = dict(steps=0, deaths=0, comfort=0.0, puz=0, ok=0, sez=0, sez_ok=0, thk=0, thk_steps=0, e_gain=0.0)
    return rows


def metacog_report(cog):
    """Üstbilişsel değerlendirme: 'sezgime güveniyorum / düşünmeliyim' kararı gerçekten sezginin doğruluğunu izliyor mu?"""
    L = cog.log
    if len(L) < 10:
        print("[üstbiliş] yeterli bilmece yok (%d)" % len(L))
        return
    conf = np.array([r["conf"] for r in L]); ok = np.array([r["g_ok"] for r in L])
    pos, neg = conf[ok], conf[~ok]
    from nora_mind import auc
    a = auc(pos, neg)
    print("\n[üstbiliş] sezgi güveni -> sezginin doğruluğu AUROC = %.2f (0.5 şans; n=%d, sezgi doğru %d)" % (a, len(L), int(ok.sum())))
    for n in range(1, 5):
        S = [r for r in L if r["n"] == n]
        if not S:
            continue
        thk = np.mean([r["mode"] != "sezgi" for r in S])
        print("    %d basamak: n=%2d | sezgi doğruluğu %.2f | 'düşünerek' çözme oranı %.2f | ortalama güven %.2f | çözüm oranı %.2f" %
              (n, len(S), np.mean([r["g_ok"] for r in S]), thk, np.mean([r["conf"] for r in S]), np.mean([r["correct"] for r in S])))
    first, last = L[:len(L) // 3], L[-len(L) // 3:]
    print("    yaşam boyunca: ilk üçte bir sezgi doğruluğu %.2f, sezgiyle yanıtlama %.2f | son üçte bir sezgi doğruluğu %.2f, sezgiyle yanıtlama %.2f" %
          (np.mean([r["g_ok"] for r in first]), np.mean([r["mode"] == "sezgi" for r in first]),
           np.mean([r["g_ok"] for r in last]), np.mean([r["mode"] == "sezgi" for r in last])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--window", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot-iters", type=int, default=3000)
    ap.add_argument("--terms", type=int, default=10)
    ap.add_argument("--sleep-every", type=int, default=12)
    args = ap.parse_args()
    world = Universe(seed=args.seed, n_term=args.terms)
    mind = Mind(seed=args.seed)
    cog = Cognition(seed=args.seed, boot_iters=args.boot_iters, sleep_every=args.sleep_every)
    mind.cog = cog
    live_bridge(mind, cog, world, args.steps, args.window)
    metacog_report(cog)
    nar = Narrator()
    txt, _ = nar.say(mind, world.observe())
    print("\n[anlatıcı] " + txt)


if __name__ == "__main__":
    main()
