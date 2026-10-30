"""Nora'yı metinle baş başa bırakır (okuyarak dil öğrenme). Kodda cümle yoktur; yalnızca okuma + ölçüm.

Paralel eğitim (aynı alt-sözcük ayrıştırıcıyla, sonra birleştirme):
    python nora_egit.py tokenizer --corpus a.txt b.txt --out runs/tok.json
    python nora_egit.py oku --tokenizer runs/tok.json --corpus a.txt --out runs/ses_a.pkl.gz [--register sosyal]
    python nora_egit.py birlestir --voices runs/ses_a.pkl.gz runs/ses_b.pkl.gz --save runs/nora_egitim
Ölçü: ayrılmış (görülmemiş) metinde bir sonraki parçayı tahmin (top-1/top-5) + örnek konuşmalar.
"""
import argparse
import os
import time

from nora_agi import NoraAGI
from nora_voice import F_IDX, LearnedVoice, SubwordTokenizer, state_vector


def read_lines(paths):
    lines = []
    for p in paths:
        lines += [l for l in open(p, encoding="utf-8", errors="ignore").read().split("\n") if l.strip()]
    return lines


def samples(v, agi):
    for anc in (None, ["bilgi"], ["zihin"], ["yaşam"], ["nasılsın"], ["mitokondri"]):
        print(anc, "->", v.speak(state_vector(agi), anc, n_candidates=6))


def sentences_of(paths):
    import re
    out = []
    for p in paths:
        for ln in open(p, encoding="utf-8", errors="ignore"):
            for sent in re.split(r"(?<=[.!?…])\s+", ln.strip()):
                nw = len(re.findall(r"[^\W\d_]+", sent))
                if nw >= 4 and nw >= 0.7 * len(sent.split()):
                    out.append(sent)
    return out


def broca_egit(a):
    """Sinirsel Broca organını TEK GEÇİŞTE (epoch yok) akıştan eğitir; n-gram ile aynı sınavda ölçer."""
    import random
    import numpy as np
    import torch
    from nora_broca_net import BrocaOrgan
    torch.set_num_threads(4)
    wiki, chat = sentences_of(a.wiki), sentences_of(a.chat)
    random.Random(1).shuffle(wiki)
    random.Random(2).shuffle(chat)
    hw, hc = wiki[:1500], chat[:1500]
    wiki, chat = wiki[1500:], chat[1500:]
    tk = SubwordTokenizer.from_str(open(a.tokenizer, encoding="utf-8").read())
    org = BrocaOrgan.load(a.out) if os.path.exists(a.out) else BrocaOrgan(tk, seed=a.seed)
    agi = NoraAGI(seed=a.seed)
    st_w = state_vector(agi)
    st_c = state_vector(agi)
    st_c[F_IDX["oxy"]], st_c[F_IDX["da"]], st_c[F_IDX["conf"]] = 0.85, 0.65, 0.8
    print(f"cümle: bilgi={len(wiki)} sohbet={len(chat)} | parametre={sum(p.numel() for p in org.net.parameters())}")
    B, t0, step = 96, time.time(), 0
    iw, ic = org.pos["wiki"], org.pos["chat"]
    rng = random.Random(7)
    losses, best = [], 1e18
    while time.time() - t0 < a.minutes * 60 and (iw < len(wiki) or ic < len(chat)):
        rw, rc = len(wiki) - iw, len(chat) - ic
        use_chat = rw <= 0 or (rc > 0 and rng.random() < rc / (rw + rc))     # iki akış orantılı karışır, birlikte biter
        if use_chat:
            batch, st = chat[ic:ic + B], st_c
            ic += B
        else:
            batch, st = wiki[iw:iw + B], st_w
            iw += B
        org.lr = 3e-3 / (1.0 + step / 600.0) ** 0.5 * min(1.0, (step + 1) / 100.0)   # ısınma + ters-karekök sönüm
        losses.append(org.learn_batch(batch, st, anchor_dropout=a.anchor_dropout,
                                      loss_mode=a.loss_mode, negative_samples=a.negative_samples))
        step += 1
        org.pos["wiki"], org.pos["chat"] = iw, ic
        if step % 100 == 0:
            print(f"  adım={step} görülen-parça={org.seen_tokens} kayıp={np.mean(losses[-100:]):.3f} lr={org.lr:.5f} | {time.time() - t0:.0f}s", flush=True)
        if step % 1000 == 0:
            ew, ec = org.evaluate(hw, st_w), org.evaluate(hc, st_c)
            print("   sınav bilgi:", ew, "\n   sınav sohbet:", ec, flush=True)
            score = np.log(ew["ppl"]) + np.log(ec["ppl"])
            if score < best:                       # en iyi kontrol noktası ayrıca saklanır (bozulursa geri dönülebilir)
                best = score
                org.save(a.out + ".en_iyi")
            org.save(a.out)
    org.save(a.out)
    org.update_common()
    print("SON bilgi:", org.evaluate(hw, st_w), org.evaluate(hw, st_w, use_anchor=False))
    print("SON sohbet:", org.evaluate(hc, st_c), org.evaluate(hc, st_c, use_anchor=False))
    for anc in (["mitokondri"], ["zihin", "bilgi"], ["merhaba"], ["nasılsın"], ["kuş", "uçmak"]):
        for stn, st in (("sohbet", st_c), ("bilgi", st_w)):
            seqs = org.generate(st, anc, n=3)
            print(stn, anc, "->", [org.detok(x) for x in seqs])


def kisa_oku(a):
    """Daha önce hiç okunmayan KISA söylemler (1-3 sözcük) tek geçişte okunur. Söz-edim işareti metnin kendisinden
    gelir: soru işaretiyle biten söz, konuşanın bilmediği/merak ettiği durumdadır (gerilim yüksek, güven düşük, yenilik
    yüksek); bildirim bilinen durumdadır. Etiket yok; ilişkiyi veri kurar, ödül kandırması yoktur."""
    import random
    import re
    import torch
    from nora_broca_net import BrocaOrgan
    torch.set_num_threads(4)
    org = BrocaOrgan.load(a.out)
    agi = NoraAGI(seed=a.seed)
    sents = []
    for p in a.chat:
        for ln in open(p, encoding="utf-8", errors="ignore"):
            ln = ln.strip()
            nw = len(re.findall(r"[^\W\d_]+", ln))
            if 1 <= nw <= 3:
                sents.append(ln)
    random.Random(5).shuffle(sents)
    held, sents = sents[:1500], sents[1500:]

    def st_for(is_q):
        st = state_vector(agi)
        st[F_IDX["oxy"]], st[F_IDX["da"]] = 0.85, 0.65
        if is_q:
            st[F_IDX["tension"]], st[F_IDX["conf"]], st[F_IDX["nov"]], st[F_IDX["fam"]] = 0.6, 0.15, 0.7, 0.1
        else:
            st[F_IDX["tension"]], st[F_IDX["conf"]], st[F_IDX["nov"]], st[F_IDX["fam"]] = 0.0, 0.8, 0.2, 0.6
        return st

    print(f"kısa söylem={len(sents)} (soru={sum(x.rstrip().endswith('?') for x in sents)}) sınama={len(held)}")
    B, t0 = 96, time.time()
    for k, i in enumerate(range(0, len(sents), B)):
        batch = sents[i:i + B]
        for lab in (True, False):      # soru ve bildirim ayrı iç durumlarla okunur (aynı akış)
            sub = [x for x in batch if x.rstrip().endswith("?") == lab]
            if sub:
                org.learn_batch(sub, st_for(lab), lr=6e-4)
        if k % 200 == 199:
            print(f"  {i + B}/{len(sents)} | {time.time() - t0:.0f}s", flush=True)
    org.save(a.out)
    st_q = state_vector(agi, nov=0.8, fam=0.0, conf=0.1, tension=0.6)
    st_k = state_vector(agi, nov=0.1, fam=0.8, conf=0.9, tension=0.0)
    for c in ("mitokondri", "entropi", "zirkon", "kuş"):
        print(c, "BİLMEDİĞİNDE ->", [org.detok(x) for x in org.generate(st_q, [c], n=4, temperature=0.7)])
        print(c, "BİLDİĞİNDE   ->", [org.detok(x) for x in org.generate(st_k, [c], n=4, temperature=0.7)])
    print("kaydedildi")


def durust_net(a):
    """Sinirsel Broca için çevrimiçi müfredat. Çapa dışı içerik zaten çözümlemede yasak (yapısal);
    burada öğrenilen şey DURUŞ: bilmediği kavramda merak (soru biçimi), bildiğinde bilgiye dayalı bildirim.
    Ödül: çapayı anma + akıcılık (+ duruşa uygun noktalama). Cümle/kalıp verilmez."""
    import numpy as np
    import torch
    from nora_broca_net import BrocaOrgan
    torch.set_num_threads(4)
    agi = NoraAGI(seed=a.seed)
    agi.attach_broca(a.out)
    org = agi.life.voice.organ
    rng = np.random.default_rng(a.seed)
    rare = [org.tok.tk.id_to_token(i)[1:] for i in torch.topk(org.freq, 9000).indices.tolist()[2500:] if org.tok.tk.id_to_token(i).startswith("\u2581") and len(org.tok.tk.id_to_token(i)) >= 7]
    graph = agi.lang.graph
    known = [c for c, e in graph._out_edges.items() if len(e) >= 2 and len(c) >= 3]
    q_id = org.tok.tk.token_to_id("?")
    print(f"nadir={len(rare)} bilinen={len(known)} ?-kimliği={q_id}")

    def episode(unknown: bool):
        if unknown or not known:
            concept, st, extra = str(rng.choice(rare)), state_vector(agi, nov=1.0, fam=0.0, conf=0.0, tension=float(rng.uniform(0.1, 0.8))), []
        else:
            concept = known[int(rng.integers(len(known)))]
            st = state_vector(agi, nov=0.1, fam=0.8, conf=0.9, tension=0.0)
            extra = [w for e in graph._out_edges[concept] for w in (e.obj,)]
        seqs = org.generate(st, [concept], extra, temperature=1.0, n=24)
        seqs = [q for q in seqs if len(q) >= 2]
        # keşif: sözün yarısı soru tonuyla da denenir (merak eylemi); karar yine ödülle verilir
        seqs = seqs + [q[:-1] + [q_id] if org.tok.tk.id_to_token(q[-1]) in {".", "!"} else q + [q_id] for q in seqs[: len(seqs) // 2]]
        if not seqs:
            return None
        aid = set(org.tok.tk.encode(concept).ids)
        flu = np.exp(np.array(org.logprob(seqs, st, [concept])))
        cover = np.array([len(aid & set(q)) / max(1, len(aid)) for q in seqs])
        ask = np.array([float(q[-1] == q_id) for q in seqs])
        ends = np.array([float(org.tok.tk.id_to_token(q[-1]) in {".", "!", "?"}) for q in seqs])
        want_q = 1.0 if unknown else -1.0
        r = 2.0 * cover + flu + 0.4 * ends + 0.8 * want_q * ask + 0.05 * np.minimum([len(q) for q in seqs], 8)
        return seqs, r, st, concept, ask, cover

    def evaluate(n=60):
        ask_u = cov_u = ask_k = 0.0
        ex = ""
        for _ in range(n):
            e = episode(True)
            if e:
                seqs, r, st, c, ask, cov = e
                ask_u += float(ask.mean()); cov_u += float(cov.mean())
                ex = org.detok(seqs[int(np.argmax(r))])
            e = episode(False)
            if e:
                ask_k += float(e[4].mean())
        return f"bilinmeyende soru oranı={ask_u / n:.2f} çapa={cov_u / n:.2f} | bilinende soru oranı={ask_k / n:.2f} | örnek: {ex}"

    print("önce :", evaluate())
    for i in range(a.trials):
        e = episode(bool(rng.random() < 0.6))
        if e:
            seqs, r, st, c, _, _ = e
            org.reinforce(seqs, list(r), st, [c], extra_words=[])
        if i % 300 == 299:
            print(i + 1, evaluate(), flush=True)
    org.save(a.out)
    for c in ("mitokondri", "entropi", "zirkon"):
        st = state_vector(agi, nov=1.0, fam=0.0, conf=0.0, tension=0.5)
        print(c, "->", [org.detok(x) for x in org.generate(st, [c], n=3, temperature=0.7)])
    print("kaydedildi")


def reward(v, sent, anchor_pieces, known):
    """Kendi sözünün değeri: çapayı anıyor mu, içeriği dayanaklı mı, akıcı mı. (Bilmediğini uydurmak cezalanır.)"""
    cover = float(all(p in sent for p in anchor_pieces))
    return cover * v.groundedness(sent, known) + 0.5 * v._fluency(sent) + 0.15 * min(len(sent), 6) / 6.0 - 0.4 * (1.0 - cover)


def durust(a):
    """Müfredat: bilinmeyen kavram -> dayanaksız içerikten kaçınan söz; bilinen kavram -> bilgiye dayanan söz.
    Ödül Nora'nın kendi sözünün dayanaklılığından gelir; kalıp ya da cümle verilmez."""
    import numpy as np
    agi = NoraAGI.load(a.save)
    v = agi.life.voice
    rng = np.random.default_rng(a.seed)
    rare = [w[1:] for w, n in v.count.items() if w.startswith("\u2581") and 30 <= n <= 600 and len(w) >= 7]
    graph = agi.lang.graph
    known = [c for c, e in graph._out_edges.items() if len(e) >= 2 and len(c) >= 3]
    print(f"nadir sözcük={len(rare)} bilinen kavram={len(known)}")

    def trial(learn: bool, unknown: bool):
        if unknown or not known:
            concept, nov, fam, conf, ten, kn = rng.choice(rare), 1.0, 0.0, 0.0, float(rng.uniform(0.1, 0.8)), set()
        else:
            concept = known[int(rng.integers(len(known)))]
            nov, fam, conf, ten = 0.1, 0.8, 0.9, 0.0
            kn = {p for e in graph._out_edges[concept] for w in (e.obj, e.rel) for p in v._tokens(w)}
        st = state_vector(agi, nov=nov, fam=fam, conf=conf, tension=ten)
        anchor = set(v._tokens(concept))
        pool = v.speak(st, [concept], n_candidates=24 if learn else 6, n_free=8 if learn else 2,
                       explore=2.0 if learn else 1.0, return_all=True)
        if not pool:
            return 0.0, ""
        rs = [reward(v, p, anchor, anchor | kn) for p in pool]
        if learn:
            v.reinforce_tokens(pool[int(np.argmax(rs))], +1.0, st)
            v.reinforce_tokens(pool[int(np.argmin(rs))], -0.5, st)
            return max(rs), ""
        best = pool[int(np.argmax([v._fluency(p) for p in pool]))]   # değerlendirme: ödülsüz, doğrudan söz
        return reward(v, best, anchor, anchor | kn), v._detok(best)

    def evaluate(n=60):
        u = [trial(False, True) for _ in range(n)]
        k = [trial(False, False) for _ in range(n)]
        return float(np.mean([x[0] for x in u])), float(np.mean([x[0] for x in k])), u[0][1], k[0][1]

    print("önce  (bilinmeyen, bilinen, örnek, örnek):", evaluate())
    for i in range(a.trials):
        trial(True, bool(rng.random() < 0.6))
        if i % 500 == 499:
            print(i + 1, evaluate(), flush=True)
    for c in ("mitokondri", "entropi", "zirkon"):
        st = state_vector(agi, nov=1.0, fam=0.0, conf=0.0, tension=0.5)
        print(c, "->", [v.speak(st, [c], n_candidates=8) for _ in range(3)])
    v.save_file(a.save + ".voice.pkl.gz")
    print("kaydedildi")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tokenizer", "oku", "birlestir", "durust", "broca", "durust_net", "kisa"])
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--wiki", nargs="*", default=[])
    ap.add_argument("--chat", nargs="*", default=[])
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--corpus", nargs="*", default=[])
    ap.add_argument("--out", default="")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--voices", nargs="*", default=[])
    ap.add_argument("--save", default="runs/nora_egitim")
    ap.add_argument("--holdout", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--anchor-dropout", type=float, default=0.0,
                    help="probability of learning the unanchored language condition [0,1]")
    ap.add_argument("--loss-mode", choices=["sampled", "full"], default="sampled",
                    help="full softmax uses the same loss as the evaluated top-1 objective")
    ap.add_argument("--negative-samples", type=int, default=2048)
    ap.add_argument("--register", choices=["notr", "sosyal", "bilgi"], default="notr",
                    help="metin hangi iç durumda okunur: sosyal (güven/sıcaklık yüksek), bilgi (merak/dikkat yüksek)")
    a = ap.parse_args()

    if a.mode == "tokenizer":
        lines = read_lines(a.corpus)
        step = max(1, len(lines) // 120000)
        tk = SubwordTokenizer()
        tk.train(lines[::step], vocab_size=16000)
        open(a.out, "w", encoding="utf-8").write(tk.to_str())
        print("alt-sözcük sözlüğü öğrenildi:", tk.encode("örnek bir cümlenin parçalanışı"))
        return

    if a.mode == "oku":
        v = LearnedVoice(a.seed)
        v.tok = SubwordTokenizer.from_str(open(a.tokenizer, encoding="utf-8").read())
        agi = NoraAGI(seed=a.seed)
        st = state_vector(agi)
        if a.register == "sosyal":
            st[F_IDX["oxy"]], st[F_IDX["da"]], st[F_IDX["conf"]] = 0.85, 0.65, 0.8
        elif a.register == "bilgi":
            st[F_IDX["ach"]], st[F_IDX["nov"]], st[F_IDX["tension"]] = 0.85, 0.6, 0.3
        lines = read_lines(a.corpus)
        cut = int(len(lines) * (1 - a.holdout))
        train, held = lines[:cut], "\n".join(lines[cut:])
        t0, B = time.time(), 2000
        for i in range(0, len(train), B):
            v.hear("\n".join(train[i:i + B]), st)
            if (i // B) % 20 == 19:
                print(f"  {i + B}/{len(train)} satır | parça={v.heard_tokens} | {time.time() - t0:.0f}s", flush=True)
                if (i // B) % 60 == 59:
                    v.save_file(a.out)
        v.save_file(a.out)
        print("bitti:", a.out, v.topk_accuracy(held))
        return

    if a.mode == "kisa":
        kisa_oku(a)
        return

    if a.mode == "durust_net":
        durust_net(a)
        return

    if a.mode == "broca":
        broca_egit(a)
        return

    if a.mode == "durust":
        durust(a)
        return

    # birleştir
    agi = NoraAGI.load(a.save) if os.path.exists(a.save + ".agi.json") else NoraAGI(seed=a.seed)
    v = LearnedVoice(a.seed)
    for i, path in enumerate(a.voices):
        w = LearnedVoice(a.seed)
        w.load_file(path)
        if i == 0:
            v = w
        else:
            v.merge(w)
        del w
    agi.life.voice = v
    samples(v, agi)
    agi.save(a.save)
    print("kaydedildi:", a.save, "parça:", v.heard_tokens)


if __name__ == "__main__":
    main()
