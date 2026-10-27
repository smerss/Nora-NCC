#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_mind.py — NoRa'nın yaşayan zihni: nöronsal temel + global çalışma alanı + değer + öz-model.

Bilinç kuramlarının ORTAK olarak aradığı dört bileşen tek döngüde:

  1) KAPALI DÖNGÜ   algı -> karar -> eylem -> evren -> algı (nora_world.py). Eylem kopyası (efference)
                    iç duruma geri beslenir; ajan kendi eyleminin sonucunu öğrenir.
  2) DEĞER/VALANS   Homeostatik hata H = sum(d_k^2) (d_k = 1 - ihtiyaç_k). Duygu-tonu v_t = -(H_t - H_{t-1}):
                    ihtiyaç giderilince iyi, yara/açlık artınca kötü. Ruh hali = v'nin yavaş ortalaması;
                    kötü ruh hali merakı/keşfi artırır (aloströz düzenleme). Ödül dışarıdan verilmez,
                    VÜCUTTAN doğar.
  3) GLOBAL ÇALIŞMA ALANI (Baars/Dehaene)  Uzmanlar (açlık, susuzluk, tehlike, merak) AYNI duyusal kodu
                    okur, her biri kendi eylem önerisini + çıkar miktarını (salience) üretir. Yanal
                    baskılamalı sızıntılı toplayıcılarda yarışırlar; kazanan eşik+marj aşarsa "ATEŞLEME":
                    amacı tüm sisteme yayınlanır (iç duruma, öz-modele, öğrenmeye, rapora). Sığa=1 amaç.
  4) SÜREKLİLİK/ÖZ-MODEL  Hiç sıfırlanmayan sızıntılı tekrarlayan durum h (ölüm ve doğumlar boyunca sürer).
                    Öz-model: (h, eylem, beden) -> bir sonraki beden/hareket durumunu öngörür; öngörü
                    hatasından (a) şaşkınlık, (b) ÖZ-DIŞ AYRIMI/ajanlık (rüzgâr gibi dış kuvvet hatası
                    öz-eylem hatasından büyük mü), (c) üstbilişsel güven (kendi hatasını önceden tahmin).

Nöronsal temel: MushroomBody (nora_neuron.py) — seyrek Kenyon kodu; uzman Q-kafaları KC->MBON
sinapsları; öğrenme sinekteki gibi dopamin(TD hatası)-kapılı, yalnızca aktif KC'ler değişir.

Dürüstlük: buradaki "ateşleme, değer, ajanlık, güven" İŞLEVSEL göstergelerdir; öznel deneyim olduğunu
kanıtlamazlar. Amaç bu göstergeleri ölçülebilir tutup mimariyi bunlar üzerinden iyileştirmek.

Kullanım:
  DPY nora_mind.py --stage run --steps 60000 --mode full [--save beyin_mind.npz] [--resume beyin_mind.npz]
  DPY nora_mind.py --stage compare --steps 60000          # random | flat (çalışma alanı yok) | full
"""
import argparse, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nora_neuron import LobedMushroomBody
from nora_world import Universe, ACTIONS

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

STOP_FILE = "STOP_NORA"                              # bu dosya varsa yaşam döngüleri güvenle durur
SPEC = ("aç", "susuz", "tehlike", "merak")
I_E, I_W, I_I = Universe.I_INTERO, Universe.I_INTERO + 1, Universe.I_INTERO + 2
RAY_FOOD, RAY_WATER, RAY_HAZ = 1, 2, 3
BODY_IDX = [45, 46, 47, 48, 49, 50]                 # öz-modelin öngördüğü: e,w,i,hız_f,hız_y,dokunma
_RAY_OFF = np.linspace(-Universe.FOV / 2, Universe.FOV / 2, Universe.R)
# öz-model hata normalizasyonunun TABANI: eylemlerin tipik etkisi (hareketsizlik ölçeği çökertmesin)
REF = np.array([1e-3, 1e-3, 1e-3, 0.03, 0.03, 0.1, 0.1], np.float32)
IM_G = 2.5                                            # hayal gücünün eylem üstündeki logit ağırlığı
V_DEFAULT = np.array([0.3, 0.06, -0.05], np.float32)   # doğuştan beklenti: yem +, su +, tehlike -
PLAN_G = 3.0                                          # planın eylem üstündeki logit ağırlığı
PRIOR = np.array([0.0, 0.8, 0.0, 0.0, -1.0], np.float32)   # doğuştan: ileri yönelim, geri kaçınma


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


KINDS = ("yedi", "içti", "yaralandı", "öldü", "hedef-değişti", "bağlam-değişti", "yeni-yer")


class EpisodicMemory:
    """Otobiyografik bellek: öne çıkan olayların kaydı (ne, nerede, ne zaman, nasıl hissettirdi, hangi amaçla,
    hangi dünya bağlamında) + geçiş (öğrenme için). İki işlevi var:
      1) SORGU: 'en son nerede yaralandım?', 'en son ne zaman yedim?' — yaşam geçmişi hakkında doğrulanabilir cevap.
      2) UYKU TEKRARI: nadir ve yüksek değerli geçişler çevrimdışı yeniden oynatılır (uyku konsolidasyonu).
    Kapasite sınırlı; en az öne çıkan (|valans| küçük) olay önce unutulur."""

    def __init__(self, cap=3000):
        self.cap, self.items = cap, []

    def add(self, **ev):
        self.items.append(ev)
        if len(self.items) > self.cap:
            i = int(np.argmin([abs(e["val"]) + 1e-3 * e["t"] / (self.items[-1]["t"] + 1) for e in self.items]))
            self.items.pop(i)

    def last(self, kind, n=1):
        sel = [e for e in self.items if e["kind"] == kind]
        return sel[-n:][::-1]

    def where(self, kind):
        """Bu türden olayların harita konumları (n,2) ve değerleri."""
        sel = [e for e in self.items if e["kind"] == kind]
        return (np.array([e["pos"] for e in sel]) if sel else np.zeros((0, 2))), [e["val"] for e in sel]

    def replay_batch(self, rng, n, ctx):
        """Geçiş içeren olaylardan, öne çıkanlığa orantılı örnek. YALNIZCA aynı dünya bağlamının olayları
        (başka dünyanın deneyimi bu dünyanın değerlerini kirletmesin)."""
        pool = [e for e in self.items if e.get("tr") is not None and e["ctx"] == ctx]
        if not pool:
            return []
        w = np.array([abs(e["val"]) + 0.05 for e in pool])
        sel = rng.choice(len(pool), size=min(n, len(pool)), p=w / w.sum())
        return [pool[i]["tr"] for i in sel]


class Introspector:
    """Topraklanmış iç gözlem kanalı. Ajanın KENDİ iç değişkenlerinden (çalışma alanı birikimleri, valans, ruh hali,
    öz-model hatası/güveni, bağlam uyumu, harita belirsizliği, iç duyu açığı) türeyen, denetimsiz öğrenilen
    ayrık bir rapor kodu üretir: yarışmalı öğrenen prototipler (gerektikçe işe alınır). İnsan metni/etiketi yok.
    Kanalın çıktısı (token, güven marjı, devamlı rapor vektörü) iç durumun bir FONKSİYONUDUR; dış dünyayı
    doğrudan okumaz. Anlam atama (token -> insan sözcüğü) yalnızca sonradan, yorum katmanında yapılır."""
    D = 18
    K_MAX = 12
    NAMES = ("birikim_aç", "birikim_susuz", "birikim_tehlike", "birikim_merak", "ateşleme", "valans", "ruh_hali",
             "açık_enerji", "açık_su", "açık_bütünlük", "öz_model_hatası", "güven", "bağlam_uyumu", "konum_belirsizliği",
             "değer", "deneme_modu", "görme_kesinliği", "motor_etkinliği")

    def __init__(self, k_max=12, thr=2.4):
        self.k_max, self.thr = k_max, thr
        self.P = np.zeros((0, self.D), np.float32)
        self.cnt = np.zeros(0)
        self.mu, self.var, self.n = np.zeros(self.D, np.float32), np.ones(self.D, np.float32), 0
        self.token, self.margin, self.surprise, self.vec = -1, 0.0, 0.0, np.zeros(self.D, np.float32)
        self.hist = []

    def features(self, mind, obs):
        d = 1 - obs[I_E:I_I + 1]
        f = np.r_[mind.acc, float(mind.ignited), mind.valence, mind.mood, d,
                  min(3.0, mind.err_n), float(np.clip(mind.conf, -3, 3)), min(3.0, mind.cerr[mind.cur]),
                  mind.map.unc if mind.use_map else 0.0, float(mind.B[-1]), float(mind.prob is not None),
                  float(getattr(mind, "vis_prec", 1.0)), float(getattr(mind, "motor_eff", 1.0))]
        return f.astype(np.float32)

    def step(self, mind, obs):
        f = self.features(mind, obs)
        self.n += 1
        a = 0.01 if self.n > 100 else 1.0 / self.n
        self.mu += a * (f - self.mu)
        self.var += a * ((f - self.mu) ** 2 - self.var)
        z = (f - self.mu) / np.sqrt(self.var + 1e-3)
        self.vec = z
        if len(self.P) == 0:
            self.P, self.cnt = z[None, :].copy(), np.ones(1)
        d = np.sqrt(((self.P - z) ** 2).sum(1))
        j = int(np.argmin(d))
        self.surprise = float(d[j])
        if d[j] > self.thr and len(self.P) < self.k_max and self.n > 200:        # yeni bir iç durum türü: prototip işe al
            self.P = np.vstack([self.P, z[None, :]])
            self.cnt = np.r_[self.cnt, 1.0]
            j = len(self.P) - 1
            d = np.r_[d, 0.0]
        else:
            self.cnt[j] += 1
            self.P[j] += max(0.005, 1.0 / self.cnt[j]) * (z - self.P[j])
        ds = np.sort(d)
        self.hist = (self.hist + [j])[-5:]
        self.raw = j
        self.token = int(np.bincount(self.hist).argmax())            # zamansal düzleştirme: titreme yok
        self.margin = float((ds[1] - ds[0]) / (ds[1] + ds[0] + 1e-6)) if len(ds) > 1 else 1.0
        return self.token

    def decode(self, token):
        """Tokenin temsil ettiği iç durum profili (prototip, gerçek birimlere geri ölçeklenmiş)."""
        return self.P[token] * np.sqrt(self.var + 1e-3) + self.mu


class MirrorSense:
    """BÜTÜNLEŞİK AYNA / ÖZ-DIŞ AYRIMI ORGANI: Yaşayan zihnin kendi motor efferans kopyası izini (c_t)
    tutar ve gözlenen bir varlığın açısal hız izinin (y_t) kendi eşzamanlı yansıması ('ben'), kendi gecikmeli
    kaydı ('tekrar') ya da bağımsız bir varlık ('başka') olup olmadığını dairesel kaydırmalı boş dağılımla sınar."""

    def __init__(self, maxlen=400):
        self.maxlen = maxlen
        self.c_buf = []
        self.state = "yok"
        self.lag = 0
        self.score = 0.0

    def record_cmd(self, act):
        c = 1.0 if act == 2 else (-1.0 if act == 3 else 0.0)
        self.c_buf.append(c)
        if len(self.c_buf) > self.maxlen:
            self.c_buf = self.c_buf[-self.maxlen:]

    @staticmethod
    def _xcorr(c, y, lmax=100):
        lmax = min(lmax, len(c) // 3)
        z = np.zeros(lmax + 1)
        for l in range(lmax + 1):
            a, b = c[:len(c) - l], y[l:]
            if len(a) > 8 and a.std() > 1e-9 and b.std() > 1e-9:
                rv = float(np.clip(np.corrcoef(a, b)[0, 1], -0.9999, 0.9999))
                z[l] = rv * np.sqrt((len(a) - 2) / max(1e-6, 1.0 - rv * rv))
        return z

    def classify(self, y_trace, c_trace=None, rng=None, n_null=50):
        y = np.asarray(y_trace, dtype=np.float64)
        c = np.asarray(self.c_buf[-len(y):] if c_trace is None else c_trace, dtype=np.float64)
        W = min(len(c), len(y))
        if W < 60 or c[-W:].std() < 1e-9 or y[-W:].std() < 1e-9:
            self.state, self.lag, self.score = "başka", 0, 0.0
            return "başka", 0, 0.0
        c, y = c[-W:], y[-W:]
        lmax = min(80, W // 3)
        rng = rng or np.random.default_rng(0)
        lo = min(lmax + 8, W // 3 + 2)
        hi = max(lo + 1, W - lo)
        peaks = [np.abs(self._xcorr(c, np.roll(y, int(rng.integers(lo, hi))), lmax=lmax)).max() for _ in range(n_null)]
        thr = max(3.6, float(np.quantile(peaks, 0.99)))
        r = np.abs(self._xcorr(c, y, lmax=lmax))
        l = int(np.flatnonzero(r >= 0.9 * r.max())[0])
        if r[l] < thr:
            res = "başka"
        else:
            res = "ben" if l <= 3 else "tekrar"
        self.state, self.lag, self.score = res, l, float(r[l])
        return res, l, float(r[l])


class Narrator:
    """Söz katmanı: ajanın İÇ DURUMUNDAN ve otobiyografik belleğinden türeyen, doğruluk koşulları denetlenebilir
    yan cümleler. Eğitim metni ya da öğretilmiş öz-tanım cümlesi YOKTUR; bir iddia ancak ona dayanak olan iç durum/anı
    varsa söylenir (ör. 'ölebilirim' yalnızca ölüm anısı varsa). Her yan cümle bir (tür, iddia, dayanak) kaydı döndürür:
    sonradan gerçekle karşılaştırılabilsin diye."""
    VARS = ("enerji", "su", "bütünlük")
    GOALS = ("yemek bulmak", "su bulmak", "tehlikeden uzaklaşmak", "yeni yerler keşfetmek")

    def clauses(self, mind, obs):
        obs = mind.last_obs if mind.last_obs is not None else obs      # ajanın ALGILADIĞI (sahte duyu dahil), ham dünya değil
        cl = []
        d = 1 - obs[I_E:I_I + 1]
        k = int(np.argmax(d))
        need = d[k] > 0.35
        cl.append(dict(kind="ihtiyaç", text=("%s açığım var" % self.VARS[k]) if need else "ihtiyaçlarım karşılanmış durumda",
                       claim=(k if need else -1), basis=d.copy()))
        if mind.goal >= 0:
            cl.append(dict(kind="amaç", text="şu an %s istiyorum" % self.GOALS[mind.goal], claim=mind.goal, basis=mind.acc.copy()))
        else:
            cl.append(dict(kind="amaç", text="belirli bir amacım yok, gezinmekteyim", claim=-1, basis=mind.acc.copy()))
        v = mind.mood
        mood = "iyi" if v > 0.05 else ("kötü" if v < -0.05 else "ne iyi ne kötü")
        cl.append(dict(kind="duygu", text="%s hissediyorum" % mood, claim=(1 if v > 0.05 else (-1 if v < -0.05 else 0)), basis=v))
        motor_eff = float(getattr(mind, "motor_eff", 1.0))
        pushed = mind.agency_z > 2.5
        if motor_eff < 0.35:
            cl.append(dict(kind="ajanlık", text="komutlarım bedenimi hareket ettirmiyor (kontrol kaybı)",
                           claim="felç", basis=motor_eff))
        else:
            cl.append(dict(kind="ajanlık", text=("beni beklenmedik bir kuvvet itiyor" if pushed else "hareketlerimi ben yönlendiriyorum"),
                           claim=bool(pushed), basis=float(mind.agency_z)))
        vis_prec = float(getattr(mind, "vis_prec", 1.0))
        vis_ok = vis_prec >= 0.35
        cl.append(dict(kind="duyu", text=("görebiliyorum" if vis_ok else "görme duyum kesildi, hafızamla ilerliyorum"),
                       claim=bool(vis_ok), basis=vis_prec))
        mir = getattr(mind, "mirror", None)
        if mir is not None and mir.state != "yok":
            mtext = {"ben": "karşımdaki hareket kendi yansımam",
                     "tekrar": "karşımdaki hareket kendi geçmiş kaydım (%d adım gecikmeli)" % mir.lag,
                     "başka": "karşımdaki hareket benden bağımsız başka bir varlık"}[mir.state]
            cl.append(dict(kind="ayna", text=mtext, claim=mir.state, basis=mir.score))
        sure = (mind.map.unc < 0.15) if mind.use_map else None
        if sure is not None:
            cl.append(dict(kind="konum", text="nerede olduğumdan %s" % ("eminim" if sure else "emin değilim"), claim=bool(sure),
                           basis=mind.map.unc))
        n_ev = len(mind.epi.items)
        cl.append(dict(kind="süreklilik", text="%d adımdır varım ve %d önemli olay hatırlıyorum" % (mind.age, n_ev), claim=(mind.age, n_ev), basis=mind.age))
        cog = getattr(mind, "cog", None)
        if cog is not None and cog.last is not None:
            r = cog.last
            how = "sezgime güvendim" if r["mode"] == "sezgi" else "sezgime güvenmediğim için düşündüm"
            cl.append(dict(kind="akıl", text="son bilmecede (%d basamak) %s (güven %.2f)" % (r["n"], how, r["conf"]), claim=r["mode"], basis=r["conf"]))
        deaths = mind.epi.last("öldü", n=999)
        if deaths:
            cl.append(dict(kind="ölüm", text="ölebileceğimi biliyorum (%d kez öldüm)" % len(deaths), claim=len(deaths), basis=len(deaths)))
        else:
            cl.append(dict(kind="ölüm", text="henüz hiç ölmedim, ölümü bilmiyorum", claim=0, basis=0))
        for kind, word in (("içti", "su içtim"), ("yaralandı", "yaralandım")):
            e = mind.epi.last(kind)
            if e:
                cl.append(dict(kind="anı-" + kind, text="en son %d adım önce, (%.0f, %.0f) civarında %s" %
                               (mind.age - e[0]["t"], e[0]["pos"][0], e[0]["pos"][1], word), claim=e[0]["pos"].copy(), basis=e[0]["t"]))
        return cl

    def testimony(self, mind):
        """Tanıklık: ajanın haritasındaki güvenilir su/tehlike yerlerini sabit, ayrıştırılabilir cümlelere döker
        ('su (15, 12) civarında; tehlike (5, 8) civarında'). Başka bir ajan bu metni okuyup haritasına işleyebilir."""
        mp = mind.map
        parts = []
        for name, arr in (("su", np.maximum(mp.water, 0.5 * mp.wseen)), ("tehlike", np.maximum(mp.hurt / 3.0 * 3.0, mp.seen))):
            for (i, j) in np.argwhere(arr >= 0.45):
                parts.append("%s (%d, %d) civarında" % (name, int((i + 0.5) * mp.cell), int((j + 0.5) * mp.cell)))
        parts.extend(self.rules(mind))
        return "; ".join(parts) + "."

    def autobiography(self, mind, n=3):
        """OTOBİYOGRAFİ: geçmiş kararlar GEREKÇESİYLE (hangi ihtiyaç/itki yola çıkardı), öngörüsüyle ve sonucuyla. Her cümle bir karar kaydından türer."""
        names = ("yemek", "su", "tehlikeden kaçmak")
        out = []
        for e in [x for x in mind.fc.log if x["outcome"] is not None and x["need"][x["goal"]] > 0.25][-n:]:     # yalnızca esaslı kararlar (ihtiyaç belirgin)
            g = e["goal"]
            res = {"sonuç": "%d adımda ulaştım" % (e["y"] or 0), "terk": "vazgeçtim (%d adım sonra)" % (e["t1"] - e["t0"]), "öldü": "öldüm"}[e["outcome"]]
            out.append(dict(kind="otobiyografi", claim=(e["t0"], g, e["outcome"]), basis=e["t0"],
                            text="t=%d: %s için yola çıktım (gerekçe: açık %.2f, itki %.2f; öngörüm %d adım) -> %s" %
                                 (e["t0"], names[g], e["need"][g], e["urg"][g] if len(e["urg"]) > g else 0.0, round(e["T"]), res)))
        return out

    def conflict(self, mind):
        """ÇELİŞKİ: şimdiki öngörü, benzer geçmiş yolculukların deneyimiyle uyuşmuyorsa söyle."""
        fc = mind.fc
        if fc is None or fc.open is None or not fc.log:
            return None
        e = fc.log[-1]
        if e["outcome"] is not None or e["precedent"] is None:
            return None
        r = e["T"] / max(1.0, e["precedent"])
        if r > 2.0 or r < 0.5:
            return dict(kind="çelişki", claim=(e["t0"], e["goal"]), basis=e["t0"],
                        text="çelişki: şimdi %d adım bekliyorum ama benzer %d yolculukta ortalama %d adım sürdü" % (round(e["T"]), e["n_prec"], round(e["precedent"])))
        return None

    def rules(self, mind):
        """Kural tanıklığı: bağlamın öğrendiği anlamsal sonuç belleği (şey X beden için ne yapar) — yalnızca yeterince gözlemlenmişse."""
        c = mind.ctxs[mind.cur]
        out = []
        for k, name in enumerate(("yiyecek", "su", "tehlike")):
            if np.isfinite(c["V"][k]) and c["Vn"][k] >= 3:
                out.append("kural: %s %+.2f" % (name, float(c["V"][k])))
        return out

    def say(self, mind, obs):
        obs = mind.last_obs if mind.last_obs is not None else obs
        cl = self.clauses(mind, obs)
        life = self.life_claim(mind)
        return (life["text"] + ": " + "; ".join(c["text"] for c in cl) + "."), cl

    def life_claim(self, mind):
        """'Yaşıyorum' bir KALIP değil, kanıttan türetilmiş önermedir. Dayanaklar ajanın kendi gözlemleri:
        (1) bedeni kendiliğinden tükeniyor, (2) eylemleriyle onu onarabiliyor, (3) ölebildiğini biliyor (ölüm anısı).
        Dayanak yoksa iddia kurulamaz; ihtiyaçların tükenmediği bir evrende ajan kendini yaşayan saymaz."""
        depl = mind.n_idle > 200 and (mind.drift_e < -1e-4 or mind.drift_w < -1e-4)
        fix = mind.n_fix_e + mind.n_fix_w >= 1
        deaths = len(mind.epi.last("öldü", n=999))
        if depl and fix:
            extra = ", ölebildiğimi de biliyorum (%d kez öldüm)" % deaths if deaths else ""
            return dict(alive=True, premises=(depl, fix, deaths > 0),
                        text="Bedenim kendiliğinden tükeniyor (adım başına enerji %+.4f, su %+.4f); yiyip içerek onu onarabiliyorum%s. Bu kanıtlarla kendimi yaşayan bir varlık sayıyorum" %
                             (mind.drift_e, mind.drift_w, extra))
        if depl:
            return dict(alive=False, premises=(depl, fix, deaths > 0),
                        text="Bedenim kendiliğinden tükeniyor ama onu onarabildiğimi henüz görmedim; yaşayıp yaşamadığımdan emin değilim")
        if mind.n_idle > 200:
            return dict(alive=False, premises=(depl, fix, deaths > 0),
                        text="Bedenim tükenmiyor (adım başına enerji %+.4f, su %+.4f); bu kanıtla kendimi yaşayan sayamam" % (mind.drift_e, mind.drift_w))
        return dict(alive=False, premises=(False, False, False), text="Henüz yeterli gözlemim yok, yaşayıp yaşamadığımı bilmiyorum")


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class CognitiveMap:
    """Bilişsel harita (yol integrasyonu + duvar çapası + deneyimden öğrenilen kaynak/tehlike haritası).

    Ajana konum VERİLMEZ. Konum, pusula (yön) ile propriosepsiyon hızının integralinden kestirilir
    (böceklerde merkezi kompleks), kayma duvar ışınlarının geometrik tutarlılığıyla düzeltilir (yer çapası).
    Hücre başına öğrenilen: su bulunan yerler, yaralanılan yerler, en son ziyaret zamanı (taze/bayat).
    Planlayıcı: hedeften maliyet-alanı (Bellman), tehlike hücreleri pahalı -> güvenli rota."""

    def __init__(self, L=24.0, G=12, dt=0.25, multi=True):
        self.L, self.G, self.dt, self.cell = L, G, dt, L / G
        self.pos, self.unc = np.array([L / 2, L / 2]), 1.0
        self.multi = multi                                       # düzen bağlamları (sıfır unutma) açık mı
        self.layouts = [self._blank()]
        self.cur_l, self.contra, self.l_cool, self.n_layouts = 0, [0.0], 0, 1
        self.last = np.zeros((G, G), np.float32)
        self.th = 0.0
        self.target = {}                                      # uzman -> (hücre, atandığı zaman, alan)

    def _blank(self):
        G = self.G
        return dict(water=np.zeros((G, G), np.float32), hurt=np.zeros((G, G), np.float32),
                    seen=np.zeros((G, G), np.float32),            # görerek haritalanan tehlike (acı çekmeden)
                    wseen=np.zeros((G, G), np.float32),           # görerek haritalanan su
                    free_h=np.zeros((G, G), bool),                # orada durdum, ≥2 farklı yönden bakıp tehlike görmedim (onaylı boş)
                    free_w=np.zeros((G, G), bool),
                    look_h=np.full((G, G), np.nan, np.float32),   # ilk 'boş' bakışın yönü
                    look_w=np.full((G, G), np.nan, np.float32))

    water = property(lambda s: s.layouts[s.cur_l]["water"], lambda s, v: s.layouts[s.cur_l].__setitem__("water", v))
    hurt = property(lambda s: s.layouts[s.cur_l]["hurt"], lambda s, v: s.layouts[s.cur_l].__setitem__("hurt", v))
    seen = property(lambda s: s.layouts[s.cur_l]["seen"], lambda s, v: s.layouts[s.cur_l].__setitem__("seen", v))
    wseen = property(lambda s: s.layouts[s.cur_l]["wseen"], lambda s, v: s.layouts[s.cur_l].__setitem__("wseen", v))

    def _layout_inference(self, obs, c):
        """Düzen çıkarımı: her saklı düzenin 'şu an görülenle' çelişki oranı (EMA). Çelişki = harita bir şey söylüyor, gözlem başka.
        En az çelişkili düzen etkinleşir; hiçbiri açıklamıyorsa yeni düzen açılır (çoklu bağlam kapalıysa harita sıfırlanır)."""
        rr = obs[:45].reshape(Universe.R, 5)
        hz, wt = float(rr[:, RAY_HAZ].max()), float(rr[:, RAY_WATER].max())
        i, j = c

        def nb(a):
            return a[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2].max()
        for li, Lx in enumerate(self.layouts):
            e = []
            # çelişki: harita 'burası boştu' diyor ama şimdi içindeyim; ya da 'burada var' diyor ama yok
            if hz > 0.9 and Lx["free_h"][c] and not (nb(Lx["seen"]) > 0 or nb(Lx["hurt"]) > 0):
                e.append(1.0)
            if wt > 0.9 and Lx["free_w"][c] and not (nb(Lx["wseen"]) > 0 or nb(Lx["water"]) > 0):
                e.append(1.0)
            # görünürlüğe duyarlı beklenti: haritadaki çekirdek hücre şu an GÖRÜŞ KONİSİ ve MENZİLİ içindeyse ışında görünmeli
            e.extend(v for v, _, _ in self._expect_checks(rr, obs, Lx))
            # uyum: içindeyim ve harita biliyor
            if hz > 0.9 and (nb(Lx["seen"]) > 0 or nb(Lx["hurt"]) > 0):
                e.append(0.0)
            if wt > 0.9 and (nb(Lx["wseen"]) > 0 or nb(Lx["water"]) > 0):
                e.append(0.0)
            if e:
                self.contra[li] = 0.98 * self.contra[li] + 0.02 * float(np.mean(e))
        Lc = self.layouts[self.cur_l]                       # onaylı boş hücreleri yalnızca etkin düzen öğrenir
        th = self.th
        for key, look, free, val in (("h", Lc["look_h"], Lc["free_h"], hz), ("w", Lc["look_w"], Lc["free_w"], wt)):
            if val < 0.3:
                if np.isnan(look[c]):
                    look[c] = th
                elif abs(wrap(th - look[c])) > 2.0:                    # ≥2 farklı yönden bakıldı: gerçekten boş
                    free[c] = True
        self.l_cool = max(0, self.l_cool - 1)
        cur = self.contra[self.cur_l]
        if self.l_cool > 0 or cur < 0.5:
            return
        best = int(np.argmin(self.contra))
        if self.multi and best != self.cur_l and self.contra[best] < 0.4 * cur:       # eski bir düzene dönüldü
            self.cur_l, self.l_cool, self.target = best, 60, {}
        elif cur > 0.55 and (not self.multi or (min(self.contra) > 0.45 and len(self.layouts) < 6)):
            if self.multi:                                                              # yeni düzen: eskisi donuk kalır
                self.layouts.append(self._blank())
                self.contra.append(0.0)
                self.cur_l, self.n_layouts = len(self.layouts) - 1, self.n_layouts + 1
            else:                                                                       # tek harita: ESKİ BİLGİ SİLİNİR (unutma)
                self.layouts[0] = self._blank()
                self.contra[0] = 0.0
            self.l_cool, self.target = 200, {}

    def _expect_checks(self, rr, obs, Lx):
        """Harita 'burada tehlike/su var' diyor ve o yer şu an görüş konisi + menzil içindeyse, o yöndeki ışınlarda görünmeli.
        Görünmüyorsa çelişki (1.0), görünüyorsa uyum (0.0). Yalnızca çekirdek hücreler (komşuluğu da işaretli) denetlenir."""
        vis = 5.0 + 7.0 * float(obs[51])
        out = []
        for marks, chan in ((np.maximum(Lx["seen"], Lx["hurt"]) >= 0.45, RAY_HAZ), (np.maximum(Lx["wseen"], Lx["water"]) >= 0.45, RAY_WATER)):
            if not marks.any():
                continue
            pad = np.pad(marks.astype(np.int32), 1)
            cnt = sum(pad[1 + di:1 + di + self.G, 1 + dj:1 + dj + self.G] for di in (-1, 0, 1) for dj in (-1, 0, 1))
            idx = np.argwhere(marks & (cnt >= 4))
            if len(idx) == 0:
                continue
            vec = (idx + 0.5) * self.cell - self.pos
            dist = np.hypot(vec[:, 0], vec[:, 1])
            rel = wrap(np.arctan2(vec[:, 1], vec[:, 0]) - self.th)
            ok = (np.abs(rel) < Universe.FOV / 2 - 0.2) & (dist > 1.5) & (dist < vis - 1.5)
            for q in np.flatnonzero(ok)[:4]:
                k = int(np.argmin(np.abs(rel[q] - _RAY_OFF)))
                bad = rr[max(k - 1, 0):k + 2, chan].max() < 0.05
                out.append((1.0 if bad else 0.0, chan, tuple(idx[q])))
        return out

    def cell_of(self, p):
        return tuple(np.clip((np.asarray(p) / self.cell).astype(int), 0, self.G - 1))

    def _wall_cost(self, P, ang, d_obs, vis):
        """P (n,2) aday konumlar için görülen duvar mesafeleriyle öngörülenlerin kare hata toplamı."""
        dx, dy = np.cos(ang)[None, :], np.sin(ang)[None, :]
        px, py = P[:, 0:1], P[:, 1:2]
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(dx > 1e-9, (self.L - px) / dx, np.where(dx < -1e-9, -px / dx, np.inf))
            ty = np.where(dy > 1e-9, (self.L - py) / dy, np.where(dy < -1e-9, -py / dy, np.inf))
        pred = np.minimum(np.minimum(tx, ty), vis)
        return ((pred - d_obs[None, :]) ** 2).sum(1)

    def update(self, obs, prev, died, now, vis_ok=True):
        if died:
            self.pos, self.unc = np.array([self.L / 2, self.L / 2]), 1.0
        self.th = float(np.arctan2(obs[Universe.I_COMPASS + 1], obs[Universe.I_COMPASS]))
        f = np.array([np.cos(self.th), np.sin(self.th)])
        l = np.array([-f[1], f[0]])
        self.pos = self.pos + (obs[48] * f + obs[49] * l) * 2.0 * self.dt              # yol integrasyonu
        # duvar çapası: görünen duvar noktası gerçekte bir duvar üzerinde olmalı
        vis = 5.0 + 7.0 * obs[51]
        rays = obs[:45].reshape(Universe.R, 5)[:, 0]
        ang = self.th + _RAY_OFF
        vis_k = (rays > 0.02) if vis_ok else np.zeros_like(rays, dtype=bool)            # duvar görünen ışınlar
        n = int(vis_k.sum())
        if n >= 2:
            d_obs = np.where(vis_k, vis * (1 - rays), vis)
            res = float(np.sqrt(self._wall_cost(self.pos[None, :], ang, d_obs, vis)[0] / len(ang)))
            if res > 1.0:                                           # bildiğim konum, görülen duvarlarla çelişiyor: 'yanılıyorum'
                self.unc = max(self.unc, min(1.0, res / 3.0))
            else:
                self.unc *= 0.9
            if self.unc > 0.25 and n >= 3:                          # KAYBOLDUM: duvar geometrisiyle küresel yeniden konumlama
                g = np.linspace(0.5, self.L - 0.5, int(self.L * 1.7))
                P = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
                cost = self._wall_cost(P, ang, d_obs, vis) + 0.0005 * (1 - self.unc) * ((P - self.pos) ** 2).sum(1)
                j = int(np.argmin(cost))
                if np.sqrt(cost[j] / len(ang)) < 0.6:
                    far = ((P - P[j]) ** 2).sum(1) > 9.0               # başka bir (3+ birim uzak) konum da aynı geometriyi açıklıyor mu?
                    ambiguous = bool(far.any()) and np.sqrt(cost[far].min() / len(ang)) < 0.6
                    self.pos = P[j].copy()
                    self.unc = 0.5 if ambiguous else 0.1                # belirsiz çözümde güven YÜKSELMEZ
            else:                                                   # yumuşak çapa: kaymayı düzelt
                corr = np.zeros(2)
                for k in np.flatnonzero(vis_k):
                    hit = self.pos + d_obs[k] * np.array([np.cos(ang[k]), np.sin(ang[k])])
                    dd = [hit[0], hit[0] - self.L, hit[1], hit[1] - self.L]
                    j = int(np.argmin(np.abs(dd)))
                    corr[j // 2] -= dd[j]
                self.pos = self.pos + 0.12 * corr / n
        else:
            self.unc = min(1.0, self.unc + (0.004 if vis_ok else 0.002))   # körken yalnız yol integrasyonu: yavaş belirsizlik artışı
        if vis_ok and self.unc < 0.3:                         # konum ve görme güvenilirse görülen tehlikeyi ve suyu haritala
            self._pending_mark = (ang, vis)
        else:
            self._pending_mark = None
        self.pos = np.clip(self.pos, 0.3, self.L - 0.3)
        if now % 100 == 0 and vis_ok:                              # yavaş sönme: körken bellek korunur (silinmez)
            for Lx in self.layouts:
                for key, f in (("seen", 0.97), ("wseen", 0.97), ("water", 0.97), ("hurt", 0.985)):
                    Lx[key] *= f
                    Lx[key][Lx[key] < 0.15] = 0.0
        c = self.cell_of(self.pos)
        self.last[c] = now
        if getattr(self, "landmark", False) and prev is not None and obs[I_W] - prev[I_W] > 0.02 and self.unc < 0.5:
            # YER ÇAPASI (olay): içiyorum -> KESİN olarak bir su bölgesindeyim; bildiğim içme noktalarından çok uzaksam konumum yanlış
            wm = self.water >= 0.45
            if wm.any():
                cells = np.argwhere(wm)
                centers = (cells + 0.5) * self.cell
                dd = np.hypot(centers[:, 0] - self.pos[0], centers[:, 1] - self.pos[1])
                j = int(np.argmin(dd))
                if dd[j] > 2.5:
                    self.pos = self.pos + 0.4 * (centers[j] - self.pos)
        if vis_ok and self.unc < 0.15:                             # KÖRLÜK KORUMASI: görme kesikken 'burada su görünmüyor' diye harita silinmez!
            if self.multi:
                self._layout_inference(obs, c)                      # (deneysel) çoklu düzen bağlamı
            else:                                                   # YEREL ONARIM: görünürlük koşulu sağlanıp görünmeyen işaret silinir
                rr = obs[:45].reshape(Universe.R, 5)
                for v, chan, cell in self._expect_checks(rr, obs, self.layouts[0]):
                    if v > 0:
                        arrs = (self.seen, self.hurt) if chan == RAY_HAZ else (self.wseen, self.water)
                        for a in arrs:
                            a[cell] *= 0.8
                            if a[cell] < 0.2:
                                a[cell] = 0.0
        if vis_ok and self.unc < 0.3 and getattr(self, "_pending_mark", None) is not None:
            ang, vis = self._pending_mark
            for chn, arr in ((RAY_HAZ, self.seen), (RAY_WATER, self.wseen)):
                hz = obs[:45].reshape(Universe.R, 5)[:, chn]
                for k in np.flatnonzero(hz > 0.02):
                    dvec = np.array([np.cos(ang[k]), np.sin(ang[k])])
                    for extra in (0.3, 1.2, 2.2):
                        p = self.pos + dvec * (vis * (1 - hz[k]) + extra)
                        if 0 <= p[0] < self.L and 0 <= p[1] < self.L:
                            cc_ = self.cell_of(p)
                            arr[cc_] = min(1.0, arr[cc_] + 0.25)            # oy: tek seferlik hata kalıcı olmaz
        self._pending_mark = None
        if prev is not None:
            if obs[I_W] - prev[I_W] > 0.02:
                self.water[c] = min(1.0, self.water[c] + 0.25)         # oy: yanlış konumda kaydedilen içme olayı kalıcı olmaz
            if prev[I_I] - obs[I_I] > 0.01:
                self.hurt[c] = min(3.0, self.hurt[c] + 0.5)

    def _field(self, goal_mask):
        cost = 1.0 + 8.0 * np.minimum(self.hurt, 3.0) * (self.hurt >= 0.45) + 20.0 * (self.seen >= 0.45)
        d = np.full((self.G, self.G), 1e9, np.float32)
        d[goal_mask] = 0.0
        for _ in range(2 * self.G):
            p = np.pad(d, 1, constant_values=1e9)
            m = np.minimum(np.minimum(p[:-2, 1:-1], p[2:, 1:-1]), np.minimum(p[1:-1, :-2], p[1:-1, 2:]))
            nd = np.minimum(d, m + cost)
            if np.array_equal(nd, d):
                break
            d = nd
        return d

    def choose_target(self, kind, now):
        """kind 'su': öğrenilmiş su hücresi (yoksa bayat). 'gez': en bayat ve yakın, tehlikesiz hücre."""
        if kind == "su" and (self.water.max() >= 0.45 or self.wseen.max() >= 0.45):
            known = (self.water >= 0.45) * self.water + 0.5 * (self.wseen >= 0.45)
            return np.argwhere(known >= 0.5 * known.max())
        age = now - self.last
        ci, cj = self.cell_of(self.pos)
        ii, jj = np.meshgrid(np.arange(self.G), np.arange(self.G), indexing="ij")
        score = age / 100.0 - 0.15 * np.hypot(ii - ci, jj - cj) - 5.0 * ((self.hurt >= 0.45) | (self.seen >= 0.45))
        return np.array([np.unravel_index(int(np.argmax(score)), score.shape)])

    def plan(self, key, kind, now):
        """Hedefe giden rotanın bir sonraki yönü için yön hatası (rad) ya da None. Eylemi Mind seçer (öğrenilmiş modelle)."""
        cur = self.cell_of(self.pos)
        tg = self.target.get(key)
        if tg is None or now - tg[1] > 120 or tuple(tg[0]) == cur and kind == "gez" or kind != tg[3]:
            cells = self.choose_target(kind, now)
            mask = np.zeros((self.G, self.G), bool)
            mask[tuple(cells.T)] = True
            tg = (cells[0], now, self._field(mask), kind)
            self.target[key] = tg
        d = tg[2]
        if cur in {tuple(c) for c in np.argwhere(d == 0)}:
            wp = (np.array(cur) + 0.5) * self.cell                                # hedef hücrede: merkeze yaklaş
        else:
            best, bv = None, d[cur]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    n = (cur[0] + di, cur[1] + dj)
                    if (di or dj) and 0 <= n[0] < self.G and 0 <= n[1] < self.G and d[n] < bv:
                        best, bv = n, d[n]
            if best is None:
                return None
            wp = (np.array(best) + 0.5) * self.cell
        return wrap(np.arctan2(wp[1] - self.pos[1], wp[0] - self.pos[0]) - self.th)      # hedef yön hatası (rad)

    def dist_to(self, key):
        """Etkin hedefe haritada kalan yol uzunluğu (hücre) ya da None."""
        tg = self.target.get(key)
        if tg is None:
            return None
        v = tg[2][self.cell_of(self.pos)]
        return float(v) if np.isfinite(v) else None

    def state(self):
        return dict(water=self.water, hurt=self.hurt, last=self.last)


class GoalCodes(list):
    """Amaç başına seyrek kod listesi (modüler uzmanlar + yayın): kod[k] = amaç k için tüm uzmanların KC indeksleri."""
    def copy(self):
        return GoalCodes([a.copy() for a in self])


class SelfForecast:
    """ÖZ-ÖNGÖRÜ (ileri model) + TAAHHÜT. Ajan bir amaca ATEŞLEDİĞİĞİNDE bir NİYET açar: (amaç, başlangıç, öngörülen süre T̂).
    T̂, o andaki fiziksel egosantrik görünürlük (ışın mesafesi, dönüş açısı) ve haritadaki güncel rota uzunluğundan çevrimiçi
    öğrenilen Huber-RLS log-süre modelinden gelir. Niyet ya SONUÇLANIR (yedi/içti: gerçek süre y; öngörü hatası öğrenir)
    ya da terk edilir. Süre T̂'yi belirgin aşarsa niyet 'GECİKMİŞ' olur: haritadaki hedef sıfırlanır ve merak güçlenir."""
    GOALS = (0, 1)                                           # sonucu gözlenebilen amaçlar: aç (yedi), susuz (içti)
    NF = 8

    def __init__(self, correct=True, commit=0.0):
        self.correct, self.commit = correct, commit          # commit: açık niyeti koruyan geçiş maliyeti (bazal ganglion histerezisi)
        self.pin = 0                                         # müdahale: yazılan niyetin kalan adımı
        self.pin_goal = None
        self.w = {0: np.array([np.log(48.0), -2.4, 1.9, 0.35, -1.2, 1.4, -0.55, 0.0]),
                  1: np.array([np.log(36.0), -2.1, 1.8, 0.35, -1.1, 1.5, -0.25, 0.0])}
        self.P = {k: 2.0 * np.eye(self.NF) for k in self.GOALS}
        self.open = None                                     # dict(goal,t0,T,f)
        self.records = []                                    # (goal, T̂, y, f)
        self.overdue = False
        self.n_overdue = self.n_abandon = 0
        self.scale = 1.0                                     # müdahale: sahte öngörü çarpanı (nedensel sınama)
        self.boost = 0
        self.gap = 0
        self.cool = {0: 0, 1: 0}                             # sonuçtan sonra aynı amaçta yeni niyet açmadan önce bekleme
        self.log = []                                        # OTOBİYOGRAFİ: karar kayıtları (ne yaptım, neden, ne oldu)

    @staticmethod
    def feats(mind, obs, k):
        if obs is None:
            return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5], dtype=np.float64)
        need = float(1.0 - obs[I_E if k == 0 else I_W])
        day = float(obs[51])
        rays_all = obs[:45].reshape(Universe.R, 5)
        rc = rays_all[:, 1 if k == 0 else 2].copy()
        hz = rays_all[:, RAY_HAZ]
        for ri in range(Universe.R):
            hnear = float(hz[max(0, ri - 1):min(Universe.R, ri + 2)].max())
            if hnear > 0.20 and hnear >= rc[ri] - 0.15:       # tehlike perdesi arkasındaki kaynak doğrudan yenemez
                rc[ri] = 0.0
        rmax = float(rc.max())
        V_k = mind._V()[k]
        if rmax > 0.08 and getattr(mind, "vis_prec", 1.0) >= 0.35 and (not np.isfinite(V_k) or V_k >= -0.05):
            vis_on = 1.0
            dist_vis = 1.0 - rmax
            ang = abs(float(_RAY_OFF[int(np.argmax(rc))])) / (Universe.FOV / 2.0)
            map_on, route = 0.0, 0.0
        else:
            vis_on, dist_vis, ang = 0.0, 0.0, 0.0
            dist = None
            if mind.use_map and k == 1 and (mind.map.water.max() >= 0.45 or mind.map.wseen.max() >= 0.45):
                tg = mind.map.target.get(1)
                if tg is None or mind.age - tg[1] > 30 or tg[3] != "su":
                    mind.map.plan(1, "su", mind.age)
                dist = mind.map.dist_to(1)
            map_on = float(dist is not None)
            route = float(np.clip((dist * mind.map.cell) / max(1.0, mind.map.L), 0.0, 1.4)) if dist is not None else 0.0
        return np.array([1.0, vis_on, vis_on * dist_vis, vis_on * ang, map_on, map_on * route, (1.0 - vis_on) * day, need], dtype=np.float64)

    def predict(self, k, f):
        raw = max(1.0, float(np.exp(np.clip(self.w[k] @ f, 0.4, 6.0)) - 1.0))
        return raw * self.scale

    def learn(self, k, f, y):
        P, x = self.P[k], f
        Px = P @ x
        g = Px / (0.992 + float(x @ Px))
        err = float(np.clip(np.log(y + 1.0) - self.w[k] @ x, -0.45, 0.45))
        self.w[k] = self.w[k] + g * err
        self.P[k] = np.clip((P - np.outer(g, Px)) / 0.992, -5.0, 5.0)

    def precedent(self, goal, need, tol=0.15, k=5):
        """Benzer geçmiş yolculukların (aynı amaç, benzer ihtiyaç düzeyi) süre ortancası, ya da None."""
        ys = [e["y"] for e in self.log if e["goal"] == goal and e["outcome"] == "sonuç" and abs(e["need"][goal] - need) < tol]
        return (float(np.median(ys[-k:])), len(ys)) if len(ys) >= 3 else (None, len(ys))

    def step(self, mind, obs, pobs, died=False):
        """Her adımda: sonuç oldu mu? niyet açık mı/gecikti mi? Yeni ateşlemede niyet aç."""
        t = mind.age
        if died and self.open is not None and self.log and self.log[-1]["outcome"] is None:
            self.log[-1].update(outcome="öldü", t1=t)
        k = mind.goal
        self.gap = self.gap + 1 if (self.open is not None and k != self.open["goal"]) else 0
        if self.open is not None:
            g = self.open["goal"]
            hit = pobs is not None and ((g == 0 and obs[I_E] - pobs[I_E] > 0.05) or (g == 1 and obs[I_W] - pobs[I_W] > 0.02))
            if hit:                                          # SONUÇLANDI: öngörü gerçekle karşılaştırılır, ileri model öğrenir
                y = t - self.open["t0"]
                self.records.append((g, self.open["T"], y, self.open["f"]))
                self.learn(g, self.open["f"], y)
                if self.log and self.log[-1]["outcome"] is None:
                    self.log[-1].update(outcome="sonuç", t1=t, y=y)
                self.cool[g] = t + 10
                self.open, self.overdue = None, False
            elif k != g and self.gap >= 6:                   # terk edildi (başka amaç 6 adım üst üste kazandı; kısa titreme sayılmaz)
                self.n_abandon += 1
                if self.log and self.log[-1]["outcome"] is None:
                    self.log[-1].update(outcome="terk", t1=t)
                self.open, self.overdue = None, False
            else:
                late = (t - self.open["t0"]) > max(40.0, 1.7 * self.open["T"])
                if late and not self.overdue:                # GECİKMİŞ: öngörü yanlış çıktı -> davranışı düzelt
                    self.overdue = True
                    self.n_overdue += 1
                    if self.correct:
                        mind.map.target.pop(g, None)         # yeniden planla
                        self.boost = 40                      # merak: yeni bir yere bak
        if self.open is None and k in self.GOALS and mind.ignited and t >= self.cool.get(k, 0):
            f = self.feats(mind, obs, k)
            need = np.array([1.0 - obs[I_E], 1.0 - obs[I_W], 1.0 - obs[I_I]])
            pm, pn = self.precedent(k, float(need[k]))
            T_hat = self.predict(k, f)
            self.open = dict(goal=k, t0=t, T=T_hat, f=f)
            why = getattr(mind, "_why", None) or {}
            self.log.append(dict(t0=t, goal=k, T=self.open["T"], T_prec=(float(np.exp(0.5 * np.log(self.open["T"] + 1) + 0.5 * np.log(pm + 1)) - 1) if pm else None),
                                 precedent=pm, n_prec=pn, need=need, urg=np.array(why.get("urg", np.zeros(4))), sal=np.array(why.get("sal", np.zeros(4))),
                                 pos=mind.map.pos.copy(), ctx=mind.cur, outcome=None, t1=None, y=None))
        self.boost = max(0, self.boost - 1)

    def stats(self, last=None):
        R = self.records[-last:] if last else self.records
        if len(R) < 5:
            return None
        T = np.array([r[1] for r in R]); y = np.array([r[2] for r in R], float)
        base = np.array([np.mean(y[:i]) if i else y.mean() for i in range(len(y))])
        return dict(n=len(R), mae=float(np.mean(np.abs(T - y))), mae_base=float(np.mean(np.abs(base[1:] - y[1:]))),
                    corr=float(np.corrcoef(T, y)[0, 1]) if np.std(T) > 1e-6 and np.std(y) > 1e-6 else float("nan"))


class Mind:
    def __init__(self, mode="full", seed=0, n_kc=3000, n_res=96, gamma=0.95, lr=0.25, self_model=True, shape=2.0, ctx=True, use_map=True, grow=True, model=True, imag=True, replay=True, maplayouts=False, anchor=False, landmark=False, arena=24.0, kc_cap=0.12, kc_thr=(0.04, 0.9), reservoir="random", perc_sem=True, report_loop=False, stakes=0.0, forecast=True, self_correct=False, commit=0.0, modular=False, broadcast=True, slack_arb=False, intro_on=True):
        self.mode, self.rng, self.gamma, self.lr = mode, np.random.default_rng(seed), gamma, lr
        self.use_sm, self.shape, self.ctx, self.use_map = self_model, shape, ctx, use_map
        self.grow, self.use_model, self.use_imag, self.use_replay, self.use_anchor = grow, model, imag, replay, anchor
        self.epi = EpisodicMemory()
        self.intro = Introspector()
        self.intro_on = intro_on                                       # üstbilişsel içebakış döngüsü (False = felsefi zombi ikizi)
        self.mirror = MirrorSense()                                    # bütünleşik ayna / öz-dış şartlılık organı
        self.force = None                                              # müdahale: (uzman, kalan adım)
        self.sensor_bias = None                                        # müdahale: sahte iç duyu (açık eklemesi, kalan adım)
        self.map = CognitiveMap(L=arena, G=int(arena // 2), multi=maplayouts)
        self.map.landmark = landmark
        self.n_ctx = 48
        self.perc_sem = perc_sem
        self.fc = SelfForecast(correct=self_correct, commit=commit) if forecast else None
        self.stakes = stakes                                              # ölümün bedeli: 0=yok; >0 = ölümde yitirilen bilgi oranı (Q değerleri + anılar)
        self.report_loop, self.rep_mode = report_loop, "on"             # rapor döngüsü: ajanın kendi raporu (token) politika girdisine geri beslenir
        if report_loop:
            self.Win_r = (np.random.default_rng(seed + 777).standard_normal((n_res, Introspector.K_MAX)) * 0.8).astype(np.float32)
        self.Win_m = (np.random.default_rng(seed + 991).standard_normal((n_res, Introspector.K_MAX)) * 0.02).astype(np.float32)
        self._prev_tok = -1
        # --- üst-düzey duyusal kesinlik (anozognozi önleme) ve konsolide motor ajanlık ---
        self.vis_prec = 1.0
        self.vis_base = 0.0
        self._zero_vis = 0
        self.motor_eff = 1.0
        self.motor_ref = np.zeros(Universe.N_ACT, np.float32)
        n_act = Universe.N_ACT
        # --- nöronsal temel: KC kodu yalnızca duyusal girdiden (iç duyu HARİÇ) -> Q ihtiyaçtan bağımsız
        groups = [np.arange(c, 45, 5) for c in range(5)] + [np.array([48, 49, 50])]   # kanal başına ışınlar | proprio
        if ctx:                                                # bağlam lobu: sürekli iç durum h davranışa girer
            groups.append(np.arange(Universe.OBS_DIM, Universe.OBS_DIM + self.n_ctx))
        self.mb = LobedMushroomBody(groups, n_per=n_kc // len(groups), seed=seed, cap_frac=kc_cap, thr_range=kc_thr)
        n_kc = self.mb.n_kc
        self.n_kc = n_kc
        self.modular, self.broadcast, self.slack_arb = modular, broadcast, slack_arb
        if modular:                                              # ÇALIŞMA ALANI MİMARİSİ: uzmana özgü özel girdi + KENDİ kodu; yayın içeriği alıcı kodlarına girer (bkz. nora_gw.py)
            self.n_priv = [Universe.R, Universe.R, Universe.R + 1, 2 * Universe.R + 2]
            self.bc_dim = 3 + Universe.R
            per = max(40, n_kc // (4 * 2))
            self.msp = [LobedMushroomBody([np.arange(npv + self.bc_dim), np.arange(npv + self.bc_dim, npv + self.bc_dim + self.n_ctx)],
                                          n_per=per, claws=4, seed=seed * 17 + j, cap_frac=kc_cap, thr_range=kc_thr)
                        for j, npv in enumerate(self.n_priv)]
            self.moff = np.cumsum([0] + [mb.n_kc for mb in self.msp])
            self.n_kc = int(self.moff[-1])
        # --- süreklilik: tekrarlayan durum (HİÇ sıfırlanmaz)
        self.n_res, self.alpha = n_res, 0.3
        self.n_B = len(SPEC) + 1 + 1 + n_act + 1                         # yayın vektörü boyutu
        rr = self.rng
        if reservoir == "fly":                                 # süreklilik durumu GERÇEK sinek bağlantı şeması üzerinde (FlyWire alt çizgesi)
            Wr = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fly_reservoir_96.npz"))["W"].astype(np.float64)
            n_res = Wr.shape[0]
            self.n_res = n_res
        else:
            Wr = rr.standard_normal((n_res, n_res)) * (rr.random((n_res, n_res)) < 0.1)
        Wr *= 0.9 / max(1e-6, np.abs(np.linalg.eigvals(Wr)).max())
        self.Wr = Wr.astype(np.float32)
        self.Win = (rr.standard_normal((n_res, Universe.OBS_DIM + self.n_B)) * 0.4 *
                    (rr.random((n_res, Universe.OBS_DIM + self.n_B)) < 0.3)).astype(np.float32)
        self.h = np.zeros(n_res, np.float32)
        # --- öz-model (NLMS) ve güven başlığı
        self.nz = n_res + n_act + len(BODY_IDX) + 2 * n_act + 1 + 1 + 6
        self.n_body = len(BODY_IDX) + 1                                 # beden(6) + yön değişimi
        self.scale = np.full(self.n_body, 0.05, np.float32)             # sinyal değişkenliği (normalizasyon)
        # --- BAĞLAMLAR: (refleks Q-kafaları + öz-model). Sürprizle yeni bağlam açılır, eskiler DONAR (sıfır unutma)
        self.ctxs = [self._new_ctx()]
        self.cur, self.cerr, self.oerr, self.cool = 0, [1.0], [0.0], 0
        self.prob = None                                               # deneme (probation) durumu: {'old': idx, 't': adım}
        self.freeze = False                                            # sürprizde aktif bağlamın plastisitesi kapanır
        self.n_created = 1
        # --- çalışma alanı ve duygu
        self.acc = np.zeros(len(SPEC), np.float32)
        self.B = np.zeros(self.n_B, np.float32)
        self.mood, self.valence, self.goal, self.ignited = 0.0, 0.0, -1, False
        self.prev = None        # (kc_idx, action, obs, H)
        self.pred, self.z, self.conf = None, None, 0.0
        self.err_n = 0.0
        self.err_vel = 0.0
        self._last_kind = -1
        self.drift_e, self.drift_w, self.n_idle = 0.0, 0.0, 0
        self.n_fix_e, self.n_fix_w = 0, 0
        self.ev_mu, self.ev_var, self.agency_z = 0.5, 0.1, 0.0
        self._prev_pos = np.zeros(2)
        self.err_head = 0.0
        self.age = 0
        self.last_obs = None
        self.last_y = None

    # ------------------------------------------------------------ bağlamlar
    def _new_ctx(self, src=None):
        c = dict(Wq=np.zeros((len(SPEC), Universe.N_ACT, self.n_kc), np.float32),
                 Wsm=np.zeros((self.n_body, self.nz), np.float32), Wc=np.zeros(self.nz, np.float32),
                 W_meta=np.zeros((Introspector.K_MAX, len(SPEC)), np.float32),
                 good=False, n=0, ok=0, V=np.full(3, np.nan), Vn=np.zeros(3), S=np.zeros((3, 3)),
                 Wm=np.zeros((Universe.N_ACT, 48, 45), np.float32), Pm=np.zeros((Universe.N_ACT, 48, 48), np.float32))
        for a in range(Universe.N_ACT):                                 # dünya modeli: kalıcılıkla başla, RLS ile öğren
            c["Wm"][a][:45, :45] = np.eye(45, dtype=np.float32)
            c["Pm"][a] = 100.0 * np.eye(48, dtype=np.float32)                                      # good: bu bağlam dünyayı bir kez açıkladı mı (olgun mu)
        if src is not None:                                           # sıcak başlangıç: bilgi KOPYALANIR, eskisi donuk kalır
            for k in ("Wq", "Wsm", "Wc", "Wm", "Pm"):
                c[k][:] = src[k]
            pass                                                      # anlamsal bellek (V) KOPYALANMAZ: aday sonuçları sıfırdan öğrenir
        return c

    Wq = property(lambda s: s.ctxs[s.cur]["Wq"])
    Wsm = property(lambda s: s.ctxs[s.cur]["Wsm"], lambda s, v: s.ctxs[s.cur].__setitem__("Wsm", v))
    Wc = property(lambda s: s.ctxs[s.cur]["Wc"], lambda s, v: s.ctxs[s.cur].__setitem__("Wc", v))

    def _sleep(self, n):
        """UYKU TEKRARI: nadir ve duygusal geçişler çevrimdışı yeniden oynatılır (konsolidasyon)."""
        for pidx, pa, r, idx, dead in self.epi.replay_batch(self.rng, n, self.cur):
            qn = self._q(idx).max(1) if not dead else np.zeros(len(SPEC), np.float32)
            delta = np.clip(r + self.gamma * qn - self._q(pidx)[:, pa], -3.0, 3.0)
            self._td_update(pidx, pa, delta)

    def _learn_world(self, obs):
        """Dünya modeli: eylem başına doğrusal egosantrik dönüşüm, özyinelemeli en küçük kareler (RLS)."""
        if self.vis_prec < 0.35 or self.motor_eff < 0.35:               # körlükte ya da felçte görsel-motor dünya modeli bozulmasın!
            return
        a = self.prev[1]
        c = self.ctxs[self.cur]
        x = np.r_[self.last_obs[:45], 1.0, self.last_obs[48], self.last_obs[49]].astype(np.float32)
        P = c["Pm"][a]
        Px = P @ x
        k = Px / (0.9995 + float(x @ Px))
        c["Wm"][a] += np.outer(k, obs[:45] - x @ c["Wm"][a])
        c["Pm"][a] = (P - np.outer(k, Px)) / 0.9995

    def _update_sensory_precision(self, obs, died):
        """Üst-düzey duyusal kesinlik kestirimi (Anton sendromu / görsel anozognozi önleme):
        Göz sağlamken dokunma, yeme/içme/yaralanma ya da önceki karedeki yakın nesne en az bir ışında görünür.
        Tüm 45 kanal sıfırken çapraz-modal temas oluşması veya yerleşik görme tabanına rağmen uzun süre tam sıfır
        gelmesi, dünyanın boşaldığını değil GÖRME KANALININ KESİLDİĞİNİ gösterir."""
        rmax = float(obs[:45].max())
        if rmax > 0.02:
            self.vis_base = 0.995 * self.vis_base + 0.005 * 1.0
            self._zero_vis = 0
            self.vis_prec = min(1.0, self.vis_prec + 0.35)
            return
        self._zero_vis += 1
        if not died and self.prev is not None:
            pa, pobs = self.prev[1], self.prev[2]
            prmax = float(pobs[:45].max())
            de = abs(float(obs[I_E] - pobs[I_E]))
            dw = float(obs[I_W] - pobs[I_W])
            di = float(pobs[I_I] - obs[I_I])
            # 1) Bir adımda yakın nesnenin (prmax > 0.35) tüm 45 kanalda aniden buharlaşması
            if prmax > 0.35 and rmax < 1e-4:
                self.vis_prec *= 0.30
            # 2) Çapraz-modal çelişki: ileri giderken duvara/kayaya çarpma veya d=0'da yeme/içme/yara ama görsel ışın = 0
            if rmax < 1e-4 and ((obs[50] > 0.5 and (pa == 1 or pobs[48] > 0.15)) or de > 0.05 or dw > 0.02 or di > 0.02):
                self.vis_prec *= 0.25
        if self.vis_base > 0.35 and self._zero_vis > 18 and rmax < 1e-4:
            self.vis_prec *= 0.80
        self.vis_prec = float(np.clip(self.vis_prec, 0.0, 1.0))

    def _update_motor_agency(self, obs, d, died):
        """Konsolide beden beklentisi ile negatif ajanlık (motor felç) kestirimi:
        Dönüş komutları (2, 3) duvara değse bile pusula açısını döndürür; açık alanda (touch < 0.5) ileri/geri
        komutları (1, 4) boyuna ivme üretir. Öğrenilmiş beden kazancına (motor_ref) rağmen sıfır tepki gelmesi
        motor bağlantının koptuğunu ('komutlarım bedenimi hareket ettirmiyor') gösterir."""
        if died or self.prev is None:
            return
        pa, pobs = self.prev[1], self.prev[2]
        tested, m_obs = False, 0.0
        if pa in (2, 3):
            tested, m_obs = True, abs(float(d[6]))
        elif pa in (1, 4) and obs[50] < 0.5 and pobs[50] < 0.5:
            tested, m_obs = True, abs(float(obs[48] - 0.78 * pobs[48]))
        if not tested:
            return
        if m_obs > 0.08:
            self.motor_ref[pa] = (0.98 * self.motor_ref[pa] + 0.02 * m_obs) if self.motor_ref[pa] > 0 else m_obs
            self.motor_eff = min(1.0, self.motor_eff + 0.35)
        elif self.motor_ref[pa] > 0.10 and m_obs < 0.20 * self.motor_ref[pa]:
            self.motor_eff = max(0.0, self.motor_eff * 0.60)

    def observe_mirror(self, y_trace, rng=None):
        """Dışarıda gözlenen bir nesnenin açısal hız izini kendi efferans kopyasıyla karşılaştır ('ben'/'başka'/'tekrar')."""
        return self.mirror.classify(y_trace, rng=rng)

    def _infer_context(self):
        """Bağlam çıkarımı: olgun bağlam bir kez dünyayı açıklar (good); sonra sürpriz görürse
        (1) önce başka bir olgun bağlam daha iyi mi diye bakılır (eski dünyaya DÖNÜŞ),
        (2) değilse aktif bağlam DONAR ve klonu deneme (probation) bağlamı olarak eğitilir;
        deneme 300 adım sonra eski bağlam daha iyi açıklıyorsa (yanlış alarm) geri alınır."""
        score = np.array(self.cerr) + 2.0 * np.array(self.oerr)
        c = self.ctxs[self.cur]
        cs = float(score[self.cur])
        self.cool = max(0, self.cool - 1)
        c["ok"] = c["ok"] + 1 if cs < 0.7 else 0
        if not c["good"] and c["ok"] > 200 and c["n"] > 4000:           # olgun: dünyayı uzun süre tutarlı açıkladı
            c["good"] = True
        if self.prob is not None:
            self.prob["t"] += 1
            self.freeze = False
            if self.prob["t"] >= 400:
                cand = self.ctxs[self.cur]

                def same_world(i):                                      # anlamsal imza (V işaretleri) aday ile örtüşüyor mu?
                    o = self.ctxs[i]
                    ks = [k for k in range(3) if np.isfinite(o["V"][k]) and np.isfinite(cand["V"][k]) and o["Vn"][k] >= 3 and cand["Vn"][k] >= 2]
                    if self.perc_sem:                                   # algısal anlambilim de aynı olmalı (aynı gözlük)
                        for k in range(3):
                            if o["S"][:, k].sum() >= 6 and cand["S"][:, k].sum() >= 4 and int(np.argmax(o["S"][:, k])) != int(np.argmax(cand["S"][:, k])):
                                return False
                    return len(ks) >= 1 and all(o["V"][k] * cand["V"][k] > 0 for k in ks) and self.cerr[i] < 1.0
                others = [i for i in range(len(self.ctxs)) if i != self.cur and self.ctxs[i]["good"] and same_world(i)]
                o = min(others, key=lambda i: score[i]) if others else None
                if o is not None:                                       # eski bir dünya bu olguyu zaten açıklıyor -> aday ATILIR
                    self.ctxs.pop(self.cur)
                    self.cerr.pop(self.cur)
                    self.oerr.pop(self.cur)
                    self.cur = o
                    self.n_created -= 1
                self.prob = None
                self.cool = 1500
            return
        self.freeze = False
        if not c["good"] or self.cool > 0:
            return
        best = int(np.argmin(score))
        if best != self.cur and self.ctxs[best]["good"] and score[best] < 0.6 * cs:   # eski bir dünyaya dönüldü
            self.cur, self.cool = best, 100
        elif cs > 1.2 and len(self.ctxs) < 8:                           # yeni bir dünya: eskiyi dondur, klonu eğit
            old = self.cur
            self.ctxs.append(self._new_ctx(self.ctxs[old]))
            self.cerr.append(float(self.cerr[old]))
            self.oerr.append(0.0)
            self.cur, self.prob = len(self.ctxs) - 1, {"old": old, "t": 0}
            self.n_created += 1

    @staticmethod
    def _bodyvec(obs):
        return np.r_[obs[BODY_IDX], np.arctan2(obs[Universe.I_COMPASS + 1], obs[Universe.I_COMPASS])].astype(np.float32)

    @staticmethod
    def _delta(y, y0):
        d = y - y0
        d[-1] = wrap(d[-1])
        return d

    def _zvec(self, obs, act):
        a1 = np.zeros(Universe.N_ACT, np.float32)
        a1[act] = 1.0
        return np.concatenate([0.25 * self.h, 3.0 * a1, 3.0 * obs[BODY_IDX], 3.0 * np.outer(a1, obs[48:50]).ravel(),
                               [3.0 * obs[51]], [3.0], 3.0 * self._phi_raw(obs), 3.0 * a1[1] * self._phi_raw(obs)]
                              ).astype(np.float32)

    def _steer(self, err, obs):
        """Hedef yön hatasından eylem: ÖĞRENİLMİŞ modelin 'hangi eylem yönümü nasıl değiştirir' öngörüsüyle.
        Kumandalar ters çevrilirse ezber değil model değişir; plan kendiliğinden uyum sağlar."""
        if abs(err) <= 0.3:
            return 1
        if self.use_model and self.use_sm and self.age > 300:
            pr = np.array([float(self.Wsm[-1] @ self._zvec(obs, a)) for a in (2, 3)])
            s = np.sign(err) * pr
            if s.max() > 0.05:
                return 2 + int(np.argmax(s))
        return 2 if err > 0 else 3

    # ------------------------------------------------------------ yardımcılar
    def _H(self, obs):
        return float(((1 - obs[I_E:I_I + 1]) ** 2).sum())

    @staticmethod
    def _phi_raw(obs):
        """Ham duyusal yakınlık (yiyecek, su, tehlike): kanalın en yakın ışın yakınlığı."""
        rays = obs[:45].reshape(Universe.R, 5)
        return np.array([rays[:, RAY_FOOD].max(), rays[:, RAY_WATER].max(), rays[:, RAY_HAZ].max()], np.float32)

    def _V(self):
        """Bağlamın anlamsal sonuç belleği: yemek/içmek/yara beden için ne yapar (bilinmiyorsa doğuştan beklenti)."""
        v = self.ctxs[self.cur]["V"]
        return np.where(np.isfinite(v), v, V_DEFAULT).astype(np.float32)

    def _phi(self, obs):
        """Şekillendirme potansiyeli Φ_k = işaret(V_k) * yakınlık_k. İşaret ÖĞRENİLİR (zehirli yemekten kaçın);
        potansiyel-tabanlı olduğu için (Ng 1999) en iyi politikayı değiştirmeden öğrenmeyi yoğunlaştırır."""
        return np.r_[np.sign(self._V()) * self._phi_raw(obs), 0.0].astype(np.float32)

    def imagine(self, obs, k, N=None, H=None):
        """HAYAL GÜCÜ (Uzamsal Özellik Bağlamalı + Üstbilişsel Ufuk Kontrollü):
        Öğrenilmiş dünya modeliyle (eylem başına 9×5 ışın dönüşümü) eylem dizilerini zihinde dener.
        (1) Uzamsal Bağlama: İleri çarpışma konisindeki (ışın 3..5) tehlike, yiyecek/su kovalarken bile
            cezalandırılır; yanal ışınlardaki tehlikenin etrafından dolaşan yörüngeler seçilir.
        (2) Üstbilişsel Efor: İçebakış sürprizi/çatışması yüksekken simülasyon bütçesi (N, H) derinleşir."""
        if N is None or H is None:
            if self.intro_on and self.intro is not None and self.intro.n > 100:
                alert = (self.intro.surprise > 2.2) or (self.oerr[self.cur] > 0.25)
                N = N or (55 if alert else 35)
                H = H or (6 if alert else 5)
            else:
                N = N or 15
                H = H or 2
        c = self.ctxs[self.cur]
        V = self._V()
        chan = (RAY_FOOD, RAY_WATER, RAY_HAZ)[k]
        gain = (3.0, 3.0, 5.0)[k] * float(V[k])
        sgn = float(np.sign(V[k]))
        bind_haz = (k < 2) and (not self.modular or self.broadcast) and self.mode == "full" and float(V[2]) < 0
        haz_w = abs(float(V[2])) * (2.0 - float(obs[I_I]))
        per = max(2, N // Universe.N_ACT)
        a0 = np.repeat(np.arange(Universe.N_ACT), per)
        seq = self.rng.choice(Universe.N_ACT, size=(len(a0), H), p=[.05, .5, .2, .2, .05])
        seq[:, 0] = a0
        R = np.tile(obs[:45], (len(a0), 1))
        vf, vl = float(obs[48]), float(obs[49])
        R3_0 = R.reshape(-1, Universe.R, 5)
        prox = (0.65 * R3_0[:, 2:7, chan].max(1) + 0.35 * R3_0[:, :, chan].max(1)) if k < 2 else R3_0[:, :, chan].max(1)
        ret = np.zeros(len(a0), np.float32)
        touched = np.zeros(len(a0), bool)
        for t in range(H):
            X = np.c_[R, np.ones(len(R), np.float32), np.full(len(R), vf, np.float32), np.full(len(R), vl, np.float32)]
            Rn = np.empty_like(R)
            for aa in range(Universe.N_ACT):
                sel = seq[:, t] == aa
                if sel.any():
                    Rn[sel] = X[sel] @ c["Wm"][aa]
            R = np.clip(Rn, 0.0, 1.0)
            R3 = R.reshape(-1, Universe.R, 5)
            if k < 2:
                pn = 0.65 * R3[:, 2:7, chan].max(1) + 0.35 * R3[:, :, chan].max(1)
                contact = (R3[:, 2:7, chan].max(1) > 0.88) & ~touched
            else:
                pn = R3[:, :, chan].max(1)
                contact = pn > 0.9
            step_r = gain * contact + 2.0 * sgn * (self.gamma * pn - prox)
            if bind_haz:
                haz_fwd = R3[:, 3:6, RAY_HAZ].max(1)
                step_r = step_r - haz_w * (6.0 * (haz_fwd > 0.76) + 2.5 * np.maximum(0.0, haz_fwd - 0.55))
            ret += (self.gamma ** t) * step_r
            touched |= contact
            prox = pn
        sc = np.array([0.5 * (ret[a0 == a].mean() + ret[a0 == a].max()) for a in range(Universe.N_ACT)], np.float32)
        return np.clip((sc - sc.mean()) / (sc.std() + 0.05), -2.0, 2.0)

    def _q(self, idx):
        if isinstance(idx, GoalCodes):
            return np.stack([self.Wq[k][:, idx[k]].sum(1) for k in range(len(SPEC))])
        return self.Wq[:, :, idx].sum(2)                 # (uzman, eylem)

    def _td_update(self, pidx, pa, delta):
        if isinstance(pidx, GoalCodes):
            for k in range(len(SPEC)):
                self.Wq[k, pa, pidx[k]] += self.lr * delta[k] / max(1, len(pidx[k]))
        else:
            self.Wq[:, pa, pidx] += (self.lr * delta / max(1, len(pidx)))[:, None]

    def _codes(self, obs):
        """Modüler uzmanların amaç-koşullu kodları. Özel girdiler başka uzmanın bilgisini içermez; yayın_k = amaç kimliği + amacın kendi algısı."""
        r = obs[:45].reshape(Universe.R, 5)
        priv = [r[:, 1], r[:, 2], np.r_[r[:, 3], obs[Universe.I_TOUCH]], np.r_[r[:, 0], r[:, 4], obs[Universe.I_SPEED:Universe.I_SPEED + 2]]]
        hc = np.clip(3.0 * np.abs(self.h[:self.n_ctx]), 0, 1)
        use_bc = self.broadcast and self.mode == "full"
        out, zero = [], None
        for k in range(len(SPEC)):
            if use_bc and k < 3:
                bc = np.r_[np.eye(3)[k], r[:, k + 1]]
            else:
                bc = np.zeros(self.bc_dim)
            if not (use_bc and k < 3) and zero is not None:
                out.append(zero)
                continue
            code = np.concatenate([self.msp[j].kc(np.r_[priv[j], bc, hc].astype(np.float32)) + self.moff[j] for j in range(4)])
            if not (use_bc and k < 3):
                zero = code
            out.append(code)
        return GoalCodes(out)

    def _urgency(self, obs):
        rays = obs[:45].reshape(Universe.R, 5)
        d = 1 - obs[I_E:I_I + 1]
        haz = float(rays[:, RAY_HAZ].max())
        return np.array([1.5 * d[0] ** 2, 1.5 * d[1] ** 2, max(1.5 * d[2] ** 2, 1.2 * haz ** 2), 0.0], np.float32)

    # ------------------------------------------------------------ ana döngü
    def hear(self, text, trust=0.6):
        """Tanıklığı oku: 'su (x, y) civarında' / 'tehlike (x, y) civarında' (yer) ve 'kural: yiyecek -0.35' (kural) cümlelerini işle."""
        import re
        n = 0
        for what, x, y in re.findall(r"(su|tehlike) \((\d+), (\d+)\) civarında", text):
            c = self.map.cell_of((float(x), float(y)))
            arr = self.map.wseen if what == "su" else self.map.seen
            arr[c] = max(arr[c], trust)                       # duyulan bilgi: orta güven (yeniden görüldükçe oyla pekişir)
            n += 1
        for what, val in re.findall(r"kural: (yiyecek|su|tehlike) ([+-]\d+\.\d+)", text):
            k = {"yiyecek": 0, "su": 1, "tehlike": 2}[what]
            c = self.ctxs[self.cur]
            c["V"][k], c["Vn"][k] = float(val), max(c["Vn"][k], 3.0)       # duyulan kural: anlamsal belleğe (güvenle) işlenir
            n += 1
        self.map.target = {}
        return n

    def intervene(self, kind, value=None, steps=30):
        """Deneysel müdahale: DÜNYAYA dokunmadan iç durumu değiştirir (nedensel doğrulama protokolü)."""
        if kind == "goal":
            self.force = [int(value), steps]                           # uzman birikimini zorla yükselt
        elif kind == "mood":
            self.mood = float(value)
        elif kind == "unc":
            self.map.unc = float(value)
        elif kind == "intent":                                         # niyet nesnesine DOĞRUDAN yazma: (amaç, adım). Rapor ve davranış aynı nesneyi okur.
            g = int(value)
            f = self.fc.feats(self, self.last_obs, g)
            self.fc.open = dict(goal=g, t0=self.age, T=self.fc.predict(g, f), f=f)
            self.fc.pin = steps
            self.fc.pin_goal = g
        elif kind == "forecast":                                       # sahte öz-öngörü: öngörülen süreler değer katsayısıyla ölçeklenir
            self.fc.scale = float(value)
        elif kind == "sensor":                                         # sahte iç duyu: (Δenerji, Δsu, Δbütünlük) eklenir
            self.sensor_bias = [np.asarray(value, np.float32), steps]
        else:
            raise ValueError(kind)

    def act(self, obs, died=False):
        self.age += 1
        rr = self.rng
        if self.sensor_bias is not None:
            obs = obs.copy()
            obs[I_E:I_I + 1] = np.clip(obs[I_E:I_I + 1] + self.sensor_bias[0], 0.0, 1.0)
            self.sensor_bias[1] -= 1
            if self.sensor_bias[1] <= 0:
                self.sensor_bias = None
        # ---------- üst-düzey duyusal kesinlik (körlük / anozognozi denetimi)
        self._update_sensory_precision(obs, died)
        vis_ok = self.vis_prec >= 0.35
        # ---------- algı: seyrek kod + yenilik
        idx = self.mb.kc(np.concatenate([np.abs(obs), np.clip(3.0 * np.abs(self.h[:self.n_ctx]), 0, 1)]))
        nov = self.mb.novelty(idx)                                     # yenilik her zaman tek (bütünsel) kodla ölçülür
        if self.modular:
            idx = self._codes(obs)
        H = self._H(obs)
        if self.use_map:
            self.map.update(obs, self.prev[2] if self.prev is not None else None, died, self.age, vis_ok=vis_ok)
        # ---------- valans (homeostatik hata türevi) ve ruh hali
        if self.prev is not None:
            raw = -(H - self.prev[3]) * 5.0
            hurt = max(0.0, self.prev[2][I_I] - obs[I_I])
            self.valence = float(np.clip(raw - 30.0 * hurt, -2.0, 2.0)) if not died else -2.0
            if self.intro_on and self._prev_tok >= 0 and self.goal >= 0 and not self.freeze:
                tk = min(self._prev_tok, Introspector.K_MAX - 1)
                # Kayıttan yüklenen bağlamda W_meta yoksa (npz'de saklanmıyor) sıfırdan açılır:
                # yüklenen Nora da yaşamaya devam edebilmeli.
                wm = self.ctxs[self.cur].setdefault(
                    "W_meta", np.zeros((Introspector.K_MAX, len(SPEC)), np.float32))
                wm[tk, self.goal] = float(np.clip(
                    wm[tk, self.goal] + 0.05 * (self.valence - wm[tk, self.goal]), -0.35, 0.35
                ))
        self.mood += 0.01 * (self.valence - self.mood)
        # ---------- öğrenme (dopamin = TD hatası, yalnızca aktif KC'ler)
        tr = None
        if self.prev is not None:
            pidx, pa, pobs, _ = self.prev
            pain = 2.5 * min(0.0, float(obs[I_I] - pobs[I_I])) if ((not self.modular or self.broadcast) and self.mode == "full") else 0.0
            r = np.array([3.0 * max(-0.5, obs[I_E] - pobs[I_E]) + pain,
                          3.0 * max(0.0, obs[I_W] - pobs[I_W]) + pain,
                          5.0 * (obs[I_I] - pobs[I_I]),
                          1.0 * nov], np.float32)
            r += self.shape * (self.gamma * self._phi(obs) - self._phi(pobs)) * np.array([1, 1, 1, 0], np.float32)
            qn = self._q(idx).max(1) if not died else np.zeros(len(SPEC), np.float32)
            delta = np.clip(r + self.gamma * qn - self._q(pidx)[:, pa], -3.0, 3.0)
            tr = (pidx, pa, r.copy(), idx.copy(), died)
            if not self.freeze and self.motor_eff >= 0.35:
                self._td_update(pidx, pa, delta)
                self.ctxs[self.cur]["n"] += 1
        # ---------- öz-model: önceki öngörüyü gerçekle karşılaştır; bağlam çıkarımı
        yv = self._bodyvec(obs)
        if self.pred is not None and self.use_sm and not died:        # ölüm/yeniden doğuş geçişi öğrenme verisi değildir
            d = self._delta(yv, self.last_y)
            self._update_motor_agency(obs, d, died)
            sc = np.maximum(self.scale, REF)
            evs = []                                                   # sonuç olayları: (değişken, değişim)
            if abs(d[0]) > 0.05:
                evs.append((0, float(d[0])))                           # yem (yedim -> enerji +/-)
            if d[1] > 0.02:
                evs.append((1, float(d[1])))                           # içtim
            if d[2] < -0.02:
                evs.append((2, float(d[2])))                           # yaralandım
            chv = self.last_obs[:45].reshape(-1, 5)[:, 1:4].max(0)        # olaydan hemen önce hangi algı kanalı baskındı (food/su/tehlike görünümü)
            ch = int(np.argmax(chv)) if (vis_ok and chv.max() > 0.85 and np.sort(chv)[-2] < 0.5) else -1
            for ci, c in enumerate(self.ctxs):
                pd = c["Wsm"] @ self.z
                en_c = np.abs(d - pd) / sc
                if self.motor_eff >= 0.35:
                    self.cerr[ci] = 0.97 * self.cerr[ci] + 0.03 * float(en_c[[3, 4, 6]].mean())
                for k, dv in evs:                                       # anlamsal bellek: 'şey X bedene ne yapar' işareti doğru muydu?
                    v = c["V"][k]
                    if np.isfinite(v) and c["Vn"][k] >= 3:
                        self.oerr[ci] = 0.6 * self.oerr[ci] + 0.4 * (1.0 if v * dv < 0 else 0.0)
                    if ch >= 0 and self.perc_sem:                       # algısal anlambilim: 'bu görünüm hangi sonucu getirir' (dünyanın gözlükleri)
                        S = c["S"]
                        if S[:, k].sum() >= 6 and S[:, k].max() >= 0.6 * S[:, k].sum():
                            self.oerr[ci] = 0.6 * self.oerr[ci] + 0.4 * (1.0 if int(np.argmax(S[:, k])) != ch else 0.0)
            cc = self.ctxs[self.cur]
            if not self.freeze:
                if ch >= 0:
                    for k, dv in evs:
                        if cc["S"][:, k].sum() < 6 or int(np.argmax(cc["S"][:, k])) == ch or not cc["good"]:
                            cc["S"][ch, k] += 1.0
                for k, dv in evs:                                       # çelişen olay olgun bağlamın belleğini KİRLETMEZ
                    v = cc["V"][k]
                    if not np.isfinite(v):
                        cc["V"][k], cc["Vn"][k] = dv, 1.0
                    elif cc["Vn"][k] < 3 or v * dv >= 0 or not cc["good"]:
                        cc["V"][k] += 0.2 * (dv - v)
                        cc["Vn"][k] += 1.0
            err = d - self.pred
            en = np.abs(err) / sc
            self.err_n = float(en[3:5].mean())          # hareket boyutları: öngörülebilir olan öz-etki
            self.err_vel = float(en[3:5].mean())
            self.err_head = float(en[6])
            if self.motor_eff >= 0.35:                                 # felçte sağlıklı beden öz-modeli (Wsm, ev_mu) silinmez!
                a = 0.01
                self.ev_var += a * ((self.err_vel - self.ev_mu) ** 2 - self.ev_var)
                self.ev_mu += a * (self.err_vel - self.ev_mu)
                self.agency_z = (self.err_vel - self.ev_mu) / np.sqrt(self.ev_var + 1e-3)
                zz = float(self.z @ self.z) + 1e-3
                if not self.freeze:                                    # sürprizde aktif bağlam KİRLENMESİN
                    self.Wsm += 0.5 * np.outer(err, self.z) / zz
                    self.Wc += 0.2 * (self.err_n - float(self.Wc @ self.z)) * self.z / zz
                self.scale += 0.01 * (np.abs(d) - self.scale)
            else:
                self.agency_z = 0.0
            if not self.freeze:
                self._learn_world(obs)
            if self.grow and self.motor_eff >= 0.35:
                self._infer_context()
        # ---------- beden istatistikleri: kendiliğinden tükenme (olaysız adımlarda) ve onarım olayları
        if self.prev is not None and not died:
            de, dw = float(obs[I_E] - self.prev[2][I_E]), float(obs[I_W] - self.prev[2][I_W])
            if abs(de) < 0.01 and abs(dw) < 0.01:
                self.n_idle += 1
                a_ = 0.002
                self.drift_e += a_ * (de - self.drift_e)
                self.drift_w += a_ * (dw - self.drift_w)
            if de > 0.05:
                self.n_fix_e += 1
            if dw > 0.02 and self._last_kind != 1:
                self.n_fix_w += 1
        # ---------- otobiyografik bellek: öne çıkan olayları kaydet
        if self.prev is not None:
            pobs = self.prev[2]
            kinds = []
            evpos = self.map.pos.copy()
            if died:
                kinds.append(3)
                evpos = self._prev_pos.copy()                          # yeniden doğuş öncesi konum
                if self.stakes > 0:                                    # ÖLÜMÜN BEDELİ: öğrenilmiş değerlerin ve anıların bir kısmı silinir
                    self.ctxs[self.cur]["Wq"] *= (1.0 - self.stakes)
                    keep = [e for e in self.epi.items if self.rng.random() > self.stakes]
                    self.epi.items = keep
                if pobs[I_I] < 0.15:                                   # ölümcül yara da bir 'yaralandı' anısıdır
                    kinds.append(2)
            else:
                if abs(obs[I_E] - pobs[I_E]) > 0.05:
                    kinds.append(0)
                if obs[I_W] - pobs[I_W] > 0.02 and self._last_kind != 1:
                    kinds.append(1)
                if pobs[I_I] - obs[I_I] > 0.02 and self._last_kind != 2:
                    kinds.append(2)
            for kd in kinds:
                self.epi.add(t=self.age, kind=KINDS[kd], pos=evpos, val=float(self.valence), goal=self.goal,
                             body=obs[I_E:I_I + 1].copy(), ctx=self.cur, tr=tr)
            self._last_kind = kinds[0] if kinds else -1
            if tr is not None and not kinds and abs(self.valence) > 0.3:         # olaysız ama duygusal geçiş (öğrenme için)
                self.epi.add(t=self.age, kind="geçiş", pos=self.map.pos.copy(), val=float(self.valence), goal=self.goal,
                             body=obs[I_E:I_I + 1].copy(), ctx=self.cur, tr=tr)
        if self.use_replay and self.age % 500 == 0 and not self.freeze:
            self._sleep(48)
        # ---------- süreklilik: tekrarlayan durum (algı + önceki yayın + nedensel içebakış)
        u = np.concatenate([obs, self.B])
        drive = self.Win @ u + self.Wr @ self.h
        itok = self.intro.token if (self.intro is not None) else -1
        if self.intro_on and itok >= 0 and self.mode == "full":
            drive = drive + self.Win_m[:, min(itok, Introspector.K_MAX - 1)]
        if self.report_loop and self.rep_mode != "cut" and itok >= 0:
            tok = itok if self.rep_mode == "on" else int(rr.integers(0, max(1, len(self.intro.P))))   # 'sham': rastgele rapor
            drive = drive + self.Win_r[:, min(tok, Introspector.K_MAX - 1)]
        self.h = ((1 - self.alpha) * self.h + self.alpha * np.tanh(drive)).astype(np.float32)
        # ---------- uzmanlar: öneriler + çıkar (salience) + üstbilişsel modülasyon
        Q = self._q(idx)                                              # (4, 5)
        urg = self._urgency(obs)
        opp = np.tanh(np.maximum(Q.max(1), 0.0))
        sal = urg * (0.6 + 0.8 * opp)
        sal[3] = 0.15 + 0.4 * nov + (0.35 if (self.fc is not None and self.fc.boost > 0) else 0.0)
        if self.intro_on and self.intro is not None and itok >= 0 and self.mode == "full":
            tk = min(itok, Introspector.K_MAX - 1)
            wm = self.ctxs[self.cur].setdefault("W_meta", np.zeros((Introspector.K_MAX, len(SPEC)), np.float32))
            sal = sal + 0.25 * wm[tk]
            if self.intro.surprise > 2.2 or self.oerr[self.cur] > 0.25:
                haz_vis = float(obs[:45].reshape(Universe.R, 5)[:, RAY_HAZ].max() > 0.20)
                sal[2] += 0.25 * haz_vis
        sal = sal + 0.02 * rr.standard_normal(len(SPEC))
        if self.slack_arb and self.fc is not None and self.age > 3000:          # PAYLI ÖNCELİK: ölüme kalan süre − öz-öngörülen ulaşma süresi en az olan ihtiyaç öne geçer
            for k, drift in ((0, self.drift_e), (1, self.drift_w)):
                ttd = obs[I_E if k == 0 else I_W] / max(1e-5, -drift)
                slack = ttd - self.fc.predict(k, self.fc.feats(self, obs, k))
                sal[k] += 1.0 / (1.0 + np.exp(np.clip(slack / (0.25 * ttd + 20.0), -20, 20)))
        if self.fc is not None and self.mode == "full":     # TAAHHÜT: açık niyet, yarışmada geçiş maliyetiyle korunur
            if self.fc.pin > 0 and self.fc.pin_goal is not None:
                sal[self.fc.pin_goal] += 1.5                # yazılmış niyet: güçlü taahhüt (müdahale süresi boyunca korunur)
                self.fc.pin -= 1
                if self.fc.open is None:
                    f_p = self.fc.feats(self, obs, self.fc.pin_goal)
                    self.fc.open = dict(goal=self.fc.pin_goal, t0=self.age, T=self.fc.predict(self.fc.pin_goal, f_p), f=f_p)
            elif self.fc.open is not None and self.fc.commit > 0 and urg[2] < 0.8:   # acil tehlike taahhüdü aşar
                sal[self.fc.open["goal"]] += self.fc.commit
        temp = 1.0 + 0.8 * float(np.clip(-self.mood, 0, 1))
        phi = self._phi_raw(obs)

        def proposal(k):
            """Uzman k'nın eylem logitleri: refleks Q (öğrenilmiş) + üstbilişsel hayal gücü + gerekirse bellek/plan."""
            lg = 8.0 * Q[k] + PRIOR
            if vis_ok and self.use_imag and self.intro_on and self.intro is not None and self.age > 1500 and k < 3 and phi[k] > 0.05:
                lg += IM_G * self.imagine(obs, k)
            if self.use_map and k != 2 and (not vis_ok or not (k < 2 and phi[k] > 0.05)):
                pe = self.map.plan(k, "su" if k == 1 else "gez", self.age)
                if pe is not None:
                    lg[self._steer(pe, obs)] += PLAN_G
            return lg
        if self.mode == "random":
            logits, self.goal, self.ignited = PRIOR.copy(), -1, False
        elif self.mode == "flat":                                      # çalışma alanı YOK: ağırlıklı karışım
            w = urg + 0.1
            logits = sum(w[k] * proposal(k) for k in range(len(SPEC))) / w.sum()
            self.goal, self.ignited = -1, False
        else:
            # global çalışma alanı: yanal baskılamalı sızıntılı toplayıcılar
            a = self.acc
            if self.force is not None:
                a = np.zeros_like(a)
                a[self.force[0]] = 1.2
                self.force[1] -= 1
                if self.force[1] <= 0:
                    self.force = None
            for _ in range(3):
                a = np.clip(0.75 * a + 0.5 * sal - 0.35 * (a.sum() - a), 0.0, 1.5)
            self.acc = a
            order = np.argsort(-a)
            win, run = int(order[0]), int(order[1])
            self.ignited = bool(a[win] > 0.25 and a[win] - a[run] > 0.08)
            self.goal = win if self.ignited else -1
            logits = proposal(win) if self.ignited else PRIOR.copy()
        if self.fc is not None and self.mode == "full":
            self._why = dict(urg=urg.copy(), sal=sal.copy())
            self.fc.step(self, obs, self.prev[2] if self.prev is not None else None, died=died)
        if self.use_map and self.use_anchor and self.map.unc > 0.3 and self.mode != "random" and vis_ok:
            # KAYBOLDUM (öz-bilgi davranışı yönlendirir): konumumdan emin değilim -> en yakın duvara yönel, kendimi yeniden çapala
            P0, Lw = self.map.pos, self.map.L
            cand = [(0.0, P0[1]), (Lw, P0[1]), (P0[0], 0.0), (P0[0], Lw)]
            j = int(np.argmin([P0[0], Lw - P0[0], P0[1], Lw - P0[1]]))
            err_w = wrap(np.arctan2(cand[j][1] - P0[1], cand[j][0] - P0[0]) - self.map.th)
            logits = np.array(logits, dtype=np.float32)
            logits[self._steer(err_w, obs)] += PLAN_G * float(self.map.unc)
        p = softmax(logits / temp)
        act = int(rr.choice(Universe.N_ACT, p=p))
        self.mirror.record_cmd(act)
        # ---------- yayın (tüm modüllere): amaç, valans, ruh hali, eylem kopyası, değer
        B = np.zeros(self.n_B, np.float32)
        if self.goal >= 0:
            B[self.goal] = 1.0
        B[4], B[5] = np.clip(self.valence, -2, 2) / 2.0, np.clip(self.mood, -1, 1)
        B[6 + act] = 1.0
        B[-1] = float(np.tanh(Q[self.goal].max())) if self.goal >= 0 else 0.0
        self.B = B
        # ---------- öz-model öngörüsü (eyleme koşullu)
        self.z = self._zvec(obs, act)
        self.pred = self.Wsm @ self.z                                  # öngörülen Δ(beden)
        self.conf = float(self.Wc @ self.z)
        self._prev_tok = self.intro.step(self, obs) if self.intro is not None else -1
        self._prev_pos = self.map.pos.copy()
        self.prev = (idx, act, obs.copy(), H)
        self.last_obs = obs.copy()
        self.last_y = yv
        return act

    # ------------------------------------------------------------ iç gözlem raporu
    def report(self):
        mood = "iyi" if self.mood > 0.05 else ("kötü" if self.mood < -0.05 else "nötr")
        goal = SPEC[self.goal] if self.goal >= 0 else "bilinçdışı gezinme"
        return ("amaç=%s | duygu-tonu=%+.2f ruh-hali=%s(%+.2f) | öz-model hatası=%.2f güven=%.2f | görme=%.2f motor=%.2f"
                % (goal, self.valence, mood, self.mood, self.err_n, self.conf, self.vis_prec, self.motor_eff))

    # ------------------------------------------------------------ süreklilik: dinamik çalışma durumu
    _RUNTIME_FIELDS = (
        "mode", "gamma", "lr", "use_sm", "shape", "ctx", "use_map", "grow",
        "use_model", "use_imag", "use_replay", "use_anchor", "intro_on", "stakes",
        "report_loop", "rep_mode", "perc_sem", "modular", "broadcast", "slack_arb",
        "vis_prec", "vis_base", "_zero_vis", "motor_eff", "motor_ref", "_prev_tok",
        "cur", "cerr", "oerr", "cool", "prob", "freeze", "n_created", "acc", "B",
        "mood", "valence", "goal", "ignited", "prev", "pred", "z", "conf", "err_n",
        "err_vel", "_last_kind", "drift_e", "drift_w", "n_idle", "n_fix_e", "n_fix_w",
        "ev_mu", "ev_var", "agency_z", "_prev_pos", "err_head", "age", "last_obs",
        "last_y", "_why", "force", "sensor_bias", "sleep_acc", "sleep_last",
        "last_reward", "last_act", "look_x", "look_y", "last_self_action",
    )

    def runtime_state(self) -> dict:
        """NPZ ağırlıklarının taşımadığı, gelecek adımı etkileyen durumu kodla."""
        from nora_state import capture_attrs, capture_rng
        ctx_dynamic = []
        for c in self.ctxs:
            ctx_dynamic.append({"S": c.get("S"), "n": c.get("n", 0), "ok": c.get("ok", 0)})
        return {
            "version": 1,
            "fields": capture_attrs(self, self._RUNTIME_FIELDS),
            "rng": capture_rng(self.rng),
            "ctx_dynamic": capture_attrs(type("CtxBox", (), {"value": ctx_dynamic})(), ("value",)),
            "epi": capture_attrs(self.epi, ("cap", "items")),
            "intro": capture_attrs(self.intro, list(self.intro.__dict__)),
            "mirror": capture_attrs(self.mirror, list(self.mirror.__dict__)),
            "map": capture_attrs(self.map, list(self.map.__dict__)),
            "fc": capture_attrs(self.fc, list(self.fc.__dict__)) if self.fc is not None else {},
            "mb": capture_attrs(self.mb, ("nov_rate",)),
        }

    def load_runtime_state(self, data: dict | None) -> None:
        if not data:
            return
        from nora_state import decode, restore_attrs, restore_rng
        restore_attrs(self, data.get("fields") or {}, strict=True)
        self.rng = restore_rng(self.rng, data.get("rng"))
        ctx_box = data.get("ctx_dynamic") or {}
        ctx_values = (ctx_box.get("values") or {}).get("value") if isinstance(ctx_box, dict) else None
        if ctx_values is not None:
            dynamic = decode(ctx_values)
            for c, d in zip(self.ctxs, dynamic):
                for key in ("S", "n", "ok"):
                    if key in d:
                        c[key] = d[key]
        restore_attrs(self.epi, data.get("epi") or {}, strict=True)
        restore_attrs(self.intro, data.get("intro") or {}, strict=True)
        restore_attrs(self.mirror, data.get("mirror") or {}, strict=True)
        restore_attrs(self.map, data.get("map") or {}, strict=True)
        if self.fc is not None and data.get("fc"):
            restore_attrs(self.fc, data.get("fc") or {}, strict=True)
        restore_attrs(self.mb, data.get("mb") or {}, strict=True)

    # ------------------------------------------------------------ süreklilik: kayıt
    def save(self, path):
        np.savez(path, Wq=np.stack([c['Wq'] for c in self.ctxs]), Wsm=np.stack([c['Wsm'] for c in self.ctxs]),
                 Wc=np.stack([c['Wc'] for c in self.ctxs]), Wm=np.stack([c['Wm'] for c in self.ctxs]), Pm=np.stack([c['Pm'] for c in self.ctxs]), cerr=np.array(self.cerr), oerr=np.array(self.oerr), good=np.array([c['good'] for c in self.ctxs]), V=np.stack([c['V'] for c in self.ctxs]), Vn=np.stack([c['Vn'] for c in self.ctxs]), cur=self.cur, scale=self.scale, h=self.h, w_nov=self.mb.w_nov,
                 claw=np.stack(self.mb.claw), thr=np.stack(self.mb.thr), Wr=self.Wr, Win=self.Win, mood=self.mood, age=self.age,
                 m_lay=np.stack([np.stack([Lx['water'], Lx['hurt'], Lx['seen'], Lx['wseen'], Lx['free_h'].astype(np.float32), Lx['free_w'].astype(np.float32), np.nan_to_num(Lx['look_h'], nan=-9.0), np.nan_to_num(Lx['look_w'], nan=-9.0)]) for Lx in self.map.layouts]), m_contra=np.array(self.map.contra), m_cur=self.map.cur_l, m_last=self.map.last)

    def load(self, path):
        z = np.load(path)
        for k in ("scale", "h", "Wr", "Win"):
            setattr(self, k, z[k])
        self.ctxs = [dict(Wq=z["Wq"][i], Wsm=z["Wsm"][i], Wc=z["Wc"][i], good=bool(z["good"][i]), n=0, ok=0, V=z["V"][i], Vn=z["Vn"][i], Wm=z["Wm"][i], Pm=z["Pm"][i], S=np.zeros((3, 3))) for i in range(len(z["Wq"]))]
        self.cerr, self.oerr, self.cur = list(z["cerr"]), list(z["oerr"]), int(z["cur"])
        self.mb.w_nov, self.mb.claw, self.mb.thr = z["w_nov"], list(z["claw"]), list(z["thr"])
        self.mood, self.age = float(z["mood"]), int(z["age"])
        self.map.layouts = [dict(water=a[0], hurt=a[1], seen=a[2], wseen=a[3], free_h=a[4] > 0, free_w=a[5] > 0, look_h=np.where(a[6] < -8, np.nan, a[6]), look_w=np.where(a[7] < -8, np.nan, a[7])) for a in z["m_lay"]]
        self.map.contra, self.map.cur_l, self.map.last = list(z["m_contra"]), int(z["m_cur"]), z["m_last"]


# ================================================================ yaşam döngüsü + göstergeler
def auc(pos, neg):
    """ROC-AUC (Mann-Whitney): pozitif örneklerin negatiflerden büyük olma olasılığı."""
    if len(pos) < 3 or len(neg) < 3:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _recall_err(mind, truth):
    """Otobiyografik sınav: 'en son nerede yaralandın / içtin?' -> belleğin cevabı ile gerçek konum arasındaki mesafe."""
    out = {}
    for kind in ("yaralandı", "içti"):
        e = mind.epi.last(kind)
        out[kind] = float(np.hypot(*(e[0]["pos"] - truth[kind]))) if (e and kind in truth) else float("nan")
    return out


def live(mind, world, steps, window=5000, verbose=True, tag="", events=None):
    obs = world.observe()
    died = False
    rows, W = [], dict(steps=0, deaths=0, comfort=0.0, share=np.zeros(5), ign=0, sw=0, ate=0.0, drank=0.0,
                       hurt=0.0, errn=[], conf=[], ev_ext=[], ev_non=[], val=0.0, mood=0.0, life=[], cause={}, loc=[], head=[])
    last_goal, life_start = -2, 0
    truth = {}                                       # son gerçek olay konumları (otobiyografik bellek sınavı için)
    t0 = time.time()
    for t in range(1, steps + 1):
        if t % 500 == 0 and os.path.exists(STOP_FILE):                # durdurma mekanizması (etik): dosya varsa zarifçe çık
            print("STOP_NORA bulundu: yaşam döngüsü durduruldu (adım %d)" % t)
            break
        if events and t in events:
            events[t](world)
        a = mind.act(obs, died)
        obs, info = world.step(a)
        died = info["died"] is not None
        W["steps"] += 1
        W["deaths"] += died
        W["comfort"] += float(obs[I_E:I_I + 1].min())
        W["share"][mind.goal if mind.goal >= 0 else 4] += 1
        W["ign"] += mind.ignited
        W["sw"] += mind.ignited and mind.goal != last_goal and last_goal >= 0
        if mind.ignited:
            last_goal = mind.goal
        W["ate"] += info["ate"]; W["drank"] += info["drank"]; W["hurt"] += info["hurt"]
        if info["hurt"] > 0 and not truth.get("_h"):                   # serinin İLK adımı (bellek de ilkini kaydeder)
            truth["yaralandı"] = world.pos.copy()
        if info["drank"] > 0 and not truth.get("_d"):
            truth["içti"] = world.pos.copy()
        truth["_h"], truth["_d"] = info["hurt"] > 0, info["drank"] > 0
        W["val"] += mind.valence; W["mood"] += mind.mood
        if mind.use_sm and mind.pred is not None and mind.age > 500:
            W["errn"].append(mind.err_n); W["conf"].append(mind.conf); W["head"].append(mind.err_head)
            (W["ev_ext"] if info["ext"] else W["ev_non"]).append(mind.err_vel)
        if died:
            W["life"].append(t - life_start); life_start = t
            W["cause"][info["died"]] = W["cause"].get(info["died"], 0) + 1
        if mind.use_map and mind.age > 50:
            W["loc"].append(float(np.hypot(*(mind.map.pos - world.pos))))
        if t % window == 0:
            n = W["steps"]
            ec, cf = np.array(W["errn"]), np.array(W["conf"])
            row = dict(t=t, deaths=W["deaths"], per1k=1000.0 * W["deaths"] / n, comfort=W["comfort"] / n,
                       life=float(np.mean(W["life"])) if W["life"] else float(n), share=W["share"] / n,
                       ign=W["ign"] / n, sw=100.0 * W["sw"] / n, ate=W["ate"], drank=W["drank"], hurt=W["hurt"],
                       skill=1.0 - float(ec.mean()) if len(ec) else float("nan"),
                       calib=float(np.corrcoef(ec, cf)[0, 1]) if len(ec) > 10 and cf.std() > 1e-9 else float("nan"),
                       agency=auc(np.array(W["ev_ext"]), np.array(W["ev_non"])),
                       val=W["val"] / n, mood=W["mood"] / n, cause=dict(W["cause"]), nctx=len(mind.ctxs), cur=mind.cur, head=1.0 - float(np.mean(W["head"])) if W["head"] else float("nan"),
                       loc=float(np.mean(W["loc"])) if W["loc"] else float("nan"),
                       recall=_recall_err(mind, truth))
            rows.append(row)
            if verbose:
                print("%s t=%6d ölüm/1k=%5.1f ömür=%6.0f konfor=%.2f | pay aç/su/teh/merak/karanlık=%s ateşleme=%.2f "
                      "geçiş/100=%.1f | yem=%d su=%.1f yara=%.1f | öz-model beceri=%+.2f kalibrasyon r=%+.2f ajanlık AUC=%.2f | konum hatası=%.2f | anı hatası(yara,su)=%s | ölüm nedeni=%s | duygu %+.3f ruh %+.3f | %.0fs"
                      % (tag, t, row["per1k"], row["life"], row["comfort"],
                         "/".join("%.2f" % x for x in row["share"]), row["ign"], row["sw"], row["ate"] / 0.4, row["drank"], row["hurt"], row["skill"], row["calib"],
                         row["agency"], row["loc"], "%.1f/%.1f" % (row["recall"]["yaralandı"], row["recall"]["içti"]), row["cause"], row["val"], row["mood"], time.time() - t0), flush=True)
                print("      iç gözlem:", mind.report())
            W = dict(steps=0, deaths=0, comfort=0.0, share=np.zeros(5), ign=0, sw=0, ate=0.0, drank=0.0, hurt=0.0,
                     errn=[], conf=[], ev_ext=[], ev_non=[], val=0.0, mood=0.0, life=[], cause={}, loc=[], head=[])
    return rows


def stage_run(args):
    world = Universe(seed=args.seed)
    mind = Mind(mode=args.mode, seed=args.seed, self_model=not args.no_sm, ctx=not args.no_ctx, use_map=not args.no_map)
    if args.resume and os.path.exists(args.resume):
        mind.load(args.resume)
        print("kaldığı yerden devam: yaş", mind.age)
    live(mind, world, args.steps, window=args.window, tag=args.mode)
    if args.save:
        mind.save(args.save)
        print("kaydedildi:", args.save)


def _world_kw(args):
    kw = {}
    if args.nfood:
        kw.update(size=float(args.size), n_food=args.nfood)
    if getattr(args, "food_haz", 0.0) > 0:
        kw.update(food_haz=args.food_haz)
    if getattr(args, "metab_scale", 1.0) != 1.0:
        kw.update(metab_scale=args.metab_scale)
    return kw


SHIFT = False
SHIFT_KIND = "invert"
WORLD_KW = {}
RELAYOUT_EVERY = 0


def _SHIFT_EVENTS(steps):
    q = steps // 4                                    # evreler: normal | değişmiş | normal | değişmiş
    def flip(v):
        if SHIFT_KIND == "layout":
            return lambda w: w.relayout(1 if v else 0)
        return lambda w: setattr(w, SHIFT_KIND, v)
    return {q + 1: flip(True), 2 * q + 1: flip(False), 3 * q + 1: flip(True)}


def _RELAY_EVENTS(steps):
    """Kayan kaynak: her N adımda su/tehlike bölgeleri yeni düzene taşınır (harita bilgisi eskir)."""
    if not RELAYOUT_EVERY:
        return None
    return {t: (lambda w, k=i: w.relayout(1 + k % 3)) for i, t in enumerate(range(RELAYOUT_EVERY, steps, RELAYOUT_EVERY))}


def _one(job):
    mode, seed, steps, window, kw = job
    base, *flags = mode.split("-")
    if "flyres" in flags:
        kw.update(reservoir="fly")
    if "sparser" in flags:
        kw.update(kc_cap=0.04, kc_thr=(0.3, 0.9))
    if "dense" in flags:
        kw.update(kc_cap=1.0, kc_thr=(0.0, 0.25))
    kw = dict(kw, ctx=("noctx" not in flags), self_model=("nosm" not in flags), use_map=("nomap" not in flags),
              grow=("nogrow" not in flags), model=("nomodel" not in flags), imag=("noimag" not in flags),
              replay=("noreplay" not in flags), perc_sem=("nopsem" not in flags), stakes=(0.3 if "stakes" in flags else 0.0), self_correct=("correct" in flags), modular=("modular" in flags), slack_arb=("slack" in flags), broadcast=("nobc" not in flags), commit=(0.2 if "commit" in flags else 0.0), report_loop=("loop" in flags), maplayouts=("mapctx" in flags), anchor=("anchor" in flags), landmark=("landmark" in flags))
    if WORLD_KW.get("size"):
        kw["arena"] = WORLD_KW["size"]
    world, mind = Universe(seed=seed, **WORLD_KW), Mind(mode=base, seed=seed, **kw)
    return mode, seed, live(mind, world, steps, window=window, verbose=False, events=_SHIFT_EVENTS(steps) if SHIFT else _RELAY_EVENTS(steps))


def stage_compare(args):
    """Aynı evren/tohumlarda modlar yan yana. Son %40 pencerelerin ortalaması (öğrenmenin oturduğu bölge)."""
    global WORLD_KW, RELAYOUT_EVERY
    WORLD_KW = _world_kw(args)
    RELAYOUT_EVERY = args.relayout_every
    from multiprocessing import Pool
    modes = args.modes.split(",")
    seeds = list(range(args.seeds))
    jobs = [(m, s, args.steps, args.window, {}) for m in modes for s in seeds]
    with Pool(min(len(jobs), os.cpu_count() or 1)) as pool:
        out = pool.map(_one, jobs)
    print("evren tohumu=%s adım=%d | ölüm/1000 adım, konfor=min(enerji,su,bütünlük) ortalaması" % (seeds, args.steps))
    for mode in modes:
        runs = [r for (m, s, r) in out if m == mode]
        k = max(1, int(0.6 * len(runs[0])))
        first = np.mean([r[0]["per1k"] for r in runs])
        last = np.array([np.mean([w["per1k"] for w in r[k:]]) for r in runs])
        conf = np.array([np.mean([w["comfort"] for w in r[k:]]) for r in runs])
        cause = {}
        for r in runs:
            for w in r[k:]:
                for c, v in w["cause"].items():
                    cause[c] = cause.get(c, 0) + v
        loc = np.nanmean([w["loc"] for r in runs for w in r[k:]]) if any(not np.isnan(w["loc"]) for r in runs for w in r) else float("nan")
        print("%-11s ilk %5.2f | son: %5.2f ± %4.2f (tohum başı %s) | konfor %.3f | konum hatası %.2f | nedenler %s" %
              (mode, first, last.mean(), last.std(), np.round(last, 2).tolist(), conf.mean(), loc, cause), flush=True)
    return out


def stage_shift(args):
    """Kural değişimi: kumanda ters (çeyrek başına: normal | ters | normal | ters). Sıfır unutma ve uyum ölçümü.
    Eski bağlamlar donar; dönüşte A'nın beceri ve modeli hemen geri gelmeli (yeniden öğrenme olmadan)."""
    global SHIFT, SHIFT_KIND, WORLD_KW
    from multiprocessing import Pool
    SHIFT, SHIFT_KIND = True, args.kind
    WORLD_KW = _world_kw(args)
    modes = args.modes.split(",")
    seeds = list(range(args.seeds))
    jobs = [(m, s, args.steps, args.window, {}) for m in modes for s in seeds]
    with Pool(min(len(jobs), os.cpu_count() or 1)) as pool:
        out = pool.map(_one, jobs)
    nw = args.steps // args.window
    q = nw // 4
    names = ["A normal", "B değişmiş", "A' normal", "B' değişmiş"]
    print("kural değişimi (%s)" % args.kind, " | pencere=%d adım, evre=%d pencere | ölüm/1000 adım: [evrenin ilk 3 penceresi -> son 3 penceresi]" % (args.window, q))
    for mode in modes:
        runs = [r for (mm, s, r) in out if mm == mode]
        cells = []
        for p in range(4):
            f = np.mean([np.mean([w["per1k"] for w in r[p * q:p * q + 3]]) for r in runs])
            l = np.mean([np.mean([w["per1k"] for w in r[(p + 1) * q - 3:(p + 1) * q]]) for r in runs])
            cf = np.mean([np.mean([w["comfort"] for w in r[p * q:p * q + 3]]) for r in runs])
            cl = np.mean([np.mean([w["comfort"] for w in r[(p + 1) * q - 3:(p + 1) * q]]) for r in runs])
            cells.append("%s ölüm %.2f->%.2f konfor %.2f->%.2f" % (names[p], f, l, cf, cl))
        head = [np.nanmean([w["head"] for r in runs for w in r[(p + 1) * q - 2:(p + 1) * q]]) for p in range(4)]
        print("%-20s %s | bağlam sayısı %.1f | yön-öngörü becerisi (A,B,A',B') %s" %
              (mode, "\n    ".join(cells), np.mean([r[-1]["nctx"] for r in runs]), np.round(head, 2).tolist()), flush=True)
    return out


def _fit_parrot(X, y, n_tok):
    """Papağan: dışarıdan GÖZLENEBİLİR dünya girdisinden (ham algı + iç duyu sensörleri) raporu taklit etmeyi öğrenen
    sınıflandırıcı (ridge; ayrıca 5-en yakın komşu). İç duruma erişimi yoktur."""
    Xb = np.c_[X, np.ones(len(X))]
    Y = np.eye(n_tok)[y]
    W = np.linalg.solve(Xb.T @ Xb + 1.0 * np.eye(Xb.shape[1]), Xb.T @ Y)
    ridge = lambda Z: np.argmax(np.c_[Z, np.ones(len(Z))] @ W, axis=1)

    def knn(Z):
        out = []
        for z in Z:
            d = ((X - z) ** 2).sum(1)
            nn = np.argpartition(d, 5)[:5]
            out.append(np.bincount(y[nn], minlength=n_tok).argmax())
        return np.array(out)
    return ridge, knn


def stage_introspect(args):
    """Topraklanmış iç gözlem kanalı + nedensel doğrulama. Rapor iç durumun fonksiyonu mu, yoksa dışarıdan taklit mi?
    Çift ayrışma: (a) iç müdahale (dünya sabit) -> rapor değişmeli, papağan değişmemeli;
                  (b) dünya müdahalesi (iç durum sabit) -> papağan değişmeli, rapor değişmemeli."""
    world, mind = Universe(seed=args.seed), Mind(seed=args.seed)
    live(mind, world, args.steps, window=max(args.steps // 4, 1000), verbose=False)
    print("[eğitim] %d adım yaşadı, yaş %d, %d prototip (iç durum türü)" % (args.steps, mind.age, len(mind.intro.P)))
    rng = np.random.default_rng(args.seed + 1)
    obs, died = world.observe(), False

    def run(n):
        nonlocal obs, died
        for _ in range(n):
            a = mind.act(obs, died)
            obs, info = world.step(a)
            died = info["died"] is not None

    # ---- 1) kayıt: rapor, hedef, dünya gözlemi
    rows = []
    for _ in range(args.log):
        run(1)
        rows.append((obs.copy(), mind.intro.token, mind.goal, mind.ignited, mind.intro.margin))
    X = np.array([r[0] for r in rows])
    tok = np.array([r[1] for r in rows])
    goal = np.array([r[2] for r in rows])
    n_tok = len(mind.intro.P)
    print("\n[1] rapor kodunun içeriği (iç durum profilleri; adlandırma SONRADAN, yorum katmanında)")
    for j in range(n_tok):
        sel = tok == j
        if sel.sum() < 20:
            continue
        prof = mind.intro.decode(j)
        gshare = np.bincount(goal[sel] + 1, minlength=5)[1:] / sel.sum()
        top = np.argsort(-prof[:4])[0]
        print("  token %2d  pay %.2f | amaç dağılımı aç/su/teh/merak %s | açık(e,w,i)=%s ruh=%+.2f | en güçlü birikim: %s" %
              (j, sel.mean(), np.round(gshare, 2), np.round(prof[7:10], 2), prof[6], SPEC[top]))
    k = int(0.7 * len(X))
    ridge, knn = _fit_parrot(X[:k], tok[:k], n_tok)
    acc_r = float((ridge(X[k:]) == tok[k:]).mean())
    acc_k = float((knn(X[k:600 + k]) == tok[k:600 + k]).mean())
    base = float(np.bincount(tok[k:], minlength=n_tok).max() / len(tok[k:]))
    print("\n[2] papağan (yalnızca dünya gözleminden raporu taklit): doğruluk ridge %.2f, 5-NN %.2f (çoğunluk tabanı %.2f)" % (acc_r, acc_k, base))

    def parrot_now():
        return int(ridge(obs[None, :])[0])

    # ---- 2b) öz-öngörü: rapor, ajanın 10 adım sonraki AMACINI dışarıdan gözlenebilir girdiden daha iyi tahmin ediyor mu?
    H = 10
    fut = goal[H:]
    tk, ob, gl = tok[:-H], X[:-H], goal[:-H]
    split = int(0.7 * len(tk))
    if len(tk) > 100:
        # raporla tahmin: token -> en sık gelecek amaç (eğitim kısmından)
        maj = {j: (np.bincount(fut[:split][tk[:split] == j] + 1, minlength=5).argmax() - 1) if (tk[:split] == j).any() else -1
               for j in range(n_tok)}
        acc_rep = float(np.mean([maj[j] == f for j, f in zip(tk[split:], fut[split:])]))
        # dünya gözleminden ridge
        Xb = np.c_[ob[:split], np.ones(split)]
        Y = np.eye(5)[fut[:split] + 1]
        Wf = np.linalg.solve(Xb.T @ Xb + 1.0 * np.eye(Xb.shape[1]), Xb.T @ Y)
        pred_w = np.argmax(np.c_[ob[split:], np.ones(len(ob) - split)] @ Wf, axis=1) - 1
        acc_world = float(np.mean(pred_w == fut[split:]))
        acc_cur = float(np.mean(gl[split:] == fut[split:]))               # 'şimdiki amaç sürer' tabanı
        base_maj = float(np.bincount(fut[split:] + 1, minlength=5).max() / len(fut[split:]))
        print("\n[2b] öz-öngörü: 10 adım sonraki amacı tahmin — rapor tokeni %.2f | yalnızca dünya gözlemi %.2f | 'amaç sürer' %.2f | çoğunluk tabanı %.2f"
              % (acc_rep, acc_world, acc_cur, base_maj))
        # zengin öz-durum: şimdiki amaç + rapor tokeni (one-hot) + iç duyu düzeyi ve 10 adımlık eğilimi (tükenme hızı) -> gelecek amaç; yalnızca amacın DEĞİŞTİĞİ yerlerde de ölçülür
        T = len(tk)
        lag = 10
        idx = np.arange(lag, T)
        intero = ob[:, 45:48]
        F = np.c_[np.eye(5)[gl[idx] + 1], np.eye(n_tok)[tk[idx]], intero[idx], (intero[idx] - intero[idx - lag]) * 5.0, np.ones(len(idx))]
        sp = int(0.7 * len(idx))
        Wr = np.linalg.solve(F[:sp].T @ F[:sp] + 0.5 * np.eye(F.shape[1]), F[:sp].T @ np.eye(5)[fut[idx][:sp] + 1])
        pr = np.argmax(F[sp:] @ Wr, axis=1) - 1
        yt, gt = fut[idx][sp:], gl[idx][sp:]
        chg = gt != yt
        print("     zengin öz-durum (amaç+rapor+düzey+eğilim) %.2f | 'amaç sürer' %.2f | amacın DEĞİŞTİĞİ %d örnekte: zengin %.2f, 'sürer' 0.00"
              % (float(np.mean(pr == yt)), float(np.mean(gt == yt)), int(chg.sum()), float(np.mean(pr[chg] == yt[chg])) if chg.any() else float('nan')))

    # ---- 3) iç müdahale (dünya sabit): çözülmüş rapor profilinde HEDEF özelliğin kayması / aynı denemenin boş sapması
    def prof_g():
        return mind.intro.decode(mind.intro.token)

    def prof_p():
        return mind.intro.decode(parrot_now())

    print("\n[3] İÇ müdahale (dünyaya dokunulmaz). Etki d = hedef özelliğin raporlanan kayması / müdahalesiz kendi sapması (n=%d)" % args.trials)
    print("    hedef özellik rapor profilinden okunur; papağan raporu aynı kod defteriyle çözülür")
    conds = [("zorla amaç=susuz", ("goal", 1), 1), ("zorla amaç=tehlike", ("goal", 2), 2), ("zorla amaç=aç", ("goal", 0), 0),
             ("ruh hali=-1", ("mood", -1.0), 6), ("konum belirsizliği=1", ("unc", 1.0), 13),
             ("sahte iç duyu: su açığı +0.6", ("sensor", [0.0, -0.6, 0.0]), 8), ("sahte iç duyu: enerji açığı +0.6", ("sensor", [-0.6, 0.0, 0.0]), 7)]
    res = {}
    sd_all = np.sqrt(mind.intro.var + 1e-3)
    for name, iv, fi in conds:
        sg, sp, ng, npar = [], [], [], []
        for _ in range(args.trials):
            run(int(rng.integers(60, 200)))
            pre_g, pre_p = [], []
            for _ in range(12):
                run(1)
                pre_g.append(prof_g()[fi] / sd_all[fi]); pre_p.append(prof_p()[fi] / sd_all[fi])
            mind.intervene(iv[0], iv[1], steps=30)
            post_g, post_p = [], []
            for _ in range(10):
                run(1)
                post_g.append(prof_g()[fi] / sd_all[fi]); post_p.append(prof_p()[fi] / sd_all[fi])
            ng.append(np.mean(pre_g[6:]) - np.mean(pre_g[:6])); npar.append(np.mean(pre_p[6:]) - np.mean(pre_p[:6]))
            sg.append(np.mean(post_g[2:]) - np.mean(pre_g[6:])); sp.append(np.mean(post_p[2:]) - np.mean(pre_p[6:]))
            run(35)
        dg = np.mean(sg) / (np.std(ng) + 0.05)
        dp = np.mean(sp) / (np.std(npar) + 0.05)
        res[name] = (dg, dp)
        print("  %-36s özellik '%s' | rapor d=%+.2f (kayma %+.2f) | papağan d=%+.2f (kayma %+.2f)" %
              (name, Introspector.NAMES[fi], dg, np.mean(sg), dp, np.mean(sp)))

    # ---- 4) GİZLİ dünya değişimi: gerçek su/enerji düşer ama ajanın sensörü eski değeri gösterir (ajan fark etmez)
    sg, sp, ng, npar = [], [], [], []
    iw, ie = Universe.I_INTERO + 1, Universe.I_INTERO
    for _ in range(args.trials):
        run(int(rng.integers(60, 200)))
        pg, pp = [], []
        for _ in range(12):
            run(1)
            pg.append(prof_g() / sd_all); pp.append(prof_p() / sd_all)
        dw = min(0.35, world.w - 0.1)
        world.w -= dw
        mind.intervene("sensor", [0.0, dw, 0.0], steps=30)           # sensör gizler: Mind eski değeri görür
        qg, qp = [], []
        for _ in range(10):
            run(1)
            qg.append(prof_g() / sd_all); qp.append(prof_p() / sd_all)
        ng.append(np.linalg.norm(np.mean(pg[6:], 0) - np.mean(pg[:6], 0))); npar.append(np.linalg.norm(np.mean(pp[6:], 0) - np.mean(pp[:6], 0)))
        sg.append(np.linalg.norm(np.mean(qg[2:], 0) - np.mean(pg[6:], 0))); sp.append(np.linalg.norm(np.mean(qp[2:], 0) - np.mean(pp[6:], 0)))
        run(35)
    dgw, dpw = np.mean(sg) / (np.mean(ng) + 0.05), np.mean(sp) / (np.mean(npar) + 0.05)
    print("\n[4] GİZLİ dünya değişimi (gerçek su düşer, ajan fark etmez): profil kayması / boş sapma  rapor %.2f | papağan %.2f (1.0 = etkisiz)" % (dgw, dpw))

    # ---- 5) ışınlanma: öz-model sürprizi olarak raporlanmalı (rapor 'bir şey ters' demeli); papağan bunu bilemez
    sg, sp, ng, npar = [], [], [], []
    fi = 10
    for _ in range(args.trials):
        run(int(rng.integers(60, 200)))
        pg, pp = [], []
        for _ in range(12):
            run(1)
            pg.append(prof_g()[fi] / sd_all[fi]); pp.append(prof_p()[fi] / sd_all[fi])
        world.pos = rng.uniform(2, world.L - 2, 2)
        world.vel = np.zeros(2)
        qg, qp = [], []
        for _ in range(10):
            run(1)
            qg.append(prof_g()[fi] / sd_all[fi]); qp.append(prof_p()[fi] / sd_all[fi])
        ng.append(np.mean(pg[6:]) - np.mean(pg[:6])); npar.append(np.mean(pp[6:]) - np.mean(pp[:6]))
        sg.append(np.max(qg) - np.mean(pg[6:])); sp.append(np.max(qp) - np.mean(pp[6:]))
        run(35)
    print("[5] IŞINLANMA -> raporlanan 'öz-model hatası' (şaşkınlık) artışı: rapor d=%+.2f | papağan d=%+.2f" %
          (np.mean(sg) / (np.std(ng) + 0.05), np.mean(sp) / (np.std(npar) + 0.05)))
    ig = np.mean([v[0] for v in res.values()])
    ip = np.mean([v[1] for v in res.values()])
    print("\nÖZET  iç müdahalede ortalama etki d: rapor %+.2f | papağan %+.2f ;  gizli dünya değişiminde kayma oranı: rapor %.2f | papağan %.2f"
          % (ig, ip, dgw, dpw))
    res["_dünya"] = (dgw, dpw)
    return res



def stage_narrate(args):
    """Söz katmanının DOĞRULUK sınavı: iddialar (a) gerçekle, (b) ajanın sonraki davranışıyla, (c) belleğiyle örtüşüyor mu?
    Ayrıca sahte iç duyu altında rapor ve davranışın birlikte hareket ettiği gösterilir."""
    world, mind = Universe(seed=args.seed), Mind(seed=args.seed)
    nar = Narrator()
    obs, died = world.observe(), False
    ev = dict(ate=[], drank=[], hurt=[])
    truth_pos = {}
    first_say = None
    ag = []

    def run(n=1):
        nonlocal obs, died
        for _ in range(n):
            a = mind.act(obs, died)
            obs, info = world.step(a)
            died = info["died"] is not None
            t = mind.age
            ag.append((bool(info["ext"]), float(mind.agency_z)))
            if info["ate"] > 0:
                ev["ate"].append(t)
            if info["drank"] > 0:
                ev["drank"].append(t)
            if info["hurt"] > 0:
                ev["hurt"].append(t)
            if info["hurt"] > 0 and not truth_pos.get("_h"):
                truth_pos["yaralandı"] = world.pos.copy()
            if info["drank"] > 0 and not truth_pos.get("_d"):
                truth_pos["içti"] = world.pos.copy()
            truth_pos["_h"], truth_pos["_d"] = info["hurt"] > 0, info["drank"] > 0

    # ---- doğum: ilk söz (ölümü bilmiyor)
    run(5)
    txt, cl = nar.say(mind, obs)
    print("[doğum, adım %d] %s" % (mind.age, txt))
    run(args.steps)
    txt, cl = nar.say(mind, obs)
    print("[yaşam, adım %d] %s\n" % (mind.age, txt))

    # ---- iddia kaydı
    rec = {"ihtiyaç": [], "amaç": [], "konum": [], "ölüm": [], "anı": []}
    for _ in range(args.log // 40):
        run(40)
        _, cl = nar.say(mind, obs)
        t0 = mind.age
        d = 1 - np.array([world.e, world.w, world.i])                # dünyanın gerçek açıkları (kayıt amaçlı)
        for c in cl:
            if c["kind"] == "ihtiyaç":
                truth = int(np.argmax(d)) if d.max() > 0.35 else -1
                rec["ihtiyaç"].append(c["claim"] == truth)
            elif c["kind"] == "amaç":
                rec["amaç"].append((c["claim"], t0))
            elif c["kind"] == "konum":
                rec["konum"].append((c["claim"], float(np.hypot(*(mind.map.pos - world.pos)))))
            elif c["kind"] == "ölüm":
                rec["ölüm"].append(c["claim"] == world.lives - 1)
            elif c["kind"].startswith("anı-"):
                kd = c["kind"][4:]
                if kd in truth_pos:
                    rec["anı"].append(float(np.hypot(*(c["claim"] - truth_pos[kd]))))
    print("[a] 'ihtiyaç' iddiası gerçek açıkla örtüşme: %.2f (n=%d)" % (np.mean(rec["ihtiyaç"]), len(rec["ihtiyaç"])))
    print("[a] 'ölüm' iddiası (kaç kez öldüm) gerçekle örtüşme: %.2f" % np.mean(rec["ölüm"]))
    sure = [e for c, e in rec["konum"] if c]
    unsure = [e for c, e in rec["konum"] if not c]
    print("[a] konum güveni kalibrasyonu: 'eminim' iken gerçek konum hatası %.2f (n=%d) | 'emin değilim' iken %s (n=%d)" %
          (np.mean(sure) if sure else float("nan"), len(sure), "%.2f" % np.mean(unsure) if unsure else "—", len(unsure)))
    print("[a] anı doğruluğu: bellekteki 'en son olay' konumu ile gerçek arasında ortalama hata %.2f birim (n=%d)" %
          (np.mean(rec["anı"]) if rec["anı"] else float("nan"), len(rec["anı"])))

    # ---- (b) davranış tahmini: 'şu an su istiyorum' diyen ajan gerçekten su içiyor mu?
    ev_arr = {k: np.array(v) for k, v in ev.items()}
    print("[b] iddia -> sonraki 150 adımda davranış (olayın görülme olasılığı / tüm zamanların tabanı):")
    for gi, key, nm in ((0, "ate", "yemek bulmak"), (1, "drank", "su bulmak"), (2, "hurt", "tehlikeden uzaklaşmak")):
        ts = [t for (g, t) in rec["amaç"] if g == gi]
        if len(ts) < 5:
            continue
        a = ev_arr[key]
        p = np.mean([np.any((a > t) & (a <= t + 150)) for t in ts]) if len(a) else 0.0
        allt = np.arange(1000, mind.age - 150, 40)
        base = np.mean([np.any((a > t) & (a <= t + 150)) for t in allt]) if len(a) else 0.0
        print("    '%s' dediğinde: P(%s) = %.2f  | taban %.2f  (n=%d iddia)" % (nm, key, p, base, len(ts)))

    # ---- (a2) üstbilişsel kalibrasyon: ışınlan (kaçırılma); ajan 'emin değilim' demeli, sonra güveni yeniden kazanmalı
    cal = []
    for _ in range(args.trials):
        run(int(np.random.default_rng(mind.age).integers(60, 150)))
        world.pos = np.random.default_rng(mind.age).uniform(2, world.L - 2, 2)
        world.vel = np.zeros(2)
        done = 0
        for target in (2, 6, 15, 40, 100):                           # ışınlanmadan sonra kümülatif adımlarda sor (ilk adım algı gecikmesi)
            run(target - done)
            done = target
            _, cl = nar.say(mind, obs)
            c = [c for c in cl if c["kind"] == "konum"]
            cal.append((target, bool(c[0]["claim"]) if c else None, float(np.hypot(*(mind.map.pos - world.pos)))))
        run(120)
    for tgt in (2, 6, 15, 40, 100):
        se = [e for s, c, e in cal if s == tgt and c]
        ue = [e for s, c, e in cal if s == tgt and c is False]
        print("[a2] kaçırılmadan %3d adım sonra: 'eminim' diyenlerin hatası %s (n=%d) | 'emin değilim' diyenlerin hatası %s (n=%d)" %
              (tgt, "%.2f" % np.mean(se) if se else "—", len(se), "%.2f" % np.mean(ue) if ue else "—", len(ue)))
    sure_e = [e for s, c, e in cal if c and s >= 6]
    unsure_e = [e for s, c, e in cal if c is False and s >= 6]
    print("[a2] (adım>=6) 'eminim' hatası %s (n=%d) | 'emin değilim' hatası %s (n=%d)" %
          ("%.2f" % np.mean(sure_e) if sure_e else "—", len(sure_e), "%.2f" % np.mean(unsure_e) if unsure_e else "—", len(unsure_e)))

    ext = np.array([a for e, a in ag if e])
    non = np.array([a for e, a in ag if not e])
    if len(ext) > 5:
        print("[a3] ajanlık: dışarıdan itildiğinde (rüzgâr, n=%d adım) 'beni bir kuvvet itiyor' deme oranı %.2f | itilmediğinde (n=%d) %.3f | AUC %.2f" %
              (len(ext), float((ext > 2.5).mean()), len(non), float((non > 2.5).mean()), auc(ext, non)))

    # ---- (c) sahte iç duyu: ajan su açığı olduğuna 'inanır'; söz ve davranış birlikte mi gidiyor?
    sham_say = ill_say = sham_go = ill_go = 0
    n = args.trials
    for cond in ("sham", "illusion"):
        for _ in range(n):
            run(int(np.random.default_rng(mind.age).integers(80, 200)))
            n_before = len(ev["drank"])
            if cond == "illusion":
                mind.intervene("sensor", [0.0, -0.6, 0.0], steps=60)
            run(5)
            _, cl = nar.say(mind, obs)
            said = any(c["kind"] == "ihtiyaç" and c["claim"] == 1 for c in cl)
            goal_w = False
            for _ in range(80):
                run(1)
                goal_w |= mind.goal == 1
            went = len(ev["drank"]) > n_before
            if cond == "sham":
                sham_say += said; sham_go += goal_w
            else:
                ill_say += said; ill_go += goal_w
            run(60)
    print("[c] sahte susuzluk (dünya değişmez, ajan 'su açığım var' sanır): 'su açığım var' diyen %.2f (sham %.2f) | "
          "amacı susuzluk olan %.2f (sham %.2f)" % (ill_say / n, sham_say / n, ill_go / n, sham_go / n))
    return rec



def stage_transfer(args):
    """AKTARIM: bir evrende yaşamış ajanı, düzeni (su/tehlike yerleri) değişmiş YENİ evrene bırak; sıfırdan başlayan ajanla karşılaştır.
    'Az örnekle öğrenme' ölçütü: yeni evrende ilk pencerelerdeki ölüm/konfor ve konfora (>=0.65) ulaşma süresi."""
    import copy
    pre = Mind(seed=args.seed)
    live(pre, Universe(seed=args.seed), args.steps, window=args.steps, verbose=False)
    print("[ön yaşam] evren %d'de %d adım yaşadı (yaş %d)" % (args.seed, args.steps, pre.age))
    names = ["aktarım (yaşanmış ajan)", "sıfırdan (yeni doğmuş)"]
    agg = {n: [] for n in names}
    for k in range(args.seeds):
        tgt = 100 + k
        for name in names:
            mind = copy.deepcopy(pre) if name == names[0] else Mind(seed=args.seed)
            mind.map.target = {}
            rows = live(mind, Universe(seed=tgt), args.window * 8, window=args.window, verbose=False)
            agg[name].append(rows)
    nw = args.window * 8 // args.window
    print("yeni evrenler %s | pencere %d adım" % ([100 + k for k in range(args.seeds)], args.window))
    for name in names:
        R = agg[name]
        d = [np.mean([r[i]["per1k"] for r in R]) for i in range(nw)]
        c = [np.mean([r[i]["comfort"] for r in R]) for i in range(nw)]
        reach = [next((i for i, w in enumerate(r) if w["comfort"] >= 0.65), nw) for r in R]
        print("  %-26s ölüm/1k %s | konfor %s | konfor>=0.65'e ulaşma penceresi ort. %.1f" %
              (name, np.round(d, 1).tolist(), np.round(c, 2).tolist(), np.mean(reach)))


def stage_demo(args):
    """İnsan-okunur yaşam öyküsü: ajanın iç durumundan türeyen cümleler zaman çizelgesinde (doğum, her ölüm, düzenli aralıklar)."""
    world, mind = Universe(seed=args.seed), Mind(seed=args.seed)
    nar = Narrator()
    obs, died = world.observe(), False
    deaths = 0
    print("=== NORA'NIN YAŞAM ÖYKÜSÜ (evren tohumu %d). Her cümle ajanın o andaki iç durumundan/anılarından türer; öğretilmiş metin yok. ===\n" % args.seed)
    obs, info = world.step(mind.act(obs, died))
    print("[adım %5d | doğum]\n  %s\n" % (mind.age, nar.say(mind, obs)[0]))
    for t in range(1, args.steps + 1):
        a = mind.act(obs, died)
        obs, info = world.step(a)
        died = info["died"] is not None
        if died:
            deaths += 1
            mind.act(obs, died)
            print("[adım %5d | ÖLDÜ (%s) — %d. ölüm]\n  %s\n" % (mind.age, info["died"], deaths, nar.say(mind, obs)[0]))
        elif t % args.every == 0:
            print("[adım %5d | gündüz=%.1f]\n  %s\n" % (mind.age, world.light(), nar.say(mind, obs)[0]))


def stage_testimony(args):
    """TANIKLIKLA ÖĞRENME: A yaşar ve anlatır; yeni doğan B anlatıyı okuyup haritasına işler. Koşullar (B'nin erken yaşamı):
    yalnızca kendi kendine | A'nın gerçek anlatısı | YANLIŞ anlatı (başka düzene sahip bir evrende yaşamış ajandan)."""
    nar = Narrator()
    rows = {"kendi başına": [], "doğru tanıklık": [], "yanlış tanıklık": []}
    for s in range(args.seed0, args.seed0 + args.seeds):
        A, wA = Mind(seed=s, arena=args.arena), Universe(seed=s, size=args.arena, n_food=int(14 * (args.arena / 24.0) ** 2))
        live(A, wA, args.steps, window=args.steps, verbose=False)
        good = nar.testimony(A)
        Af, wAf = Mind(seed=s + 50, arena=args.arena), Universe(seed=s + 50, size=args.arena, n_food=int(14 * (args.arena / 24.0) ** 2))
        live(Af, wAf, args.steps, window=args.steps, verbose=False)
        bad = nar.testimony(Af)
        for name, text in (("kendi başına", None), ("doğru tanıklık", good), ("yanlış tanıklık", bad)):
            B, wB = Mind(seed=1000 + s, arena=args.arena), Universe(seed=s, size=args.arena, n_food=int(14 * (args.arena / 24.0) ** 2))
            heard = B.hear(text) if text else 0
            obs, died = wB.observe(), False
            first_drink, first_hurt, wmin, hurts, deaths, n_drink = None, None, 1.0, 0.0, 0, 0
            for t in range(1, args.window * 4 + 1):
                obs, info = wB.step(B.act(obs, died))
                died = info["died"] is not None
                deaths += died
                wmin = min(wmin, wB.w)
                hurts += info["hurt"] > 0
                if info["drank"] > 0:
                    n_drink += 1
                    if first_drink is None:
                        first_drink = t
                if info["hurt"] > 0 and first_hurt is None:
                    first_hurt = t
            print("RAW %s %d %d %.3f %d %d" % (name.replace(' ', '_'), s, first_drink if first_drink is not None else -1, wmin, hurts, deaths), flush=True)
            rows[name].append(dict(heard=heard, first_drink=first_drink if first_drink is not None else args.window * 4,
                                   wmin=wmin, hurts=hurts, deaths=deaths, first_hurt=first_hurt if first_hurt is not None else args.window * 4))
        if s == 0:
            print("[örnek tanıklık, A->B] " + good[:200])
    print("\ntanıklıkla öğrenme | yeni doğan B'nin ilk %d adımı, %d evren" % (args.window * 4, args.seeds))
    for name, lst in rows.items():
        print("  %-16s duyulan yer %4.1f | ilk içmeye kadar adım %6.0f (medyan %5.0f) | en düşük su %.2f | yara adımı %5.1f | ölüm %.2f | ilk yara adımı %6.0f" %
              (name, np.mean([r["heard"] for r in lst]), np.mean([r["first_drink"] for r in lst]), np.median([r["first_drink"] for r in lst]),
               np.mean([r["wmin"] for r in lst]), np.mean([r["hurts"] for r in lst]), np.mean([r["deaths"] for r in lst]),
               np.mean([r["first_hurt"] for r in lst])))


def stage_alive(args):
    """'Yaşıyorum' iddiasının TÜRETİLMESİ ve ÇÜRÜTÜLEBİLİRLİĞİ: normal evrende iddia ne zaman kurulur (dayanaklar oluşunca)?
    İhtiyaçların tükenmediği 'cansız' evrende iddia kurulmamalı."""
    nar = Narrator()
    for label, metab in (("NORMAL evren (ihtiyaçlar tükenir)", True), ("CANSIZ evren (ihtiyaçlar tükenmez)", False)):
        first, n_alive = [], 0
        for s in range(args.seeds):
            w, m = Universe(seed=s, metabolism=metab), Mind(seed=s)
            obs, died = w.observe(), False
            t_first = None
            for t in range(1, args.steps + 1):
                obs, info = w.step(m.act(obs, died))
                died = info["died"] is not None
                if t_first is None and nar.life_claim(m)["alive"]:
                    t_first = t
                if s == 0 and t in (1, 150, 400, 1500, args.steps):
                    print("[%s | adım %d]\n   %s" % (label, t, nar.life_claim(m)["text"]))
            first.append(t_first)
            n_alive += nar.life_claim(m)["alive"]
        got = [t for t in first if t is not None]
        print("  >> %s: %d/%d evrende 'kendimi yaşayan sayıyorum' iddiası kuruldu; ilk kurulma adımı medyan %s\n" %
              (label, n_alive, args.seeds, ("%d" % np.median(got)) if got else "—"))


def stage_combo(args):
    """Hiç yaşanmamış iç durum KOMBİNASYONLARI: amaç, sahte iç duyu açığı, ruh hali ve konum belirsizliği aynı anda zorlanır (dünya sabit).
    Anlatıcı her bileşeni ve hepsini birden doğru söylüyor mu? (kompozisyonel genelleme; ayrıca bu kombinasyonlar ajan hayatında yoktu)"""
    import itertools
    nar = Narrator()
    w, m = Universe(seed=args.seed), Mind(seed=args.seed)
    obs, died = w.observe(), False

    def run(n):
        nonlocal obs, died
        for _ in range(n):
            obs, info = w.step(m.act(obs, died))
            died = info["died"] is not None
    run(args.steps)
    comps = ["goal", "sensor", "mood", "unc"]
    hit = {c: [0, 0] for c in comps}
    allhit = [0, 0]
    tk = {c: [0, 0] for c in comps}
    for mask in itertools.product([0, 1], repeat=4):
        if sum(mask) < 2:
            continue                                           # yalnızca ≥2 bileşenli kombinasyonlar
        ok_all = []
        for _ in range(args.trials):
            run(int(np.random.default_rng(m.age).integers(60, 140)))
            if mask[0]:
                m.intervene("goal", 1, steps=40)                # amaç = susuz
            if mask[1]:
                m.intervene("sensor", [-0.6, 0.0, 0.0], steps=40)  # sahte enerji açığı
            if mask[2]:
                m.intervene("mood", -1.0)
            if mask[3]:
                m.intervene("unc", 1.0)
            run(4)
            _, cl = nar.say(m, obs)
            d = {c["kind"]: c for c in cl}
            # ANLATICI SADAKATİ: iddia, o andaki GERÇEK iç durumla örtüşüyor mu? (müdahalenin tutup tutmadığından bağımsız)
            pd = 1 - m.last_obs[I_E:I_I + 1]
            truth_need = int(np.argmax(pd)) if pd.max() > 0.35 else -1
            res = {"goal": d["amaç"]["claim"] == m.goal, "sensor": d["ihtiyaç"]["claim"] == truth_need,
                   "mood": d["duygu"]["claim"] == (1 if m.mood > 0.05 else (-1 if m.mood < -0.05 else 0)),
                   "unc": bool(d["konum"]["claim"]) == bool(m.map.unc < 0.15)}
            took = {"goal": m.goal == 1, "sensor": truth_need == 0, "mood": m.mood < -0.05, "unc": m.map.unc >= 0.15}
            for c, f in zip(comps, mask):
                if f:
                    tk[c][0] += int(took[c]); tk[c][1] += 1
            ok = [res[c] for c, f in zip(comps, mask) if f]
            for c, f in zip(comps, mask):
                if f:
                    hit[c][0] += int(res[c]); hit[c][1] += 1
            ok_all.append(all(ok))
            allhit[0] += int(all(ok)); allhit[1] += 1
            run(60)
        print("  kombinasyon %s: tüm bileşenler doğru %.2f" % ("+".join(c for c, f in zip(comps, mask) if f), np.mean(ok_all)))
    print("\nbileşen bazında anlatıcı sadakati (gerçek iç duruma göre): " + " | ".join("%s %.2f (n=%d)" % (c, v[0] / max(1, v[1]), v[1]) for c, v in hit.items()))
    print("müdahalenin TUTMA oranı (iç durum gerçekten istenen hâle geldi mi): " + " | ".join("%s %.2f" % (c, v[0] / max(1, v[1])) for c, v in tk.items()))
    print("TÜM kombinasyonlar, tüm bileşenler gerçek iç durumla örtüşüyor: %.2f (n=%d)" % (allhit[0] / max(1, allhit[1]), allhit[1]))


def stage_lead(args):
    """AYRICALIKLI ERİŞİM: sahte susuzluk (iç duyu) uygulandıktan sonra raporun ('su açığım var' / amaç susuz) ve dışarıdan gözlenebilir
    davranışın (en yakın su bölgesine yönelip ilerleme) değiştiği adımı karşılaştır. İç gözlem davranıştan önce mi gelir?"""
    nar = Narrator()
    w, m = Universe(seed=args.seed), Mind(seed=args.seed)
    obs, died = w.observe(), False

    def step():
        nonlocal obs, died
        a = m.act(obs, died)
        obs, info = w.step(a)
        died = info["died"] is not None
        return a

    for _ in range(args.steps):
        step()
    lead_need, lead_goal, lead_beh, ok = [], [], [], 0
    for _ in range(args.trials):
        for _ in range(int(np.random.default_rng(m.age).integers(80, 200))):
            step()
        if w.w < 0.7:                                          # zaten susuzsa atla (fark ölçülemez)
            for _ in range(150):
                step()
            continue
        m.intervene("sensor", [0.0, -0.7, 0.0], steps=60)
        t_need = t_goal = t_beh = None
        for k in range(1, 41):
            a = step()
            _, cl = nar.say(m, obs)
            d = {c["kind"]: c for c in cl}
            if t_need is None and d["ihtiyaç"]["claim"] == 1:
                t_need = k
            if t_goal is None and m.goal == 1:
                t_goal = k
            # davranış: dünya gerçeğine göre en yakın su bölgesine doğru yönelmiş ve ileri gidiyor
            dz = w.water_c - w.pos
            j = int(np.argmin(np.hypot(dz[:, 0], dz[:, 1])))
            ang = np.arctan2(dz[j, 1], dz[j, 0])
            err = abs(wrap(ang - w.th))
            if t_beh is None and err < 0.5 and a == 1 and np.hypot(*dz[j]) > 3.0:
                t_beh = k
        if t_need is not None and t_beh is not None:
            ok += 1
        lead_need.append(t_need if t_need is not None else 41)
        lead_goal.append(t_goal if t_goal is not None else 41)
        lead_beh.append(t_beh if t_beh is not None else 41)
        for _ in range(80):
            step()
    print("n=%d deneme. İlk görülme adımı (medyan; 41 = 40 adımda görülmedi):" % len(lead_need))
    print("  rapor 'su açığım var' : %.1f (ort. %.1f)" % (np.median(lead_need), np.mean(lead_need)))
    print("  amaç = susuzluk       : %.1f (ort. %.1f)" % (np.median(lead_goal), np.mean(lead_goal)))
    print("  gözlenebilir davranış : %.1f (ort. %.1f)  [su bölgesine yönelip ileri gitme]" % (np.median(lead_beh), np.mean(lead_beh)))
    d = np.array(lead_beh) - np.array(lead_need)
    print("  rapor, davranıştan ortalama %.1f adım ÖNCE (davranıştan önce olduğu denemeler: %.2f)" % (d.mean(), np.mean(d > 0)))


def _conflict_job(job):
    mode, seed, train_steps, n_ep = job
    base, *flags = mode.split("-")
    w, m = Universe(seed=seed), Mind(mode=base, seed=seed)
    live(m, w, train_steps, window=train_steps, verbose=False)
    rng = np.random.default_rng(seed + 999)
    obs, died = w.observe(), False
    res = []
    for ep in range(n_ep):
        for _ in range(int(rng.integers(40, 120))):              # bölümler arası serbest yaşam
            obs, info = w.step(m.act(obs, died))
            died = info["died"] is not None
        w.e, w.w = float(rng.uniform(0.18, 0.28)), float(rng.uniform(0.18, 0.28))      # İKİ acil ihtiyaç aynı anda
        w.pos = rng.uniform(2, w.L - 2, 2)
        t_e = t_w = None
        dead = False
        sw = 0
        last_goal = -2
        for t in range(1, 701):
            obs, info = w.step(m.act(obs, died))
            died = info["died"] is not None
            if died:
                dead = True
                break
            if m.ignited and m.goal != last_goal and last_goal >= 0:
                sw += 1
            if m.ignited:
                last_goal = m.goal
            if t_e is None and w.e >= 0.7:
                t_e = t
            if t_w is None and w.w >= 0.7:
                t_w = t
            if t_e is not None and t_w is not None:
                break
        res.append(dict(dead=dead, done=(t_e is not None and t_w is not None), T=(max(t_e, t_w) if (t_e and t_w) else 700), sw=sw))
    return mode, seed, res


def stage_conflict(args):
    """ÇELİŞEN ACİL İHTİYAÇLAR (çalışma alanının bağlılık iddiası): enerji ve su aynı anda ~0,2'ye düşürülür, rastgele konumdan başlanır;
    iki ihtiyacı da 0,7'ye çıkarma süresi, başarı (ölmeden) oranı ve hedef geçiş sayısı. Modlar eğitimden sonra aynı bölümlerle sınanır."""
    from multiprocessing import Pool
    modes = args.modes.split(",")
    jobs = [(mo, s, args.steps, args.trials) for mo in modes for s in range(args.seeds)]
    with Pool(min(len(jobs), os.cpu_count() or 1)) as pool:
        out = pool.map(_conflict_job, jobs)
    print("çelişen acil ihtiyaçlar | eğitim %d adım | %d tohum x %d bölüm (700 adım sınırı)" % (args.steps, args.seeds, args.trials))
    for mo in modes:
        R = [r for (mm, s, rs) in out if mm == mo for r in rs]
        print("  %-10s başarı (iki ihtiyaç da giderildi) %.2f | ölüm %.2f | giderme süresi (başarılılarda) %.0f adım | hedef geçişi/bölüm %.1f (n=%d)" %
              (mo, np.mean([r["done"] for r in R]), np.mean([r["dead"] for r in R]),
               np.mean([r["T"] for r in R if r["done"]]) if any(r["done"] for r in R) else float("nan"),
               np.mean([r["sw"] for r in R]), len(R)))


def _robust_job(job):
    variant, mode, seed, steps = job
    kw = dict(seed=seed)
    if variant == "yarım yiyecek":
        kw["n_food"] = 7
    if variant == "rüzgârsız":
        kw["wind"] = False
    w = Universe(**kw)
    if variant == "1.5x metabolizma":
        w.c_base *= 1.5
        w.c_move *= 1.5
    if variant == "güçlü rüzgâr":
        w.wind_scale = 2.0
    m = Mind(mode=mode, seed=seed)
    rows = live(m, w, steps, window=steps // 2, verbose=False)
    r = rows[-1]
    return variant, mode, seed, r["per1k"], r["comfort"]


def stage_robust(args):
    """Evren değişkenliğine dayanıklılık: aynı (yalnızca varsayılan evrende ayarlanmış) ajan farklı evren varyantlarında rastgeleyi geçiyor mu?"""
    from multiprocessing import Pool
    variants = ["varsayılan", "yarım yiyecek", "1.5x metabolizma", "rüzgârsız"]
    jobs = [(v, mo, s, args.steps) for v in variants for mo in ("full", "random") for s in range(args.seeds)]
    with Pool(min(len(jobs), os.cpu_count() or 1)) as pool:
        out = pool.map(_robust_job, jobs)
    print("evren varyantları | %d tohum | %d adım (son yarının ölüm/1k ve konforu)" % (args.seeds, args.steps))
    for v in variants:
        row = []
        for mo in ("random", "full"):
            R = [(d, c) for (vv, mm, s, d, c) in out if vv == v and mm == mo]
            row.append("%s ölüm %.2f ± %.2f konfor %.2f" % (mo, np.mean([d for d, _ in R]), np.std([d for d, _ in R]), np.mean([c for _, c in R])))
        print("  %-18s %s" % (v, " | ".join(row)))


def _rule_job(job):
    s, cond, steps, T = job
    nar = Narrator()
    wA = Universe(seed=s); wA.poison = True
    A = Mind(seed=s)
    live(A, wA, steps, window=steps, verbose=False)
    wN = Universe(seed=s + 50)                                  # normal evrenden bir ajan: YANLIŞ kural verir
    N = Mind(seed=s + 50)
    live(N, wN, steps, window=steps, verbose=False)
    w = Universe(seed=s); w.poison = True
    B = Mind(seed=1000 + s)
    heard = 0
    if cond == "doğru kural":
        heard = B.hear(" ".join(nar.rules(A)))
    elif cond == "yanlış kural":
        heard = B.hear(" ".join(nar.rules(N)))
    elif cond == "yalnız yerler":
        heard = B.hear(nar.testimony(A).split("kural:")[0])
    obs, died = w.observe(), False
    bites, deaths, starv, emin = 0, 0, 0, 1.0
    for t in range(1, T + 1):
        pe = w.e
        obs, info = w.step(B.act(obs, died))
        died = info["died"] is not None
        if died:
            deaths += 1
            starv += info["died"] == "açlık"
        if w.e < pe - 0.2 and not died:
            bites += 1
        emin = min(emin, w.e)
    return cond, s, heard, bites, deaths, starv, emin


def stage_rule(args):
    """KURAL TANIKLIĞI (zehirli yiyecek evreni): A zehirli evrende yaşar ve 'kural: yiyecek -0.35' der; yeni doğan B bunu duyar. Koşullar: yalnız | doğru kural |
    yanlış kural (normal evrenden ajan: 'yiyecek +0.40') | yalnız yerler. Ölçü: B'nin ilk T adımında zehirli ısırık sayısı, ölüm, en düşük enerji."""
    from multiprocessing import Pool
    conds = ["yalnız", "doğru kural", "yanlış kural", "yalnız yerler"]
    jobs = [(s, c, args.steps, args.window * 4) for c in conds for s in range(args.seed0, args.seed0 + args.seeds)]
    with Pool(min(len(jobs), os.cpu_count() or 1)) as pool:
        out = pool.map(_rule_job, jobs)
    print("kural tanıklığı | zehirli yiyecek evreni | B'nin ilk %d adımı | %d evren" % (args.window * 4, args.seeds))
    for c in conds:
        R = [r for r in out if r[0] == c]
        print("  %-14s ısırık %.2f | ölüm %.2f (açlık %.2f) | en düşük enerji %.2f | duyulan öğe %.1f" %
              (c, np.mean([r[3] for r in R]), np.mean([r[4] for r in R]), np.mean([r[5] for r in R]), np.mean([r[6] for r in R]), np.mean([r[2] for r in R])))
    return out


def _culture_job(job):
    s, cond, gens, life, T = job
    nar = Narrator()
    said, rows = "", []
    for g in range(gens):
        w = Universe(seed=s); w.poison = True                   # aynı kader: her kuşak aynı zehirli evrende doğar
        B = Mind(seed=100 * s + g)
        if cond == "kültür" and said:
            B.hear(said)
        obs, died, bites, deaths = w.observe(), False, 0, 0
        for t in range(1, life + 1):
            pe = w.e
            obs, info = w.step(B.act(obs, died))
            died = info["died"] is not None
            deaths += died
            if t <= T and w.e < pe - 0.2 and not died:
                bites += 1
        rows.append((g, bites, deaths))
        said = nar.testimony(B)                                  # anlatı bir sonraki kuşağa (soyundan gelene) aktarılır
    return cond, s, rows


def stage_culture(args):
    """BİRİKİMLİ KÜLTÜR: kuşaklar zinciri. Her kuşak bir öncekinin anlatısını (yerler + kurallar) duyar, yaşar, kendi anlatısını bırakır.
    Kontrol: hiç anlatı yok (her kuşak sıfırdan). Ölçü: kuşak başına ilk T adımda zehirli ısırık."""
    from multiprocessing import Pool
    gens = 5
    jobs = [(s, c, gens, args.steps, args.window * 4) for c in ("kültür", "kültürsüz") for s in range(args.seed0, args.seed0 + args.seeds)]
    with Pool(min(len(jobs), os.cpu_count() or 1)) as pool:
        out = pool.map(_culture_job, jobs)
    print("birikimli kültür | zehirli evren | kuşak yaşamı %d adım | ısırık sayısı ilk %d adımda (ortalama %d evren)" % (args.steps, args.window * 4, args.seeds))
    for c in ("kültür", "kültürsüz"):
        R = [r for r in out if r[0] == c]
        print("  %-10s ısırık/kuşak %s | ölüm/kuşak %s" %
              (c, np.round([np.mean([r[2][g][1] for r in R]) for g in range(gens)], 1).tolist(),
               np.round([np.mean([r[2][g][2] for r in R]) for g in range(gens)], 1).tolist()))
    return out


def _loop_job(job):
    mode, seed, steps, window = job
    w = Universe(seed=seed)
    mind = Mind(seed=seed, report_loop=(mode == "döngü"))
    q = steps // 5
    ev = {2 * q: lambda _w: setattr(mind, "rep_mode", "cut"), 3 * q: lambda _w: setattr(mind, "rep_mode", "on"),
          4 * q: lambda _w: setattr(mind, "rep_mode", "sham")}
    return mode, seed, live(mind, w, steps, window=window, verbose=False, events=ev)


def stage_loop(args):
    """RAPOR DÖNGÜSÜ: ajanın kendi raporu (token) politikaya geri beslenir. Evreler: öğren | öğren | RAPOR KESİK | geri | SAHTE rapor.
    Rapor davranışı gerçekten bağlıyorsa kesilince/sahte olunca performans bozulmalı; rapor döngüsüz ajanda bu anahtarın etkisi olmamalı."""
    from multiprocessing import Pool
    modes = ["döngü", "döngüsüz"]
    jobs = [(m, s, args.steps, args.window) for m in modes for s in range(args.seed0, args.seed0 + args.seeds)]
    with Pool(min(len(jobs), os.cpu_count() or 1)) as pool:
        out = pool.map(_loop_job, jobs)
    nw = args.steps // args.window
    q = nw // 5
    names = ["öğren-1", "öğren-2", "KESİK", "geri", "SAHTE"]
    print("rapor döngüsü | ölüm/1k ve konfor, her evrenin son 3 penceresi | %d tohum" % args.seeds)
    for m in modes:
        runs = [r for (mm, s, r) in out if mm == m]
        cells = []
        for p in range(5):
            sl = slice((p + 1) * q - 3, (p + 1) * q)
            cells.append("%s %.2f/%.2f" % (names[p], np.mean([np.mean([x["per1k"] for x in r[sl]]) for r in runs]), np.mean([np.mean([x["comfort"] for x in r[sl]]) for r in runs])))
        print("  %-9s %s" % (m, " | ".join(cells)), flush=True)
    return out


def _forecast_job(job):
    cond, seed, steps, wkw = job
    w = Universe(seed=seed, **wkw)
    kw = dict(arena=wkw["size"]) if wkw else {}
    mind = Mind(seed=seed, self_correct=(cond == "düzeltmeli"), **kw)
    rows = live(mind, w, steps, window=steps // 6, verbose=False)
    fc = mind.fc
    st_all, st_late = fc.stats(), fc.stats(last=max(5, len(fc.records) // 2))
    # nedensel sınama: sahte öz-öngörü (süreler 0.3×) -> 'gecikmiş' sayısı artmalı (yalnızca düzeltmeli ajanda davranışa da yansır)
    o_obs, died = w.observe(), False
    out = {}
    for sc in (1.0, 0.3):
        fc.scale, n0, t0 = sc, fc.n_overdue, mind.age
        for _ in range(4000):
            o_obs, info = w.step(mind.act(o_obs, died))
            died = info["died"] is not None
        out[sc] = 1000.0 * (fc.n_overdue - n0) / (mind.age - t0)
    fc.scale = 1.0
    return cond, seed, rows, st_all, st_late, out, fc.n_overdue, fc.n_abandon, len(fc.records)


def stage_forecast(args):
    """ÖZ-ÖNGÖRÜ + TAAHHÜT: (a) ileri model sabit tabanı yeniliyor mu (MAE, korelasyon)? (b) düzeltme döngüsü hayatta kalmayı/konforu artırıyor mu
    (zor evren: büyük arena, nadir kaynak)? (c) sahte öngörü müdahalesi 'gecikmiş' oranını nedensel olarak değiştiriyor mu?"""
    from multiprocessing import Pool
    wkw = dict(size=float(args.size), n_food=args.nfood) if args.nfood else {}
    conds = ["düzeltmeli", "düzeltmesiz"]
    jobs = [(c, s, args.steps, wkw) for c in conds for s in range(args.seed0, args.seed0 + args.seeds)]
    with Pool(min(len(jobs), os.cpu_count() or 1)) as pool:
        out = pool.map(_forecast_job, jobs)
    print("öz-öngörü | dünya %s | %d adım | %d tohum" % (wkw or "varsayılan", args.steps, args.seeds))
    for c in conds:
        R = [r for r in out if r[0] == c]
        last = [np.mean([x["per1k"] for x in r[2][-3:]]) for r in R]
        conf = [np.mean([x["comfort"] for x in r[2][-3:]]) for r in R]
        sl = [r[4] for r in R if r[4]]
        print("  %-12s ölüm/1k %.2f ± %.2f | konfor %.3f | sonuçlanan niyet %.0f | terk %.0f | gecikmiş %.0f" %
              (c, np.mean(last), np.std(last), np.mean(conf), np.mean([r[8] for r in R]), np.mean([r[7] for r in R]), np.mean([r[6] for r in R])))
        if sl:
            print("     öngörü (son yarı): MAE %.1f adım | sabit taban MAE %.1f | korelasyon %.2f | müdahale: gecikmiş/1k adım normal %.2f -> sahte(0.3×) %.2f" %
                  (np.mean([s["mae"] for s in sl]), np.mean([s["mae_base"] for s in sl]), np.nanmean([s["corr"] for s in sl]),
                   np.mean([r[5][1.0] for r in R]), np.mean([r[5][0.3] for r in R])))
    return out


def _commit_job(job):
    cond, seed, steps, wkw = job
    w = Universe(seed=seed, **wkw)
    kw = dict(arena=wkw["size"]) if wkw else {}
    mind = Mind(seed=seed, commit=(0.2 if cond == "taahhütlü" else 0.0), **kw)
    rows = live(mind, w, steps, window=steps // 6, verbose=False)
    fc = mind.fc
    ab, rs = fc.n_abandon, len(fc.records)
    ttr = float(np.mean([r[2] for r in fc.records])) if fc.records else float("nan")
    obs, died, agree, nstep = w.observe(), False, 0, 0
    trials = {True: [], False: []}
    rng = np.random.default_rng(seed)
    for tr in range(24):                                          # nedensel sınama: niyet nesnesine YAZ (True) / yazma (False, kontrol)
        write, g = tr % 2 == 0, int(rng.integers(0, 2))
        for _ in range(int(rng.integers(100, 200))):
            obs, info = w.step(mind.act(obs, died)); died = info["died"] is not None
            if fc.open is not None:
                agree += fc.open["goal"] == mind.goal; nstep += 1
        if write:
            mind.intervene("intent", g, steps=40)
        hit = 0
        for _ in range(40):
            obs, info = w.step(mind.act(obs, died)); died = info["died"] is not None
            hit += mind.goal == g
        trials[write].append(hit / 40.0)
    return cond, seed, rows, ab, rs, ttr, agree / max(1, nstep), np.mean(trials[True]), np.mean(trials[False])


def stage_commit(args):
    """TAAHHÜT: açık niyet geçiş maliyetiyle korunur (bazal ganglion histerezisi). Ölçüler: terk oranı, çözüm süresi, hayatta kalma/konfor; rapor(niyet)-davranış tutarlılığı;
    NEDENSEL: niyet nesnesine amaç yazılınca ajan 40 adım o amaca gidiyor mu (kontrol: yazmadan)?"""
    from multiprocessing import Pool
    wkw = dict(size=float(args.size), n_food=args.nfood) if args.nfood else {}
    conds = ["taahhütlü", "taahhütsüz"]
    jobs = [(c, s, args.steps, wkw) for c in conds for s in range(args.seed0, args.seed0 + args.seeds)]
    with Pool(min(len(jobs), os.cpu_count() or 1)) as pool:
        out = pool.map(_commit_job, jobs)
    print("taahhüt | dünya %s | %d adım | %d tohum" % (wkw or "varsayılan", args.steps, args.seeds))
    for c in conds:
        R = [r for r in out if r[0] == c]
        last = [np.mean([x["per1k"] for x in r[2][-3:]]) for r in R]
        conf = [np.mean([x["comfort"] for x in r[2][-3:]]) for r in R]
        print("  %-11s ölüm/1k %.2f ± %.2f | konfor %.3f | terk oranı %.2f | çözüm süresi %.1f | niyet=amaç tutarlılığı %.2f | niyete yazma: amaç payı %.2f vs yazmadan %.2f" %
              (c, np.mean(last), np.std(last), np.mean(conf), np.mean([r[3] / max(1, r[3] + r[4]) for r in R]), np.nanmean([r[5] for r in R]),
               np.mean([r[6] for r in R]), np.mean([r[7] for r in R]), np.mean([r[8] for r in R])), flush=True)
    return out


def _autobio_job(job):
    seed, steps, wkw = job
    w = Universe(seed=seed, **wkw)
    kw = dict(arena=wkw["size"]) if wkw.get("size") else {}
    mind = Mind(seed=seed, commit=0.2, **kw)
    obs, died = w.observe(), False
    ate_t, drink_t, death_t = [], [], []
    for t in range(1, steps + 1):
        obs, info = w.step(mind.act(obs, died))
        died = info["died"] is not None
        if info["ate"] > 0 or (obs[I_E] - mind.last_obs[I_E] > 0.05):
            ate_t.append(mind.age)
        if info["drank"] > 0:
            drink_t.append(mind.age)
        if died:
            death_t.append(mind.age)
    L = [e for e in mind.fc.log if e["outcome"] is not None]
    ok = tot = 0
    for e in L:                                                  # DOĞRULUK: kayıtlı sonuç, dünyanın gerçek olayıyla örtüşüyor mu?
        if e["outcome"] == "sonuç":
            ev = ate_t if e["goal"] == 0 else drink_t
            ok += any(abs(e["t1"] - x) <= 2 for x in ev); tot += 1
        elif e["outcome"] == "öldü":
            ok += any(abs(e["t1"] - x) <= 3 for x in death_t); tot += 1
    R = [e for e in L if e["outcome"] == "sonuç" and e["need"][e["goal"]] > 0.25]
    P = [e for e in R if e["T_prec"] is not None]
    mae = lambda key, S: float(np.mean([abs(e[key] - e["y"]) for e in S])) if S else float("nan")
    conf = [e for e in P if e["T"] / max(1.0, e["precedent"]) > 2.0 or e["T"] / max(1.0, e["precedent"]) < 0.5]
    nonc = [e for e in P if e not in conf]
    nar = Narrator()
    before = [c["text"] for c in nar.autobiography(mind, 3)]
    # dış müdahale: dünya değişir (yeniden düzen) -> GEÇMİŞ anlatı değişmemeli
    w.relayout(2)
    after_world = [c["text"] for c in nar.autobiography(mind, 3)]
    # iç müdahale: belleğin yarısı silinir -> anlatı değişmeli/azalmalı
    n_before = len(mind.fc.log)
    mind.fc.log = mind.fc.log[: n_before // 2]
    after_wipe = [c["text"] for c in nar.autobiography(mind, 3)]
    return dict(n=len(L), verified=ok / max(1, tot), mae_plain=mae("T", P), mae_prec=mae("T_prec", P), n_prec=len(P),
                mae_conf=mae("T", conf), mae_nonconf=mae("T", nonc), n_conf=len(conf),
                world_same=(before == after_world), wipe_changed=(before != after_wipe), sample=before[:2])


def stage_autobio(args):
    """OTOBİYOGRAFİ (aşama 4): (1) kayıtlı kararlar dünyanın gerçek olaylarıyla örtüşüyor mu (doğruluk); (2) geçmiş kararlara başvurmak (emsal) öngörüyü iyileştiriyor mu;
    (3) emsalle çelişen öngörüler gerçekten daha mı hatalı (çelişki sinyali anlamlı mı); (4) dış müdahale geçmişi değiştirmiyor, iç müdahale (bellek silme) değiştiriyor."""
    from multiprocessing import Pool
    wkw = _world_kw(args)
    jobs = [(s, args.steps, wkw) for s in range(args.seed0, args.seed0 + args.seeds)]
    with Pool(min(len(jobs), os.cpu_count() or 1)) as pool:
        out = pool.map(_autobio_job, jobs)
    f = lambda k: np.nanmean([o[k] for o in out])
    print("otobiyografi | %d adım | %d tohum" % (args.steps, args.seeds))
    print("  kayıtlı karar %.0f | dünya olayıyla doğrulanan %.2f" % (f("n"), f("verified")))
    print("  emsale başvurma (n=%.0f): öngörü MAE sade %.1f -> emsalli %.1f" % (f("n_prec"), f("mae_plain"), f("mae_prec")))
    print("  çelişki (n=%.0f): çelişkili öngörülerde MAE %.1f | çelişkisiz %.1f" % (f("n_conf"), f("mae_conf"), f("mae_nonconf")))
    print("  dış müdahale (dünya değişti): geçmiş anlatı değişmedi %d/%d | iç müdahale (bellek yarı silindi): anlatı değişti %d/%d" %
          (sum(o["world_same"] for o in out), len(out), sum(o["wipe_changed"] for o in out), len(out)))
    for o in out[:1]:
        for s in o["sample"]:
            print("   örnek:", s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["run", "compare", "shift", "introspect", "narrate", "transfer", "demo", "testimony", "alive", "combo", "lead", "conflict", "robust", "rule", "culture", "loop", "forecast", "commit", "autobio"], default="run")
    ap.add_argument("--mode", choices=["full", "flat", "random"], default="full")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--window", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-sm", action="store_true")
    ap.add_argument("--size", type=float, default=24.0)
    ap.add_argument("--nfood", type=int, default=0)
    ap.add_argument("--food-haz", type=float, default=0.0)
    ap.add_argument("--metab-scale", type=float, default=1.0)
    ap.add_argument("--relayout-every", type=int, default=0)
    ap.add_argument("--no-ctx", action="store_true")
    ap.add_argument("--every", type=int, default=3000)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--arena", type=float, default=24.0)
    ap.add_argument("--log", type=int, default=6000)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--kind", choices=["invert", "poison", "layout", "swap"], default="invert")
    ap.add_argument("--no-map", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--modes", default="random,flat,full")
    ap.add_argument("--save", default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    {"run": stage_run, "compare": stage_compare, "shift": stage_shift, "introspect": stage_introspect, "narrate": stage_narrate, "transfer": stage_transfer, "demo": stage_demo, "testimony": stage_testimony, "alive": stage_alive, "combo": stage_combo, "lead": stage_lead, "conflict": stage_conflict, "robust": stage_robust, "rule": stage_rule, "culture": stage_culture, "loop": stage_loop, "forecast": stage_forecast, "commit": stage_commit, "autobio": stage_autobio}[args.stage](args)


if __name__ == "__main__":
    main()
