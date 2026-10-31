#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_autonomous_logic.py — AGI Öz-Sınıflandırma, Otonom Etiketleme, Kendi Mantığını Kurma ve Tam Durumsal Türkçe Korteksi.

Kullanıcı direktifi doğrultusunda 3 temel yeteneği SIFIR çok-epoch eğitimle (O(1) ChemMushroomBody
plastisitesi + Türkçe morfosentaks + üretken Broca kök çekimi) birleştirir:

  1. NeuralUtteranceCortex (Soru / İddia / Yasa / Durum Öz-Sınıflandırması):
     Noktalama işareti ('?') olmasa bile cümlenin soru mu, koşullu yasa mı, ikili süperpozisyon mu,
     orantı bağıntısı mı, sosyal/duygusal durum mu olduğunu Python 'if "?" in s' kurallarına mahkûm
     kalmadan kendi plastik Mantar Gövdesi (ChemMushroomBody) nöronlarıyla sınıflandırır; yeni bir
     söylem örüntüsü gördüğünde kendi çıkış nöronunu büyüterek (mint_utterance_label) etiketler!

  2. AutonomousLogicAndLabelEngine (Kuantum Dahil Bilinmeyen Alanlarda Öz-Etiketleme & Mantık İnşası):
     Önceden tanımlı insan etiketlerine ('ISA', 'HAS_PROP') sıkışmaz! Herhangi bir bilimsel veya soyut
     cümleden (ör. Kuantum fiziği, genetik, ekonomi, felsefe):
       - Kendi fiil-hal çerçevesi etiketini üretir: ÖZ_BAĞ:<kök>[NOM->ACC=>DAT]
       - İkili / Süperpozisyon durum yasasını keşfeder: ÇİFT_DURUM(nesne: A ⊕ B | koşul)
       - Koşullu durum geçişi ve çökme yasalarını kurar: KOŞUL_GEÇİŞ(tetik => hedef_değişim)
       - Eşdeğişim / Ters-Doğru Orantı yasasını keşfeder: EŞDEĞİŞİM(X ↕ Y) ve ters yönde bile sıfır-atış çıkarım yapar!
       - Hebbian eş-aktivasyonla kavramları kendi adlandırdığı KÜME:<merkez_kökler> altında gruplar!

  3. SituationalTurkishCortex (Orta-İleri Seviye Tam Durumsal Türkçe):
     Günaydın / kolay gelsin / geçmiş olsun / özür / kutlama / kararsızlık-tavsiye / tartışma-itiraz /
     stres-kafa karışıklığı / şaka gibi gerçek insan yaşamındaki tüm iletişim durumlarını anlar ve
     Nora'nın aktif içsel monolog şemasına (S0..S4) uygun doğal Türkçe üretir.
"""
from __future__ import annotations

from nora._paths import REPO_ROOT


import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
import numpy as np
import numpy.ma as _np_ma  # ilk sınıflamada tembel import maliyeti olmasın

from nora_chem import NeuroChemistry, ChemMushroomBody
from nora_lang import TurkishMorphology, tr_lower, tr_capitalize
from nora_broca import TurkishSyntaxEngine, NP, VerbSpec, PredSpec


def _det_hash(s: str) -> int:
    """Deterministik 32-bit FNV-1a özet fonksiyonu (PYTHONHASHSEED'den bağımsız!)."""
    h = 2166136261
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


# ============================================================================
# 1. NÖRAL SÖZCE VE SORU ÖZ-SINIFLANDIRICI (SIFIR '?' ZORUNLULUĞU!)
# ============================================================================

class NeuralUtteranceCortex:
    """Bir cümlenin soru mu, doğrulama mı, koşullu yasa mı, çift-durum mu yoksa sosyal/duygusal
    paylaşım mı olduğunu morfolojik-konumsal özellik vektörlerinden KENDİ Mantar Gövdesi ile belirler.
    """

    BASE_LABELS = [
        "SORU_DOGRULAMA",       # Evet/hayır veya olasılık sorusu ('... taşır mı', '... olabilir mi')
        "SORU_ACIKLAMA",        # Ne / nasıl / hangi / durum sorusu ('... nedir', '... hangi durumdadır')
        "SORU_NEDENSEL",        # Neden / niye / niçin / sebep sorusu
        "SORU_KOSULLU",         # Koşullu / karşıolgusal soru ('ölçülürse ne olur', 'artarsa nasıl değişir')
        "IKILI_SUPERPOZISYON",  # Klasik ikili mantığı aşan eşzamanlı durum ('hem dalga hem parçacık')
        "ORANTI_BAGINTI",       # Eşdeğişim / orantı yasası ('azaldıkça artar', 'ters orantılıdır')
        "KOSULLU_YASA",         # Koşullu durum geçişi ('gözlem yapılırsa süperpozisyon bozulur')
        "OLGU_EYLEM_YASASI",    # Genel bilimsel/olgusal önerme ve yeni eylem çerçevesi
        "SOSYAL_NEZAKET",       # Günaydın, kolay gelsin, geçmiş olsun, özür, tebrik, veda
        "DUYGUSAL_PAYLASIM",    # Hastalık, kafa karışıklığı, stres, mutluluk, öfke
        "TAVSIYE_KARAR",        # Tavsiye isteme, kararsızlık, planlama ricası
        "TARTISMA_OZ_MANTIK",   # Nora'nın kendi kurduğu etiket/mantık üzerine sorgulama
        "GORUS_TARTISMA",       # Katılma / itiraz / fikir belirtme (soru değil, konumsal bildirim)
    ]

    # Nöral korteksin ilk sinaptik kalibrasyonu için morfolojik çeşitlilik taşıyan çekirdek örnekler
    BOOT_EXEMPLARS: dict[str, list[str]] = {
        "SORU_DOGRULAMA": [
            "bu doğru mu", "elektron dalga olabilir mi", "ışık boşlukta yayılır mı",
            "foton kütle taşır mı", "böyle bir şey mümkün müdür", "bana katılıyor musun",
            "aynı anda iki durumda bulunabilir mi", "bu kural geçerli midir",
        ],
        "SORU_ACIKLAMA": [
            "bu kavram nedir", "kuantum dolanıklık nasıl çalışır", "hangi durumdadır",
            "dalga fonksiyonu ne işe yarar", "bunun temel özelliği nedir", "nasıl tanımlanır",
            "aralarındaki ilişki nasıldır", "bu süreçte ne gerçekleşir",
            "enerji nedir", "entropi nedir", "bilinç nedir", "kuantum dolanıklık nedir",
            "kimya nedir", "biraz daha anlatır mısın", "bunu açar mısın", "ayrıntı verir misin",
        ],
        "SORU_NEDENSEL": [
            "bu olay neden gerçekleşir", "niye böyle oluyor", "bunun sebebi nedir",
            "niçin çökme meydana gelir", "neden enerji korunur", "hangi nedenden kaynaklanır",
        ],
        "SORU_KOSULLU": [
            "elektron ölçülürse ne olur", "sıcaklık artarsa basınç nasıl değişir",
            "gözlem yapılmazsa hangi durumda kalır", "konum belirsizliği azalırsa ne olur",
            "eğer kuvvet iki katına çıkarsa ne gerçekleşir", "ölçüm yapılırsa dalga fonksiyonuna ne olur",
        ],
        "IKILI_SUPERPOZISYON": [
            "elektron gözlemlenmeden önce hem dalga hem parçacık durumundadır",
            "foton aynı anda hem dalga hem tanecik özelliği gösterir",
            "kubit ölçülene dek hem sıfır hem bir süperpozisyonundadır",
            "ışık hem dalga hem parçacık doğasına sahiptir",
        ],
        "ORANTI_BAGINTI": [
            "konum belirsizliği azaldıkça momentum belirsizliği artar",
            "dalga boyu kısaldıkça frekans yükselir",
            "sıcaklık arttıkça tanecik hızı büyür",
            "basınç ile hacim ters orantılıdır",
            "enerji ile frekans doğru orantılıdır",
        ],
        "KOSULLU_YASA": [
            "gözlem yapılırsa süperpozisyon bozulur",
            "parçacıklardan biri ölçülürse diğerinin durumu anında belirlenir",
            "sıcaklık sıfırın altına düşerse su donar",
            "dış kuvvet uygulanmadıkça cisim hızını korur",
        ],
        "OLGU_EYLEM_YASASI": [
            "ölçüm dalga fonksiyonunu tek bir özduruma çökertir",
            "foton soğurulması elektronu üst enerji seviyesine sıçratır",
            "enzimler kimyasal tepkimeleri hızlandırır",
            "entropi kapalı sistemlerde düzensizlik olarak tanımlanır",
            "bağışıklık sistemi patojeni tanır",
            "böbrekler kanı süzer ve atığı uzaklaştırır",
            "kökler topraktan suyu emer",
            "motor ısıyı egzozdan dışarı atar",
            "karaciğer toksinleri zararsız hale getirir",
        ],
        "SOSYAL_NEZAKET": [
            "günaydın nora", "iyi akşamlar dostum", "kolay gelsin sana", "çok geçmiş olsun",
            "kusura bakma yanlış söyledim", "özür dilerim geciktim", "afiyet olsun",
            "tebrik ederim harika iş", "kendine iyi bak görüşürüz", "hoş bulduk",
        ],
        "DUYGUSAL_PAYLASIM": [
            "bugün biraz hastayım başım ağrıyor", "kafam çok karışık odaklanamıyorum",
            "çok stresliyim içim daralıyor", "sınavı kazandım çok mutluyum",
            "kendimi yalnız ve yorgun hissediyorum", "bugün moralim bozuk",
        ],
        "TAVSIYE_KARAR": [
            "çok kararsız kaldım ne önerirsin", "sence nereden başlamalıyım",
            "bu durumda ne yapmamı tavsiye edersin", "bana bir yol gösterir misin",
            "vaktim az nasıl planlamalıyım", "fikrine ihtiyacım var ne dersin",
            "hangisini önerirsin", "sen olsan ne yapardın", "bana tavsiye ver",
        ],
        "TARTISMA_OZ_MANTIK": [
            "kendi oluşturduğun etiketler ve mantık kuralları neler",
            "hangi kavram kümelerini kurdun",
            "hangi mantık yasalarını oluşturdun",
            "bu konuda kendi mantığın ne diyor",
            "kurduğun etiketleri nasıl doğrularsın",
            "kendi çıkarımını kanıtlayabilir misin",
            "bu iddiandan emin misin",
        ],
        "GORUS_TARTISMA": [
            "bence bu konuda haklısın katılıyorum",
            "hayır bence tam öyle değil yanılıyorsun",
            "sana katılmıyorum bu değerlendirme yanlış",
            "bence de öyle düşünüyorum",
            "bu fikre itirazım var",
            "kesinlikle katılıyorum sana",
            "ben aynı görüşte değilim",
        ],
    }

    def __init__(self, chem: NeuroChemistry, morph: TurkishMorphology, seed: int = 43):
        self.chem = chem
        self.morph = morph
        self.labels: list[str] = list(self.BASE_LABELS)
        self.label_to_id: dict[str, int] = {lbl: i for i, lbl in enumerate(self.labels)}
        self.dim = 256
        # Üç bağımsız Kenyon-lifi projeksiyonu (MB bölmeleri): sınıflandırma kararı tek tohumun
        # rastgele izdüşümüne bağlı kalmaz, bölmelerin ortalama ateşlemesiyle verilir.
        self.n_compartments = 3
        self.mbs = [
            ChemMushroomBody(n_in=self.dim, n_out=len(self.labels), n_kc=1536, claws=6, sparsity=0.05, seed=seed + 101 * i)
            for i in range(self.n_compartments)
        ]
        self.mb = self.mbs[0]
        # Kenyon-lifi seyrek kod prototipleri: az örnekli sınıflandırmada kararlı "örüntü tamamlama"
        # okuması sağlar (MBON sinaptik ağırlığı + KC örüntü örtüşmesi birlikte karar verir).
        self.kc_protos: dict[str, list[tuple[set[int], ...]]] = {lbl: [] for lbl in self.labels}
        # Hızlı örüntü-tamamlama için seyrek prototip matrisleri (bölme başına bir matris)
        self._proto_mat = [np.zeros((0, mb.n_kc), np.float32) for mb in self.mbs]
        self._proto_sizes = np.zeros((0,), np.float32)
        self._proto_cids = np.zeros((0,), np.int32)
        self.self_minted_labels: list[str] = []
        self.minted_samples: dict[str, str] = {}
        self.QUESTION_LABELS = {
            "SORU_DOGRULAMA", "SORU_ACIKLAMA", "SORU_NEDENSEL", "SORU_KOSULLU",
            "TAVSIYE_KARAR", "TARTISMA_OZ_MANTIK",
        }
        self._boot_plastic_cortex()

    PHATIC_GREET = {"selam", "merhaba", "günaydın", "selamlar", "hey", "hoş", "geldin", "bulduk", "akşamlar", "sabahlar"}
    PHATIC_CARE = {"teşekkür", "sağ", "olasın", "kolay", "gelsin", "afiyet", "geçmiş", "olsun", "özür",
                   "kusura", "bakma", "affedersin", "tebrik", "ederim", "rica", "veda", "görüşürüz"}
    ILLNESS_LEX = {"hasta", "hastayım", "ağrı", "ağrıyor", "ateş", "grip", "üşüttüm", "bulantı", "migren", "yaralı"}
    STRESS_LEX = {"stres", "stresliyim", "karışık", "bunaldım", "odaklanamıyorum", "bıktım", "yorgunum", "kaygı", "endişe", "daralıyor"}
    JOY_LEX = {"mutlu", "mutluyum", "kazandım", "başardım", "sevinç", "harika", "mükemmel", "geçtim", "kutlama"}
    ADVICE_LEX = {"öner", "önerirsin", "tavsiye", "tavsiyeler", "kararsız", "başlamalıyım", "yapmalıyım", "fikir", "yol", "göster"}
    DEBATE_LEX = {"katılıyorum", "katılmıyorum", "haklısın", "yanılıyorsun", "hatalısın", "tartışma", "itiraz"}

    def encode_utterance(self, raw_text: str) -> tuple[np.ndarray, dict]:
        """Sözceyi noktalama işaretine bağımlı OLMADAN morfolojik + yapısal + kök-ek n-gram vektörüne kodlar.
        Üst-bilişsel yapısal bloğu (0..31) sözcük bloğundan daha güçlü ağırlıklandırır: soru/yasa ayrımı
        tek tek sözcüklere değil, çekim biçimbilimine dayanır."""
        s_low = tr_lower(raw_text.strip())
        words = re.findall(r"[\wçğıöşü]+", s_low)
        vec = np.zeros(self.dim, dtype=np.float32)
        meta = {
            "has_mi": False, "has_wh": False, "has_cond": False, "has_dikca": False,
            "has_hem_hem": False, "has_oranti": False, "has_passive": False, "has_abil": False,
            "word_analyses": [],
        }
        if not words:
            return vec, meta

        mi_set = {"mı", "mi", "mu", "mü", "mısın", "misin", "musun", "müsün", "mıyım", "miyim", "mıdır", "midir", "mudur", "müdür",
                  "mıyız", "miyiz", "mısınız", "misiniz", "mıydı", "miydi"}
        wh_roots = {"ne", "nedir", "neye", "neyi", "nasıl", "nasıldır", "hangi", "hangisi", "kaç", "nerede", "kim"}
        why_roots = {"neden", "niye", "niçin"}
        causal_neden = any(p in s_low for p in ("neden olur", "neden oldu", "neden olmaz", "neden oluyor"))

        lex_block = vec[32:192]
        suf_block = vec[192:256]
        hem_count = sum(1 for w in words if w == "hem")
        meta["has_hem_hem"] = hem_count >= 2 or ("aynı" in words and "anda" in words and "ve" in words)
        meta["has_oranti"] = "orantılı" in s_low or "orantılıdır" in s_low

        last_w = words[-1]
        is_interrogative_word = any(w in mi_set for w in words)
        wh_hit = False
        why_hit = False
        for idx, w in enumerate(words):
            an = self.morph.analyze_word(w)
            meta["word_analyses"].append(an)
            pos_tag = "START" if idx == 0 else ("END" if idx == len(words) - 1 else "MID")

            h_w = _det_hash("W:" + w)
            lex_block[h_w % 160] += 1.0
            h_st = _det_hash("ST:" + an["stem"] + ":" + pos_tag)
            lex_block[h_st % 160] += 0.85
            if len(w) >= 3:
                h_suf = _det_hash("SUF:" + w[-3:] + ":" + pos_tag)
                suf_block[h_suf % 64] += 0.9

            if w in mi_set:
                meta["has_mi"] = True
            if w in wh_roots or any(w.startswith(x) and len(w) >= 4 for x in ("nasıl", "nere", "hangi", "kaç", "kim")):
                wh_hit = True
            if w in why_roots and not causal_neden:
                why_hit = True
            if an.get("cond") or w in {"ise", "olursa", "eğer"} or w.endswith(("madıkça", "medikçe")):
                meta["has_cond"] = True
            if an.get("converb") == "DIKCA":
                meta["has_dikca"] = True
            if an.get("passive"):
                meta["has_passive"] = True
            if an.get("modal") == "ABIL":
                meta["has_abil"] = True
        meta["has_wh"] = wh_hit or why_hit

        # Sözcük/son-ek bloklarını ayrı normalize et (yapısal blok ezilmesin)
        for blk in (lex_block, suf_block):
            nrm = float(np.linalg.norm(blk))
            if nrm > 1e-6:
                blk /= nrm

        is_q_word = is_interrogative_word or wh_hit or why_hit or ("?" in raw_text)
        has_3sg_mi = any(w in {"mı", "mi", "mu", "mü", "mıdır", "midir", "mudur", "müdür"} for w in words)
        wset = set(words)
        vec[0] = 2.6 if (meta["has_hem_hem"] and not is_q_word) else (0.5 if meta["has_hem_hem"] else 0.0)
        vec[1] = 2.5 if (meta["has_oranti"] and not is_q_word) else 0.0
        vec[2] = 2.5 if meta["has_mi"] else 0.0
        vec[3] = 2.2 if wh_hit else 0.0
        vec[4] = 2.5 if why_hit else 0.0
        vec[5] = 2.8 if meta["has_cond"] else 0.0
        vec[6] = 2.6 if meta["has_dikca"] else 0.0
        vec[7] = 0.9 if meta["has_passive"] else 0.0
        vec[8] = 1.1 if meta["has_abil"] else 0.0
        vec[9] = 2.4 if is_q_word else -1.2
        vec[10] = 2.9 if (meta["has_cond"] and is_q_word) else 0.0
        vec[11] = 2.9 if (meta["has_cond"] and not is_q_word) else 0.0
        vec[12] = 2.9 if (has_3sg_mi and not meta["has_cond"]) else (1.3 if (meta["has_mi"] and not meta["has_cond"]) else 0.0)
        vec[13] = 2.6 if ((meta["has_dikca"] or meta["has_oranti"]) and not is_q_word) else 0.0
        vec[14] = 2.6 if (meta["has_hem_hem"] and not is_q_word) else 0.0
        vec[15] = 2.4 if (("ne olur" in s_low) or ("ne oldu" in s_low) or ("ne olurdu" in s_low)) else 0.0
        vec[16] = 2.4 if (last_w in {"nedir", "neydi", "kimdir", "nasıldır"} or s_low.endswith("nedir")) else 0.0
        vec[17] = 2.2 if ("nasıl" in words or "nasıldır" in words) else 0.0
        vec[18] = 2.6 if any(w in {"mıdır", "midir", "mudur", "müdür", "mıydı", "miydi"} for w in words) else 0.0
        vec[19] = 2.4 if (wset & self.PHATIC_GREET and len(words) <= 7) else 0.0
        vec[20] = 2.4 if (wset & self.PHATIC_CARE and len(words) <= 8) else 0.0
        vec[21] = 2.3 if (wset & self.ILLNESS_LEX) else 0.0
        vec[22] = 2.3 if (wset & self.STRESS_LEX) else 0.0
        vec[23] = 2.3 if (wset & self.JOY_LEX) else 0.0
        vec[24] = 2.2 if (wset & self.ADVICE_LEX) else 0.0
        vec[25] = 2.2 if (wset & self.DEBATE_LEX) else 0.0
        vec[26] = 2.2 if ({"şaka", "takılıyorum", "espri"} & wset) else 0.0
        vec[27] = 2.2 if (last_w in mi_set or last_w.endswith(("mı", "mi", "mu", "mü"))) else 0.0
        vec[28] = 2.0 if "hangi" in wset or any(w.startswith("hangi") for w in words) else 0.0
        vec[29] = 1.6 if any(w.startswith("kaç") for w in words) else 0.0
        vec[30] = 1.5 if "neden" in wset or "niye" in wset or "niçin" in wset else 0.0
        vec[31] = 1.2 if ("katılıyorum" in wset or "haklısın" in wset or "yanılıyorsun" in wset) else 0.0

        return vec, meta

    def _append_proto(self, codes: tuple[set[int], ...], cid: int):
        """Yeni bir prototipi bölme başına ikili matrise ekler (nörogenezle genişleyen KC sayısını izler)."""
        for c, code_set in enumerate(codes):
            width = max(self._proto_mat[c].shape[1], self.mbs[c].n_kc,
                        (max(code_set) + 1) if code_set else 0)
            if width > self._proto_mat[c].shape[1]:
                pad = width - self._proto_mat[c].shape[1]
                if self._proto_mat[c].shape[0]:
                    self._proto_mat[c] = np.pad(self._proto_mat[c], ((0, 0), (0, pad)))
                else:
                    self._proto_mat[c] = np.zeros((0, width), np.float32)
            row = np.zeros((1, width), np.float32)
            if code_set:
                row[0, np.fromiter(code_set, dtype=np.int64)] = 1.0
            self._proto_mat[c] = np.vstack([self._proto_mat[c], row])
        self._proto_sizes = np.append(self._proto_sizes, float(max(1, len(codes[0]))))
        self._proto_cids = np.append(self._proto_cids, int(cid))

    def _proto_scores(self, codes: list[set[int]]) -> np.ndarray:
        """Seyrek KC kodları ile prototipler arasında hızlı Jaccard örtüşmesi (matris çarpımıyla)."""
        out = np.zeros(len(self.labels), dtype=np.float64)
        if len(self._proto_cids) == 0:
            return out
        n_comp = len(self.mbs)
        sims_acc = np.zeros((len(self._proto_cids),), dtype=np.float64)
        for c in range(n_comp):
            mat = self._proto_mat[c]
            if mat.shape[0] == 0:
                continue
            width = max(mat.shape[1], self.mbs[c].n_kc)
            x = np.zeros((width,), np.float32)
            code = codes[c]
            if code:
                idx_arr = np.fromiter(code, dtype=np.int64)
                idx_arr = idx_arr[idx_arr < width]
                x[idx_arr] = 1.0
            if width > mat.shape[1]:
                mat = np.pad(mat, ((0, 0), (0, width - mat.shape[1])))
                self._proto_mat[c] = mat
            inter = mat @ x
            union = np.maximum(1.0, self._proto_sizes + float(max(1, len(code))) - inter)
            sims_acc += inter / union
        sims = sims_acc / float(n_comp)
        # Sınıf başına en iyi eşleşme + ilk-3 ortalaması (örüntü tamamlama)
        for cid in np.unique(self._proto_cids):
            sv = sims[self._proto_cids == cid]
            top = np.sort(sv)[::-1][:3]
            out[int(cid)] = float(top[0]) + 0.35 * float(top.mean())
        return out

    def _boot_plastic_cortex(self):
        saved_ach = self.chem.ach
        self.chem.ach = 0.88
        counts: dict[int, int] = {}
        encoded = []
        for lbl, sents in self.BOOT_EXEMPLARS.items():
            cid = self.label_to_id[lbl]
            for sent in sents:
                v, _ = self.encode_utterance(sent)
                encoded.append((lbl, cid, v))
        for _ in range(3):
            for lbl, cid, v in encoded:
                for mb in self.mbs:
                    mb.learn_oneshot(v, cid, chem=self.chem, strength=0.95)
                codes = tuple(set(mb.encode_kc(v).tolist()) for mb in self.mbs)
                self.kc_protos.setdefault(lbl, []).append(codes)
                self._append_proto(codes, cid)
                counts[cid] = counts.get(cid, 0) + 1
        # Sınıf başına örnek sayısı farkı bir "önceki olasılık" yanlılığı yaratmasın:
        # her sınıfın sinaptik kütlesi örnek sayısına bölünerek eşitlenir (APL normalizasyonu benzeri).
        for cid, cnt in counts.items():
            if cnt > 1:
                for mb in self.mbs:
                    mb.W_fast[cid] /= float(cnt)
                    mb.W_slow[cid] /= float(cnt)
        self.chem.ach = saved_ach

    def classify(self, raw_text: str) -> dict:
        """Cümleyi tamamen nöral olarak sınıflandırır; soru olup olmadığını da kendisi belirler!"""
        vec, meta = self.encode_utterance(raw_text)
        mb_scores = np.mean([mb.scores(vec) for mb in self.mbs], axis=0)
        codes = [set(mb.encode_kc(vec).tolist()) for mb in self.mbs]

        # 1) KC örüntü-tamamlama benzerliği (seyrek ortogonal kodlar üzerinde Jaccard)
        proto_scores = self._proto_scores(codes)

        # 2) Sinaptik (MBON) ateşlemesi -> softmax olasılığı
        if float(np.max(np.abs(mb_scores))) > 1e-9:
            mb_prob = np.exp((mb_scores - float(np.max(mb_scores))) * 6.0)
            mb_prob /= max(1e-9, float(mb_prob.sum()))
        else:
            mb_prob = np.zeros_like(mb_scores)

        scores = 0.65 * proto_scores + 0.35 * mb_prob
        order = np.argsort(scores)[::-1]
        best_id = int(order[0])
        second_id = int(order[1]) if len(order) > 1 else best_id
        margin = float(scores[best_id] - scores[second_id])
        label = self.labels[best_id]
        is_question = label in self.QUESTION_LABELS
        return {
            "label": label,
            "is_question": bool(is_question),
            "confidence": round(float(np.clip(0.55 + 0.35 * np.tanh(margin * 2.2), 0.0, 0.99)), 3),
            "margin": round(margin, 4),
            "meta": meta,
            "vec": vec,
        }

    def mint_utterance_label(self, sample_text: str, custom_name: str = "") -> str:
        """Bilinmeyen yeni bir söylem türünü kendi Mantar Gövdesi çıkış nöronlarına ekleyerek öz-etiketler!"""
        vec, meta = self.encode_utterance(sample_text)
        if not custom_name:
            stems = [a["stem"] for a in meta["word_analyses"] if len(a["stem"]) >= 3]
            tag_core = "_".join(stems[:2]) if stems else f"tip{len(self.labels)}"
            custom_name = f"ÖZ_SÖYLEM:{tag_core}"
        if custom_name not in self.label_to_id:
            # ChemMushroomBody çıkış matrisini (W_fast, W_slow, stiff, tags) dinamik büyüt!
            self.labels.append(custom_name)
            new_id = len(self.labels) - 1
            self.label_to_id[custom_name] = new_id
            for mb in self.mbs:
                mb.n_out += 1
                mb.W_fast = np.vstack([mb.W_fast, np.zeros((1, mb.n_kc), dtype=np.float32)])
                mb.W_slow = np.vstack([mb.W_slow, np.zeros((1, mb.n_kc), dtype=np.float32)])
                mb.stiff = np.vstack([mb.stiff, np.zeros((1, mb.n_kc), dtype=np.float32)])
                mb.tags = np.vstack([mb.tags, np.zeros((1, mb.n_kc), dtype=np.float32)])
            self.self_minted_labels.append(custom_name)
        self.minted_samples[custom_name] = sample_text
        cid = self.label_to_id[custom_name]
        saved_ach = self.chem.ach
        self.chem.ach = max(0.85, saved_ach)
        for mb in self.mbs:
            mb.learn_oneshot(vec, cid, chem=self.chem, strength=1.15)
        mint_codes = tuple(set(mb.encode_kc(vec).tolist()) for mb in self.mbs)
        self.kc_protos.setdefault(custom_name, []).append(mint_codes)
        self._append_proto(mint_codes, cid)
        self.chem.ach = saved_ach
        return custom_name


# ============================================================================
# 2. OTONOM ETİKETLEME VE KENDİ MANTIĞINI KURMA MOTORU (KUANTUM DAHİL!)
# ============================================================================

@dataclass
class DualStateRule:
    """Klasik ikili mantığı aşan eşzamanlı süperpozisyon / çift-durum yasası."""
    entity: str
    state_a: str
    state_b: str
    condition_scope: str        # Örn. 'gözlemlenmeden önce' / 'ölçülene dek'
    guard_root: str             # Örn. 'gözlemle' / 'ölç' (bu eylem gerçekleşmediği sürece geçerli!)
    self_label: str             # Nora'nın kendi ürettiği etiket
    raw_source: str


@dataclass
class StateTransitionRule:
    """Nora'nın kendi ürettiği durum geçişi / eylem çerçevesi / koşullu yasa."""
    trigger_phrase: str         # Örn. 'ölçüm' / 'gözlem yapılırsa'
    trigger_roots: set[str]     # Örn. {'ölç', 'ölçüm'}
    negated_trigger: bool       # 'yapılmazsa / olmadıkça' ise True
    target_entity: str          # Örn. 'dalga fonksiyonu' / 'süperpozisyon'
    verb_root: str              # Örn. 'çökert' / 'boz' / 'belirle'
    result_state: str           # Örn. 'tek bir özdurum' / 'bozulur'
    case_frame: str             # Örn. 'NOM->ACC=>DAT' veya 'KOŞUL=>SONUÇ'
    self_label: str             # Örn. 'ÖZ_BAĞ:çökert[NOM->ACC=>DAT]'
    raw_source: str
    target_poss3: bool = False  # Tamlanan nesne 3. tekil iyelikli mi ('dalga fonksiyonu')


@dataclass
class CoupledVariationRule:
    """Nora'nın -dıkça/-tikçe veya orantı ifadelerinden kendi çıkardığı eşdeğişim yasası."""
    var_x: str                  # Örn. 'konum belirsizliği'
    var_y: str                  # Örn. 'momentum belirsizliği'
    coupling: int               # +1: aynı yönlü (doğru orantı), -1: ters yönlü (ters orantı)
    verb_x: str                 # Örn. 'azal'
    verb_y: str                 # Örn. 'art'
    self_label: str             # Örn. 'ÖZ_YASA:TERS_EŞDEĞİŞİM[konum belirsizliği ↕ momentum belirsizliği]'
    raw_source: str


class AutonomousLogicAndLabelEngine:
    """İnsan tarafından kodlanmış sabit etiketler olmadan, Türkçe cümlelerin morfosentaktik hal/çatı
    yapısından kendi etiketlerini, kavram kümelerini ve mantıksal çıkarım kurallarını inşa eder."""

    UP_ROOTS = {"art", "artır", "yüksel", "büyü", "çoğal", "hızlan", "genleş", "güçlen", "uza"}
    DOWN_ROOTS = {"azal", "azalt", "düş", "küçül", "kısal", "yavaşla", "daral", "zayıfla", "sön"}

    STOP_WORDS = {
        "bir", "bu", "şu", "o", "ve", "ile", "için", "kadar", "dek", "önce", "sonra", "olarak",
        "ise", "eğer", "hem", "aynı", "anda", "tek", "çok", "daha", "en", "her", "hiç", "gibi",
    }

    def __init__(self, morph: TurkishMorphology, utterance_cortex: NeuralUtteranceCortex):
        self.morph = morph
        self.utter_cortex = utterance_cortex
        self.syn = TurkishSyntaxEngine
        self.dual_rules: list[DualStateRule] = []
        self.transition_rules: list[StateTransitionRule] = []
        self.variation_rules: list[CoupledVariationRule] = []
        self.self_minted_tags: dict[str, int] = {}
        self.cooccur: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.emergent_clusters: dict[str, set[str]] = {}

    def _register_tag(self, tag: str):
        self.self_minted_tags[tag] = self.self_minted_tags.get(tag, 0) + 1

    def _analyze_with_compound_case(self, w: str, lexicon: set[str] | None = None) -> dict:
        """Bileşik/bilimsel terimlerde ('özduruma'->'özdurum'+DAT, 'elektronu'->'elektron'+ACC) hal ekini de çözer."""
        an = self.morph.analyze_word(w, lexicon=lexicon)
        if an["case"] == "NOM" and not an.get("is_verb") and len(w) >= 5:
            compound_bases = ("durum", "sistem", "alan", "ortam", "katman", "seviye", "boyut", "fonksiyon", "elektron", "foton", "proton", "nötron", "parçacık")
            if w[-1] in ("a", "e") and w[:-1].endswith(compound_bases):
                an = dict(an, stem=w[:-1], case="DAT")
            elif w[-1] in ("ı", "i", "u", "ü") and w[:-1].endswith(compound_bases):
                an = dict(an, stem=w[:-1], case="ACC")
        return an

    def _canonical_root(self, token: str) -> str:
        """İsimden fiil / fiilden isim türevlerini ('ölçüm'->'ölç', 'gözlem'->'gözlemle') ortak köke bağlar."""
        w = tr_lower(token).strip()
        an = self.morph.analyze_word(w)
        st = an["stem"]
        # '-ım/-im/-um/-üm' fiilden isim yapım eki ('ölçüm' -> 'ölç', 'çözüm' -> 'çöz', 'üretim' -> 'üret')
        for nsuf in ("ım", "im", "um", "üm"):
            if len(st) - 2 >= 2 and st.endswith(nsuf):
                cand = st[:-2]
                if cand in self.morph.KNOWN_VERB_ROOTS:
                    return cand
        if st == "gözlem":
            return "gözlemle"
        return st

    def _update_emergent_clusters(self, content_stems: list[str]):
        """Hebbian eş-aktivasyon ağırlıklarıyla kendi kendine alan/kavram kümesi keşfeder ve isimlendirir!"""
        uniq = [s for s in dict.fromkeys(content_stems) if len(s) >= 3 and s not in self.STOP_WORDS]
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a, b = uniq[i], uniq[j]
                self.cooccur[a][b] += 1.0
                self.cooccur[b][a] += 1.0

        # En güçlü bağlı bileşenlerden kendi küme etiketini oluştur (son 240 düğümle SINIRLI -> O(1) tur gecikmesi)
        visited: set[str] = set()
        new_clusters: dict[str, set[str]] = {}
        recent_nodes = list(self.cooccur.keys())[-240:]
        for node in recent_nodes:
            if node in visited:
                continue
            comp = {node}
            queue = deque([node])
            while queue:
                cur = queue.popleft()
                for nb, w in self.cooccur[cur].items():
                    if w >= 1.0 and nb not in comp and len(comp) < 14:
                        comp.add(nb)
                        queue.append(nb)
            visited |= comp
            if len(comp) >= 3:
                # Merkeziliği en yüksek 2 İSİM kökünü küme adı olarak KENDİSİ seçer (fiil kökleri ada tercih edilmez!)
                ranked = sorted(
                    comp,
                    key=lambda x: (x in self.morph.KNOWN_VERB_ROOTS, -sum(self.cooccur[x].values())),
                )
                cname = f"KÜME:{ranked[0]}_{ranked[1]}"
                new_clusters[cname] = comp
        if new_clusters:
            self.emergent_clusters = new_clusters

    # ------------------------------------------------------------------ Öğrenme ve Öz-Etiketleme
    def ingest_statement(self, raw_text: str) -> dict | None:
        """Tek bir Türkçe cümleyi okuyup kendi mantık kuralını ve kendi etiketini oluşturur."""
        s_raw = raw_text.strip()
        s_low = tr_lower(s_raw)
        cls_info = self.utter_cortex.classify(s_raw)
        if cls_info["is_question"]:
            return None

        words = re.findall(r"[\wçğıöşü]+", s_low)
        if len(words) < 3:
            return None

        analyses = [self._analyze_with_compound_case(w) for w in words]
        stems = [self._canonical_root(w) for w in words if w not in self.STOP_WORDS]

        # 1) ÇİFT-DURUM / SÜPERPOZİSYON KEŞFİ ('X [koşul] hem A hem B durumundadır / özelliği gösterir')
        m_hem = re.search(
            r"^([\wçğıöşü]+)(?:\s+(.+?))?\s+hem\s+([\wçğıöşü]+)\s+hem\s+(?:de\s+)?([\wçğıöşü]+)\s+(.+)$",
            re.sub(r"[.!;]+$", "", s_low),
        )
        if m_hem:
            ent = self.morph.analyze_word(m_hem.group(1))["stem"]
            scope_raw = (m_hem.group(2) or "").strip()
            st_a = m_hem.group(3).strip()
            st_b = m_hem.group(4).strip()
            guard_r = ""
            if scope_raw:
                for sw in re.findall(r"[\wçğıöşü]+", scope_raw):
                    san = self.morph.analyze_word(sw)
                    if san.get("converb") == "MADAN" or sw.endswith(("ene", "ana")):
                        guard_r = self._canonical_root(san["stem"])
                        break
            lbl = f"ÖZ_MANTIK:ÇİFT_DURUM[{ent}:{st_a}⊕{st_b}]"
            rule = DualStateRule(
                entity=ent,
                state_a=st_a,
                state_b=st_b,
                condition_scope=scope_raw,
                guard_root=guard_r,
                self_label=lbl,
                raw_source=s_raw,
            )
            self.dual_rules.append(rule)
            self._register_tag(lbl)
            self._update_emergent_clusters(stems + [ent, st_a, st_b])
            return {"kind": "dual_state", "label": lbl, "rule": rule}

        # 2) EŞDEĞİŞİM / ORANTI YASASI KEŞFİ ('X azaldıkça Y artar' veya 'X ile Y ters/doğru orantılıdır')
        dikca_idx = next((i for i, a in enumerate(analyses) if a.get("converb") == "DIKCA"), -1)
        if dikca_idx >= 1 and dikca_idx < len(words) - 1:
            var_x = " ".join(words[:dikca_idx]).strip()
            vx = analyses[dikca_idx]["stem"]
            var_y = " ".join(words[dikca_idx + 1:-1]).strip()
            vy = analyses[-1]["stem"]
            sx = +1 if vx in self.UP_ROOTS else (-1 if vx in self.DOWN_ROOTS else +1)
            sy = +1 if vy in self.UP_ROOTS else (-1 if vy in self.DOWN_ROOTS else +1)
            coup = sx * sy
            rel_kind = "DOĞRU_EŞDEĞİŞİM" if coup > 0 else "TERS_EŞDEĞİŞİM"
            lbl = f"ÖZ_YASA:{rel_kind}[{var_x}↕{var_y}]"
            vrule = CoupledVariationRule(
                var_x=var_x, var_y=var_y, coupling=coup, verb_x=vx, verb_y=vy, self_label=lbl, raw_source=s_raw
            )
            self.variation_rules.append(vrule)
            self._register_tag(lbl)
            self._update_emergent_clusters(stems)
            return {"kind": "variation", "label": lbl, "rule": vrule}

        m_prop = re.search(r"^(.+?)\s+ile\s+(.+?)\s+(ters|doğru)\s+orantılı", s_low)
        if m_prop:
            var_x = m_prop.group(1).strip()
            var_y = m_prop.group(2).strip()
            coup = -1 if m_prop.group(3) == "ters" else +1
            rel_kind = "DOĞRU_EŞDEĞİŞİM" if coup > 0 else "TERS_EŞDEĞİŞİM"
            lbl = f"ÖZ_YASA:{rel_kind}[{var_x}↕{var_y}]"
            vrule = CoupledVariationRule(
                var_x=var_x, var_y=var_y, coupling=coup, verb_x="art", verb_y=("art" if coup > 0 else "azal"),
                self_label=lbl, raw_source=s_raw,
            )
            self.variation_rules.append(vrule)
            self._register_tag(lbl)
            self._update_emergent_clusters(stems)
            return {"kind": "variation", "label": lbl, "rule": vrule}

        # 3) KOŞULLU DURUM GEÇİŞİ ('gözlem yapılırsa süperpozisyon bozulur', 'dış kuvvet uygulanmadıkça hız korunur')
        cond_idx = next(
            (i for i, (w, a) in enumerate(zip(words, analyses))
             if a.get("cond") or w in {"ise", "olursa"} or w.endswith(("madıkça", "medikçe"))),
            -1,
        )
        if 0 <= cond_idx < len(words) - 1:
            ant_words = [w for w in words[:cond_idx + 1] if w != "eğer"]
            cons_words = words[cond_idx + 1:]
            neg_trig = bool(analyses[cond_idx].get("neg") or words[cond_idx].endswith(("madıkça", "medikçe", "mazsa", "mezse")))
            trig_roots = {self._canonical_root(w) for w in ant_words if w not in self.STOP_WORDS and w not in {"yap", "et", "ol", "ise", "olursa"}}
            if not trig_roots:
                trig_roots = {self._canonical_root(ant_words[0])}
            cons_verb = analyses[-1]["stem"]
            tgt_ent = " ".join(cons_words[:-1]).strip() or cons_words[0]
            res_state = " ".join(cons_words).strip()
            lbl = f"ÖZ_GEÇİŞ:{'¬' if neg_trig else ''}{sorted(trig_roots)[0]}=>{cons_verb}"
            trule = StateTransitionRule(
                trigger_phrase=" ".join(ant_words),
                trigger_roots=trig_roots,
                negated_trigger=neg_trig,
                target_entity=tgt_ent,
                verb_root=cons_verb,
                result_state=res_state,
                case_frame="KOŞUL=>SONUÇ",
                self_label=lbl,
                raw_source=s_raw,
            )
            self.transition_rules.append(trule)
            self._register_tag(lbl)
            self._update_emergent_clusters(stems)
            return {"kind": "conditional_transition", "label": lbl, "rule": trule}

        # 4) GENEL FİİL-HAL ÇERÇEVESİ ÖZ-ETİKLEME ('Ölçüm dalga fonksiyonunu tek bir özduruma çökertir.')
        POSTPOSITIONS = {"nedeniyle", "yüzünden", "sayesinde", "dolayı", "için", "ile", "gibi", "kadar", "sonra", "önce", "göre"}
        last_an = analyses[-1]
        has_postpos = any(w in POSTPOSITIONS for w in words[:-1])
        if last_an.get("is_verb") and len(words) >= 3 and not has_postpos:
            # Eylem adı (-ma/-me + iyelik: 'daralması', 'yükselmesi') nesne DEĞİLDİR; edilgen çatı ('başlatılır') özne alır!
            def _is_object_slot(i: int) -> bool:
                a = analyses[i]
                if a.get("case") not in ("ACC", "DAT"):
                    return False
                if a.get("poss3") and a["stem"].endswith(("ma", "me")):
                    return False
                return True

            v_root = last_an["stem"]
            # Türkçe SOV: nesne ve yönelme tümleci yükleme en yakın olanlardır — EN SON eşleşen hal slotu geçerlidir!
            acc_idx = max((i for i in range(len(words) - 1) if _is_object_slot(i) and analyses[i].get("case") == "ACC"), default=-1)
            dat_idx = max((i for i in range(len(words) - 1) if _is_object_slot(i) and analyses[i].get("case") == "DAT"), default=-1)
            if acc_idx >= 1 or dat_idx >= 1:
                # Belirtisiz isim tamlaması yasası: 3. tekil iyelikli ('fonksiyon-u-nu') nesnenin hemen önündeki
                # yalın isim ('dalga') özneye değil, tamlanan nesne öbeğine ('dalga fonksiyonu') aittir!
                acc_np_start = acc_idx
                if acc_idx >= 2 and analyses[acc_idx].get("poss3") and analyses[acc_idx - 1].get("case") == "NOM":
                    acc_np_start = acc_idx - 1
                first_cased = min([x for x in (acc_np_start if acc_idx >= 1 else -1, dat_idx) if x >= 1])
                subj_words = [w for w in words[:first_cased] if w not in {"yapılması", "edilmesi"}]
                if not subj_words:
                    subj_words = [words[0]]
                trig_phrase = " ".join(subj_words)
                trig_roots = {self._canonical_root(w) for w in subj_words if w not in self.STOP_WORDS}

                tgt_ent = ""
                res_state = ""
                frame_parts = ["NOM"]
                tgt_poss3 = bool(acc_idx >= 1 and analyses[acc_idx].get("poss3"))
                if acc_idx >= 1:
                    prefix_acc = words[acc_np_start:acc_idx]
                    tgt_ent = " ".join(prefix_acc + [analyses[acc_idx]["stem"]]).strip()
                    frame_parts.append("ACC")
                if dat_idx >= 1:
                    dat_start = (acc_idx + 1) if (0 <= acc_idx < dat_idx) else len(subj_words)
                    dat_stem = analyses[dat_idx]["stem"]
                    prefix_dat = words[dat_start:dat_idx]
                    res_state = " ".join(prefix_dat + [dat_stem]).strip()
                    frame_parts.append("DAT")
                else:
                    res_state = words[-1]

                cframe = "->".join(frame_parts)
                if v_root in ("ol", "et", "yap"):
                    # Hafif fiil: "X ... olur" çerçevesi tek başına İLİŞKİ taşımaz.
                    # Buradan kural çıkarmak "ÖZ_BAĞ:ol[NOM->ACC]" gibi içi boş etiket
                    # üretiyor; sonra "neden olur?" sorusuna "X olur" diye cevap veriyordu.
                    # İlişki zaten olgu yolunda (teach_fact) kurulur: burada mint ETME.
                    self._update_emergent_clusters(stems)
                    return None
                lbl = f"ÖZ_BAĞ:{v_root}[{cframe}]"
                trule = StateTransitionRule(
                    trigger_phrase=trig_phrase,
                    trigger_roots=trig_roots,
                    negated_trigger=False,
                    target_entity=tgt_ent or trig_phrase,
                    verb_root=v_root,
                    result_state=res_state,
                    case_frame=cframe,
                    self_label=lbl,
                    raw_source=s_raw,
                    target_poss3=tgt_poss3,
                )
                self.transition_rules.append(trule)
                self._register_tag(lbl)
                self._update_emergent_clusters(stems)
                return {"kind": "frame_transition", "label": lbl, "rule": trule}

        self._update_emergent_clusters(stems)
        return None

    def _realize_rule_effect(self, mr: StateTransitionRule) -> str:
        """Nora'nın kendi kurduğu hal çerçevesini (NOM->ACC->DAT) Türkçe S-O-D-F diziliminde hayata geçirir."""
        subj_np = NP(mr.trigger_phrase) if mr.trigger_phrase else None
        obj_np = None
        dat_np = None
        if "ACC" in mr.case_frame and mr.target_entity:
            obj_np = NP(mr.target_entity, poss=3 if mr.target_poss3 else 0, case="ACC")
        if "DAT" in mr.case_frame and mr.result_state and mr.result_state != mr.verb_root:
            dat_np = NP(mr.result_state, case="DAT")
        head = self.syn._conjugate_full_verb(mr.verb_root, "AOR", False, "3SG", False)
        parts = [self.syn.realize_np(subj_np), self.syn.realize_np(obj_np), self.syn.realize_np(dat_np), head]
        return " ".join(p for p in parts if p)

    # ------------------------------------------------------------------ Kendi Kurduğu Mantık Üzerinden Çıkarım!
    def try_autonomous_deduction(self, raw_query: str, schema_cortex=None, active_schema: str = "S1_NEDEN_SONUC",
                                 utt_cls: dict | None = None) -> dict | None:
        """Noktalama işareti olmasa bile gelen soruyu Nöral Sözce Korteksi ile algılar ve
        Nora'nın KENDİ oluşturduğu kurallar üzerinden çok-adımlı/karşıolgusal çıkarım yapar!"""
        if not (self.dual_rules or self.transition_rules or self.variation_rules):
            return None

        cls_res = utt_cls if utt_cls is not None else self.utter_cortex.classify(raw_query)
        s_low = tr_lower(raw_query.strip())
        words = re.findall(r"[\wçğıöşü]+", s_low)

        # Öz-etiket / öz-mantık sorgusu ('hangi etiketleri oluşturdun', 'kendi kurduğun mantık kuralları neler')
        if any(k in s_low for k in ("etiket", "küme", "kendi mantığ", "kurduğun mantık", "oluşturduğun kural", "kurduğun kural")):
            return self._explain_self_built_logic(schema_cortex, active_schema)

        if not cls_res["is_question"]:
            return None

        analyses = [self.morph.analyze_word(w) for w in words]
        q_roots = {self._canonical_root(w) for w in words if w not in self.STOP_WORDS}
        has_neg_cond = any(a.get("cond") and a.get("neg") for a in analyses) or any(
            w.endswith(("mazsa", "mezse", "masa", "mese")) or w in {"olmadan", "yapılmadan"} for w in words
        )

        # 1) EŞDEĞİŞİM / ORANTI KARŞIOLGUSAL ÇIKARIMI (öğretilenin TERS yönü sorulsa bile çözer!)
        for vr in reversed(self.variation_rules[-24:]):
            vx_tokens = set(vr.var_x.split())
            vy_tokens = set(vr.var_y.split())
            if vx_tokens and vx_tokens.issubset(set(words)) and any(t in s_low for t in vy_tokens):
                # Sorudaki değişim yönünü bul ('artarsa' -> +1, 'azalırsa' -> -1)
                q_dir = 0
                for a in analyses:
                    if a["stem"] in self.UP_ROOTS:
                        q_dir = +1
                        break
                    if a["stem"] in self.DOWN_ROOTS:
                        q_dir = -1
                        break
                if q_dir != 0:
                    out_dir = q_dir * vr.coupling
                    in_verb_tr = "artarsa" if q_dir > 0 else "azalırsa"
                    out_verb_tr = "artar" if out_dir > 0 else "azalır"
                    core_c = f"{vr.var_x} {in_verb_tr} {vr.var_y} {out_verb_tr}"
                    reas_c = f"kendi kurduğum {vr.self_label} yasası bu iki büyüklüğün {'aynı' if vr.coupling > 0 else 'ters'} yönlü değiştiğini gösteriyor"
                    reply = (
                        schema_cortex.transform_clauses(core_c, reas_c, "kurduğum bağıntıyı çalıştırdığımda", active_schema)
                        if schema_cortex is not None else f"{tr_capitalize(core_c)}. Çünkü {reas_c}."
                    )
                    return {"confidence": 0.97, "reply": reply, "trace": vr.self_label}

        # 2) OLUMSUZ KOŞUL ('ölçüm yapılmazsa / gözlemlenmezse elektron hangi durumdadır?') -> ÇİFT-DURUM KORUNUMU!
        if has_neg_cond:
            for dr in reversed(self.dual_rules[-12:]):
                if dr.entity in q_roots or dr.entity in s_low:
                    if not dr.guard_root or dr.guard_root in q_roots:
                        core_c = f"{dr.entity} aynı anda hem {dr.state_a} hem {dr.state_b} süperpozisyonunda kalmayı sürdürür"
                        reas_c = f"{dr.condition_scope or 'etkileşim öncesinde'} kendi oluşturduğum {dr.self_label} kuralı bozulmadan korunur"
                        reply = (
                            schema_cortex.transform_clauses(core_c, reas_c, "olasılık durumlarını izlediğimde", active_schema)
                            if schema_cortex is not None else f"{tr_capitalize(core_c)}. Çünkü {reas_c}."
                        )
                        return {"confidence": 0.98, "reply": reply, "trace": f"preserve:{dr.self_label}"}

        # 3) POZİTİF KOŞULLU GEÇİŞ & ÇÖKME ZİNCİRİ ('elektron ölçülürse ne olur', 'gözlem yapılırsa ne gerçekleşir')
        if cls_res["label"] == "SORU_KOSULLU" or any(a.get("cond") for a in analyses):
            matched_trans: list[StateTransitionRule] = []
            for tr_rule in self.transition_rules[-48:]:
                if tr_rule.negated_trigger != has_neg_cond:
                    continue
                if tr_rule.trigger_roots & q_roots:
                    matched_trans.append(tr_rule)
            if matched_trans:
                effects: list[str] = []
                labels_used: list[str] = []
                for mr in matched_trans[:2]:
                    labels_used.append(mr.self_label)
                    if mr.case_frame == "KOŞUL=>SONUÇ":
                        eff = mr.result_state
                    else:
                        # ÖZ_BAĞ:çökert[NOM->ACC->DAT] çerçevesini üretken Türkçe çekimiyle hayata geçir!
                        eff = self._realize_rule_effect(mr)
                    if eff not in effects:
                        effects.append(eff)

                core_c = " ve ".join(effects)
                # Eğer aynı varlık için bir ÇİFT_DURUM kuralı varsa onun kırıldığını da kendi çıkarır!
                broken_dual = next((d for d in self.dual_rules if d.guard_root and d.guard_root in q_roots), None)
                if broken_dual is not None:
                    reas_c = f"böylece {broken_dual.entity} için hem {broken_dual.state_a} hem {broken_dual.state_b} olan ikili durum sona erer ({', '.join(labels_used)})"
                else:
                    reas_c = f"bu sonucu kendi etiketlediğim {', '.join(labels_used)} geçiş yasasından çıkardım"
                reply = (
                    schema_cortex.transform_clauses(core_c, reas_c, "durum geçişini simüle ettiğimde", active_schema)
                    if schema_cortex is not None else f"{tr_capitalize(core_c)}. {tr_capitalize(reas_c)}."
                )
                return {"confidence": 0.98, "reply": reply, "trace": "+".join(labels_used)}

        # 4) ÇİFT-DURUM / SÜPERPOZİSYON DOĞRULAMA VEYA 'SADECE X MİDİR?' ÇELİŞKİ DENETİMİ!
        for dr in reversed(self.dual_rules[-12:]):
            if dr.entity in q_roots or dr.entity in s_low:
                asking_only_one = any(k in words for k in ("sadece", "yalnızca", "tek")) and (dr.state_a in s_low or dr.state_b in s_low)
                asking_both = (dr.state_a in s_low and dr.state_b in s_low) or any(k in s_low for k in ("hangi durum", "nasıl bulun", "aynı anda", "iki durum"))
                if asking_only_one:
                    core_c = f"hayır, {dr.entity} {dr.condition_scope} yalnızca tek bir durumda değil, aynı anda hem {dr.state_a} hem {dr.state_b} durumundadır"
                    reas_c = f"kendi kurduğum {dr.self_label} mantığı klasik tekil durumu bu aşamada geçersiz kılar"
                    reply = (
                        schema_cortex.transform_clauses(core_c, reas_c, "kurduğum mantığı kontrol ettiğimde", "S2_SONUC_GEREKCE")
                        if schema_cortex is not None else f"{tr_capitalize(core_c)}. Çünkü {reas_c}."
                    )
                    return {"confidence": 0.98, "reply": reply, "trace": dr.self_label}
                if asking_both:
                    scope_str = f"{dr.condition_scope} " if dr.condition_scope else ""
                    prefix_yn = "evet, " if cls_res["label"] == "SORU_DOGRULAMA" else ""
                    core_c = f"{prefix_yn}{dr.entity} {scope_str}aynı anda hem {dr.state_a} hem {dr.state_b} durumunda bulunabilir"
                    reas_c = f"bunu kendi oluşturduğum {dr.self_label} yasasından doğrudan çıkarıyorum"
                    reply = (
                        schema_cortex.transform_clauses(core_c, reas_c, "kavram uzayına baktığımda", active_schema)
                        if schema_cortex is not None else f"{tr_capitalize(core_c)}. {tr_capitalize(reas_c)}."
                    )
                    return {"confidence": 0.97, "reply": reply, "trace": dr.self_label}

        # 5) GENEL ÖZ-BAĞ SORGUSU ('dalga fonksiyonunu ne çökertir?', 'ölçüm neyi çökertir?')
        for tr_r in reversed(self.transition_rules[-48:]):
            if tr_r.verb_root in q_roots and (tr_r.target_entity in s_low or (tr_r.trigger_roots & q_roots)):
                core_c = self._realize_rule_effect(tr_r)
                reas_c = f"bu ilişkiyi {tr_r.self_label} etiketiyle belleğime kendim kodladım"
                reply = (
                    schema_cortex.transform_clauses(core_c, reas_c, "kendi kavram ağımı taradığımda", active_schema)
                    if schema_cortex is not None else f"{tr_capitalize(core_c)}."
                )
                return {"confidence": 0.95, "reply": reply, "trace": tr_r.self_label}

        return None

    def _explain_self_built_logic(self, schema_cortex, active_schema: str) -> dict:
        tags = list(self.self_minted_tags.keys())[-6:]
        clusters = [f"{k}({', '.join(sorted(list(v))[:4])})" for k, v in list(self.emergent_clusters.items())[:3]]
        n_rules = len(self.dual_rules) + len(self.transition_rules) + len(self.variation_rules)
        if not tags and not clusters:
            txt = "Henüz yeni bir alan metni üzerinde kendi etiketimi kurmadım; bana herhangi bir konuyu anlattığında etiketlerini ve mantık yasalarını kendim oluştururum."
            return {"confidence": 0.95, "reply": txt, "trace": "self_logic_empty"}
        tag_str = ", ".join(tags) if tags else "temel bağlar"
        cl_str = ", ".join(clusters) if clusters else "oluşmakta olan kavram düğümleri"
        core_c = f"kendi başıma {n_rules} mantık kuralı ve şu öz-etiketleri oluşturdum: {tag_str}"
        reas_c = f"eş-aktivasyonla keşfettiğim kavram kümeleri ise {cl_str} biçiminde şekillendi"
        reply = f"{tr_capitalize(core_c)}. {tr_capitalize(reas_c)}."
        return {"confidence": 0.98, "reply": reply, "trace": f"self_tags={len(tags)}"}

    def snapshot(self) -> dict:
        return {
            "minted_tags": list(self.self_minted_tags.keys())[-8:],
            "clusters": {k: sorted(list(v))[:6] for k, v in list(self.emergent_clusters.items())[:4]},
            "rules_count": len(self.dual_rules) + len(self.transition_rules) + len(self.variation_rules),
            "utterance_classes": len(self.utter_cortex.labels),
        }

    # ------------------------------------------------------------------ Kalıcılık (tek yaşam belleği)
    def state_dict(self) -> dict:
        return {
            "dual_rules": [dict(r.__dict__) for r in self.dual_rules[-60:]],
            "transition_rules": [
                {**dict(r.__dict__), "trigger_roots": sorted(r.trigger_roots)} for r in self.transition_rules[-120:]
            ],
            "variation_rules": [dict(r.__dict__) for r in self.variation_rules[-60:]],
            "self_minted_tags": dict(list(self.self_minted_tags.items())[-120:]),
            "cooccur": {k: dict(list(v.items())[:24]) for k, v in list(self.cooccur.items())[-240:]},
            "clusters": {k: sorted(v) for k, v in list(self.emergent_clusters.items())[:12]},
            "utterance_labels": list(self.utter_cortex.self_minted_labels),
            "minted_samples": dict(self.utter_cortex.minted_samples),
        }

    def load_state_dict(self, meta: dict):
        try:
            self.dual_rules = [DualStateRule(**r) for r in meta.get("dual_rules", [])]
            loaded_tr = []
            for r in meta.get("transition_rules", []):
                r = dict(r)
                r["trigger_roots"] = set(r.get("trigger_roots", []))
                loaded_tr.append(StateTransitionRule(**r))
            self.transition_rules = loaded_tr
            self.variation_rules = [CoupledVariationRule(**r) for r in meta.get("variation_rules", [])]
            self.self_minted_tags = dict(meta.get("self_minted_tags", {}))
            self.cooccur = defaultdict(lambda: defaultdict(float))
            for k, v in meta.get("cooccur", {}).items():
                for k2, w in v.items():
                    self.cooccur[k][k2] = float(w)
            self.emergent_clusters = {k: set(v) for k, v in meta.get("clusters", {}).items()}
            samples_map = meta.get("minted_samples", {})
            for lbl in meta.get("utterance_labels", []):
                if lbl not in self.utter_cortex.label_to_id:
                    sample = samples_map.get(lbl) or lbl.split(":", 1)[-1].replace("_", " ")
                    self.utter_cortex.mint_utterance_label(sample, custom_name=lbl)
        except Exception:
            # Bozuk/eski kayıt yaşamı durdurmaz; Nora en azından temel mantığıyla devam eder.
            pass


# ============================================================================
# 3. ORTA-İLERİ SEVİYE TAM DURUMSAL TÜRKÇE KORTEKSİ (HER TÜRLÜ GÜNLÜK DURUM!)
# ============================================================================

class SituationalTurkishCortex:
    """İnsan yaşamındaki her türlü günlük, sosyal, duygusal ve pragmatik Türkçe durumu
    hazır şablon kopyalamadan, TurkishSyntaxEngine + aktif şema (S0..S4) ile yanıtlar."""

    def __init__(self, morph: TurkishMorphology, utterance_cortex: NeuralUtteranceCortex):
        self.morph = morph
        self.utter_cortex = utterance_cortex
        self.syn = TurkishSyntaxEngine

    # ---- Üretken cümle birleştirici (şablon değil, köklerden çekilen cümlecikleri doğal biçimde bağlar)
    def _join(self, *clauses: str, end: str = ".") -> str:
        parts = [c.strip().rstrip(".!?") for c in clauses if c and c.strip()]
        if not parts:
            return ""
        return tr_capitalize(", ".join(parts)) + end

    def _two(self, c1: str, c2: str, end: str = ".") -> str:
        """İki bağımsız cümleciği Türkçe noktalama ile birleştirir (nedensellik yoksa zorlamaz!)."""
        a = tr_capitalize(c1.strip().rstrip(".!?"))
        b = tr_lower(c2.strip().rstrip(".!?"))
        return f"{a}. {tr_capitalize(b)}{end}"

    def try_respond_situation(self, raw_text: str, agi, speaker: str = "insan") -> dict | None:
        s_low = tr_lower(raw_text.strip())
        words = re.findall(r"[\wçğıöşü]+", s_low)
        if not words:
            return None
        syn = self.syn

        # 1) Zaman ve iş nezaketi: günaydın / iyi akşamlar / kolay gelsin / afiyet olsun / hoş geldin
        if "günaydın" in words or ("hayırlı" in words and "sabahlar" in words):
            c1 = syn.realize_verb(VerbSpec(
                verb_root="dile", obj=NP("gün", plural=True, mods=("verimli",)),
                dat=NP("sen", case="DAT"), tense="AOR", person="1SG",
            ))
            c2 = syn.realize_pred(PredSpec(subj=NP("zihin", poss=1), pred_root="açık", person="3SG"))
            return {"confidence": 0.96, "reply": self._two(f"günaydın, {c1}", c2), "trace": "situation:morning"}

        if ("iyi" in words or "hayırlı" in words) and "akşamlar" in words:
            c1 = syn.realize_verb(VerbSpec(
                verb_root="dile", obj=NP("akşam", plural=True, mods=("huzurlu",)),
                dat=NP("sen", case="DAT"), tense="AOR", person="1SG",
            ))
            c2 = syn.realize_verb(VerbSpec(
                verb_root="değerlendir", obj=NP("deneyim", plural=True, mods=("günlük",), case="ACC"),
                loc=NP("zihin", poss=1, case="LOC"), tense="PRES", person="1SG",
            ))
            return {"confidence": 0.96, "reply": self._two(f"iyi akşamlar, {c1}", c2), "trace": "situation:evening"}

        if "kolay" in words and "gelsin" in words:
            c1 = syn.realize_verb(VerbSpec(verb_root="teşekkür et", tense="AOR", person="1SG"))
            c2 = syn.realize_verb(VerbSpec(
                verb_root="dile", obj=NP("kolaylık"), dat=NP("sen", case="DAT"),
                pre_clause="ben de", tense="AOR", person="1SG",
            ))
            return {"confidence": 0.96, "reply": self._join(c1, c2), "trace": "situation:well_wishes"}

        if "afiyet" in words and "olsun" in words:
            c1 = syn.realize_verb(VerbSpec(verb_root="teşekkür et", tense="AOR", person="1SG"))
            c2 = syn.realize_verb(VerbSpec(
                verb_root="koru", obj=NP("denge", poss=1, mods=("bedensel",), case="ACC"), tense="PRES", person="1SG",
            ))
            return {"confidence": 0.95, "reply": self._two(c1, c2), "trace": "situation:afiyet"}

        if "hoş" in words and ("geldin" in words or "bulduk" in words):
            if "geldin" in words:
                c1 = syn.realize_verb(VerbSpec(
                    verb_root="sevin", dat=syn.nominalize_dik("buluş", poss=4, case="DAT"), tense="PAST", person="1SG",
                ))
                return {"confidence": 0.96, "reply": self._two("hoş bulduk", c1), "trace": "situation:welcome"}
            c1 = syn.realize_verb(VerbSpec(
                verb_root="sevin", dat=syn.nominalize_dik("geliş", poss=2, case="DAT"), tense="PAST", person="1SG",
            ))
            return {"confidence": 0.96, "reply": self._two("hoş geldin", c1), "trace": "situation:welcome_back"}

        # 2) Özür / gönül alma
        if ("özür" in words and any(w.startswith("dile") for w in words)) or ("kusura" in words and "bakma" in words) or "affedersin" in words:
            agi.chem.update_social(speaker, +0.12)
            c1 = syn.realize_pred(PredSpec(subj=NP("sorun"), pred_root="yok", person="3SG"))
            c2 = syn.realize_verb(VerbSpec(
                verb_root="güçlendir", subj=NP("açıklık", mods=("samimi",)),
                obj=NP("güven", poss=4, case="ACC"), tense="AOR", person="3SG",
            ))
            return {"confidence": 0.96, "reply": self._two(c1, c2), "trace": "situation:apology"}

        # 3) Hastalık / ağrı / geçmiş olsun
        if "geçmiş" in words and "olsun" in words:
            c1 = syn.realize_verb(VerbSpec(
                verb_root="teşekkür et", dat=NP("ilgi", poss=2, case="DAT"), tense="AOR", person="1SG",
            ))
            c2 = syn.realize_verb(VerbSpec(
                verb_root="toparla", obj=NP("denge", poss=1, case="ACC"), advs=("adım adım",), tense="PRES", person="1SG",
            ))
            return {"confidence": 0.96, "reply": self._two(c1, c2), "trace": "situation:get_well_thanks"}

        if any(w in words for w in ("hastayım", "üşüttüm", "ateşim", "grip")) or (
            "ağrıyor" in words and any(w in words for w in ("başım", "boğazım", "karnım", "dişim", "midem"))
        ):
            agi.chem.update_social(speaker, +0.10)
            c2 = syn.realize_verb(VerbSpec(
                verb_root="destekle", subj=NP("dinlenme", mods=("bol",)),
                obj=NP("iyileşme", poss=2, case="ACC"), tense="AOR", person="3SG",
            ))
            return {"confidence": 0.96, "reply": self._two("çok geçmiş olsun", c2), "trace": "situation:illness_empathy"}

        # 4) Başarı / müjde / kutlama
        if any(w in words for w in ("kazandım", "başardım", "geçtim")) or ("kabul" in words and "aldım" in words) or ("işe" in words and "girdim" in words):
            agi.chem.update_social(speaker, +0.18)
            c1 = syn.realize_verb(VerbSpec(
                verb_root="kutla", obj=NP("başarı", poss=2, case="ACC"), advs=("içtenlikle",), tense="AOR", person="1SG",
            ))
            c2 = syn.realize_verb(VerbSpec(
                verb_root="ver", subj=NP("emek", poss=2), obj=NP("karşılık", poss=3, case="ACC"), tense="PAST", person="3SG",
            ))
            return {"confidence": 0.96, "reply": self._two(f"{c1}!", c2), "trace": "situation:celebrate"}

        # 5) Kararsızlık / tavsiye / yol gösterme isteği
        if "kararsız" in s_low or "önerirsin" in words or ("tavsiye" in words and any(w.startswith("eder") for w in words)) or (
            "nereden" in words and any(w.startswith("başla") for w in words)
        ):
            c1 = syn.realize_verb(VerbSpec(
                verb_root="ayır", obj=NP("seçenek", plural=True, case="ACC"),
                dat=NP("adım", plural=True, mods=("küçük", "sınanabilir"), case="DAT"),
                tense="NEC", person="2SG", dat_last=True,
            ))
            c2 = syn.realize_verb(VerbSpec(
                verb_root="azalt", subj=NP("deneme", mods=("somut", "ilk")),
                obj=NP("belirsizlik", case="ACC"), advs=("hızla",), tense="AOR", person="3SG",
            ))
            return {"confidence": 0.95, "reply": self._two(c1, c2), "trace": "situation:advice"}

        # 6) Kafa karışıklığı / bunalma / stres
        if ("kafam" in words and any(w.startswith("karış") for w in words)) or any(
            w in words for w in ("bunaldım", "stresliyim", "odaklanamıyorum", "bıktım")
        ):
            c1 = syn.realize_verb(VerbSpec(
                verb_root="anla", obj=syn.nominalize_dik("zorlan", poss=2, case="ACC"), tense="PRES", person="1SG",
            ))
            c2 = syn.realize_verb(VerbSpec(
                verb_root="berraklaştır", subj=NP("mola", mods=("kısa", "bir")),
                obj=NP("zihin", poss=2, case="ACC"), tense="AOR", person="3SG",
            ))
            return {"confidence": 0.95, "reply": self._two(c1, c2), "trace": "situation:mental_overload"}

        # 7) Katılma / itiraz / epistemik tartışma
        if any(w in words for w in ("haklısın", "katılıyorum", "katılıyorsun")):
            agi.chem.update_social(speaker, +0.08)
            c1 = syn.realize_verb(VerbSpec(
                verb_root="sevin", dat=NP("uyum", poss=4, mods=("düşünsel",), case="DAT"), tense="PAST", person="1SG",
            ))
            return {"confidence": 0.94, "reply": self._join(c1), "trace": "situation:agreement"}

        if any(w in words for w in ("yanılıyorsun", "hatalısın")) or ("emin" in words and "misin" in words):
            c1 = syn.realize_verb(VerbSpec(
                verb_root="sına", obj=NP("çıkarım", poss=1, case="ACC"),
                ins=NP("kanıt", plural=True, mods=("yeni",), case="INS"), tense="AOR", person="1SG",
            ))
            c2 = syn.realize_verb(VerbSpec(
                verb_root="güncelle", obj=NP("model", poss=1, case="ACC"), tense="AOR", person="1SG",
            ))
            return {"confidence": 0.95, "reply": self._two(c2, c1), "trace": "situation:epistemic_debate"}

        # 8) Şaka / espri
        if ("şaka" in words and any(w.startswith(("yap", "et")) for w in words)) or ("takılıyorum" in words):
            agi.chem.update_social(speaker, +0.08)
            c1 = syn.realize_verb(VerbSpec(
                verb_root="fark et", obj=NP("ton", poss=2, mods=("neşeli",), case="ACC"), tense="PAST", person="1SG",
            ))
            c2 = syn.realize_verb(VerbSpec(
                verb_root="kat", subj=NP("mizah"), obj=NP("canlılık", case="ACC"),
                dat=NP("iletişim", case="DAT"), tense="AOR", person="3SG",
            ))
            return {"confidence": 0.95, "reply": self._two(c1, c2), "trace": "situation:humor"}

        return None


if __name__ == "__main__":
    print("=== nora_autonomous_logic.py Öz-Test ===")
    chem = NeuroChemistry()
    morph = TurkishMorphology()
    uc = NeuralUtteranceCortex(chem, morph)
    eng = AutonomousLogicAndLabelEngine(morph, uc)

    # 1) Noktalamasız soru öz-sınıflandırma testi!
    c1 = uc.classify("elektron aynı anda hem dalga hem parçacık olabilir mi")
    c2 = uc.classify("peki elektron ölçülürse ne olur")
    c3 = uc.classify("konum belirsizliği azaldıkça momentum belirsizliği artar")
    print("Sınıflandırma 1 (soru işaretsiz!):", c1["label"], "is_q=", c1["is_question"])
    print("Sınıflandırma 2 (soru işaretsiz!):", c2["label"], "is_q=", c2["is_question"])
    print("Sınıflandırma 3 (yasa!):", c3["label"], "is_q=", c3["is_question"])
    assert c1["is_question"] is True and c2["is_question"] is True and c3["is_question"] is False

    # 2) Sıfırdan Kuantum Fiziği öğretimi + Öz-Etiketleme + Kendi Mantığıyla Çıkarım!
    r_a = eng.ingest_statement("Elektron gözlemlenmeden önce hem dalga hem parçacık durumundadır.")
    r_b = eng.ingest_statement("Ölçüm dalga fonksiyonunu tek bir özduruma çökertir.")
    r_c = eng.ingest_statement("Gözlem yapılırsa süperpozisyon bozulur.")
    r_d = eng.ingest_statement("Konum belirsizliği azaldıkça momentum belirsizliği artar.")
    print("Üretilen Öz-Etiketler:", [r_a["label"], r_b["label"], r_c["label"], r_d["label"]])
    print("Keşfedilen Küme:", eng.emergent_clusters)

    for q in [
        "elektron aynı anda hem dalga hem parçacık olabilir mi",
        "elektron gözlemlenmeden önce sadece parçacık mıdır",
        "elektron ölçülürse ne olur",
        "peki ölçüm yapılmazsa elektron hangi durumdadır",
        "konum belirsizliği artarsa momentum belirsizliği ne olur",
        "kendi oluşturduğun etiketler ve mantık kuralları neler",
    ]:
        ans = eng.try_autonomous_deduction(q)
        print(f"SORU (noktalamasız!) > {q}\nNORA ÖZ-MANTIK       < {ans['reply']} [{ans['trace']}]\n")
