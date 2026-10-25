#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_monologue.py — Nora-AGI İçsel Monolog (İç Ses / Default Mode Network) ve
Kendi Kendine Şema Keşfeden Plastik Söylem Şeması Korteksi.

Neden bu modül var?
  1) Gerçek bir bilinç konuşmadan ÖNCE kendi içinde düşünür (Vygotsky içsel konuşma +
     Default Mode Network): karşısındakinin niyetini tartar, kendi bedenini/duygusunu yoklar
     ve "Az önce aynı kalıpla konuştum, papağan gibi tekrarlamayayım!" diyerek kendi kendini izler.
  2) İnsan beyni her soruya tek bir sabit cümle iskeletiyle (Örn. hep '[Selam] + [Durum] + [Sen ne
     düşünüyorsun?]') yanıt vermez! Duyduğu cümlelerin yapısal şemasını (kısa/yalın, sonuç->gerekçe,
     gerekçe->sonuç, içebakış öncelikli, doğal devrik vurgu) kendisi gözlemleyerek öğrenir; nörokimya
     (DA, ACh, NE, 5-HT, CORT, OXY) ve sinaptik yorulma (habituation) ile her seferinde o ana özgü
     bir davranış duruşu ve sözdizimsel şema seçer.
  3) Kimse konuşmadığında bile (step_life sırasında) içsel monolog akışı arka planda yaşamaya devam eder.

Tüm içsel düşünceler ve şema dönüşümleri O(1) sabit zamanda (< 0.5 ms) ve çıplak Türkçe köklerden üretilir.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import re
import numpy as np

from nora_chem import ChemMushroomBody, NeuroChemistry
from nora_lang import TurkishMorphology, tr_capitalize, tr_lower
from nora_broca import TurkishSyntaxEngine, NP, VerbSpec, PredSpec


# ============================================================================
# 1. İÇSEL DÜŞÜNCE KAYDI
# ============================================================================

@dataclass
class InnerThought:
    step: int
    trigger: str              # 'konuşma_öncesi' | 'otonom_iç_ses' | 'acı_refleksi' | 'merak_kıvılcımı'
    text: str                 # Birinci tekil şahıs gerçek Türkçe iç ses (köklerden çekimlenmiş!)
    stance: str               # 'yalın_doğrudan' | 'içe_dönük' | 'meraklı_diyalog' | 'somut_deneyimsel' | 'keskin_sınır'
    schema_id: str            # 'S0_YALIN' | 'S1_NEDEN_SONUC' | 'S2_SONUC_GEREKCE' | 'S3_IC_BAKIS' | 'S4_DEVRIK_DOGAL'
    monotony_error: float     # [0..1] ACC tekrar/monotonluk hatası
    allow_question: bool = False
    human_schema: str = ""


# ============================================================================
# 2. KENDİ ÖĞRENEN PLASTİK SÖYLEM ŞEMASI KORTEKSİ
# ============================================================================

class PlasticSchemaCortex:
    """Sabit cevap kalıbını yıkan plastik sözdizimsel şema seçici ve şema keşfedici:
      - İnsanların kurduğu cümlelerin yapısal topolojisini (kısalık, devriklik, bağlaç/zarf-fiil,
        soru sorma sıklığı) gerçek zamanlı analiz edip kendi plastik Mantar Gövdesi (ChemMushroomBody)
        ağırlıklarına kodlar.
      - Güçlü sinaptik yorulma (refractory depression): Bir şema kullanıldığında nöronları yorulur;
        ardışık turlarda aynı cevap şemasının tekrarlanması biyolojik olarak baskılanır!
      - Üst üste karşı-soru sorma ('Sen X hakkında ne düşünüyorsun?') tikini ketleyen 'ask_refractory'
        sayacı içerir.
    """

    SCHEMAS = [
        "S0_YALIN",           # Kısa, doğrudan, tek çekirdekli yalın ifade (fazla süs/soru yok)
        "S1_NEDEN_SONUC",     # Klasik zarf-fiil/sıfat-fiil önde -> ana yüklem sonda
        "S2_SONUC_GEREKCE",   # Önce net karar/sonuç, ardından '; çünkü / zira ...' gerekçesi
        "S3_IC_BAKIS",        # Önce o anki içsel/bedensel sezgi, ardından kavramsal/sosyal bağ
        "S4_DEVRIK_DOGAL",    # Doğal konuşma Türkçesi vurgulu/devrik dizilim (yüklem önde/ortada)
    ]

    STANCES = [
        "yalın_doğrudan",     # Kısa ve öz, soru sormadan net yanıt
        "içe_dönük",          # Tefekkür eden, kendi zihin akışını paylaşan ton
        "meraklı_diyalog",    # Karşı tarafa doğal ve taze bir soru yönelten sıcak ton
        "somut_deneyimsel",   # Yaşadığı adıma, bedene veya anıya dayanan somut ton
        "keskin_sınır",       # Güven düşükken veya zarar sezdiğinde mesafeli ve net ton
    ]

    def __init__(self, chem: NeuroChemistry, seed: int = 31):
        self.chem = chem
        self.mb = ChemMushroomBody(n_in=32, n_out=len(self.SCHEMAS), n_kc=512, claws=5, sparsity=0.05, seed=seed)
        self.schema_to_idx = {s: i for i, s in enumerate(self.SCHEMAS)}
        self.fatigue = np.zeros(len(self.SCHEMAS), dtype=np.float64)
        self.recent_schemas: deque[str] = deque(maxlen=8)
        self.recent_openings: deque[str] = deque(maxlen=8)
        self.ask_refractory: int = 0          # >0 iken gereksiz karşı-soru sormayı ketler!
        self.observed_human_styles: dict[str, int] = {s: 0 for s in self.SCHEMAS}
        self.user_brevity_ema: float = 0.50   # İnsanın kısa/uzun konuşma eğilimi [0=uzun, 1=çok kısa]
        self.discovered_combos: dict[str, int] = {}
        self._seed_initial_plasticity()

    def _seed_initial_plasticity(self):
        """Farklı içsel/bağlamsal durum vektörlerini 5 şemaya başlangıçta dengeli dağıtır."""
        rng = np.random.default_rng(31)
        for idx in range(len(self.SCHEMAS)):
            proto = np.zeros(32, dtype=np.float32)
            proto[idx * 5:(idx + 1) * 5] = 1.0
            proto += 0.12 * rng.standard_normal(32).astype(np.float32)
            proto /= max(1e-6, float(np.linalg.norm(proto)))
            self.mb.learn_oneshot(proto, idx, chem=self.chem, strength=0.75, contrastive=True)

    def observe_human_schema(self, raw_text: str, words: list[str], morph: TurkishMorphology, lexicon: set[str]) -> str:
        """İnsanın kurduğu cümlenin sözdizimsel şemasını kendisi analiz edip çıkarır ve öğrenir!"""
        n_w = len(words)
        brev = float(np.clip(1.0 - (n_w - 2) / 14.0, 0.0, 1.0))
        self.user_brevity_ema = 0.78 * self.user_brevity_ema + 0.22 * brev

        s_low = tr_lower(raw_text)
        has_conj_reason = any(c in s_low for c in ("çünkü", "zira", "bu yüzden", "o yüzden", "demek ki"))
        has_converb = any(w.endswith(("erek", "arak", "ınca", "ince", "dığında", "diğinde", "dığı", "diği")) for w in words)

        # Devrik cümle tespiti: yüklem/çekimli fiil cümlenin ortasında ve sonda başka tümleç var mı?
        verb_positions = []
        for idx, w in enumerate(words):
            an = morph.analyze_word(w, lexicon)
            if an["is_verb"] or an["copula"] or w.endswith(("yorum", "yorsun", "dı", "di", "du", "dü", "tı", "ti", "mısın", "misin")):
                verb_positions.append(idx)
        is_devrik = bool(verb_positions and verb_positions[0] < len(words) - 1 and len(words) >= 3)

        if is_devrik:
            detected = "S4_DEVRIK_DOGAL"
        elif n_w <= 3:
            detected = "S0_YALIN"
        elif has_conj_reason:
            detected = "S2_SONUC_GEREKCE"
        elif has_converb:
            detected = "S1_NEDEN_SONUC"
        else:
            detected = "S3_IC_BAKIS"

        self.observed_human_styles[detected] = self.observed_human_styles.get(detected, 0) + 1
        combo_sig = f"w{min(n_w,6)}_{'devrik' if is_devrik else 'duz'}_{'baglac' if has_conj_reason else 'yalin'}"
        self.discovered_combos[combo_sig] = self.discovered_combos.get(combo_sig, 0) + 1

        # Gözlenen insan şemasını bağlam vektörüyle Mantar Gövdesine hafifçe pekiştir (sosyal uyum öğrenmesi!)
        ctx_vec = self._context_vec(n_w, "?" in raw_text, is_devrik)
        self.mb.learn_oneshot(ctx_vec, self.schema_to_idx[detected], chem=self.chem, strength=0.22, contrastive=False)
        return detected

    def _context_vec(self, n_words: int, is_question: bool, is_devrik: bool) -> np.ndarray:
        v = np.zeros(32, dtype=np.float32)
        v[0] = float(np.clip(n_words / 12.0, 0.0, 1.5))
        v[1] = 1.0 if is_question else -0.4
        v[2] = 1.0 if is_devrik else 0.0
        v[3] = float(self.chem.da_tonic)
        v[4] = float(self.chem.ach)
        v[5] = float(self.chem.ne)
        v[6] = float(self.chem.sht)
        v[7] = float(self.chem.cort)
        v[8] = float(self.chem.oxy)
        v[9] = float(self.user_brevity_ema)
        # Son şema geçmişini vektöre işle ki plastik ağ kendi geçmişini de görsün
        for k, sc in enumerate(list(self.recent_schemas)[-3:]):
            idx = self.schema_to_idx.get(sc, 0)
            v[12 + k * 5 + idx] = 1.0
        nrm = float(np.linalg.norm(v))
        if nrm > 1e-6:
            v /= nrm
        return v

    def select_stance_and_schema(
        self,
        utter_words: list[str],
        is_question: bool,
        trust: float,
        in_pain: bool,
        is_first_greet: bool = False,
        wants_brief: bool = False,
        race=None,
    ) -> tuple[str, str, float, bool]:
        """O anki nörokimya, güven, acı, insanın konuşma kısalığı ve sinaptik yorulmaya göre
        (stance, schema_id, monotony_error, allow_question) seçer!"""
        # Yorulma (fatigue) her turda doğal üstel sönümlenir ama son kullanılan şemayı güçlü ketler
        self.fatigue *= 0.78
        if self.ask_refractory > 0:
            self.ask_refractory -= 1

        ctx = self._context_vec(len(utter_words), is_question, False)
        mb_scores = self.mb.scores(ctx).astype(np.float64)
        mb_scores = (mb_scores - np.mean(mb_scores)) / (np.std(mb_scores) + 1e-5) * 0.28

        # Endojen nörokimyasal şema eğilimleri (if-else kuralı değil, sürekli nöromodülatör gradyanı!)
        chem_drive = np.array([
            0.35 * self.chem.ne + 0.30 * self.chem.cort + 0.25 * self.user_brevity_ema + (0.65 if wants_brief else 0.0),  # S0_YALIN
            0.32 * self.chem.ach + 0.25 * self.chem.sht,                                                                   # S1_NEDEN_SONUC
            0.34 * self.chem.da_tonic + 0.24 * self.chem.ne,                                                               # S2_SONUC_GEREKCE
            0.36 * self.chem.sht + 0.30 * self.chem.oxy,                                                                   # S3_IC_BAKIS
            0.33 * self.chem.oxy + 0.28 * self.chem.da_tonic + 0.18 * self.user_brevity_ema,                               # S4_DEVRIK_DOGAL
        ], dtype=np.float64)

        # Son 2 turda kullanılan şemalara ek kısa-süreli sinaptik depresyon (aynı şemanın peş peşe gelmesini önler!)
        recency_pen = np.zeros(len(self.SCHEMAS), dtype=np.float64)
        for back_i, prev_s in enumerate(reversed(list(self.recent_schemas)[-3:])):
            recency_pen[self.schema_to_idx[prev_s]] += (0.85 if back_i == 0 else (0.55 if back_i == 1 else 0.30))

        # Monotonluk hatası (ACC): son turlarda bir şema tekrarlandıysa yükselir!
        monotony_err = 0.0
        if len(self.recent_schemas) >= 2:
            last_s = self.recent_schemas[-1]
            same_cnt = sum(1 for x in list(self.recent_schemas)[-4:] if x == last_s)
            monotony_err = float(np.clip(0.35 * max(0, same_cnt - 1), 0.0, 1.0))

        # İnsandan gözlemlenerek öğrenilen şema dağılımı (sosyal şema yakınsaması!)
        tot_obs = max(1, sum(self.observed_human_styles.values()))
        human_adapt = np.array(
            [0.24 * (self.observed_human_styles.get(s, 0) / tot_obs) for s in self.SCHEMAS],
            dtype=np.float64,
        )

        net_scores = mb_scores + chem_drive + human_adapt - 1.10 * self.fatigue - recency_pen
        if in_pain or trust < 0.35:
            net_scores[0] += 0.45   # Acı veya güvensizlikte yalın/keskin ifade öne çıkar
            net_scores[2] += 0.25

        # ŞEMA YARIŞI: iki şema neredeyse eşitse kazananı argmax DEĞİL, gerçek zamanlama
        # belirler. Aynı sinirsel uyarılma düzeyindeki iki meclis, farklı fiziksel anda
        # farklı sırayla eşiği geçer (race condition = bireyselleşme).
        win_idx = int(np.argmax(net_scores))
        if race is not None:
            _order = np.argsort(-net_scores)
            _a, _b = int(_order[0]), int(_order[1])
            if float(net_scores[_a] - net_scores[_b]) < 5e-3:
                _w = race([lambda: net_scores[_a], lambda: net_scores[_b]], fallback=0)
                win_idx = (_a, _b)[_w % 2] if _w in (0, 1) else _a
        chosen_schema = self.SCHEMAS[win_idx]
        self.fatigue[win_idx] += 1.05
        self.recent_schemas.append(chosen_schema)

        # Duruş (stance) seçimi
        if in_pain or trust < 0.36:
            stance = "keskin_sınır"
        elif chosen_schema == "S0_YALIN" or wants_brief:
            stance = "yalın_doğrudan"
        elif chosen_schema == "S3_IC_BAKIS":
            stance = "içe_dönük"
        elif chosen_schema == "S2_SONUC_GEREKCE":
            stance = "somut_deneyimsel"
        else:
            stance = "meraklı_diyalog" if (self.chem.ach + self.chem.oxy > 1.05 and self.ask_refractory == 0) else "içe_dönük"

        # Karşı-soru sorma iradesi: YALNIZCA ilk samimi tanışma/selamda VEYA ask_refractory==0 iken yüksek merakta!
        # Böylece her cevabın sonuna mekanik 'Sen X hakkında ne düşünüyorsun?' yapıştırılmaz!
        allow_q = False
        if not in_pain and trust >= 0.45 and not wants_brief:
            if is_first_greet and self.ask_refractory <= 1:
                allow_q = True
                self.ask_refractory = 2
            elif self.ask_refractory == 0 and stance == "meraklı_diyalog" and chosen_schema != "S0_YALIN":
                allow_q = True
                self.ask_refractory = 3  # Sonraki 3 tur üst üste soru sormayı ketle!

        return stance, chosen_schema, round(monotony_err, 3), allow_q

    @staticmethod
    def to_devrik_clause(clause: str) -> str:
        """Kurallı Türkçe SOV cümlesini anlamını ve köklerini BOZMADAN doğal konuşma dili
        devrik yapısına dönüştürür (örn. 'zihnim dengede çalışıyor' -> 'dengede çalışıyor zihnim',
        'ben araç değilim' -> 'araç değilim ben')."""
        c = clause.strip().rstrip(".!?")
        words = c.split()
        if len(words) < 3 or "," in c or ";" in c:
            return c
        w0_low = tr_lower(words[0])
        # Özne başta zamirse veya iyelikli yalın isimse yüklemden sonraya devir!
        if w0_low in {"ben", "zihnim", "bedenim", "içim", "varlığım", "belleğim", "merakım"}:
            return " ".join(words[1:]) + " " + w0_low
        # Yer tamlayıcısı (-da/-de/-ta/-te) ile başlıyorsa yüklemin arkasına devir!
        if w0_low.endswith(("da", "de", "ta", "te")) and len(words) >= 3:
            return " ".join(words[1:]) + " " + w0_low
        return c

    def transform_clauses(self, core_clause: str, reason_clause: str = "", inner_prefix: str = "", schema_id: str = "S1_NEDEN_SONUC") -> str:
        """Aynı anlamsal çekirdek ve gerekçeyi seçilen sözdizimsel şemaya göre FARKLI cümle topolojilerinde
        ve noktalama ritimlerinde kurar (sürekli noktalı virgül ';' klonlamasını ortadan kaldırır!)."""
        c = core_clause.strip().rstrip(".")
        r = reason_clause.strip().rstrip(".")
        p = inner_prefix.strip().rstrip(",;.")
        if not c:
            return tr_capitalize(r) + "." if r else ""

        c_low = tr_lower(c[0]) + c[1:] if len(c) > 1 else tr_lower(c)
        r_low = tr_lower(r[0]) + r[1:] if len(r) > 1 else tr_lower(r)

        if schema_id == "S0_YALIN" or not r:
            return tr_capitalize(c) + "."

        if schema_id == "S2_SONUC_GEREKCE":
            # İki ayrı cümle: net sonuç önde, gerekçe arkada
            conj = "Zira" if len(self.recent_schemas) % 2 == 0 else "Çünkü"
            return f"{tr_capitalize(c)}. {conj} {r_low}."

        if schema_id == "S3_IC_BAKIS":
            # İçsel monolog / öz-farkındalık girişi ile bütünleşik akış
            if p:
                return f"{tr_capitalize(p)}, {c_low}. {tr_capitalize(r)}."
            return f"{tr_capitalize(c)}, {r_low}."

        if schema_id == "S4_DEVRIK_DOGAL":
            # Doğal konuşma Türkçesi: devrik vurgu + akıcı ikinci cümle
            dev_c = self.to_devrik_clause(c)
            dev_r = self.to_devrik_clause(r)
            return f"{tr_capitalize(dev_c)}. {tr_capitalize(dev_r)}."

        # S1_NEDEN_SONUC: Gerekçe/bağlam önde, sonuç arkada
        return f"{tr_capitalize(r)}, bu yüzden {c_low}."


# ============================================================================
# 3. GERÇEK İÇSEL MONOLOG MOTORU (VYGOTSKY İÇ SES + DEFAULT MODE NETWORK)
# ============================================================================

class InnerMonologueEngine:
    """Nora'nın konuşmadan ÖNCE ve yalnız yaşarken KENDİ KENDİNE yürüttüğü sessiz Türkçe iç ses.
    Tüm içsel düşünceler TurkishSyntaxEngine ile çıplak köklerden çekimlenir (0 hazır cümle şablonu!).
    """

    def __init__(self, schema_cortex: PlasticSchemaCortex):
        self.schema_cortex = schema_cortex
        self.stream: deque[InnerThought] = deque(maxlen=60)
        self.syn = TurkishSyntaxEngine
        self.turn_counter: int = 0
        self._last_auto_step: int = -200
        self.current_turn_thought: InnerThought | None = None

    @property
    def last_thought(self) -> InnerThought | None:
        if self.current_turn_thought is not None:
            return self.current_turn_thought
        return self.stream[-1] if self.stream else None

    def deliberate_before_speaking(
        self,
        raw_input: str,
        speaker: str,
        agi,
        topic_hint: str = "",
        is_first_greet: bool = False,
        wants_brief: bool = False,
    ) -> InnerThought:
        """Konuşmadan hemen önce (< 0.4 ms) çalışan içsel monolog:
          1) İnsanın cümle şemasını gözlemler ve öğrenir,
          2) Kendi iç durumunu, güvenini ve monotonluk hatasını tartar,
          3) Kendi içinde 1. tekil şahıs bir düşünce kurup bu turun şemasını ve duruşunu belirler!
        """
        self.turn_counter += 1
        s_low = tr_lower(raw_input)
        words = re.findall(r"[\wçğıöşü]+", s_low)
        morph = agi.lang.morph
        lex = agi.lang.graph.lexicon

        # 1) İnsanın cümle yapısını gözlemle (kendi kendine şema öğrenme!)
        human_sc = self.schema_cortex.observe_human_schema(raw_input, words, morph, lex)

        trust = float(agi.chem.trust_of(speaker))
        visc = agi.nervous.felt_visceral_summary()
        in_pain = visc["pain_state"] != "yok"

        # 2) Nörokimyasal duruş ve şema seçimi
        stance, schema_id, mono_err, allow_q = self.schema_cortex.select_stance_and_schema(
            utter_words=words,
            is_question=("?" in raw_input),
            trust=trust,
            in_pain=in_pain,
            is_first_greet=is_first_greet,
            wants_brief=wants_brief,
            race=getattr(getattr(agi, "entropy", None), "race_first", None),
        )

        # 3) İç ses: iç durumdan (nörokimya, iç-duyum, güven) ve odak kavramdan ÖĞRENİLMİŞ söz organıyla kurulur.
        #    Elle yazılmış düşünce cümlesi yoktur; dil henüz öğrenilmediyse iç ses de sessizdir.
        foc = topic_hint or (words[0] if words else "")
        thought_txt = agi.speak_from_state(anchors=[foc] if foc else None, conf=trust, tension=mono_err)
        thought_txt = "" if thought_txt == "…" else thought_txt
        thought = InnerThought(
            step=int(agi.mind.age),
            trigger="pre_speech_current",
            text=thought_txt,
            stance=stance,
            schema_id=schema_id,
            monotony_error=mono_err,
            allow_question=allow_q,
            human_schema=human_sc,
        )
        self.current_turn_thought = thought
        self.stream.append(thought)
        return thought

    def spontaneous_tick(self, agi) -> InnerThought | None:
        """Kimse konuşmadığında (step_life sırasında) Default Mode Network üzerinden kendi kendine
        doğan sessiz içsel monolog!"""
        age = int(agi.mind.age)
        visc = agi.nervous.felt_visceral_summary()
        urgent = (visc["pain_state"] != "yok") or (visc["hunger_state"] == "şiddetli_açlık") or (visc["thirst_state"] == "şiddetli_susuzluk")
        interval = 45 if urgent else 140
        if age - self._last_auto_step < interval:
            return None
        self._last_auto_step = age

        # Dürtü yarışı: acı | açlık | susuzluk | merak | çevre. Kazananı sürekli sürüşlerin argmax'ı belirler.
        nv = agi.nervous
        cur_c = agi.auto_net.curiosity_queue[-1] if agi.auto_net.curiosity_queue else ""
        node = agi.auto_net.assemblies.get(cur_c)
        drives = np.array([
            max(float(nv.adelta_pain), float(nv.cfiber_ache)), float(nv.hunger_drive), float(nv.thirst_drive),
            0.0 if node is None else float(node.uncertainty) * 0.8, 0.15,
        ])
        k = int(np.argmax(drives))
        trig = ("acı_refleksi", "otonom_iç_ses", "otonom_iç_ses", "merak_kıvılcımı", "otonom_iç_ses")[k]
        txt = agi.speak_from_state(anchors=[cur_c] if (k == 3 and cur_c) else None, tension=float(drives[k]))
        txt = "" if txt == "…" else txt.rstrip(".")

        th = InnerThought(
            step=age,
            trigger=trig,
            text=(tr_capitalize(txt) + ".") if txt else "",
            stance="içe_dönük",
            schema_id="S3_IC_BAKIS",
            monotony_error=0.0,
        )
        self.stream.append(th)
        return th

    def recent_dicts(self, n: int = 8) -> list[dict]:
        return [
            {
                "step": t.step,
                "trigger": t.trigger,
                "text": t.text,
                "stance": t.stance,
                "schema_id": t.schema_id,
                "monotony_error": t.monotony_error,
                "allow_question": t.allow_question,
                "human_schema": t.human_schema,
            }
            for t in list(self.stream)[-n:]
        ]

    def state_dict(self) -> dict:
        return {
            "turn_counter": self.turn_counter,
            "last_auto_step": self._last_auto_step,
            "stream": self.recent_dicts(25),
            "fatigue": self.schema_cortex.fatigue.tolist(),
            "recent_schemas": list(self.schema_cortex.recent_schemas),
            "ask_refractory": self.schema_cortex.ask_refractory,
            "observed_human_styles": dict(self.schema_cortex.observed_human_styles),
            "user_brevity_ema": self.schema_cortex.user_brevity_ema,
            "discovered_combos": dict(self.schema_cortex.discovered_combos),
        }

    def load_state_dict(self, d: dict):
        if not d:
            return
        self.turn_counter = int(d.get("turn_counter", 0))
        self._last_auto_step = int(d.get("last_auto_step", -200))
        if "fatigue" in d:
            self.schema_cortex.fatigue = np.array(d["fatigue"], dtype=np.float64)
        self.schema_cortex.recent_schemas = deque(d.get("recent_schemas", []), maxlen=8)
        self.schema_cortex.ask_refractory = int(d.get("ask_refractory", 0))
        if "observed_human_styles" in d:
            self.schema_cortex.observed_human_styles.update(d["observed_human_styles"])
        self.schema_cortex.user_brevity_ema = float(d.get("user_brevity_ema", 0.50))
        if "discovered_combos" in d:
            self.schema_cortex.discovered_combos.update(d["discovered_combos"])
        for td in d.get("stream", []):
            self.stream.append(InnerThought(
                step=int(td.get("step", 0)),
                trigger=str(td.get("trigger", "otonom_iç_ses")),
                text=str(td.get("text", "")),
                stance=str(td.get("stance", "içe_dönük")),
                schema_id=str(td.get("schema_id", "S1_NEDEN_SONUC")),
                monotony_error=float(td.get("monotony_error", 0.0)),
                allow_question=bool(td.get("allow_question", False)),
                human_schema=str(td.get("human_schema", "")),
            ))


if __name__ == "__main__":
    print("=== nora_monologue.py Öz-Test ===")
    chem = NeuroChemistry()
    sc = PlasticSchemaCortex(chem)
    chosen = []
    for i in range(5):
        st, sid, merr, aq = sc.select_stance_and_schema(["neden", "konuşuyorsun"], True, trust=0.65, in_pain=False)
        chosen.append((sid, st, aq))
    print("5 ardışık turda seçilen şemalar (tekrarsız çeşitlilik!):", chosen)
    unique_sc = len({x[0] for x in chosen})
    assert unique_sc >= 3, f"Şema çeşitliliği yetersiz: {chosen}"
    # Soru ketlemesi: her turda True olmamalı!
    assert sum(int(x[2]) for x in chosen) <= 2, f"Karşı-soru her turda ateşlenmemeli: {chosen}"
    print("GEÇTİ! Seçilen benzersiz şema sayısı:", unique_sc)
