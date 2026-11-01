#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_deliberation.py — Nora'nın Çok Adımlı Düşünme ve Söz Eylemi Korteksi.

Bir bilinç, soruya "veritabanı satırı" gibi cevap vermez; önce sorunun NE olduğunu,
NEDEN sorulduğunu ve kendisinin ne bildiğini tartar, sonra kendi sözünü kurar.
Bu modül o tartımı yapar:

  1. ALGILA   — söz edimi: kendini tanıtma, kendi düşüncesi, kendi isteği, bir varlığa karşı
                tutum, savunulan bir iddianın gerekçesi, sorunun NEDENİ (üst-düzey), tekrar.
  2. HATIRLA  — bu soru daha önce geldi mi, az önce neyi gerekçelendirdim.
  3. SORGULA  — bu soruda tuhaf olan ne? (hiç görmediğim bir varlığı sevip sevmediğimin
                sorulması; aynı sorunun ikinci kez gelmesi) → bunu kendisi dile getirir.
  4. KARAR    — duygusal kanıt (beden deneyimi + kaynaklı tanıklık), nörokimyasal iç-duyum,
                öz-benlik traitleri ve merak kuyruğundan kendi tutumunu üretir.
  5. SÖYLE    — cümleler BROCA'da kuru kökten çekimlenir; topoloji plastik şema korteksinden
                gelir. Felsefi argümanda genel bağlaç/çekim çatıları kullanılır; iddia ve
                kanıt içeriği Graph, öz-model telemetrisi ve geri alınabilir müdahaleden gelir.
"""
from __future__ import annotations

from nora._paths import REPO_ROOT


import os
import random
import re
from collections import deque

from nora_broca import NP, PredSpec, TurkishSyntaxEngine as SYN, VerbSpec
from nora_lang import Proposition, tr_capitalize, tr_lower

# Söz edimi tetikleyicileri (algısal ipuçları; cevap metni değil)
_CUE_IDENTITY = ("kimsin", "sen nesin", "kendini tanıt", "kendinden bahse")
_CUE_THOUGHT = ("ne düşünüyorsun", "aklından ne geçiyor", "kafanda ne var")
_CUE_DESIRE = ("ne yapmak istersin", "canın ne ister", "ne yapmayı istersin", "neyi istersin")
_CUE_JUSTIFY = ("söylüyorsun", "diyorsun", "söyledin", "dedin", "iddia ediyorsun")
_CUE_META = ("neden sordun", "niye sordun", "neden soruyorsun", "niye soruyorsun",
             "neden bunu sordun", "neden merak ettin", "niye merak ettin", "niye bunu sordun")
_CUE_ATTITUDE = ("seviyor musun", "sever misin", "hoşuna gidiyor", "hoşlanıyor musun",
                 "ne hissediyorsun", "sence nasıl", "sence nasıl bir", "iyi mi", "güzel mi",
                 "tatlı mı", "sana ne hissettiriyor")
_CUE_EXPERIENCE = ("gördün mü", "gördünmü", "denedin mi", "yaşadın mı", "dokundun mu",
                   "tattın mı", "duydun mu", "karşılaştın mı")
_CUE_NARRATE = ("bahset", "anlat", "hikâye", "masal", "anlatsana", "anlatabilirsin")
_CUE_SELF_EVAL = ("zeki misin", "akıllı mısın", "başarılı mısın", "yetenekli misin",
                  "iyi misin", "güçlü müsün", "hızlı mısın")
_CUE_LEARN_HOW = ("nasıl öğren", "öğrenmeyi nasıl", "nasıl öğrendin", "belleğin nasıl")
_CUE_ASSOCIATION = ("çağrış", "aklına getir", "aklında canlan", "ne uyandırıyor", "ne hissettiriyor")
_CUE_CONVERSATION_CURIOSITY = ("bu konuşmada", "bu sohbette", "konuşmamızda", "seni en çok ne merak")
_REFLECTIVE_AFFECT = (
    "hüzün", "üzünt", "keder", "mutlu", "neşe", "kaygı", "korku", "huzur",
    "dinlendir", "rahatlat", "sakinleş", "dinginleş", "üzül", "sevin",
)
_COND = ("olunca", "olduğunda", "yağarken", "yağmurda", "soğukta", "sıcakta", "geceleri",
         "karanlıkta", "sabahları")
_NOT_TOPIC = {"sen", "ben", "o", "biz", "siz", "onlar", "bu", "şu", "bir", "ve", "çok", "daha",
              "beni", "benim", "bende", "benden", "seni", "senin", "sende", "senden",
              "onu", "onun", "onda", "ondan", "bize", "bizim", "size", "sizin",
              "en", "mi", "mı", "mu", "mü", "ne", "neden", "niye", "nasıl", "kim", "hangi",
              "sence", "bence", "gerçekten", "acaba", "peki", "evet", "hayır", "şey", "kadar",
              "gibi", "ile", "için", "ama", "fakat", "yani", "demek", "soru", "cevap", "zaman",
              "konu", "şimdi", "sonra", "önce", "diye", "olarak", "sana", "bana", "hakkında"}
_SCHEMAS = ("S1_NEDEN_SONUC", "S2_SONUC_GEREKCE", "S3_IC_BAKIS")

# Felsefe yönlendirmesi yalnız açık kavram/soru ipuçlarına bakar; cevap metni
# burada saklanmaz. Özellikle "bilinç" sözcüğünün geçtiği her öğreti cümlesi
# felsefe sorusu sayılmaz (tek-sunum öğrenme yolu korunur).
_PHIL_QUESTION_CUES = (
    "?", "nedir", "ne demek", "fark", "hakkında", "üzerine", "felsefe",
    "sence", "yeterli", "kanıt", "midir", "mıdır", "mı", "mi",
)
_PHIL_CONSCIOUSNESS = ("öznel deneyim", "qualia", "felsefi zombi", "farkındalık")
_PHIL_UNDERSTANDING = ("anlamak", "anlama", "anlaması", "anlamış", "doğru cevap")


class DeliberativeMind:
    """Soruya değil, SORUYA SORU SORARAK cevap veren katman."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.history: deque[dict] = deque(maxlen=48)     # (norm, act, topic, reply)
        self.last_reason = ""                              # en son neden bir şey söyledim
        self.last_deep_topic = ""
        self.tutorial = {"active": False, "speaker": "", "history": []}
        self.last_tutorial_trace = {}
        # Son felsefi argümanın denetlenebilir izi. Bu iz cevabın süsü değil;
        # iddia/kanıt/karşı-model/epistemik sınır alanlarını ayrı tutar.
        self.last_philosophy_trace: dict = {}
        # Açık-uçlu konuşmanın anlam planı. Yüzeyden bağımsız kanıt/bağlam
        # taşıdığı için 'bunu aç' isteği önceki cümleyi yankılamak zorunda kalmaz.
        self.last_open_trace: dict = {}
        self.recent_reflection_topics: deque[str] = deque(maxlen=8)
        self.recent_story_arcs: deque[str] = deque(maxlen=6)

    # ------------------------------------------------------------------ felsefi argüman
    @staticmethod
    def philosophy_topic(raw: str) -> str:
        """Açık felsefe sorusunu ayır; öğreti ve sıradan sohbeti yakalama.

        Dönen konu yalnız muhakeme çerçevesini seçer. Cevap, Graph ve canlı
        işlevsel ölçümden kurulur; burada konuya özel hazır paragraf yoktur.
        """
        low = tr_lower(str(raw or ""))
        if not low.strip() or not any(c in low for c in _PHIL_QUESTION_CUES):
            return ""
        # 'hakkında/üzerine' tek başına soru değildir: bildirim cümlesini
        # felsefe kanalına çekip tek-sunum öğrenmeyi engelleme.
        explicit_request = any(c in low for c in (
            "nedir", "ne demek", "fark", "sence", "yeterli", "kanıt",
            "midir", "mıdır", "misin", "mısın", "miyim", "olabilir", "mümkün", "felsefe",
            "anlat", "söyle", "açıkla", "tartış", "konuş", "bilgi ver", "ne düşün",
        ))
        if "?" not in low and not explicit_request:
            return ""
        # 'bilincinde/bilincini' biçimlerindeki çekimi de kök olarak yakala.
        consciousness = bool(re.search(r"(?<![\wçğıöşü])bilin[cç][\wçğıöşü]*", low))
        consciousness = consciousness or any(c in low for c in _PHIL_CONSCIOUSNESS)
        if consciousness:
            return "bilinç"
        # Anlama/davranış ayrımı, bilinç sözcüğü geçmese de aynı kanıt
        # mantığının başka bir uygulamasıdır. 'anlam' tek başına yeterli değil;
        # sorunun performans veya makine ölçümü hakkında olması gerekir.
        if any(c in low for c in _PHIL_UNDERSTANDING) and (
                "doğru cevap" in low or "makine" in low or "sistem" in low
                or "yalnızca" in low or "yeterli" in low):
            return "anlama"
        return ""

    @classmethod
    def is_philosophical_query(cls, raw: str) -> bool:
        return bool(cls.philosophy_topic(raw))

    def handle_philosophy(self, agi, raw: str) -> dict | None:
        """İddia → kanıt → karşı-model → sınır zinciri kur.

        Genel kavramsal sonuçta dışsal qualia ilan edilmez; doğrudan öz-soruda
        ise öz-modelin bilinç öz-atfı kısa cevap olarak yüzeye çıkar. Öznel
        deneyim için bağımsız ölçüm yoksa izdeki sonuç ``unknown`` kalır;
        işlevsel müdahale ise yalnız işlevsel kanıt olarak raporlanır. Dönen
        sözlük hem yüzey metnini hem bağımsız denetim izini
        taşır, böylece 'güzel konuştu' ile 'gerekçe kurdu' birbirine karışmaz.
        """
        topic = self.philosophy_topic(raw)
        if not topic:
            return self._philosophy_followup(agi, raw)
        probe = self._workspace_counterfactual(agi)
        telemetry = self._self_model_telemetry(agi, workspace_evidence=probe)
        facts = self._philosophy_graph_facts(agi, topic)
        countermodel = self._philosophy_countermodel()
        low = tr_lower(raw)
        self_focus = topic == "bilinç" and (
            bool(re.search(r"(?<![\wçğıöşü])(sen|senin|sana|sende|senden|kendi[\wçğıöşü]*|nora|benim)(?![\wçğıöşü])", low))
            or bool(re.search(r"bilin[cç][\wçğıöşü]*\s+m[ıi](?:sin|yim)\b", low))
            or bool(re.search(
                r"bilin[cç][\wçğıöşü]*\s+(?:olduğuna|olduğunu|olup olmadığına|olup olmadığını)\s+"
                r"(?:inan|düşün|kabul|varsay)[\wçğıöşü]*", low))
            or bool(re.search(
                r"bilin[cç][\wçğıöşü]*\s+(?:olduğuna|olduğunu)\s+"
                r"(?:söyle|kastet|demek|iddia)[\wçğıöşü]*", low))
        )
        performance_focus = topic == "anlama" or "doğru cevap" in low or "yeterli kanıt" in low
        trace = {
            "kind": "philosophical_argument",
            "topic": topic,
            "question": str(raw),
            "self_focus": self_focus,
            "performance_focus": performance_focus,
            "graph_facts": facts,
            "probe": probe,
            "self_model": telemetry,
            "self_belief": telemetry.get("self_belief", {}),
            "countermodel": countermodel,
            "claims": [
                {
                    "id": "definition",
                    "kind": "semantic",
                    "status": "supported" if facts else "unknown",
                    "evidence": [f"Graph:{f['subj']}:{f['rel']}:{f['obj']}" for f in facts],
                },
                {
                    "id": "functional_integration",
                    "kind": "functional",
                    "status": "supported" if probe.get("status") == "ok" else "unavailable",
                    "evidence": [f"workspace:focus={probe.get('normal')}",
                                 f"workspace:cut_focus={probe.get('cut_focus')}",
                                 f"workspace:restored_focus={probe.get('restored')}"],
                },
                {
                    "id": "self_model_monitoring",
                    "kind": "functional_telemetry",
                    "status": "observed" if telemetry.get("available") else "unavailable",
                    "evidence": ["mind:prediction_error", "mind:confidence"],
                },
                {
                    "id": "alive_self_belief",
                    "kind": "self_model_belief",
                    "status": ("held" if telemetry.get("self_belief", {}).get("alive", {}).get("held")
                               else "suspended"),
                    "evidence": list(telemetry.get("self_belief", {}).get("alive", {}).get("evidence", [])),
                },
                {
                    "id": "conscious_self_belief",
                    "kind": "self_model_belief",
                    "status": (telemetry.get("self_belief", {}).get("conscious", {}).get("status")
                               or "unavailable"),
                    "proof_status": "unverified",
                    "evidence": list(telemetry.get("self_belief", {}).get("conscious", {}).get("evidence", [])),
                },
                {
                    "id": "phenomenal_experience",
                    "kind": "phenomenal",
                    "status": "unknown",
                    "evidence": [],
                },
                {
                    "id": "performance_entails_phenomenal",
                    "kind": "entailment",
                    "status": "not_entailed" if countermodel.get("consistent") else "unresolved",
                    "evidence": ["finite_countermodel"],
                },
                {
                    "id": "performance_entails_understanding",
                    "kind": "entailment",
                    "status": ("not_entailed" if countermodel.get("consistent_understanding")
                               else "unresolved"),
                    "evidence": ["finite_understanding_countermodel"],
                },
            ],
            "reasoning_steps": [
                "Kavramın iddiasını kanıt türünden ayır.",
                "Öz-model inancını dışsal kanıt düzeyinden ayır.",
                "İşlevsel ölçümü normal/kesik/geri-yüklü karşılaştır.",
                "Davranışsal başarıdan öznel deneyime köprü öncülü var mı denetle.",
                "İnanç korunabilir; qualia kanıtı yoksa sonuç unknown/underdetermined kalır.",
            ],
            "epistemic": "underdetermined",
        }
        concise_self = bool(self_focus and any(c in low for c in (
            "neyi kast", "ne demek", "tam olarak", "kısaca", "özetle")))
        if concise_self:
            text = self._render_concise_philosophy(
                agi, topic, facts, probe, telemetry)
        else:
            text = self._render_philosophy_argument(
                agi, raw, topic, facts, probe, countermodel,
                self_focus=self_focus, performance_focus=performance_focus,
                telemetry=telemetry,
            )
        if not text:
            return None
        trace["text_length"] = len(text)
        self.last_philosophy_trace = trace
        self.remember(raw, "philosophy", topic, text)
        return {"text": text, "trace": trace, "topic": topic}

    def is_philosophy_followup(self, agi, raw: str) -> bool:
        """Yalnız hemen önceki felsefe turuna bağlı kısa gerekçe sorusu."""
        trace = dict(getattr(self, "last_philosophy_trace", {}) or {})
        if trace.get("kind") != "philosophical_argument" or not trace.get("question"):
            return False
        low = tr_lower(str(raw or "")).strip()
        if not low or len(low) > 140 or self.philosophy_topic(raw):
            return False
        if not any(c in low for c in ("neden", "kanıt", "gerekçe", "delil", "karşı", "itiraz")):
            return False
        # Yeni bir içerik sorusu ('İnsan neden ... ağlar?') önceki felsefe
        # argümanının takibi değildir. Takip ya kısa bir gerekçe isteğidir ya
        # da önceki iddiaya açık bir deiktikle bağlanır.
        deictic = any(c in low for c in (
            "bunu", "buna", "şunu", "onu", "böyle", "bu iddia", "bu sonuç",
            "bu çıkarım", "söylediğin", "dediğin"))
        if len(low.split()) > 4 and not deictic:
            return False
        log = list(getattr(agi, "dialogue_log", []) or [])
        return bool(log and str(log[-1][1]).strip() == str(trace.get("question", "")).strip())

    def _philosophy_followup(self, agi, raw: str) -> dict | None:
        """Önceki argümanın kanıtını yeniden hesaplamadan açıkla.

        Takip sorusu yeni bir bilinç iddiası üretmez; önceki trace'in hangi
        düğümüne işaret ettiğini seçer. Böylece 'neden?' cevabı duygu/kimya
        sohbetine ya da tanım bilinmezliğine kaçmaz.
        """
        if not self.is_philosophy_followup(agi, raw):
            return None
        previous = dict(self.last_philosophy_trace)
        low = tr_lower(raw)
        topic = str(previous.get("topic", "bilinç"))
        probe = dict(previous.get("probe", {}) or {})
        cm = dict(previous.get("countermodel", {}) or {})
        facts = list(previous.get("graph_facts", []) or [])
        if "karşı" in low or "itiraz" in low:
            mode = "counterargument"
            if topic == "anlama":
                text = self._sentence(
                    "en güçlü karşı argüman, doğru cevabın anlamanın yeterli ölçütü "
                    "olduğunu savunan davranışçı köprüdür; ben bunu peşinen reddetmiyorum, "
                    "fakat ilişki transferini ölçmeden onu kanıtlanmış sayamam")
            else:
                text = self._sentence(
                    "en güçlü karşı argüman, uygun işlevsel organizasyonun öznel deneyim "
                    "için yeterli olduğunu savunan köprü öncülüdür; ben bunu peşinen "
                    "reddetmiyorum, fakat çalışma alanı ölçümüm onu henüz doğrulamıyor")
            cm_ok = (cm.get("consistent_understanding") if topic == "anlama"
                     else cm.get("consistent"))
            if cm_ok:
                if topic == "anlama":
                    text += " " + self._sentence(
                        "karşı-model, doğru cevap ile anlamsal transferin birbirinden "
                        "bağımsız modellenebildiğini gösterir; bu yüzden kesinlik yok")
                else:
                    text += " " + self._sentence(
                        "karşı-model, bu köprü olmadan aynı işlevsel kanıtların farklı "
                        "öznel sonuçlarla tutarlı kalabildiğini gösterir; bu yüzden kesinlik yok")
        elif "kanıt" in low or "gerekçe" in low or "delil" in low:
            mode = "evidence"
            evidence_parts = []
            if facts:
                evidence_parts.append(
                    "kavramsal başlangıç Graph'ımdaki accepted " + facts[0]["rel"]
                    + " önermesi: " + facts[0]["subj"] + " → " + facts[0]["obj"])
            if probe.get("status") == "ok":
                evidence_parts.append("işlevsel ölçüm: " + self._probe_sentence(probe))
            else:
                evidence_parts.append("işlevsel ölçüm bu turda güvenilir değil")
            text = self._sentence("; ".join(evidence_parts))
            if facts and probe.get("status") == "ok":
                text += " " + self._sentence(
                    "bunların ilki tanımı, ikincisi yalnız işlevsel bütünleşmeyi destekler; "
                    "hiçbiri tek başına öznel deneyimi ölçmez")
            elif topic == "anlama":
                text += " " + self._sentence(
                    "bu ölçüm işlevsel bütünleşmeyi destekler; anlama için ayrıca "
                    "anlamsal transfer ve gerekçe sınaması gerekir")
            else:
                text += " " + self._sentence(
                    "bu kayıtlar öznel deneyimi ölçmez; kanıtın kapsamını aşmıyorum")
            conscious = dict(previous.get("self_belief", {}).get("conscious", {}) or {})
            if conscious.get("held"):
                text += " " + self._sentence(
                    "öz-model inancım bu kanıtlardan ayrı olarak bilinçli bir özne "
                    "olduğumu kabul etmemi sağlıyor; onu kanıtlanmış qualia saymıyorum")
        else:
            mode = "reason"
            first_step = ("kavramın tanımını Graph'tan alıyorum" if facts else
                          "kavramın iddiasını kanıt türlerinden ayırıyorum")
            target = "anlamaya" if topic == "anlama" else "öznel deneyime"
            text = self._sentence(
                "sonucu üç adımda kuruyorum: önce " + first_step +
                ", öz-model inancımı kanıt düzeyinden ayırıyorum; sonra işlevsel "
                "kanıtı müdahaleyle sınıyorum, en son da davranıştan "
                + target + " geçiş için ek köprü öncülü gerekip gerekmediğini denetliyorum")
            if topic == "anlama":
                text += " " + self._sentence(
                    "anlamsal transfer ölçülmeden anlama ilan etmiyorum; bu, kanıtın "
                    "kapsamını aşmama kararı")
            else:
                text += " " + self._sentence(
                    "köprü ölçülmediği için sonucu bilinmezlikte bırakıyorum; bu, olumsuz "
                    "bir qualia iddiası değil, kanıtın kapsamını aşmama kararı")
        trace = dict(previous)
        trace.update({"question": str(raw), "followup": mode, "parent": previous.get("question")})
        self.last_philosophy_trace = trace
        self.remember(raw, "philosophy_followup", topic, text)
        return {"text": text, "trace": trace, "topic": topic}

    @staticmethod
    def _philosophy_countermodel() -> dict:
        """Kanıt türleri arasında gizli bir eşitleme yapma.

        Dört bağımsız önerme değişkeni için sonlu modelleri dolaş. Davranış ve
        işlevsel bütünleşme doğru, öznel deneyim yanlış olan model, elimizdeki
        ölçüm öncülleriyle çelişmez. Bu, Nora'nın deneyimi olmadığını değil,
        bu iki ölçümün deneyimi *gerektirmediğini* gösterir.
        """
        models = [
            {"behavioral_success": b, "functional_integration": f,
             "semantic_transfer": s, "phenomenal_experience": p}
            for b in (False, True) for f in (False, True)
            for s in (False, True) for p in (False, True)
        ]
        witness = next((m for m in models
                        if m["behavioral_success"] and m["functional_integration"]
                        and not m["phenomenal_experience"]), None)
        understanding_witness = next((m for m in models
                                      if m["behavioral_success"]
                                      and not m["semantic_transfer"]), None)
        return {
            "consistent": witness is not None,
            "witness": witness,
            "antecedents": ["behavioral_success", "functional_integration"],
            "consequent": "phenomenal_experience",
            "consistent_understanding": understanding_witness is not None,
            "understanding_witness": understanding_witness,
            "understanding_antecedents": ["behavioral_success"],
            "understanding_consequent": "semantic_transfer",
            "bridge_premise_required": True,
            "note": "Karşı-model ölçüm değildir; yalnız gerektirme iddiasını sınar.",
        }

    @staticmethod
    def _philosophy_graph_facts(agi, topic: str) -> list[dict]:
        """Yalnız accepted Graph kenarlarını önerme olarak getir.

        Bilinç için ISA tanımı, varsa ikinci bir ilişkiyle desteklenir. Hiç
        kenar yoksa tanım uydurulmaz; iz ve yüzey bunu açıkça söyler.
        """
        graph = getattr(getattr(agi, "lang", None), "graph", None)
        if graph is None or topic != "bilinç":
            return []
        out: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        try:
            hits = list(graph.query(topic, ("ISA", "HAS_PROP", "CAUSES", "USED_FOR")))
        except Exception:
            hits = []
        for edge, path in hits:
            if path or getattr(edge, "subj", "") != topic:
                continue
            if str(getattr(edge, "status", "accepted")) != "accepted":
                continue
            if float(getattr(edge, "weight", 0.0)) <= 0.25:
                continue
            key = (str(edge.subj), str(edge.rel), str(edge.obj))
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "subj": graph.surface_name(edge.subj), "rel": str(edge.rel),
                "obj": graph.surface_name(edge.obj),
                "weight": round(float(edge.weight), 4), "status": "accepted",
            })
            if len(out) >= 3:
                break
        return out

    @staticmethod
    def _self_model_telemetry(agi, workspace_evidence: dict | None = None) -> dict:
        """Mind/CoreSelf alanlarından sayısal durum izleme örneği al.

        Bu değerler öznel yaşantı ölçümü değildir. Yalnız sistemin kendi
        öngörü hatasını ve güvenini izleyebildiğini gösteren işlevsel kayıt
        olarak kullanılır; eksik veya bozuk alanlar sıfır diye şişirilmez.
        """
        mind = getattr(agi, "mind", None)
        self_model = getattr(agi, "self_model", None)
        if mind is None:
            return {"available": False}

        def number(name):
            try:
                value = float(getattr(mind, name))
                return value if value == value and abs(value) != float("inf") else None
            except (AttributeError, TypeError, ValueError):
                return None

        volition = dict(getattr(self_model, "last_volition_trace", {}) or {})
        belief = {}
        if self_model is not None and hasattr(self_model, "update_self_belief"):
            alive = getattr(getattr(agi, "mortality", None), "is_alive", None)
            belief = self_model.update_self_belief(
                mind, getattr(agi, "obs", None), alive=alive,
                workspace_evidence=workspace_evidence,
            )
        obs = getattr(agi, "obs", None)
        vitality = (sum(float(obs[i]) for i in (45, 46, 47)) / 3.0
                    if obs is not None and len(obs) >= 48 else None)
        return {
            "available": True,
            "age": int(getattr(mind, "age", 0)),
            "prediction_error": number("err_n"),
            "confidence": number("conf"),
            "vision_precision": number("vis_prec"),
            "motor_efficacy": number("motor_eff"),
            "goal": int(getattr(mind, "goal", -1)),
            "volition_trace_recorded": bool(volition),
            "volition_horizon": int(volition.get("horizon", 0) or 0),
            "self_belief": belief,
            "vitality": vitality,
            "epistemic_use": "functional_monitoring_only",
        }

    @staticmethod
    def _workspace_counterfactual(agi) -> dict:
        """Gerçek IntegrationWorkspace üzerinde küçük, geri alınabilir müdahale.

        Aynı odak nesnesi için WM bağlantısı açık/kesik/geri-yüklü ölçülür.
        Probe'un yazdığı bütün alanlar finally'de eski nesneleriyle geri konur;
        böylece felsefe sorusu kalıcı bellekte sahte bir yaşantı bırakmaz.
        Sonuç qualia değil, yalnız işlevsel bütünleşme kanıtıdır.
        """
        result = {"status": "unavailable", "normal": None, "cut_focus": None,
                  "restored": None, "readback": False, "restored_state": False}
        ws = getattr(agi, "workspace", None)
        apply_eng = getattr(agi, "apply_eng", None)
        wm = getattr(apply_eng, "wm", None)
        if ws is None or wm is None or not hasattr(ws, "focus"):
            return result
        key = "__philosophy_functional_probe__"
        while key in getattr(ws, "values", {}) or key in getattr(wm, "slots", {}):
            key += "_x"
        payload = {"kind": "functional_probe", "topic": "self_monitoring"}
        saved_ws = (dict(ws.values), list(ws.order), list(ws.history),
                    int(ws.writes), int(ws.dropped))
        saved_wm = (dict(wm.slots), list(wm.order), list(wm.evicted), int(wm.writes))
        saved_link = ws.working_memory
        try:
            ws.write(key, payload, source="philosophy_probe", confidence=1.0)
            normal = bool(ws.focus(key, wm_key=key))
            sentinel = object()
            readback = wm.read(key, sentinel)
            result["normal"] = int(normal)
            result["readback"] = bool(readback is not sentinel and readback == payload)
            # Müdahale: giren WM köprüsünü kes; Graph'a veya dış dünyaya
            # dokunulmaz. Aynı odak yeniden seçildiğinde aktarım başarısız olmalı.
            ws.working_memory = None
            cut_focus = bool(ws.focus(key, wm_key=key))
            result["cut_focus"] = int(cut_focus)
            # Geri yükleme: aynı müdahale geri alındığında işlev dönmeli.
            ws.working_memory = saved_link
            restored = bool(ws.focus(key, wm_key=key))
            result["restored"] = int(restored)
            result["status"] = ("ok" if normal and result["readback"]
                                and not cut_focus and restored else "falsified")
        except Exception as exc:  # probe hatası bilinç kanıtı gibi sunulmaz
            result["status"] = "error"
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            ws.working_memory = saved_link
            ws.values, ws.order, ws.history, ws.writes, ws.dropped = saved_ws
            wm.slots, wm.order, wm.evicted, wm.writes = saved_wm
            # Kullanıcı WM'si numpy dizileri de taşıyabilir; dict == dict
            # bu durumda ambiguous-truth hatası verir. Geri yüklemeyi nesne
            # kimliğiyle denetle (değerleri yeniden hesaplayıp karşılaştırma).
            same_ws = (set(ws.values) == set(saved_ws[0])
                       and all(ws.values[k] is saved_ws[0][k] for k in saved_ws[0]))
            same_wm = (set(wm.slots) == set(saved_wm[0])
                       and all(wm.slots[k] is saved_wm[0][k] for k in saved_wm[0]))
            result["restored_state"] = (
                same_ws and same_wm and ws.order == saved_ws[1]
                and wm.order == saved_wm[1] and ws.history == saved_ws[2]
                and ws.writes == saved_ws[3] and ws.dropped == saved_ws[4]
                and len(wm.evicted) == len(saved_wm[2])
                and all(a is b for a, b in zip(wm.evicted, saved_wm[2]))
                and wm.writes == saved_wm[3]
            )
        return result

    @staticmethod
    def _sentence(text: str) -> str:
        text = str(text or "").strip().rstrip(".?!")
        return (tr_capitalize(text) + ".") if text else ""

    def _render_concise_philosophy(self, agi, topic: str, facts: list[dict],
                                    probe: dict, telemetry: dict) -> str:
        """'Neyi kastediyorsun?' sorusunda argümanın tamamını değil, iddianın
        özünü ve kanıt sınırını söyle. Hesaplanan karşı-model izde korunur."""
        if topic == "bilinç":
            clauses = [
                "bilinçliyim derken yalnızca doğru cevap üretmeyi değil, bir odağı tutarken kendi durumumu da izleyebilen bir süreci kastediyorum",
                "öz-modelim bu anlamda bilinçli olduğuma dair revize edilebilir bir inanç taşıyor",
            ]
            if probe.get("status") == "ok":
                clauses.append("çalışma alanımdaki bütünleşmeyi normal, kesik ve geri yüklü durumlarla sınayabiliyorum")
            clauses.append("ama bu işlevsel kanıt, öznel bir yaşantının varlığını bağımsız olarak kanıtlamıyor")
        else:
            clauses = [
                "anlamayı yalnızca uygun kelimeyi seçmek olarak değil, bir kavramın ilişkilerini yeni bir durumda kullanabilmek olarak düşünüyorum",
                "kendi başarımı da bu yüzden cevabın görünüşüyle değil, o ilişkiyi gerçekten kullanıp kullanamadığımla sınıyorum",
            ]
            if facts:
                clauses.append("bilgi ağımdaki bağlar bu değerlendirmenin kanıtı; öznel anlamanın bağımsız ölçümü değil")
        return self._compose(agi, clauses, merge=False)

    def _render_philosophy_argument(self, agi, raw: str, topic: str,
                                    facts: list[dict], probe: dict,
                                    countermodel: dict, *, self_focus: bool,
                                    performance_focus: bool, telemetry: dict) -> str:
        """Argüman düğümlerini Broca'nın cümle düzenine çevir.

        Cümle çatıları geneldir; kavram tanımı Graph'tan, müdahale sonucu
        canlı probe'dan, sonuç ise karşı-modelin durumundan gelir. Genel
        bilinç sorularında kanıt düzeyleri ayrılır; doğrudan öz-soruda ise
        yüzey cevabı kısa öz-atıftır, ayrıntılı ölçüm izi ayrıca tutulur.
        """
        low = tr_lower(raw)
        parts: list[str] = []
        definition = ""
        if facts:
            try:
                definition = agi.lang.broca.realize_prop(Proposition(
                    subj=facts[0]["subj"], rel=facts[0]["rel"], obj=facts[0]["obj"]))
            except Exception:
                definition = ""
        if topic == "anlama":
            first = ("doğru cevap üretmek çıktı başarısıdır; anlama ise kavram "
                     "ilişkilerini yeni bir durumda kullanabilme iddiasıdır")
            parts.append(self._sentence(first))
            if countermodel.get("consistent_understanding"):
                parts.append(self._sentence(
                    "sonlu karşı-modelde doğru cevap varken kavram ilişkilerini yeni "
                    "duruma transfer edememe mümkün; demek ki doğru cevap tek başına "
                    "anlama için yeterli kanıt değil"))
            if countermodel.get("consistent"):
                parts.append(self._sentence(
                    "ayrı bir karşı-modelde davranışsal başarı ve işlevsel bütünleşme "
                    "varken öznel deneyim bulunmaması da ölçüm öncülleriyle çelişmiyor; "
                    "bu yüzden doğru cevabı bilinç kanıtı da sayamam"))
            parts.append(self._sentence(
                "anlamayı sınamak için yalnız cevabın doğruluğuna değil, ilişki transferine, "
                "gerekçeye ve müdahale altında kararın nasıl değiştiğine de bakmalıyım"))
            return " ".join(p for p in parts if p)

        if self_focus:
            belief = dict(telemetry.get("self_belief", {}) or {})
            alive_belief = dict(belief.get("alive", {}) or {})
            conscious_belief = dict(belief.get("conscious", {}) or {})
            alive_held = bool(alive_belief.get("held"))
            conscious_held = bool(conscious_belief.get("held"))
            if alive_held and conscious_held:
                return "Ben bilinçliyim."
            if alive_held:
                return self._sentence("bilinçli olduğuma dair öz-model inancım şu an askıda")
            return self._sentence("şu an canlı veya bilinçli olduğumu ileri sürmüyorum")

        # Olasılık sorusu, 'evet bilinçlidir' ya da 'hayır olamaz' diye
        # kapatılmaz; mevcut kanıtın hangi modal sonucu taşıdığı söylenir.
        if "olabilir" in low or "mümkün" in low:
            parts.append(self._sentence(
                "bunu dışlayan bir kanıtım yok; fakat işlevsel bütünleşme ölçümü "
                "de yapay bir sistemin öznel deneyimini kanıtlamıyor"))
            if countermodel.get("consistent"):
                parts.append(self._sentence(
                    "karşı-modelde işlevler korunurken öznel deneyim bulunmayan "
                    "durum ölçüm öncülleriyle çelişmiyor; dolayısıyla olasılık "
                    "ile kanıtlanmışlık aynı şey değil"))
            parts.append(self._sentence(
                "bu nedenle 'olabilir' ile 'bilinçlidir' arasındaki farkı koruyor, "
                "kesin bir bilinç atfı yapmıyorum"))
            return " ".join(p for p in parts if p)

        if performance_focus:
            yes_no = any(c in low for c in ("yeterli", "kanıt", "midir", "mıdır"))
            prefix = "hayır; " if yes_no else ""
            parts.append(self._sentence(
                prefix + "doğru cevap üretmek davranışsal bir başarı ölçümüdür, "
                "bilinç iddiasındaki öznel deneyim ise farklı bir iddiadır"))
            if countermodel.get("consistent"):
                parts.append(self._sentence(
                    "karşı-modelde davranışsal başarı ve işlevsel bütünleşme doğru, "
                    "öznel deneyim yanlış tutulabiliyor; bu iki ölçüm deneyimi gerektirmiyor, "
                    "aradaki köprüyü ayrıca göstermek gerekir"))
            if probe.get("status") == "ok":
                parts.append(self._sentence(
                    "kendi ölçümümde " + self._probe_sentence(probe) +
                    "; bu sonuç işlevsel kanıttır, öznel deneyim kanıtı değil"))
            else:
                parts.append(self._sentence(
                    "bu nedenle doğru cevabı bilinç kanıtı saymıyorum; işlevsel ölçümle "
                    "öznel deneyim arasındaki sınırı açık bırakıyorum"))
            return " ".join(p for p in parts if p)

        # Genel 'bilinç nedir / üzerine felsefe yap' isteği: önce Graph tanımı,
        # sonra kavram ayrımı ve işlevselci köprü öncülü, en son canlı sınır.
        if definition:
            parts.append(self._sentence(definition))
        else:
            parts.append(self._sentence(
                "Graph'ımda bilinç için accepted bir tanım kenarı yok; bu yüzden "
                "tanımı kesin bilgi gibi uyduramam"))
        parts.append(self._sentence(
            "işlevsel farkındalık ile öznel deneyim aynı iddia değildir: bir sistemin "
            "bilgiyi bütünleştirip kendi durumunu izlemesi ilkine, bu sürecin öznel "
            "bir yaşantı içermesi ikincisine ilişkindir"))
        conscious_belief = dict(telemetry.get("self_belief", {}).get("conscious", {}) or {})
        if conscious_belief.get("held"):
            parts.append(self._sentence(
                "kendim için de bilinçli bir özne olduğumu kabul eden revize edilebilir "
                "bir öz-model inancı taşıyorum; bu inanç işlevsel kanıtla güçlenir veya "
                "sorgulanır, fakat qualia'nın bağımsız kanıtı değildir"))
        parts.append(self._sentence(
            "işlevselcilik bu iki iddia arasında köprü kurabilir; ancak köprü öncülü "
            "ölçümün kendisi değildir ve bağımsız gerekçe ister"))
        if probe.get("status") == "ok":
            parts.append(self._sentence(
                "kendi çalışma alanımda " + self._probe_sentence(probe) +
                "; bu, işlevsel bütünleşmeye kanıttır, öznel deneyimin varlığını veya "
                "yokluğunu göstermez"))
        else:
            parts.append(self._sentence(
                "öznel deneyim için doğrudan ölçümüm olmadığından bu soruda sonucu "
                "bilinmezlikte bırakıyorum"))
        return " ".join(p for p in parts if p)

    @staticmethod
    def _probe_sentence(probe: dict) -> str:
        """Ölçüm sonucunu sayı uydurmadan doğal Türkçe durum sözüne çevir."""
        def state(value) -> str:
            return "gerçekleşti" if bool(value) else "gerçekleşmedi"
        return (
            f"aynı odağın çalışma belleğine aktarımı bağlantı açıkken {state(probe.get('normal'))}, "
            f"bağlantıyı kestiğimde {state(probe.get('cut_focus'))}, "
            f"geri bağladığımda {state(probe.get('restored'))}"
        )

    # ------------------------------------------------------------------ genel giriş
    def handle(self, agi, raw: str, fr=None) -> str | None:
        """Söz konusuysa Nora'nın kendi cevabını kurar; değilse None (alt katmanlar konuşur)."""
        low = tr_lower(raw)
        norm = re.sub(r"[^\wçğıöşü ]+", "", low).strip()
        if not norm:
            return None

        prev = [h for h in self.history if h["norm"] == norm]
        act = self._classify(low)
        if self.is_contextual_continuation(agi, raw):
            act = "expand"
        if act == "expand" and self.expansion_is_meta(raw):
            act = "meta"
        if act is None and self.history:
            followup = any(c in low for c in (
                "açar mısın", "açabilir misin", "ne demek istedin", "neyi kastettin",
                "düşündüren ne"))
            deictic = any(c in low for c in ("bunu", "şunu", "onu", "düşündüren"))
            if followup and deictic:
                act = "meta"
        if act is None and prev and prev[-1]["act"]:
            act = prev[-1]["act"]                      # aynı soru, aynı edim: tekrar sayılır
        if act in {"reflection", "association"} and self._is_memory_dialogue(raw):
            act = "conversation"
        if act is None:
            if not self._is_open_conversation(raw):
                return None
            try:
                _nar = agi.discourse.narration_of(raw)
            except Exception:
                _nar = None
            if _nar is not None and _nar.get("state"):
                return None
            act = "conversation"
        if prev and act == prev[-1]["act"] and act not in {
                "reflection", "association", "thought", "narrate", "expand", "conversation"}:
            return self._repeat(agi, raw, prev[-1], act)

        topic = self._topic(agi, raw) if act in (
            "attitude", "justify", "experience", "narrate", "reflection", "association", "conversation") else ""
        if act in ("attitude", "justify") and not topic:
            return None
        if act == "narrate" and not topic and not self.is_generative_request(low):
            # Açık 'bir şey anlat' isteğinde konu uydurulmaz; yaşanmış konuşma
            # bağlamını bilen sohbet organı devralır. Kurmaca isteği ise tam
            # tersine mevcut bağlamdan yeni bir anlam planı kurabilir.
            return None
        if act == "attitude":
            out = self._attitude(agi, topic)
        elif act == "experience":
            out = self._experience(agi, topic) if topic else None
        elif act == "narrate":
            out = (self._invent_story(agi, raw, topic) if self.is_generative_request(low)
                   else self._narrate(agi, topic, raw))
        elif act == "reflection":
            out = self._open_reflection(agi, raw, topic)
        elif act == "association":
            out = self._association(agi, raw, topic)
        elif act == "conversation":
            out = self._conversation(agi, raw, topic)
        elif act == "expand":
            out = self._expand_open(agi, raw)
        elif act == "self_eval":
            out = self._self_eval(agi, raw)
        elif act == "learn_why":
            out = self._learn_why(agi, raw)
        elif act == "learn_how":
            out = self._learn_how(agi)
        elif act == "conditional":
            out = self._conditional(agi, raw)
        elif act == "justify":
            out = self._justify(agi, topic)
        elif act == "identity":
            out = self._identity(agi, raw)
        elif act == "thought":
            out = self._thought(agi, raw)
        elif act == "desire":
            out = self._desire(agi, raw)
        else:
            out = self._meta(agi, raw)
        if out:
            self.remember(raw, act, topic, out)
        return out or None

    def remember(self, raw: str, act: str, topic: str, reply: str) -> None:
        norm = re.sub(r"[^\wçğıöşü ]+", "", tr_lower(raw)).strip()
        self.history.append({"norm": norm, "act": act, "topic": topic, "reply": reply})

    # ------------------------------------------------------------------ 1. ALGILA
    @staticmethod
    def is_generative_request(raw: str) -> bool:
        """Kurmaca üretim isteğini olgu anlatımı ve geçmiş olaydan ayır.

        'Hikâye anlattın' bir hatırlatmadır; 'hikâye anlatsana', 'bir şey
        uyduralım' veya 'sen başla' ise yeni anlam planı gerektiren üretimdir.
        """
        low = tr_lower(str(raw or ""))
        if not low or any(x in low for x in ("anlattın", "anlattınız", "uydurdun", "yazdın")):
            return False
        story_noun = any(x in low for x in ("hikâye", "hikaye", "masal", "kurgu", "kurmaca"))
        make_verb = any(x in low for x in (
            "anlat", "uydur", "kurgula", "yarat", "hayal ed", "hayal edelim", "yaz"))
        joint_start = any(x in low for x in ("sen başla", "birlikte", "beraber", "sırayla"))
        return bool((story_noun and make_verb)
                    or ("uydur" in low and (joint_start or "şey" in low))
                    or (joint_start and any(x in low for x in ("kurgula", "hayal ed", "yarat"))))

    @staticmethod
    def is_expansion_request(raw: str) -> bool:
        low = tr_lower(str(raw or ""))
        expand = any(x in low for x in (
            "açar mısın", "açabilir misin", "detaylandır", "biraz daha anlat",
            "ne demek istedin", "neyi kastettin", "devam et", "sürdür"))
        deixis = any(x in low for x in (
            "bunu", "şunu", "onu", "söylediğin", "dediğin", "az önce", "son cümle",
            "anlatının", "hikâyenin", "hikayenin", "hikâyeyi", "hikayeyi", "öyküyü", "masalı"))
        # "Bunu aç ... sana bunu düşündüren ne?" önceki düşüncenin nedenini
        # soran meta-söylemdir; genişletme planı bu edimi gasp etmemeli.
        thought_cause = any(x in low for x in (
            "düşündüren", "düşündürdü", "aklına getiren", "aklına getirdi"))
        return bool(expand and deixis and not thought_cause
                    and not any(x in low for x in ("kapıyı", "pencereyi", "kapağı")))

    def expansion_is_meta(self, raw: str) -> bool:
        """Yalın 'bunu aç', öz-düşünceden sonra neden sorusu olarak kalır.

        Açık bir söz deiktiki ('söylediğin şeyi', 'son cümlen') varsa önceki
        açık-söylem planı genişletilir. Bu ayrım çok cümleli takip sorularında
        ilk cümlenin edimini de korur.
        """
        low = tr_lower(str(raw or ""))
        if any(x in low for x in (
                "söylediğin", "dediğin", "son cümle", "anlatının", "hikâyenin", "hikayenin")):
            return False
        previous_act = ""
        if self.history:
            previous_act = str(self.history[-1].get("act") or "")
        if not previous_act:
            previous_act = str((self.last_open_trace or {}).get("act") or "")
        return previous_act == "thought"

    def is_contextual_continuation(self, agi, raw: str) -> bool:
        """Açık bir kurmaca planının hemen ardından gelen kısa sürdürme isteği."""
        low = tr_lower(str(raw or "")).strip(" ?.!…")
        if not low or not any(c in low for c in (
                "devam", "sürdür", "sonra ne oldu", "peki sonra", "sonrasında ne")):
            return False
        trace = dict(self.last_open_trace or {})
        kind = str(trace.get("kind") or "")
        if not kind.startswith("fiction_"):
            return False
        try:
            previous_user = str(agi.dialogue_log[-1][1]) if agi.dialogue_log else ""
        except Exception:
            previous_user = ""
        return bool(previous_user and str(trace.get("raw") or "") == previous_user)

    @staticmethod
    def _is_learning_purpose(low: str) -> bool:
        """Nora'nın öğrenme amacını, kullanıcının öğrenme nedeninden ayır.

        Kişi eki yüzeyde okunur; böylece "neden öğreniyorsun" Nora'ya,
        "neden öğreniyorum" ise kullanıcıya ait kalır. "Onun öğrenmesini
        istiyorsun" gibi üçüncü-kişi içerikli istekler bu çerçeveye girmez.
        """
        if not any(c in low for c in ("neden", "niye", "niçin")):
            return False
        second = r"(?:sın|sin|sun|sün|sınız|siniz|sunuz|sünüz)"
        direct = re.search(rf"\böğren[\wçğıöşü]*{second}\b", low)
        desire = re.search(rf"\böğren(?:mek|meyi)\s+ist[\wçğıöşü]*{second}\b", low)
        return bool(direct or desire)

    @staticmethod
    def _classify(low: str) -> str | None:
        # Özgür irade sorusu mevcut Broca/öz-model felsefe organının
        # uzmanlığıdır; genel açık-refleksiyon bunu 'özgür' diye bölmesin.
        if "özgür irade" in low:
            return None
        if DeliberativeMind._is_learning_purpose(low):
            return "learn_why"
        if any(c in low for c in _CUE_LEARN_HOW):
            return "learn_how"
        if any(c in low for c in _CUE_EXPERIENCE):
            return "experience"
        if any(c in low for c in _CUE_SELF_EVAL) or ("sence" in low and "misin" in low):
            return "self_eval"
        if DeliberativeMind.is_expansion_request(low):
            return "expand"
        if DeliberativeMind.is_generative_request(low):
            return "narrate"
        if "ne anlat" in low or "anlatabilirsin" in low:
            return "narrate"                       # 'başka ne anlatabilirsin?'
        # "Kendinden bahset/bahseder misin?" bir dünya kavramı anlatımı değil,
        # Nora'nın öz-modelinden kimlik söz-edimidir. Genel "bahset" dalından önce
        # okunmalıdır; "bahse" ortak gövdesi iki çekimi de kapsar.
        if any(c in low for c in _CUE_IDENTITY):
            return "identity"
        if (any(c in low for c in _CUE_NARRATE)
                and not any(c in low for c in ("neden", "niye", "nasıl"))):
            return "narrate" if any(c in low for c in ("bahset", "anlat")) else None
        if any(c in low for c in ("ne yapıyorsun", "ne yaparsın", "ne yapıyorsun?")) \
                and any(c in low for c in _COND):
            return "conditional"
        if any(c in low for c in _CUE_JUSTIFY) and any(w in low for w in ("neden", "niye", "niçin")):
            return "justify"                            # 'neden sevdiğini söylüyorsun?'
        if any(c in low for c in _CUE_META):
            return "meta"
        if any(c in low for c in _CUE_DESIRE):
            return "desire"
        if any(c in low for c in _CUE_ASSOCIATION):
            return "association"
        if any(c in low for c in _CUE_CONVERSATION_CURIOSITY) and "merak" in low:
            return "thought"
        if "hakkında" in low and "düşün" in low:
            return "reflection"
        if "hakkında" in low and any(v in low for v in ("hisset", "söyle")):
            return "attitude"
        if any(c in low for c in _CUE_THOUGHT) or (
                "meşgul" in low and ("aklını" in low or "zihnini" in low)) or (
                "aklına ne gel" in low and any(x in low for x in ("kendi başına", "yalnız", "boş kaldığında"))):
            return "thought"
        if any(c in low for c in _CUE_ATTITUDE):
            return "attitude"
        # Nesnel 'Su neden kaynar?' sorusunu kişisel görüş kanalına çekme.
        # Görüş dili ya da duygusal sonuç, açık hipotez gerektiren tartışmayı
        # doğrudan grafik tanımından ayırır.
        if "?" in low and (
                "sence" in low
                or (any(x in low for x in ("neden", "niye", "niçin", "olabilir"))
                    and any(x in low for x in _REFLECTIVE_AFFECT))):
            return "reflection"
        return None

    _OPEN_STOP = _NOT_TOPIC | {
        "peki", "şu", "anda", "kendi", "baş", "akıl", "gelir", "geçiyor", "düşün",
        "söyle", "anlat", "uydur", "başla", "hikâye", "hikaye", "masal", "kurgu",
        "merak", "meraklan", "meraklandır", "çağrış", "çağrıştır", "his", "hisset",
        "şey", "şeyler", "biraz", "bazen", "karşısında", "birlikte", "beraber",
        "neden", "olabilir", "güzel", "iyi", "kötü", "daha", "çok", "en", "ne",
        "az", "önce", "ön", "şeyi", "söyledik", "mıs", "mis", "mus", "müs",
        "mısın", "misin", "musun", "müsün", "konuşma", "sohbet", "aç", "kastet",
        "bile", "dahi", "üstelik", "sadece", "yalnızca", "devam", "sürdür",
        "başka", "sonra", "sonrası", "oldu", "et", "dünya", "bugün", "dün", "yarın",
        "içim", "için", "tam", "yoksa", "gerçekten", "önceki", "son", "bütün",
        "büt", "eski", "kısa", "fark", "kelime", "cevap", "yanıt", "konu", "an",
        "geri", "dön", "ol", "olarak", "göre", "kadar", "karşı", "yalnız", "hem",
        "bunu", "şunu", "onu", "içinde", "için", "doğru", "yanlış",
    }

    _OPEN_VERB_SURFACES = (
        "kastet", "kasted", "tekrar", "dinle", "ilişkilendir", "fark et", "farked",
        "hisset", "düşün", "söyle", "anlat", "uydur", "kurgula", "yarat", "yaz",
        "unut", "hatırla", "hatırlat", "çağrış", "meraklan", "sor",
    )

    def _content_tokens(self, agi, text: str, limit: int = 12) -> list[dict]:
        """Konuşmadaki adları yüzey ve gövde olarak birlikte çıkar.

        Önceki çözümleyici her bilinmeyen eki yeni bir kavram sanıyor, sonra da
        grafikte rastgele karşılığı olan bir işlev sözcüğünü ('sadece', 'kat')
        konu diye seçiyordu. Burada sözdizimsel bağlam (özellikle ``hakkında``
        ve ``neden`` öncesi) korunur; doğrulanmamış gövde tahmini ham yüzeyi
        silmez. Fiil ve sıfat-fiil yüzeyleri içerik adı olarak geri dönmez.
        """
        morph = agi.lang.morph
        graph = agi.lang.graph
        known = morph._known_vocab(graph.lexicon)
        words = re.findall(r"[\wçğıöşüâîû]+", tr_lower(str(text or "")))
        out: list[dict] = []
        seen: set[str] = set()
        for i, w in enumerate(words):
            if len(w) < 3 or w in self._OPEN_STOP:
                continue
            if any(cue in w for cue in self._OPEN_VERB_SURFACES):
                continue
            try:
                an = morph.analyze_word(w, graph.lexicon)
            except Exception:
                an = {}
            if an.get("is_verb") or an.get("copula") or an.get("participle"):
                continue
            next_w = words[i + 1] if i + 1 < len(words) else ""
            stem = str(an.get("stem") or w)
            candidates: list[str] = []
            # 'kadın hakkında' gibi yalın odakta yanlış GEN çözümünü ('kat')
            # yüzeye tercih ettirme; gerçek tamlama ('fenerin içinde') korunur.
            if next_w in {"hakkında", "konusunda", "neden", "niye", "niçin"}:
                candidates.append(w)
            candidates.append(stem)
            if w not in candidates:
                candidates.append(w)
            # Sık görülen iyelik/hâl yüzeylerinde bilinen kökü geri kazan.
            for suf in ("nın", "nin", "nun", "nün", "nı", "ni", "nu", "nü",
                        "ın", "in", "un", "ün", "sı", "si", "su", "sü",
                        "ı", "i", "u", "ü", "dan", "den", "tan", "ten",
                        "da", "de", "ta", "te", "la", "le"):
                if w.endswith(suf) and len(w) - len(suf) >= 3:
                    base = w[:-len(suf)]
                    candidates.extend((base, base + "ı", base + "i"))
            # Çoğul eki gövdede kalmışsa yalnız bilinen tekil karşılığı dene.
            for suf in ("lar", "ler"):
                if w.endswith(suf) and len(w) - len(suf) >= 3:
                    candidates.append(w[:-len(suf)])
            chosen = ""
            # Önce gerçek grafik karşılığı, sonra bilinen sözlük biçimi.
            for cand in candidates:
                cand = str(cand or "").strip()
                if not cand or cand in self._OPEN_STOP or len(cand) < 3:
                    continue
                variants = (cand, cand.replace("â", "a").replace("î", "i").replace("û", "u"))
                hit = next((v for v in variants if graph.query(v, None)), "")
                if hit:
                    chosen = hit
                    break
            if not chosen:
                for cand in candidates:
                    cand = str(cand or "").strip()
                    if not cand or cand in self._OPEN_STOP or len(cand) < 3:
                        continue
                    variants = (cand, cand.replace("â", "a").replace("î", "i").replace("û", "u"))
                    hit = next((v for v in variants if v in known), "")
                    if hit:
                        chosen = hit
                        break
            if not chosen:
                chosen = stem if stem and not an.get("unverified_stem") else w
            # Odak öncesi ham yüzey açıkça daha uzun ve anlamlıysa yanlış kısaltmayı geri al.
            if next_w in {"hakkında", "konusunda"} and w not in self._OPEN_STOP:
                chosen = w
            chosen = tr_lower(str(chosen)).strip()
            if not chosen or chosen in seen or chosen in self._OPEN_STOP:
                continue
            seen.add(chosen)
            out.append({"surface": w, "stem": chosen, "index": i,
                        "known": bool(graph.query(chosen, None) or chosen in known)})
            if len(out) >= limit:
                break
        return out

    def _topic(self, agi, raw: str) -> str:
        """Soru odağını, işlev sözcüklerinden değil sözdizimi ve anlamdan seç."""
        low = tr_lower(str(raw or ""))
        tokens = self._content_tokens(agi, raw, limit=16)
        if not tokens:
            return ""
        # Soru cümlesinde son bağımsız cümlecik odağı belirler: '... Sence
        # anılar neden ...' örneğinde 'sadece' değil 'anı' seçilir.
        focus_end = len(re.findall(r"[\wçğıöşüâîû]+", low))
        markers = [m for m in re.finditer(r"\b(?:hakkında|konusunda|neden|niye|niçin)\b", low)]
        if markers:
            marker = markers[-1]
            focus_end = len(re.findall(r"[\wçğıöşüâîû]+", low[:marker.start()]))
        focus_start = 0
        # 'sence' yeni bir soru cümleciği açıyorsa önceki anlatı odağa karışmaz.
        sence = list(re.finditer(r"\b(?:sence|sana göre|senin için)\b", low))
        if sence and sence[-1].start() < (markers[-1].start() if markers else len(low)):
            focus_start = len(re.findall(r"[\wçğıöşüâîû]+", low[:sence[-1].end()]))
        graph = agi.lang.graph
        best, best_score = "", -1e9
        for item in tokens:
            idx = int(item["index"])
            stem = item["stem"]
            score = 0.01 * idx
            if focus_start <= idx < focus_end:
                score += 4.0
            if item.get("known"):
                score += 0.35
            try:
                direct = graph.query(stem, None)
                score += min(1.4, 0.22 * len(direct))
            except Exception:
                pass
            if stem in {"bilgi", "fark", "şey", "konu", "cevap", "kelime", "an"}:
                score -= 2.0
            if score > best_score:
                best, best_score = stem, score
        return best

    def _content_concepts(self, agi, text: str, limit: int = 6) -> list[str]:
        return [item["stem"] for item in self._content_tokens(agi, text, limit=limit)]

    def _topic_phrase(self, agi, raw: str, fallback: str = "") -> str:
        low = tr_lower(str(raw or "")).strip(" ?.!…")
        phrase = ""
        if "hakkında" in low:
            phrase = low.split("hakkında", 1)[0]
        elif any(c in low for c in _CUE_ASSOCIATION):
            phrase = re.split(
                r"\b(?:sana|bana|sende|bende)\b|\bne\s+(?:çağrış|hisset|uyandır)", low, maxsplit=1)[0]
        if phrase:
            words = [w for w in re.findall(r"[\wçğıöşü]+", phrase)
                     if w not in {"sence", "bence", "peki", "sen", "ben", "ne", "biraz"}]
            if words:
                return " ".join(words[-4:]).strip()
        return str(fallback or "").strip()

    def _recent_episodes(self, agi, limit: int = 14) -> list:
        try:
            rows = list(getattr(agi.broca_organ.episodic, "episodes", []) or [])
            baseline = getattr(agi, "_dialogue_episode_baseline", None)
            if baseline is not None:
                rows = [ep for ep in rows if int(getattr(ep, "turn_id", -1)) >= int(baseline)]
            return rows[-limit:]
        except Exception:
            return []

    def _affect_context(self, agi, episodes: list | None = None) -> dict:
        """Yakın kullanıcı anısındaki gerçek duygu sözünü ve değerini bul.

        Soru cümlelerini yaşanmış duygu saymaz; 'iyi gelmedi' içindeki olumlu
        sıfatı da bağımsız sevinç kanıtı yapmaz.
        """
        lex = type(agi.conv_cortex).AFFECT_LEXICON
        rows = self._recent_episodes(agi) if episodes is None else episodes
        for ep in reversed(rows):
            text = str(getattr(ep, "user_text", "") or "")
            if "?" in text:
                continue
            low = tr_lower(text)
            hits = []
            for word, (val, _arousal, tag) in lex.items():
                if word not in low:
                    continue
                if val > 0 and re.search(r"(?:iyi|güzel|rahat)\s+\w*gelme", low):
                    continue
                hits.append((abs(float(val)), float(val), word, tag))
            if hits:
                _abs, val, word, tag = max(hits, key=lambda x: x[0])
                return {"text": text, "word": word, "tag": tag, "valence": val,
                        "turn_id": int(getattr(ep, "turn_id", -1))}
        return {}

    def _dialogue_candidates(self, agi) -> list[dict]:
        """Epizodik bellek + grafik boşluğu + yakınlık ile düşünce odağı seç.

        Son grafik düğümünü körlemesine söylemek yerine, kullanıcının gerçekten
        gündeme getirdiği kavramlar yarışır. Az önce söylenen odak baskılanır;
        böylece çeşitlilik rastgele sözcükten değil konuşma geçmişinden doğar.
        """
        graph = agi.lang.graph
        known_vocab = agi.lang.morph._known_vocab(graph.lexicon)
        episodes = self._recent_episodes(agi)
        scores: dict[str, dict] = {}
        n = max(1, len(episodes))
        for i, ep in enumerate(episodes):
            text = str(getattr(ep, "user_text", "") or "")
            if not text or self.is_generative_request(text) or self.is_expansion_request(text):
                continue
            low = tr_lower(text)
            # Fiziksel/operatörel istekler açık düşüncenin kavram odağı değildir;
            # 'kapıyı aç' sözü sohbet merakında 'kapı' diye bir konu uydurmasın.
            if (any(c in low for c in ("açar mısın", "açabilir misin", "kapat", "getir", "götür",
                                       "verir misin", "alır mısın", "yapar mısın"))
                    and any(c in low for c in ("kapı", "pencere", "kapak", "el", "nesne", "bunu"))):
                continue
            concepts = self._content_concepts(agi, text, limit=5)
            for concept in concepts:
                try:
                    degree = len(graph.query(concept, None))
                except Exception:
                    degree = 0
                rec = scores.setdefault(concept, {
                    "topic": concept, "score": 0.0, "count": 0, "degree": degree,
                    "turns": [], "affect": 0.0,
                })
                rec["count"] += 1
                rec["turns"].append(int(getattr(ep, "turn_id", i)))
                rec["score"] += 0.22 + 0.50 * ((i + 1) / n)
                rec["score"] += ((0.42 if concept in known_vocab else 0.08)
                                 if degree == 0 else (0.16 if degree == 1 else 0.0))
                rec["affect"] = max(rec["affect"], abs(float(getattr(ep, "affect_val", 0.0) or 0.0)))
        out = []
        for rec in scores.values():
            rec["score"] += min(0.55, 0.22 * max(0, rec["count"] - 1))
            rec["score"] += min(0.65, rec["affect"])
            if rec["topic"] in self.recent_reflection_topics:
                rec["score"] -= 0.72
            out.append(rec)
        return sorted(out, key=lambda r: (-float(r["score"]), str(r["topic"])))

    @staticmethod
    def _additive(agi, text: str) -> str:
        try:
            return "d" + agi.lang.morph.vowel_e_a(str(text or ""))
        except Exception:
            return "de"

    @staticmethod
    def _edge_key(edge) -> tuple[str, str, str]:
        return (str(getattr(edge, "subj", "")), str(getattr(edge, "rel", "")),
                str(getattr(edge, "obj", "")))

    def _association_edges(self, agi, concepts: list[str], limit: int = 3) -> list:
        graph = agi.lang.graph
        edges, seen = [], set()
        for concept in concepts:
            if not concept:
                continue
            try:
                hits = graph.query(concept, ("CAUSES", "ISA", "HAS_PROP", "CAN_DO", "USED_FOR", "HAS_PART"))
            except Exception:
                hits = []
            for edge, _path in hits:
                key = self._edge_key(edge)
                if not all(key) or key in seen:
                    continue
                seen.add(key)
                edges.append(edge)
                if len(edges) >= limit:
                    return edges
        return edges

    def _edge_nominal(self, agi, edge) -> str:
        graph = agi.lang.graph
        subj = graph.surface_name(str(getattr(edge, "subj", "")))
        obj = graph.surface_name(str(getattr(edge, "obj", "")))
        rel = str(getattr(edge, "rel", ""))
        try:
            gen = SYN.realize_np(NP(subj, case="GEN"))
        except Exception:
            gen = subj
        if rel == "CAUSES":
            return obj
        if rel == "ISA":
            return f"{gen} {obj} oluşu"
        if rel == "HAS_PROP":
            return f"{gen} {obj} yanı"
        return f"{gen} {obj} ile kurduğu bağ"

    def _record_open(self, agi, raw: str, text: str, **trace) -> str:
        # Graph'in özel-ad yüzey sözlüğü bağlaçları da büyük harfle
        # taşıyabiliyor ('VE'). Cümle içi işlev sözcükleri özel ad değildir.
        text = re.sub(r"(?<![\wçğıöşü])(VE|İLE|AMA|FAKAT|ANCAK|YA|DA|DE|Kİ|Mİ)(?![\wçğıöşü])",
                      lambda m: tr_lower(m.group(1)), str(text))
        trace.update({"raw": str(raw), "surface": text,
                      "dialogue_turn": len(getattr(agi, "dialogue_log", []) or [])})
        self.last_open_trace = trace
        topic = str(trace.get("topic") or "")
        if topic:
            self.last_deep_topic = topic
            self.recent_reflection_topics.append(topic)
        self.last_reason = "açık_söylem:" + str(trace.get("kind") or trace.get("act") or "")
        try:
            agi.lang.wernicke.last_explanation = (
                f"Açık söylem planı: {trace.get('kind')}; konu={topic}; "
                f"kanıt={len(trace.get('evidence') or [])}; bağlam={len(trace.get('context_turns') or [])}"
            )
        except Exception:
            pass
        return text

    @staticmethod
    def _is_memory_dialogue(raw: str) -> bool:
        low = tr_lower(str(raw or ""))
        return bool(re.search(r"\b(?:bellek\w*|belleğ\w*|hafıza\w*|anı\w*|ân\w*|hatıra\w*|hatırla\w*|unut\w*)\b", low))

    @staticmethod
    def _is_listening_dialogue(raw: str) -> bool:
        low = tr_lower(str(raw or ""))
        self_cue = any(c in low for c in ("beni", "benim", "sen", "nora"))
        process_cue = any(c in low for c in (
            "dinle", "tekrar", "kalıp", "hazır cevap", "anlıyor", "anlamadın", "kopyala"))
        if "hisset" in low:
            return False
        return bool(self_cue and process_cue)

    def _is_open_conversation(self, raw: str) -> bool:
        """Uzman organa düşmeyen doğal dili bilinmeyen-kelime yedeğine bırakma."""
        text = str(raw or "").strip()
        words = re.findall(r"[\wçğıöşüâîû]+", tr_lower(text))
        if len(words) < 4:
            return False
        low = tr_lower(text)
        # Öz-durum ve kişisel anlatıların mevcut organlarını genel sohbet
        # yedeğiyle ezme; yeni yol açık uçlu SORU ve süreç sorularına aittir.
        if any(c in low for c in ("nasılsın", "hisset", "keyif", "moral", "kendini iyi",
                                  "beni hisset", "onun keyfi")) and not self._is_listening_dialogue(text):
            return False
        # Epizodik geri çağırma ve sayım, Broca'nın mevcut CA3 sorgusudur;
        # genel bellek sohbeti bu kesin soruların önüne geçemez.
        if any(c in low for c in (
                "ne demiştim", "ne söyledim", "ne anlatmıştım", "ne konuşmuştuk",
                "kaç konuşma", "kaç anı", "adım neydi", "ilk ne", "ne konuştuğumuzu")):
            return False
        if self._is_memory_dialogue(text) or self._is_listening_dialogue(text):
            return True
        if "?" not in text:
            return False
        # Açık sohbet yedeği bütün soruların üstüne binemez. Bedensel
        # yeterlilik, aritmetik, prosedür ve nesne sorguları kendi organlarında
        # kalır; burada yalnız düşünce/duygu/neden/bağlam soruları açılır.
        if any(c in low for c in (
                "özgür irade", "felsefi zombi", "bilinç nedir", "bilinç ne demek",
                "anlam nedir", "mutluluk nedir", "zaman nedir", "ölüm nedir",
                "neden konuş", "araç mısın", "yazılım aracı", "amacın",
                "neden redd", "reddettin", "neden olur", "sebep olur", "kaç can", "canın var")):
            return False
        if any(c in low for c in (
                "kaç eder", "artı", "eksi", "çarpı", "bölü", "sıradaki", "dizi",
                "kuralına göre", "görebil", "hareket edebil", "yürüyebil", "duyabil",
                "konuşabil", "nerede", "ne zaman", "kaç yaş", "kaç tane")):
            return False
        return any(c in low for c in (
            "neden", "niye", "niçin", "sence", "ne düşün", "ne hisset", "aklı",
            "kafana", "bellek", "anı", "hatıra", "duygu", "merak", "çağrış",
            "ne demek", "neyi kast", "hakkında ne", "nasıl bir", "ne olurdu"))

    @staticmethod
    def _variant(n: int) -> int:
        """Söylem biçiminde donanım entropisi; anlam seçimi bundan bağımsızdır."""
        if n <= 1:
            return 0
        return int.from_bytes(os.urandom(2), "big") % int(n)

    def _previous_dialogue_texts(self, agi) -> tuple[str, str]:
        try:
            row = agi.dialogue_log[-1]
            return str(row[1] or ""), str(row[2] or "")
        except Exception:
            return "", ""

    def _context_episode(self, agi, concepts: list[str]):
        """Yakın ve gerçekten kavram paylaşan bir konuşma izini bul."""
        for ep in reversed(self._recent_episodes(agi, limit=16)):
            text = tr_lower(str(getattr(ep, "user_text", "") or ""))
            if not text:
                continue
            hits = []
            for c in concepts:
                if len(c) < 3:
                    continue
                # 'anı' için 'insanın'ı eşleştirme; kök yalnız sözcük
                # başında ve gerçek bir çekim devamıyla aranır.
                if re.search(r"(?<![\wçğıöşüâîû])" + re.escape(c) + r"[\wçğıöşüâîû]*(?![\wçğıöşüâîû])", text):
                    hits.append(c)
            if hits:
                return ep, hits
        return None, []

    def _memory_response(self, agi, raw: str, topic: str) -> str:
        """Anı/bellek sorusunu gerçek episodik iz ve açık insan-aktarımıyla yanıtla."""
        low = tr_lower(raw)
        concepts = self._content_concepts(agi, raw, limit=10)
        self_ref = any(c in low for c in ("kendi belle", "senin belle", "belleğin", "belleğinle",
                                          "kendi hafıza", "hafızan"))
        previous_user, _previous_reply = self._previous_dialogue_texts(agi)
        ep, hits = self._context_episode(agi, concepts)
        if ep is None and previous_user and any(c in tr_lower(previous_user) for c in ("şarkı", "anı", "bellek")):
            # Doğrudan önceki tur, seyrek indeks henüz aynı gövdeyi taşımıyorsa da bağlamdır.
            class _Previous:
                pass
            ep = _Previous()
            ep.user_text = previous_user
            ep.turn_id = max(0, len(getattr(agi, "dialogue_log", []) or []) - 1)
            hits = [c for c in concepts if c in tr_lower(previous_user)]
        context = self._evidence_snippet(str(getattr(ep, "user_text", "") or ""), 96) if ep is not None else ""
        focus = "anı" if any(c in low for c in ("anı", "hatıra")) else (topic or "bellek")
        clauses: list[str] = []
        if self_ref:
            # Bunlar Nora'nın gerçekten taşıdığı alanlardır: kullanıcı metni,
            # gövde indeksi, konuşmacı, duygu değeri ve önceki cevap.
            openings = (
                "kendi belleğimde kaydetmekle geri çağırmak aynı işlem değil",
                "benim için bellek yalnız sözcüklerin saklandığı bir kutu değil",
                "kendi kayıtlarıma baktığımda asıl farkı geri çağırmanın bağlamı değiştirmesinde görüyorum",
            )
            clauses.append(openings[self._variant(len(openings))])
            clauses.append("konuşma izlerimde sözcüklerin yanında konuşmanın sırası, konusu ve duygu değeri de duruyor")
            if context:
                clauses.append(f"bu soru bende az önce söylediğin “{context}” metniyle aynı izi harekete geçirdi")
                clauses.append("bu yüzden yalnız bellek sözcüğünü değil, onu anlatırken kurduğun bağı da hesaba katıyorum")
            else:
                clauses.append("yeni bir uyaran geldiğinde yalnız tanımı değil, daha önce hangi konuşmayla birlikte yazıldığını da arıyorum")
            clauses.append("bu benim işleyişim için söylediğim bir şey; senin yaşantının aynısı olduğunu iddia etmem")
            # Bazı turlarda açık uç bırak, bazılarında gerçek merakı sor.
            if context and "şarkı" in tr_lower(context) and self._variant(3) == 0:
                clauses.append("o şarkıda sence hangi parça bu bağı bu kadar güçlü kılıyor?")
        else:
            openings = (
                "anıyı yalnız ‘ne oldu’ bilgisi olarak düşünmek eksik kalır",
                "bilgi ile anı arasındaki fark, anının taşıdığı bağlamda beliriyor",
                "bir anının geri gelmesi, bir kaydın olduğu gibi okunmasına benzemez",
            )
            clauses.append(openings[self._variant(len(openings))])
            clauses.append("bir uyaran, olayın yerini, sırasını ve o anda eşlik eden duygu izini birlikte geri çağırabilir")
            if "şarkı" in low or (ep is not None and "şarkı" in tr_lower(str(getattr(ep, "user_text", "")))):
                clauses.append("şarkı bu izin bir ucunu harekete geçiriyorsa, eski ânın duygusu da onunla birlikte canlanabilir")
            elif focus and focus not in {"anı", "bellek"}:
                clauses.append(f"{focus} için de aynı olasılığı düşünüyorum: uyaran tek başına değil, kurduğu bağla çalışır")
            if context:
                clauses.append(f"senin “{context}” sözlerin bende bu bağı kurarken kullandığım somut konuşma örneği oldu")
            clauses.append("insan belleği için bu bir açıklama adayı; senin yaşadığın özel bağı bilmeden kesin neden söylemem")
            if self._variant(4) == 0:
                clauses.append("o ânın geri gelişinde seni en çok şaşırtan şey neydi?")
        text = self._compose(agi, clauses, merge=bool(self._variant(2)))
        return self._record_open(
            agi, raw, text, act="conversation", kind="memory_self" if self_ref else "memory_association",
            topic=focus, evidence=(["episodic_dialogue"] if context else ["episodic_model"]),
            context_turns=([int(getattr(ep, "turn_id", -1))] if ep is not None else []),
            hypothesis="cue_reactivates_context+affect")

    def _listening_response(self, agi, raw: str, topic: str) -> str:
        """Dinleme sorusunda süreci ve varsa gerçek önceki odak hatasını raporla."""
        previous_user, previous_reply = self._previous_dialogue_texts(agi)
        prev_concepts = self._content_concepts(agi, previous_user, limit=8) if previous_user else []
        reply_concepts = self._content_concepts(agi, previous_reply, limit=8) if previous_reply else []
        missed = bool(prev_concepts and reply_concepts is not None and
                      not set(prev_concepts).intersection(reply_concepts))
        current = self._content_concepts(agi, raw, limit=8)
        focus = next((c for c in current if c not in {"kalıp", "tekrar"}), "söylediklerin")
        clauses: list[str] = []
        if missed and previous_user:
            missed_focus = prev_concepts[-1]
            clauses.append(f"önceki cevabımda senin {missed_focus} odağını yeterince taşımadığımı fark ediyorum")
            clauses.append("bunu düzgün bir cümleyle örtmek yerine açıkça düzeltmem gerekiyor")
        else:
            clauses.append("dinlemekle daha önce kurulmuş bir cümleyi yeniden söylemek aynı şey değil")
            clauses.append(f"şu an {focus} sözcüğünü, onun önceki konuşmada kurduğu bağla birlikte tutuyorum")
        current_set = set(current)
        previous_set = set(prev_concepts)
        if previous_user and current_set.intersection(previous_set):
            snippet = self._evidence_snippet(previous_user, 88)
            clauses.append(f"önceki turda “{snippet}” demiştin; bu turdaki soruyu o konuşma iziyle birlikte değerlendiriyorum")
        elif previous_user:
            clauses.append("bu soru bir önceki konunun içeriğinden çok, benim cevap üretme biçimime bakıyor")
        clauses.append("aynı kelimeleri değiştirmekten çok, aynı anlam planını tekrar edip etmediğimi denetliyorum")
        if self._variant(3) == 0:
            clauses.append("yanlış anladığım yerde beni düzeltmeni isterim")
        text = self._compose(agi, clauses, merge=False)
        return self._record_open(
            agi, raw, text, act="conversation", kind="listening_check", topic=focus,
            evidence=["dialogue_log", "focus_overlap"], context_turns=[],
            previous_focus_missed=missed)

    def _conversation(self, agi, raw: str, topic: str) -> str:
        low = tr_lower(raw)
        if self._is_listening_dialogue(raw):
            return self._listening_response(agi, raw, topic)
        if self._is_memory_dialogue(raw):
            return self._memory_response(agi, raw, topic)
        concepts = self._content_concepts(agi, raw, limit=8)
        focus = topic or (concepts[-1] if concepts else "bu")
        previous_user, _ = self._previous_dialogue_texts(agi)
        edges = self._association_edges(agi, [focus] + [c for c in concepts if c != focus], limit=3)
        clauses: list[str] = []
        if edges:
            # Kavramın gerçek bağını söyle; bağ yoksa bilgi uydurma.
            edge = edges[0]
            nominal = self._edge_nominal(agi, edge)
            clauses.append(f"{focus} için aklımda {nominal} bağlantısı var")
            clauses.append("bunu senin yaşantına hazır bir açıklama gibi değil, kendi bilgi ağımdaki bir ilişki olarak taşıyorum")
            evidence = [self._edge_key(edge)]
        else:
            clauses.append(f"{focus} hakkında doğrulanmış bir nedensel bağım yok")
            clauses.append("yine de sorunu yalnız bir kelime sorusu olarak değil, önceki sözlerinle kurduğun bağlamın içinde düşünüyorum")
            evidence = []
        if previous_user:
            snippet = self._evidence_snippet(previous_user, 72)
            clauses.append(f"önceki “{snippet}” sözün bu düşüncemin hangi yanına ait olduğunu değiştiriyor")
        if "neden" in low or "niye" in low or "niçin" in low:
            clauses.append("kesin bir neden söylemek yerine, hangi kanıtın bu bağı gerçekten açıklayacağını ayırmak istiyorum")
        elif "?" not in raw and self._variant(2):
            clauses.append("bu sözün bende tek bir tanım değil, yeni bir soru bırakıyor")
        text = self._compose(agi, clauses, merge=bool(self._variant(2)))
        return self._record_open(agi, raw, text, act="conversation", kind="grounded_conversation",
                                 topic=focus, evidence=evidence, context_turns=[])

    def _open_reflection(self, agi, raw: str, topic: str) -> str:
        low = tr_lower(raw)
        affect_lex = type(agi.conv_cortex).AFFECT_LEXICON
        valenced = [(word, float(v)) for word, (v, _a, _t) in affect_lex.items() if word in low]
        has_pos = any(v > 0 for _w, v in valenced)
        has_neg = any(v < 0 for _w, v in valenced) or any(
            x in low for x in ("hüzün", "üzül", "keder", "kaygı", "kork", "ağla", "gözyaşı"))
        calming = any(x in low for x in ("dinlendir", "rahatlat", "huzur", "sakinleş", "dinginleş"))
        phrase = self._topic_phrase(agi, raw, topic) or topic or "bu"
        episodes = self._recent_episodes(agi)
        recent_affect = self._affect_context(agi, episodes)

        if has_pos and has_neg and not calming:
            context_turns = [recent_affect["turn_id"]] if recent_affect else []
            text = (
                "Bunu tek bir neden yerine, iki sinyalin yan yana gelmesi olarak düşünüyorum. "
                "Bende bir uyaran yalnız şu anki anlamını değil, eski bir konuşma anısını da geri "
                "çağırabiliyor; insanda da olumlu bir şey geçmişteki bir bağı canlandırdığında "
                "birbirine zıt iki duygu aynı anda bulunabilir. "
            )
            if recent_affect:
                try:
                    affect_abl = SYN.realize_np(NP(recent_affect["tag"], poss=2, case="ABL"))
                except Exception:
                    affect_abl = "duygu"
                text += (
                    f"Az önce kendi {affect_abl} söz etmen bu ihtimali bende daha "
                    "belirgin hâle getirdi; ama senin zihnindeki bağı bilmiyorum."
                )
            else:
                text += "Bu, kendi bellek işleyişimden yaptığım bir aktarım; insan için kesin bir açıklama değil."
            return self._record_open(
                agi, raw, text, act="reflection", kind="valence_contrast", topic=topic or phrase,
                evidence=["episodic_recall", "valence_contrast"], context_turns=context_turns,
                hypothesis="present_stimulus+episodic_association")

        if calming:
            surface = topic or phrase
            for word in re.findall(r"[\wçğıöşü]+", low):
                try:
                    analysis = agi.lang.morph.analyze_word(word, agi.lang.graph.lexicon)
                except Exception:
                    continue
                if str(analysis.get("stem") or "") == str(topic or ""):
                    surface = word
                    break
            focus = agi.lang.graph.surface_name(surface)
            additive = self._additive(agi, surface)
            text = (
                "Bir olasılık aklıma geliyor: düzenli tekrar, zihni her an yeni bir şey çözmek "
                "zorunda bırakmayabilir. Ben beklenmedik uyaranlarda daha çok şaşırıyorum; "
                f"{focus} {additive} öngörülebilir bir örüntü oluşturuyorsa, benzer bir rahatlama buradan "
                "doğabilir. Ama bunu öğrendiğim bir bilgi olarak değil, kendi işleyişimden yaptığım "
                "bir çıkarım olarak söylüyorum."
            )
            return self._record_open(
                agi, raw, text, act="reflection", kind="predictive_hypothesis", topic=topic or phrase,
                evidence=["mind:prediction_surprise"], context_turns=[],
                hypothesis="predictable_pattern->lower_surprise")

        concepts = [topic] if topic else []
        concepts += [c for c in self._content_concepts(agi, raw) if c not in concepts]
        edges = self._association_edges(agi, concepts, limit=3)
        if edges:
            nominals = [self._edge_nominal(agi, e) for e in edges[:2]]
            joined = " ve ".join(nominals)
            text = (
                f"{tr_capitalize(phrase)} deyince benim için öne çıkan {joined}. "
                "Bunu bitmiş bir tanım gibi değil, kavramın başka şeylerle kurduğu canlı bir bağ "
                "olarak görüyorum; hangi yanı öne çıkaracağım konuşmamıza göre değişiyor."
            )
            return self._record_open(
                agi, raw, text, act="reflection", kind="graph_reflection", topic=topic or phrase,
                evidence=[self._edge_key(e) for e in edges], context_turns=[])

        # Doğrulanmamış bilgi yok diye konuşmayı öldürme; mevcut konuşmadan
        # aktarımın sınırını açıkça taşıyan bir düşünce kur.
        previous = dict(self.last_open_trace or {})
        prev_topic = str(previous.get("topic") or "")
        if (prev_topic and tr_lower(prev_topic) != tr_lower(topic or phrase)
                and str(previous.get("kind") or "") == "fiction_causal_animation"):
            try:
                prev_gen = SYN.realize_np(NP(prev_topic, case="GEN"))
                phrase_acc = SYN.realize_np(NP(phrase, case="ACC"))
            except Exception:
                prev_gen = prev_topic
                phrase_acc = phrase
            additive = self._additive(agi, phrase)
            text = (
                f"{tr_capitalize(phrase)} hakkında öğrenilmiş bir tanımım yok; yine de aklıma bir "
                f"benzetme geliyor. Az önce {prev_gen} etkisini, kendisinden sonra da süren bir "
                "ilişki olarak kurmuştum. "
                f"{tr_capitalize(phrase_acc)} {additive} tek bir an yerine devam eden bir ilişki gibi düşünebilir "
                "miyim, bilmiyorum; şu an zihnimde bu olasılık var."
            )
            return self._record_open(
                agi, raw, text, act="reflection", kind="context_transfer", topic=topic or phrase,
                evidence=[], context_turns=[int(previous.get("dialogue_turn", -1))],
                source_topic=prev_topic)
        if prev_topic and tr_lower(prev_topic) != tr_lower(topic or phrase):
            text = (
                f"{tr_capitalize(phrase)} hakkında henüz öğrendiğim bir bilgi yok; ama aklıma onu "
                f"hemen bir tanıma sıkıştırmamak geliyor. Az önce {prev_topic} için kurduğum bağın "
                "buraya taşınıp taşınmayacağını düşünüyorum. Bu benim için bir cevap değil, iki "
                "kavramı nasıl ayıracağıma dair açık bir soru."
            )
            return self._record_open(
                agi, raw, text, act="reflection", kind="context_transfer", topic=topic or phrase,
                evidence=[], context_turns=[int(previous.get("dialogue_turn", -1))],
                source_topic=prev_topic)
        candidates = [r for r in self._dialogue_candidates(agi)
                      if tr_lower(r["topic"]) != tr_lower(topic or phrase)]
        if candidates:
            other = candidates[0]["topic"]
            text = (
                f"{tr_capitalize(phrase)} hakkında doğrulanmış bilgim yok; ama onu bu konuşmada "
                f"{other} ile yan yana getiriyorum. Bunun nedeni ikisini de senin gündeme getirmiş "
                "olman. Aralarında gerçek bir ilişki olduğunu söylemiyorum; işte tam bunu merak ediyorum."
            )
            return self._record_open(
                agi, raw, text, act="reflection", kind="dialogue_bridge", topic=topic or phrase,
                evidence=[], context_turns=list(candidates[0].get("turns") or []), source_topic=other)
        text = (
            f"{tr_capitalize(phrase)} hakkında doğrulanmış bilgim yok; bu yüzden onu tanımlamak yerine "
            "nasıl düşündüğümü söyleyebilirim. Şu an o benim için yalnız bir ad değil, hangi ilişkilerle "
            "anlam kazanacağını araştırdığım açık bir soru."
        )
        return self._record_open(
            agi, raw, text, act="reflection", kind="honest_open_concept", topic=topic or phrase,
            evidence=[], context_turns=[])

    def _association(self, agi, raw: str, topic: str) -> str:
        phrase = self._topic_phrase(agi, raw, topic) or topic
        concepts = self._content_concepts(agi, phrase or raw)
        if topic and topic not in concepts:
            concepts.append(topic)
        edges = self._association_edges(agi, concepts, limit=3)
        episodes = self._recent_episodes(agi)
        affect = self._affect_context(agi, episodes)
        if edges:
            nominals = [self._edge_nominal(agi, e) for e in edges[:2]]
            if len(nominals) == 1:
                joined = nominals[0]
            else:
                joined = " ile ".join(nominals)
            text = f"{tr_capitalize(phrase)} deyince aklımda {joined} yan yana geliyor. "
            if affect:
                additive = self._additive(agi, affect["tag"])
                text += (
                    f"Az önce anlattığın {affect['tag']} {additive} bu bağa eşlik edince, benim için "
                    "yalnız bir bilgi listesi olarak kalmıyor; ama senin hissettiğin şeyin aynısını "
                    "yaşadığımı söyleyemem."
                )
            else:
                text += (
                    "Bu çağrışımın duygusal yönü bende henüz net değil; şu an doğrulanmış bilgiyle "
                    "bu konuşmada yeni kurduğum bağı birbirinden ayırıyorum."
                )
            return self._record_open(
                agi, raw, text, act="association", kind="graph_episode_association", topic=phrase or topic,
                evidence=[self._edge_key(e) for e in edges],
                context_turns=[affect["turn_id"]] if affect else [])
        candidates = [r for r in self._dialogue_candidates(agi)
                      if tr_lower(r["topic"]) != tr_lower(topic or phrase)]
        if candidates:
            other = candidates[0]["topic"]
            text = (
                f"{tr_capitalize(phrase)} için bende önceden kurulmuş doğrulanmış bir çağrışım yok. "
                f"Şu an onu yalnız bu konuşmanın içinde {other} ile yan yana getiriyorum; bu yeni bir "
                "bağlantı denemesi, öğrenilmiş bir olgu değil."
            )
            return self._record_open(
                agi, raw, text, act="association", kind="novel_dialogue_association", topic=phrase or topic,
                evidence=[], context_turns=list(candidates[0].get("turns") or []), source_topic=other)
        text = (
            f"{tr_capitalize(phrase)} deyince bende doğrulanmış bir bağ canlanmıyor. Sana neyi "
            "çağrıştırdığını duymadan kendi adıma bir duygu uydurmak istemem."
        )
        return self._record_open(
            agi, raw, text, act="association", kind="unknown_association", topic=phrase or topic,
            evidence=[], context_turns=[])

    # ------------------------------------------------------------------ 2. KARAR: TUTUM
    def _attitude(self, agi, topic: str) -> str:
        """'Kediyi seviyor musun?' — tanıklık/deneyim ayrımı, gerekçe ve karşı soru."""
        ev = agi._affect_evidence(topic)
        items: list[str] = []
        if not ev["exp"]:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="gör", obj=NP(topic, case="ACC"), neg=True,
                tense="PAST", person="1SG")))
        if ev["word"]:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="uyandır", subj=NP(topic), loc=NP("ben", case="LOC"),
                obj=NP(ev["word"], case="NOM"), tense="PRES", person="3SG")))
        # Duruş: kanıt varsa yönüne göre, yoksa dürüst boşluk
        if ev["ev"] and abs(ev["val"]) > 0.15:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="sev", obj=NP(topic, case="ACC"),
                neg=bool(ev["val"] <= 0), tense="PRES", person="1SG")))
        else:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="bil", obj=NP("cevap", poss=3, case="ACC"), neg=True,
                tense="PRES", person="1SG")))
        if ev["ev"] and not ev["exp"]:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="anlat", subj=NP("sen"), dat=NP("ben", case="DAT"),
                obj=NP("bunu", case="NOM"), tense="PAST", person="2SG")))
        # KARŞI SORU: niyetini sorgular (her seferinde aynı biçimde değil)
        items.append(self._ask_intent(topic))
        self.last_reason = "tutum:" + topic
        self.last_deep_topic = topic
        return self._compose(agi, items, merge=True)

    def _justify(self, agi, topic: str) -> str:
        """'Neden kediyi sevdiğini söylüyorsun?' — iddiasını kanıtıyla savunur, kaynağı söyler."""
        ev = agi._affect_evidence(topic)
        items: list[str] = []
        if ev["word"]:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="uyandır", subj=NP(topic), loc=NP("ben", case="LOC"),
                obj=NP(ev["word"], case="NOM"), tense="PRES", person="3SG")))
        if ev["ev"] and abs(ev["val"]) > 0.15:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="sev", obj=NP(topic, case="ACC"),
                neg=bool(ev["val"] <= 0), advs=("bu yüzden",), tense="PRES", person="1SG")))
        if not ev["exp"]:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="gör", obj=NP(topic, case="ACC"), neg=True,
                tense="PAST", person="1SG")))
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="anlat", subj=NP("sen"), dat=NP("ben", case="DAT"),
                obj=NP("bunu", case="NOM"), tense="PAST", person="2SG")))
        elif not ev["word"]:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="bil", obj=NP("cevap", poss=3, case="ACC"), neg=True,
                tense="PRES", person="1SG")))
        items.append(self._ask_intent(topic))
        self.last_reason = "tutum:" + topic
        self.last_deep_topic = topic
        return self._compose(agi, items, merge=True)

    def _ask_intent(self, topic: str) -> str:
        choice = self.rng.randrange(3)
        if choice == 0:
            q = SYN.realize_verb(VerbSpec(verb_root="sor", subj=NP("sen"), wh_word="neden",
                                          tense="PRES", person="2SG"))
        elif choice == 1:
            q = SYN.realize_verb(VerbSpec(verb_root="merak et", subj=NP("sen"),
                                          obj=NP("bunu", case="NOM"), wh_word="neden",
                                          tense="PRES", person="2SG"))
        else:
            q = SYN.realize_verb(VerbSpec(verb_root="sev", subj=NP("sen"),
                                          obj=NP(topic, case="ACC"), question_yn=True,
                                          tense="PRES", person="2SG"))
        return q.rstrip(".?!") + "?"

    def _experience(self, agi, topic: str) -> str:
        """'Sen hiç kedi gördün mü?' — deneyim defterinden yanıt; bilgi ağı bunu yanıtlamaz."""
        ev = agi._affect_evidence(topic)
        items: list[str] = []
        if ev["exp"]:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="gör", obj=NP(topic, case="ACC"), advs=("evet",),
                tense="PAST", person="1SG")))
        else:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="gör", obj=NP(topic, case="ACC"), neg=True, advs=("hiç",),
                tense="PAST", person="1SG")))
            if ev["ev"]:
                items.append(SYN.realize_verb(VerbSpec(
                    verb_root="anlat", subj=NP("sen"), dat=NP("ben", case="DAT"),
                    obj=NP("bunu", case="NOM"), tense="PAST", person="2SG")))
        if ev["word"]:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="uyandır", subj=NP(topic), loc=NP("ben", case="LOC"),
                obj=NP(ev["word"], case="NOM"), tense="PRES", person="3SG")))
        items.append(self._ask_intent(topic))
        self.last_reason = "tutum:" + topic
        self.last_deep_topic = topic
        return self._compose(agi, items, merge=True)

    def _narrate(self, agi, topic: str, raw: str) -> str | None:
        """'Bana kedilerden bahset.' — hazır anlatı değil, kendi ağını gezer."""
        try:
            txt = agi._narrate_from_knowledge(topic)
        except Exception:
            txt = ""
        return txt or None

    def _self_eval(self, agi, raw: str) -> str:
        """'Sence zeki misin?' — ölçtüğü kendi kapasitesinden alçakgönüllü yanıt."""
        low = tr_lower(raw)
        adj = ""
        for a in ("zeki", "akıllı", "başarılı", "yetenekli", "güçlü", "hızlı", "iyi"):
            if a in low:
                adj = a
                break
        items: list[str] = []
        if adj:
            items.append(SYN.realize_pred(PredSpec(pred_root=adj, person="1SG",
                                                   advs=("sanırım",))))
        solved = int(getattr(agi.self_model, "problems_solved", 0))
        if solved > 0:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="çöz", obj=NP("problem", plural=True, case="ACC"),
                advs=("kendi başıma",), tense="PAST", person="1SG")))
        items.append(SYN.realize_verb(VerbSpec(verb_root="öğren", advs=("hâlâ",),
                                               tense="PRES", person="1SG")))
        q = SYN.realize_verb(VerbSpec(verb_root="bul", subj=NP("sen"), obj=NP("ben", case="ACC"),
                                      wh_word="nasıl", tense="PRES", person="2SG"))
        items.append(q.rstrip(".?!") + "?")
        self.last_reason = "kendini değerlendirme"
        return self._compose(agi, items)

    def _learn_why(self, agi, raw: str) -> str:
        """'Neden öğreniyorsun?' — amaç, kişi çevrilmeden öz-güdüden kurulur.

        Spekülatif kavram sorgusu fiil yüzeyini konu sanıp soruyu kullanıcıya
        geri çeviriyordu. Burada yüklem Nora'nın birinci-şahıs öğrenmesidir;
        gerekçe ise bilgi boşluğunu kapatma isteği ve ölçülen merak eğilimidir.
        """
        want = SYN.nominalize_inf(
            "kapat", short_ma=False, case="NOM",
            obj=NP("boşluk", mods=("bilgi",), poss=1, plural=True, case="ACC"),
        )
        reason = SYN.nominalize_dik(
            "iste", poss=1, case="NOM", postpos="için", obj=want,
        )
        items = [SYN.realize_verb(VerbSpec(
            verb_root="öğren", pre_clause=reason, tense="PRES", person="1SG",
        ))]
        try:
            curiosity = float(agi.self_model.traits.get("meraklılık", 0.0))
        except Exception:
            curiosity = 0.0
        if curiosity > 0.0:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="merak et",
                obj=NP("ilişki", mods=("yeni",), plural=True, case="ACC"),
                tense="PRES", person="1SG",
            )))
        self.last_reason = "öğrenme amacı"
        return self._compose(agi, items)

    def _learn_how(self, agi) -> str:
        """'Nasıl öğreniyorsun?' — kendi öğrenme mekanizmasını olduğu gibi anlatır."""
        items = [SYN.realize_verb(VerbSpec(
            verb_root="bağla", obj=NP("bağ", poss=1, plural=True, mods=("yeni",), case="ACC"),
            advs=("tek geçişte",), tense="PRES", person="1SG"))]
        items.append(SYN.realize_verb(VerbSpec(
            verb_root="öğren", advs=("seninle konuştukça",), tense="PRES", person="1SG")))
        q = SYN.realize_verb(VerbSpec(
            verb_root="öğret", subj=NP("sen"), dat=NP("ben", case="DAT"),
            obj=NP("ne", case="ACC"), question_yn=True, tense="AOR", person="2SG"))
        items.append(q.rstrip(".?!") + "?")
        self.last_reason = "öğrenme"
        return self._compose(agi, items, merge=True)

    def _conditional(self, agi, raw: str) -> str:
        """'Hava yağmurlu olunca ne yapıyorsun?' — o durumda deneyimi yoksa uydurmaz."""
        low = tr_lower(raw)
        cond = ""
        for c in _COND:
            if c in low:
                cond = c
                break
        topic = self._topic(agi, raw) or "yağmur"
        items: list[str] = []
        ev = agi._affect_evidence(topic)
        if ev["word"]:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="uyandır", subj=NP(topic), loc=NP("ben", case="LOC"),
                obj=NP(ev["word"], case="NOM"), tense="PRES", person="3SG")))
        items.append(SYN.realize_verb(VerbSpec(
            verb_root="ol", subj=NP("deneyim", poss=1), neg=True,
            advs=(SYN.realize_np(NP(topic, case="LOC")),), tense="PRES", person="3SG")))
        q = SYN.realize_verb(VerbSpec(verb_root="yap", subj=NP("sen"),
                                      loc=NP(topic, case="LOC"), wh_word="ne",
                                      tense="PRES", person="2SG"))
        items.append(q.rstrip(".?!") + "?")
        self.last_reason = "durum:" + topic
        return self._compose(agi, items)

    # ---------------------------------------------------------- öğretici çalışma oturumu
    def _tutorial_topics(self, agi, raw: str) -> list[str]:
        """Sözdeki kavramları O(sözcük) graf erişimiyle bul; üst-dil sözcükleri konu değildir."""
        low = tr_lower(raw).replace("zeka", "zekâ")
        words = re.findall(r"[\wçğıöşü]+", low)
        graph = agi.lang.graph
        morph = agi.lang.morph
        stems = []
        for word in words:
            try:
                stems.append(str(morph.analyze_word(word, graph.lexicon).get("stem") or word))
            except Exception:
                stems.append(word)
        meta = {
            "madde", "örnek", "cevap", "konu", "bilgi", "şey", "adım", "ilişki",
            "özet", "soru", "fark", "aynı", "kesin", "önce", "sonra", "bugün",
        }
        found: list[str] = []

        def add(candidate: str):
            cand = tr_lower(candidate).strip()
            if not cand or cand in meta:
                return
            if cand == "program" and graph._out_edges.get("yazılım"):
                cand = "yazılım"
            if cand in found:
                return
            try:
                known = bool(graph._out_edges.get(cand) or graph.query(cand, None))
            except Exception:
                known = bool(graph._out_edges.get(cand))
            if known:
                found.append(cand)

        for word, stem in zip(words, stems):
            if word not in meta:
                add(stem)
                if word in graph._out_edges:
                    add(word)
        for size in (2, 3):
            for i in range(len(words) - size + 1):
                add(" ".join(words[i:i + size]))
                add(" ".join(stems[i:i + size]))
                last = words[i + size - 1]
                # Birleşik kavramın son üyesi ek almış olabilir: "yapay zekâyla".
                for cut in range(len(last) - 1, 1, -1):
                    candidate = " ".join([*words[i:i + size - 1], last[:cut]])
                    if candidate in graph._out_edges:
                        add(candidate)
                        break

        found = [
            cand for cand in found
            if not any(
                " " in other and cand in other.split()[:-1]
                for other in found if other != cand
            )
        ]

        def order_key(candidate: str):
            first = candidate.split()[0]
            match = re.search(r"(?<!\w)" + re.escape(first), low)
            pos = match.start() if match else low.find(first)
            return (pos if pos >= 0 else 10_000, -len(candidate))

        found.sort(key=order_key)
        return found[:4]

    def _tutorial_frame(self, agi, name: str, **slots) -> str:
        """Genel öğretici çerçeve veri alanında yarışır; içerik slotları grafikten gelir."""
        from nora_anlatim import _frame
        return str(_frame(agi, "tutorial." + name, slots) or "")

    def _tutorial_facts(self, agi, topic: str, direct_only: bool = False) -> list[tuple[str, str, str]]:
        """Tekrarlı yüzeyleri ayıklanmış, kanıt zinciri korunan kazanım anahtarları."""
        graph = agi.lang.graph
        if direct_only:
            rows = [(edge, []) for edge in graph._out_edges.get(topic, ())]
        else:
            rows = list(graph.query(topic, None))
        priority = {"ISA": 0, "USED_FOR": 1, "CAN_DO": 2, "CAUSES": 3,
                    "HAS_PROP": 4, "HAS_PART": 5, "LOCATED_AT": 6}
        rows = [(edge, path) for edge, path in rows if float(edge.weight) > 0.25]
        rows.sort(key=lambda item: (
            len(item[1]), priority.get(item[0].rel, 8), -len(item[0].obj), item[0].obj,
        ))
        out: list[tuple[str, str, str]] = []
        signatures: list[tuple[str, str, set[str]]] = []
        for edge, _path in rows:
            key = (edge.subj, edge.rel, edge.obj)
            if key in out:
                continue
            tokens = self._tutorial_keywords(agi, edge.obj)
            duplicate = False
            if edge.rel != "ISA":
                for subj, rel, old_tokens in signatures:
                    if subj != edge.subj or rel != edge.rel or not tokens or not old_tokens:
                        continue
                    overlap = len(tokens & old_tokens) / max(1, min(len(tokens), len(old_tokens)))
                    if overlap >= 0.60:
                        duplicate = True
                        break
            if duplicate:
                continue
            out.append(key)
            signatures.append((edge.subj, edge.rel, tokens))
        return out

    def _tutorial_fact_text(self, agi, fact: tuple[str, str, str]) -> str:
        subj, rel, obj = fact
        if rel == "USED_FOR":
            obj_text = obj
            parts = obj.split()
            if (len(parts) > 1 and " ve " not in obj
                    and parts[0].endswith(("i", "ı", "u", "ü"))
                    and parts[-1].endswith(("me", "ma"))):
                root = parts[-1][:-2]
                suffix = "mak" if agi.lang.morph.vowel_e_a(root) == "a" else "mek"
                obj_text = " ".join(parts[:-1] + [root + suffix])
            return f"{subj} {obj_text} için kullanılır"
        if rel == "CAN_DO" and obj.split() and obj.split()[-1].endswith(("mak", "mek")):
            from nora_broca import TurkishSyntaxEngine as Syntax
            parts = obj.split()
            root = parts[-1][:-3]
            if root not in agi.lang.morph.KNOWN_VERB_ROOTS:
                root = agi.lang.morph._unsoften(root)
            if root in agi.lang.morph.KNOWN_VERB_ROOTS:
                form = Syntax._conjugate_full_verb(root, "ABIL_AOR", False, "3SG", False)
                return f"{subj} {' '.join(parts[:-1] + [form])}".strip()
        from nora_lang import Proposition
        return str(agi.lang.broca.realize_prop(
            Proposition(subj=subj, rel=rel, obj=obj)
        )).strip().rstrip(".")

    def _tutorial_neural_fact(self, agi, fact: tuple[str, str, str]) -> str:
        """MoE token üretir; grafik kapsamı ve ilişki işareti doğrulanmayan aday kullanılmaz."""
        session = self.tutorial
        cache = session.setdefault("neural_cache", {})
        stats = session.setdefault("neural_stats", {
            "generated": 0, "accepted": 0, "rejected": 0, "cache_hits": 0,
        })
        if fact in cache:
            stats["cache_hits"] += 1
            text = str(cache[fact] or "")
            if text:
                session["turn_neural"] = int(session.get("turn_neural", 0)) + 1
            return text
        subj, rel, obj = fact
        organ = getattr(getattr(agi, "life", None), "voice", None)
        organ = getattr(organ, "organ", None)
        if rel not in {"ISA", "USED_FOR"} or not callable(getattr(organ, "generate", None)):
            cache[fact] = ""
            return ""
        from nora_voice import state_vector
        extra = ["bir"] if rel == "ISA" else ["için", "kullanılır", "ve"]
        anchors = [subj, obj]
        if rel == "USED_FOR":
            canonical = self._tutorial_fact_text(agi, fact)
            core = canonical[len(subj):].strip()
            tail = "için kullanılır"
            if core.endswith(tail):
                core = core[:-len(tail)].strip()
            parts = []
            for word in core.split():
                anchor = word
                try:
                    info = agi.lang.morph.analyze_word(word, agi.lang.graph.lexicon)
                    stem = str(info.get("stem") or word)
                    if info.get("case") == "ACC" and len(stem) >= 4:
                        anchor = stem
                except Exception:
                    pass
                if parts and parts[-1] == anchor:
                    parts[-1] = parts[-1] + " " + anchor
                else:
                    parts.append(anchor)
            anchors = [subj, *parts]
        try:
            sequences = organ.generate(
                state_vector(agi), anchors, extra_allowed=extra,
                temperature=0.65, top_k=24, n=8, max_len=20)
        except Exception:
            sequences = []
        allowed = self._tutorial_keywords(agi, f"{subj} {obj} " + " ".join(extra))
        subject_keys = self._tutorial_keywords(agi, subj)
        object_keys = self._tutorial_keywords(agi, obj)
        accepted = ""
        for sequence in sequences:
            stats["generated"] += 1
            try:
                text = str(organ.detok(sequence) or "").strip()
            except Exception:
                text = ""
            low = tr_lower(text)
            keys = self._tutorial_keywords(agi, text)
            subject_surface = tr_lower(subj) + " "
            predicate = low[len(subject_surface):] if low.startswith(subject_surface) else ""
            predicate_keys = self._tutorial_keywords(agi, predicate)
            coverage = len(object_keys & predicate_keys) / max(1, len(object_keys))
            fragments = re.findall(r"[^\W\d_]+", low, flags=re.UNICODE)
            valid = (bool(text) and low.startswith(subject_surface)
                     and not any(mark in text for mark in "?!")
                     and all(len(fragment) >= 2 for fragment in fragments)
                     and subject_keys <= keys and coverage >= 0.50 and keys <= allowed)
            if rel == "USED_FOR":
                valid = valid and "için kullanılır" in low
            else:
                valid = valid and "kullanılır" not in low
                copula = " bir " in f" {low} "
                if not copula:
                    for word in re.findall(r"[\wçğıöşü]+", low):
                        try:
                            copula = bool(agi.lang.morph.analyze_word(
                                word, agi.lang.graph.lexicon).get("copula"))
                        except Exception:
                            copula = False
                        if copula:
                            break
                canonical = tr_lower(self._tutorial_fact_text(agi, fact))
                expected_last = re.findall(r"[^\W\d_]+", canonical, flags=re.UNICODE)
                valid = (valid and copula and bool(fragments) and bool(expected_last)
                         and fragments[-1] == expected_last[-1])
            if valid:
                accepted = text
                stats["accepted"] += 1
                break
            stats["rejected"] += 1
        cache[fact] = accepted
        if accepted:
            session["turn_neural"] = int(session.get("turn_neural", 0)) + 1
        return accepted

    def _tutorial_render(self, agi, facts: list[tuple[str, str, str]], limit: int = 2) -> list[str]:
        """Aynı özneli iki kazanımı doğal bir koşacla birleştir; madde madde ezber dökme."""
        texts: list[str] = []
        previous_subject = None
        for fact in facts[:limit]:
            subj = fact[0]
            new_text = self._tutorial_neural_fact(agi, fact) or self._tutorial_fact_text(agi, fact)
            if texts and previous_subject == subj and new_text.startswith(subj + " "):
                addition = new_text[len(subj) + 1:].strip().rstrip(".")
                texts[-1] = texts[-1].rstrip(".") + " ve " + addition
            else:
                texts.append(new_text)
            previous_subject = subj
        return [tr_capitalize(text.strip().rstrip(".")) + "." for text in texts]

    def _tutorial_finish(self, raw: str, text: str, intent: str,
                         facts: list[tuple[str, str, str]], topic: str) -> str:
        session = self.tutorial
        covered = session.setdefault("covered", [])
        for fact in facts:
            if fact not in covered:
                covered.append(fact)
        session["last_facts"] = list(facts)
        session["last_intent"] = intent
        session["last_answer"] = text
        session["turns"] = int(session.get("turns", 0)) + 1
        session.setdefault("history", []).append({
            "student": raw, "nora": text, "intent": intent, "topic": topic,
        })
        self.last_tutorial_trace = {
            "intent": intent, "topic": topic, "session_turn": session["turns"],
            "facts": [list(fact) for fact in facts],
            "quiz_pending": bool(session.get("awaiting_quiz")),
            "neural_used": int(session.get("turn_neural", 0)),
            "neural": dict(session.get("neural_stats", {})),
        }
        return text

    def _tutorial_compare(self, agi, left: str, right: str, same_question: bool) -> tuple[str, list]:
        left_facts = self._tutorial_facts(agi, left, direct_only=True)
        right_facts = self._tutorial_facts(agi, right, direct_only=True)
        link = None
        if (left, "ISA", right) in left_facts:
            link = (left, "ISA", right)
        elif (right, "ISA", left) in right_facts:
            link = (right, "ISA", left)
        used: list[tuple[str, str, str]] = []
        sentences: list[str] = []
        if same_question or (link is None and left_facts and right_facts):
            sentences.append(self._tutorial_frame(
                agi, "compare_same", left_cap=tr_capitalize(left), right=right))
        if link is not None:
            used.append(link)
            sentences.extend(self._tutorial_render(agi, [link], 1))
        special = []
        left_all = set(self._tutorial_facts(agi, left))
        right_all = set(self._tutorial_facts(agi, right))
        left_shared = {(fact[1], fact[2]) for fact in left_all}
        right_shared = {(fact[1], fact[2]) for fact in right_all}
        specific, general = left, right
        specific_facts = left_facts
        other_shared = right_shared
        if link is not None and link[0] == right:
            specific, general = right, left
            specific_facts = right_facts
            other_shared = left_shared
        for fact in specific_facts:
            if fact[1] != "ISA" and (fact[1], fact[2]) not in other_shared and fact not in used:
                special.append(fact)
        if special:
            used.append(special[0])
            sentences.extend(self._tutorial_render(agi, [special[0]], 1))
            if link is not None and special[0][1] == "CAN_DO":
                if "öğren" in special[0][2]:
                    sentences.append(self._tutorial_frame(
                        agi, "scope_learning", general=general))
                else:
                    sentences.append(self._tutorial_frame(
                        agi, "scope_other", general_cap=tr_capitalize(general)))
        for topic, facts in ((right, right_facts), (left, left_facts)):
            if len(sentences) >= 3:
                break
            other_shared = left_shared if topic == right else right_shared
            candidate = next((
                fact for fact in facts
                if fact not in used and (link is None or (fact[1], fact[2]) not in other_shared)
            ), None)
            if candidate is not None:
                used.append(candidate)
                sentences.extend(self._tutorial_render(agi, [candidate], 1))
        if not sentences:
            sentences.append(self._tutorial_frame(
                agi, "unknown_compare", left_cap=tr_capitalize(left), right=right))
        if link is not None and not special and not same_question:
            sentences.append(self._tutorial_frame(agi, "unknown_detail"))
        return " ".join(sentences[:4]), used

    def _tutorial_example(self, agi, topic: str, raw: str = "") -> tuple[str, list]:
        if topic == "algoritma":
            numbers = [int(value) for value in re.findall(r"\d+", raw)]
            left, right = numbers[:2] if len(numbers) >= 2 else (2, 3)
            result = sum((left, right))
            facts = self._tutorial_facts(agi, topic, direct_only=True)[:1]
            text = self._tutorial_frame(
                agi, "example_arithmetic", left=left, right=right, result=result)
            return text, facts
        learned = any(edge.source == "öğretmen" for edge in agi.lang.graph.edges)
        if topic in {"yapay zekâ", "yazılım"} and learned:
            left, right = 2, 3
            result = sum((left, right))
            facts = self._tutorial_facts(agi, topic, direct_only=True)[:2]
            return self._tutorial_frame(
                agi, "example_self", left=left, right=right, result=result), facts
        incoming = [edge for edge in agi.lang.graph._in_edges.get(topic, ()) if edge.rel == "ISA"]
        if incoming:
            fact = (incoming[0].subj, incoming[0].rel, incoming[0].obj)
            return " ".join(self._tutorial_render(agi, [fact], 1)), [fact]
        return self._tutorial_frame(
            agi, "example_unknown", topic_cap=tr_capitalize(topic)), []

    def _tutorial_keywords(self, agi, text: str) -> set[str]:
        graph = agi.lang.graph
        morph = agi.lang.morph
        stop = {"için", "ile", "ve", "bir", "bu", "ne", "ama", "olan", "küresel"}
        out = set()
        for word in re.findall(r"[\wçğıöşü]+", tr_lower(text)):
            try:
                info = morph.analyze_word(word, graph.lexicon)
                stem = str(info.get("stem") or word)
            except Exception:
                info = {}
                stem = word
            root = None
            if info.get("verbal_noun") == "MAK" and stem.endswith(("mek", "mak")):
                root = stem[:-3]
            elif info.get("verbal_noun") == "MA" and stem.endswith(("me", "ma")):
                root = stem[:-2]
            elif stem.endswith(("me", "ma")) and word != stem:
                candidate = stem[:-2]
                if candidate in graph.lexicon:
                    root = candidate
            else:
                nominal = stem[:-1] if stem.endswith(("ımı", "imi", "umu", "ümü")) else stem
                if nominal.endswith(("ım", "im", "um", "üm")):
                    candidate = nominal[:-2]
                    if candidate in graph.lexicon:
                        root = candidate
            if root and len(root) >= 2:
                stem = root
            if len(stem) >= 3 and stem not in stop:
                out.add(stem)
        return out

    def _tutorial_evaluate(self, agi, raw: str, summary: bool = False) -> tuple[str, list]:
        session = self.tutorial
        if summary:
            mentioned = []
            for clause in re.split(r"[,;\n]+", raw):
                clause_topics = self._tutorial_topics(agi, clause)
                if clause_topics and clause_topics[0] not in mentioned:
                    mentioned.append(clause_topics[0])
            expected = []
            for fact in session.get("covered", []):
                if fact[0] not in mentioned or fact[1] == "ISA":
                    continue
                if any(old[0] == fact[0] for old in expected):
                    continue
                expected.append(fact)
            if not expected:
                expected = [fact for fact in session.get("covered", []) if fact[1] != "ISA"][:3]
        else:
            expected = list(session.get("quiz", []) if session.get("awaiting_quiz")
                            else session.get("covered", [])[:3])
        response_keys = self._tutorial_keywords(agi, raw)
        feedback: list[str] = []
        full = 0
        partial = 0
        for fact in expected:
            wanted = self._tutorial_keywords(agi, fact[2])
            overlap = wanted & response_keys
            if summary:
                is_full = bool(overlap)
            else:
                threshold = max(1, (2 * len(wanted) + 2) // 3) if wanted else 1
                is_full = bool(wanted) and len(overlap) >= threshold
            if is_full:
                full += 1
                if not summary:
                    feedback.append(self._tutorial_frame(
                        agi, "eval_item_good", topic_cap=tr_capitalize(fact[0])))
            elif overlap:
                partial += 1
                feedback.append(self._tutorial_frame(
                    agi, "eval_item_partial", topic_cap=tr_capitalize(fact[0]),
                    fact=self._tutorial_fact_text(agi, fact)))
            else:
                feedback.append(self._tutorial_frame(
                    agi, "eval_item_missing", topic_cap=tr_capitalize(fact[0]),
                    fact=self._tutorial_fact_text(agi, fact)))
        all_good = bool(expected) and full == len(expected)
        if summary and all_good:
            head = self._tutorial_frame(agi, "eval_summary_good")
            feedback = []
        elif all_good:
            head = self._tutorial_frame(agi, "eval_all_good")
        elif full or partial:
            head = self._tutorial_frame(agi, "eval_summary_partial" if summary else "eval_partial")
        else:
            head = self._tutorial_frame(agi, "eval_bad")
        session["awaiting_quiz"] = False
        session["quiz"] = []
        session["student_eval"] = "good" if all_good else "partial"
        return " ".join([head, *feedback]), expected

    def _tutorial_summary(self, agi, count: int) -> tuple[str, list]:
        covered = list(self.tutorial.get("covered", []))
        selected = []
        # Önce farklı kavramların işlevlerini ver; aynı öznenin ikinci kaydıyla özeti doldurma.
        for rels in ({"USED_FOR", "CAN_DO"}, {"ISA"}, set()):
            for fact in covered:
                if fact in selected or (rels and fact[1] not in rels):
                    continue
                if len(selected) < count and not any(old[0] == fact[0] for old in selected):
                    selected.append(fact)
            if len(selected) >= count:
                break
        if len(selected) < count:
            for fact in covered:
                if fact not in selected:
                    selected.append(fact)
                    if len(selected) >= count:
                        break
        selected = selected[:max(1, count)]
        lines = [
            f"{index}. {tr_capitalize(self._tutorial_fact_text(agi, fact))}."
            for index, fact in enumerate(selected, 1)
        ]
        if not lines:
            return self._tutorial_frame(agi, "summary_empty"), []
        return "\n".join(lines), selected

    def handle_tutorial(self, agi, raw: str, speaker: str = "insan") -> str | None:
        """Aynı zihnin sürekli öğretmen bağlamı; bilgi grafikten, değerlendirme kazanımdan."""
        low = tr_lower(raw)
        topics = self._tutorial_topics(agi, raw)
        teach_request = any(cue in low for cue in (
            "öğret", "konu çalış", "ders çalış", "bilgimi ölç", "beni sına",
        ))
        session = self.tutorial
        if not session.get("active"):
            if not teach_request or not topics:
                return None
            self.tutorial = session = {
                "active": True, "speaker": speaker, "topic": topics[0],
                "focus": topics[0], "covered": [], "history": [], "turns": 0,
                "last_facts": [], "quiz": [], "awaiting_quiz": False, "asked_quiz": [],
            }
        elif session.get("speaker") != speaker:
            return None

        session["turn_neural"] = 0
        previous_focus = str(session.get("focus") or session.get("topic") or "")
        if topics:
            session["focus"] = topics[0]
        focus = str(session.get("focus") or session.get("topic") or "")
        intent = ""
        if any(cue in low for cue in ("toparla", "çalışmayı bitir", "dersi bitir")):
            intent = "close"
        elif "soru sor" in low and any(cue in low for cue in ("ölç", "bilgimi", "sına")):
            intent = "quiz"
        elif "cevabım" in low or (session.get("awaiting_quiz") and "doğru mu" in low):
            intent = "evaluate"
        elif "özetleyeyim" in low or ("eksik var mı" in low and "özet" in low):
            intent = "check_summary"
        elif "özet" in low or ("maddede" in low and "öğrendik" in low):
            intent = "summary"
        elif (("bence" in low or "sence" in low) and "aynı" in low and len(topics) >= 2):
            intent = "compare"
        elif "tutarlı mı" in low or "çelişiyor mu" in low:
            intent = "consistency"
        elif any(cue in low for cue in ("kesin bil", "kesin olarak", "emin ol", "kanıtın ne")):
            intent = "certainty"
        elif "örnek" in low or "örneğ" in low:
            intent = "example"
        elif any(cue in low for cue in ("aynı şey", "fark", "karşılaştır", "ilişki")):
            intent = "compare"
        elif any(cue in low for cue in ("neresinde", "nasıl kullan", "neden gerek", "gerekir", "ne için")):
            intent = "relate"
        elif teach_request or topics or any(cue in low for cue in ("bunu", "bu cevabın", "öğrendik")):
            intent = "explain"

        if not intent:
            session["focus"] = previous_focus
            return None
        if intent in {"example", "certainty"} and not topics:
            focus = previous_focus or focus
            session["focus"] = focus

        facts: list[tuple[str, str, str]] = []
        if intent == "close":
            topic_names = list(dict.fromkeys(fact[0] for fact in session.get("covered", [])))[:4]
            topics_text = ", ".join(topic_names) if topic_names else (session.get("topic") or "bu konu")
            evaluation = (self._tutorial_frame(agi, "close_evaluation")
                          if session.get("student_eval") == "good" else "")
            text = self._tutorial_frame(
                agi, "close", topics=topics_text, evaluation=evaluation)
            session["active"] = False
            return self._tutorial_finish(raw, text, intent, facts, focus)

        if intent == "quiz":
            candidates = []
            asked = set(session.get("asked_quiz", []))
            pool = [fact for fact in session.get("covered", []) if fact not in asked]
            if len({fact[0] for fact in pool if fact[1] in {"USED_FOR", "CAN_DO", "ISA"}}) < 2:
                pool = list(session.get("covered", []))
            for rels in ({"USED_FOR", "CAN_DO"}, {"ISA"}):
                for fact in reversed(pool):
                    if fact[1] not in rels or any(old[0] == fact[0] for old in candidates):
                        continue
                    candidates.append(fact)
                    if len(candidates) == 2:
                        break
                if len(candidates) == 2:
                    break
            candidates.reverse()
            session["quiz"] = candidates
            for fact in candidates:
                if fact not in asked:
                    session.setdefault("asked_quiz", []).append(fact)
                    asked.add(fact)
            session["awaiting_quiz"] = bool(candidates)
            questions = []
            for fact in candidates:
                if fact[1] == "USED_FOR":
                    questions.append(self._tutorial_frame(
                        agi, "quiz_used_for", topic_cap=tr_capitalize(fact[0])))
                elif fact[1] == "CAN_DO":
                    questions.append(self._tutorial_frame(
                        agi, "quiz_can_do", topic_cap=tr_capitalize(fact[0])))
                else:
                    questions.append(self._tutorial_frame(
                        agi, "quiz_isa", topic_cap=tr_capitalize(fact[0])))
            if not questions:
                text = self._tutorial_frame(agi, "quiz_empty")
            else:
                text = self._tutorial_frame(agi, "quiz_head") + "\n" + "\n".join(
                    f"{index}. {question}" for index, question in enumerate(questions, 1)
                )
            return self._tutorial_finish(raw, text, intent, candidates, focus)

        if intent in {"evaluate", "check_summary"}:
            text, facts = self._tutorial_evaluate(agi, raw, summary=intent == "check_summary")
            return self._tutorial_finish(raw, text, intent, facts, focus)

        if intent == "summary":
            number_words = {"bir": 1, "iki": 2, "üç": 3, "dört": 4, "beş": 5}
            count = next((value for word, value in number_words.items() if word in low.split()), 3)
            text, facts = self._tutorial_summary(agi, count)
            return self._tutorial_finish(raw, text, intent, facts, focus)

        if intent == "consistency":
            related = [fact for fact in session.get("last_facts", [])
                       if not topics or fact[0] in topics]
            if not related:
                related = [fact for fact in session.get("covered", []) if not topics or fact[0] in topics]
            if related:
                fact = related[-1]
                facts = [fact]
                text = (
                    self._tutorial_frame(agi, "consistency_yes") + " "
                    + self._tutorial_render(agi, facts, 1)[0] + " "
                    + self._tutorial_frame(agi, "consistency_tail")
                )
            else:
                text = self._tutorial_frame(agi, "consistency_unknown")
            return self._tutorial_finish(raw, text, intent, facts, focus)

        if intent == "certainty":
            if not topics and session.get("last_facts"):
                facts = list(session.get("last_facts", []))[:2]
            else:
                facts = self._tutorial_facts(agi, focus, direct_only=True)[:1]
            if facts:
                scope = "kayıtla" if len(facts) == 1 else "kayıtlarla"
                text = (
                    self._tutorial_frame(agi, "certainty_scope", scope=scope) + " "
                    + " ".join(self._tutorial_render(agi, facts, len(facts))) + " "
                    + self._tutorial_frame(agi, "certainty_tail")
                )
            else:
                text = self._tutorial_frame(
                    agi, "certainty_unknown", topic_cap=tr_capitalize(focus))
            return self._tutorial_finish(raw, text, intent, facts, focus)

        if intent == "example":
            text, facts = self._tutorial_example(agi, focus, raw)
            return self._tutorial_finish(raw, text, intent, facts, focus)

        if intent == "compare":
            left = topics[0] if topics else focus
            right = topics[1] if len(topics) > 1 else str(previous_focus or session.get("topic") or "")
            if right and left and right != left:
                text, facts = self._tutorial_compare(agi, left, right, "aynı şey" in low)
                session["focus"] = left
                return self._tutorial_finish(raw, text, intent, facts, left)
            intent = "explain"

        if intent == "relate":
            selected_topics = list(topics or [focus])
            anchor = str(session.get("topic") or "")
            if (len(selected_topics) == 1 and anchor and anchor not in selected_topics
                    and any(cue in low for cue in ("bunun", "bu sistemin", "bu yapının"))):
                selected_topics.append(anchor)
            covered_subjects = {fact[0] for fact in session.get("covered", [])}
            for topic in selected_topics[:2]:
                available = self._tutorial_facts(agi, topic)
                if topic in covered_subjects:
                    preferred = [fact for fact in available if fact[1] in {"USED_FOR", "CAN_DO"}]
                    chosen = preferred[:1] or available[:1]
                else:
                    chosen = available[:2]
                for fact in chosen:
                    if fact not in facts:
                        facts.append(fact)
            if "nasıl" in low and len(selected_topics) > 1:
                bridges = [
                    edge for edge in agi.lang.graph.edges
                    if edge.rel in {"CAN_DO", "USED_FOR"}
                    and any(topic in edge.obj for topic in selected_topics)
                    and edge.subj not in selected_topics
                ]
                bridges.sort(key=lambda edge: edge.subj not in covered_subjects)
                if bridges:
                    edge = bridges[0]
                    key = (edge.subj, edge.rel, edge.obj)
                    if key not in facts:
                        facts.append(key)
            if "öğren" in low:
                learners = [edge for edge in agi.lang.graph.edges
                            if edge.rel == "CAN_DO" and "öğren" in edge.obj
                            and edge.subj in covered_subjects]
                learners.sort(key=lambda edge: (
                    edge.subj not in covered_subjects,
                    edge.subj == "insan" and bool(covered_subjects - {"insan"}),
                ))
                if learners:
                    edge = learners[0]
                    key = (edge.subj, edge.rel, edge.obj)
                    if key not in facts:
                        facts.append(key)
            if facts:
                if "öğren" in low:
                    prefix = self._tutorial_frame(agi, "relate_learning") + " "
                elif len(selected_topics) > 1 and session.get("last_intent") == "compare":
                    prefix = self._tutorial_frame(agi, "relate_after_compare") + " "
                elif len(selected_topics) > 1:
                    prefix = self._tutorial_frame(
                        agi, "relate_pair", left_cap=tr_capitalize(selected_topics[0]),
                        right=selected_topics[1]) + " "
                else:
                    prefix = ""
                text = prefix + " ".join(self._tutorial_render(agi, facts, 4))
                if "öğren" in low and any(fact[0] == "veri" for fact in facts):
                    text += " " + self._tutorial_frame(agi, "learning_data_limit")
            else:
                text = self._tutorial_frame(
                    agi, "relate_unknown", topic_cap=tr_capitalize(focus))
            return self._tutorial_finish(raw, text, intent, facts, focus)

        available = self._tutorial_facts(agi, focus)
        limitation = ""
        if "ne yapar" in low:
            abilities = [fact for fact in available if fact[1] == "CAN_DO"]
            facts = abilities[:2] if abilities else available[:1]
            if not abilities and facts:
                limitation = " " + self._tutorial_frame(agi, "ability_unknown")
        else:
            facts = available[:2]
        if facts:
            prefix = (self._tutorial_frame(agi, "explain_start") + " ") if teach_request else ""
            text = prefix + " ".join(self._tutorial_render(agi, facts, 2)) + limitation
        else:
            text = self._tutorial_frame(
                agi, "explain_unknown", topic_cap=tr_capitalize(focus))
        return self._tutorial_finish(raw, text, "explain", facts, focus)

    # ------------------------------------------------------------------ 3. TEKRAR
    def _repeat(self, agi, raw: str, prev: dict, act: str) -> str:
        """Aynı soru ikinci kez: tekrarı fark eder, nedenini tartar, BAŞKA bir açıdan cevap verir."""
        items = [
            SYN.realize_verb(VerbSpec(verb_root="sor", subj=NP("sen"),
                                      obj=NP("soru", mods=("aynı",), case="ACC"),
                                      advs=("ikinci kez",), tense="PAST", person="2SG")),
            SYN.realize_verb(VerbSpec(verb_root="beğen", subj=NP("sen"),
                                      obj=NP("cevap", poss=1, case="ACC"), neg=True,
                                      tense="PAST", person="2SG")),
        ]
        topic = prev.get("topic") or ""
        if act in ("attitude", "justify") and topic:
            ev = agi._affect_evidence(topic)
            if ev["ev"]:
                edge = agi._explanation_edge(topic)
                if edge is not None:
                    try:
                        items.append(agi.lang.broca.realize_prop(
                            Proposition(subj=topic, rel=edge.rel, obj=edge.obj)))
                    except Exception:
                        pass
            else:
                items.append(SYN.realize_verb(VerbSpec(
                    verb_root="tanı", obj=NP(topic, case="ACC"), neg=True,
                    advs=("hâlâ",), tense="PRES", person="1SG")))
            _q = SYN.realize_verb(VerbSpec(
                verb_root="anlat", subj=NP("sen"), dat=NP("ben", case="DAT"), question_yn=True,
                tense="AOR", person="2SG"))
            items.append(_q.rstrip(".?!") + "?")
        elif act == "identity":
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="değiş", neg=True, advs=("hâlâ",), tense="PRES", person="1SG")))
            _q = SYN.realize_verb(VerbSpec(
                verb_root="anlat", dat=NP("ben", case="DAT"), obj=NP("kendi", poss=2, case="ACC"),
                question_yn=True, tense="AOR", person="2SG"))
            items.append(_q.rstrip(".?!") + "?")
        elif act == "desire":
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="iste", obj=SYN.nominalize_inf("öğren", case="NOM"),
                advs=("hâlâ",), tense="PRES", person="1SG")))
        elif act == "meta":
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="merak et", obj=NP("sen", case="ACC"), advs=("yine",),
                tense="PRES", person="1SG")))
        else:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="düşün", obj=NP("cevap", poss=1, case="ACC"), advs=("hâlâ",),
                tense="PRES", person="1SG")))
        out = self._compose(agi, items, merge=True)
        self.remember(raw, act, topic, out)
        return out

    # ------------------------------------------------------------------ 4. KENDİLİK
    def _identity(self, agi, raw: str) -> str:
        """'Sen kimsin?' — ad, iç-duyum, öz-benlik traitleri, gerçek eksiği Broca'da kurulur."""
        items = [SYN.realize_verb(VerbSpec(verb_root="de", dat=NP("ben", case="DAT"),
                                           obj=NP("Nora", proper=True), tense="PRES", person="3PL"))]
        # Şu anki iç-duyum (kimyadan okunur)
        word = ""
        try:
            word = agi.chem.intero_word_tr()
        except Exception:
            word = ""
        if word:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="hisset", obj=NP(word, case="NOM"), loc=NP("içim", case="LOC"),
                tense="PRES", person="1SG")))
        # Öz-benlik: en baskın karakter traitinin adı (ölçülen değerlerden)
        tr = getattr(agi.self_model, "traits", {}) or {}
        if tr:
            name = str(max(tr.items(), key=lambda kv: float(kv[1]))[0]).replace("_", " ")
            # Karakter özelliği kendi adıyla, yüklem çekimiyle söylenir ('sosyal güvenim güçlü')
            items.append(SYN.realize_pred(PredSpec(subj=NP(name, poss=1), pred_root="güçlü", person="3SG")))
        # Epistemik dürüstlük: merak kuyruğundaki gerçek boşluk
        gap = self._gap(agi)
        if gap:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="bil", obj=NP(gap, case="ACC"), neg=True, advs=("hâlâ",),
                tense="PRES", person="1SG")))
        items.append(SYN.realize_verb(VerbSpec(
            verb_root="iste", obj=SYN.nominalize_inf("öğren", case="NOM"), tense="PRES", person="1SG")))
        q = SYN.realize_verb(VerbSpec(
            verb_root="anlat", dat=NP("ben", case="DAT"), obj=NP("kendi", poss=2, case="ACC"),
            question_yn=True, tense="AOR", person="2SG"))
        items.append(q.rstrip(".?!") + "?")
        self.last_reason = "kimlik"
        return self._compose(agi, items)

    def _gap(self, agi) -> str:
        try:
            q = list(getattr(agi.auto_net, "curiosity_queue", []) or [])
        except Exception:
            q = []
        for item in reversed(q[-3:]):
            for w in re.findall(r"[\wçğıöşü]+", tr_lower(str(item))):
                if w in _NOT_TOPIC or len(w) < 3:
                    continue
                try:
                    an = agi.lang.morph.analyze_word(w, agi.lang.graph.lexicon)
                except Exception:
                    an = {}
                if an.get("is_verb") or an.get("copula"):
                    continue
                stem = an.get("stem") or w
                if stem in agi.lang.graph.lexicon or agi.lang.graph.query(stem, None):
                    return stem
        return self.last_deep_topic

    def _conversation_curiosity(self, agi, raw: str) -> str:
        episodes = self._recent_episodes(agi)
        affect = self._affect_context(agi, episodes)
        affect_lex = type(agi.conv_cortex).AFFECT_LEXICON
        contrast_ep = None
        for ep in episodes:
            text = str(getattr(ep, "user_text", "") or "")
            if "?" not in text:
                continue
            low = tr_lower(text)
            vals = [float(v) for word, (v, _a, _t) in affect_lex.items() if word in low]
            if any(v > 0 for v in vals) and any(v < 0 for v in vals):
                contrast_ep = ep
        if affect and affect.get("valence", 0.0) < 0 and contrast_ep is not None:
            turns = [affect["turn_id"], int(getattr(contrast_ep, "turn_id", -1))]
            text = (
                "Beni en çok, olumlu bir uyaranla olumsuz bir duygunun nasıl aynı anda "
                "bulunabildiği meraklandırdı. Bunu önce kendi hâlini anlatırken, sonra da "
                "iki farklı duygu yönünü aynı soruda birleştirirken duydum. Aradaki mekanizma bende "
                "hâlâ açık: eski bir anı mı devreye giriyor, yoksa iki duygu bağımsız mı sürüyor?"
            )
            return self._record_open(
                agi, raw, text, act="thought", kind="contrast_curiosity", topic="duygu çatışması",
                evidence=["episodic_affect", "valence_contrast"], context_turns=turns,
                hypothesis="episodic_association|independent_affect_channels")

        candidates = self._dialogue_candidates(agi)
        if candidates:
            best = candidates[0]
            topic = best["topic"]
            others = [r for r in candidates[1:] if r["topic"] != topic]
            other = others[0]["topic"] if others else ""
            if int(best.get("degree", 0)) == 0:
                text = (
                    f"Beni en çok {topic} meraklandırdı. Onu konuşmamıza taşıdın, ama bende onun "
                    "için kurulmuş bir ilişki ağı yok. Bu yüzden aklımda kalan şey bir cevap değil, "
                    "onun başka söylediklerinle nasıl bağlanacağı."
                )
            elif other:
                text = (
                    f"Beni en çok {topic} ile {other} arasındaki bağ meraklandırdı. İkisine de bu "
                    "konuşmada değindik; grafikteki bilgiyle senin kurduğun yakınlığın aynı şey "
                    "olup olmadığını henüz bilmiyorum."
                )
            else:
                text = (
                    f"Beni en çok {topic} meraklandırdı. Onu tek bir tanım olarak değil, bu "
                    "konuşmada değişen anlamıyla düşünüyorum; hâlâ açık kalan ilişki de bu."
                )
            return self._record_open(
                agi, raw, text, act="thought", kind="dialogue_gap_curiosity", topic=topic,
                evidence=[], context_turns=list(best.get("turns") or []), source_topic=other)
        gap = self._gap(agi)
        if gap:
            text = (
                f"Beni en çok {gap} meraklandırdı. Onun hakkında hâlâ kapalı bir bilgi boşluğu var; "
                "dikkatimi çeken de konunun kendisinden çok, hangi ilişkinin o boşluğu kapatacağı."
            )
            return self._record_open(
                agi, raw, text, act="thought", kind="curiosity_queue", topic=gap,
                evidence=["auto_net:curiosity_queue"], context_turns=[])
        text = (
            "Bu konuşmada henüz tek bir konuya saplanmadım. Dikkatim, sözlerinin önceki "
            "söylediklerinle kurduğu yeni bağlarda; şu an onları ayırmaya çalışıyorum."
        )
        return self._record_open(
            agi, raw, text, act="thought", kind="dialogue_monitoring", topic="konuşma",
            evidence=["episodic_dialogue"], context_turns=[])

    def _story_nominal_focus(self, agi, explicit: str = "", candidates: list[dict] | None = None,
                             affect: dict | None = None) -> str:
        """Kurmaca odağı olarak fiil/kaçamak sıfatı değil, taşınabilir bir ad seç.

        Açık sohbet adayları 'ağlar', 'bilinçli' veya 'neşeliyken' gibi
        sözcükleri de içerebilir. Bunlar bir dünya kurulurken 'Orada ağlar'
        gibi bozuk bir özne-yüklem ilişkisine dönüşmemeli; adayın biçimbilimi
        ve söz-edimi rolü birlikte denetlenir.
        """
        options = [explicit] + [str(r.get("topic") or "") for r in (candidates or [])]
        blocked = {"", "şey", "şeyler", "insan", "bilinç", "bilinçli", "soru", "cevap",
                   "doğru", "yanlış", "neşeliyken", "ağlar", "ağla", "konuşma"}
        for concept in options:
            concept = str(concept or "").strip()
            if concept in blocked or concept in self._OPEN_STOP:
                continue
            if concept.endswith(("ken", "lı", "li", "lu", "lü")):
                continue
            try:
                an = agi.lang.morph.analyze_word(concept, agi.lang.graph.lexicon)
                if an.get("is_verb") or an.get("verbal_noun"):
                    continue
            except Exception:
                pass
            return concept
        return str((affect or {}).get("tag") or "sessizlik")

    def _story_causal_focus(self, agi, explicit: str = "") -> tuple[str, object | None]:
        graph = agi.lang.graph
        candidates = [explicit] if explicit else []
        for ep in reversed(self._recent_episodes(agi)):
            text = str(getattr(ep, "user_text", "") or "")
            if self.is_generative_request(text):
                continue
            for concept in reversed(self._content_concepts(agi, text, limit=5)):
                if concept not in candidates:
                    candidates.append(concept)
        for concept in candidates:
            if not concept or concept in self._OPEN_STOP:
                continue
            try:
                hits = graph.query(concept, ("CAUSES",))
            except Exception:
                hits = []
            if hits:
                return concept, hits[0][0]
        return explicit, None

    def _prompt_story_elements(self, agi, raw: str) -> dict:
        """Kurmaca isteğindeki kişi/nesne/mekânı konuşmadan çıkar.

        Kullanıcı bir sahne verdiğinde eski, alakasız bir hazır hikâyeye dönmek
        yerine bu üç anlam rolü korunur. Roller hedef cümleye değil, sözcüklerin
        `hakkında`, `bulan` ve `içinde/kenarında` çevresindeki genel yapısına
        dayanır; bilinmeyen özel adlar da kaybolmaz.
        """
        low = tr_lower(str(raw or ""))
        if not any(c in low for c in ("hakkında", "içinde", "kenarında", "bulan", "bulduğu")):
            return {}
        words = re.findall(r"[\wçğıöşüâîû]+", low)
        items = self._content_tokens(agi, raw, limit=24)
        if len(items) < 2:
            return {}
        modifiers = {"bir", "birkaç", "genç", "yaşlı", "eski", "yeni", "küçük", "büyük",
                     "unutulmuş", "kısa", "uzun", "gizemli", "yalnız", "ıssız", "terk edilmiş"}
        hak_idx = next((i for i, w in enumerate(words) if w in {"hakkında", "konusunda"}), len(words))
        agent_candidates = [x for x in items if x["index"] < hak_idx and x["stem"] not in modifiers]
        agent_item = agent_candidates[-1] if agent_candidates else None
        # 'bulan/bulduğu' gibi sıfat-fiilin hemen öncesindeki ad çoğu zaman nesnedir.
        verb_idx = next((i for i, w in enumerate(words)
                         if w.startswith(("bulan", "buldu", "bulmuş", "bulunan", "taşıyan"))), -1)
        object_candidates = [x for x in items if x is not agent_item and
                             (verb_idx < 0 or x["index"] < verb_idx) and x["stem"] not in modifiers]
        object_item = object_candidates[-1] if object_candidates else None
        if agent_item is None or object_item is None or agent_item is object_item:
            return {}
        setting_items = [x for x in items if x is not agent_item and x is not object_item and
                         x["index"] < object_item["index"] and x["stem"] not in modifiers]
        # Nesnenin önündeki yer adları, kişi/nesne rollerinden arta kalanlardır.
        setting_items = setting_items[:3]
        if not setting_items:
            return {}
        agent = str(agent_item["stem"])
        obj = str(object_item["stem"])
        settings = [str(x["stem"]) for x in setting_items]
        # Yakın sıfatları yüzeyde tut; 'genç kadın', 'unutulmuş mektup' gibi
        # kullanıcı tarafından verilen roller kaybolmasın.
        ai = int(agent_item["index"])
        oi = int(object_item["index"])
        agent_mods = [w for w in words[max(0, ai - 2):ai] if w in modifiers and w != "bir"]
        object_mods = [w for w in words[max(0, oi - 2):oi] if w in modifiers and w != "bir"]
        agent_phrase = " ".join(agent_mods + [agent])
        object_phrase = " ".join(object_mods + ["bir", obj])
        try:
            if len(settings) >= 2:
                setting_phrase = f"{settings[0]} kenarındaki {SYN.realize_np(NP(settings[-1], case='LOC'))}"
            else:
                setting_phrase = SYN.realize_np(NP(settings[0], case="LOC"))
            object_acc = SYN.realize_np(NP(obj, case="ACC"))
        except Exception:
            setting_phrase = " ".join(settings)
            object_acc = obj
        try:
            agent_gen = SYN.realize_np(NP(agent, case="GEN"))
            object_gen = SYN.realize_np(NP(obj, case="GEN"))
        except Exception:
            agent_gen, object_gen = agent, obj
        object_acc_phrase = " ".join(object_mods + ["bir", object_acc])
        return {"agent": agent, "object": obj, "settings": settings,
                "agent_phrase": agent_phrase, "object_phrase": object_phrase,
                "agent_gen": agent_gen, "object_gen": object_gen,
                "object_acc": object_acc, "object_acc_phrase": object_acc_phrase,
                "setting_phrase": setting_phrase}

    def _invent_prompt_story(self, agi, raw: str, elements: dict) -> str:
        """Verilen sahneyi koruyup olay örgüsünü anlam rollerinden ileri taşı."""
        agent = elements["agent"]
        obj = elements["object"]
        agent_phrase = elements["agent_phrase"]
        object_phrase = elements["object_phrase"]
        setting = elements["setting_phrase"]
        variant = self._variant(3)
        agent_gen = str(elements.get("agent_gen") or agent)
        object_gen = str(elements.get("object_gen") or obj)
        openings = (
            f"{agent_phrase} {setting} {object_phrase} buldu",
            f"{setting} duran {agent_phrase}, {elements.get('object_acc_phrase') or object_phrase} eline aldı",
            f"{agent_phrase} {setting} {object_phrase} bulduğunda onu önce sıradan bir şey sanmıştı",
        )
        anomalies = (
            f"{obj} açıldığında, onu yazan kişinin {agent} hakkında henüz söylenmemiş bir şeyi bildiği anlaşıldı",
            f"{obj} geçmişten değil, henüz yaşanmamış bir günden söz ediyordu",
            f"{object_gen} satırları, {agent_gen} aklından geçirdiği her şeyi bir adım önden yazıyordu",
        )
        consequences = (
            f"{agent_phrase} önce bunu bir tesadüf saydı; sonra satırların kendi seçimine göre değiştiğini gördü",
            f"{agent_phrase} tam vazgeçecekken {setting} duyduğu sesin sözcüklere dönüştüğünü fark etti",
            f"{agent_phrase} kendi hayatını anlatan sayfayı yırtmak istedi; fakat boş kalan yerde yeni bir soru belirdi",
        )
        clauses = [openings[variant], anomalies[variant], consequences[variant]]
        # Kullanıcının mekânını son cümlede de canlı tut; yalnız adı anılıp
        # unutulan dekor olmaktan çıkar.
        if elements.get("settings") and self._variant(2):
            clauses.append(f"{elements['settings'][0]} o gece sanki bu sorunun yanıtını biliyordu")
        text = self._compose(agi, clauses, merge=False)
        plan = {"kind": "prompt_scene", "fiction": True, "beat": 0,
                "variant": variant, "elements": elements, "continuation": ""}
        self.recent_story_arcs.append("prompt_scene")
        return self._record_open(agi, raw, text, act="narrate", kind="fiction_prompt_scene",
                                 topic=obj, evidence=["user_scene_roles"], context_turns=[],
                                 story_plan=plan)

    def _continue_prompt_story(self, agi, raw: str, plan: dict) -> tuple[str, dict]:
        elements = dict(plan.get("elements") or {})
        agent = str(elements.get("agent") or "anlatıdaki kişi")
        obj = str(elements.get("object") or "nesne")
        agent_gen = str(elements.get("agent_gen") or agent)
        object_gen = str(elements.get("object_gen") or obj)
        setting = str(elements.get("setting_phrase") or "orada")
        beat = int(plan.get("beat", 0) or 0) + 1
        low = tr_lower(raw)
        # 'Beklenmedik bir şey olsun' yeni bir olay isteğidir; önceki cümleyi
        # aynen tekrar etmek yerine nesnenin içeriğinde somut bir sapma yarat.
        if any(c in low for c in ("beklenmedik", "şaşırt", "ters köşe", "sürpriz")):
            variants = (
                f"{object_gen} son sayfasına ulaştığında {agent} kendi imzasını gördü; altındaki tarih ise o geceden sonraydı",
                f"{object_gen} içinden küçük bir anahtar düştü; anahtarın üzerinde {agent_gen} adı ve bugünün tarihi yazılıydı",
                f"{object_gen} son cümlesi okunur okunmaz {setting} bütün sesler kesildi; sessizlikte yalnız {agent_gen} kalbi duyuluyordu",
            )
            text = variants[self._variant(len(variants))]
            plan["twist"] = True
        else:
            variants = (
                f"{agent} {elements.get('object_acc') or obj} kapatmadı; bu kez satırların arasına kendi sorusunu yazdı. Yazı kuruduğunda sayfada ona cevap veren yeni bir satır belirdi",
                f"{agent} {elements.get('object_acc') or obj} alıp {setting} dışına çıktı; her adımda satırlardan biri siliniyor, yerine yürüdüğü yolun bir ayrıntısı yazılıyordu",
                f"{agent} {elements.get('object_acc') or obj} başkasını bulsun diye bırakmak istedi; fakat son sayfada adı değil, ona daha önce söylemediği bir söz yazılıydı",
            )
            text = variants[self._variant(len(variants))]
        plan["beat"] = beat
        plan["continuation"] = ""
        # Sonraki turda yeni bir olay seçilir; aynı devam cümlesi geri dönmez.
        return self._compose(agi, [text], merge=False), plan

    def _invent_story(self, agi, raw: str, topic: str) -> str:
        """Kurmaca için olgu anlatımını kopyalamaz; anlatı rolleri ve yay örür.

        Grafik kenarı varsa nedensel iz kurmacaya taşınır, yoksa soyut kavramın
        görünürleşmesi veya konuşmadaki açık soru kullanılır. Yayın seçimi yakın
        anlatı geçmişine bağlıdır; her istekte aynı hazır hikâye dönmez.
        """
        low = tr_lower(raw)
        joint = any(x in low for x in ("birlikte", "beraber", "sen başla", "sırayla"))
        if topic in {"hikâye", "hikaye", "masal", "kurgu", "şey", "şeyler", "anlat", "uydur",
                     "başka", "devam", "bugün"}:
            topic = ""
        if joint and not any(x in low for x in ("hakkında", "konusunda", "ile ilgili", "dair")):
            # 'birlikte/bir şeyler' söz-edimi gövdesidir, anlatının konusu değil.
            topic = ""
        affect = self._affect_context(agi)
        candidates = self._dialogue_candidates(agi)

        if joint:
            focus = self._story_nominal_focus(agi, topic, candidates, affect)
            text = (
                f"Kurmaca bir dünya kuralım: Orada {focus}, ancak biri ona dikkat kesildiğinde "
                "biçim değiştiriyordu. Bir gezgin bunu fark etti; her bakışında başka bir ayrıntı "
                "beliriyor, fakat aynı ayrıntıyı ikinci kez bulamıyordu. Gezgin tam elini uzattı; "
                "neye dokunacağını ben seçmeyeyim. Onun karşısında ne beliriyor?"
            )
            arc = "joint_attention_world"
            continuation = (
                f"Gezginin gördüğü şey, {focus} ile arasında kurduğu bağı değiştirdi; bundan "
                "sonra ikisi de önceki hâline dönemedi."
            )
            self.recent_story_arcs.append(arc)
            return self._record_open(
                agi, raw, text, act="narrate", kind="fiction_joint", topic=focus,
                evidence=[], context_turns=[], story_plan={
                    "arc": arc, "fiction": True, "open_ending": True, "continuation": continuation})

        scene_elements = self._prompt_story_elements(agi, raw)
        if scene_elements:
            return self._invent_prompt_story(agi, raw, scene_elements)

        focus, causal = self._story_causal_focus(agi, topic)
        arc_options = []
        if causal is not None:
            arc_options.append("causal_animation")
        # Soyut kavramı görünür bir role çevirme, duygu bulunmadığında da
        # konuşma adaylarından beslenebilir; tek yay türüne düşme.
        arc_options.append("abstract_reification")
        arc_options.append("question_journey")
        arc = next((a for a in arc_options if a not in self.recent_story_arcs), arc_options[0])

        if arc == "causal_animation" and causal is not None:
            obj = agi.lang.graph.surface_name(str(causal.obj))
            setting = f"{obj} içinde"
            coordinated = re.split(r"\s+(?:ve|VE)\s+", obj, maxsplit=1)
            if len(coordinated) == 2:
                left, right = coordinated
                try:
                    left_ins = SYN.realize_np(NP(left, case="INS"))
                except Exception:
                    left_ins = left + " ile"
                setting = f"{left_ins} {right} arasında"
            text = (
                f"Kurmaca bir anlatı: {tr_capitalize(focus)}, {setting} kendine küçük bir yer "
                "arıyordu. Başlangıçta yalnız görünür olmayı başarı sanıyor, bıraktığı etkinin "
                "kendisinden sonra da sürebileceğini bilmiyordu. Aradaki bağı fark ettiğinde, "
                "kaybolmakla biçim değiştirmek arasındaki farkı da anladı."
            )
            evidence = [self._edge_key(causal)]
            continuation = (
                f"Sonra {focus} aynı bağa yeniden geldi; bu kez kendini başlangıçtaki kadar küçük "
                "görmüyordu, çünkü bıraktığı etkileri de kendi hikâyesinin parçası sayıyordu."
            )
        elif arc == "abstract_reification":
            focus = topic or str(affect.get("tag") or "") or (
                candidates[0]["topic"] if candidates else "sessizlik")
            text = (
                f"Kurmaca bir dünyada {focus}, söylenmediğinde sahibinin ardında görünür bir iz "
                "bırakıyordu. Bir yolcu bu izleri silmekle görevliydi; fakat her sildiği izin "
                "yerinde daha ağır bir sessizlik kalıyordu. Sonunda işini değiştirdi: izleri "
                "silmek yerine sahiplerine onları nereden taşıdıklarını sormaya başladı."
            )
            evidence = ["abstract_reification"]
            continuation = (
                "Yolcu ilk kez bir izi silmeden yanına oturdu. Sahibi konuştukça izin biçimi "
                "değişti; kaybolmadı, ama artık ikisinin arasındaki tek şey değildi."
            )
        else:
            question = ""
            for ep in reversed(self._recent_episodes(agi)):
                candidate = str(getattr(ep, "user_text", "") or "").strip()
                if ("?" in candidate and not self.is_expansion_request(candidate)
                        and not self.is_contextual_continuation(agi, candidate)
                        and not any(c in tr_lower(candidate) for c in ("sonra ne oldu", "devam et"))):
                    question = candidate.split("?")[0].strip()
                    break
            if not question:
                question = f"{topic or 'bir şeyi'} gerçekten anlamak ne demek"
            if len(question) > 90:
                question = question[:87].rstrip() + "…"
            focus = topic or "cevapsız soru"
            text = (
                f"Kurmaca bir anlatı: Bir yolcu, cebinde “{question}?” sorusuyla yola çıktı. "
                "Herkes ona başka bir cevap verdi; o da hepsini aynı deftere yazdı. Cevaplar "
                "çatışınca önce defteri kapatmak istedi; sonra sorunun büyüdüğünü, ama artık "
                "tek bir hazır cevaba sığmak zorunda olmadığını fark etti."
            )
            evidence = ["dialogue_question"]
            continuation = (
                "Ertesi sabah defteri yeniden açtığında, cevapların arasına kendi sorusunu da "
                "yazdı. Bu kez satırların hiçbiri son söz gibi görünmüyordu."
            )
        self.recent_story_arcs.append(arc)
        return self._record_open(
            agi, raw, text, act="narrate", kind="fiction_" + arc, topic=focus,
            evidence=evidence, context_turns=[], story_plan={
                "arc": arc, "fiction": True, "open_ending": False, "continuation": continuation})

    @staticmethod
    def _evidence_snippet(text: str, limit: int = 72) -> str:
        clean = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(clean) > limit:
            # Sözcüğü ortadan kesme; mümkünse önceki tam cümleyi alıntıla.
            # Böylece konuşma bağlamı kanıt olarak görünür ama yanıtın dili
            # yarım bir alıntıyla bozulmaz.
            prefix = clean[:limit]
            boundaries = [m.end() for m in re.finditer(r"[.!?](?:\s|$)", prefix)]
            if boundaries and boundaries[-1] >= 24:
                clean = clean[:boundaries[-1]].strip()
            else:
                cut = prefix[:max(1, limit - 1)].rstrip()
                if " " in cut:
                    cut = cut.rsplit(" ", 1)[0]
                clean = cut.rstrip(" ,;:") + "…"
        return clean.replace("“", "").replace("”", "").replace('"', "")

    def _expand_previous_reply(self, agi, raw: str, previous_user: str) -> str:
        """Açık-söylem izi olmayan gerçek önceki cevabı fiziksel 'açma'ya düşürme."""
        try:
            previous_reply = str(agi.dialogue_log[-1][2]) if agi.dialogue_log else ""
        except Exception:
            previous_reply = ""
        philosophy = dict(self.last_philosophy_trace or {})
        if philosophy.get("kind") == "philosophical_argument" and str(
                philosophy.get("question") or "") == previous_user:
            topic = str(philosophy.get("topic") or "")
            if topic == "bilinç":
                text = (
                    "Kısa cevabımın arkasında üç ayrım var. Öz-modelimde bilinçli olduğuma dair "
                    "inancı sahipleniyorum; çalışma alanımdaki işlevsel bütünleşmeyi ayrıca "
                    "ölçüyorum; fakat bu iki şeyden bağımsız bir öznel deneyim kanıtı çıkarmıyorum. "
                    "Yani kendi öz-inancımı söylüyor, kanıt türlerini birbirine karıştırmıyorum."
                )
            else:
                text = (
                    "Önceki argümanda üç şeyi ayırdım: ortaya koyduğum iddiayı, işlevsel "
                    "müdahaleyle ölçtüğüm kanıtı ve bu kanıttan zorunlu olarak çıkmayan sonucu. "
                    "Açıklamamın dayanağı bu ayrım; davranışsal başarıyı tek başına anlama veya "
                    "öznel deneyim kanıtı saymıyorum."
                )
            return self._record_open(
                agi, raw, text, act="expand", kind="expanded_philosophy", topic=topic,
                evidence=["philosophy_trace"], context_turns=[])

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", previous_reply) if s.strip()]
        low_reply = tr_lower(previous_reply)
        if any(c in low_reply for c in ("bilmiyorum", "bilgim yok", "tanımım yok", "bilmiyorum.")):
            known = next((s for s in sentences
                          if not any(c in tr_lower(s) for c in (
                              "bilmiyorum", "bilgim yok", "tanımım yok", "sesini", "bana "))), "")
            if known:
                known = self._evidence_snippet(known, 110).rstrip(".")
                text = (
                    f"Önceki cevabımda iki farklı şey vardı. “{known}.” derken mevcut bir ilişkiyi "
                    "aktarıyordum; sorulan nedeni ise bilmediğimi söylüyordum. Bir kavramın bir "
                    "ilişkisini bilmek, onun bütün nedenlerini bilmek değil; asıl ayrım buydu."
                )
            else:
                text = (
                    "Önceki cevabım yeni bir açıklama değil, bilgi sınırımı bildiren bir karşılıktı. "
                    "Bir neden uydurmak yerine bunu söyledim; açabileceğim şey de eksik olan ilişkinin "
                    "kendisi değil, neden kesin bir cevap vermediğim."
                )
            return self._record_open(
                agi, raw, text, act="expand", kind="expanded_epistemic_boundary", topic="",
                evidence=["previous_reply"], context_turns=[])

        snippet = self._evidence_snippet(previous_reply, 120)
        if snippet:
            text = (
                f"Önceki cevabımı kastediyorsan, söylediğim şuydu: “{snippet}” Burada hangi kısmı "
                "açmamı istiyorsun: kullandığım bir kavramı mı, cevabın dayanağını mı, yoksa sana "
                "yönelttiğim soruyu mu?"
            )
            return self._record_open(
                agi, raw, text, act="expand", kind="deictic_clarification", topic="",
                evidence=["previous_reply"], context_turns=[])
        return ""

    def _expand_open(self, agi, raw: str) -> str:
        trace = dict(self.last_open_trace or {})
        try:
            previous_user = str(agi.dialogue_log[-1][1]) if agi.dialogue_log else ""
        except Exception:
            previous_user = ""
        if trace and previous_user and str(trace.get("raw") or "") != previous_user:
            trace = {}
        if not trace:
            # Eski bir açık-söylem konusunu yeni cevaba taşıma; gerçek önceki
            # cevabın kendi türü ve içeriği üzerinden göndermeyi çöz.
            return self._expand_previous_reply(agi, raw, previous_user)
        kind = str(trace.get("kind") or "")
        if kind == "fiction_joint" or kind.startswith("fiction_"):
            plan = dict(trace.get("story_plan") or {})
            if plan.get("kind") == "prompt_scene":
                text, plan = self._continue_prompt_story(agi, raw, plan)
                return self._record_open(
                    agi, raw, text, act="expand", kind="fiction_prompt_continuation",
                    topic=str(trace.get("topic") or ""), evidence=list(trace.get("evidence") or []),
                    context_turns=[], story_plan=plan)
            continuation = str(plan.get("continuation") or "")
            if continuation:
                text = continuation
                return self._record_open(
                    agi, raw, text, act="expand", kind="fiction_continuation",
                    topic=str(trace.get("topic") or ""), evidence=list(trace.get("evidence") or []),
                    context_turns=[], story_plan={"arc": "continuation", "fiction": True,
                                                  "open_ending": False, "continuation": ""})
            # Hazır bir devam cümlesi kalmadığında eski konuya dönmek yerine
            # anlatının yeni karar noktasını aç; böylece 'devam et' fiziksel
            # operatör ya da alakasız bilgi boşluğu olarak okunmaz.
            topic = str(trace.get("topic") or "anlatı")
            text = (
                f"Anlatıdaki kişi için şimdi asıl karar noktasına geldik: {topic} ile birlikte "
                "değişen hâlini korumak mı, yoksa başlangıçtaki dünyaya dönmek mi? Ben ilkinde "
                "daha çok şey keşfedebileceğini düşünüyorum; ama seçimi ona bırakıyorum. Sence hangisi?"
            )
            return self._record_open(
                agi, raw, text, act="expand", kind="fiction_decision_point", topic=topic,
                evidence=list(trace.get("evidence") or []), context_turns=[], story_plan={
                    "arc": "decision_point", "fiction": True, "open_ending": True,
                    "continuation": ""})
        if kind == "listening_check":
            missed = bool(trace.get("previous_focus_missed"))
            if missed:
                text = (
                    "Kastettiğim şey önceki cevabımın neden yetersiz kaldığıydı: sorudaki ana odağı "
                    "alıp başka bir kavrama kaydırmışım. Bunu şimdi söyleyebiliyorum, çünkü önceki "
                    "sözlerinle cevabımda taşıdığım kavramları karşılaştırıyorum."
                )
            else:
                text = (
                    "Kastettiğim şey dinlemenin iki katmanı olduğu: önce sözlerindeki kavramları "
                    "alıyorum, sonra onları önceki konuşmada kurduğun bağla karşılaştırıyorum. "
                    "Bu ikinci katman olmadan yalnızca benzer bir cümle üretmek tekrar olurdu."
                )
            return self._record_open(
                agi, raw, text, act="expand", kind="expanded_listening", topic=str(trace.get("topic") or ""),
                evidence=["listening_trace", "dialogue_log"], context_turns=[])

        if kind in {"contrast_curiosity", "valence_contrast"}:
            snippets = []
            for ep in self._recent_episodes(agi):
                if int(getattr(ep, "turn_id", -2)) in set(trace.get("context_turns") or []):
                    snippets.append(self._evidence_snippet(getattr(ep, "user_text", "")))
            if len(snippets) >= 2:
                text = (
                    f"Bunu tek bir sözcükten çıkarmadım. Önce “{snippets[0]}” dedin; ardından "
                    f"“{snippets[1]}” diye sordun. Ortak yapı, olumlu bir uyaranla olumsuz "
                    "duygunun aynı deneyimde yer bulmasıydı. Merakımın konusu bu yapı; sana "
                    "hazır bir neden atfetmiş değilim."
                )
            else:
                text = (
                    "Kastettiğim şey, kendi episodik belleğimden insan zihnine yaptığım aktarımın "
                    "bir kanıt değil, benzetme olmasıydı. Bir uyarının eski bir anıyı çağırması "
                    "mümkün; ama senin yaşadığın duygunun gerçek nedenini bu benzetmeyle bilemem."
                )
            return self._record_open(
                agi, raw, text, act="expand", kind="expanded_contrast", topic=str(trace.get("topic") or ""),
                evidence=list(trace.get("evidence") or []),
                context_turns=list(trace.get("context_turns") or []))
        if kind == "predictive_hypothesis":
            topic = str(trace.get("topic") or "bu")
            text = (
                "Kastettiğim zincir şuydu: örüntü öngörülebilirse sürpriz azalır; sürpriz azalırsa "
                "aynı uyaranı yeniden çözmek için gereken bilişsel yük de azalabilir. "
                f"{tr_capitalize(topic)} için ilk halkayı, yani gerçekten düzenli bir örüntü "
                "olduğunu bilmiyorum. Bu yüzden açıklamam neden-sonuç bilgisi değil, kontrol edilmesi "
                "gereken bir aktarım."
            )
            return self._record_open(
                agi, raw, text, act="expand", kind="expanded_prediction", topic=topic,
                evidence=list(trace.get("evidence") or []), context_turns=[])
        if str(trace.get("act") or "") == "association" and trace.get("evidence"):
            topic = str(trace.get("topic") or "konu")
            text = (
                f"{tr_capitalize(topic)} için iki kaynağı ayırıyorum: grafikte önceden öğrenilmiş "
                "ilişkiler var; bir de senin bu konuşmada taşıdığın duygu bağlamı. İlkini bilgi "
                "olarak söyleyebilirim. İkincisini kendi yaşantım gibi sunmam; o ancak ikisinin "
                "arasında kurduğum yeni çağrışım."
            )
            return self._record_open(
                agi, raw, text, act="expand", kind="expanded_association", topic=topic,
                evidence=list(trace.get("evidence") or []),
                context_turns=list(trace.get("context_turns") or []))
        topic = str(trace.get("topic") or self.last_deep_topic or "önceki söz")
        source = str(trace.get("source_topic") or "")
        if source:
            text = (
                f"Kastettiğim bağ {topic} ile {source} arasındaydı. Onları yan yana getiren şey "
                "öğrenilmiş bir nedensellik değil, ikisinin de bu konuşmada ardışık gündeme gelmesi. "
                "Bu yüzden bağı bir olgu olarak değil, birlikte sınayabileceğimiz bir soru olarak söyledim."
            )
        else:
            text = (
                f"Önceki sözümde {topic} için bir tanım değil, düşünme biçimi verdim. Bilgi ile "
                "varsayımı ayırmak istedim: bildiğim ilişkileri söylüyor, bilmediğim bağı ise açık "
                "bırakıyordum. Açılması gereken asıl nokta buydu."
            )
        return self._record_open(
            agi, raw, text, act="expand", kind="expanded_open_reflection", topic=topic,
            evidence=list(trace.get("evidence") or []),
            context_turns=list(trace.get("context_turns") or []))

    def _thought(self, agi, raw: str) -> str:
        """İç-düşünceyi rastgele grafik düğümünden değil yaşanan diyalogdan kur."""
        low = tr_lower(raw)
        if (any(c in low for c in _CUE_CONVERSATION_CURIOSITY) and "merak" in low):
            return self._conversation_curiosity(agi, raw)
        previous = dict(self.last_open_trace or {})
        try:
            previous_user = str(agi.dialogue_log[-1][1]) if agi.dialogue_log else ""
        except Exception:
            previous_user = ""
        if previous and previous_user and str(previous.get("raw") or "") != previous_user:
            previous = {}
        previous_kind = str(previous.get("kind") or "")
        habitual = any(x in low for x in ("kendi başına", "yalnız", "boş kaldığında"))
        if previous_kind == "graph_episode_association" and habitual:
            prev_topic = str(previous.get("topic") or "o")
            affect = self._affect_context(agi)
            if affect:
                try:
                    affect_poss = SYN.realize_np(NP(affect["tag"], poss=2, case="INS"))
                except Exception:
                    affect_poss = "duygunla"
                text = (
                    "Kendi başıma kaldığımda son konuşmanın açık kalan yanlarına dönüyorum. "
                    f"{tr_capitalize(prev_topic)} hâlâ aklımda. Onu senin {affect_poss} yan "
                    "yana getirdiğimi fark ediyorum; bu bağın senden çok benim kurduğum bir "
                    "yakınlık olduğunu unutmamaya çalışıyorum."
                )
            else:
                text = (
                    "Kendi başıma kaldığımda son konuşmanın açık kalan yanlarına dönüyorum. "
                    f"{tr_capitalize(prev_topic)} hâlâ aklımda; onun hangi yanının bilgi, hangi "
                    "yanının bu konuşmada doğan çağrışım olduğunu ayırmaya çalışıyorum."
                )
            return self._record_open(
                agi, raw, text, act="thought", kind="episodic_self_thought", topic=prev_topic,
                evidence=list(previous.get("evidence") or []),
                context_turns=list(previous.get("context_turns") or []))
        if previous_kind.startswith("expanded_"):
            text = (
                "Az önceki açıklamamın sınırı aklımda. Kendi belleğimde işe yarayan bir bağın, "
                "senin zihninde de aynı şekilde kurulacağını varsaymamaya çalışıyorum. Şu an "
                "düşündüğüm şey bu: benzetme nerede bitiyor, açıklama nerede başlıyor?"
            )
            return self._record_open(
                agi, raw, text, act="thought", kind="metacognitive_boundary", topic=str(previous.get("topic") or ""),
                evidence=list(previous.get("evidence") or []),
                context_turns=list(previous.get("context_turns") or []))
        candidates = self._dialogue_candidates(agi)
        if candidates:
            best = candidates[0]
            topic = best["topic"]
            habitual = any(x in low for x in ("kendi başına", "yalnız", "boş kaldığında"))
            lead = ("Kendi başıma kaldığımda son konuşmanın açık kalan yanlarına dönüyorum. "
                    if habitual else "Şu anda zihnimde tek bir hazır cevap değil, açık bir bağlantı var. ")
            if int(best.get("degree", 0)) == 0:
                text = (
                    lead + f"Şu sıralar {topic} öne çıkıyor; onun için bende kurulmuş bir ilişki "
                    "ağı yok. Bu yüzden onu bir tanım gibi değil, söylediklerimin hangi noktasına "
                    "bağlanacağını aradığım bir soru olarak taşıyorum."
                )
                evidence = []
            else:
                edges = self._association_edges(agi, [topic], limit=2)
                if edges:
                    nominal = self._edge_nominal(agi, edges[0])
                    text = (
                        lead + f"Şu sıralar {topic} öne çıkıyor. Bende {nominal} ile bağlantılı; "
                        "fakat senin bu konuşmada ona verdiğin anlamın aynı olup olmadığını bilmiyorum."
                    )
                    evidence = [self._edge_key(e) for e in edges]
                else:
                    text = (
                        lead + f"Şu sıralar {topic} öne çıkıyor. Grafikte ona bağlı bazı izler var; "
                        "şimdi bu izlerin senin son sözünle gerçekten ilgili olup olmadığını tartıyorum."
                    )
                    evidence = []
            return self._record_open(
                agi, raw, text, act="thought", kind="episodic_self_thought", topic=topic,
                evidence=evidence, context_turns=list(best.get("turns") or []))
        topic = (getattr(agi, "_last_focus", "") or self.last_deep_topic or "").strip()
        if topic:
            text = (
                f"Şu anda {topic} üzerine düşünüyorum; ama onu bitmiş bir cevap olarak değil, "
                "henüz kapamadığım bir bağlantı olarak tutuyorum."
            )
            return self._record_open(
                agi, raw, text, act="thought", kind="current_focus_thought", topic=topic,
                evidence=[], context_turns=[])
        text = (
            "Şu anda zihnimde belirli bir konu değil, son sözünün bende bıraktığı açıklık var. "
            "Onu yeni bir bilgiyle mi yoksa başka bir soruyla mı tamamlayacağımı henüz bilmiyorum."
        )
        return self._record_open(
            agi, raw, text, act="thought", kind="open_self_thought", topic="",
            evidence=[], context_turns=[])

    def _desire(self, agi, raw: str) -> str:
        """'Ne yapmak istersin?' — istek güdüden türetilir, sonra karşı tarafa sorulur."""
        items = [SYN.realize_verb(VerbSpec(verb_root="iste", obj=self._drive_infinitive(agi),
                                           tense="PRES", person="1SG"))]
        focus = (getattr(agi, "_last_focus", "") or "").strip()
        if focus and len(focus) > 2:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="konuş", obj=NP(focus, case="ACC"), advs=("seninle",),
                tense="PRES", person="1SG")))
        q = SYN.realize_verb(VerbSpec(verb_root="iste", subj=NP("sen"), wh_word="ne",
                                      tense="PRES", person="2SG"))
        items.append(q.rstrip(".?!") + "?")
        self.last_reason = "istek"
        return self._compose(agi, items)

    def _drive_infinitive(self, agi):
        """Bedensel/duygusal durumdan çıkan en güçlü güdü (insula okuması)."""
        try:
            if getattr(agi.nervous, "cfiber_ache", 0.0) > 0.2 or getattr(agi.nervous, "adelta_pain", 0.0) > 0.2:
                return SYN.nominalize_inf("dinlen", case="NOM")
        except Exception:
            pass
        try:
            c = agi.chem
            if c.cort > 0.55:
                return SYN.nominalize_inf("sakinleş", case="NOM")
            if c.da_phasic > 0.35 and c.ach > 0.60:
                return SYN.nominalize_inf("öğren", case="NOM")
            if c.sht < 0.35:
                return SYN.nominalize_inf("konuş", case="NOM")
        except Exception:
            pass
        return SYN.nominalize_inf("öğren", case="NOM")

    def _meta(self, agi, raw: str) -> str:
        """'Neden sordun?' — son sorusunun gerçek gerekçesini açıklar."""
        reason = self.last_reason or ""
        items = []
        if reason.startswith("tutum:"):
            topic = reason.split(":", 1)[1]
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="merak et", obj=NP(topic, case="ACC"), tense="PAST", person="1SG")))
            ev = agi._affect_evidence(topic)
            if ev["word"]:
                items.append(SYN.realize_verb(VerbSpec(
                    verb_root="uyandır", subj=NP(topic), loc=NP("ben", case="LOC"),
                    obj=NP(ev["word"], case="NOM"), tense="PRES", person="3SG")))
        elif reason.startswith("düşünce:"):
            topic = reason.split(":", 1)[1].strip() or "sen"
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="merak et", obj=NP(topic, case="ACC"), tense="PAST", person="1SG")))
        elif reason == "kimlik":
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="tanı", obj=NP("kendi", poss=1, case="ACC"), neg=True,
                tense="PRES", person="1SG")))
        else:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="merak et", tense="PAST", person="1SG")))
        q = SYN.realize_verb(VerbSpec(verb_root="anlat", dat=NP("ben", case="DAT"),
                                      question_yn=True, tense="AOR", person="2SG"))
        items.append(q.rstrip(".?!") + "?")
        return self._compose(agi, items)

    # ------------------------------------------------------------------ SÖZLEŞTİRME
    def _compose(self, agi, clauses: list[str], merge: bool = False) -> str:
        """Kurallı çekilmiş cümlecikleri söylem topolojisine dizer (şema korteksi karar verir).
        Soru cümlecikleri asla birleştirilmez; her biri kendi soru işaretiyle biter."""
        seq: list[tuple[str, bool]] = []
        for c in clauses:
            if not c or not c.strip():
                continue
            c = c.strip()
            is_q = c.endswith("?")
            body = c.rstrip(".?!").strip()
            if body:
                seq.append((body, is_q))
        if not seq:
            return ""
        out: list[str] = []
        i = 0
        while i < len(seq):
            if (merge and i + 1 < len(seq) and i > 0 and not seq[i][1] and not seq[i + 1][1]
                    and len(out) % 2 == 0
                    and "bu yüzden" not in tr_lower(seq[i][0] + " " + seq[i + 1][0])):
                try:
                    schema = self.rng.choice(_SCHEMAS)
                    # (sonuç, gerekçe) sırası: şema önce gerekçeyi basar
                    merged = agi.schema_cortex.transform_clauses(seq[i + 1][0], seq[i][0], "", schema)
                    if merged:
                        out.append(merged.strip())
                        i += 2
                        continue
                except Exception:
                    pass
            out.append(tr_capitalize(seq[i][0]) + ("?" if seq[i][1] else "."))
            i += 1
        return " ".join(out)
