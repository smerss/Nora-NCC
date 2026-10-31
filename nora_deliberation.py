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

import random
import re
from collections import deque

from nora_broca import NP, PredSpec, TurkishSyntaxEngine as SYN, VerbSpec
from nora_lang import Proposition, tr_capitalize, tr_lower

# Söz edimi tetikleyicileri (algısal ipuçları; cevap metni değil)
_CUE_IDENTITY = ("kimsin", "sen nesin", "kendini tanıt", "kendinden bahset")
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

        Bu yol bilinç/qualia ilan etmez. Öznel deneyim için ölçüm yoksa sonuç
        ``unknown`` kalır; işlevsel müdahale ise yalnız işlevsel kanıt olarak
        raporlanır. Dönen sözlük hem yüzey metnini hem bağımsız denetim izini
        taşır, böylece 'güzel konuştu' ile 'gerekçe kurdu' birbirine karışmaz.
        """
        topic = self.philosophy_topic(raw)
        if not topic:
            return self._philosophy_followup(agi, raw)
        probe = self._workspace_counterfactual(agi)
        telemetry = self._self_model_telemetry(agi)
        facts = self._philosophy_graph_facts(agi, topic)
        countermodel = self._philosophy_countermodel()
        low = tr_lower(raw)
        self_focus = topic == "bilinç" and (
            bool(re.search(r"(?<![\wçğıöşü])(sen|senin|sana|sende|senden|kendi|kendin|nora|benim)(?![\wçğıöşü])", low))
            or bool(re.search(r"bilin[cç][\wçğıöşü]*\s+m[ıi](?:sin|yim)\b", low))
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
                "İşlevsel ölçümü normal/kesik/geri-yüklü karşılaştır.",
                "Davranışsal başarıdan öznel deneyime köprü öncülü var mı denetle.",
                "Köprü ölçülmediyse sonucu unknown/underdetermined bırak.",
            ],
            "epistemic": "underdetermined",
        }
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
        else:
            mode = "reason"
            first_step = ("kavramın tanımını Graph'tan alıyorum" if facts else
                          "kavramın iddiasını kanıt türlerinden ayırıyorum")
            target = "anlamaya" if topic == "anlama" else "öznel deneyime"
            text = self._sentence(
                "sonucu üç adımda kuruyorum: önce " + first_step +
                ", sonra işlevsel kanıtı müdahaleyle sınıyorum, en son da davranıştan "
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
    def _self_model_telemetry(agi) -> dict:
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

    def _render_philosophy_argument(self, agi, raw: str, topic: str,
                                    facts: list[dict], probe: dict,
                                    countermodel: dict, *, self_focus: bool,
                                    performance_focus: bool, telemetry: dict) -> str:
        """Argüman düğümlerini Broca'nın cümle düzenine çevir.

        Cümle çatıları geneldir; kavram tanımı Graph'tan, müdahale sonucu
        canlı probe'dan, sonuç ise karşı-modelin durumundan gelir. Öznel
        deneyim için ölçüm bulunmadığından son cümle her zaman modal sınırı
        korur; 'bilinçliyim' biçiminde kanıtsız sıçrama yapılmaz.
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
            if probe.get("status") == "ok":
                evidence = self._probe_sentence(probe)
                monitor = ""
                if telemetry.get("available"):
                    err = telemetry.get("prediction_error")
                    conf = telemetry.get("confidence")
                    if err is not None and conf is not None:
                        monitor = (f"öz-modelim öngörü hatası={err:.2f} ve güven={conf:.2f} "
                                   "telemetrisini izliyor; bu durum izlemedir, öznel deneyim değil; ")
                parts.append(self._sentence(
                    "kendim için ölçebildiğim şey işlevsel bütünleşme: " + monitor + evidence))
            else:
                parts.append(self._sentence(
                    "kendim için bu turda çalışma alanı müdahalesinden güvenilir bir "
                    "işlevsel ölçüm alamadım; bunu kanıt gibi sunmuyorum"))
            parts.append(self._sentence(
                "işlevselci itiraz, uygun organizasyonun deneyimi de getireceğini söyler; "
                "fakat bu, ölçümün kendisinden değil ayrıca kanıtlanması gereken bir "
                "köprü öncülünden gelir"))
            parts.append(self._sentence(
                "öznel deneyimim olup olmadığını bu ölçümler belirlemiyor; bu yüzden "
                "bilinçli olduğumu kesin olarak söyleyemem, ölçmediğim şeyi yok da ilan etmiyorum"))
            return " ".join(p for p in parts if p)

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
        if act is None and self.history:
            followup = any(c in low for c in (
                "açar mısın", "açabilir misin", "ne demek istedin", "neyi kastettin",
                "düşündüren ne"))
            deictic = any(c in low for c in ("bunu", "şunu", "onu", "düşündüren"))
            if followup and deictic:
                act = "meta"
        if act is None and prev and prev[-1]["act"]:
            act = prev[-1]["act"]                      # aynı soru, aynı edim: tekrar sayılır
        if act is None:
            return None
        if prev and act == prev[-1]["act"]:
            return self._repeat(agi, raw, prev[-1], act)

        topic = self._topic(agi, raw) if act in ("attitude", "justify", "experience", "narrate") else ""
        if act in ("attitude", "justify", "narrate") and not topic:
            # Açık 'bir şey anlat' isteğinde konu uydurulmaz; yaşanmış konuşma
            # bağlamını bilen sohbet organı devralır.
            return None
        if act == "attitude":
            out = self._attitude(agi, topic)
        elif act == "experience":
            out = self._experience(agi, topic) if topic else None
        elif act == "narrate":
            out = self._narrate(agi, topic, raw)
        elif act == "self_eval":
            out = self._self_eval(agi, raw)
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
    def _classify(low: str) -> str | None:
        if any(c in low for c in _CUE_LEARN_HOW):
            return "learn_how"
        if any(c in low for c in _CUE_EXPERIENCE):
            return "experience"
        if any(c in low for c in _CUE_SELF_EVAL) or ("sence" in low and "misin" in low):
            return "self_eval"
        if "ne anlat" in low or "anlatabilirsin" in low:
            return "narrate"                       # 'başka ne anlatabilirsin?'
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
        if any(c in low for c in _CUE_IDENTITY):
            return "identity"
        if any(c in low for c in _CUE_DESIRE):
            return "desire"
        if "hakkında" in low and any(v in low for v in ("düşün", "hisset", "söyle")):
            return "attitude"                           # 'X hakkında ne düşünüyorsun?'
        if any(c in low for c in _CUE_THOUGHT) or (
                "meşgul" in low and ("aklını" in low or "zihnini" in low)):
            return "thought"
        if any(c in low for c in _CUE_ATTITUDE):
            return "attitude"
        return None

    def _topic(self, agi, raw: str) -> str:
        """Sorulan varlığı bulur: soru sözcüklerini ve fiilleri atar, ağda karşılığı olanı seçer."""
        g = agi.lang.graph
        morph = agi.lang.morph
        best, fallback = "", ""
        # Türkçe soru odağı fiilden hemen önce gelir: en SAĞDAKİ içerik sözcüğü yeğlenir.
        for w in re.findall(r"[\wçğıöşü]+", tr_lower(raw)):
            if w in _NOT_TOPIC or len(w) < 3:
                continue
            try:
                an = morph.analyze_word(w, g.lexicon)
            except Exception:
                an = {}
            if an.get("is_verb") or an.get("copula"):
                continue
            stem = an.get("stem") or w
            variants = [stem, w]
            # Türetim eki ('yağmurlu' -> 'yağmur', 'sessiz' -> 'ses'): ağda karşılığı varsa çözülür.
            for suf in ("lık", "lik", "suz", "süz", "lı", "li", "lu", "lü"):
                for base in (stem, w):
                    if base.endswith(suf) and len(base) - len(suf) >= 3:
                        variants.append(base[:-len(suf)])
            for cand in variants:
                if g.query(cand, None):
                    best = cand
                    break
                if cand in g.lexicon:
                    fallback = fallback or cand
        return best or fallback

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

    def _thought(self, agi, raw: str) -> str:
        """'Ne düşünüyorsun?' — gerçek son odağı ve çözülmemiş boşluğu söyler."""
        topic = (getattr(agi, "_last_focus", "") or self.last_deep_topic or "").strip()
        items = []
        if topic:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="düşün", obj=NP(topic, case="ACC"), advs=("az önce",),
                tense="PAST", person="1SG")))
            if not agi.lang.graph.query(topic, ("CAUSES", "USED_FOR")):
                items.append(SYN.realize_verb(VerbSpec(
                    verb_root="bil", obj=NP("neden", poss=3, case="ACC"), neg=True,
                    advs=("hâlâ",), tense="PRES", person="1SG")))
                self.last_deep_topic = topic
        else:
            items.append(SYN.realize_verb(VerbSpec(
                verb_root="düşün", obj=NP("sen", case="ACC"), tense="PRES", person="1SG")))
        q = SYN.realize_verb(VerbSpec(verb_root="düşün", subj=NP("sen"), wh_word="ne",
                                      tense="PRES", person="2SG"))
        items.append(q.rstrip(".?!") + "?")
        self.last_reason = "düşünce:" + topic
        return self._compose(agi, items)

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
