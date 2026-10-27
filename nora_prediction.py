#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_prediction.py — gövde, nedensel grafik ve fiziği birleştiren öngörü.

Tek dış arayüz ``predict(durum, eylem)`` şu üçlüleri döndürür::

    [(değişken, delta, güven), ...]

Bu organ yeni bir nedensel olgu tabanı değildir. Üç mevcut kaynağı okur:

* Mind'in öğrenilmiş beden geçişleri ve deneyimsel V belleği;
* GroundedConceptGraph'ın genel CAUSES/IF_THEN yolları;
* Universe'in salt-okunur fizik önizlemesi.

Grafik etkileri beden kanallarına, CoreSelf'teki somatik kayıtların ve genel
homeostatik değişken adlarının mevcut anlamlarıyla bağlanır. Bir kavram için
özel ``if edge(...)`` kuralı yoktur. Deneyimsel beden kanıtı, aynı değişken
için duyumdan gelen grafik öngörüsünden daha yüksek kanıt ağırlığına sahiptir.

Fizik kopyası yalnız mekanik çıktı ve geometrik ilerleme için kullanılır.
Kopya dünyanın gizli ``poison`` bayrağı veya ondan doğan beden değişimi, ajanın
önceden bildiği bir sonuç gibi modele verilmez.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from nora_lang import tr_lower
from nora_world import Universe


class UnifiedPredictor:
    """Nora'nın mevcut organlarını değiştirmeden öngörülerini aynı imzada birleştirir."""

    BODY_VARS = ("enerji", "su", "bütünlük")
    BODY_INDEX = {name: i for i, name in enumerate(BODY_VARS)}
    # Bunlar nesnelerin fiziksel kanallarıdır; nedensel sonuçları değildir.
    # ISA ile yeni kavramlar da aynı kanallara bağlanabilir.
    TARGET_CHANNELS = {
        "yiyecek": 0, "yem": 0, "besin": 0, "elma": 0,
        "su": 1,
        "tehlike": 2, "diken": 2,
    }
    CHANNEL_NAMES = ("yiyecek", "su", "tehlike")
    DISCRETE_ACTIONS = {
        "dur": 0, "bekle": 0, "ileri": 1, "git": 1,
        "ye": 1, "iç": 1, "sola": 2, "sağa": 3,
        "geri": 4, "uzak_dur": 4, "kaç": 4, "sakın": 4,
    }
    SOURCE_PRECISION = {
        "deneyim": 0.96,
        "duyum": 0.72,
        "doğuştan": 0.42,
        "çıkarım": 0.58,
    }
    LOSS_MARKERS = (
        "kayıp", "kaybı", "azalma", "azalış", "tükenme", "eksiklik",
        "bozulma", "zarar",
    )
    REPAIR_MARKERS = ("iyileş", "onarı", "gider", "kazanç", "kazan")

    def __init__(self, owner: Any):
        self.owner = owner
        self.last_trace: dict[str, Any] = {}

    # ------------------------------------------------------------------ girdi normalizasyonu
    def _obs(self, durum: Any) -> np.ndarray:
        """Bir gözlem veya yalnız üç iç-duyu değeri verilse de aynı durumu kur."""
        if durum is None:
            return np.asarray(self.owner.obs, dtype=np.float64).copy()
        state = durum if isinstance(durum, dict) else {}
        raw = state.get("obs", state.get("observation")) if state else durum
        if raw is None:
            obs = np.asarray(self.owner.obs, dtype=np.float64).copy()
        else:
            arr = np.asarray(raw, dtype=np.float64)
            if arr.size == 3:
                obs = np.asarray(self.owner.obs, dtype=np.float64).copy()
                obs[45:48] = arr.reshape(3)
                return obs
            if arr.size != Universe.OBS_DIM:
                raise ValueError(f"obs_dim={Universe.OBS_DIM} or body_dim=3")
            obs = arr.reshape(Universe.OBS_DIM).copy()
        if state:
            for i, name in enumerate(self.BODY_VARS):
                if name in state:
                    obs[45 + i] = float(state[name])
            # Kısa İngilizce adlar yalnız API kolaylığıdır; grafik anlamı değildir.
            for i, name in enumerate(("e", "w", "i")):
                if name in state:
                    obs[45 + i] = float(state[name])
        return obs

    def _action(self, eylem: Any, target: str = "") -> tuple[Any, str, int | None]:
        """(normalleşmiş eylem, hedef, ayrık motor eylemi) döndürür."""
        action = eylem
        if isinstance(eylem, dict):
            target = str(
                eylem.get("target", eylem.get("hedef", eylem.get("concept", target))) or ""
            )
            action = eylem.get(
                "act", eylem.get("action", eylem.get("eylem", eylem.get("kind", "git")))
            )
        target = tr_lower(str(target or "")).strip()
        if isinstance(action, (int, np.integer)):
            a = int(action)
            if not 0 <= a < Universe.N_ACT:
                raise ValueError(f"bilinmeyen motor eylemi: {a}")
            return a, target, a
        text = tr_lower(str(action or "git")).strip()
        if text.isdigit():
            a = int(text)
            if not 0 <= a < Universe.N_ACT:
                raise ValueError(f"bilinmeyen motor eylemi: {a}")
            return a, target, a
        return text, target, self.DISCRETE_ACTIONS.get(text)

    def _context_target(self, obs: np.ndarray) -> str:
        """Açık hedef yoksa mevcut niyeti, sonra görünür nedensel hedefi kullan."""
        explicit = tr_lower(str(getattr(self.owner, "_prediction_target", "") or "")).strip()
        if explicit:
            return explicit
        mind = self.owner.mind
        fc = getattr(mind, "fc", None)
        pin = getattr(fc, "pin_goal", None)
        if isinstance(pin, (int, np.integer)) and 0 <= int(pin) < 3:
            return self.CHANNEL_NAMES[int(pin)]
        goal = getattr(mind, "goal", -1)
        if isinstance(goal, (int, np.integer)) and 0 <= int(goal) < 3:
            return self.CHANNEL_NAMES[int(goal)]
        # Görünür her şeyi hedef ilan etme: yalnız grafikte ileri nedensel yolu
        # olan baskın kanal, eylem-değerleme için aday olur.
        rays = np.asarray(obs[:45]).reshape(Universe.R, 5)
        proximity = rays[:, 1:4].max(axis=0)
        for ch in np.argsort(-proximity):
            if proximity[ch] < 0.55:
                break
            name = self.CHANNEL_NAMES[int(ch)]
            if self._graph_paths(name):
                return name
        return ""

    # ------------------------------------------------------------------ fiziksel hedef ve önizleme
    def _target_channel(self, target: str) -> int | None:
        target = tr_lower(str(target or "")).strip()
        if not target:
            return None
        if target in self.TARGET_CHANNELS:
            return self.TARGET_CHANNELS[target]
        # Yeni kavram için elle kanal eklemek yerine mevcut ISA taksonomisini izle.
        try:
            for ancestor, _ in self.owner.lang.graph.ancestors(target):
                if ancestor in self.TARGET_CHANNELS:
                    return self.TARGET_CHANNELS[ancestor]
        except Exception:
            pass
        return None

    def _target_positions(self, channel: int | None) -> np.ndarray:
        w = self.owner.world
        if channel == 0:
            return np.asarray(w.food, dtype=np.float64)
        if channel == 1:
            return np.asarray(w.water_c, dtype=np.float64)
        if channel == 2:
            return np.asarray(w.haz_c, dtype=np.float64)
        return np.zeros((0, 2), dtype=np.float64)

    @staticmethod
    def _distance(pos: np.ndarray, points: np.ndarray) -> float:
        if points.size == 0:
            return float("nan")
        return float(np.min(np.hypot(*(points - pos).T)))

    def _physics_source(self, obs: np.ndarray, motor_action: int | None,
                        target: str) -> tuple[list[dict], dict]:
        """Universe kopyasından yalnız mekanik/geometrik öngörü al."""
        if motor_action is None:
            return [], {"approach": 1.0, "progress": None}
        try:
            preview = self.owner.world.predict_transition(motor_action)
        except Exception:
            return [], {"approach": 0.0, "progress": None}
        candidates: list[dict] = []
        next_obs = np.asarray(preview["obs"], dtype=np.float64)
        for i, name in enumerate(("hız_ileri", "hız_yanal")):
            delta = float(next_obs[48 + i] - obs[48 + i])
            if abs(delta) > 1e-8:
                candidates.append({
                    "variable": name, "delta": delta, "confidence": 0.90,
                    "authority": 1.0, "source": "fizik", "target": target,
                })
        channel = self._target_channel(target)
        points = self._target_positions(channel)
        progress = None
        approach = 0.0
        if points.size:
            before = self._distance(np.asarray(self.owner.world.pos), points)
            after = self._distance(np.asarray(preview["position"]), points)
            progress = before - after
            # Yaklaşma, sonucun adından bağımsız geometrik bir ölçümdür.
            approach = 1.0 if progress > 1e-7 else 0.0
            if abs(progress) > 1e-8:
                candidates.append({
                    "variable": "ilerleme", "delta": float(progress), "confidence": 0.90,
                    "authority": 1.0, "source": "fizik", "target": target,
                })
        return candidates, {"approach": approach, "progress": progress}

    # ------------------------------------------------------------------ gövdesel kaynak
    def _body_source(self, obs: np.ndarray, action: Any, motor_action: int | None,
                     target: str) -> list[dict]:
        mind = self.owner.mind
        out: list[dict] = []
        channel = self._target_channel(target)
        # Doğrudan deneyimsel V, grafik duyumundan daha güçlü kanıttır.
        if channel is not None:
            try:
                ctx = mind.ctxs[mind.cur]
                v = float(ctx["V"][channel])
                n = int(ctx["Vn"][channel])
                if np.isfinite(v) and n >= 1 and abs(v) > 1e-6:
                    delta = float(np.clip(2.4 * v, -1.0, 1.0))
                    out.append({
                        "variable": self.BODY_VARS[channel], "delta": delta,
                        "confidence": float(min(0.98, 0.30 + 0.14 * n)),
                        "authority": 4.0, "source": "beden", "target": target,
                    })
            except Exception:
                pass
        # Wsm, Mind.imagine ile aynı öğrenilmiş geçiş/ışın modelinin beden
        # kısmıdır. Eğitim yoksa sıfır matrisi bilgi gibi sunulmaz.
        if motor_action is not None:
            try:
                ctx = mind.ctxs[mind.cur]
                n = int(ctx.get("n", 0))
                if n >= 3:
                    z = mind._zvec(obs, motor_action)
                    predicted = np.asarray(mind.Wsm @ z, dtype=np.float64)
                    confidence = float(np.clip(
                        (1.0 - min(1.0, float(mind.err_n) / 3.0)) * min(1.0, n / 40.0),
                        0.03, 0.90,
                    ))
                    for i, name in enumerate(self.BODY_VARS):
                        delta = float(predicted[i])
                        if abs(delta) > 0.005:
                            out.append({
                                "variable": name, "delta": float(np.clip(delta, -1.0, 1.0)),
                                "confidence": confidence, "authority": 3.0,
                                "source": "beden", "target": target,
                            })
            except Exception:
                pass
        # Aynı beden deneyiminin ikinci kanalı: ham önce/sonra geçişlerinden
        # öğrenilen model. Grafik/yiyecek adı bilmez; yalnız gerçek müdahale
        # kayıtlarıyla ve yeterli karşılaştırma varsa aday üretir.
        causal = getattr(self.owner, "causal", None)
        if causal is not None and motor_action is not None:
            for pred in causal.predict(obs, motor_action):
                name = str(pred.get("variable", ""))
                if not (name.startswith("f") and name[1:].isdigit()):
                    continue
                index = int(name[1:]) - 45
                if not 0 <= index < len(self.BODY_VARS):
                    continue
                out.append({
                    "variable": self.BODY_VARS[index],
                    "delta": float(np.clip(float(pred["delta"]), -1.0, 1.0)),
                    "confidence": float(pred["confidence"]), "authority": 2.8,
                    "source": "beden-geçiş", "target": target,
                    "novel": pred.get("evidence") == "intervention",
                    "evidence": pred.get("evidence", "association"),
                    "support": int(pred.get("support", 0)),
                })
        return out

    # ------------------------------------------------------------------ grafik etkisini bedene bağlama
    @classmethod
    def _concept_in_text(cls, concept: str, text: str) -> bool:
        concept = tr_lower(concept).strip()
        if not concept:
            return False
        for token in re.findall(r"[\wçğıöşü]+", tr_lower(text), flags=re.UNICODE):
            if token == concept or token.startswith(concept):
                # 'su' kökünün 'susuzluk' biçiminde olumlu su etkisi gibi
                # okunması yerine aşağıdaki genel eksiklik kuralı çalışır.
                if concept == "su" and token.startswith("susuz"):
                    continue
                return True
        return False

    def _effect_vector(self, effect: str) -> np.ndarray | None:
        """Grafik nesnesini mevcut somatik/beden anlamına çevir; bilinmeyeni uydurma."""
        text = tr_lower(str(effect or "")).strip()
        if not text:
            return None
        loss = any(marker in text for marker in self.LOSS_MARKERS)
        repair = any(marker in text for marker in self.REPAIR_MARKERS)
        # Eksiklik biçimleri, 'su'/'enerji' kazanmakla aynı değildir.
        if text.startswith("susuz"):
            return np.array([0.0, -1.0, 0.0], dtype=np.float64)
        if text.startswith("açlık") or text.startswith("açlığ"):
            return np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        # Doğrudan homeostatik değişken adları birim etkidir; büyüklüğü
        # kenar güveni belirler, test için özel sayı yazılmaz.
        for name, idx in self.BODY_INDEX.items():
            if text.startswith(name) or re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text):
                sign = -1.0 if loss and not repair else 1.0
                vec = np.zeros(3, dtype=np.float64)
                vec[idx] = sign
                return vec
        # En uzun kökü seçmek 'su' ile 'susuzluk' gibi çakışmaları azaltır.
        records = sorted(
            getattr(self.owner.self_model, "somatic", {}).items(),
            key=lambda item: len(item[0]), reverse=True,
        )
        for concept, rec in records:
            if concept in self.BODY_INDEX or not self._concept_in_text(concept, text):
                continue
            try:
                vec = np.asarray(rec.vec3, dtype=np.float64).copy()
            except Exception:
                continue
            if vec.size != 3 or not np.any(np.isfinite(vec)) or np.linalg.norm(vec) < 1e-6:
                continue
            vec = np.nan_to_num(vec, nan=0.0)
            if loss and not repair and float(np.sum(vec)) > 0:
                vec = -vec
            elif repair and not loss and float(np.sum(vec)) < 0:
                vec = -vec
            return np.clip(vec, -1.0, 1.0)
        return None

    def _graph_paths(self, target: str) -> list[list[Any]]:
        if not target:
            return []
        graph = self.owner.lang.graph
        paths: list[list[Any]] = []
        try:
            for _effect, path in graph.causal_chain(target):
                paths.append(list(path))
            # causal_chain doğrudan düğümleri izler; query ISA ile kalıtılan
            # genel kenarları da aynı kaynak olarak getirir.
            for edge, isa_path in graph.query(target, ("CAUSES", "IF_THEN")):
                paths.append([*isa_path, edge])
        except Exception:
            return []
        unique: list[list[Any]] = []
        seen = set()
        for path in paths:
            if not path:
                continue
            key = tuple((e.subj, e.rel, e.obj) for e in path)
            if key not in seen:
                seen.add(key)
                unique.append(path)
        return unique

    def _graph_source(self, target: str, approach: float) -> list[dict]:
        if not target or approach <= 0.0:
            return []
        out: list[dict] = []
        for path in self._graph_paths(target):
            causal = [e for e in path if e.rel in ("CAUSES", "IF_THEN")]
            if not causal:
                continue
            effect = causal[-1].obj
            vec = self._effect_vector(effect)
            if vec is None:
                continue
            confidence = 1.0
            for edge in path:
                confidence *= float(np.clip(edge.weight, 0.0, 1.0))
            source = getattr(causal[-1], "source", "duyum")
            confidence *= self.SOURCE_PRECISION.get(source, 0.50)
            confidence *= 0.85 ** max(0, len(causal) - 1)
            if any(e.rel == "IF_THEN" for e in causal):
                confidence *= 0.65
            confidence = float(np.clip(confidence, 0.02, 0.98))
            for i, name in enumerate(self.BODY_VARS):
                if abs(vec[i]) < 1e-8:
                    continue
                out.append({
                    "variable": name,
                    "delta": float(np.clip(vec[i] * confidence * approach, -1.0, 1.0)),
                    "confidence": confidence, "authority": 2.0,
                    "source": "grafik", "target": target,
                    # Doğuştan önsel, mevcut refleks/somatik politikada zaten
                    # temsil edilir. Alt düzey seçici bunu ikinci kez ödül
                    # gibi uygulamamak için kanıt türünü izde taşır.
                    "novel": any(
                        e.rel in ("CAUSES", "IF_THEN") and getattr(e, "source", "duyum") != "doğuştan"
                        for e in path
                    ),
                    "path": [(e.subj, e.rel, e.obj) for e in path],
                })
        return out

    # ------------------------------------------------------------------ tek arayüz ve değerleme
    def predict(self, durum: Any = None, eylem: Any = None) -> list[tuple[str, float, float]]:
        """``(değişken, delta, güven)`` listesiyle birleşik ileri öngörü yap.

        ``eylem`` bir motor indeksi, eylem adı veya ``{"act": ..., "target":
        ...}`` sözlüğü olabilir. ``durum`` tam gözlem, üç iç-duyu değeri ya
        da bunları içeren bir sözlüktür. Orijinal Universe ve Mind öğrenme
        durumları bu çağrıda ilerletilmez.
        """
        obs = self._obs(durum)
        state_target = ""
        if isinstance(durum, dict):
            state_target = str(
                durum.get("target", durum.get("hedef", durum.get("concept", ""))) or ""
            )
        action, target, motor_action = self._action(eylem, state_target)
        if not target:
            target = self._context_target(obs)
        # Üst düzey eylem hedefe yönelme niyetidir; ayrık motor eylemde ise
        # gerçek geometrik yaklaşma, grafik sonucunun uygulanırlığını belirler.
        high_level = not isinstance(action, (int, np.integer))
        physics, physical = self._physics_source(obs, motor_action, target)
        approach = 1.0 if high_level else float(physical.get("approach", 0.0))
        candidates = [
            *self._body_source(obs, action, motor_action, target),
            *self._graph_source(target, approach),
            *physics,
        ]
        # Aynı değişken için kanıt ağırlığı x güveni en yüksek olan kaynağı
        # seç. Böylece bir grafik zinciri kendi ara düğümlerini tekrar tekrar
        # toplayıp yapay olarak büyük bir delta üretmez.
        selected: dict[str, dict] = {}
        for candidate in candidates:
            name = candidate["variable"]
            score = float(candidate["authority"]) * float(candidate["confidence"])
            old = selected.get(name)
            if old is None or score > float(old["authority"]) * float(old["confidence"]):
                selected[name] = candidate
        order = {name: i for i, name in enumerate(self.BODY_VARS)}
        chosen = sorted(
            selected.values(),
            key=lambda c: (order.get(c["variable"], len(order)), c["variable"]),
        )
        result = [
            (str(c["variable"]), float(c["delta"]), float(np.clip(c["confidence"], 0.0, 1.0)))
            for c in chosen
            if abs(float(c["delta"])) > 1e-8
        ]
        self.last_trace = {
            "target": target, "action": action, "motor_action": motor_action,
            "approach": approach, "physical": physical,
            "candidates": candidates, "selected": chosen, "fused": result,
        }
        return result

    @staticmethod
    def value(predictions: list[tuple[str, float, float]]) -> float:
        """Öngörüyü homeostatik değerle çarp; mekanik değişkenler ödül değildir."""
        weights = np.array([1.0, 1.0, 1.35], dtype=np.float64)
        total = 0.0
        for name, delta, _confidence in predictions:
            if name in UnifiedPredictor.BODY_INDEX:
                total += weights[UnifiedPredictor.BODY_INDEX[name]] * float(delta)
        return float(total)

    def guided_action(self, proposed: int) -> int:
        """Öğrenilmiş grafik/geçiş kanıtı varsa eylemleri tahmin × değerle karşılaştır.

        Sonradan öğrenilmiş ilgili ileri yol veya yeterli müdahale-geçiş kanıtı
        yoksa mevcut politika aynen korunur; doğuştan önsel ikinci kez sayılmaz.
        Beraberlikte de önerilen eylem korunur; bu organ sürücüyü gereksiz yere
        ele geçirmez.
        """
        proposed = int(proposed)
        if not 0 <= proposed < Universe.N_ACT:
            return proposed
        target = self._context_target(np.asarray(self.owner.obs))
        paths = self._graph_paths(target)
        learned_path = any(
            any(
                e.rel in ("CAUSES", "IF_THEN") and getattr(e, "source", "duyum") != "doğuştan"
                for e in path
            )
            for path in paths
        )
        causal = getattr(self.owner, "causal", None)
        enough_transition = bool(causal is not None and
                                 getattr(causal, "transitions", 0) >= 2 * getattr(causal, "min_support", 3))
        if (not target or not learned_path) and not enough_transition:
            return proposed
        predictions: list[list[tuple[str, float, float]]] = []
        graph_found = False
        transition_found = False
        for action in range(Universe.N_ACT):
            pred = self.predict(self.owner.obs, {"act": action, "target": target})
            predictions.append(pred)
            for c in self.last_trace.get("candidates", []):
                if c.get("variable") not in self.BODY_INDEX:
                    continue
                if c.get("source") == "grafik" and c.get("novel", False):
                    graph_found = True
                if c.get("source") == "beden-geçiş" and c.get("novel", False):
                    transition_found = True
        if not graph_found and not transition_found:
            return proposed
        utilities = np.asarray([self.value(p) for p in predictions], dtype=np.float64)
        best = float(np.max(utilities))
        if float(utilities[proposed]) >= best - 1e-8:
            return proposed
        return int(np.argmax(utilities))
